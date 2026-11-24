"Train the constant-coefficient inverse models with staged transfer and early stopping."
import argparse
import csv
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
import data as data_module
import models
import physics
from common import (atomic_json,atomic_torch,capture_rng,cpu_tree,digest,metrics,
                    output_lock,restore_rng,seed_all,write_csv)
from problem import exact_numpy,independent_points
from protocol import TRUE,METHODS,RUN_ROOT,plan,unknown_names,gradient_contract
from resource_tracking import TrainingMeter,hardware_info

FORMAT='I01_supplied_scripts_constant_inverse_v1'
STOP=False
SOURCE_FILES=['train_inverse.py','original_components.py','models.py','physics.py','fractional.py',
              'data.py','problem.py','protocol.py','common.py','resource_tracking.py']
TRAJECTORY_FIELDS=['global_step','stage','epoch','alpha','beta','gamma',
                   'alpha_relative_error','beta_relative_error','gamma_relative_error','unknown_relative_l2']


def parse_args(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--method',choices=METHODS,default='go_multi')
    q.add_argument('--unknown',choices=['alpha','beta','gamma','joint'],default='alpha')
    q.add_argument('--initial',default=None,help='Single value, or alpha,beta,gamma for joint.')
    q.add_argument('--seed',type=int,default=1234);q.add_argument('--data-seed',type=int,default=77123)
    q.add_argument('--output',default=None);q.add_argument('--resume',action='store_true')
    q.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    q.add_argument('--width',type=int,default=60);q.add_argument('--extra-hidden-layers',type=int,default=3)
    q.add_argument('--network-dtype',choices=['float32','float64'],default='float32')
    q.add_argument('--shared-update',choices=['legacy_double','once'],default='legacy_double')
    q.add_argument('--physical-lr-scale',type=float,default=1.)
    q.add_argument('--freeze-hypernet',action='store_true',help='Explicit source-script comparison: do not optimize the weighting MLP.')
    q.add_argument('--w-grad',type=float,default=.001)
    q.add_argument('--gradient-mode',choices=['script','x','xt','none'],default='script',help='Default script=xt: local GE in x+t for alpha/beta/gamma/joint; x is the explicit spatial-only comparison; none disables GE. FL1 always disables GE.')
    q.add_argument('--consistency-mode',choices=['rolling','frozen','none'],default='frozen')
    q.add_argument('--noise-level',type=float,default=0.,help='Relative Gaussian noise on terminal complex observations only.')
    q.add_argument('--condition-points',type=int,default=100)
    q.add_argument('--history-rule',choices=['soe','direct'],default='soe')
    q.add_argument('--history-evaluation',choices=['fft','stream'],default='fft')
    q.add_argument('--quad-order',type=int,default=8);q.add_argument('--panel-width',type=float,default=1.)
    q.add_argument('--low-tail',type=float,default=1e-6)
    q.add_argument('--alpha-min',type=float,default=1e-4);q.add_argument('--alpha-max',type=float,default=1-1e-4)
    q.add_argument('--spatial-batch',type=int,default=None,help='0=full batch; default CUDA 0, CPU 7; complete history in every spatial batch.')
    q.add_argument('--max-epochs-per-stage',type=int,default=10000)
    q.add_argument('--eval-every',type=int,default=100);q.add_argument('--patience-evals',type=int,default=10)
    q.add_argument('--checkpoint-every',type=int,default=100)
    q.add_argument('--reference-points',type=int,default=4096);q.add_argument('--reference-seed',type=int,default=77201)
    q.add_argument('--threads',type=int,default=1);q.add_argument('--smoke',action='store_true')
    a=q.parse_args(argv)
    if a.spatial_batch is None:a.spatial_batch=0 if str(a.device).startswith('cuda') else 7
    if a.initial is None:a.initial='.5,.5,.5' if a.unknown=='joint' else '.5'
    try:a.initial_values=[float(v) for v in a.initial.split(',')]
    except ValueError:q.error('Invalid initial value.')
    if len(a.initial_values)!=len(unknown_names(a.unknown)):q.error('Initial-value count does not match the unknowns.')
    if not all(np.isfinite(v) for v in a.initial_values):q.error('Nonfinite initial value.')
    if not 0<a.alpha_min<a.alpha_max<1:q.error('Require 0 < alpha-min < alpha-max < 1.')
    if a.unknown in ['alpha','joint'] and not a.alpha_min<a.initial_values[0]<a.alpha_max:q.error('Initial alpha must lie inside its fractional domain.')
    if min(a.width,a.condition_points,a.eval_every,a.patience_evals,a.checkpoint_every,a.reference_points,a.quad_order,a.threads)<1:q.error('Counts must be positive.')
    if min(a.extra_hidden_layers,a.spatial_batch,a.seed,a.data_seed,a.reference_seed)<0:q.error('Negative count or seed.')
    if not 1<=a.max_epochs_per_stage<=10000:q.error('The requested per-stage cap must be between 1 and 10000.')
    if not all(np.isfinite(v) for v in [a.physical_lr_scale,a.w_grad,a.noise_level,a.panel_width,a.low_tail]):q.error('Nonfinite setting.')
    if min(a.w_grad,a.noise_level)<0 or min(a.physical_lr_scale,a.panel_width)<=0 or not 0<a.low_tail<1e-2:q.error('Invalid numerical setting.')
    if a.history_rule=='direct' and a.history_evaluation!='fft':q.error('Direct reference uses fft evaluation.')
    if a.smoke:
        a.width=8;a.extra_hidden_layers=1;a.max_epochs_per_stage=2;a.eval_every=1
        a.checkpoint_every=1;a.reference_points=32;a.condition_points=8;a.spatial_batch=2
    if a.output is None:
        tag=a.initial.replace('.','p').replace(',','_')
        a.output=RUN_ROOT+'/'+a.unknown+'_'+tag+'/'+a.method+'/seed_'+str(a.seed)
    return a


def source_hashes():
    import hashlib
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def signature(a):
    values=vars(a).copy()
    for name in ['output','resume']:values.pop(name,None)
    return digest(dict(args=values,sources=source_hashes()))


def errors(values,unknown):
    result={k+'_relative_error':abs(values[k]-TRUE[k])/abs(TRUE[k]) for k in TRUE}
    names=unknown_names(unknown)
    result['unknown_relative_l2']=float(np.linalg.norm([values[k]-TRUE[k] for k in names])/np.linalg.norm([TRUE[k] for k in names]))
    return result


def snapshot(state,a,stage,epoch,updates,monitor=None):
    return dict(format=FORMAT,args=vars(a),stage=stage,epoch=epoch,updates=updates,
                state=models.snapshot(state),parameters=models.values(state),monitor=monitor)


def load_bundle(saved,device,optimizers=True):
    if saved.get('format')!=FORMAT:raise ValueError('Not an I01 inverse checkpoint.')
    args=argparse.Namespace(**saved['args']);state=models.create(args,device)
    models.restore(state,saved['state'],optimizers);return state


def load_for_prediction(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    return load_bundle(saved,torch.device(device),False),saved


def predict_numpy(state,tx,batch=8192):
    output=[]
    with torch.no_grad():
        for start in range(0,len(tx),batch):
            point=torch.as_tensor(tx[start:start+batch],device=state['device'],dtype=torch.float64)
            output.append(models.predict(state,point).cpu().numpy())
    return np.concatenate(output)


def move(v,device):
    if isinstance(v,torch.Tensor):return v.to(device).clone()
    if isinstance(v,dict):return {k:move(x,device) for k,x in v.items()}
    if isinstance(v,list):return [move(x,device) for x in v]
    return v


def _stop(signum,frame):
    global STOP
    STOP=True


def run(argv=None,pause_after_updates=None,pause_after_stage=None):
    a=parse_args(argv)
    with output_lock(a.output):return _run(a,pause_after_updates,pause_after_stage)


def _run(a,pause_after_updates=None,pause_after_stage=None):
    global STOP
    STOP=False;device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
    torch.set_num_threads(a.threads);seed_all(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);requested=signature(a);items=plan(a.method,a.smoke)
    config_path=out/'config.json';resumed=None
    if config_path.exists():
        if not a.resume:raise FileExistsError('Existing run; use --resume or a new output directory.')
        config=json.loads(config_path.read_text())
        if config['signature']!=requested:raise ValueError('Code/configuration changed; use a new output directory.')
        if (out/'summary.json').exists():
            print('Matching run completed; skipped.',flush=True);return json.loads((out/'summary.json').read_text())
        resumed=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
        now=hardware_info(device)
        if now['gpu']!=config['hardware']['gpu'] or str(now['torch'])!=config['hardware']['torch']:
            raise ValueError('Exact continuation requires the recorded GPU/PyTorch version.')
    else:
        if any(p.name!='.run.lock' for p in out.iterdir()):raise FileExistsError('Nonempty output without matching configuration.')
        hw=hardware_info(device);hw['network_dtype']=a.network_dtype;hw['history_dtype']='float64'
        config=dict(format=FORMAT,signature=requested,args=vars(a),plan=items,source_hashes=source_hashes(),hardware=hw,
                    gate='g=0 fixed-parameter coordinate branch',unknowns=unknown_names(a.unknown),true_parameters=TRUE,
                    observation='+M denotes M terminal complex observations at t=1; nested common pool',
                    conditions='100 IC and 100 per physical spatial face by default, additional to +M',
                    forcing='fixed true-parameter manufactured function evaluated at current scheme coordinates',
                    gradient_enhancement=gradient_contract(a.unknown,a.method,a.gradient_mode,a.w_grad),
                    alpha_history='fully differentiable SOE weights, low-frequency node, sigma, local/history coefficients; never frozen at initial alpha',
                    physical_optimizer='one independent Adam; no duplicate physical parameter update',
                    hypernet='4-16-4 softmax attention with symmetric perturbation; optimized unless --freeze-hypernet',
                    stopping='validation physical loss; strict historical minimum; 10 checks; max10000 per stage',
                    selection='final-stage best PRE-UPDATE validation-loss checkpoint; true parameter/solution errors never select or stop',
                    transfer='stage-LAST weights, all physical estimates and all Adam states; consistency on inherited prediction points')
        atomic_json(config_path,config)
    reference=independent_points(a.reference_seed,a.reference_points);target=exact_numpy(reference)
    meter=TrainingMeter(device);clock=time.perf_counter();wall_before=0.
    rows=[];stages=[];stage0=epoch0=global_step=point_work=new_updates=0
    context=dataset=teacher=stage_best=None
    if resumed is None:
        pool=data_module.make_pool(a,device);state=models.create(a,device)
        np.savez_compressed(out/'observations.npz',tx=pool['obs_tx'].cpu().numpy(),target=pool['obs_target'].cpu().numpy(),clean=pool['obs_clean'].cpu().numpy())
        np.savez_compressed(out/'reference_coordinates.npz',tx=reference)
    else:
        state=load_bundle(resumed['current'],device);pool=move(resumed['pool'],device)
        rows=resumed['rows'];stages=resumed['stages'];stage0=resumed['next_stage'];epoch0=resumed['next_epoch']
        context=resumed['context'];dataset=move(resumed['data'],device);teacher=move(resumed['teacher'],device)
        stage_best=resumed['stage_best'];global_step=resumed['global_step'];point_work=resumed['point_work'];wall_before=resumed['wall_seconds']
        m=resumed['meter'];meter.seconds=m['optimization_seconds'];meter.steps=m['updates']
        meter.peak_allocated=m['peak_training_allocated_bytes'];meter.peak_reserved=m['peak_training_reserved_bytes']
        meter.peak_increment=m['peak_step_increment_allocated_bytes'];meter.max_step_seconds=m['max_measured_update_seconds']
        restore_rng(resumed['rng'])
    trajectory=out/'trajectory.csv'
    if resumed is None:
        with trajectory.open('w',newline='') as f:csv.DictWriter(f,fieldnames=TRAJECTORY_FIELDS).writeheader()
    else:
        with trajectory.open(newline='') as f:kept=[r for r in csv.DictReader(f) if int(r['global_step'])<=global_step]
        if len(kept)!=global_step:raise ValueError('Trajectory does not match the saved optimizer checkpoint.')
        with trajectory.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=TRAJECTORY_FIELDS);w.writeheader();w.writerows(kept)
        write_csv(out/'training.csv',rows);atomic_json(out/'stage_history.json',stages)
    validation_args=argparse.Namespace(**vars(a))
    validation_args.data_seed+=104729
    validation_args.w_grad=0.
    validation_pool=data_module.make_pool(validation_args,device)
    validation_item=dict(items[-1],M=items[-1]['M']+1)
    validation_data=data_module.build(validation_item,validation_pool,validation_args,device)
    def elapsed():return wall_before+time.perf_counter()-clock
    with trajectory.open('a',newline='') as trajectory_handle:
        writer=csv.DictWriter(trajectory_handle,fieldnames=TRAJECTORY_FIELDS)
        def save_resume(sid,epoch):
            trajectory_handle.flush()
            atomic_torch(out/'resume.pt',dict(format=FORMAT,current=snapshot(state,a,sid,epoch-1,global_step),
                         next_stage=sid,next_epoch=epoch,context=context,data=cpu_tree(dataset),teacher=cpu_tree(teacher),
                         pool=cpu_tree(pool),stage_best=stage_best,rows=rows,stages=stages,global_step=global_step,
                         point_work=point_work,meter=meter.report(),wall_seconds=elapsed(),rng=capture_rng()))
        print('%s | unknown=%s | initial=%s | stages=%d | cap/stage=%d' % (a.method,a.unknown,a.initial,len(items),a.max_epochs_per_stage),flush=True)
        for sid in range(stage0,len(items)):
            item=items[sid]
            if dataset is None:
                before=models.hashes(state)
                previous=(data_module.build(items[sid-1],pool,a,device) if sid and a.method=='go_multi' else None)
                teacher=physics.make_teacher(state,previous,a) if a.method=='go_multi' else None
                dataset=data_module.build(item,pool,a,device);models.set_lr(state,item['lr'])
                for group in state['optimizers'][-1].param_groups:group['lr']=item['lr']*a.physical_lr_scale
                assert models.hashes(state)==before
                context=dict(best_loss=float('inf'),bad=0,start_step=global_step,start_seconds=meter.seconds,
                             entry=before,entry_parameters=models.values(state))
                stage_best=None;epoch0=0
                atomic_torch(out/('stage_%02d_entry.pt'%sid),snapshot(state,a,sid,-1,global_step));save_resume(sid,0)
            print('S%d | %dx%d + %d terminal observations | lr=%.3e' % (sid,item['M'],item['N'],item['observations'],item['lr']),flush=True)
            epoch=epoch0;stopped=False
            while epoch<a.max_epochs_per_stage:
                meter.begin()
                for opt in state['optimizers']:opt.zero_grad(set_to_none=True)
                values=physics.backward(state,dataset,a,epoch,teacher)
                by_cap=epoch+1==a.max_epochs_per_stage;check=epoch%a.eval_every==0 or by_cap
                new_best=False
                if check:
                    validation_loss=physics.validation_loss(state,validation_data,validation_args)
                    values['validation_loss']=validation_loss
                    if validation_loss<context['best_loss']:
                        context['best_loss']=validation_loss;context['bad']=0
                        
                        stage_best=snapshot(state,a,sid,epoch,global_step,dict(loss=values['loss'],timing='pre-update'))
                        new_best=True
                    else:context['bad']+=1
                for opt in state['optimizers']:
                    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for g in opt.param_groups for p in g['params']):
                        raise FloatingPointError('Nonfinite gradient.')
                    opt.step()
                models.project_alpha(state,a);meter.end();global_step+=1;new_updates+=1
                if new_best:atomic_torch(out/('stage_%02d_best_loss.pt'%sid),stage_best)
                point_work+=(item['M']-1)*item['N']
                current=models.values(state)
                writer.writerow(dict(global_step=global_step,stage=sid,epoch=epoch,**current,**errors(current,a.unknown)))
                if check:
                    prediction=predict_numpy(state,reference)
                    solution_l2=metrics(prediction[:,0]+1j*prediction[:,1],target)['relative_l2']
                    physics.refresh_teacher(state,teacher,a)
                    row=dict(stage=sid,epoch=epoch,global_step=global_step,M=item['M'],N=item['N'],observations=item['observations'],
                             lr=item['lr'],physical_lr=item['lr']*a.physical_lr_scale,**values,**current,**errors(current,a.unknown),
                             reference_solution_relative_l2=solution_l2,bad_checks=context['bad'],
                             optimization_seconds=meter.seconds,elapsed_seconds=elapsed(),residual_points_processed=point_work)
                    rows.append(row);write_csv(out/'training.csv',rows);trajectory_handle.flush()
                    print('S%d epoch %5d | loss %.3e | a %.7f b %.7f g %.7f | solution L2 %.3e | bad %d/%d' %
                          (sid,epoch,values['loss'],current['alpha'],current['beta'],current['gamma'],solution_l2,context['bad'],a.patience_evals),flush=True)
                epoch+=1;stopped=check and context['bad']>=a.patience_evals
                if stopped or by_cap:break
                if check or epoch%a.checkpoint_every==0 or STOP:save_resume(sid,epoch)
                if STOP or (pause_after_updates is not None and new_updates>=pause_after_updates):
                    save_resume(sid,epoch);return dict(status='PAUSED',global_step=global_step)
            last=snapshot(state,a,sid,epoch-1,global_step)
            atomic_torch(out/('stage_%02d_last.pt'%sid),last)
            if stage_best is None:raise AssertionError('No validation-loss checkpoint.')
            last_values=models.values(state);best_values=stage_best['parameters']
            stages.append(dict(**item,epochs=epoch,global_step=global_step,stop_reason='loss_patience' if stopped else 'epoch_cap',
                               entry=context['entry'],last=models.hashes(state),entry_parameters=context['entry_parameters'],
                               last_parameters=last_values,last_parameter_errors=errors(last_values,a.unknown),
                               best_parameters=best_values,best_parameter_errors=errors(best_values,a.unknown),
                               best_loss=context['best_loss'],best_epoch=stage_best['epoch'],best_updates=stage_best['updates'],
                               optimization_seconds=meter.seconds-context['start_seconds'],cumulative_optimization_seconds=meter.seconds,
                               trainable_parameters=models.parameter_count(state)))
            atomic_json(out/'stage_history.json',stages)
            dataset=teacher=context=None;epoch0=0;save_resume(sid+1,0)
            if STOP or pause_after_stage==sid:return dict(status='PAUSED_BETWEEN_STAGES',global_step=global_step)
    atomic_torch(out/'final.pt',stage_best)
    selected=load_bundle(stage_best,device,False);pred=predict_numpy(selected,reference)
    final_values=models.values(selected)
    summary=dict(format=FORMAT,status='SMOKE_ONLY' if a.smoke else 'COMPLETED',method=a.method,unknown=a.unknown,
                 gradient_enhancement=gradient_contract(a.unknown,a.method,a.gradient_mode,a.w_grad),
                 initial_values=a.initial_values,seed=a.seed,total_updates=global_step,stages=stages,
                 selected_stage=stage_best['stage'],selected_epoch=stage_best['epoch'],selected_updates=stage_best['updates'],
                 selection='finest-stage best pre-update validation physical loss; no ground-truth selection',
                 parameters=final_values,parameter_errors=errors(final_values,a.unknown),
                 reference_solution_metrics=metrics(pred[:,0]+1j*pred[:,1],target),resources=meter.report(),wall_seconds=elapsed(),
                 residual_points_processed=point_work,accuracy_statement='Actual measured errors; no guarantee of matching the supplied manuscript curves.')
    atomic_json(out/'summary.json',summary)
    np.savez_compressed(out/'reference_predictions.npz',tx=reference,prediction=pred,reference=target)
    print('Completed '+a.method+' | '+str(final_values),flush=True)
    return summary


def main(argv=None):
    signal.signal(signal.SIGINT,_stop);signal.signal(signal.SIGTERM,_stop)
    return run(argv)


if __name__=='__main__':main()
