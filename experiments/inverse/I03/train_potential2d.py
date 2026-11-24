"Train the two-dimensional potential reconstruction with tensor-product basis enrichment."
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
from common import atomic_json,atomic_torch,capture_rng,cpu_tree,digest,metrics,output_lock,restore_rng,seed_all,write_csv
from protocol import METHODS,plan
from problem import eta_truth
from resource_tracking import TrainingMeter,hardware_info

FORMAT='I03_2D_potential_inverse_supplied_source_v1'
STOP=False
SOURCE_FILES=['train_potential2d.py','models.py','original_components.py','physics.py','fractional.py','data.py',
              'problem.py','protocol.py','common.py','resource_tracking.py']
TRAJECTORY_FIELDS=['global_step','stage','epoch']+['c%d%d'%(m,n) for m in range(4) for n in range(4)]


def parse_args(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--method',choices=METHODS,default='go_multi')
    q.add_argument('--supervised-points',type=int,choices=[4,9],default=9)
    q.add_argument('--seed',type=int,default=1234);q.add_argument('--data-seed',type=int,default=77123)
    q.add_argument('--output',default=None);q.add_argument('--resume',action='store_true')
    q.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    q.add_argument('--width',type=int,default=60);q.add_argument('--extra-hidden-layers',type=int,default=3)
    q.add_argument('--network-dtype',choices=['float32','float64'],default='float32')
    q.add_argument('--freeze-hypernet',action='store_true')
    q.add_argument('--activation-initial',type=float,default=.1)
    q.add_argument('--activation-scale',type=float,default=10.)
    q.add_argument('--activation-lr-ratio',type=float,default=.1)
    q.add_argument('--hyper-lr-ratio',type=float,default=1.)
    q.add_argument('--matmul-precision',choices=['highest','high','medium'],default='high')
    q.add_argument('--eta-weight',type=float,default=1.)
    q.add_argument('--w-grad',type=float,default=.001,help='GO base GE weight (active weight = w_grad*exp(-3*detached loss_f)); matches local minus fixed true-local derivatives in x, y and t, excluding the Caputo forcing part. FL1 always disables GE.')
    q.add_argument('--consistency-mode',choices=['frozen','rolling','none'],default='frozen')
    q.add_argument('--condition-points',type=int,default=100)
    q.add_argument('--noise-level',type=float,default=0.,help='Noise on terminal field observations only; eta supervision remains exact.')
    q.add_argument('--history-rule',choices=['soe','direct'],default='soe')
    q.add_argument('--history-evaluation',choices=['fft','stream'],default='fft')
    q.add_argument('--quad-order',type=int,default=8);q.add_argument('--panel-width',type=float,default=1.)
    q.add_argument('--low-tail',type=float,default=1e-6)
    q.add_argument('--spatial-batch',type=int,default=None)
    q.add_argument('--max-epochs-per-stage',type=int,default=10000)
    q.add_argument('--eval-every',type=int,default=100);q.add_argument('--patience-evals',type=int,default=10)
    q.add_argument('--checkpoint-every',type=int,default=100)
    q.add_argument('--eta-eval-nodes',type=int,default=101)
    q.add_argument('--reference-points',type=int,default=4096);q.add_argument('--reference-seed',type=int,default=77201)
    q.add_argument('--threads',type=int,default=1);q.add_argument('--smoke',action='store_true')
    a=q.parse_args(argv)
    if a.spatial_batch is None:a.spatial_batch=0 if str(a.device).startswith('cuda') else 7
    if min(a.width,a.condition_points,a.eval_every,a.patience_evals,a.checkpoint_every,a.reference_points,a.quad_order,a.threads)<1:q.error('Counts must be positive.')
    if min(a.extra_hidden_layers,a.spatial_batch,a.seed,a.data_seed,a.reference_seed)<0:q.error('Negative count or seed.')
    if a.eta_eval_nodes<2:q.error('At least two eta evaluation nodes required.')
    if not 1<=a.max_epochs_per_stage<=10000:q.error('Per-stage cap must be 1..10000.')
    values=[a.eta_weight,a.w_grad,a.noise_level,a.panel_width,a.low_tail,a.activation_initial,a.activation_scale,a.activation_lr_ratio,a.hyper_lr_ratio]
    if not all(np.isfinite(values)) or min(a.w_grad,a.noise_level)<0 or min(a.eta_weight,a.panel_width,a.activation_initial,a.activation_scale,a.activation_lr_ratio,a.hyper_lr_ratio)<=0 or not 0<a.low_tail<1e-2:q.error('Invalid loss/numerical setting.')
    if a.history_rule=='direct' and a.history_evaluation!='fft':q.error('Direct history uses fft evaluation.')
    if a.smoke:
        a.width=8;a.extra_hidden_layers=1;a.max_epochs_per_stage=2;a.eval_every=1
        a.checkpoint_every=1;a.reference_points=24;a.condition_points=8;a.spatial_batch=3;a.eta_eval_nodes=21
    if a.output is None:a.output='runs/I03_gradient_v2/eta_'+str(a.supervised_points)+'/'+a.method+'/seed_'+str(a.seed)
    return a


def source_hashes():
    import hashlib
    root=Path(__file__).parent
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def signature(args):
    a=vars(args).copy()
    for k in ['output','resume']:a.pop(k,None)
    return digest(dict(args=a,sources=source_hashes()))


def snapshot(state,a,stage,epoch,updates,monitor=None):
    return dict(format=FORMAT,args=vars(a),stage=stage,epoch=epoch,updates=updates,
                state=models.snapshot(state),coefficients=models.coefficients(state),monitor=monitor)


def load_bundle(saved,device,optimizers=True):
    if saved.get('format')!=FORMAT:raise ValueError('Not an I03 checkpoint.')
    args=argparse.Namespace(**saved['args']);degree=saved['state']['degree']
    state=models.create(args,device,degree or 0);models.restore(state,saved['state'],optimizers);return state


def load_for_prediction(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    return load_bundle(saved,torch.device(device),False),saved


def predict_numpy(state,tx,batch=8192):
    result=[]
    with torch.no_grad():
        for start in range(0,len(tx),batch):
            q=torch.as_tensor(tx[start:start+batch],dtype=torch.float64,device=state['device'])
            result.append(models.predict(state,q).cpu().numpy())
    return np.concatenate(result)


def potential_numpy(state,xy):
    with torch.no_grad():
        return models.eta(state,torch.as_tensor(xy,dtype=torch.float64,device=state['device']).reshape(-1,2)).cpu().numpy().ravel()


def diagnostics(state,pool):
    with torch.no_grad():
        prediction=models.predict(state,pool['eval_tx']).cpu().numpy();target=pool['eval_target'].cpu().numpy()
    xy=pool['eta_eval_xy'].cpu().numpy();potential=potential_numpy(state,xy);truth=eta_truth(xy)
    return dict(eta_relative_l2=float(np.linalg.norm(potential-truth)/np.linalg.norm(truth)),
                eta_mae=float(np.mean(np.abs(potential-truth))),
                solution_relative_l2=metrics(prediction[:,0]+1j*prediction[:,1],target[:,0]+1j*target[:,1])['relative_l2'])


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
    torch.set_num_threads(a.threads);torch.set_float32_matmul_precision(a.matmul_precision);seed_all(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);items=plan(a.method,a.smoke);requested=signature(a)
    config_path=out/'config.json';resumed=None
    if config_path.exists():
        if not a.resume:raise FileExistsError('Existing run; use --resume or a new output.')
        config=json.loads(config_path.read_text())
        if config['signature']!=requested:raise ValueError('Configuration/code changed; use a new output directory.')
        if (out/'summary.json').exists():print('Matching completed run skipped.',flush=True);return json.loads((out/'summary.json').read_text())
        resumed=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
        now=hardware_info(device)
        if now['gpu']!=config['hardware']['gpu'] or str(now['torch'])!=config['hardware']['torch']:raise ValueError('Exact continuation requires the recorded GPU and PyTorch version.')
    else:
        if any(p.name!='.run.lock' for p in out.iterdir()):raise FileExistsError('Nonempty output without matching config.')
        hw=hardware_info(device);hw['network_dtype']=a.network_dtype
        config=dict(format=FORMAT,signature=requested,args=vars(a),plan=items,source_hashes=source_hashes(),hardware=hw,
                    gradient_enhancement=dict(enabled=a.method=='go_multi' and bool(a.w_grad),base_weight=a.w_grad,
                        active_weight='w_grad*exp(-3*loss_f.detach()); zero for FL1',
                        target='local minus fixed true-local x/y/t coordinate derivatives; includes eta_x, eta_y'),
                    gate='g=0 coordinate-only fixed-parameter inverse',
                    representation='sum c_mn cos(2pi*m*x) cos(2pi*n*y), k=0->1->2->3' if a.method=='go_multi' else '3 hidden layers x20 potential MLP',
                    observation='10->25->50->100 nested terminal complex-field observations at t=1; plus 4/9 sparse eta values',
                    stopping='validation physical loss; strict historical minimum; 10 checks; max10000 per stage',
                    selection='finest-stage minimum pre-update validation physical loss; true eta/field errors are diagnostics only',
                    transfer='LAST field, hypernet, activation and eta; inherited Adam moments padded by zeros for new basis functions')
        atomic_json(config_path,config)
    meter=TrainingMeter(device);clock=time.perf_counter();wall_before=0.;rows=[];stages=[]
    dataset=teacher=context=stage_best=None;stage0=epoch0=global_step=point_work=new_updates=0
    if resumed is None:
        pool=data_module.make_pool(a,device);state=models.create(a,device)
        np.savez_compressed(out/'observations.npz',**{k:pool[k].cpu().numpy() for k in ['obs_tx','obs_target','obs_clean','eta_xy','eta_target','eval_tx','eval_target','eta_eval_xy']})
    else:
        state=load_bundle(resumed['current'],device);pool=move(resumed['pool'],device)
        rows=resumed['rows'];stages=resumed['stages'];stage0=resumed['next_stage'];epoch0=resumed['next_epoch']
        context=resumed['context'];dataset=move(resumed['data'],device);teacher=move(resumed['teacher'],device);stage_best=resumed['stage_best']
        global_step=resumed['global_step'];point_work=resumed['point_work'];wall_before=resumed['wall_seconds']
        m=resumed['meter'];meter.seconds=m['optimization_seconds'];meter.steps=m['updates']
        meter.peak_allocated=m['peak_training_allocated_bytes'];meter.peak_reserved=m['peak_training_reserved_bytes']
        meter.peak_increment=m['peak_step_increment_allocated_bytes'];meter.max_step_seconds=m['max_measured_update_seconds']
        restore_rng(resumed['rng'])
    trajectory=out/'trajectory.csv'
    if resumed is None:
        with trajectory.open('w',newline='') as f:csv.DictWriter(f,fieldnames=TRAJECTORY_FIELDS).writeheader()
    else:
        with trajectory.open(newline='') as f:kept=[r for r in csv.DictReader(f) if int(r['global_step'])<=global_step]
        if len(kept)!=global_step:raise ValueError('Trajectory and checkpoint disagree.')
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
    print('%s | eta supervision %d | cap/stage %d'%(a.method,a.supervised_points,a.max_epochs_per_stage),flush=True)
    with trajectory.open('a',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=TRAJECTORY_FIELDS)
        def save_resume(sid,epoch):
            handle.flush()
            atomic_torch(out/'resume.pt',dict(format=FORMAT,current=snapshot(state,a,sid,epoch-1,global_step),
                         next_stage=sid,next_epoch=epoch,context=context,data=cpu_tree(dataset),teacher=cpu_tree(teacher),pool=cpu_tree(pool),
                         stage_best=stage_best,rows=rows,stages=stages,global_step=global_step,point_work=point_work,
                         meter=meter.report(),wall_seconds=elapsed(),rng=capture_rng()))
        for sid in range(stage0,len(items)):
            item=items[sid]
            if dataset is None:
                inherited=models.hashes(state)
                previous=data_module.build(items[sid-1],pool,a,device) if sid and a.method=='go_multi' else None
                teacher=physics.make_teacher(state,previous,a)
                enrichment=models.enrich(state,item['degree']) if a.method=='go_multi' else dict(expanded=False)
                dataset=data_module.build(item,pool,a,device);models.set_lr(state,item['lr'],a)
                context=dict(best_loss=float('inf'),bad=0,start_step=global_step,start_seconds=meter.seconds,
                             transferred_from=inherited,entry=models.hashes(state),enrichment=enrichment)
                stage_best=None;epoch0=0
                atomic_torch(out/('stage_%02d_entry.pt'%sid),snapshot(state,a,sid,-1,global_step));save_resume(sid,0)
            print('S%d | %d^2 x %d + %d field observations | k=%s | lr %.3e'%(sid,item['M'],item['N'],item['observations'],item['degree'],item['lr']),flush=True)
            epoch=epoch0;stopped=False
            while epoch<a.max_epochs_per_stage:
                meter.begin()
                for opt in state['optimizers']:opt.zero_grad(set_to_none=True)
                values=physics.backward(state,dataset,a,epoch,teacher)
                by_cap=epoch+1==a.max_epochs_per_stage;check=epoch%a.eval_every==0 or by_cap;new_best=False
                if check:
                    validation_loss=physics.validation_loss(state,validation_data,validation_args)
                    values['validation_loss']=validation_loss
                    if validation_loss<context['best_loss']:
                        context['best_loss']=validation_loss;context['bad']=0
                        stage_best=snapshot(state,a,sid,epoch,global_step,dict(loss=values['loss'],timing='pre-update'));new_best=True
                    else:context['bad']+=1
                for opt in state['optimizers']:
                    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for g in opt.param_groups for p in g['params']):raise FloatingPointError('Nonfinite gradient.')
                    opt.step()
                meter.end();global_step+=1;new_updates+=1;point_work+=(item['M']-1)**2*item['N']
                if new_best:atomic_torch(out/('stage_%02d_best_loss.pt'%sid),stage_best)
                cs=models.coefficients(state)
                writer.writerow(dict(global_step=global_step,stage=sid,epoch=epoch,**{'c%d%d'%(m,n):cs[m][n] if cs is not None and m<len(cs) and n<len(cs) else None for m in range(4) for n in range(4)}))
                if check:
                    diagnostic=diagnostics(state,pool);physics.refresh_teacher(state,teacher,a)
                    rows.append(dict(stage=sid,epoch=epoch,global_step=global_step,**values,**diagnostic,
                                     coefficients=models.coefficients(state),bad_checks=context['bad'],optimization_seconds=meter.seconds,
                                     elapsed_seconds=elapsed(),residual_points_processed=point_work))
                    write_csv(out/'training.csv',rows);handle.flush()
                    print('S%d epoch %5d | loss %.3e | eta L2 %.3e | field L2 %.3e | bad %d/%d'%(sid,epoch,values['loss'],diagnostic['eta_relative_l2'],diagnostic['solution_relative_l2'],context['bad'],a.patience_evals),flush=True)
                epoch+=1;stopped=check and context['bad']>=a.patience_evals
                if stopped or by_cap:break
                if check or epoch%a.checkpoint_every==0 or STOP:save_resume(sid,epoch)
                if STOP or (pause_after_updates is not None and new_updates>=pause_after_updates):
                    save_resume(sid,epoch);return dict(status='PAUSED',global_step=global_step)
            atomic_torch(out/('stage_%02d_last.pt'%sid),snapshot(state,a,sid,epoch-1,global_step))
            if stage_best is None:raise AssertionError('No loss checkpoint.')
            rng=capture_rng();best=load_bundle(stage_best,device,False);restore_rng(rng)
            stages.append(dict(**item,epochs=epoch,global_step=global_step,stop_reason='loss_patience' if stopped else 'epoch_cap',
                               transferred_from=context['transferred_from'],entry=context['entry'],last=models.hashes(state),
                               enrichment=context['enrichment'],last_coefficients=models.coefficients(state),best_coefficients=stage_best['coefficients'],
                               best_metrics=diagnostics(best,pool),best_loss=context['best_loss'],best_epoch=stage_best['epoch'],best_updates=stage_best['updates'],
                               optimization_seconds=meter.seconds-context['start_seconds'],trainable_parameters=models.parameter_count(state)))
            atomic_json(out/'stage_history.json',stages)
            dataset=teacher=context=None;epoch0=0;save_resume(sid+1,0)
            if STOP or pause_after_stage==sid:return dict(status='PAUSED_BETWEEN_STAGES',global_step=global_step)
    atomic_torch(out/'final.pt',stage_best);selected=load_bundle(stage_best,device,False)
    summary=dict(format=FORMAT,status='SMOKE_ONLY' if a.smoke else 'COMPLETED',method=a.method,
                 supervised_points=a.supervised_points,seed=a.seed,total_updates=global_step,stages=stages,
                 coefficients=models.coefficients(selected),metrics=diagnostics(selected,pool),eta_evaluation='fixed full Cartesian grid, Frobenius norm',
                 selected_stage=stage_best['stage'],selected_epoch=stage_best['epoch'],selected_updates=stage_best['updates'],
                 selection='finest-stage best pre-update validation physical loss; no ground-truth selection',
                 resources=meter.report(),wall_seconds=elapsed(),residual_points_processed=point_work,
                 manuscript_accuracy_reproduced=False)
    atomic_json(out/'summary.json',summary)
    axis=np.linspace(0,1,a.eta_eval_nodes);xy=pool['eta_eval_xy'].cpu().numpy()
    np.savez_compressed(out/'potential_prediction.npz',x=axis,y=axis,prediction=potential_numpy(selected,xy).reshape(len(axis),len(axis)),reference=eta_truth(xy).reshape(len(axis),len(axis)))
    print('Completed | '+str(summary['metrics']),flush=True);return summary


def main(argv=None):
    signal.signal(signal.SIGINT,_stop);signal.signal(signal.SIGTERM,_stop);return run(argv)


if __name__=='__main__':main()
