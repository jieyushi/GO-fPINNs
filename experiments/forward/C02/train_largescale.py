"""Train the large-domain forward models with spatial enrichment."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
import backend as b
from common import (atomic_json,atomic_torch,capture_rng,cpu_tree,digest,
                    independent_points,metrics,output_lock,restore_rng,seed_all,write_csv)
from problem import reference_field,audit,SOURCE_MODEL
from protocol import plan,method_scheme,XL,XR

FORMAT='C02_manufactured_source_spatial_C1_DD_v2'
STOP=False
SOURCE_FILES=['train_largescale.py','backend.py','protocol.py','problem.py','common.py',
              'fractional_schemes.py','resource_tracking.py','c01_source/baseline_original.py',
              'c01_source/weight_schemes.py','c01_source/direct_reference.py']


def parse_args(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--method',choices=['go','fpinn','fl1'],default='go')
    q.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    q.add_argument('--seed',type=int,default=1234)
    q.add_argument('--output',default=None)
    q.add_argument('--width',type=int,default=60)
    q.add_argument('--extra-hidden-layers',type=int,default=2,help='C01 convention: one input hidden block plus this many blocks.')
    q.add_argument('--network-dtype',choices=['float32','float64'],default='float32')
    q.add_argument('--kernel-mode',choices=['original','stable64'],default='original')
    q.add_argument('--shared-update',choices=['legacy_double','once'],default='legacy_double')
    q.add_argument('--w-grad',type=float,default=.001,
                   help='GO full discrete residual x-gradient and sigma-time differences; weight=w_grad*exp(-3*LF.detach()); 0 disables GE. Baselines always disable GE.')
    q.add_argument('--noise-amplitude',type=float,default=.05,help='Adaptive loss-weight perturbation, NOT observation noise.')
    q.add_argument('--weight-lower-bound',type=float,default=1e-6)
    q.add_argument('--interface-weight',type=float,default=1.)
    q.add_argument('--interface-points',type=int,default=101,help='Fixed times including 0 and 1 on EACH current interface.')
    q.add_argument('--eval-every',type=int,default=100)
    q.add_argument('--patience-evals',type=int,default=10)
    q.add_argument('--max-epochs-per-stage',type=int,default=0,help='0 retains C01 uncapped loss-patience stopping; positive adds an explicit safety cap.')
    q.add_argument('--checkpoint-every',type=int,default=100)
    q.add_argument('--eval-points',type=int,default=4096)
    q.add_argument('--validation-spatial-sites',type=int,default=32)
    q.add_argument('--validation-time-intervals',type=int,default=50)
    q.add_argument('--validation-seed',type=int,default=98101)
    q.add_argument('--test-seed',type=int,default=98102)
    q.add_argument('--threads',type=int,default=1)
    q.add_argument('--resume',action='store_true')
    q.add_argument('--smoke',action='store_true')
    a=q.parse_args(argv)
    if min(a.width,a.interface_points,a.eval_every,a.patience_evals,a.checkpoint_every,a.eval_points,
           a.validation_spatial_sites,a.validation_time_intervals,a.threads)<1 or a.extra_hidden_layers<0 or a.max_epochs_per_stage<0:
        q.error('Invalid counts/layers/cap.')
    if a.interface_points<2 or a.validation_time_intervals<2:q.error('At least two interface times and time intervals required.')
    if min(a.seed,a.validation_seed,a.test_seed)<0 or a.validation_seed==a.test_seed:q.error('Independent nonnegative evaluation seeds required.')
    if not all(np.isfinite(v) for v in [a.w_grad,a.noise_amplitude,a.weight_lower_bound,a.interface_weight]):q.error('Nonfinite weights.')
    if min(a.w_grad,a.noise_amplitude)<0 or min(a.weight_lower_bound,a.interface_weight)<=0:q.error('Invalid weights.')
    if a.network_dtype=='float64' and a.kernel_mode=='original':q.error('Use stable64 for float64 networks.')
    a.alpha=.9;a.scheme=method_scheme(a.method)
    if a.smoke:
        a.width=8;a.extra_hidden_layers=1;a.max_epochs_per_stage=3
        a.eval_every=1;a.checkpoint_every=1;a.eval_points=64
        a.validation_spatial_sites=6;a.validation_time_intervals=4;a.interface_points=9
    if a.output is None:a.output='runs/C02_gradient_v2/'+a.method+'/seed_'+str(a.seed)
    return a


def source_hashes():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def signature(a):
    args=vars(a).copy()
    for name in ['resume','output']:args.pop(name,None)
    return hashlib.sha256(json.dumps({'args':args,'sources':source_hashes()},sort_keys=True).encode()).hexdigest()


def snapshot(states,domains,a,stage,epoch,validation=None):
    return dict(format=FORMAT,args=vars(a),stage=stage,epoch=epoch,domains=domains,
                states=[b.snapshot(s) for s in states],validation=validation)


def load_bundle(saved,device,optimizers=True):
    if saved.get('format')!=FORMAT:raise ValueError('Not a C02 C1-domain-decomposition checkpoint.')
    args=argparse.Namespace(**saved['args'])
    states=[b.create_state(args,device) for _ in saved['states']]
    for s,row in zip(states,saved['states']):b.restore(s,row,optimizers)
    return states


def load_for_prediction(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    return load_bundle(saved,torch.device(device),False),saved


def unpack_data(saved,device):
    result=[]
    for data in saved:
        def to_device(value):
            if isinstance(value,torch.Tensor):return value.to(device).clone()
            if isinstance(value,dict):return {k:to_device(v) for k,v in value.items()}
            return value
        d=to_device(data)
        if 'sigma_tx' in d:d['sigma_tx'].requires_grad_(True)
        result.append(d)
    return result


def model_hash(states):return digest([b.weight_hash(b.snapshot(s)) for s in states])
def optimizer_hash(states):return digest([b.moment_hash(b.snapshot(s)) for s in states])


def _stop(signum,frame):
    global STOP
    STOP=True


def run(argv=None,pause_after_updates=None,pause_after_stage=None):
    a=parse_args(argv)
    with output_lock(a.output):return _run(a,pause_after_updates,pause_after_stage)


def _run(a,pause_after_updates=None,pause_after_stage=None):
    # Linux-only resource accounting is needed by a training run, not CLI help
    # or the small cross-platform physics/backward verification.
    from resource_tracking import TrainingMeter,hardware_info
    global STOP
    STOP=False
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
    torch.set_num_threads(a.threads);seed_all(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);requested=signature(a)
    resumed=None;stage_plan=plan(a.method,a.smoke)
    if any(p.name!='.run.lock' for p in out.iterdir()):
        if not a.resume:raise FileExistsError('Existing output. Resume the identical configuration or choose a new directory.')
        config=json.loads((out/'config.json').read_text())
        if config['signature']!=requested:raise ValueError('Code/configuration changed. Start a new output directory.')
        if (out/'summary.json').exists():
            print('Matching run already completed; skipped.',flush=True)
            return json.loads((out/'summary.json').read_text())
        resumed=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
        now=hardware_info(device)
        if str(now['torch'])!=config['hardware']['torch'] or now['gpu']!=config['hardware']['gpu']:
            raise ValueError('Exact continuation requires recorded PyTorch version/GPU.')
    else:
        config=dict(format=FORMAT,signature=requested,args=vars(a),plan=stage_plan,
                    source_hashes=source_hashes(),hardware=hardware_info(device),problem_audit=audit(),
                    gate='task-fixed g=0 coordinate branch',
                    stopping='validation physical loss; ten checks without a strict new minimum; reset per stage',
                    final_selection='fixed independent physical LF+BC+IC+lambda_if*IF; no series interior labels',
                    transfer='C01 stage-LAST parameters/Adam; at splits each child copies its own parent',
                    consistency='fixed previous-stage predictions; 1/(epoch+1), removed below 1e-2',
                    spatial_interface='sum adjacent pairs mean points [sum real/imag value jump squared + sum real/imag spatial derivative jump squared]',
                    spatial_interface_excludes=['time-derivative jump','PDE-residual jump','exact interface values'],
                    spatial_part_weighting='volume-weighted PDE/IC/GE; sum actual external-face BC; sum interfaces',
                    gradient_enhancement=dict(operator='full discrete manufactured-source PDE residual',
                        spatial='g_u=-H(v_x)+d_x N_u-f_R,x; g_v=H(u_x)+d_x N_v-f_I,x; independent nodal/sigma coordinate leaves',
                        temporal='complete sigma residual sequence, centered differences inside and one-sided at endpoints, spacing=tau=1/N',
                        weight='w_grad*exp(-3*loss_f.detach()); default w_grad=0.001',
                        baselines='disabled for fpinn and fl1'),
                    time_history='every spatial child keeps all times from t=0; never reset at spatial split',
                    physical_boundary='given Dirichlet endpoint values; no added periodic derivative constraint')
        config=json.loads(json.dumps(config));atomic_json(out/'config.json',config)
        atomic_json(out/'problem_audit.json',audit())
    dtype=torch.float32 if a.network_dtype=='float32' else torch.float64
    val_array=independent_points(a.validation_seed,a.eval_points)
    test_array=independent_points(a.test_seed,a.eval_points)
    val_tx=torch.tensor(val_array,device=device,dtype=dtype)
    if resumed is None:
        np.savez_compressed(out/'evaluation_coordinates.npz',validation=val_array,test=test_array)
        states=[b.create_state(a,device)];domains=stage_plan[0]['domains']
    else:
        states=load_bundle(resumed['current'],device);domains=resumed['current']['domains']
    meter=TrainingMeter(device);rows=[];history=[];teacher=None;best=None;best_score=float('inf')
    global_step=0;point_work=0;first_stage=0;next_epoch=0;current_data=None;context=None
    wall_before=0.;clock=time.perf_counter();new_updates=0
    if resumed is not None:
        rows=resumed['rows'];history=resumed['history'];teacher=unpack_data([resumed['teacher']],device)[0] if resumed['teacher'] is not None else None
        # Teacher only contains tx/uv; it does not have a differentiable sigma grid.
        first_stage=resumed['next_stage'];next_epoch=resumed['next_epoch']
        if resumed['data'] is not None:current_data=unpack_data(resumed['data'],device)
        context=resumed['context'];best=resumed['best'];best_score=resumed['best_score']
        global_step=resumed['global_step'];point_work=resumed['point_work'];wall_before=resumed['wall_seconds']
        m=resumed['meter'];meter.seconds=m['optimization_seconds'];meter.steps=m['updates']
        meter.peak_allocated=m['peak_training_allocated_bytes'];meter.peak_reserved=m['peak_training_reserved_bytes']
        meter.peak_increment=m['peak_step_increment_allocated_bytes'];meter.max_step_seconds=m['max_measured_update_seconds']
        restore_rng(resumed['rng'])
        write_csv(out/'training.csv',rows);atomic_json(out/'stage_history.json',history)
    def elapsed():return wall_before+time.perf_counter()-clock
    def save_resume(sid,epoch,data,ctx):
        model_stage=sid if data is not None else sid-1
        atomic_torch(out/'resume.pt',dict(format=FORMAT,current=snapshot(states,domains,a,model_stage,epoch-1),
                     next_stage=sid,next_epoch=epoch,data=cpu_tree(data),context=ctx,
                     teacher=cpu_tree(teacher),rows=rows,history=history,best=best,best_score=best_score,
                     global_step=global_step,point_work=point_work,meter=meter.report(),wall_seconds=elapsed(),rng=capture_rng()))
    print('%s | stages=%d | alpha=.9 beta=.5 gamma=1 | source=manufactured_series | loss-patience=%d | cap=%s' %
          (a.method,len(stage_plan),a.patience_evals,a.max_epochs_per_stage or 'none'),flush=True)
    for sid in range(first_stage,len(stage_plan)):
        item=stage_plan[sid];M,N=item['M'],item['N']
        continuing=current_data is not None and sid==first_stage
        if continuing:
            datasets=current_data;epoch=next_epoch;current_data=None
        else:
            parent_weights=[b.weight_hash(b.snapshot(s)) for s in states]
            parent_moments=[b.moment_hash(b.snapshot(s)) for s in states]
            if len(states)!=item['parts']:
                states=[b.clone_parent(states[p],a,device) for p in item['parents']]
            domains=item['domains'];b.set_learning_rate(states,item['lr'],sid)
            for i,s in enumerate(states):
                parent=item['parents'][i] if sid>=3 else 0
                assert b.weight_hash(b.snapshot(s))==parent_weights[parent]
                assert b.moment_hash(b.snapshot(s))==parent_moments[parent]
            datasets=[b.build_data(d,M,N,a,device,dtype) for d in domains]
            entry_previous=float(b.teacher_loss(states,domains,teacher).detach())
            if entry_previous>1e-10:raise AssertionError('Parent-to-child function changed at inherited points.')
            context=dict(best_loss=float('inf'),bad=0,start_step=global_step,start_seconds=meter.seconds,
                         entry_hash=model_hash(states),entry_moments=optimizer_hash(states),
                         parent_weight_hashes=parent_weights,parent_moment_hashes=parent_moments,
                         child_parent_ids=item['parents'],entry_previous_loss=entry_previous,
                         stage_start_wall=elapsed())
            epoch=0
            atomic_torch(out/('stage_%02d_entry.pt'%sid),snapshot(states,domains,a,sid,-1))
            save_resume(sid,0,datasets,context)
        print('S%d | phase=%s | models=%d | each intervals=%dx%d | lr=%.3e | interfaces=%s' %
              (sid,item['phase'],len(states),M,N,item['lr'],item['interfaces']),flush=True)
        stopped=False
        while not a.max_epochs_per_stage or epoch<a.max_epochs_per_stage:
            meter.begin()
            loss,values=b.objective(states,domains,datasets,a,epoch,teacher)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss.')
            for s in states:
                for optimizer in s['optimizers']:optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for s in states:
                for optimizer in s['optimizers']:
                    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for group in optimizer.param_groups for p in group['params']):
                        raise FloatingPointError('Nonfinite trainable gradient.')
                    optimizer.step()
            meter.end();global_step+=1;new_updates+=1;point_work+=len(states)*(M-1)*N
            by_cap=bool(a.max_epochs_per_stage and epoch+1==a.max_epochs_per_stage)
            scheduled=epoch%a.eval_every==0 or by_cap
            if scheduled:
                validation=b.physical_validation(states,domains,a,device)
                with torch.no_grad():
                    val_pred=b.predict(states,domains,val_tx).cpu().double().numpy()
                    sigma=b.refresh_teacher(states,domains,datasets)
                    sigma_pred=sigma['uv'].cpu().double().numpy()
                sigma_reference=reference_field(sigma['tx'].cpu().double().numpy())
                exact_l2=metrics(val_pred[:,0]+1j*val_pred[:,1],reference_field(val_array))['relative_l2']
                sigma_l2=metrics(sigma_pred[:,0]+1j*sigma_pred[:,1],sigma_reference)['relative_l2']
                if validation['score']<context['best_loss']:context['best_loss']=validation['score'];context['bad']=0
                else:context['bad']+=1
                saved=snapshot(states,domains,a,sid,epoch,validation)
                stage_path=out/('stage_%02d_best_physical.pt'%sid)
                stage_score=context.get('best_physical',float('inf'))
                if validation['score']<stage_score:
                    context['best_physical']=validation['score'];atomic_torch(stage_path,saved)
                if validation['score']<best_score:
                    best_score=validation['score'];best=saved;atomic_torch(out/'best_physical.pt',best)
                row=dict(stage=sid,epoch=epoch,completed_epochs=epoch+1,global_step=global_step,
                         phase=item['phase'],parts=len(states),M=M,N=N,lr=item['lr'],**values,
                         exact_series_sigma_l2=sigma_l2,exact_validation_series_l2=exact_l2,
                         validation_physical=validation['score'],validation_interface=validation['loss_interface'],
                         no_improvement_count=context['bad'],optimization_seconds=meter.seconds,elapsed_seconds=elapsed(),
                         residual_points_processed=point_work)
                rows.append(row);write_csv(out/'training.csv',rows)
                monitor=out/'monitor_predictions';monitor.mkdir(exist_ok=True)
                cache_dtype=np.float64 if a.network_dtype=='float64' else np.float32
                np.savez_compressed(monitor/('s%d_e%d.npz'%(sid,epoch)),prediction=val_pred.astype(cache_dtype),global_step=np.array(global_step))
                print('S%d epoch %6d | loss %.3e | physical %.3e | series_L2 %.3e (manufactured exact) | IF %.3e | bad %d/%d' %
                      (sid,epoch,values['loss'],validation['score'],exact_l2,values['loss_interface'],context['bad'],a.patience_evals),flush=True)
            epoch+=1
            stopped=scheduled and context['bad']>=a.patience_evals
            if stopped or by_cap:break
            if epoch%a.checkpoint_every==0 or scheduled or STOP:
                save_resume(sid,epoch,datasets,context)
            if STOP or (pause_after_updates is not None and new_updates>=pause_after_updates):
                save_resume(sid,epoch,datasets,context)
                return dict(status='PAUSED',global_step=global_step)
        teacher=sigma if a.method=='go' else None
        reason='loss_patience' if stopped else 'explicit_safety_cap'
        last=snapshot(states,domains,a,sid,epoch-1)
        atomic_torch(out/('stage_%02d_last.pt'%sid),last)
        times=torch.linspace(0,1,a.interface_points,device=device,dtype=dtype)
        _,jumps=b.interface_loss(states,domains,times,True)
        history.append(dict(stage=sid,phase=item['phase'],parts=len(states),M=M,N=N,lr=item['lr'],
                            epochs=epoch,stop_reason=reason,global_step=global_step,
                            interfaces=jumps,parent_ids=context['child_parent_ids'],
                            parent_weight_hashes=context['parent_weight_hashes'],parent_moment_hashes=context['parent_moment_hashes'],
                            entry_weights=context['entry_hash'],entry_moments=context['entry_moments'],
                            final_weights=model_hash(states),final_moments=optimizer_hash(states),
                            entry_previous_loss=context['entry_previous_loss'],
                            unique_ensemble_parameters=sum(p.numel() for p in {id(p):p for s in states for o in s['optimizers'] for g in o.param_groups for p in g['params']}.values()),
                            optimization_seconds=meter.seconds-context['start_seconds'],
                            no_improvement_count=context['bad']))
        atomic_json(out/'stage_history.json',history)
        save_resume(sid+1,0,None,None)
        if STOP or pause_after_stage==sid:return dict(status='PAUSED_BETWEEN_STAGES',global_step=global_step)
    if best is None:raise AssertionError('No physical validation checkpoint selected.')
    atomic_torch(out/'final.pt',best)
    selected=load_bundle(best,device,False)
    test_tx=torch.tensor(test_array,device=device,dtype=dtype)
    with torch.no_grad():uv=b.predict(selected,best['domains'],test_tx).cpu().double().numpy()
    reference=reference_field(test_array)
    exact_metrics=metrics(uv[:,0]+1j*uv[:,1],reference)
    np.savez_compressed(out/'test_predictions.npz',tx=test_array,prediction=uv,reference_field=reference)
    summary=dict(format=FORMAT,status='SMOKE_ONLY' if a.smoke else 'COMPLETED_ACCURACY_UNVERIFIED',
                 method=a.method,seed=a.seed,signature=requested,total_updates=global_step,
                 stages=history,selected_stage=best['stage'],selected_epoch=best['epoch'],
                 best_validation_physical=best_score,exact_series_test_metrics=exact_metrics,
                 reference_is_analytic=True,source_model=SOURCE_MODEL,residual_points_processed=point_work,
                 unique_terminal_ensemble_parameters=sum(p.numel() for p in {id(p):p for s in states for o in s['optimizers'] for g in o.param_groups for p in g['params']}.values()),
                 unique_selected_ensemble_parameters=sum(p.numel() for p in {id(p):p for s in selected for o in s['optimizers'] for g in o.param_groups for p in g['params']}.values()),
                 resources=meter.report(),wall_seconds=elapsed(),problem_audit=audit(),
                 reference_role='ANALYTIC_MANUFACTURED_SERIES')
    atomic_json(out/'summary.json',summary)
    print('Completed '+a.method+'; selected by physical validation. Errors use the analytic manufactured reference.',flush=True)
    return summary


def main():
    signal.signal(signal.SIGINT,_stop);signal.signal(signal.SIGTERM,_stop)
    return run()


if __name__=='__main__':main()
