"Train GO-fPINNs with temporal refinement followed by time-window decomposition."
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
import models
import physics
from common import (atomic_json,atomic_torch,capture_rng,cpu_tree,digest,metrics,
                    output_lock,restore_rng,seed_all,write_csv)
from problem import exact_numpy,independent_points
from protocol import plan,ALPHA,T
from resource_tracking import TrainingMeter,hardware_info

FORMAT='C03_C01_full_history_time_windows_v1'
STOP=False
SOURCE_FILES=['train_longtime.py','models.py','physics.py','history.py','problem.py','protocol.py',
              'common.py','resource_tracking.py','c01_source/baseline_original.py',
              'c01_source/weight_schemes.py','c01_source/direct_reference.py']


def parse_args(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    q.add_argument('--seed',type=int,default=1234)
    q.add_argument('--output',default=None)
    q.add_argument('--width',type=int,default=60)
    q.add_argument('--extra-hidden-layers',type=int,default=2,help='One input hidden block plus two additional hidden blocks.')
    q.add_argument('--network-dtype',choices=['float64','float32'],default='float64')
    q.add_argument('--shared-update',choices=['legacy_double','once'],default='legacy_double')
    q.add_argument('--history-mode',choices=['fft','stream','direct'],default='fft')
    q.add_argument('--soe-eps',type=float,default=1e-10)
    q.add_argument('--potential-sign',type=int,choices=[1,-1],default=1)
    q.add_argument('--w-grad',type=float,default=.001)
    q.add_argument('--noise-amplitude',type=float,default=.05,help='Loss-weight perturbation; no observation noise.')
    q.add_argument('--weight-lower-bound',type=float,default=.2,help='Explicit positive floor prevents learned weights from discarding PDE or boundaries.')
    q.add_argument('--continuity-weight',type=float,default=1.)
    q.add_argument('--interface-points',type=int,default=51)
    q.add_argument('--boundary-points',type=int,default=128,help='Points on EACH physical spatial boundary in the active window.')
    q.add_argument('--spatial-batch',type=int,default=None,help='Auto: 7 on CUDA, 1 on CPU. 0 = all x; every batch retains FULL history.')
    q.add_argument('--eval-every',type=int,default=100)
    q.add_argument('--patience-evals',type=int,default=10)
    q.add_argument('--max-epochs-per-window',type=int,default=5000,help='Per active model/window cap; 0 disables this cap. Global stage has one window.')
    q.add_argument('--checkpoint-every',type=int,default=100)
    q.add_argument('--eval-points',type=int,default=4096)
    q.add_argument('--validation-seed',type=int,default=98401)
    q.add_argument('--test-seed',type=int,default=98402)
    q.add_argument('--threads',type=int,default=1)
    q.add_argument('--resume',action='store_true')
    q.add_argument('--smoke',action='store_true')
    a=q.parse_args(argv)
    if a.spatial_batch is None:a.spatial_batch=7 if str(a.device).startswith('cuda') else 1
    if min(a.width,a.eval_every,a.patience_evals,a.checkpoint_every,a.eval_points,a.threads)<1:q.error('Counts must be positive.')
    if min(a.interface_points,a.boundary_points)<2:q.error('At least two condition points required.')
    if min(a.extra_hidden_layers,a.spatial_batch,a.max_epochs_per_window,a.seed,a.validation_seed,a.test_seed)<0:q.error('Negative count.')
    if a.validation_seed==a.test_seed:q.error('Validation and test seeds must differ.')
    if not all(np.isfinite(v) for v in [a.w_grad,a.soe_eps,a.noise_amplitude,a.weight_lower_bound,a.continuity_weight]):q.error('Nonfinite numerical setting.')
    if min(a.w_grad,a.noise_amplitude)<0 or min(a.weight_lower_bound,a.continuity_weight)<=0 or not 0<a.soe_eps<1:q.error('Invalid numerical setting.')
    if a.smoke:
        a.width=8;a.extra_hidden_layers=1;a.max_epochs_per_window=2;a.eval_every=1
        a.checkpoint_every=1;a.eval_points=64;a.boundary_points=8;a.interface_points=5;a.spatial_batch=2
    if a.output is None:a.output='runs/C03/seed_'+str(a.seed)
    return a


def source_hashes():
    root=Path(__file__).parent
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in SOURCE_FILES}


def signature(a):
    values=vars(a).copy()
    for n in ['resume','output']:values.pop(n,None)
    return digest(dict(args=values,sources=source_hashes()))


def snapshot(states,windows,a,stage,window,epoch,validation=None):
    return dict(format=FORMAT,args=vars(a),stage=stage,window=window,epoch=epoch,
                windows=windows,states=[models.snapshot(s) for s in states],validation=validation)


def load_bundle(saved,device,optimizers=True):
    if saved.get('format')!=FORMAT:raise ValueError('Not a C03 full-history checkpoint.')
    args=argparse.Namespace(**saved['args']);states=[]
    for row in saved['states']:
        s=models.create(args,device);models.restore(s,row,optimizers);states.append(s)
    return states


def load_for_prediction(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    return load_bundle(saved,torch.device(device),False),saved


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device).clone()
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device) for v in value]
    return value


def prediction_numpy(states,windows,tx,batch=8192):
    out=[];s=states[0]
    with torch.no_grad():
        for start in range(0,len(tx),batch):
            t=torch.as_tensor(tx[start:start+batch],device=s['device'],dtype=s['dtype'])
            out.append(models.predict(states,windows,t).cpu().double().numpy())
    return np.concatenate(out)


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
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);sig=signature(a);items=plan(a.smoke)
    config_path=out/'config.json';resumed=None
    if config_path.exists():
        if not a.resume:raise FileExistsError('Output exists. Use --resume or a new output.')
        config=json.loads(config_path.read_text())
        if config['signature']!=sig:raise ValueError('Code/configuration differs; use a new output directory.')
        if (out/'summary.json').exists():
            print('Matching run already completed.',flush=True);return json.loads((out/'summary.json').read_text())
        resumed=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
        now=hardware_info(device)
        if now['gpu']!=config['hardware']['gpu'] or str(now['torch'])!=config['hardware']['torch']:
            raise ValueError('Exact continuation requires the recorded GPU and PyTorch version.')
    else:
        if any(p.name!='.run.lock' for p in out.iterdir()):raise FileExistsError('Nonempty output without matching config.')
        hw=hardware_info(device);hw['network_dtype']=a.network_dtype
        config=dict(format=FORMAT,signature=sig,args=vars(a),plan=items,source_hashes=source_hashes(),hardware=hw,
                    equation='i D_C^.9 psi + psi_xx + |psi|^2 psi + potential_sign*x*psi = f',
                    gate='task-configured g=0 coordinate branch',source='analytic manufactured forcing; no interior labels in loss',
                    representation='psi=100*N(2*t/100-1,2*x-1), common global mapping for all windows',
                    history='lower limit 0; frozen predictions in earlier current-stage windows plus differentiable current-window predictions',
                    history_mode=a.history_mode,soe_horizon=100.,
                    conditioning='PDE/GE scale 100*(2+t^2+abs(x)); IC/BC/continuity/transfer scale 100^2',
                    stopping='validation physical loss; ten checks without a strict historical minimum; cap per active window',
                    selection='minimum relative L2 on common independent validation coordinates; test never selects or stops',
                    transfer='parent stage-LAST model + Adam; earlier windows frozen after chronological optimization',
                    consistency='fixed previous-stage predictions; 1/(epoch+1), removed below 1e-2',
                    continuity='value only, current left endpoint vs detached optimized previous right endpoint; no derivative/residual jump')
        atomic_json(config_path,config)
    validation=independent_points(a.validation_seed,a.eval_points);test=independent_points(a.test_seed,a.eval_points)
    target=exact_numpy(validation)
    if resumed is None:
        np.savez_compressed(out/'evaluation_coordinates.npz',validation=validation,test=test)
        states=[models.create(a,device)];windows=items[0]['windows']
    else:
        states=load_bundle(resumed['current'],device);windows=resumed['current']['windows']
    meter=TrainingMeter(device);clock=time.perf_counter();wall_before=0.
    rows=[];stage_history=[];window_history=[];sid0=wid0=epoch0=global_step=point_work=0
    stage_ctx=window_ctx=data=teacher=best=None;best_score=float('inf');new_updates=0
    if resumed is not None:
        rows=resumed['rows'];stage_history=resumed['stage_history'];window_history=resumed['window_history']
        sid0=resumed['next_stage'];wid0=resumed['next_window'];epoch0=resumed['next_epoch']
        stage_ctx=resumed['stage_ctx'];window_ctx=resumed['window_ctx']
        data=move(resumed['data'],device);teacher=move(resumed['teacher'],device)
        global_step=resumed['global_step'];point_work=resumed['point_work'];best=resumed['best'];best_score=resumed['best_score']
        wall_before=resumed['wall_seconds'];m=resumed['meter'];meter.seconds=m['optimization_seconds'];meter.steps=m['updates']
        meter.peak_allocated=m['peak_training_allocated_bytes'];meter.peak_reserved=m['peak_training_reserved_bytes']
        meter.peak_increment=m['peak_step_increment_allocated_bytes'];meter.max_step_seconds=m['max_measured_update_seconds']
        restore_rng(resumed['rng'])
    def elapsed():return wall_before+time.perf_counter()-clock
    def save_resume(sid,wid,epoch):
        atomic_torch(out/'resume.pt',dict(format=FORMAT,current=snapshot(states,windows,a,sid,wid,epoch-1),
                     next_stage=sid,next_window=wid,next_epoch=epoch,stage_ctx=stage_ctx,window_ctx=window_ctx,
                     data=cpu_tree(data),teacher=cpu_tree(teacher),rows=rows,stage_history=stage_history,
                     window_history=window_history,global_step=global_step,point_work=point_work,best=best,best_score=best_score,
                     meter=meter.report(),wall_seconds=elapsed(),rng=capture_rng()))
    print('GO-fPINNs | T=100 | 9 stages | full Caputo history | cap/window=%s' % (a.max_epochs_per_window or 'none'),flush=True)
    for sid in range(sid0,len(items)):
        item=items[sid]
        if stage_ctx is None:
            parents=[models.hashes(s) for s in states]
            if item['parts']!=len(states):states=[models.clone(states[p],a,device) for p in item['parents']]
            windows=item['windows'];models.set_lr(states,item['lr'],sid)
            entry=[models.hashes(s) for s in states]
            for j,h in enumerate(entry):assert h==parents[item['parents'][j]]
            stage_ctx=dict(start_seconds=meter.seconds,start_step=global_step,parents=parents,entry=entry)
            wid0=0;epoch0=0
            atomic_torch(out/('stage_%02d_entry.pt'%sid),snapshot(states,windows,a,sid,0,-1))
            save_resume(sid,0,0)
        print('S%d | windows=%d | each grid=%dx%d intervals | lr=%.3e' % (sid,len(states),item['M'],item['N'],item['lr']),flush=True)
        for wid in range(wid0,len(states)):
            state=states[wid]
            if data is None:
                data,teacher=physics.build_data(states,item,wid,a,device,items[sid-1] if sid else None)
                window_ctx=dict(best_loss=float('inf'),bad=0,start_step=global_step,start_seconds=meter.seconds,
                                entry=models.hashes(state),prefix_hash=digest(data['prefix']),
                                frozen_past=[models.hashes(s) for s in states[:wid]])
                epoch0=0;save_resume(sid,wid,0)
            epoch=epoch0;stagnated=False
            validation_data,_=physics.build_data(states,item,wid,a,device,validation=True)
            while not a.max_epochs_per_window or epoch<a.max_epochs_per_window:
                meter.begin()
                for opt in state['optimizers']:opt.zero_grad(set_to_none=True)
                values=physics.gradients(state,data,a,epoch,teacher)
                for opt in state['optimizers']:
                    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for g in opt.param_groups for p in g['params']):
                        raise FloatingPointError('Nonfinite trainable gradient.')
                    opt.step()
                meter.end();global_step+=1;new_updates+=1;point_work+=(item['M']-1)*item['N']
                by_cap=bool(a.max_epochs_per_window and epoch+1==a.max_epochs_per_window)
                check=epoch%a.eval_every==0 or by_cap
                if check:
                    pred=prediction_numpy(states,windows,validation)
                    val=metrics(pred[:,0]+1j*pred[:,1],target)['relative_l2']
                    validation_loss=physics.validation_loss(state,validation_data,a)
                    values['validation_loss']=validation_loss
                    if validation_loss<window_ctx['best_loss']:window_ctx['best_loss']=validation_loss;window_ctx['bad']=0
                    else:window_ctx['bad']+=1
                    saved=snapshot(states,windows,a,sid,wid,epoch,val)
                    if val<stage_ctx.get('best_validation',float('inf')):
                        stage_ctx['best_validation']=val;atomic_torch(out/('stage_%02d_best.pt'%sid),saved)
                    if val<best_score:
                        best_score=val;best=saved;atomic_torch(out/'best_validation.pt',best)
                    row=dict(stage=sid,window=wid,windows=len(states),epoch=epoch,completed_epochs=epoch+1,
                             global_step=global_step,M=item['M'],N=item['N'],t_left=data['lo'],t_right=data['hi'],
                             lr=item['lr'],**values,validation_relative_l2=val,bad_checks=window_ctx['bad'],
                             optimization_seconds=meter.seconds,elapsed_seconds=elapsed(),residual_points_processed=point_work)
                    rows.append(row);write_csv(out/'training.csv',rows)
                    print('S%d W%02d epoch %5d | loss %.3e | val L2 %.3e | cont %.3e | bad %d/%d' %
                          (sid,wid,epoch,values['loss'],val,values['loss_cont'],window_ctx['bad'],a.patience_evals),flush=True)
                epoch+=1;stagnated=check and window_ctx['bad']>=a.patience_evals
                if stagnated or by_cap:break
                if check or epoch%a.checkpoint_every==0 or STOP:save_resume(sid,wid,epoch)
                if STOP or (pause_after_updates is not None and new_updates>=pause_after_updates):
                    save_resume(sid,wid,epoch);return dict(status='PAUSED',global_step=global_step)
            if [models.hashes(s) for s in states[:wid]]!=window_ctx['frozen_past']:raise AssertionError('A previous optimized window was updated.')
            if digest(data['prefix'])!=window_ctx['prefix_hash']:raise AssertionError('Frozen memory prefix mutated.')
            window_history.append(dict(stage=sid,window=wid,t_left=data['lo'],t_right=data['hi'],N=item['N'],
                                       epochs=epoch,stop_reason='loss_patience' if stagnated else 'explicit_window_cap',
                                       bad_checks=window_ctx['bad'],entry=window_ctx['entry'],last=models.hashes(state),
                                       prefix_hash=window_ctx['prefix_hash'],past_models=window_ctx['frozen_past'],
                                       optimization_seconds=meter.seconds-window_ctx['start_seconds'],global_step=global_step))
            atomic_json(out/'window_history.json',window_history)
            data=teacher=window_ctx=None;epoch0=0;save_resume(sid,wid+1,0)
            if STOP or (pause_after_updates is not None and new_updates>=pause_after_updates):
                return dict(status='PAUSED_BETWEEN_WINDOWS',global_step=global_step)
        last_epoch=window_history[-1]['epochs']-1
        last=snapshot(states,windows,a,sid,len(states)-1,last_epoch)
        atomic_torch(out/('stage_%02d_last.pt'%sid),last)
        pred=prediction_numpy(states,windows,validation)
        val=metrics(pred[:,0]+1j*pred[:,1],target)['relative_l2']
        stage_history.append(dict(**item,global_step=global_step,validation_relative_l2=val,
                                  optimization_seconds=meter.seconds-stage_ctx['start_seconds'],cumulative_optimization_seconds=meter.seconds,
                                  parent_hashes=stage_ctx['parents'],entry_hashes=stage_ctx['entry'],last_hashes=[models.hashes(s) for s in states],
                                  continuity=physics.continuity_report(states,windows,a.interface_points),
                                  trainable_parameters=models.parameter_count(states)))
        atomic_json(out/'stage_history.json',stage_history)
        stage_ctx=None;wid0=0;save_resume(sid+1,0,0)
        if STOP or pause_after_stage==sid:return dict(status='PAUSED_BETWEEN_STAGES',global_step=global_step)
    if best is None:raise AssertionError('No validation checkpoint.')
    atomic_torch(out/'final.pt',best)
    selected=load_bundle(best,device,False);uv=prediction_numpy(selected,best['windows'],test)
    test_metrics=metrics(uv[:,0]+1j*uv[:,1],exact_numpy(test))
    summary=dict(format=FORMAT,status='SMOKE_ONLY' if a.smoke else 'COMPLETED',seed=a.seed,total_updates=global_step,
                 selected_stage=best['stage'],selected_window=best['window'],selected_epoch=best['epoch'],
                 best_validation_relative_l2=best_score,test_metrics=test_metrics,
                 target_relative_l2=1e-4,target_reached_on_independent_test=test_metrics['relative_l2']<=1e-4,
                 resources=meter.report(),wall_seconds=elapsed(),stages=stage_history,
                 residual_points_processed=point_work,accuracy_statement='Measured values only; no guarantee of matching supplied manuscript plots.')
    np.savez_compressed(out/'test_predictions.npz',tx=test,prediction=uv,reference=exact_numpy(test))
    atomic_json(out/'summary.json',summary)
    print('Completed. Test L2=%.6e; target reached=%s' % (test_metrics['relative_l2'],summary['target_reached_on_independent_test']),flush=True)
    return summary


def main():
    signal.signal(signal.SIGINT,_stop);signal.signal(signal.SIGTERM,_stop)
    return run()


if __name__=='__main__':main()
