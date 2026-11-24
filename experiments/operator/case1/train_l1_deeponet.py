"""Optional L1-DeepONet reconstruction; never labelled as original-method code."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import operator_alpha_core as core
import training_support as s
import single_support as p
from validation_support import ValidationLossStop
from l1_deeponet import L1DeepONet, build_l1_dataset, physics_terms, validate_physics
from resource_tracking import TrainingMeter, hardware_info

FORMAT = 'Case1_L1_DeepONet_reimplementation_v1'


def parse_args(argv=None):
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument('--operator', choices=list(p.ACTIVE), default='alpha')
    q.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    q.add_argument('--seed', type=int, default=1234)
    q.add_argument('--split-seed', type=int, default=20261017)
    q.add_argument('--output', default=None)
    q.add_argument('--epochs-per-stage', type=int, default=5000)
    q.add_argument('--initial-lr', type=float, default=1e-3)
    q.add_argument('--grid', type=int, default=9, help='Intervals: 9 means 10^3 nodes.')
    q.add_argument('--boundary-points', type=int, default=150)
    q.add_argument('--width', type=int, default=60)
    q.add_argument('--hidden-layers', type=int, default=5)
    q.add_argument('--latent-dim', type=int, default=64)
    q.add_argument('--eval-every', type=int, default=100)
    q.add_argument('--patience-evals', type=int, default=10)
    q.add_argument('--eval-points', type=int, default=4096)
    q.add_argument('--validation-count', type=int, default=6)
    q.add_argument('--test-count', type=int, default=30)
    q.add_argument('--threads', type=int, default=1)
    q.add_argument('--resume', action='store_true')
    q.add_argument('--smoke', action='store_true')
    a = q.parse_args(argv)
    if min(a.seed, a.split_seed) < 0 or not 1 <= a.epochs_per_stage <= 5000:
        q.error('Nonnegative seeds and a stage cap in [1,5000] required.')
    if min(a.grid, a.boundary_points, a.width, a.hidden_layers, a.latent_dim,
           a.eval_every, a.patience_evals, a.eval_points, a.validation_count, a.test_count, a.threads) < 1 or a.grid < 2:
        q.error('Positive counts and grid >=2 required.')
    if not np.isfinite(a.initial_lr) or a.initial_lr <= 0:
        q.error('Positive finite learning rate required.')
    if a.smoke:
        a.epochs_per_stage=4; a.grid=4; a.boundary_points=16
        a.width=8; a.hidden_layers=2; a.latent_dim=4
        a.eval_every=1; a.eval_points=32; a.validation_count=3; a.test_count=5
    if a.output is None:
        a.output='runs/Case1/operators/%s/l1_reimplementation/seed_%d' % (a.operator, a.seed)
    return a


def create_model(a, device):
    return L1DeepONet(a.width, a.hidden_layers, a.latent_dim).to(device)


def load_for_prediction(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != FORMAT:
        raise ValueError('Not an L1-DeepONet reconstruction checkpoint.')
    model = create_model(argparse.Namespace(**saved['config']['args']), torch.device(device))
    model.load_state_dict(saved['model']); model.eval()
    return model, saved


def run(argv=None, pause_after_new_updates=None):
    a = parse_args(argv)
    with p.output_lock(a.output):
        return _run(a, pause_after_new_updates)


def _run(a, pause_after_new_updates=None):
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    torch.set_num_threads(a.threads);s.seed_all(a.seed);core.device=device;core.SEED=a.seed
    out=Path(a.output);out.mkdir(exist_ok=True, parents=True)
    source_names=['train_l1_deeponet.py','l1_deeponet.py','single_support.py',
                  'training_support.py','operator_alpha_core.py','validation_support.py','resource_tracking.py']
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in source_names}
    args=vars(a).copy();args.pop('resume');args.pop('output')
    signature=hashlib.sha256(json.dumps({'args':args,'sources':hashes}, sort_keys=True).encode()).hexdigest()
    checkpoint=None
    if any(f.name!='.run.lock' for f in out.iterdir()):
        if not a.resume:raise FileExistsError('Choose a new output or resume the identical configuration.')
        config=json.loads((out/'config.json').read_text())
        if config['run_signature']!=signature:raise ValueError('Resume code/configuration changed.')
        if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
        checkpoint=torch.load(out/'resume.pt', map_location='cpu', weights_only=True)
    else:
        config={'format':FORMAT,'args':vars(a),'run_signature':signature,'source_hashes':hashes,
                'method':'L1-DeepONet reconstruction; original source unavailable',
                'original_method_reproduction_verified':False,'hardware':hardware_info(device),
                'single_stage':True,'training_parameter_count':4 if a.smoke else 6,
                'learning_rate':a.initial_lr,'early_stopping':True,'maximum_epochs':a.epochs_per_stage,
                'validation_monitor':'LF+LBC+LIC, unweighted, clean independent validation',
                'checkpoint_selection':'independent validation macro relative L2',
                'data_noise':0.,'training_grid_nodes_per_axis':a.grid+1}
        config=json.loads(json.dumps(config))
        s.atomic_json(out/'config.json',config)
    # Same design as GO, including coordinates and all split IDs.
    counts=[2,3,4] if a.smoke else [4,5,6]
    grids=[2,3,4] if a.smoke else [5,7,9]
    design=p.make_design(a.split_seed,counts,a.validation_count,a.test_count,a.eval_points,grids,a.operator)
    coordinates={role:p.coordinate_array(design,role) for role in ['validation','test','training_diagnostic']}
    if checkpoint is not None:
        if json.loads((out/'parameter_split.json').read_text())!=design:raise ValueError('Parameter design changed.')
        if config['hardware']['torch']!=hardware_info(device)['torch'] or config['hardware']['gpu']!=hardware_info(device)['gpu']:
            raise ValueError('Resume requires the same PyTorch version and GPU.')
    else:p.save_design(out,design,coordinates)
    model=create_model(a,device);optimizer=torch.optim.Adam(model.parameters(),lr=a.initial_lr)
    train=[build_l1_dataset(row,a.grid,a.boundary_points) for row in design['training']]
    old_seed=core.SEED;core.SEED=a.split_seed+700001
    try:val=[build_l1_dataset(row,a.grid,a.boundary_points) for row in design['validation']]
    finally:core.SEED=old_seed
    val_data=p.make_evaluation(design['validation'],torch.tensor(coordinates['validation'],device=device))
    meter=TrainingMeter(device);rows=[];epoch=0;point_work=0;new_updates=0;clock=time.perf_counter()
    def capture(completed, metrics):
        return {'format':FORMAT,'config':config,'epoch':completed,'metrics':metrics,
                'model':s.tree_to_cpu(model.state_dict()),'optimizer':s.tree_to_cpu(optimizer.state_dict()),'rng':s.capture_rng()}
    if checkpoint is None:
        initial,_=p.evaluate(model,val_data)
        initial_loss=validate_physics(model,val)['validation_loss']
        stopper=ValidationLossStop(a.patience_evals,best=initial_loss)
        best=capture(0,initial)
    else:
        model.load_state_dict(checkpoint['current']['model']);optimizer.load_state_dict(checkpoint['current']['optimizer'])
        s.restore_rng(checkpoint['current']['rng']);epoch=checkpoint['next_epoch'];rows=checkpoint['rows'];point_work=checkpoint['point_work']
        best=checkpoint['best'];stopper=ValidationLossStop(**checkpoint['stopper'])
        m=checkpoint['meter'];meter.seconds=m['optimization_seconds'];meter.steps=m['updates']
        meter.peak_allocated=m['peak_training_allocated_bytes'];meter.peak_reserved=m['peak_training_reserved_bytes']
        meter.peak_increment=m['peak_step_increment_allocated_bytes'];meter.max_step_seconds=m['max_measured_update_seconds']
    def save_resume():
        s.atomic_checkpoint(out/'resume.pt',{'format':FORMAT,'current':capture(epoch,{}),'best':best,
                           'next_epoch':epoch,'rows':rows,'stopper':stopper.state_dict(),
                           'point_work':point_work,'meter':meter.report()})
    save_resume()
    while epoch<a.epochs_per_stage and stopper.count<a.patience_evals:
        model.train();meter.begin();optimizer.zero_grad(set_to_none=True);loss_total=0.
        for data in train:
            terms=physics_terms(model,data);loss=terms['validation_loss']
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite L1 training loss.')
            (loss/len(train)).backward();loss_total+=float(loss.detach())/len(train)
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        optimizer.step();meter.end();epoch+=1;new_updates+=1;point_work+=len(train)*(a.grid-1)**2*a.grid
        if epoch%a.eval_every==0 or epoch==a.epochs_per_stage:
            metrics,_=p.evaluate(model,val_data);vloss=validate_physics(model,val);stopper.update(vloss['validation_loss'])
            if metrics['relative_l2']<best['metrics']['relative_l2']:best=capture(epoch,metrics)
            rows.append({'epoch':epoch,'training_loss':loss_total,**vloss,
                         'validation_l2':metrics['relative_l2'],'no_improvement_count':stopper.count,'lr':a.initial_lr})
            s.write_csv(out/'training.csv',rows);save_resume()
            print('L1 reconstruction %s | epoch %d | val_L2 %.4e | val_loss %.4e | no_improve %d/%d' %
                  (a.operator,epoch,metrics['relative_l2'],vloss['validation_loss'],stopper.count,a.patience_evals),flush=True)
        if pause_after_new_updates is not None and new_updates>=pause_after_new_updates:
            save_resume();return {'status':'PAUSED_FOR_FUNCTIONAL_VERIFICATION','total_updates':epoch}
    s.atomic_checkpoint(out/'last.pt',capture(epoch,{}));s.atomic_checkpoint(out/'final.pt',best)
    model.load_state_dict(best['model'])
    # Test solutions only constructed AFTER selection has fixed final.pt.
    test_data=p.make_evaluation(design['test'],torch.tensor(coordinates['test'],device=device))
    metrics,per_parameter=p.evaluate(model,test_data);s.write_csv(out/'test_per_parameter.csv',per_parameter)
    summary={'format':FORMAT,'operator':a.operator,'method':'l1_reimplementation',
             'status':'SMOKE_ONLY' if a.smoke else 'TRAINING_COMPLETED_ACCURACY_UNVERIFIED',
             'run_signature':signature,'design_sha256':design['design_sha256'],
             'total_updates':epoch,'selected_epoch':best['epoch'],
             'stop_reason':'validation_loss_no_improvement' if stopper.count>=a.patience_evals else 'maximum_stage_updates',
             'test_parameter_count':len(per_parameter),'test_macro_over_parameters':metrics,
             'validation_best':best['metrics']['relative_l2'],'resources':meter.report(),
             'residual_point_evaluations':point_work,'parameter_count':sum(x.numel() for x in model.parameters()),
             'wall_seconds_this_invocation':time.perf_counter()-clock,
             'original_method_reproduction_verified':False}
    s.atomic_json(out/'summary.json',summary);s.atomic_json(out/'resources.json',meter.report());return summary


if __name__=='__main__':run()
