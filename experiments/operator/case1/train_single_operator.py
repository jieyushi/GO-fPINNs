"""Case 1 single alpha, beta or gamma operator, paper protocol.
Early stopping in S0--S2, at most 5000 epochs per stage.
Fixed stage rates: 1e-3, 5e-4, 2.5e-4. No warmup or plateau scheduler.
Keep the original physics, Softmax-noise weights, local GE and stage transfer.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import operator_alpha_core as core
import training_support as s
import single_support as j
import validation_support as validation
from resource_tracking import TrainingMeter,hardware_info,history_inventory,sync

FORMAT='Case1_single_operator_earlystop_20261017_v1'
LEGACY_FORMAT=FORMAT


def three_counts(text,minimum=1):
    values=[int(x) for x in text.split(',')]
    if len(values)!=3 or min(values)<minimum:raise ValueError('Exactly three valid positive counts required.')
    return values


def parse_args(argv=None,forced_mode=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',choices=['alpha','beta','gamma'],default='alpha',
        help='Only this paper parameter varies; the other two stay fixed.')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--output',default=None)
    p.add_argument('--budget-mode',choices=['early_stop'],default='early_stop')
    p.add_argument('--epochs-per-stage',type=int,default=5000,
        help='Maximum epochs in each stage; must be <=5000. Not a fixed update budget.')
    p.add_argument('--stage-epochs',default=None)
    p.add_argument('--grids',default='5,7,9',
        help='Intervals per axis at S0/S1/S2. 5,7,9 gives full grids of 6^3,8^3,10^3 points including endpoints.')
    p.add_argument('--boundary-points',default='80,120,150')
    p.add_argument('--split-seed',type=int,default=20261017,
        help='Independent validation/test streams; fixed across training seeds and methods.')
    p.add_argument('--parameter-counts',default='4,5,6',help='Nested active parameter values.')
    p.add_argument('--validation-count',type=int,default=6)
    p.add_argument('--test-count',type=int,default=30)
    p.add_argument('--width',type=int,default=60)
    p.add_argument('--hidden-layers',type=int,default=5)
    p.add_argument('--latent-dim',type=int,default=16)
    p.add_argument('--initial-lr',type=float,default=1e-3)
    p.add_argument('--stage-lr-factors',default='1,0.5,0.25',
        help='Three fixed stage LR multipliers; paper default gives 0.001,0.0005,0.00025.')
    p.add_argument('--w-grad',type=float,default=1e-4)
    p.add_argument('--noise-amplitude',type=float,default=.05)
    p.add_argument('--weight-lower-bound',type=float,default=1e-6)
    p.add_argument('--warmup-epochs',type=int,default=0,help='Must be zero for the stated stage learning rates.')
    p.add_argument('--warmup-start-factor',type=float,default=.02)
    p.add_argument('--consistency-initial',type=float,default=.1)
    p.add_argument('--consistency-decay',type=float,default=300.)
    p.add_argument('--consistency-cutoff',type=float,default=.01)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--patience-evals',type=int,default=10,
        help='Consecutive checks without a strictly lower best validation physical loss. No relative tolerance.')
    p.add_argument('--eval-points',type=int,default=4096)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--resume',action='store_true',help='Restore the identical Case 1 configuration, Adam, teacher, stopping counter and RNG.')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(argv)
    if forced_mode is not None and a.budget_mode!=forced_mode:
        p.error('This entry point requires --budget-mode '+forced_mode)
    if a.output is None:a.output='runs/Case1/operators/'+a.operator+'/go/seed_'+str(a.seed)
    try:
        a.grids=three_counts(a.grids,2)
        if not a.grids[0]<a.grids[1]<a.grids[2]:raise ValueError('Grids must increase.')
        a.boundary_points=three_counts(a.boundary_points)
        a.stage_epochs=three_counts(a.stage_epochs) if a.stage_epochs else [a.epochs_per_stage]*3
        a.parameter_counts=three_counts(a.parameter_counts)
        a.stage_lr_factors=[float(v) for v in a.stage_lr_factors.split(',')]
        if len(a.stage_lr_factors)!=3 or not all(math.isfinite(v) and v>0 for v in a.stage_lr_factors):
            raise ValueError('Three positive stage LR factors required.')
        if not a.parameter_counts[0]<a.parameter_counts[1]<a.parameter_counts[2]:
            raise ValueError('Training parameter counts must strictly increase.')
    except ValueError as error:p.error(str(error))
    if min(a.epochs_per_stage,a.width,a.hidden_layers,a.latent_dim,a.eval_every,a.eval_points,a.threads,a.validation_count,a.test_count,a.patience_evals)<1:p.error('Counts must be positive.')
    if max(a.stage_epochs)>5000 or a.epochs_per_stage>5000:p.error('Every stage is capped at 5000 epochs.')
    if min(a.seed,a.split_seed)<0:p.error('Seeds must be nonnegative.')
    if a.warmup_epochs!=0:p.error('Paper stage rates require --warmup-epochs 0.')
    positive=[a.initial_lr,a.consistency_initial,a.consistency_decay,a.consistency_cutoff]
    if not all(math.isfinite(x) and x>0 for x in positive):p.error('LR and consistency settings must be finite and positive.')
    if not math.isfinite(a.noise_amplitude) or a.noise_amplitude<0:p.error('Invalid noise amplitude.')
    if not math.isfinite(a.weight_lower_bound) or a.weight_lower_bound<=0:p.error('Invalid weight lower bound.')
    if not 0<a.warmup_start_factor<=1:p.error('Warmup start factor must lie in (0,1].')
    if not math.isfinite(a.w_grad) or a.w_grad<0:p.error('Invalid gradient-enhancement weight.')
    if a.consistency_initial<a.consistency_cutoff:p.error('Initial consistency weight must be >= cutoff.')
    if a.smoke:
        a.stage_epochs=[4,4,4];a.epochs_per_stage=4;a.grids=[2,3,4];a.boundary_points=[8,12,16]
        a.width=8;a.hidden_layers=2;a.latent_dim=4;a.eval_every=1;a.eval_points=32
        a.consistency_decay=.7
        a.parameter_counts=[2,3,4];a.validation_count=3;a.test_count=5
    return a


def source_hashes():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ['train_single_operator.py','single_support.py','training_support.py','validation_support.py','operator_alpha_core.py','resource_tracking.py']}


def signature(args):
    values=vars(args).copy();values.pop('resume',None);values.pop('output',None)
    return hashlib.sha256(json.dumps({'args':values,'sources':source_hashes()},sort_keys=True).encode()).hexdigest()


def create_objects(a,device):
    a1=torch.tensor([.2],device=device,requires_grad=True);a2=torch.tensor([.2],device=device,requires_grad=True)
    fn=torch.tensor([5.],device=device)
    model=core.OperatorEnhancedFPINN(latent_dim=a.latent_dim,hidden_size=a.width,
        num_hidden_layers=a.hidden_layers,a_1=a1,a_2=a2,fn=fn).to(device)
    hyper=s.StagewiseHyperNet(a.noise_amplitude,a.weight_lower_bound).to(device);core.hypernet=hyper
    predictor=list(model.parameters())+[a1,a2]
    trainables=list(model.parameters())+list(hyper.parameters())+[a1,a2]
    optimizer=torch.optim.Adam(trainables,lr=a.initial_lr)
    return {'model':model,'hyper':hyper,'a1':a1,'a2':a2,'fn':fn,
            'predictor':predictor,'trainables':trainables,'optimizer':optimizer}


def capture(state,config,stage,epoch,metrics):
    return {'format':FORMAT,'config':config,'stage':stage,'epoch':epoch,'metrics':metrics,
        'weighting_stage':state['hyper'].training_stage,'weight_mode':state['hyper'].weight_mode,
        'model':s.tree_to_cpu(state['model'].state_dict()),'hypernet':s.tree_to_cpu(state['hyper'].state_dict()),
        'a_1':s.tree_to_cpu(state['a1']),'a_2':s.tree_to_cpu(state['a2']),'fn':s.tree_to_cpu(state['fn']),
        'optimizer':s.tree_to_cpu(state['optimizer'].state_dict()),'rng':s.capture_rng()}


def restore(state,saved,optimizer=True):
    if saved['format'] not in (FORMAT,LEGACY_FORMAT):raise ValueError('Expected a Case 1 single-parameter checkpoint.')
    state['model'].load_state_dict(saved['model']);state['hyper'].load_state_dict(saved['hypernet'])
    state['hyper'].set_stage(saved['weighting_stage'])
    if state['hyper'].weight_mode!=saved['weight_mode']:raise ValueError('Checkpoint stage weight policy mismatch.')
    with torch.no_grad():
        for key,name in [('a1','a_1'),('a2','a_2'),('fn','fn')]:state[key].copy_(saved[name].to(state[key]))
    if optimizer:state['optimizer'].load_state_dict(saved['optimizer'])


def load_checkpoint(path):
    # All new checkpoints contain ordinary dict/list/scalars and tensors.
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format') not in (FORMAT,LEGACY_FORMAT):raise ValueError('Checkpoint is not a Case 1 single-parameter run.')
    return saved


def load_for_prediction(path,device='cpu'):
    saved=load_checkpoint(path);a=argparse.Namespace(**saved['config']['args'])
    objects=create_objects(a,torch.device(device));restore(objects,saved,optimizer=False)
    objects['model'].eval();return objects,saved


def run(argv=None,pause_after_new_updates=None,forced_mode=None,pause_after_stage=None):
    a=parse_args(argv,forced_mode=forced_mode)
    with j.output_lock(a.output):
        return _run(a,pause_after_new_updates,pause_after_stage)


def _run(a,pause_after_new_updates=None,pause_after_stage=None):
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
    torch.set_num_threads(a.threads);s.seed_all(a.seed);core.SEED=a.seed;core.device=device
    core._FAST_CACHE.clear();core.KERNEL_AUDIT.clear()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    requested_signature=signature(a);resuming=False;resume=None
    if any(p.name!='.run.lock' for p in out.iterdir()):
        if not a.resume:raise FileExistsError('Existing output; choose a new directory or use --resume for this same single-parameter run.')
        config=json.loads((out/'config.json').read_text())
        if config['run_signature']!=requested_signature:raise ValueError('Resume code/configuration differs. Use a new output directory.')
        if (out/'summary.json').exists():
            completed=json.loads((out/'summary.json').read_text())
            if completed.get('status') in ['SMOKE_ONLY','TRAINING_COMPLETED_ACCURACY_UNVERIFIED']:
                print('Matching run already completed; no updates repeated.',flush=True);return completed
        resume=load_checkpoint(out/'resume.pt');resuming=True
    clock=time.perf_counter();state=create_objects(a,device);model=state['model'];optimizer=state['optimizer']
    design=j.make_design(a.split_seed,a.parameter_counts,a.validation_count,a.test_count,a.eval_points,a.grids,a.operator)
    sets=[design['training'][:n] for n in a.parameter_counts]
    coordinates={role:j.coordinate_array(design,role) for role in ['validation','test','training_diagnostic']}
    # Check coordinate partitions too; no solution labels are generated here.
    coordinate_sets=[{tuple(row) for row in coordinates[role]} for role in coordinates]
    if any(coordinate_sets[i]&coordinate_sets[k] for i in range(3) for k in range(i)):
        raise AssertionError('Independent coordinate partitions overlap.')
    if resuming:
        saved_design=json.loads((out/'parameter_split.json').read_text())
        if saved_design!=design:raise ValueError('Saved parameter split differs from requested design.')
        with np.load(out/'evaluation_coordinates.npz') as arrays:
            if any(not np.array_equal(arrays[role],values) for role,values in coordinates.items()):
                raise ValueError('Saved evaluation coordinates differ.')
    else:j.save_design(out,design,coordinates)
    stages=[{'stage':i,'grid':a.grids[i],'boundary_points_per_face':a.boundary_points[i],
             'parameters':sets[i],'update_cap':a.stage_epochs[i]} for i in range(3)]
    if not resuming:
        configuration={'format':FORMAT,'revision':'case1_single_earlystop_5000','run_signature':requested_signature,
            'source_hashes':source_hashes(),'args':vars(a),'stages':stages,
            'architecture':{'branch_dimensions':[9,64,64,a.latent_dim],'trunk_input_dimensions':3+a.latent_dim,
                'trunk_hidden_layers':a.hidden_layers,'trunk_width':a.width,'trunk_count':2,
                'trunk_activation':'original adaptive dual-tanh','shared_activation_initial':[.2,.2],
                'fixed_fn':5.,'loss_weight_dimensions':[3,16,3],
                'prediction_parameters':sum(p.numel() for p in state['predictor']),
                'loss_weight_parameters':sum(p.numel() for p in state['hyper'].parameters()),
                'total_optimizer_parameters':sum(p.numel() for p in state['trainables']),
                'model_structure':str(model),'loss_weight_structure':str(state['hyper'])},
            'hardware':hardware_info(device),'design_sha256':design['design_sha256'],
            'parameter_protocol':{'operator':a.operator,'split_seed':a.split_seed,'training_parameter_counts':a.parameter_counts,
                'validation_count':a.validation_count,'test_count':a.test_count,
                'independent_training_seed':a.seed,'split_fixed_across_training_seeds':True},
            'validation_protocol':'same independent active-parameter values and coordinates at ALL stages',
            'evaluation_point_streams':design['coordinate_rng_streams'],
            'discretization':'unchanged checked fast L2-1sigma/SOE, float64 history; network float32',
            'gradient_enhancement':'local x/y/time derivatives; network and source targets exclude all Caputo derivative contributions; w0*exp(-3*Lf.detach())',
            'weighting':{'mode':'softmax_noise','stages':[0,1,2],
                'loss_order':['loss_f','loss_bc','loss_ic'],
                'normalization':'losses / max(mean(losses),1e-8); gradients retained',
                'attention':'softmax(MLP(normalized_losses)); same trainable 3-16-3 MLP',
                'formula':'max(3*phi + xi, epsilon), xi uniform[-noise_amplitude,+noise_amplitude]',
                'noise_amplitude':a.noise_amplitude,'epsilon':a.weight_lower_bound,
                'sum_exactly_three':False,'later_stage_affine_floor':False,
                'transition':'same formula at ALL stages; selected MLP parameters and Adam moments inherited'},
            'optimization':{'optimizer':'Adam','betas':[.9,.999],'epsilon':1e-8,'weight_decay':0.,
                'moment_transfer':'same selected best checkpoint as model, HyperNet and shared activations',
                'stage_target_lr_factors':a.stage_lr_factors,'warmup_stages':[],
                'warmup_epochs':a.warmup_epochs,'warmup_start_factor':a.warmup_start_factor,
                'scheduler':None,
                'scheduler_during_warmup':False,'scheduler_reset_at_stage_transition':False,
                'gradient_clip_norm':10.,'early_stopping':a.budget_mode=='early_stop',
                'budget_mode':a.budget_mode,'maximum_update_budget':sum(a.stage_epochs),
                'fixed_budget':sum(a.stage_epochs) if a.budget_mode=='fixed_steps' else None,
                'batching':'all current active-parameter instances; mean objective, sequential backward, one Adam update'},
            'stopping':{'enabled':a.budget_mode=='early_stop','eval_every_completed_updates':a.eval_every,
                'patience_evals':a.patience_evals,'minimum_delta':0.,'comparison':'strictly lower than historical best within stage',
                'loss':'equal-weight mean of LF+LBC+LIC over the independent held-out validation instances; no HyperNet/noise/GE/teacher',
                'grid_intervals_fixed_all_stages':a.grids[-1],
                'boundary_points_per_face':a.boundary_points[-1],
                'boundary_seed_stream':a.split_seed+700001,
                'stage_entry_loss_initializes_best':True,'counter_reset_per_stage':True,
                'counter_checkpointed':True,'stage_caps':a.stage_epochs,
                'model_selection':'validation macro relative L2; learning rate fixed within each stage'},
            'consistency':{'teacher':'previous stage selection-best, fixed full-grid predictions',
                'paired_parameters':'exact inherited full (alpha,beta,gamma) tuples for this single operator only','complex_loss':'mean over parameter instances and grid points, sum real/imag squared errors',
                'weight':'rho0*exp(-epoch/decay); removed when below cutoff',
                'rho0':a.consistency_initial,'decay_epochs':a.consistency_decay,'cutoff':a.consistency_cutoff,
                'first_removed_epoch':int(math.floor(a.consistency_decay*math.log(a.consistency_initial/a.consistency_cutoff))+1)},
            'model_selection':'stage-best and final global best use macro relative L2 on the SAME independent validation set, including stage entries',
            'phase_primary':'wrapped angle MAE over ALL evaluation points, as in the manuscript',
            'phase_secondary':'phase_mae_masked_radians retains old threshold: reference amplitude > 1e-6*maximum per parameter',
            'parameter_mapping':'code [alpha,D,beta] = paper [alpha,dispersion beta,nonlinearity gamma]; exactly one entry varies, selected by --operator',
            'complete_manuscript_formula_alignment':False,
            'test_targets_used_for_training_or_selection':False}
        # Remove TorchVersion subclasses / NumPy scalar classes from metadata.
        config=json.loads(json.dumps(configuration));s.atomic_json(out/'config.json',config)
    else:
        current=hardware_info(device)
        if str(current['torch'])!=config['hardware']['torch'] or current['gpu']!=config['hardware']['gpu']:
            raise ValueError('Resume requires the same recorded GPU and PyTorch version.')
    val_tx=torch.tensor(coordinates['validation'],device=device)
    train_tx=torch.tensor(coordinates['training_diagnostic'],device=device)
    val_data=j.make_evaluation(design['validation'],val_tx)
    val_physics=validation.build_validation_datasets(design,a.grids[-1],a.boundary_points[-1])
    train_evaluation=j.make_evaluation(sets[-1],train_tx)
    meter=TrainingMeter(device);records=[];validation_rows=[];history=[];previous_teacher=[]
    global_best=float('inf');global_checkpoint=None;global_step=0;point_work=0;wall_before=0.;new_updates=0
    starting_stage=0;resume_current=False
    if resuming:
        restore(state,resume['current']);s.restore_rng(resume['current']['rng'])
        records=resume['records'];validation_rows=resume['validation_rows'];history=resume['history']
        previous_teacher=s.tree_to_device(resume['previous_teacher'],device)
        global_best=resume['global_best'];global_checkpoint=resume['global_checkpoint']
        global_step=resume['global_step'];point_work=resume['point_work'];wall_before=resume['wall_seconds']
        measured=resume['meter'];meter.seconds=measured['optimization_seconds'];meter.steps=measured['updates']
        meter.peak_allocated=measured['peak_training_allocated_bytes'];meter.peak_reserved=measured['peak_training_reserved_bytes']
        meter.peak_increment=measured['peak_step_increment_allocated_bytes'];meter.max_step_seconds=measured['max_measured_update_seconds']
        starting_stage=resume['stage'];resume_current=resume['phase']=='training'
        core.KERNEL_AUDIT.update(resume['kernel_audit'])
        s.write_csv(out/'training.csv',records);s.write_csv(out/'validation_per_parameter.csv',validation_rows)
        s.atomic_json(out/'stage_history.json',history)
        if global_checkpoint is not None:s.atomic_checkpoint(out/'best_validation.pt',global_checkpoint)
        print('Resumed stage=%d next_epoch=%d global_step=%d with saved Adam/RNG/teacher.' %
              (starting_stage,resume.get('next_epoch',0),global_step),flush=True)

    def elapsed():return wall_before+time.perf_counter()-clock
    def save_resume(phase,sid,next_epoch,context=None,best=None,scheduler=None,stopper=None):
        payload={'format':FORMAT,'phase':phase,'stage':sid,'next_epoch':next_epoch,
            'current':capture(state,config,sid,next_epoch-1,{}),'stage_context':context,'stage_best':best,
            'scheduler':scheduler.state_dict() if scheduler is not None else None,
            'stopper':stopper.state_dict() if stopper is not None else None,
            'previous_teacher':s.tree_to_cpu(previous_teacher),'global_best':global_best,'global_checkpoint':global_checkpoint,
            'global_step':global_step,'point_work':point_work,'records':records,'validation_rows':validation_rows,
            'history':history,'meter':meter.report(),'wall_seconds':elapsed(),'kernel_audit':core.KERNEL_AUDIT}
        s.atomic_checkpoint(out/'resume.pt',payload)

    print('Case1 Full | active_parameters=%s | val=%d test=%d | mode=%s | stage_caps=%s | weights=softmax_noise ALL stages | split_seed=%d train_seed=%d' %
          (a.parameter_counts,a.validation_count,a.test_count,a.budget_mode,a.stage_epochs,a.split_seed,a.seed),flush=True)
    for sid in range(starting_stage,3):
        state['hyper'].set_stage(sid)
        validation_protocol='independent_single_mu_common_all_stages'
        stage=stages[sid];g=stage['grid'];parameters=stage['parameters'];cap=stage['update_cap']
        data_list=[]
        for parameter in parameters:
            data=core.build_dataset(*parameter['mu'],M1=g,M2=g,N=g,boundnum=stage['boundary_points_per_face'])
            data['parameter_id']=parameter['parameter_id']
            data_list.append(data)
        inventory=history_inventory(core,data_list)
        for item,parameter in zip(inventory,parameters):
            item.update(parameter_id=parameter['parameter_id'],beta_dispersion=parameter['beta_dispersion'],gamma_nonlinearity=parameter['gamma_nonlinearity'])
        inherited_keys={j.mu_key(p['mu']) for p in parameters}
        teacher=[row for row in previous_teacher if j.mu_key(row['mu']) in inherited_keys]
        previous_keys={j.mu_key(p['mu']) for p in sets[sid-1]} if sid else set()
        if {j.mu_key(row['mu']) for row in teacher}!=previous_keys:
            raise AssertionError('Inherited single-parameter teacher pairing failed.')
        old_evaluation=[row for row in train_evaluation if j.mu_key(row['mu']) in previous_keys]
        new_evaluation=[row for row in train_evaluation if j.mu_key(row['mu']) in inherited_keys-previous_keys]
        target_lr=a.initial_lr*a.stage_lr_factors[sid]
        scheduler=None
        if resume_current and sid==starting_stage:
            context=resume['stage_context'];best=resume['stage_best'];epoch=resume['next_epoch']
            stopper=validation.ValidationLossStop(**resume['stopper'])
            s.atomic_checkpoint(out/f'stage_{sid:02d}_best.pt',best)
            resume_current=False
        else:
            for group in optimizer.param_groups:group['lr']=target_lr
            entry_metrics,_=j.evaluate(model,val_data)
            entry_physics,_=validation.validate_physics(model,val_physics)
            stopper=validation.ValidationLossStop(a.patience_evals,best=entry_physics['validation_loss'])
            common_entry_metrics=entry_metrics
            if not math.isfinite(entry_metrics['relative_l2']):raise FloatingPointError('Nonfinite stage-entry metric.')
            best=capture(state,config,sid,-1,entry_metrics);epoch=0
            entry_hash=s.weight_fingerprint(best);moment_hash=s.moment_fingerprint(best['optimizer'])
            if sid:
                if entry_hash!=history[-1]['selected_weights_sha256']:raise AssertionError('Model / HyperNet / activations not inherited.')
                if moment_hash!=history[-1]['selected_adam_moments_sha256']:raise AssertionError('Selected Adam moments not inherited.')
            with torch.no_grad():entry_drift=float(torch.stack([s.teacher_loss(model,row) for row in teacher]).mean()) if teacher else 0.
            if entry_drift>1e-12:raise AssertionError('Teacher differs from stage-entry prediction.')
            context={'entry_metrics':entry_metrics,'entry_common_metrics':common_entry_metrics,
                'entry_validation_physics':entry_physics,
                'weight_mode':state['hyper'].weight_mode,'validation_protocol':validation_protocol,
                'entry_weights_sha256':entry_hash,'entry_adam_moments_sha256':moment_hash,
                'entry_consistency_loss':entry_drift,'teacher_sha256':s.tensor_hash(teacher),
                'start_optimization_seconds':meter.seconds,'start_global_step':global_step,
                'entry_lr':optimizer.param_groups[0]['lr'],'target_lr':target_lr}
            s.atomic_checkpoint(out/f'stage_{sid:02d}_entry.pt',best);s.atomic_checkpoint(out/f'stage_{sid:02d}_best.pt',best)
            if common_entry_metrics['relative_l2']<global_best:
                global_best=common_entry_metrics['relative_l2']
                global_checkpoint=dict(best,metrics=common_entry_metrics,stage_selection_metrics=entry_metrics)
                s.atomic_checkpoint(out/'best_validation.pt',global_checkpoint)
            save_resume('training',sid,0,context,best,scheduler,stopper)
        print('\nStage %d | grid_intervals=%d | full_grid=%d^3 | residual_points_per_tuple=%d | parameter_count=%d | weights=%s | val_L2=%.6e | inherited_pairs=%d | target_lr=%.2e' %
              (sid,g,g+1,(g-1)**2*g,len(parameters),state['hyper'].weight_mode,
               context['entry_metrics']['relative_l2'],len(teacher),target_lr),flush=True)
        point_cost=len(parameters)*(g-1)**2*g
        while epoch<cap and not (a.budget_mode=='early_stop' and stopper.count>=a.patience_evals):
            if any(group['lr']!=target_lr for group in optimizer.param_groups):
                raise AssertionError('Learning rate changed inside a stage.')
            lr_used=optimizer.param_groups[0]['lr'];rho=s.consistency_weight(sid,epoch,a.consistency_initial,a.consistency_decay,a.consistency_cutoff)
            scheduled=(epoch+1)%a.eval_every==0 or epoch+1==cap
            model.train();state['hyper'].train();meter.begin();optimizer.zero_grad(set_to_none=True)
            totals={key:0. for key in ['loss','loss_f','loss_ic','loss_bc','loss_grad','loss_grad_space','loss_grad_time','w_f','w_bc','w_ic','w_g']}
            weighted={key:0. for key in ['weighted_loss_f','weighted_loss_ic','weighted_loss_bc','weighted_loss_grad']}
            for data in data_list:
                terms=s.physics_terms(model,data,a.w_grad)
                if not bool(torch.isfinite(terms['loss'])):raise FloatingPointError('Nonfinite physics loss.')
                (terms['loss']/len(data_list)).backward()
                for key in totals:totals[key]+=float(terms[key].detach().cpu())/len(data_list)
                for key,wkey in [('loss_f','w_f'),('loss_ic','w_ic'),('loss_bc','w_bc'),('loss_grad','w_g')]:
                    weighted['weighted_'+key]+=float((terms[key]*terms[wkey]).detach().cpu())/len(data_list)
            predictor_gradient=s.gradient_vector(state['predictor']) if scheduled else None
            inter_loss=0.
            if rho:
                for row in teacher:
                    part=s.teacher_loss(model,row)
                    (rho*part/len(teacher)).backward();inter_loss+=float(part.detach().cpu())/len(teacher)
            diagnostics={'physics_prediction_gradient_norm':float('nan'),'consistency_prediction_gradient_norm':float('nan'),
                         'consistency_to_physics_gradient_ratio':float('nan'),'gradient_cosine':float('nan')}
            if scheduled:
                inter_gradient=s.gradient_vector(state['predictor'])-predictor_gradient
                pn=float(predictor_gradient.norm());cn=float(inter_gradient.norm())
                diagnostics.update(physics_prediction_gradient_norm=pn,consistency_prediction_gradient_norm=cn,
                    consistency_to_physics_gradient_ratio=cn/max(pn,1e-30),
                    gradient_cosine=float(torch.dot(predictor_gradient,inter_gradient)/(pn*cn)) if pn>0 and cn>0 else float('nan'))
            grad_norm=torch.nn.utils.clip_grad_norm_(state['trainables'],10.,error_if_nonfinite=True)
            optimizer.step();step_seconds=meter.end();global_step+=1;new_updates+=1;point_work+=point_cost
            if scheduled:
                metrics,per_parameter=j.evaluate(model,val_data);vl=metrics['relative_l2']
                validation_physics,physics_rows=validation.validate_physics(model,val_physics)
                stopper.update(validation_physics['validation_loss'])
                common_metrics=metrics;common_vl=vl
                if not math.isfinite(vl):raise FloatingPointError('Nonfinite validation error.')
                if not math.isfinite(common_vl):raise FloatingPointError('Nonfinite common validation error.')
                sigma=float(np.mean([core.relative_l2_on_dataset(model,data) for data in data_list]))
                old_l2=j.evaluate(model,old_evaluation)[0]['relative_l2'] if old_evaluation else float('nan')
                new_l2=j.evaluate(model,new_evaluation)[0]['relative_l2'] if new_evaluation else float('nan')
                with torch.no_grad():post_drift=float(torch.stack([s.teacher_loss(model,row) for row in teacher]).mean()) if teacher else 0.
                current=capture(state,config,sid,epoch,metrics)
                if vl<best['metrics']['relative_l2']:
                    best=current;s.atomic_checkpoint(out/f'stage_{sid:02d}_best.pt',best)
                if common_vl<global_best:
                    global_best=common_vl
                    global_checkpoint=dict(current,metrics=common_metrics,stage_selection_metrics=metrics)
                    s.atomic_checkpoint(out/'best_validation.pt',global_checkpoint)
                row={'stage':sid,'epoch':epoch,'completed_epochs':epoch+1,'global_step':global_step,'grid_intervals':g,'parameter_count':len(parameters),
                    **totals,**weighted,'loss_task':totals['loss'],'loss_total':totals['loss']+rho*inter_loss,
                    'loss_prev':inter_loss,'w_prev':rho,'teacher_drift_after_update':post_drift,
                    'teacher_pairs':len(teacher),'teacher_grid_points_per_parameter':teacher[0]['tx'].shape[0] if teacher else 0,
                    'sigma_train_l2':sigma,'validation_l2':vl,'validation_full_range_l2':common_vl,
                    **validation_physics,'best_validation_physics_loss':stopper.best,
                    'no_improvement_count':stopper.count,'early_stopping_enabled':a.budget_mode=='early_stop',
                    'validation_protocol':validation_protocol,'weight_mode':state['hyper'].weight_mode,
                    'weight_floor_applied':a.weight_lower_bound,
                    'inherited_parameter_l2':old_l2,'new_parameter_l2':new_l2,
                    'lr_used':lr_used,'next_lr':optimizer.param_groups[0]['lr'],'warmup_active':bool(sid and epoch<a.warmup_epochs),
                    'pre_clip_gradient_norm':float(grad_norm.detach().cpu()),**diagnostics,
                    'a1':float(state['a1'].detach().cpu()),'a2':float(state['a2'].detach().cpu()),
                    'optimization_seconds':meter.seconds,'last_update_seconds':step_seconds,'residual_point_evaluations':point_work,
                    'peak_training_allocated_bytes':meter.peak_allocated,'peak_training_reserved_bytes':meter.peak_reserved,
                    'elapsed_seconds':elapsed()}
                records.append(row)
                physics_by_id={item['parameter_id']:item for item in physics_rows}
                for item in per_parameter:
                    validation_rows.append({'stage':sid,'epoch':epoch,'completed_epochs':epoch+1,'global_step':global_step,
                                            **item,**{k:value for k,value in physics_by_id[item['parameter_id']].items() if k!='parameter_id'}})
                s.write_csv(out/'training.csv',records);s.write_csv(out/'validation_per_parameter.csv',validation_rows)
                s.atomic_checkpoint(out/'latest_model.pt',current);s.atomic_json(out/'resources_partial.json',meter.report())
                save_resume('training',sid,epoch+1,context,best,scheduler,stopper)
                print('[Stage %d] Epoch %5d | %s | Loss %.3e | sigma_L2 %.3e | stage_val_L2 %.3e | common_val_L2 %.3e | old_L2 %.3e | new_L2 %.3e | w_f %.3f w_bc %.3f w_ic %.3f | rho %.3e | lr %.2e | val_loss %.3e | no_improve %d/%d | mode %s' %
                    (sid,epoch+1,state['hyper'].weight_mode,row['loss_total'],sigma,vl,common_vl,
                     old_l2,new_l2,totals['w_f'],totals['w_bc'],totals['w_ic'],rho,lr_used,
                     validation_physics['validation_loss'],stopper.count,a.patience_evals,a.budget_mode),flush=True)
            epoch+=1
            if pause_after_new_updates is not None and new_updates>=pause_after_new_updates:
                # Do not insert an extra loss check when pausing between scheduled checks.
                save_resume('training',sid,epoch,context,best,scheduler,stopper)
                return {'status':'PAUSED_FOR_FUNCTIONAL_VERIFICATION','global_step':global_step}
        stage_updates=global_step-context['start_global_step']
        if stage_updates!=epoch:raise AssertionError('Completed stage updates and epoch counter disagree.')
        stop_reason=('validation_loss_no_improvement' if a.budget_mode=='early_stop' and stopper.count>=a.patience_evals
                     else 'maximum_stage_updates' if a.budget_mode=='early_stop' else 'completed_fixed_updates')
        last=capture(state,config,sid,epoch-1,{});s.atomic_checkpoint(out/f'stage_{sid:02d}_last.pt',last)
        if s.tensor_hash(teacher)!=context['teacher_sha256']:raise AssertionError('Frozen teacher was modified.')
        restore(state,best,optimizer=True)
        selected_common_metrics,_=j.evaluate(model,val_data)
        previous_teacher=j.frozen_full_grid_targets(model,data_list)
        history.append({'stage':sid,'grid':g,'parameters':parameters,'updates':stage_updates,'update_cap':cap,'stop_reason':stop_reason,
            'budget_mode':a.budget_mode,'best_validation_physics_loss':stopper.best,'no_improvement_count':stopper.count,
            'entry_validation_physics_loss':context['entry_validation_physics']['validation_loss'],
            'weight_mode':context['weight_mode'],'weight_floor_applied':a.weight_lower_bound,
            'validation_protocol':context['validation_protocol'],
            'entry_validation_full_range_l2':context['entry_common_metrics']['relative_l2'],
            'selected_validation_full_range_l2':selected_common_metrics['relative_l2'],
            'entry_validation_l2':context['entry_metrics']['relative_l2'],'best_validation_l2':best['metrics']['relative_l2'],
            'selected_epoch':best['epoch'],'entry_weights_sha256':context['entry_weights_sha256'],
            'selected_weights_sha256':s.weight_fingerprint(best),
            'entry_adam_moments_sha256':context['entry_adam_moments_sha256'],
            'selected_adam_moments_sha256':s.moment_fingerprint(best['optimizer']),
            'entry_consistency_loss':context['entry_consistency_loss'],'teacher_pairs':len(teacher),
            'teacher_grid_points_per_parameter':teacher[0]['tx'].shape[0] if teacher else 0,
            'entry_lr':context['entry_lr'],'target_lr':target_lr,'selected_checkpoint_next_lr':best['optimizer']['param_groups'][0]['lr'],
            'optimization_seconds':meter.seconds-context['start_optimization_seconds'],'history_state_inventory':inventory})
        s.atomic_json(out/'stage_history.json',history)
        save_resume('between_stages' if sid<2 else 'finished_training',sid+1,0)
        if pause_after_stage==sid:return {'status':'PAUSED_BETWEEN_STAGES_FOR_VERIFICATION','global_step':global_step}
    if len(history)!=3 or global_step!=sum(item['updates'] for item in history):raise AssertionError('Stage update accounting failed.')
    if a.budget_mode=='fixed_steps' and global_step!=sum(a.stage_epochs):raise AssertionError('Full fixed update budget not completed.')
    if global_step>sum(a.stage_epochs):raise AssertionError('Maximum update budget exceeded.')
    restore(state,global_checkpoint,optimizer=True);s.atomic_checkpoint(out/'final.pt',global_checkpoint)
    # Do not evaluate test solution labels until final.pt is fixed by validation.
    selected_fingerprint=s.weight_fingerprint(global_checkpoint)
    s.atomic_json(out/'test_selection_audit.json',{
        'design_sha256':design['design_sha256'],'final_weights_sha256':selected_fingerprint,
        'selection_metric':'validation macro relative L2','test_labels_used_for_selection':False,
        'training_updates_finished':global_step})
    test_tx=torch.tensor(coordinates['test'],device=device)
    test_data=j.make_evaluation(design['test'],test_tx)
    test_metrics,test_rows=j.evaluate(model,test_data)
    s.write_csv(out/'test_per_parameter.csv',test_rows)
    train_metrics,train_rows=j.evaluate(model,train_evaluation)
    s.write_csv(out/'training_parameters_evaluation.csv',train_rows)
    with torch.no_grad():
        predictions=[]
        for item in test_data:
            u,v=model(item['tx'],item['mu']);predictions.append(torch.cat([u,v],1).cpu().numpy())
        np.savez_compressed(out/'single_test_predictions.npz',
            parameter_names=np.asarray(j.PARAMETER_NAMES),
            parameter_ids=np.asarray([p['parameter_id'] for p in design['test']]),
            mu=np.asarray([p['mu'] for p in design['test']]),tx=coordinates['test'],
            prediction=np.stack(predictions),
            exact=np.stack([torch.cat([item['true'],item['true']],1).cpu().numpy() for item in test_data]))
    if s.weight_fingerprint(capture(state,config,global_checkpoint['stage'],global_checkpoint['epoch'],{}))!=selected_fingerprint:
        raise AssertionError('Test evaluation changed selected weights.')
    sync(device);resources=meter.report();resources['wall_seconds_total_checkpointed']=elapsed()
    resources['resumed_from_checkpoint']=resuming
    resources['interruption_accounting_note']='Cumulative saved scopes; work after the last checkpoint before an interruption is not recoverable in these totals.'
    s.atomic_json(out/'resources.json',resources);s.atomic_json(out/'kernel_audit.json',core.KERNEL_AUDIT)
    summary={'format':FORMAT,'revision':'case1_single_earlystop_5000','seed':a.seed,'run_signature':requested_signature,
        'status':'SMOKE_ONLY' if a.smoke else 'TRAINING_COMPLETED_ACCURACY_UNVERIFIED',
        'operator':a.operator,'budget_mode':a.budget_mode,'weighting_mode':'softmax_noise',
        'total_updates':global_step,'nominal_updates':sum(a.stage_epochs),'stages':history,
        'validation_best':global_best,'selected_stage':global_checkpoint['stage'],'selected_epoch':global_checkpoint['epoch'],
        'design_sha256':design['design_sha256'],'test_parameter_count':len(test_rows),
        'test_macro_over_parameters':test_metrics,'test_parameter_distribution':j.test_distribution(test_rows),'training_parameter_macro':train_metrics,
        'residual_point_evaluations':point_work,'resources':resources,
        'publication_model_selected_by':'common independent validation; identical points/parameters across all stages',
        'memory_scope':'Measured neural optimization including consistency and backward; not ideal SOE streaming state alone',
        'accuracy_improvement_over_previous_run_verified':False}
    s.atomic_json(out/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ['status','total_updates','validation_best','selected_stage','selected_epoch','test_macro_over_parameters','resources']},indent=2),flush=True)
    return summary


if __name__=='__main__':run()
