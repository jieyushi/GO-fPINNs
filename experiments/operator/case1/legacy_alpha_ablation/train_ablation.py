"""Eleven V6 local-x/y/t comparisons, Softmax+noise weighting in all stages.

Only the main-loss weighting rule changes; residual/grid/transfer/scheduler
and the existing eleven ablation switches retain their original definitions.
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
import ablation_support as variants
import train_alpha_operator as original_v6
from resource_tracking import TrainingMeter,hardware_info,history_inventory,sync

FORMAT='R2_alpha_ablation_softmax_noise_local_xt'
DEFAULT_ORDER=[.6,.7,.8,.9,.65,.85]
DEFAULT_VALIDATION=[.625,.675,.725,.775,.825,.875]


def three_counts(text,minimum=1):
    values=[int(x) for x in text.split(',')]
    if len(values)!=3 or min(values)<minimum:raise ValueError('Exactly three valid positive counts required.')
    return values


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--output',default='runs/R2_alpha_softmax_noise_grid6810/matched_steps/full/seed_1234')
    p.add_argument('--epochs-per-stage',type=int,default=5000)
    p.add_argument('--stage-epochs',default=None)
    p.add_argument('--grids',default='5,7,9',
        help='Intervals per axis at S0/S1/S2. 5,7,9 gives full grids of 6^3,8^3,10^3 points including endpoints.')
    p.add_argument('--boundary-points',default='80,120,150')
    p.add_argument('--alpha-order',default=','.join(str(x) for x in DEFAULT_ORDER))
    p.add_argument('--validation-alphas',default=','.join(str(x) for x in DEFAULT_VALIDATION),
        help='Common validation and S1/S2 selection. S0 uses V3 original five alphas.')
    p.add_argument('--width',type=int,default=60)
    p.add_argument('--hidden-layers',type=int,default=5)
    p.add_argument('--latent-dim',type=int,default=16)
    p.add_argument('--initial-lr',type=float,default=1e-3)
    p.add_argument('--w-grad',type=float,default=1e-4)
    p.add_argument('--weight-floor',type=float,default=.2,help='Unused legacy argument; no late-stage affine floor.')
    p.add_argument('--noise-amplitude',type=float,default=.05)
    p.add_argument('--weight-lower-bound',type=float,default=1e-6)
    p.add_argument('--warmup-epochs',type=int,default=0)
    p.add_argument('--warmup-start-factor',type=float,default=.02)
    p.add_argument('--consistency-initial',type=float,default=.1)
    p.add_argument('--consistency-decay',type=float,default=300.)
    p.add_argument('--consistency-cutoff',type=float,default=.01)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--eval-points',type=int,default=4096)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--resume',action='store_true',help='Restore this v6 resume.pt, including stage weight policy, Adam, scheduler, teacher and RNG.')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--variant',choices=list(variants.VARIANTS),default='full')
    p.add_argument('--budget-mode',choices=['matched_steps','matched_point_work','matched_time'],default='matched_steps')
    p.add_argument('--phase-seconds',default=None,help='Three measured optimization time limits from the completed full reference.')
    p.add_argument('--validation-only',action='store_true',help='Tuning trial: do not construct or evaluate test targets.')
    a=p.parse_args(argv)
    try:
        a.grids=three_counts(a.grids,2)
        if not a.grids[0]<a.grids[1]<a.grids[2]:raise ValueError('Grids must increase.')
        a.boundary_points=three_counts(a.boundary_points)
        a.stage_epochs=three_counts(a.stage_epochs) if a.stage_epochs else [a.epochs_per_stage]*3
        a.alpha_order=[float(x) for x in a.alpha_order.split(',')];s.nested_sets(a.alpha_order)
        a.validation_alphas=[float(x) for x in a.validation_alphas.split(',')]
        if len(a.validation_alphas)<3 or not all(math.isfinite(x) and .6<x<.9 for x in a.validation_alphas):
            raise ValueError('At least three validation alphas strictly inside (0.6,0.9) required.')
        if len(set(a.validation_alphas))!=len(a.validation_alphas):raise ValueError('Validation alphas must be distinct.')
        test=s.test_parameters(a.alpha_order)
        if any(min(abs(x-y) for y in a.alpha_order+test)<1e-5 for x in a.validation_alphas):
            raise ValueError('Validation alphas must be separate from training and test alphas.')
    except ValueError as error:p.error(str(error))
    if min(a.epochs_per_stage,a.width,a.hidden_layers,a.latent_dim,a.eval_every,a.eval_points,a.threads)<1:p.error('Counts must be positive.')
    if a.warmup_epochs<0:p.error('--warmup-epochs cannot be negative.')
    positive=[a.initial_lr,a.consistency_initial,a.consistency_decay,a.consistency_cutoff]
    if not all(math.isfinite(x) and x>0 for x in positive):p.error('LR and consistency settings must be finite and positive.')
    if not 0<a.weight_floor<1:p.error('--weight-floor must lie in (0,1).')
    if not math.isfinite(a.noise_amplitude) or a.noise_amplitude<0:p.error('Noise amplitude must be finite and nonnegative.')
    if not math.isfinite(a.weight_lower_bound) or a.weight_lower_bound<=0:p.error('Weight lower bound must be positive and finite.')
    if not 0<a.warmup_start_factor<=1:p.error('Warmup start factor must lie in (0,1].')
    if not math.isfinite(a.w_grad) or a.w_grad<0:p.error('Invalid gradient-enhancement weight.')
    if a.consistency_initial<a.consistency_cutoff:p.error('Initial consistency weight must be >= cutoff.')
    if a.smoke:
        a.stage_epochs=[4,4,4];a.epochs_per_stage=4;a.grids=[2,3,4];a.boundary_points=[8,12,16]
        a.width=8;a.hidden_layers=2;a.latent_dim=4;a.eval_every=1;a.eval_points=32
        a.warmup_epochs=0;a.consistency_decay=.7
    if a.phase_seconds:
        try:a.phase_seconds=[float(x) for x in a.phase_seconds.split(',')]
        except ValueError:p.error('--phase-seconds requires three positive finite numbers.')
    if a.budget_mode=='matched_time':
        if not a.phase_seconds or len(a.phase_seconds)!=3 or not all(math.isfinite(x) and x>0 for x in a.phase_seconds):
            p.error('matched_time requires --phase-seconds from the full reference.')
    elif a.phase_seconds is not None:p.error('--phase-seconds is only valid with matched_time.')
    try:a.stage_plan,a.reference_plan=variants.make_plan(a)
    except ValueError as error:p.error(str(error))
    return a


def source_hashes():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ['train_alpha_operator.py','train_ablation.py','ablation_support.py','training_support.py','operator_alpha_core.py','resource_tracking.py']}


def signature(args):
    values=vars(args).copy();values.pop('resume',None);values.pop('output',None)
    return hashlib.sha256(json.dumps({'args':values,'sources':source_hashes()},sort_keys=True).encode()).hexdigest()


def create_objects(a,device):
    return variants.configure_state(a,device,original_v6.create_objects)


def capture(state,config,stage,epoch,metrics):
    return {'format':FORMAT,'config':config,'stage':stage,'epoch':epoch,'metrics':metrics,
        'weighting_stage':state['hyper'].training_stage,'weight_mode':state['hyper'].weight_mode,
        'model':s.tree_to_cpu(state['model'].state_dict()),'hypernet':s.tree_to_cpu(state['hyper'].state_dict()),
        'a_1':s.tree_to_cpu(state['a1']),'a_2':s.tree_to_cpu(state['a2']),'fn':s.tree_to_cpu(state['fn']),
        'optimizer':s.tree_to_cpu(state['optimizer'].state_dict()),'rng':s.capture_rng()}


def restore(state,saved,optimizer=True):
    if saved['format']!=FORMAT:raise ValueError('Expected this local-xt V6 checkpoint; spatial-only V6/V3/V5 are not exact resumes.')
    state['model'].load_state_dict(saved['model']);state['hyper'].load_state_dict(saved['hypernet'])
    state['hyper'].set_stage(saved['weighting_stage'])
    if state['hyper'].weight_mode!=saved['weight_mode']:raise ValueError('Checkpoint stage weight policy mismatch.')
    with torch.no_grad():
        for key,name in [('a1','a_1'),('a2','a_2'),('fn','fn')]:state[key].copy_(saved[name].to(state[key]))
    if optimizer:state['optimizer'].load_state_dict(saved['optimizer'])


def load_checkpoint(path):
    # All new checkpoints contain ordinary dict/list/scalars and tensors.
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format')!=FORMAT:raise ValueError('Checkpoint is not local-xt hybrid v6.')
    return saved


def load_for_prediction(path,device='cpu'):
    saved=load_checkpoint(path);a=argparse.Namespace(**saved['config']['args'])
    objects=create_objects(a,torch.device(device));restore(objects,saved,optimizer=False)
    objects['model'].eval();return objects,saved


def run(argv=None,pause_after_new_updates=None):
    a=parse_args(argv);device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
    torch.set_num_threads(a.threads);s.seed_all(a.seed);core.SEED=a.seed;core.device=device
    core._FAST_CACHE.clear();core.KERNEL_AUDIT.clear()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    requested_signature=signature(a);resuming=False;resume=None
    if any(out.iterdir()):
        if not a.resume:raise FileExistsError('Existing output; choose a new directory or use --resume for this same v6 run.')
        config=json.loads((out/'config.json').read_text())
        if config['run_signature']!=requested_signature:raise ValueError('Resume code/configuration differs. Use a new output directory.')
        if (out/'summary.json').exists():
            completed=json.loads((out/'summary.json').read_text())
            if completed.get('status') in ['SMOKE_ONLY','TRAINING_COMPLETED_ACCURACY_UNVERIFIED','VALIDATION_ONLY_COMPLETED']:
                print('Matching run already completed; no updates repeated.',flush=True);return completed
        resume=load_checkpoint(out/'resume.pt');resuming=True
    clock=time.perf_counter();state=create_objects(a,device);model=state['model'];optimizer=state['optimizer']
    sets=s.nested_sets(a.alpha_order);test_alphas=s.test_parameters(sets[-1])
    v3_val_alphas=s.v3_validation_parameters(sets[-1])
    stages=a.stage_plan
    sets=[row['alphas'] for row in stages]
    active=variants.flags(a.variant)
    if not resuming:
        configuration={'format':FORMAT,'revision':'v6_local_xt_softmax_noise','run_signature':requested_signature,
            'source_hashes':source_hashes(),'args':vars(a),'stages':stages,
            'variant':a.variant,'switches':active,
            'budget_mode':a.budget_mode,'reference_stages':a.reference_plan,
            'controlled_protocol':{
                'no_enrichment_definition':'Single stage on the final grid and all six parameters, with the matched total optimizer-update budget.',
                'plain_definition':'Single-stage raw-parameter tanh operator without enhancement components, using the matched total optimizer-update budget.',
                'phases':'Optimization phases do not imply enrichment when grid and parameters are fixed.',
                'capacity':'Exactly identical predictor for enrichment ablations. Ordinary tanh removes two scalar parameters; raw branch uses the nearest-capacity integer width, no dummy padding.',
                'tuning':'One shared predeclared configuration in fixed-setting suite; optional equal-trial validation-only learning-rate sweep.',
                'historical_tuning':'V6 was selected after prior development; current equal-trial protocol does not erase that history.',
                'point_budget':'PDE residual sample evaluations, a workload proxy; not FLOPs or guaranteed equal runtime.',
                'time_budget':'Synchronized optimization-step time, excludes setup/validation/checkpoint IO; at most one whole-step overshoot per phase.'},
            'architecture':variants.architecture(state,a),
            'hardware':hardware_info(device),'validation_alphas':a.validation_alphas,'test_alphas':test_alphas,
            'validation_protocol':{'S0_scheduler_and_stage_best':'original V3 five-alpha validation',
                'S0_validation_alphas':v3_val_alphas,'S1_S2_scheduler_and_stage_best':'V5 six-alpha validation by default',
                'common_all_stage_and_global_best':'validation_full_range_l2 on configured common validation alphas',
                'warning':'validation_l2 changes parameter set at S1; use validation_full_range_l2 for cross-stage comparisons'},
            'evaluation_point_seeds':{'validation':99101,'test':99102,'training_parameters':99103},
            'discretization':'unchanged checked fast L2-1sigma/SOE, float64 history; network float32',
            'gradient_enhancement':'local x/y/time derivatives; network and source targets exclude all Caputo derivative contributions; w0*exp(-3*Lf.detach())',
            'weighting':s.weighting_config(a),
            'optimization':{'optimizer':'Adam','betas':[.9,.999],'epsilon':1e-8,'weight_decay':0.,
                'moment_transfer':'same selected best checkpoint as model, HyperNet and shared activations',
                'stage_target_lr_factors':[1.,.5,.25],'warmup_stages':[],
                'warmup_epochs':a.warmup_epochs,'warmup_start_factor':a.warmup_start_factor,
                'scheduler':None,
                'scheduler_during_warmup':False,'scheduler_reset_at_stage_transition':False,
                'gradient_clip_norm':10.,'early_stopping':False,'fixed_budget':sum(a.stage_epochs),
                'batching':'all current alphas; mean objective, sequential backward, one Adam update'},
            'consistency':{'teacher':'previous stage selection-best, fixed full-grid predictions',
                'paired_parameters':'inherited alphas only','complex_loss':'mean over alphas and grid points, sum real/imag squared errors',
                'weight':'rho0*exp(-epoch/decay); removed when below cutoff',
                'rho0':a.consistency_initial,'decay_epochs':a.consistency_decay,'cutoff':a.consistency_cutoff,
                'first_removed_epoch':int(math.floor(a.consistency_decay*math.log(a.consistency_initial/a.consistency_cutoff))+1)},
            'model_selection':'Stage-best uses V3 five alphas for S0 and common validation for S1/S2; final global best uses common validation in ALL stages, including entries',
            'phase_mask':'reference amplitude > max(1e-3*maximum reference amplitude,1e-8)',
            'phase_main_report':'phase_mae_radians; masked wrapped phase error',
            'parameter_mapping':'code [alpha,D,beta] = paper [alpha,dispersion beta,nonlinearity gamma]; D=beta(code)=1',
            
            'test_targets_used_for_training_or_selection':False}
        if not active['adaptive_weights']:
            configuration['weighting']={'all_phases':'w_f=w_bc=w_ic=1; no trainable loss-weight parameters'}
        if not active['consistency']:
            configuration['consistency']={'enabled':False,'teacher':'none'}
        if not active['gradient']:
            configuration['gradient_enhancement']='disabled; local third derivatives are not computed'
        configuration['optimization']['nominal_reference_updates']=sum(a.reference_stage_epochs)
        configuration['optimization']['phase_time_limits']=a.phase_seconds
        # Remove TorchVersion subclasses / NumPy scalar classes from metadata.
        config=json.loads(json.dumps(configuration));s.atomic_json(out/'config.json',config)
    else:
        current=hardware_info(device)
        if str(current['torch'])!=config['hardware']['torch'] or current['gpu']!=config['hardware']['gpu']:
            raise ValueError('Resume requires the same recorded GPU and PyTorch version.')
    val_data=s.make_evaluation(a.validation_alphas,a.eval_points,99101,device)
    v3_val_data=s.make_evaluation(v3_val_alphas,a.eval_points,99101,device)
    train_evaluation=s.make_evaluation(sets[-1],a.eval_points,99103,device)
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
        s.write_csv(out/'training.csv',records);s.write_csv(out/'validation_per_alpha.csv',validation_rows)
        s.atomic_json(out/'stage_history.json',history)
        if global_checkpoint is not None:s.atomic_checkpoint(out/'best_validation.pt',global_checkpoint)
        print('Resumed stage=%d next_epoch=%d global_step=%d with saved Adam/RNG/teacher.' %
              (starting_stage,resume.get('next_epoch',0),global_step),flush=True)

    def elapsed():return wall_before+time.perf_counter()-clock
    def save_resume(phase,sid,next_epoch,context=None,best=None,scheduler=None):
        payload={'format':FORMAT,'phase':phase,'stage':sid,'next_epoch':next_epoch,
            'current':capture(state,config,sid,next_epoch-1,{}),'stage_context':context,'stage_best':best,
            'scheduler':scheduler.state_dict() if scheduler is not None else None,
            'previous_teacher':s.tree_to_cpu(previous_teacher),'global_best':global_best,'global_checkpoint':global_checkpoint,
            'global_step':global_step,'point_work':point_work,'records':records,'validation_rows':validation_rows,
            'history':history,'meter':meter.report(),'wall_seconds':elapsed(),'kernel_audit':core.KERNEL_AUDIT}
        s.atomic_checkpoint(out/'resume.pt',payload)

    print('V6 ablation | variant=%s | budget=%s | grids=%s | alpha_counts=%s | phase_caps=%s' %
          (a.variant,a.budget_mode,[x['grid'] for x in stages],[len(x['alphas']) for x in stages],a.stage_epochs),flush=True)
    for sid in range(starting_stage,len(stages)):
        state['hyper'].set_stage(sid)
        selection_data=val_data
        selection_evaluate=s.evaluate
        validation_protocol='common_validation_alphas'
        stage=stages[sid];g=stage['grid'];alphas=stage['alphas'];cap=stage['update_cap']
        data_list=[core.build_dataset(alpha,1.,1.,g,g,g,stage['boundary_points_per_face']) for alpha in alphas]
        inventory=history_inventory(core,data_list) if active['soe'] else {'history_mode':'direct_l21'}
        inherited_keys={round(x,7) for x in alphas}
        teacher=[row for row in previous_teacher if round(row['alpha'],7) in inherited_keys]
        if not active['consistency']:teacher=[]
        if sid and active['consistency'] and len(teacher)!=len(sets[sid-1]):raise AssertionError('Inherited parameter teacher pairing failed.')
        old_evaluation=[row for row in train_evaluation if sid and round(row[0],7) in {round(x,7) for x in sets[sid-1]}]
        new_evaluation=[row for row in train_evaluation if round(row[0],7) in {round(x,7) for x in alphas}
                        and (not sid or round(row[0],7) not in {round(x,7) for x in sets[sid-1]})]
        target_lr=a.initial_lr*0.5**sid
        if resume_current and sid==starting_stage:
            context=resume['stage_context'];best=resume['stage_best'];epoch=resume['next_epoch']
            scheduler=None
            s.atomic_checkpoint(out/f'stage_{sid:02d}_best.pt',best)
            resume_current=False
        else:
            for group in optimizer.param_groups:group['lr']=s.warmup_lr(sid,0,target_lr,a.warmup_epochs,a.warmup_start_factor)
            scheduler=None
            entry_metrics,_=selection_evaluate(model,selection_data)
            common_entry_metrics,_=s.evaluate(model,val_data)
            if not math.isfinite(entry_metrics['relative_l2']):raise FloatingPointError('Nonfinite stage-entry metric.')
            best=capture(state,config,sid,-1,entry_metrics);epoch=0
            entry_hash=s.weight_fingerprint(best);moment_hash=s.moment_fingerprint(best['optimizer'])
            if sid:
                if entry_hash!=history[-1]['selected_weights_sha256']:raise AssertionError('Model / HyperNet / activations not inherited.')
                if moment_hash!=history[-1]['selected_adam_moments_sha256']:raise AssertionError('Selected Adam moments not inherited.')
            with torch.no_grad():entry_drift=float(torch.stack([s.teacher_loss(model,row) for row in teacher]).mean()) if teacher else 0.
            if entry_drift>1e-12:raise AssertionError('Teacher differs from stage-entry prediction.')
            context={'entry_metrics':entry_metrics,'entry_common_metrics':common_entry_metrics,
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
            save_resume('training',sid,0,context,best,scheduler)
        print('\nStage %d | grid_intervals=%d | full_grid=%d^3 | residual_points_per_alpha=%d | alphas=%s | weights=%s | stage_val=%.6e | common_val=%.6e | inherited_pairs=%d | target_lr=%.2e' %
              (sid,g,g+1,(g-1)**2*g,alphas,state['hyper'].weight_mode,context['entry_metrics']['relative_l2'],
               context['entry_common_metrics']['relative_l2'],len(teacher),target_lr),flush=True)
        point_cost=len(alphas)*(g-1)**2*g
        while epoch<cap and (a.budget_mode!='matched_time' or
                meter.seconds-context['start_optimization_seconds']<a.phase_seconds[sid]):
            if sid and epoch<a.warmup_epochs:
                for group in optimizer.param_groups:group['lr']=s.warmup_lr(sid,epoch,target_lr,a.warmup_epochs,a.warmup_start_factor)
            elif sid and epoch==a.warmup_epochs:
                for group in optimizer.param_groups:group['lr']=target_lr
            lr_used=optimizer.param_groups[0]['lr'];rho=s.consistency_weight(sid,epoch,a.consistency_initial,a.consistency_decay,a.consistency_cutoff)
            if not active['consistency']:rho=0.
            scheduled=epoch%a.eval_every==0 or epoch+1==cap
            model.train();state['hyper'].train();meter.begin();optimizer.zero_grad(set_to_none=True)
            totals={key:0. for key in ['loss','loss_f','loss_ic','loss_bc','loss_grad','loss_grad_space','loss_grad_time','w_f','w_bc','w_ic','w_g']}
            weighted={key:0. for key in ['weighted_loss_f','weighted_loss_ic','weighted_loss_bc','weighted_loss_grad']}
            for data in data_list:
                terms=variants.physics_terms(model,data,a.w_grad,a.variant)
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
            if a.budget_mode=='matched_time' and meter.seconds-context['start_optimization_seconds']>=a.phase_seconds[sid]:
                scheduled=True
            if scheduled:
                metrics,_=selection_evaluate(model,selection_data);vl=metrics['relative_l2']
                common_metrics,per_alpha=s.evaluate(model,val_data);common_vl=common_metrics['relative_l2']
                if not math.isfinite(vl):raise FloatingPointError('Nonfinite validation error.')
                if not math.isfinite(common_vl):raise FloatingPointError('Nonfinite common validation error.')
                sigma=float(np.mean([core.relative_l2_on_dataset(model,data) for data in data_list]))
                old_l2=s.evaluate(model,old_evaluation)[0]['relative_l2'] if old_evaluation else float('nan')
                new_l2=s.evaluate(model,new_evaluation)[0]['relative_l2'] if new_evaluation else float('nan')
                with torch.no_grad():post_drift=float(torch.stack([s.teacher_loss(model,row) for row in teacher]).mean()) if teacher else 0.
                current=capture(state,config,sid,epoch,metrics)
                if vl<best['metrics']['relative_l2']:
                    best=current;s.atomic_checkpoint(out/f'stage_{sid:02d}_best.pt',best)
                if common_vl<global_best:
                    global_best=common_vl
                    global_checkpoint=dict(current,metrics=common_metrics,stage_selection_metrics=metrics)
                    s.atomic_checkpoint(out/'best_validation.pt',global_checkpoint)
                row={'stage':sid,'epoch':epoch,'global_step':global_step,'grid_intervals':g,'alpha_count':len(alphas),
                    **totals,**weighted,'loss_task':totals['loss'],'loss_total':totals['loss']+rho*inter_loss,
                    'loss_prev':inter_loss,'w_prev':rho,'teacher_drift_after_update':post_drift,
                    'teacher_pairs':len(teacher),'teacher_grid_points_per_parameter':teacher[0]['tx'].shape[0] if teacher else 0,
                    'sigma_train_l2':sigma,'validation_l2':vl,'validation_full_range_l2':common_vl,
                    'validation_protocol':validation_protocol,'weight_mode':state['hyper'].weight_mode,
                    'weight_floor_applied':0.,
                    'weight_noise_amplitude':a.noise_amplitude if active['adaptive_weights'] else 0.,
                    'main_weight_lower_bound':a.weight_lower_bound if active['adaptive_weights'] else 0.,
                    'inherited_alpha_l2':old_l2,'new_alpha_l2':new_l2,
                    'lr_used':lr_used,'next_lr':optimizer.param_groups[0]['lr'],'warmup_active':bool(sid and epoch<a.warmup_epochs),
                    'pre_clip_gradient_norm':float(grad_norm.detach().cpu()),**diagnostics,
                    'a1':float(state['a1'].detach().cpu()),'a2':float(state['a2'].detach().cpu()),
                    'optimization_seconds':meter.seconds,'last_update_seconds':step_seconds,'residual_point_evaluations':point_work,
                    'peak_training_allocated_bytes':meter.peak_allocated,'peak_training_reserved_bytes':meter.peak_reserved,
                    'elapsed_seconds':elapsed()}
                records.append(row)
                for item in per_alpha:validation_rows.append({'stage':sid,'epoch':epoch,'global_step':global_step,**item})
                s.write_csv(out/'training.csv',records);s.write_csv(out/'validation_per_alpha.csv',validation_rows)
                s.atomic_checkpoint(out/'latest_model.pt',current);s.atomic_json(out/'resources_partial.json',meter.report())
                save_resume('training',sid,epoch+1,context,best,scheduler)
                print('[Stage %d] Epoch %5d | %s | Loss %.3e | sigma_L2 %.3e | stage_val_L2 %.3e | common_val_L2 %.3e | old_L2 %.3e | new_L2 %.3e | w_f %.3f w_bc %.3f w_ic %.3f | rho %.3e | lr %.2e' %
                    (sid,epoch,state['hyper'].weight_mode,row['loss_total'],sigma,vl,common_vl,
                     old_l2,new_l2,totals['w_f'],totals['w_bc'],totals['w_ic'],rho,lr_used),flush=True)
                if pause_after_new_updates is not None and new_updates>=pause_after_new_updates:
                    return {'status':'PAUSED_FOR_FUNCTIONAL_VERIFICATION','global_step':global_step}
            epoch+=1
        last=capture(state,config,sid,epoch-1,{});s.atomic_checkpoint(out/f'stage_{sid:02d}_last.pt',last)
        if s.tensor_hash(teacher)!=context['teacher_sha256']:raise AssertionError('Frozen teacher was modified.')
        transition=best if active['best_transfer'] else last
        restore(state,transition,optimizer=True)
        s.atomic_checkpoint(out/f'stage_{sid:02d}_transition.pt',transition)
        selected_common_metrics,_=s.evaluate(model,val_data)
        previous_teacher=s.frozen_full_grid_targets(model,data_list) if active['consistency'] else []
        history.append({'stage':sid,'grid':g,'alphas':alphas,'updates':epoch,'stop_reason':'completed_phase_time' if a.budget_mode=='matched_time' else 'completed_fixed_updates',
            'weight_mode':context['weight_mode'],'weight_floor_applied':0.,
            'weight_noise_amplitude':a.noise_amplitude if active['adaptive_weights'] else 0.,
            'validation_protocol':context['validation_protocol'],
            'entry_validation_full_range_l2':context['entry_common_metrics']['relative_l2'],
            'selected_validation_full_range_l2':selected_common_metrics['relative_l2'],
            'entry_validation_l2':context['entry_metrics']['relative_l2'],'best_validation_l2':best['metrics']['relative_l2'],
            'selected_epoch':transition['epoch'],'entry_weights_sha256':context['entry_weights_sha256'],
            'selected_weights_sha256':s.weight_fingerprint(transition),
            'entry_adam_moments_sha256':context['entry_adam_moments_sha256'],
            'selected_adam_moments_sha256':s.moment_fingerprint(transition['optimizer']),
            'entry_consistency_loss':context['entry_consistency_loss'],'teacher_pairs':len(teacher),
            'teacher_grid_points_per_parameter':teacher[0]['tx'].shape[0] if teacher else 0,
            'entry_lr':context['entry_lr'],'target_lr':target_lr,'selected_checkpoint_next_lr':transition['optimizer']['param_groups'][0]['lr'],
            'optimization_seconds':meter.seconds-context['start_optimization_seconds'],'history_state_inventory':inventory})
        s.atomic_json(out/'stage_history.json',history)
        save_resume('between_stages' if sid+1<len(stages) else 'finished_training',sid+1,0)
    if a.budget_mode!='matched_time' and global_step!=sum(a.stage_epochs):raise AssertionError('Full fixed update budget not completed.')
    restore(state,global_checkpoint,optimizer=True);s.atomic_checkpoint(out/'final.pt',global_checkpoint)
    test_metrics=None;test_global_max=None
    if not a.validation_only:
        # Test tensors and targets are created only after model selection is fixed.
        test_data=s.make_evaluation(test_alphas,a.eval_points,99102,device)
        test_metrics,test_rows=s.evaluate(model,test_data);s.write_csv(out/'test_per_alpha.csv',test_rows)
        train_metrics,train_rows=s.evaluate(model,train_evaluation);s.write_csv(out/'training_parameters_evaluation.csv',train_rows)
        with torch.no_grad():
            predictions=[]
            for alpha,tx,true in test_data:
                u,v=model(tx,tx.new_tensor([[alpha,1.,1.]]));predictions.append(torch.cat([u,v],1).cpu().numpy())
            np.savez_compressed(out/'unseen_alpha_predictions.npz',alpha=np.asarray(test_alphas),tx=test_data[0][1].cpu().numpy(),
                prediction=np.stack(predictions),exact=np.stack([torch.cat([y,y],1).cpu().numpy() for _,_,y in test_data]))
        test_global_max=max(row['max_absolute_complex_error'] for row in test_rows)
    sync(device);resources=meter.report();resources['wall_seconds_total_checkpointed']=elapsed()
    resources['resumed_from_checkpoint']=resuming
    resources['history_forward_scope']='Per-parameter ideal SOE state O(P*N_exp); includes two complex components. Neural training uses full time-grid predictions and autograd; streaming complexity is not an end-to-end training bound.'
    resources['reference_point_work']=sum(row['reference_residual_point_budget'] for row in stages)
    resources['actual_point_work']=point_work
    resources['phase_time_limits']=a.phase_seconds
    resources['point_work_is_compute_proxy_not_FLOPs']=True
    resources['phase_time_overshoot_seconds']=[max(0.,h['optimization_seconds']-a.phase_seconds[i]) for i,h in enumerate(history)] if a.phase_seconds else None
    resources['interruption_accounting_note']='Cumulative saved scopes; work after the last checkpoint before an interruption is not recoverable in these totals.'
    s.atomic_json(out/'resources.json',resources);s.atomic_json(out/'kernel_audit.json',core.KERNEL_AUDIT)
    summary={'format':FORMAT,'revision':'v6_local_xt_softmax_noise','seed':a.seed,'run_signature':requested_signature,
        'status':'SMOKE_ONLY' if a.smoke else ('VALIDATION_ONLY_COMPLETED' if a.validation_only else 'TRAINING_COMPLETED_ACCURACY_UNVERIFIED'),
        'variant':a.variant,'budget_mode':a.budget_mode,'reference_updates':sum(a.reference_stage_epochs),
        'architecture':config['architecture'],'hardware':config['hardware'],
        'actual_initial_lr':a.initial_lr,'switches':active,
        'total_updates':global_step,'nominal_updates':sum(a.stage_epochs),'stages':history,
        'validation_best':global_best,'selected_stage':global_checkpoint['stage'],'selected_epoch':global_checkpoint['epoch'],
        'test_macro_over_unseen_alphas':test_metrics,
        'test_global_max_absolute_complex_error':test_global_max,
        'training_parameter_macro':None if a.validation_only else train_metrics,
        'residual_point_evaluations':point_work,'resources':resources,
        'publication_model_selected_by':'common independent six-alpha validation by default; identical points/parameters across all stages',
        'memory_scope':'Measured neural optimization including consistency and backward; not ideal SOE streaming state alone',
        'accuracy_improvement_over_previous_run_verified':False}
    s.atomic_json(out/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ['status','total_updates','validation_best','selected_stage','selected_epoch','test_macro_over_unseen_alphas','resources']},indent=2),flush=True)
    return summary


if __name__=='__main__':
    import fcntl
    import sys
    # Serialize only the SAME output path; distinct seeds/variants are independent.
    parsed=parse_args()
    destination=Path(parsed.output).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    with open(str(destination)+'.lock','a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process owns this output directory.')
        run(sys.argv[1:])
