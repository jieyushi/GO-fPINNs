"""V6 transfer support with C01-matched Softmax+noise weights in every stage."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import operator_alpha_core as core


LOSS_INPUTS = [
    'pt_t0_collocation','pt_x1t0_collocation','pt_x2t0_collocation',
    'pt_tbc_collocation','pt_x1bc_collocation','pt_x2bc_collocation',
    'pt_t_collocation','pt_x1_collocation','pt_x2_collocation',
    'pt_ubc_collocation','pt_vbc_collocation','pt_ut0_collocation','pt_vt0_collocation',
    'pt_f_real_collocation','pt_f_real_x1_collocation','pt_f_real_x2_collocation',
    'pt_f_real_t_collocation','pt_f_image_collocation','pt_f_image_x1_collocation',
    'pt_f_image_x2_collocation','pt_f_image_t_collocation']


class StagewiseHyperNet(core.HyperNet):
    """Same 3-16-3 trainable MLP/initialization, C01 Softmax+noise rule.

    Loss inputs retain gradients. The old parameter-free Softplus is replaced
    by Identity; the same Linear/Tanh/Linear layers produce raw logits.
    Every stage uses max(3*softmax(MLP(loss/mean(loss)))+xi, epsilon).
    weight_floor remains an unused legacy argument, not a late-stage floor.
    """
    def __init__(self, weight_floor=.2, noise_amplitude=.05, lower_bound=1e-6):
        super().__init__()
        self.net[-1]=nn.Identity()
        self.weight_floor=float(weight_floor)
        self.noise_amplitude=float(noise_amplitude)
        self.lower_bound=float(lower_bound)
        self.training_stage=0

    def set_stage(self, stage):
        stage=int(stage)
        if stage not in (0,1,2):raise ValueError('Expected training stage 0, 1 or 2.')
        self.training_stage=stage

    @property
    def weight_mode(self):
        return 'softmax_noise'

    def forward(self, losses):
        normalized=losses/losses.mean(dim=-1,keepdim=True).clamp_min(1e-8)
        logits=self.net(normalized.view(1,3)).view(-1)
        phi=torch.softmax(logits,dim=-1)
        xi=self.noise_amplitude*(2*torch.rand_like(phi)-1)
        weights=(3*phi+xi).clamp_min(self.lower_bound)
        return tuple(weights.unbind())


def weighting_config(a):
    return {'all_stages':'softmax_noise','main_loss_order':['PDE','BC','IC'],'K':3,
        'input_transform':'loss/mean(loss); mean denominator clamped at 1e-8',
        'detach_loss_inputs':False,'MLP':[3,16,3],'hidden_activation':'tanh',
        'output_activation':'softmax','formula':'max(3*phi+xi,epsilon)',
        'perturbation':'Uniform(-noise_amplitude,+noise_amplitude)',
        'noise_amplitude':a.noise_amplitude,'epsilon':a.weight_lower_bound,
        'gradient_and_consistency_excluded':True,
        'sum_of_weights':'not constrained to exactly 3 after perturbation/clipping',
        'legacy_weight_floor_unused':a.weight_floor,
        'transition':'same MLP and selected Adam moments; weighting formula unchanged at transitions'}


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def nested_sets(order):
    if len(order)!=6 or not all(math.isfinite(x) and .6<=x<=.9 for x in order):
        raise ValueError('Six finite training alphas in [0.6,0.9] required.')
    if min(np.diff(sorted(order)))<1e-6:raise ValueError('Training alphas must be distinct.')
    return [sorted(order[:n]) for n in [4,5,6]]


def test_parameters(master):
    # Preserve the SAME twenty test alphas used in the preceding nested-v3.
    rng=np.random.default_rng(99173);values=[]
    while len(values)<25:
        candidate=float(rng.uniform(.6,.9))
        if min(abs(candidate-x) for x in list(master)+values)>1e-5:values.append(candidate)
    return sorted(values[5:])


def v3_validation_parameters(master):
    # Same generator, rejection rule and first FIVE values as uploaded V3.
    rng=np.random.default_rng(99173);values=[]
    while len(values)<25:
        candidate=float(rng.uniform(.6,.9))
        if min(abs(candidate-x) for x in list(master)+values)>1e-5:values.append(candidate)
    return sorted(values[:5])


def make_evaluation(alphas,points,seed,device):
    tx=torch.tensor(np.random.default_rng(seed).uniform(0,1,(points,3)),dtype=torch.float32,device=device)
    t,x,y=[tx[:,i:i+1] for i in range(3)]
    return [(float(alpha),tx,(t**(1+alpha)*torch.sin(np.pi*x)*torch.sin(np.pi*y)).detach()) for alpha in alphas]


def metric_values(u,v,true):
    u=u.double();v=v.double();true=true.double()
    error=torch.sqrt((u-true)**2+(v-true)**2)
    amplitude=torch.sqrt(u*u+v*v);reference_amplitude=np.sqrt(2)*true.abs()
    phase=torch.atan2(v,u)-torch.atan2(true,true)
    phase=torch.atan2(torch.sin(phase),torch.cos(phase))
    mask=reference_amplitude>max(1e-3*float(reference_amplitude.max()),1e-8)
    denominator=torch.linalg.vector_norm(reference_amplitude).clamp_min(1e-30)
    return {'relative_l2':float(torch.linalg.vector_norm(error)/denominator),
        'amplitude_relative_l2':float(torch.linalg.vector_norm(amplitude-reference_amplitude)/denominator),
        'phase_mae_radians':float(phase[mask].abs().mean()),
        'phase_mae_all_radians':float(phase.abs().mean()),
        'mean_absolute_complex_error':float(error.mean()),
        'max_absolute_complex_error':float(error.max())}


def evaluate(model,data):
    model.eval();rows=[]
    with torch.no_grad():
        for alpha,tx,true in data:
            u,v=model(tx,tx.new_tensor([[alpha,1.,1.]]))
            rows.append({'alpha':alpha,**metric_values(u,v,true)})
    return {key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='alpha'},rows


def evaluate_v3(model,data):
    """V3's exact evaluation arithmetic for the S0 scheduler/selection."""
    model.eval();rows=[]
    with torch.no_grad():
        for alpha,tx,true in data:
            u,v=model(tx,tx.new_tensor([[alpha,1.,1.]]))
            u=u.double();v=v.double();y=true.double()
            target_amp=np.sqrt(2)*y.abs();amp=torch.sqrt(u*u+v*v)
            err2=(u-y)**2+(v-y)**2
            rel=float(torch.sqrt(err2.sum()/(2*(y*y).sum()).clamp_min(1e-30)))
            amp_rel=float(torch.linalg.vector_norm(amp-target_amp)/torch.linalg.vector_norm(target_amp).clamp_min(1e-30))
            mask=target_amp>max(1e-3*float(target_amp.max()),1e-8)
            angle=torch.atan2(v,u)-torch.atan2(y,y)
            wrapped=torch.atan2(torch.sin(angle),torch.cos(angle))
            rows.append({'alpha':alpha,'relative_l2':rel,'amplitude_relative_l2':amp_rel,
                'phase_mae_radians':float(wrapped[mask].abs().mean()),
                'phase_mae_all_radians':float(wrapped.abs().mean()),
                'mean_absolute_complex_error':float(torch.sqrt(err2).mean()),
                'max_absolute_complex_error':float(torch.sqrt(err2).max())})
    return {key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='alpha'},rows


def physics_terms(model,data,w_grad):
    real=core.ComponentWrapper(model,data['mu'],'real');imag=core.ComponentWrapper(model,data['mu'],'imag')
    return core.compute_loss(real,imag,*[data[k] for k in LOSS_INPUTS],nn.MSELoss(),
        data['M1'],data['M2'],data['N'],data['tau'],data['alpha'],w_grad,data['D'],data['beta'],
        data['pt_t_sigma'],data['pt_x1_sigma'],data['pt_x2_sigma'],core.device)


def frozen_full_grid_targets(model,data_list):
    """Teacher values cached ONLY after previous-stage best restoration."""
    result=[];model.eval()
    with torch.no_grad():
        for data in data_list:
            tx=torch.cat([data['pt_t_collocation'],data['pt_x1_collocation'],data['pt_x2_collocation']],1).detach().clone()
            mu=data['mu'].detach().clone();u,v=model(tx,mu)
            result.append({'alpha':float(data['alpha'].detach().cpu()),'tx':tx,'mu':mu,
                           'target':torch.cat([u,v],1).detach().clone()})
    return result


def teacher_loss(model,row):
    u,v=model(row['tx'],row['mu'])
    # Mean over grid POINTS, sum over the TWO real/imag components.
    return ((torch.cat([u,v],1)-row['target'])**2).sum(1).mean()


def consistency_weight(stage,epoch,initial=.1,decay=300.,cutoff=.01):
    if stage==0:return 0.
    value=initial*math.exp(-epoch/decay)
    return value if value>=cutoff else 0.


def warmup_lr(stage,epoch,target,epochs=200,start_factor=.02):
    if stage==0 or epochs==0 or epoch>=epochs:return target
    if epochs==1:return target*start_factor
    return target*(start_factor+(1-start_factor)*epoch/(epochs-1))


def gradient_vector(parameters):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad.detach()).flatten().double().cpu() for p in parameters])


def tree_to_cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:tree_to_cpu(item) for key,item in value.items()}
    if isinstance(value,list):return [tree_to_cpu(item) for item in value]
    if isinstance(value,tuple):return tuple(tree_to_cpu(item) for item in value)
    return value


def tree_to_device(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {key:tree_to_device(item,device) for key,item in value.items()}
    if isinstance(value,list):return [tree_to_device(item,device) for item in value]
    if isinstance(value,tuple):return tuple(tree_to_device(item,device) for item in value)
    return value


def tensor_hash(value):
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            digest.update(str(item.dtype).encode());digest.update(str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=str):digest.update(str(key).encode());visit(item[key])
        elif isinstance(item,(list,tuple)):
            for x in item:visit(x)
        else:digest.update(repr(item).encode())
    visit(value);return digest.hexdigest()


def weight_fingerprint(checkpoint):
    return tensor_hash({key:checkpoint[key] for key in ['model','hypernet','a_1','a_2','fn']})


def moment_fingerprint(optimizer_state):
    # LR changes intentionally at transitions; hash moment tensors and steps.
    return tensor_hash(optimizer_state['state'])


def capture_rng():
    state=np.random.get_state()
    return {'python':random.getstate(),'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]],
        'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(saved):
    random.setstate(saved['python']);s=saved['numpy']
    np.random.set_state((s[0],np.asarray(s[1],dtype=np.uint32),s[2],s[3],s[4]))
    torch.set_rng_state(saved['torch'].cpu())
    if saved['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda']])


def atomic_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def atomic_checkpoint(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)


def write_csv(path,rows):
    if not rows:return
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tmp.replace(path)
