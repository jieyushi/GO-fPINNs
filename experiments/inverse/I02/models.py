"Construct solution and inverse-parameter models and transfer inherited optimizer states."
import copy
import types
import torch
from torch import nn
from original_components import RealNet,ImageNet,HyperNet
from common import cpu_tree,digest


def plain_forward(self,x):
    out=torch.tanh(self.input_layer(x))
    for layer in self.hidden_layers:out=torch.tanh(layer(out))
    return self.output_layer(out)


class CosinePotential(nn.Module):
    def __init__(self,degree,device):
        super().__init__();self.coefficients=nn.Parameter(torch.zeros(degree+1,dtype=torch.float64,device=device))
    def forward(self,x):
        n=torch.arange(len(self.coefficients),dtype=x.dtype,device=x.device)
        return torch.cos(2*x*n).matmul(self.coefficients[:,None])


def create(args,device,degree=0):
    dtype=torch.float32 if args.network_dtype=='float32' else torch.float64
    go=args.method=='go_multi'
    a1=torch.tensor([.1],device=device,dtype=dtype,requires_grad=go)
    a2=torch.tensor([.1],device=device,dtype=dtype,requires_grad=go)
    fn=torch.tensor([10.],device=device,dtype=dtype)
    real=RealNet(args.extra_hidden_layers,2,args.width,torch.tanh,a1,a2,fn).to(device=device,dtype=dtype)
    imag=ImageNet(args.extra_hidden_layers,2,args.width,torch.tanh,a1,a2,fn).to(device=device,dtype=dtype)
    hyper=HyperNet(.6,1.5).to(device=device,dtype=dtype)
    if go:potential=CosinePotential(degree,device)
    else:
        potential=nn.Sequential(nn.Linear(1,20),nn.Tanh(),nn.Linear(20,20),nn.Tanh(),nn.Linear(20,20),nn.Tanh(),nn.Linear(20,1)).to(device=device,dtype=dtype)
        real.forward=types.MethodType(plain_forward,real);imag.forward=types.MethodType(plain_forward,imag)
        hyper.requires_grad_(False)
    opts=[torch.optim.Adam(real.parameters(),lr=.001),torch.optim.Adam(imag.parameters(),lr=.001)]
    if go:
        opts[0].add_param_group({'params':[a1,a2]})
        if args.freeze_hypernet:hyper.requires_grad_(False)
        else:opts.append(torch.optim.Adam(hyper.parameters(),lr=.001))
    opts.append(torch.optim.Adam(potential.parameters(),lr=.001))
    return dict(realnet=real,imagenet=imag,hypernet=hyper,potential=potential,a1=a1,a2=a2,fn=fn,
                optimizers=opts,device=device,dtype=dtype,degree=degree if go else None)


def predict(state,tx):
    tx=tx.to(state['dtype'])
    return torch.cat([state['realnet'](tx),state['imagenet'](tx)],1).to(torch.float64)


def eta(state,x):
    dtype=torch.float64 if state['degree'] is not None else state['dtype']
    return state['potential'](x.to(dtype)).to(torch.float64)


def enrich(state,degree):
    if state['degree'] is None or degree==state['degree']:return dict(expanded=False)
    old=state['potential'];old_n=len(old.coefficients)
    if degree+1<old_n:raise ValueError('Cannot remove inherited basis functions.')
    old_opt=state['optimizers'][-1];saved=copy.deepcopy(old_opt.state_dict())
    new=CosinePotential(degree,state['device'])
    with torch.no_grad():new.coefficients[:old_n].copy_(old.coefficients)
    optimizer=torch.optim.Adam(new.parameters(),lr=old_opt.param_groups[0]['lr'])
    for values in saved['state'].values():
        for key in ['exp_avg','exp_avg_sq','max_exp_avg_sq']:
            if key in values:
                padding=values[key].new_zeros(degree+1);padding[:old_n]=values[key];values[key]=padding
    optimizer.load_state_dict(saved)
    state['potential']=new;state['optimizers'][-1]=optimizer;state['degree']=degree
    return dict(expanded=True,old_coefficient_count=old_n,new_coefficient_count=degree+1,
                old_coefficients_copied=True,new_coefficients_initialized_to_zero=True,
                inherited_Adam_moments_padded_with_zero=True)


def set_lr(state,lr):
    for opt in state['optimizers']:
        for group in opt.param_groups:group['lr']=lr


def snapshot(state):
    return cpu_tree(dict(**{k:state[k].state_dict() for k in ['realnet','imagenet','hypernet','potential']},
            a1=state['a1'],a2=state['a2'],fn=state['fn'],degree=state['degree'],
            optimizers=[o.state_dict() for o in state['optimizers']]))


def restore(state,saved,optimizers=True):
    if state['degree']!=saved['degree']:raise ValueError('Basis dimension mismatch.')
    for key in ['realnet','imagenet','hypernet','potential']:state[key].load_state_dict(saved[key])
    with torch.no_grad():
        for key in ['a1','a2','fn']:state[key].copy_(saved[key].to(state[key]))
    if optimizers:
        if len(state['optimizers'])!=len(saved['optimizers']):raise ValueError('Optimizer ownership changed.')
        for opt,row in zip(state['optimizers'],saved['optimizers']):opt.load_state_dict(row)


def coefficients(state):
    return state['potential'].coefficients.detach().cpu().tolist() if state['degree'] is not None else None


def hashes(state):
    saved=snapshot(state)
    return dict(weights=digest({k:v for k,v in saved.items() if k!='optimizers'}),moments=digest([o['state'] for o in saved['optimizers']]))


def parameter_count(state):
    return sum(p.numel() for p in {id(p):p for o in state['optimizers'] for g in o.param_groups for p in g['params']}.values())
