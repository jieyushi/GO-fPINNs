"Construct solution and inverse-parameter models and transfer inherited optimizer states."
import types
import torch
from torch import nn
from original_components import FastNetwork,HyperNet,EtaBasis2D
from common import cpu_tree,digest


def _plain(self,x):return torch.tanh(self.linear(x))


def create(args,device,degree=0):
    dtype=torch.float32 if args.network_dtype=='float32' else torch.float64;go=args.method=='go_multi'
    a1=torch.tensor([args.activation_initial],device=device,dtype=dtype,requires_grad=go)
    a2=torch.tensor([args.activation_initial],device=device,dtype=dtype,requires_grad=go)
    fn=torch.tensor([args.activation_scale],device=device,dtype=dtype)
    real=FastNetwork(args.extra_hidden_layers,3,args.width,torch.tanh,a1,a2,fn).to(device=device,dtype=dtype)
    imag=FastNetwork(args.extra_hidden_layers,3,args.width,torch.tanh,a1,a2,fn).to(device=device,dtype=dtype)
    hyper=HyperNet().to(device=device,dtype=dtype)
    if go:potential=EtaBasis2D(degree,degree,device=device,dtype=dtype)
    else:
        potential=nn.Sequential(nn.Linear(2,20),nn.Tanh(),nn.Linear(20,20),nn.Tanh(),nn.Linear(20,20),nn.Tanh(),nn.Linear(20,1)).to(device=device,dtype=dtype)
        for net in [real,imag]:
            for block in net.layers:block.forward=types.MethodType(_plain,block)
        hyper.requires_grad_(False)
    groups=[dict(params=list(real.parameters())+list(imag.parameters()),lr=.001),
            dict(params=list(potential.parameters()),lr=.001)]
    if go:groups.extend([dict(params=[a1],lr=.0001),dict(params=[a2],lr=.0001)])
    opts=[torch.optim.Adam(groups)]
    if go:
        if args.freeze_hypernet:hyper.requires_grad_(False)
        else:opts.append(torch.optim.Adam(hyper.parameters(),lr=.001))
    return dict(realnet=real,imagenet=imag,hypernet=hyper,potential=potential,a1=a1,a2=a2,fn=fn,
                optimizers=opts,device=device,dtype=dtype,degree=degree if go else None)


def predict(state,txy):
    q=txy.to(state['dtype'])
    return torch.cat([state['realnet'](q),state['imagenet'](q)],1).to(torch.float64)


def eta(state,xy):
    xy=xy.to(state['dtype'])
    value=state['potential'](xy[:,0:1],xy[:,1:2]) if state['degree'] is not None else state['potential'](xy)
    return value.to(torch.float64)


def enrich(state,degree):
    if state['degree'] is None or state['degree']==degree:return dict(expanded=False)
    old=state['potential'];old_shape=tuple(old.c.shape)
    if degree<state['degree']:raise ValueError('Cannot shrink inherited basis.')
    new=EtaBasis2D.from_prev(old,degree,degree)
    opt=state['optimizers'][0]
    saved=opt.state.pop(old.c,{})
    
    for name in ['exp_avg','exp_avg_sq','max_exp_avg_sq']:
        if name in saved:
            padded=torch.zeros_like(new.c);padded[:old_shape[0],:old_shape[1]].copy_(saved[name]);saved[name]=padded
    opt.param_groups[1]['params']=[new.c]
    if saved:opt.state[new.c]=saved
    state['potential']=new;state['degree']=degree
    return dict(expanded=True,old_shape=list(old_shape),new_shape=list(new.c.shape),
                coefficient_top_left_block_copied=True,Adam_top_left_blocks_copied=True,new_entries_zero=True)


def set_lr(state,lr,args):
    main=state['optimizers'][0]
    for i,group in enumerate(main.param_groups):group['lr']=lr*(args.activation_lr_ratio if i>=2 else 1.)
    for opt in state['optimizers'][1:]:
        for group in opt.param_groups:group['lr']=lr*args.hyper_lr_ratio


def coefficients(state):return state['potential'].c.detach().cpu().tolist() if state['degree'] is not None else None


def snapshot(state):
    return cpu_tree(dict(**{k:state[k].state_dict() for k in ['realnet','imagenet','hypernet','potential']},
              a1=state['a1'],a2=state['a2'],fn=state['fn'],degree=state['degree'],optimizers=[o.state_dict() for o in state['optimizers']]))


def restore(state,saved,optimizers=True):
    if state['degree']!=saved['degree']:raise ValueError('Basis mismatch.')
    for key in ['realnet','imagenet','hypernet','potential']:state[key].load_state_dict(saved[key])
    with torch.no_grad():
        for key in ['a1','a2','fn']:state[key].copy_(saved[key].to(state[key]))
    if optimizers:
        if len(state['optimizers'])!=len(saved['optimizers']):raise ValueError('Optimizer ownership changed.')
        for opt,row in zip(state['optimizers'],saved['optimizers']):opt.load_state_dict(row)


def hashes(state):
    saved=snapshot(state)
    return dict(weights=digest({k:v for k,v in saved.items() if k!='optimizers'}),moments=digest([o['state'] for o in saved['optimizers']]))


def parameter_count(state):return sum(p.numel() for p in {id(p):p for o in state['optimizers'] for g in o.param_groups for p in g['params']}.values())
