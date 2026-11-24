"Construct solution and inverse-parameter models and transfer inherited optimizer states."
import types
import torch
from original_components import FastNetwork,HyperNet
from common import cpu_tree,digest
from protocol import TRUE,unknown_names


def _plain(self,x):return torch.tanh(self.linear(x))


def create(args,device):
    dtype=torch.float32 if args.network_dtype=='float32' else torch.float64
    a1=torch.tensor([.1],device=device,dtype=dtype,requires_grad=args.method!='fl1')
    a2=torch.tensor([.1],device=device,dtype=dtype,requires_grad=args.method!='fl1')
    fn=torch.tensor([10.],device=device,dtype=dtype)
    real=FastNetwork(args.extra_hidden_layers,2,args.width,torch.tanh,a1,a2,fn).to(device=device,dtype=dtype)
    imag=FastNetwork(args.extra_hidden_layers,2,args.width,torch.tanh,a1,a2,fn).to(device=device,dtype=dtype)
    hyper=HyperNet(.6,1.5).to(device=device,dtype=dtype)
    names=unknown_names(args.unknown);initial=dict(TRUE)
    initial.update(zip(names,args.initial_values))
    physical={k:torch.tensor(float(initial[k]),dtype=torch.float64,device=device,requires_grad=k in names) for k in TRUE}
    if args.method=='fl1':
        for net in [real,imag]:
            for block in net.layers:block.forward=types.MethodType(_plain,block)
        hyper.requires_grad_(False)
    opts=[torch.optim.Adam(real.parameters(),lr=1e-3),torch.optim.Adam(imag.parameters(),lr=1e-3)]
    if args.method!='fl1':
        opts[0].add_param_group({'params':[a1,a2]})
        if args.shared_update=='legacy_double':opts[1].add_param_group({'params':[a1,a2]})
        if args.freeze_hypernet:hyper.requires_grad_(False)
        else:opts.append(torch.optim.Adam(hyper.parameters(),lr=1e-3))
    
    opts.append(torch.optim.Adam([physical[k] for k in names],lr=1e-3))
    return dict(realnet=real,imagenet=imag,hypernet=hyper,a1=a1,a2=a2,fn=fn,physical=physical,
                optimizers=opts,device=device,dtype=dtype)


def predict(state,tx):
    
    q=tx.to(dtype=state['dtype'])
    return torch.cat([state['realnet'](q),state['imagenet'](q)],1).to(torch.float64)


def values(state):return {k:float(v.detach()) for k,v in state['physical'].items()}


def project_alpha(state,args):
    if state['physical']['alpha'].requires_grad:
        with torch.no_grad():state['physical']['alpha'].clamp_(args.alpha_min,args.alpha_max)


def set_lr(state,lr):
    for opt in state['optimizers']:
        for group in opt.param_groups:group['lr']=lr


def snapshot(state):
    return cpu_tree(dict(realnet=state['realnet'].state_dict(),imagenet=state['imagenet'].state_dict(),
        hypernet=state['hypernet'].state_dict(),a1=state['a1'],a2=state['a2'],fn=state['fn'],
        physical=state['physical'],optimizers=[o.state_dict() for o in state['optimizers']]))


def restore(state,saved,optimizers=True):
    for key in ['realnet','imagenet','hypernet']:state[key].load_state_dict(saved[key])
    with torch.no_grad():
        for key in ['a1','a2','fn']:state[key].copy_(saved[key].to(state[key]))
        for key in TRUE:state['physical'][key].copy_(saved['physical'][key].to(state['physical'][key]))
    if optimizers:
        if len(state['optimizers'])!=len(saved['optimizers']):raise ValueError('Optimizer ownership changed.')
        for opt,row in zip(state['optimizers'],saved['optimizers']):opt.load_state_dict(row)


def hashes(state):
    saved=snapshot(state)
    return dict(weights=digest({k:v for k,v in saved.items() if k!='optimizers'}),
                moments=digest([o['state'] for o in saved['optimizers']]))


def parameter_count(state):
    return sum(p.numel() for p in {id(p):p for o in state['optimizers'] for g in o.param_groups for p in g['params']}.values())
