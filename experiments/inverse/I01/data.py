"Generate reproducible observation pools and stage-specific training samples."
import numpy as np
import torch
import fractional
from problem import known_values


def make_pool(args,device):
    n=7 if args.smoke else 50
    rng=np.random.default_rng(args.data_seed)
    x=rng.uniform(0,1,n)
    tensor=lambda a:torch.tensor(a,dtype=torch.float64,device=device)
    obs=tensor(np.column_stack([np.ones(n),x]));clean=known_values(obs)
    noisy=clean.clone()
    if args.noise_level:
        noise=np.random.default_rng(args.data_seed+11).standard_normal((n,2))
        scale=clean.square().mean(0).sqrt()
        noisy=noisy+args.noise_level*tensor(noise)*scale
    count=args.condition_points
    bt=np.random.default_rng(args.data_seed+1).uniform(0,1,count)
    ix=np.random.default_rng(args.data_seed+2).uniform(0,1,count)
    return dict(obs_tx=obs,obs_target=noisy,obs_clean=clean,
                ic_tx=tensor(np.column_stack([np.zeros(count),ix])),
                left_tx=tensor(np.column_stack([bt,np.zeros(count)])),
                right_tx=tensor(np.column_stack([bt,np.ones(count)])))


def mesh(space,times):
    xx,tt=np.meshgrid(space,times,indexing='ij')
    return np.column_stack([tt.ravel(),xx.ravel()])


def build(item,pool,args,device):
    M,N=item['M'],item['N'];tau=1/N
    axis=np.linspace(0,1,M+1)[1:-1];times=np.linspace(0,1,N+1)
    tensor=lambda a:torch.tensor(a,dtype=torch.float64,device=device)
    return dict(M=M,N=N,tau=tau,scheme=item['scheme'],space=tensor(axis),times=tensor(times),
                nodal=tensor(mesh(axis,times)).view(M-1,N+1,2),
                basis=fractional.quadrature(tau,device,args.quad_order,args.panel_width,args.low_tail),
                obs_tx=pool['obs_tx'][:item['observations']],obs_target=pool['obs_target'][:item['observations']],
                ic_tx=pool['ic_tx'],left_tx=pool['left_tx'],right_tx=pool['right_tx'])


def shifted(data,alpha,rows):
    sigma=1-alpha/2 if data['scheme']=='fast_l21' else alpha.new_tensor(1.)
    times=data['times'][:-1]+sigma*data['tau'];x=data['space'][rows]
    tt=times[None,:].expand(len(x),-1);xx=x[:,None].expand(-1,len(times))
    tx=torch.stack([tt.reshape(-1),xx.reshape(-1)],1)
    return tx.requires_grad_(True)
