"Generate reproducible observation pools and stage-specific training samples."
import numpy as np
import torch
import fractional
from protocol import ALPHA,supervised_xy
from problem import eta_truth,known_values,independent_points


def make_pool(args,device):
    tensor=lambda q:torch.tensor(q,dtype=torch.float64,device=device)
    count=10 if args.smoke else 100;n=args.condition_points
    xy=np.random.default_rng(args.data_seed).uniform(0,1,(max(n,count),2))
    tx=np.column_stack([np.ones(count),xy[:count]]);clean=known_values(tensor(tx));target=clean.clone()
    if args.noise_level:
        noise=np.random.default_rng(args.data_seed+11).standard_normal((count,2))
        target=target+args.noise_level*tensor(noise)*clean.square().mean(0).sqrt()
    t=np.random.default_rng(args.data_seed+1).uniform(0,1,n);base=xy[:n]
    faces=[np.column_stack([t,np.zeros(n),base[:,1]]),np.column_stack([t,np.ones(n),base[:,1]]),
           np.column_stack([t,base[:,0],np.zeros(n)]),np.column_stack([t,base[:,0],np.ones(n)])]
    supervised=supervised_xy(args.supervised_points)
    evaluation=tensor(independent_points(args.reference_seed,args.reference_points))
    axis=np.linspace(0,1,args.eta_eval_nodes);xx,yy=np.meshgrid(axis,axis,indexing='ij');eta_xy=np.column_stack([xx.ravel(),yy.ravel()])
    return dict(obs_tx=tensor(tx),obs_target=target,obs_clean=clean,
                eta_xy=tensor(supervised),eta_target=tensor(eta_truth(supervised)[:,None]),
                ic_tx=tensor(np.column_stack([np.zeros(n),base])),bc_tx=tensor(np.concatenate(faces)),
                eval_tx=evaluation,eval_target=known_values(evaluation),eta_eval_xy=tensor(eta_xy))


def build(item,pool,args,device):
    M,N=item['M'],item['N'];dtype=torch.float64
    axis=torch.linspace(0,1,M+1,dtype=dtype,device=device)[1:-1]
    xx,yy=torch.meshgrid(axis,axis,indexing='ij');xy=torch.stack([xx.ravel(),yy.ravel()],1);ns=len(xy)
    time=torch.linspace(0,1,N+1,dtype=dtype,device=device)
    nodal=torch.cat([time[None,:,None].expand(ns,-1,-1),xy[:,None,:].expand(-1,N+1,-1)],2)
    sigma=1-ALPHA/2 if item['scheme']=='fast_l21' else 1.
    shifted=torch.cat([(time[:-1]+sigma/N)[None,:,None].expand(ns,-1,-1),xy[:,None,:].expand(-1,N,-1)],2)
    alpha=torch.tensor(ALPHA,dtype=dtype,device=device)
    basis=fractional.quadrature(1/N,device,args.quad_order,args.panel_width,args.low_tail)
    kernel=fractional.build(alpha,basis,N,item['scheme'],args.history_rule=='direct')
    return dict(M=M,N=N,spatial_sites=ns,nodal=nodal,shifted=shifted,kernel=kernel,
                obs_tx=pool['obs_tx'][:item['observations']],obs_target=pool['obs_target'][:item['observations']],
                **{k:pool[k] for k in ['eta_xy','eta_target','ic_tx','bc_tx']})
