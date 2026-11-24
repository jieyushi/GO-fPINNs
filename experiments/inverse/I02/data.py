"Generate reproducible observation pools and stage-specific training samples."
import math
import numpy as np
import torch
import fractional
import reference_solver as reference
from problem import series_field,eta_truth
from protocol import supervised_x,ALPHA


def make_pool(args,device):
    tensor=lambda q:torch.tensor(q,dtype=torch.float64,device=device)
    count=8 if args.smoke else 100
    rng=np.random.default_rng(args.data_seed);x=rng.uniform(0,math.pi,count)
    tx=np.column_stack([np.ones(count),x])
    evaluation=np.random.default_rng(args.reference_seed).uniform([0,0],[1,math.pi],(args.reference_points,2))
    if args.problem=='zero_source':
        field=reference.ensure(args.reference,args.reference_nx,args.reference_nt,'zero_source')
        target=reference.query(field,tx);evaluated=reference.query(field,evaluation);kind=field['metadata']
    else:target=series_field(tx);evaluated=series_field(evaluation);kind=dict(analytic_reference=True,problem='manufactured_series')
    uv=np.column_stack([target.real,target.imag]);noisy=uv.copy()
    if args.noise_level:noisy+=args.noise_level*np.sqrt((uv**2).mean(0))*np.random.default_rng(args.data_seed+11).standard_normal(uv.shape)
    n=args.condition_points
    bt=np.random.default_rng(args.data_seed+1).uniform(0,1,n)
    ix=np.random.default_rng(args.data_seed+2).uniform(0,math.pi,n)
    ex=np.asarray(supervised_x(args.supervised_points))[:,None]
    return dict(obs_tx=tensor(tx),obs_target=tensor(noisy),obs_clean=tensor(uv),
                eval_tx=tensor(evaluation),eval_target=tensor(np.column_stack([evaluated.real,evaluated.imag])),
                eta_x=tensor(ex),eta_target=tensor(eta_truth(ex)),reference_metadata=kind,
                ic_tx=tensor(np.column_stack([np.zeros(n),ix])),
                left_tx=tensor(np.column_stack([bt,np.zeros(n)])),right_tx=tensor(np.column_stack([bt,np.full(n,math.pi)])))


def build(item,pool,args,device):
    M,N=item['M'],item['N'];space=torch.linspace(0,math.pi,M+1,device=device,dtype=torch.float64)[1:-1]
    times=torch.linspace(0,1,N+1,device=device,dtype=torch.float64);xx,tt=torch.meshgrid(space,times,indexing='ij')
    nodal=torch.stack([tt,xx],-1);sig=1-ALPHA/2 if item['scheme']=='fast_l21' else 1.
    xs,ts=torch.meshgrid(space,times[:-1]+sig/N,indexing='ij')
    tx=torch.stack([ts,xs],-1)
    a=torch.tensor(ALPHA,dtype=torch.float64,device=device)
    basis=fractional.quadrature(1/N,device,args.quad_order,args.panel_width,args.low_tail)
    kernel=fractional.build(a,basis,N,item['scheme'],args.history_rule=='direct')
    return dict(M=M,N=N,nodal=nodal,shifted=tx,kernel=kernel,
                obs_tx=pool['obs_tx'][:item['observations']],obs_target=pool['obs_target'][:item['observations']],
                **{k:pool[k] for k in ['eta_x','eta_target','ic_tx','left_tx','right_tx']})
