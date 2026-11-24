"Define stage grids, parameter settings, observation counts, and learning rates."
import numpy as np
ALPHA=.9
BETA=1.
GAMMA=1.
XL,XR,T=0.,1.,1.
METHODS=('fl1','go_multi')
LABELS={'fl1':'FL1-fPINN','go_multi':'GO-fPINNs'}


def plan(method,smoke=False):
    if method not in METHODS:raise ValueError('Unknown method.')
    grid=[6,8,10,12] if method=='go_multi' else [12]
    observations=[10,25,50,100] if method=='go_multi' else [100]
    if smoke:grid=[3,4,5,6] if method=='go_multi' else [6];observations=[4,6,8,10] if method=='go_multi' else [10]
    return [dict(stage=s,M=n,N=n,observations=observations[s],degree=s if method=='go_multi' else None,
                 lr=.001/2**s if method=='go_multi' else .0005,
                 scheme='fast_l21' if method=='go_multi' else 'fast_l1') for s,n in enumerate(grid)]


def supervised_xy(count):
    if count not in [4,9]:raise ValueError('Use four or nine eta points.')
    axis=np.linspace(0,1,2 if count==4 else 3)
    xx,yy=np.meshgrid(axis,axis,indexing='ij');return np.column_stack([xx.ravel(),yy.ravel()])
