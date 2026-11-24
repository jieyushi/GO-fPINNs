"""Forward Case 2; grid entries count INTERVALS, as in the supplied C01."""
import numpy as np

XL, XR, T = -3*np.pi, 3*np.pi, 1.
ALPHA, BETA, GAMMA = .9, .5, 1.
METHODS = ('go', 'fpinn', 'fl1')
LABELS = {'go':'GO-fPINNs','fpinn':'fPINNs (L1)','fl1':'FL1-fPINN (Fast L1)'}


def plan(method='go', smoke=False):
    if method not in METHODS:raise ValueError('Unknown method.')
    if method!='go':
        return [dict(stage=0,phase='single_global',parts=1,M=16 if smoke else 200,
                     N=4 if smoke else 50,lr=1e-3,domains=[[XL,XR]],interfaces=[],parents=[0])]
    counts=[4,8,16,8,4] if smoke else [50,100,200,100,50]
    result=[]
    for sid,(parts,M,lr) in enumerate(zip([1,1,1,2,4],counts,[5e-3,2.5e-3,1.25e-3,6.25e-4,6.25e-4])):
        bounds=np.linspace(XL,XR,parts+1)
        result.append(dict(stage=sid,phase='global_refinement' if sid<3 else 'spatial_decomposition',
                           parts=parts,M=M,N=4 if smoke else 50,lr=lr,
                           domains=[[float(bounds[j]),float(bounds[j+1])] for j in range(parts)],
                           interfaces=[float(v) for v in bounds[1:-1]],
                           parents=[0]*parts if sid<4 else [0,0,1,1]))
    return result


def method_scheme(method):
    return {'go':'fast_l21','fpinn':'l1','fl1':'fast_l1'}[method]


def owner(x, domains):
    """Shared endpoints go to the right region; global right endpoint stays last."""
    return np.searchsorted(np.asarray(domains)[:-1,1],np.asarray(x),side='right')
