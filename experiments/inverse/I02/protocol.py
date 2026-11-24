"Define stage grids, parameter settings, observation counts, and learning rates."
import math
ALPHA=.5
BETA=.5
GAMMA=-1.
XL,XR,T=0.,math.pi,1.
METHODS=('fl1','go_multi')
PROBLEMS=('manufactured_series','zero_source')
LABELS={'fl1':'FL1-fPINN','go_multi':'GO-fPINNs'}


def plan(method,smoke=False):
    if method not in METHODS:raise ValueError('Unknown method.')
    sizes=[10,20,30] if method=='go_multi' else [30]
    observations=[25,50,100] if method=='go_multi' else [100]
    if smoke:sizes=[3,4,5] if method=='go_multi' else [5];observations=[4,6,8] if method=='go_multi' else [8]
    return [dict(stage=s,M=n,N=n,observations=observations[s],degree=s if method=='go_multi' else None,
                 lr=.001/2**s if method=='go_multi' else .0005,
                 scheme='fast_l21' if method=='go_multi' else 'fast_l1') for s,n in enumerate(sizes)]


def supervised_x(count):
    if count==2:return [0.,math.pi]
    if count==3:return [0.,math.pi/2,math.pi]
    raise ValueError('Use two or three supervised potential points.')
