"Define stage grids, parameter settings, observation counts, and learning rates."
XL, XR, T = 0., 1., 100.
ALPHA, BETA, GAMMA = .9, 1., 1.


def plan(smoke=False):
    result=[]
    for sid in range(9):
        parts=1 if sid<5 else 2**(sid-4)
        n=(50*2**sid) if sid<5 else 800//parts
        if smoke:n=2 if sid<5 else 32//parts
        bounds=[T*j/parts for j in range(parts+1)]
        result.append(dict(stage=sid,phase='temporal_refinement' if sid<5 else 'time_windows',
                           parts=parts,M=4 if smoke else 50,N=n,total_N=n*parts,
                           tau=T/(n*parts),lr=5e-3/2**sid if sid<5 else 1e-4,
                           windows=[[bounds[j],bounds[j+1]] for j in range(parts)],
                           parents=[0]*parts if sid<6 else [j//2 for j in range(parts)]))
    return result
