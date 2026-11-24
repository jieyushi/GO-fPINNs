"Define the equation, initial and boundary conditions, and fixed reference or source."
import math
import numpy as np
import torch
from protocol import ALPHA,BETA,GAMMA,T


def exact_numpy(tx):
    tx=np.asarray(tx,dtype=np.float64);t,x=tx[:,0],tx[:,1]
    return t*(np.sin(x)+1j*np.cos(x))


def boundary_target(tx):
    "Boundary target."
    t,x=tx[:,0:1],tx[:,1:2]
    return t*torch.cat([torch.sin(x),torch.cos(x)],1)


def source(tx,potential_sign=1.):
    "Evaluate the fixed source and its local derivative terms."
    t,x=tx[:,0:1],tx[:,1:2];s,c=torch.sin(x),torch.cos(x)
    cap=t.pow(1-ALPHA)/math.gamma(2-ALPHA)
    q=GAMMA*t.pow(3)+(potential_sign*x-BETA)*t
    forcing=torch.cat([-cap*c+q*s,cap*s+q*c],1)
    
    qx=potential_sign*t;qt=3*GAMMA*t.square()+potential_sign*x-BETA
    local_dx=torch.cat([qx*s+q*c,qx*c-q*s],1)
    local_dt=torch.cat([qt*s,qt*c],1)
    return forcing,local_dx,local_dt


def residual_scale(tx):
    "Residual scale."
    t=tx[:,0:1]
    return T*(1+BETA+GAMMA*t.square()+tx[:,1:2].abs())


def independent_points(seed,count):
    rng=np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0,T,count),rng.uniform(0,1,count)])
