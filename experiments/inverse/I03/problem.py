"Define the equation, initial and boundary conditions, and fixed reference or source."
import math
import numpy as np
import torch
from protocol import ALPHA,BETA,GAMMA


def eta_truth(xy):
    xy=np.asarray(xy,dtype=float);return -2*np.exp(-((xy[...,0]-.5)**2+(xy[...,1]-.5)**2)/2)


def eta_truth_torch(xy):return -2*torch.exp(-((xy[:,0:1]-.5).square()+(xy[:,1:2]-.5).square())/2)


def exact_numpy(txy):
    q=np.asarray(txy,dtype=float);t,x,y=q[...,0],q[...,1],q[...,2]
    return t*(np.sin(np.pi*x*y)+1j*np.cos(np.pi*x*y))


def known_values(txy):
    t,x,y=txy[:,0:1],txy[:,1:2],txy[:,2:3];s=math.pi*x*y
    return t*torch.cat([torch.sin(s),torch.cos(s)],1)


def forcing(txy):
    "Forcing."
    t,x,y=txy[:,0:1],txy[:,1:2],txy[:,2:3]
    s=math.pi*x*y;sin,cos=torch.sin(s),torch.cos(s)
    multiplier=-BETA*math.pi**2*(x.square()+y.square())+GAMMA*t.square()+eta_truth_torch(txy[:,1:3])
    local=multiplier*t*torch.cat([sin,cos],1)
    cap=t.pow(1-ALPHA)/math.gamma(2-ALPHA)
    full=local+cap*torch.cat([-cos,sin],1)
    return full,local


def independent_points(seed,count,dimensions=3):
    return np.random.default_rng(seed).uniform(0,1,(count,dimensions))
