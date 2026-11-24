"Define the equation, initial and boundary conditions, and fixed reference or source."
import math
import numpy as np
import torch
from protocol import TRUE


def exact_numpy(tx):
    tx=np.asarray(tx,dtype=np.float64);t,x=tx[:,0],tx[:,1]
    return t**2*(np.sin(2*np.pi*x)+1j*np.cos(2*np.pi*x))


def known_values(tx):
    t,x=tx[:,0:1],tx[:,1:2]
    return t.square()*torch.cat([torch.sin(2*math.pi*x),torch.cos(2*math.pi*x)],1)


def forcing(tx):
    "Forcing."
    qtx=tx.to(torch.float64);t,x=qtx[:,0:1],qtx[:,1:2]
    s,c=torch.sin(2*math.pi*x),torch.cos(2*math.pi*x)
    cap=2*t.pow(2-TRUE['alpha'])/math.gamma(3-TRUE['alpha'])
    q=-4*math.pi**2*TRUE['beta']*t.square()+TRUE['gamma']*t.pow(6)
    source=torch.cat([-cap*c+q*s,cap*s+q*c],1)
    
    local_dx=2*math.pi*q*torch.cat([c,-s],1)
    qt=-8*math.pi**2*TRUE['beta']*t+6*TRUE['gamma']*t.pow(5)
    local_dt=qt*torch.cat([s,c],1)
    return source,local_dt,local_dx


def independent_points(seed,count):
    rng=np.random.default_rng(seed)
    return rng.uniform(0,1,(count,2))
