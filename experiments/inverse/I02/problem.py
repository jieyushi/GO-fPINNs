"Define the equation, initial and boundary conditions, and fixed reference or source."
import math
import numpy as np
from scipy.special import erfcx


def eta_truth(x):return -np.cos(np.asarray(x,dtype=float))**2


def series_field(tx):
    tx=np.asarray(tx,dtype=float);t,x=tx[...,0],tx[...,1]
    
    return np.sin(x)*erfcx(1.5j*np.sqrt(t))


def series_forcing_numpy(tx):
    tx=np.asarray(tx,dtype=float);field=series_field(tx)
    temporal=erfcx(1.5j*np.sqrt(tx[...,0]))
    return np.sin(tx[...,1])**2*(1-np.abs(temporal)**2)*field


def forcing_numpy(tx,problem='manufactured_series'):
    if problem not in ('zero_source','manufactured_series'):raise ValueError('Unknown problem mode.')
    return np.zeros(np.asarray(tx).shape[:-1],dtype=complex) if problem=='zero_source' else series_forcing_numpy(tx)


def forcing(tx,problem='manufactured_series'):
    """Fixed known f; depends on coordinates only, never on the inferred eta."""
    import torch
    if problem not in ('zero_source','manufactured_series'):raise ValueError('Unknown problem mode.')
    if problem=='zero_source':return torch.zeros((len(tx),2),dtype=torch.float64,device=tx.device)
    
    t,x=tx[:,0:1],tx[:,1:2];z=torch.complex(torch.zeros_like(t),-1.5*torch.sqrt(t))
    result=torch.ones_like(z);power=torch.ones_like(z)
    for k in range(1,65):
        power=power*z;result=result+power/math.gamma(k*.5+1)
    source=torch.sin(x).pow(3)*(1-result.abs().square())*result
    return torch.cat([source.real,source.imag],1)


def series_values_torch(tx):
    """Differentiable fixed manufactured field; same converged 64-term series."""
    import torch
    t,x=tx[:,0:1],tx[:,1:2]
    z=torch.complex(torch.zeros_like(t),-1.5*torch.sqrt(t))
    temporal=torch.ones_like(z);power=torch.ones_like(z)
    for k in range(1,65):
        power=power*z;temporal=temporal+power/math.gamma(k*.5+1)
    field=torch.sin(x)*temporal
    return torch.cat([field.real,field.imag],1)


def manufactured_local(tx):
    """Legacy local target; the current GE objective differentiates the full residual.

    The stated field has psi_xx=-psi and eta_*=-cos(x)^2. No numerical
    zero-source reference, trainable potential, or full forcing is differentiated.
    """
    import torch
    field=series_values_torch(tx)
    potential=-torch.cos(tx[:,1:2]).square()
    return (-.5-field.square().sum(1,keepdim=True)+potential)*field


def initial_values(tx):
    import torch
    x=tx[:,1:2];return torch.cat([torch.sin(x),torch.zeros_like(x)],1)


def audit():
    rows=[]
    for t in [.25,.5,1.]:
        tx=np.array([[t,math.pi/2]]);v=series_field(tx)[0]
        rows.append(dict(x=math.pi/2,t=t,temporal_modulus_squared=float(abs(v)**2),
                         zero_source_continuous_residual_abs=float(abs(series_forcing_numpy(tx)[0]))))
    return dict(default_problem='manufactured_series',source_depends_on_trainable_potential=False,
                series_is_exact_for_manufactured_source=True,series_is_exact_for_zero_source=False,
                residual='sin(x)^2 * (1 - |E_(1/2)(-1.5 i sqrt(t))|^2) * psi',checks=rows,
                zero_source='f=0; use independent numerical field data, not the stated series as exact',
                manufactured_series='preserve stated series and eta; use its fixed nonzero manufactured source')
