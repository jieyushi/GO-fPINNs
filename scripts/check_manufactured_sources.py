"""Verify C02/I02 source terms without training or generating experiment results.

NumPy/SciPy checks use independent Caputo quadrature. When PyTorch is available,
small residual/backward checks also run; no optimizer updates are performed.
"""
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx

ROOT=Path(__file__).resolve().parents[1]
C02=ROOT/'experiments/forward/C02'
I02=ROOT/'experiments/inverse/I02'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


sys.path.insert(0,str(C02))
forward=module('c02_source_check',C02/'problem.py')
sys.path.pop(0)
inverse=module('i02_source_check',I02/'problem.py')


def caputo_by_quadrature(t,alpha,regular_derivative):
    # u=s/t; regular_derivative(s)=s**(1-alpha)*dA/ds.
    options=dict(weight='alg',wvar=(alpha-1,-alpha),epsabs=2e-12,epsrel=2e-12)
    real=quad(lambda u:float(np.real(regular_derivative(t*u))),0,1,**options)[0]
    imag=quad(lambda u:float(np.imag(regular_derivative(t*u))),0,1,**options)[0]
    return (real+1j*imag)/math.gamma(1-alpha)


class SourceChecks(unittest.TestCase):
    def test_c02_full_pde_against_caputo_integral(self):
        alpha=.9
        def regular(s):
            return sum((.5j)**k*s**(alpha*(k-1))/math.gamma(alpha*k) for k in range(1,64))
        for t in [.03,.25,.7,1.]:
            derivative=caputo_by_quadrature(t,alpha,regular)
            for x in [-2.7,0.,1.3]:
                tx=np.array([[t,x]])
                psi=forward.reference_field(tx)[0]
                lhs=1j*np.exp(1j*x)*derivative-.5*psi+abs(psi)**2*psi
                self.assertLess(abs(lhs-forward.forcing_numpy(tx)[0]),2e-10)

    def test_i02_full_pde_against_caputo_integral(self):
        def regular(s):
            amplitude=erfcx(1.5j*np.sqrt(s))
            return -2.25*np.sqrt(s)*amplitude-1.5j/np.sqrt(np.pi)
        for t in [.03,.25,.7,1.]:
            derivative=caputo_by_quadrature(t,.5,regular)
            for x in [.2,1.,np.pi/2]:
                tx=np.array([[t,x]])
                psi=inverse.series_field(tx)[0]
                lhs=1j*np.sin(x)*derivative-.5*psi-abs(psi)**2*psi-np.cos(x)**2*psi
                self.assertLess(abs(lhs-inverse.forcing_numpy(tx)[0]),2e-10)

    def test_c02_fixed_source_spatial_derivative(self):
        tx=np.array([[.1,-2.],[.5,.7],[1.,2.8]])
        step=np.array([0.,1e-5])
        finite=(forward.forcing_numpy(tx+step)-forward.forcing_numpy(tx-step))/(2e-5)
        np.testing.assert_allclose(forward.forcing_x_numpy(tx),finite,rtol=2e-8,atol=2e-10)

    def test_initial_and_boundary_data(self):
        x=np.linspace(-3*np.pi,3*np.pi,11)
        tx=np.column_stack([np.zeros_like(x),x])
        np.testing.assert_allclose(forward.reference_field(tx),np.exp(1j*x),atol=1e-14)
        np.testing.assert_allclose(forward.forcing_numpy(tx),0.,atol=1e-14)
        tx=np.array([[t,x] for t in [0.,.5,1.] for x in [0.,np.pi]])
        np.testing.assert_allclose(inverse.series_field(tx),0.,atol=1e-14)
        np.testing.assert_allclose(inverse.forcing_numpy(tx),0.,atol=1e-14)
        np.testing.assert_allclose(inverse.forcing_numpy(np.array([[1.,1.]]),'zero_source'),0.)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable: residual/backward check skipped')
    def test_c02_residual_and_backward(self):
        code=r'''
import numpy as np
import torch
import backend as b
from train_largescale import parse_args
from problem import forcing_numpy
from protocol import XL,XR
torch.set_num_threads(1)
a=parse_args(['--device','cpu','--network-dtype','float64','--kernel-mode','stable64'])
d=b.build_data([XL,XR],4,6,a,torch.device('cpu'),torch.float64)
w=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
def field(tx):
    t,x=tx[:,0:1],tx[:,1:2]
    return w*torch.cat([(1+t)**2*torch.sin(x),(1+t)**3*torch.cos(x)],1)
forced=b.discrete_residual(field,d,a,True)
unforced=b.discrete_residual(field,dict(d,source=torch.zeros_like(d['source'])),a,True)
f=d['source'].view(3,6,2)
torch.testing.assert_close(forced['residual']-unforced['residual'],-f)
fx=torch.stack([-f[...,1],f[...,0]],-1)
torch.testing.assert_close(forced['spatial_gradient']-unforced['spatial_gradient'],-fx)
torch.testing.assert_close(forced['temporal_gradient']-unforced['temporal_gradient'],-b.residual_time_difference(f,d['tau']))
loss=sum(forced[k].square().mean() for k in ['residual','spatial_gradient','temporal_gradient'])
loss.backward()
assert w.grad is not None and torch.isfinite(w.grad)
assert not d['source'].requires_grad
'''
        subprocess.run([sys.executable,'-c',code],cwd=C02,check=True)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable: residual/backward check skipped')
    def test_i02_residual_and_backward(self):
        code=r'''
import numpy as np
import torch
from unittest.mock import patch
import data,physics
from train_potential import parse_args
from protocol import plan
from problem import forcing,forcing_numpy
torch.set_num_threads(1)
a=parse_args(['--device','cpu','--network-dtype','float64','--smoke'])
assert a.problem=='manufactured_series'
pool=data.make_pool(a,torch.device('cpu'))
assert pool['reference_metadata']['analytic_reference']
d=data.build(plan(a.method,True)[0],pool,a,torch.device('cpu'))
w=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
eta=torch.tensor(-.7,dtype=torch.float64,requires_grad=True)
def field(tx):
    t,x=tx[:,0:1],tx[:,1:2]
    return w*torch.cat([(1+t)**2*torch.sin(x),(1+t)**3*torch.sin(x)],1)
def potential(x):return eta*torch.cos(x).square()
rows=slice(0,d['M']-1)
forced=physics.residual_fields(field,potential,d,a,rows)
with patch.object(physics,'forcing',lambda tx,mode:torch.zeros_like(tx)):
    unforced=physics.residual_fields(field,potential,d,a,rows)
tx=d['shifted'].reshape(-1,2).detach().requires_grad_(True)
f=forcing(tx,a.problem)
truth=forcing_numpy(tx.detach().numpy())
np.testing.assert_allclose(f.detach().numpy(),np.column_stack([truth.real,truth.imag]),rtol=2e-11,atol=2e-12)
fx=torch.stack([torch.autograd.grad(f[:,c].sum(),tx,retain_graph=True)[0][:,1] for c in [0,1]],1)
torch.testing.assert_close(forced['residual']-unforced['residual'],-f.view_as(forced['residual']))
torch.testing.assert_close(forced['space_gradient']-unforced['space_gradient'],-fx.view_as(forced['space_gradient']))
torch.testing.assert_close(forced['time_gradient']-unforced['time_gradient'],-physics.time_difference(f.view_as(forced['residual']),1/d['N']))
loss=sum(forced[k].square().mean() for k in ['residual','space_gradient','time_gradient'])
loss.backward()
assert w.grad is not None and torch.isfinite(w.grad)
assert eta.grad is not None and torch.isfinite(eta.grad)
assert torch.autograd.grad(f.sum(),eta,allow_unused=True)[0] is None
'''
        subprocess.run([sys.executable,'-c',code],cwd=I02,check=True)


if __name__=='__main__':
    unittest.main(verbosity=2)
