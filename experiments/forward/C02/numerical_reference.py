"""Independent 1D reference: implicit L1 time history + centered spatial FD.

This optional solver checks the manufactured-source equation with the
same series-valued Dirichlet data. Main error evaluation uses the analytic field.
Use convergence_check.py before treating a chosen resolution as a paper reference.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gamma as gamma_function
from problem import ml_amplitude,forcing_numpy,audit,SOURCE_MODEL
from protocol import XL,XR,ALPHA,BETA,GAMMA
from common import atomic_json


def solve(M=800,N=2048,alpha=ALPHA,beta=BETA,gamma=GAMMA,tolerance=1e-10,max_newton=20):
    if min(M,N)<2 or not 0<alpha<=1:raise ValueError('Invalid reference grid/order.')
    x=np.linspace(XL,XR,M+1);t=np.linspace(0,1,N+1);h=(XR-XL)/M;tau=1/N
    k=np.arange(N,dtype=float);weights=(k+1)**(1-alpha)-k**(1-alpha)
    if alpha==1:weights[:]=0;weights[0]=1
    c=tau**(-alpha)/gamma_function(2-alpha);d=beta/h**2
    psi=np.empty((N+1,M+1),dtype=np.complex128);psi[0]=np.exp(1j*x)
    boundary=-ml_amplitude(t,alpha);psi[:,0]=boundary;psi[:,-1]=boundary
    increments=np.zeros((N,M-1),dtype=np.complex128)
    maximum_residual=0.;maximum_iterations=0
    try:
        from threadpoolctl import threadpool_limits
        limits=threadpool_limits(limits=1)
    except ImportError:limits=contextlib.nullcontext()
    with limits:
        for n in range(1,N+1):
            memory=weights[1:n][::-1]@increments[:n-1] if n>1 else np.zeros(M-1,dtype=np.complex128)
            history=c*(memory-psi[n-1,1:-1])
            source=forcing_numpy(np.column_stack([np.full(M-1,t[n]),x[1:-1]]),alpha,beta,gamma)
            z=psi[n-1,1:-1].copy()
            def residual(value):
                full=np.concatenate(([boundary[n]],value,[boundary[n]]))
                return 1j*(c*value+history)+d*(full[:-2]-2*value+full[2:])+gamma*np.abs(value)**2*value-source
            for iteration in range(max_newton+1):
                r=residual(z);norm=float(np.max(np.abs(r)))
                if norm<tolerance:break
                if iteration==max_newton:raise ArithmeticError('Reference Newton failed at time step %d: %.3e'%(n,norm))
                u,v=z.real,z.imag;size=2*(M-1)
                band=np.zeros((5,size),dtype=float)
                band[2,0::2]=-2*d+gamma*(3*u*u+v*v)
                band[2,1::2]=-2*d+gamma*(u*u+3*v*v)
                band[1,1::2]=-c+2*gamma*u*v
                band[3,0::2]=c+2*gamma*u*v
                band[0,2:]=d;band[4,:-2]=d
                rhs=np.empty(size);rhs[0::2]=-r.real;rhs[1::2]=-r.imag
                update=solve_banded((2,2),band,rhs,check_finite=False)
                dz=update[0::2]+1j*update[1::2]
                step=1.
                while step>1/128 and np.max(np.abs(residual(z+step*dz)))>norm:step*=.5
                z=z+step*dz
            psi[n,1:-1]=z;increments[n-1]=z-psi[n-1,1:-1]
            maximum_residual=max(maximum_residual,norm);maximum_iterations=max(maximum_iterations,iteration)
    return x,t,psi,dict(M=M,N=N,alpha=alpha,beta=beta,gamma=gamma,
                       method='implicit L1 / centered second-order spatial FD / real block-banded Newton',
                       source='manufactured_series',source_model=SOURCE_MODEL,potential=0.,boundary='prescribed series-valued Dirichlet',
                       nonlinear_tolerance=tolerance,max_nonlinear_residual=maximum_residual,
                       max_Newton_iterations=maximum_iterations,mesh_convergence_certified=False)


class Reference:
    def __init__(self,path,alpha=ALPHA):
        self.path=Path(path)
        with np.load(self.path,allow_pickle=False) as a:
            self.x=a['x'].copy();self.t=a['t'].copy();self.psi=a['psi'].copy()
            self.alpha=float(a['alpha'])
        if abs(self.alpha-alpha)>1e-14:raise ValueError('Reference alpha differs.')
        metadata=json.loads(self.path.with_suffix('.json').read_text())
        if (metadata.get('source_model')!=SOURCE_MODEL or metadata.get('source')!='manufactured_series' or metadata['boundary']!='prescribed series-valued Dirichlet'
                or metadata['beta']!=BETA or metadata['gamma']!=GAMMA):
            raise ValueError('Reference solves another source or boundary problem.')
        if metadata['reference_sha256']!=hashlib.sha256(self.path.read_bytes()).hexdigest():
            raise ValueError('Reference data/metadata hash mismatch.')
        if self.psi.shape!=(len(self.t),len(self.x)) or not np.isfinite(self.psi).all():
            raise ValueError('Invalid reference array.')
        if not np.allclose([self.x[0],self.x[-1],self.t[0],self.t[-1]],[XL,XR,0,1],rtol=0,atol=1e-13):
            raise ValueError('Reference domain differs.')
        self.metadata=metadata
        self.interpolator=RegularGridInterpolator((self.t,self.x),self.psi,bounds_error=True)
    def __call__(self,tx):
        return self.interpolator(np.asarray(tx,dtype=float))


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--output',type=Path,default=Path('runs/C02_gradient_v2/reference/reference.npz'))
    q.add_argument('--space-intervals',type=int,default=800)
    q.add_argument('--time-intervals',type=int,default=2048)
    q.add_argument('--alpha',type=float,default=ALPHA)
    q.add_argument('--smoke',action='store_true')
    a=q.parse_args(argv)
    if a.smoke:a.space_intervals=32;a.time_intervals=32
    if a.output.exists():
        ref=Reference(a.output,a.alpha)
        if ref.metadata['M']!=a.space_intervals or ref.metadata['N']!=a.time_intervals:
            raise ValueError('Existing reference resolution differs; choose a new path.')
        print('Matching independent reference reused.');return
    start=time.perf_counter()
    x,t,psi,meta=solve(a.space_intervals,a.time_intervals,a.alpha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,x=x,t=t,psi=psi,alpha=np.array(a.alpha))
    meta.update(elapsed_seconds=time.perf_counter()-start,smoke_only=a.smoke,
                reference_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                series_audit=audit(a.alpha))
    atomic_json(a.output.with_suffix('.json'),meta)
    print('Independent manufactured-source numerical reference saved: '+str(a.output))


if __name__=='__main__':main()
