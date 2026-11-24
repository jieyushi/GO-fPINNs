"Generate an independent numerical reference for the zero-source potential problem."
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import RegularGridInterpolator
from common import atomic_json,output_lock
from problem import eta_truth,forcing_numpy,series_field,audit

FORMAT='I02_direct_Alikhanov_FD_reference_v1'


def direct_weights(N,alpha=.5):
    
    q=np.longdouble;sig=q(1-alpha/2);a=q(alpha);tau=q(1)/N
    j=np.arange(1,N+1,dtype=q);hi=j+sig;lo=j-1+sig
    w0=(hi**(1-a)-lo**(1-a))/(1-a)
    w1=(j-q(.5)+sig)*w0-(hi**(2-a)-lo**(2-a))/(2-a)
    scale=tau**(-a)/math.gamma(1-alpha)
    local=sig**(1-a)*tau**(-a)/math.gamma(2-alpha)
    taps=np.r_[local+scale*w1[0],scale*(w0[:-1]-w1[:-1]+w1[1:])].astype(float)
    return float(local),taps,np.asarray(scale*w1,dtype=float)


def solve(nx=256,nt=1024,problem='zero_source',tol=1e-12,max_iterations=100):
    if min(nx,nt)<4:raise ValueError('At least four intervals in each axis.')
    x=np.linspace(0,math.pi,nx+1);t=np.linspace(0,1,nt+1);h=x[1]-x[0];sig=.75
    interior=x[1:-1];potential=eta_truth(interior)
    field=np.zeros((nt+1,nx+1),dtype=np.complex128);field[0]=np.sin(x);field[:,[0,-1]]=0
    increments=np.empty((nt,nx-1),dtype=np.complex128)
    local,taps,correction=direct_weights(nt);max_error=0.;most=0
    off=.5/h**2
    for n in range(1,nt+1):
        if n==1:c=local;history=np.zeros(nx-1,dtype=complex)
        else:
            c=taps[0]
            history=np.einsum('i,ij->j',taps[1:n][::-1],increments[:n-1],optimize=False)-correction[n-1]*increments[0]
        previous=field[n-1,1:-1];ts=(n-1+sig)/nt
        source=forcing_numpy(np.column_stack([np.full(nx-1,ts),interior]),problem)
        rhs=source+1j*(c/sig*previous-history)
        w=previous.copy()
        for it in range(max_iterations):
            band=np.zeros((3,nx-1),dtype=complex)
            band[0,1:]=off;band[2,:-1]=off
            band[1]=1j*c/sig-2*off+potential-np.abs(w)**2
            new=solve_banded((1,1),band,rhs,check_finite=False)
            error=np.max(np.abs(new-w));w=new
            if error<tol*(1+np.max(np.abs(w))):break
        else:raise RuntimeError('Reference nonlinear solve failed at n=%d'%n)
        delta=w-previous
        lap=(-2*w+np.r_[0,w[:-1]]+np.r_[w[1:],0])/h**2
        residual=1j*(c/sig*delta+history)+.5*lap+(potential-np.abs(w)**2)*w-source
        max_error=max(max_error,float(np.max(np.abs(residual))));most=max(most,it+1)
        field[n,1:-1]=previous+delta/sig;increments[n-1]=field[n,1:-1]-previous
    return dict(x=x,t=t,field=field,metadata=dict(format=FORMAT,problem=problem,nx=nx,nt=nt,
                 space_intervals=True,time_intervals=True,spatial_order=2,
                 history='direct L2-1sigma coefficients; no SOE; full history',
                 nonlinear_max_iterations=most,discrete_residual_max_abs=max_error,
                 analytic_reference=False))


def ensure(path,nx=256,nt=1024,problem='zero_source'):
    path=Path(path)
    with output_lock(path.parent/('.'+path.stem+'_lock')):
        if path.exists():
            data=load(path);m=data['metadata']
            if (m['format'],m['nx'],m['nt'],m['problem'])!=(FORMAT,nx,nt,problem):raise ValueError('Reference configuration changed; use a new reference path.')
            return data
        result=solve(nx,nt,problem);path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+'.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,x=result['x'],t=result['t'],field=result['field'],metadata=json.dumps(result['metadata']))
        temp.replace(path);return result


def load(path):
    with np.load(path,allow_pickle=False) as a:return dict(x=a['x'].copy(),t=a['t'].copy(),field=a['field'].copy(),metadata=json.loads(str(a['metadata'].item())))


def query(data,tx):
    return RegularGridInterpolator((data['t'],data['x']),data['field'],bounds_error=True)(np.asarray(tx,dtype=float))


def convergence(output,nx=128,nt=512):
    
    xe=np.linspace(0,math.pi,65);te=np.linspace(0,1,129)
    tt,xx=np.meshgrid(te,xe,indexing='ij');points=np.column_stack([tt.ravel(),xx.ravel()])
    rows=[]
    for direction,sizes in [('space',[nx//2,nx,2*nx]),('time',[nt//2,nt,2*nt])]:
        fields=[]
        for count in sizes:
            values=solve(count if direction=='space' else nx,count if direction=='time' else nt)
            fields.append(query(values,points))
        d1=float(np.linalg.norm(fields[1]-fields[0])/np.linalg.norm(fields[2]))
        d2=float(np.linalg.norm(fields[2]-fields[1])/np.linalg.norm(fields[2]))
        rows.append(dict(refinement=direction,intervals=sizes,delta1=d1,delta2=d2,observed_order=float(np.log2(d1/d2))))
    result=dict(problem='zero_source',common_evaluation_nodes=[129,65],nx_fixed_for_time=nx,nt_fixed_for_space=nt,
                denominator='finest sequence field on the common evaluation grid',results=rows,
                note='Measured self-refinement, not an analytical-solution error or a fourth-order spatial method.')
    atomic_json(output,result);return result


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--output',default='runs/I02/reference.npz')
    q.add_argument('--nx',type=int,default=256);q.add_argument('--nt',type=int,default=1024)
    q.add_argument('--problem',choices=['zero_source','manufactured_series'],default='zero_source')
    q.add_argument('--audit',action='store_true');q.add_argument('--convergence',action='store_true');a=q.parse_args(argv)
    if a.audit:print(json.dumps(audit(),indent=2));return
    if a.convergence:print(json.dumps(convergence(a.output,a.nx,a.nt),indent=2));return
    r=ensure(a.output,a.nx,a.nt,a.problem);print(json.dumps(r['metadata'],indent=2))


if __name__=='__main__':main()
