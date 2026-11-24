"""Manufactured Mittag-Leffler wave, prescribed data, and fixed source.

For B(t)=E_alpha(i*t**alpha/2), Caputo(B)=i*B/2. The reference
psi_*=exp(i*x)*B solves i*Caputo(psi)+beta*psi_xx+gamma*|psi|^2*psi=f
with f=(-1/2-beta+gamma*|B|^2)*psi_*. At beta=1/2 and gamma=1 this
is (|B|^2-1)*psi_*. The source never depends on network predictions.
"""
import numpy as np
from scipy.special import gammaln
from protocol import ALPHA,BETA,GAMMA

SOURCE_MODEL='C02_manufactured_MittagLeffler_v1'


def ml_amplitude(t,alpha=ALPHA,tol=1e-15,max_terms=128):
    t=np.asarray(t,dtype=np.float64)
    if np.any(t<0):raise ValueError('Negative time.')
    z=.5j*t**alpha
    result=np.ones_like(z,dtype=np.complex128);term=result.copy()
    for k in range(1,max_terms):
        term=term*z*np.exp(gammaln(alpha*(k-1)+1)-gammaln(alpha*k+1))
        result=result+term
        if k>4 and np.max(np.abs(term),initial=0)<tol:return result
    raise ArithmeticError('Mittag-Leffler series did not converge.')


def reference_field(tx,alpha=ALPHA):
    tx=np.asarray(tx,dtype=np.float64)
    return np.exp(1j*tx[...,1])*ml_amplitude(tx[...,0],alpha)


def forcing_numpy(tx,alpha=ALPHA,beta=BETA,gamma=GAMMA):
    """Known source, evaluated from the prescribed field, in (t,x) order."""
    tx=np.asarray(tx,dtype=np.float64)
    amplitude=ml_amplitude(tx[...,0],alpha)
    return (-.5-beta+gamma*np.abs(amplitude)**2)*np.exp(1j*tx[...,1])*amplitude


def forcing_x_numpy(tx,alpha=ALPHA,beta=BETA,gamma=GAMMA):
    """Exact spatial derivative of the fixed source: f_x=i*f."""
    return 1j*forcing_numpy(tx,alpha,beta,gamma)


def audit(alpha=ALPHA):
    t=np.linspace(0,1,1001);tx=np.column_stack([t,np.zeros_like(t)])
    amplitude=ml_amplitude(t,alpha);source=forcing_numpy(tx,alpha)
    return dict(alpha=alpha,beta=BETA,gamma=GAMMA,source_model=SOURCE_MODEL,
                equation='i*Caputo(psi)+0.5*psi_xx+|psi|^2*psi=f(t,x)',
                reference_field='exp(i*x)*E_alpha(i*t^alpha/2)',
                source_formula='(|E_alpha(i*t^alpha/2)|^2-1)*exp(i*x)*E_alpha(i*t^alpha/2)',
                source_depends_on_network=False,
                amplitude_squared_at_t1=float(abs(amplitude[-1])**2),
                absolute_source_at_t1=float(abs(source[-1])),maximum_source=float(abs(source).max()),
                series_is_exact_for_prescribed_source=True,
                boundary_implementation='prescribed Dirichlet values from series at both endpoints; no extra periodic derivative constraint',
                reference_policy='analytic manufactured field; optional numerical reference solves the same nonzero-source PDE')


if __name__=='__main__':
    import json
    print(json.dumps(audit(),indent=2))
