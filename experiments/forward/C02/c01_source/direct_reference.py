import torch

def sigma_of(a,scheme):
    return 1-a/2

def direct_caputo(values,alpha,tau,scheme='l21'):
    """Analytic integrals of the piecewise quadratic/linear interpolant.
    Rebuilt every call: alpha, sigma, lgamma, all history weights retain autograd.
    O(N^2) work; chosen deliberately for trustworthy alpha inversion.
    """
    v=values.to(torch.float64)
    a=torch.as_tensor(alpha,dtype=v.dtype,device=v.device).reshape(())
    if not 0<float(a.detach())<1: raise ValueError('Require 0<alpha<1.')
    d=v[...,1:]-v[...,:-1];N=d.shape[-1]
    sig=sigma_of(a,scheme); scale=tau**(-a)*torch.exp(-torch.lgamma(1-a))
    out=[]
    for n in range(1,N+1):
        val=sig**(1-a)/(1-a)*d[...,n-1]
        if n>1:
            j=torch.arange(1,n,dtype=v.dtype,device=v.device)
            s=n-1+sig
            hi=s-j+1;lo=s-j
            w0=(hi**(1-a)-lo**(1-a))/(1-a)
            val=val+(d[...,:n-1]*w0).sum(-1)
            if scheme=='l21':
                w1=(s-j+0.5)*w0-(hi**(2-a)-lo**(2-a))/(2-a)
                val=val+((d[...,1:n]-d[...,:n-1])*w1).sum(-1)
        out.append(val*scale)
    return torch.stack(out,dim=-1)
