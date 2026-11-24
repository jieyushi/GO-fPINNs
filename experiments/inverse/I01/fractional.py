"Differentiable sum-of-exponentials Caputo history and Alikhanov coefficients."
import math
import numpy as np
import torch


def quadrature(tau,device,order=8,panel_width=1.,tail=1e-6):
    
    lo,hi=math.log(tail),math.log(60/(.5*tau))
    count=math.ceil((hi-lo)/panel_width);edges=np.linspace(lo,hi,count+1)
    z,w=np.polynomial.legendre.leggauss(order)
    nodes=[];weights=[]
    for left,right in zip(edges[:-1],edges[1:]):
        nodes.extend((.5*(right-left)*z+.5*(right+left)).tolist())
        weights.extend((.5*(right-left)*w).tolist())
    return dict(log_nodes=torch.tensor(nodes,dtype=torch.float64,device=device),
                log_weights=torch.tensor(weights,dtype=torch.float64,device=device),tail=tail,tau=tau)


def soe(alpha,basis):
    a=alpha.to(torch.float64).reshape(());z=basis['log_nodes'];w=basis['log_weights'];tail=basis['tail']
    factor=torch.sin(math.pi*a)/math.pi
    
    tail_node=a/(a+1)*tail
    tail_weight=factor*torch.exp(a*math.log(tail))/a
    nodes=torch.cat([tail_node[None],torch.exp(z)])
    weights=torch.cat([tail_weight[None],factor*w*torch.exp(a*z)])
    return nodes,weights


def _ab(q,sigma):
    "Ab."
    small=q<.05;safe=torch.where(small,torch.ones_like(q),q)
    aa=.5*torch.exp(-sigma*safe)*((2+safe)-(2+3*safe)*torch.exp(-safe))/safe.square()
    bb=.5*torch.exp(-sigma*safe)*((safe-2)+(safe+2)*torch.exp(-safe))/safe.square()
    
    z=torch.where(small,q,torch.zeros_like(q));pa=torch.zeros_like(z);pb=torch.zeros_like(z)
    for j in range(2,15):
        pa=pa-(2*(-1)**j/math.factorial(j)+3*(-1)**(j-1)/math.factorial(j-1))*z.pow(j-2)
        pb=pb+(2*(-1)**j/math.factorial(j)+(-1)**(j-1)/math.factorial(j-1))*z.pow(j-2)
    return torch.where(small,.5*torch.exp(-sigma*z)*pa,aa),torch.where(small,.5*torch.exp(-sigma*z)*pb,bb)


def build(alpha,basis,N,scheme='fast_l21',direct=False):
    a=alpha.to(torch.float64).reshape(());tau=basis['tau'];sigma=1-a/2 if scheme=='fast_l21' else a.new_tensor(1.)
    local=sigma.pow(1-a)*tau**(-a)*torch.exp(-torch.lgamma(2-a))
    lag=torch.arange(N,device=a.device,dtype=a.dtype)
    if direct:
        if scheme=='fast_l1':
            j=lag[1:]
            taps=local*torch.cat([a.new_ones(1),(j+1).pow(1-a)-j.pow(1-a)])
            correction=torch.zeros_like(taps)
        else:
            j=torch.arange(1,N+1,device=a.device,dtype=a.dtype)
            hi=j+sigma;lo=j-1+sigma
            w0=(hi.pow(1-a)-lo.pow(1-a))/(1-a)
            w1=(j-.5+sigma)*w0-(hi.pow(2-a)-lo.pow(2-a))/(2-a)
            scale=tau**(-a)*torch.exp(-torch.lgamma(1-a))
            taps=torch.cat([(local+scale*w1[0])[None],scale*(w0[:-1]-w1[:-1]+w1[1:])])
            correction=scale*w1
        info=dict(scheme=scheme,direct=True)
    else:
        nodes,weights=soe(a,basis);q=nodes*tau;decay=torch.exp(-q)
        powers=decay[:,None].pow(lag[None,:])
        if scheme=='fast_l21':
            A,B=_ab(q,sigma)
            taps=torch.cat([(local+(weights*B).sum())[None],
                            ((weights*(A+B*decay))[:,None]*powers[:,:N-1]).sum(0)])
            correction=((weights*B)[:,None]*powers).sum(0)
            info=dict(scheme=scheme,direct=False,A=A,B=B)
        else:
            phi=-torch.expm1(-q)/q
            taps=torch.cat([local[None],((weights*phi)[:,None]*powers[:,1:]).sum(0)])
            correction=torch.zeros_like(taps)
            info=dict(scheme=scheme,direct=False,phi=phi)
        info.update(nodes=nodes,weights=weights,decay=decay,terms=len(nodes))
    nfft=1<<(2*N-1).bit_length()
    return dict(info,N=N,sigma=sigma,local=local,taps=taps,correction=correction,
                nfft=nfft,fft_taps=torch.fft.rfft(taps,n=nfft))


def apply(values,k,mode='fft'):
    v=values.to(torch.float64);d=v[...,1:]-v[...,:-1];N=d.shape[-1]
    if N!=k['N']:raise ValueError('History node count mismatch.')
    if mode=='fft':
        result=torch.fft.irfft(torch.fft.rfft(d,n=k['nfft'])*k['fft_taps'],n=k['nfft'])[...,:N]
        return result-d[...,:1]*k['correction']
    if mode!='stream' or k['direct']:raise ValueError('Stream mode requires the SOE rule.')
    memory=v.new_zeros((*v.shape[:-1],k['terms']));out=[]
    for n in range(N):
        if n:
            if k['scheme']=='fast_l21':memory=memory*k['decay']+d[...,n-1,None]*k['A']+d[...,n,None]*k['B']
            else:memory=k['decay']*(memory+d[...,n-1,None]*k['phi'])
        out.append((memory@k['weights'])+k['local']*d[...,n])
    return torch.stack(out,-1)
