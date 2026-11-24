"Evaluate the fractional history over the complete interval and inherited time windows."
import math
import torch
from c01_source.baseline_original import build_stable_kernel
from c01_source.direct_reference import direct_caputo
from protocol import ALPHA,T


def build(tau,N,device,eps=1e-10):
    k=build_stable_kernel(ALPHA,tau,device,Tfinal=T,eps=eps)
    dtype=torch.float64
    decay=k['exp_xs_tau'];a=k['A_l'].ravel();b=k['B_l'].ravel()
    w=k['ws'].ravel()/math.gamma(1-ALPHA)
    local=(1-ALPHA/2)**(1-ALPHA)*tau**(-ALPHA)/math.gamma(2-ALPHA)
    powers=decay[:,None].pow(torch.arange(N,dtype=dtype,device=device)[None,:])
    taps=torch.cat([torch.tensor([local],device=device,dtype=dtype)+(w*b).sum(),
                    ((w*(a+b*decay))[:,None]*powers[:,:N-1]).sum(0)])
    correction=((w*b)[:,None]*powers).sum(0)
    nfft=1<<(2*N-1).bit_length()
    k.update(N=N,local=local,gamma_inv=1/math.gamma(1-ALPHA),nfft=nfft,
             fft_taps=torch.fft.rfft(taps,n=nfft),startup_correction=correction,
             full_horizon=T)
    return k


def fft(values,k):
    "Fft."
    v=values.to(torch.float64);d=v[...,1:]-v[...,:-1];N=d.shape[-1]
    if N>k['N']:raise ValueError('History longer than its SOE horizon.')
    result=torch.fft.irfft(torch.fft.rfft(d,n=k['nfft'])*k['fft_taps'],n=k['nfft'])[...,:N]
    return result-d[...,:1]*k['startup_correction'][:N]


def stream(values,k):
    "Stream."
    v=values.to(torch.float64);d=v[...,1:]-v[...,:-1]
    memory=v.new_zeros((*v.shape[:-1],k['nexp']));out=[]
    for n in range(d.shape[-1]):
        if n:memory=memory*k['exp_xs_tau']+d[...,n-1,None]*k['A_l']+d[...,n,None]*k['B_l']
        out.append(k['gamma_inv']*(memory@k['ws']).squeeze(-1)+k['local']*d[...,n])
    return torch.stack(out,-1)


def apply(values,k,mode='fft'):
    if mode=='fft':return fft(values,k)
    if mode=='stream':return stream(values,k)
    if mode=='direct':return direct_caputo(values,ALPHA,float(k['tau']),scheme='l21')
    raise ValueError('Unknown history mode.')
