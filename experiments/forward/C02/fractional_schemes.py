"""Fixed-alpha history rules for the C01 comparison; Python 3.8+.

fast_l21 delegates to the uploaded solver unchanged. The other three rules
are defined here, including their actual collocation time levels.
"""
import math

import numpy as np
import torch

SCHEMES = ('fast_l21', 'l21', 'l1', 'fast_l1')


def sigma(alpha, scheme):
    return 1.0 if scheme in ('l1', 'fast_l1') else 1.0-float(alpha)/2.0


def l1_matrix(N, alpha, tau, device):
    """Map successive value increments to D^alpha u at t_1,...,t_N."""
    j = np.arange(N)
    lag = j[:, None]-j[None, :]
    k = np.maximum(lag, 0).astype(np.float64)
    weights = ((k+1.0)**(1.0-alpha)-k**(1.0-alpha))
    weights *= (lag >= 0)*tau**(-alpha)/math.gamma(2.0-alpha)
    return torch.tensor(weights, dtype=torch.float64, device=device)


def build_kernel(alpha, tau, N, device, scheme, baseline):
    if scheme == 'l1':
        return {'scheme': scheme, 'matrix': l1_matrix(N, alpha, tau, device)}
    if scheme == 'l21':
        return {'scheme': scheme}
    if scheme != 'fast_l1':
        raise ValueError('fast_l21 uses the original kernel builder.')
    xs, ws, nexp = baseline.sumofexpappr2new(float(alpha), 1e-8, tau, 1.0)
    q = np.asarray(xs, dtype=np.float64)*tau
    if np.any(q <= 0) or not all(np.isfinite(a).all() for a in (q, ws)):
        raise ArithmeticError('Invalid Fast L1 SOE coefficients.')
    tensor = lambda a: torch.tensor(a, dtype=torch.float64, device=device)
    return {'scheme': scheme, 'decay': tensor(np.exp(-q)),
            'interval_average': tensor(-np.expm1(-q)/q),
            'weights': tensor(ws).reshape(-1, 1), 'nexp': int(nexp),
            'gamma_inv': tensor(1.0/math.gamma(1.0-alpha)),
            'local_coefficient': tensor(tau**(-alpha)/math.gamma(2.0-alpha))}


def apply_history(values, alpha, tau, kernel, direct_caputo):
    scheme = kernel['scheme']
    if scheme == 'l21':
        return direct_caputo(values, alpha, tau, scheme='l21')
    v = values.to(torch.float64)
    increments = v[..., 1:]-v[..., :-1]
    if scheme == 'l1':
        return increments @ kernel['matrix'].T
    if scheme != 'fast_l1':
        raise ValueError('Unknown history scheme: '+str(scheme))
    history = torch.zeros((*v.shape[:-1], kernel['nexp']),
                          device=v.device, dtype=v.dtype)
    result = []
    for n in range(increments.shape[-1]):
        if n:
            history = kernel['decay']*(history + increments[..., n-1, None]*kernel['interval_average'])
        memory = (history @ kernel['weights']).squeeze(-1)*kernel['gamma_inv']
        result.append(memory+kernel['local_coefficient']*increments[..., n])
    return torch.stack(result, dim=-1)


def manufactured_data(t, x, alpha):
    """Source and the ORIGINAL local x/t regularizer targets, not full history gradients."""
    sine, cosine = np.sin(2*np.pi*x), np.cos(2*np.pi*x)
    local = -4*np.pi**2*t**2+t**6
    caputo = 2*t**(2-alpha)/math.gamma(3-alpha)
    return {'Exact1_real_sigma': t**2*sine, 'Exact1_image_sigma': t**2*cosine,
            'pt_f_real': -caputo*cosine+local*sine,
            'pt_f_image': caputo*sine+local*cosine,
            'pt_f_real_x': 2*np.pi*local*cosine,
            'pt_f_image_x': -2*np.pi*local*sine,
            'pt_f_real_t': (-8*np.pi**2*t+6*t**5)*sine,
            'pt_f_image_t': (-8*np.pi**2*t+6*t**5)*cosine}
