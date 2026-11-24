"""Optional L1/DeepONet reconstruction, NOT the absent original baseline code.

Standard branch/trunk dot product; uniform L1 time history, ordinary tanh,
uniform PDE/BC/IC weights. No GO features, enrichment, GE or consistency.
The architecture is an explicit implementation choice, not a recovered one.
L1 formula: ScienceDirect PII S009630032400081X (see REFERENCES.md).
"""
import math
import numpy as np
import torch
from torch import nn
import operator_alpha_core as core


class L1DeepONet(nn.Module):
    def __init__(self, width=60, depth=5, latent=64):
        super().__init__()
        self.register_buffer('lower', torch.tensor([.6, .5, 1.]))
        self.register_buffer('upper', torch.tensor([.9, 1.5, 2.]))
        self.branch = nn.Sequential(nn.Linear(3, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2*latent))
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 2*latent))
        self.trunk = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(2))
        self.latent = latent

    def forward(self, tx, mu):
        branch = self.branch(2*(mu-self.lower)/(self.upper-self.lower)-1)
        branch = branch.view(-1, 2, self.latent)
        trunk = self.trunk(2*tx-1).view(-1, 2, self.latent)
        result = (branch*trunk).sum(-1) + self.bias
        return result[:, :1], result[:, 1:2]


def l1_caputo(values, alpha, tau):
    """Uniform L1 at t_1,...,t_N; values[...,0:N+1], float64 history.

    D^alpha v_n = tau^-alpha/Gamma(2-alpha) *
                 sum_{j=1}^n [(n-j+1)^(1-alpha)-(n-j)^(1-alpha)]*(v_j-v_{j-1}).
    Matrix form remains differentiable with respect to ALL history values.
    """
    if not 0 < alpha < 1 or tau <= 0 or values.shape[-1] < 2:
        raise ValueError('L1 needs 0<alpha<1, tau>0 and at least two times.')
    n = values.shape[-1]-1
    index = torch.arange(n, device=values.device, dtype=torch.float64)
    lag = index[:, None]-index[None, :]
    nonnegative = lag.clamp_min(0)
    weights = ((nonnegative+1).pow(1-alpha)-nonnegative.pow(1-alpha))*(lag >= 0)
    weights = weights/(math.gamma(2-alpha)*tau**alpha)
    increments = values.double()[..., 1:]-values.double()[..., :-1]
    return increments @ weights.T


def build_l1_dataset(parameter, grid, boundary_points):
    data = core.build_dataset(*parameter['mu'], M1=grid, M2=grid, N=grid,
                              boundnum=boundary_points)
    alpha, beta, gamma = parameter['mu']
    times = np.linspace(0, 1, grid+1)[1:]
    space = np.linspace(0, 1, grid+1)[1:-1]
    tt, xx, yy = np.meshgrid(times, space, space)
    t, x, y = [a.reshape(-1, 1) for a in (tt, xx, yy)]
    spatial = np.sin(np.pi*beta*x)*np.sin(np.pi*gamma*y)
    caputo = math.gamma(2+alpha)*t*spatial
    local = -beta*np.pi**2*(beta**2+gamma**2)*t**(1+alpha)*spatial \
            +2*gamma*t**(3+3*alpha)*spatial**3
    data['l1_tx'] = torch.tensor(np.concatenate([t, x, y], 1),
                                 dtype=torch.float32, device=core.device, requires_grad=True)
    data['l1_f_real'] = torch.tensor(local-caputo, dtype=torch.float32, device=core.device)
    data['l1_f_imag'] = torch.tensor(local+caputo, dtype=torch.float32, device=core.device)
    data['parameter_id'] = parameter['parameter_id']
    return data


def physics_terms(model, data):
    """PDE at integer L1 times, exact parameter-dependent source and BC/IC."""
    full_tx = torch.cat([data[k] for k in
                        ['pt_t_collocation', 'pt_x1_collocation', 'pt_x2_collocation']], 1)
    full_u, full_v = model(full_tx, data['mu'])
    tx = data['l1_tx']
    u, v = model(tx, data['mu'])
    m, n, m2 = data['M1'], data['N'], data['M2']
    derivatives = []
    for full in [full_u, full_v]:
        values = full.view(m+1, n+1, m2+1)[1:m, :, 1:m2].permute(0, 2, 1)
        derivative = l1_caputo(values, float(data['alpha']), data['tau'])
        derivatives.append(derivative.permute(0, 2, 1).reshape(-1, 1).to(u))
    laplacians = []
    for field in [u, v]:
        grad = torch.autograd.grad(field, tx, torch.ones_like(field), create_graph=True)[0]
        lap = 0
        for dim in [1, 2]:
            first = grad[:, dim:dim+1]
            lap = lap + torch.autograd.grad(first, tx, torch.ones_like(first),
                                            create_graph=True)[0][:, dim:dim+1]
        laplacians.append(lap)
    amp2 = u*u+v*v
    real = -derivatives[1]+data['D']*laplacians[0]+data['beta']*amp2*u
    imag = derivatives[0]+data['D']*laplacians[1]+data['beta']*amp2*v
    lf = ((real-data['l1_f_real'])**2).mean()+((imag-data['l1_f_imag'])**2).mean()
    bc = torch.cat([data[k] for k in
                    ['pt_tbc_collocation', 'pt_x1bc_collocation', 'pt_x2bc_collocation']], 1)
    ic = torch.cat([data[k] for k in
                    ['pt_t0_collocation', 'pt_x1t0_collocation', 'pt_x2t0_collocation']], 1)
    ub, vb = model(bc, data['mu']); ui, vi = model(ic, data['mu'])
    lbc = ((ub-data['pt_ubc_collocation'])**2).mean()+((vb-data['pt_vbc_collocation'])**2).mean()
    lic = ((ui-data['pt_ut0_collocation'])**2).mean()+((vi-data['pt_vt0_collocation'])**2).mean()
    return {'validation_loss': lf+lbc+lic, 'validation_loss_f': lf,
            'validation_loss_bc': lbc, 'validation_loss_ic': lic}


def validate_physics(model, data_list):
    was_training = model.training; model.eval(); rows = []
    try:
        with torch.enable_grad():
            for data in data_list:
                values = physics_terms(model, data)
                rows.append({key: float(v.detach().cpu()) for key, v in values.items()})
    finally:
        model.train(was_training)
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
