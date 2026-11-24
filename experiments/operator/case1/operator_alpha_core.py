"""Parameterized wave model and fast Caputo residual."""


import os
import argparse
import copy
import csv
import json
import hashlib
import random
from pathlib import Path
import time
import math
import platform
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.special import gamma
from scipy.linalg import hankel

# ============================================================
# Basic settings
# ============================================================
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
torch.set_float32_matmul_precision("high")

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Device is selected by the training entry point.


# ============================================================
# Network components from the single-equation code
# ============================================================
class DualTanhLayer(nn.Module):
    def __init__(self, in_features, out_features, a_1=None, a_2=None, fn=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.a_1 = a_1
        self.a_2 = a_2
        self.fn = fn

    def forward(self, x):
        z = self.linear(x)
        a_1 = self.a_1 if self.a_1 is not None else torch.tensor(0.2, device=z.device)
        a_2 = self.a_2 if self.a_2 is not None else torch.tensor(0.2, device=z.device)
        fn = self.fn if self.fn is not None else torch.tensor(5.0, device=z.device)
        return torch.tanh(fn * a_1 * z * (torch.tanh(fn * a_2 * z) + 1.0))


class OperatorBranch(nn.Module):
    """
    Encode the physical configuration mu=[alpha,D,beta] into a latent vector.
    A small Fourier-feature lift is used to improve parameter conditioning.
    """
    def __init__(self, mu_dim=3, hidden_size=64, latent_dim=16):
        super().__init__()
        self.register_buffer("mu_min", torch.tensor([0.6, 0.5, 1], dtype=torch.float32))
        self.register_buffer("mu_max", torch.tensor([0.9, 1.5, 2], dtype=torch.float32))
        lifted_dim = mu_dim * 3
        self.net = nn.Sequential(
            nn.Linear(lifted_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(self, mu):
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
        mu_min = self.mu_min.to(mu.device)
        mu_max = self.mu_max.to(mu.device)
        mu_n = 2.0 * (mu - mu_min) / (mu_max - mu_min) - 1.0
        feat = torch.cat([mu_n, torch.sin(math.pi * mu_n), torch.cos(math.pi * mu_n)], dim=-1)
        return self.net(feat)


class OperatorEnhancedNet(nn.Module):
    """
    Trunk network: (t,x,y,z_mu) -> one real-valued component.
    """
    def __init__(
        self,
        num_hidden_layers=3,
        coord_dim=3,
        latent_dim=16,
        hidden_size=60,
        activation=torch.tanh,
        a_1=None,
        a_2=None,
        fn=None,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.a_1 = a_1
        self.a_2 = a_2
        self.fn = fn

        in_dim = coord_dim + latent_dim
        layers = [DualTanhLayer(in_dim, hidden_size, a_1=a_1, a_2=a_2, fn=fn)]
        for _ in range(num_hidden_layers - 1):
            layers.append(DualTanhLayer(hidden_size, hidden_size, a_1=a_1, a_2=a_2, fn=fn))
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, tx, z_mu):
        if z_mu.dim() == 1:
            z_mu = z_mu.unsqueeze(0)
        if z_mu.shape[0] == 1 and tx.shape[0] != 1:
            z_mu = z_mu.expand(tx.shape[0], -1)
        inp = torch.cat([tx, z_mu], dim=1)
        return self.out(self.hidden(inp))


class OperatorEnhancedFPINN(nn.Module):
    """
    Complex-valued operator-enhanced fPINN:
        mu -> z_mu,
        (t,x,y,z_mu) -> (u,v).
    """
    def __init__(
        self,
        mu_dim=3,
        latent_dim=16,
        hidden_size=60,
        num_hidden_layers=3,
        activation=torch.tanh,
        a_1=None,
        a_2=None,
        fn=None,
    ):
        super().__init__()
        self.branch = OperatorBranch(mu_dim=mu_dim, hidden_size=64, latent_dim=latent_dim)
        self.real_trunk = OperatorEnhancedNet(
            num_hidden_layers=num_hidden_layers,
            coord_dim=3,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            activation=activation,
            a_1=a_1,
            a_2=a_2,
            fn=fn,
        )
        self.imag_trunk = OperatorEnhancedNet(
            num_hidden_layers=num_hidden_layers,
            coord_dim=3,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            activation=activation,
            a_1=a_1,
            a_2=a_2,
            fn=fn,
        )

    def forward(self, tx, mu):
        z_mu = self.branch(mu)
        u = self.real_trunk(tx, z_mu)
        v = self.imag_trunk(tx, z_mu)
        return u, v


class ComponentWrapper(nn.Module):
    """
    Wrapper used to keep fpde(...) and compute_loss(...) unchanged.
    compute_loss still calls realnet(tx) and imagenet(tx), but internally the
    operator model receives both tx and the current parameter mu.
    """
    def __init__(self, model, mu, component="real"):
        super().__init__()
        self.model = model
        self.mu = mu
        self.component = component

    def forward(self, tx):
        u, v = self.model(tx, self.mu)
        return u if self.component == "real" else v


class HyperNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
            nn.Softplus(),
        )

    def forward(self, losses_tensor):
        x = torch.log(losses_tensor.detach() + 1e-8).view(1, 3)
        w = self.net(x).view(-1)
        # Normalization avoids extremely large weights and keeps the scale stable.
        w = 3.0 * w / (torch.sum(w) + 1e-12)
        return w[0], w[1], w[2]


hypernet = None


def myls_qr(A, b, eps=1e-13):
    m, n = A.shape
    Q, R = np.linalg.qr(A)
    if eps is None:
        eps = 1e-13
    s = np.diag(R)
    r = np.sum(np.abs(s) > eps)
    Q = Q[:, :r]
    R = R[:r, :r]
    b1 = b[r:m + r]
    x = np.linalg.solve(R, Q.T @ b1)
    return x


def myls_svd(A, b, eps=1e-13):
    m, n = A.shape
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    if eps is None:
        eps = 1e-13
    s = S
    r = np.sum(s > eps)
    x = np.zeros(n, dtype=complex)
    for i in range(r):
        x += (U[:, i].T @ b) / s[i] * Vh[i]
    res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    return x, res


def prony(xs, ws):
    M = len(xs)
    errbnd = 1e-12
    h = np.zeros((2 * M, 1))
    for j in range(1, 2 * M + 1):
        h[j - 1] = np.sum(xs ** (j - 1) * ws)
    C = h[:M]
    R = h[M:2 * M - 1]
    H = hankel(C.ravel(), R.ravel())
    b = -h
    q = myls_qr(H, b, errbnd)
    r = len(q)
    Coef = np.concatenate((np.array([[1]]), np.flipud(q[:])), axis=0)
    xsnew = np.roots(Coef.flatten())
    A = np.zeros((2 * M, r), dtype=complex)
    for j in range(1, 2 * M + 1):
        A[j - 1, :] = xsnew.ravel() ** (j - 1)
    wsnew, _ = myls_svd(A, h, errbnd)
    ind = np.where(np.real(xsnew) >= 0)[0]
    p = len(ind)
    if p == 0:
        ind = np.where(np.real(xsnew) < 0)[0]
        xsnew = xsnew[ind]
        wsnew = wsnew[ind]
    else:
        ind = np.where(np.real(xsnew) < 0)[0]
        xsnew = xsnew[ind]
        wsnew = wsnew[ind]
    return wsnew, xsnew


def sumofexpappr2new(alpha, reps, dt, Tfinal):
    delta = dt / Tfinal
    h = 2 * np.pi / (np.log(3) + alpha * np.log(1 / np.cos(1)) + np.log(1 / reps))
    tlower = 1 / alpha * np.log(reps * gamma(1 + alpha))
    if alpha >= 1:
        tupper = np.log(1 / delta) + np.log(np.log(1 / reps)) + np.log(alpha) + 0.5
    else:
        tupper = np.log(1 / delta) + np.log(np.log(1 / reps))
    M = int(np.floor(tlower / h)); N = int(np.ceil(tupper / h))

    n1  = np.arange(M, 0, 1)
    xs1 = -np.exp(h * n1)
    ws1 = h / gamma(alpha) * np.exp(alpha * h * n1)
    ws1new, xs1new = prony(xs1, ws1)

    n2  = np.arange(N + 1)
    xs2 = -np.exp(h * n2)
    ws2 = h / gamma(alpha) * np.exp(alpha * h * n2)

    xs = np.concatenate((-np.real(xs1new), -np.real(xs2)))
    ws = np.concatenate((np.real(ws1new),  np.real(ws2)))
    xs = xs / Tfinal
    ws = ws / Tfinal ** alpha
    return xs, ws, len(ws)


def gamma_torch(x):
    return torch.exp(torch.lgamma(x))


def a_hat(sigma, alpha, tau):
    return (sigma ** (1 - alpha)) * (tau ** (-alpha)) / (1 - alpha)



_FAST_CACHE = {}


KERNEL_AUDIT = {}


def _positive_soe(alpha, eps, dt):
    # Same logarithmic trapezoidal quadrature as the upload, without Prony
    # compression of the small-rate tail. Used only if its compressed version
    # fails the numerical kernel check below.
    h = 2*np.pi/(np.log(3)+alpha*np.log(1/np.cos(1))+np.log(1/eps))
    low = np.log(eps*gamma(1+alpha))/alpha
    high = np.log(1/dt)+np.log(np.log(1/eps))
    k = np.arange(int(np.floor(low/h)), int(np.ceil(high/h))+1)
    return np.exp(h*k), h/gamma(alpha)*np.exp(alpha*h*k)


def _get_fast_coeff(alpha_val, sigma_val, tau_val, device):
    key = (float(alpha_val), float(sigma_val), float(tau_val), str(device))
    if key in _FAST_CACHE:
        return _FAST_CACHE[key]
    eps = 1e-8
    dt = sigma_val*tau_val
    probe = np.geomspace(dt, 1.0, 1024)
    xs, ws, _ = sumofexpappr2new(alpha_val, eps, dt, 1.0)
    error = np.max(np.abs(np.exp(-probe[:,None]*xs)@ws-probe**(-alpha_val)))
    method = 'uploaded_prony'
    if (not np.isfinite(error)) or error > 1.05*eps or np.any(xs <= 0) or np.any(ws <= 0):
        xs, ws = _positive_soe(alpha_val, eps, dt)
        error = np.max(np.abs(np.exp(-probe[:,None]*xs)@ws-probe**(-alpha_val)))
        method = 'same_quadrature_without_prony'
    if not np.isfinite(error) or error > 1.05*eps:
        raise ArithmeticError(f'SOE kernel check failed: {error}')
    # Stable evaluation of the SAME A/B integrals. Original float32 expressions
    # suffer catastrophic cancellation at small q = rate*tau.
    q = xs*tau_val
    small = np.abs(q) < 0.05
    A = np.empty_like(q); B = np.empty_like(q)
    z = q[small]; pa = np.zeros_like(z); pb = np.zeros_like(z)
    for j in range(2,15):
        ca = -(2*(-1)**j/math.factorial(j)+3*(-1)**(j-1)/math.factorial(j-1))
        cb = 2*(-1)**j/math.factorial(j)+(-1)**(j-1)/math.factorial(j-1)
        pa += ca*z**(j-2); pb += cb*z**(j-2)
    A[small] = 0.5*np.exp(-sigma_val*z)*pa
    B[small] = 0.5*np.exp(-sigma_val*z)*pb
    z = q[~small]
    A[~small] = 0.5*np.exp(-sigma_val*z)*((2+z)-(2+3*z)*np.exp(-z))/(z*z)
    B[~small] = 0.5*np.exp(-sigma_val*z)*((z-2)+(z+2)*np.exp(-z))/(z*z)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    gi = 1/gamma(1-alpha_val)
    local = sigma_val**(1-alpha_val)*tau_val**(-alpha_val)/gamma(2-alpha_val)
    result = (tensor(xs),tensor(ws).view(-1,1),tensor(A).view(1,-1),
              tensor(B).view(1,-1),tensor(np.exp(-q)),tensor(gi),tensor(local))
    KERNEL_AUDIT[str(key)] = {'alpha':alpha_val,'tau':tau_val,
        'soe_terms':len(xs),'method':method,'probe_max_abs_error':float(error),
        'probe_count':len(probe),'history_dtype':'float64'}
    _FAST_CACHE[key] = result
    return result


def fast_caputo(values, alpha, tau):
    """Input shape (..., N+1); output (..., N) at t[n-1]+sigma*tau."""
    alpha = float(alpha)
    coeff = _get_fast_coeff(alpha,1-alpha/2,float(tau),values.device)
    xs, ws, A, B, decay, gi, local = coeff
    y = values.to(torch.float64)
    history = torch.zeros((*y.shape[:-1], len(xs)),dtype=y.dtype,device=y.device)
    output = []
    for n in range(1,y.shape[-1]):
        delta = (y[...,n]-y[...,n-1]).unsqueeze(-1)
        if n > 1:
            old_delta = (y[...,n-1]-y[...,n-2]).unsqueeze(-1)
            history = history*decay+old_delta*A+delta*B
        output.append(gi*(history@ws).squeeze(-1)+local*delta.squeeze(-1))
    return torch.stack(output,dim=-1)



def fpde(tx, tx_sigma, realnet, imagenet, M, M2, N, tau, D, alpha, beta, device):
    alpha_numpy = alpha.detach().cpu().numpy()
    sigma = 1 - alpha / 2
    sigma_numpy = sigma.detach().cpu().numpy()

    u = realnet(tx)       # (B,1)
    v = imagenet(tx)      # (B,1)
    u_sigma = realnet(tx_sigma)
    v_sigma = imagenet(tx_sigma)

    u_mat = u.view(M + 1, N + 1, M2 + 1)
    v_mat = v.view(M + 1, N + 1, M2 + 1)

    # meshgrid flatten order is (x1,t,x2), as in the uploaded code.
    Dtu_sigma = fast_caputo(u_mat[1:M,:,1:M2].permute(0,2,1),
        float(alpha.detach().cpu().item()),tau).permute(0,2,1).reshape(-1,1).to(u_sigma)
    Dtv_sigma = fast_caputo(v_mat[1:M,:,1:M2].permute(0,2,1),
        float(alpha.detach().cpu().item()),tau).permute(0,2,1).reshape(-1,1).to(v_sigma)

    
    ones_u = torch.ones_like(u_sigma)
    ones_v = torch.ones_like(v_sigma)
    u_grad = torch.autograd.grad(u_sigma, tx_sigma, grad_outputs=ones_u, create_graph=True, allow_unused=False)[0]
    v_grad = torch.autograd.grad(v_sigma, tx_sigma, grad_outputs=ones_v, create_graph=True, allow_unused=False)[0]

    u_x1 = u_grad[:, 1:2]; u_x2 = u_grad[:, 2:3]
    v_x1 = v_grad[:, 1:2]; v_x2 = v_grad[:, 2:3]

    u_xx1 = torch.autograd.grad(u_x1, tx_sigma, grad_outputs=torch.ones_like(u_x1), create_graph=True, allow_unused=False)[0][:, 1:2]
    u_xx2 = torch.autograd.grad(u_x2, tx_sigma, grad_outputs=torch.ones_like(u_x2), create_graph=True, allow_unused=False)[0][:, 2:3]
    v_xx1 = torch.autograd.grad(v_x1, tx_sigma, grad_outputs=torch.ones_like(v_x1), create_graph=True, allow_unused=False)[0][:, 1:2]
    v_xx2 = torch.autograd.grad(v_x2, tx_sigma, grad_outputs=torch.ones_like(v_x2), create_graph=True, allow_unused=False)[0][:, 2:3]

    uu  = u_sigma * u_sigma + v_sigma * v_sigma
    uuu = uu * u_sigma
    vvv = uu * v_sigma

    fl_real = -Dtv_sigma + D * (u_xx1 + u_xx2) + beta * uuu
    fl_image =  Dtu_sigma + D * (v_xx1 + v_xx2) + beta * vvv

    fll_real = D * (u_xx1 + u_xx2) + beta * uuu
    fll_image = D * (v_xx1 + v_xx2) + beta * vvv

    grad_f_real  = torch.autograd.grad(fll_real,  tx_sigma, grad_outputs=torch.ones_like(fll_real),  create_graph=True, allow_unused=False)[0]
    grad_f_image = torch.autograd.grad(fll_image, tx_sigma, grad_outputs=torch.ones_like(fll_image), create_graph=True, allow_unused=False)[0]

    grad_x1_f_real = grad_f_real[:, 1:2]
    grad_x2_f_real = grad_f_real[:, 2:3]
    grad_x1_f_image = grad_f_image[:, 1:2]
    grad_x2_f_image = grad_f_image[:, 2:3]

    # Local time derivatives: SAME fll terms, without any Caputo contribution.
    grad_t_f_real = grad_f_real[:, 0:1]
    grad_t_f_image = grad_f_image[:, 0:1]
    return (fl_real, fl_image, grad_x1_f_real, grad_x2_f_real,
            grad_x1_f_image, grad_x2_f_image, grad_t_f_real, grad_t_f_image)


# ============================================================
# Loss: keep the original function definition
# ============================================================
def compute_loss(realnet, imagenet,
                 pt_t0_collocation, pt_x1t0_collocation, pt_x2t0_collocation,
                 pt_tbc_collocation, pt_x1bc_collocation, pt_x2bc_collocation,
                 pt_t_collocation, pt_x1_collocation, pt_x2_collocation,
                 pt_ubc_collocation, pt_vbc_collocation, pt_ut0_collocation, pt_vt0_collocation,
                 pt_f_real_collocation, pt_f_real_x1_collocation, pt_f_real_x2_collocation, pt_f_real_t_collocation,
                 pt_f_image_collocation, pt_f_image_x1_collocation, pt_f_image_x2_collocation, pt_f_image_t_collocation,
                 mse_cost_function1, M1, M2, N, tau, alpha, w_grad, D, beta,
                 pt_t_sigma, pt_x1_sigma, pt_x2_sigma, device):
    tx_bc = torch.cat([pt_tbc_collocation, pt_x1bc_collocation, pt_x2bc_collocation], 1)
    tx_t0 = torch.cat([pt_t0_collocation, pt_x1t0_collocation, pt_x2t0_collocation], 1)

    net_u_xbc_real = realnet(tx_bc)
    net_u_xbc_image = imagenet(tx_bc)
    net_u_t0_real = realnet(tx_t0)
    net_u_t0_image = imagenet(tx_t0)

    mse_xbc_real = mse_cost_function1(pt_ubc_collocation, net_u_xbc_real)
    mse_xbc_image = mse_cost_function1(pt_vbc_collocation, net_u_xbc_image)
    mse_t0_real = mse_cost_function1(pt_ut0_collocation, net_u_t0_real)
    mse_t0_image = mse_cost_function1(pt_vt0_collocation, net_u_t0_image)

    tx = torch.cat([pt_t_collocation, pt_x1_collocation, pt_x2_collocation], 1)
    tx_sig = torch.cat([pt_t_sigma, pt_x1_sigma, pt_x2_sigma], 1)

    (f_out_real, f_out_image,
     grad_x1_f_real, grad_x2_f_real,
     grad_x1_f_image, grad_x2_f_image,
     grad_t_f_real, grad_t_f_image) = fpde(
        tx, tx_sig, realnet, imagenet, M1, M2, N, tau, D, alpha, beta, device
    )

    mse_f_real_x1 = mse_cost_function1(grad_x1_f_real, pt_f_real_x1_collocation)
    mse_f_real_x2 = mse_cost_function1(grad_x2_f_real, pt_f_real_x2_collocation)
    mse_f_image_x1 = mse_cost_function1(grad_x1_f_image, pt_f_image_x1_collocation)
    mse_f_image_x2 = mse_cost_function1(grad_x2_f_image, pt_f_image_x2_collocation)
    mse_f_real_t = mse_cost_function1(grad_t_f_real, pt_f_real_t_collocation)
    mse_f_image_t = mse_cost_function1(grad_t_f_image, pt_f_image_t_collocation)
    mse_f_1_real = mse_cost_function1(f_out_real, pt_f_real_collocation)
    mse_f_1_image = mse_cost_function1(f_out_image, pt_f_image_collocation)

    loss_f = mse_f_1_real + mse_f_1_image
    loss_ic = mse_t0_real + mse_t0_image
    loss_bc = mse_xbc_real + mse_xbc_image

    losses_tensor = torch.stack([loss_f, loss_bc, loss_ic], dim=-1)
    w_f, w_bc, w_ic = hypernet(losses_tensor)
    w_g = w_grad * torch.exp(-3 * loss_f.detach())

    # Keep the old four spatial MSEs and add two local time MSEs.
    # No division by the number of directions and no change to w_g.
    loss_grad_space = mse_f_real_x1 + mse_f_real_x2 + mse_f_image_x1 + mse_f_image_x2
    loss_grad_time = mse_f_real_t + mse_f_image_t
    loss_grad = loss_grad_space + loss_grad_time
    loss = w_f * loss_f + w_ic * loss_ic + w_bc * loss_bc + w_g * loss_grad
    return {"loss":loss,"loss_f":loss_f,"loss_ic":loss_ic,"loss_bc":loss_bc,
        "loss_grad":loss_grad,"loss_grad_space":loss_grad_space,"loss_grad_time":loss_grad_time,
        "w_f":w_f,"w_bc":w_bc,"w_ic":w_ic,"w_g":w_g}


# ============================================================
# Data construction for each parameter configuration
# ============================================================
def to_torch(arr, requires_grad=False):
    return Variable(torch.from_numpy(arr).float(), requires_grad=requires_grad).to(device)


def exact_uv(t, x1, x2, alpha_val, D_val, beta_val):
    S = np.sin(np.pi * D_val * x1) * np.sin(np.pi * beta_val * x2)
    amp = t ** (1.0 + alpha_val) * S
    return amp, amp


def build_dataset(alpha_val, D_val, beta_val, M1=8, M2=8, N=8, boundnum=100):
    x1l, x2l = 0.0, 0.0
    x1r, x2r = 1.0, 1.0

    rng = np.random.default_rng(SEED + int(1000 * alpha_val) + int(100 * D_val) + int(10 * beta_val))
    t_ic_var = rng.uniform(0, 1, (boundnum, 1))
    x1_bc_var = rng.uniform(x1l, x1r, (boundnum, 1))
    x2_bc_var = rng.uniform(x2l, x2r, (boundnum, 1))
    t0_np = np.zeros_like(x1_bc_var)

    x1ll = x1l * np.ones_like(x1_bc_var)
    x1rr = x1r * np.ones_like(x1_bc_var)
    x2ll = x2l * np.ones_like(x2_bc_var)
    x2rr = x2r * np.ones_like(x2_bc_var)

    x1_bc_coll = np.concatenate([x1ll, x1rr, x1_bc_var, x1_bc_var], axis=0)
    x2_bc_coll = np.concatenate([x2_bc_var, x2_bc_var, x2ll, x2rr], axis=0)
    t_bc_coll = np.concatenate([t_ic_var, t_ic_var, t_ic_var, t_ic_var], axis=0)

    Exactu_bc_coll, Exactv_bc_coll = exact_uv(t_bc_coll, x1_bc_coll, x2_bc_coll, alpha_val, D_val, beta_val)
    Exactt0_u, Exactt0_v = exact_uv(t0_np, x1_bc_var, x2_bc_var, alpha_val, D_val, beta_val)

    t = np.linspace(0.0, 1.0, N + 1)
    x1 = np.linspace(0.0, 1.0, M1 + 1)
    x2 = np.linspace(0.0, 1.0, M2 + 1)
    x1_in = x1[1:M1]
    x2_in = x2[1:M2]
    tau = 1.0 / N
    sigma = 1.0 - alpha_val / 2.0
    t_sigma = (t + tau * sigma)[:N]

    ms_t, ms_x1, ms_x2 = np.meshgrid(t, x1, x2)
    t_np = ms_t.reshape(-1, 1)
    x1_np = ms_x1.reshape(-1, 1)
    x2_np = ms_x2.reshape(-1, 1)

    ms_ts, ms_x1s, ms_x2s = np.meshgrid(t_sigma, x1_in, x2_in)
    t_sigma_np = ms_ts.reshape(-1, 1)
    x1_sigma_np = ms_x1s.reshape(-1, 1)
    x2_sigma_np = ms_x2s.reshape(-1, 1)

    Exact1_real_sigma, Exact1_image_sigma = exact_uv(t_sigma_np, x1_sigma_np, x2_sigma_np, alpha_val, D_val, beta_val)

    S = np.sin(np.pi * D_val * x1_sigma_np) * np.sin(np.pi * beta_val * x2_sigma_np)
    caputo = gamma(2.0 + alpha_val) * t_sigma_np

    f_real = -caputo * S - (np.pi ** 2 * (D_val ** 2 + beta_val ** 2)) * D_val * t_sigma_np ** (1.0 + alpha_val) * S \
             + 2.0 * beta_val * t_sigma_np ** (3.0 + 3.0 * alpha_val) * (S ** 3)
    f_image = caputo * S - (np.pi ** 2 * (D_val ** 2 + beta_val ** 2)) * D_val * t_sigma_np ** (1.0 + alpha_val) * S \
              + 2.0 * beta_val * t_sigma_np ** (3.0 + 3.0 * alpha_val) * (S ** 3)

    Sx = np.pi * D_val * np.cos(np.pi * D_val * x1_sigma_np) * np.sin(np.pi * beta_val * x2_sigma_np)
    Sy = np.pi * beta_val * np.sin(np.pi * D_val * x1_sigma_np) * np.cos(np.pi * beta_val * x2_sigma_np)
    coeff = t_sigma_np ** (1.0 + alpha_val) * (-np.pi ** 2 * (D_val ** 2 + beta_val ** 2)) * D_val \
            + 6.0 * beta_val * t_sigma_np ** (3.0 + 3.0 * alpha_val) * (S ** 2)

    f_real_x1 = coeff * Sx
    f_real_x2 = coeff * Sy
    # Analytic time targets for the LOCAL source; fractional source is excluded.
    coeff_t = -(np.pi**2*(D_val**2+beta_val**2))*D_val*(1+alpha_val)*t_sigma_np**alpha_val \
        + 6*beta_val*(1+alpha_val)*t_sigma_np**(2+3*alpha_val)*S**2
    f_real_t = coeff_t * S
    f_image_x1 = coeff * Sx
    f_image_x2 = coeff * Sy
    f_image_t = coeff_t * S

    data = dict(
        alpha=torch.tensor([alpha_val], dtype=torch.float32, device=device),
        D=torch.tensor([D_val], dtype=torch.float32, device=device),
        beta=torch.tensor([beta_val], dtype=torch.float32, device=device),
        mu=torch.tensor([[alpha_val, D_val, beta_val]], dtype=torch.float32, device=device),
        tau=tau,
        M1=M1,
        M2=M2,
        N=N,
        Exact1_real_sigma=Exact1_real_sigma,
        Exact1_image_sigma=Exact1_image_sigma,
        pt_tbc_collocation=to_torch(t_bc_coll, True),
        pt_x1bc_collocation=to_torch(x1_bc_coll, True),
        pt_x2bc_collocation=to_torch(x2_bc_coll, True),
        pt_ubc_collocation=to_torch(Exactu_bc_coll, False),
        pt_vbc_collocation=to_torch(Exactv_bc_coll, False),
        pt_t0_collocation=to_torch(t0_np, True),
        pt_x1t0_collocation=to_torch(x1_bc_var, True),
        pt_x2t0_collocation=to_torch(x2_bc_var, True),
        pt_ut0_collocation=to_torch(Exactt0_u, False),
        pt_vt0_collocation=to_torch(Exactt0_v, False),
        pt_t_collocation=to_torch(t_np, True),
        pt_x1_collocation=to_torch(x1_np, True),
        pt_x2_collocation=to_torch(x2_np, True),
        pt_t_sigma=to_torch(t_sigma_np, True),
        pt_x1_sigma=to_torch(x1_sigma_np, True),
        pt_x2_sigma=to_torch(x2_sigma_np, True),
        pt_u_real_sigma=to_torch(Exact1_real_sigma, False),
        pt_u_image_sigma=to_torch(Exact1_image_sigma, False),
        pt_f_real_collocation=to_torch(f_real, False),
        pt_f_real_x1_collocation=to_torch(f_real_x1, False),
        pt_f_real_x2_collocation=to_torch(f_real_x2, False),
        pt_f_real_t_collocation=to_torch(f_real_t, False),
        pt_f_image_collocation=to_torch(f_image, False),
        pt_f_image_x1_collocation=to_torch(f_image_x1, False),
        pt_f_image_x2_collocation=to_torch(f_image_x2, False),
        pt_f_image_t_collocation=to_torch(f_image_t, False),
    )
    return data


def relative_l2_on_dataset(model, data):
    model.eval()
    with torch.no_grad():
        tx_eval = torch.cat([data["pt_t_sigma"], data["pt_x1_sigma"], data["pt_x2_sigma"]], dim=1)
        u_pred, v_pred = model(tx_eval, data["mu"])
        u_true = data["pt_u_real_sigma"]
        v_true = data["pt_u_image_sigma"]
        err = torch.sum((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2)
        den = torch.sum(u_true ** 2 + v_true ** 2) + 1e-12
        return float(torch.sqrt(err / den).detach().cpu())


