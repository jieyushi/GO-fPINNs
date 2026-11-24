"""C01 v3: retain the uploaded training algorithm, with isolated audit switches.
Default: float32 networks, uploaded float32 kernel, learned HyperNet,
local x/t operator regularization, rolling consistency, continuing Adam,
legacy double updates of shared a1/a2, original loss-based stage stopping.
Use --kernel-mode stable64 to change ONLY coefficient/history arithmetic.
This reproduction mode is not a certificate that the original kernel is accurate.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.special import gamma
from scipy.linalg import hankel

class FastGatedBlock(nn.Module):
    "Fastgatedblock."
    def __init__(self, in_dim, out_dim, activation, a1, a2, fn):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.a1 = a1
        self.a2 = a2
        self.fn = fn

    def forward(self, x):
        z = self.linear(x)                           
        inner = self.activation(self.fn * self.a1 * z)
        out = self.activation(self.fn * self.a2 * z * (inner + 1))
        return out

class FastNetwork(nn.Module):
    def __init__(self, num_hidden_layers, input_size, hidden_size,
                 activation, a_1, a_2, fn):
        super().__init__()

        
        layers = [FastGatedBlock(input_size, hidden_size, activation, a_1, a_2, fn)]

        
        for _ in range(num_hidden_layers):
            layers.append(FastGatedBlock(hidden_size, hidden_size, activation, a_1, a_2, fn))

        self.layers = nn.Sequential(*layers)

        
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layers(x)
        out = self.output_layer(out)
        return out

class HyperNet(nn.Module):

    def __init__(self,w_a, w_b ):
        super().__init__()
        # MLP layers
        self.fc1_phys = nn.Linear(3, 16)
        self.fc2_phys = nn.Linear(16, 3)
        # Weight interval parameters
        self.w_a = w_a
        self.w_b = w_b
        # self.gamma = nn.Parameter(torch.tensor(0.2))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:

        losses_phys = losses[..., :3]  # (...,3)
        loss_grad = losses[..., 3:]  # (...,1)

        
        mean_phys = losses_phys.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        tilde_phys = losses_phys / mean_phys  # (...,3)
        tilde_grad = loss_grad / mean_phys  # (...,1)

        
        x_phys = torch.tanh(self.fc1_phys(tilde_phys))
        logits_phys = self.fc2_phys(x_phys)
        alphas_phys = F.softmax(logits_phys, dim=-1)  # (...,3)
        w_phys_low = 1 -  alphas_phys
        w_phys_high = 1+  alphas_phys
        w_a = w_phys_low + 0.1 * torch.rand_like(w_phys_low)
        w_b = w_phys_high - 0.1 * torch.rand_like(w_phys_high)
        w_phys = torch.where(tilde_phys > 1.0,w_a,w_b)  # (...,3)

        return torch.cat([w_phys], dim=-1)

def sumofexpappr2new(alpha, reps, dt, Tfinal):
    delta = dt / Tfinal
    h = 2 * np.pi / (np.log(3) + alpha * np.log(1 / np.cos(1)) + np.log(1 / reps))
    tlower = 1 / alpha * np.log(reps * gamma(1 + alpha))
    if alpha >= 1:
        tupper = np.log(1 / delta) + np.log(np.log(1 / reps)) + np.log(alpha) + 0.5
    else:
        tupper = np.log(1 / delta) + np.log(np.log(1 / reps))
    M = int(np.floor(tlower / h))
    N = int(np.ceil(tupper / h))

    n1 = np.arange(M, 0, 1)
    xs1 = -np.exp(h * n1)
    ws1 = h / gamma(alpha) * np.exp(alpha * h * n1)
    ws1new, xs1new = prony(xs1, ws1)

    n2 = np.arange(N + 1)
    xs2 = -np.exp(h * n2)
    ws2 = h / gamma(alpha) * np.exp(alpha * h * n2)

    xs = np.concatenate((-np.real(xs1new), -np.real(xs2)))
    ws = np.concatenate((np.real(ws1new), np.real(ws2)))
    xs = xs / Tfinal
    ws = ws / Tfinal ** alpha
    nexp = len(ws)
    return xs, ws, nexp

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
    [wsnew, res] = myls_svd(A, h, errbnd)
    ind = np.where(np.real(xsnew) >= 0)[0]
    # ind = np.argsort(np.abs(xsnew)).astype(int)
    p = len(ind)
    if p == 0:
        ind = np.where(np.real(xsnew) < 0)[0]
        xsnew = xsnew[ind]
        wsnew = wsnew[ind]
    else:
        # assert np.sum(np.abs(wsnew[ind]) < 1e-15) == p
        ind = np.where(np.real(xsnew) < 0)[0]
        xsnew = xsnew[ind]
        wsnew = wsnew[ind]
    return wsnew, xsnew

def gamma_torch(x):
    return torch.exp(torch.lgamma(x))

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

def aaa(l, alpha):
    output = (l + 1) ** (1 - alpha) - l ** (1 - alpha)
    return output

def a_hat(sigma,alpha, tau):
    return (sigma ** (1-alpha))*(tau**(-alpha)) / (1 - alpha)

def build_original_kernel(alpha, tau, device):
    if isinstance(alpha, torch.Tensor):
        alpha_val = float(alpha.detach().cpu())
    else:
        alpha_val = float(alpha)

    sigma = 1 - alpha_val / 2
    tau_hat = sigma * tau
    epsilon = 1e-8
    T_final = 1.0

    xs_np, ws_np, nexp = sumofexpappr2new(alpha_val, epsilon, tau_hat, T_final)

    xs = torch.tensor(xs_np, dtype=torch.float32, device=device)
    ws = torch.tensor(ws_np, dtype=torch.float32, device=device).view(nexp,1)

    tau_t   = torch.tensor(tau, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)

    e_st = torch.exp(-sigma_t * tau_t * xs)
    e_t  = torch.exp(-tau_t * xs)
    e_pt = torch.exp(tau_t * xs)

    denom = (tau_t**2) * (xs**2)

    A_l = 0.5 * e_st * e_t * (2*e_pt - 3*tau_t*xs + tau_t*xs*e_pt - 2) / denom
    B_l = 0.5 * e_st * e_t * (tau_t*xs - 2*e_pt + tau_t*xs*e_pt + 2) / denom

    A_l = A_l.view(1,-1)
    B_l = B_l.view(1,-1)
    exp_xs_tau = torch.exp(-xs * tau_t)

    kernel = {
        "xs": xs,
        "ws": ws,
        "A_l": A_l,
        "B_l": B_l,
        "exp_xs_tau": exp_xs_tau,
        "nexp": nexp,
        "sigma": sigma_t,
        "tau": tau_t,
    }
    return kernel

def fpde_fast(tx, tx_sigma, realnet, imagenet,
              M, N, tau, alpha, kernel, device):

    
    u = realnet(tx).view(M+1, N+1)
    v = imagenet(tx).view(M+1, N+1)

    u_sigma = realnet(tx_sigma)
    v_sigma = imagenet(tx_sigma)

    # ---- kernel setup ----
    xs = kernel["xs"]
    ws = kernel["ws"]
    A_l = kernel["A_l"]
    B_l = kernel["B_l"]
    exp_xs_tau = kernel["exp_xs_tau"]
    nexp = kernel["nexp"]
    kdtype = kernel["ws"].dtype
    alpha = torch.tensor([alpha], device=device, dtype=kdtype)  
    sigma = 1 - alpha/2
    gamma_inv = 1.0 / gamma_torch(1-alpha)
    a_hat_inv_gamma = a_hat(sigma, alpha, kernel["tau"]) / gamma_torch(1-alpha)

    
    F_u = torch.zeros((M-1, nexp), device=device, dtype=kdtype)
    F_v = torch.zeros((M-1, nexp), device=device, dtype=kdtype)

    Dtu = torch.zeros((M+1, N+1), device=device, dtype=kdtype)
    Dtv = torch.zeros((M+1, N+1), device=device, dtype=kdtype)

    
    for n in range(1, N+1):

        if n > 1:
            F_u *= exp_xs_tau
            F_v *= exp_xs_tau

            u1 = (u[1:M, n] - u[1:M, n-1]).unsqueeze(-1).to(dtype=kdtype)
            u2 = (u[1:M, n-1] - u[1:M, n-2]).unsqueeze(-1).to(dtype=kdtype)
            v1 = (v[1:M, n] - v[1:M, n-1]).unsqueeze(-1).to(dtype=kdtype)
            v2 = (v[1:M, n-1] - v[1:M, n-2]).unsqueeze(-1).to(dtype=kdtype)

            F_u = F_u + u2 @ A_l + u1 @ B_l
            F_v = F_v + v2 @ A_l + v1 @ B_l

        Dtu[1:M,n] = gamma_inv * (F_u @ ws).squeeze(-1)
        Dtv[1:M,n] = gamma_inv * (F_v @ ws).squeeze(-1)

        Dtu[1:M,n] += a_hat_inv_gamma * (u[1:M,n] - u[1:M,n-1])
        Dtv[1:M,n] += a_hat_inv_gamma * (v[1:M,n] - v[1:M,n-1])

    
    Dtu_sigma = Dtu[1:M,1:N+1].reshape(-1,1)
    Dtv_sigma = Dtv[1:M,1:N+1].reshape(-1,1)

    # ---- autodifferentiation terms ----
    grads_u = torch.autograd.grad(u_sigma, tx_sigma,
                                  grad_outputs=torch.ones_like(u_sigma),
                                  create_graph=True)[0]
    grads_v = torch.autograd.grad(v_sigma, tx_sigma,
                                  grad_outputs=torch.ones_like(v_sigma),
                                  create_graph=True)[0]

    u_x = grads_u[:,1:2]
    v_x = grads_v[:,1:2]

    u_xx = torch.autograd.grad(u_x, tx_sigma, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:,1:2]
    v_xx = torch.autograd.grad(v_x, tx_sigma, grad_outputs=torch.ones_like(v_x),
                               create_graph=True)[0][:,1:2]

    amp2 = u_sigma**2 + v_sigma**2
    uuu = amp2*u_sigma
    vvv = amp2*v_sigma

    f_real = -Dtv_sigma + u_xx + uuu
    f_imag =  Dtu_sigma + v_xx + vvv

    # gradient residuals:
    f_real_g = u_xx + uuu
    f_imag_g = v_xx + vvv

    grad_fr = torch.autograd.grad(f_real_g, tx_sigma,
                                  grad_outputs=torch.ones_like(f_real),
                                  create_graph=True)[0]
    grad_fi = torch.autograd.grad(f_imag_g, tx_sigma,
                                  grad_outputs=torch.ones_like(f_imag),
                                  create_graph=True)[0]

    return (
        f_real, f_imag,
        grad_fr[:,1:2], grad_fr[:,0:1],
        grad_fi[:,1:2], grad_fi[:,0:1],
    )

def compute_loss(realnet, imagenet,hypernet,t0_collocation, xic_collocation,tic_collocation, x1l_collocation, x1r_collocation,
                 Exact_t0_real_collocation,Exact_t0_image_collocation,
                 Exact_x1l_real_collocation, Exact_x1l_image_collocation,
                 Exact_x1r_real_collocation, Exact_x1r_image_collocation,
                 pt_t, pt_x, pt_t_sigma, pt_x_sigma, pt_f_real_x,
                 pt_f_real_t, pt_f_image_x, pt_f_image_t,
                 pt_f_real, pt_f_image, mse_cost_function1,
                 M, N, tau, alpha, w_grad,kernel_fast, prev_tx_sigma=None,
                 prev_u_real=None,
                 prev_u_imag=None,epoch=0):
    
    net_u_x1l_real = realnet(torch.cat([tic_collocation, x1l_collocation], 1))
    net_u_x1r_real = realnet(torch.cat([tic_collocation, x1r_collocation], 1))
    net_u_x1l_image = imagenet(torch.cat([tic_collocation, x1l_collocation], 1))
    net_u_x1r_image = imagenet(torch.cat([tic_collocation, x1r_collocation], 1))
    net_u_t0_real = realnet(torch.cat([t0_collocation, xic_collocation], 1))
    net_u_t0_image = imagenet(torch.cat([t0_collocation, xic_collocation], 1))

    


    mse_xbc_losses = [mse_cost_function1(Exact_x1l_real_collocation, net_u_x1l_real),
                       mse_cost_function1(Exact_x1r_real_collocation, net_u_x1r_real),
                       mse_cost_function1(Exact_x1l_image_collocation, net_u_x1l_image),
                       mse_cost_function1(Exact_x1r_image_collocation, net_u_x1r_image)]

    mse_t0_real = mse_cost_function1(Exact_t0_real_collocation, net_u_t0_real)
    mse_t0_image = mse_cost_function1(Exact_t0_image_collocation, net_u_t0_image)


    
    f_out_real, f_out_image, grad_x_f_real, grad_t_f_real, grad_x_f_image, grad_t_f_image = fpde_fast(
        torch.cat([pt_t, pt_x], 1),torch.cat([pt_t_sigma, pt_x_sigma], 1), realnet,imagenet, M, N, tau, alpha,kernel_fast, pt_t.device)
    mse_f_real_losses = [mse_cost_function1(grad_x_f_real, pt_f_real_x),mse_cost_function1(grad_t_f_real, pt_f_real_t)]#
    mse_f_image_losses = [mse_cost_function1(grad_x_f_image, pt_f_image_x),mse_cost_function1(grad_t_f_image, pt_f_image_t)]#
    mse_f_real_total = sum(mse_f_real_losses)
    mse_f_image_total = sum(mse_f_image_losses)
    mse_f_1_real = mse_cost_function1(f_out_real, pt_f_real.to(f_out_real))
    mse_f_1_image = mse_cost_function1(f_out_image, pt_f_image.to(f_out_image))
    loss_f=mse_f_1_real+mse_f_1_image
    loss_ic=mse_t0_real+mse_t0_image

    loss_bc= sum(mse_xbc_losses)
    loss_grad = mse_f_real_total+mse_f_image_total
    losses_tensor = torch.stack([loss_f, loss_bc, loss_ic], dim=-1)  # shape (4,),loss_grad
    w_f, w_bc, w_ic = hypernet(losses_tensor.to(dtype=next(hypernet.parameters()).dtype))
    w_g=w_grad*np.exp(-3*loss_f.cpu().data)
    loss_prev = torch.tensor(0.0, device=loss_f.device)

    if prev_tx_sigma is not None:
        u_real_prev_pred = realnet(prev_tx_sigma)
        u_imag_prev_pred = imagenet(prev_tx_sigma)

        loss_prev = mse_cost_function1(u_real_prev_pred, prev_u_real) + \
                    mse_cost_function1(u_imag_prev_pred, prev_u_imag)

        
        w_prev = 1.0 / (epoch + 1)
        if w_prev < 1e-2:w_prev = 0.0
    else:
        w_prev = 0.0
    loss = w_f*loss_f + w_ic*loss_ic +  w_bc*loss_bc + w_g * loss_grad+ w_prev * loss_prev#+w_t1*loss_t1

    return {
        "loss":loss,"loss_f":loss_f,"loss_bc":loss_bc,"loss_ic":loss_ic,
        "loss_grad":loss_grad,"loss_prev":loss_prev,"w_f":w_f,"w_bc":w_bc,
        "w_ic":w_ic,"w_grad":loss_f.new_tensor(w_grad),"w_g":w_g,"w_prev":loss_f.new_tensor(w_prev),
    }

def get_stage_lr(stage_id):
    """
    stage_id = 0,1,2,...
    """
    return max(1e-5, 1e-3 * (0.5 ** stage_id))

def build_stage_data(M, N, alpha, device, network_dtype=torch.float32, kernel_mode='original'):
    """Exact uploaded resampling order; even zero-noise RNG draws are retained."""
    x1l, x1r = 0.0, 1.0
    boundnum = M

    t_ic_var = np.random.uniform(low=0, high=1, size=(boundnum, 1))
    x1_bc_var = np.random.uniform(low=x1l, high=x1r, size=(boundnum, 1))
    x1ll = x1l * np.ones_like(x1_bc_var)
    x1rr = x1r * np.ones_like(x1_bc_var)
    t0ic = 0 * np.ones_like(x1_bc_var)

    Exact_x1l_real = t_ic_var * 0
    Exact_x1r_real = t_ic_var * 0
    Exact_x1l_image = t_ic_var ** 2
    Exact_x1r_image = t_ic_var ** 2
    Exact_ic_real = t_ic_var * 0
    Exact_ic_image = t_ic_var * 0

    x1l_collocation, x1r_collocation, t0ic_collocation, tic_collocation, xbc_collocation = map(
        lambda data: Variable(torch.from_numpy(data).to(dtype=network_dtype), requires_grad=True).to(device),
        [x1ll, x1rr, t0ic, t_ic_var, x1_bc_var]
    )

    Exact_x1l_real_collocation, Exact_x1r_real_collocation, Exact_x1l_image_collocation, Exact_x1r_image_collocation, Exact_ic_real_collocation, Exact_ic_image_collocation = map(
        lambda data: Variable(torch.from_numpy(data).to(dtype=network_dtype), requires_grad=True).to(device),
        [Exact_x1l_real, Exact_x1r_real, Exact_x1l_image, Exact_x1r_image, Exact_ic_real, Exact_ic_image]
    )

    # ---- rebuild grid ----
    t = np.linspace(0, 1, N+1)
    x = np.linspace(x1l, x1r, M+1)
    tau = t[2] - t[1]

    sigma = 1 - alpha / 2
    
    alpha_val = float(alpha.detach().cpu().item()) if torch.is_tensor(alpha) else float(alpha)
    sigma = 1.0 - alpha_val / 2.0
    t_sigma = t + tau * sigma

    t_sigma = t_sigma[0:N]
    x_sigma = x[1:M]

    ms_t, ms_x = np.meshgrid(t, x)
    t = np.ravel(ms_t).reshape(-1,1)
    x = np.ravel(ms_x).reshape(-1,1)

    ms_t_sigma, ms_x_sigma = np.meshgrid(t_sigma, x_sigma)
    t_sigma = np.ravel(ms_t_sigma).reshape(-1,1)
    x_sigma = np.ravel(ms_x_sigma).reshape(-1,1)

    # ---- exact solution ----
    Exact1_real = t**2 * np.sin(2*np.pi*x)
    Exact1_image = t**2 * np.cos(2*np.pi*x)
    Exact1_real_sigma = t_sigma**2 * np.sin(2*np.pi*x_sigma)
    Exact1_image_sigma = t_sigma**2 * np.cos(2*np.pi*x_sigma)

    # ---- torch variables ----
    pt_x, pt_t, pt_u_real, pt_u_image = map(
        lambda d: Variable(torch.from_numpy(d).to(dtype=network_dtype), requires_grad=True).to(device),
        [x, t, Exact1_real, Exact1_image]
    )

    pt_x_sigma, pt_t_sigma, pt_u_real_sigma, pt_u_image_sigma = map(
        lambda d: Variable(torch.from_numpy(d).to(dtype=network_dtype), requires_grad=True).to(device),
        [x_sigma, t_sigma, Exact1_real_sigma, Exact1_image_sigma]
    )
    f_real_sigma = -2 * t_sigma ** (2 - alpha) / gamma(3 - alpha) * np.cos(2 * np.pi * x_sigma) + (
            -4 * np.pi ** 2 * t_sigma ** 2 + t_sigma ** 6) * np.sin(2 * np.pi * x_sigma)
    f_image_sigma = 2 * t_sigma ** (2 - alpha) / gamma(3 - alpha) * np.sin(2 * np.pi * x_sigma) + (
            -4 * np.pi ** 2 * t_sigma ** 2 + t_sigma ** 6) * np.cos(2 * np.pi * x_sigma)

    
    f_real_sigma = -2 * t_sigma ** (2 - alpha) / gamma(3 - alpha) * np.cos(2 * np.pi * x_sigma) + \
                   (-4 * np.pi ** 2 * t_sigma ** 2 + t_sigma ** 6) * np.sin(2 * np.pi * x_sigma)

    f_image_sigma = 2 * t_sigma ** (2 - alpha) / gamma(3 - alpha) * np.sin(2 * np.pi * x_sigma) + \
                    (-4 * np.pi ** 2 * t_sigma ** 2 + t_sigma ** 6) * np.cos(2 * np.pi * x_sigma)

    
    noise_level = 0 # 2%
    f_real_sigma = f_real_sigma + noise_level * np.std(f_real_sigma) * np.random.randn(*np.shape(f_real_sigma))
    f_image_sigma = f_image_sigma + noise_level * np.std(f_image_sigma) * np.random.randn(*np.shape(f_image_sigma))

    f_real_x_sigma = - 2 * np.pi * np.cos(2 * np.pi * x_sigma) * (
            - t_sigma ** 6 + 4 * np.pi ** 2 * t_sigma ** 2)  # +(4 * np.pi * t_sigma **(2 - alpha) * np.sin(2 * np.pi * x_sigma)) / gamma(3 - alpha)
    f_real_t_sigma = - np.sin(2 * np.pi * x_sigma) * (
            - 6 * t_sigma ** 5 + 8 * np.pi ** 2 * t_sigma)  # +(2 * t_sigma ** (1 - alpha) * np.cos(2 * np.pi * x_sigma) * (alpha - 2)) / gamma(3 - alpha)
    f_image_x_sigma = 2 * np.pi * np.sin(2 * np.pi * x_sigma) * (
                - t_sigma ** 6 + 4 * np.pi ** 2 * t_sigma ** 2)  # + (4 * np.pi * t_sigma ** (2 - alpha) * np.cos(2 * np.pi * x_sigma)) / gamma(3 - alpha)
    f_image_t_sigma = - np.cos(2 * np.pi * x_sigma) * (
                - 6 * t_sigma ** 5 + 8 * np.pi ** 2 * t_sigma)  # - (2 * t_sigma ** (1 - alpha) * np.sin(2 * np.pi * x_sigma) * (alpha - 2)) / gamma( 3 - alpha)

    

    pt_f_real, pt_f_real_x, pt_f_real_t = map(
        lambda data: Variable(torch.from_numpy(data).to(dtype=network_dtype), requires_grad=True).to(device),
        [f_real_sigma, f_real_x_sigma, f_real_t_sigma]
    )
    pt_f_image, pt_f_image_x, pt_f_image_t = map(
        lambda data: Variable(torch.from_numpy(data).to(dtype=network_dtype), requires_grad=True).to(device),
        [f_image_sigma, f_image_x_sigma, f_image_t_sigma]
    )
    # ---- rebuild kernel ----
    kernel_fast = build_fastL21sigma_kernel(alpha, tau, device, mode=kernel_mode)

    return {
        "t0ic_collocation":t0ic_collocation,
        "xbc_collocation":xbc_collocation,
        "tic_collocation":tic_collocation,
        "x1l_collocation":x1l_collocation,
        "x1r_collocation":x1r_collocation,
        "Exact_ic_real_collocation":Exact_ic_real_collocation,
        "Exact_ic_image_collocation":Exact_ic_image_collocation,
        "Exact_x1l_real_collocation":Exact_x1l_real_collocation,
        "Exact_x1l_image_collocation":Exact_x1l_image_collocation,
        "Exact_x1r_real_collocation":Exact_x1r_real_collocation,
        "Exact_x1r_image_collocation":Exact_x1r_image_collocation,
        "pt_t":pt_t,
        "pt_x":pt_x,
        "pt_t_sigma":pt_t_sigma,
        "pt_x_sigma":pt_x_sigma,
        "pt_f_real_x":pt_f_real_x,
        "pt_f_real_t":pt_f_real_t,
        "pt_f_image_x":pt_f_image_x,
        "pt_f_image_t":pt_f_image_t,
        "pt_f_real":pt_f_real,
        "pt_f_image":pt_f_image,
        "pt_u_real_sigma":pt_u_real_sigma,
        "pt_u_image_sigma":pt_u_image_sigma,
        "Exact1_real_sigma":Exact1_real_sigma,
        "Exact1_image_sigma":Exact1_image_sigma,
        "tau":tau,
        "kernel_fast":kernel_fast,
    }


# ===== Isolated numerical repair; disabled in the default reproduction mode =====
def build_stable_kernel(alpha, tau, device, Tfinal=1.0, eps=1e-8):
    """Same uploaded SOE nodes/weights. Only stable A/B + float64 history.
    No new Prony tolerance, quadrature rule, architecture, weighting or loss.
    """
    a=float(alpha);sigma=1-a/2
    xs,ws,nexp=sumofexpappr2new(a,eps,sigma*tau,Tfinal)
    q=xs*tau;small=np.abs(q)<.05
    A=np.empty_like(q);B=np.empty_like(q)
    z=q[small];pa=np.zeros_like(z);pb=np.zeros_like(z)
    for j in range(2,15):
        ca=-(2*(-1)**j/math.factorial(j)+3*(-1)**(j-1)/math.factorial(j-1))
        cb=2*(-1)**j/math.factorial(j)+(-1)**(j-1)/math.factorial(j-1)
        pa+=ca*z**(j-2);pb+=cb*z**(j-2)
    A[small]=.5*np.exp(-sigma*z)*pa;B[small]=.5*np.exp(-sigma*z)*pb
    z=q[~small]
    A[~small]=.5*np.exp(-sigma*z)*((2+z)-(2+3*z)*np.exp(-z))/(z*z)
    B[~small]=.5*np.exp(-sigma*z)*((z-2)+(z+2)*np.exp(-z))/(z*z)
    if not all(np.isfinite(v).all() for v in [xs,ws,A,B]):raise ArithmeticError('Nonfinite SOE coefficients.')
    tensor=lambda v:torch.tensor(v,dtype=torch.float64,device=device)
    return {'xs':tensor(xs),'ws':tensor(ws).view(nexp,1),'A_l':tensor(A).view(1,-1),
        'B_l':tensor(B).view(1,-1),'exp_xs_tau':tensor(np.exp(-q)),
        'sigma':tensor(sigma),'tau':tensor(tau),'nexp':nexp,'mode':'stable64'}


def build_fastL21sigma_kernel(alpha,tau,device,mode='original'):
    if mode=='original':return build_original_kernel(alpha,tau,device)
    if mode=='stable64':return build_stable_kernel(alpha,tau,device)
    raise ValueError('Choose kernel mode original or stable64.')


def create_training_objects(args,device):
    """Keep original construction order, external shared tensors and 3 Adam states."""
    dtype=torch.float32 if args.network_dtype=='float32' else torch.float64
    a1=torch.from_numpy(np.array([.1])).to(device=device,dtype=dtype).requires_grad_(True)
    a2=torch.from_numpy(np.array([.1])).to(device=device,dtype=dtype).requires_grad_(True)
    fn=torch.from_numpy(np.array([10])).to(device=device,dtype=dtype).requires_grad_(True)
    a1.grad=torch.ones_like(a1);a2.grad=torch.ones_like(a2);fn.grad=torch.ones_like(fn)
    real=FastNetwork(args.extra_hidden_layers,2,args.width,torch.tanh,a1,a2,fn).to(device=device,dtype=dtype)
    imag=FastNetwork(args.extra_hidden_layers,2,args.width,torch.tanh,a1,a2,fn).to(device=device,dtype=dtype)
    hyper=HyperNet(w_a=.6,w_b=1.5).to(device=device,dtype=dtype)
    real_optimizer=torch.optim.Adam(real.parameters(),lr=1e-3)
    real_optimizer.add_param_group({'params':[a1],'lr':5e-4})
    real_optimizer.add_param_group({'params':[a2],'lr':5e-4})
    imag_optimizer=torch.optim.Adam(imag.parameters(),lr=1e-3)
    if args.shared_update=='legacy_double':
        imag_optimizer.add_param_group({'params':[a1],'lr':5e-4})
        imag_optimizer.add_param_group({'params':[a2],'lr':5e-4})
    # shared_update=once makes only the real optimizer own shared a1/a2;
    # their gradient still includes BOTH real and imaginary paths.
    hyper_optimizer=torch.optim.Adam(hyper.parameters(),lr=1e-3)
    return {'realnet':real,'imagenet':imag,'hypernet':hyper,'a1':a1,'a2':a2,'fn':fn,
        'optimizers':[real_optimizer,imag_optimizer,hyper_optimizer],'dtype':dtype,'device':device}


LOSS_FIELDS=['t0ic_collocation','xbc_collocation','tic_collocation','x1l_collocation','x1r_collocation',
    'Exact_ic_real_collocation','Exact_ic_image_collocation','Exact_x1l_real_collocation','Exact_x1l_image_collocation',
    'Exact_x1r_real_collocation','Exact_x1r_image_collocation','pt_t','pt_x','pt_t_sigma','pt_x_sigma',
    'pt_f_real_x','pt_f_real_t','pt_f_image_x','pt_f_image_t','pt_f_real','pt_f_image']


def stage_loss(state,data,M,N,alpha,w_grad,prev_tx=None,prev_u=None,prev_v=None,epoch=0):
    return compute_loss(state['realnet'],state['imagenet'],state['hypernet'],
        *[data[k] for k in LOSS_FIELDS],torch.nn.MSELoss(reduction='mean'),
        M,N,data['tau'],alpha,w_grad,data['kernel_fast'],prev_tx,prev_u,prev_v,epoch)


def training_step(state,data,M,N,alpha,w_grad,prev=None,epoch=0):
    # Original order: compute the loss, then clear all three optimizers,
    # backward once, step real, image, HyperNet in that order.
    prev=(None,None,None) if prev is None else prev
    result=stage_loss(state,data,M,N,alpha,w_grad,*prev,epoch=epoch)
    if not bool(torch.isfinite(result['loss'])):raise FloatingPointError('Nonfinite training loss.')
    for optimizer in state['optimizers']:optimizer.zero_grad()
    result['loss'].backward()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for optimizer in state['optimizers'] for group in optimizer.param_groups for p in group['params']):
        raise FloatingPointError('Nonfinite trainable gradients.')
    for optimizer in state['optimizers']:optimizer.step()
    return result


def prediction(state,tx):
    with torch.no_grad():return torch.cat([state['realnet'](tx),state['imagenet'](tx)],1)


def exact_solution(tx):
    t=tx[:,0:1];x=tx[:,1:2]
    return t*t*torch.cat([torch.sin(2*np.pi*x),torch.cos(2*np.pi*x)],1)


def relative_l2(pred,target):
    # Compute report metrics in double without changing network/optimizer arithmetic.
    p=pred.detach().to(torch.float64);y=target.detach().to(torch.float64)
    return float(torch.linalg.vector_norm(p-y)/torch.linalg.vector_norm(y).clamp_min(1e-30))


def uploaded_sigma_l2(state,data):
    """Same numpy complex norm and target arrays as the pasted code."""
    tx=torch.cat([data['pt_t_sigma'],data['pt_x_sigma']],1)
    pred=prediction(state,tx)
    er=(pred[:,0:1]-data['pt_u_real_sigma']).detach().cpu().numpy()
    ei=(pred[:,1:2]-data['pt_u_image_sigma']).detach().cpu().numpy()
    return float(np.linalg.norm(er+1j*ei)/np.linalg.norm(data['Exact1_real_sigma']+1j*data['Exact1_image_sigma'])),tx,pred


def independent_points(seed,count,device,dtype):
    # Separate generator so logging/evaluation never consumes the original
    # np.random training stream (including its zero-noise randn draws).
    rng=np.random.default_rng(seed)
    tx=torch.tensor(rng.uniform(0,1,(count,2)),device=device,dtype=dtype)
    return tx,exact_solution(tx).detach()


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def make_checkpoint(state,args,stage,epoch,M,N,metrics,prev):
    return copy.deepcopy({'format':'C01_original_strategy_v3','config':vars(args).copy(),
        'stage':stage,'epoch':epoch,'M':M,'N':N,'metrics':metrics,
        'realnet':state['realnet'].state_dict(),'imagenet':state['imagenet'].state_dict(),
        'hypernet':state['hypernet'].state_dict(),
        'a1':state['a1'].detach(),'a2':state['a2'].detach(),'fn':state['fn'].detach(),
        'optimizers':[x.state_dict() for x in state['optimizers']],
        'previous':tuple(x.detach() if x is not None else None for x in prev),
        'rng':rng_state()})


def apply_checkpoint(state,ckpt,with_optimizers=False):
    # Copy values into the SAME external activation tensors referenced by every block.
    state['realnet'].load_state_dict(ckpt['realnet']);state['imagenet'].load_state_dict(ckpt['imagenet'])
    state['hypernet'].load_state_dict(ckpt['hypernet'])
    with torch.no_grad():
        for k in ('a1','a2','fn'):state[k].copy_(ckpt[k].to(state[k]))
    if with_optimizers:
        for opt,saved in zip(state['optimizers'],ckpt['optimizers']):opt.load_state_dict(saved)


def load_for_prediction(path,device='cpu'):
    """Use only checkpoints generated by this script. Legacy whole-model pickles
    have a different format and are deliberately not loaded implicitly.
    """
    ckpt=torch.load(path,map_location=device,weights_only=False)
    if ckpt.get('format')!='C01_original_strategy_v3':raise ValueError('Wrong checkpoint format.')
    args=argparse.Namespace(**ckpt['config']);state=create_training_objects(args,torch.device(device))
    apply_checkpoint(state,ckpt)
    return state,ckpt


def flush_log(out,records):
    keys=['stage','epoch','global_step','loss','loss_f','loss_bc','loss_ic','loss_grad','loss_prev',
          'w_f','w_bc','w_ic','w_grad','w_g','w_prev','a1','a2','l2','sigma_l2','validation_l2','elapsed_seconds']
    with (out/'training.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)
    np.savez(out/'train_losses_by_stage.npz',**{k:np.asarray([r.get(k,np.nan) for r in records]) for k in keys})


def parse_grids(value):
    try:
        result=[tuple(int(n) for n in s.lower().split('x')) for s in value.split(',')]
        if not all(len(x)==2 and min(x)>=2 for x in result):raise ValueError()
        return result
    except ValueError:raise argparse.ArgumentTypeError('Example: 10x10,20x20,30x30,40x40,50x50')


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description='C01 v3 based on the uploaded successful solver. Defaults preserve its algorithm.')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--output',default='runs/C01_v3_original_seed1234')
    parser.add_argument('--kernel-mode',choices=['original','stable64'],default='original')
    parser.add_argument('--network-dtype',choices=['float32','float64'],default='float32')
    parser.add_argument('--shared-update',choices=['legacy_double','once'],default='legacy_double')
    parser.add_argument('--transfer-mode',choices=['rolling','frozen_stage'],default='frozen_stage')
    parser.add_argument('--grids',type=parse_grids,default=[(10,10),(20,20),(30,30),(40,40),(50,50)])
    parser.add_argument('--width',type=int,default=60)
    parser.add_argument('--extra-hidden-layers',type=int,default=2)
    parser.add_argument('--alpha',type=float,default=.9)
    parser.add_argument('--w-grad',type=float,default=.001)
    parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--patience-evals',type=int,default=10)
    parser.add_argument('--tol',type=float,default=0.)
    parser.add_argument('--max-epochs-per-stage',type=int,default=0,help='0 keeps original uncapped loss-stagnation stopping; positive value enables an explicit cap.')
    parser.add_argument('--independent-points',type=int,default=4096)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(argv)
    if args.smoke:
        args.grids=[(4,4),(6,6)];args.width=8;args.extra_hidden_layers=1
        args.max_epochs_per_stage=3;args.eval_every=1;args.independent_points=64
    if not 0<args.alpha<1:parser.error('0<alpha<1 is required.')
    if min(args.width,args.eval_every,args.patience_evals,args.independent_points)<1:parser.error('Counts must be positive.')
    if args.extra_hidden_layers<0 or args.max_epochs_per_stage<0 or args.w_grad<0 or args.tol<0:parser.error('Negative configuration value.')
    if args.network_dtype=='float64' and args.kernel_mode=='original':parser.error('Keep network float32 for original-mode reproduction; use stable64 when testing a double network.')
    return args


def run(args):
    device=torch.device(args.device)
    if args.threads>0:torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'run_config.json').exists() and not args.overwrite:raise FileExistsError('Existing run directory; choose a new --output.')
    print(f'C01 v3 | kernel={args.kernel_mode} | network={args.network_dtype} | shared_update={args.shared_update} | transfer={args.transfer_mode} | stage_cap={args.max_epochs_per_stage}',flush=True)
    config={'version':'v3_uploaded_strategy','args':vars(args),
        'source':'pasted Markdown (1)(7), restored from HTML/Markdown escapes',
        'original_stopping':'pre-update training loss at scheduled evaluations; stage models/Adam states are NOT reset or restored',
        'regularizer':'local (u_xx+|psi|^2*u, v_xx+|psi|^2*v) derivatives in x AND t; targets match local operator; NOT full PDE residual gradient',
        'weighting':'uploaded trainable HyperNet; stochastic 1-phi/1+phi intervals; w_g=w0*exp(-3*Lf.detach())',
        'shared_updates_per_step':2 if args.shared_update=='legacy_double' else 1,
        'kernel_warning':'original float32 coefficients have a demonstrated cancellation error; this mode is for faithful reproduction and controlled comparison',
        'test_used_for_training_or_selection':False,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':str(device)}}
    (out/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    state=create_training_objects(args,device);dtype=state['dtype']
    val_tx,val_y=independent_points(args.seed+10000,args.independent_points,device,dtype)
    test_tx,test_y=independent_points(args.seed+20000,args.independent_points,device,dtype)
    records=[];stages=[];global_step=0;start=time.time();prev=(None,None,None)
    best_sigma=float('inf');best_val=float('inf');sigma_ckpt=None;val_ckpt=None
    for sid,(M,N) in enumerate(args.grids):
        new_lr=get_stage_lr(sid)
        for optimizer in state['optimizers']:
            for group in optimizer.param_groups:group['lr']=new_lr
        print(f'\n========== Stage {sid}: Grid {M} x {N} ==========')
        print(f'[Stage {sid}] Learning rate set to {new_lr:.2e}',flush=True)
        data=build_stage_data(M,N,args.alpha,device,dtype,args.kernel_mode)
        # Keep global Adam state, a1/a2 and model weights across every grid stage.
        best_loss=float('inf');no_improve=0;epoch=0
        frozen_prev=prev;reason=None
        while True:
            active_prev=(None,None,None) if sid==0 else (prev if args.transfer_mode=='rolling' else frozen_prev)
            terms=training_step(state,data,M,N,args.alpha,args.w_grad,active_prev,epoch)
            global_step+=1
            row={k:float(terms[k].detach().cpu()) for k in ['loss','loss_f','loss_bc','loss_ic','loss_grad','loss_prev','w_f','w_bc','w_ic','w_grad','w_g','w_prev']}
            row.update(stage=sid,epoch=epoch,global_step=global_step,a1=float(state['a1'].detach().cpu()),
                       a2=float(state['a2'].detach().cpu()),elapsed_seconds=time.time()-start,
                       l2=float('nan'),sigma_l2=float('nan'),validation_l2=float('nan'))
            scheduled=epoch%args.eval_every==0
            if scheduled:
                sigma_l2,tx,pred=uploaded_sigma_l2(state,data)
                val_l2=relative_l2(prediction(state,val_tx),val_y)
                prev=(tx.detach(),pred[:,0:1].detach(),pred[:,1:2].detach())
                row.update(l2=sigma_l2,sigma_l2=sigma_l2,validation_l2=val_l2)
                metrics={'sigma_l2':sigma_l2,'validation_l2':val_l2,'pre_update_loss':row['loss']}
                ckpt=make_checkpoint(state,args,sid,epoch,M,N,metrics,prev)
                if sigma_l2<best_sigma:
                    best_sigma=sigma_l2;sigma_ckpt=ckpt
                    torch.save(ckpt,out/'original_best_sigma.pt')
                if val_l2<best_val:
                    best_val=val_l2;val_ckpt=ckpt
                    torch.save(ckpt,out/'best_validation.pt')
                # Main console L2 is the uploaded sigma-grid metric. Independent
                # validation stays in result files, with a separate field name.
                print(f'[Stage {sid}] Epoch {epoch:6d} | '
                      f'Loss {row["loss"]:.3e} | L2 {sigma_l2:.3e} |'
                      f'w_f {row["w_f"]:.3e} | w_ic {row["w_ic"]:.3e} | w_bc {row["w_bc"]:.3e}',flush=True)
                if row['loss']<best_loss-args.tol:best_loss=row['loss'];no_improve=0
                else:no_improve+=1
            records.append(row)
            if scheduled and no_improve>=args.patience_evals:reason='original_loss_stagnation';break
            if args.max_epochs_per_stage and epoch+1>=args.max_epochs_per_stage:reason='explicit_epoch_cap';break
            epoch+=1
        # Stage transition uses LAST weights/moments, not a validation-best restore.
        last_ckpt=make_checkpoint(state,args,sid,epoch,M,N,{'pre_update_loss':row['loss']},prev)
        torch.save(last_ckpt,out/f'stage_{sid:02d}_last.pt')
        flush_log(out,records)
        stages.append({'stage':sid,'M':M,'N':N,'epochs':epoch+1,'lr':new_lr,'stop_reason':reason,
                       'boundary_points':M,'kernel_mode':args.kernel_mode,'last_loss':row['loss']})
    # Model choices are fixed before any test values are used. Report all choices
    # for diagnosis; publication model is chosen only by the independent validation.
    last_test=relative_l2(prediction(state,test_tx),test_y)
    apply_checkpoint(state,sigma_ckpt)
    sigma_test=relative_l2(prediction(state,test_tx),test_y)
    apply_checkpoint(state,val_ckpt)
    val_test=relative_l2(prediction(state,test_tx),test_y)
    torch.save(val_ckpt,out/'final.pt')
    np.savez(out/'independent_predictions.npz',tx=test_tx.detach().cpu().numpy(),
        prediction=prediction(state,test_tx).cpu().numpy(),exact=test_y.cpu().numpy())
    summary={'version':'v3_uploaded_strategy','kernel_mode':args.kernel_mode,'shared_update':args.shared_update,
        'transfer_mode':args.transfer_mode,'seed':args.seed,'stages':stages,'total_steps':global_step,
        'original_min_sigma_l2':best_sigma,'independent_best_validation_l2':best_val,
        'test':{'last_model_l2':last_test,'original_sigma_selected_model_l2':sigma_test,
                'independent_validation_selected_model_l2':val_test},
        'publication_model_selection':'independent validation only; final.pt mirrors best_validation.pt',
        'full_gpu_training_performed_here':device.type=='cuda','reproduced_user_best_result':False,
        'status':'SMOKE_CHECK_NOT_ACCURACY_RESULT' if args.smoke else 'TRAINING_COMPLETED_REQUIRES_ACCURACY_REVIEW',
        'elapsed_seconds':time.time()-start}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


if __name__=='__main__':
    run(parse_args())
