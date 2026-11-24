"""Adapters around pristine uploaded C01 numerical/network source files."""
import copy
from pathlib import Path
import sys
import types

import numpy as np
import torch

LEGACY = Path(__file__).resolve().parent/'legacy_ablation'
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
import baseline_original as b
import run_c01_ablation as original
from direct_reference import direct_caputo
from weight_schemes import install_weights
import fractional_schemes as schemes

ORIGINAL_BUILD_DATA = b.build_stage_data
ORIGINAL_FPDE = b.fpde_fast
_ACTIVE_ARGS = None


def _tanh_forward(self, value):
    return torch.tanh(self.linear(value))


def configure(args, device):
    global _ACTIVE_ARGS
    _ACTIVE_ARGS = args
    factory_args = copy.copy(args)
    factory_args.variant = 'fixed_operator_branch' if args.branch == 'operator' else 'full_original'
    state = original.configure(factory_args, device)
    if args.branch == 'operator':
        for key in ('realnet', 'imagenet', 'branch'):
            state[key].to(device=device, dtype=state['dtype'])
    if args.network_activation == 'tanh':
        for key in ('realnet', 'imagenet'):
            net = state[key].trunk if args.branch == 'operator' else state[key]
            for block in net.layers:
                block.forward = types.MethodType(_tanh_forward, block)
    if not args.use_attention:
        state['hypernet'] = original.FixedWeights(state['dtype'], device)
        state['optimizers'] = state['optimizers'][:2]
    else:
        install_weights(state, 'softmax_noise', args.noise_amplitude,
                        args.weight_lower_bound, args.weight_floor)
    b.build_stage_data = build_stage_data
    b.fpde_fast = fpde
    return state


def build_stage_data(M, N, alpha, device, network_dtype=torch.float32, kernel_mode='original'):
    data = ORIGINAL_BUILD_DATA(M, N, alpha, device, network_dtype, kernel_mode)
    scheme = _ACTIVE_ARGS.fractional_scheme
    if scheme == 'fast_l21':
        return data
    data['kernel_fast'] = schemes.build_kernel(alpha, data['tau'], N, device, scheme, b)
    if scheme in ('l1', 'fast_l1'):
        times = np.tile(np.linspace(0, 1, N+1)[1:], M-1).reshape(-1, 1)
        space = data['pt_x_sigma'].detach().cpu().double().numpy()
        targets = schemes.manufactured_data(times, space, alpha)
        data['pt_t_sigma'] = torch.tensor(times, device=device, dtype=network_dtype, requires_grad=True)
        for key, value in targets.items():
            if key.startswith('Exact1_'):
                data[key] = value
            else:
                data[key] = torch.tensor(value, device=device, dtype=network_dtype, requires_grad=True)
        data['pt_u_real_sigma'] = torch.tensor(targets['Exact1_real_sigma'], device=device, dtype=network_dtype, requires_grad=True)
        data['pt_u_image_sigma'] = torch.tensor(targets['Exact1_image_sigma'], device=device, dtype=network_dtype, requires_grad=True)
    return data


def fpde(tx, tx_sigma, realnet, imagenet, M, N, tau, alpha, kernel, device):
    if _ACTIVE_ARGS.fractional_scheme == 'fast_l21' and _ACTIVE_ARGS.use_gradient:
        return ORIGINAL_FPDE(tx, tx_sigma, realnet, imagenet, M, N, tau, alpha, kernel, device)
    # Preserve the uploaded forward order and local regularizer arithmetic.
    u = realnet(tx).view(M+1, N+1)
    v = imagenet(tx).view(M+1, N+1)
    us, vs = realnet(tx_sigma), imagenet(tx_sigma)
    if _ACTIVE_ARGS.fractional_scheme == 'fast_l21':
        # Same recurrence, with only the unused third derivatives omitted.
        kdtype = kernel['ws'].dtype
        aa = torch.tensor([alpha], device=device, dtype=kdtype)
        gi = 1.0/b.gamma_torch(1-aa)
        coeff = b.a_hat(1-aa/2, aa, kernel['tau'])/b.gamma_torch(1-aa)
        histories = []
        for values in (u, v):
            memory = torch.zeros((M-1, kernel['nexp']), device=device, dtype=kdtype)
            derivatives = []
            for n in range(1, N+1):
                current = (values[1:M, n]-values[1:M, n-1]).unsqueeze(-1).to(kdtype)
                if n > 1:
                    previous = (values[1:M, n-1]-values[1:M, n-2]).unsqueeze(-1).to(kdtype)
                    memory = memory*kernel['exp_xs_tau']+previous@kernel['A_l']+current@kernel['B_l']
                derivatives.append(gi*(memory@kernel['ws']).squeeze(-1)+coeff*current.squeeze(-1))
            histories.append(torch.stack(derivatives, dim=1).reshape(-1, 1))
        du, dv = histories
    else:
        du = schemes.apply_history(u[1:M], alpha, tau, kernel, direct_caputo).reshape(-1, 1)
        dv = schemes.apply_history(v[1:M], alpha, tau, kernel, direct_caputo).reshape(-1, 1)
    ux = torch.autograd.grad(us, tx_sigma, torch.ones_like(us), create_graph=True)[0][:, 1:2]
    vx = torch.autograd.grad(vs, tx_sigma, torch.ones_like(vs), create_graph=True)[0][:, 1:2]
    uxx = torch.autograd.grad(ux, tx_sigma, torch.ones_like(ux), create_graph=True)[0][:, 1:2]
    vxx = torch.autograd.grad(vx, tx_sigma, torch.ones_like(vx), create_graph=True)[0][:, 1:2]
    amp2 = us**2+vs**2
    local_u, local_v = uxx+amp2*us, vxx+amp2*vs
    fr, fi = -dv+local_u, du+local_v
    if not _ACTIVE_ARGS.use_gradient:
        zero = torch.zeros_like(us)
        return fr, fi, zero, zero, zero, zero
    gu = torch.autograd.grad(local_u, tx_sigma, torch.ones_like(fr), create_graph=True)[0]
    gv = torch.autograd.grad(local_v, tx_sigma, torch.ones_like(fi), create_graph=True)[0]
    return fr, fi, gu[:, 1:2], gu[:, 0:1], gv[:, 1:2], gv[:, 0:1]
