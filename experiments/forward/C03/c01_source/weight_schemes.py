"""Two requested loss-weight rules; reuse the original C01 MLP parameters.

The wrappers do not initialize any new trainable parameters. This retains
the same predictor/MLP initialization and Adam parameter ownership.
"""
import torch
from torch import nn
from torch.nn import functional as F


SCHEMES = ('softmax_noise', 'softplus_hybrid', 'original')


class SharedWeightMLP(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.fc1_phys = original.fc1_phys
        self.fc2_phys = original.fc2_phys
        self.stage = 0

    def set_stage(self, stage):
        if int(stage) < 0:
            raise ValueError('Stage must be nonnegative.')
        self.stage = int(stage)

    def logits(self, inputs):
        return self.fc2_phys(torch.tanh(self.fc1_phys(inputs)))


class SoftmaxNoiseWeights(SharedWeightMLP):
    """Method 1: mean-normalized loss -> MLP -> softmax -> K*phi+xi.

    Loss inputs are NOT detached, consistent with the original C01 path.
    The manuscript leaves the perturbation amplitude unspecified. Here it
    is explicitly configurable, with xi ~ Uniform(-0.05, 0.05) by default.
    """
    def __init__(self, original, noise_amplitude=0.05, lower_bound=1e-6):
        super().__init__(original)
        self.noise_amplitude = float(noise_amplitude)
        self.lower_bound = float(lower_bound)

    def forward(self, losses):
        normalized = losses / losses.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        phi = F.softmax(self.logits(normalized), dim=-1)
        xi = self.noise_amplitude * (2.0 * torch.rand_like(phi) - 1.0)
        return (losses.shape[-1] * phi + xi).clamp_min(self.lower_bound)


class SoftplusHybridWeights(SharedWeightMLP):
    """Method 2: detached log-loss -> MLP -> Softplus -> normalized scores.

    S0 is the exact operator-V3 rule. S1 and all subsequent C01 stages use
    the operator-V5 affine floor. C01 retains its original FIVE grids.
    """
    def __init__(self, original, weight_floor=0.2):
        super().__init__(original)
        self.weight_floor = float(weight_floor)

    def forward(self, losses):
        positive = F.softplus(self.logits(torch.log(losses.detach() + 1e-8)))
        if self.stage == 0:
            return losses.shape[-1] * positive / (positive.sum(dim=-1, keepdim=True) + 1e-12)
        positive = positive.clamp_min(torch.finfo(positive.dtype).tiny)
        normalized = losses.shape[-1] * positive / positive.sum(dim=-1, keepdim=True)
        return self.weight_floor + (1.0 - self.weight_floor) * normalized


def install_weights(state, scheme, noise_amplitude=0.05, lower_bound=1e-6, weight_floor=0.2):
    if scheme not in SCHEMES:
        raise ValueError('Unknown weight scheme: ' + str(scheme))
    original = state['hypernet']
    # The original no-attention variants already replace HyperNet with ones.
    if scheme == 'original' or not hasattr(original, 'fc1_phys'):
        return state
    if scheme == 'softmax_noise':
        replacement = SoftmaxNoiseWeights(original, noise_amplitude, lower_bound)
    else:
        replacement = SoftplusHybridWeights(original, weight_floor)
    assert list(replacement.parameters()) == list(original.parameters())
    state['hypernet'] = replacement
    return state


def set_stage(state, stage):
    if hasattr(state['hypernet'], 'set_stage'):
        state['hypernet'].set_stage(stage)


def description(scheme, noise_amplitude=0.05, lower_bound=1e-6, weight_floor=0.2):
    common = {'loss_order': ['PDE', 'BC', 'IC'], 'K': 3,
              'architecture': 'Linear(3,16)-Tanh-Linear(16,3)',
              'same_initial_MLP_parameters': True,
              'gradient_and_consistency_terms_excluded': True}
    if scheme == 'softmax_noise':
        common.update(input_transform='loss/mean(loss), mean clamped at 1e-8',
                      detach_loss_inputs=False, output_activation='softmax',
                      formula='max(3*phi + xi, epsilon)',
                      perturbation='Uniform(-amplitude, amplitude)',
                      noise_amplitude=noise_amplitude, epsilon=lower_bound,
                      stages='same rule in every stage')
    elif scheme == 'softplus_hybrid':
        common.update(input_transform='log(loss+1e-8)', detach_loss_inputs=True,
                      output_activation='Softplus', stochastic_perturbation=False,
                      S0_formula='3*q/(sum(q)+1e-12)',
                      later_stage_formula='delta+(1-delta)*3*q/sum(q)',
                      weight_floor=weight_floor, floor_start_stage=1,
                      later_stage_sum=3.0, later_stage_upper_bound=3-2*weight_floor)
    else:
        common.update(formula='original C01 two-branch softmax rule',
                      stages='original unmodified HyperNet')
    return common
