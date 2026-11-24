"""Network components for constant-coefficient identification."""
import torch
from torch import nn
from torch.nn import functional as F

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
    """
    Hypernetwork generating adaptive weights for [w_f, w_bc, w_ic, w_t1].
    Inputs: losses tensor [..., 4]
    Outputs: weights tensor [..., 4]
    """
    def __init__(self, w_a, w_b):
        super().__init__()
        self.fc1_phys = nn.Linear(4, 16)
        self.fc2_phys = nn.Linear(16, 4)
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        losses_phys=losses[..., :4]
        normalized=losses_phys/losses_phys.mean(dim=-1,keepdim=True).clamp_min(1e-8)
        logits=self.fc2_phys(torch.tanh(self.fc1_phys(normalized)))
        attention=F.softmax(logits,dim=-1)
        noise=torch.empty_like(attention).uniform_(-.05,.05)
        return (4*attention+noise).clamp_min(1e-6)
