"""Network definitions extracted unchanged from supplied C07; no training side effects."""
import math
import torch
import torch.nn as nn

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
        self.mu_min = torch.tensor([0.6, 0.5, 1], dtype=torch.float32)
        self.mu_max = torch.tensor([0.9, 1.5, 2], dtype=torch.float32)
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
