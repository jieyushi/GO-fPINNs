"""Network components for two-dimensional potential reconstruction."""
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
    def __init__(self):
        super().__init__()
        self.fc1_phys = nn.Linear(4, 16)
        self.fc2_phys = nn.Linear(16, 4)

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        losses_phys=losses[..., :4]
        normalized=losses_phys/losses_phys.mean(dim=-1,keepdim=True).clamp_min(1e-8)
        logits=self.fc2_phys(torch.tanh(self.fc1_phys(normalized)))
        attention=F.softmax(logits,dim=-1)
        noise=torch.empty_like(attention).uniform_(-.05,.05)
        return (4*attention+noise).clamp_min(1e-6)

class EtaBasis2D(nn.Module):
    """
    Represents a 2D potential eta(x,y) using a truncated Fourier cosine series.
    The potential is parameterized as:
      eta(x,y) = sum_{m=0 to M} sum_{n=0 to N} c_{m,n} * cos(m*x) * cos(n*y)

    This class is designed for multi-stage training where the number of basis
    functions (M and N) is progressively increased. The `from_prev` classmethod
    facilitates transfer learning between these stages.
    """

    def __init__(self, M, N, device=None, dtype=torch.float32):
        """
        Initializes the basis with maximum frequencies M for x and N for y.

        Args:
            M (int): Maximum frequency for the x-dimension.
            N (int): Maximum frequency for the y-dimension.
            device (torch.device, optional): The device to place tensors on.
            dtype (torch.dtype, optional): The data type for the tensors.
        """
        super().__init__()
        self.M = M
        self.N = N
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype

        # Coefficients c_{m,n} are stored in a single (M+1) x (N+1) parameter matrix.
        self.c = nn.Parameter(
            torch.zeros(self.M + 1, self.N + 1, dtype=self.dtype, device=self.device)
        )

    @classmethod
    def from_prev(cls, prev_model, M_new, N_new):
        """
        Creates a new model by warm-starting from a previous, smaller model.

        This method is the core of the multi-stage transfer learning. It takes the
        coefficients from the previous model and places them in the top-left
        corner of the new, larger coefficient matrix. All newly added
        coefficients are initialized to zero.

        Args:
            prev_model (EtaBasis2D): The trained model from the previous stage.
            M_new (int): The new maximum frequency for the x-dimension.
            N_new (int): The new maximum frequency for the y-dimension.

        Returns:
            EtaBasis2D: A new model instance ready for the next training stage.
        """
        M_prev, N_prev = prev_model.M, prev_model.N
        assert M_new >= M_prev and N_new >= N_prev, "New frequencies must be >= previous ones."

        # Create a new model instance with the new dimensions and device/dtype
        model = cls(
            M=M_new,
            N=N_new,
            device=prev_model.device,
            dtype=prev_model.dtype
        )

        # Create a new zero-initialized coefficient matrix
        c_init = torch.zeros(M_new + 1, N_new + 1, dtype=model.dtype, device=model.device)

        # Copy the learned coefficients from the previous model
        with torch.no_grad():
            c_prev = prev_model.c.data
            c_init[:M_prev + 1, :N_prev + 1] = c_prev

            # Assign the warm-started matrix to the new model's parameter
            model.c.data.copy_(c_init)

        return model

    def forward(self, xcol: torch.Tensor, ycol: torch.Tensor) -> torch.Tensor:
        """
        Computes the potential eta(x,y) for the given input coordinates.

        Args:
            xcol (torch.Tensor): A tensor of x-coordinates, shape (num_points, 1).
            ycol (torch.Tensor): A tensor of y-coordinates, shape (num_points, 1).

        Returns:
            torch.Tensor: The computed potential eta, shape (num_points, 1).
        """
        # Generate basis matrices for x and y coordinates
        # Phi_x will have shape (num_points, M+1)
        # Phi_y will have shape (num_points, N+1)
        m_range = torch.arange(0, self.M + 1, device=self.device, dtype=self.dtype).view(1, -1)
        n_range = torch.arange(0, self.N + 1, device=self.device, dtype=self.dtype).view(1, -1)

        Phi_x = torch.cos(xcol * m_range*torch.pi*2)
        Phi_y = torch.cos(ycol * n_range*torch.pi*2)

        # Efficiently compute the sum using einsum:
        # For each point 'p', we want sum_{m,n} Phi_x[p,m] * c[m,n] * Phi_y[p,n]
        # This is equivalent to the diagonal of Phi_x @ c @ Phi_y.T
        eta = torch.einsum('pm,mn,pn->p', Phi_x, self.c, Phi_y).unsqueeze(1)

        return eta
