"""Network components for one-dimensional potential reconstruction."""
import torch
from torch import nn
from torch.nn import functional as F

class RealNet(nn.Module):
    def __init__(self, num_hidden_layers, input_size, hidden_size, activation, a_1,a_2, fn):
        super(RealNet, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 1)
        self.active = activation
        self.a_1 = a_1
        self.a_2 = a_2
        self.fn = fn

    def forward(self, x):
        out = self.active(self.fn * self.a_2* self.input_layer(x)*(self.active(self.fn * self.a_1 * self.input_layer(x))+1))
        for hidden_layer in self.hidden_layers:
            out = self.active(self.fn * self.a_2 * hidden_layer(out)*(self.active(self.fn * self.a_1 * hidden_layer(out))+1))
        out_NN = self.output_layer(out)
        out_real = out_NN[:, 0]  
        xs_real = 1#x[:, 0]
        out_final_real = torch.mul(xs_real, out_real)
        size_out_real = out_final_real.shape[0]
        out_final_real = out_final_real.reshape(size_out_real, 1)
        return out_final_real

class ImageNet(nn.Module):
    def __init__(self, num_hidden_layers, input_size, hidden_size, activation, a_1,a_2, fn):
        super(ImageNet, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 1)
        self.active = activation
        self.a_1 = a_1
        self.a_2 = a_2
        self.fn = fn

    def forward(self, x):

        out = self.active(self.fn * self.a_2* self.input_layer(x)*(self.active(self.fn * self.a_1 * self.input_layer(x))+1))
        for hidden_layer in self.hidden_layers:
            out = self.active(self.fn * self.a_2 * hidden_layer(out)*(self.active(self.fn * self.a_1 * hidden_layer(out))+1))
        out_NN = self.output_layer(out)
        out_image = out_NN[:,0]  
        xs_image =1#x[:, 0]
        out_final_image = torch.mul(xs_image, out_image)
        size_out_image = out_final_image.shape[0]
        out_final_image = out_final_image.reshape(size_out_image, 1)
        return out_final_image

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
        losses_phys=losses[..., :3]
        normalized=losses_phys/losses_phys.mean(dim=-1,keepdim=True).clamp_min(1e-8)
        logits=self.fc2_phys(torch.tanh(self.fc1_phys(normalized)))
        attention=F.softmax(logits,dim=-1)
        noise=torch.empty_like(attention).uniform_(-.05,.05)
        return (3*attention+noise).clamp_min(1e-6)
