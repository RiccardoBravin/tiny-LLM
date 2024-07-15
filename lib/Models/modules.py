import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    

class NoNorm(nn.Module):
    def __init__(self, feat_size, eps=None):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(feat_size))
        self.weight = nn.Parameter(torch.ones(feat_size)/ feat_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor * self.weight + self.bias