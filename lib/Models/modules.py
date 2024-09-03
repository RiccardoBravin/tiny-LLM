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
    

def make_score_mask(input_tensor):
    """
    Args:
        input_tensor: the input tensor of shape (batch_size, max_len)
    Returns:
        Tensor of shape (batch_size, max_len, max_len)
    """
    # Get batch size and sentence length
    batch_size, max_length = input_tensor.shape

    # Initialize the output tensor with zeros
    output_tensor = torch.zeros((batch_size, max_length, max_length), dtype=input_tensor.dtype, device=input_tensor.device)

    # Fill the output tensor according to the input tensor values
    for i in range(batch_size):
        sentence_len = (input_tensor[i] != 0).sum()  # Get the non-zero length (count of 1s)
        output_tensor[i, :sentence_len, :] = input_tensor[i]

    return output_tensor