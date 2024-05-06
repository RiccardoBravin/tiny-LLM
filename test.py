import torch
from lib.BRAV import Reader

x = torch.randn(32, 64, 128)

cls = Reader(128)

tokens = cls(x)
print(tokens.size())