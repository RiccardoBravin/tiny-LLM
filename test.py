import torch
from lib.BRAV import Reader

from lib.Electra import Electra, SimpleElectra
from lib.MLP import MLPSwiGLU

# x = torch.randn(32, 64, 128)

# cls = Reader(128)

# tokens = cls(x)
# print(tokens.size())



x = torch.randint(0, 512, (1, 32))

gen = MLPSwiGLU(512, 128, 1, 64, 1)

disc = MLPSwiGLU(512, 128, 1, 64, 1)

electra = SimpleElectra(disc, 128, 512)


out = electra(x)

#print(out)