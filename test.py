# import torch
# import numpy as np

# x = torch.randn(100)
# y = torch.randn(100)

# print(x,y)

# def _get_ranks(x: torch.Tensor) -> torch.Tensor:
#     tmp = x.argsort()
#     ranks = torch.zeros_like(tmp)
#     ranks[tmp] = torch.arange(len(x))
#     return ranks

# def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
#     """Compute correlation between 2 1-D vectors
#     Args:
#         x: Shape (N, )
#         y: Shape (N, )
#     """
#     x_rank = _get_ranks(x)
#     y_rank = _get_ranks(y)
    
#     n = x.size(0)
#     upper = 6 * torch.sum((x_rank - y_rank).pow(2))
#     down = n * (n ** 2 - 1.0)
#     return 1.0 - (upper / down)


# print(spearman_correlation(x,y))


#graphing the learning rate schedule

import matplotlib.pyplot as plt
import numpy as np
import torch

STEPS = 3125
optim = torch.optim.Adam([torch.zeros(1)], lr=1e-3)
pct_start = 0.05
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=8e-4, total_steps=STEPS,pct_start=pct_start, three_phase=False, final_div_factor=1)

s1 = torch.optim.lr_scheduler.LinearLR(optim, 1e-2, 1, STEPS*pct_start)
# s2 = torch.optim.lr_scheduler.LinearLR(optim, 1, 1e-2, STEPS*(1-pct_start))
s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim, STEPS*(1-pct_start), 1e-5)
scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [s1, s2], [int(STEPS*pct_start)])

lrs = []
for i in range(STEPS):
    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0])

plt.plot(np.arange(STEPS), lrs)
plt.xlabel('Step')
plt.ylabel('Learning rate')
print(max(lrs))
print(min(lrs))

plt.show()

