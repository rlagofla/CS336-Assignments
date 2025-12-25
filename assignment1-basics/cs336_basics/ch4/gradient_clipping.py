from typing import Iterable
import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # 是每个参数都查看一次 norm，还是全部参数查看一次 norm
    total_norm = 0
    for p in parameters:
        if p.grad is None: continue
        total_norm += (p.grad**2).sum()
    total_norm = total_norm ** 0.5

    eps = 1e-6
    if total_norm >= max_l2_norm:
        factor =  max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is None: continue
            p.grad *= factor