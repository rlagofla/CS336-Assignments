import torch
import torch.nn as nn
from .linear import Linear


class SwiGLU(nn.Module):
    # 偷看了一下 adpaters，里头提供了 d_ff，所以我就不算 8/3 了
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        # 左升维矩阵
        self.W1 = Linear(d_model, d_ff, device, dtype)
        # 右升维矩阵
        self.W3 = Linear(d_model, d_ff, device, dtype)
        # 降维矩阵
        self.W2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x):
        return self.W2(self.silu(self.W1(x)) * self.W3(x))
    
    def silu(self, x):
        return x * torch.sigmoid(x)
