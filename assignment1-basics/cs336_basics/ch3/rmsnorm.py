import torch
import torch.nn as nn
from einops import einsum


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.g = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
        std = (2 / d_model) ** 0.5
        nn.init.trunc_normal_(self.g, std=std, a=-3*std, b=3*std)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # 感觉这么写反而不通用了，标准的 LN 的 d_model 可以是 tuple
        # einsum lose
        # rms = (einsum(x, x, '... i, ... i -> ... 1') / self.d_model + self.eps) ** 0.5
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.g
        return result.to(in_dtype)