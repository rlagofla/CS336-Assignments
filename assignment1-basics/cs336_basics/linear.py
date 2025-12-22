import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = (2 / (in_features+out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, 'o i,... i -> ... o')
    
if __name__ == '__main__':
    linear = Linear(2, 3)
    print(linear(torch.rand(2)))