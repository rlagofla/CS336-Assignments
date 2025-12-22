import torch
import torch.nn as nn
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embdM = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        # pdf 没明确说，我也不知道这么做对不对
        std = (2 / (num_embeddings + embedding_dim)) ** 0.5
        nn.init.trunc_normal_(self.embdM, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 这个索引没学过，但是由于我知道 embedding 是 (B, T) -> (B, T, C)，所以可以反过来理解
        return self.embdM[token_ids]