import torch
import torch.nn as nn
from .softmax import softmax
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    # 诡异，作业和 pdf 说的对不上
    # 而且维度这么变来变去的，其实 queries=keys=values，不然乘不起来
    pre_sftmax = einsum(Q, K, '... q d, ... k d -> ... q k')
    pre_sftmax = pre_sftmax / d_k ** 0.5
    
    if mask is not None:
        pre_sftmax = torch.masked_fill(pre_sftmax, ~mask, float('-inf'))
        sftmax = softmax(pre_sftmax, dim=-1)  # (... q)
        return sftmax @ V
    else:
        sftmax = softmax(pre_sftmax, dim=-1)  # (... q)
        return sftmax @ V
