import torch
import torch.nn as nn
from .linear import Linear
from .scaled_dot_product_attention import scaled_dot_product_attention
from .rope import RoPE
from einops import rearrange

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq, theta=None, device=None):
        super().__init__()
        self.num_heads = num_heads
        # 多头合并
        self.QKV = Linear(d_model, 3 * d_model, device=device)
        self.O = Linear(d_model, d_model, device=device)

        self.rope = None
        if theta:
            self.rope = RoPE(theta, d_model // num_heads, max_seq, device=device)

        # pdf 和实现就是转置来转置去，好讨厌
        self.register_buffer('mask', torch.tril(torch.ones((max_seq, max_seq))).to(bool), persistent=False)

    def forward(self, x, token_positions=None):
        T = x.shape[-2]

        # t 是序列，h 是头数，d 是头内运算的特征数
        # 这个牛逼啊，首先 rearrange 还是老样子，这个解包是 tensor 本来就能根据 dim0 做解包
        # 所以拆分矩阵乱七八糟的一套解决
        q, k, v = rearrange(
            self.QKV(x),
            '... t (qkv h d) -> qkv ... h t d',
            h=self.num_heads,
            qkv=3
        )
        
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        n_x = scaled_dot_product_attention(q, k, v, self.mask[:T, :T])
        n_x = rearrange(n_x, '... h t d -> ... t (h d)')
        return self.O(n_x)

        
        