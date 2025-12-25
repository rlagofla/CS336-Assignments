import torch.nn as nn
from .rmsnorm import RMSNorm
from .multihead_self_attention import MultiHeadSelfAttention
from .positionwise_feedforward import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_seq, theta, d_ff, device=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq, theta, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)

    def forward(self, x):
        pre_half = x + self.attn(self.ln1(x))
        out = pre_half + self.ffn(self.ln2(pre_half))
        return out
