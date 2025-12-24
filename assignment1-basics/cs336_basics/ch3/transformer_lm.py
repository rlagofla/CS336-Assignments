import torch.nn as nn
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size, d_model, context_length, num_layers, 
        num_heads, theta, d_ff
    ):
        super().__init__()
        self.embd = Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(*[TransformerBlock(d_model, num_heads, context_length, theta, d_ff) for i in range(num_layers)])
        self.ln = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x):
        x_embd = self.embd(x)
        y = self.layers(x_embd)
        y = self.ln(y)
        y = self.lm_head(y)
        # 哦，原来这里不用套 softmax，是给 attention 用的，后面再 loss
        return y