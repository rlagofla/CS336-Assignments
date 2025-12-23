import torch
import torch.nn as nn
from einops import einsum, rearrange


class RoPE(nn.Module):
    '''
    Gemini 说 Llama 啥的，选择两两配对旋转是前半后半中取
    但 pdf，包括测试还是按照原始的相邻两个配对
    '''
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        theta_k = theta ** (- torch.arange(0, d_k, 2).float() / d_k)
        i = torch.arange(0, max_seq_len)
        rotate_angle = einsum(i, theta_k, 'i,k -> i k')  # 要外积 (seq, d/2)
        # 这样搞一下是因为 dim0,1 用的都是 k=0，dim2,3 用的都是 k=1，所以需要交替穿插后面维数才匹配的上
        rotate_angle = torch.stack([rotate_angle, rotate_angle], dim=-1)
        rotate_angle = rearrange(rotate_angle, '... i j -> ... (i j)')

        self.register_buffer('cos', rotate_angle.cos())
        self.register_buffer('sin', rotate_angle.sin())


    # 没有用到 token_positions，测试用例这么简单吗
    # token_positions 的作用是，推理阶段或者 KV cache 的时候，x 不一定是完整的，有可能是中途开始的
    # 这个 token_positions 就是标记 x 的位置的
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        return cos * x + sin * self.rotate_half(x)

    def rotate_half(self, x):
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        # 交替穿插可以考虑用 stack + rearrange
        new_x = torch.stack([-x1, x0], dim=-1)
        return rearrange(new_x, '... i j -> ... (i j)')