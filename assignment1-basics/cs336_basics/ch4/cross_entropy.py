import torch
from einops import reduce, rearrange

def cross_entropy(logits, targets):
    '''
    logits 理论上应该是 (B, T, C) 的，我看之前的实现会 (B*T, C) 使得每个词都有一个 logits
    但是，adapters 直接给的就是 (B, C) 感觉。。。看不懂
    '''
    max_val = reduce(logits, '... o -> ... 1', 'max')
    fenmu = torch.log(reduce(torch.exp(logits - max_val), '... eo -> ... 1', 'sum'))
    # 没学过这个 gather，大概是，dim 为多少，index 的 dim 就得为 1
    # 表示这个维度按 index 取一个，其他维度要匹配
    targets = rearrange(targets, '... -> ... 1')
    fenzi = torch.gather(logits, dim=-1, index=targets) - max_val
    loss = fenmu - fenzi
    return reduce(loss, '... -> 1', 'mean')