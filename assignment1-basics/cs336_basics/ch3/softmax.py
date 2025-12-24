import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x = x - max_val

    exp_x = torch.exp(x)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp