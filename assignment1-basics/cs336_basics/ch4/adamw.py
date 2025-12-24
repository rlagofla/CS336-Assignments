import torch.optim as optim
from collections.abc import Callable
from typing import Optional


class AdamW(optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'betas': betas,
            'eps': eps,
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]
                m = beta1 * state.get('m', 0) + (1 - beta1) * grad
                v = beta2 * state.get('v', 0) + (1 - beta2) * grad**2
                t = state.get('t', 1)
                a = lr * (1-beta2**t)**0.5 / (1-beta1**t)
                p.data -= a * m / (v**0.5+eps) + lr * lmbda * p.data

                state['m'] = m
                state['v'] = v
                state['t'] = t + 1
            return loss
