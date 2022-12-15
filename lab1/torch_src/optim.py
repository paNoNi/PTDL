from typing import Optional, Callable

import torch
from torch.optim import Optimizer


class AdaSmooth(Optimizer):
    def __init__(self, params, lr=1e-3, p=(0.5, 0.99), eps: float = 1e-12, window: int = 10):
        default = dict(lr=lr, p=p, eps=eps, window=window)
        super().__init__(params, default)

    def __setstate__(self, state):
        super(AdaSmooth, self).__setstate__(state)

    def step(self, closure=...):
        for group in self.param_groups:
            for i, p in enumerate(group['params']):

                state = self.state[p]
                if p.grad is None:
                    continue

                grad = p.grad.data
                weights = p.data
                if len(state) == 0:
                    state['Egs'] = 0
                    state['weights_hist'] = None
                    state['delta_data'] = None

                if state['delta_data'] is None or state['delta_data'].shape[0] < group['window']:
                    delta_data = grad * group['lr']
                else:

                    abs_sum_x = torch.abs(weights - state['weights_hist'][0])
                    sum_abs_x = torch.sum(torch.abs(state['delta_data']), dim=0)

                    e = torch.divide(abs_sum_x, sum_abs_x + group['eps'])
                    c2 = torch.pow((group['p'][1] - group['p'][0]) * e + (1 - group['p'][1]), 2)
                    state['Egs'] = torch.multiply(c2, torch.pow(grad, 2)) + torch.multiply((1 - c2), state['Egs'])

                    # print('=' * 100)
                    # print(sum_abs_x.min())
                    # print(e.max())
                    # print(c2.max())
                    # print(((state['Egs'] + group['eps']) < 0).sum())
                    # print(torch.isnan(torch.sqrt(state['Egs'] + group['eps'])).sum())
                    lregs = group['lr'] / (torch.sqrt(state['Egs'] + group['eps']))
                    # print(torch.isnan(lregs).sum())
                    # print('=' * 100)

                    delta_data = torch.multiply(grad, lregs)

                p.data = p.data - delta_data

                if state['weights_hist'] is None:
                    state['weights_hist'] = torch.unsqueeze(weights, dim=0)
                    state['delta_data'] = torch.unsqueeze(delta_data, dim=0)
                else:
                    state['weights_hist'] = torch.concat([state['weights_hist'], torch.unsqueeze(weights, dim=0)], dim=0)
                    state['delta_data'] = torch.concat([state['delta_data'], torch.unsqueeze(delta_data, dim=0)], dim=0)

                state['weights_hist'] = state['weights_hist'][-group['window']:]
                state['delta_data'] = state['delta_data'][-group['window']:]
