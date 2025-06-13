import math
import torch
from torch.optim.optimizer import Optimizer

import numpy as np

class AdamHDN(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hypergrad_lr (float, optional): hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        `Online Learning Rate Adaptation with Hypergradient Descent`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Online Learning Rate Adaptation with Hypergradient Descent:
        https://openreview.net/forum?id=BkrsAzWAb
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, hypergrad_lr=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hypergrad_lr=hypergrad_lr)
        self.step_count = 0
        super(AdamHDN, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        self.step_count += 1
        inv_sqrt_step = 1.0 / math.sqrt(self.step_count)

        for group in self.param_groups:
            h = 0.0
            normalizer = 0.0
            weight_decay = group["weight_decay"]

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(p.data, alpha=group['weight_decay'])

                if state['step'] > 1:
                    prev_bias_correction1 = 1 - beta1 ** (state['step'] - 1)
                    prev_bias_correction2 = 1 - beta2 ** (state['step'] - 1)
                    # Hypergradient for Adam:
                    momentum_term = torch.div(exp_avg, exp_avg_sq.sqrt().add_(group['eps']))
                    if weight_decay != 0:
                        momentum_term.add_(p.data, alpha=weight_decay)
                    momentum_coef = math.sqrt(prev_bias_correction2) / prev_bias_correction1
                    h += torch.dot(grad.view(-1), momentum_term.view(-1)) * momentum_coef
                    normalizer += (momentum_coef**2) * (torch.norm(momentum_term)**2)
                    # Hypergradient descent of the learning rate:
            
            group['lr'] += group['hypergrad_lr'] * h *inv_sqrt_step #/ (normalizer + 1e-1)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
