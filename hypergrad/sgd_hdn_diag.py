import torch
from torch.optim import Optimizer
import math

class SGDHDN(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, hypergrad_lr=1e-6,
                 normalize=False):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        hypergrad_lr=hypergrad_lr, normalize=normalize)
        self.step_count = 0
        super(SGDHDN, self).__init__(params, defaults)

    # def step(self, closure=None):
    #     assert closure is not None
    #     loss = closure()

    #     # normalizer = 0.0
    #     self.step_count += 1

    #     for group in self.param_groups:
    #         lr = group["lr"]
    #         weight_decay = group["weight_decay"]
    #         if "normalizer" not in group:
    #             group["normalizer"] = 0.0
    #         group["normalizer"] = 0.0
            
    #         for p in group["params"]:
    #             if p.grad is None:
    #                 continue 
    #             grad = p.grad.data
    #             if grad.is_sparse:
    #                 raise RuntimeError(
    #                     "SGD_HDN does not support sparse gradients."
    #                 )
                
    #             state = self.state[p]
    #             if "grad" not in state:
    #                 state["grad"] = torch.zeros_like(p)

    #             if weight_decay != 0:
    #                 grad.add_(p.data, alpha=weight_decay)
                
    #             p.data.add_(grad, alpha=-lr)

    #             group["normalizer"] += grad.norm().pow(2)
    #             state["grad"].copy_(grad)
        
    #     loss_new = closure()
    #     inv_sqrt_step = 1.0 / math.sqrt(self.step_count)

    #     for group in self.param_groups:
    #         lr = group["lr"]
    #         hyper_lr = group["hypergrad_lr"]
    #         weight_decay = group["weight_decay"]
    #         normalizer = group["normalizer"]
    #         hyper_grad = 0.0

    #         for p in group["params"]:
    #             if p.grad is None:
    #                 continue

    #             grad_new = p.grad.data
                
    #             if weight_decay != 0:
    #                 grad_new.add_(p.data, alpha=weight_decay)

    #             grad = self.state[p]["grad"]
    #             hyper_grad += torch.sum(grad_new * grad)
        
    #         lr_update = hyper_lr * hyper_grad * inv_sqrt_step / (normalizer + 1e-1)
    #         group['lr'] += lr_update

    #     return loss

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
            lr = group["lr"]
            hyper_lr = group["hypergrad_lr"]
            weight_decay = group["weight_decay"]
            normalizer = 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "SGD_HDN does not support sparse gradients."
                    )
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)
                
                state = self.state[p]

                if len(state) == 0:
                    state['grad_prev'] = torch.zeros_like(grad)
                    state['lr'] = torch.ones_like(grad) * lr
                    state['hyper_grad'] = torch.zeros_like(grad)
                
                grad_prev = state['grad_prev']
                state['hyper_grad'] = grad * grad_prev
                normalizer += grad_prev.norm().pow(2)

                grad_prev.copy_(grad)
            
            normalizer += 1e-1

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                state['lr'].add_(state['hyper_grad'], alpha=hyper_lr * inv_sqrt_step  / normalizer)

                p.data.sub_(grad * state['lr'])
        
        return loss