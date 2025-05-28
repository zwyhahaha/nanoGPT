import torch
from torch.optim import Optimizer
import math

class SGDMHDN(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.995, dampening=0,
                 weight_decay=0, nesterov=False, hypergrad_lr=1e-6,
                 online_optimizer='decay'):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        hypergrad_lr=hypergrad_lr, online_optimizer=online_optimizer,
                        )
        self.step_count = 0
        super(SGDMHDN, self).__init__(params, defaults)

    def step(self, closure=None):
        assert closure is not None
        loss = closure()
        beta1 = 0.9
        beta2 = 0.99

        self.step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            if "hypergrad_square" not in group:
                group["hypergrad_square"] = 0.0
                group["hypergrad_expavg"] = 0.0
                group["hypergrad_expavg_sq"] = 0.0
            normalizer = 0.0
            
            for p in group["params"]:
                if p.grad is None:
                    continue 
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "SGD_HDN does not support sparse gradients."
                    )
                
                state = self.state[p]
                if "grad" not in state:
                    state["grad"] = torch.zeros_like(p)
                    state["prev_params"] = torch.zeros_like(p)
                    state["momentum_buffer"] = torch.zeros_like(p)

                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)
                
                state['momentum_buffer'].mul_(momentum).add_(grad, alpha=-lr)
                momentum_buffer = state['momentum_buffer']
                
                state["prev_params"].copy_(p.data)

                # print(grad.norm())
                
                p.data.add_(momentum_buffer)

                normalizer += grad.norm().pow(2) + momentum_buffer.norm().pow(2)
                state["grad"].copy_(grad)
                
            # print(math.sqrt(normalizer)/len(group["params"]))
        
        loss_new = closure()

        if loss_new > loss * 1.0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue 
                    state = self.state[p]
                    p.data.copy_(state["prev_params"])

        for group in self.param_groups:
            hyper_lr = group["hypergrad_lr"]
            weight_decay = group["weight_decay"]
            online_optimizer = group["online_optimizer"]
            hyper_grad = 0.0
            momentum_hyper_grad = 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad_new = p.grad.data
                
                if weight_decay != 0:
                    grad_new.add_(p.data, alpha=weight_decay)

                grad = self.state[p]["grad"]
                momentum_buffer = self.state[p]['momentum_buffer']
                hyper_grad += torch.sum(grad_new * grad)
                momentum_hyper_grad += torch.sum(grad_new * momentum_buffer)

            hyper_grad /= (normalizer + 1e-1)
            momentum_hyper_grad /= (normalizer + 1e-1)

            # print(hyper_grad)

            # hyper_grad = min(max(hyper_grad, -1.0), 1.0)
            if online_optimizer == 'adagrad':
                group['hypergrad_square'] += hyper_grad ** 2
                lr_update = hyper_lr * hyper_grad / (math.sqrt(group["hypergrad_square"])+1e-8)
                group['lr'] += lr_update
            elif online_optimizer == 'adam': # and self.step_count<=1000 for mlp task
                group["hypergrad_expavg"] = beta1*group["hypergrad_expavg"] + (1-beta1)*hyper_grad
                group["hypergrad_expavg_sq"] = beta2*group["hypergrad_expavg_sq"] + (1-beta2)*hyper_grad**2
                bias_correction1 = 1 - beta1 ** self.step_count
                bias_correction2 = 1 - beta2 ** self.step_count
                lr_update = hyper_lr * group["hypergrad_expavg"] / (math.sqrt(group["hypergrad_expavg_sq"])+1e-8)
                group['lr'] += lr_update * math.sqrt(bias_correction2) / bias_correction1
            else:
                lr_update = hyper_lr * hyper_grad #/ math.sqrt(self.step_count)
                group['lr'] += lr_update
            momentum_update = 0.1 * hyper_lr * momentum_hyper_grad #/ math.sqrt(self.step_count)
            group["momentum"] += momentum_update
            # group['momentum'] = max(0.0, min(0.995, group['momentum']))
            # print(group['momentum'])
        return loss

    # def step(self, closure=None):
    #     """Performs a single optimization step.

    #     Arguments:
    #         closure (callable, optional): A closure that reevaluates the model
    #             and returns the loss.
    #     """
    #     loss = None
    #     if closure is not None:
    #         loss = closure()
    #     beta1 = 0.9
    #     beta2 = 0.999

    #     self.step_count += 1

    #     for group in self.param_groups:
    #         lr = group["lr"]
    #         hyper_lr = group["hypergrad_lr"]
    #         weight_decay = group["weight_decay"]
    #         online_optimizer = group["online_optimizer"]
    #         normalizer = 0.0
    #         hyper_grad = 0.0

    #         if "hypergrad_square" not in group:
    #             group["hypergrad_square"] = 0.0
    #             group["hypergrad_expavg"] = 0.0
    #             group["hypergrad_expavg_sq"] = 0.0

    #         for p in group["params"]:
    #             if p.grad is None:
    #                 continue
    #             grad = p.grad.data
    #             if grad.is_sparse:
    #                 raise RuntimeError(
    #                     "SGD_HDN does not support sparse gradients."
    #                 )
    #             if weight_decay != 0:
    #                 grad.add_(p.data, alpha=weight_decay)
                
    #             state = self.state[p]

    #             if len(state) == 0:
    #                 state['grad_prev'] = torch.zeros_like(grad)
                
    #             grad_prev = state['grad_prev']
    #             hyper_grad += torch.sum(grad * grad_prev)

    #             normalizer += grad_prev.norm().pow(2)

    #             grad_prev.copy_(grad)

    #         hyper_grad /= (normalizer + 1e-1)
    #         if online_optimizer == 'adagrad':
    #             group['hypergrad_square'] += hyper_grad ** 2
    #             lr_update = hyper_lr * hyper_grad / (math.sqrt(group["hypergrad_square"])+1e-8)
    #             group['lr'] += lr_update
    #         elif online_optimizer == 'adam': # and self.step_count<=1000 for mlp task
    #             group["hypergrad_expavg"] = beta1*group["hypergrad_expavg"] + (1-beta1)*hyper_grad
    #             group["hypergrad_expavg_sq"] = beta2*group["hypergrad_expavg_sq"] + (1-beta2)*hyper_grad**2
    #             bias_correction1 = 1 - beta1 ** self.step_count
    #             bias_correction2 = 1 - beta2 ** self.step_count
    #             lr_update = hyper_lr * group["hypergrad_expavg"] / (math.sqrt(group["hypergrad_expavg_sq"])+1e-8)
    #             group['lr'] += lr_update * math.sqrt(bias_correction2) / bias_correction1
    #         else:
    #             lr_update = hyper_lr * hyper_grad / math.sqrt(self.step_count)
    #             group['lr'] += lr_update
    #         lr = group["lr"]

    #         for p in group["params"]:
    #             if p.grad is None:
    #                 continue
    #             grad = p.grad.data

    #             p.data.add_(grad, alpha=-lr)
        
    #     return loss