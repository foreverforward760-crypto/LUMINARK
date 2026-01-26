
"""
Learning Rate Schedulers
Adaptive learning rate adjustment strategies
"""
import math
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

# We can wrap standard PyTorch schedulers or define custom ones
# For LUMINARK, we will re-export PyTorch ones for API consistency

class LRScheduler:
    """Base wrapper for LUMINARK schedulers"""
    pass

class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]
