"""
Learning Rate Schedulers
Adaptive learning rate adjustment strategies
"""
import numpy as np
from typing import Optional


class LRScheduler:
    """Base class for learning rate schedulers"""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.initial_lr = optimizer.lr
        self.current_step = 0

    def step(self):
        """Update learning rate"""
        raise NotImplementedError

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.lr


class StepLR(LRScheduler):
    """
    Decays learning rate by gamma every step_size epochs

    Example:
        >>> scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        """
        Args:
            optimizer: Optimizer instance
            step_size: Period of learning rate decay
            gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        """Decay learning rate if at step boundary"""
        self.current_step += 1
        if self.current_step % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate scheduler
    Smoothly decreases LR following a cosine curve

    Example:
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        >>> for epoch in range(50):
        >>>     train(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, T_max: int, eta_min: float = 0):
        """
        Args:
            optimizer: Optimizer instance
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
        """
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self):
        """Update learning rate using cosine annealing"""
        self.current_step += 1
        lr = self.eta_min + (self.initial_lr - self.eta_min) * \
             (1 + np.cos(np.pi * self.current_step / self.T_max)) / 2
        self.optimizer.lr = float(lr)


class ExponentialLR(LRScheduler):
    """
    Exponential learning rate decay

    Example:
        >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, gamma: float):
        """
        Args:
            optimizer: Optimizer instance
            gamma: Multiplicative factor of learning rate decay (< 1.0)
        """
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        """Decay learning rate exponentially"""
        self.optimizer.lr *= self.gamma


class ReduceLROnPlateau(LRScheduler):
    """
    Reduce learning rate when a metric has stopped improving

    Example:
        >>> scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
        >>> for epoch in range(100):
        >>>     val_loss = validate(...)
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode: str = 'min', factor: float = 0.1,
                 patience: int = 10, threshold: float = 1e-4, min_lr: float = 0):
        """
        Args:
            optimizer: Optimizer instance
            mode: 'min' for loss, 'max' for accuracy
            factor: Factor by which to reduce LR
            patience: Number of epochs with no improvement after which LR is reduced
            threshold: Threshold for measuring improvement
            min_lr: Minimum learning rate
        """
        super().__init__(optimizer)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr

        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        self.last_epoch = 0

    def step(self, metric: float):
        """
        Update learning rate based on metric

        Args:
            metric: Current metric value (loss or accuracy)
        """
        current = metric

        # Check if metric improved
        if self.mode == 'min':
            improved = current < self.best - self.threshold
        else:  # mode == 'max'
            improved = current > self.best + self.threshold

        if improved:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Reduce LR if patience exceeded
        if self.num_bad_epochs >= self.patience:
            new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
            if new_lr < self.optimizer.lr:
                self.optimizer.lr = new_lr
                print(f"Reducing learning rate to {new_lr:.2e}")
            self.num_bad_epochs = 0


class OneCycleLR(LRScheduler):
    """
    One-cycle learning rate policy
    Increases LR then decreases it (useful for super-convergence)

    Example:
        >>> scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=1000)
        >>> for batch in data_loader:
        >>>     train_batch(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, div_factor: float = 25.0,
                 final_div_factor: float = 1e4):
        """
        Args:
            optimizer: Optimizer instance
            max_lr: Maximum learning rate
            total_steps: Total number of training steps
            pct_start: Percentage of cycle spent increasing LR
            div_factor: Initial LR = max_lr / div_factor
            final_div_factor: Final LR = max_lr / final_div_factor
        """
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up

        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor

        self.optimizer.lr = self.initial_lr

    def step(self):
        """Update learning rate for one-cycle policy"""
        self.current_step += 1

        if self.current_step <= self.step_size_up:
            # Increasing phase
            progress = self.current_step / self.step_size_up
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Decreasing phase
            progress = (self.current_step - self.step_size_up) / self.step_size_down
            lr = self.max_lr - (self.max_lr - self.final_lr) * progress

        self.optimizer.lr = float(lr)


class WarmupLR(LRScheduler):
    """
    Linear warmup followed by constant LR

    Example:
        >>> scheduler = WarmupLR(optimizer, warmup_steps=1000)
        >>> for step in range(total_steps):
        >>>     train_step(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, warmup_steps: int, start_lr: float = 0):
        """
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            start_lr: Initial learning rate (default 0)
        """
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.optimizer.lr = start_lr

    def step(self):
        """Linearly increase LR during warmup"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            lr = self.start_lr + (self.initial_lr - self.start_lr) * progress
            self.optimizer.lr = float(lr)
