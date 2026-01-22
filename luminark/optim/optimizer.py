"""
Optimizer base class and implementations
"""
import numpy as np
from typing import List
from luminark.nn.module import Parameter


class Optimizer:
    """Base class for optimizers"""
    
    def __init__(self, parameters):
        self.parameters = list(parameters)
    
    def step(self):
        """Perform one optimization step"""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero out gradients"""
        for param in self.parameters:
            param.zero_grad()


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data
            
            # Momentum
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                grad = self.velocity[i]
            
            # Update parameters
            param.data -= self.lr * grad
    
    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum}, weight_decay={self.weight_decay})"


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [np.zeros_like(p.data) for p in self.parameters]  # First moment
        self.v = [np.zeros_like(p.data) for p in self.parameters]  # Second moment
        self.t = 0  # Timestep
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def __repr__(self):
        return f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), eps={self.eps})"
