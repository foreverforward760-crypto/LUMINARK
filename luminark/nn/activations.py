"""
Activation functions
"""
import numpy as np
from luminark.nn.module import Module
from luminark.core.tensor import Tensor


class ReLU(Module):
    """Rectified Linear Unit"""
    
    def forward(self, x):
        data = np.maximum(0, x.data)
        out = Tensor(data, requires_grad=x.requires_grad, _children=(x,), _op='relu')
        
        def _backward():
            if x.requires_grad:
                x.grad = x.grad + (x.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation"""
    
    def forward(self, x):
        data = 1 / (1 + np.exp(-x.data))
        out = Tensor(data, requires_grad=x.requires_grad, _children=(x,), _op='sigmoid')
        
        def _backward():
            if x.requires_grad:
                x.grad = x.grad + (data * (1 - data)) * out.grad
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Hyperbolic tangent activation"""
    
    def forward(self, x):
        data = np.tanh(x.data)
        out = Tensor(data, requires_grad=x.requires_grad, _children=(x,), _op='tanh')
        
        def _backward():
            if x.requires_grad:
                x.grad = x.grad + (1 - data ** 2) * out.grad
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "Tanh()"


class Softmax(Module):
    """Softmax activation"""
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        # Numerical stability: subtract max
        exp_x = np.exp(x.data - np.max(x.data, axis=self.dim, keepdims=True))
        data = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
        out = Tensor(data, requires_grad=x.requires_grad, _children=(x,), _op='softmax')
        
        def _backward():
            if x.requires_grad:
                # Simplified gradient for now
                x.grad = x.grad + out.grad
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"
