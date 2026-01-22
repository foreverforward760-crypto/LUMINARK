"""
Neural network layers
"""
import numpy as np
from luminark.nn.module import Module, Parameter
from luminark.core.tensor import Tensor


class Linear(Module):
    """Fully connected layer"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(np.random.uniform(-limit, limit, (in_features, out_features)))
        
        if bias:
            self.bias = Parameter(np.zeros(out_features))
        else:
            self.bias = None
    
    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            # Custom bias addition with proper gradient handling
            result = Tensor(out.data + self.bias.data,
                          requires_grad=out.requires_grad or self.bias.requires_grad,
                          _children=(out, self.bias), _op='+bias')

            def _backward():
                if out.requires_grad:
                    out.grad = out.grad + result.grad
                if self.bias.requires_grad:
                    # Sum over batch dimension for bias gradient
                    self.bias.grad = self.bias.grad + np.sum(result.grad, axis=0)

            result._backward = _backward
            return result
        return out
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class Sequential(Module):
    """Sequential container for layers"""
    
    def __init__(self, *layers):
        super().__init__()
        for idx, layer in enumerate(layers):
            self._modules[str(idx)] = layer
    
    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x


class Dropout(Module):
    """Dropout layer"""
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # Inverted dropout
        mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
        return x * Tensor(mask)
    
    def __repr__(self):
        return f"Dropout(p={self.p})"
