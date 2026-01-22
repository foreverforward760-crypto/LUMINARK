"""
Core Tensor implementation with automatic differentiation
"""
import numpy as np
from typing import Union, Tuple, Optional, List


class Tensor:
    """
    Tensor with automatic differentiation support
    """
    
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        else:
            data = np.array(data, dtype=np.float32)
            
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None if not requires_grad else np.zeros_like(data, dtype=np.float32)
        
        # Autograd graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (other.data * out.grad)
            if other.requires_grad:
                other.grad = other.grad + (self.data * out.grad)
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='@')
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (out.grad @ other.data.T)
            if other.requires_grad:
                other.grad = other.grad + (self.data.T @ out.grad)
        
        out._backward = _backward
        return out
    
    def __pow__(self, power):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad,
                    _children=(self,), _op=f'**{power}')
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (power * self.data ** (power - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return Tensor(other) - self
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad,
                    _children=(self,), _op='sum')
        
        def _backward():
            if self.requires_grad:
                if axis is None:
                    self.grad = self.grad + np.ones_like(self.data) * out.grad
                else:
                    grad_shape = list(self.data.shape)
                    if not keepdims:
                        grad_shape[axis] = 1
                    self.grad = self.grad + np.broadcast_to(out.grad.reshape(grad_shape), self.data.shape)
        
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad,
                    _children=(self,), _op='mean')
        
        def _backward():
            if self.requires_grad:
                n = self.data.size if axis is None else self.data.shape[axis]
                if axis is None:
                    self.grad = self.grad + np.ones_like(self.data) * out.grad / n
                else:
                    grad_shape = list(self.data.shape)
                    if not keepdims:
                        grad_shape[axis] = 1
                    self.grad = self.grad + np.broadcast_to(out.grad.reshape(grad_shape), self.data.shape) / n
        
        out._backward = _backward
        return out
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad,
                    _children=(self,), _op='reshape')
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad.reshape(self.data.shape)
        
        out._backward = _backward
        return out
    
    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes) if axes else self.data.T,
                    requires_grad=self.requires_grad, _children=(self,), _op='transpose')
        
        def _backward():
            if self.requires_grad:
                if axes:
                    # Inverse permutation
                    inv_axes = [0] * len(axes)
                    for i, ax in enumerate(axes):
                        inv_axes[ax] = i
                    self.grad = self.grad + out.grad.transpose(*inv_axes)
                else:
                    self.grad = self.grad + out.grad.T
        
        out._backward = _backward
        return out
    
    @property
    def T(self):
        return self.transpose()
    
    def backward(self):
        """
        Compute gradients via backpropagation
        """
        # Topological sort
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient
        self.grad = np.ones_like(self.data, dtype=np.float32)
        
        # Backpropagate
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """Reset gradients to zero"""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)
    
    def detach(self):
        """Detach from computational graph"""
        return Tensor(self.data.copy(), requires_grad=False)
    
    def numpy(self):
        """Get numpy array"""
        return self.data.copy()
    
    def item(self):
        """Get Python scalar"""
        return self.data.item()


def tensor(data, requires_grad=False):
    """Create a tensor"""
    return Tensor(data, requires_grad=requires_grad)
