"""
Base Module and Parameter classes for neural networks
"""
from luminark.core.tensor import Tensor
from typing import Iterator, Tuple


class Parameter(Tensor):
    """
    A trainable parameter (a Tensor that requires gradients)
    """
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    """
    Base class for all neural network modules
    """
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self) -> Iterator[Parameter]:
        """Return an iterator over module parameters"""
        for name, param in self._parameters.items():
            yield param
        for name, module in self._modules.items():
            yield from module.parameters()
    
    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        """Return an iterator over module parameters with names"""
        for name, param in self._parameters.items():
            yield name, param
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param
    
    def zero_grad(self):
        """Zero out gradients of all parameters"""
        for param in self.parameters():
            param.zero_grad()
    
    def train(self, mode=True):
        """Set module to training mode"""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        """Set module to evaluation mode"""
        return self.train(False)
    
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)
    
    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            if name in self._parameters:
                return self._parameters[name]
        if '_modules' in self.__dict__:
            if name in self._modules:
                return self._modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __repr__(self):
        lines = [self.__class__.__name__ + '(']
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = '\n  '.join(mod_str.split('\n'))
            lines.append(f'  ({key}): {mod_str}')
        lines.append(')')
        return '\n'.join(lines)
