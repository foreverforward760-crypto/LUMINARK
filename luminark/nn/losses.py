"""
Loss functions
"""
import numpy as np
from luminark.nn.module import Module
from luminark.core.tensor import Tensor


class MSELoss(Module):
    """Mean Squared Error Loss"""
    
    def forward(self, predictions, targets):
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
        
        diff = predictions - targets
        loss = (diff ** 2).mean()
        return loss
    
    def __repr__(self):
        return "MSELoss()"


class CrossEntropyLoss(Module):
    """Cross Entropy Loss (combines LogSoftmax and NLLLoss)"""
    
    def forward(self, predictions, targets):
        """
        predictions: (batch_size, num_classes) logits
        targets: (batch_size,) class indices
        """
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)
        
        batch_size = predictions.shape[0]
        
        # LogSoftmax for numerical stability
        logits = predictions.data
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        
        # Negative log likelihood
        loss_data = -log_probs[np.arange(batch_size), targets].mean()
        loss = Tensor(loss_data, requires_grad=predictions.requires_grad,
                     _children=(predictions,), _op='cross_entropy')
        
        def _backward():
            if predictions.requires_grad:
                # Gradient of cross entropy
                grad = np.exp(log_probs)
                grad[np.arange(batch_size), targets] -= 1
                grad /= batch_size
                predictions.grad = predictions.grad + grad * loss.grad
        
        loss._backward = _backward
        return loss
    
    def __repr__(self):
        return "CrossEntropyLoss()"


class BCELoss(Module):
    """Binary Cross Entropy Loss"""
    
    def forward(self, predictions, targets):
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
        
        # Numerical stability
        eps = 1e-7
        pred_data = np.clip(predictions.data, eps, 1 - eps)
        loss_data = -(targets.data * np.log(pred_data) + (1 - targets.data) * np.log(1 - pred_data)).mean()
        
        loss = Tensor(loss_data, requires_grad=predictions.requires_grad,
                     _children=(predictions,), _op='bce')
        
        def _backward():
            if predictions.requires_grad:
                grad = -(targets.data / pred_data - (1 - targets.data) / (1 - pred_data)) / predictions.shape[0]
                predictions.grad = predictions.grad + grad * loss.grad
        
        loss._backward = _backward
        return loss
    
    def __repr__(self):
        return "BCELoss()"
