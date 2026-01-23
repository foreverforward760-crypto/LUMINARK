"""
Model Checkpointing and Serialization
Save and load model weights, optimizer states, and training metadata
"""
import pickle
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np


class Checkpoint:
    """
    Complete checkpoint manager for models, optimizers, and training state

    Example:
        >>> checkpoint = Checkpoint(model, optimizer, epoch=10, metrics={'acc': 0.95})
        >>> checkpoint.save('checkpoints/model_epoch10.pt')
        >>>
        >>> # Later...
        >>> checkpoint = Checkpoint.load('checkpoints/model_epoch10.pt', model, optimizer)
        >>> print(f"Resumed from epoch {checkpoint.epoch}")
    """

    def __init__(self, model, optimizer=None, epoch=0, metrics=None, metadata=None):
        """
        Args:
            model: Model instance with parameters
            optimizer: Optional optimizer instance
            epoch: Current epoch number
            metrics: Dict of training metrics
            metadata: Additional metadata to save
        """
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.metrics = metrics or {}
        self.metadata = metadata or {}

    def save(self, path: str):
        """
        Save checkpoint to disk

        Args:
            path: File path to save checkpoint
        """
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Collect model parameters
        model_state = {}
        for name, param in self.model.named_parameters():
            model_state[name] = {
                'data': param.data,
                'requires_grad': param.requires_grad
            }

        # Collect optimizer state if present
        optimizer_state = None
        if self.optimizer is not None:
            optimizer_state = {
                'class_name': self.optimizer.__class__.__name__,
                'lr': self.optimizer.lr,
                'state': {}
            }

            # Save Adam-specific state
            if hasattr(self.optimizer, 'betas'):
                optimizer_state['betas'] = self.optimizer.betas
                optimizer_state['eps'] = self.optimizer.eps
                optimizer_state['m'] = {name: m for name, m in self.optimizer.m.items()}
                optimizer_state['v'] = {name: v for name, v in self.optimizer.v.items()}
                optimizer_state['t'] = self.optimizer.t

            # Save SGD-specific state
            if hasattr(self.optimizer, 'momentum'):
                optimizer_state['momentum'] = self.optimizer.momentum
                optimizer_state['velocities'] = {name: v for name, v in self.optimizer.velocities.items()}

        # Create checkpoint dictionary
        checkpoint_data = {
            'model_state': model_state,
            'model_class': self.model.__class__.__name__,
            'optimizer_state': optimizer_state,
            'epoch': self.epoch,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'version': '0.1.0'
        }

        # Save to file
        with open(path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        print(f"✓ Checkpoint saved to {path}")
        print(f"  Epoch: {self.epoch}")
        if self.metrics:
            print(f"  Metrics: {self.metrics}")

    @classmethod
    def load(cls, path: str, model, optimizer=None):
        """
        Load checkpoint from disk

        Args:
            path: File path to load checkpoint from
            model: Model instance to load weights into
            optimizer: Optional optimizer instance to restore state

        Returns:
            Checkpoint instance with loaded state
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load checkpoint data
        with open(path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        # Verify model class matches
        if checkpoint_data['model_class'] != model.__class__.__name__:
            print(f"⚠️  Warning: Model class mismatch. "
                  f"Checkpoint: {checkpoint_data['model_class']}, "
                  f"Current: {model.__class__.__name__}")

        # Restore model parameters
        model_state = checkpoint_data['model_state']
        for name, param in model.named_parameters():
            if name in model_state:
                param.data = model_state[name]['data']
                param.requires_grad = model_state[name]['requires_grad']
            else:
                print(f"⚠️  Warning: Parameter '{name}' not found in checkpoint")

        # Restore optimizer state if present
        optimizer_state = checkpoint_data.get('optimizer_state')
        if optimizer is not None and optimizer_state is not None:
            optimizer.lr = optimizer_state['lr']

            # Restore Adam state
            if hasattr(optimizer, 'betas') and 'betas' in optimizer_state:
                optimizer.betas = optimizer_state['betas']
                optimizer.eps = optimizer_state['eps']
                optimizer.m = optimizer_state['m']
                optimizer.v = optimizer_state['v']
                optimizer.t = optimizer_state['t']

            # Restore SGD state
            if hasattr(optimizer, 'momentum') and 'momentum' in optimizer_state:
                optimizer.momentum = optimizer_state['momentum']
                optimizer.velocities = optimizer_state['velocities']

        # Create checkpoint instance
        checkpoint = cls(
            model=model,
            optimizer=optimizer,
            epoch=checkpoint_data['epoch'],
            metrics=checkpoint_data['metrics'],
            metadata=checkpoint_data['metadata']
        )

        print(f"✓ Checkpoint loaded from {path}")
        print(f"  Epoch: {checkpoint.epoch}")
        if checkpoint.metrics:
            print(f"  Metrics: {checkpoint.metrics}")

        return checkpoint


def save_model(model, path: str, metadata: Optional[Dict] = None):
    """
    Quick save - just save model weights

    Args:
        model: Model instance
        path: File path to save to
        metadata: Optional metadata dict
    """
    checkpoint = Checkpoint(model, metadata=metadata)
    checkpoint.save(path)


def load_model(path: str, model):
    """
    Quick load - just load model weights

    Args:
        path: File path to load from
        model: Model instance to load into

    Returns:
        Model with loaded weights
    """
    checkpoint = Checkpoint.load(path, model)
    return checkpoint.model


def save_checkpoint(model, optimizer, epoch: int, metrics: Dict, path: str):
    """
    Save complete training checkpoint

    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch
        metrics: Training metrics dict
        path: File path to save to
    """
    checkpoint = Checkpoint(model, optimizer, epoch, metrics)
    checkpoint.save(path)


def load_checkpoint(path: str, model, optimizer):
    """
    Load complete training checkpoint

    Args:
        path: File path to load from
        model: Model instance
        optimizer: Optimizer instance

    Returns:
        Tuple of (model, optimizer, epoch, metrics)
    """
    checkpoint = Checkpoint.load(path, model, optimizer)
    return checkpoint.model, checkpoint.optimizer, checkpoint.epoch, checkpoint.metrics
