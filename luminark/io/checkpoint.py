
"""
Model Checkpointing and Serialization
Save and load model weights, optimizer states, and training metadata
"""
import torch
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

class Checkpoint:
    """
    Complete checkpoint manager for models, optimizers, and training state
    """
    def __init__(self, model, optimizer=None, epoch=0, metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.metrics = metrics or {}

    def save(self, filepath: str):
        """Save state to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        state = {
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'metrics': self.metrics
        }
        if self.optimizer:
            state['optimizer_state'] = self.optimizer.state_dict()
            
        torch.save(state, filepath)
        print(f"📁 Checkpoint saved at: {filepath}")

    @classmethod
    def load(cls, filepath: str, model, optimizer=None):
        """Load state from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
            
        print(f"📂 Loading checkpoint: {filepath}")
        state = torch.load(filepath)
        
        model.load_state_dict(state['model_state'])
        if optimizer and 'optimizer_state' in state:
            optimizer.load_state_dict(state['optimizer_state'])
            
        return cls(model, optimizer, state['epoch'], state['metrics'])

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    ckpt = Checkpoint(model, optimizer, epoch, metrics)
    ckpt.save(filepath)

def load_checkpoint(filepath, model, optimizer=None):
    return Checkpoint.load(filepath, model, optimizer)
