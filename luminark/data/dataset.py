"""
Dataset base classes
"""
from typing import Any


class Dataset:
    """Base class for datasets"""
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError
