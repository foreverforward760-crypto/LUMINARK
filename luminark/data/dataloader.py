"""
DataLoader for batch processing
"""
import numpy as np
from typing import Optional


class DataLoader:
    """DataLoader for iterating over datasets in batches"""
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            batch_indices = indices[start_idx:end_idx]
            
            # Collect batch
            batch = [self.dataset[int(i)] for i in batch_indices]
            
            # Collate batch
            if batch:
                # Assume batch items are tuples (x, y)
                xs = np.stack([item[0] for item in batch])
                ys = np.array([item[1] for item in batch])
                yield xs, ys
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
