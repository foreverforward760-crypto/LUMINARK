
"""
LUMINARK Data Loading Utilities
"""
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

class DataLoader(TorchDataLoader):
    """LUMINARK Data Loader (Wrapper)"""
    pass

class MNISTDigits(Dataset):
    """
    Mock MNIST Dataset for Demonstration.
    Generates synthetic digit-like data for immediate testing without downloads.
    """
    def __init__(self, train=True, normalize=True):
        self.train = train
        self.size = 1000 if train else 200
        # Synthetic data: 1x28x28 images (flattened to 784 in some models, but we keep shape)
        self.data = torch.randn(self.size, 1, 28, 28)
        self.targets = torch.randint(0, 10, (self.size,))
        print(f"📊 Initialized Mock MNIST ({'Train' if train else 'Test'}): {self.size} samples")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
