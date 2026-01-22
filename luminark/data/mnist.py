"""
MNIST-like dataset (uses sklearn digits for simplicity)
"""
import numpy as np
from luminark.data.dataset import Dataset


class MNISTDigits(Dataset):
    """
    Simple MNIST-like dataset using sklearn's digits dataset
    8x8 grayscale images of digits 0-9
    """
    
    def __init__(self, train=True, normalize=True):
        try:
            from sklearn.datasets import load_digits
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError("Please install scikit-learn: pip install scikit-learn")
        
        # Load dataset
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if train:
            self.data = X_train
            self.targets = y_train
        else:
            self.data = X_test
            self.targets = y_test
        
        # Normalize
        if normalize:
            self.data = self.data / 16.0  # Digits are 0-16
        
        self.data = self.data.astype(np.float32)
        self.targets = self.targets.astype(np.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
