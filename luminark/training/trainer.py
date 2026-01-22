"""
Training system with metrics integration
"""
import time
import numpy as np
from typing import Optional
from luminark.core.tensor import Tensor


class Trainer:
    """
    Trainer for neural network models with real-time metrics
    """
    
    def __init__(self, model, optimizer, criterion, train_loader,
                 val_loader=None, metrics_callback=None, defense_system=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics_callback = metrics_callback
        self.defense_system = defense_system
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        self.iteration = 0
        self.start_time = time.time()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = 0
        
        epoch_start = time.time()
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            batch_start = time.time()
            self.iteration += 1
            
            # Forward pass
            data_tensor = Tensor(data, requires_grad=True)
            predictions = self.model(data_tensor)
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            pred_classes = np.argmax(predictions.data, axis=1)
            correct = np.sum(pred_classes == targets)
            accuracy = correct / len(targets)
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += len(targets)
            batch_count += 1
            
            # Calculate metrics
            batch_time = time.time() - batch_start
            throughput = len(targets) / batch_time if batch_time > 0 else 0
            
            # Emit metrics to callback (for dashboard)
            if self.metrics_callback and batch_idx % 5 == 0:
                metrics = {
                    'timestamp': time.time(),
                    'iteration': self.iteration,
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'accuracy': accuracy * 100,  # Convert to percentage
                    'throughput': throughput,
                    'learning_rate': self.optimizer.lr,
                }
                self.metrics_callback(metrics)
            
            # Defense system monitoring
            if self.defense_system and batch_idx % 10 == 0:
                # Calculate stability metrics
                grad_norms = [np.linalg.norm(p.grad) for p in self.model.parameters() if p.grad is not None]
                avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
                
                stability = max(0, min(1, 1.0 - avg_grad_norm / 10.0))  # Normalize
                tension = min(1, loss.item())  # Use loss as tension
                coherence = min(1, accuracy)  # Use accuracy as coherence
                
                defense_response = self.defense_system.analyze_threat(stability, tension, coherence)
                
                if defense_response['defense_mode'] != 'NOMINAL':
                    print(f"\n⚠️  Defense Alert: {defense_response['defense_mode']}")
                    print(f"   Strategy: {defense_response['strategy']}")
        
        # Epoch statistics
        avg_loss = total_loss / batch_count
        avg_acc = total_correct / total_samples
        epoch_time = time.time() - epoch_start
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(avg_acc)
        
        return avg_loss, avg_acc, epoch_time
    
    def validate(self):
        """Validate the model"""
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = 0
        
        for data, targets in self.val_loader:
            # Forward pass only
            data_tensor = Tensor(data)
            predictions = self.model(data_tensor)
            loss = self.criterion(predictions, targets)
            
            # Calculate accuracy
            pred_classes = np.argmax(predictions.data, axis=1)
            correct = np.sum(pred_classes == targets)
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += len(targets)
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        avg_acc = total_correct / total_samples
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(avg_acc)
        
        return avg_loss, avg_acc
    
    def fit(self, epochs):
        """Train the model for multiple epochs"""
        print("=" * 80)
        print(f"{'LUMINARK Training Started':^80}")
        print("=" * 80)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Criterion: {self.criterion}")
        print(f"Epochs: {epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print("=" * 80)
        print()
        
        best_val_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_acc, epoch_time = self.train_epoch(epoch)
            
            # Validate
            if self.val_loader:
                val_loss, val_acc = self.validate()
                
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
                print(f"Epoch Time: {epoch_time:.2f}s")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"✓ New best validation accuracy: {val_acc*100:.2f}%")
            else:
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
                print(f"Epoch Time: {epoch_time:.2f}s")
        
        print()
        print("=" * 80)
        print(f"{'Training Complete':^80}")
        print("=" * 80)
        total_time = time.time() - self.start_time
        print(f"Total training time: {total_time:.2f}s")
        if self.val_loader:
            print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
        print("=" * 80)
        
        return self.history
