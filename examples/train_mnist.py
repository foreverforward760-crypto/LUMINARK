#!/usr/bin/env python3
"""
LUMINARK Example: Train a Neural Network on MNIST Digits
This example demonstrates the full AI framework capabilities
"""
import sys
sys.path.insert(0, '/home/user/LUMINARK')

from luminark.nn import Module, Linear, ReLU, Sequential
from luminark.nn import CrossEntropyLoss
from luminark.optim import Adam
from luminark.data import MNISTDigits, DataLoader
from luminark.training import Trainer
from mycelial_defense import MycelialDefenseSystem


class SimpleNN(Module):
    """Simple feedforward neural network"""
    
    def __init__(self, input_size=64, hidden_size=128, num_classes=10):
        super().__init__()
        self.network = Sequential(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def main():
    print("\n🌟 LUMINARK AI Framework - MNIST Training Example\n")
    
    # Load datasets
    print("Loading MNIST digits dataset...")
    train_dataset = MNISTDigits(train=True, normalize=True)
    val_dataset = MNISTDigits(train=False, normalize=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("\nCreating neural network model...")
    model = SimpleNN(input_size=64, hidden_size=128, num_classes=10)
    print(model)
    
    # Count parameters
    num_params = sum(p.data.size for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    # Setup training
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Initialize defense system
    defense = MycelialDefenseSystem()
    
    # Metrics callback for real-time monitoring
    metrics_history = []
    
    def metrics_callback(metrics):
        metrics_history.append(metrics)
        if metrics['batch'] % 10 == 0:
            print(f"  Iter {metrics['iteration']:4d} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Acc: {metrics['accuracy']:6.2f}% | "
                  f"Throughput: {metrics['throughput']:.1f} samples/s")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        metrics_callback=metrics_callback,
        defense_system=defense
    )
    
    # Train the model
    history = trainer.fit(epochs=10)
    
    # Print final results
    print("\n" + "=" * 80)
    print("Training History Summary")
    print("=" * 80)
    for epoch in range(len(history['train_loss'])):
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {history['train_loss'][epoch]:.4f}, "
              f"Train Acc: {history['train_acc'][epoch]*100:.2f}%, "
              f"Val Loss: {history['val_loss'][epoch]:.4f}, "
              f"Val Acc: {history['val_acc'][epoch]*100:.2f}%")
    print("=" * 80)
    
    print("\n✅ Training complete! The LUMINARK AI framework is fully operational.")
    print("🎯 You now have a working AI framework that can:")
    print("   • Build neural network architectures")
    print("   • Train models with automatic differentiation")
    print("   • Monitor training with real-time metrics")
    print("   • Detect threats with the mycelial defense system")
    print("   • Optimize with SGD and Adam")
    print("   • Load and process datasets")
    print()


if __name__ == '__main__':
    main()
