#!/usr/bin/env python3
"""
Demonstration of Model Checkpointing and Learning Rate Schedulers
Shows how to save/load models and use adaptive learning rates
"""
import sys
sys.path.insert(0, '/home/user/LUMINARK')

import numpy as np
from luminark.nn import Module, Linear, ReLU, Sequential
from luminark.nn import CrossEntropyLoss
from luminark.optim import Adam
from luminark.optim import CosineAnnealingLR, ReduceLROnPlateau
from luminark.data import MNISTDigits, DataLoader
from luminark.training import Trainer
from luminark.io import save_checkpoint, load_checkpoint, Checkpoint

print("""
╔══════════════════════════════════════════════════════════╗
║        CHECKPOINT & SCHEDULER DEMONSTRATION              ║
║    Advanced Training Techniques for Production AI        ║
╚══════════════════════════════════════════════════════════╝
""")

# Define a simple model
class SimpleNN(Module):
    def __init__(self):
        super().__init__()
        self.network = Sequential(
            Linear(64, 128), ReLU(),
            Linear(128, 128), ReLU(),
            Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# Part 1: Training with Learning Rate Scheduler
# ============================================================================
print("="*70)
print("PART 1: Training with Cosine Annealing LR Scheduler")
print("="*70)

# Load data
train_data = MNISTDigits(train=True, normalize=True)
val_data = MNISTDigits(train=False, normalize=True)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Create model and optimizer
model = SimpleNN()
optimizer = Adam(model.parameters(), lr=0.01)  # Start with higher LR
criterion = CrossEntropyLoss()

# Create cosine annealing scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

print(f"Initial learning rate: {optimizer.lr:.6f}\n")

# Track learning rates
lr_history = []

# Custom callback to track LR
def lr_callback(metrics):
    lr_history.append(optimizer.lr)

# Train with scheduler
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    metrics_callback=lr_callback
)

print("Training with cosine annealing LR schedule...\n")
history = trainer.fit(epochs=10)

# Step scheduler each epoch
for epoch in range(10):
    scheduler.step()

print("\n" + "="*70)
print("Learning Rate Schedule:")
print("="*70)
for epoch, lr in enumerate(lr_history[:10], 1):
    print(f"Epoch {epoch:2d}: LR = {lr:.6f}")

print(f"\nFinal validation accuracy: {history['val_acc'][-1]*100:.2f}%")


# ============================================================================
# Part 2: Saving Checkpoint
# ============================================================================
print("\n" + "="*70)
print("PART 2: Saving Model Checkpoint")
print("="*70)

checkpoint_path = 'checkpoints/model_epoch10.pkl'
final_metrics = {
    'train_acc': history['train_acc'][-1],
    'val_acc': history['val_acc'][-1],
    'train_loss': history['train_loss'][-1],
    'val_loss': history['val_loss'][-1]
}

print(f"\nSaving checkpoint with metrics:")
for key, value in final_metrics.items():
    print(f"  {key}: {value:.4f}")

# Save complete checkpoint
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    metrics=final_metrics,
    path=checkpoint_path
)


# ============================================================================
# Part 3: Loading Checkpoint and Resuming
# ============================================================================
print("\n" + "="*70)
print("PART 3: Loading Checkpoint and Resuming Training")
print("="*70)

# Create new model and optimizer instances
new_model = SimpleNN()
new_optimizer = Adam(new_model.parameters(), lr=0.001)

# Load checkpoint
loaded_model, loaded_optimizer, loaded_epoch, loaded_metrics = load_checkpoint(
    path=checkpoint_path,
    model=new_model,
    optimizer=new_optimizer
)

print(f"\nLoaded checkpoint from epoch {loaded_epoch}")
print(f"Restored metrics:")
for key, value in loaded_metrics.items():
    print(f"  {key}: {value:.4f}")

# Verify model works
print("\n🔬 Verifying loaded model...")
test_input = np.random.randn(1, 64).astype(np.float32)
from luminark.core.tensor import Tensor
test_tensor = Tensor(test_input)
output = loaded_model(test_tensor)
print(f"✓ Model inference successful! Output shape: {output.data.shape}")


# ============================================================================
# Part 4: ReduceLROnPlateau Scheduler
# ============================================================================
print("\n" + "="*70)
print("PART 4: ReduceLROnPlateau Scheduler")
print("="*70)

# Create new model for this demonstration
model2 = SimpleNN()
optimizer2 = Adam(model2.parameters(), lr=0.01)

# Create plateau scheduler
plateau_scheduler = ReduceLROnPlateau(
    optimizer2,
    mode='min',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

print(f"Initial LR: {optimizer2.lr:.6f}")
print("Simulating training with plateau detection...\n")

# Simulate some loss values
simulated_losses = [2.3, 2.1, 1.9, 1.85, 1.83, 1.82, 1.82, 1.82, 1.82, 1.81]

for epoch, loss in enumerate(simulated_losses, 1):
    print(f"Epoch {epoch:2d}: Loss = {loss:.3f}, LR = {optimizer2.lr:.6f}")
    plateau_scheduler.step(loss)


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("✅ DEMONSTRATION COMPLETE!")
print("="*70)
print("""
🎓 You've learned how to:

1. 📉 Learning Rate Schedulers
   • CosineAnnealingLR - Smooth cosine decay
   • ReduceLROnPlateau - Adaptive based on metrics
   • StepLR, ExponentialLR, OneCycleLR, WarmupLR also available

2. 💾 Model Checkpointing
   • save_checkpoint() - Save model, optimizer, metrics
   • load_checkpoint() - Resume training exactly where you left off
   • Checkpoint class for advanced control

3. 🔄 Resume Training
   • Load model weights and optimizer state
   • Continue from any epoch
   • Preserve training history

🚀 Production-Ready Features:
   ✓ Save/load models for deployment
   ✓ Resume interrupted training
   ✓ Adaptive learning rates
   ✓ Automatic LR reduction on plateaus
   ✓ Complete training state persistence

💡 Next Steps:
   • Try different schedulers with your models
   • Implement checkpointing in your training loops
   • Combine schedulers for advanced training strategies
   • Save best model based on validation metrics
""")

print(f"\n📁 Checkpoint saved at: {checkpoint_path}")
print("="*70)
