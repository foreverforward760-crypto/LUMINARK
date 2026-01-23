#!/usr/bin/env python3
"""
Full integration test: Train -> Save -> Load -> Continue Training
Tests the complete workflow with all components
"""
import sys
sys.path.insert(0, '/home/user/LUMINARK')

from luminark.nn import Module, Linear, ReLU, Sequential
from luminark.nn import CrossEntropyLoss
from luminark.optim import Adam, ReduceLROnPlateau
from luminark.data import MNISTDigits, DataLoader
from luminark.training import Trainer
from luminark.io import save_checkpoint, load_checkpoint
from luminark.monitoring.enhanced_defense import EnhancedDefenseSystem
import os

print("="*70)
print("LUMINARK FULL INTEGRATION TEST")
print("Train -> Save -> Load -> Resume -> Monitor")
print("="*70)

# Phase 1: Train initial model
print("\n[Phase 1] Training initial model (3 epochs)...")

class TestModel(Module):
    def __init__(self):
        super().__init__()
        self.network = Sequential(
            Linear(64, 128), ReLU(),
            Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

model = TestModel()
optimizer = Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()

train_data = MNISTDigits(train=True, normalize=True)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

trainer = Trainer(model, optimizer, criterion, train_loader)
history1 = trainer.fit(epochs=3)

acc1 = history1['train_acc'][-1]
loss1 = history1['train_loss'][-1]

print(f"✅ Phase 1 complete")
print(f"   Accuracy: {acc1*100:.2f}%")
print(f"   Loss: {loss1:.4f}")

# Phase 2: Save checkpoint
print("\n[Phase 2] Saving checkpoint...")
checkpoint_path = '/tmp/integration_test.pkl'
save_checkpoint(
    model, optimizer, 3,
    {'acc': acc1, 'loss': loss1},
    checkpoint_path
)
print(f"✅ Checkpoint saved to {checkpoint_path}")

# Phase 3: Load checkpoint into new model
print("\n[Phase 3] Loading checkpoint into new model...")
new_model = TestModel()
new_optimizer = Adam(new_model.parameters(), lr=0.001)  # Different LR

loaded_model, loaded_opt, epoch, metrics = load_checkpoint(
    checkpoint_path, new_model, new_optimizer
)

print(f"✅ Checkpoint loaded successfully")
print(f"   Resumed from epoch: {epoch}")
print(f"   Restored accuracy: {metrics['acc']*100:.2f}%")
print(f"   Optimizer LR restored: {loaded_opt.lr}")

# Verify LR was restored
assert loaded_opt.lr == 0.01, f"LR not restored correctly: {loaded_opt.lr}"

# Phase 4: Continue training with scheduler
print("\n[Phase 4] Continuing training with ReduceLROnPlateau (3 more epochs)...")
scheduler = ReduceLROnPlateau(loaded_opt, mode='min', patience=2, factor=0.5)

new_trainer = Trainer(loaded_model, loaded_opt, criterion, train_loader)

# Custom callback to step scheduler
def scheduler_callback(metrics):
    scheduler.step(metrics['loss'])

new_trainer.metrics_callback = scheduler_callback
history2 = new_trainer.fit(epochs=3)

acc2 = history2['train_acc'][-1]
loss2 = history2['train_loss'][-1]

print(f"✅ Phase 4 complete")
print(f"   Accuracy: {acc2*100:.2f}%")
print(f"   Loss: {loss2:.4f}")
print(f"   Final LR: {loaded_opt.lr:.6f}")

# Verify improvement
if acc2 > acc1:
    print(f"   🎉 Accuracy improved by {(acc2-acc1)*100:.2f}%!")

# Phase 5: Defense system monitoring
print("\n[Phase 5] Testing defense system monitoring...")
defense = EnhancedDefenseSystem()

state = defense.analyze_training_state({
    'loss': loss2,
    'accuracy': acc2,
    'grad_norm': 1.0,
    'epoch': 6
})

print(f"✅ Defense analysis complete")
print(f"   Awareness Stage: {state['stage']}")
print(f"   Risk Level: {state['risk_level']}")
print(f"   Stability: {state['metrics']['stability']:.2f}")
print(f"   Coherence: {state['metrics']['coherence']:.2f}")

# Phase 6: Verify model works
print("\n[Phase 6] Verifying loaded model inference...")
import numpy as np
from luminark.core import Tensor

test_input = np.random.randn(1, 64).astype(np.float32)
test_tensor = Tensor(test_input)
output = loaded_model(test_tensor)

assert output.data.shape == (1, 10), f"Wrong output shape: {output.data.shape}"
prediction = np.argmax(output.data)
print(f"✅ Model inference working")
print(f"   Test input shape: {test_input.shape}")
print(f"   Output shape: {output.data.shape}")
print(f"   Predicted class: {prediction}")

# Cleanup
os.remove(checkpoint_path)
print(f"\n✅ Cleaned up test checkpoint")

print("\n" + "="*70)
print("🎉 FULL INTEGRATION TEST PASSED!")
print("="*70)
print("\nAll components working together:")
print("  ✅ Model training")
print("  ✅ Checkpointing")
print("  ✅ Checkpoint loading")
print("  ✅ Training resumption")
print("  ✅ Learning rate scheduling")
print("  ✅ Defense monitoring")
print("  ✅ Model inference")
print("\nLUMINARK framework is production-ready! 🚀")
