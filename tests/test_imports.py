#!/usr/bin/env python3
"""
Quick import verification test
Run this immediately after restructuring to verify all imports work
"""
import sys
sys.path.insert(0, '/home/user/LUMINARK')

print("="*70)
print("LUMINARK Import Verification Test")
print("="*70)

# Core imports
try:
    from luminark.core import Tensor
    from luminark.core.quantum import QuantumUncertaintyEstimator
    print("✅ Core imports: OK")
except Exception as e:
    print(f"❌ Core imports: FAILED - {e}")
    sys.exit(1)

# NN imports
try:
    from luminark.nn import Module, Linear, ReLU, Sequential
    from luminark.nn import CrossEntropyLoss, MSELoss, BCELoss
    from luminark.nn.advanced_layers import ToroidalAttention, GatedLinear
    from luminark.nn.advanced_layers import AttentionPooling, ResidualBlock
    print("✅ NN imports: OK")
except Exception as e:
    print(f"❌ NN imports: FAILED - {e}")
    sys.exit(1)

# Optimizer imports
try:
    from luminark.optim import SGD, Adam
    from luminark.optim import (
        CosineAnnealingLR, ReduceLROnPlateau, StepLR,
        ExponentialLR, OneCycleLR, WarmupLR
    )
    print("✅ Optimizer imports: OK")
except Exception as e:
    print(f"❌ Optimizer imports: FAILED - {e}")
    sys.exit(1)

# IO imports (NEW)
try:
    from luminark.io import save_checkpoint, load_checkpoint, Checkpoint
    from luminark.io import save_model, load_model
    print("✅ IO imports: OK")
except Exception as e:
    print(f"❌ IO imports: FAILED - {e}")
    sys.exit(1)

# Data imports
try:
    from luminark.data import MNISTDigits, DataLoader, Dataset
    print("✅ Data imports: OK")
except Exception as e:
    print(f"❌ Data imports: FAILED - {e}")
    sys.exit(1)

# Training imports
try:
    from luminark.training import Trainer
    from luminark.training.meta_learner import MetaLearner
    print("✅ Training imports: OK")
except Exception as e:
    print(f"❌ Training imports: FAILED - {e}")
    sys.exit(1)

# Monitoring imports
try:
    from luminark.monitoring.enhanced_defense import EnhancedDefenseSystem
    print("✅ Monitoring imports: OK")
except Exception as e:
    print(f"❌ Monitoring imports: FAILED - {e}")
    sys.exit(1)

# Memory imports
try:
    from luminark.memory.associative_memory import AssociativeMemory
    print("✅ Memory imports: OK")
except Exception as e:
    print(f"❌ Memory imports: FAILED - {e}")
    sys.exit(1)

print("="*70)
print("✅ ALL IMPORTS SUCCESSFUL!")
print("="*70)
print("\nPackage structure verified. All modules accessible.")
