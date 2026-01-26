
print("Testing LUMINARK Package Imports...")
try:
    from luminark.nn.layers import ToroidalAttention
    print("✅ luminark.nn.layers.ToroidalAttention imported")
    
    from luminark.training.trainer import LuminarkTrainer
    print("✅ luminark.training.trainer.LuminarkTrainer imported")
    
    from luminark.monitoring.defense import LuminarkSafetySystem
    print("✅ luminark.monitoring.defense.LuminarkSafetySystem imported")
    
    from luminark.io.checkpoint import Checkpoint
    print("✅ luminark.io.checkpoint.Checkpoint imported")
    
    print("🎉 All production modules are accessible.")
except ImportError as e:
    print(f"❌ Import Failed: {e}")
    exit(1)
