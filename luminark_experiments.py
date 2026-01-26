import torch
from luminark_core import ToroidalAttention, GatedLinear, Module, Linear, LuminarkSafetySystem

print("🧪 STARTING LUMINARK EXPERIMENTS...")

# ==========================================
# Experiment 1: Custom AI with Toroidal Attn
# ==========================================
print("\n[Experiment 1] Building Custom AI with Toroidal Attention...")

class MyCustomAI(Module):
    def __init__(self):
        super().__init__()
        self.attention = ToroidalAttention(hidden_dim=32, num_heads=4, window_size=2)
        self.gated = GatedLinear(32, 16)
        self.output = Linear(16, 2) # Binary classification
    
    def forward(self, x):
        x = self.attention(x)
        x = self.gated(x)
        return self.output(x)

try:
    model = MyCustomAI()
    dummy_input = torch.randn(1, 10, 32) # Batch 1, Seq 10, Dim 32
    output = model(dummy_input)
    print("✅ Experiment 1 SUCCESS: Model Forward Pass Complete")
    print(f"   Output Shape: {output.shape}")
except Exception as e:
    print(f"❌ Experiment 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# ==========================================
# Experiment 2: Monitor Training Safety
# ==========================================
print("\n[Experiment 2] Testing Enhanced Defense System...")

defense = LuminarkSafetySystem()
try:
    # Simulate a "Hallucination" state
    analysis = defense.analyze_training_state({
        'loss': 0.8,         # High Loss
        'confidence': 0.995, # High Confidence
        'grad_norm': 2.0
    })

    print(f"   Simulated State: {analysis}")
    if analysis['stage_value'] >= 7:
        print(f"✅ Experiment 2 SUCCESS: Warning Triggered correctly: {analysis['description']}")
    else:
        print("❌ Experiment 2 FAILED: Warning not triggered")
except Exception as e:
    print(f"❌ Experiment 2 FAILED: {e}")

# ==========================================
# Experiment 3: Quantum Confidence
# ==========================================
print("\n[Experiment 3] Estimating Quantum Confidence...")

try:
    # Fake logits
    logits = torch.tensor([2.0, 1.0, 0.5]) 
    q_conf = defense.estimate_quantum_confidence(logits)
    print(f"✅ Experiment 3 SUCCESS: Quantum Confidence Calculated: {q_conf:.2%}")
except Exception as e:
    print(f"❌ Experiment 3 FAILED: {e}")

print("\n🧪 EXPERIMENTS COMPLETE.")

# ==========================================
# Experiment 4: Beast Mode Activation
# ==========================================
from luminark_core import LuminarkBeast, LuminarkTrainer
import torch

print("\n[Experiment 4] Testing LUMINARK BEAST...")
try:
    # 1. Initialize Beast
    beast = LuminarkBeast(vocab_size=100, hidden_dim=32, layers=2)
    safety = LuminarkSafetySystem()
    trainer = LuminarkTrainer(beast, safety)
    
    # 2. Train Loop
    x = torch.randint(0, 100, (4, 10))
    y = torch.randint(0, 100, (4, 10))
    
    metrics = trainer.train_step(x, y)
    print(f"✅ Experiment 4 SUCCESS: Beast Trained 1 Step.")
    print(f"   - Metrics: {metrics}")
except Exception as e:
    print(f"❌ Experiment 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

