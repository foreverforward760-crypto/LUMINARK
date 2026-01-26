
import torch
from luminark_core import LuminarkBeast, LuminarkTrainer, LuminarkSafetySystem

# Mock Streamlit Processor
class TextProcessor:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars) + 1
        self.stoi = { ch:i+1 for i,ch in enumerate(chars) }
        self.itos = { i+1:ch for i,ch in enumerate(chars) }
        self.data_tensor = self.encode(text)
        
    def encode(self, s):
        return torch.tensor([self.stoi.get(c, 0) for c in s], dtype=torch.long)
    
    def get_batch(self, batch_size=32, block_size=16):
        ix = torch.randint(len(self.data_tensor) - block_size, (batch_size,))
        x = torch.stack([self.data_tensor[i:i+block_size] for i in ix])
        y = torch.stack([self.data_tensor[i+1:i+block_size+1] for i in ix])
        return x, y

print("Loading Untitled-1.py...")
try:
    with open("Untitled-1.py", "r", encoding="utf-8") as f:
        text_data = f.read()
    print(f"File loaded. Length: {len(text_data)}")
except Exception as e:
    print(f"Failed to read file: {e}")
    exit(1)

print("Initializing Processor...")
processor = TextProcessor(text_data)
print(f"Vocab Size: {processor.vocab_size}")

print("Initializing Model...")
vocab_size = processor.vocab_size + 10
model = LuminarkBeast(vocab_size=vocab_size, hidden_dim=64, layers=4)
safety = LuminarkSafetySystem()
trainer = LuminarkTrainer(model, safety)

print("Running Training Step...")
try:
    x, y = processor.get_batch(batch_size=32, block_size=16)
    metrics = trainer.train_step(x, y)
    print("SUCCESS! Step complete.")
    print(metrics)
except Exception as e:
    print("\nCRASH DETECTED:")
    print(e)
    import traceback
    traceback.print_exc()
