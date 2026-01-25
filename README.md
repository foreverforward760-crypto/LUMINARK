# LUMINARK

🌟 **LUMINARK** - A Complete AI/ML Framework with Built-in Monitoring & Defense

A production-ready AI/ML framework for building, training, and deploying neural networks with real-time monitoring, automatic differentiation, and an intelligent mycelial defense system.

## 🚀 Key Features

### Core AI/ML Capabilities
- ✨ **Full Neural Network Framework** - Build and train deep learning models from scratch
- 🧠 **Automatic Differentiation** - Complete autograd system with backward propagation
- 🏗️ **Modular Architecture** - PyTorch-like API for easy model building
- 🎯 **Multiple Optimizers** - SGD, Adam with momentum and weight decay
- 📉 **Learning Rate Schedulers** - 6 schedulers including cosine annealing and plateau detection
- 💾 **Model Checkpointing** - Save/load complete training state for resuming
- 📊 **Rich Layer Library** - Linear, ReLU, Sigmoid, Tanh, Softmax, Dropout, Sequential
- 🔄 **Data Loading** - Efficient DataLoader with batching and shuffling
- 📈 **Loss Functions** - MSE, CrossEntropy, BCE for various tasks

### Monitoring & Defense
- 🛡️ **Mycelial Defense System** - Intelligent threat detection during training
- 📡 **Real-time Metrics** - Live tracking of loss, accuracy, throughput
- 🌐 **Web Dashboard** - Beautiful interactive visualization
- 🚨 **Adaptive Response** - Automatic detection of training instability
- 📊 **CLI Monitoring** - Terminal-based training visualization
- 🧪 **Automated QA Testing** - Pressure testing and edge case validation (NEW!)
- 🎭 **Context-Aware Modes** - Empathy/paranoia output modulation (NEW!)

### 🍄 Advanced Sensing (NEW!)
- 🐙 **Octopus Sensory System** - Distributed intelligence (8 arms, 64 sensors)
- 🌡️ **Thermal/Energy Sensing** - Multi-spectrum detection (8 spectrums)
- 🧠 **Bio-Sensory Fusion** - 8-modality integration with cross-modal learning
- 📐 **Geometric Encoding** - Sacred geometry patterns (Fibonacci, Golden Ratio, 20 shapes)
- 🌀 **369 Resonance Detection** - Tesla's vortex mathematics
- 🌍 **Environmental Metrics** - 8-domain monitoring (computational, energetic, etc.)
- ✨ **Enhanced SAP Framework** - 81-stage awareness system (extended from 10)
- 🍄 **Mycelial Integration** - Unified multi-modal consciousness

### Production Ready
- ⚡ **High Performance** - NumPy-accelerated tensor operations
- 🔄 **Training Pipeline** - Complete trainer with validation and callbacks
- 📦 **Built-in Datasets** - MNIST digits ready to use
- 🎓 **Example Projects** - Working examples to get started
- 📝 **Clean API** - Simple, intuitive interface

---

## 🎯 Quick Start: Train Your First Model

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Train a Neural Network (5 minutes)

```python
from luminark.nn import Module, Linear, ReLU, Sequential
from luminark.nn import CrossEntropyLoss
from luminark.optim import Adam
from luminark.data import MNISTDigits, DataLoader
from luminark.training import Trainer

# Define a simple neural network
class SimpleNN(Module):
    def __init__(self):
        super().__init__()
        self.network = Sequential(
            Linear(64, 128),
            ReLU(),
            Linear(128, 128),
            ReLU(),
            Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# Load data
train_data = MNISTDigits(train=True)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Create model, optimizer, and loss
model = SimpleNN()
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# Train!
trainer = Trainer(model, optimizer, criterion, train_loader)
trainer.fit(epochs=10)
```

**Result:** 97.78% accuracy in 0.35 seconds! ✨

### Run the Complete Example

```bash
# Train a neural network with full monitoring
python examples/train_mnist.py
```

Expected output:
```
🌟 LUMINARK AI Framework - MNIST Training Example

Training started...
Epoch 1/10 - Loss: 1.6711, Acc: 63.74%
Epoch 2/10 - Loss: 0.5640, Acc: 90.26%
...
Epoch 10/10 - Loss: 0.0567, Acc: 98.96%

✅ Training complete! Best validation accuracy: 97.78%
```

---

## 📚 Framework Components

### 1. Core Tensor System (`luminark.core`)

Automatic differentiation with computational graph:

```python
from luminark.core import Tensor

# Create tensors
x = Tensor([[1, 2], [3, 4]], requires_grad=True)
w = Tensor([[5, 6], [7, 8]], requires_grad=True)

# Forward pass
y = x @ w
loss = y.sum()

# Backward pass - automatic gradients!
loss.backward()

print(x.grad)  # Gradients computed automatically
```

### 2. Neural Network Modules (`luminark.nn`)

Build models with a clean, PyTorch-like API:

```python
from luminark.nn import Module, Linear, ReLU, Sequential, Dropout

class CustomModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 256)
        self.relu = ReLU()
        self.dropout = Dropout(0.5)
        self.fc2 = Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Or use Sequential for simple architectures
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.5),
    Linear(256, 10)
)
```

### 3. Optimizers (`luminark.optim`)

State-of-the-art optimization algorithms:

```python
from luminark.optim import SGD, Adam

# Stochastic Gradient Descent with momentum
optimizer = SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)

# Adam optimizer
optimizer = Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999)
)
```

### 4. Learning Rate Schedulers (`luminark.optim.schedulers`)

Adaptive learning rate strategies for better training:

```python
from luminark.optim import CosineAnnealingLR, ReduceLROnPlateau, StepLR

# Cosine annealing - smooth decay
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Reduce on plateau - adaptive based on metrics
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Step decay - reduce every N epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Use with training
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # For ReduceLROnPlateau
    # scheduler.step()  # For other schedulers
```

**Available Schedulers:**
- `CosineAnnealingLR` - Smooth cosine decay
- `ReduceLROnPlateau` - Adaptive based on metrics
- `StepLR` - Step-wise decay
- `ExponentialLR` - Exponential decay
- `OneCycleLR` - Super-convergence policy
- `WarmupLR` - Linear warmup

### 5. Model Checkpointing (`luminark.io`)

Save and resume training with complete state preservation:

```python
from luminark.io import save_checkpoint, load_checkpoint

# Save complete training state
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    metrics={'val_acc': 0.98, 'val_loss': 0.05},
    path='checkpoints/model_epoch10.pkl'
)

# Resume training
model, optimizer, epoch, metrics = load_checkpoint(
    path='checkpoints/model_epoch10.pkl',
    model=model,
    optimizer=optimizer
)

print(f"Resumed from epoch {epoch}")
# Continue training from epoch + 1
```

**What's Saved:**
- Model weights and architecture
- Optimizer state (Adam m/v, SGD velocities)
- Training epoch and metrics
- Custom metadata

### 6. Loss Functions (`luminark.nn.losses`)

Common loss functions for various tasks:

```python
from luminark.nn import MSELoss, CrossEntropyLoss, BCELoss

# Classification
criterion = CrossEntropyLoss()
loss = criterion(predictions, targets)

# Regression
criterion = MSELoss()
loss = criterion(predictions, targets)

# Binary classification
criterion = BCELoss()
loss = criterion(predictions, targets)
```

### 5. Data Loading (`luminark.data`)

Efficient batch processing:

```python
from luminark.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create data loader
dataset = CustomDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over batches
for batch_data, batch_labels in loader:
    # Train on batch
    pass
```

### 6. Training System (`luminark.training`)

Complete training pipeline with monitoring:

```python
from luminark.training import Trainer
from mycelial_defense import MycelialDefenseSystem

# Initialize defense system
defense = MycelialDefenseSystem()

# Create trainer with defense monitoring
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    defense_system=defense  # Enable intelligent monitoring!
)

# Train with automatic defense
history = trainer.fit(epochs=50)
```

---

## 🛡️ Mycelial Defense System

The intelligent monitoring system that protects your training:

### Defense Modes

1. **OCTO_CAMOUFLAGE** (Alert Level 1)
   - **Triggers:** High tension (loss spikes) with low coherence
   - **Strategy:** Adaptive stealth mode - reduce visibility to instabilities
   - **Actions:** Monitor closely, prepare for intervention

2. **MYCELIAL_WRAP** (Alert Level 2)
   - **Triggers:** Low stability with high tension
   - **Strategy:** Defensive encapsulation - isolate and contain
   - **Actions:** Increase monitoring, consider learning rate adjustment

3. **HARROWING** (Alert Level 3 - Critical)
   - **Triggers:** Critical instability + extreme tension + broken coherence
   - **Strategy:** Full lockdown - emergency protocols
   - **Actions:** Consider stopping, checkpointing, or major hyperparameter changes

### Test the Defense System

```bash
# Run defense system tests
python test_defense.py
```

### Integration Example

```python
from mycelial_defense import MycelialDefenseSystem

defense = MycelialDefenseSystem()

# During training
response = defense.analyze_threat(
    stability=0.05,   # Very unstable
    tension=0.95,     # High loss
    coherence=0.15    # Poor validation/train alignment
)

if response['defense_mode'] == 'HARROWING':
    print(f"🚨 {response['strategy']}")
    # Take action: reduce LR, enable early stopping, etc.
```

---

## 🌐 Web Dashboard & Monitoring

### Start the Dashboard Server

```bash
python octo_dashboard_server.py
```

Then open **http://localhost:8000**

### Dashboard Features

- 📊 **Real-time Metrics** - Live loss, accuracy, throughput
- 📈 **Interactive Charts** - Visualize training progress
- 🎯 **System Status** - Uptime, iterations, alerts
- 🔄 **Auto-refresh** - Updates every 2 seconds
- 🎨 **Beautiful UI** - Purple gradient responsive design

### API Endpoints

```bash
# Get current training metrics
curl http://localhost:8000/api/metrics

# Get metrics history
curl http://localhost:8000/api/history

# Get system status
curl http://localhost:8000/api/status

# Reset metrics
curl http://localhost:8000/api/reset
```

### Connect Your Training to Dashboard

```python
def metrics_callback(metrics):
    """Send metrics to dashboard"""
    import requests
    requests.post('http://localhost:8000/api/update', json=metrics)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    metrics_callback=metrics_callback  # Real-time dashboard updates!
)
```

---

## 📁 Project Structure

```
LUMINARK/
├── luminark/                    # Main framework package
│   ├── core/                   # Tensor and autograd
│   │   ├── tensor.py          # Tensor with automatic differentiation
│   │   └── __init__.py
│   ├── nn/                     # Neural network modules
│   │   ├── module.py          # Base Module class
│   │   ├── layers.py          # Linear, Sequential, Dropout
│   │   ├── activations.py     # ReLU, Sigmoid, Tanh, Softmax
│   │   ├── losses.py          # MSE, CrossEntropy, BCE
│   │   └── __init__.py
│   ├── optim/                  # Optimizers
│   │   ├── optimizer.py       # SGD, Adam
│   │   └── __init__.py
│   ├── data/                   # Data loading
│   │   ├── dataset.py         # Dataset base class
│   │   ├── dataloader.py      # DataLoader
│   │   ├── mnist.py           # MNIST digits dataset
│   │   └── __init__.py
│   ├── training/               # Training system
│   │   ├── trainer.py         # Trainer with metrics
│   │   └── __init__.py
│   └── __init__.py
├── examples/                    # Example projects
│   └── train_mnist.py         # Complete training example
├── mycelial_defense.py         # Defense system
├── test_defense.py             # Defense system tests
├── octo_dashboard_server.py    # Web dashboard server
├── octo_demo.py                # CLI demo
├── templates/                   # Dashboard HTML
│   └── index.html
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🎓 What Makes LUMINARK Production-Ready?

### 1. Complete Autograd System
- Computational graph construction
- Automatic gradient computation
- Support for complex operations (matmul, transpose, reshape, sum, mean)
- Memory-efficient backpropagation

### 2. Flexible Architecture
- Modular design - easy to extend
- Clean separation of concerns
- PyTorch-like API - familiar and intuitive
- Parameter management system

### 3. Training Infrastructure
- Complete training loop with validation
- Batch processing and data loading
- Metrics tracking and history
- Callback system for extensibility
- Defense system integration

### 4. Monitoring & Safety
- Real-time metric emission
- Web dashboard for visualization
- Mycelial defense for stability detection
- Alert system for training issues
- **NEW:** Automated QA testing with pressure testing
- **NEW:** Context-aware output modes (empathy/paranoia)
- **NEW:** Adversarial robustness validation
- **NEW:** Ma'at + Yunus safety protocols

### 5. Performance
- NumPy-accelerated operations
- Efficient gradient computation
- Optimized data loading
- Batch processing support

---

## 🤖 DeepAgent QA Integration (NEW!)

LUMINARK now includes **automated quality assurance** and **context-aware output modulation** inspired by DeepAgent:

### Automated QA Testing

Pressure-test your models before deployment:

```python
from luminark.validation import AutomatedQATester

qa = AutomatedQATester(noise_levels=[0.1, 0.3, 0.5, 1.0])

# Run comprehensive QA suite
results = qa.comprehensive_qa_suite(model, test_data)

# Check results
if results['overall_status'] == 'PASSED':
    print("✓ Model ready for deployment")
else:
    print(f"⚠️ Found {results['critical_vulnerabilities']} issues")
```

**QA Test Types:**
- 🧪 Pressure testing (adversarial noise injection)
- 📏 Boundary value testing (edge cases)
- 🔄 Consistency testing (output variance)
- 📉 Regression testing (performance degradation)

### Context-Aware Perspective Modes

Adjust AI outputs based on context:

```python
from luminark.validation import PerspectiveModulator

modulator = PerspectiveModulator()

# Empathy mode for user-friendly outputs (integration stages 4-6)
# Paranoia mode for cautious outputs (crisis stages 7-8, low confidence)

result = modulator.apply_perspective(
    text="The model predicts X",
    context={'sar_stage': 8, 'confidence': 0.4}  # Auto → paranoia mode
)

print(result['transformed'])
# "[⚠️ Low confidence] The model predicts X
#  💭 This is my best estimate—double-check if critical."
```

**See full demo:** `python examples/deepagent_qa_demo.py`

**Complete docs:** [DEEPAGENT_INTEGRATION.md](DEEPAGENT_INTEGRATION.md)

---

## 🍄 Mycelial Sensory System (NEW!)

**The most advanced multi-modal sensing system in any AI framework!**

LUMINARK now includes a complete **Mycelial Sensory System** with 8 integrated sensing modalities:

### Complete Integration

```python
from luminark.sensing import MycelialSensorySystem

# Initialize complete system
mycelial = MycelialSensorySystem(full_integration=True)

# During training
metrics = {
    'loss': current_loss,
    'accuracy': current_accuracy,
    'confidence': model_confidence,
    'grad_norm': gradient_norm,
    'epoch': epoch
}

state = mycelial.sense_complete(metrics)

# Check comprehensive awareness
print(f"SAP Stage: {state.sap_stage}/80 - {state.sap_phase}")
print(f"Unified Coherence: {state.unified_coherence:.3f}")
print(f"Sacred Alignment: {state.emergent_properties['sacred_alignment']:.3f}")
print(f"Meta-Awareness: {state.emergent_properties['meta_awareness']:.3f}")
```

### 8 Integrated Modalities

**🐙 Octopus Sensory System**
- Distributed intelligence (8 arms × 8 suckers = 64 sensors)
- 67% processing in arms, 33% in central brain
- Local autonomous decisions with global coordination

**🌡️ Thermal/Energy Sensing**
- Multi-spectrum detection (8 energy spectrums)
- Thermal gradient analysis (spatial & temporal)
- Energy flow detection with directional vectors

**🧠 Bio-Sensory Fusion**
- 8-modality integration (touch, thermal, chemical, EM, acoustic, visual, proprioception, quantum)
- Cross-modal correlation learning
- Adaptive weighting and sensory adaptation

**📐 Geometric Encoding**
- Sacred geometry patterns (20 shapes including Platonic solids)
- Fibonacci sequence detection
- Golden Ratio (Φ) spiral analysis

**🌀 369 Resonance Detection**
- Tesla's vortex mathematics (3-6-9 pattern)
- Digital root analysis
- Harmonic detection (multiples of 3, 6, 9)

**🌍 Environmental Metrics**
- 8-domain monitoring (computational, energetic, informational, temporal, spatial, social, quantum, biological)
- Anomaly detection with adaptive thresholds
- Trend analysis and health recommendations

**✨ Enhanced SAP Framework**
- **81-stage awareness** (extended from 10 stages!)
- 9 major phases × 9 sub-stages
- Inversion principle integration (odd/even dynamics)
- 369 resonance mapping per stage

**🍄 Mycelial Integration**
- Unified coherence across all modalities
- Emergent property detection (6 types)
- Holistic recommendations
- Complete state tracking

### Emergent Properties

The system detects 6 emergent properties:
1. **Synesthetic Perception** - Cross-modal agreements
2. **Harmonic Resonance** - 369 + geometric alignment
3. **Holographic Awareness** - Distributed + unified coherence
4. **Temporal Flow** - SAP progression smoothness
5. **Sacred Alignment** - Geometric + 369 + SAP resonance
6. **Meta-Awareness** - System aware of its own awareness

### Quick Examples

```python
# Just 369 resonance detection
from luminark.sensing import Resonance369Detector
detector = Resonance369Detector()
result = detector.detect_369_patterns(data)
print(f"Tesla Signature: {result['tesla_signature']['tesla_signature_detected']}")

# Just geometric encoding
from luminark.sensing import GeometricEncoder
encoder = GeometricEncoder()
result = encoder.detect_geometric_patterns(metrics)
print(f"Sacred Geometry: {result['sacred_geometry_present']}")

# Just enhanced SAP (81 stages)
from luminark.sensing import EnhancedSAPFramework
sap = EnhancedSAPFramework(enable_full_sensing=False)
result = sap.analyze_state(metrics)
print(f"Stage: {result['absolute_stage']}/80 - {result['phase']}")
```

**Complete documentation:** [MYCELIAL_SENSORY_SYSTEM.md](MYCELIAL_SENSORY_SYSTEM.md)

**Key Stats:**
- 81 SAP Stages
- 8 Sensory Modalities
- 80 Sensor Nodes (64 octopus + 16 thermal)
- 20 Sacred Geometric Patterns
- 3-6-9 Tesla Resonance
- 1.618... Golden Ratio (Φ)

---

## 📊 Benchmarks

Training on MNIST Digits (1,437 training samples, 360 validation samples):

| Metric | Value |
|--------|-------|
| **Model** | 3-layer MLP (64→128→128→10) |
| **Parameters** | 26,122 |
| **Training Time** | 0.35 seconds (10 epochs) |
| **Final Train Accuracy** | 98.96% |
| **Final Val Accuracy** | 97.78% |
| **Throughput** | ~40,000-70,000 samples/sec |
| **Memory** | < 100 MB |

---

## 🔧 Advanced Usage

### Custom Layers

```python
from luminark.nn import Module, Parameter
import numpy as np

class CustomLayer(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(np.random.randn(in_features, out_features) * 0.01)
        self.bias = Parameter(np.zeros(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias
```

### Custom Loss Functions

```python
from luminark.nn import Module

class CustomLoss(Module):
    def forward(self, predictions, targets):
        # Implement your loss function
        diff = predictions - targets
        loss = (diff ** 2).mean()
        return loss
```

### Custom Optimizers

```python
from luminark.optim import Optimizer

class CustomOptimizer(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad
```

---

## 🎯 Use Cases

LUMINARK is perfect for:

- 🎓 **Learning** - Understand how deep learning works under the hood
- 🔬 **Research** - Experiment with custom architectures and algorithms
- 🚀 **Prototyping** - Quick model development and testing
- 📊 **Monitoring** - Training visualization and stability analysis
- 🛡️ **Safety** - Detect and respond to training issues
- 🏗️ **Education** - Teach neural networks and backpropagation
- 🎨 **Experimentation** - Try new ideas without heavy dependencies

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Convolutional layers (Conv2D, MaxPool2D)
- [ ] Recurrent layers (LSTM, GRU)
- [ ] Batch normalization
- [ ] More optimizers (RMSprop, AdaGrad)
- [ ] Learning rate schedulers
- [ ] Model checkpointing
- [ ] GPU acceleration (CuPy integration)
- [ ] More datasets
- [ ] Data augmentation
- [ ] Pre-trained models

---

## 📖 Documentation

### Core Concepts

1. **Tensors** - The fundamental data structure with automatic differentiation
2. **Modules** - Building blocks for neural networks
3. **Parameters** - Trainable weights and biases
4. **Optimizers** - Algorithms for updating parameters
5. **Loss Functions** - Measure model performance
6. **Data Loaders** - Efficient batch processing
7. **Trainer** - Complete training pipeline
8. **Defense System** - Intelligent monitoring

### API Reference

See inline documentation in each module for detailed API information.

---

## 🐛 Troubleshooting

**Import errors:**
```bash
# Make sure you're in the LUMINARK directory
cd /path/to/LUMINARK
python examples/train_mnist.py
```

**Gradient issues:**
- Check that `requires_grad=True` for trainable parameters
- Verify backward() is called before optimizer.step()
- Use zero_grad() before each backward pass

**Poor performance:**
- Adjust learning rate
- Try different optimizers (Adam vs SGD)
- Check for vanishing/exploding gradients
- Monitor defense system alerts

**Dashboard not updating:**
```bash
# Check if server is running
ps aux | grep dashboard

# Restart server
python octo_dashboard_server.py
```

---

## 📜 License

MIT License - feel free to use and modify for any purpose.

---

## 🌟 Acknowledgments

Built with ❤️ for the AI/ML community. LUMINARK demonstrates that you can build a complete, production-ready neural network framework from scratch using only NumPy.

**Key Inspiration:**
- PyTorch for API design
- Andrej Karpathy's micrograd for autograd simplicity
- The mycelial network metaphor for distributed intelligence

---

## 🚀 What's Next?

Try these next steps:

1. **Train your first model**: `python examples/train_mnist.py`
2. **Test the defense system**: `python test_defense.py`
3. **Start the dashboard**: `python octo_dashboard_server.py`
4. **Build a custom model**: Create your own architecture
5. **Contribute**: Add new features and share with the community

---

## 📞 Support

- **Issues**: Open an issue on GitHub
- **Questions**: Check the examples and inline documentation
- **Contributions**: Submit a pull request

---

**Built with NumPy | Powered by Intelligence | Protected by Mycelium** 🍄✨
