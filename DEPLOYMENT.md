# LUMINARK Deployment Guide

Complete guide for deploying LUMINARK in production environments.

---

## 🐳 Docker Deployment

### Quick Start with Docker

**Build the image:**
```bash
docker build -t luminark:latest .
```

**Run training:**
```bash
docker run luminark:latest
```

**Run dashboard:**
```bash
docker run -p 8000:8000 luminark:latest python octo_dashboard_server.py
```

**Interactive shell:**
```bash
docker run -it luminark:latest /bin/bash
```

### Docker Compose (Recommended)

**Start all services:**
```bash
docker-compose up
```

**Start specific service:**
```bash
# Training only
docker-compose up luminark-train

# Dashboard only
docker-compose up luminark-dashboard

# Advanced AI
docker-compose up luminark-advanced
```

**Run in background:**
```bash
docker-compose up -d
```

**View logs:**
```bash
docker-compose logs -f luminark-dashboard
```

**Stop services:**
```bash
docker-compose down
```

---

## 📦 Package Installation

### Install as Python Package

**Development install:**
```bash
pip install -e .
```

**Production install:**
```bash
pip install .
```

**From PyPI (when published):**
```bash
pip install luminark
```

### Optional Dependencies

**Minimal (core only):**
```bash
pip install -r requirements-prod.txt
```

**With quantum features:**
```bash
pip install luminark[quantum]
```

**With all features:**
```bash
pip install luminark[all]
```

---

## 🚀 Production Deployment

### 1. Model Training Service

**Create training script:**
```python
# train_production.py
from luminark.nn import Module, Sequential, Linear, ReLU
from luminark.optim import Adam
from luminark.optim import ReduceLROnPlateau
from luminark.training import Trainer
from luminark.io import save_checkpoint
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

class ProductionModel(Module):
    def __init__(self):
        super().__init__()
        self.network = Sequential(
            Linear(784, 256), ReLU(),
            Linear(256, 128), ReLU(),
            Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# Training configuration
model = ProductionModel()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, patience=5)

# Checkpoint callback
def save_best_model(metrics):
    if metrics['val_acc'] > save_best_model.best_acc:
        save_best_model.best_acc = metrics['val_acc']
        save_checkpoint(
            model, optimizer,
            epoch=metrics['epoch'],
            metrics=metrics,
            path='checkpoints/best_model.pkl'
        )
        logging.info(f"New best model saved! Val Acc: {metrics['val_acc']:.4f}")

save_best_model.best_acc = 0.0

# Train
trainer = Trainer(
    model, optimizer, criterion,
    train_loader, val_loader,
    metrics_callback=save_best_model
)

try:
    history = trainer.fit(epochs=100)
    logging.info("Training completed successfully")
except Exception as e:
    logging.error(f"Training failed: {e}")
    raise
```

**Run with monitoring:**
```bash
nohup python train_production.py > logs/training.log 2>&1 &
```

### 2. Model Serving API

**Create Flask API:**
```python
# serve_model.py
from flask import Flask, request, jsonify
from luminark.io import load_model
from luminark.core import Tensor
import numpy as np

app = Flask(__name__)

# Load model
model = YourModel()
load_model('checkpoints/best_model.pkl', model)
model.eval()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        input_tensor = Tensor(np.array(data, dtype=np.float32))

        # Inference
        predictions = model(input_tensor)

        return jsonify({
            'predictions': predictions.data.tolist(),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Production server with Gunicorn:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 serve_model:app
```

### 3. Batch Inference Service

**Create batch processor:**
```python
# batch_inference.py
from luminark.io import load_model
from luminark.core import Tensor
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)

class BatchInferenceService:
    def __init__(self, model_path, batch_size=32):
        self.model = YourModel()
        load_model(model_path, self.model)
        self.batch_size = batch_size

    def process_file(self, input_path, output_path):
        """Process data from file and save predictions"""
        # Load data
        data = np.load(input_path)

        # Batch process
        results = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_tensor = Tensor(batch)
            predictions = self.model(batch_tensor)
            results.extend(predictions.data.tolist())

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f)

        logging.info(f"Processed {len(data)} samples")

if __name__ == '__main__':
    service = BatchInferenceService('checkpoints/best_model.pkl')
    service.process_file('input.npy', 'predictions.json')
```

---

## ☁️ Cloud Deployment

### AWS Deployment

**1. EC2 Instance:**
```bash
# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start

# Deploy LUMINARK
git clone https://github.com/foreverforward760-crypto/LUMINARK.git
cd LUMINARK
docker-compose up -d
```

**2. ECS (Container Service):**
```yaml
# task-definition.json
{
  "family": "luminark-training",
  "containerDefinitions": [
    {
      "name": "luminark",
      "image": "your-registry/luminark:latest",
      "memory": 2048,
      "cpu": 1024,
      "essential": true
    }
  ]
}
```

**3. Lambda (Serverless):**
```python
# lambda_handler.py
import json
from luminark.io import load_model
from luminark.core import Tensor
import numpy as np

# Load model once (outside handler)
model = YourModel()
load_model('/tmp/model.pkl', model)

def lambda_handler(event, context):
    data = np.array(event['data'], dtype=np.float32)
    input_tensor = Tensor(data)
    predictions = model(input_tensor)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'predictions': predictions.data.tolist()
        })
    }
```

### Google Cloud Platform

**Cloud Run deployment:**
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/luminark
gcloud run deploy luminark --image gcr.io/PROJECT_ID/luminark --platform managed
```

### Azure Deployment

**Container Instances:**
```bash
az container create \
  --resource-group myResourceGroup \
  --name luminark-container \
  --image your-registry/luminark:latest \
  --cpu 2 --memory 4 \
  --ports 8000
```

---

## 📊 Monitoring & Logging

### Application Logging

**Setup structured logging:**
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module
        }
        return json.dumps(log_data)

handler = logging.FileHandler('logs/app.log')
handler.setFormatter(JSONFormatter())
logging.root.addHandler(handler)
```

### Metrics Collection

**Track training metrics:**
```python
from luminark.training import Trainer

metrics_history = []

def collect_metrics(metrics):
    metrics_history.append({
        'timestamp': time.time(),
        'epoch': metrics['epoch'],
        'loss': metrics['loss'],
        'accuracy': metrics['accuracy']
    })

    # Send to monitoring service
    # send_to_prometheus(metrics)
    # send_to_cloudwatch(metrics)

trainer = Trainer(..., metrics_callback=collect_metrics)
```

---

## 🔐 Security Best Practices

### 1. Model Security
- ✅ Version control your models
- ✅ Sign model checkpoints
- ✅ Validate model inputs
- ✅ Rate limit API endpoints

### 2. Environment Security
```python
# Use environment variables
import os

MODEL_PATH = os.getenv('MODEL_PATH', 'models/default.pkl')
API_KEY = os.getenv('API_KEY')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
```

### 3. Container Security
```dockerfile
# Run as non-root user
RUN useradd -m -u 1000 luminark
USER luminark

# Read-only root filesystem
docker run --read-only luminark:latest
```

---

## 🔧 Performance Optimization

### 1. Model Optimization
- Use smaller batch sizes for lower latency
- Implement model quantization
- Cache frequently used models
- Use model ensembles for better accuracy

### 2. Infrastructure Optimization
- Auto-scaling based on load
- Load balancing across instances
- Caching predictions for common inputs
- GPU acceleration when available

### 3. Code Optimization
```python
# Batch predictions
def predict_batch(data_list):
    # More efficient than individual predictions
    batch = np.array(data_list)
    return model(Tensor(batch))
```

---

## 📋 Deployment Checklist

- [ ] Requirements installed
- [ ] Model trained and validated
- [ ] Checkpoints saved
- [ ] Docker image built
- [ ] Tests passing
- [ ] Logging configured
- [ ] Monitoring setup
- [ ] Security hardened
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Backup strategy in place
- [ ] Rollback plan ready

---

## 🆘 Troubleshooting

### Common Issues

**Out of Memory:**
```python
# Reduce batch size
train_loader = DataLoader(data, batch_size=16)  # Instead of 32
```

**Slow Inference:**
```python
# Disable gradient computation
with torch.no_grad():  # If using PyTorch compatibility
    predictions = model(input_tensor)
```

**Model Not Loading:**
```python
# Verify checkpoint integrity
import pickle
with open('checkpoint.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
print(checkpoint.keys())
```

---

## 📞 Support

- GitHub Issues: https://github.com/foreverforward760-crypto/LUMINARK/issues
- Documentation: See README.md and ADVANCED_FEATURES.md
- Examples: Check examples/ directory

---

**You're now ready for production deployment! 🚀**
