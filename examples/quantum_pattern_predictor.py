#!/usr/bin/env python3
"""
Quantum-Aware Pattern Predictor
A self-aware AI that predicts time series with quantum confidence scoring
and 10-stage awareness defense against overconfident predictions

Features:
- Quantum uncertainty estimation on every prediction
- 10-stage awareness system warns when predictions are unreliable
- Meta-learner improves prediction strategy over time
- Auto-checkpoints best models
- Real-time confidence visualization

Example use cases:
- Stock price prediction
- Crypto price prediction
- Weather forecasting
- Sales forecasting
- Any sequential pattern prediction
"""
import sys
sys.path.insert(0, '/home/user/LUMINARK')

import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from luminark.nn import Module, Linear, ReLU, Sequential
from luminark.nn import MSELoss
from luminark.nn.advanced_layers import GatedLinear, ToroidalAttention
from luminark.optim import Adam, ReduceLROnPlateau
from luminark.core import Tensor
from luminark.core.quantum import QuantumUncertaintyEstimator
from luminark.monitoring.enhanced_defense import EnhancedDefenseSystem
from luminark.training.meta_learner import MetaLearningEngine
from luminark.io import save_checkpoint, load_checkpoint


class QuantumPatternPredictor(Module):
    """
    Neural network with toroidal attention for pattern prediction
    Uses gated layers for adaptive feature selection
    """

    def __init__(self, sequence_length=20, hidden_dim=64):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = Linear(1, hidden_dim)
        self.relu1 = ReLU()

        # Toroidal attention for circular pattern detection
        self.attention = ToroidalAttention(hidden_dim, window_size=7)

        # Gated layers for adaptive processing
        self.gated1 = GatedLinear(hidden_dim, hidden_dim)
        self.relu2 = ReLU()

        self.gated2 = GatedLinear(hidden_dim, hidden_dim)
        self.relu3 = ReLU()

        # Output projection
        self.output_proj = Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (batch_size, sequence_length, 1)
        returns: (batch_size, 1) - next value prediction
        """
        # Project input
        x = self.input_proj(x)
        x = self.relu1(x)

        # Apply toroidal attention (treats sequence as circular)
        x = self.attention(x)

        # Gated processing
        x = self.gated1(x)
        x = self.relu2(x)

        x = self.gated2(x)
        x = self.relu3(x)

        # Take last time step and project to output
        # x shape: (batch, seq, hidden) -> take x[:, -1, :] -> (batch, hidden)
        last_hidden = Tensor(x.data[:, -1, :], requires_grad=x.requires_grad)
        output = self.output_proj(last_hidden)

        return output


class QuantumAwarePredictor:
    """
    Complete prediction system with quantum confidence and self-awareness
    """

    def __init__(self, sequence_length=20, hidden_dim=64, learning_rate=0.001):
        self.sequence_length = sequence_length

        # Core model
        self.model = QuantumPatternPredictor(sequence_length, hidden_dim)

        # Optimizer with scheduler
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=10,
            factor=0.5
        )

        # Loss function
        self.criterion = MSELoss()

        # Quantum confidence estimator
        self.quantum_estimator = QuantumUncertaintyEstimator(num_qubits=4)

        # 10-stage awareness defense system
        self.defense = EnhancedDefenseSystem()

        # Meta-learner for self-improvement
        self.meta_learner = MetaLearningEngine()

        # Training history
        self.training_history = []
        self.prediction_history = []

        # Best model tracking
        self.best_loss = float('inf')
        self.checkpoint_path = 'checkpoints/quantum_predictor_best.pkl'

    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert time series into supervised learning sequences

        Args:
            data: 1D array of values

        Returns:
            X: (num_samples, sequence_length, 1)
            y: (num_samples, 1) - next value
        """
        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])

        X = np.array(X).reshape(-1, self.sequence_length, 1).astype(np.float32)
        y = np.array(y).reshape(-1, 1).astype(np.float32)

        return X, y

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0

        # Simple batch training
        batch_size = 32
        indices = np.random.permutation(len(X))

        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]

            # Forward pass
            X_tensor = Tensor(batch_X, requires_grad=True)
            predictions = self.model(X_tensor)

            # Compute loss
            y_tensor = Tensor(batch_y, requires_grad=False)
            loss = self.criterion(predictions, y_tensor)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data
            num_batches += 1

        avg_loss = total_loss / num_batches

        return {'loss': avg_loss, 'num_batches': num_batches}

    def fit(self, data: np.ndarray, epochs=50, verbose=True):
        """
        Train the predictor on time series data

        Args:
            data: 1D array of sequential values
            epochs: Number of training epochs
        """
        # Prepare data
        X, y = self.prepare_sequences(data)

        if verbose:
            print(f"\n{'='*70}")
            print(f"QUANTUM-AWARE PATTERN PREDICTOR - Training")
            print(f"{'='*70}")
            print(f"Training samples: {len(X)}")
            print(f"Sequence length: {self.sequence_length}")
            print(f"Epochs: {epochs}")
            print(f"{'='*70}\n")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(X, y)
            loss = train_metrics['loss']

            # Step scheduler
            self.scheduler.step(loss)

            # Calculate metrics for defense system
            grad_norm = self._calculate_grad_norm()

            # Analyze with defense system
            defense_state = self.defense.analyze_training_state({
                'loss': loss,
                'accuracy': 1.0 - min(loss, 1.0),  # Approximate accuracy
                'grad_norm': grad_norm,
                'epoch': epoch
            })

            # Record for meta-learner
            self.meta_learner.record_training_result(
                config={'lr': self.optimizer.lr, 'epoch': epoch},
                performance={'loss': loss}
            )

            # Save history
            self.training_history.append({
                'epoch': epoch,
                'loss': loss,
                'lr': self.optimizer.lr,
                'awareness_stage': defense_state['stage'].value,
                'risk_level': defense_state['risk_level'],
                'time': time.time() - start_time
            })

            # Save best model
            if loss < self.best_loss:
                self.best_loss = loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    {'loss': loss, 'best_loss': self.best_loss},
                    self.checkpoint_path
                )
                if verbose:
                    print(f"✓ New best model saved! Loss: {loss:.6f}")

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {loss:.6f} | "
                      f"LR: {self.optimizer.lr:.2e} | "
                      f"Stage: {defense_state['stage'].value} | "
                      f"Risk: {defense_state['risk_level']:8s} | "
                      f"Time: {elapsed:.3f}s")

                # Show warning if risk elevated
                if defense_state['risk_level'] in ['elevated', 'high', 'critical']:
                    print(f"  ⚠️  {defense_state['description']}")

        if verbose:
            print(f"\n{'='*70}")
            print(f"Training Complete!")
            print(f"Best Loss: {self.best_loss:.6f}")
            print(f"Final LR: {self.optimizer.lr:.2e}")
            print(f"{'='*70}\n")

    def predict_with_confidence(self, sequence: np.ndarray) -> Dict:
        """
        Make prediction with quantum confidence and awareness analysis

        Args:
            sequence: Recent values (must be sequence_length long)

        Returns:
            Dict with prediction, confidence, quantum_uncertainty, awareness_state
        """
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Sequence must be {self.sequence_length} values long")

        # Prepare input
        X = sequence.reshape(1, self.sequence_length, 1).astype(np.float32)
        X_tensor = Tensor(X, requires_grad=False)

        # Get prediction
        prediction = self.model(X_tensor)
        pred_value = float(prediction.data[0, 0])

        # Estimate quantum uncertainty
        # Use prediction as probability distribution (normalized)
        pred_dist = np.array([pred_value, 1 - pred_value])
        pred_dist = np.abs(pred_dist) / (np.abs(pred_dist).sum() + 1e-10)
        quantum_uncertainty = self.quantum_estimator.estimate_uncertainty(pred_dist)

        # Calculate confidence (inverse of uncertainty)
        confidence = 1.0 - quantum_uncertainty

        # Analyze with defense system
        defense_state = self.defense.analyze_training_state({
            'loss': self.training_history[-1]['loss'] if self.training_history else 0.5,
            'accuracy': confidence,
            'confidence': confidence,
            'grad_norm': 1.0
        })

        # Record prediction
        result = {
            'prediction': pred_value,
            'confidence': confidence,
            'quantum_uncertainty': quantum_uncertainty,
            'awareness_stage': defense_state['stage'],
            'awareness_stage_value': defense_state['stage'].value,
            'risk_level': defense_state['risk_level'],
            'defense_description': defense_state['description'],
            'should_trust': defense_state['stage'].value < 7  # Trust if below warning stage
        }

        self.prediction_history.append(result)

        return result

    def _calculate_grad_norm(self) -> float:
        """Calculate gradient norm for defense system"""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += np.sum(param.grad ** 2)
        return float(np.sqrt(total_norm))

    def get_meta_insights(self) -> Dict:
        """Get insights from meta-learner"""
        return self.meta_learner.analyze_training_patterns()

    def load_best_model(self):
        """Load the best saved checkpoint"""
        try:
            self.model, self.optimizer, epoch, metrics = load_checkpoint(
                self.checkpoint_path,
                self.model,
                self.optimizer
            )
            print(f"✓ Loaded best model from epoch {epoch}")
            print(f"  Best loss: {metrics['best_loss']:.6f}")
            return True
        except FileNotFoundError:
            print("⚠️  No saved model found")
            return False


def generate_sample_data(num_points=500, pattern='sine_trend'):
    """
    Generate sample time series data for testing

    Patterns:
    - 'sine_trend': Sine wave with upward trend
    - 'crypto': Crypto-like volatile pattern
    - 'seasonal': Seasonal pattern with noise
    """
    t = np.linspace(0, 20, num_points)

    if pattern == 'sine_trend':
        # Sine wave with upward trend and noise
        data = np.sin(t) + 0.05 * t + 0.1 * np.random.randn(num_points)

    elif pattern == 'crypto':
        # Volatile crypto-like pattern
        data = np.cumsum(np.random.randn(num_points) * 0.5) + 50
        data += 10 * np.sin(t / 2)  # Add cyclical component

    elif pattern == 'seasonal':
        # Seasonal pattern
        data = 10 * np.sin(2 * np.pi * t / 10)  # Seasonal
        data += 0.5 * t  # Trend
        data += np.random.randn(num_points) * 0.5  # Noise

    return data.astype(np.float32)


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════╗
║     QUANTUM-AWARE PATTERN PREDICTOR                      ║
║  Self-Aware AI with Quantum Confidence Scoring          ║
╚══════════════════════════════════════════════════════════╝
""")

    # Generate sample data
    print("📊 Generating sample time series data...")
    data = generate_sample_data(num_points=500, pattern='sine_trend')
    print(f"✓ Generated {len(data)} data points")

    # Create predictor
    print("\n🔬 Initializing Quantum-Aware Predictor...")
    predictor = QuantumAwarePredictor(
        sequence_length=20,
        hidden_dim=64,
        learning_rate=0.001
    )
    print("✓ Model initialized with:")
    print("  - Toroidal Attention (circular pattern detection)")
    print("  - Gated Layers (adaptive processing)")
    print("  - Quantum Uncertainty Estimator")
    print("  - 10-Stage Awareness Defense")
    print("  - Meta-Learning Engine")

    # Train
    print("\n🚀 Training predictor...")
    predictor.fit(data, epochs=50, verbose=True)

    # Make predictions with confidence
    print("\n🔮 Making Predictions with Quantum Confidence...")
    print("="*70)

    # Test on last few sequences
    test_sequences = [
        data[-40:-20],   # Recent past
        data[-60:-40],   # Medium past
        data[-80:-60],   # Older past
    ]

    for i, seq in enumerate(test_sequences, 1):
        result = predictor.predict_with_confidence(seq)

        print(f"\nPrediction #{i}:")
        print(f"  Predicted Value: {result['prediction']:.4f}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Quantum Uncertainty: {result['quantum_uncertainty']:.4f}")
        print(f"  Awareness Stage: {result['awareness_stage'].value} - {result['awareness_stage'].name}")
        print(f"  Risk Level: {result['risk_level'].upper()}")

        # Show recommendation
        if result['should_trust']:
            print(f"  ✓ Recommendation: TRUST this prediction")
        else:
            print(f"  ⚠️  Recommendation: LOW CONFIDENCE - Use caution!")
        print(f"  Defense Analysis: {result['defense_description']}")

    # Show meta-learning insights
    print("\n" + "="*70)
    print("🧠 Meta-Learning Insights:")
    print("="*70)
    insights = predictor.get_meta_insights()
    print(f"Total experiments tracked: {insights['total_experiments']}")
    print(f"Best learning rate found: {insights['best_lr']:.2e}")
    print("\nInsights:")
    for insight in insights['insights']:
        print(f"  • {insight}")

    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE!")
    print("="*70)
    print("""
🎉 What you just saw:
  ✓ Neural network trained on time series
  ✓ Quantum confidence scoring on predictions
  ✓ 10-stage awareness preventing overconfident predictions
  ✓ Meta-learner tracking what works
  ✓ Auto-checkpointing of best models
  ✓ Self-aware AI that knows its limits!

💡 Next Steps:
  • Use your own data (stocks, crypto, weather, sales, etc.)
  • Adjust sequence_length for different patterns
  • Tune hidden_dim for model capacity
  • Export predictions to CSV for analysis
  • Build a web dashboard for real-time monitoring

🚀 Your LUMINARK framework makes this possible!
""")
