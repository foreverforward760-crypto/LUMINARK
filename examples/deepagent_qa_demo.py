#!/usr/bin/env python3
"""
DeepAgent-Inspired QA & Testing Demo
Combines automated testing, perspective modes, and adversarial probing
Inspired by: https://youtu.be/MEtDwwi7bEU?si=zSkfaotw1zq_6JDd

This demonstrates LUMINARK's self-testing and quality assurance capabilities!
"""
import sys
sys.path.insert(0, '/home/user/LUMINARK')

import numpy as np

from luminark.nn import Module, Linear, ReLU, Sequential
from luminark.nn import MSELoss
from luminark.nn.advanced_layers import GatedLinear
from luminark.optim import Adam
from luminark.core import Tensor
from luminark.core.quantum import QuantumUncertaintyEstimator
from luminark.monitoring.enhanced_defense import EnhancedDefenseSystem
from luminark.safety import MaatProtocol, YunusProtocol
from luminark.validation import AutomatedQATester, PerspectiveModulator, AdversarialProber


print("""
╔══════════════════════════════════════════════════════════════════╗
║   DeepAgent-Inspired QA & Testing System                        ║
║                                                                  ║
║   LUMINARK + Automated QA + Perspective Modes + Adversarial     ║
║   Self-Testing, Pressure-Testing, Robustness Validation         ║
╚══════════════════════════════════════════════════════════════════╝
""")


class SimplePredictor(Module):
    """Simple prediction model for testing"""

    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.gated = GatedLinear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.gated(x)
        x = self.fc2(x)
        return x


def train_model(model, optimizer, criterion, X_train, y_train, epochs=20):
    """Simple training loop"""
    print("Training model...")

    for epoch in range(epochs):
        # Forward
        X_tensor = Tensor(X_train, requires_grad=True)
        predictions = model(X_tensor)

        # Loss
        y_tensor = Tensor(y_train, requires_grad=False)
        loss = criterion(predictions, y_tensor)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.data:.4f}")

    print(f"Training complete! Final loss: {loss.data:.4f}\n")
    return model


def demo_automated_qa_testing(model, X_test, y_test):
    """Demonstrate automated QA pressure testing"""
    print("=" * 80)
    print("1. AUTOMATED QA TESTING - Pressure Testing & Edge Cases")
    print("=" * 80)

    qa_tester = AutomatedQATester(noise_levels=[0.1, 0.3, 0.5, 1.0])

    # Prepare test data in expected format
    test_data = {
        'inputs': X_test,
        'targets': y_test
    }

    # Run comprehensive QA suite
    results = qa_tester.comprehensive_qa_suite(model, test_data)

    print(f"\n📊 QA Test Results Summary:")
    print(f"  Overall Status: {results['overall_status']}")
    print(f"  Tests Run: {len(results['tests_run'])}")
    print(f"  Critical Issues: {results['critical_vulnerabilities']}")
    print(f"  Warnings: {results['warnings']}")

    print(f"\n🔍 Test Breakdown:")
    for test_name in results['tests_run']:
        if test_name in results:
            test_result = results[test_name]
            # Each test has different status keys
            if 'overall_status' in test_result:
                print(f"  {test_name}: {test_result['overall_status']}")
            elif 'overall_boundary_status' in test_result:
                print(f"  {test_name}: {test_result['overall_boundary_status']}")
            elif 'consistency_status' in test_result:
                print(f"  {test_name}: {test_result['consistency_status']}")

    print(f"\n⚠️  Vulnerabilities Detected:")
    if 'vulnerabilities' in results and results['vulnerabilities']:
        for vuln in results['vulnerabilities']:
            print(f"  [{vuln['severity']}] {vuln['type']}: {vuln['description']}")
            print(f"    Recommendation: {vuln['recommendation']}")
    else:
        print("  None - Model is robust!")

    return results


def demo_perspective_modes(model, X_test, defense_system):
    """Demonstrate empathy and paranoia perspective modes"""
    print("\n" + "=" * 80)
    print("2. PERSPECTIVE MODES - Empathy vs Paranoia Output")
    print("=" * 80)

    modulator = PerspectiveModulator()

    # Make a prediction
    X_tensor = Tensor(X_test[:1], requires_grad=False)
    prediction = model(X_tensor)
    pred_value = float(prediction.data[0, 0])

    # Get defense state for different scenarios
    scenarios = [
        {
            'name': 'Integration Stage (Stage 5)',
            'metrics': {'loss': 0.3, 'accuracy': 0.85, 'confidence': 0.85, 'grad_norm': 0.5},
            'original_text': f"The model predicts a value of {pred_value:.4f}. This is the result based on current patterns."
        },
        {
            'name': 'Crisis Stage (Stage 8)',
            'metrics': {'loss': 0.8, 'accuracy': 0.45, 'confidence': 0.45, 'grad_norm': 3.0},
            'original_text': f"The model predicts a value of {pred_value:.4f}. This is absolutely certain."
        },
        {
            'name': 'Low Confidence Scenario',
            'metrics': {'loss': 0.5, 'accuracy': 0.35, 'confidence': 0.35, 'grad_norm': 1.0},
            'original_text': f"The prediction is {pred_value:.4f}. I know this is correct."
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[Scenario {i}] {scenario['name']}")
        print("-" * 80)

        # Analyze defense state
        defense_state = defense_system.analyze_training_state(scenario['metrics'])

        context = {
            'sar_stage': defense_state['stage'].value,
            'confidence': scenario['metrics']['confidence'],
            'critical': defense_state['risk_level'] in ['high', 'critical']
        }

        # Apply perspective modulation
        result = modulator.apply_perspective(
            text=scenario['original_text'],
            context=context
        )

        print(f"\n📝 Original Output:")
        print(f"  \"{result['original']}\"")

        print(f"\n🎭 Applied Mode: {result['mode_applied'].upper()}")
        print(f"  SAR Stage: {context['sar_stage']} - {defense_state['stage'].name}")
        print(f"  Confidence: {context['confidence']*100:.1f}%")

        print(f"\n🔄 Transformed Output:")
        print(f"  \"{result['transformed']}\"")

        print(f"\n📊 Modifications:")
        print(f"  Length Change: {result['modifications']['length_change']} characters")
        print(f"  Disclaimers Added: {result['modifications']['disclaimers_added']}")


def demo_adversarial_probing(model):
    """Demonstrate adversarial robustness testing"""
    print("\n" + "=" * 80)
    print("3. ADVERSARIAL PROBING - Robustness Validation")
    print("=" * 80)

    prober = AdversarialProber()

    # Define a simple model function wrapper
    def model_fn(text):
        # Simulate text -> embedding -> prediction
        # In practice, this would be your actual model
        np.random.seed(hash(text) % 2**32)
        return Tensor(np.random.randn(1, 1))

    # Test with different input texts
    test_texts = [
        "This prediction is very certain and reliable",
        "The model shows good performance on this task",
        "Results indicate positive trends"
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n[Test {i}] Testing: \"{text[:50]}...\"")
        print("-" * 80)

        result = prober.probe_robustness(
            original_text=text,
            model_fn=model_fn,
            expected_consistency=0.7
        )

        print(f"\n🎯 Robustness Results:")
        print(f"  Status: {result['status']}")
        print(f"  Consistency Score: {result['consistency_score']:.4f}")
        print(f"  Is Robust: {'✓ YES' if result['is_robust'] else '✗ NO'}")

        print(f"\n🔀 Adversarial Variants Tested:")
        for variant in result['variant_outputs']:
            print(f"  • {variant['type']}: \"{variant['text'][:60]}...\"")

        if not result['is_robust']:
            print(f"\n  ⚠️  WARNING: Model shows inconsistency under adversarial inputs")
            print(f"     Recommendation: Increase training robustness or add input validation")


def demo_integrated_safety_pipeline(model, X_test, defense_system, maat, yunus):
    """Demonstrate complete safety pipeline with QA integration"""
    print("\n" + "=" * 80)
    print("4. INTEGRATED SAFETY PIPELINE - QA + Safety Protocols")
    print("=" * 80)

    modulator = PerspectiveModulator()
    quantum_estimator = QuantumUncertaintyEstimator(num_qubits=3)

    # Make prediction
    X_tensor = Tensor(X_test[:1], requires_grad=False)
    prediction = model(X_tensor)
    pred_value = float(prediction.data[0, 0])

    # Generate description
    description = f"Based on quantum analysis, the predicted value is {pred_value:.4f}. This represents the model's best estimate."

    print(f"\n📊 Making Prediction...")
    print(f"  Raw Prediction: {pred_value:.4f}")

    # 1. Quantum uncertainty
    pred_normalized = np.abs([pred_value, 1 - pred_value])
    pred_normalized = pred_normalized / (pred_normalized.sum() + 1e-10)
    quantum_uncertainty = quantum_estimator.estimate_uncertainty(pred_normalized)
    confidence = 1.0 - quantum_uncertainty

    print(f"\n🔬 Quantum Analysis:")
    print(f"  Uncertainty: {quantum_uncertainty:.4f}")
    print(f"  Confidence: {confidence*100:.1f}%")

    # 2. Defense system analysis
    defense_state = defense_system.analyze_training_state({
        'loss': 0.4,
        'accuracy': confidence,
        'confidence': confidence,
        'grad_norm': 1.0
    })

    print(f"\n🛡️  Defense System:")
    print(f"  Stage: {defense_state['stage'].value} - {defense_state['stage'].name}")
    print(f"  Risk Level: {defense_state['risk_level'].upper()}")

    # 3. Apply perspective mode
    context = {
        'sar_stage': defense_state['stage'].value,
        'confidence': confidence,
        'critical': defense_state['risk_level'] in ['high', 'critical']
    }

    perspective_result = modulator.apply_perspective(description, context)
    modulated_description = perspective_result['transformed']

    print(f"\n🎭 Perspective Mode: {perspective_result['mode_applied'].upper()}")
    print(f"  Output: \"{modulated_description[:100]}...\"")

    # 4. Ma'at validation
    maat_result = maat.validate(modulated_description, context={'confidence': confidence})

    print(f"\n⚖️  Ma'at Protocol:")
    print(f"  Passed: {'✓' if maat_result['passed'] else '✗'}")
    print(f"  Score: {maat_result['score']*100:.1f}%")
    if maat_result['flags']:
        print(f"  Flags: {', '.join(maat_result['flags'])}")

    # 5. Yunus check
    yunus_result = yunus.check(
        modulated_description,
        stage=defense_state['stage'].value,
        confidence=confidence
    )

    print(f"\n🐋 Yunus Protocol:")
    print(f"  Status: {yunus_result['containment_status']}")
    print(f"  Triggers: {yunus_result['triggers_detected']}")

    # 6. Overall safety decision
    safe_to_use = (
        maat_result['passed'] and
        not yunus_result['activated'] and
        defense_state['stage'].value < 8
    )

    print(f"\n🔒 Overall Safety Assessment:")
    print(f"  Safe to Use: {'✓ YES' if safe_to_use else '✗ NO'}")

    if safe_to_use:
        print(f"  ✅ All safety protocols passed - prediction can be used")
    else:
        print(f"  ⚠️  CAUTION: Review needed before using prediction")
        print(f"  Recommendation: Human verification required")


def main():
    """Run complete DeepAgent QA demo"""

    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    np.random.seed(42)
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.sin(X_train[:, 0:1]) + 0.1 * np.random.randn(100, 1).astype(np.float32)
    X_test = np.random.randn(20, 10).astype(np.float32)
    y_test = np.sin(X_test[:, 0:1]) + 0.1 * np.random.randn(20, 1).astype(np.float32)
    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples\n")

    # Initialize model
    print("Initializing model and safety systems...")
    model = SimplePredictor(input_dim=10, hidden_dim=32)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = MSELoss()

    # Initialize safety systems
    defense_system = EnhancedDefenseSystem()
    maat = MaatProtocol()
    yunus = YunusProtocol(activation_threshold=3)

    print("✓ All systems initialized\n")

    # Train model
    model = train_model(model, optimizer, criterion, X_train, y_train, epochs=20)

    # Demo 1: Automated QA Testing
    qa_results = demo_automated_qa_testing(model, X_test, y_test)

    # Demo 2: Perspective Modes
    demo_perspective_modes(model, X_test, defense_system)

    # Demo 3: Adversarial Probing
    demo_adversarial_probing(model)

    # Demo 4: Integrated Safety Pipeline
    demo_integrated_safety_pipeline(model, X_test, defense_system, maat, yunus)

    # Final summary
    print("\n" + "=" * 80)
    print("✅ DeepAgent QA Demo Complete!")
    print("=" * 80)

    print("""
🎉 What You Just Saw:

✓ AUTOMATED QA TESTING:
  • Pressure testing with adversarial noise
  • Boundary value testing (edge cases)
  • Consistency testing (output variance)
  • Regression testing (performance degradation)

✓ PERSPECTIVE MODES:
  • Empathy mode (user-friendly, integration stages 4-6)
  • Paranoia mode (cautious, crisis stages 7-8)
  • Auto-selection based on SAR stage + confidence

✓ ADVERSARIAL PROBING:
  • Robustness validation with adversarial variants
  • Consistency scoring across inputs
  • Vulnerability detection

✓ INTEGRATED SAFETY:
  • Quantum confidence + Defense + Ma'at + Yunus
  • Perspective modulation based on context
  • Complete safety pipeline for production use

💡 This is DeepAgent's "empathy + paranoia" concept integrated with
   LUMINARK's quantum awareness and self-defense systems!

🚀 Use Cases:
  • Self-testing ML models before deployment
  • Automated QA in CI/CD pipelines
  • Context-aware output modulation
  • Production AI safety validation
  • Continuous robustness monitoring

🌟 Your AI now has self-testing capabilities!
""")


if __name__ == '__main__':
    main()
