#!/usr/bin/env python3
"""
Comprehensive unit tests for LUMINARK framework
Run: python tests/test_framework.py
"""
import numpy as np
import sys
sys.path.insert(0, '/home/user/LUMINARK')

def test_tensor_autograd():
    """Test automatic differentiation"""
    from luminark.core import Tensor

    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)
    y = x @ w
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "x gradient not computed"
    assert w.grad is not None, "w gradient not computed"
    print("✅ test_tensor_autograd: PASSED")

def test_nn_forward():
    """Test neural network forward pass"""
    from luminark.nn import Linear, ReLU, Sequential
    from luminark.core import Tensor

    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    )

    x = Tensor(np.random.randn(2, 10).astype(np.float32))
    output = model(x)

    assert output.data.shape == (2, 5), f"Wrong output shape: {output.data.shape}"
    print("✅ test_nn_forward: PASSED")

def test_optimizer_step():
    """Test optimizer parameter updates"""
    from luminark.nn import Linear
    from luminark.optim import Adam
    from luminark.core import Tensor

    layer = Linear(5, 3)
    optimizer = Adam(layer.parameters(), lr=0.01)

    # Forward and backward
    x = Tensor(np.random.randn(2, 5).astype(np.float32), requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = layer.weight.data.copy()

    # Optimizer step
    optimizer.step()

    # Weight should have changed
    assert not np.allclose(original_weight, layer.weight.data), "Weights didn't update"
    print("✅ test_optimizer_step: PASSED")

def test_scheduler():
    """Test learning rate scheduler"""
    from luminark.nn import Linear
    from luminark.optim import Adam, CosineAnnealingLR

    layer = Linear(5, 3)
    optimizer = Adam(layer.parameters(), lr=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)

    initial_lr = optimizer.lr

    # Step multiple times
    for _ in range(5):
        scheduler.step()

    assert optimizer.lr < initial_lr, "LR didn't decrease"
    print("✅ test_scheduler: PASSED")

def test_checkpoint():
    """Test checkpointing save/load"""
    from luminark.nn import Linear, Sequential
    from luminark.optim import Adam
    from luminark.io import save_checkpoint, load_checkpoint
    import os

    # Create model
    model = Sequential(Linear(10, 5), Linear(5, 2))
    optimizer = Adam(model.parameters(), lr=0.01)

    # Save checkpoint
    test_path = '/tmp/test_checkpoint.pkl'
    save_checkpoint(model, optimizer, epoch=10,
                   metrics={'acc': 0.95}, path=test_path)

    # Create new model and load
    new_model = Sequential(Linear(10, 5), Linear(5, 2))
    new_optimizer = Adam(new_model.parameters(), lr=0.001)

    loaded_model, loaded_opt, epoch, metrics = load_checkpoint(
        test_path, new_model, new_optimizer
    )

    assert epoch == 10, f"Wrong epoch: {epoch}"
    assert metrics['acc'] == 0.95, f"Wrong accuracy: {metrics['acc']}"
    assert loaded_opt.lr == 0.01, f"LR not restored: {loaded_opt.lr}"

    # Cleanup
    os.remove(test_path)
    print("✅ test_checkpoint: PASSED")

def test_data_loader():
    """Test data loading"""
    from luminark.data import MNISTDigits, DataLoader

    dataset = MNISTDigits(train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    batch = next(iter(loader))
    data, labels = batch

    assert data.shape[0] == 32, f"Wrong batch size: {data.shape[0]}"
    assert labels.shape[0] == 32, f"Wrong label batch size: {labels.shape[0]}"
    assert data.shape[1] == 64, f"Wrong input dim: {data.shape[1]}"
    print("✅ test_data_loader: PASSED")

def test_quantum_uncertainty():
    """Test quantum uncertainty estimator"""
    try:
        from luminark.core.quantum import QuantumUncertaintyEstimator

        estimator = QuantumUncertaintyEstimator(num_qubits=3)
        predictions = np.array([0.8, 0.1, 0.1])
        uncertainty = estimator.estimate_uncertainty(predictions)

        assert isinstance(uncertainty, (int, float)), f"Wrong uncertainty type: {type(uncertainty)}"
        assert 0 <= uncertainty <= 1, f"Uncertainty out of range: {uncertainty}"
        print("✅ test_quantum_uncertainty: PASSED")
    except ImportError:
        print("⚠️  test_quantum_uncertainty: SKIPPED (Qiskit not available)")

def test_defense_system():
    """Test 10-stage defense system"""
    from luminark.monitoring.enhanced_defense import EnhancedDefenseSystem

    defense = EnhancedDefenseSystem()

    # Test normal state
    result = defense.analyze_training_state({
        'loss': 0.5,
        'accuracy': 0.85,
        'grad_norm': 1.0
    })

    assert 'stage' in result, "Missing stage"
    assert 'risk_level' in result, "Missing risk_level"
    assert 'metrics' in result, "Missing metrics"
    assert 'stability' in result['metrics'], "Missing stability in metrics"
    print("✅ test_defense_system: PASSED")

def test_advanced_layers():
    """Test advanced layer functionality"""
    from luminark.nn.advanced_layers import GatedLinear, ToroidalAttention
    from luminark.core import Tensor

    # Test GatedLinear
    gated = GatedLinear(10, 10)
    x = Tensor(np.random.randn(2, 10).astype(np.float32))
    output = gated(x)
    assert output.data.shape == (2, 10), f"Wrong GatedLinear output: {output.data.shape}"

    # Test ToroidalAttention
    attention = ToroidalAttention(10, window_size=3)
    x = Tensor(np.random.randn(2, 5, 10).astype(np.float32))  # (batch, seq_len, hidden)
    output = attention(x)
    assert output.data.shape == (2, 5, 10), f"Wrong attention output: {output.data.shape}"

    print("✅ test_advanced_layers: PASSED")

def test_associative_memory():
    """Test associative memory system"""
    from luminark.memory.associative_memory import AssociativeMemory

    memory = AssociativeMemory(capacity=100)

    # Store experience
    exp1 = {'state': [1, 2, 3], 'action': 0, 'reward': 1.0}
    memory.store(exp1, tags=['success'])

    # Retrieve using replay_batch
    samples = memory.replay_batch(batch_size=1)
    assert len(samples) == 1, f"Wrong sample count: {len(samples)}"

    print("✅ test_associative_memory: PASSED")

def test_loss_functions():
    """Test loss function implementations"""
    from luminark.nn import CrossEntropyLoss, MSELoss, BCELoss
    from luminark.core import Tensor
    import numpy as np

    # Test CrossEntropyLoss
    ce_loss = CrossEntropyLoss()
    pred = Tensor(np.random.randn(4, 10).astype(np.float32))
    target = np.array([1, 3, 5, 7])
    loss = ce_loss(pred, target)
    assert loss.data.shape == (), "CE loss should be scalar"

    # Test MSELoss
    mse_loss = MSELoss()
    pred = Tensor(np.random.randn(4, 5).astype(np.float32))
    target = Tensor(np.random.randn(4, 5).astype(np.float32))
    loss = mse_loss(pred, target)
    assert loss.data.shape == (), "MSE loss should be scalar"

    print("✅ test_loss_functions: PASSED")

if __name__ == '__main__':
    print("="*70)
    print("LUMINARK Framework Unit Tests")
    print("="*70)
    print()

    tests = [
        test_tensor_autograd,
        test_nn_forward,
        test_optimizer_step,
        test_scheduler,
        test_checkpoint,
        test_data_loader,
        test_quantum_uncertainty,
        test_defense_system,
        test_advanced_layers,
        test_associative_memory,
        test_loss_functions,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__}: FAILED - {e}")
            failed += 1
        except Exception as e:
            if "SKIPPED" in str(e):
                skipped += 1
            else:
                print(f"❌ {test.__name__}: ERROR - {e}")
                failed += 1

    print()
    print("="*70)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
    print("="*70)
