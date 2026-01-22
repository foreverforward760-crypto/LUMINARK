"""Core tensor operations"""
from luminark.core.tensor import Tensor, tensor

__all__ = ['Tensor', 'tensor']
# Quantum components (optional dependency)
try:
    from luminark.core.quantum import QuantumUncertaintyEstimator, QuantumPatternDetector, estimate_model_confidence
    __all__.extend(['QuantumUncertaintyEstimator', 'QuantumPatternDetector', 'estimate_model_confidence'])
except ImportError:
    pass  # Qiskit not installed
