"""
LUMINARK - Quantum Module
Experimental Qiskit circuits for character-encoded measurement distributions and quantum repetition-code demonstrations

Components:
- circuits: Circuit-output entropy experiments, pattern concentration, and repetition-code demonstrations
"""

from .circuits import (
    QuantumEntropyAnalyzer,
    QuantumTruthVerifier,
    QuantumRepetitionCode,
    QISKIT_AVAILABLE
)

__all__ = [
    'QuantumEntropyAnalyzer',
    'QuantumTruthVerifier',
    'QuantumRepetitionCode',
    'QISKIT_AVAILABLE'
]
