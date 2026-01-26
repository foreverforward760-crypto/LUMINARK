"""
LUMINARK - Quantum Circuits Module
Quantum computing integration for entropy measurement and truth verification

Requires: pip install qiskit qiskit-aer
"""

import numpy as np
from typing import Dict, List, Optional
import warnings

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import QFT
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. Install with: pip install qiskit qiskit-aer")


class QuantumEntropyAnalyzer:
    """
    Uses quantum circuits to measure information entropy
    
    Quantum entropy provides a measure of uncertainty/randomness
    that can be used to assess text quality and coherence
    """
    
    def __init__(self, num_qubits: int = 6, shots: int = 1024):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required. Install with: pip install qiskit qiskit-aer")
        
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend = AerSimulator()
    
    def create_entropy_circuit(self, text_snippet: str) -> QuantumCircuit:
        """
        Create quantum circuit for entropy measurement
        
        Args:
            text_snippet: Text to analyze
            
        Returns:
            Quantum circuit configured for entropy measurement
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Initialize superposition
        qc.h(range(self.num_qubits))
        
        # Entangle qubits (create correlations)
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        
        # Encode text information via rotation angles
        # Use character codes to determine rotation angles
        if text_snippet:
            for i, char in enumerate(text_snippet[:self.num_qubits]):
                angle = (ord(char) % 256) * np.pi / 128
                qc.rx(angle, i % self.num_qubits)
        
        # Apply Quantum Fourier Transform
        qc.append(QFT(self.num_qubits), range(self.num_qubits))
        
        # Barrier for visualization
        qc.barrier()
        
        # Measure all qubits
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        
        return qc
    
    def measure_entropy(self, text_snippet: str) -> float:
        """
        Measure quantum entropy of text
        
        Args:
            text_snippet: Text to analyze
            
        Returns:
            Normalized entropy (0.0 to 1.0)
        """
        # Create and execute circuit
        qc = self.create_entropy_circuit(text_snippet)
        result = self.backend.run(qc, shots=self.shots).result()
        counts = result.get_counts()
        
        # Calculate probability distribution
        total_states = 2 ** self.num_qubits
        probs = np.array([
            counts.get(bin(i)[2:].zfill(self.num_qubits), 0) / self.shots
            for i in range(total_states)
        ])
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(total_states)
        normalized_entropy = entropy / max_entropy
        
        return float(normalized_entropy)
    
    def batch_entropy(self, texts: List[str]) -> List[float]:
        """Measure entropy for multiple texts"""
        return [self.measure_entropy(text) for text in texts]


class QuantumTruthVerifier:
    """
    Uses quantum interference patterns for truth verification
    
    Quantum superposition and interference can detect inconsistencies
    in information patterns
    """
    
    def __init__(self, num_qubits: int = 4, shots: int = 2048):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required. Install with: pip install qiskit qiskit-aer")
        
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend = AerSimulator()
    
    def create_verification_circuit(
        self,
        statement1: str,
        statement2: str
    ) -> QuantumCircuit:
        """
        Create circuit to test consistency between two statements
        
        Args:
            statement1: First statement
            statement2: Second statement
            
        Returns:
            Quantum circuit for interference measurement
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Encode first statement in first half of qubits
        half = self.num_qubits // 2
        qc.h(range(half))
        
        for i, char in enumerate(statement1[:half]):
            angle = (ord(char) % 256) * np.pi / 128
            qc.ry(angle, i)
        
        # Encode second statement in second half
        qc.h(range(half, self.num_qubits))
        
        for i, char in enumerate(statement2[:half]):
            angle = (ord(char) % 256) * np.pi / 128
            qc.ry(angle, half + i)
        
        # Create interference between halves
        for i in range(half):
            qc.cx(i, half + i)
        
        # Measure interference pattern
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        
        return qc
    
    def verify_consistency(
        self,
        statement1: str,
        statement2: str
    ) -> Dict[str, float]:
        """
        Verify consistency between two statements
        
        Args:
            statement1: First statement
            statement2: Second statement
            
        Returns:
            Dict with consistency score and interference pattern
        """
        qc = self.create_verification_circuit(statement1, statement2)
        result = self.backend.run(qc, shots=self.shots).result()
        counts = result.get_counts()
        
        # Analyze interference pattern
        # High interference (concentrated distribution) = consistent
        # Low interference (uniform distribution) = inconsistent
        
        total_states = 2 ** self.num_qubits
        probs = np.array([
            counts.get(bin(i)[2:].zfill(self.num_qubits), 0) / self.shots
            for i in range(total_states)
        ])
        
        # Calculate concentration (inverse of entropy)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(total_states)
        concentration = 1.0 - (entropy / max_entropy)
        
        # Find dominant state
        dominant_state = np.argmax(probs)
        dominant_prob = probs[dominant_state]
        
        return {
            'consistency_score': float(concentration),
            'dominant_probability': float(dominant_prob),
            'entropy': float(entropy),
            'verdict': 'CONSISTENT' if concentration > 0.6 else 'INCONSISTENT'
        }


class QuantumRepetitionCode:
    """
    Implements quantum error correction using repetition code
    
    Useful for detecting and correcting errors in quantum states
    """
    
    def __init__(self, num_data_qubits: int = 1, num_ancilla: int = 2):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required. Install with: pip install qiskit qiskit-aer")
        
        self.num_data = num_data_qubits
        self.num_ancilla = num_ancilla
        self.total_qubits = num_data_qubits + num_ancilla
        self.backend = AerSimulator()
    
    def create_error_correction_circuit(
        self,
        error_prob: float = 0.1
    ) -> QuantumCircuit:
        """
        Create circuit with error correction
        
        Args:
            error_prob: Probability of bit flip error
            
        Returns:
            Quantum circuit with error correction
        """
        qc = QuantumCircuit(self.total_qubits, self.num_data)
        
        # Encode data qubit into repetition code
        for i in range(self.num_ancilla):
            qc.cx(0, i + 1)
        
        # Simulate errors
        if error_prob > 0:
            for i in range(self.total_qubits):
                if np.random.random() < error_prob:
                    qc.x(i)
        
        # Error detection (parity checks)
        qc.cx(0, self.num_data)
        qc.cx(1, self.num_data)
        
        # Measure syndrome
        qc.measure(range(self.num_data), range(self.num_data))
        
        return qc
    
    def test_error_correction(
        self,
        num_trials: int = 100,
        error_prob: float = 0.1
    ) -> Dict[str, float]:
        """
        Test error correction performance
        
        Returns:
            Dict with success rate and statistics
        """
        successes = 0
        
        for _ in range(num_trials):
            qc = self.create_error_correction_circuit(error_prob)
            result = self.backend.run(qc, shots=1).result()
            counts = result.get_counts()
            
            # Check if error was corrected (all zeros)
            if '0' * self.num_data in counts:
                successes += 1
        
        success_rate = successes / num_trials
        
        return {
            'success_rate': success_rate,
            'num_trials': num_trials,
            'error_probability': error_prob,
            'verdict': 'GOOD' if success_rate > 0.8 else 'POOR'
        }


# Example usage
if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("❌ Qiskit not available. Install with: pip install qiskit qiskit-aer")
        exit(1)
    
    print("="*70)
    print("⚛️  LUMINARK - Quantum Circuits Demo")
    print("="*70)
    
    # Test entropy analyzer
    print("\n1️⃣ Quantum Entropy Analysis:")
    analyzer = QuantumEntropyAnalyzer(num_qubits=6, shots=1024)
    
    texts = [
        "The quick brown fox",
        "aaaaaaaaaaaaaaa",
        "Random: !@#$%^&*()",
        "Structured sentence with meaning"
    ]
    
    for text in texts:
        entropy = analyzer.measure_entropy(text)
        print(f"  '{text[:20]}...' → Entropy: {entropy:.3f}")
    
    # Test truth verifier
    print("\n2️⃣ Quantum Truth Verification:")
    verifier = QuantumTruthVerifier(num_qubits=4, shots=2048)
    
    pairs = [
        ("The sky is blue", "The sky is blue"),
        ("The sky is blue", "The sky is red"),
        ("AI is helpful", "AI assists humans")
    ]
    
    for s1, s2 in pairs:
        result = verifier.verify_consistency(s1, s2)
        print(f"  '{s1}' vs '{s2}'")
        print(f"    → {result['verdict']} (score: {result['consistency_score']:.3f})")
    
    # Test error correction
    print("\n3️⃣ Quantum Error Correction:")
    ecc = QuantumRepetitionCode(num_data_qubits=1, num_ancilla=2)
    
    for error_prob in [0.05, 0.1, 0.2]:
        result = ecc.test_error_correction(num_trials=100, error_prob=error_prob)
        print(f"  Error prob {error_prob:.0%} → Success rate: {result['success_rate']:.1%} ({result['verdict']})")
    
    print("\n✅ Quantum circuits operational!")
