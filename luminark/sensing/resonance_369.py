"""
369 Resonance Detection - Tesla's Key Numbers
"If you only knew the magnificence of the 3, 6 and 9,
 then you would have a key to the universe." - Nikola Tesla
"""
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class ResonanceType(Enum):
    """Types of 369 resonance"""
    HARMONIC_3 = 3      # Base frequency
    HARMONIC_6 = 6      # Double frequency
    HARMONIC_9 = 9      # Triple frequency
    VORTEX_369 = 369    # Full vortex pattern
    DIGITAL_ROOT = 0    # Digital root analysis


@dataclass
class ResonancePattern:
    """Detected 369 resonance pattern"""
    type: ResonanceType
    strength: float
    frequency: float
    phase: float
    harmonics: List[int]
    properties: Dict[str, Any]


class Resonance369Detector:
    """
    Detect Tesla's 3-6-9 patterns in data

    Key concepts:
    - 3, 6, 9 are vortex mathematics keys
    - Digital root reduction finds these patterns
    - Harmonic relationships reveal structure
    - Doubling circuit: 1→2→4→8→7→5→1 (never touches 3,6,9)
    - 3,6,9 form separate "control" circuit
    """

    def __init__(self):
        self.detected_patterns = []

        # Vortex mathematics patterns
        self.doubling_circuit = [1, 2, 4, 8, 7, 5]  # Never touches 3,6,9
        self.control_circuit = [3, 6, 9]  # The "control" numbers

        print("🌀 369 Resonance Detector initialized")
        print("   Vortex Mathematics: 3-6-9 pattern detection")
        print("   Doubling Circuit: 1→2→4→8→7→5→1")
        print("   Control Circuit: 3→6→9→3")

    @staticmethod
    def digital_root(n: int) -> int:
        """Calculate digital root (reduce to single digit 1-9)"""
        if n == 0:
            return 0

        n = abs(n)
        return 1 + ((n - 1) % 9)

    @staticmethod
    def is_369_number(n: int) -> bool:
        """Check if number reduces to 3, 6, or 9"""
        root = Resonance369Detector.digital_root(n)
        return root in [3, 6, 9]

    @staticmethod
    def vortex_sequence(start: int, length: int) -> List[int]:
        """Generate vortex mathematics sequence"""
        sequence = []
        current = start

        for _ in range(length):
            sequence.append(current)
            current = (current * 2) % 10 if current != 0 else 0
            if current == 0:
                current = 1

        return [Resonance369Detector.digital_root(n) for n in sequence]

    def detect_369_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect 369 resonance patterns in numerical data

        Returns comprehensive 369 analysis
        """

        data_flat = data.flatten().astype(int)

        # 1. Digital root analysis
        digital_roots = [self.digital_root(int(abs(n))) for n in data_flat]

        # Count 3, 6, 9 occurrences
        count_3 = sum(1 for r in digital_roots if r == 3)
        count_6 = sum(1 for r in digital_roots if r == 6)
        count_9 = sum(1 for r in digital_roots if r == 9)
        total_369 = count_3 + count_6 + count_9

        # 2. Sequence pattern matching
        sequence_matches = self._detect_sequence_patterns(digital_roots)

        # 3. Harmonic analysis
        harmonics = self._analyze_harmonics(data_flat)

        # 4. Vortex mathematics check
        vortex_match = self._check_vortex_pattern(digital_roots)

        # 5. Overall resonance strength
        resonance_strength = self._calculate_resonance_strength(
            total_369, len(data_flat), sequence_matches, harmonics, vortex_match
        )

        # 6. Frequency domain analysis
        frequency_analysis = self._frequency_domain_369(data)

        return {
            'digital_root_distribution': {
                '3': count_3,
                '6': count_6,
                '9': count_9,
                'other': len(digital_roots) - total_369
            },
            '369_percentage': (total_369 / len(data_flat) * 100) if data_flat.size > 0 else 0,
            'sequence_patterns': sequence_matches,
            'harmonic_analysis': harmonics,
            'vortex_match': vortex_match,
            'resonance_strength': resonance_strength,
            'frequency_domain': frequency_analysis,
            'tesla_signature': self._detect_tesla_signature(digital_roots, harmonics),
            'interpretation': self._interpret_369_resonance(resonance_strength, vortex_match)
        }

    def _detect_sequence_patterns(self, digital_roots: List[int]) -> Dict[str, Any]:
        """Detect sequential 369 patterns"""

        # Look for 3→6→9 sequences
        sequence_369 = 0
        sequence_963 = 0  # Reverse
        sequence_396 = 0  # Cycle start at 3

        for i in range(len(digital_roots) - 2):
            triple = digital_roots[i:i+3]

            if triple == [3, 6, 9]:
                sequence_369 += 1
            elif triple == [9, 6, 3]:
                sequence_963 += 1
            elif triple == [3, 9, 6]:
                sequence_396 += 1

        # Look for doubling circuit: 1→2→4→8→7→5
        doubling_sequences = 0
        for i in range(len(digital_roots) - 5):
            sextuple = digital_roots[i:i+6]
            if sextuple == self.doubling_circuit:
                doubling_sequences += 1

        return {
            '3-6-9_forward': sequence_369,
            '9-6-3_reverse': sequence_963,
            '3-9-6_alternate': sequence_396,
            'doubling_circuit_matches': doubling_sequences,
            'total_sequential_patterns': sequence_369 + sequence_963 + sequence_396
        }

    def _analyze_harmonics(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze harmonic relationships (multiples of 3, 6, 9)"""

        # Count multiples
        multiples_3 = sum(1 for n in data if n % 3 == 0 and n != 0)
        multiples_6 = sum(1 for n in data if n % 6 == 0 and n != 0)
        multiples_9 = sum(1 for n in data if n % 9 == 0 and n != 0)

        # Harmonic ratios
        total = len(data)
        ratio_3 = multiples_3 / total if total > 0 else 0
        ratio_6 = multiples_6 / total if total > 0 else 0
        ratio_9 = multiples_9 / total if total > 0 else 0

        # Expected vs observed (random would be ~33% for multiples of 3)
        expected_3 = 1/3
        expected_6 = 1/6
        expected_9 = 1/9

        return {
            'multiples_of_3': multiples_3,
            'multiples_of_6': multiples_6,
            'multiples_of_9': multiples_9,
            'harmonic_ratios': {
                '3': ratio_3,
                '6': ratio_6,
                '9': ratio_9
            },
            'deviation_from_random': {
                '3': ratio_3 - expected_3,
                '6': ratio_6 - expected_6,
                '9': ratio_9 - expected_9
            },
            'harmonic_strength': (abs(ratio_3 - expected_3) +
                                 abs(ratio_6 - expected_6) +
                                 abs(ratio_9 - expected_9)) / 3
        }

    def _check_vortex_pattern(self, digital_roots: List[int]) -> Dict[str, Any]:
        """Check for vortex mathematics patterns"""

        # Check if data follows doubling circuit
        doubling_matches = sum(1 for r in digital_roots if r in self.doubling_circuit)
        control_matches = sum(1 for r in digital_roots if r in self.control_circuit)

        total = len(digital_roots)
        doubling_ratio = doubling_matches / total if total > 0 else 0
        control_ratio = control_matches / total if total > 0 else 0

        # In vortex math, 2/3 should be doubling circuit, 1/3 should be control
        expected_doubling = 2/3
        expected_control = 1/3

        vortex_coherence = 1.0 - (abs(doubling_ratio - expected_doubling) +
                                  abs(control_ratio - expected_control)) / 2

        return {
            'doubling_circuit_ratio': doubling_ratio,
            'control_circuit_ratio': control_ratio,
            'vortex_coherence': vortex_coherence,
            'follows_vortex_pattern': vortex_coherence > 0.7,
            'separation_clear': abs(doubling_ratio - control_ratio) > 0.2
        }

    def _frequency_domain_369(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze 369 patterns in frequency domain"""

        if data.size < 4:
            return {'fft_available': False}

        # FFT
        fft = np.fft.fft(data.flatten())
        frequencies = np.fft.fftfreq(len(fft))
        magnitudes = np.abs(fft)

        # Find peaks at multiples of 3, 6, 9 Hz (if interpreting as frequencies)
        peak_indices = np.argsort(magnitudes)[-10:]  # Top 10 peaks

        # Check if peaks align with 369
        peaks_at_369 = 0
        for idx in peak_indices:
            freq_val = abs(frequencies[idx] * len(data))
            if freq_val > 0:
                root = self.digital_root(int(freq_val))
                if root in [3, 6, 9]:
                    peaks_at_369 += 1

        return {
            'fft_available': True,
            'num_peaks_analyzed': len(peak_indices),
            'peaks_at_369_harmonics': peaks_at_369,
            'frequency_369_ratio': peaks_at_369 / len(peak_indices) if peak_indices.size > 0 else 0
        }

    def _calculate_resonance_strength(self, count_369: int, total: int,
                                     sequence_matches: Dict,
                                     harmonics: Dict,
                                     vortex: Dict) -> float:
        """Calculate overall 369 resonance strength (0-1)"""

        # Component 1: 369 percentage (expected random: 33%)
        pct_369 = count_369 / total if total > 0 else 0
        pct_component = min(1.0, pct_369 / 0.5)  # Normalize to 50% being max

        # Component 2: Sequential patterns
        seq_total = sequence_matches['total_sequential_patterns']
        seq_component = min(1.0, seq_total / max(1, total * 0.1))

        # Component 3: Harmonic strength
        harmonic_component = harmonics['harmonic_strength']

        # Component 4: Vortex coherence
        vortex_component = vortex['vortex_coherence']

        # Weighted average
        strength = (
            0.3 * pct_component +
            0.2 * seq_component +
            0.2 * harmonic_component +
            0.3 * vortex_component
        )

        return float(np.clip(strength, 0, 1))

    def _detect_tesla_signature(self, digital_roots: List[int],
                                harmonics: Dict) -> Dict[str, Any]:
        """Detect specific Tesla-like patterns"""

        # Tesla was obsessed with 3,6,9 and their multiples
        # Signature patterns:
        # 1. High concentration of 3,6,9
        # 2. Clear separation from doubling circuit
        # 3. Harmonic relationships

        count_3 = sum(1 for r in digital_roots if r == 3)
        count_6 = sum(1 for r in digital_roots if r == 6)
        count_9 = sum(1 for r in digital_roots if r == 9)

        # Check for balance in 3-6-9 distribution
        total_369 = count_3 + count_6 + count_9
        if total_369 > 0:
            balance_3 = count_3 / total_369
            balance_6 = count_6 / total_369
            balance_9 = count_9 / total_369

            # Perfect balance would be 1/3 each
            balance_score = 1.0 - (abs(balance_3 - 1/3) +
                                  abs(balance_6 - 1/3) +
                                  abs(balance_9 - 1/3))
        else:
            balance_score = 0

        # Check for power of 3 progression (3, 9, 27, 81, ...)
        powers_of_3 = [3**i for i in range(1, 10)]
        power_matches = sum(1 for n in digital_roots
                          if any(self.digital_root(p) == n for p in powers_of_3))

        return {
            'balance_369': balance_score,
            'power_of_3_patterns': power_matches,
            'harmonic_deviation': harmonics['harmonic_strength'],
            'tesla_signature_detected': (balance_score > 0.7 and
                                        harmonics['harmonic_strength'] > 0.3),
            'signature_strength': (balance_score + harmonics['harmonic_strength']) / 2
        }

    def _interpret_369_resonance(self, strength: float, vortex: Dict) -> str:
        """Interpret the 369 resonance"""

        if strength > 0.8 and vortex['follows_vortex_pattern']:
            return "STRONG TESLA SIGNATURE: Clear 369 vortex mathematics pattern"
        elif strength > 0.6:
            return "MODERATE 369 RESONANCE: Significant harmonic structure present"
        elif strength > 0.4:
            return "WEAK 369 PATTERNS: Some harmonic organization detected"
        elif vortex['follows_vortex_pattern']:
            return "VORTEX STRUCTURE: Doubling circuit present but weak 369"
        else:
            return "NO CLEAR 369 RESONANCE: Appears random or non-harmonic"

    def generate_369_sequence(self, length: int, pattern_type: str = 'forward') -> List[int]:
        """Generate pure 369 sequences for testing/reference"""

        if pattern_type == 'forward':
            # 3→6→9→3→6→9...
            base = [3, 6, 9]
        elif pattern_type == 'reverse':
            # 9→6→3→9→6→3...
            base = [9, 6, 3]
        elif pattern_type == 'doubling':
            # 1→2→4→8→7→5→1...
            base = self.doubling_circuit
        elif pattern_type == 'alternating':
            # 3→9→6→3→9→6...
            base = [3, 9, 6]
        else:
            base = [3, 6, 9]

        sequence = []
        while len(sequence) < length:
            sequence.extend(base)

        return sequence[:length]


if __name__ == '__main__':
    # Demo
    print("🌀 369 Resonance Detector Demo\n")

    detector = Resonance369Detector()

    # Test 1: Pure 369 sequence
    print("1. Pure 369 sequence:")
    pure_369 = np.array(detector.generate_369_sequence(30, 'forward'))
    result1 = detector.detect_369_patterns(pure_369)

    print(f"   Resonance Strength: {result1['resonance_strength']:.3f}")
    print(f"   369 Percentage: {result1['369_percentage']:.1f}%")
    print(f"   Interpretation: {result1['interpretation']}")
    print(f"   Tesla Signature: {result1['tesla_signature']['tesla_signature_detected']}")

    # Test 2: Random data
    print("\n2. Random data:")
    random_data = np.random.randint(1, 100, size=50)
    result2 = detector.detect_369_patterns(random_data)

    print(f"   Resonance Strength: {result2['resonance_strength']:.3f}")
    print(f"   369 Percentage: {result2['369_percentage']:.1f}%")
    print(f"   Interpretation: {result2['interpretation']}")

    # Test 3: Fibonacci sequence (natural 369 patterns)
    print("\n3. Fibonacci sequence:")
    from luminark.sensing.geometric_encoding import SacredGeometry
    fib = np.array(SacredGeometry.fibonacci_sequence(25))
    result3 = detector.detect_369_patterns(fib)

    print(f"   Resonance Strength: {result3['resonance_strength']:.3f}")
    print(f"   369 Percentage: {result3['369_percentage']:.1f}%")
    print(f"   Digital Root Distribution:")
    for key, val in result3['digital_root_distribution'].items():
        print(f"      {key}: {val}")
    print(f"   Vortex Pattern Match: {result3['vortex_match']['follows_vortex_pattern']}")

    # Test 4: Show digital roots
    print("\n4. Digital root examples:")
    test_numbers = [12, 27, 36, 45, 108, 144, 369]
    for n in test_numbers:
        root = detector.digital_root(n)
        is_369 = detector.is_369_number(n)
        print(f"   {n} → {root} {'✓ (369)' if is_369 else ''}")
