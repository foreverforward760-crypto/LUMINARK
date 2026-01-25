"""
Geometric Encoding - Sacred Geometry for SAP Framework
Encodes patterns using Fibonacci, Golden Ratio, Platonic Solids, etc.
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SacredShape(Enum):
    """Sacred geometric shapes"""
    CIRCLE = 0              # Unity, wholeness
    VESICA_PISCIS = 1      # Intersection, creation
    TRIANGLE = 2            # Trinity, stability
    SQUARE = 3              # Earth, foundation
    PENTAGON = 4            # Life, phi ratio
    HEXAGON = 5             # Harmony, beehive
    HEPTAGON = 6            # Mysticism, 7-fold
    OCTAGON = 7             # Regeneration, infinity
    ENNEAGON = 8            # Completion, 9-fold
    TETRAHEDRON = 9         # Fire, 4 faces
    CUBE = 10               # Earth, 6 faces
    OCTAHEDRON = 11         # Air, 8 faces
    DODECAHEDRON = 12       # Ether, 12 faces
    ICOSAHEDRON = 13        # Water, 20 faces
    METATRONS_CUBE = 14     # All platonic solids
    FLOWER_OF_LIFE = 15     # Creation pattern
    SEED_OF_LIFE = 16       # Genesis pattern
    TORUS = 17              # Energy flow
    SPIRAL = 18             # Growth, evolution
    MERKABA = 19            # Star tetrahedron


@dataclass
class GeometricPattern:
    """Encoded geometric pattern"""
    shape: SacredShape
    ratio: float            # Related ratio (phi, pi, etc.)
    resonance: float        # How strongly it resonates
    dimensions: int         # 2D, 3D, etc.
    properties: Dict[str, Any]


class SacredGeometry:
    """Sacred geometry constants and calculations"""

    # Golden Ratio (Phi)
    PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895

    # Pi
    PI = np.pi

    # Fibonacci sequence
    @staticmethod
    def fibonacci(n: int) -> int:
        """Generate nth Fibonacci number"""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    @staticmethod
    def fibonacci_sequence(length: int) -> List[int]:
        """Generate Fibonacci sequence"""
        seq = []
        for i in range(length):
            seq.append(SacredGeometry.fibonacci(i))
        return seq

    @staticmethod
    def phi_spiral(theta: float, a: float = 1.0) -> Tuple[float, float]:
        """Generate point on golden spiral"""
        r = a * (SacredGeometry.PHI ** (theta / (np.pi / 2)))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return (x, y)

    @staticmethod
    def platonic_solid_vertices(shape: SacredShape) -> Optional[np.ndarray]:
        """Get vertices of Platonic solids"""
        phi = SacredGeometry.PHI

        if shape == SacredShape.TETRAHEDRON:
            # 4 vertices
            return np.array([
                [1, 1, 1],
                [1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1]
            ]) / np.sqrt(3)

        elif shape == SacredShape.CUBE:
            # 8 vertices
            return np.array([
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
            ]) / np.sqrt(3)

        elif shape == SacredShape.OCTAHEDRON:
            # 6 vertices
            return np.array([
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1]
            ])

        elif shape == SacredShape.DODECAHEDRON:
            # 20 vertices using golden ratio
            return np.array([
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
                [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
                [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
            ]) / np.sqrt(3)

        elif shape == SacredShape.ICOSAHEDRON:
            # 12 vertices using golden ratio
            return np.array([
                [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
                [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
            ]) / np.sqrt(1 + phi**2)

        return None

    @staticmethod
    def vesica_piscis(radius: float = 1.0, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Vesica Piscis (two overlapping circles)"""
        theta = np.linspace(0, 2*np.pi, num_points)

        # Circle 1 centered at origin
        x1 = radius * np.cos(theta)
        y1 = radius * np.sin(theta)

        # Circle 2 offset by radius
        x2 = radius * np.cos(theta) + radius
        y2 = radius * np.sin(theta)

        return (np.array([x1, y1]), np.array([x2, y2]))

    @staticmethod
    def flower_of_life(radius: float = 1.0, layers: int = 2) -> List[Tuple[float, float]]:
        """Generate Flower of Life pattern (overlapping circles)"""
        centers = [(0, 0)]  # Center circle

        # First layer (6 circles)
        for i in range(6):
            angle = i * np.pi / 3
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            centers.append((x, y))

        # Additional layers
        if layers > 1:
            # Second layer (12 circles)
            for i in range(12):
                angle = i * np.pi / 6
                distance = radius * np.sqrt(3)
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)
                centers.append((x, y))

        return centers


class GeometricEncoder:
    """
    Encode patterns using sacred geometry
    Maps data to geometric forms for deeper pattern recognition
    """

    def __init__(self):
        self.phi = SacredGeometry.PHI
        self.pi = SacredGeometry.PI

        # Shape resonance database
        self.shape_resonances = {}

        # Encoded patterns
        self.encoded_patterns = []

        print("📐 Geometric Encoder initialized")
        print(f"   Golden Ratio (Φ): {self.phi:.10f}")
        print(f"   Pi (π): {self.pi:.10f}")

    def encode_data(self, data: np.ndarray, method: str = 'fibonacci') -> Dict[str, Any]:
        """
        Encode numerical data using sacred geometry

        Args:
            data: Numerical array to encode
            method: Encoding method ('fibonacci', 'phi_spiral', 'platonic')

        Returns:
            Geometric encoding with pattern analysis
        """

        if method == 'fibonacci':
            return self._encode_fibonacci(data)
        elif method == 'phi_spiral':
            return self._encode_phi_spiral(data)
        elif method == 'platonic':
            return self._encode_platonic(data)
        elif method == 'fractal':
            return self._encode_fractal(data)
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def _encode_fibonacci(self, data: np.ndarray) -> Dict[str, Any]:
        """Encode using Fibonacci sequence"""
        data_flat = data.flatten()
        n = len(data_flat)

        # Generate Fibonacci sequence of same length
        fib_seq = SacredGeometry.fibonacci_sequence(n)
        fib_ratios = [fib_seq[i+1] / max(fib_seq[i], 1) for i in range(n-1)]

        # Calculate how well data follows Fibonacci pattern
        data_normalized = (data_flat - np.min(data_flat)) / (np.max(data_flat) - np.min(data_flat) + 1e-10)
        fib_normalized = np.array(fib_seq) / max(fib_seq)

        # Correlation with Fibonacci
        correlation = np.corrcoef(data_normalized, fib_normalized)[0, 1] if n > 1 else 0

        # Find phi ratio approximations in data
        data_ratios = [data_flat[i+1] / max(data_flat[i], 1e-10) for i in range(n-1)]
        phi_matches = sum(1 for r in data_ratios if abs(r - self.phi) < 0.1)

        return {
            'method': 'fibonacci',
            'fibonacci_correlation': float(correlation),
            'phi_ratio_matches': phi_matches,
            'fibonacci_resonance': float(abs(correlation)),
            'encoded_sequence': fib_seq[:10],  # First 10 for display
            'shape': SacredShape.SPIRAL,
            'properties': {
                'growth_pattern': 'fibonacci' if correlation > 0.5 else 'non-fibonacci',
                'golden_ratio_present': phi_matches > n * 0.1
            }
        }

    def _encode_phi_spiral(self, data: np.ndarray) -> Dict[str, Any]:
        """Encode using golden spiral"""
        data_flat = data.flatten()
        n = len(data_flat)

        # Generate spiral coordinates
        theta_vals = np.linspace(0, 4*np.pi, n)
        spiral_points = [SacredGeometry.phi_spiral(t) for t in theta_vals]

        # Calculate spiral characteristics
        distances = [np.sqrt(p[0]**2 + p[1]**2) for p in spiral_points]

        # Map data to spiral
        data_normalized = (data_flat - np.min(data_flat)) / (np.max(data_flat) - np.min(data_flat) + 1e-10)
        distances_normalized = np.array(distances) / max(distances)

        # Measure how well data fits spiral growth
        correlation = np.corrcoef(data_normalized, distances_normalized)[0, 1] if n > 1 else 0

        return {
            'method': 'phi_spiral',
            'spiral_correlation': float(correlation),
            'spiral_resonance': float(abs(correlation)),
            'spiral_points': spiral_points[:10],  # First 10 for display
            'shape': SacredShape.SPIRAL,
            'properties': {
                'growth_type': 'logarithmic' if correlation > 0.5 else 'linear',
                'expansion_rate': float(self.phi)
            }
        }

    def _encode_platonic(self, data: np.ndarray) -> Dict[str, Any]:
        """Encode using Platonic solid symmetries"""
        data_flat = data.flatten()

        # Try each Platonic solid
        solid_shapes = [
            SacredShape.TETRAHEDRON,
            SacredShape.CUBE,
            SacredShape.OCTAHEDRON,
            SacredShape.DODECAHEDRON,
            SacredShape.ICOSAHEDRON
        ]

        best_match = None
        best_resonance = 0

        for shape in solid_shapes:
            vertices = SacredGeometry.platonic_solid_vertices(shape)
            if vertices is None:
                continue

            # Calculate resonance based on dimensional match
            num_vertices = len(vertices)

            # Check if data length is multiple of vertices
            if len(data_flat) % num_vertices == 0:
                resonance = 1.0
            else:
                resonance = 1.0 - abs(len(data_flat) % num_vertices) / num_vertices

            if resonance > best_resonance:
                best_resonance = resonance
                best_match = {
                    'shape': shape,
                    'vertices': vertices,
                    'num_vertices': num_vertices,
                    'resonance': resonance
                }

        return {
            'method': 'platonic',
            'best_match_shape': best_match['shape'] if best_match else None,
            'platonic_resonance': float(best_resonance),
            'num_vertices': best_match['num_vertices'] if best_match else 0,
            'shape': best_match['shape'] if best_match else SacredShape.CUBE,
            'properties': {
                'symmetry_group': best_match['shape'].name if best_match else 'none',
                'dimensional_match': best_resonance > 0.8
            }
        }

    def _encode_fractal(self, data: np.ndarray) -> Dict[str, Any]:
        """Encode using fractal/self-similar patterns"""
        data_flat = data.flatten()
        n = len(data_flat)

        # Check for self-similarity at different scales
        scales = [2, 3, 4, 5]
        self_similarity_scores = []

        for scale in scales:
            if n >= scale * 2:
                # Split data into chunks
                chunk_size = n // scale
                chunks = [data_flat[i*chunk_size:(i+1)*chunk_size] for i in range(scale)]

                # Calculate correlation between chunks
                correlations = []
                for i in range(len(chunks)-1):
                    if len(chunks[i]) == len(chunks[i+1]):
                        corr = np.corrcoef(chunks[i], chunks[i+1])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

                if correlations:
                    self_similarity_scores.append(np.mean(correlations))

        avg_self_similarity = np.mean(self_similarity_scores) if self_similarity_scores else 0

        return {
            'method': 'fractal',
            'self_similarity': float(avg_self_similarity),
            'fractal_resonance': float(avg_self_similarity),
            'shape': SacredShape.FLOWER_OF_LIFE,  # Fractal-like structure
            'properties': {
                'fractal_dimension_estimate': 1 + avg_self_similarity,  # Simplified
                'scale_invariant': avg_self_similarity > 0.6
            }
        }

    def detect_geometric_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect all geometric patterns in data

        Returns comprehensive analysis using all encoding methods
        """

        # Try all encoding methods
        encodings = {
            'fibonacci': self.encode_data(data, 'fibonacci'),
            'phi_spiral': self.encode_data(data, 'phi_spiral'),
            'platonic': self.encode_data(data, 'platonic'),
            'fractal': self.encode_data(data, 'fractal')
        }

        # Find strongest resonance
        resonances = {
            method: result.get(f'{method}_resonance', 0)
            for method, result in encodings.items()
        }

        dominant_method = max(resonances.items(), key=lambda x: x[1])[0]
        dominant_encoding = encodings[dominant_method]

        # Overall geometric signature
        signature = {
            'dominant_pattern': dominant_method,
            'dominant_shape': dominant_encoding['shape'],
            'resonances': resonances,
            'encodings': encodings,
            'overall_geometric_coherence': np.mean(list(resonances.values())),
            'sacred_geometry_present': any(r > 0.6 for r in resonances.values()),
            'interpretation': self._interpret_geometric_signature(resonances, dominant_method)
        }

        return signature

    def _interpret_geometric_signature(self, resonances: Dict[str, float],
                                      dominant: str) -> str:
        """Interpret what the geometric signature means"""

        if resonances[dominant] > 0.8:
            return f"STRONG {dominant.upper()} pattern - highly ordered geometric structure"
        elif resonances[dominant] > 0.6:
            return f"MODERATE {dominant.upper()} pattern - partial geometric order"
        elif max(resonances.values()) > 0.4:
            return "WEAK geometric patterns - some order present"
        else:
            return "CHAOTIC - no clear geometric structure"


if __name__ == '__main__':
    # Demo
    print("📐 Geometric Encoding Demo\n")

    encoder = GeometricEncoder()

    # Test with Fibonacci-like sequence
    print("\n1. Fibonacci-like sequence:")
    fib_data = np.array(SacredGeometry.fibonacci_sequence(20), dtype=float)
    result = encoder.detect_geometric_patterns(fib_data)

    print(f"   Dominant Pattern: {result['dominant_pattern']}")
    print(f"   Dominant Shape: {result['dominant_shape'].name}")
    print(f"   Overall Coherence: {result['overall_geometric_coherence']:.3f}")
    print(f"   Interpretation: {result['interpretation']}")
    print(f"   Resonances:")
    for method, resonance in result['resonances'].items():
        print(f"      {method}: {resonance:.3f}")

    # Test with random data
    print("\n2. Random data:")
    random_data = np.random.randn(50)
    result2 = encoder.detect_geometric_patterns(random_data)

    print(f"   Dominant Pattern: {result2['dominant_pattern']}")
    print(f"   Overall Coherence: {result2['overall_geometric_coherence']:.3f}")
    print(f"   Interpretation: {result2['interpretation']}")

    # Test phi spiral generation
    print("\n3. Golden Spiral visualization:")
    for i in range(5):
        theta = i * np.pi / 4
        x, y = SacredGeometry.phi_spiral(theta)
        print(f"   θ={theta:.3f}: ({x:.3f}, {y:.3f})")
