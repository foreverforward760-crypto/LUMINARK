"""
SAP SPAT Vector Calculations

Calculates system state vectors for defense trigger logic:
- Complexity (C): System differentiation
- Stability (S): Resistance to change
- Tension (T): Internal contradictions
- Adaptability (A): Capacity for change
- Coherence (Coh): Truth alignment

Used by MycelialDefenseSystem to determine appropriate defense mode.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import numpy as np


@dataclass
class SPATVectors:
    """System state vectors for defense triggers"""
    complexity: float      # 0.0-1.0: System differentiation level
    stability: float       # 0.0-1.0: Resistance to change
    tension: float         # 0.0-1.0: Internal contradictions
    adaptability: float    # 0.0-1.0: Capacity for change
    coherence: float       # 0.0-1.0: Truth/purpose alignment
    timestamp: float
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "complexity": self.complexity,
            "stability": self.stability,
            "tension": self.tension,
            "adaptability": self.adaptability,
            "coherence": self.coherence,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class SAPCalculator:
    """
    Calculate SAP SPAT vectors from system metrics.

    Translates real system measurements into the 5 SPAT dimensions
    for intelligent defense triggering.
    """

    def __init__(self):
        self.history: List[SPATVectors] = []
        self.baseline: Optional[SPATVectors] = None

    def calculate_from_components(
        self,
        components: List[Dict],
        alignment_scores: Dict[str, float],
        resource_usage: Dict[str, float],
        connection_map: Optional[Dict[str, List[str]]] = None
    ) -> SPATVectors:
        """
        Calculate SPAT vectors from component state.

        Args:
            components: List of component dictionaries
            alignment_scores: Component alignment scores (0.0-1.0)
            resource_usage: Resource usage per component (0.0-1.0)
            connection_map: Optional map of component connections

        Returns:
            SPATVectors representing current system state
        """
        if not components:
            return SPATVectors(
                complexity=0.0,
                stability=1.0,
                tension=0.0,
                adaptability=1.0,
                coherence=1.0,
                timestamp=time.time()
            )

        # Calculate Complexity: Based on number of components and connections
        complexity = self._calculate_complexity(components, connection_map)

        # Calculate Stability: Based on alignment consistency
        stability = self._calculate_stability(alignment_scores)

        # Calculate Tension: Based on resource contention and misalignment
        tension = self._calculate_tension(alignment_scores, resource_usage)

        # Calculate Adaptability: Based on resource headroom
        adaptability = self._calculate_adaptability(resource_usage)

        # Calculate Coherence: Based on overall alignment
        coherence = self._calculate_coherence(alignment_scores)

        vectors = SPATVectors(
            complexity=complexity,
            stability=stability,
            tension=tension,
            adaptability=adaptability,
            coherence=coherence,
            timestamp=time.time(),
            metadata={
                "component_count": len(components),
                "avg_alignment": np.mean(list(alignment_scores.values())) if alignment_scores else 1.0,
                "avg_resources": np.mean(list(resource_usage.values())) if resource_usage else 0.0
            }
        )

        self.history.append(vectors)

        return vectors

    def _calculate_complexity(
        self,
        components: List[Dict],
        connection_map: Optional[Dict[str, List[str]]]
    ) -> float:
        """
        Calculate system complexity.

        Higher complexity = more components and connections.
        """
        component_count = len(components)

        # Normalize component count (assume 1-100 components)
        component_factor = min(1.0, component_count / 100.0)

        if connection_map:
            # Count total connections
            total_connections = sum(len(conns) for conns in connection_map.values())
            max_possible = component_count * (component_count - 1)  # Fully connected
            connection_factor = total_connections / max_possible if max_possible > 0 else 0.0
        else:
            connection_factor = 0.5  # Assume moderate connectivity

        # Weight: 60% components, 40% connections
        complexity = 0.6 * component_factor + 0.4 * connection_factor

        return min(1.0, complexity)

    def _calculate_stability(self, alignment_scores: Dict[str, float]) -> float:
        """
        Calculate system stability.

        Higher stability = consistent alignment across components.
        Low variance in alignment = high stability.
        """
        if not alignment_scores:
            return 1.0

        scores = list(alignment_scores.values())

        # Calculate variance
        mean_score = np.mean(scores)
        variance = np.var(scores)

        # Low variance = high stability
        # Normalize variance (assume max variance ~0.25 for 0-1 range)
        stability = 1.0 - min(1.0, variance / 0.25)

        return stability

    def _calculate_tension(
        self,
        alignment_scores: Dict[str, float],
        resource_usage: Dict[str, float]
    ) -> float:
        """
        Calculate system tension.

        Higher tension = contradictions and conflicts.
        Misalignment + high resource usage = high tension.
        """
        if not alignment_scores:
            return 0.0

        scores = list(alignment_scores.values())
        resources = list(resource_usage.values()) if resource_usage else [0.0] * len(scores)

        # Misalignment factor (1.0 - avg alignment)
        misalignment = 1.0 - np.mean(scores)

        # Resource pressure (avg resource usage)
        resource_pressure = np.mean(resources)

        # Variance in alignment (inconsistency creates tension)
        variance_tension = np.var(scores)

        # Combine factors
        tension = (
            0.4 * misalignment +
            0.3 * resource_pressure +
            0.3 * variance_tension
        )

        return min(1.0, tension)

    def _calculate_adaptability(self, resource_usage: Dict[str, float]) -> float:
        """
        Calculate system adaptability.

        Higher adaptability = more capacity to change.
        Low resource usage = high adaptability (headroom available).
        """
        if not resource_usage:
            return 1.0

        resources = list(resource_usage.values())
        avg_usage = np.mean(resources)

        # Invert: low usage = high adaptability
        adaptability = 1.0 - avg_usage

        return adaptability

    def _calculate_coherence(self, alignment_scores: Dict[str, float]) -> float:
        """
        Calculate system coherence.

        Higher coherence = aligned with truth/purpose.
        Average alignment score = coherence.
        """
        if not alignment_scores:
            return 1.0

        scores = list(alignment_scores.values())
        coherence = np.mean(scores)

        return coherence

    def calculate_from_metrics(
        self,
        complexity: Optional[float] = None,
        stability: Optional[float] = None,
        tension: Optional[float] = None,
        adaptability: Optional[float] = None,
        coherence: Optional[float] = None
    ) -> SPATVectors:
        """
        Create SPAT vectors from direct metric inputs.

        Useful for testing or when metrics are already calculated.

        Args:
            complexity: System complexity (0.0-1.0)
            stability: System stability (0.0-1.0)
            tension: System tension (0.0-1.0)
            adaptability: System adaptability (0.0-1.0)
            coherence: System coherence (0.0-1.0)

        Returns:
            SPATVectors with provided or default values
        """
        vectors = SPATVectors(
            complexity=complexity if complexity is not None else 0.5,
            stability=stability if stability is not None else 0.5,
            tension=tension if tension is not None else 0.5,
            adaptability=adaptability if adaptability is not None else 0.5,
            coherence=coherence if coherence is not None else 0.5,
            timestamp=time.time()
        )

        self.history.append(vectors)

        return vectors

    def set_baseline(self, vectors: SPATVectors):
        """Set baseline SPAT vectors for healthy system state"""
        self.baseline = vectors

    def calculate_drift(self, current: SPATVectors) -> Dict[str, float]:
        """
        Calculate drift from baseline.

        Returns:
            Dictionary of dimension -> drift amount
        """
        if not self.baseline:
            return {
                "complexity": 0.0,
                "stability": 0.0,
                "tension": 0.0,
                "adaptability": 0.0,
                "coherence": 0.0
            }

        return {
            "complexity": abs(current.complexity - self.baseline.complexity),
            "stability": abs(current.stability - self.baseline.stability),
            "tension": abs(current.tension - self.baseline.tension),
            "adaptability": abs(current.adaptability - self.baseline.adaptability),
            "coherence": abs(current.coherence - self.baseline.coherence)
        }

    def get_trend(self, dimension: str, window: int = 10) -> str:
        """
        Get trend for a specific dimension.

        Args:
            dimension: SPAT dimension name
            window: Number of recent measurements

        Returns:
            "increasing", "decreasing", or "stable"
        """
        if len(self.history) < 2:
            return "stable"

        recent = self.history[-window:]
        values = [getattr(v, dimension) for v in recent]

        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"

    def get_analysis(self, vectors: SPATVectors) -> Dict:
        """
        Get human-readable analysis of SPAT vectors.

        Returns:
            Dictionary with analysis and recommendations
        """
        warnings = []
        recommendations = []

        # Analyze each dimension
        if vectors.complexity > 0.8:
            warnings.append("Very high complexity - system may be difficult to manage")
            recommendations.append("Consider simplifying architecture")

        if vectors.stability < 0.2:
            warnings.append("Very low stability - system is volatile")
            recommendations.append("Reduce rate of change, allow system to settle")

        if vectors.tension > 0.8:
            warnings.append("Very high tension - system under severe stress")
            recommendations.append("Activate defensive measures immediately")

        if vectors.adaptability < 0.2:
            warnings.append("Very low adaptability - system cannot respond to changes")
            recommendations.append("Free up resources, reduce load")

        if vectors.coherence < 0.3:
            warnings.append("Very low coherence - system lost alignment with purpose")
            recommendations.append("Re-evaluate goals, check for corruption")

        # Overall health assessment
        avg_health = np.mean([
            vectors.stability,
            1.0 - vectors.tension,  # Invert tension
            vectors.adaptability,
            vectors.coherence
        ])

        if avg_health > 0.7:
            health_status = "healthy"
        elif avg_health > 0.4:
            health_status = "degraded"
        else:
            health_status = "critical"

        return {
            "health_status": health_status,
            "health_score": avg_health,
            "warnings": warnings,
            "recommendations": recommendations,
            "vectors": vectors.to_dict()
        }
