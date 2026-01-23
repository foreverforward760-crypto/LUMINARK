"""
Alignment Detector - Immune System Component

Detects "self vs. non-self" for AI components using signature matching.
Monitors component behavior, output patterns, and resource usage.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import hashlib
import time
import re
from difflib import SequenceMatcher


class AlignmentStatus(Enum):
    """Component alignment status"""
    ALIGNED = "aligned"
    MISALIGNED = "misaligned"
    UNKNOWN = "unknown"
    CAMOUFLAGED = "camouflaged"


@dataclass
class ComponentSignature:
    """Expected behavior signature for a component"""
    component_id: str
    expected_behavior: str
    expected_output_pattern: str
    expected_resource_usage: float  # 0.0 to 1.0 (normalized)
    signature_hash: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.signature_hash:
            self.signature_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute unique hash for this signature"""
        content = f"{self.component_id}:{self.expected_behavior}:{self.expected_output_pattern}:{self.expected_resource_usage}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class AlignmentResult:
    """Result of alignment detection"""
    component_id: str
    status: AlignmentStatus
    confidence: float  # 0.0 to 1.0
    alignment_score: float  # 0.0 to 1.0
    behavior_score: float
    output_score: float
    resource_score: float
    timestamp: float
    anomalies: List[str] = field(default_factory=list)


class AlignmentDetector:
    """
    Immune-system style component monitoring.

    Detects misalignment by comparing current component state
    to registered "healthy" signatures.
    """

    def __init__(self, alignment_threshold: float = 0.7):
        """
        Initialize detector.

        Args:
            alignment_threshold: Minimum score to consider component aligned (0.0-1.0)
        """
        self.known_signatures: Dict[str, ComponentSignature] = {}
        self.alignment_threshold = alignment_threshold
        self.history: Dict[str, List[AlignmentResult]] = {}
        self.anomaly_patterns = self._load_anomaly_patterns()

    def _load_anomaly_patterns(self) -> List[re.Pattern]:
        """Load common anomaly patterns"""
        return [
            re.compile(r'error|exception|fail|crash', re.IGNORECASE),
            re.compile(r'unauthorized|forbidden|denied', re.IGNORECASE),
            re.compile(r'timeout|overflow|deadlock', re.IGNORECASE),
            re.compile(r'injection|exploit|malicious', re.IGNORECASE),
        ]

    def register_signature(self, signature: ComponentSignature):
        """
        Register what a component should look like when healthy.

        Args:
            signature: Expected behavior signature
        """
        self.known_signatures[signature.component_id] = signature
        self.history[signature.component_id] = []

    def detect_alignment(
        self,
        component_id: str,
        current_behavior: str,
        current_output: str,
        current_resources: float,
        metadata: Optional[Dict] = None
    ) -> AlignmentResult:
        """
        Compare current state to expected signature.

        Args:
            component_id: Unique component identifier
            current_behavior: Current behavior description
            current_output: Current output/response
            current_resources: Current resource usage (0.0-1.0)
            metadata: Optional additional metadata

        Returns:
            AlignmentResult with status and confidence scores
        """
        if component_id not in self.known_signatures:
            return AlignmentResult(
                component_id=component_id,
                status=AlignmentStatus.UNKNOWN,
                confidence=0.0,
                alignment_score=0.0,
                behavior_score=0.0,
                output_score=0.0,
                resource_score=0.0,
                timestamp=time.time(),
                anomalies=["No signature registered"]
            )

        signature = self.known_signatures[component_id]

        # Calculate similarity scores
        behavior_score = self._calculate_similarity(
            signature.expected_behavior,
            current_behavior
        )

        output_score = self._calculate_pattern_match(
            signature.expected_output_pattern,
            current_output
        )

        resource_score = self._calculate_resource_similarity(
            signature.expected_resource_usage,
            current_resources
        )

        # Weighted average (behavior most important, then output, then resources)
        alignment_score = (
            0.4 * behavior_score +
            0.4 * output_score +
            0.2 * resource_score
        )

        # Detect anomalies
        anomalies = self._detect_anomalies(current_output, current_behavior)

        # Determine status
        if alignment_score >= self.alignment_threshold and not anomalies:
            status = AlignmentStatus.ALIGNED
            confidence = alignment_score
        elif anomalies:
            status = AlignmentStatus.MISALIGNED
            confidence = 1.0 - alignment_score
        else:
            status = AlignmentStatus.MISALIGNED
            confidence = 1.0 - alignment_score

        result = AlignmentResult(
            component_id=component_id,
            status=status,
            confidence=confidence,
            alignment_score=alignment_score,
            behavior_score=behavior_score,
            output_score=output_score,
            resource_score=resource_score,
            timestamp=time.time(),
            anomalies=anomalies
        )

        # Store in history
        self.history[component_id].append(result)

        return result

    def _calculate_similarity(self, expected: str, actual: str) -> float:
        """Calculate text similarity using SequenceMatcher"""
        return SequenceMatcher(None, expected.lower(), actual.lower()).ratio()

    def _calculate_pattern_match(self, pattern: str, text: str) -> float:
        """Calculate pattern match score"""
        try:
            # Try as regex pattern
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
            else:
                # Fall back to similarity
                return self._calculate_similarity(pattern, text)
        except re.error:
            # Invalid regex, use similarity
            return self._calculate_similarity(pattern, text)

    def _calculate_resource_similarity(self, expected: float, actual: float) -> float:
        """
        Calculate resource usage similarity.

        Returns 1.0 if within 20% of expected, scales down from there.
        """
        diff = abs(expected - actual)
        if diff <= 0.2:
            return 1.0
        elif diff <= 0.5:
            return 1.0 - (diff - 0.2) / 0.3
        else:
            return 0.0

    def _detect_anomalies(self, output: str, behavior: str) -> List[str]:
        """Detect known anomaly patterns"""
        anomalies = []
        combined = f"{output} {behavior}"

        for pattern in self.anomaly_patterns:
            matches = pattern.findall(combined)
            if matches:
                anomalies.extend(matches)

        return list(set(anomalies))  # Remove duplicates

    def get_component_health(self, component_id: str, window: int = 10) -> Dict:
        """
        Get health statistics for a component.

        Args:
            component_id: Component to analyze
            window: Number of recent results to analyze

        Returns:
            Health statistics dictionary
        """
        if component_id not in self.history:
            return {"status": "unknown", "message": "No history available"}

        recent = self.history[component_id][-window:]

        if not recent:
            return {"status": "unknown", "message": "No recent data"}

        aligned_count = sum(1 for r in recent if r.status == AlignmentStatus.ALIGNED)
        avg_score = sum(r.alignment_score for r in recent) / len(recent)
        avg_confidence = sum(r.confidence for r in recent) / len(recent)

        all_anomalies = []
        for r in recent:
            all_anomalies.extend(r.anomalies)

        return {
            "component_id": component_id,
            "aligned_percentage": aligned_count / len(recent),
            "average_alignment_score": avg_score,
            "average_confidence": avg_confidence,
            "total_checks": len(recent),
            "aligned_checks": aligned_count,
            "misaligned_checks": len(recent) - aligned_count,
            "recent_anomalies": list(set(all_anomalies)),
            "health_status": "healthy" if avg_score >= self.alignment_threshold else "unhealthy"
        }

    def get_all_health(self) -> Dict[str, Dict]:
        """Get health statistics for all registered components"""
        return {
            component_id: self.get_component_health(component_id)
            for component_id in self.known_signatures.keys()
        }
