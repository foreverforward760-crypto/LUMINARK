"""
Temporal Anchoring & Timeline Integrity

Blockchain-style immutable timestamps detect timeline manipulation,
rollback attacks, and temporal inconsistencies.

Key concepts:
- Merkle trees for decision history
- Cryptographic temporal anchors
- Timeline manipulation detection
- Causal chain verification
- Memory integrity checking
- Fractal time patterns (As Above, So Below - patterns repeat at different time scales)

Inspired by:
- Blockchain immutability
- Merkle tree verification
- Lamport clocks (distributed systems)
- Causal consistency models
- Fractal time theory

Author: Richard Leroy Stanfield Jr. / Meridian Axiom
Part of: SAP V4.0 Enhancement Suite
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import secrets
import json
from collections import defaultdict


class AnchorType(Enum):
    """Types of temporal anchors"""
    GENESIS = "genesis"  # First anchor
    DECISION = "decision"  # Decision point
    STATE_CHANGE = "state_change"  # System state transition
    CONSCIOUSNESS_SHIFT = "consciousness_shift"  # SAP stage change
    THREAT_EVENT = "threat_event"  # Security event
    MERKLE_ROOT = "merkle_root"  # Periodic tree root


class TamperType(Enum):
    """Types of timeline tampering"""
    ROLLBACK = "rollback"  # Attempt to revert to earlier state
    INSERTION = "insertion"  # Inject false history
    DELETION = "deletion"  # Remove events from timeline
    REORDERING = "reordering"  # Change causal order
    BIFURCATION = "bifurcation"  # Timeline split/fork


@dataclass
class TemporalAnchor:
    """Immutable timestamp anchor"""
    anchor_id: str
    anchor_type: AnchorType
    timestamp: float  # Unix timestamp
    previous_hash: str  # Hash of previous anchor
    data_hash: str  # Hash of anchored data
    merkle_root: str  # Root of merkle tree at this point
    nonce: str  # Random nonce for uniqueness
    signature: str  # Cryptographic signature

    def compute_hash(self) -> str:
        """Compute hash of this anchor"""
        content = f"{self.anchor_id}:{self.timestamp}:{self.previous_hash}:{self.data_hash}:{self.merkle_root}:{self.nonce}"
        return hashlib.sha256(content.encode()).hexdigest()

    def verify(self, expected_hash: str) -> bool:
        """Verify anchor hasn't been tampered"""
        return self.compute_hash() == expected_hash


@dataclass
class TimelineEvent:
    """Event in timeline"""
    event_id: str
    event_type: str
    timestamp: float
    data: Dict[str, Any]
    anchor_id: Optional[str] = None  # Which anchor covers this
    causal_parents: List[str] = field(default_factory=list)  # Events that caused this
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of event"""
        content = json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "data": self.data,
            "causal_parents": sorted(self.causal_parents)
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class MerkleNode:
    """Node in Merkle tree"""
    hash: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    is_leaf: bool = False
    data: Optional[Any] = None


@dataclass
class TamperDetection:
    """Detected tampering attempt"""
    detection_id: str
    tamper_type: TamperType
    timestamp: float
    affected_anchors: List[str]
    affected_events: List[str]
    evidence: Dict[str, Any]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL


@dataclass
class FractalTimePattern:
    """Fractal pattern detected across time scales"""
    pattern_id: str
    pattern_type: str
    occurrences: List[Tuple[float, float]]  # (timestamp, scale)
    fractal_dimension: float
    self_similarity_score: float


class TemporalAnchoringSystem:
    """
    Blockchain-style temporal anchoring for timeline integrity

    Every significant event gets cryptographically anchored
    Timeline tampering is mathematically detectable
    """

    def __init__(
        self,
        system_id: str,
        anchor_interval: float = 60.0  # Anchor every 60 seconds
    ):
        self.system_id = system_id
        self.anchor_interval = anchor_interval

        # Anchor chain
        self.anchors: List[TemporalAnchor] = []
        self.anchor_index: Dict[str, TemporalAnchor] = {}

        # Event timeline
        self.events: List[TimelineEvent] = []
        self.event_index: Dict[str, TimelineEvent] = {}

        # Merkle trees (periodic snapshots)
        self.merkle_roots: Dict[str, str] = {}  # anchor_id -> root hash

        # Tampering detection
        self.tamper_detections: List[TamperDetection] = []

        # Fractal patterns
        self.fractal_patterns: List[FractalTimePattern] = []

        # State
        self.genesis_time: float = time.time()
        self.last_anchor_time: float = self.genesis_time

        # Create genesis anchor
        self._create_genesis_anchor()

    def _create_genesis_anchor(self):
        """Create first anchor in chain"""
        genesis = TemporalAnchor(
            anchor_id=f"anchor_genesis_{self.system_id}",
            anchor_type=AnchorType.GENESIS,
            timestamp=self.genesis_time,
            previous_hash="0" * 64,  # Genesis has no previous
            data_hash="0" * 64,  # No data yet
            merkle_root="0" * 64,  # Empty tree
            nonce=secrets.token_hex(16),
            signature=self._sign("genesis")
        )

        self.anchors.append(genesis)
        self.anchor_index[genesis.anchor_id] = genesis

    def _sign(self, content: str) -> str:
        """Sign content (simplified - in production use real crypto)"""
        return hashlib.sha256(f"{self.system_id}:{content}:{secrets.token_hex(16)}".encode()).hexdigest()

    def record_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        causal_parents: Optional[List[str]] = None
    ) -> TimelineEvent:
        """
        Record event in timeline

        Args:
            event_type: Type of event
            data: Event data
            causal_parents: Events that caused this one

        Returns:
            TimelineEvent recorded
        """
        event = TimelineEvent(
            event_id=f"event_{secrets.token_hex(8)}",
            event_type=event_type,
            timestamp=time.time(),
            data=data,
            causal_parents=causal_parents or []
        )

        self.events.append(event)
        self.event_index[event.event_id] = event

        # Check if we need to create anchor
        if time.time() - self.last_anchor_time >= self.anchor_interval:
            self.create_anchor(AnchorType.MERKLE_ROOT)

        return event

    def create_anchor(
        self,
        anchor_type: AnchorType,
        data: Optional[Dict] = None
    ) -> TemporalAnchor:
        """
        Create temporal anchor

        Args:
            anchor_type: Type of anchor
            data: Optional data to anchor

        Returns:
            TemporalAnchor created
        """
        # Get previous anchor
        previous = self.anchors[-1] if self.anchors else None
        previous_hash = previous.compute_hash() if previous else "0" * 64

        # Build merkle tree of recent events
        recent_events = [
            e for e in self.events
            if e.timestamp > self.last_anchor_time
        ]
        merkle_root = self._build_merkle_tree(recent_events)

        # Hash data
        data_hash = hashlib.sha256(
            json.dumps(data or {}, sort_keys=True).encode()
        ).hexdigest()

        # Create anchor
        anchor = TemporalAnchor(
            anchor_id=f"anchor_{secrets.token_hex(8)}",
            anchor_type=anchor_type,
            timestamp=time.time(),
            previous_hash=previous_hash,
            data_hash=data_hash,
            merkle_root=merkle_root,
            nonce=secrets.token_hex(16),
            signature=self._sign(f"{anchor_type.value}:{time.time()}")
        )

        # Mark events with this anchor
        for event in recent_events:
            event.anchor_id = anchor.anchor_id

        self.anchors.append(anchor)
        self.anchor_index[anchor.anchor_id] = anchor
        self.merkle_roots[anchor.anchor_id] = merkle_root
        self.last_anchor_time = anchor.timestamp

        return anchor

    def _build_merkle_tree(self, events: List[TimelineEvent]) -> str:
        """
        Build Merkle tree from events

        Args:
            events: Events to include

        Returns:
            Root hash
        """
        if not events:
            return "0" * 64

        # Create leaf nodes
        leaves = [
            MerkleNode(hash=event.hash, is_leaf=True, data=event)
            for event in events
        ]

        # Build tree bottom-up
        current_level = leaves

        while len(current_level) > 1:
            next_level = []

            # Pair up nodes
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left

                # Create parent
                combined = left.hash + right.hash
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                parent = MerkleNode(hash=parent_hash, left=left, right=right)

                next_level.append(parent)

            current_level = next_level

        return current_level[0].hash

    def verify_chain_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify entire anchor chain integrity

        Returns:
            (is_valid, list of errors)
        """
        errors = []

        # Check each anchor
        for i, anchor in enumerate(self.anchors):
            # Verify hash
            expected_hash = anchor.compute_hash()
            if not anchor.verify(expected_hash):
                errors.append(f"Anchor {anchor.anchor_id} hash mismatch")

            # Verify links to previous (except genesis)
            if i > 0:
                previous = self.anchors[i - 1]
                expected_prev_hash = previous.compute_hash()

                if anchor.previous_hash != expected_prev_hash:
                    errors.append(
                        f"Anchor {anchor.anchor_id} previous_hash doesn't match {previous.anchor_id}"
                    )

            # Verify merkle root if we have events
            anchored_events = [
                e for e in self.events
                if e.anchor_id == anchor.anchor_id
            ]

            if anchored_events:
                recomputed_root = self._build_merkle_tree(anchored_events)
                if anchor.merkle_root != recomputed_root:
                    errors.append(
                        f"Anchor {anchor.anchor_id} merkle root mismatch"
                    )

        return (len(errors) == 0, errors)

    def verify_event_integrity(self, event_id: str) -> bool:
        """
        Verify specific event hasn't been tampered

        Args:
            event_id: Event to verify

        Returns:
            True if event is valid
        """
        if event_id not in self.event_index:
            return False

        event = self.event_index[event_id]

        # Recompute hash
        expected_hash = event._compute_hash()

        return event.hash == expected_hash

    def detect_rollback_attack(self) -> Optional[TamperDetection]:
        """
        Detect if timeline has been rolled back

        Returns:
            TamperDetection if rollback detected
        """
        # Check for discontinuities in timestamps
        for i in range(1, len(self.anchors)):
            current = self.anchors[i]
            previous = self.anchors[i - 1]

            # Timestamp should be monotonically increasing
            if current.timestamp < previous.timestamp:
                return TamperDetection(
                    detection_id=f"tamper_{secrets.token_hex(8)}",
                    tamper_type=TamperType.ROLLBACK,
                    timestamp=time.time(),
                    affected_anchors=[current.anchor_id, previous.anchor_id],
                    affected_events=[],
                    evidence={
                        "current_timestamp": current.timestamp,
                        "previous_timestamp": previous.timestamp,
                        "delta": current.timestamp - previous.timestamp
                    },
                    severity="CRITICAL"
                )

        # Check for hash chain breaks
        is_valid, errors = self.verify_chain_integrity()
        if not is_valid:
            affected = [a.anchor_id for a in self.anchors]
            return TamperDetection(
                detection_id=f"tamper_{secrets.token_hex(8)}",
                tamper_type=TamperType.ROLLBACK,
                timestamp=time.time(),
                affected_anchors=affected,
                affected_events=[],
                evidence={"errors": errors},
                severity="CRITICAL"
            )

        return None

    def detect_causal_violations(self) -> List[TamperDetection]:
        """
        Detect events that violate causal ordering

        Returns:
            List of violations detected
        """
        violations = []

        for event in self.events:
            for parent_id in event.causal_parents:
                if parent_id not in self.event_index:
                    # Parent doesn't exist - deletion?
                    violations.append(TamperDetection(
                        detection_id=f"tamper_{secrets.token_hex(8)}",
                        tamper_type=TamperType.DELETION,
                        timestamp=time.time(),
                        affected_anchors=[event.anchor_id] if event.anchor_id else [],
                        affected_events=[event.event_id],
                        evidence={
                            "event": event.event_id,
                            "missing_parent": parent_id
                        },
                        severity="HIGH"
                    ))
                    continue

                parent = self.event_index[parent_id]

                # Parent must occur before child
                if parent.timestamp > event.timestamp:
                    violations.append(TamperDetection(
                        detection_id=f"tamper_{secrets.token_hex(8)}",
                        tamper_type=TamperType.REORDERING,
                        timestamp=time.time(),
                        affected_anchors=[],
                        affected_events=[event.event_id, parent_id],
                        evidence={
                            "child": event.event_id,
                            "child_time": event.timestamp,
                            "parent": parent_id,
                            "parent_time": parent.timestamp
                        },
                        severity="HIGH"
                    ))

        return violations

    def detect_fractal_patterns(
        self,
        pattern_type: str,
        min_occurrences: int = 3
    ) -> List[FractalTimePattern]:
        """
        Detect fractal patterns across time scales

        "As Above, So Below" - patterns repeat at different scales

        Args:
            pattern_type: Type of pattern to detect
            min_occurrences: Minimum pattern repetitions

        Returns:
            List of fractal patterns detected
        """
        patterns = []

        # Get events of this type
        typed_events = [
            e for e in self.events
            if pattern_type in e.event_type
        ]

        if len(typed_events) < min_occurrences:
            return patterns

        # Calculate intervals between occurrences
        intervals = []
        for i in range(1, len(typed_events)):
            interval = typed_events[i].timestamp - typed_events[i-1].timestamp
            intervals.append(interval)

        # Look for self-similar patterns at different scales
        # (simplified - real fractal analysis would use wavelet transforms)

        # Check if intervals cluster at multiple scales
        scale_clusters = defaultdict(list)

        for interval in intervals:
            # Categorize by order of magnitude
            scale = int(np.log10(interval)) if interval > 0 else 0
            scale_clusters[scale].append(interval)

        # If we have clusters at multiple scales with similar patterns
        if len(scale_clusters) >= 2:
            # Calculate self-similarity
            scales = sorted(scale_clusters.keys())
            similarity_scores = []

            for i in range(len(scales) - 1):
                scale_a = scales[i]
                scale_b = scales[i + 1]

                # Compare distributions
                mean_a = np.mean(scale_clusters[scale_a])
                mean_b = np.mean(scale_clusters[scale_b])

                # Scale-adjusted similarity
                ratio = mean_b / mean_a if mean_a > 0 else 0
                expected_ratio = 10  # Order of magnitude

                similarity = 1.0 - abs(ratio - expected_ratio) / expected_ratio
                similarity_scores.append(max(0, similarity))

            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0

            if avg_similarity > 0.5:  # Threshold for fractal pattern
                pattern = FractalTimePattern(
                    pattern_id=f"fractal_{secrets.token_hex(8)}",
                    pattern_type=pattern_type,
                    occurrences=[
                        (e.timestamp, scale_clusters.get(int(np.log10(e.timestamp - self.genesis_time)), [0])[0])
                        for e in typed_events
                    ],
                    fractal_dimension=len(scale_clusters),
                    self_similarity_score=avg_similarity
                )
                patterns.append(pattern)
                self.fractal_patterns.append(pattern)

        return patterns

    def get_timeline_snapshot(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get snapshot of timeline

        Args:
            start_time: Start of window (None = genesis)
            end_time: End of window (None = now)

        Returns:
            Timeline snapshot
        """
        start = start_time or self.genesis_time
        end = end_time or time.time()

        # Filter anchors and events
        anchors_in_range = [
            a for a in self.anchors
            if start <= a.timestamp <= end
        ]

        events_in_range = [
            e for e in self.events
            if start <= e.timestamp <= end
        ]

        return {
            "start_time": start,
            "end_time": end,
            "duration": end - start,
            "anchors": len(anchors_in_range),
            "events": len(events_in_range),
            "integrity_verified": self.verify_chain_integrity()[0],
            "tamper_detections": len(self.tamper_detections),
            "fractal_patterns": len(self.fractal_patterns)
        }

    def export_audit_log(self, filepath: str):
        """
        Export complete audit log

        Args:
            filepath: Output file path
        """
        audit_data = {
            "system_id": self.system_id,
            "genesis_time": self.genesis_time,
            "export_time": time.time(),
            "anchors": [
                {
                    "anchor_id": a.anchor_id,
                    "type": a.anchor_type.value,
                    "timestamp": a.timestamp,
                    "hash": a.compute_hash()
                }
                for a in self.anchors
            ],
            "events": [
                {
                    "event_id": e.event_id,
                    "type": e.event_type,
                    "timestamp": e.timestamp,
                    "hash": e.hash,
                    "anchor_id": e.anchor_id
                }
                for e in self.events
            ],
            "integrity": {
                "chain_valid": self.verify_chain_integrity()[0],
                "tamper_detections": [
                    {
                        "type": t.tamper_type.value,
                        "timestamp": t.timestamp,
                        "severity": t.severity
                    }
                    for t in self.tamper_detections
                ]
            }
        }

        with open(filepath, 'w') as f:
            json.dump(audit_data, f, indent=2)


# Import numpy for fractal pattern detection
try:
    import numpy as np
except ImportError:
    # Fallback if numpy not available
    class np:
        @staticmethod
        def log10(x):
            import math
            return math.log10(x) if x > 0 else 0

        @staticmethod
        def mean(lst):
            return sum(lst) / len(lst) if lst else 0
