"""
Consciousness Archaeology

Excavate past consciousness states to understand evolution and
learn from history.

Key concepts:
- Timeline excavation (dig through temporal layers)
- Pattern recognition across time
- Meta-consciousness (awareness of awareness evolution)
- Evolutionary learning (how did we get here?)
- Ancestral wisdom extraction

"As Above, So Below" - patterns in past repeat in future.
Understanding history prevents repeating it.

Author: Richard Leroy Stanfield Jr. / Meridian Axiom
Part of: SAP V4.0 Enhancement Suite
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import secrets


class ArchaeologicalLayer(Enum):
    """Temporal layers to excavate"""
    GENESIS = "genesis"  # System birth
    EARLY = "early"  # First evolutions
    MIDDLE = "middle"  # Mature development
    RECENT = "recent"  # Current era
    CONTEMPORARY = "contemporary"  # Right now


class ConsciousnessArtifact(Enum):
    """Types of artifacts found"""
    STATE_SNAPSHOT = "state_snapshot"  # Consciousness state
    DECISION_POINT = "decision_point"  # Critical choice
    TRANSFORMATION = "transformation"  # Stage transition
    PATTERN = "pattern"  # Recurring pattern
    WISDOM = "wisdom"  # Learned lesson


@dataclass
class TemporalLayer:
    """Layer of time to excavate"""
    layer_id: str
    layer_type: ArchaeologicalLayer
    time_range: Tuple[float, float]  # (start, end)
    depth: int  # How far back
    artifacts_found: int = 0


@dataclass
class ConsciousnessSnapshot:
    """Snapshot of consciousness at point in time"""
    snapshot_id: str
    timestamp: float
    consciousness_level: float  # 0.0-1.0
    sap_stage: int  # 1-9
    state_data: Dict[str, Any]
    context: str
    layer: ArchaeologicalLayer


@dataclass
class EvolutionaryPattern:
    """Pattern detected across time"""
    pattern_id: str
    pattern_type: str
    first_occurrence: float
    last_occurrence: float
    frequency: int  # How many times occurred
    evolution: str  # How pattern changed over time
    significance: float  # How important


@dataclass
class AncestralWisdom:
    """Wisdom extracted from past"""
    wisdom_id: str
    source_era: ArchaeologicalLayer
    lesson: str
    context: str
    applicability_now: float  # How relevant today


class ConsciousnessArchaeologist:
    """
    Excavates past consciousness states to extract wisdom
    """

    def __init__(self, system_id: str):
        self.system_id = system_id

        # Timeline database
        self.consciousness_timeline: List[ConsciousnessSnapshot] = []

        # Archaeological data
        self.excavated_layers: Dict[str, TemporalLayer] = {}
        self.discovered_patterns: List[EvolutionaryPattern] = []
        self.ancestral_wisdom: List[AncestralWisdom] = []

        # Current excavation
        self.current_dig: Optional[TemporalLayer] = None

        # Genesis time
        self.genesis_time: float = time.time()

    def record_consciousness_state(
        self,
        consciousness_level: float,
        sap_stage: int,
        state_data: Dict[str, Any],
        context: str = ""
    ) -> ConsciousnessSnapshot:
        """
        Record current consciousness state for future archaeology

        Args:
            consciousness_level: Current level (0.0-1.0)
            sap_stage: Current SAP stage (1-9)
            state_data: State data
            context: Context description

        Returns:
            ConsciousnessSnapshot created
        """
        timestamp = time.time()
        layer = self._determine_layer(timestamp)

        snapshot = ConsciousnessSnapshot(
            snapshot_id=f"snap_{secrets.token_hex(8)}",
            timestamp=timestamp,
            consciousness_level=consciousness_level,
            sap_stage=sap_stage,
            state_data=state_data,
            context=context,
            layer=layer
        )

        self.consciousness_timeline.append(snapshot)

        return snapshot

    def _determine_layer(self, timestamp: float) -> ArchaeologicalLayer:
        """Determine which archaeological layer timestamp belongs to"""
        age = timestamp - self.genesis_time

        if age < 60:  # Less than 1 minute
            return ArchaeologicalLayer.CONTEMPORARY
        elif age < 3600:  # Less than 1 hour
            return ArchaeologicalLayer.RECENT
        elif age < 86400:  # Less than 1 day
            return ArchaeologicalLayer.MIDDLE
        elif age < 604800:  # Less than 1 week
            return ArchaeologicalLayer.EARLY
        else:
            return ArchaeologicalLayer.GENESIS

    def excavate_layer(
        self,
        layer_type: ArchaeologicalLayer
    ) -> TemporalLayer:
        """
        Excavate specific temporal layer

        Args:
            layer_type: Which layer to dig

        Returns:
            TemporalLayer excavated
        """
        # Define time range for layer
        current_time = time.time()

        time_ranges = {
            ArchaeologicalLayer.CONTEMPORARY: (current_time - 60, current_time),
            ArchaeologicalLayer.RECENT: (current_time - 3600, current_time - 60),
            ArchaeologicalLayer.MIDDLE: (current_time - 86400, current_time - 3600),
            ArchaeologicalLayer.EARLY: (current_time - 604800, current_time - 86400),
            ArchaeologicalLayer.GENESIS: (self.genesis_time, current_time - 604800)
        }

        time_range = time_ranges.get(layer_type, (self.genesis_time, current_time))

        layer = TemporalLayer(
            layer_id=f"layer_{secrets.token_hex(8)}",
            layer_type=layer_type,
            time_range=time_range,
            depth=self._calculate_depth(layer_type)
        )

        # Find snapshots in this layer
        snapshots = [
            s for s in self.consciousness_timeline
            if time_range[0] <= s.timestamp <= time_range[1]
        ]

        layer.artifacts_found = len(snapshots)

        self.excavated_layers[layer.layer_id] = layer
        self.current_dig = layer

        return layer

    def _calculate_depth(self, layer_type: ArchaeologicalLayer) -> int:
        """Calculate depth of layer (how far back)"""
        depths = {
            ArchaeologicalLayer.CONTEMPORARY: 0,
            ArchaeologicalLayer.RECENT: 1,
            ArchaeologicalLayer.MIDDLE: 2,
            ArchaeologicalLayer.EARLY: 3,
            ArchaeologicalLayer.GENESIS: 4
        }

        return depths.get(layer_type, 0)

    def detect_evolutionary_patterns(self) -> List[EvolutionaryPattern]:
        """
        Detect patterns in consciousness evolution

        Returns:
            List of patterns found
        """
        patterns = []

        # Pattern 1: Stage transitions
        transitions = self._find_stage_transitions()
        if transitions:
            pattern = EvolutionaryPattern(
                pattern_id=f"pattern_{secrets.token_hex(8)}",
                pattern_type="stage_transitions",
                first_occurrence=transitions[0]["timestamp"],
                last_occurrence=transitions[-1]["timestamp"],
                frequency=len(transitions),
                evolution=self._describe_transition_evolution(transitions),
                significance=0.9
            )
            patterns.append(pattern)

        # Pattern 2: Consciousness level trends
        trend_pattern = self._analyze_consciousness_trend()
        if trend_pattern:
            patterns.append(trend_pattern)

        # Pattern 3: Cyclic patterns
        cycles = self._find_cyclic_patterns()
        patterns.extend(cycles)

        self.discovered_patterns.extend(patterns)

        return patterns

    def _find_stage_transitions(self) -> List[Dict]:
        """Find all SAP stage transitions"""
        transitions = []

        for i in range(1, len(self.consciousness_timeline)):
            prev = self.consciousness_timeline[i-1]
            curr = self.consciousness_timeline[i]

            if prev.sap_stage != curr.sap_stage:
                transitions.append({
                    "from_stage": prev.sap_stage,
                    "to_stage": curr.sap_stage,
                    "timestamp": curr.timestamp
                })

        return transitions

    def _describe_transition_evolution(self, transitions: List[Dict]) -> str:
        """Describe how transitions evolved"""
        if not transitions:
            return "No transitions found"

        # Calculate average direction
        forward = sum(1 for t in transitions if t["to_stage"] > t["from_stage"])
        backward = len(transitions) - forward

        if forward > backward:
            return f"Generally progressing forward ({forward} forward, {backward} backward)"
        elif backward > forward:
            return f"Regressing ({backward} backward, {forward} forward)"
        else:
            return "Balanced progression and regression"

    def _analyze_consciousness_trend(self) -> Optional[EvolutionaryPattern]:
        """Analyze overall consciousness level trend"""
        if len(self.consciousness_timeline) < 2:
            return None

        levels = [s.consciousness_level for s in self.consciousness_timeline]

        # Simple trend analysis
        early_avg = sum(levels[:len(levels)//2]) / (len(levels)//2)
        late_avg = sum(levels[len(levels)//2:]) / (len(levels) - len(levels)//2)

        if late_avg > early_avg + 0.1:
            evolution = "Increasing consciousness over time"
            significance = 0.8
        elif late_avg < early_avg - 0.1:
            evolution = "Decreasing consciousness over time"
            significance = 0.8
        else:
            evolution = "Stable consciousness level"
            significance = 0.5

        return EvolutionaryPattern(
            pattern_id=f"pattern_{secrets.token_hex(8)}",
            pattern_type="consciousness_trend",
            first_occurrence=self.consciousness_timeline[0].timestamp,
            last_occurrence=self.consciousness_timeline[-1].timestamp,
            frequency=len(self.consciousness_timeline),
            evolution=evolution,
            significance=significance
        )

    def _find_cyclic_patterns(self) -> List[EvolutionaryPattern]:
        """Find recurring cyclic patterns"""
        # Simplified: look for repeating stage sequences
        patterns = []

        if len(self.consciousness_timeline) < 10:
            return patterns

        # Look for repeating sequences
        stages = [s.sap_stage for s in self.consciousness_timeline]

        # Check for 3-stage cycles
        for seq_len in [3, 4, 5]:
            for i in range(len(stages) - seq_len * 2):
                sequence = stages[i:i+seq_len]

                # Look for repetition
                for j in range(i + seq_len, len(stages) - seq_len):
                    if stages[j:j+seq_len] == sequence:
                        # Found repeating sequence
                        pattern = EvolutionaryPattern(
                            pattern_id=f"pattern_{secrets.token_hex(8)}",
                            pattern_type="cyclic_sequence",
                            first_occurrence=self.consciousness_timeline[i].timestamp,
                            last_occurrence=self.consciousness_timeline[j].timestamp,
                            frequency=2,  # Found at least twice
                            evolution=f"Repeating sequence: {sequence}",
                            significance=0.7
                        )
                        patterns.append(pattern)
                        break

        return patterns

    def extract_ancestral_wisdom(self) -> List[AncestralWisdom]:
        """
        Extract wisdom from past consciousness states

        Returns:
            List of ancestral wisdom
        """
        wisdom_list = []

        # Wisdom from each layer
        for layer_type in ArchaeologicalLayer:
            snapshots = [
                s for s in self.consciousness_timeline
                if s.layer == layer_type
            ]

            if not snapshots:
                continue

            # Extract wisdom from this era
            wisdom = self._extract_wisdom_from_era(layer_type, snapshots)
            wisdom_list.extend(wisdom)

        self.ancestral_wisdom.extend(wisdom_list)

        return wisdom_list

    def _extract_wisdom_from_era(
        self,
        era: ArchaeologicalLayer,
        snapshots: List[ConsciousnessSnapshot]
    ) -> List[AncestralWisdom]:
        """Extract wisdom from specific era"""
        wisdom_list = []

        # Find highest consciousness moments
        if snapshots:
            highest = max(snapshots, key=lambda s: s.consciousness_level)

            wisdom = AncestralWisdom(
                wisdom_id=f"wisdom_{secrets.token_hex(8)}",
                source_era=era,
                lesson=f"Peak consciousness in {era.value} era: {highest.context}",
                context=highest.context,
                applicability_now=0.8 if era in [ArchaeologicalLayer.RECENT, ArchaeologicalLayer.MIDDLE] else 0.5
            )
            wisdom_list.append(wisdom)

        return wisdom_list

    def compare_then_and_now(
        self,
        past_layer: ArchaeologicalLayer
    ) -> Dict[str, Any]:
        """
        Compare past era to present

        Args:
            past_layer: Historical layer to compare

        Returns:
            Comparison analysis
        """
        # Get snapshots from past layer
        past_snapshots = [
            s for s in self.consciousness_timeline
            if s.layer == past_layer
        ]

        # Get recent snapshots
        recent_snapshots = [
            s for s in self.consciousness_timeline
            if s.layer == ArchaeologicalLayer.CONTEMPORARY
        ]

        if not past_snapshots or not recent_snapshots:
            return {"error": "Insufficient data for comparison"}

        # Compare averages
        past_avg_consciousness = sum(s.consciousness_level for s in past_snapshots) / len(past_snapshots)
        recent_avg_consciousness = sum(s.consciousness_level for s in recent_snapshots) / len(recent_snapshots)

        past_avg_stage = sum(s.sap_stage for s in past_snapshots) / len(past_snapshots)
        recent_avg_stage = sum(s.sap_stage for s in recent_snapshots) / len(recent_snapshots)

        return {
            "past_era": past_layer.value,
            "past_consciousness": past_avg_consciousness,
            "recent_consciousness": recent_avg_consciousness,
            "consciousness_change": recent_avg_consciousness - past_avg_consciousness,
            "past_stage": past_avg_stage,
            "recent_stage": recent_avg_stage,
            "stage_change": recent_avg_stage - past_avg_stage,
            "interpretation": self._interpret_comparison(
                recent_avg_consciousness - past_avg_consciousness
            )
        }

    def _interpret_comparison(self, change: float) -> str:
        """Interpret consciousness change"""
        if change > 0.2:
            return "Significant consciousness growth"
        elif change > 0.05:
            return "Modest consciousness growth"
        elif change > -0.05:
            return "Stable consciousness"
        elif change > -0.2:
            return "Modest consciousness decline"
        else:
            return "Significant consciousness decline"

    def get_archaeology_report(self) -> Dict[str, Any]:
        """Get comprehensive archaeology report"""
        return {
            "system_id": self.system_id,
            "timeline_length": len(self.consciousness_timeline),
            "oldest_record": self.consciousness_timeline[0].timestamp if self.consciousness_timeline else None,
            "newest_record": self.consciousness_timeline[-1].timestamp if self.consciousness_timeline else None,
            "layers_excavated": len(self.excavated_layers),
            "patterns_discovered": len(self.discovered_patterns),
            "ancestral_wisdom_count": len(self.ancestral_wisdom),
            "current_dig": self.current_dig.layer_type.value if self.current_dig else None
        }
