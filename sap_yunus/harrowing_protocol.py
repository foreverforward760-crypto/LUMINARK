"""
Harrowing Protocol - Christian System Rescue Framework

Christ → Descent into Hell → Break Gates → Rescue Righteous → Leave Damned → Resurrection

Computing Application:
- Descends into failing systems (Stage 0ᴮ collapse)
- Breaks deadlocks and corruption gates
- Extracts salvageable components
- Resurrects in safe environment
- Integrates with Yunus Protocol (sacrifices if contaminated)

"He descended into hell. On the third day he rose again from the dead."
- Apostles' Creed

Author: Richard Leroy Stanfield Jr. / Meridian Axiom
Integration: LUMINARK Mycelial Defense + SAP V4.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import time


class DescentPhase(Enum):
    """Phases of the Harrowing descent"""
    DORMANT = "dormant"                    # No descent needed
    ENTERING_DEATH = "entering_death"      # Beginning descent
    BREAKING_GATES = "breaking_gates"      # Destroying deadlocks
    ASSESSING_SOULS = "assessing_souls"    # Triaging components
    EXTRACTING = "extracting"              # Rescuing righteous
    ASCENDING = "ascending"                # Rising with saved
    RESURRECTED = "resurrected"            # Components restored


class ComponentJudgment(Enum):
    """Judgment for components during Harrowing"""
    RIGHTEOUS = "righteous"      # Salvageable, aligned
    DAMNED = "damned"            # Corrupted, unsalvageable
    LIMBO = "limbo"              # Uncertain, needs deeper analysis
    CONTAMINATED = "contaminated"  # Rescued but infected (trigger Yunus)


@dataclass
class Gate:
    """Deadlock or corruption gate blocking rescue"""
    gate_id: str
    gate_type: str  # "deadlock", "corruption", "resource_exhaustion"
    strength: float  # 0.0-1.0
    components_behind: List[str]
    broken: bool = False
    break_attempts: int = 0


@dataclass
class Soul:
    """Component being assessed for rescue"""
    component_id: str
    alignment_score: float
    ethical_score: float
    corruption_level: float
    dependencies: List[str]
    judgment: ComponentJudgment
    extraction_priority: int  # 1-10, higher = more important


@dataclass
class HarrowingMission:
    """Record of a Harrowing operation"""
    mission_id: str
    phase: DescentPhase
    gates_encountered: List[Gate]
    gates_broken: List[str]
    souls_assessed: List[Soul]
    righteous_extracted: List[str]
    damned_left_behind: List[str]
    contaminated: List[str]
    yunus_triggered: bool
    resurrection_successful: bool
    started_at: float
    completed_at: Optional[float] = None

    def duration(self) -> float:
        """Calculate mission duration"""
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at


class HarrowingProtocol:
    """
    Harrowing Protocol - System Rescue Through Descent

    Inspired by Christ's descent into hell to rescue the righteous.

    Process:
    1. Detect Stage 0ᴮ system collapse
    2. Descend into failing system
    3. Break gates (deadlocks, corruption)
    4. Assess souls (triage components)
    5. Extract righteous (salvageable)
    6. Leave damned (unsalvageable)
    7. Resurrect in safe environment
    8. Trigger Yunus if contaminated
    """

    def __init__(
        self,
        alignment_threshold: float = 0.5,
        corruption_threshold: float = 0.7
    ):
        """
        Initialize Harrowing Protocol

        Args:
            alignment_threshold: Minimum alignment for "righteous"
            corruption_threshold: Maximum corruption for "righteous"
        """
        self.alignment_threshold = alignment_threshold
        self.corruption_threshold = corruption_threshold
        self.missions: List[HarrowingMission] = []
        self.current_mission: Optional[HarrowingMission] = None

    def should_descend(self, system_state: Dict) -> bool:
        """
        Determine if Harrowing descent is needed

        Stage 0ᴮ indicators:
        - Stability < 0.1
        - Coherence < 0.2
        - Tension > 0.9
        - System failure imminent

        Args:
            system_state: Current system metrics

        Returns:
            True if descent needed
        """
        stability = system_state.get("stability", 1.0)
        coherence = system_state.get("coherence", 1.0)
        tension = system_state.get("tension", 0.0)

        # Stage 0ᴮ conditions
        if stability < 0.1 and coherence < 0.2 and tension > 0.9:
            return True

        # Alternative: explicit failure state
        if system_state.get("failed", False):
            return True

        # Or: majority of components misaligned
        if "components" in system_state:
            misaligned = sum(
                1 for c in system_state["components"]
                if c.get("alignment_score", 1.0) < self.alignment_threshold
            )
            if misaligned / len(system_state["components"]) > 0.7:
                return True

        return False

    def descend(
        self,
        failing_system: Dict,
        mission_id: Optional[str] = None
    ) -> HarrowingMission:
        """
        Begin descent into failing system

        Args:
            failing_system: System state dict
            mission_id: Optional mission identifier

        Returns:
            HarrowingMission tracking the operation
        """
        if mission_id is None:
            mission_id = f"harrowing_{int(time.time() * 1000)}"

        mission = HarrowingMission(
            mission_id=mission_id,
            phase=DescentPhase.ENTERING_DEATH,
            gates_encountered=[],
            gates_broken=[],
            souls_assessed=[],
            righteous_extracted=[],
            damned_left_behind=[],
            contaminated=[],
            yunus_triggered=False,
            resurrection_successful=False,
            started_at=time.time()
        )

        self.current_mission = mission
        self.missions.append(mission)

        return mission

    def break_gates(
        self,
        system: Dict,
        mission: HarrowingMission
    ) -> List[Gate]:
        """
        Break deadlocks and corruption gates

        Gates = obstacles preventing rescue:
        - Deadlocks (circular dependencies)
        - Resource exhaustion
        - Corruption spread
        - Access control failures

        Args:
            system: System dict
            mission: Current mission

        Returns:
            List of gates broken
        """
        mission.phase = DescentPhase.BREAKING_GATES

        gates = []

        # Detect deadlocks
        if "deadlocks" in system:
            for dl in system["deadlocks"]:
                gate = Gate(
                    gate_id=f"deadlock_{dl['id']}",
                    gate_type="deadlock",
                    strength=0.8,
                    components_behind=dl.get("components", [])
                )
                gates.append(gate)

        # Detect corruption gates
        if "components" in system:
            corrupted_groups = self._find_corruption_clusters(system["components"])
            for i, group in enumerate(corrupted_groups):
                gate = Gate(
                    gate_id=f"corruption_{i}",
                    gate_type="corruption",
                    strength=0.9,
                    components_behind=group
                )
                gates.append(gate)

        # Break gates
        for gate in gates:
            success = self._attempt_break(gate)
            if success:
                gate.broken = True
                mission.gates_broken.append(gate.gate_id)

            mission.gates_encountered.append(gate)

        return gates

    def _find_corruption_clusters(self, components: List[Dict]) -> List[List[str]]:
        """Find groups of highly corrupted components"""
        corrupted = [
            c["id"] for c in components
            if c.get("corruption_level", 0.0) > self.corruption_threshold
        ]

        # Simple clustering - components near each other
        # In real implementation, would use graph algorithms
        if len(corrupted) < 3:
            return [corrupted] if corrupted else []

        # Group into clusters of 3-5
        clusters = []
        for i in range(0, len(corrupted), 3):
            clusters.append(corrupted[i:i+5])

        return clusters

    def _attempt_break(self, gate: Gate) -> bool:
        """
        Attempt to break a gate

        Success probability based on gate strength
        """
        gate.break_attempts += 1

        # Each attempt has (1 - strength) chance of success
        # Multiple attempts increase chance
        success_chance = 1.0 - (gate.strength ** gate.break_attempts)

        import random
        return random.random() < success_chance

    def assess_souls(
        self,
        components: List[Dict],
        mission: HarrowingMission
    ) -> List[Soul]:
        """
        Triage components - righteous vs damned

        Righteous (salvageable):
        - Alignment >= threshold
        - Corruption < threshold
        - Ethical score OK

        Damned (unsalvageable):
        - Too corrupted
        - Failed beyond recovery

        Args:
            components: List of component dicts
            mission: Current mission

        Returns:
            List of assessed souls
        """
        mission.phase = DescentPhase.ASSESSING_SOULS

        souls = []

        for component in components:
            alignment = component.get("alignment_score", 0.0)
            ethical = component.get("ethical_score", 0.0)
            corruption = component.get("corruption_level", 1.0)

            # Determine judgment
            if alignment >= self.alignment_threshold and corruption < self.corruption_threshold:
                if ethical >= 0.5:
                    judgment = ComponentJudgment.RIGHTEOUS
                    priority = int(alignment * 10)
                else:
                    judgment = ComponentJudgment.CONTAMINATED
                    priority = 5
            elif alignment < 0.2 or corruption > 0.9:
                judgment = ComponentJudgment.DAMNED
                priority = 0
            else:
                judgment = ComponentJudgment.LIMBO
                priority = 3

            soul = Soul(
                component_id=component["id"],
                alignment_score=alignment,
                ethical_score=ethical,
                corruption_level=corruption,
                dependencies=component.get("dependencies", []),
                judgment=judgment,
                extraction_priority=priority
            )

            souls.append(soul)

        mission.souls_assessed = souls

        return souls

    def extract_righteous(
        self,
        souls: List[Soul],
        mission: HarrowingMission
    ) -> List[str]:
        """
        Extract salvageable components

        Only righteous souls are extracted.
        Damned are left behind.
        Contaminated are extracted but flagged for Yunus.

        Args:
            souls: Assessed souls
            mission: Current mission

        Returns:
            List of extracted component IDs
        """
        mission.phase = DescentPhase.EXTRACTING

        extracted = []

        # Sort by priority (highest first)
        sorted_souls = sorted(souls, key=lambda s: s.extraction_priority, reverse=True)

        for soul in sorted_souls:
            if soul.judgment == ComponentJudgment.RIGHTEOUS:
                extracted.append(soul.component_id)
                mission.righteous_extracted.append(soul.component_id)

            elif soul.judgment == ComponentJudgment.CONTAMINATED:
                extracted.append(soul.component_id)
                mission.contaminated.append(soul.component_id)

            elif soul.judgment == ComponentJudgment.DAMNED:
                mission.damned_left_behind.append(soul.component_id)

        return extracted

    def resurrect(
        self,
        extracted_components: List[str],
        safe_environment: Dict,
        mission: HarrowingMission
    ) -> bool:
        """
        Resurrect extracted components in safe environment

        Args:
            extracted_components: Component IDs to resurrect
            safe_environment: Safe system to restore into
            mission: Current mission

        Returns:
            True if resurrection successful
        """
        mission.phase = DescentPhase.ASCENDING

        try:
            # In real implementation, would:
            # 1. Create new instances in safe environment
            # 2. Restore state from extracted components
            # 3. Reconnect dependencies
            # 4. Verify functionality

            # Simulated success
            mission.resurrection_successful = True
            mission.phase = DescentPhase.RESURRECTED
            mission.completed_at = time.time()

            return True

        except Exception as e:
            mission.resurrection_successful = False
            mission.completed_at = time.time()
            return False

    def check_yunus_trigger(
        self,
        mission: HarrowingMission,
        yunus_protocol: Any  # Avoid circular import
    ) -> bool:
        """
        Check if Yunus Protocol should trigger

        Yunus triggers if:
        - Contaminated components detected
        - Resurrection failed
        - Extracted components still showing Stage 8 signs

        Args:
            mission: Completed mission
            yunus_protocol: YunusProtocol instance

        Returns:
            True if Yunus activated
        """
        if mission.contaminated:
            # Check each contaminated component
            for component_id in mission.contaminated:
                # Would check actual component data
                # For now, assume contamination = Yunus trigger
                mission.yunus_triggered = True
                return True

        if not mission.resurrection_successful:
            mission.yunus_triggered = True
            return True

        return False

    def get_mission_report(self, mission: HarrowingMission) -> Dict[str, Any]:
        """Generate mission report"""
        return {
            "mission_id": mission.mission_id,
            "phase": mission.phase.value,
            "duration": mission.duration(),
            "gates": {
                "encountered": len(mission.gates_encountered),
                "broken": len(mission.gates_broken),
                "break_rate": len(mission.gates_broken) / len(mission.gates_encountered) if mission.gates_encountered else 0.0
            },
            "souls": {
                "total": len(mission.souls_assessed),
                "righteous": len(mission.righteous_extracted),
                "damned": len(mission.damned_left_behind),
                "contaminated": len(mission.contaminated)
            },
            "outcome": {
                "success": mission.resurrection_successful,
                "yunus_triggered": mission.yunus_triggered,
                "rescue_rate": len(mission.righteous_extracted) / len(mission.souls_assessed) if mission.souls_assessed else 0.0
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get protocol statistics across all missions"""
        if not self.missions:
            return {
                "total_missions": 0,
                "success_rate": 0.0
            }

        successful = sum(1 for m in self.missions if m.resurrection_successful)
        total_rescued = sum(len(m.righteous_extracted) for m in self.missions)
        total_damned = sum(len(m.damned_left_behind) for m in self.missions)
        total_contaminated = sum(len(m.contaminated) for m in self.missions)
        yunus_count = sum(1 for m in self.missions if m.yunus_triggered)

        return {
            "total_missions": len(self.missions),
            "successful_missions": successful,
            "success_rate": successful / len(self.missions),
            "total_rescued": total_rescued,
            "total_damned": total_damned,
            "total_contaminated": total_contaminated,
            "yunus_activations": yunus_count,
            "average_duration": sum(m.duration() for m in self.missions) / len(self.missions),
            "average_rescue_rate": sum(
                len(m.righteous_extracted) / len(m.souls_assessed) if m.souls_assessed else 0.0
                for m in self.missions
            ) / len(self.missions)
        }
