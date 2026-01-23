"""
Mycelial Defense System - Integrated Orchestration

Complete defense system with intelligent trigger logic based on SPAT vectors.
Orchestrates AlignmentDetector, MycelialNetwork, and OctoCamouflage.

Defense Modes:
1. OCTO_CAMOUFLAGE: High tension + Low coherence → Hide in void
2. MYCELIAL_WRAP: Low stability + High tension → Surround and isolate
3. FULL_HARROWING: Critical collapse → Complete rescue operation
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import time
import logging

from .alignment import AlignmentDetector, AlignmentStatus, ComponentSignature
from .mycelial import MycelialNetwork, MycelialWall, ContainmentZone
from .octo import OctoCamouflage, CamouflagePattern, CamouflageProfile
from .sap import SAPCalculator, SPATVectors


class DefenseMode(Enum):
    """Defense operation modes"""
    DORMANT = "dormant"                      # No active threats
    OCTO_CAMOUFLAGE = "octo_camouflage"      # Hide healthy components
    MYCELIAL_WRAP = "mycelial_wrap"          # Surround and contain
    FULL_HARROWING = "full_harrowing"        # Complete rescue operation
    MONITORING = "monitoring"                # Active monitoring only


@dataclass
class DefenseAction:
    """Record of a defense action taken"""
    action_id: str
    mode: DefenseMode
    trigger: str
    components_affected: List[str]
    timestamp: float
    spat_vectors: SPATVectors
    success: bool
    metadata: Dict = field(default_factory=dict)


@dataclass
class ThreatAssessment:
    """Assessment of current threat level"""
    threat_level: float  # 0.0-1.0
    recommended_mode: DefenseMode
    trigger_conditions: List[str]
    spat_vectors: SPATVectors
    timestamp: float
    analysis: Dict = field(default_factory=dict)


class MycelialDefenseSystem:
    """
    Complete integrated defense with intelligent trigger logic.

    Combines immune detection, fungal containment, and weaponized emptiness
    into a unified defense system triggered by SAP SPAT vectors.
    """

    def __init__(self, system_id: str, alignment_threshold: float = 0.7):
        """
        Initialize defense system.

        Args:
            system_id: Unique system identifier
            alignment_threshold: Alignment score threshold (0.0-1.0)
        """
        self.system_id = system_id
        self.detector = AlignmentDetector(alignment_threshold)
        self.mycelium = MycelialNetwork()
        self.octo = OctoCamouflage()
        self.sap = SAPCalculator()

        self.mode = DefenseMode.DORMANT
        self.active = False
        self.history: List[DefenseAction] = []
        self.logger = logging.getLogger(f"mycelial_defense.{system_id}")

        # Defense trigger thresholds
        self.thresholds = {
            "octo_camouflage": {
                "tension_min": 0.8,
                "coherence_max": 0.3
            },
            "mycelial_wrap": {
                "stability_max": 0.2,
                "tension_min": 0.7
            },
            "full_harrowing": {
                "stability_max": 0.1,
                "coherence_max": 0.2,
                "tension_min": 0.9
            }
        }

    def assess_threat(
        self,
        complexity: float,
        stability: float,
        tension: float,
        adaptability: float,
        coherence: float
    ) -> ThreatAssessment:
        """
        Assess threat level and determine defense mode from SPAT vectors.

        Trigger Logic:
        1. OCTO_CAMOUFLAGE: tension > 0.8 AND coherence < 0.3
           (High velocity + Lost truth = Hide)

        2. MYCELIAL_WRAP: stability < 0.2 AND tension > 0.7
           (Collapsing + Overload = Surround)

        3. FULL_HARROWING: stability < 0.1 AND coherence < 0.2 AND tension > 0.9
           (Total collapse = Rescue)

        Args:
            complexity: System complexity (0.0-1.0)
            stability: System stability (0.0-1.0)
            tension: System tension (0.0-1.0)
            adaptability: System adaptability (0.0-1.0)
            coherence: System coherence (0.0-1.0)

        Returns:
            ThreatAssessment with recommended defense mode
        """
        vectors = self.sap.calculate_from_metrics(
            complexity, stability, tension, adaptability, coherence
        )

        trigger_conditions = []
        recommended_mode = DefenseMode.DORMANT
        threat_level = 0.0

        # Check FULL_HARROWING first (most critical)
        if (stability < self.thresholds["full_harrowing"]["stability_max"] and
            coherence < self.thresholds["full_harrowing"]["coherence_max"] and
            tension > self.thresholds["full_harrowing"]["tension_min"]):

            trigger_conditions.append("Critical collapse detected")
            trigger_conditions.append(f"Stability: {stability:.2f} < 0.1")
            trigger_conditions.append(f"Coherence: {coherence:.2f} < 0.2")
            trigger_conditions.append(f"Tension: {tension:.2f} > 0.9")
            recommended_mode = DefenseMode.FULL_HARROWING
            threat_level = 1.0

        # Check MYCELIAL_WRAP (high severity)
        elif (stability < self.thresholds["mycelial_wrap"]["stability_max"] and
              tension > self.thresholds["mycelial_wrap"]["tension_min"]):

            trigger_conditions.append("System overload detected")
            trigger_conditions.append(f"Stability: {stability:.2f} < 0.2")
            trigger_conditions.append(f"Tension: {tension:.2f} > 0.7")
            recommended_mode = DefenseMode.MYCELIAL_WRAP
            threat_level = 0.8

        # Check OCTO_CAMOUFLAGE (moderate severity)
        elif (tension > self.thresholds["octo_camouflage"]["tension_min"] and
              coherence < self.thresholds["octo_camouflage"]["coherence_max"]):

            trigger_conditions.append("High velocity with lost direction")
            trigger_conditions.append(f"Tension: {tension:.2f} > 0.8")
            trigger_conditions.append(f"Coherence: {coherence:.2f} < 0.3")
            recommended_mode = DefenseMode.OCTO_CAMOUFLAGE
            threat_level = 0.6

        # Low-level monitoring
        elif tension > 0.5 or coherence < 0.5:
            trigger_conditions.append("Elevated metrics detected")
            recommended_mode = DefenseMode.MONITORING
            threat_level = 0.3

        analysis = self.sap.get_analysis(vectors)

        return ThreatAssessment(
            threat_level=threat_level,
            recommended_mode=recommended_mode,
            trigger_conditions=trigger_conditions,
            spat_vectors=vectors,
            timestamp=time.time(),
            analysis=analysis
        )

    def execute_defense(
        self,
        components: List[Dict],
        threat_assessment: Optional[ThreatAssessment] = None
    ) -> DefenseAction:
        """
        Execute defense based on threat assessment.

        Args:
            components: List of component dictionaries (must have 'id' field)
            threat_assessment: Optional pre-computed threat assessment

        Returns:
            DefenseAction record
        """
        if not threat_assessment:
            # Calculate SPAT vectors from component state
            alignment_scores = {}
            resource_usage = {}

            for comp in components:
                comp_id = comp['id']
                # Get or estimate alignment
                if 'alignment_score' in comp:
                    alignment_scores[comp_id] = comp['alignment_score']
                else:
                    alignment_scores[comp_id] = 1.0

                # Get or estimate resources
                if 'resource_usage' in comp:
                    resource_usage[comp_id] = comp['resource_usage']
                else:
                    resource_usage[comp_id] = 0.5

            vectors = self.sap.calculate_from_components(
                components, alignment_scores, resource_usage
            )

            threat_assessment = self.assess_threat(
                vectors.complexity,
                vectors.stability,
                vectors.tension,
                vectors.adaptability,
                vectors.coherence
            )

        # Execute appropriate defense mode
        if threat_assessment.recommended_mode == DefenseMode.FULL_HARROWING:
            return self._activate_full_harrowing(components, threat_assessment)

        elif threat_assessment.recommended_mode == DefenseMode.MYCELIAL_WRAP:
            return self._activate_mycelial_wrap(components, threat_assessment)

        elif threat_assessment.recommended_mode == DefenseMode.OCTO_CAMOUFLAGE:
            return self._activate_octo_camouflage(components, threat_assessment)

        elif threat_assessment.recommended_mode == DefenseMode.MONITORING:
            return self._activate_monitoring(components, threat_assessment)

        else:
            return self._remain_dormant(threat_assessment)

    def _activate_octo_camouflage(
        self,
        components: List[Dict],
        threat: ThreatAssessment
    ) -> DefenseAction:
        """Activate Octo-Camouflage defense"""
        self.logger.info("Activating OCTO_CAMOUFLAGE defense")
        self.mode = DefenseMode.OCTO_CAMOUFLAGE

        # Find aligned components to protect
        aligned_components = [
            comp for comp in components
            if comp.get('alignment_score', 1.0) >= self.detector.alignment_threshold
        ]

        component_ids = [comp['id'] for comp in aligned_components]

        # Apply void mimicry to healthy components
        profiles = self.octo.mass_cloak(
            component_ids,
            pattern=CamouflagePattern.MIMIC_VOID,
            intensity=0.95
        )

        action = DefenseAction(
            action_id=f"action_{int(time.time() * 1000)}",
            mode=DefenseMode.OCTO_CAMOUFLAGE,
            trigger="High tension + Low coherence",
            components_affected=component_ids,
            timestamp=time.time(),
            spat_vectors=threat.spat_vectors,
            success=True,
            metadata={
                "camouflaged_count": len(profiles),
                "avg_deception": sum(p.deception_score for p in profiles.values()) / len(profiles) if profiles else 0.0
            }
        )

        self.history.append(action)
        self.active = True

        return action

    def _activate_mycelial_wrap(
        self,
        components: List[Dict],
        threat: ThreatAssessment
    ) -> DefenseAction:
        """Activate Mycelial Wrap defense"""
        self.logger.info("Activating MYCELIAL_WRAP defense")
        self.mode = DefenseMode.MYCELIAL_WRAP

        # Get alignment scores
        alignment_scores = {
            comp['id']: comp.get('alignment_score', 1.0)
            for comp in components
        }

        # Detect misaligned zones
        zone = self.mycelium.detect_zone(
            components,
            alignment_scores,
            threshold=self.detector.alignment_threshold
        )

        walls_created = []
        if zone:
            # Surround with containment wall
            wall = self.mycelium.surround_zone(zone)
            walls_created.append(wall.wall_id)

        action = DefenseAction(
            action_id=f"action_{int(time.time() * 1000)}",
            mode=DefenseMode.MYCELIAL_WRAP,
            trigger="Low stability + High tension",
            components_affected=zone.components if zone else [],
            timestamp=time.time(),
            spat_vectors=threat.spat_vectors,
            success=True,
            metadata={
                "zones_created": 1 if zone else 0,
                "walls_created": len(walls_created),
                "contained_count": len(zone.components) if zone else 0
            }
        )

        self.history.append(action)
        self.active = True

        return action

    def _activate_full_harrowing(
        self,
        components: List[Dict],
        threat: ThreatAssessment
    ) -> DefenseAction:
        """
        Activate Full Harrowing - Complete rescue operation.

        Steps:
        1. Surround all misaligned zones with Mycelium
        2. Camouflage all aligned components
        3. Create extraction pathways
        4. Extract healthy components to safe zone
        """
        self.logger.warning("Activating FULL_HARROWING - Critical rescue operation")
        self.mode = DefenseMode.FULL_HARROWING

        alignment_scores = {
            comp['id']: comp.get('alignment_score', 1.0)
            for comp in components
        }

        # Step 1: Surround misaligned zones
        zone = self.mycelium.detect_zone(
            components,
            alignment_scores,
            threshold=self.detector.alignment_threshold
        )

        walls = []
        if zone:
            wall = self.mycelium.surround_zone(zone)
            walls.append(wall)

        # Step 2: Camouflage aligned components
        aligned_components = [
            comp for comp in components
            if comp.get('alignment_score', 1.0) >= self.detector.alignment_threshold
        ]

        aligned_ids = [comp['id'] for comp in aligned_components]
        profiles = self.octo.mass_cloak(
            aligned_ids,
            pattern=CamouflagePattern.MIMIC_VOID,
            intensity=1.0  # Maximum camouflage
        )

        # Step 3: Create extraction pathways
        pathways = []
        if zone:
            pathway = self.mycelium.create_pathway(
                from_zone=zone.zone_id,
                to_zone="safe_zone",
                bandwidth=1.0,
                encrypted=True,
                hidden=True
            )
            pathways.append(pathway)

        # Step 4: Extract healthy components
        extracted = []
        for comp_id in aligned_ids:
            if self.mycelium.extract_component(comp_id, "safe_zone"):
                extracted.append(comp_id)

        action = DefenseAction(
            action_id=f"action_{int(time.time() * 1000)}",
            mode=DefenseMode.FULL_HARROWING,
            trigger="Critical collapse - Total system failure imminent",
            components_affected=aligned_ids + (zone.components if zone else []),
            timestamp=time.time(),
            spat_vectors=threat.spat_vectors,
            success=True,
            metadata={
                "zones_surrounded": len(walls),
                "components_camouflaged": len(profiles),
                "pathways_created": len(pathways),
                "components_extracted": len(extracted),
                "rescue_rate": len(extracted) / len(aligned_ids) if aligned_ids else 0.0
            }
        )

        self.history.append(action)
        self.active = True

        return action

    def _activate_monitoring(
        self,
        components: List[Dict],
        threat: ThreatAssessment
    ) -> DefenseAction:
        """Activate monitoring mode"""
        self.mode = DefenseMode.MONITORING

        action = DefenseAction(
            action_id=f"action_{int(time.time() * 1000)}",
            mode=DefenseMode.MONITORING,
            trigger="Elevated metrics",
            components_affected=[comp['id'] for comp in components],
            timestamp=time.time(),
            spat_vectors=threat.spat_vectors,
            success=True,
            metadata={
                "component_count": len(components)
            }
        )

        self.history.append(action)
        self.active = True

        return action

    def _remain_dormant(self, threat: ThreatAssessment) -> DefenseAction:
        """Remain in dormant mode"""
        self.mode = DefenseMode.DORMANT

        action = DefenseAction(
            action_id=f"action_{int(time.time() * 1000)}",
            mode=DefenseMode.DORMANT,
            trigger="No threats detected",
            components_affected=[],
            timestamp=time.time(),
            spat_vectors=threat.spat_vectors,
            success=True
        )

        self.history.append(action)
        self.active = False

        return action

    def get_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "system_id": self.system_id,
            "mode": self.mode.value,
            "active": self.active,
            "alignment_detector": {
                "registered_components": len(self.detector.known_signatures),
                "threshold": self.detector.alignment_threshold,
                "health": self.detector.get_all_health()
            },
            "mycelial_network": self.mycelium.get_network_status(),
            "octo_camouflage": self.octo.get_system_status(),
            "recent_actions": [
                {
                    "mode": action.mode.value,
                    "trigger": action.trigger,
                    "components_affected": len(action.components_affected),
                    "success": action.success,
                    "timestamp": action.timestamp
                }
                for action in self.history[-5:]
            ],
            "total_actions": len(self.history)
        }

    def reset(self):
        """Reset defense system to initial state"""
        self.mode = DefenseMode.DORMANT
        self.active = False
        self.octo.mass_decloak(list(self.octo.camouflaged.keys()))
        self.logger.info("Defense system reset")
