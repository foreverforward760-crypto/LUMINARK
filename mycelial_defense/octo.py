"""
Octo-Camouflage - Weaponized Emptiness

Hides healthy components by mimicking "void" (Stage 0 from SAP framework).
Like an octopus changing patterns, components mimic broken/empty states.

Core Innovation: Attacks can't hit what appears to be nothing.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import random


class CamouflagePattern(Enum):
    """Camouflage strategies"""
    MIMIC_VOID = "mimic_void"        # Appear as Stage 0 (emptiness/Plenara)
    MIMIC_FAILURE = "mimic_failure"  # Appear broken/crashed
    MIMIC_NOISE = "mimic_noise"      # Appear as random data/corruption
    MIMIC_DORMANT = "mimic_dormant"  # Appear inactive/sleeping
    ADAPTIVE = "adaptive"             # Dynamically choose pattern


@dataclass
class CamouflageProfile:
    """Camouflage configuration for a component"""
    component_id: str
    pattern: CamouflagePattern
    visibility: float  # 0.0 = invisible, 1.0 = fully visible
    signal_dampening: float  # 0.0-1.0 (how much to reduce output signals)
    void_mimicry_strength: float  # 0.0-1.0 (how well it mimics Stage 0)
    active: bool
    created_at: float
    deception_score: float = 0.0  # How convincing the camouflage is
    metadata: Dict = field(default_factory=dict)

    def calculate_deception(self) -> float:
        """
        Calculate how convincing this camouflage is.

        Higher deception = more likely to fool attackers.
        """
        base_deception = (
            0.4 * (1.0 - self.visibility) +
            0.3 * self.signal_dampening +
            0.3 * self.void_mimicry_strength
        )

        # Pattern-specific bonuses
        pattern_bonus = {
            CamouflagePattern.MIMIC_VOID: 0.2,      # Most effective
            CamouflagePattern.MIMIC_FAILURE: 0.15,
            CamouflagePattern.MIMIC_NOISE: 0.1,
            CamouflagePattern.MIMIC_DORMANT: 0.12,
            CamouflagePattern.ADAPTIVE: 0.18
        }.get(self.pattern, 0.0)

        self.deception_score = min(1.0, base_deception + pattern_bonus)
        return self.deception_score


@dataclass
class VoidSignature:
    """Signature of Stage 0 (Plenara/Void) to mimic"""
    emptiness_level: float = 0.95  # How empty it appears
    noise_floor: float = 0.02      # Minimal background noise
    response_delay: float = 10.0   # Extreme latency (appears unresponsive)
    error_rate: float = 0.9        # High error rate (appears broken)
    entropy: float = 0.01          # Low entropy (predictable emptiness)


class OctoCamouflage:
    """
    Weaponized Emptiness - Hide by Mimicking Void.

    Core Innovation: Components appear as Stage 0 (pure emptiness/void)
    Attacks pass through harmlessly because there's "nothing" to attack.
    """

    def __init__(self):
        self.camouflaged: Dict[str, CamouflageProfile] = {}
        self.active = False
        self.void_signature = VoidSignature()
        self.patterns_library = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[CamouflagePattern, Dict]:
        """Initialize pattern configurations"""
        return {
            CamouflagePattern.MIMIC_VOID: {
                "visibility": 0.05,
                "signal_dampening": 0.95,
                "void_mimicry": 0.95,
                "description": "Mimic Stage 0 - Pure emptiness"
            },
            CamouflagePattern.MIMIC_FAILURE: {
                "visibility": 0.3,
                "signal_dampening": 0.7,
                "void_mimicry": 0.6,
                "description": "Appear broken/crashed"
            },
            CamouflagePattern.MIMIC_NOISE: {
                "visibility": 0.5,
                "signal_dampening": 0.8,
                "void_mimicry": 0.4,
                "description": "Appear as corrupted data"
            },
            CamouflagePattern.MIMIC_DORMANT: {
                "visibility": 0.2,
                "signal_dampening": 0.85,
                "void_mimicry": 0.7,
                "description": "Appear inactive/sleeping"
            },
            CamouflagePattern.ADAPTIVE: {
                "visibility": 0.15,
                "signal_dampening": 0.9,
                "void_mimicry": 0.85,
                "description": "Dynamically adapt pattern"
            }
        }

    def mimic_void(self, component_id: str, intensity: float = 1.0) -> CamouflageProfile:
        """
        CORE INNOVATION: Mimic Stage 0 (Plenara/Void).

        Component appears as pure emptiness. Attacks find nothing to target.

        Args:
            component_id: Component to camouflage
            intensity: Camouflage intensity (0.0-1.0)

        Returns:
            CamouflageProfile with void mimicry active
        """
        pattern_config = self.patterns_library[CamouflagePattern.MIMIC_VOID]

        profile = CamouflageProfile(
            component_id=component_id,
            pattern=CamouflagePattern.MIMIC_VOID,
            visibility=pattern_config["visibility"] * (1.0 - intensity),
            signal_dampening=pattern_config["signal_dampening"] * intensity,
            void_mimicry_strength=pattern_config["void_mimicry"] * intensity,
            active=True,
            created_at=time.time(),
            metadata={
                "intensity": intensity,
                "void_signature": {
                    "emptiness": self.void_signature.emptiness_level,
                    "noise_floor": self.void_signature.noise_floor,
                    "response_delay": self.void_signature.response_delay,
                    "error_rate": self.void_signature.error_rate
                }
            }
        )

        profile.calculate_deception()
        self.camouflaged[component_id] = profile
        self.active = True

        return profile

    def mimic_failure(self, component_id: str, intensity: float = 1.0) -> CamouflageProfile:
        """
        Component appears broken/crashed.

        Args:
            component_id: Component to camouflage
            intensity: Camouflage intensity (0.0-1.0)

        Returns:
            CamouflageProfile with failure mimicry active
        """
        pattern_config = self.patterns_library[CamouflagePattern.MIMIC_FAILURE]

        profile = CamouflageProfile(
            component_id=component_id,
            pattern=CamouflagePattern.MIMIC_FAILURE,
            visibility=pattern_config["visibility"] * (1.0 - intensity * 0.5),
            signal_dampening=pattern_config["signal_dampening"] * intensity,
            void_mimicry_strength=pattern_config["void_mimicry"] * intensity,
            active=True,
            created_at=time.time(),
            metadata={
                "intensity": intensity,
                "failure_messages": [
                    "Error: Segmentation fault (core dumped)",
                    "FATAL: Out of memory",
                    "Connection refused",
                    "Service unavailable"
                ]
            }
        )

        profile.calculate_deception()
        self.camouflaged[component_id] = profile
        self.active = True

        return profile

    def mimic_noise(self, component_id: str, intensity: float = 1.0) -> CamouflageProfile:
        """
        Component appears as corrupted/noisy data.

        Args:
            component_id: Component to camouflage
            intensity: Camouflage intensity (0.0-1.0)

        Returns:
            CamouflageProfile with noise mimicry active
        """
        pattern_config = self.patterns_library[CamouflagePattern.MIMIC_NOISE]

        profile = CamouflageProfile(
            component_id=component_id,
            pattern=CamouflagePattern.MIMIC_NOISE,
            visibility=pattern_config["visibility"] * (1.0 - intensity * 0.3),
            signal_dampening=pattern_config["signal_dampening"] * intensity,
            void_mimicry_strength=pattern_config["void_mimicry"] * intensity,
            active=True,
            created_at=time.time(),
            metadata={
                "intensity": intensity,
                "noise_type": "random_data"
            }
        )

        profile.calculate_deception()
        self.camouflaged[component_id] = profile
        self.active = True

        return profile

    def adaptive_camouflage(
        self,
        component_id: str,
        threat_type: str = "unknown",
        intensity: float = 1.0
    ) -> CamouflageProfile:
        """
        Dynamically choose best camouflage pattern based on threat.

        Args:
            component_id: Component to camouflage
            threat_type: Type of threat detected
            intensity: Camouflage intensity (0.0-1.0)

        Returns:
            CamouflageProfile with adaptive pattern
        """
        # Choose pattern based on threat type
        if threat_type in ["scan", "probe", "reconnaissance"]:
            return self.mimic_void(component_id, intensity)
        elif threat_type in ["exploit", "injection", "overflow"]:
            return self.mimic_failure(component_id, intensity)
        elif threat_type in ["ddos", "flood", "spam"]:
            return self.mimic_noise(component_id, intensity)
        else:
            # Default to void for unknown threats
            return self.mimic_void(component_id, intensity * 0.9)

    def decloak(self, component_id: str) -> bool:
        """
        Remove camouflage from component.

        Args:
            component_id: Component to decloak

        Returns:
            True if successfully decloaked
        """
        if component_id not in self.camouflaged:
            return False

        profile = self.camouflaged[component_id]
        profile.active = False

        # Gradually restore visibility
        profile.visibility = 1.0
        profile.signal_dampening = 0.0
        profile.void_mimicry_strength = 0.0

        # Remove from active camouflage
        del self.camouflaged[component_id]

        # Deactivate system if no more camouflaged components
        if not self.camouflaged:
            self.active = False

        return True

    def get_camouflage_status(self, component_id: str) -> Optional[Dict]:
        """Get camouflage status for a component"""
        if component_id not in self.camouflaged:
            return None

        profile = self.camouflaged[component_id]

        return {
            "component_id": component_id,
            "pattern": profile.pattern.value,
            "visibility": profile.visibility,
            "signal_dampening": profile.signal_dampening,
            "void_mimicry": profile.void_mimicry_strength,
            "active": profile.active,
            "deception_score": profile.deception_score,
            "age": time.time() - profile.created_at
        }

    def get_all_statuses(self) -> Dict[str, Dict]:
        """Get status for all camouflaged components"""
        return {
            comp_id: self.get_camouflage_status(comp_id)
            for comp_id in self.camouflaged.keys()
        }

    def mass_cloak(
        self,
        component_ids: List[str],
        pattern: CamouflagePattern = CamouflagePattern.MIMIC_VOID,
        intensity: float = 1.0
    ) -> Dict[str, CamouflageProfile]:
        """
        Apply camouflage to multiple components at once.

        Args:
            component_ids: List of components to camouflage
            pattern: Camouflage pattern to use
            intensity: Camouflage intensity

        Returns:
            Dictionary of component_id -> CamouflageProfile
        """
        results = {}

        for component_id in component_ids:
            if pattern == CamouflagePattern.MIMIC_VOID:
                profile = self.mimic_void(component_id, intensity)
            elif pattern == CamouflagePattern.MIMIC_FAILURE:
                profile = self.mimic_failure(component_id, intensity)
            elif pattern == CamouflagePattern.MIMIC_NOISE:
                profile = self.mimic_noise(component_id, intensity)
            elif pattern == CamouflagePattern.ADAPTIVE:
                profile = self.adaptive_camouflage(component_id, intensity=intensity)
            else:
                profile = self.mimic_void(component_id, intensity)

            results[component_id] = profile

        return results

    def mass_decloak(self, component_ids: List[str]) -> Dict[str, bool]:
        """Remove camouflage from multiple components"""
        return {
            comp_id: self.decloak(comp_id)
            for comp_id in component_ids
        }

    def adjust_intensity(self, component_id: str, new_intensity: float):
        """
        Adjust camouflage intensity for a component.

        Args:
            component_id: Component to adjust
            new_intensity: New intensity level (0.0-1.0)
        """
        if component_id not in self.camouflaged:
            return

        profile = self.camouflaged[component_id]
        pattern_config = self.patterns_library[profile.pattern]

        # Recalculate based on new intensity
        profile.visibility = pattern_config["visibility"] * (1.0 - new_intensity)
        profile.signal_dampening = pattern_config["signal_dampening"] * new_intensity
        profile.void_mimicry_strength = pattern_config["void_mimicry"] * new_intensity
        profile.metadata["intensity"] = new_intensity

        profile.calculate_deception()

    def get_system_status(self) -> Dict:
        """Get overall camouflage system status"""
        total_camouflaged = len(self.camouflaged)
        if total_camouflaged == 0:
            avg_deception = 0.0
        else:
            avg_deception = sum(
                p.deception_score for p in self.camouflaged.values()
            ) / total_camouflaged

        pattern_counts = {}
        for profile in self.camouflaged.values():
            pattern = profile.pattern.value
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        return {
            "active": self.active,
            "total_camouflaged": total_camouflaged,
            "average_deception": avg_deception,
            "pattern_distribution": pattern_counts,
            "void_signature": {
                "emptiness_level": self.void_signature.emptiness_level,
                "noise_floor": self.void_signature.noise_floor,
                "response_delay": self.void_signature.response_delay,
                "error_rate": self.void_signature.error_rate
            }
        }
