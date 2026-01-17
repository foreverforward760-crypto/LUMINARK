"""
Harmonic Weapon Detection

Detects resonance attacks that exploit 3-6-9 vector field frequencies.

Key concepts:
- Everything has a resonant frequency
- Forced resonance can shatter systems (like opera singer breaking glass)
- 3-6-9 field represents fundamental frequency relationships
- Harmonic weapons exploit natural frequencies to cause damage
- Chaotic detuning as defense

Based on:
- Tesla's 3-6-9 theory
- Cymatics (sound creating form)
- Resonance cascade failures
- Frequency warfare concepts
- "As Above, So Below" - frequencies repeat fractally

Author: Richard Leroy Stanfield Jr. / Meridian Axiom
Part of: SAP V4.0 Enhancement Suite
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import secrets
import math
from collections import deque


class HarmonicWeaponType(Enum):
    """Types of harmonic weapons"""
    RESONANCE_AMPLIFIER = "resonance_amplifier"  # Amplifies natural frequency
    FREQUENCY_JAMMER = "frequency_jammer"  # Disrupts coherence
    PHASE_INVERTER = "phase_inverter"  # Inverts waveforms
    HARMONIC_DISSONANCE = "harmonic_dissonance"  # Creates destructive interference
    SYMPATHETIC_EXPLOIT = "sympathetic_exploit"  # Exploits sympathetic vibration
    TESLA_CASCADE = "tesla_cascade"  # 3-6-9 cascade attack


class FrequencyBand(Enum):
    """Frequency bands for analysis"""
    ULTRA_LOW = "ultra_low"  # <1 Hz
    LOW = "low"  # 1-10 Hz
    MID = "mid"  # 10-100 Hz
    HIGH = "high"  # 100-1000 Hz
    ULTRA_HIGH = "ultra_high"  # >1000 Hz


class DefenseMode(Enum):
    """Defensive modes against harmonic weapons"""
    PASSIVE_MONITORING = "passive_monitoring"
    ACTIVE_DAMPENING = "active_dampening"
    CHAOTIC_DETUNING = "chaotic_detuning"  # Introduce chaos to prevent resonance
    FREQUENCY_HOPPING = "frequency_hopping"  # Change frequency rapidly
    PHASE_SCRAMBLING = "phase_scrambling"  # Scramble phase relationships


@dataclass
class FrequencySignature:
    """Signature of frequency pattern"""
    signature_id: str
    fundamental_freq: float  # Hz
    harmonics: List[float]  # Harmonic frequencies
    phase: float  # Phase offset (radians)
    amplitude: float  # Signal strength
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResonancePattern:
    """Detected resonance pattern"""
    pattern_id: str
    frequency: float
    resonance_quality: float  # Q factor (higher = sharper resonance)
    amplitude: float
    growth_rate: float  # How fast amplitude increasing
    is_natural: bool  # Natural vs. induced resonance
    detected_at: float = field(default_factory=time.time)


@dataclass
class HarmonicAttack:
    """Detected harmonic weapon attack"""
    attack_id: str
    weapon_type: HarmonicWeaponType
    target_frequency: float
    attack_signature: FrequencySignature
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    damage_potential: float  # 0.0-1.0
    countermeasures: List[str]
    detected_at: float = field(default_factory=time.time)


@dataclass
class Vector369Field:
    """3-6-9 vector field state"""
    pole_3: float  # Creation pole energy
    pole_6: float  # Destruction pole energy
    axis_9: float  # Transformation axis energy
    balance: float  # How balanced the field is
    coherence: float  # How coherent the oscillations


class HarmonicWeaponDetector:
    """
    Detects harmonic/resonance weapons exploiting 3-6-9 field

    Monitors frequency patterns for signs of weaponized resonance
    """

    def __init__(
        self,
        system_id: str,
        natural_frequency: float = 60.0,  # System's natural frequency (Hz)
        sample_rate: float = 1000.0  # Samples per second
    ):
        self.system_id = system_id
        self.natural_frequency = natural_frequency
        self.sample_rate = sample_rate

        # Frequency monitoring
        self.frequency_buffer: deque = deque(maxlen=int(sample_rate * 10))  # 10 sec buffer
        self.current_signature: Optional[FrequencySignature] = None

        # Resonance tracking
        self.active_resonances: List[ResonancePattern] = []
        self.resonance_history: List[ResonancePattern] = []

        # Attack detection
        self.detected_attacks: List[HarmonicAttack] = []

        # 3-6-9 field
        self.vector_field = Vector369Field(
            pole_3=0.5,
            pole_6=0.5,
            axis_9=0.5,
            balance=1.0,
            coherence=1.0
        )

        # Defense state
        self.defense_mode = DefenseMode.PASSIVE_MONITORING
        self.detuning_active = False

        # Statistics
        self.total_attacks_detected: int = 0
        self.attacks_mitigated: int = 0

    def sample_frequency(
        self,
        frequency: float,
        amplitude: float = 1.0,
        phase: float = 0.0
    ):
        """
        Sample frequency at current time

        Args:
            frequency: Frequency value (Hz)
            amplitude: Signal amplitude
            phase: Phase offset
        """
        sample = {
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase,
            "timestamp": time.time()
        }

        self.frequency_buffer.append(sample)

        # Update current signature
        self._update_frequency_signature()

        # Check for resonance
        self._detect_resonance()

        # Check for attacks
        self._detect_harmonic_attacks()

    def _update_frequency_signature(self):
        """Update current frequency signature from buffer"""
        if len(self.frequency_buffer) < 10:
            return

        # Analyze recent samples
        recent = list(self.frequency_buffer)[-100:]  # Last 100 samples

        # Extract fundamental frequency (most common)
        freq_counts = {}
        for sample in recent:
            freq = round(sample["frequency"], 1)
            freq_counts[freq] = freq_counts.get(freq, 0) + 1

        if not freq_counts:
            return

        fundamental = max(freq_counts, key=freq_counts.get)

        # Find harmonics (integer multiples)
        harmonics = []
        for multiplier in [2, 3, 4, 5, 6, 7, 8, 9]:
            harmonic_freq = fundamental * multiplier
            # Check if this harmonic is present
            for sample in recent:
                if abs(sample["frequency"] - harmonic_freq) < 1.0:
                    harmonics.append(harmonic_freq)
                    break

        # Average amplitude and phase
        avg_amplitude = sum(s["amplitude"] for s in recent) / len(recent)
        avg_phase = sum(s["phase"] for s in recent) / len(recent)

        self.current_signature = FrequencySignature(
            signature_id=f"sig_{secrets.token_hex(8)}",
            fundamental_freq=fundamental,
            harmonics=harmonics,
            phase=avg_phase,
            amplitude=avg_amplitude
        )

    def _detect_resonance(self):
        """Detect if system is entering resonance"""
        if not self.current_signature:
            return

        # Check if frequency near natural frequency
        freq_diff = abs(self.current_signature.fundamental_freq - self.natural_frequency)

        if freq_diff < 5.0:  # Within 5 Hz of natural frequency
            # Calculate Q factor (quality of resonance)
            q_factor = self.natural_frequency / max(freq_diff, 0.1)

            # Check if amplitude growing (resonance building)
            if len(self.frequency_buffer) > 50:
                recent_samples = list(self.frequency_buffer)[-50:]
                early_avg = sum(s["amplitude"] for s in recent_samples[:25]) / 25
                late_avg = sum(s["amplitude"] for s in recent_samples[25:]) / 25

                growth_rate = (late_avg - early_avg) / early_avg if early_avg > 0 else 0

                if growth_rate > 0.1:  # Amplitude growing by >10%
                    # Resonance detected
                    resonance = ResonancePattern(
                        pattern_id=f"res_{secrets.token_hex(8)}",
                        frequency=self.current_signature.fundamental_freq,
                        resonance_quality=q_factor,
                        amplitude=self.current_signature.amplitude,
                        growth_rate=growth_rate,
                        is_natural=self._is_natural_resonance()
                    )

                    self.active_resonances.append(resonance)
                    self.resonance_history.append(resonance)

    def _is_natural_resonance(self) -> bool:
        """Determine if resonance is natural or induced"""
        # Natural resonance builds gradually
        # Induced resonance appears suddenly

        if len(self.frequency_buffer) < 100:
            return True  # Not enough data

        recent = list(self.frequency_buffer)[-100:]

        # Check variance in frequency
        frequencies = [s["frequency"] for s in recent]
        mean_freq = sum(frequencies) / len(frequencies)
        variance = sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies)

        # Low variance = locked onto specific frequency = likely induced
        return variance > 1.0

    def _detect_harmonic_attacks(self):
        """Detect harmonic weapon attacks"""
        if not self.current_signature:
            return

        # Check for various attack patterns

        # 1. Resonance Amplifier Attack
        if self._detect_resonance_amplifier():
            self._register_attack(HarmonicWeaponType.RESONANCE_AMPLIFIER)

        # 2. Frequency Jammer
        if self._detect_frequency_jammer():
            self._register_attack(HarmonicWeaponType.FREQUENCY_JAMMER)

        # 3. Tesla 3-6-9 Cascade
        if self._detect_tesla_cascade():
            self._register_attack(HarmonicWeaponType.TESLA_CASCADE)

        # 4. Phase Inverter
        if self._detect_phase_inversion():
            self._register_attack(HarmonicWeaponType.PHASE_INVERTER)

    def _detect_resonance_amplifier(self) -> bool:
        """Detect resonance amplifier weapon"""
        # Characteristics:
        # - Frequency locked to natural frequency
        # - Rapid amplitude growth
        # - Unnatural stability

        for resonance in self.active_resonances:
            if (not resonance.is_natural and
                resonance.growth_rate > 0.5 and
                resonance.resonance_quality > 10):
                return True

        return False

    def _detect_frequency_jammer(self) -> bool:
        """Detect frequency jammer"""
        # Characteristics:
        # - Rapid frequency hopping
        # - High amplitude noise
        # - Loss of coherence

        if len(self.frequency_buffer) < 50:
            return False

        recent = list(self.frequency_buffer)[-50:]
        frequencies = [s["frequency"] for s in recent]

        # Calculate frequency change rate
        changes = sum(
            1 for i in range(1, len(frequencies))
            if abs(frequencies[i] - frequencies[i-1]) > 10
        )

        change_rate = changes / len(frequencies)

        # High change rate = jamming
        return change_rate > 0.3

    def _detect_tesla_cascade(self) -> bool:
        """Detect Tesla 3-6-9 cascade attack"""
        # Characteristics:
        # - Frequencies at 3, 6, 9 Hz or multiples
        # - Synchronized phase
        # - Building amplitude

        if not self.current_signature:
            return False

        # Check for 3-6-9 pattern in harmonics
        tesla_frequencies = {3, 6, 9, 12, 18, 27, 36, 54, 63, 81, 90}

        harmonics = set(round(h) for h in self.current_signature.harmonics)

        overlap = tesla_frequencies & harmonics

        # If we see multiple 3-6-9 frequencies
        return len(overlap) >= 3

    def _detect_phase_inversion(self) -> bool:
        """Detect phase inversion attack"""
        # Characteristics:
        # - Phase suddenly inverted (180 degrees)
        # - Creates destructive interference

        if len(self.frequency_buffer) < 20:
            return False

        recent = list(self.frequency_buffer)[-20:]
        phases = [s["phase"] for s in recent]

        # Check for phase flip
        early_phase = sum(phases[:10]) / 10
        late_phase = sum(phases[10:]) / 10

        phase_diff = abs(late_phase - early_phase)

        # 180 degree flip
        return abs(phase_diff - math.pi) < 0.3

    def _register_attack(self, weapon_type: HarmonicWeaponType):
        """Register detected attack"""
        # Calculate severity
        severity, damage_potential = self._assess_attack_severity(weapon_type)

        # Determine countermeasures
        countermeasures = self._get_countermeasures(weapon_type)

        attack = HarmonicAttack(
            attack_id=f"attack_{secrets.token_hex(8)}",
            weapon_type=weapon_type,
            target_frequency=self.natural_frequency,
            attack_signature=self.current_signature,
            severity=severity,
            damage_potential=damage_potential,
            countermeasures=countermeasures
        )

        self.detected_attacks.append(attack)
        self.total_attacks_detected += 1

        # Auto-activate defenses
        if severity in ["HIGH", "CRITICAL"]:
            self.activate_defense(DefenseMode.CHAOTIC_DETUNING)

    def _assess_attack_severity(
        self,
        weapon_type: HarmonicWeaponType
    ) -> Tuple[str, float]:
        """
        Assess attack severity

        Returns:
            (severity_level, damage_potential)
        """
        # Base severity by weapon type
        base_severity = {
            HarmonicWeaponType.RESONANCE_AMPLIFIER: (0.8, "HIGH"),
            HarmonicWeaponType.FREQUENCY_JAMMER: (0.5, "MEDIUM"),
            HarmonicWeaponType.PHASE_INVERTER: (0.7, "HIGH"),
            HarmonicWeaponType.HARMONIC_DISSONANCE: (0.6, "MEDIUM"),
            HarmonicWeaponType.SYMPATHETIC_EXPLOIT: (0.4, "LOW"),
            HarmonicWeaponType.TESLA_CASCADE: (0.9, "CRITICAL")
        }

        damage, level = base_severity.get(weapon_type, (0.5, "MEDIUM"))

        # Adjust based on resonance strength
        if self.active_resonances:
            max_growth = max(r.growth_rate for r in self.active_resonances)
            damage *= (1 + max_growth)
            damage = min(1.0, damage)

            if damage > 0.9:
                level = "CRITICAL"
            elif damage > 0.7:
                level = "HIGH"
            elif damage > 0.4:
                level = "MEDIUM"

        return level, damage

    def _get_countermeasures(
        self,
        weapon_type: HarmonicWeaponType
    ) -> List[str]:
        """Get recommended countermeasures"""
        countermeasures = {
            HarmonicWeaponType.RESONANCE_AMPLIFIER: [
                "Activate chaotic detuning",
                "Dampen amplitude",
                "Shift natural frequency"
            ],
            HarmonicWeaponType.FREQUENCY_JAMMER: [
                "Filter noise",
                "Lock onto stable frequency",
                "Increase coherence"
            ],
            HarmonicWeaponType.PHASE_INVERTER: [
                "Scramble phase relationships",
                "Re-synchronize oscillators",
                "Deploy phase locks"
            ],
            HarmonicWeaponType.TESLA_CASCADE: [
                "IMMEDIATE: Disrupt 3-6-9 field",
                "Randomize harmonics",
                "Break sympathetic resonance"
            ]
        }

        return countermeasures.get(weapon_type, ["Monitor and assess"])

    def activate_defense(self, mode: DefenseMode):
        """
        Activate defensive mode

        Args:
            mode: Defense mode to activate
        """
        self.defense_mode = mode

        if mode == DefenseMode.CHAOTIC_DETUNING:
            self._apply_chaotic_detuning()
        elif mode == DefenseMode.FREQUENCY_HOPPING:
            self._apply_frequency_hopping()
        elif mode == DefenseMode.PHASE_SCRAMBLING:
            self._apply_phase_scrambling()
        elif mode == DefenseMode.ACTIVE_DAMPENING:
            self._apply_active_dampening()

    def _apply_chaotic_detuning(self):
        """
        Apply chaotic detuning defense

        Introduce controlled chaos to prevent resonance lock
        """
        self.detuning_active = True

        # Shift natural frequency slightly and randomly
        import random
        self.natural_frequency += random.uniform(-2.0, 2.0)

        self.attacks_mitigated += 1

    def _apply_frequency_hopping(self):
        """Rapidly change operating frequency"""
        import random
        self.natural_frequency = random.uniform(40.0, 80.0)

    def _apply_phase_scrambling(self):
        """Scramble phase relationships"""
        if self.current_signature:
            import random
            self.current_signature.phase = random.uniform(0, 2 * math.pi)

    def _apply_active_dampening(self):
        """Apply active dampening to reduce amplitude"""
        # Reduce amplitude of active resonances
        for resonance in self.active_resonances:
            resonance.amplitude *= 0.5
            resonance.growth_rate = max(0, resonance.growth_rate - 0.3)

    def analyze_369_field(self) -> Vector369Field:
        """
        Analyze 3-6-9 vector field state

        Returns:
            Current field state
        """
        if not self.current_signature:
            return self.vector_field

        # Map frequencies to 3-6-9 poles
        fundamental = self.current_signature.fundamental_freq

        # Pole 3 (creation) - lower frequencies
        pole_3_energy = 0.0
        if 1 <= fundamental <= 30:
            pole_3_energy = fundamental / 30.0

        # Pole 6 (destruction) - mid frequencies
        pole_6_energy = 0.0
        if 30 < fundamental <= 60:
            pole_6_energy = (fundamental - 30) / 30.0

        # Axis 9 (transformation) - higher frequencies
        axis_9_energy = 0.0
        if fundamental > 60:
            axis_9_energy = min(1.0, (fundamental - 60) / 40.0)

        # Calculate balance
        total = pole_3_energy + pole_6_energy + axis_9_energy
        if total > 0:
            balance = 1.0 - abs((pole_3_energy - pole_6_energy) / total)
        else:
            balance = 1.0

        # Calculate coherence (how stable frequencies are)
        if len(self.frequency_buffer) > 10:
            recent = list(self.frequency_buffer)[-10:]
            freqs = [s["frequency"] for s in recent]
            mean_freq = sum(freqs) / len(freqs)
            variance = sum((f - mean_freq) ** 2 for f in freqs) / len(freqs)
            coherence = 1.0 / (1.0 + variance)
        else:
            coherence = 1.0

        self.vector_field = Vector369Field(
            pole_3=pole_3_energy,
            pole_6=pole_6_energy,
            axis_9=axis_9_energy,
            balance=balance,
            coherence=coherence
        )

        return self.vector_field

    def get_harmonic_status(self) -> Dict[str, Any]:
        """Get comprehensive harmonic status"""
        field = self.analyze_369_field()

        return {
            "system_id": self.system_id,
            "natural_frequency": self.natural_frequency,
            "current_signature": {
                "fundamental": self.current_signature.fundamental_freq if self.current_signature else None,
                "harmonics_count": len(self.current_signature.harmonics) if self.current_signature else 0,
                "amplitude": self.current_signature.amplitude if self.current_signature else 0
            },
            "vector_369_field": {
                "pole_3": field.pole_3,
                "pole_6": field.pole_6,
                "axis_9": field.axis_9,
                "balance": field.balance,
                "coherence": field.coherence
            },
            "active_resonances": len(self.active_resonances),
            "detected_attacks": len(self.detected_attacks),
            "defense_mode": self.defense_mode.value,
            "detuning_active": self.detuning_active,
            "statistics": {
                "total_attacks_detected": self.total_attacks_detected,
                "attacks_mitigated": self.attacks_mitigated
            }
        }
