"""
Bio-Sensory Fusion - Multi-Modal Sensory Integration
Combines octopus sensory, thermal sensing, and other modalities
"""
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class SensoryModality(Enum):
    """Different sensory input modalities"""
    DISTRIBUTED_TOUCH = 0    # Octopus arm sensors
    THERMAL = 1              # Thermal/energy sensors
    CHEMICAL = 2             # Chemical sensing
    ELECTROMAGNETIC = 3      # EM field sensing
    ACOUSTIC = 4             # Sound/vibration
    VISUAL = 5               # Light/visual
    PROPRIOCEPTION = 6       # Self-state awareness
    QUANTUM = 7              # Quantum field sensing


@dataclass
class FusedSensoryState:
    """Combined multi-modal sensory state"""
    timestamp: float
    modalities: Dict[SensoryModality, float]  # Activation levels
    confidence: float
    coherence: float  # How well modalities agree
    dominant_modality: SensoryModality
    cross_modal_correlations: Dict[str, float]
    integrated_perception: Dict[str, Any]


class BioSensoryFusion:
    """
    Multi-modal sensory fusion inspired by biological systems
    - Combines multiple sensory streams
    - Cross-modal validation
    - Adaptive weighting based on reliability
    - Emergent perception from combined inputs
    """

    def __init__(self, modalities: Optional[List[SensoryModality]] = None):
        if modalities is None:
            self.modalities = list(SensoryModality)
        else:
            self.modalities = modalities

        # Sensory weights (adaptive)
        self.modality_weights = {m: 1.0 / len(self.modalities) for m in self.modalities}

        # Fusion history
        self.fusion_history = []
        self.max_history = 1000

        # Cross-modal learning
        self.correlation_matrix = self._initialize_correlation_matrix()

        # Sensory adaptation (habituation/sensitization)
        self.adaptation_state = {m: 1.0 for m in self.modalities}

        print(f"🧠 Bio-Sensory Fusion initialized")
        print(f"   Modalities: {len(self.modalities)}")
        for m in self.modalities:
            print(f"      {m.name}: weight={self.modality_weights[m]:.3f}")

    def _initialize_correlation_matrix(self) -> Dict[tuple, float]:
        """Initialize cross-modal correlation matrix"""
        matrix = {}

        for m1 in self.modalities:
            for m2 in self.modalities:
                if m1 != m2:
                    # Start with weak correlations
                    matrix[(m1, m2)] = 0.1

        return matrix

    def fuse_sensory_inputs(self, sensory_data: Dict[str, Any]) -> FusedSensoryState:
        """
        Fuse multi-modal sensory inputs into integrated perception

        Args:
            sensory_data: Dict with keys matching modality names or custom data

        Returns:
            FusedSensoryState with integrated perception
        """

        timestamp = sensory_data.get('timestamp', np.random.random())

        # Extract activation levels for each modality
        modality_activations = self._extract_modality_activations(sensory_data)

        # Apply adaptive weights
        weighted_activations = self._apply_adaptive_weights(modality_activations)

        # Calculate cross-modal correlations
        correlations = self._calculate_cross_modal_correlations(modality_activations)

        # Update correlation matrix (learning)
        self._update_correlation_matrix(modality_activations)

        # Determine coherence (how well modalities agree)
        coherence = self._calculate_coherence(modality_activations, correlations)

        # Find dominant modality
        dominant = max(weighted_activations.items(), key=lambda x: x[1])[0]

        # Create integrated perception
        integrated = self._integrate_perception(
            modality_activations,
            weighted_activations,
            correlations,
            coherence
        )

        # Apply sensory adaptation
        self._apply_adaptation(modality_activations)

        # Calculate overall confidence
        confidence = self._calculate_fusion_confidence(
            modality_activations,
            coherence,
            correlations
        )

        # Create fused state
        fused_state = FusedSensoryState(
            timestamp=timestamp,
            modalities=modality_activations,
            confidence=confidence,
            coherence=coherence,
            dominant_modality=dominant,
            cross_modal_correlations=correlations,
            integrated_perception=integrated
        )

        # Store in history
        self.fusion_history.append(fused_state)
        if len(self.fusion_history) > self.max_history:
            self.fusion_history = self.fusion_history[-self.max_history:]

        return fused_state

    def _extract_modality_activations(self, sensory_data: Dict[str, Any]) -> Dict[SensoryModality, float]:
        """Extract activation level for each modality from raw sensory data"""
        activations = {}

        for modality in self.modalities:
            # Try multiple ways to extract activation
            activation = 0.0

            # Direct key match
            key = modality.name.lower()
            if key in sensory_data:
                activation = float(sensory_data[key])

            # Check for nested data structures
            elif 'modalities' in sensory_data and key in sensory_data['modalities']:
                activation = float(sensory_data['modalities'][key])

            # Try to infer from related keys
            elif modality == SensoryModality.DISTRIBUTED_TOUCH:
                activation = sensory_data.get('octopus_activation', 0.5)
            elif modality == SensoryModality.THERMAL:
                activation = sensory_data.get('thermal_intensity', 0.5)
            elif modality == SensoryModality.CHEMICAL:
                activation = sensory_data.get('chemical_concentration', 0.5)
            elif modality == SensoryModality.ELECTROMAGNETIC:
                activation = sensory_data.get('em_field_strength', 0.5)
            elif modality == SensoryModality.ACOUSTIC:
                activation = sensory_data.get('sound_level', 0.5)
            elif modality == SensoryModality.VISUAL:
                activation = sensory_data.get('light_intensity', 0.5)
            elif modality == SensoryModality.PROPRIOCEPTION:
                activation = sensory_data.get('self_state', 0.5)
            elif modality == SensoryModality.QUANTUM:
                activation = sensory_data.get('quantum_field', 0.5)

            # Add noise to simulate biological uncertainty
            activation += np.random.normal(0, 0.02)
            activation = np.clip(activation, 0, 1)

            activations[modality] = activation

        return activations

    def _apply_adaptive_weights(self, activations: Dict[SensoryModality, float]) -> Dict[SensoryModality, float]:
        """Apply adaptive weights based on reliability and adaptation state"""
        weighted = {}

        for modality, activation in activations.items():
            weight = self.modality_weights[modality]
            adaptation = self.adaptation_state[modality]

            weighted[modality] = activation * weight * adaptation

        return weighted

    def _calculate_cross_modal_correlations(self,
                                           activations: Dict[SensoryModality, float]) -> Dict[str, float]:
        """Calculate correlations between different modalities"""
        correlations = {}

        for m1 in self.modalities:
            for m2 in self.modalities:
                if m1 != m2:
                    # Get historical correlation
                    hist_corr = self.correlation_matrix.get((m1, m2), 0.0)

                    # Calculate current correlation (simplified)
                    act1 = activations[m1]
                    act2 = activations[m2]

                    # Pearson-like correlation measure
                    current_corr = 1.0 - abs(act1 - act2)

                    # Blend with historical
                    blended_corr = 0.8 * hist_corr + 0.2 * current_corr

                    key = f"{m1.name}_{m2.name}"
                    correlations[key] = blended_corr

        return correlations

    def _update_correlation_matrix(self, activations: Dict[SensoryModality, float]):
        """Update correlation matrix with new observations (learning)"""
        learning_rate = 0.1

        for m1 in self.modalities:
            for m2 in self.modalities:
                if m1 != m2:
                    act1 = activations[m1]
                    act2 = activations[m2]

                    # Observed correlation
                    observed_corr = 1.0 - abs(act1 - act2)

                    # Update with learning rate
                    old_corr = self.correlation_matrix.get((m1, m2), 0.0)
                    new_corr = old_corr + learning_rate * (observed_corr - old_corr)

                    self.correlation_matrix[(m1, m2)] = new_corr

    def _calculate_coherence(self, activations: Dict[SensoryModality, float],
                            correlations: Dict[str, float]) -> float:
        """Calculate how coherent/consistent the multi-modal inputs are"""
        if len(activations) < 2:
            return 1.0

        # Variance in activations (low variance = high coherence)
        activation_values = list(activations.values())
        variance = np.var(activation_values)
        variance_coherence = 1.0 / (1.0 + variance)

        # Average correlation (high correlation = high coherence)
        correlation_values = list(correlations.values())
        avg_correlation = np.mean(correlation_values) if correlation_values else 0.5

        # Combined coherence
        coherence = 0.5 * variance_coherence + 0.5 * avg_correlation

        return float(coherence)

    def _integrate_perception(self, activations: Dict[SensoryModality, float],
                             weighted: Dict[SensoryModality, float],
                             correlations: Dict[str, float],
                             coherence: float) -> Dict[str, Any]:
        """Create integrated multi-modal perception"""

        # Overall activation level
        overall_activation = np.mean(list(weighted.values()))

        # Identify active modalities (above threshold)
        active_modalities = [m.name for m, val in activations.items() if val > 0.5]

        # Detect cross-modal patterns
        patterns = self._detect_cross_modal_patterns(activations, correlations)

        # Emergent properties from fusion
        emergent = {
            'salience': overall_activation * coherence,  # How important/noticeable
            'clarity': coherence,  # How clear/unambiguous
            'novelty': self._calculate_novelty(activations),
            'intensity': max(activations.values()),
            'stability': self._calculate_stability(activations)
        }

        return {
            'overall_activation': overall_activation,
            'active_modalities': active_modalities,
            'cross_modal_patterns': patterns,
            'emergent_properties': emergent,
            'interpretation': self._interpret_fused_state(activations, patterns, emergent)
        }

    def _detect_cross_modal_patterns(self, activations: Dict[SensoryModality, float],
                                    correlations: Dict[str, float]) -> List[str]:
        """Detect emergent patterns from cross-modal correlations"""
        patterns = []

        # High activation in multiple modalities
        high_activation_modalities = [m.name for m, val in activations.items() if val > 0.7]
        if len(high_activation_modalities) >= 3:
            patterns.append(f"multi_modal_activation: {', '.join(high_activation_modalities)}")

        # Strong correlations
        strong_correlations = [k for k, v in correlations.items() if v > 0.8]
        if strong_correlations:
            patterns.append(f"strong_correlations: {len(strong_correlations)} pairs")

        # Specific pattern detection
        thermal_high = activations.get(SensoryModality.THERMAL, 0) > 0.7
        em_high = activations.get(SensoryModality.ELECTROMAGNETIC, 0) > 0.7
        if thermal_high and em_high:
            patterns.append("thermal_em_coupling: possible energy field interaction")

        touch_high = activations.get(SensoryModality.DISTRIBUTED_TOUCH, 0) > 0.7
        acoustic_high = activations.get(SensoryModality.ACOUSTIC, 0) > 0.7
        if touch_high and acoustic_high:
            patterns.append("tactile_acoustic_coupling: vibrational interaction")

        return patterns

    def _calculate_novelty(self, activations: Dict[SensoryModality, float]) -> float:
        """Calculate how novel this sensory state is"""
        if not self.fusion_history:
            return 1.0

        # Compare to recent history
        recent_states = self.fusion_history[-20:]

        differences = []
        for past_state in recent_states:
            diff = 0.0
            for modality, current_activation in activations.items():
                past_activation = past_state.modalities.get(modality, 0.5)
                diff += abs(current_activation - past_activation)

            differences.append(diff)

        # Novelty is average difference from history
        avg_diff = np.mean(differences) if differences else 0.5
        novelty = min(1.0, avg_diff / len(activations))

        return float(novelty)

    def _calculate_stability(self, activations: Dict[SensoryModality, float]) -> float:
        """Calculate how stable the sensory state is over time"""
        if len(self.fusion_history) < 5:
            return 0.5

        # Look at variance over recent history
        recent_states = self.fusion_history[-5:]

        variances = []
        for modality in activations.keys():
            values = [state.modalities.get(modality, 0.5) for state in recent_states]
            values.append(activations[modality])
            variance = np.var(values)
            variances.append(variance)

        # Low variance = high stability
        avg_variance = np.mean(variances)
        stability = 1.0 / (1.0 + avg_variance)

        return float(stability)

    def _interpret_fused_state(self, activations: Dict[SensoryModality, float],
                               patterns: List[str],
                               emergent: Dict[str, Any]) -> str:
        """High-level interpretation of fused sensory state"""

        salience = emergent['salience']
        novelty = emergent['novelty']
        intensity = emergent['intensity']

        if salience > 0.8 and novelty > 0.7:
            return "ALERT: High salience novel stimulus detected"
        elif intensity > 0.9:
            return "WARNING: Very high intensity sensory input"
        elif emergent['clarity'] > 0.8 and emergent['stability'] > 0.7:
            return "STABLE: Clear and stable sensory environment"
        elif emergent['clarity'] < 0.3:
            return "AMBIGUOUS: Conflicting or unclear sensory inputs"
        elif len(patterns) > 3:
            return f"COMPLEX: Multiple cross-modal patterns detected ({len(patterns)})"
        else:
            return "NORMAL: Routine sensory processing"

    def _apply_adaptation(self, activations: Dict[SensoryModality, float]):
        """Apply sensory adaptation (habituation/sensitization)"""
        adaptation_rate = 0.05

        for modality, activation in activations.items():
            current_adaptation = self.adaptation_state[modality]

            if activation > 0.8:  # High activation → habituation (reduce sensitivity)
                new_adaptation = current_adaptation - adaptation_rate
            elif activation < 0.2:  # Low activation → sensitization (increase sensitivity)
                new_adaptation = current_adaptation + adaptation_rate
            else:  # Recovery toward baseline
                new_adaptation = current_adaptation + adaptation_rate * (1.0 - current_adaptation)

            self.adaptation_state[modality] = np.clip(new_adaptation, 0.1, 2.0)

    def _calculate_fusion_confidence(self, activations: Dict[SensoryModality, float],
                                    coherence: float,
                                    correlations: Dict[str, float]) -> float:
        """Calculate confidence in the fused sensory state"""

        # High coherence = high confidence
        coherence_confidence = coherence

        # Multiple active modalities = high confidence
        active_count = sum(1 for val in activations.values() if val > 0.3)
        multi_modal_confidence = min(1.0, active_count / 4)

        # Strong correlations = high confidence
        strong_corr_count = sum(1 for val in correlations.values() if val > 0.7)
        correlation_confidence = min(1.0, strong_corr_count / 5)

        # Combined confidence
        confidence = (
            0.4 * coherence_confidence +
            0.3 * multi_modal_confidence +
            0.3 * correlation_confidence
        )

        return float(confidence)

    def get_sensory_health(self) -> Dict[str, Any]:
        """Get health status of sensory fusion system"""
        return {
            'num_modalities': len(self.modalities),
            'modality_weights': {m.name: w for m, w in self.modality_weights.items()},
            'adaptation_states': {m.name: a for m, a in self.adaptation_state.items()},
            'fusion_history_size': len(self.fusion_history),
            'average_coherence': np.mean([s.coherence for s in self.fusion_history[-10:]]) if self.fusion_history else 0,
            'system_status': 'healthy' if self.fusion_history else 'initializing'
        }


if __name__ == '__main__':
    # Demo
    print("🧠 Bio-Sensory Fusion Demo\n")

    fusion = BioSensoryFusion()

    # Simulate multi-modal sensory input
    sensory_input = {
        'distributed_touch': 0.75,
        'thermal': 0.65,
        'chemical': 0.45,
        'electromagnetic': 0.70,
        'acoustic': 0.55,
        'visual': 0.80,
        'proprioception': 0.60,
        'quantum': 0.50,
        'timestamp': 1.0
    }

    print("📡 Fusing multi-modal sensory inputs...")
    for key, val in sensory_input.items():
        if key != 'timestamp':
            print(f"   {key}: {val:.2f}")

    result = fusion.fuse_sensory_inputs(sensory_input)

    print(f"\n🎯 Fused Sensory State:")
    print(f"   Timestamp: {result.timestamp:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Coherence: {result.coherence:.3f}")
    print(f"   Dominant Modality: {result.dominant_modality.name}")

    print(f"\n🔗 Integrated Perception:")
    for key, val in result.integrated_perception.items():
        if isinstance(val, dict):
            print(f"   {key}:")
            for k2, v2 in val.items():
                print(f"      {k2}: {v2}")
        elif isinstance(val, list):
            print(f"   {key}: {len(val)} items")
            for item in val[:3]:
                print(f"      - {item}")
        else:
            print(f"   {key}: {val}")

    print(f"\n📊 System Health:")
    health = fusion.get_sensory_health()
    print(f"   Status: {health['system_status']}")
    print(f"   Average Coherence: {health['average_coherence']:.3f}")
