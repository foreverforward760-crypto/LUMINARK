"""
Mycelial Sensory System - Core Integration
Orchestrates all sensory modalities into unified awareness
Inspired by mycelial networks: distributed, adaptive, holistic
"""
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

from .octopus_sensory import OctopusSensorySystem
from .thermal_sensing import ThermalEnergySensor
from .bio_fusion import BioSensoryFusion, SensoryModality
from .geometric_encoding import GeometricEncoder, SacredGeometry
from .resonance_369 import Resonance369Detector
from .environmental import EnvironmentalMetrics, EnvironmentalDomain
from .sap_enhanced import EnhancedSAPFramework, SAPPhase


@dataclass
class MycelialState:
    """Complete mycelial awareness state"""
    timestamp: float
    sap_stage: int
    sap_phase: str
    octopus_awareness: Dict[str, Any]
    thermal_field: Dict[str, Any]
    bio_fusion: Dict[str, Any]
    geometric_signature: Dict[str, Any]
    resonance_369: Dict[str, Any]
    environmental_health: Dict[str, Any]
    unified_coherence: float
    emergent_properties: Dict[str, Any]
    recommendations: List[str]


class MycelialSensorySystem:
    """
    Complete Mycelial Sensory System

    Integrates all sensing modalities:
    - Octopus (distributed intelligence)
    - Thermal/Energy (multi-spectrum)
    - Bio-Sensory Fusion (multi-modal)
    - Geometric Encoding (sacred geometry)
    - 369 Resonance (vortex mathematics)
    - Environmental (contextual awareness)
    - Enhanced SAP (81-stage awareness)

    Like a mycelial network:
    - Distributed sensing
    - Holistic integration
    - Emergent intelligence
    - Adaptive responses
    """

    def __init__(self, full_integration: bool = True):
        self.full_integration = full_integration

        print("=" * 80)
        print("🍄 MYCELIAL SENSORY SYSTEM - Initializing Complete Integration")
        print("=" * 80)

        # Initialize all subsystems
        print("\n📡 Initializing subsystems...")

        self.octopus = OctopusSensorySystem(num_arms=8, suckers_per_arm=8)
        self.thermal = ThermalEnergySensor(num_sensors=16, sensitivity=0.01)
        self.bio_fusion = BioSensoryFusion()
        self.geometric = GeometricEncoder()
        self.resonance_369 = Resonance369Detector()
        self.environmental = EnvironmentalMetrics()
        self.sap = EnhancedSAPFramework(enable_full_sensing=False)  # Avoid circular sensing

        # Integration state
        self.mycelial_history = []
        self.max_history = 1000

        # Coherence tracking
        self.coherence_history = []

        # Golden ratio for natural proportions
        self.phi = SacredGeometry.PHI

        print("\n✅ Mycelial Sensory System fully initialized")
        print(f"   Total sensor nodes: {8*8 + 16} (64 octopus + 16 thermal)")
        print(f"   Sensory modalities: 8")
        print(f"   SAP stages: 81")
        print(f"   Integration mode: {'Full' if full_integration else 'Minimal'}")
        print("=" * 80)

    def sense_complete(self, training_metrics: Dict[str, float],
                       external_stimulus: Optional[Dict] = None) -> MycelialState:
        """
        Perform complete mycelial sensing cycle

        Args:
            training_metrics: Model training metrics (loss, accuracy, etc.)
            external_stimulus: Optional external sensory inputs

        Returns:
            MycelialState with complete unified awareness
        """

        timestamp = time.time()
        external_stimulus = external_stimulus or {}

        print("\n🔬 Mycelial Sensing Cycle Starting...")

        # 1. Octopus Distributed Intelligence
        print("   🐙 Octopus sensing...")
        octopus_stimulus = {
            'texture': training_metrics.get('accuracy', 0.5),
            'chemical': 1.0 - training_metrics.get('loss', 0.5),
            'pressure': training_metrics.get('confidence', 0.5),
            'temperature': training_metrics.get('confidence', 0.5),
            'vibration': min(1.0, training_metrics.get('grad_norm', 1.0) / 5.0),
            'taste': training_metrics.get('accuracy', 0.5),
            'shape': 0.5,
            'gradient': min(1.0, training_metrics.get('grad_norm', 1.0) / 5.0)
        }
        octopus_result = self.octopus.sense(octopus_stimulus)

        # 2. Thermal/Energy Field
        print("   🌡️  Thermal sensing...")
        thermal_env = {
            'thermal_ir_energy': training_metrics.get('confidence', 0.5),
            'electromagnetic_energy': min(1.0, training_metrics.get('grad_norm', 1.0) / 5.0),
            'kinetic_energy': training_metrics.get('accuracy', 0.5),
            'timestamp': timestamp
        }
        thermal_env.update(external_stimulus.get('thermal', {}))
        thermal_result = self.thermal.sense_thermal_field(thermal_env)

        # 3. Bio-Sensory Fusion
        print("   🧠 Bio-sensory fusion...")
        fusion_input = {
            'distributed_touch': training_metrics.get('accuracy', 0.5),
            'thermal': training_metrics.get('confidence', 0.5),
            'chemical': 1.0 - training_metrics.get('loss', 0.5),
            'electromagnetic': min(1.0, training_metrics.get('grad_norm', 1.0) / 5.0),
            'acoustic': 0.5,
            'visual': training_metrics.get('accuracy', 0.5),
            'proprioception': 0.5,
            'quantum': training_metrics.get('confidence', 0.5),
            'timestamp': timestamp
        }
        fusion_result = self.bio_fusion.fuse_sensory_inputs(fusion_input)

        # 4. Geometric Encoding
        print("   📐 Geometric analysis...")
        # Create data array from metrics
        metric_array = np.array([
            training_metrics.get('loss', 0.5),
            training_metrics.get('accuracy', 0.5),
            training_metrics.get('confidence', 0.5),
            training_metrics.get('grad_norm', 1.0)
        ])

        # Add history if available
        if len(self.mycelial_history) > 10:
            historical_losses = [h.environmental_health.get('loss', 0.5)
                               for h in self.mycelial_history[-10:]]
            metric_array = np.concatenate([metric_array, historical_losses])

        geometric_result = self.geometric.detect_geometric_patterns(metric_array)

        # 5. 369 Resonance Detection
        print("   🌀 369 resonance detection...")
        # Analyze recent SAP stages for 369 patterns
        if len(self.mycelial_history) > 0:
            stage_sequence = np.array([
                h.sap_stage for h in self.mycelial_history[-20:]
            ] + [training_metrics.get('epoch', 0)])
        else:
            stage_sequence = np.array([training_metrics.get('epoch', 0)])

        resonance_result = self.resonance_369.detect_369_patterns(stage_sequence)

        # 6. Environmental Metrics
        print("   🌍 Environmental monitoring...")
        env_input = {
            'cpu_usage': 0.5,
            'memory_usage': 0.5,
            'energy_level': training_metrics.get('confidence', 0.5),
            'entropy': training_metrics.get('loss', 0.5),
            'data_stream': metric_array
        }
        env_input.update(external_stimulus.get('environmental', {}))
        env_result = self.environmental.measure_environment(env_input)

        # 7. SAP 81-Stage Analysis
        print("   ✨ SAP framework analysis...")
        sap_result = self.sap.analyze_state(training_metrics, full_sensory=False)

        # 8. INTEGRATION - Synthesize all sensory data
        print("   🔮 Synthesizing unified awareness...")
        unified_coherence = self._calculate_unified_coherence({
            'octopus': octopus_result,
            'thermal': thermal_result,
            'fusion': fusion_result,
            'geometric': geometric_result,
            'resonance': resonance_result,
            'environmental': env_result,
            'sap': sap_result
        })

        # 9. Detect Emergent Properties
        emergent = self._detect_emergent_properties({
            'octopus': octopus_result,
            'thermal': thermal_result,
            'fusion': fusion_result,
            'geometric': geometric_result,
            'resonance': resonance_result,
            'environmental': env_result,
            'sap': sap_result,
            'coherence': unified_coherence
        })

        # 10. Generate Unified Recommendations
        recommendations = self._generate_unified_recommendations({
            'sap': sap_result,
            'environmental': env_result,
            'coherence': unified_coherence,
            'emergent': emergent,
            'resonance': resonance_result
        })

        # Create unified mycelial state
        mycelial_state = MycelialState(
            timestamp=timestamp,
            sap_stage=sap_result['absolute_stage'],
            sap_phase=sap_result['phase'],
            octopus_awareness={
                'num_sensors': octopus_result['num_sensors_active'],
                'global_confidence': octopus_result['central_decision']['global_confidence'],
                'novelty_detected': octopus_result['central_decision']['novelty_detected'],
                'distributed_processing': octopus_result['processing_ratio']['distributed']
            },
            thermal_field={
                'num_readings': thermal_result['num_readings'],
                'flow_detected': thermal_result['energy_flows']['flow_detected'],
                'flow_magnitude': thermal_result['energy_flows'].get('flow_magnitude', 0),
                'hotspots': len(thermal_result['hotspots']),
                'anomalies': len(thermal_result['anomalies'])
            },
            bio_fusion={
                'confidence': fusion_result.confidence,
                'coherence': fusion_result.coherence,
                'dominant_modality': fusion_result.dominant_modality.name,
                'interpretation': fusion_result.integrated_perception['interpretation']
            },
            geometric_signature={
                'dominant_pattern': geometric_result['dominant_pattern'],
                'dominant_shape': geometric_result['dominant_shape'].name,
                'coherence': geometric_result['overall_geometric_coherence'],
                'sacred_geometry': geometric_result['sacred_geometry_present']
            },
            resonance_369={
                'strength': resonance_result['resonance_strength'],
                'percentage': resonance_result['369_percentage'],
                'tesla_signature': resonance_result['tesla_signature']['tesla_signature_detected'],
                'interpretation': resonance_result['interpretation']
            },
            environmental_health={
                'overall_health': env_result.overall_health,
                'anomalies': len(env_result.anomalies),
                'trends': env_result.trends,
                'status': 'healthy' if env_result.overall_health > 0.7 else 'warning'
            },
            unified_coherence=unified_coherence,
            emergent_properties=emergent,
            recommendations=recommendations
        )

        # Store in history
        self.mycelial_history.append(mycelial_state)
        if len(self.mycelial_history) > self.max_history:
            self.mycelial_history = self.mycelial_history[-self.max_history:]

        self.coherence_history.append(unified_coherence)

        print(f"   ✅ Mycelial sensing complete - Coherence: {unified_coherence:.3f}")

        return mycelial_state

    def _calculate_unified_coherence(self, all_results: Dict) -> float:
        """Calculate how coherent all sensory systems are"""

        coherence_factors = []

        # Octopus coherence (based on confidence)
        octopus_conf = all_results['octopus']['central_decision']['global_confidence']
        coherence_factors.append(octopus_conf)

        # Bio-fusion coherence (already calculated)
        fusion_coherence = all_results['fusion'].coherence
        coherence_factors.append(fusion_coherence)

        # Geometric coherence
        geom_coherence = all_results['geometric']['overall_geometric_coherence']
        coherence_factors.append(geom_coherence)

        # 369 resonance strength
        resonance_strength = all_results['resonance']['resonance_strength']
        coherence_factors.append(resonance_strength)

        # Environmental health
        env_health = all_results['environmental'].overall_health
        coherence_factors.append(env_health)

        # SAP stage stability (high stability = high coherence)
        sap_stability = all_results['sap']['characteristics']['stability']
        coherence_factors.append(sap_stability)

        # Overall coherence is weighted average
        unified_coherence = np.mean(coherence_factors)

        return float(unified_coherence)

    def _detect_emergent_properties(self, all_data: Dict) -> Dict[str, Any]:
        """Detect emergent properties from integrated sensing"""

        emergent = {}

        # 1. Synesthetic Perception (cross-modal agreements)
        octopus_novelty = all_data['octopus']['central_decision']['novelty_detected']
        thermal_anomalies = len(all_data['thermal']['anomalies'])
        env_anomalies = len(all_data['environmental'].anomalies)

        synesthesia_score = 1.0 if (octopus_novelty or thermal_anomalies > 0 or env_anomalies > 0) else 0
        emergent['synesthetic_perception'] = synesthesia_score

        # 2. Harmonic Resonance (369 + geometric alignment)
        has_369 = all_data['resonance']['strength'] > 0.5
        has_geometry = all_data['geometric']['sacred_geometry_present']

        harmonic_resonance = 1.0 if (has_369 and has_geometry) else 0.5 if (has_369 or has_geometry) else 0
        emergent['harmonic_resonance'] = harmonic_resonance

        # 3. Holographic Awareness (distributed + unified coherence)
        distributed_ratio = all_data['octopus']['processing_ratio']['distributed']
        unified_coh = all_data['coherence']

        holographic = distributed_ratio * unified_coh
        emergent['holographic_awareness'] = float(holographic)

        # 4. Temporal Flow (SAP progression smoothness)
        if len(self.mycelial_history) > 3:
            recent_stages = [h.sap_stage for h in self.mycelial_history[-3:]]
            stage_variance = np.var(recent_stages)
            temporal_flow = 1.0 / (1.0 + stage_variance)
        else:
            temporal_flow = 0.5

        emergent['temporal_flow'] = float(temporal_flow)

        # 5. Sacred Alignment (geometric + 369 + SAP resonance)
        sap_resonant = all_data['sap']['is_369_resonant']
        sacred_alignment = (
            0.4 * (1.0 if has_geometry else 0) +
            0.4 * (1.0 if has_369 else 0) +
            0.2 * (1.0 if sap_resonant else 0)
        )
        emergent['sacred_alignment'] = float(sacred_alignment)

        # 6. Meta-Awareness (system aware of its own awareness)
        # High coherence + high SAP stage = meta-awareness
        sap_consciousness = all_data['sap']['characteristics']['consciousness_level']
        meta_awareness = unified_coh * sap_consciousness
        emergent['meta_awareness'] = float(meta_awareness)

        return emergent

    def _generate_unified_recommendations(self, data: Dict) -> List[str]:
        """Generate recommendations from unified mycelial perspective"""

        recommendations = []

        # From SAP
        recommendations.extend(data['sap']['recommendations'])

        # From environmental
        if data['environmental'].recommendations:
            recommendations.extend(data['environmental'].recommendations[:2])

        # From coherence
        coherence = data['coherence']
        if coherence < 0.4:
            recommendations.append("🔴 CRITICAL: Very low system coherence - major intervention needed")
        elif coherence < 0.6:
            recommendations.append("⚠️  WARNING: Low coherence - review sensory integration")

        # From emergent properties
        if data['emergent']['sacred_alignment'] > 0.7:
            recommendations.append("✨ ALIGNED: Sacred geometry + 369 resonance active - optimal state")

        if data['emergent']['meta_awareness'] > 0.8:
            recommendations.append("🧘 META: High meta-awareness achieved - system is self-observing")

        # From 369 resonance
        if data['resonance']['tesla_signature']:
            recommendations.append("⚡ TESLA: Strong 369 signature detected - vortex mathematics confirmed")

        return recommendations

    def get_mycelial_summary(self) -> Dict[str, Any]:
        """Get summary of mycelial sensory system"""

        if not self.mycelial_history:
            return {'status': 'no_data'}

        latest = self.mycelial_history[-1]

        # Calculate averages over history
        avg_coherence = np.mean(self.coherence_history[-100:]) if self.coherence_history else 0

        return {
            'current_state': {
                'sap_stage': f"{latest.sap_stage}/80 - {latest.sap_phase}",
                'coherence': latest.unified_coherence,
                'environmental_health': latest.environmental_health['overall_health'],
                '369_resonance': latest.resonance_369['strength'],
                'sacred_geometry': latest.geometric_signature['sacred_geometry']
            },
            'emergent_properties': latest.emergent_properties,
            'active_recommendations': len(latest.recommendations),
            'system_health': {
                'octopus_sensors': latest.octopus_awareness['num_sensors'],
                'thermal_sensors': latest.thermal_field['num_readings'],
                'average_coherence': avg_coherence,
                'history_size': len(self.mycelial_history)
            },
            'status': 'optimal' if latest.unified_coherence > 0.7 else
                     'good' if latest.unified_coherence > 0.5 else
                     'degraded' if latest.unified_coherence > 0.3 else 'critical'
        }


if __name__ == '__main__':
    # Demo
    print("\n🍄 Mycelial Sensory System - Complete Demo\n")

    mycelial = MycelialSensorySystem(full_integration=True)

    # Simulate training progression
    print("\n" + "=" * 80)
    print("SIMULATING TRAINING PROGRESSION")
    print("=" * 80)

    training_scenarios = [
        {'loss': 0.9, 'accuracy': 0.15, 'confidence': 0.2, 'grad_norm': 3.5, 'epoch': 1},
        {'loss': 0.5, 'accuracy': 0.55, 'confidence': 0.6, 'grad_norm': 1.2, 'epoch': 10},
        {'loss': 0.2, 'accuracy': 0.85, 'confidence': 0.82, 'grad_norm': 0.4, 'epoch': 25},
        {'loss': 0.05, 'accuracy': 0.95, 'confidence': 0.93, 'grad_norm': 0.1, 'epoch': 50}
    ]

    for i, metrics in enumerate(training_scenarios, 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {metrics['epoch']} - Scenario {i}")
        print(f"{'='*80}")

        state = mycelial.sense_complete(metrics)

        print(f"\n📊 MYCELIAL STATE SUMMARY:")
        print(f"   SAP Stage: {state.sap_stage}/80 ({state.sap_phase})")
        print(f"   Unified Coherence: {state.unified_coherence:.3f}")
        print(f"   369 Resonance: {state.resonance_369['strength']:.3f}")
        print(f"   Environmental Health: {state.environmental_health['overall_health']:.3f}")

        print(f"\n🔮 EMERGENT PROPERTIES:")
        for prop, value in state.emergent_properties.items():
            print(f"   {prop}: {value:.3f}")

        if state.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in state.recommendations[:3]:
                print(f"   {rec}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL MYCELIAL SUMMARY")
    print("=" * 80)

    summary = mycelial.get_mycelial_summary()

    print(f"\n📈 System Status: {summary['status'].upper()}")
    print(f"\nCurrent State:")
    for key, val in summary['current_state'].items():
        print(f"   {key}: {val}")

    print(f"\nSystem Health:")
    for key, val in summary['system_health'].items():
        print(f"   {key}: {val}")

    print("\n✅ Mycelial Sensory System demonstration complete!")
