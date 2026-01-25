"""
Octopus Sensory System - Distributed Intelligence
Inspired by octopus neurophysiology: 2/3 of neurons in arms, 1/3 in brain
Each sensor node can process locally while contributing to global awareness
"""
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class SensorNodeType(Enum):
    """Types of distributed sensor nodes"""
    CENTRAL_BRAIN = 0      # Central processing hub (1/3 processing)
    ARM_NODE = 1           # Distributed arm node (local intelligence)
    PERIPHERAL_NODE = 2    # Edge sensor (basic processing)
    INTEGRATION_NODE = 3   # Cross-modal fusion point


@dataclass
class SensorReading:
    """Individual sensor reading with metadata"""
    node_id: str
    node_type: SensorNodeType
    value: float
    confidence: float
    timestamp: float
    local_decision: Optional[str] = None
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


class OctopusArm:
    """
    Individual octopus arm with local processing capability
    Can make autonomous decisions while coordinating with central brain
    """

    def __init__(self, arm_id: int, num_suckers: int = 8):
        self.arm_id = arm_id
        self.num_suckers = num_suckers
        self.suckers = []  # Local sensor nodes
        self.local_memory = []  # Short-term local memory
        self.autonomy_level = 0.67  # 67% autonomous processing

        # Initialize suckers (sensor nodes)
        for i in range(num_suckers):
            self.suckers.append({
                'id': f'arm{arm_id}_sucker{i}',
                'position': i / num_suckers,  # Normalized position on arm
                'sensitivity': np.random.uniform(0.7, 1.0),
                'specialization': self._assign_specialization(i)
            })

    def _assign_specialization(self, sucker_idx: int) -> str:
        """Assign sensory specialization to each sucker"""
        specializations = [
            'texture',      # Surface texture detection
            'chemical',     # Chemical sensing
            'pressure',     # Pressure/force sensing
            'temperature',  # Thermal sensing
            'vibration',    # Vibration/frequency
            'taste',        # Chemical taste
            'shape',        # Object shape recognition
            'gradient'      # Gradient detection
        ]
        return specializations[sucker_idx % len(specializations)]

    def sense_local(self, stimulus: Dict[str, float]) -> Dict[str, Any]:
        """
        Local sensing and processing (autonomous)
        Doesn't need central brain approval for basic responses
        """
        local_readings = []

        for sucker in self.suckers:
            # Get relevant stimulus for this sucker's specialization
            stim_value = stimulus.get(sucker['specialization'], 0.0)

            # Apply sucker sensitivity
            processed_value = stim_value * sucker['sensitivity']

            # Local decision making
            local_decision = self._make_local_decision(
                processed_value,
                sucker['specialization']
            )

            reading = SensorReading(
                node_id=sucker['id'],
                node_type=SensorNodeType.ARM_NODE,
                value=processed_value,
                confidence=sucker['sensitivity'],
                timestamp=np.random.random(),  # Would be time.time() in production
                local_decision=local_decision,
                context={'specialization': sucker['specialization']}
            )

            local_readings.append(reading)

        # Store in local memory
        self.local_memory.append({
            'readings': local_readings,
            'summary': self._summarize_readings(local_readings)
        })

        # Keep memory bounded
        if len(self.local_memory) > 100:
            self.local_memory = self.local_memory[-100:]

        return {
            'arm_id': self.arm_id,
            'readings': local_readings,
            'local_summary': self._summarize_readings(local_readings),
            'needs_central_processing': self._needs_central_input(local_readings)
        }

    def _make_local_decision(self, value: float, specialization: str) -> str:
        """Local autonomous decision (67% of processing)"""
        if value > 0.8:
            return f'strong_{specialization}_detected'
        elif value > 0.5:
            return f'moderate_{specialization}_detected'
        elif value < 0.2:
            return f'no_{specialization}_detected'
        else:
            return f'weak_{specialization}_detected'

    def _summarize_readings(self, readings: List[SensorReading]) -> Dict[str, float]:
        """Summarize local readings"""
        if not readings:
            return {}

        return {
            'mean_value': np.mean([r.value for r in readings]),
            'max_value': np.max([r.value for r in readings]),
            'mean_confidence': np.mean([r.confidence for r in readings]),
            'num_readings': len(readings)
        }

    def _needs_central_processing(self, readings: List[SensorReading]) -> bool:
        """Determine if central brain needs to be involved"""
        # Send to central brain if:
        # 1. Very high values (novel/important)
        # 2. Conflicting signals
        # 3. Low confidence

        values = [r.value for r in readings]
        confidences = [r.confidence for r in readings]

        max_value = np.max(values) if values else 0
        variance = np.var(values) if len(values) > 1 else 0
        min_confidence = np.min(confidences) if confidences else 1.0

        return (max_value > 0.9 or  # Novel/important signal
                variance > 0.5 or    # Conflicting signals
                min_confidence < 0.5)  # Low confidence


class OctopusSensorySystem:
    """
    Complete octopus-inspired distributed sensory system
    - 8 arms (distributed processing)
    - Central brain (global coordination)
    - 2/3 processing in arms, 1/3 in brain
    - Local autonomy with global awareness
    """

    def __init__(self, num_arms: int = 8, suckers_per_arm: int = 8):
        self.num_arms = num_arms
        self.suckers_per_arm = suckers_per_arm

        # Create distributed arms
        self.arms = [OctopusArm(i, suckers_per_arm) for i in range(num_arms)]

        # Central brain processing
        self.central_memory = []
        self.global_state = {}
        self.decision_history = []

        # Processing distribution
        self.arm_processing_ratio = 2/3  # 67% in arms
        self.brain_processing_ratio = 1/3  # 33% in brain

        print(f"🐙 Octopus Sensory System initialized")
        print(f"   {num_arms} arms × {suckers_per_arm} suckers = {num_arms * suckers_per_arm} sensor nodes")
        print(f"   Processing: {self.arm_processing_ratio*100:.1f}% distributed, {self.brain_processing_ratio*100:.1f}% central")

    def sense(self, environmental_stimulus: Dict[str, float]) -> Dict[str, Any]:
        """
        Complete sensing cycle:
        1. All arms sense locally (distributed)
        2. Central brain receives summaries
        3. Global decision made
        4. Context distributed back to arms
        """

        # Phase 1: Distributed sensing (67% of processing)
        arm_results = []
        needs_central = []

        for arm in self.arms:
            result = arm.sense_local(environmental_stimulus)
            arm_results.append(result)

            if result['needs_central_processing']:
                needs_central.append(result)

        # Phase 2: Central brain processing (33% of processing)
        central_decision = self._central_processing(arm_results, needs_central)

        # Phase 3: Global integration
        global_state = self._integrate_global_state(arm_results, central_decision)

        # Store in central memory
        self.central_memory.append(global_state)
        if len(self.central_memory) > 1000:
            self.central_memory = self.central_memory[-1000:]

        return global_state

    def _central_processing(self, arm_results: List[Dict], priority_items: List[Dict]) -> Dict[str, Any]:
        """
        Central brain processing (33% of total)
        Only processes what arms can't handle locally
        """

        # Extract all readings
        all_readings = []
        for result in arm_results:
            all_readings.extend(result['readings'])

        # Global statistics
        all_values = [r.value for r in all_readings]
        all_confidences = [r.confidence for r in all_readings]

        # Central decision focuses on:
        # 1. Cross-arm coordination
        # 2. Novel patterns
        # 3. Low-confidence resolution

        decision = {
            'global_mean': np.mean(all_values) if all_values else 0,
            'global_max': np.max(all_values) if all_values else 0,
            'global_confidence': np.mean(all_confidences) if all_confidences else 0,
            'cross_arm_variance': self._calculate_cross_arm_variance(arm_results),
            'novelty_detected': self._detect_novelty(all_values),
            'priority_count': len(priority_items),
            'coordination_needed': len(priority_items) > 2
        }

        # Store decision
        self.decision_history.append(decision)

        return decision

    def _calculate_cross_arm_variance(self, arm_results: List[Dict]) -> float:
        """Measure variance across arms (coordination check)"""
        arm_means = [result['local_summary'].get('mean_value', 0)
                     for result in arm_results]
        return float(np.var(arm_means)) if len(arm_means) > 1 else 0.0

    def _detect_novelty(self, values: List[float]) -> bool:
        """Detect novel patterns by comparing to memory"""
        if not self.central_memory or not values:
            return False

        current_mean = np.mean(values)

        # Compare to recent history
        recent_means = [m['central_decision']['global_mean']
                       for m in self.central_memory[-10:]
                       if 'central_decision' in m]

        if not recent_means:
            return True

        historical_mean = np.mean(recent_means)
        historical_std = np.std(recent_means) if len(recent_means) > 1 else 0.1

        # Novel if >2 standard deviations from norm
        return abs(current_mean - historical_mean) > 2 * historical_std

    def _integrate_global_state(self, arm_results: List[Dict],
                                central_decision: Dict) -> Dict[str, Any]:
        """Integrate distributed and central processing"""

        # Count sensor specializations
        specialization_counts = {}
        all_readings = []

        for result in arm_results:
            for reading in result['readings']:
                spec = reading.context.get('specialization', 'unknown')
                specialization_counts[spec] = specialization_counts.get(spec, 0) + 1
                all_readings.append(reading)

        return {
            'timestamp': np.random.random(),
            'num_sensors_active': len(all_readings),
            'arm_results': arm_results,
            'central_decision': central_decision,
            'specialization_distribution': specialization_counts,
            'global_state': {
                'distributed_processing': f"{len([r for r in arm_results if not r['needs_central_processing']])} arms autonomous",
                'central_coordination': f"{central_decision['priority_count']} items escalated",
                'novelty': central_decision['novelty_detected'],
                'confidence': central_decision['global_confidence']
            },
            'processing_ratio': {
                'distributed': self.arm_processing_ratio,
                'central': self.brain_processing_ratio
            }
        }

    def get_sensor_health(self) -> Dict[str, Any]:
        """Get health status of distributed sensor network"""
        total_suckers = self.num_arms * self.suckers_per_arm

        # Get recent activity
        recent_activity = len(self.central_memory[-10:]) if self.central_memory else 0

        return {
            'total_sensors': total_suckers,
            'num_arms': self.num_arms,
            'sensors_per_arm': self.suckers_per_arm,
            'recent_activity': recent_activity,
            'central_memory_size': len(self.central_memory),
            'decisions_made': len(self.decision_history),
            'distribution_healthy': recent_activity > 0
        }


if __name__ == '__main__':
    # Demo
    print("🐙 Octopus Sensory System Demo\n")

    system = OctopusSensorySystem(num_arms=8, suckers_per_arm=8)

    # Simulate environmental stimulus
    stimulus = {
        'texture': 0.8,
        'chemical': 0.6,
        'pressure': 0.9,
        'temperature': 0.4,
        'vibration': 0.7,
        'taste': 0.3,
        'shape': 0.85,
        'gradient': 0.5
    }

    print("\n📡 Sensing environmental stimulus:")
    for key, val in stimulus.items():
        print(f"   {key}: {val:.2f}")

    result = system.sense(stimulus)

    print(f"\n🧠 Global State:")
    print(f"   Sensors Active: {result['num_sensors_active']}")
    print(f"   Novelty Detected: {result['central_decision']['novelty_detected']}")
    print(f"   Global Confidence: {result['central_decision']['global_confidence']:.2f}")
    print(f"   {result['global_state']['distributed_processing']}")
    print(f"   {result['global_state']['central_coordination']}")

    print(f"\n📊 Sensor Health:")
    health = system.get_sensor_health()
    for key, val in health.items():
        print(f"   {key}: {val}")
