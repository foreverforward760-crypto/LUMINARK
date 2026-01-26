"""
LUMINARK - Thermal/Energy Sensing & Bio-Sensory Fusion
Combines mycelium and octopus sensory capabilities with attention mechanism
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from .mycelium import MyceliumSensorySystem
    from .octopus import OctopusSensorySystem
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from mycelium import MyceliumSensorySystem
    from octopus import OctopusSensorySystem

@dataclass
class ThreatAssessment:
    """Complete threat assessment from fused sensors"""
    threat_level: float  # 0.0 to 1.0
    threat_type: str  # 'chemical', 'thermal', 'energy', 'vibration', 'visual'
    affected_nodes: List[int]
    confidence: float  # 0.0 to 1.0
    recommended_action: str

class ThermalEnergySensing:
    """
    Detects heat signatures and energy surges (combined biological sensing)
    """
    
    def __init__(self):
        self.thermal_baseline: Optional[float] = None
        self.energy_baseline: Optional[float] = None
        self.thermal_history: List[np.ndarray] = []
        self.energy_history: List[np.ndarray] = []
        self.max_history = 100
        
    def detect_thermal_anomalies(self, 
                                node_temperatures: np.ndarray, 
                                ambient_temperature: float) -> Dict:
        """
        Detect thermal anomalies (heat signatures)
        
        Args:
            node_temperatures: Temperature at each node
            ambient_temperature: Expected ambient temperature
            
        Returns:
            Dict with thermal anomalies, gradients, high gradient nodes
        """
        if len(node_temperatures) == 0:
            return {
                'thermal_anomalies': [],
                'thermal_gradients': [],
                'high_gradient_nodes': [],
                'ambient_temperature': ambient_temperature
            }
        
        # Initialize baseline
        if self.thermal_baseline is None:
            self.thermal_baseline = np.mean(node_temperatures)
        
        # Calculate temperature anomalies
        temp_anomalies = node_temperatures - ambient_temperature
        thermal_anomaly_nodes = np.where(np.abs(temp_anomalies) > 2.0)[0]
        
        # Calculate thermal gradient (rate of change)
        thermal_gradient = np.gradient(node_temperatures)
        high_gradient_nodes = np.where(np.abs(thermal_gradient) > 1.0)[0]
        
        # Store in history
        self.thermal_history.append(node_temperatures)
        if len(self.thermal_history) > self.max_history:
            self.thermal_history.pop(0)
        
        return {
            'thermal_anomalies': thermal_anomaly_nodes.tolist(),
            'thermal_gradients': thermal_gradient.tolist(),
            'high_gradient_nodes': high_gradient_nodes.tolist(),
            'ambient_temperature': ambient_temperature,
            'max_deviation': float(np.max(np.abs(temp_anomalies)))
        }
    
    def detect_energy_surges(self, node_energy: np.ndarray) -> Dict:
        """
        Detect energy surges and power anomalies
        
        Args:
            node_energy: Energy level at each node
            
        Returns:
            Dict with energy surges, total energy, variance
        """
        if len(node_energy) == 0:
            return {
                'energy_surges': [],
                'total_energy': 0.0,
                'energy_variance': 0.0
            }
        
        # Initialize baseline
        if self.energy_baseline is None:
            self.energy_baseline = np.mean(node_energy)
        
        # Store in history
        self.energy_history.append(node_energy)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
        
        # Detect surges (3 sigma above mean)
        if len(self.energy_history) > 1:
            energy_change = node_energy - self.energy_history[-2]
            surge_threshold = np.std(node_energy) * 3
            surge_nodes = np.where(energy_change > surge_threshold)[0]
        else:
            surge_nodes = np.array([])
        
        return {
            'energy_surges': surge_nodes.tolist(),
            'total_energy': float(np.sum(node_energy)),
            'energy_variance': float(np.var(node_energy)),
            'baseline_deviation': float(np.mean(node_energy) - self.energy_baseline) if self.energy_baseline else 0.0
        }

class BioSensoryFusion:
    """
    Fuses mycelium and octopus sensory capabilities with attention mechanism
    
    Attention weights determine which sensory modality is prioritized
    based on current threat landscape and system state
    """
    
    def __init__(self, network_size: int = 100):
        self.network_size = network_size
        
        # Initialize sensory systems
        self.mycelium_sensors = MyceliumSensorySystem(network_size)
        self.octopus_sensors = OctopusSensorySystem()
        self.thermal_energy = ThermalEnergySensing()
        
        # Attention weights (sum to 1.0)
        self.attention_weights = {
            'vibration': 0.25,      # Mycelium vibration sensing
            'chemical': 0.20,       # Mycelium + Octopus chemical
            'electrical': 0.15,     # Mycelium electrical
            'visual': 0.20,         # Octopus polarized vision
            'proprioceptive': 0.10, # Octopus proprioception
            'thermal': 0.10         # Thermal/energy
        }
        
        # Calibration history
        self.calibration_history: List[Dict] = []
        self.threat_history: List[ThreatAssessment] = []
        
    def sense_environment(self, network_state: Dict) -> Dict:
        """
        Comprehensive environmental sensing using all biological modalities
        
        Args:
            network_state: Dict containing:
                - node_positions: np.ndarray
                - node_health: np.ndarray
                - node_activity: np.ndarray
                - node_temperatures: np.ndarray
                - node_energy: np.ndarray
                - node_velocities: np.ndarray
                - threat_signatures: Dict
                - light_field: Optional[np.ndarray]
                
        Returns:
            Dict with all sensory data and fused threat assessment
        """
        sensory_data = {}
        
        # Extract network data
        node_positions = network_state.get('node_positions', np.array([]))
        node_health = network_state.get('node_health', np.array([]))
        node_activity = network_state.get('node_activity', np.array([]))
        node_temperatures = network_state.get('node_temperatures', np.array([]))
        node_energy = network_state.get('node_energy', np.array([]))
        node_velocities = network_state.get('node_velocities', np.array([]))
        threat_signatures = network_state.get('threat_signatures', {})
        
        # === MYCELIUM SENSING ===
        if len(node_positions) > 0:
            # Chemical gradient detection
            if 'chemical_signatures' in threat_signatures:
                chemical_gradients = self.mycelium_sensors.detect_chemical_gradient(
                    node_positions, threat_signatures['chemical_signatures']
                )
                sensory_data['chemical_gradients'] = chemical_gradients
            
            # Electrical pattern sensing
            if len(node_activity) > 0:
                electrical_patterns = self.mycelium_sensors.sense_electrical_patterns(node_activity)
                sensory_data['electrical_patterns'] = electrical_patterns
            
            # Vibration detection
            if len(node_activity) > 0:
                vibrations = self.mycelium_sensors.detect_vibrations(node_activity)
                sensory_data['vibrations'] = vibrations
            
            # Mineral concentration sensing
            if len(node_health) > 0:
                mineral_deficiencies = self.mycelium_sensors.sense_mineral_concentrations(node_health)
                sensory_data['mineral_deficiencies'] = mineral_deficiencies
        
        # === OCTOPUS SENSING ===
        if len(node_positions) > 0:
            # Polarized light vision
            if 'light_field' in network_state and network_state['light_field'] is not None:
                polarized_vision = self.octopus_sensors.polarized_light_vision(
                    network_state['light_field']
                )
                sensory_data['polarized_vision'] = polarized_vision
            
            # Chemotactile sensing
            chemotactile = self.octopus_sensors.chemotactile_sensing(
                node_positions, threat_signatures
            )
            sensory_data['chemotactile_detections'] = chemotactile
            
            # Proprioceptive awareness
            if len(node_velocities) > 0:
                proprioceptive = self.octopus_sensors.proprioceptive_awareness(
                    node_positions, node_velocities
                )
                sensory_data['proprioceptive_awareness'] = proprioceptive
        
        # === THERMAL/ENERGY SENSING ===
        if len(node_temperatures) > 0:
            thermal = self.thermal_energy.detect_thermal_anomalies(
                node_temperatures, 
                network_state.get('ambient_temperature', 25.0)
            )
            sensory_data['thermal_sensing'] = thermal
        
        if len(node_energy) > 0:
            energy = self.thermal_energy.detect_energy_surges(node_energy)
            sensory_data['energy_sensing'] = energy
        
        # === SENSOR FUSION ===
        fused_threat_assessment = self.fuse_sensors(sensory_data)
        sensory_data['fused_threat_assessment'] = fused_threat_assessment
        
        return sensory_data
    
    def fuse_sensors(self, sensory_data: Dict) -> Dict:
        """
        Sensor fusion using attention-weighted integration
        
        Combines all sensory modalities into unified threat assessment
        
        Args:
            sensory_data: Dict with all sensor outputs
            
        Returns:
            Dict with threat scores, categories, overall threat level
        """
        num_nodes = self.network_size
        threat_scores = {i: 0.0 for i in range(num_nodes)}
        
        # === VIBRATION-BASED THREAT SCORING ===
        if 'vibrations' in sensory_data:
            vibrations = sensory_data['vibrations']
            vibration_map = vibrations.get('vibration_map', np.zeros(num_nodes))
            
            if len(vibration_map) > 0:
                # Normalize vibration intensity
                max_vib = np.max(vibration_map) if np.max(vibration_map) > 0 else 1.0
                for i in range(min(num_nodes, len(vibration_map))):
                    threat_scores[i] += (vibration_map[i] / max_vib) * self.attention_weights['vibration']
        
        # === ELECTRICAL-BASED THREAT SCORING ===
        if 'electrical_patterns' in sensory_data:
            electrical = sensory_data['electrical_patterns']
            energy_surges = electrical.get('energy_surges', [])
            
            for node in energy_surges:
                if node < num_nodes:
                    threat_scores[node] += self.attention_weights['electrical']
        
        # === THERMAL-BASED THREAT SCORING ===
        if 'thermal_sensing' in sensory_data:
            thermal = sensory_data['thermal_sensing']
            thermal_anomalies = thermal.get('thermal_anomalies', [])
            
            for node in thermal_anomalies:
                if node < num_nodes:
                    threat_scores[node] += self.attention_weights['thermal']
        
        # === ENERGY-BASED THREAT SCORING ===
        if 'energy_sensing' in sensory_data:
            energy = sensory_data['energy_sensing']
            energy_surges = energy.get('energy_surges', [])
            
            for node in energy_surges:
                if node < num_nodes:
                    threat_scores[node] += self.attention_weights['thermal']  # Share thermal weight
        
        # === VISUAL-BASED THREAT SCORING ===
        if 'polarized_vision' in sensory_data:
            vision = sensory_data['polarized_vision']
            anomaly_indices = vision.get('anomaly_indices', [])
            
            for node in anomaly_indices:
                if node < num_nodes:
                    threat_scores[node] += self.attention_weights['visual']
        
        # Normalize threat scores
        max_threat = max(threat_scores.values()) if threat_scores else 1.0
        if max_threat > 0:
            threat_scores = {k: v / max_threat for k, v in threat_scores.items()}
        
        # Categorize threat levels
        threat_categories = {}
        for node, score in threat_scores.items():
            if score > 0.8:
                threat_categories[node] = 'CRITICAL'
            elif score > 0.6:
                threat_categories[node] = 'HIGH'
            elif score > 0.4:
                threat_categories[node] = 'MEDIUM'
            elif score > 0.2:
                threat_categories[node] = 'LOW'
            else:
                threat_categories[node] = 'NORMAL'
        
        # Calculate overall threat level
        overall_threat = np.mean(list(threat_scores.values())) if threat_scores else 0.0
        
        return {
            'threat_scores': threat_scores,
            'threat_categories': threat_categories,
            'overall_threat_level': float(overall_threat),
            'critical_nodes': [k for k, v in threat_categories.items() if v == 'CRITICAL'],
            'high_threat_nodes': [k for k, v in threat_categories.items() if v == 'HIGH']
        }
    
    def update_attention_weights(self, threat_feedback: Dict):
        """
        Adaptive attention mechanism - adjust weights based on threat feedback
        
        Args:
            threat_feedback: Dict with actual threat outcomes
        """
        # Simplified adaptive weighting
        # In production, would use reinforcement learning
        
        if 'successful_modality' in threat_feedback:
            modality = threat_feedback['successful_modality']
            if modality in self.attention_weights:
                # Increase weight for successful modality
                self.attention_weights[modality] = min(0.4, self.attention_weights[modality] * 1.1)
                
                # Normalize weights
                total = sum(self.attention_weights.values())
                self.attention_weights = {k: v/total for k, v in self.attention_weights.items()}

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("🔬 LUMINARK - Bio-Sensory Fusion Demo")
    print("="*70)
    
    fusion = BioSensoryFusion(network_size=50)
    
    # Simulate network state
    network_state = {
        'node_positions': np.random.randn(50, 2) * 10,
        'node_health': np.random.rand(50),
        'node_activity': np.random.randn(50) * 0.5 + 1.0,
        'node_temperatures': np.random.randn(50) * 2 + 37.0,
        'node_energy': np.random.rand(50) * 100,
        'node_velocities': np.random.randn(50, 2) * 0.1,
        'ambient_temperature': 25.0,
        'threat_signatures': {
            'chemical_signatures': np.random.randn(50, 5) * 0.1
        },
        'light_field': np.random.rand(50)
    }
    
    print("\n🌐 Performing comprehensive environmental sensing...")
    sensory_data = fusion.sense_environment(network_state)
    
    print("\n📊 Sensory Data Collected:")
    print(f"  Modalities Active: {len([k for k in sensory_data.keys() if k != 'fused_threat_assessment'])}")
    
    if 'fused_threat_assessment' in sensory_data:
        assessment = sensory_data['fused_threat_assessment']
        print(f"\n⚠️ Threat Assessment:")
        print(f"  Overall Threat Level: {assessment['overall_threat_level']:.3f}")
        print(f"  Critical Nodes: {len(assessment['critical_nodes'])}")
        print(f"  High Threat Nodes: {len(assessment['high_threat_nodes'])}")
    
    print("\n✅ Bio-Sensory Fusion operational!")
