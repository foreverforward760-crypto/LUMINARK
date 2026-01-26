"""
LUMINARK - Octopus Sensory System
Inspired by Cephalopoda (500M neurons, distributed intelligence)

Capabilities:
- Polarized light vision (invisible to humans)
- Chemotactile sensing (10,000+ receptors per sucker)
- Proprioceptive awareness (distributed processing)
- Adaptive camouflage (pattern matching)
- Distributed decision-making (2/3 neurons in arms)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial import distance

@dataclass
class CamouflagePattern:
    """Represents an octopus camouflage pattern"""
    pattern_type: str  # 'mimicry', 'disruptive', 'countershading'
    complexity: float  # 0.0 to 1.0
    frequency: float  # Pattern frequency
    effectiveness: float  # 0.0 to 1.0

class OctopusSensorySystem:
    """
    Octopus Sensory Capabilities (500M neurons, distributed intelligence)
    
    Biological basis:
    - 500 million neurons (2/3 in arms, distributed processing)
    - Polarized light vision (invisible patterns to humans)
    - 10,000+ chemoreceptors per sucker
    - Chromatophores for instant camouflage
    - Proprioceptive awareness without central control
    """
    
    def __init__(self):
        self.total_neurons = 500_000_000
        self.arm_neurons = int(self.total_neurons * 2/3)  # 2/3 in arms
        self.brain_neurons = int(self.total_neurons * 1/3)  # 1/3 in brain
        
        # Polarization angles for vision (36 angles, 10° increments)
        self.polarization_angles = np.linspace(0, 180, 36)
        
        # Camouflage pattern library
        self.camouflage_patterns = {
            'mimicry': CamouflagePattern('mimicry', 0.8, 0.3, 0.9),
            'disruptive': CamouflagePattern('disruptive', 0.6, 0.5, 0.8),
            'countershading': CamouflagePattern('countershading', 0.4, 0.2, 0.7),
            'uniform': CamouflagePattern('uniform', 0.2, 0.1, 0.5)
        }
        
        # Sucker chemical memory
        self.chemical_memory: Dict[int, List[Dict]] = {}
        
        # Chromatophore states (color-changing cells)
        self.chromatophore_states = np.random.rand(1000)  # 1000 chromatophores
        
    def polarized_light_vision(self, light_field: np.ndarray) -> Dict:
        """
        Octopus sees polarized light patterns (invisible to humans)
        
        Biological basis:
        - Rhabdomeric photoreceptors sensitive to polarization
        - Can detect polarization angle and degree
        - Used for communication and prey detection
        
        Args:
            light_field: Light intensity field
            
        Returns:
            Dict with polarization patterns, entropy, anomalies
        """
        if light_field is None or len(light_field) == 0:
            return {
                'polarization_pattern': np.array([]),
                'polarization_entropy': np.array([]),
                'anomaly_indices': [],
                'pattern_complexity': 0.0
            }
        
        # Calculate polarization vectors for each angle
        polarization_vectors = np.array([
            light_field * np.cos(np.deg2rad(angle))
            for angle in self.polarization_angles
        ])
        
        # Calculate polarization entropy (information content)
        # Normalize to probability distribution
        prob_dist = np.abs(polarization_vectors) / (np.sum(np.abs(polarization_vectors), axis=0) + 1e-10)
        
        # Shannon entropy
        polarization_entropy = -np.sum(
            prob_dist * np.log2(prob_dist + 1e-10),
            axis=0
        )
        
        # Detect anomalies (high entropy = unusual patterns)
        entropy_threshold = np.mean(polarization_entropy) * 1.5
        anomalies = np.where(polarization_entropy > entropy_threshold)[0]
        
        return {
            'polarization_pattern': polarization_vectors,
            'polarization_entropy': polarization_entropy,
            'anomaly_indices': anomalies.tolist(),
            'pattern_complexity': float(np.std(polarization_entropy)),
            'dominant_angle': float(self.polarization_angles[np.argmax(np.mean(np.abs(polarization_vectors), axis=1))])
        }
    
    def chemotactile_sensing(self, 
                            node_positions: np.ndarray, 
                            chemical_signatures: Dict) -> Dict:
        """
        Octopus suckers taste what they touch (10,000+ receptors per sucker)
        
        Biological basis:
        - Chemoreceptors in sucker epithelium
        - Can identify objects by taste alone
        - Chemical memory for learned substances
        
        Args:
            node_positions: Positions of contact points
            chemical_signatures: Chemical profiles at each point
            
        Returns:
            Dict with chemical detections and novelty scores
        """
        if len(node_positions) == 0:
            return {
                'detections': {},
                'novel_chemicals': [],
                'familiar_chemicals': []
            }
        
        chemical_detections = {}
        novel_chemicals = []
        familiar_chemicals = []
        
        for node_id, position in enumerate(node_positions):
            detections = []
            
            # Check each chemical signature
            for chem_name, chem_field in chemical_signatures.items():
                concentration = chem_field.get(node_id, 0) if isinstance(chem_field, dict) else 0
                
                if concentration > 0.1:  # Detection threshold
                    # Check if chemical is in memory
                    is_novel = node_id not in self.chemical_memory or \
                              chem_name not in [d['chemical'] for d in self.chemical_memory.get(node_id, [])]
                    
                    detection = {
                        'chemical': chem_name,
                        'concentration': float(concentration),
                        'novelty': 1.0 if is_novel else 0.3,
                        'position': position.tolist() if isinstance(position, np.ndarray) else position
                    }
                    
                    detections.append(detection)
                    
                    if is_novel:
                        novel_chemicals.append(chem_name)
                    else:
                        familiar_chemicals.append(chem_name)
                    
                    # Store in memory
                    if node_id not in self.chemical_memory:
                        self.chemical_memory[node_id] = []
                    self.chemical_memory[node_id].append(detection)
            
            if detections:
                chemical_detections[node_id] = detections
        
        return {
            'detections': chemical_detections,
            'novel_chemicals': list(set(novel_chemicals)),
            'familiar_chemicals': list(set(familiar_chemicals)),
            'total_detections': sum(len(d) for d in chemical_detections.values())
        }
    
    def proprioceptive_awareness(self, 
                                node_positions: np.ndarray, 
                                node_velocities: np.ndarray) -> Dict:
        """
        Octopus knows arm positions without looking (distributed processing)
        
        Biological basis:
        - Each arm has autonomous nervous system
        - Proprioceptors in muscles and skin
        - Distributed decision-making (arms can act independently)
        
        Args:
            node_positions: Current positions of all nodes
            node_velocities: Velocity vectors for each node
            
        Returns:
            Dict with position uncertainty, movement synchrony, collective awareness
        """
        if len(node_positions) == 0:
            return {
                'position_uncertainty': np.array([]),
                'movement_synchrony': np.array([]),
                'proprioceptive_anomalies': [],
                'collective_awareness': 0.0
            }
        
        num_nodes = len(node_positions)
        position_uncertainty = np.zeros(num_nodes)
        movement_synchrony = np.zeros((num_nodes, num_nodes))
        
        # Calculate position uncertainty (distance from center)
        center = np.mean(node_positions, axis=0)
        for i in range(num_nodes):
            distance_from_center = np.linalg.norm(node_positions[i] - center)
            # Uncertainty increases with distance (harder to sense distant parts)
            position_uncertainty[i] = distance_from_center * 0.1
        
        # Calculate movement synchrony (how coordinated are movements?)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and i < len(node_velocities) and j < len(node_velocities):
                    vel_i = node_velocities[i]
                    vel_j = node_velocities[j]
                    
                    norm_i = np.linalg.norm(vel_i)
                    norm_j = np.linalg.norm(vel_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        # Cosine similarity of velocity vectors
                        velocity_correlation = np.dot(vel_i, vel_j) / (norm_i * norm_j)
                        movement_synchrony[i, j] = velocity_correlation
        
        # Detect proprioceptive anomalies (high uncertainty)
        anomaly_threshold = np.mean(position_uncertainty) + np.std(position_uncertainty)
        proprioceptive_anomalies = np.where(position_uncertainty > anomaly_threshold)[0]
        
        # Calculate collective awareness (average synchrony)
        positive_synchrony = movement_synchrony[movement_synchrony > 0]
        collective_awareness = np.mean(positive_synchrony) if len(positive_synchrony) > 0 else 0.0
        
        return {
            'position_uncertainty': position_uncertainty,
            'movement_synchrony': movement_synchrony,
            'proprioceptive_anomalies': proprioceptive_anomalies.tolist(),
            'collective_awareness': float(collective_awareness),
            'distributed_intelligence': float(self.arm_neurons / self.total_neurons)
        }
    
    def adaptive_camouflage(self, 
                           background_pattern: np.ndarray,
                           threat_level: float = 0.5) -> Dict:
        """
        Octopus adaptive camouflage (instant pattern matching)
        
        Biological basis:
        - Chromatophores (color), iridophores (iridescence), leucophores (white)
        - Can change color/pattern in <1 second
        - Pattern matching without visual feedback (distributed control)
        
        Args:
            background_pattern: Pattern to match
            threat_level: 0.0 (calm) to 1.0 (extreme threat)
            
        Returns:
            Dict with selected pattern, effectiveness, chromatophore states
        """
        # Select pattern based on threat level and background complexity
        background_complexity = np.std(background_pattern) if len(background_pattern) > 0 else 0.5
        
        if threat_level > 0.7:
            # High threat: Disruptive coloration
            selected_pattern = self.camouflage_patterns['disruptive']
        elif background_complexity > 0.6:
            # Complex background: Mimicry
            selected_pattern = self.camouflage_patterns['mimicry']
        elif threat_level < 0.3:
            # Low threat: Uniform coloration
            selected_pattern = self.camouflage_patterns['uniform']
        else:
            # Medium threat: Countershading
            selected_pattern = self.camouflage_patterns['countershading']
        
        # Update chromatophore states
        # Simulate pattern generation
        pattern_frequency = selected_pattern.frequency
        self.chromatophore_states = np.sin(
            np.linspace(0, 2 * np.pi * pattern_frequency, len(self.chromatophore_states))
        ) * selected_pattern.complexity
        
        # Calculate effectiveness (how well does it match background?)
        if len(background_pattern) > 0:
            # Simplified matching score
            pattern_match = 1.0 - np.abs(np.mean(self.chromatophore_states) - np.mean(background_pattern))
            effectiveness = selected_pattern.effectiveness * pattern_match
        else:
            effectiveness = selected_pattern.effectiveness
        
        return {
            'pattern_type': selected_pattern.pattern_type,
            'complexity': selected_pattern.complexity,
            'effectiveness': float(effectiveness),
            'chromatophore_states': self.chromatophore_states,
            'activation_time': 0.8,  # seconds (biological reality)
            'threat_response': 'active' if threat_level > 0.5 else 'passive'
        }
    
    def distributed_decision(self, 
                            arm_inputs: List[Dict],
                            consensus_threshold: float = 0.6) -> Dict:
        """
        Distributed decision-making (arms can act semi-independently)
        
        Biological basis:
        - Each arm has ~40 million neurons
        - Can make local decisions without brain input
        - Consensus emerges from distributed processing
        
        Args:
            arm_inputs: List of sensory inputs from each arm
            consensus_threshold: Required agreement level (0.0 to 1.0)
            
        Returns:
            Dict with decision, consensus level, dissenting arms
        """
        if not arm_inputs:
            return {
                'decision': 'no_action',
                'consensus': 0.0,
                'participating_arms': 0,
                'dissenting_arms': []
            }
        
        # Count votes for each action
        action_votes = {}
        for i, arm_input in enumerate(arm_inputs):
            action = arm_input.get('suggested_action', 'no_action')
            if action not in action_votes:
                action_votes[action] = []
            action_votes[action].append(i)
        
        # Find majority action
        majority_action = max(action_votes.items(), key=lambda x: len(x[1]))
        decision = majority_action[0]
        supporting_arms = majority_action[1]
        
        # Calculate consensus
        consensus = len(supporting_arms) / len(arm_inputs)
        
        # Find dissenting arms
        dissenting_arms = [i for i in range(len(arm_inputs)) if i not in supporting_arms]
        
        # Determine if consensus is sufficient
        decision_approved = consensus >= consensus_threshold
        
        return {
            'decision': decision if decision_approved else 'no_consensus',
            'consensus': float(consensus),
            'participating_arms': len(arm_inputs),
            'supporting_arms': supporting_arms,
            'dissenting_arms': dissenting_arms,
            'decision_approved': decision_approved,
            'distributed_processing': True
        }

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("🐙 LUMINARK - Octopus Sensory System Demo")
    print("="*70)
    
    octopus = OctopusSensorySystem()
    
    print(f"\n🧠 Neural Architecture:")
    print(f"  Total Neurons: {octopus.total_neurons:,}")
    print(f"  Brain Neurons: {octopus.brain_neurons:,} (33%)")
    print(f"  Arm Neurons: {octopus.arm_neurons:,} (67%)")
    
    # Test polarized light vision
    print("\n👁️ Polarized Light Vision:")
    light_field = np.random.rand(100)
    vision = octopus.polarized_light_vision(light_field)
    print(f"  Pattern Complexity: {vision['pattern_complexity']:.3f}")
    print(f"  Anomalies Detected: {len(vision['anomaly_indices'])}")
    print(f"  Dominant Angle: {vision['dominant_angle']:.1f}°")
    
    # Test camouflage
    print("\n🎨 Adaptive Camouflage:")
    background = np.random.rand(100)
    camouflage = octopus.adaptive_camouflage(background, threat_level=0.8)
    print(f"  Pattern Type: {camouflage['pattern_type']}")
    print(f"  Effectiveness: {camouflage['effectiveness']:.2f}")
    print(f"  Activation Time: {camouflage['activation_time']}s")
    
    # Test distributed decision
    print("\n🤝 Distributed Decision-Making:")
    arm_inputs = [
        {'suggested_action': 'retreat'},
        {'suggested_action': 'retreat'},
        {'suggested_action': 'attack'},
        {'suggested_action': 'retreat'},
    ]
    decision = octopus.distributed_decision(arm_inputs)
    print(f"  Decision: {decision['decision']}")
    print(f"  Consensus: {decision['consensus']:.1%}")
    print(f"  Dissenting Arms: {decision['dissenting_arms']}")
    
    print("\n✅ Octopus Sensory System operational!")
