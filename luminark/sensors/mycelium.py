"""
LUMINARK - Mycelial Sensory System
Inspired by Armillaria ostoyae (world's largest organism: 2,400 acres, 2,500 years)

Capabilities:
- Chemical gradient detection (calcium, potassium, pH)
- Electrical signal sensing (conductivity, resonance)
- Vibration detection (0.1-100 Hz biological range)
- Mineral concentration sensing
- Network-wide threat propagation
"""

import numpy as np
from scipy import signal, fft
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MycelialSignal:
    """Represents a signal detected by the mycelial network"""
    signal_type: str  # 'chemical', 'electrical', 'vibration', 'mineral'
    intensity: float  # 0.0 to 1.0
    location: Tuple[float, float]  # (x, y) coordinates
    timestamp: float
    propagation_speed: float  # m/s
    
class MyceliumSensorySystem:
    """
    World's Largest Mycelium (2,400 acres, 2,500 years) Sensory Capabilities
    
    Biological basis:
    - Conductivity: 0.85 S/m (biological tissue)
    - Signal velocity: 0.5 m/s in fungal hyphae
    - Resonance frequencies: 7, 14, 28, 42 Hz (mycelial network)
    """
    
    def __init__(self, network_size: int = 100):
        self.network_size = network_size
        self.conductivity = 0.85  # S/m - biological tissue conductivity
        self.signal_velocity = 0.5  # m/s in fungal hyphae
        self.resonance_frequencies = [7, 14, 28, 42]  # Hz - mycelial network resonance
        
        # Signal history for pattern detection
        self.signal_history: List[MycelialSignal] = []
        self.max_history = 1000
        
    def detect_chemical_gradient(self, 
                                node_positions: np.ndarray, 
                                threat_chemicals: np.ndarray) -> np.ndarray:
        """
        Mycelium detects chemical gradients (calcium, potassium, pH changes)
        
        Args:
            node_positions: Array of (x, y) positions for each node
            threat_chemicals: Chemical concentrations at each position
            
        Returns:
            Gradient field showing chemical threat distribution
        """
        if len(node_positions) == 0 or len(threat_chemicals) == 0:
            return np.array([])
            
        gradient = np.zeros((len(node_positions), len(threat_chemicals)))
        
        for i, pos in enumerate(node_positions):
            # Calculate distances to all other nodes
            distances = np.linalg.norm(node_positions - pos, axis=1)
            
            # Exponential attenuation with distance (biological diffusion)
            attenuation = np.exp(-distances / 10.0)
            
            # Calculate chemical field at this position
            chemical_field = np.sum(threat_chemicals * attenuation[:, np.newaxis], axis=0)
            gradient[i] = chemical_field
            
        return gradient
    
    def sense_electrical_patterns(self, node_activity: np.ndarray) -> Dict:
        """
        Mycelium conducts electrical signals, detects energy surges
        
        Biological basis:
        - Action potentials in fungal hyphae
        - Electrical signaling for nutrient coordination
        - Resonance at specific frequencies
        
        Args:
            node_activity: Electrical activity at each node
            
        Returns:
            Dict with resonance frequencies, energy surges, dominant frequency
        """
        if len(node_activity) == 0:
            return {
                'resonance_frequencies': [],
                'energy_surges': [],
                'total_power': 0.0,
                'dominant_frequency': 0.0
            }
        
        # FFT analysis for frequency detection
        frequencies = fft.fftfreq(len(node_activity), 0.01)  # 100 Hz sampling
        power_spectrum = np.abs(fft.fft(node_activity))**2
        
        # Detect resonance at mycelial frequencies
        resonance_detected = []
        for freq in self.resonance_frequencies:
            idx = np.argmin(np.abs(frequencies - freq))
            if power_spectrum[idx] > np.mean(power_spectrum) * 3:
                resonance_detected.append(freq)
        
        # Detect energy surges (3 sigma above mean)
        surge_threshold = np.mean(node_activity) + 3 * np.std(node_activity)
        surge_mask = node_activity > surge_threshold
        surge_nodes = np.where(surge_mask)[0]
        
        return {
            'resonance_frequencies': resonance_detected,
            'energy_surges': surge_nodes.tolist(),
            'total_power': float(np.sum(power_spectrum)),
            'dominant_frequency': float(frequencies[np.argmax(power_spectrum)])
        }
    
    def detect_vibrations(self, node_movements: np.ndarray) -> Dict:
        """
        Mycelium senses soil vibrations (0.1-100 Hz biological range)
        
        Biological basis:
        - Mechanoreceptors in fungal cell walls
        - Vibration sensing for predator/prey detection
        - Rhythmic pattern recognition
        
        Args:
            node_movements: Movement/vibration data for each node
            
        Returns:
            Dict with vibration intensity, rhythmic patterns, anomalies
        """
        if len(node_movements) == 0:
            return {
                'vibration_intensity': 0.0,
                'rhythmic_patterns': None,
                'vibration_map': np.array([]),
                'anomalous_vibrations': []
            }
        
        vibrations = np.zeros(len(node_movements))
        
        # Continuous Wavelet Transform for vibration analysis
        for i, movement in enumerate(node_movements):
            # Use Ricker wavelet (Mexican hat) for biological vibrations
            coefficients, _ = signal.cwt(movement, signal.ricker, np.arange(1, 31))
            
            # Focus on biological frequency range (5-25 Hz)
            dominant_vibration = np.mean(np.abs(coefficients[5:25]))
            vibrations[i] = dominant_vibration
        
        # Detect rhythmic patterns using autocorrelation
        autocorr = np.correlate(node_movements, node_movements, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
        
        # Find periodicity (peaks in autocorrelation)
        peaks = signal.find_peaks(autocorr, height=np.max(autocorr) * 0.5)[0]
        periodicity = peaks[0] if len(peaks) > 0 else None
        
        # Detect anomalous vibrations (2 sigma above mean)
        anomaly_threshold = np.mean(vibrations) + 2 * np.std(vibrations)
        anomalous_nodes = np.where(vibrations > anomaly_threshold)[0]
        
        return {
            'vibration_intensity': float(np.mean(vibrations)),
            'rhythmic_patterns': int(periodicity) if periodicity else None,
            'vibration_map': vibrations,
            'anomalous_vibrations': anomalous_nodes.tolist()
        }
    
    def sense_mineral_concentrations(self, node_health: np.ndarray) -> Dict:
        """
        Armillaria detects mineral imbalances (Ca²⁺, K⁺, Mg²⁺, Fe³⁺)
        
        Biological basis:
        - Ion channels in fungal membranes
        - Nutrient sensing for growth optimization
        - Mineral deficiency detection
        
        Args:
            node_health: Health status of each node (0.0 to 1.0)
            
        Returns:
            Dict with mineral deficiencies by type
        """
        if len(node_health) == 0:
            return {
                'calcium_deficit': [],
                'potassium_deficit': [],
                'magnesium_deficit': [],
                'iron_deficit': []
            }
        
        # Simulate mineral concentrations based on health
        # (In real system, would use actual sensor data)
        calcium = node_health * 0.7 + np.random.normal(0, 0.1, len(node_health))
        potassium = node_health * 0.5 + np.random.normal(0, 0.08, len(node_health))
        magnesium = node_health * 0.3 + np.random.normal(0, 0.05, len(node_health))
        iron = node_health * 0.4 + np.random.normal(0, 0.06, len(node_health))
        
        return {
            'calcium_deficit': np.where(calcium < 0.4)[0].tolist(),
            'potassium_deficit': np.where(potassium < 0.3)[0].tolist(),
            'magnesium_deficit': np.where(magnesium < 0.2)[0].tolist(),
            'iron_deficit': np.where(iron < 0.25)[0].tolist()
        }
    
    def propagate_threat_signal(self, 
                                threat_location: Tuple[float, float],
                                node_positions: np.ndarray) -> Dict:
        """
        Simulate threat signal propagation through mycelial network
        
        Args:
            threat_location: (x, y) coordinates of threat
            node_positions: Positions of all nodes
            
        Returns:
            Dict with arrival times and signal strength at each node
        """
        if len(node_positions) == 0:
            return {
                'arrival_times': np.array([]),
                'signal_strength': np.array([])
            }
        
        threat_pos = np.array(threat_location)
        
        # Calculate distances from threat to each node
        distances = np.linalg.norm(node_positions - threat_pos, axis=1)
        
        # Calculate arrival times (distance / signal velocity)
        arrival_times = distances / self.signal_velocity
        
        # Calculate signal strength (exponential decay with distance)
        signal_strength = np.exp(-distances / 20.0)
        
        return {
            'arrival_times': arrival_times,
            'signal_strength': signal_strength,
            'max_propagation_time': float(np.max(arrival_times)),
            'affected_nodes': np.where(signal_strength > 0.1)[0].tolist()
        }
    
    def compartmentalize_threat(self, 
                                threat_nodes: List[int],
                                network_graph) -> Dict:
        """
        Mycelial compartmentalization: Isolate infected sections
        
        Biological basis:
        - Septa formation to seal off infected hyphae
        - Programmed cell death in damaged areas
        - Resource reallocation away from threats
        
        Args:
            threat_nodes: List of compromised node IDs
            network_graph: NetworkX graph of the system
            
        Returns:
            Dict with isolation strategy and affected connections
        """
        isolated_connections = []
        preserved_nodes = []
        
        for threat_node in threat_nodes:
            if threat_node in network_graph.nodes:
                # Get all neighbors
                neighbors = list(network_graph.neighbors(threat_node))
                
                # Remove edges to isolate threat
                for neighbor in neighbors:
                    if neighbor not in threat_nodes:
                        isolated_connections.append((threat_node, neighbor))
                        network_graph.remove_edge(threat_node, neighbor)
                        preserved_nodes.append(neighbor)
        
        return {
            'isolated_nodes': threat_nodes,
            'severed_connections': isolated_connections,
            'preserved_nodes': list(set(preserved_nodes)),
            'compartments_created': len(threat_nodes)
        }

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("🍄 LUMINARK - Mycelial Sensory System Demo")
    print("="*70)
    
    mycelium = MyceliumSensorySystem(network_size=50)
    
    # Simulate network
    node_positions = np.random.randn(50, 2) * 10
    node_activity = np.random.randn(50) * 0.5 + 1.0
    node_health = np.random.rand(50)
    
    # Test electrical sensing
    print("\n🔌 Electrical Pattern Sensing:")
    electrical = mycelium.sense_electrical_patterns(node_activity)
    print(f"  Resonance Frequencies: {electrical['resonance_frequencies']} Hz")
    print(f"  Energy Surges: {len(electrical['energy_surges'])} nodes")
    print(f"  Dominant Frequency: {electrical['dominant_frequency']:.2f} Hz")
    
    # Test vibration sensing
    print("\n🌊 Vibration Detection:")
    vibrations = mycelium.detect_vibrations(node_activity)
    print(f"  Vibration Intensity: {vibrations['vibration_intensity']:.3f}")
    print(f"  Rhythmic Pattern: {vibrations['rhythmic_patterns']}")
    print(f"  Anomalous Nodes: {len(vibrations['anomalous_vibrations'])}")
    
    # Test threat propagation
    print("\n⚠️ Threat Signal Propagation:")
    threat_loc = (5.0, 5.0)
    propagation = mycelium.propagate_threat_signal(threat_loc, node_positions)
    print(f"  Max Propagation Time: {propagation['max_propagation_time']:.2f}s")
    print(f"  Affected Nodes: {len(propagation['affected_nodes'])}")
    
    print("\n✅ Mycelial Sensory System operational!")
