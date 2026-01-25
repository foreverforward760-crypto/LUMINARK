"""
Thermal & Energy Sensing System
Multi-spectrum energy detection with thermal gradients
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class EnergySpectrum(Enum):
    """Different energy spectrums to sense"""
    THERMAL_IR = 0          # Thermal infrared
    NEAR_IR = 1             # Near infrared
    VISIBLE = 2             # Visible light
    UV = 3                  # Ultraviolet
    ELECTROMAGNETIC = 4     # EM fields
    KINETIC = 5            # Movement/vibration energy
    POTENTIAL = 6          # Stored/potential energy
    QUANTUM_FIELD = 7      # Quantum field fluctuations


@dataclass
class ThermalReading:
    """Thermal/energy sensor reading"""
    spectrum: EnergySpectrum
    temperature: float  # Kelvin or normalized
    intensity: float    # 0-1 normalized
    gradient: float     # Rate of change
    direction: Optional[Tuple[float, float, float]] = None  # 3D gradient vector
    timestamp: float = 0.0
    confidence: float = 1.0


class ThermalEnergySensor:
    """
    Advanced thermal and energy sensing
    - Multi-spectrum detection
    - Gradient analysis
    - Temporal tracking
    - Energy flow patterns
    """

    def __init__(self, num_sensors: int = 16, sensitivity: float = 0.01):
        self.num_sensors = num_sensors
        self.sensitivity = sensitivity

        # Initialize sensor array (distributed spatial sensors)
        self.sensors = self._initialize_sensor_array()

        # Historical readings for gradient calculation
        self.history = []
        self.max_history = 1000

        # Energy flow tracking
        self.flow_vectors = []
        self.thermal_map = {}

        print(f"🌡️  Thermal/Energy Sensor initialized")
        print(f"   {num_sensors} multi-spectrum sensors")
        print(f"   Sensitivity: {sensitivity}")

    def _initialize_sensor_array(self) -> List[Dict[str, Any]]:
        """Initialize spatially distributed sensors"""
        sensors = []

        # Create 3D sensor grid
        grid_size = int(np.ceil(self.num_sensors ** (1/3)))

        sensor_id = 0
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    if sensor_id >= self.num_sensors:
                        break

                    sensors.append({
                        'id': sensor_id,
                        'position': np.array([
                            x / max(grid_size-1, 1),
                            y / max(grid_size-1, 1),
                            z / max(grid_size-1, 1)
                        ]),
                        'spectrums': list(EnergySpectrum),
                        'calibration': np.random.uniform(0.95, 1.05),
                        'last_reading': None
                    })
                    sensor_id += 1

        return sensors[:self.num_sensors]

    def sense_thermal_field(self, environmental_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sense complete thermal/energy field
        Returns multi-spectrum analysis with gradients
        """

        readings = []
        current_time = environmental_state.get('timestamp', np.random.random())

        # Read from each sensor
        for sensor in self.sensors:
            sensor_readings = self._read_sensor(sensor, environmental_state, current_time)
            readings.extend(sensor_readings)

        # Calculate thermal gradients
        gradients = self._calculate_thermal_gradients(readings)

        # Detect energy flows
        flows = self._detect_energy_flows(readings, gradients)

        # Analyze spectral distribution
        spectral_analysis = self._analyze_spectrum(readings)

        # Update thermal map
        self._update_thermal_map(readings)

        # Store in history
        self.history.append({
            'timestamp': current_time,
            'readings': readings,
            'gradients': gradients,
            'flows': flows
        })

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return {
            'timestamp': current_time,
            'num_readings': len(readings),
            'readings': readings,
            'thermal_gradients': gradients,
            'energy_flows': flows,
            'spectral_analysis': spectral_analysis,
            'thermal_map': self.thermal_map,
            'hotspots': self._detect_hotspots(readings),
            'coldspots': self._detect_coldspots(readings),
            'anomalies': self._detect_thermal_anomalies(readings)
        }

    def _read_sensor(self, sensor: Dict, environment: Dict, timestamp: float) -> List[ThermalReading]:
        """Read all spectrums from single sensor"""
        readings = []

        pos = sensor['position']
        calibration = sensor['calibration']

        for spectrum in sensor['spectrums']:
            # Get base value from environment
            base_value = environment.get(f'{spectrum.name.lower()}_energy', 0.5)

            # Add spatial variation based on position
            spatial_factor = np.sin(pos[0] * np.pi) * np.cos(pos[1] * np.pi) * np.sin(pos[2] * np.pi)
            spatial_factor = (spatial_factor + 1) / 2  # Normalize to [0, 1]

            # Combine with noise
            noise = np.random.normal(0, self.sensitivity)
            measured_value = (base_value * spatial_factor + noise) * calibration

            # Clamp to [0, 1]
            measured_value = np.clip(measured_value, 0, 1)

            # Calculate gradient if we have history
            gradient = self._calculate_temporal_gradient(
                sensor['id'],
                spectrum,
                measured_value,
                timestamp
            )

            # Convert to temperature (simplified)
            temperature = 273.15 + measured_value * 100  # 0-100°C range mapped to Kelvin

            reading = ThermalReading(
                spectrum=spectrum,
                temperature=temperature,
                intensity=measured_value,
                gradient=gradient,
                direction=self._estimate_gradient_direction(sensor, measured_value),
                timestamp=timestamp,
                confidence=calibration
            )

            readings.append(reading)

        sensor['last_reading'] = readings
        return readings

    def _calculate_temporal_gradient(self, sensor_id: int, spectrum: EnergySpectrum,
                                    current_value: float, current_time: float) -> float:
        """Calculate rate of change over time"""
        if not self.history:
            return 0.0

        # Find last reading from this sensor for this spectrum
        for entry in reversed(self.history[-10:]):  # Check last 10 entries
            for reading in entry['readings']:
                if (hasattr(reading, 'spectrum') and
                    reading.spectrum == spectrum):
                    dt = current_time - entry['timestamp']
                    if dt > 0:
                        dvalue = current_value - reading.intensity
                        return dvalue / dt

        return 0.0

    def _estimate_gradient_direction(self, sensor: Dict, value: float) -> Optional[Tuple[float, float, float]]:
        """Estimate 3D gradient direction using neighboring sensors"""
        pos = sensor['position']

        # Find nearby sensors
        directions = []

        for other_sensor in self.sensors:
            if other_sensor['id'] == sensor['id']:
                continue

            other_pos = other_sensor['position']
            distance = np.linalg.norm(pos - other_pos)

            if distance < 0.5 and other_sensor['last_reading']:  # Within threshold
                # Get comparable reading
                other_value = np.mean([r.intensity for r in other_sensor['last_reading']])

                # Calculate gradient vector
                diff = other_value - value
                direction = (other_pos - pos) / (distance + 1e-10)

                directions.append(direction * diff)

        if directions:
            avg_direction = np.mean(directions, axis=0)
            return tuple(avg_direction)

        return None

    def _calculate_thermal_gradients(self, readings: List[ThermalReading]) -> Dict[str, Any]:
        """Calculate spatial and temporal thermal gradients"""
        temporal_gradients = [r.gradient for r in readings]

        return {
            'mean_temporal_gradient': np.mean(temporal_gradients) if temporal_gradients else 0,
            'max_temporal_gradient': np.max(np.abs(temporal_gradients)) if temporal_gradients else 0,
            'gradient_variance': np.var(temporal_gradients) if len(temporal_gradients) > 1 else 0,
            'heating_rate': np.sum([g for g in temporal_gradients if g > 0]),
            'cooling_rate': np.sum([g for g in temporal_gradients if g < 0])
        }

    def _detect_energy_flows(self, readings: List[ThermalReading],
                            gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Detect energy flow patterns"""

        # Extract direction vectors
        vectors_with_direction = [r for r in readings if r.direction is not None]

        if not vectors_with_direction:
            return {
                'flow_detected': False,
                'primary_direction': None,
                'flow_magnitude': 0
            }

        # Calculate mean flow direction
        directions = np.array([r.direction for r in vectors_with_direction])
        mean_direction = np.mean(directions, axis=0)

        # Normalize
        flow_magnitude = np.linalg.norm(mean_direction)
        if flow_magnitude > 0:
            mean_direction = mean_direction / flow_magnitude

        return {
            'flow_detected': flow_magnitude > 0.1,
            'primary_direction': tuple(mean_direction),
            'flow_magnitude': float(flow_magnitude),
            'num_vectors': len(vectors_with_direction),
            'flow_type': self._classify_flow_type(mean_direction, flow_magnitude)
        }

    def _classify_flow_type(self, direction: np.ndarray, magnitude: float) -> str:
        """Classify type of energy flow"""
        if magnitude < 0.1:
            return 'stagnant'
        elif magnitude < 0.3:
            return 'diffuse'
        elif magnitude < 0.6:
            return 'laminar'
        else:
            return 'turbulent'

    def _analyze_spectrum(self, readings: List[ThermalReading]) -> Dict[str, Any]:
        """Analyze spectral distribution of energy"""
        spectrum_data = {}

        for spectrum in EnergySpectrum:
            spectrum_readings = [r for r in readings if r.spectrum == spectrum]

            if spectrum_readings:
                intensities = [r.intensity for r in spectrum_readings]
                temperatures = [r.temperature for r in spectrum_readings]

                spectrum_data[spectrum.name] = {
                    'mean_intensity': np.mean(intensities),
                    'max_intensity': np.max(intensities),
                    'mean_temperature': np.mean(temperatures),
                    'num_sensors': len(spectrum_readings)
                }

        return spectrum_data

    def _update_thermal_map(self, readings: List[ThermalReading]):
        """Update 3D thermal map"""
        for reading in readings:
            key = reading.spectrum.name
            if key not in self.thermal_map:
                self.thermal_map[key] = {
                    'current': reading.intensity,
                    'history': []
                }

            self.thermal_map[key]['current'] = reading.intensity
            self.thermal_map[key]['history'].append(reading.intensity)

            # Keep history bounded
            if len(self.thermal_map[key]['history']) > 100:
                self.thermal_map[key]['history'] = self.thermal_map[key]['history'][-100:]

    def _detect_hotspots(self, readings: List[ThermalReading]) -> List[Dict[str, Any]]:
        """Detect thermal hotspots (high energy regions)"""
        intensities = [r.intensity for r in readings]
        if not intensities:
            return []

        threshold = np.mean(intensities) + 2 * np.std(intensities) if len(intensities) > 1 else 0.8

        hotspots = []
        for reading in readings:
            if reading.intensity > threshold:
                hotspots.append({
                    'spectrum': reading.spectrum.name,
                    'intensity': reading.intensity,
                    'temperature': reading.temperature,
                    'severity': 'critical' if reading.intensity > 0.9 else 'warning'
                })

        return hotspots

    def _detect_coldspots(self, readings: List[ThermalReading]) -> List[Dict[str, Any]]:
        """Detect cold spots (low energy regions)"""
        intensities = [r.intensity for r in readings]
        if not intensities:
            return []

        threshold = np.mean(intensities) - 2 * np.std(intensities) if len(intensities) > 1 else 0.2

        coldspots = []
        for reading in readings:
            if reading.intensity < threshold:
                coldspots.append({
                    'spectrum': reading.spectrum.name,
                    'intensity': reading.intensity,
                    'temperature': reading.temperature
                })

        return coldspots

    def _detect_thermal_anomalies(self, readings: List[ThermalReading]) -> List[Dict[str, Any]]:
        """Detect unusual thermal patterns"""
        anomalies = []

        # Check for rapid temperature changes
        rapid_changes = [r for r in readings if abs(r.gradient) > 0.5]
        if rapid_changes:
            anomalies.append({
                'type': 'rapid_thermal_change',
                'count': len(rapid_changes),
                'max_gradient': max([abs(r.gradient) for r in rapid_changes]),
                'severity': 'high'
            })

        # Check for spectrum inconsistencies
        spectrum_means = {}
        for reading in readings:
            if reading.spectrum.name not in spectrum_means:
                spectrum_means[reading.spectrum.name] = []
            spectrum_means[reading.spectrum.name].append(reading.intensity)

        for spectrum, values in spectrum_means.items():
            variance = np.var(values) if len(values) > 1 else 0
            if variance > 0.3:  # High variance
                anomalies.append({
                    'type': 'spectrum_inconsistency',
                    'spectrum': spectrum,
                    'variance': variance,
                    'severity': 'medium'
                })

        return anomalies


if __name__ == '__main__':
    # Demo
    print("🌡️  Thermal/Energy Sensing Demo\n")

    sensor = ThermalEnergySensor(num_sensors=16, sensitivity=0.01)

    # Simulate environment
    environment = {
        'thermal_ir_energy': 0.7,
        'near_ir_energy': 0.6,
        'visible_energy': 0.8,
        'uv_energy': 0.3,
        'electromagnetic_energy': 0.5,
        'kinetic_energy': 0.65,
        'potential_energy': 0.4,
        'quantum_field_energy': 0.55,
        'timestamp': 1.0
    }

    print("📡 Sensing thermal/energy field...")
    result = sensor.sense_thermal_field(environment)

    print(f"\n📊 Results:")
    print(f"   Total Readings: {result['num_readings']}")
    print(f"   Thermal Gradients: {result['thermal_gradients']}")
    print(f"\n   Energy Flows:")
    for key, val in result['energy_flows'].items():
        print(f"      {key}: {val}")

    print(f"\n   Spectral Analysis:")
    for spectrum, data in result['spectral_analysis'].items():
        print(f"      {spectrum}: intensity={data['mean_intensity']:.3f}, temp={data['mean_temperature']:.1f}K")

    if result['hotspots']:
        print(f"\n   🔥 Hotspots Detected: {len(result['hotspots'])}")
        for hotspot in result['hotspots'][:3]:
            print(f"      {hotspot['spectrum']}: {hotspot['intensity']:.3f} ({hotspot['severity']})")

    if result['anomalies']:
        print(f"\n   ⚠️  Anomalies Detected: {len(result['anomalies'])}")
        for anomaly in result['anomalies']:
            print(f"      {anomaly['type']}: {anomaly.get('severity', 'unknown')}")
