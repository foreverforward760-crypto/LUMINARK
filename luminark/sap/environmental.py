"""
LUMINARK - Environmental Metrics Module
Monitors environmental harmony for SAP framework alignment

Metrics:
- Temperature harmony (thermal gradient smoothness)
- Light quality (full spectrum + circadian appropriate)
- Air vitality (O2/CO2 balance, negative ions)
- Sound harmonics (beneficial frequency presence)
- Spatial flow (Qi/prana flow through spaces)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EnvironmentalState:
    """Complete environmental state"""
    temperature_harmony: float  # 0-1
    light_quality: float  # 0-1
    air_vitality: float  # 0-1
    sound_harmonics: float  # 0-1
    spatial_flow: float  # 0-1
    overall_harmony: float  # 0-1
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'temperature_harmony': self.temperature_harmony,
            'light_quality': self.light_quality,
            'air_vitality': self.air_vitality,
            'sound_harmonics': self.sound_harmonics,
            'spatial_flow': self.spatial_flow,
            'overall_harmony': self.overall_harmony,
            'timestamp': self.timestamp.isoformat()
        }

class EnvironmentalMetrics:
    """
    Monitors environmental conditions for optimal SAP alignment
    """
    
    def __init__(self):
        self.history: List[EnvironmentalState] = []
        self.max_history = 1000
        
        # Optimal ranges
        self.optimal_temperature_range = (20.0, 24.0)  # Celsius
        self.optimal_humidity_range = (40.0, 60.0)  # Percent
        self.optimal_co2_range = (400, 1000)  # ppm
        self.optimal_light_temp_day = (5000, 6500)  # Kelvin
        self.optimal_light_temp_night = (2700, 3500)  # Kelvin
        
        # Beneficial frequencies (Hz)
        self.beneficial_frequencies = {
            'solfeggio': [174, 285, 396, 417, 528, 639, 741, 852, 963],
            'schumann': [7.83, 14.3, 20.8, 27.3, 33.8],  # Earth's resonance
            'binaural': [4, 8, 13, 30, 40]  # Brain wave entrainment
        }
    
    def assess_temperature_harmony(self, 
                                   temperatures: np.ndarray,
                                   ambient_temp: float = 22.0) -> Dict:
        """
        Assess thermal gradient smoothness
        
        Smooth gradients = better harmony
        Sharp gradients = discomfort
        
        Args:
            temperatures: Temperature readings from different locations
            ambient_temp: Target ambient temperature
            
        Returns:
            Dict with harmony score and analysis
        """
        if len(temperatures) == 0:
            return {
                'harmony_score': 0.5,
                'gradient_smoothness': 0.5,
                'deviation_from_optimal': 0.0
            }
        
        # Calculate gradient smoothness
        gradients = np.gradient(temperatures)
        gradient_variance = np.var(gradients)
        smoothness = 1.0 / (1.0 + gradient_variance)  # Lower variance = smoother
        
        # Calculate deviation from optimal range
        in_range = np.logical_and(
            temperatures >= self.optimal_temperature_range[0],
            temperatures <= self.optimal_temperature_range[1]
        )
        range_compliance = np.mean(in_range)
        
        # Overall harmony score
        harmony_score = (smoothness * 0.6 + range_compliance * 0.4)
        
        return {
            'harmony_score': float(harmony_score),
            'gradient_smoothness': float(smoothness),
            'range_compliance': float(range_compliance),
            'mean_temperature': float(np.mean(temperatures)),
            'temperature_variance': float(np.var(temperatures))
        }
    
    def assess_light_quality(self, 
                            light_spectrum: np.ndarray,
                            color_temperature: float,
                            hour_of_day: int) -> Dict:
        """
        Assess light quality for circadian alignment
        
        Args:
            light_spectrum: Full spectrum analysis
            color_temperature: Color temperature in Kelvin
            hour_of_day: Current hour (0-23)
            
        Returns:
            Dict with quality score and circadian alignment
        """
        # Determine if daytime or nighttime
        is_daytime = 6 <= hour_of_day < 18
        
        # Check color temperature appropriateness
        if is_daytime:
            optimal_range = self.optimal_light_temp_day
        else:
            optimal_range = self.optimal_light_temp_night
        
        temp_in_range = optimal_range[0] <= color_temperature <= optimal_range[1]
        temp_score = 1.0 if temp_in_range else 0.5
        
        # Assess spectrum completeness (full spectrum is better)
        if len(light_spectrum) > 0:
            spectrum_completeness = np.mean(light_spectrum > 0.1)
        else:
            spectrum_completeness = 0.5
        
        # Circadian alignment (right light at right time)
        if is_daytime and color_temperature > 5000:
            circadian_alignment = 1.0
        elif not is_daytime and color_temperature < 4000:
            circadian_alignment = 1.0
        else:
            circadian_alignment = 0.6
        
        # Overall quality
        quality_score = (temp_score * 0.3 + spectrum_completeness * 0.3 + circadian_alignment * 0.4)
        
        return {
            'quality_score': float(quality_score),
            'color_temperature': color_temperature,
            'spectrum_completeness': float(spectrum_completeness),
            'circadian_alignment': float(circadian_alignment),
            'is_daytime': is_daytime
        }
    
    def assess_air_vitality(self,
                           o2_percent: float = 21.0,
                           co2_ppm: float = 400.0,
                           humidity: float = 50.0,
                           negative_ions: Optional[int] = None) -> Dict:
        """
        Assess air quality and vitality
        
        Args:
            o2_percent: Oxygen percentage
            co2_ppm: CO2 concentration in ppm
            humidity: Relative humidity percentage
            negative_ions: Negative ion count (optional)
            
        Returns:
            Dict with vitality score and metrics
        """
        # O2 score (normal is ~21%)
        o2_score = 1.0 if 20.5 <= o2_percent <= 21.5 else 0.7
        
        # CO2 score (lower is better, but not too low)
        co2_in_range = self.optimal_co2_range[0] <= co2_ppm <= self.optimal_co2_range[1]
        co2_score = 1.0 if co2_in_range else max(0.3, 1.0 - (abs(co2_ppm - 700) / 1000))
        
        # Humidity score
        humidity_in_range = self.optimal_humidity_range[0] <= humidity <= self.optimal_humidity_range[1]
        humidity_score = 1.0 if humidity_in_range else 0.6
        
        # Negative ions score (higher is better)
        if negative_ions is not None:
            # Optimal: 1000-5000 ions/cm³
            ion_score = min(1.0, negative_ions / 3000)
        else:
            ion_score = 0.5  # Unknown
        
        # Overall vitality
        vitality_score = (o2_score * 0.3 + co2_score * 0.3 + humidity_score * 0.2 + ion_score * 0.2)
        
        return {
            'vitality_score': float(vitality_score),
            'o2_score': o2_score,
            'co2_score': co2_score,
            'humidity_score': humidity_score,
            'ion_score': ion_score,
            'co2_ppm': co2_ppm,
            'humidity': humidity
        }
    
    def assess_sound_harmonics(self,
                              frequency_spectrum: np.ndarray,
                              frequencies: np.ndarray) -> Dict:
        """
        Assess presence of beneficial frequencies
        
        Args:
            frequency_spectrum: Power spectrum of sound
            frequencies: Frequency bins
            
        Returns:
            Dict with harmonic score and detected frequencies
        """
        if len(frequency_spectrum) == 0 or len(frequencies) == 0:
            return {
                'harmonic_score': 0.5,
                'beneficial_frequencies_present': [],
                'total_beneficial_power': 0.0
            }
        
        beneficial_present = []
        total_beneficial_power = 0.0
        
        # Check for Solfeggio frequencies
        for freq in self.beneficial_frequencies['solfeggio']:
            idx = np.argmin(np.abs(frequencies - freq))
            if frequency_spectrum[idx] > np.mean(frequency_spectrum) * 1.5:
                beneficial_present.append(f"Solfeggio {freq}Hz")
                total_beneficial_power += frequency_spectrum[idx]
        
        # Check for Schumann resonances
        for freq in self.beneficial_frequencies['schumann']:
            idx = np.argmin(np.abs(frequencies - freq))
            if frequency_spectrum[idx] > np.mean(frequency_spectrum) * 1.5:
                beneficial_present.append(f"Schumann {freq}Hz")
                total_beneficial_power += frequency_spectrum[idx]
        
        # Calculate harmonic score
        harmonic_score = min(1.0, len(beneficial_present) / 5.0)
        
        return {
            'harmonic_score': float(harmonic_score),
            'beneficial_frequencies_present': beneficial_present,
            'total_beneficial_power': float(total_beneficial_power),
            'frequency_diversity': len(beneficial_present)
        }
    
    def assess_spatial_flow(self,
                           flow_vectors: np.ndarray,
                           obstruction_map: Optional[np.ndarray] = None) -> Dict:
        """
        Assess Qi/prana flow through space
        
        Args:
            flow_vectors: Directional flow vectors
            obstruction_map: Map of obstructions (optional)
            
        Returns:
            Dict with flow score and analysis
        """
        if len(flow_vectors) == 0:
            return {
                'flow_score': 0.5,
                'flow_smoothness': 0.5,
                'obstruction_level': 0.0
            }
        
        # Calculate flow smoothness (consistent direction = better)
        flow_directions = flow_vectors / (np.linalg.norm(flow_vectors, axis=1, keepdims=True) + 1e-10)
        direction_variance = np.var(flow_directions, axis=0)
        smoothness = 1.0 / (1.0 + np.mean(direction_variance))
        
        # Calculate obstruction level
        if obstruction_map is not None:
            obstruction_level = np.mean(obstruction_map)
        else:
            obstruction_level = 0.0
        
        # Flow score (high smoothness, low obstruction = good)
        flow_score = smoothness * (1.0 - obstruction_level)
        
        return {
            'flow_score': float(flow_score),
            'flow_smoothness': float(smoothness),
            'obstruction_level': float(obstruction_level),
            'mean_flow_magnitude': float(np.mean(np.linalg.norm(flow_vectors, axis=1)))
        }
    
    def get_comprehensive_assessment(self,
                                    temperature_data: np.ndarray,
                                    light_spectrum: np.ndarray,
                                    color_temp: float,
                                    hour: int,
                                    o2: float = 21.0,
                                    co2: float = 400.0,
                                    humidity: float = 50.0,
                                    sound_spectrum: Optional[np.ndarray] = None,
                                    sound_freqs: Optional[np.ndarray] = None,
                                    flow_vectors: Optional[np.ndarray] = None) -> EnvironmentalState:
        """
        Get comprehensive environmental assessment
        
        Returns:
            Complete EnvironmentalState object
        """
        # Assess each metric
        temp_result = self.assess_temperature_harmony(temperature_data)
        light_result = self.assess_light_quality(light_spectrum, color_temp, hour)
        air_result = self.assess_air_vitality(o2, co2, humidity)
        
        if sound_spectrum is not None and sound_freqs is not None:
            sound_result = self.assess_sound_harmonics(sound_spectrum, sound_freqs)
        else:
            sound_result = {'harmonic_score': 0.5}
        
        if flow_vectors is not None:
            flow_result = self.assess_spatial_flow(flow_vectors)
        else:
            flow_result = {'flow_score': 0.5}
        
        # Calculate overall harmony
        overall_harmony = (
            temp_result['harmony_score'] * 0.25 +
            light_result['quality_score'] * 0.25 +
            air_result['vitality_score'] * 0.25 +
            sound_result['harmonic_score'] * 0.125 +
            flow_result['flow_score'] * 0.125
        )
        
        state = EnvironmentalState(
            temperature_harmony=temp_result['harmony_score'],
            light_quality=light_result['quality_score'],
            air_vitality=air_result['vitality_score'],
            sound_harmonics=sound_result['harmonic_score'],
            spatial_flow=flow_result['flow_score'],
            overall_harmony=overall_harmony,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return state

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("🌿 LUMINARK - Environmental Metrics Demo")
    print("="*70)
    
    env = EnvironmentalMetrics()
    
    # Simulate environmental data
    temperatures = np.random.randn(10) * 2 + 22.0
    light_spectrum = np.random.rand(7)  # 7-color spectrum
    sound_spectrum = np.random.rand(100)
    sound_freqs = np.linspace(0, 1000, 100)
    flow_vectors = np.random.randn(20, 2)
    
    # Get comprehensive assessment
    state = env.get_comprehensive_assessment(
        temperature_data=temperatures,
        light_spectrum=light_spectrum,
        color_temp=5500,
        hour=14,  # 2 PM
        o2=21.0,
        co2=450,
        humidity=50.0,
        sound_spectrum=sound_spectrum,
        sound_freqs=sound_freqs,
        flow_vectors=flow_vectors
    )
    
    print("\n🌍 Environmental Assessment:")
    print(f"  Temperature Harmony: {state.temperature_harmony:.2f}")
    print(f"  Light Quality: {state.light_quality:.2f}")
    print(f"  Air Vitality: {state.air_vitality:.2f}")
    print(f"  Sound Harmonics: {state.sound_harmonics:.2f}")
    print(f"  Spatial Flow: {state.spatial_flow:.2f}")
    print(f"  Overall Harmony: {state.overall_harmony:.2f}")
    
    print("\n✅ Environmental Metrics operational!")
