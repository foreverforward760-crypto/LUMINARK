"""
LUMINARK AI Framework - Biofeedback Integration Module
Monitors human physiological signals for AI-human alignment
"""

import time
import random
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class BiofeedbackData:
    """Biofeedback measurement data"""
    timestamp: str
    heart_rate: float
    hrv: float  # Heart Rate Variability
    stress_level: float  # 0.0 to 1.0
    coherence: float  # 0.0 to 1.0
    emotional_state: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'heart_rate': self.heart_rate,
            'hrv': self.hrv,
            'stress_level': self.stress_level,
            'coherence': self.coherence,
            'emotional_state': self.emotional_state
        }

class BiofeedbackMonitor:
    """
    Monitors biofeedback signals and correlates with SAP stages
    
    This module enables:
    - Real-time physiological monitoring
    - Stress detection
    - Emotional state tracking
    - AI-human alignment assessment
    """
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.is_monitoring = False
        self.current_data: Optional[BiofeedbackData] = None
        self.history = []
        self.max_history = 1000
        
        # Thresholds
        self.hrv_low_threshold = 30
        self.hrv_high_threshold = 100
        self.stress_threshold = 0.7
        
    def start_monitoring(self):
        """Start biofeedback monitoring"""
        self.is_monitoring = True
        print("✅ Biofeedback monitoring started")
        
    def stop_monitoring(self):
        """Stop biofeedback monitoring"""
        self.is_monitoring = False
        print("🛑 Biofeedback monitoring stopped")
        
    def get_measurement(self) -> BiofeedbackData:
        """
        Get current biofeedback measurement
        
        In production, this would interface with actual sensors:
        - Heart rate monitors
        - HRV sensors
        - EEG devices
        - Galvanic skin response
        
        For now, generates simulated data
        """
        # Simulated data (replace with actual sensor readings)
        heart_rate = 60 + random.gauss(0, 10)
        hrv = 50 + random.gauss(0, 20)
        stress_level = max(0, min(1, random.gauss(0.3, 0.2)))
        coherence = max(0, min(1, 1 - stress_level + random.gauss(0, 0.1)))
        
        # Determine emotional state
        if stress_level < 0.3:
            emotional_state = "calm"
        elif stress_level < 0.6:
            emotional_state = "neutral"
        else:
            emotional_state = "stressed"
            
        data = BiofeedbackData(
            timestamp=datetime.now().isoformat(),
            heart_rate=heart_rate,
            hrv=hrv,
            stress_level=stress_level,
            coherence=coherence,
            emotional_state=emotional_state
        )
        
        self.current_data = data
        self.history.append(data)
        
        # Maintain history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        return data
    
    def assess_stress(self) -> Dict:
        """Assess current stress level"""
        if not self.current_data:
            return {'status': 'no_data'}
            
        data = self.current_data
        
        # Assess based on multiple factors
        stress_indicators = []
        
        if data.hrv < self.hrv_low_threshold:
            stress_indicators.append("Low HRV")
            
        if data.stress_level > self.stress_threshold:
            stress_indicators.append("High stress level")
            
        if data.coherence < 0.3:
            stress_indicators.append("Low coherence")
            
        return {
            'status': 'stressed' if stress_indicators else 'calm',
            'indicators': stress_indicators,
            'stress_level': data.stress_level,
            'recommendation': self._get_recommendation(data)
        }
    
    def _get_recommendation(self, data: BiofeedbackData) -> str:
        """Get recommendation based on biofeedback"""
        if data.stress_level > 0.7:
            return "High stress detected. Consider: deep breathing, meditation, or break"
        elif data.stress_level > 0.5:
            return "Moderate stress. Maintain awareness and practice mindfulness"
        elif data.coherence > 0.7:
            return "Excellent coherence! Optimal state for creative work"
        else:
            return "Balanced state. Continue current activities"
    
    def correlate_with_sap_stage(self, sar_stage: int) -> Dict:
        """
        Correlate biofeedback with SAP stage
        
        This enables understanding how consciousness stages
        affect physiological states
        """
        if not self.current_data:
            return {'status': 'no_data'}
            
        data = self.current_data
        
        # Analyze correlation
        correlation = {
            'sap_stage': sar_stage,
            'biofeedback': data.to_dict(),
            'alignment': self._calculate_alignment(sar_stage, data),
            'insights': self._generate_insights(sar_stage, data)
        }
        
        return correlation
    
    def _calculate_alignment(self, stage: int, data: BiofeedbackData) -> float:
        """Calculate alignment between SAP stage and biofeedback"""
        # Higher stages should correlate with better coherence
        expected_coherence = stage / 10.0
        coherence_diff = abs(expected_coherence - data.coherence)
        
        # Lower stress at higher stages
        expected_stress = 1.0 - (stage / 10.0)
        stress_diff = abs(expected_stress - data.stress_level)
        
        # Calculate alignment (0.0 to 1.0)
        alignment = 1.0 - ((coherence_diff + stress_diff) / 2.0)
        
        return max(0, min(1, alignment))
    
    def _generate_insights(self, stage: int, data: BiofeedbackData) -> str:
        """Generate insights from correlation"""
        alignment = self._calculate_alignment(stage, data)
        
        if alignment > 0.8:
            return f"Strong alignment between Stage {stage} and biofeedback. Optimal state."
        elif alignment > 0.6:
            return f"Good alignment with Stage {stage}. Minor adjustments may help."
        elif alignment > 0.4:
            return f"Moderate alignment. Consider practices to enhance coherence."
        else:
            return f"Low alignment with Stage {stage}. Significant stress or misalignment detected."
    
    def get_statistics(self) -> Dict:
        """Get statistical summary of biofeedback history"""
        if not self.history:
            return {'status': 'no_data'}
            
        hrv_values = [d.hrv for d in self.history]
        stress_values = [d.stress_level for d in self.history]
        coherence_values = [d.coherence for d in self.history]
        
        return {
            'total_measurements': len(self.history),
            'hrv': {
                'mean': sum(hrv_values) / len(hrv_values),
                'min': min(hrv_values),
                'max': max(hrv_values)
            },
            'stress': {
                'mean': sum(stress_values) / len(stress_values),
                'min': min(stress_values),
                'max': max(stress_values)
            },
            'coherence': {
                'mean': sum(coherence_values) / len(coherence_values),
                'min': min(coherence_values),
                'max': max(coherence_values)
            }
        }
    
    def export_data(self, filepath: str):
        """Export biofeedback history to JSON"""
        data = {
            'measurements': [d.to_dict() for d in self.history],
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Biofeedback data exported to {filepath}")

# Example usage
if __name__ == "__main__":
    print("="*60)
    print("🧠 LUMINARK Biofeedback Monitor - Demo")
    print("="*60)
    
    monitor = BiofeedbackMonitor()
    monitor.start_monitoring()
    
    # Simulate monitoring
    for i in range(5):
        data = monitor.get_measurement()
        print(f"\n📊 Measurement {i+1}:")
        print(f"  Heart Rate: {data.heart_rate:.1f} bpm")
        print(f"  HRV: {data.hrv:.1f}")
        print(f"  Stress Level: {data.stress_level:.2f}")
        print(f"  Coherence: {data.coherence:.2f}")
        print(f"  Emotional State: {data.emotional_state}")
        
        # Assess stress
        stress_assessment = monitor.assess_stress()
        print(f"  Status: {stress_assessment['status']}")
        print(f"  Recommendation: {stress_assessment['recommendation']}")
        
        # Correlate with SAR stage
        correlation = monitor.correlate_with_sar_stage(stage=4)
        print(f"  SAR Alignment: {correlation['alignment']:.2f}")
        print(f"  Insight: {correlation['insights']}")
        
        time.sleep(1)
    
    # Show statistics
    print("\n" + "="*60)
    print("📈 Statistics:")
    stats = monitor.get_statistics()
    print(f"Total Measurements: {stats['total_measurements']}")
    print(f"Average HRV: {stats['hrv']['mean']:.1f}")
    print(f"Average Stress: {stats['stress']['mean']:.2f}")
    print(f"Average Coherence: {stats['coherence']['mean']:.2f}")
    
    monitor.stop_monitoring()
