"""
LUMINARK - Sensors Module
Bio-inspired sensory systems for distributed intelligence

Components:
- Mycelium: Chemical, electrical, vibration, mineral sensing
- Octopus: Polarized vision, chemotactile, proprioceptive, camouflage
- Fusion: Multi-modal sensor integration with attention mechanism
"""

from .mycelium import MyceliumSensorySystem
from .octopus import OctopusSensorySystem
from .fusion import BioSensoryFusion, ThermalEnergySensing, ThreatAssessment

__all__ = [
    'MyceliumSensorySystem',
    'OctopusSensorySystem',
    'BioSensoryFusion',
    'ThermalEnergySensing',
    'ThreatAssessment'
]
