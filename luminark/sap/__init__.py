"""
LUMINARK - SAP (Stanfield's Axiom of Perpetuity) Module
Experimental SAP stage and micro-stage prototype

Components:
- framework_81: Experimental stage 0-9 mapping with separate 0B boundary; 81 microstates not yet implemented
- environmental: Environmental harmony metrics
- Project-defined geometry labels and 3/6/9 digit-pattern calculation
"""

from .framework_81 import (
    SAP81Framework,
    SAPState81,
    Gate81,
    SAPArc,
    GeometricEncoding81
)
from .environmental import EnvironmentalMetrics, EnvironmentalState

__all__ = [
    'SAP81Framework',
    'SAPState81',
    'Gate81',
    'SAPArc',
    'GeometricEncoding81',
    'EnvironmentalMetrics',
    'EnvironmentalState'
]
