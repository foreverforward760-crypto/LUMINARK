"""
LUMINARK - SAP (Stanfield's Axiom of Perpetuity) Module

Experimental stage and micro-stage prototype. The numbered 0-9 mapping and
separate 0B boundary are available without loading optional scientific packages.
The stated 81 discrete microstates are not yet implemented or validated.
"""

from .framework_81 import (
    SAP81Framework,
    SAPState81,
    Gate81,
    SAPArc,
    GeometricEncoding81,
)

__all__ = [
    "SAP81Framework",
    "SAPState81",
    "Gate81",
    "SAPArc",
    "GeometricEncoding81",
    "EnvironmentalMetrics",
    "EnvironmentalState",
]


def __getattr__(name):
    if name in {"EnvironmentalMetrics", "EnvironmentalState"}:
        from .environmental import EnvironmentalMetrics, EnvironmentalState
        return {
            "EnvironmentalMetrics": EnvironmentalMetrics,
            "EnvironmentalState": EnvironmentalState,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
