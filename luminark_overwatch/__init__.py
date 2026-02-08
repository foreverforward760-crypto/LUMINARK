"""
LUMINARK OVERWATCH
==================
AI Regulatory System - Monitors systems and AI outputs for alignment.

Modules:
- analyzers: Response quality analyzers (Personal Assumption, Fake Empathy, etc.)
- core: Main OverwatchEngine (system monitoring, SAP diagnostics)
"""

# Analyzers are imported from submodule
from .analyzers import (
    analyze_personal_assumption,
    PersonalAssumptionResult,
    analyze_fake_empathy,
    FakeEmpathyResult,
)

__version__ = "1.1.0"
__all__ = [
    "analyze_personal_assumption",
    "PersonalAssumptionResult",
    "analyze_fake_empathy",
    "FakeEmpathyResult",
]
