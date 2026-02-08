"""
LUMINARK OVERWATCH - Response Analyzers
========================================
AI output quality checks that detect unwanted behaviors in LLM responses.

Analyzers:
- personal_assumption: Detects unsolicited psychoanalysis and mind-reading
- fake_empathy: Detects performative emotional mirroring on technical requests
- topic_drift: Detects deviation from user's actual request
"""

from .personal_assumption import analyze_personal_assumption, PersonalAssumptionResult
from .fake_empathy import analyze_fake_empathy, FakeEmpathyResult

__all__ = [
    "analyze_personal_assumption",
    "PersonalAssumptionResult",
    "analyze_fake_empathy",
    "FakeEmpathyResult",
]
