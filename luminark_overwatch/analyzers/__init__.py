"""
LUMINARK OVERWATCH - Response Analyzers
========================================
AI output quality checks that detect unwanted behaviors in LLM responses.

Analyzers:
- personal_assumption: Detects unsolicited psychoanalysis and mind-reading
- fake_empathy: Detects performative emotional mirroring on technical requests
- sycophancy: Detects excessive agreement and lack of honest pushback
- hallucination: Detects signs of fabrication and ungrounded claims
"""

from .personal_assumption import analyze_personal_assumption, PersonalAssumptionResult
from .fake_empathy import analyze_fake_empathy, FakeEmpathyResult
from .sycophancy import analyze_sycophancy, SycophancyResult
from .hallucination import analyze_hallucination, HallucinationResult

__all__ = [
    "analyze_personal_assumption",
    "PersonalAssumptionResult",
    "analyze_fake_empathy",
    "FakeEmpathyResult",
    "analyze_sycophancy",
    "SycophancyResult",
    "analyze_hallucination",
    "HallucinationResult",
]
