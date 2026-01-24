"""
Validation and Quality Assurance Module
Integrates DeepAgent-inspired testing and perspective modes
"""
from .qa_tester import AutomatedQATester
from .perspective_modes import PerspectiveModulator, AdversarialProber

__all__ = ['AutomatedQATester', 'PerspectiveModulator', 'AdversarialProber']
