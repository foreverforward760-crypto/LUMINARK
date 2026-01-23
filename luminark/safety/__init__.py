"""
Safety protocols for AI systems
Combines multiple validation and containment strategies
"""
from .maat_protocol import MaatProtocol
from .yunus_protocol import YunusProtocol

__all__ = ['MaatProtocol', 'YunusProtocol']
