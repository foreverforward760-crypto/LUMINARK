"""
Mycelial Defense System - Bio-Inspired Active Defense for AI Systems

A complete defense system inspired by biological immune systems, fungal networks,
and octopus camouflage for protecting AI systems from attacks and misalignment.

Core Components:
- AlignmentDetector: Immune-system style "self vs. non-self" detection
- MycelialNetwork: Fungal containment and isolation system
- OctoCamouflage: Weaponized emptiness - hide by mimicking void
- MycelialDefenseSystem: Integrated orchestration with intelligent triggers

Author: Richard Leroy Stanfield Jr. / Meridian Axiom
Date: January 2026
License: MIT (after IP protection)
"""

__version__ = "0.1.0"
__author__ = "Richard Leroy Stanfield Jr."
__license__ = "MIT"

from .alignment import AlignmentDetector, AlignmentStatus, ComponentSignature
from .mycelial import MycelialNetwork, MycelialWall, MycelialPathway
from .octo import OctoCamouflage, CamouflagePattern, CamouflageProfile
from .defense import MycelialDefenseSystem, DefenseMode
from .sap import SAPCalculator, SPATVectors

__all__ = [
    "AlignmentDetector",
    "AlignmentStatus",
    "ComponentSignature",
    "MycelialNetwork",
    "MycelialWall",
    "MycelialPathway",
    "OctoCamouflage",
    "CamouflagePattern",
    "CamouflageProfile",
    "MycelialDefenseSystem",
    "DefenseMode",
    "SAPCalculator",
    "SPATVectors",
]
