"""Memory and experience systems"""
from luminark.memory.associative_memory import AssociativeMemory

try:
    from luminark.memory.faiss_memory import FAISSMemory
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

__all__ = ['AssociativeMemory', 'FAISSMemory', 'FAISS_AVAILABLE']
