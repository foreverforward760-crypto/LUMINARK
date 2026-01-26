"""
LUMINARK - Memory Module
Memory systems for retrieval and context

Components:
- rag: FAISS-based retrieval-augmented generation
"""

from .rag import RAGMemoryBank, Memory, FAISS_AVAILABLE

__all__ = [
    'RAGMemoryBank',
    'Memory',
    'FAISS_AVAILABLE'
]
