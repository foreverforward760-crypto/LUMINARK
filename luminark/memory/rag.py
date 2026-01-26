"""
LUMINARK - RAG (Retrieval-Augmented Generation) Module
FAISS-based vector similarity search for memory retrieval

Requires: pip install faiss-cpu (or faiss-gpu for GPU support)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Install with: pip install faiss-cpu")


@dataclass
class Memory:
    """Represents a stored memory"""
    text: str
    embedding: np.ndarray
    timestamp: datetime
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class RAGMemoryBank:
    """
    Retrieval-Augmented Generation memory system
    
    Uses FAISS for efficient similarity search over embeddings
    """
    
    def __init__(self, embedding_dim: int = 256, index_type: str = 'flat'):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS required. Install with: pip install faiss-cpu")
        
        self.embedding_dim = embedding_dim
        self.memories: List[Memory] = []
        
        # Create FAISS index
        if index_type == 'flat':
            # Exact search (slower but accurate)
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == 'ivf':
            # Approximate search (faster for large datasets)
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.is_trained = False
    
    def add_memory(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ):
        """
        Add a memory to the bank
        
        Args:
            text: Text content
            embedding: Vector embedding (must be embedding_dim)
            metadata: Optional metadata dict
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding must be {self.embedding_dim}-dimensional")
        
        # Ensure embedding is 2D for FAISS
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Create memory object
        memory = Memory(
            text=text,
            embedding=embedding.flatten(),
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.memories.append(memory)
        
        # Add to FAISS index
        self.index.add(embedding.astype(np.float32))
    
    def add_batch(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata_list: Optional[List[Dict]] = None
    ):
        """
        Add multiple memories at once
        
        Args:
            texts: List of text contents
            embeddings: (N, embedding_dim) array
            metadata_list: Optional list of metadata dicts
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embeddings must be {self.embedding_dim}-dimensional")
        
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        for text, emb, meta in zip(texts, embeddings, metadata_list):
            memory = Memory(
                text=text,
                embedding=emb,
                timestamp=datetime.now(),
                metadata=meta
            )
            self.memories.append(memory)
        
        # Add all to FAISS index
        self.index.add(embeddings.astype(np.float32))
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        return_distances: bool = False
    ) -> List[Tuple[Memory, float]]:
        """
        Search for similar memories
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            return_distances: Whether to return distances
            
        Returns:
            List of (Memory, distance) tuples
        """
        if len(self.memories) == 0:
            return []
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search FAISS index
        k = min(k, len(self.memories))
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            k
        )
        
        # Retrieve memories
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.memories):
                memory = self.memories[idx]
                results.append((memory, float(dist)))
        
        return results
    
    def search_by_text(
        self,
        query_text: str,
        embedding_fn,
        k: int = 5
    ) -> List[Tuple[Memory, float]]:
        """
        Search using text query (requires embedding function)
        
        Args:
            query_text: Text query
            embedding_fn: Function that converts text to embedding
            k: Number of results
            
        Returns:
            List of (Memory, distance) tuples
        """
        query_embedding = embedding_fn(query_text)
        return self.search(query_embedding, k)
    
    def get_context(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        max_length: int = 500
    ) -> str:
        """
        Get context string from top-k memories
        
        Args:
            query_embedding: Query vector
            k: Number of memories to retrieve
            max_length: Maximum total character length
            
        Returns:
            Concatenated context string
        """
        results = self.search(query_embedding, k)
        
        context_parts = []
        total_length = 0
        
        for memory, _ in results:
            text = memory.text
            if total_length + len(text) > max_length:
                # Truncate to fit
                remaining = max_length - total_length
                text = text[:remaining]
            
            context_parts.append(text)
            total_length += len(text)
            
            if total_length >= max_length:
                break
        
        return " ".join(context_parts)
    
    def clear(self):
        """Clear all memories"""
        self.memories = []
        self.index.reset()
    
    def save(self, path: str):
        """Save index and memories to disk"""
        import pickle
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save memories
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self.memories, f)
    
    def load(self, path: str):
        """Load index and memories from disk"""
        import pickle
        
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")
        
        # Load memories
        with open(f"{path}.pkl", 'rb') as f:
            self.memories = pickle.load(f)
    
    def get_stats(self) -> Dict:
        """Get memory bank statistics"""
        return {
            'total_memories': len(self.memories),
            'embedding_dim': self.embedding_dim,
            'index_type': type(self.index).__name__,
            'oldest_memory': self.memories[0].timestamp.isoformat() if self.memories else None,
            'newest_memory': self.memories[-1].timestamp.isoformat() if self.memories else None
        }


# Example usage
if __name__ == "__main__":
    if not FAISS_AVAILABLE:
        print("❌ FAISS not available. Install with: pip install faiss-cpu")
        exit(1)
    
    print("="*70)
    print("🧠 LUMINARK - RAG Memory Bank Demo")
    print("="*70)
    
    # Create memory bank
    embedding_dim = 128
    bank = RAGMemoryBank(embedding_dim=embedding_dim)
    
    # Add some memories (with random embeddings for demo)
    memories = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Neural networks are inspired by biological neurons",
        "Deep learning uses multiple layers of neural networks"
    ]
    
    print(f"\n📝 Adding {len(memories)} memories...")
    for text in memories:
        # Generate random embedding (in real use, use actual model)
        embedding = np.random.randn(embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        bank.add_memory(text, embedding, metadata={'source': 'demo'})
    
    print(f"✅ Added {len(bank.memories)} memories")
    
    # Search for similar memories
    print(f"\n🔍 Searching for similar memories...")
    query_embedding = np.random.randn(embedding_dim)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    results = bank.search(query_embedding, k=3)
    
    print(f"Top 3 results:")
    for i, (memory, distance) in enumerate(results, 1):
        print(f"  {i}. '{memory.text[:50]}...' (distance: {distance:.3f})")
    
    # Get context
    print(f"\n📚 Getting context (max 200 chars)...")
    context = bank.get_context(query_embedding, k=2, max_length=200)
    print(f"Context: '{context}'")
    
    # Stats
    print(f"\n📊 Memory Bank Statistics:")
    stats = bank.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ RAG Memory Bank operational!")
