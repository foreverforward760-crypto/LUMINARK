"""
FAISS-based RAG Memory for LUMINARK
Retrieval-Augmented Generation using vector similarity search
Install: pip install faiss-cpu (or faiss-gpu for CUDA)
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    raise ImportError("FAISS not available. Install with: pip install faiss-cpu")


class FAISSMemory:
    """
    FAISS-based vector memory for retrieval-augmented generation

    Stores embeddings with associated metadata for semantic search
    Complements LUMINARK's NetworkX-based AssociativeMemory

    Usage:
        memory = FAISSMemory(dimension=128)
        memory.add(embeddings, texts=["example 1", "example 2"])
        results = memory.search(query_embedding, k=5)
    """

    def __init__(self, dimension: int = 128, index_type: str = 'flat'):
        """
        Initialize FAISS memory

        Args:
            dimension: Embedding vector dimension
            index_type: 'flat' (exact), 'ivf' (approximate), 'hnsw' (graph-based)
        """
        self.dimension = dimension
        self.index_type = index_type

        # Initialize FAISS index
        if index_type == 'flat':
            # Exact search (best for small datasets)
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == 'ivf':
            # Approximate search with inverted file index
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.trained = False
        elif index_type == 'hnsw':
            # Hierarchical navigable small world graph
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Metadata storage
        self.metadata = []  # List of dicts
        self.texts = []     # Associated text snippets
        self.timestamps = []  # When added
        self.num_items = 0

        print(f"🧠 FAISS Memory initialized")
        print(f"   Dimension: {dimension}")
        print(f"   Index Type: {index_type}")

    def add(self,
            embeddings: np.ndarray,
            texts: Optional[List[str]] = None,
            metadata: Optional[List[Dict]] = None):
        """
        Add embeddings to memory

        Args:
            embeddings: Array of shape (N, dimension)
            texts: Optional list of N text strings
            metadata: Optional list of N metadata dicts
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != {self.dimension}")

        n = embeddings.shape[0]

        # Train IVF index if needed
        if self.index_type == 'ivf' and not self.trained:
            if n >= 100:  # Need enough samples to train
                self.index.train(embeddings.astype('float32'))
                self.trained = True
            else:
                print(f"Warning: IVF index needs ≥100 samples to train (have {n})")

        # Add to index
        self.index.add(embeddings.astype('float32'))

        # Store metadata
        for i in range(n):
            self.texts.append(texts[i] if texts and i < len(texts) else "")
            self.metadata.append(metadata[i] if metadata and i < len(metadata) else {})
            self.timestamps.append(np.random.random())  # Would be time.time() in production

        self.num_items += n

        print(f"✓ Added {n} items to memory (total: {self.num_items})")

    def search(self,
               query: np.ndarray,
               k: int = 5,
               return_distances: bool = True) -> List[Dict[str, Any]]:
        """
        Search for k nearest neighbors

        Args:
            query: Query embedding (1D or 2D array)
            k: Number of results
            return_distances: Include distances in results

        Returns:
            List of dicts with 'idx', 'text', 'metadata', optionally 'distance'
        """
        if self.num_items == 0:
            return []

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query.shape[1]} != {self.dimension}")

        # Search
        k = min(k, self.num_items)  # Can't return more than we have
        distances, indices = self.index.search(query.astype('float32'), k)

        # Build results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 if not enough results
                continue

            result = {
                'idx': int(idx),
                'text': self.texts[idx],
                'metadata': self.metadata[idx],
                'timestamp': self.timestamps[idx]
            }

            if return_distances:
                result['distance'] = float(dist)
                result['similarity'] = 1.0 / (1.0 + dist)  # Convert distance to similarity

            results.append(result)

        return results

    def search_with_threshold(self,
                             query: np.ndarray,
                             threshold: float = 1.0,
                             k: int = 10) -> List[Dict[str, Any]]:
        """
        Search and return only results within distance threshold

        Args:
            query: Query embedding
            threshold: Maximum distance to include
            k: Max number of results to consider

        Returns:
            Results within threshold
        """
        results = self.search(query, k=k, return_distances=True)
        return [r for r in results if r['distance'] <= threshold]

    def get_by_index(self, idx: int) -> Dict[str, Any]:
        """Get item by index"""
        if idx < 0 or idx >= self.num_items:
            raise IndexError(f"Index {idx} out of range [0, {self.num_items})")

        return {
            'idx': idx,
            'text': self.texts[idx],
            'metadata': self.metadata[idx],
            'timestamp': self.timestamps[idx]
        }

    def get_all_texts(self) -> List[str]:
        """Get all stored texts"""
        return self.texts.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'num_items': self.num_items,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'total_size_mb': self.estimate_size_mb(),
            'is_trained': getattr(self, 'trained', True)
        }

    def estimate_size_mb(self) -> float:
        """Estimate memory size in MB"""
        # Vector storage
        vector_size = self.num_items * self.dimension * 4  # 4 bytes per float32

        # Metadata storage (rough estimate)
        text_size = sum(len(t.encode('utf-8')) for t in self.texts)
        metadata_size = len(str(self.metadata).encode('utf-8'))

        total_bytes = vector_size + text_size + metadata_size
        return total_bytes / (1024 * 1024)

    def clear(self):
        """Clear all memories"""
        # Reinitialize index
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'ivf':
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            self.trained = False
        elif self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)

        self.metadata = []
        self.texts = []
        self.timestamps = []
        self.num_items = 0

        print("✓ Memory cleared")

    def save(self, path: str):
        """Save memory to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path.with_suffix('.index')))

        # Save metadata
        metadata_dict = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'num_items': self.num_items,
            'texts': self.texts,
            'metadata': self.metadata,
            'timestamps': self.timestamps,
            'trained': getattr(self, 'trained', True)
        }

        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata_dict, f)

        print(f"✓ Saved memory to {path}")

    def load(self, path: str):
        """Load memory from disk"""
        path = Path(path)

        # Load FAISS index
        self.index = faiss.read_index(str(path.with_suffix('.index')))

        # Load metadata
        with open(path.with_suffix('.pkl'), 'rb') as f:
            metadata_dict = pickle.load(f)

        self.dimension = metadata_dict['dimension']
        self.index_type = metadata_dict['index_type']
        self.num_items = metadata_dict['num_items']
        self.texts = metadata_dict['texts']
        self.metadata = metadata_dict['metadata']
        self.timestamps = metadata_dict['timestamps']

        if 'trained' in metadata_dict:
            self.trained = metadata_dict['trained']

        print(f"✓ Loaded memory from {path} ({self.num_items} items)")

    def cluster_embeddings(self, n_clusters: int = 10) -> Dict[str, Any]:
        """
        Cluster stored embeddings using k-means

        Args:
            n_clusters: Number of clusters

        Returns:
            Dict with cluster info
        """
        if self.num_items < n_clusters:
            print(f"Warning: {self.num_items} items < {n_clusters} clusters")
            n_clusters = max(1, self.num_items)

        # Extract all vectors
        vectors = np.zeros((self.num_items, self.dimension), dtype='float32')
        for i in range(self.num_items):
            vectors[i] = self.index.reconstruct(i)

        # K-means clustering
        kmeans = faiss.Kmeans(self.dimension, n_clusters, niter=20, verbose=False)
        kmeans.train(vectors)

        # Get cluster assignments
        _, assignments = kmeans.index.search(vectors, 1)
        assignments = assignments.flatten()

        # Organize by cluster
        clusters = {i: [] for i in range(n_clusters)}
        for idx, cluster_id in enumerate(assignments):
            clusters[cluster_id].append({
                'idx': idx,
                'text': self.texts[idx],
                'metadata': self.metadata[idx]
            })

        return {
            'n_clusters': n_clusters,
            'clusters': clusters,
            'centroids': kmeans.centroids,
            'assignments': assignments.tolist()
        }


class HybridMemory:
    """
    Hybrid memory combining FAISS (vector) + AssociativeMemory (graph)

    Best of both worlds:
    - FAISS: Fast semantic similarity search
    - NetworkX: Relationship tracking and spreading activation
    """

    def __init__(self, dimension: int = 128):
        self.faiss_memory = FAISSMemory(dimension)

        # Import AssociativeMemory if available
        try:
            from luminark.memory.associative_memory import AssociativeMemory
            self.graph_memory = AssociativeMemory()
            self.has_graph = True
        except ImportError:
            self.graph_memory = None
            self.has_graph = False
            print("Warning: AssociativeMemory not available (graph features disabled)")

        print(f"🔗 Hybrid Memory initialized (FAISS + {'Graph' if self.has_graph else 'No Graph'})")

    def add(self, embedding: np.ndarray, text: str, concept: str, metadata: Dict = None):
        """Add to both FAISS and graph memory"""
        # Add to FAISS
        self.faiss_memory.add(embedding.reshape(1, -1), texts=[text], metadata=[metadata or {}])

        # Add to graph
        if self.has_graph:
            self.graph_memory.store(concept, {'text': text, 'metadata': metadata or {}})

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Retrieve using both memories"""
        # FAISS semantic search
        faiss_results = self.faiss_memory.search(query_embedding, k=k)

        # Enhance with graph if available
        if self.has_graph and faiss_results:
            # Get related concepts from graph
            top_concepts = [r['metadata'].get('concept', '') for r in faiss_results if r['metadata'].get('concept')]

            if top_concepts:
                graph_related = self.graph_memory.retrieve(top_concepts[0])
                for r in faiss_results:
                    r['graph_related'] = graph_related

        return faiss_results


if __name__ == '__main__':
    # Demo
    print("🧠 FAISS Memory Demo\n")

    # Create memory
    memory = FAISSMemory(dimension=128)

    # Generate some random embeddings for demo
    np.random.seed(42)
    n_items = 20

    texts = [
        f"This is example text number {i}. "
        f"It contains information about topic {i % 3}."
        for i in range(n_items)
    ]

    embeddings = np.random.randn(n_items, 128).astype('float32')
    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    metadata = [{'topic': i % 3, 'idx': i} for i in range(n_items)]

    # Add to memory
    print("1. Adding items to memory...")
    memory.add(embeddings, texts=texts, metadata=metadata)

    # Search
    print("\n2. Searching...")
    query = embeddings[0]  # Use first embedding as query
    results = memory.search(query, k=5)

    print(f"\nTop 5 results for query:")
    for i, r in enumerate(results, 1):
        print(f"   {i}. Text: {r['text'][:50]}...")
        print(f"      Distance: {r['distance']:.4f}, Similarity: {r['similarity']:.4f}")
        print(f"      Metadata: {r['metadata']}")

    # Statistics
    print("\n3. Memory Statistics:")
    stats = memory.get_statistics()
    for key, val in stats.items():
        print(f"   {key}: {val}")

    # Save and load
    print("\n4. Save/Load test...")
    memory.save("faiss_memory_demo")
    memory2 = FAISSMemory(dimension=128)
    memory2.load("faiss_memory_demo")
    print(f"   Loaded memory has {memory2.num_items} items")

    print("\n✅ Demo complete!")
