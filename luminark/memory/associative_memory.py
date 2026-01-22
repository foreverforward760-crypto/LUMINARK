"""
Associative Memory System with Experience Replay
Stores training experiences with semantic associations for better learning
"""
import numpy as np
import networkx as nx
from collections import deque
from typing import Dict, List, Any, Optional
import hashlib


class AssociativeMemory:
    """
    Memory system that stores experiences with associations
    Enables experience replay with semantic similarity
    """

    def __init__(self, capacity=10000, embedding_dim=64):
        self.capacity = capacity
        self.embedding_dim = embedding_dim

        # Memory storage
        self.memories = deque(maxlen=capacity)
        self.embeddings = np.zeros((capacity, embedding_dim))
        self.next_index = 0

        # Association graph
        self.association_graph = nx.Graph()

        # Tags and metadata
        self.tag_index = {}  # tag -> list of memory indices
        self.metadata_index = {}  # metadata key -> list of indices

    def store(self, experience: Dict, tags: List[str] = None, metadata: Dict = None):
        """Store an experience with associations"""
        exp_id = hashlib.md5(str(experience).encode()).hexdigest()[:8]
        embedding = self._create_embedding(experience)

        memory_idx = self.next_index % self.capacity
        self.memories.append({
            'id': exp_id,
            'experience': experience,
            'tags': tags or [],
            'metadata': metadata or {},
            'embedding': embedding,
            'index': memory_idx
        })
        self.embeddings[memory_idx] = embedding

        self.association_graph.add_node(exp_id, index=memory_idx)
        self._create_associations(exp_id, embedding, memory_idx)

        if tags:
            for tag in tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = []
                self.tag_index[tag].append(memory_idx)

        self.next_index += 1
        return exp_id

    def recall(self, query: Dict = None, tags: List[str] = None,
              num_memories: int = 10) -> List[Dict]:
        """Recall memories based on query or tags"""
        if len(self.memories) == 0:
            return []

        candidate_indices = set()
        if tags:
            for tag in tags:
                if tag in self.tag_index:
                    candidate_indices.update(self.tag_index[tag])

        if not candidate_indices:
            candidate_indices = set(range(min(self.next_index, self.capacity)))

        top_indices = list(candidate_indices)[:num_memories]
        recalled = []
        for idx in top_indices:
            if idx < len(self.memories):
                for mem in self.memories:
                    if mem['index'] == idx:
                        recalled.append(mem)
                        break

        return recalled[:num_memories]

    def replay_batch(self, batch_size: int = 32) -> List[Dict]:
        """Get a batch of experiences for replay"""
        if len(self.memories) == 0:
            return []
        indices = np.random.choice(len(self.memories),
                                  size=min(batch_size, len(self.memories)),
                                  replace=False)
        return [self.memories[i] for i in indices]

    def _create_embedding(self, data: Dict) -> np.ndarray:
        """Create embedding from experience data"""
        data_str = str(sorted(data.items()))
        hash_val = hashlib.md5(data_str.encode()).digest()
        embedding = np.frombuffer(hash_val, dtype=np.uint8)
        embedding = np.tile(embedding, (self.embedding_dim // len(embedding)) + 1)
        embedding = embedding[:self.embedding_dim].astype(np.float32)
        return embedding / 255.0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity"""
        norm_a = np.linalg.norm(a) + 1e-10
        norm_b = np.linalg.norm(b) + 1e-10
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _create_associations(self, exp_id: str, embedding: np.ndarray, memory_idx: int):
        """Create associations with similar memories"""
        similarities = []
        for mem in self.memories:
            if mem['id'] != exp_id:
                sim = self._cosine_similarity(embedding, mem['embedding'])
                if sim > 0.7:
                    similarities.append((mem['id'], sim))

        for other_id, sim in similarities[:5]:
            self.association_graph.add_edge(exp_id, other_id, weight=sim)

    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        return {
            'total_memories': len(self.memories),
            'capacity': self.capacity,
            'fill_percentage': len(self.memories) / self.capacity * 100,
            'num_tags': len(self.tag_index),
            'num_associations': self.association_graph.number_of_edges()
        }
