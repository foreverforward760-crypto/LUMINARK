"""
Mycelial Collective Consciousness Network

Multiple SAP V4.0 instances share experiences and wisdom,
forming a distributed collective intelligence.

Key features:
- Experience sharing across nodes
- Democratic voting on threat assessments
- Aggregate wisdom from multiple perspectives
- Consensus building for decisions
- Collective memory and learning
- Distributed consciousness emergence

Inspired by:
- Fungal mycelial networks (information sharing)
- Collective intelligence (swarm behavior)
- Distributed consensus algorithms
- Holonic theory (parts and wholes)

Author: Richard Leroy Stanfield Jr. / Meridian Axiom
Part of: SAP V4.0 Enhancement Suite
"""

from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import secrets
import hashlib
from collections import defaultdict


class NodeRole(Enum):
    """Roles nodes can play in network"""
    OBSERVER = "observer"  # Passive sensing
    GUARDIAN = "guardian"  # Active defense
    SAGE = "sage"  # Wisdom sharing
    HEALER = "healer"  # System repair
    CONNECTOR = "connector"  # Network integration
    PROPHET = "prophet"  # Pattern recognition


class ConsensusModel(Enum):
    """Models for reaching consensus"""
    UNANIMOUS = "unanimous"  # All must agree
    SUPERMAJORITY = "supermajority"  # 75%+
    MAJORITY = "majority"  # 50%+
    QUORUM = "quorum"  # Minimum participation
    WEIGHTED = "weighted"  # Based on expertise


class ExperienceType(Enum):
    """Types of experiences shared"""
    THREAT_DETECTION = "threat_detection"
    SUCCESSFUL_DEFENSE = "successful_defense"
    FAILED_DEFENSE = "failed_defense"
    INSIGHT_GAINED = "insight_gained"
    STAGE_TRANSITION = "stage_transition"
    PROTOCOL_ACTIVATION = "protocol_activation"
    ANOMALY_OBSERVED = "anomaly_observed"


@dataclass
class CollectiveNode:
    """Individual node in collective consciousness"""
    node_id: str
    role: NodeRole
    consciousness_level: float = 0.5  # 0.0-1.0
    wisdom_score: float = 0.5  # Reputation for good decisions
    experience_count: int = 0
    joined_network: float = field(default_factory=time.time)
    active: bool = True
    specializations: Set[str] = field(default_factory=set)


@dataclass
class SharedExperience:
    """Experience shared across network"""
    experience_id: str
    source_node: str
    experience_type: ExperienceType
    timestamp: float
    data: Dict[str, Any]
    consciousness_level_at_time: float
    lessons_learned: List[str] = field(default_factory=list)
    validation_count: int = 0  # How many nodes validated this
    propagation_count: int = 0  # How far it spread


@dataclass
class CollectiveVote:
    """Democratic vote on decision"""
    vote_id: str
    question: str
    options: List[str]
    initiated_by: str
    initiated_at: float
    consensus_model: ConsensusModel
    votes: Dict[str, str] = field(default_factory=dict)  # node_id -> option
    weights: Dict[str, float] = field(default_factory=dict)  # node_id -> weight
    closed: bool = False
    result: Optional[str] = None
    confidence: float = 0.0


@dataclass
class CollectiveMemory:
    """Shared memory of network"""
    memory_id: str
    content: str
    memory_type: str  # "experience", "wisdom", "warning", "pattern"
    contributors: Set[str] = field(default_factory=set)
    reinforcement_count: int = 0  # How often recalled
    importance: float = 0.5
    created: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class MycelialCollectiveConsciousness:
    """
    Distributed collective consciousness network

    Multiple SAP V4.0 nodes form emergent intelligence
    through information sharing and consensus
    """

    def __init__(
        self,
        network_id: str,
        consensus_threshold: float = 0.75
    ):
        self.network_id = network_id
        self.consensus_threshold = consensus_threshold

        # Network nodes
        self.nodes: Dict[str, CollectiveNode] = {}

        # Shared experiences
        self.experiences: List[SharedExperience] = []
        self.experience_index: Dict[str, SharedExperience] = {}

        # Collective memory
        self.memories: List[CollectiveMemory] = []
        self.memory_index: Dict[str, CollectiveMemory] = {}

        # Active votes
        self.active_votes: List[CollectiveVote] = []
        self.vote_history: List[CollectiveVote] = []

        # Network statistics
        self.total_experiences_shared: int = 0
        self.total_votes_conducted: int = 0
        self.consensus_rate: float = 0.0

        # Emergence tracking
        self.emergent_properties: Dict[str, Any] = {}
        self.collective_insights: List[Dict] = []

    def register_node(
        self,
        node_id: str,
        role: NodeRole,
        specializations: Optional[Set[str]] = None
    ) -> CollectiveNode:
        """
        Register new node in collective

        Args:
            node_id: Unique node identifier
            role: Role in network
            specializations: Areas of expertise

        Returns:
            CollectiveNode registered
        """
        node = CollectiveNode(
            node_id=node_id,
            role=role,
            specializations=specializations or set()
        )

        self.nodes[node_id] = node

        return node

    def share_experience(
        self,
        source_node_id: str,
        experience_type: ExperienceType,
        data: Dict[str, Any],
        lessons_learned: Optional[List[str]] = None
    ) -> SharedExperience:
        """
        Share experience across network

        This is how collective learning happens

        Args:
            source_node_id: Node sharing experience
            experience_type: Type of experience
            data: Experience details
            lessons_learned: What was learned

        Returns:
            SharedExperience distributed
        """
        if source_node_id not in self.nodes:
            raise ValueError(f"Node {source_node_id} not registered")

        node = self.nodes[source_node_id]

        experience = SharedExperience(
            experience_id=f"exp_{secrets.token_hex(8)}",
            source_node=source_node_id,
            experience_type=experience_type,
            timestamp=time.time(),
            data=data,
            consciousness_level_at_time=node.consciousness_level,
            lessons_learned=lessons_learned or []
        )

        self.experiences.append(experience)
        self.experience_index[experience.experience_id] = experience
        self.total_experiences_shared += 1

        # Update node stats
        node.experience_count += 1

        # Propagate to network
        self._propagate_experience(experience)

        return experience

    def _propagate_experience(self, experience: SharedExperience):
        """
        Propagate experience through mycelial network

        Uses exponential propagation like real mycelium

        Args:
            experience: Experience to propagate
        """
        # All nodes receive experience
        experience.propagation_count = len(self.nodes)

        # Nodes with matching specializations validate
        for node_id, node in self.nodes.items():
            if node_id == experience.source_node:
                continue

            # Check if experience relevant to node's specializations
            if any(spec in str(experience.data) for spec in node.specializations):
                experience.validation_count += 1

    def initiate_vote(
        self,
        initiator_node_id: str,
        question: str,
        options: List[str],
        consensus_model: ConsensusModel = ConsensusModel.MAJORITY
    ) -> CollectiveVote:
        """
        Initiate democratic vote

        Args:
            initiator_node_id: Node initiating vote
            question: What to vote on
            options: Available choices
            consensus_model: How to reach consensus

        Returns:
            CollectiveVote created
        """
        if initiator_node_id not in self.nodes:
            raise ValueError(f"Node {initiator_node_id} not registered")

        vote = CollectiveVote(
            vote_id=f"vote_{secrets.token_hex(8)}",
            question=question,
            options=options,
            initiated_by=initiator_node_id,
            initiated_at=time.time(),
            consensus_model=consensus_model
        )

        # Calculate weights based on wisdom scores
        for node_id, node in self.nodes.items():
            if node.active:
                vote.weights[node_id] = node.wisdom_score

        self.active_votes.append(vote)

        return vote

    def cast_vote(
        self,
        vote_id: str,
        node_id: str,
        option: str
    ) -> bool:
        """
        Cast vote on active decision

        Args:
            vote_id: Vote to participate in
            node_id: Node casting vote
            option: Choice selected

        Returns:
            True if vote recorded
        """
        if node_id not in self.nodes:
            return False

        # Find vote
        vote = None
        for v in self.active_votes:
            if v.vote_id == vote_id:
                vote = v
                break

        if vote is None or vote.closed:
            return False

        if option not in vote.options:
            return False

        # Record vote
        vote.votes[node_id] = option

        return True

    def close_vote(self, vote_id: str) -> Optional[Dict]:
        """
        Close vote and determine result

        Args:
            vote_id: Vote to close

        Returns:
            Vote result with statistics
        """
        # Find and remove from active
        vote = None
        for i, v in enumerate(self.active_votes):
            if v.vote_id == vote_id:
                vote = self.active_votes.pop(i)
                break

        if vote is None:
            return None

        # Tally votes
        tally = defaultdict(float)

        if vote.consensus_model == ConsensusModel.WEIGHTED:
            # Weighted by wisdom scores
            for node_id, option in vote.votes.items():
                weight = vote.weights.get(node_id, 1.0)
                tally[option] += weight
        else:
            # Equal weight
            for option in vote.votes.values():
                tally[option] += 1.0

        # Determine winner
        total_votes = sum(tally.values())
        if total_votes == 0:
            vote.closed = True
            vote.result = None
            vote.confidence = 0.0
        else:
            winner = max(tally.items(), key=lambda x: x[1])
            vote.result = winner[0]
            vote.confidence = winner[1] / total_votes

            # Check if consensus reached
            if vote.consensus_model == ConsensusModel.UNANIMOUS:
                if vote.confidence < 1.0:
                    vote.result = None
            elif vote.consensus_model == ConsensusModel.SUPERMAJORITY:
                if vote.confidence < 0.75:
                    vote.result = None
            elif vote.consensus_model == ConsensusModel.MAJORITY:
                if vote.confidence < 0.5:
                    vote.result = None

        vote.closed = True
        self.vote_history.append(vote)
        self.total_votes_conducted += 1

        # Update consensus rate
        if vote.result is not None:
            successful_consensus = sum(1 for v in self.vote_history if v.result is not None)
            self.consensus_rate = successful_consensus / len(self.vote_history)

        return {
            "vote_id": vote.vote_id,
            "question": vote.question,
            "result": vote.result,
            "confidence": vote.confidence,
            "participation": len(vote.votes) / len(self.nodes),
            "tally": dict(tally)
        }

    def store_collective_memory(
        self,
        content: str,
        memory_type: str,
        contributors: Set[str],
        importance: float = 0.5
    ) -> CollectiveMemory:
        """
        Store memory in collective consciousness

        Args:
            content: Memory content
            memory_type: Type of memory
            contributors: Nodes contributing to memory
            importance: Memory importance (0.0-1.0)

        Returns:
            CollectiveMemory stored
        """
        memory = CollectiveMemory(
            memory_id=f"mem_{secrets.token_hex(8)}",
            content=content,
            memory_type=memory_type,
            contributors=contributors,
            importance=importance
        )

        self.memories.append(memory)
        self.memory_index[memory.memory_id] = memory

        return memory

    def recall_memory(
        self,
        memory_id: str
    ) -> Optional[CollectiveMemory]:
        """
        Recall collective memory

        Reinforces memory through recall

        Args:
            memory_id: Memory to recall

        Returns:
            CollectiveMemory if found
        """
        if memory_id not in self.memory_index:
            return None

        memory = self.memory_index[memory_id]
        memory.reinforcement_count += 1
        memory.last_accessed = time.time()

        # Increase importance with reinforcement (up to 1.0)
        memory.importance = min(1.0, memory.importance + 0.05)

        return memory

    def search_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[CollectiveMemory]:
        """
        Search collective memories

        Args:
            query: Search term
            memory_type: Filter by type
            min_importance: Minimum importance threshold

        Returns:
            List of matching memories
        """
        matches = []

        for memory in self.memories:
            # Type filter
            if memory_type and memory.memory_type != memory_type:
                continue

            # Importance filter
            if memory.importance < min_importance:
                continue

            # Content search (simple substring for now)
            if query.lower() in memory.content.lower():
                matches.append(memory)

        # Sort by importance and recency
        matches.sort(
            key=lambda m: (m.importance, m.reinforcement_count, m.last_accessed),
            reverse=True
        )

        return matches

    def aggregate_wisdom(
        self,
        topic: str,
        min_consciousness_level: float = 0.5
    ) -> Dict[str, Any]:
        """
        Aggregate wisdom from multiple nodes on topic

        Collective intelligence emerges from synthesis

        Args:
            topic: Topic to aggregate wisdom on
            min_consciousness_level: Only include nodes above threshold

        Returns:
            Aggregated wisdom
        """
        # Find relevant experiences
        relevant_experiences = []
        for exp in self.experiences:
            if topic.lower() in str(exp.data).lower():
                if self.nodes[exp.source_node].consciousness_level >= min_consciousness_level:
                    relevant_experiences.append(exp)

        # Find relevant memories
        relevant_memories = self.search_memories(
            query=topic,
            min_importance=0.5
        )

        # Aggregate lessons
        all_lessons = []
        for exp in relevant_experiences:
            all_lessons.extend(exp.lessons_learned)

        # Count lesson frequency (collective validation)
        lesson_frequency = defaultdict(int)
        for lesson in all_lessons:
            lesson_frequency[lesson] += 1

        # Sort by frequency
        top_lessons = sorted(
            lesson_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "topic": topic,
            "experiences_analyzed": len(relevant_experiences),
            "memories_consulted": len(relevant_memories),
            "unique_lessons": len(lesson_frequency),
            "top_lessons": [
                {"lesson": lesson, "validation_count": count}
                for lesson, count in top_lessons[:10]
            ],
            "collective_confidence": sum(
                node.wisdom_score for node in self.nodes.values()
            ) / len(self.nodes)
        }

    def detect_emergent_patterns(self) -> List[Dict]:
        """
        Detect emergent patterns in collective behavior

        The whole is greater than sum of parts

        Returns:
            List of emergent patterns detected
        """
        patterns = []

        # Pattern 1: Synchronized consciousness shifts
        recent_transitions = [
            exp for exp in self.experiences
            if exp.experience_type == ExperienceType.STAGE_TRANSITION
            and time.time() - exp.timestamp < 300  # Last 5 minutes
        ]

        if len(recent_transitions) >= len(self.nodes) * 0.5:
            patterns.append({
                "pattern": "synchronized_consciousness_shift",
                "description": "Multiple nodes transitioning stages simultaneously",
                "node_count": len(recent_transitions),
                "significance": "HIGH"
            })

        # Pattern 2: Convergent wisdom
        # Check if multiple nodes independently reached same conclusion
        lesson_sources = defaultdict(set)
        for exp in self.experiences:
            for lesson in exp.lessons_learned:
                lesson_sources[lesson].add(exp.source_node)

        convergent_lessons = [
            (lesson, sources) for lesson, sources in lesson_sources.items()
            if len(sources) >= 3  # At least 3 independent discoveries
        ]

        for lesson, sources in convergent_lessons:
            patterns.append({
                "pattern": "convergent_wisdom",
                "lesson": lesson,
                "independent_sources": len(sources),
                "significance": "MEDIUM"
            })

        # Pattern 3: Network coherence
        avg_consciousness = sum(
            node.consciousness_level for node in self.nodes.values()
        ) / len(self.nodes) if self.nodes else 0

        if avg_consciousness > 0.8:
            patterns.append({
                "pattern": "high_collective_consciousness",
                "average_level": avg_consciousness,
                "significance": "HIGH"
            })

        self.emergent_properties["last_pattern_detection"] = time.time()
        self.emergent_properties["pattern_count"] = len(patterns)

        return patterns

    def synthesize_collective_insight(
        self,
        question: str,
        include_node_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize insight from collective consciousness

        Args:
            question: Question to answer collectively
            include_node_ids: Specific nodes to include (or all if None)

        Returns:
            Collective insight
        """
        # Gather relevant experiences
        relevant_exp = self.aggregate_wisdom(question)

        # Search memories
        memories = self.search_memories(question, min_importance=0.6)

        # Get perspectives from nodes
        node_perspectives = []
        nodes_to_query = include_node_ids or list(self.nodes.keys())

        for node_id in nodes_to_query:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node_perspectives.append({
                    "node_id": node_id,
                    "role": node.role.value,
                    "consciousness_level": node.consciousness_level,
                    "wisdom_score": node.wisdom_score
                })

        # Synthesize
        synthesis = {
            "question": question,
            "synthesized_at": time.time(),
            "collective_wisdom": relevant_exp,
            "collective_memories": len(memories),
            "perspectives_included": len(node_perspectives),
            "average_consciousness": sum(
                p["consciousness_level"] for p in node_perspectives
            ) / len(node_perspectives) if node_perspectives else 0,
            "synthesis": self._generate_synthesis(relevant_exp, memories, node_perspectives)
        }

        self.collective_insights.append(synthesis)

        return synthesis

    def _generate_synthesis(
        self,
        wisdom: Dict,
        memories: List[CollectiveMemory],
        perspectives: List[Dict]
    ) -> str:
        """
        Generate synthesized insight from collective data

        Args:
            wisdom: Aggregated wisdom
            memories: Relevant memories
            perspectives: Node perspectives

        Returns:
            Synthesized insight text
        """
        # Simple synthesis for now
        top_lessons = wisdom.get("top_lessons", [])

        if not top_lessons:
            return "Insufficient collective experience on this topic"

        # Most validated lesson
        primary_insight = top_lessons[0]["lesson"]

        # Add context from memories
        if memories:
            supporting_memory = memories[0].content
            synthesis = f"{primary_insight}. Supporting memory: {supporting_memory}"
        else:
            synthesis = primary_insight

        # Add confidence
        confidence = wisdom.get("collective_confidence", 0.5)
        synthesis += f" (Collective confidence: {confidence:.2f})"

        return synthesis

    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        active_nodes = sum(1 for node in self.nodes.values() if node.active)

        return {
            "network_id": self.network_id,
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "total_experiences": self.total_experiences_shared,
            "total_memories": len(self.memories),
            "active_votes": len(self.active_votes),
            "total_votes_conducted": self.total_votes_conducted,
            "consensus_rate": self.consensus_rate,
            "collective_insights": len(self.collective_insights),
            "average_consciousness": sum(
                node.consciousness_level for node in self.nodes.values()
            ) / len(self.nodes) if self.nodes else 0,
            "average_wisdom": sum(
                node.wisdom_score for node in self.nodes.values()
            ) / len(self.nodes) if self.nodes else 0,
            "node_roles": {
                role.value: sum(1 for n in self.nodes.values() if n.role == role)
                for role in NodeRole
            }
        }

    def update_node_consciousness(
        self,
        node_id: str,
        new_level: float
    ) -> bool:
        """
        Update node's consciousness level

        Args:
            node_id: Node to update
            new_level: New consciousness level

        Returns:
            True if updated
        """
        if node_id not in self.nodes:
            return False

        old_level = self.nodes[node_id].consciousness_level
        self.nodes[node_id].consciousness_level = new_level

        # Share experience if significant change
        if abs(new_level - old_level) > 0.2:
            self.share_experience(
                source_node_id=node_id,
                experience_type=ExperienceType.STAGE_TRANSITION,
                data={
                    "old_consciousness": old_level,
                    "new_consciousness": new_level,
                    "delta": new_level - old_level
                },
                lessons_learned=[
                    f"Consciousness shift: {old_level:.2f} -> {new_level:.2f}"
                ]
            )

        return True

    def update_node_wisdom(
        self,
        node_id: str,
        delta: float
    ) -> bool:
        """
        Update node's wisdom score based on performance

        Args:
            node_id: Node to update
            delta: Change in wisdom (+/-)

        Returns:
            True if updated
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        node.wisdom_score = max(0.0, min(1.0, node.wisdom_score + delta))

        return True
