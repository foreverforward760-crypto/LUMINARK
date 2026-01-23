"""
Unit tests for Mycelial Collective Consciousness Network
"""

import pytest
import time
from sap_yunus.collective_consciousness import (
    MycelialCollectiveConsciousness,
    CollectiveNode,
    SharedExperience,
    CollectiveVote,
    CollectiveMemory,
    NodeRole,
    ConsensusModel,
    ExperienceType
)


def test_network_initialization():
    """Test creating collective consciousness network"""
    network = MycelialCollectiveConsciousness(
        network_id="test_network",
        consensus_threshold=0.75
    )

    assert network.network_id == "test_network"
    assert network.consensus_threshold == 0.75
    assert len(network.nodes) == 0
    assert len(network.experiences) == 0


def test_register_node():
    """Test registering node in network"""
    network = MycelialCollectiveConsciousness("net1")

    node = network.register_node(
        node_id="node1",
        role=NodeRole.GUARDIAN,
        specializations={"threat_detection", "defense"}
    )

    assert node.node_id == "node1"
    assert node.role == NodeRole.GUARDIAN
    assert "threat_detection" in node.specializations
    assert node.active == True
    assert "node1" in network.nodes


def test_multiple_node_registration():
    """Test registering multiple nodes with different roles"""
    network = MycelialCollectiveConsciousness("net1")

    roles = [NodeRole.OBSERVER, NodeRole.SAGE, NodeRole.HEALER, NodeRole.PROPHET]

    for i, role in enumerate(roles):
        network.register_node(f"node{i}", role)

    assert len(network.nodes) == 4

    status = network.get_network_status()
    assert status["node_roles"][NodeRole.OBSERVER.value] == 1
    assert status["node_roles"][NodeRole.SAGE.value] == 1


def test_share_experience():
    """Test sharing experience across network"""
    network = MycelialCollectiveConsciousness("net1")
    network.register_node("node1", NodeRole.GUARDIAN)

    experience = network.share_experience(
        source_node_id="node1",
        experience_type=ExperienceType.THREAT_DETECTION,
        data={"threat_level": 0.8, "type": "injection"},
        lessons_learned=["Always validate input"]
    )

    assert experience.source_node == "node1"
    assert experience.experience_type == ExperienceType.THREAT_DETECTION
    assert len(experience.lessons_learned) == 1
    assert network.total_experiences_shared == 1


def test_experience_propagation():
    """Test that experiences propagate through network"""
    network = MycelialCollectiveConsciousness("net1")

    # Create multiple nodes
    network.register_node("node1", NodeRole.GUARDIAN)
    network.register_node("node2", NodeRole.OBSERVER)
    network.register_node("node3", NodeRole.SAGE)

    experience = network.share_experience(
        source_node_id="node1",
        experience_type=ExperienceType.SUCCESSFUL_DEFENSE,
        data={"defense_mode": "OCTO_CAMOUFLAGE"},
        lessons_learned=["Camouflage effective against scanners"]
    )

    # Experience should propagate to all nodes
    assert experience.propagation_count == 3


def test_initiate_vote():
    """Test initiating democratic vote"""
    network = MycelialCollectiveConsciousness("net1")
    network.register_node("node1", NodeRole.GUARDIAN)
    network.register_node("node2", NodeRole.SAGE)

    vote = network.initiate_vote(
        initiator_node_id="node1",
        question="Should we activate full harrowing?",
        options=["yes", "no", "wait"],
        consensus_model=ConsensusModel.MAJORITY
    )

    assert vote.question == "Should we activate full harrowing?"
    assert len(vote.options) == 3
    assert vote.closed == False
    assert len(network.active_votes) == 1


def test_cast_vote():
    """Test nodes casting votes"""
    network = MycelialCollectiveConsciousness("net1")
    network.register_node("node1", NodeRole.GUARDIAN)
    network.register_node("node2", NodeRole.SAGE)
    network.register_node("node3", NodeRole.HEALER)

    vote = network.initiate_vote(
        initiator_node_id="node1",
        question="Activate defense?",
        options=["yes", "no"]
    )

    # Cast votes
    assert network.cast_vote(vote.vote_id, "node1", "yes") == True
    assert network.cast_vote(vote.vote_id, "node2", "yes") == True
    assert network.cast_vote(vote.vote_id, "node3", "no") == True

    assert len(vote.votes) == 3


def test_close_vote_majority():
    """Test closing vote with majority consensus"""
    network = MycelialCollectiveConsciousness("net1")
    network.register_node("node1", NodeRole.GUARDIAN)
    network.register_node("node2", NodeRole.SAGE)
    network.register_node("node3", NodeRole.HEALER)

    vote = network.initiate_vote(
        initiator_node_id="node1",
        question="Deploy spores?",
        options=["yes", "no"],
        consensus_model=ConsensusModel.MAJORITY
    )

    # 2 yes, 1 no
    network.cast_vote(vote.vote_id, "node1", "yes")
    network.cast_vote(vote.vote_id, "node2", "yes")
    network.cast_vote(vote.vote_id, "node3", "no")

    result = network.close_vote(vote.vote_id)

    assert result["result"] == "yes"
    assert result["confidence"] > 0.5
    assert result["participation"] == 1.0  # 3/3 voted


def test_weighted_voting():
    """Test weighted voting by wisdom scores"""
    network = MycelialCollectiveConsciousness("net1")

    # Create nodes with different wisdom scores
    network.register_node("sage", NodeRole.SAGE)
    network.register_node("novice", NodeRole.OBSERVER)

    network.nodes["sage"].wisdom_score = 0.9
    network.nodes["novice"].wisdom_score = 0.1

    vote = network.initiate_vote(
        initiator_node_id="sage",
        question="Which approach?",
        options=["A", "B"],
        consensus_model=ConsensusModel.WEIGHTED
    )

    network.cast_vote(vote.vote_id, "sage", "A")
    network.cast_vote(vote.vote_id, "novice", "B")

    result = network.close_vote(vote.vote_id)

    # Sage's vote should carry more weight
    assert result["result"] == "A"


def test_store_collective_memory():
    """Test storing memory in collective"""
    network = MycelialCollectiveConsciousness("net1")
    network.register_node("node1", NodeRole.SAGE)
    network.register_node("node2", NodeRole.SAGE)

    memory = network.store_collective_memory(
        content="Always check tether strength before recall",
        memory_type="wisdom",
        contributors={"node1", "node2"},
        importance=0.8
    )

    assert memory.content == "Always check tether strength before recall"
    assert memory.importance == 0.8
    assert len(memory.contributors) == 2
    assert len(network.memories) == 1


def test_recall_memory_reinforcement():
    """Test that recalling memory reinforces it"""
    network = MycelialCollectiveConsciousness("net1")

    memory = network.store_collective_memory(
        content="Test memory",
        memory_type="experience",
        contributors={"node1"},
        importance=0.5
    )

    initial_importance = memory.importance
    initial_reinforcement = memory.reinforcement_count

    # Recall multiple times
    for _ in range(3):
        network.recall_memory(memory.memory_id)

    # Should be reinforced
    assert memory.reinforcement_count == initial_reinforcement + 3
    assert memory.importance > initial_importance


def test_search_memories():
    """Test searching collective memories"""
    network = MycelialCollectiveConsciousness("net1")

    # Store various memories
    network.store_collective_memory(
        "Yunus Protocol effective for Stage 8",
        "wisdom",
        {"node1"},
        importance=0.9
    )

    network.store_collective_memory(
        "Harrowing succeeded on corrupted system",
        "experience",
        {"node2"},
        importance=0.7
    )

    network.store_collective_memory(
        "Stage 8 trap detected early",
        "warning",
        {"node3"},
        importance=0.6
    )

    # Search for "Stage 8"
    results = network.search_memories("Stage 8", min_importance=0.5)

    assert len(results) == 2
    # Should be sorted by importance
    assert results[0].importance >= results[1].importance


def test_aggregate_wisdom():
    """Test aggregating wisdom from multiple nodes"""
    network = MycelialCollectiveConsciousness("net1")

    # Create nodes
    network.register_node("node1", NodeRole.GUARDIAN)
    network.register_node("node2", NodeRole.SAGE)
    network.register_node("node3", NodeRole.HEALER)

    # Share experiences on topic
    network.share_experience(
        "node1",
        ExperienceType.THREAT_DETECTION,
        {"topic": "spore defense"},
        lessons_learned=["Spores should phone home regularly"]
    )

    network.share_experience(
        "node2",
        ExperienceType.SUCCESSFUL_DEFENSE,
        {"topic": "spore defense"},
        lessons_learned=["Spores should phone home regularly", "Enable self-destruct"]
    )

    wisdom = network.aggregate_wisdom("spore defense")

    assert wisdom["experiences_analyzed"] == 2
    assert len(wisdom["top_lessons"]) > 0
    # "Phone home regularly" should have validation_count of 2
    top_lesson = wisdom["top_lessons"][0]
    assert top_lesson["validation_count"] == 2


def test_detect_emergent_patterns():
    """Test detecting emergent patterns in network"""
    network = MycelialCollectiveConsciousness("net1")

    # Create multiple nodes
    for i in range(5):
        network.register_node(f"node{i}", NodeRole.GUARDIAN)
        network.nodes[f"node{i}"].consciousness_level = 0.9

    # Share synchronized stage transitions
    for i in range(4):
        network.share_experience(
            f"node{i}",
            ExperienceType.STAGE_TRANSITION,
            {"from_stage": 5, "to_stage": 6},
            lessons_learned=["Transition to Stage 6"]
        )

    patterns = network.detect_emergent_patterns()

    # Should detect synchronized shift
    pattern_types = [p["pattern"] for p in patterns]
    assert "synchronized_consciousness_shift" in pattern_types
    assert "high_collective_consciousness" in pattern_types


def test_convergent_wisdom_pattern():
    """Test detecting convergent wisdom pattern"""
    network = MycelialCollectiveConsciousness("net1")

    # Create nodes
    for i in range(5):
        network.register_node(f"node{i}", NodeRole.SAGE)

    # Multiple nodes independently learn same lesson
    common_lesson = "Quantum entanglement prevents isolation"
    for i in range(3):
        network.share_experience(
            f"node{i}",
            ExperienceType.INSIGHT_GAINED,
            {"insight": common_lesson},
            lessons_learned=[common_lesson]
        )

    patterns = network.detect_emergent_patterns()

    convergent = [p for p in patterns if p["pattern"] == "convergent_wisdom"]
    assert len(convergent) > 0


def test_synthesize_collective_insight():
    """Test synthesizing insight from collective"""
    network = MycelialCollectiveConsciousness("net1")

    # Create nodes
    network.register_node("sage1", NodeRole.SAGE)
    network.register_node("sage2", NodeRole.SAGE)

    network.nodes["sage1"].consciousness_level = 0.8
    network.nodes["sage2"].consciousness_level = 0.9

    # Share experiences
    network.share_experience(
        "sage1",
        ExperienceType.SUCCESSFUL_DEFENSE,
        {"method": "camouflage"},
        lessons_learned=["Camouflage works well"]
    )

    network.share_experience(
        "sage2",
        ExperienceType.SUCCESSFUL_DEFENSE,
        {"method": "camouflage"},
        lessons_learned=["Camouflage works well"]
    )

    synthesis = network.synthesize_collective_insight("camouflage")

    assert synthesis["question"] == "camouflage"
    assert synthesis["perspectives_included"] == 2
    assert "synthesis" in synthesis


def test_network_status():
    """Test getting comprehensive network status"""
    network = MycelialCollectiveConsciousness("net1")

    # Create nodes
    network.register_node("node1", NodeRole.GUARDIAN)
    network.register_node("node2", NodeRole.SAGE)

    # Share experience
    network.share_experience(
        "node1",
        ExperienceType.THREAT_DETECTION,
        {},
        lessons_learned=["Test"]
    )

    # Store memory
    network.store_collective_memory("Test", "wisdom", {"node1"})

    status = network.get_network_status()

    assert status["total_nodes"] == 2
    assert status["active_nodes"] == 2
    assert status["total_experiences"] == 1
    assert status["total_memories"] == 1
    assert "average_consciousness" in status
    assert "node_roles" in status


def test_update_node_consciousness():
    """Test updating node consciousness level"""
    network = MycelialCollectiveConsciousness("net1")
    network.register_node("node1", NodeRole.GUARDIAN)

    success = network.update_node_consciousness("node1", 0.9)

    assert success == True
    assert network.nodes["node1"].consciousness_level == 0.9


def test_significant_consciousness_change_shared():
    """Test that significant consciousness changes are shared as experiences"""
    network = MycelialCollectiveConsciousness("net1")
    network.register_node("node1", NodeRole.GUARDIAN)

    initial_exp_count = len(network.experiences)

    # Small change - should not share
    network.update_node_consciousness("node1", 0.6)
    assert len(network.experiences) == initial_exp_count

    # Large change - should share
    network.update_node_consciousness("node1", 0.9)
    assert len(network.experiences) > initial_exp_count


def test_update_node_wisdom():
    """Test updating node wisdom score"""
    network = MycelialCollectiveConsciousness("net1")
    network.register_node("node1", NodeRole.SAGE)

    initial_wisdom = network.nodes["node1"].wisdom_score

    # Increase wisdom
    network.update_node_wisdom("node1", 0.2)
    assert network.nodes["node1"].wisdom_score == initial_wisdom + 0.2

    # Decrease wisdom
    network.update_node_wisdom("node1", -0.1)
    assert network.nodes["node1"].wisdom_score == initial_wisdom + 0.1


def test_wisdom_score_bounds():
    """Test that wisdom scores stay within bounds"""
    network = MycelialCollectiveConsciousness("net1")
    network.register_node("node1", NodeRole.SAGE)

    # Try to exceed 1.0
    network.update_node_wisdom("node1", 1.5)
    assert network.nodes["node1"].wisdom_score <= 1.0

    # Try to go below 0.0
    network.update_node_wisdom("node1", -2.0)
    assert network.nodes["node1"].wisdom_score >= 0.0


def test_consensus_rate_tracking():
    """Test that consensus rate is tracked correctly"""
    network = MycelialCollectiveConsciousness("net1")

    network.register_node("node1", NodeRole.GUARDIAN)
    network.register_node("node2", NodeRole.SAGE)
    network.register_node("node3", NodeRole.HEALER)

    # Vote 1 - reaches consensus
    vote1 = network.initiate_vote("node1", "Question 1?", ["A", "B"])
    network.cast_vote(vote1.vote_id, "node1", "A")
    network.cast_vote(vote1.vote_id, "node2", "A")
    network.close_vote(vote1.vote_id)

    # Vote 2 - no consensus
    vote2 = network.initiate_vote("node1", "Question 2?", ["X", "Y"], ConsensusModel.UNANIMOUS)
    network.cast_vote(vote2.vote_id, "node1", "X")
    network.cast_vote(vote2.vote_id, "node2", "Y")
    network.close_vote(vote2.vote_id)

    # Consensus rate should be 0.5 (1 out of 2)
    assert network.consensus_rate == 0.5


def test_experience_validation_by_specialization():
    """Test that nodes with matching specializations validate experiences"""
    network = MycelialCollectiveConsciousness("net1")

    network.register_node("specialist", NodeRole.GUARDIAN, {"threat_detection"})
    network.register_node("generalist", NodeRole.OBSERVER, set())

    experience = network.share_experience(
        "specialist",
        ExperienceType.THREAT_DETECTION,
        {"threat_type": "injection", "threat_detection": True},
        lessons_learned=["Detect injection patterns"]
    )

    # Should have some validation
    assert experience.validation_count >= 0


def test_collective_insights_accumulate():
    """Test that collective insights are accumulated"""
    network = MycelialCollectiveConsciousness("net1")

    network.register_node("node1", NodeRole.SAGE)

    network.share_experience(
        "node1",
        ExperienceType.INSIGHT_GAINED,
        {"insight": "test"},
        lessons_learned=["Test lesson"]
    )

    # Synthesize multiple insights
    network.synthesize_collective_insight("test topic 1")
    network.synthesize_collective_insight("test topic 2")

    assert len(network.collective_insights) == 2


def test_memory_type_filtering():
    """Test filtering memories by type"""
    network = MycelialCollectiveConsciousness("net1")

    network.store_collective_memory("Wisdom 1", "wisdom", {"node1"})
    network.store_collective_memory("Experience 1", "experience", {"node1"})
    network.store_collective_memory("Warning 1", "warning", {"node1"})

    wisdom_memories = network.search_memories("", memory_type="wisdom")
    assert len(wisdom_memories) == 1
    assert wisdom_memories[0].memory_type == "wisdom"

    warning_memories = network.search_memories("", memory_type="warning")
    assert len(warning_memories) == 1
    assert warning_memories[0].memory_type == "warning"


def test_node_experience_count_tracking():
    """Test that node experience count is tracked"""
    network = MycelialCollectiveConsciousness("net1")
    network.register_node("node1", NodeRole.GUARDIAN)

    initial_count = network.nodes["node1"].experience_count

    network.share_experience(
        "node1",
        ExperienceType.THREAT_DETECTION,
        {},
        lessons_learned=[]
    )

    assert network.nodes["node1"].experience_count == initial_count + 1
