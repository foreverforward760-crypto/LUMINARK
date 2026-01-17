"""
Unit tests for Quantum Entangled Spores
"""

import pytest
from sap_yunus.quantum_spore import (
    QuantumEntangledSpore,
    QuantumSporeNetwork,
    QuantumState,
    QuantumCorrelation,
    QuantumBroadcast
)
from sap_yunus.spore_protocol import SporeIdentity


def test_quantum_spore_creation():
    """Test creating quantum-entangled spore"""
    identity = SporeIdentity.generate(
        creator_id="user1",
        network_id="net1",
        data=b"Test data",
        classification="confidential"
    )

    spore = QuantumEntangledSpore(identity, enable_self_destruct=True)

    assert spore.quantum_state == QuantumState.ENTANGLED
    assert len(spore.entangled_spores) == 0
    assert spore.entanglement_strength == 1.0
    assert spore.quantum_signature is not None
    assert len(spore.quantum_signature) == 64  # SHA256 hex


def test_entanglement_creation():
    """Test creating entanglement between spores"""
    identity1 = SporeIdentity.generate("user1", "net1", b"Data1", "private")
    identity2 = SporeIdentity.generate("user1", "net1", b"Data2", "private")

    spore1 = QuantumEntangledSpore(identity1)
    spore2 = QuantumEntangledSpore(identity2)

    # Entangle spore1 with spore2
    success = spore1.entangle_with(identity2.spore_id)

    assert success == True
    assert identity2.spore_id in spore1.entangled_spores
    assert identity2.spore_id in spore1.correlations
    assert spore1.correlations[identity2.spore_id].correlation_coefficient == 1.0


def test_cannot_entangle_with_self():
    """Test that spore cannot entangle with itself"""
    identity = SporeIdentity.generate("user1", "net1", b"Data", "private")
    spore = QuantumEntangledSpore(identity)

    success = spore.entangle_with(identity.spore_id)

    assert success == False
    assert len(spore.entangled_spores) == 0


def test_quantum_broadcast():
    """Test quantum state broadcasting"""
    identity1 = SporeIdentity.generate("user1", "net1", b"Data1", "private")
    identity2 = SporeIdentity.generate("user1", "net1", b"Data2", "private")

    spore1 = QuantumEntangledSpore(identity1)
    spore2 = QuantumEntangledSpore(identity2)

    spore1.entangle_with(identity2.spore_id)

    # Broadcast state update
    state_update = {"test_value": 42, "status": "active"}
    broadcast = spore1.broadcast_state(state_update, instantaneous=True)

    assert broadcast.source_spore == identity1.spore_id
    assert broadcast.propagation_speed == "instantaneous"
    assert identity2.spore_id in broadcast.receivers
    assert len(spore1.broadcast_history) == 1


def test_receive_broadcast():
    """Test receiving quantum broadcast"""
    identity1 = SporeIdentity.generate("user1", "net1", b"Data1", "private")
    identity2 = SporeIdentity.generate("user1", "net1", b"Data2", "private")

    spore1 = QuantumEntangledSpore(identity1)
    spore2 = QuantumEntangledSpore(identity2)

    # Create bidirectional entanglement
    spore1.entangle_with(identity2.spore_id)
    spore2.entangle_with(identity1.spore_id)

    # Broadcast from spore1
    state_update = {"tether_strength": 0.95}
    broadcast = spore1.broadcast_state(state_update)

    # Receive on spore2
    received = spore2.receive_broadcast(broadcast, identity1.spore_id)

    assert received == True
    assert broadcast.broadcast_id in spore2.received_broadcasts
    assert identity2.spore_id in broadcast.confirmed_receipts


def test_measure_entanglement():
    """Test measuring entanglement strength"""
    identity1 = SporeIdentity.generate("user1", "net1", b"Data1", "private")
    identity2 = SporeIdentity.generate("user1", "net1", b"Data2", "private")

    spore1 = QuantumEntangledSpore(identity1)
    spore2 = QuantumEntangledSpore(identity2)

    spore1.entangle_with(identity2.spore_id)

    # Measure entanglement
    strength = spore1.measure_entanglement(identity2.spore_id)

    assert strength is not None
    assert 0.0 <= strength <= 1.0
    assert spore1.correlations[identity2.spore_id].measurement_count == 1


def test_entanglement_collapse_detection():
    """Test detecting entanglement collapse"""
    identity1 = SporeIdentity.generate("user1", "net1", b"Data1", "confidential")
    identity2 = SporeIdentity.generate("user1", "net1", b"Data2", "confidential")

    spore1 = QuantumEntangledSpore(identity1, enable_self_destruct=True)
    spore1.entangle_with(identity2.spore_id)

    # Manually weaken entanglement
    spore1.correlations[identity2.spore_id].correlation_coefficient = 0.2

    # Measure should detect collapse
    strength = spore1.measure_entanglement(identity2.spore_id)

    assert strength < spore1.collapse_threshold
    assert len(spore1.collapse_alerts) > 0
    assert spore1.collapse_alerts[0]["event"] == "entanglement_collapse"
    assert spore1.collapse_alerts[0]["severity"] == "CRITICAL"


def test_complete_isolation_triggers_defense():
    """Test that complete isolation triggers defensive measures"""
    identity1 = SporeIdentity.generate("user1", "net1", b"Secret", "confidential")
    identity2 = SporeIdentity.generate("user1", "net1", b"Data2", "confidential")

    spore1 = QuantumEntangledSpore(identity1, enable_self_destruct=True)
    spore1.entangle_with(identity2.spore_id)

    initial_accesses = spore1.audit_trail.total_accesses

    # Simulate complete entanglement collapse (isolation attack)
    spore1.correlations[identity2.spore_id].correlation_coefficient = 0.01
    spore1.measure_entanglement(identity2.spore_id)

    # Should trigger security response
    assert len(spore1.collapse_alerts) > 0
    # Camouflage should activate or self-destruct should occur


def test_superposition_entry():
    """Test entering quantum superposition"""
    identity = SporeIdentity.generate("user1", "net1", b"Data", "private")
    spore = QuantumEntangledSpore(identity)

    states = [
        {"name": "defensive", "camouflage_active": True},
        {"name": "monitoring", "beacon_active": True},
        {"name": "dormant", "beacon_active": False}
    ]

    success = spore.enter_superposition(states)

    assert success == True
    assert spore.quantum_state == QuantumState.SUPERPOSITION
    assert len(spore.superposition_states) == 3


def test_superposition_requires_multiple_states():
    """Test that superposition requires at least 2 states"""
    identity = SporeIdentity.generate("user1", "net1", b"Data", "private")
    spore = QuantumEntangledSpore(identity)

    success = spore.enter_superposition([{"name": "single"}])

    assert success == False
    assert spore.quantum_state != QuantumState.SUPERPOSITION


def test_collapse_superposition():
    """Test collapsing superposition to single state"""
    identity = SporeIdentity.generate("user1", "net1", b"Data", "private")
    spore = QuantumEntangledSpore(identity)

    states = [
        {"name": "state1", "test_value": 1},
        {"name": "state2", "test_value": 2}
    ]

    spore.enter_superposition(states)
    collapsed = spore.collapse_superposition(measurement="state2")

    assert collapsed["name"] == "state2"
    assert len(spore.superposition_states) == 0
    assert spore.quantum_state != QuantumState.SUPERPOSITION


def test_quantum_status():
    """Test getting quantum status"""
    identity1 = SporeIdentity.generate("user1", "net1", b"Data1", "private")
    identity2 = SporeIdentity.generate("user1", "net1", b"Data2", "private")

    spore1 = QuantumEntangledSpore(identity1)
    spore1.entangle_with(identity2.spore_id)

    status = spore1.get_quantum_status()

    assert "quantum_state" in status
    assert "entangled_partners" in status
    assert "correlations" in status
    assert "isolation_risk" in status
    assert status["entangled_partners"] == 1
    assert status["quantum_signature"] == spore1.quantum_signature


def test_verify_entanglement():
    """Test verifying active entanglement"""
    identity1 = SporeIdentity.generate("user1", "net1", b"Data1", "private")
    identity2 = SporeIdentity.generate("user1", "net1", b"Data2", "private")

    spore1 = QuantumEntangledSpore(identity1)
    spore1.entangle_with(identity2.spore_id)

    # Should verify successfully
    verified = spore1.verify_entanglement(identity2.spore_id)
    assert verified == True

    # Break entanglement
    spore1.correlations[identity2.spore_id].correlation_coefficient = 0.1

    # Should fail verification
    verified = spore1.verify_entanglement(identity2.spore_id)
    assert verified == False


def test_quantum_network_creation():
    """Test creating quantum spore network"""
    network = QuantumSporeNetwork("net1", "user1")

    assert network.network_id == "net1"
    assert network.creator_id == "user1"
    assert len(network.quantum_spores) == 0


def test_network_creates_quantum_spore():
    """Test network creating quantum spore"""
    network = QuantumSporeNetwork("net1", "user1")

    spore = network.create_quantum_spore(
        data=b"Test data",
        classification="confidential",
        auto_entangle=False
    )

    assert spore.identity.spore_id in network.quantum_spores
    assert isinstance(spore, QuantumEntangledSpore)


def test_auto_entangle_on_creation():
    """Test automatic entanglement when creating spores"""
    network = QuantumSporeNetwork("net1", "user1")

    # Create first spore
    spore1 = network.create_quantum_spore(b"Data1", "private", auto_entangle=True)

    # Create second spore - should auto-entangle with first
    spore2 = network.create_quantum_spore(b"Data2", "private", auto_entangle=True)

    assert spore2.identity.spore_id in spore1.entangled_spores
    assert spore1.identity.spore_id in spore2.entangled_spores


def test_network_entangle_spores():
    """Test network entangling two spores"""
    network = QuantumSporeNetwork("net1", "user1")

    spore1 = network.create_quantum_spore(b"Data1", "private", auto_entangle=False)
    spore2 = network.create_quantum_spore(b"Data2", "private", auto_entangle=False)

    success = network.entangle_spores(
        spore1.identity.spore_id,
        spore2.identity.spore_id
    )

    assert success == True
    assert spore2.identity.spore_id in network.entanglement_map[spore1.identity.spore_id]
    assert spore1.identity.spore_id in network.entanglement_map[spore2.identity.spore_id]


def test_network_broadcast():
    """Test broadcasting across quantum network"""
    network = QuantumSporeNetwork("net1", "user1")

    spore1 = network.create_quantum_spore(b"Data1", "private", auto_entangle=True)
    spore2 = network.create_quantum_spore(b"Data2", "private", auto_entangle=True)
    spore3 = network.create_quantum_spore(b"Data3", "private", auto_entangle=True)

    state_update = {"network_alert": "test"}
    result = network.broadcast_to_network(spore1.identity.spore_id, state_update)

    assert result["sent"] == 2  # Sent to spore2 and spore3
    assert "broadcast_id" in result


def test_network_coherence():
    """Test calculating network quantum coherence"""
    network = QuantumSporeNetwork("net1", "user1")

    # Create fully entangled network
    spore1 = network.create_quantum_spore(b"Data1", "private", auto_entangle=True)
    spore2 = network.create_quantum_spore(b"Data2", "private", auto_entangle=True)
    spore3 = network.create_quantum_spore(b"Data3", "private", auto_entangle=True)

    coherence = network.get_network_coherence()

    # Should be high (near 1.0) for fresh entanglements
    assert coherence > 0.9


def test_detect_isolation_attacks():
    """Test detecting isolation attacks on network"""
    network = QuantumSporeNetwork("net1", "user1")

    spore1 = network.create_quantum_spore(b"Data1", "private", auto_entangle=True)
    spore2 = network.create_quantum_spore(b"Data2", "private", auto_entangle=True)
    spore3 = network.create_quantum_spore(b"Data3", "private", auto_entangle=False)

    # spore3 is isolated (not entangled)
    alerts = network.detect_isolation_attacks()

    assert len(alerts) > 0
    # Should detect spore3 as isolated
    isolated_ids = [alert["spore_id"] for alert in alerts if alert["event"] == "complete_isolation"]
    assert spore3.identity.spore_id in isolated_ids


def test_quantum_network_status():
    """Test getting quantum network status"""
    network = QuantumSporeNetwork("net1", "user1")

    network.create_quantum_spore(b"Data1", "private", auto_entangle=True)
    network.create_quantum_spore(b"Data2", "private", auto_entangle=True)

    status = network.get_quantum_network_status()

    assert status["network_id"] == "net1"
    assert status["total_quantum_spores"] == 2
    assert status["total_entanglements"] == 1  # One pair
    assert "network_coherence" in status
    assert "fully_connected" in status


def test_fully_connected_network():
    """Test that auto-entangle creates fully connected network"""
    network = QuantumSporeNetwork("net1", "user1")

    # Create 4 spores with auto-entangle
    spores = []
    for i in range(4):
        spore = network.create_quantum_spore(
            data=f"Data{i}".encode(),
            classification="private",
            auto_entangle=True
        )
        spores.append(spore)

    # Each spore should be entangled with all others
    for spore in spores:
        expected_partners = len(spores) - 1  # All except self
        assert len(spore.entangled_spores) == expected_partners

    status = network.get_quantum_network_status()
    assert status["fully_connected"] == True


def test_correlation_degrades_over_measurements():
    """Test that correlation degrades slightly with measurements"""
    identity1 = SporeIdentity.generate("user1", "net1", b"Data1", "private")
    identity2 = SporeIdentity.generate("user1", "net1", b"Data2", "private")

    spore1 = QuantumEntangledSpore(identity1)
    spore1.entangle_with(identity2.spore_id)

    # Initial correlation
    initial = spore1.measure_entanglement(identity2.spore_id)

    # Many measurements
    for _ in range(10):
        spore1.measure_entanglement(identity2.spore_id)

    final = spore1.measure_entanglement(identity2.spore_id)

    # Should degrade slightly (observer effect)
    assert final < initial


def test_quantum_signature_unique():
    """Test that each spore gets unique quantum signature"""
    identity1 = SporeIdentity.generate("user1", "net1", b"Data", "private")
    identity2 = SporeIdentity.generate("user1", "net1", b"Data", "private")

    spore1 = QuantumEntangledSpore(identity1)
    spore2 = QuantumEntangledSpore(identity2)

    assert spore1.quantum_signature != spore2.quantum_signature
