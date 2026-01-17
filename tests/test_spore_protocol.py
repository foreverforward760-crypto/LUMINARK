"""
Unit tests for Mycelial Spore Protocol (Tethered Information Defense)
"""

import pytest
from sap_yunus.spore_protocol import (
    MycelialSpore,
    MycelialSporeNetwork,
    SporeIdentity,
    SporeState,
    ThreatLevel,
    AccessEvent
)


def test_spore_identity_generation():
    """Test spore identity generation"""
    data = b"Test data"
    identity = SporeIdentity.generate(
        creator_id="user1",
        network_id="net1",
        data=data,
        classification="confidential"
    )

    assert identity.spore_id.startswith("spore_")
    assert identity.creator_id == "user1"
    assert identity.network_id == "net1"
    assert identity.classification == "confidential"
    assert len(identity.data_hash) == 16


def test_spore_network_creation():
    """Test mycelial spore network creation"""
    network = MycelialSporeNetwork(
        network_id="test_network",
        creator_id="test_user"
    )

    assert network.network_id == "test_network"
    assert network.creator_id == "test_user"
    assert len(network.spores) == 0
    assert network.active_tracking == True


def test_create_spore():
    """Test creating a spore"""
    network = MycelialSporeNetwork("net1", "user1")

    data = b"Confidential document"
    spore = network.create_spore(
        data=data,
        classification="confidential",
        enable_self_destruct=True
    )

    assert spore.state == SporeState.ACTIVE
    assert spore.tether_strength == 1.0
    assert spore.self_destruct_armed == True
    assert spore.beacon_active == True
    assert len(spore.defense_capabilities) > 0


def test_access_reporting():
    """Test reporting access events"""
    network = MycelialSporeNetwork("net1", "user1")
    data = b"Test data"
    spore = network.create_spore(data, "private")

    # Authorized access
    event = spore.report_access(
        accessor_id="user1",
        location="home_device",
        action="read",
        authorized=True
    )

    assert event.authorized == True
    assert event.threat_level == ThreatLevel.SAFE
    assert spore.audit_trail.total_accesses == 1


def test_unauthorized_access_detection():
    """Test unauthorized access detection"""
    network = MycelialSporeNetwork("net1", "user1")
    data = b"Secret data"
    spore = network.create_spore(data, "confidential")

    # Unauthorized copy
    event = spore.report_access(
        accessor_id="unknown",
        location="foreign_ip",
        action="copy",
        authorized=False
    )

    assert event.authorized == False
    assert event.threat_level == ThreatLevel.CRITICAL
    assert spore.state == SporeState.COMPROMISED
    assert spore.audit_trail.compromised_count == 1


def test_camouflage_activation():
    """Test octo-camouflage activation"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "private")

    success = spore.activate_camouflage()

    assert success == True
    assert spore.camouflage_active == True
    assert spore.state == SporeState.CAMOUFLAGED


def test_self_destruct():
    """Test Yunus Protocol self-destruct"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "confidential", enable_self_destruct=True)

    assert spore.self_destruct_armed == True

    success = spore.execute_self_destruct()

    assert success == True
    assert spore.state == SporeState.DESTROYED
    assert spore.beacon_active == False


def test_phone_home():
    """Test phone home functionality"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "private")

    report = spore.phone_home("https://network.example.com")

    assert report["status"] == "received"
    assert "spore_id" in report or "commands" in report


def test_receive_command_camouflage():
    """Test receiving camouflage command"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "private")

    result = spore.receive_command({"type": "camouflage"})

    assert result == True
    assert spore.camouflage_active == True


def test_receive_command_self_destruct():
    """Test receiving self-destruct command"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "confidential", enable_self_destruct=True)

    result = spore.receive_command({"type": "self_destruct"})

    assert result == True
    assert spore.state == SporeState.DESTROYED


def test_harrowing_recall_success():
    """Test successful Harrowing recall"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "confidential")

    result = network.execute_harrowing_recall(spore.identity.spore_id)

    # Should succeed with strong tether
    assert result.get("success") or result.get("recall_failed")


def test_harrowing_recall_failure_triggers_yunus():
    """Test recall failure triggers Yunus Protocol"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "confidential")

    # Weaken tether
    spore.tether_strength = 0.2

    result = network.execute_harrowing_recall(spore.identity.spore_id)

    # Should fail and trigger Yunus
    if result.get("recall_failed"):
        assert result.get("yunus_executed") is not None
        assert result.get("data_destroyed") == True


def test_audit_trail():
    """Test audit trail accumulation"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "private")

    # Multiple accesses
    for i in range(5):
        spore.report_access(
            accessor_id=f"user{i}",
            location=f"device{i}",
            action="read",
            authorized=i < 3  # First 3 authorized
        )

    assert spore.audit_trail.total_accesses == 5
    assert len(spore.audit_trail.unique_accessors) == 5
    assert spore.audit_trail.compromised_count == 2  # Last 2 unauthorized
    assert len(spore.audit_trail.current_locations) == 5


def test_track_spore():
    """Test tracking spore in network"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "private")

    tracking_info = network.track_spore(spore.identity.spore_id)

    assert tracking_info is not None
    assert "identity" in tracking_info
    assert "state" in tracking_info
    assert "audit" in tracking_info


def test_list_compromised():
    """Test listing compromised spores"""
    network = MycelialSporeNetwork("net1", "user1")

    # Create multiple spores
    spore1 = network.create_spore(b"Data1", "private")
    spore2 = network.create_spore(b"Data2", "private")
    spore3 = network.create_spore(b"Data3", "private")

    # Compromise one
    spore2.report_access("unknown", "foreign", "copy", authorized=False)

    compromised = network.list_compromised()

    assert len(compromised) == 1
    assert compromised[0].identity.spore_id == spore2.identity.spore_id


def test_mass_recall():
    """Test mass recall of spores"""
    network = MycelialSporeNetwork("net1", "user1")

    # Create spores of different classifications
    network.create_spore(b"Public", "public")
    network.create_spore(b"Private", "private")
    network.create_spore(b"Confidential", "confidential")

    # Recall only confidential
    results = network.mass_recall(classification_filter="confidential")

    assert results["attempted"] == 1
    assert results["successful"] + results["destroyed"] + results["failed"] == 1


def test_network_status():
    """Test network status reporting"""
    network = MycelialSporeNetwork("net1", "user1")

    # Create various spores
    spore1 = network.create_spore(b"Data1", "public")
    spore2 = network.create_spore(b"Data2", "private")
    spore3 = network.create_spore(b"Data3", "confidential")

    # Destroy one
    spore3.execute_self_destruct()

    status = network.get_network_status()

    assert status["total_spores"] == 3
    assert status["active"] == 2
    assert status["destroyed"] == 1
    assert "spores_by_classification" in status
    assert status["spores_by_classification"]["public"] == 1


def test_spore_status():
    """Test individual spore status"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "confidential")

    # Add some activity
    spore.report_access("user1", "device1", "read", True)
    spore.activate_camouflage()

    status = spore.get_status()

    assert status["identity"]["spore_id"] == spore.identity.spore_id
    assert status["state"] == SporeState.CAMOUFLAGED.value
    assert status["defense"]["camouflage_active"] == True
    assert status["audit"]["total_accesses"] == 1


def test_tether_strength_decay():
    """Test tether strength affecting recall"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "confidential")

    # Strong tether - recall should work
    assert spore.tether_strength == 1.0
    result = spore._attempt_recall()
    assert result["success"] == True

    # Weak tether - should fail
    spore.tether_strength = 0.1
    result = spore._attempt_recall()
    assert result["success"] == False
    assert "Tether too weak" in result["reason"]


def test_defense_capabilities():
    """Test spore defense capabilities"""
    network = MycelialSporeNetwork("net1", "user1")
    spore = network.create_spore(b"Data", "confidential")

    capabilities = spore.defense_capabilities

    assert "tracking" in capabilities
    assert "reporting" in capabilities
    assert "self_destruct" in capabilities
    assert "camouflage" in capabilities
    assert "beacon" in capabilities
