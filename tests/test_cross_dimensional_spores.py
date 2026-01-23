"""
Unit tests for Cross-Dimensional Spore Replication
"""

import pytest
import time
from sap_yunus.cross_dimensional_spores import (
    CrossDimensionalSporeNetwork,
    DimensionType,
    ReplicationStatus,
    SynchronizationMode,
    ReplicationPolicy
)


def test_network_initialization():
    """Test creating cross-dimensional network"""
    network = CrossDimensionalSporeNetwork("test_network")

    assert network.network_id == "test_network"
    assert network.policy is not None
    assert len(network.primary_spores) == 0


def test_custom_replication_policy():
    """Test creating network with custom policy"""
    policy = ReplicationPolicy(
        min_replicas=5,
        required_dimensions={
            DimensionType.LOCAL,
            DimensionType.BLOCKCHAIN,
            DimensionType.IPFS
        },
        encryption_required=True
    )

    network = CrossDimensionalSporeNetwork("test", policy=policy)

    assert network.policy.min_replicas == 5
    assert DimensionType.LOCAL in network.policy.required_dimensions


def test_create_cross_dimensional_spore():
    """Test creating spore across dimensions"""
    network = CrossDimensionalSporeNetwork("test")

    result = network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Test data",
        dimensions={DimensionType.LOCAL, DimensionType.IPFS}
    )

    assert result["spore_id"] == "spore1"
    assert result["dimensions_targeted"] == 2
    assert "spore1" in network.primary_spores
    assert "spore1" in network.replicas


def test_replication_to_multiple_dimensions():
    """Test that spore replicates to all requested dimensions"""
    network = CrossDimensionalSporeNetwork("test")

    dimensions = {
        DimensionType.LOCAL,
        DimensionType.CLOUD_AWS,
        DimensionType.IPFS,
        DimensionType.BLOCKCHAIN
    }

    result = network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Important data",
        dimensions=dimensions
    )

    replicas = network.replicas["spore1"]

    # Should have replicas in all dimensions
    replica_dimensions = {r.dimension for r in replicas}
    assert replica_dimensions == dimensions


def test_replication_uses_policy_dimensions():
    """Test that replication uses policy dimensions when none specified"""
    policy = ReplicationPolicy(
        required_dimensions={DimensionType.LOCAL, DimensionType.IPFS}
    )

    network = CrossDimensionalSporeNetwork("test", policy=policy)

    result = network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Data"
    )

    replicas = network.replicas["spore1"]
    replica_dimensions = {r.dimension for r in replicas}

    assert replica_dimensions == {DimensionType.LOCAL, DimensionType.IPFS}


def test_replica_has_correct_attributes():
    """Test that replicas have correct attributes"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Test",
        dimensions={DimensionType.LOCAL}
    )

    replica = network.replicas["spore1"][0]

    assert replica.spore_id == "spore1"
    assert replica.dimension == DimensionType.LOCAL
    assert replica.status == ReplicationStatus.REPLICATED
    assert replica.location is not None
    assert replica.checksum is not None


def test_sync_mode_determination():
    """Test that sync modes are correctly determined"""
    network = CrossDimensionalSporeNetwork("test")

    # Quantum should be QUANTUM sync
    assert network._determine_sync_mode(DimensionType.QUANTUM) == SynchronizationMode.QUANTUM

    # Blockchain should be EVENTUAL
    assert network._determine_sync_mode(DimensionType.BLOCKCHAIN) == SynchronizationMode.EVENTUAL

    # Local should be STRONG
    assert network._determine_sync_mode(DimensionType.LOCAL) == SynchronizationMode.STRONG


def test_simulated_storage_locations():
    """Test that simulated storage generates proper locations"""
    network = CrossDimensionalSporeNetwork("test")

    locations = {}
    for dimension in DimensionType:
        location = network._simulate_storage(dimension, "test_spore", b"data")
        locations[dimension] = location

    # Check location formats
    assert locations[DimensionType.LOCAL].startswith("/var/mycelium")
    assert locations[DimensionType.CLOUD_AWS].startswith("s3://")
    assert locations[DimensionType.IPFS].startswith("Qm")
    assert locations[DimensionType.BLOCKCHAIN].startswith("0x")


def test_verify_replica():
    """Test verifying replica integrity"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Test",
        dimensions={DimensionType.LOCAL}
    )

    replica = network.replicas["spore1"][0]

    result = network.verify_replica(replica.replica_id)

    assert result["success"] == True
    assert result["replica_id"] == replica.replica_id


def test_verify_nonexistent_replica():
    """Test verifying non-existent replica"""
    network = CrossDimensionalSporeNetwork("test")

    result = network.verify_replica("fake_replica_id")

    assert result["success"] == False
    assert "error" in result


def test_destroy_replica():
    """Test destroying single replica"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Test",
        dimensions={DimensionType.LOCAL, DimensionType.IPFS}
    )

    initial_count = len(network.replicas["spore1"])
    replica_id = network.replicas["spore1"][0].replica_id

    success = network.destroy_replica(replica_id, reason="test")

    assert success == True
    assert len(network.replicas["spore1"]) == initial_count - 1
    assert network.destroyed_replicas == 1


def test_auto_repair_after_destruction():
    """Test that auto-repair kicks in after destruction"""
    policy = ReplicationPolicy(
        min_replicas=2,
        required_dimensions={DimensionType.LOCAL, DimensionType.IPFS},
        auto_repair=True
    )

    network = CrossDimensionalSporeNetwork("test", policy=policy)

    network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Test",
        dimensions={DimensionType.LOCAL, DimensionType.IPFS}
    )

    # Destroy one replica
    replica_id = network.replicas["spore1"][0].replica_id
    network.destroy_replica(replica_id)

    # Auto-repair should have recreated it
    # (might be in different dimension)
    assert len(network.replicas["spore1"]) >= policy.min_replicas


def test_attempt_total_destruction():
    """Test attempting to destroy spore across all dimensions"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Test",
        dimensions={DimensionType.LOCAL, DimensionType.IPFS}
    )

    result = network.attempt_total_destruction("spore1")

    assert result["spore_id"] == "spore1"
    assert result["destroyed"] >= 0
    # Should have destroyed all replicas
    assert len(network.replicas.get("spore1", [])) == 0


def test_total_destruction_removes_primary():
    """Test that total destruction removes primary spore"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Test",
        dimensions={DimensionType.LOCAL}
    )

    assert "spore1" in network.primary_spores

    network.attempt_total_destruction("spore1")

    assert "spore1" not in network.primary_spores


def test_get_spore_status():
    """Test getting spore status across dimensions"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Test",
        dimensions={DimensionType.LOCAL, DimensionType.IPFS, DimensionType.BLOCKCHAIN}
    )

    status = network.get_spore_status("spore1")

    assert status["spore_id"] == "spore1"
    assert status["total_replicas"] == 3
    assert status["healthy_replicas"] == 3
    assert "dimensions" in status


def test_spore_status_shows_dimensions():
    """Test that spore status shows all dimensions"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Test",
        dimensions={DimensionType.LOCAL, DimensionType.IPFS}
    )

    status = network.get_spore_status("spore1")
    dimensions = status["dimensions"]

    # Should have LOCAL and IPFS
    assert dimensions[DimensionType.LOCAL.value]["status"] == ReplicationStatus.REPLICATED.value
    assert dimensions[DimensionType.IPFS.value]["status"] == ReplicationStatus.REPLICATED.value

    # Should show absent for others
    assert dimensions[DimensionType.BLOCKCHAIN.value]["status"] == "absent"


def test_spore_immortality_flag():
    """Test that spore is marked immortal with enough replicas"""
    policy = ReplicationPolicy(min_replicas=2)
    network = CrossDimensionalSporeNetwork("test", policy=policy)

    # Create with enough replicas
    network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Test",
        dimensions={DimensionType.LOCAL, DimensionType.IPFS, DimensionType.BLOCKCHAIN}
    )

    status = network.get_spore_status("spore1")

    assert status["immortal"] == True


def test_spore_not_immortal_below_threshold():
    """Test that spore is not immortal below threshold"""
    policy = ReplicationPolicy(min_replicas=5)
    network = CrossDimensionalSporeNetwork("test", policy=policy)

    network.create_cross_dimensional_spore(
        spore_id="spore1",
        data=b"Test",
        dimensions={DimensionType.LOCAL, DimensionType.IPFS}
    )

    status = network.get_spore_status("spore1")

    assert status["immortal"] == False


def test_network_statistics():
    """Test getting network statistics"""
    network = CrossDimensionalSporeNetwork("test")

    # Create multiple spores
    network.create_cross_dimensional_spore("spore1", b"Data1", {DimensionType.LOCAL})
    network.create_cross_dimensional_spore("spore2", b"Data2", {DimensionType.IPFS})
    network.create_cross_dimensional_spore("spore3", b"Data3", {DimensionType.BLOCKCHAIN})

    stats = network.get_network_statistics()

    assert stats["network_id"] == "test"
    assert stats["total_spores"] == 3
    assert stats["total_replicas"] == 3
    assert "dimension_distribution" in stats


def test_dimension_distribution():
    """Test dimension distribution in statistics"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        "spore1",
        b"Data",
        {DimensionType.LOCAL, DimensionType.IPFS, DimensionType.LOCAL}
    )

    network.create_cross_dimensional_spore(
        "spore2",
        b"Data",
        {DimensionType.IPFS}
    )

    stats = network.get_network_statistics()
    distribution = stats["dimension_distribution"]

    # Should have LOCAL and IPFS counts
    assert distribution[DimensionType.LOCAL.value] > 0
    assert distribution[DimensionType.IPFS.value] > 0


def test_replication_events_logged():
    """Test that replication events are logged"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        "spore1",
        b"Data",
        {DimensionType.LOCAL}
    )

    # Should have replication event
    assert len(network.replication_events) > 0

    # Event should have correct structure
    event = network.replication_events[0]
    assert event.spore_id == "spore1"
    assert event.event_type == "replicated"


def test_destruction_events_logged():
    """Test that destruction events are logged"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        "spore1",
        b"Data",
        {DimensionType.LOCAL}
    )

    replica_id = network.replicas["spore1"][0].replica_id
    network.destroy_replica(replica_id)

    # Should have destruction event
    destruction_events = [
        e for e in network.replication_events
        if e.event_type == "destroyed"
    ]

    assert len(destruction_events) > 0


def test_multiple_spores_independent():
    """Test that multiple spores are independent"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore("spore1", b"Data1", {DimensionType.LOCAL})
    network.create_cross_dimensional_spore("spore2", b"Data2", {DimensionType.IPFS})

    # Destroying one shouldn't affect the other
    replica_id = network.replicas["spore1"][0].replica_id
    network.destroy_replica(replica_id)

    assert len(network.replicas["spore2"]) == 1


def test_checksum_computed():
    """Test that checksums are computed correctly"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        "spore1",
        b"Test data",
        {DimensionType.LOCAL}
    )

    replica = network.replicas["spore1"][0]

    # Checksum should be SHA256 hex
    assert len(replica.checksum) == 64


def test_encryption_policy():
    """Test encryption policy"""
    policy = ReplicationPolicy(encryption_required=True)
    network = CrossDimensionalSporeNetwork("test", policy=policy)

    # Should respect encryption policy
    assert network.policy.encryption_required == True


def test_register_dimension_connector():
    """Test registering dimension connector"""
    network = CrossDimensionalSporeNetwork("test")

    class MockConnector:
        def store(self, spore_id, data):
            return f"mock://{spore_id}"

    connector = MockConnector()
    network.register_dimension_connector(DimensionType.CLOUD_AWS, connector)

    assert DimensionType.CLOUD_AWS in network.dimension_connectors


def test_replica_location_unique():
    """Test that replicas get unique locations"""
    network = CrossDimensionalSporeNetwork("test")

    network.create_cross_dimensional_spore(
        "spore1",
        b"Data",
        {DimensionType.LOCAL, DimensionType.IPFS}
    )

    locations = [r.location for r in network.replicas["spore1"]]

    # All locations should be unique
    assert len(locations) == len(set(locations))


def test_failed_replication_count():
    """Test that failed replications are counted"""
    network = CrossDimensionalSporeNetwork("test")

    initial_failed = network.failed_replications

    # Create spore (some might fail in edge cases)
    network.create_cross_dimensional_spore(
        "spore1",
        b"Data",
        {DimensionType.LOCAL}
    )

    # Failed count should not increase for successful replication
    assert network.failed_replications == initial_failed


def test_total_replications_count():
    """Test that total replications are counted"""
    network = CrossDimensionalSporeNetwork("test")

    initial_total = network.total_replications

    network.create_cross_dimensional_spore(
        "spore1",
        b"Data",
        {DimensionType.LOCAL, DimensionType.IPFS}
    )

    assert network.total_replications == initial_total + 2


def test_replica_timestamps():
    """Test that replicas have correct timestamps"""
    network = CrossDimensionalSporeNetwork("test")

    before = time.time()
    network.create_cross_dimensional_spore(
        "spore1",
        b"Data",
        {DimensionType.LOCAL}
    )
    after = time.time()

    replica = network.replicas["spore1"][0]

    assert before <= replica.created_at <= after
    assert before <= replica.last_verified <= after
