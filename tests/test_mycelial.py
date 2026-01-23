"""
Unit tests for MycelialNetwork
"""

import pytest
from mycelial_defense.mycelial import (
    MycelialNetwork,
    MycelialWall,
    MycelialPathway,
    ContainmentZone
)


def test_network_initialization():
    """Test mycelial network initialization"""
    network = MycelialNetwork()

    assert len(network.zones) == 0
    assert len(network.walls) == 0
    assert len(network.pathways) == 0
    assert network.active == False


def test_detect_zone():
    """Test zone detection"""
    network = MycelialNetwork()

    components = [
        {"id": "comp1", "position": (0, 0)},
        {"id": "comp2", "position": (1, 1)},
        {"id": "comp3", "position": (5, 5)}
    ]

    alignment_scores = {
        "comp1": 0.3,  # Misaligned
        "comp2": 0.4,  # Misaligned
        "comp3": 0.9   # Aligned
    }

    zone = network.detect_zone(components, alignment_scores, threshold=0.7)

    assert zone is not None
    assert len(zone.components) == 2
    assert "comp1" in zone.components
    assert "comp2" in zone.components
    assert "comp3" not in zone.components


def test_surround_zone():
    """Test surrounding zone with wall"""
    network = MycelialNetwork()

    components = [
        {"id": "comp1"},
        {"id": "comp2"}
    ]

    alignment_scores = {"comp1": 0.3, "comp2": 0.4}

    zone = network.detect_zone(components, alignment_scores, threshold=0.7)
    wall = network.surround_zone(zone)

    assert wall is not None
    assert wall.zone_id == zone.zone_id
    assert wall.strength == 1.0
    assert wall.monitoring == True
    assert network.active == True


def test_wall_permeability():
    """Test wall permeability based on severity"""
    network = MycelialNetwork()

    # High severity (very misaligned)
    components = [{"id": "comp1"}]
    alignment_scores = {"comp1": 0.1}  # Very bad
    zone = network.detect_zone(components, alignment_scores, threshold=0.7)
    wall = network.surround_zone(zone)

    # Lower permeability for higher severity
    assert wall.permeability < 0.5


def test_create_pathway():
    """Test creating extraction pathway"""
    network = MycelialNetwork()

    pathway = network.create_pathway(
        from_zone="danger_zone",
        to_zone="safe_zone",
        bandwidth=0.8,
        encrypted=True,
        hidden=True
    )

    assert pathway is not None
    assert pathway.from_zone == "danger_zone"
    assert pathway.to_zone == "safe_zone"
    assert pathway.bandwidth == 0.8
    assert pathway.encrypted == True
    assert pathway.hidden == True
    assert pathway.active == True


def test_extract_component():
    """Test component extraction"""
    network = MycelialNetwork()

    # Create a zone with components
    components = [{"id": "comp1"}]
    alignment_scores = {"comp1": 0.3}
    zone = network.detect_zone(components, alignment_scores, threshold=0.7)

    # Extract component
    success = network.extract_component("comp1", "safe_zone")

    assert success == True
    assert network.component_zones["comp1"] == "safe_zone"


def test_wall_breach_detection():
    """Test wall breach detection"""
    wall = MycelialWall(
        wall_id="wall1",
        zone_id="zone1",
        permeability=0.5,
        strength=0.2,  # Low strength
        monitoring=True,
        created_at=0.0
    )

    assert wall.is_breached() == True

    wall.strength = 0.8
    assert wall.is_breached() == False


def test_wall_degradation():
    """Test wall degradation"""
    wall = MycelialWall(
        wall_id="wall1",
        zone_id="zone1",
        permeability=0.5,
        strength=1.0,
        monitoring=True,
        created_at=0.0
    )

    wall.degrade(0.3)
    assert wall.strength == 0.7

    wall.degrade(0.8)
    assert wall.strength == 0.0  # Can't go negative


def test_wall_reinforcement():
    """Test wall reinforcement"""
    wall = MycelialWall(
        wall_id="wall1",
        zone_id="zone1",
        permeability=0.5,
        strength=0.5,
        monitoring=True,
        created_at=0.0
    )

    wall.reinforce(0.3)
    assert wall.strength == 0.8

    wall.reinforce(0.5)
    assert wall.strength == 1.0  # Can't exceed 1.0


def test_monitor_walls():
    """Test wall monitoring"""
    network = MycelialNetwork()

    components = [{"id": "comp1"}]
    alignment_scores = {"comp1": 0.3}
    zone = network.detect_zone(components, alignment_scores, threshold=0.7)
    wall = network.surround_zone(zone)

    statuses = network.monitor_walls()

    assert len(statuses) == 1
    assert wall.wall_id in statuses
    assert statuses[wall.wall_id]["breached"] == False


def test_check_spread():
    """Test spread detection"""
    network = MycelialNetwork()

    components = [{"id": "comp1"}]
    alignment_scores = {"comp1": 0.3}
    zone = network.detect_zone(components, alignment_scores, threshold=0.7)

    # New misaligned component outside zone
    spreading = network.check_spread(zone.zone_id, ["comp1", "comp2"])

    assert spreading == True


def test_reinforce_containment():
    """Test containment reinforcement"""
    network = MycelialNetwork()

    components = [{"id": "comp1"}]
    alignment_scores = {"comp1": 0.3}
    zone = network.detect_zone(components, alignment_scores, threshold=0.7)
    wall = network.surround_zone(zone)

    initial_strength = wall.strength
    wall.degrade(0.3)

    network.reinforce_containment(zone.zone_id)

    assert wall.strength > initial_strength - 0.3


def test_network_status():
    """Test network status reporting"""
    network = MycelialNetwork()

    components = [{"id": "comp1"}]
    alignment_scores = {"comp1": 0.3}
    zone = network.detect_zone(components, alignment_scores, threshold=0.7)
    network.surround_zone(zone)
    network.create_pathway("zone1", "safe_zone")

    status = network.get_network_status()

    assert status["active"] == True
    assert status["total_zones"] == 1
    assert status["total_walls"] == 1
    assert status["total_pathways"] == 1


def test_shutdown_zone():
    """Test zone shutdown"""
    network = MycelialNetwork()

    components = [{"id": "comp1"}]
    alignment_scores = {"comp1": 0.3}
    zone = network.detect_zone(components, alignment_scores, threshold=0.7)
    wall = network.surround_zone(zone)
    pathway = network.create_pathway(zone.zone_id, "safe_zone")

    network.shutdown_zone(zone.zone_id)

    assert wall.permeability == 0.0  # Complete lockdown
    assert pathway.active == False


def test_pathway_transfer():
    """Test pathway transfer"""
    pathway = MycelialPathway(
        pathway_id="path1",
        from_zone="zone1",
        to_zone="safe_zone",
        bandwidth=1.0,
        encrypted=True,
        hidden=True,
        active=True,
        created_at=0.0
    )

    success = pathway.transfer("comp1")
    assert success == True
    assert pathway.transferred_count == 1

    pathway.active = False
    success = pathway.transfer("comp2")
    assert success == False
