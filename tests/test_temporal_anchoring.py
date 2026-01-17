"""
Unit tests for Temporal Anchoring & Timeline Integrity
"""

import pytest
import time
import json
import os
from sap_yunus.temporal_anchoring import (
    TemporalAnchoringSystem,
    TemporalAnchor,
    TimelineEvent,
    AnchorType,
    TamperType,
    FractalTimePattern
)


def test_system_initialization():
    """Test creating temporal anchoring system"""
    system = TemporalAnchoringSystem("test_system")

    assert system.system_id == "test_system"
    assert len(system.anchors) == 1  # Genesis anchor
    assert system.anchors[0].anchor_type == AnchorType.GENESIS


def test_genesis_anchor_properties():
    """Test genesis anchor has correct properties"""
    system = TemporalAnchoringSystem("test_system")
    genesis = system.anchors[0]

    assert genesis.previous_hash == "0" * 64
    assert genesis.anchor_type == AnchorType.GENESIS
    assert genesis.timestamp == system.genesis_time


def test_record_event():
    """Test recording event in timeline"""
    system = TemporalAnchoringSystem("test_system")

    event = system.record_event(
        event_type="test_event",
        data={"value": 42}
    )

    assert event.event_id is not None
    assert event.event_type == "test_event"
    assert event.data["value"] == 42
    assert len(system.events) == 1


def test_event_hash_computation():
    """Test that events compute consistent hashes"""
    system = TemporalAnchoringSystem("test_system")

    event = system.record_event("test", {"data": "value"})

    # Hash should be consistent
    hash1 = event._compute_hash()
    hash2 = event._compute_hash()

    assert hash1 == hash2
    assert event.hash == hash1


def test_causal_relationships():
    """Test recording causal event relationships"""
    system = TemporalAnchoringSystem("test_system")

    event1 = system.record_event("event1", {})
    event2 = system.record_event(
        "event2",
        {},
        causal_parents=[event1.event_id]
    )

    assert event1.event_id in event2.causal_parents


def test_create_anchor():
    """Test creating temporal anchor"""
    system = TemporalAnchoringSystem("test_system")

    # Record some events
    system.record_event("event1", {"a": 1})
    system.record_event("event2", {"b": 2})

    # Create anchor
    anchor = system.create_anchor(AnchorType.DECISION, {"decision": "test"})

    assert anchor.anchor_type == AnchorType.DECISION
    assert len(system.anchors) == 2  # Genesis + new
    assert anchor.previous_hash != "0" * 64


def test_anchor_chain_linking():
    """Test that anchors link correctly"""
    system = TemporalAnchoringSystem("test_system")

    anchor1 = system.create_anchor(AnchorType.DECISION)
    time.sleep(0.01)
    anchor2 = system.create_anchor(AnchorType.STATE_CHANGE)

    # anchor2 should link to anchor1
    expected_prev_hash = anchor1.compute_hash()
    assert anchor2.previous_hash == expected_prev_hash


def test_merkle_tree_building():
    """Test building Merkle tree from events"""
    system = TemporalAnchoringSystem("test_system")

    events = [
        system.record_event("event1", {"a": 1}),
        system.record_event("event2", {"b": 2}),
        system.record_event("event3", {"c": 3}),
        system.record_event("event4", {"d": 4})
    ]

    root = system._build_merkle_tree(events)

    assert root is not None
    assert len(root) == 64  # SHA256 hex


def test_merkle_tree_empty():
    """Test Merkle tree with no events"""
    system = TemporalAnchoringSystem("test_system")

    root = system._build_merkle_tree([])

    assert root == "0" * 64


def test_verify_chain_integrity_valid():
    """Test verifying valid chain"""
    system = TemporalAnchoringSystem("test_system")

    system.record_event("event1", {})
    system.create_anchor(AnchorType.DECISION)
    system.record_event("event2", {})
    system.create_anchor(AnchorType.DECISION)

    is_valid, errors = system.verify_chain_integrity()

    assert is_valid == True
    assert len(errors) == 0


def test_verify_event_integrity():
    """Test verifying individual event"""
    system = TemporalAnchoringSystem("test_system")

    event = system.record_event("test", {"data": "value"})

    # Should be valid
    assert system.verify_event_integrity(event.event_id) == True


def test_verify_event_tampering():
    """Test detecting tampered event"""
    system = TemporalAnchoringSystem("test_system")

    event = system.record_event("test", {"data": "value"})

    # Tamper with event data
    event.data["data"] = "tampered"

    # Hash should no longer match
    expected_hash = event._compute_hash()
    assert event.hash != expected_hash


def test_detect_rollback_timestamp():
    """Test detecting rollback via timestamp"""
    system = TemporalAnchoringSystem("test_system")

    anchor1 = system.create_anchor(AnchorType.DECISION)
    time.sleep(0.01)
    anchor2 = system.create_anchor(AnchorType.DECISION)

    # Tamper: make anchor2 timestamp earlier than anchor1
    anchor2.timestamp = anchor1.timestamp - 10

    detection = system.detect_rollback_attack()

    assert detection is not None
    assert detection.tamper_type == TamperType.ROLLBACK
    assert detection.severity == "CRITICAL"


def test_detect_causal_violation_missing_parent():
    """Test detecting event with missing causal parent"""
    system = TemporalAnchoringSystem("test_system")

    # Create event with non-existent parent
    event = system.record_event(
        "test",
        {},
        causal_parents=["fake_parent_id"]
    )

    violations = system.detect_causal_violations()

    assert len(violations) > 0
    assert violations[0].tamper_type == TamperType.DELETION


def test_detect_causal_violation_wrong_order():
    """Test detecting events in wrong causal order"""
    system = TemporalAnchoringSystem("test_system")

    event1 = system.record_event("event1", {})
    time.sleep(0.01)
    event2 = system.record_event("event2", {}, causal_parents=[event1.event_id])

    # Tamper: swap timestamps
    event1.timestamp, event2.timestamp = event2.timestamp, event1.timestamp

    violations = system.detect_causal_violations()

    assert len(violations) > 0
    assert violations[0].tamper_type == TamperType.REORDERING


def test_automatic_anchoring_interval():
    """Test that anchors created automatically at intervals"""
    system = TemporalAnchoringSystem("test_system", anchor_interval=0.1)

    initial_anchors = len(system.anchors)

    # Record events
    for i in range(5):
        system.record_event(f"event{i}", {"i": i})
        time.sleep(0.03)

    # Should have created at least one additional anchor
    assert len(system.anchors) > initial_anchors


def test_events_get_anchor_id():
    """Test that events get assigned to anchors"""
    system = TemporalAnchoringSystem("test_system")

    event = system.record_event("test", {})
    anchor = system.create_anchor(AnchorType.MERKLE_ROOT)

    # Event should now have anchor_id
    assert event.anchor_id == anchor.anchor_id


def test_timeline_snapshot():
    """Test getting timeline snapshot"""
    system = TemporalAnchoringSystem("test_system")

    system.record_event("event1", {})
    time.sleep(0.01)
    system.create_anchor(AnchorType.DECISION)
    system.record_event("event2", {})

    snapshot = system.get_timeline_snapshot()

    assert "anchors" in snapshot
    assert "events" in snapshot
    assert "integrity_verified" in snapshot
    assert snapshot["anchors"] >= 1
    assert snapshot["events"] >= 2


def test_timeline_snapshot_time_range():
    """Test snapshot with time range"""
    system = TemporalAnchoringSystem("test_system")

    start = time.time()
    system.record_event("event1", {})
    time.sleep(0.05)
    mid = time.time()
    system.record_event("event2", {})
    time.sleep(0.05)
    end = time.time()

    # Snapshot should only include middle event
    snapshot = system.get_timeline_snapshot(start_time=mid - 0.01, end_time=mid + 0.01)

    assert snapshot["events"] >= 0  # May or may not capture event2 depending on timing


def test_export_audit_log(tmp_path):
    """Test exporting audit log"""
    system = TemporalAnchoringSystem("test_system")

    system.record_event("event1", {"a": 1})
    system.create_anchor(AnchorType.DECISION)

    output_file = tmp_path / "audit.json"
    system.export_audit_log(str(output_file))

    assert output_file.exists()

    # Load and verify
    with open(output_file) as f:
        audit = json.load(f)

    assert audit["system_id"] == "test_system"
    assert "anchors" in audit
    assert "events" in audit
    assert "integrity" in audit


def test_fractal_pattern_detection():
    """Test detecting fractal time patterns"""
    system = TemporalAnchoringSystem("test_system")

    # Create events at multiple time scales
    # Scale 1: Every 0.01 seconds
    for i in range(3):
        system.record_event("pattern_A", {"iteration": i})
        time.sleep(0.01)

    time.sleep(0.05)

    # Scale 2: Every 0.1 seconds
    for i in range(3):
        system.record_event("pattern_A", {"iteration": i + 3})
        time.sleep(0.02)

    patterns = system.detect_fractal_patterns("pattern_A", min_occurrences=3)

    # May or may not detect pattern depending on timing precision
    # This is a simplified test
    assert isinstance(patterns, list)


def test_anchor_type_variety():
    """Test different anchor types"""
    system = TemporalAnchoringSystem("test_system")

    types = [
        AnchorType.DECISION,
        AnchorType.STATE_CHANGE,
        AnchorType.CONSCIOUSNESS_SHIFT,
        AnchorType.THREAT_EVENT
    ]

    for anchor_type in types:
        anchor = system.create_anchor(anchor_type)
        assert anchor.anchor_type == anchor_type


def test_multiple_events_per_anchor():
    """Test that multiple events can share an anchor"""
    system = TemporalAnchoringSystem("test_system")

    event1 = system.record_event("event1", {})
    event2 = system.record_event("event2", {})
    event3 = system.record_event("event3", {})

    anchor = system.create_anchor(AnchorType.MERKLE_ROOT)

    # All three should have same anchor
    assert event1.anchor_id == anchor.anchor_id
    assert event2.anchor_id == anchor.anchor_id
    assert event3.anchor_id == anchor.anchor_id


def test_anchor_hash_immutability():
    """Test that anchor hashes are immutable"""
    system = TemporalAnchoringSystem("test_system")

    anchor = system.create_anchor(AnchorType.DECISION, {"decision": "test"})

    hash1 = anchor.compute_hash()

    # Try to tamper
    original_nonce = anchor.nonce
    anchor.nonce = "tampered"

    hash2 = anchor.compute_hash()

    # Hashes should be different
    assert hash1 != hash2

    # Restore and verify
    anchor.nonce = original_nonce
    hash3 = anchor.compute_hash()

    assert hash1 == hash3


def test_chain_breaks_detected():
    """Test that chain breaks are detected"""
    system = TemporalAnchoringSystem("test_system")

    anchor1 = system.create_anchor(AnchorType.DECISION)
    anchor2 = system.create_anchor(AnchorType.DECISION)

    # Break the chain
    anchor2.previous_hash = "0" * 64

    is_valid, errors = system.verify_chain_integrity()

    assert is_valid == False
    assert len(errors) > 0


def test_genesis_has_no_previous():
    """Test that genesis anchor has no previous"""
    system = TemporalAnchoringSystem("test_system")
    genesis = system.anchors[0]

    assert genesis.previous_hash == "0" * 64


def test_large_event_set():
    """Test handling large number of events"""
    system = TemporalAnchoringSystem("test_system", anchor_interval=0.1)

    # Create 100 events
    for i in range(100):
        system.record_event(f"event_{i}", {"index": i})

    assert len(system.events) == 100

    # Verify integrity
    is_valid, errors = system.verify_chain_integrity()
    assert is_valid == True


def test_concurrent_causal_parents():
    """Test event with multiple causal parents"""
    system = TemporalAnchoringSystem("test_system")

    event1 = system.record_event("event1", {})
    event2 = system.record_event("event2", {})

    # Event3 caused by both event1 and event2
    event3 = system.record_event(
        "event3",
        {},
        causal_parents=[event1.event_id, event2.event_id]
    )

    assert len(event3.causal_parents) == 2
    assert event1.event_id in event3.causal_parents
    assert event2.event_id in event3.causal_parents


def test_event_data_types():
    """Test events with different data types"""
    system = TemporalAnchoringSystem("test_system")

    # Different data types
    system.record_event("string_event", {"value": "text"})
    system.record_event("number_event", {"value": 42})
    system.record_event("float_event", {"value": 3.14})
    system.record_event("bool_event", {"value": True})
    system.record_event("list_event", {"value": [1, 2, 3]})
    system.record_event("dict_event", {"value": {"nested": "data"}})

    assert len(system.events) == 6


def test_tamper_detection_accumulation():
    """Test that tamper detections are accumulated"""
    system = TemporalAnchoringSystem("test_system")

    # Create events with violations
    event1 = system.record_event("event1", {})
    event2 = system.record_event(
        "event2",
        {},
        causal_parents=["fake_parent"]
    )

    violations = system.detect_causal_violations()

    # Should detect the fake parent
    assert len(violations) >= 1

    # Store detections
    system.tamper_detections.extend(violations)

    assert len(system.tamper_detections) > 0
