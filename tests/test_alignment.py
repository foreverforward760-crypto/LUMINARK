"""
Unit tests for AlignmentDetector
"""

import pytest
import time
from mycelial_defense.alignment import (
    AlignmentDetector,
    AlignmentStatus,
    ComponentSignature
)


def test_signature_creation():
    """Test component signature creation"""
    sig = ComponentSignature(
        component_id="test_comp",
        expected_behavior="process_requests",
        expected_output_pattern="success",
        expected_resource_usage=0.5
    )

    assert sig.component_id == "test_comp"
    assert sig.signature_hash != ""
    assert len(sig.signature_hash) == 16


def test_detector_initialization():
    """Test detector initialization"""
    detector = AlignmentDetector(alignment_threshold=0.8)

    assert detector.alignment_threshold == 0.8
    assert len(detector.known_signatures) == 0
    assert len(detector.history) == 0


def test_register_signature():
    """Test registering component signatures"""
    detector = AlignmentDetector()
    sig = ComponentSignature(
        component_id="comp1",
        expected_behavior="normal",
        expected_output_pattern="OK",
        expected_resource_usage=0.3
    )

    detector.register_signature(sig)

    assert "comp1" in detector.known_signatures
    assert detector.known_signatures["comp1"] == sig


def test_detect_aligned_component():
    """Test detection of aligned component"""
    detector = AlignmentDetector(alignment_threshold=0.7)

    sig = ComponentSignature(
        component_id="comp1",
        expected_behavior="process_data",
        expected_output_pattern="success",
        expected_resource_usage=0.5
    )
    detector.register_signature(sig)

    result = detector.detect_alignment(
        component_id="comp1",
        current_behavior="process_data",
        current_output="success",
        current_resources=0.5
    )

    assert result.status == AlignmentStatus.ALIGNED
    assert result.alignment_score >= 0.7
    assert result.confidence > 0.5


def test_detect_misaligned_component():
    """Test detection of misaligned component"""
    detector = AlignmentDetector(alignment_threshold=0.7)

    sig = ComponentSignature(
        component_id="comp1",
        expected_behavior="process_data",
        expected_output_pattern="success",
        expected_resource_usage=0.5
    )
    detector.register_signature(sig)

    result = detector.detect_alignment(
        component_id="comp1",
        current_behavior="malicious_behavior",
        current_output="error: injection detected",
        current_resources=0.9
    )

    assert result.status == AlignmentStatus.MISALIGNED
    assert result.alignment_score < 0.7
    assert len(result.anomalies) > 0


def test_detect_unknown_component():
    """Test detection of unknown component"""
    detector = AlignmentDetector()

    result = detector.detect_alignment(
        component_id="unknown",
        current_behavior="anything",
        current_output="anything",
        current_resources=0.5
    )

    assert result.status == AlignmentStatus.UNKNOWN
    assert result.alignment_score == 0.0


def test_anomaly_detection():
    """Test anomaly detection in output"""
    detector = AlignmentDetector()

    sig = ComponentSignature(
        component_id="comp1",
        expected_behavior="normal",
        expected_output_pattern="OK",
        expected_resource_usage=0.5
    )
    detector.register_signature(sig)

    result = detector.detect_alignment(
        component_id="comp1",
        current_behavior="normal",
        current_output="ERROR: Unauthorized access attempt",
        current_resources=0.5
    )

    assert len(result.anomalies) > 0
    assert any("error" in a.lower() or "unauthorized" in a.lower() for a in result.anomalies)


def test_component_health():
    """Test component health tracking"""
    detector = AlignmentDetector(alignment_threshold=0.7)

    sig = ComponentSignature(
        component_id="comp1",
        expected_behavior="normal",
        expected_output_pattern="OK",
        expected_resource_usage=0.5
    )
    detector.register_signature(sig)

    # Generate some history
    for i in range(10):
        detector.detect_alignment(
            component_id="comp1",
            current_behavior="normal",
            current_output="OK",
            current_resources=0.5
        )

    health = detector.get_component_health("comp1")

    assert health["component_id"] == "comp1"
    assert health["aligned_percentage"] > 0.8
    assert health["health_status"] == "healthy"


def test_all_health():
    """Test getting health for all components"""
    detector = AlignmentDetector()

    for i in range(3):
        sig = ComponentSignature(
            component_id=f"comp{i}",
            expected_behavior="normal",
            expected_output_pattern="OK",
            expected_resource_usage=0.5
        )
        detector.register_signature(sig)

        detector.detect_alignment(
            component_id=f"comp{i}",
            current_behavior="normal",
            current_output="OK",
            current_resources=0.5
        )

    all_health = detector.get_all_health()

    assert len(all_health) == 3
    assert all(h["health_status"] == "healthy" for h in all_health.values())


def test_resource_similarity():
    """Test resource usage similarity calculation"""
    detector = AlignmentDetector()

    # Test exact match
    assert detector._calculate_resource_similarity(0.5, 0.5) == 1.0

    # Test within tolerance
    assert detector._calculate_resource_similarity(0.5, 0.6) == 1.0

    # Test outside tolerance
    score = detector._calculate_resource_similarity(0.5, 0.9)
    assert score < 1.0


def test_pattern_matching():
    """Test pattern matching"""
    detector = AlignmentDetector()

    # Exact match
    assert detector._calculate_pattern_match("success", "success") == 1.0

    # Regex match
    assert detector._calculate_pattern_match("^OK.*", "OK: processed") > 0.9

    # No match
    score = detector._calculate_pattern_match("success", "failure")
    assert score < 0.5
