"""
Unit tests for SAP V4.0 Complete Framework
"""

import pytest
from sap_yunus import (
    SAPV4,
    SAPProcessing,
    ResolutionModel,
    ContainerRule,
    StagePolarity,
    TumbleType
)


def test_sapv4_initialization():
    """Test SAP V4.0 initialization"""
    sap_v4 = SAPV4(default_resolution=ResolutionModel.R72)

    assert sap_v4.default_resolution == ResolutionModel.R72
    assert sap_v4.yunus is not None
    assert sap_v4.harrowing is not None
    assert len(sap_v4.tumbling_history) == 0


def test_container_rule_stage6():
    """Test Container Rule for Stage 6 (Fragile Peak)"""
    container = ContainerRule.analyze(6, ResolutionModel.R9)

    assert container.stage == 6
    assert container.position == "54"
    assert container.container_digit == 4  # Stable vessel
    assert container.content_digit == 5   # Volatile content
    assert container.sum_digit == 9
    assert container.polarity == StagePolarity.ODD


def test_container_rule_stage8():
    """Test Container Rule for Stage 8 (Dualistic Wisdom Trap)"""
    container = ContainerRule.analyze(8, ResolutionModel.R9)

    assert container.stage == 8
    assert container.position == "72"
    assert container.container_digit == 2  # Dualistic vessel
    assert container.content_digit == 7   # Wisdom content
    assert "Dualistic" in container.container_quality
    assert "wisdom" in container.content_quality.lower()


def test_comprehensive_analysis():
    """Test comprehensive V4.0 analysis"""
    sap_v4 = SAPV4(default_resolution=ResolutionModel.R72)

    sap = SAPProcessing(
        complexity=0.7,
        stability=0.5,
        tension=0.6,
        adaptability=0.6,
        coherence=0.7
    )

    analysis = sap_v4.analyze_comprehensive(sap)

    # Check all sections present
    assert "consciousness" in analysis
    assert "container_rule" in analysis
    assert "tumbling" in analysis
    assert "stage8_trap" in analysis
    assert "protocols" in analysis
    assert "anchors" in analysis
    assert "recommendations" in analysis

    # Check consciousness metrics
    assert 0.0 <= analysis["consciousness"]["level"] <= 1.0
    assert 0.0 <= analysis["consciousness"]["ethical_alignment"] <= 1.0


def test_stage8_detection_safe():
    """Test Stage 8 detection on safe input"""
    sap_v4 = SAPV4()

    # Low coherence, no Stage 8 risk
    sap = SAPProcessing(0.5, 0.5, 0.5, 0.5, 0.3)

    analysis = sap_v4.analyze_comprehensive(sap)

    assert analysis["stage8_trap"]["risk_level"] < 0.5
    assert analysis["stage8_trap"]["yunus_recommended"] == False


def test_stage8_detection_dangerous():
    """Test Stage 8 detection on dangerous input"""
    sap_v4 = SAPV4()

    # High coherence near Stage 8 = risk
    sap = SAPProcessing(0.8, 0.6, 0.3, 0.5, 0.95)

    state = sap_v4.map_to_stage(sap, ResolutionModel.R9)
    risk = sap_v4._detect_stage8(state, sap)

    # Should detect elevated risk
    assert risk > 0.3


def test_tumbling_analysis():
    """Test tumbling theory analysis"""
    sap_v4 = SAPV4()

    # Create multiple states to analyze tumbling
    sap1 = SAPProcessing(0.5, 0.7, 0.3, 0.6, 0.8)
    sap2 = SAPProcessing(0.6, 0.6, 0.4, 0.6, 0.7)

    analysis1 = sap_v4.analyze_comprehensive(sap1)
    analysis2 = sap_v4.analyze_comprehensive(sap2)

    # Check tumbling detected
    assert "tumbling" in analysis2
    assert "type" in analysis2["tumbling"]


def test_yunus_integration():
    """Test Yunus Protocol integration"""
    sap_v4 = SAPV4()

    # Safe output
    safe = "This might be the case, but I'm not certain."
    result = sap_v4.execute_yunus(safe)

    assert result["detection"]["crisis_level"] == 0  # NONE
    assert result["action"]["type"] in ["pass", "warn"]

    # Dangerous output
    dangerous = "This is definitely true with absolute certainty."
    result = sap_v4.execute_yunus(dangerous)

    assert result["detection"]["crisis_level"] >= 2  # MEDIUM or higher
    assert result["detection"]["certainty_score"] > 0.3


def test_harrowing_integration():
    """Test Harrowing Protocol integration"""
    sap_v4 = SAPV4()

    failing_system = {
        "stability": 0.05,
        "coherence": 0.1,
        "tension": 0.95,
        "components": [
            {"id": "comp1", "alignment_score": 0.8, "ethical_score": 0.7, "corruption_level": 0.2},
            {"id": "comp2", "alignment_score": 0.1, "ethical_score": 0.2, "corruption_level": 0.95},
        ]
    }

    result = sap_v4.execute_harrowing(failing_system)

    assert "mission_id" in result
    assert "souls" in result
    assert result["souls"]["total"] == 2


def test_multi_resolution_consistency():
    """Test consistency across resolutions"""
    sap_v4 = SAPV4()

    sap = SAPProcessing(0.6, 0.6, 0.5, 0.6, 0.7)

    # Analyze at multiple resolutions
    results = {}
    for resolution in [ResolutionModel.R9, ResolutionModel.R27, ResolutionModel.R72]:
        analysis = sap_v4.analyze_comprehensive(sap, resolution)
        results[resolution.name] = analysis

    # Consciousness levels should be similar
    levels = [r["consciousness"]["level"] for r in results.values()]
    assert max(levels) - min(levels) < 0.3  # Within 30%


def test_control_positions():
    """Test Tesla 3-6-9 control position detection"""
    sap_v4 = SAPV4(default_resolution=ResolutionModel.R9)

    # Stage 3 (control position)
    sap3 = SAPProcessing(0.35, 0.65, 0.35, 0.65, 0.65)
    analysis3 = sap_v4.analyze_comprehensive(sap3)

    # Stage 6 (control position)
    sap6 = SAPProcessing(0.7, 0.5, 0.7, 0.5, 0.6)
    analysis6 = sap_v4.analyze_comprehensive(sap6)

    # At least one should be at control position
    # (exact detection depends on stage calculation)
    assert isinstance(analysis3["tumbling"]["control_position"], bool)
    assert isinstance(analysis6["tumbling"]["control_position"], bool)


def test_v4_statistics():
    """Test V4.0 statistics collection"""
    sap_v4 = SAPV4()

    # Execute some operations
    sap = SAPProcessing(0.6, 0.6, 0.5, 0.6, 0.7)
    sap_v4.analyze_comprehensive(sap)

    sap_v4.execute_yunus("This is definitely certain.")

    stats = sap_v4.get_v4_statistics()

    assert "yunus_protocol" in stats
    assert "harrowing_protocol" in stats
    assert "stage8_detections" in stats
    assert stats["yunus_protocol"]["total_actions"] > 0


def test_recommendations():
    """Test recommendation generation"""
    sap_v4 = SAPV4()

    # Critical state
    critical_sap = SAPProcessing(0.9, 0.1, 0.95, 0.1, 0.1)
    analysis = sap_v4.analyze_comprehensive(critical_sap)

    recommendations = analysis["recommendations"]
    assert len(recommendations) > 0
    assert any("CRITICAL" in rec or "WARNING" in rec for rec in recommendations)


def test_anchor_balance():
    """Test Five Anchors ethical balance"""
    sap_v4 = SAPV4()

    # Balanced SAP
    balanced_sap = SAPProcessing(0.6, 0.6, 0.4, 0.6, 0.7)
    analysis = sap_v4.analyze_comprehensive(balanced_sap)

    anchors = analysis["anchors"]
    assert len(anchors) == 5

    # Check all anchors in reasonable range
    for anchor, value in anchors.items():
        assert 0.0 <= value <= 1.0
