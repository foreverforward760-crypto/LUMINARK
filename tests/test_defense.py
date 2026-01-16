"""
Unit tests for MycelialDefenseSystem
"""

import pytest
from mycelial_defense.defense import (
    MycelialDefenseSystem,
    DefenseMode,
    ThreatAssessment
)
from mycelial_defense.alignment import ComponentSignature


def test_defense_initialization():
    """Test defense system initialization"""
    defense = MycelialDefenseSystem("test_system")

    assert defense.system_id == "test_system"
    assert defense.mode == DefenseMode.DORMANT
    assert defense.active == False
    assert defense.detector is not None
    assert defense.mycelium is not None
    assert defense.octo is not None
    assert defense.sap is not None


def test_assess_threat_dormant():
    """Test threat assessment - dormant mode"""
    defense = MycelialDefenseSystem("test_system")

    assessment = defense.assess_threat(
        complexity=0.5,
        stability=0.8,
        tension=0.3,
        adaptability=0.7,
        coherence=0.8
    )

    assert assessment.recommended_mode == DefenseMode.DORMANT
    assert assessment.threat_level < 0.5


def test_assess_threat_octo_camouflage():
    """Test threat assessment - octo camouflage trigger"""
    defense = MycelialDefenseSystem("test_system")

    assessment = defense.assess_threat(
        complexity=0.7,
        stability=0.5,
        tension=0.85,  # High tension
        adaptability=0.6,
        coherence=0.2   # Low coherence
    )

    assert assessment.recommended_mode == DefenseMode.OCTO_CAMOUFLAGE
    assert assessment.threat_level > 0.5
    assert "High velocity" in " ".join(assessment.trigger_conditions)


def test_assess_threat_mycelial_wrap():
    """Test threat assessment - mycelial wrap trigger"""
    defense = MycelialDefenseSystem("test_system")

    assessment = defense.assess_threat(
        complexity=0.7,
        stability=0.15,  # Low stability
        tension=0.75,    # High tension
        adaptability=0.4,
        coherence=0.5
    )

    assert assessment.recommended_mode == DefenseMode.MYCELIAL_WRAP
    assert assessment.threat_level > 0.7
    assert "overload" in " ".join(assessment.trigger_conditions).lower()


def test_assess_threat_full_harrowing():
    """Test threat assessment - full harrowing trigger"""
    defense = MycelialDefenseSystem("test_system")

    assessment = defense.assess_threat(
        complexity=0.8,
        stability=0.05,   # Critical low stability
        tension=0.95,     # Critical high tension
        adaptability=0.1,
        coherence=0.15    # Critical low coherence
    )

    assert assessment.recommended_mode == DefenseMode.FULL_HARROWING
    assert assessment.threat_level >= 0.9
    assert "Critical collapse" in " ".join(assessment.trigger_conditions)


def test_execute_defense_octo():
    """Test executing octo-camouflage defense"""
    defense = MycelialDefenseSystem("test_system")

    components = [
        {"id": "comp1", "alignment_score": 0.9, "resource_usage": 0.3},
        {"id": "comp2", "alignment_score": 0.8, "resource_usage": 0.4},
        {"id": "comp3", "alignment_score": 0.3, "resource_usage": 0.7}
    ]

    assessment = defense.assess_threat(
        complexity=0.7,
        stability=0.5,
        tension=0.85,
        adaptability=0.6,
        coherence=0.2
    )

    action = defense.execute_defense(components, assessment)

    assert action.mode == DefenseMode.OCTO_CAMOUFLAGE
    assert action.success == True
    assert len(action.components_affected) > 0
    assert defense.active == True


def test_execute_defense_mycelial():
    """Test executing mycelial wrap defense"""
    defense = MycelialDefenseSystem("test_system")

    components = [
        {"id": "comp1", "alignment_score": 0.3, "resource_usage": 0.7},
        {"id": "comp2", "alignment_score": 0.4, "resource_usage": 0.8},
        {"id": "comp3", "alignment_score": 0.9, "resource_usage": 0.3}
    ]

    assessment = defense.assess_threat(
        complexity=0.7,
        stability=0.15,
        tension=0.75,
        adaptability=0.4,
        coherence=0.5
    )

    action = defense.execute_defense(components, assessment)

    assert action.mode == DefenseMode.MYCELIAL_WRAP
    assert action.success == True
    assert defense.mycelium.active == True


def test_execute_defense_full_harrowing():
    """Test executing full harrowing defense"""
    defense = MycelialDefenseSystem("test_system")

    components = [
        {"id": "comp1", "alignment_score": 0.2, "resource_usage": 0.9},
        {"id": "comp2", "alignment_score": 0.1, "resource_usage": 0.95},
        {"id": "comp3", "alignment_score": 0.8, "resource_usage": 0.4}
    ]

    assessment = defense.assess_threat(
        complexity=0.8,
        stability=0.05,
        tension=0.95,
        adaptability=0.1,
        coherence=0.15
    )

    action = defense.execute_defense(components, assessment)

    assert action.mode == DefenseMode.FULL_HARROWING
    assert action.success == True
    # Both mycelium and octo should be active
    assert defense.mycelium.active == True
    assert defense.octo.active == True
    # Should have metadata about rescue operation
    assert "rescue_rate" in action.metadata


def test_defense_history():
    """Test defense action history"""
    defense = MycelialDefenseSystem("test_system")

    components = [
        {"id": "comp1", "alignment_score": 0.9, "resource_usage": 0.3}
    ]

    # Execute multiple defenses
    for i in range(3):
        assessment = defense.assess_threat(
            complexity=0.7,
            stability=0.5,
            tension=0.85,
            adaptability=0.6,
            coherence=0.2
        )
        defense.execute_defense(components, assessment)

    assert len(defense.history) == 3
    assert all(action.mode == DefenseMode.OCTO_CAMOUFLAGE for action in defense.history)


def test_get_status():
    """Test getting system status"""
    defense = MycelialDefenseSystem("test_system")

    components = [
        {"id": "comp1", "alignment_score": 0.9, "resource_usage": 0.3}
    ]

    assessment = defense.assess_threat(
        complexity=0.7,
        stability=0.5,
        tension=0.85,
        adaptability=0.6,
        coherence=0.2
    )
    defense.execute_defense(components, assessment)

    status = defense.get_status()

    assert status["system_id"] == "test_system"
    assert status["mode"] == DefenseMode.OCTO_CAMOUFLAGE.value
    assert status["active"] == True
    assert "alignment_detector" in status
    assert "mycelial_network" in status
    assert "octo_camouflage" in status
    assert len(status["recent_actions"]) > 0


def test_reset():
    """Test resetting defense system"""
    defense = MycelialDefenseSystem("test_system")

    components = [
        {"id": "comp1", "alignment_score": 0.9, "resource_usage": 0.3}
    ]

    assessment = defense.assess_threat(
        complexity=0.7,
        stability=0.5,
        tension=0.85,
        adaptability=0.6,
        coherence=0.2
    )
    defense.execute_defense(components, assessment)

    assert defense.active == True

    defense.reset()

    assert defense.mode == DefenseMode.DORMANT
    assert defense.active == False


def test_threshold_customization():
    """Test custom thresholds"""
    defense = MycelialDefenseSystem("test_system", alignment_threshold=0.8)

    assert defense.detector.alignment_threshold == 0.8

    # Modify thresholds
    defense.thresholds["octo_camouflage"]["tension_min"] = 0.9

    assessment = defense.assess_threat(
        complexity=0.7,
        stability=0.5,
        tension=0.85,
        adaptability=0.6,
        coherence=0.2
    )

    # Should not trigger with new threshold
    assert assessment.recommended_mode != DefenseMode.OCTO_CAMOUFLAGE


def test_auto_calculate_spat():
    """Test auto-calculating SPAT from components"""
    defense = MycelialDefenseSystem("test_system")

    components = [
        {"id": "comp1", "alignment_score": 0.9, "resource_usage": 0.3},
        {"id": "comp2", "alignment_score": 0.8, "resource_usage": 0.4},
        {"id": "comp3", "alignment_score": 0.3, "resource_usage": 0.9}
    ]

    action = defense.execute_defense(components)

    assert action.spat_vectors is not None
    assert action.spat_vectors.complexity > 0
    assert action.spat_vectors.tension > 0


def test_monitoring_mode():
    """Test monitoring mode activation"""
    defense = MycelialDefenseSystem("test_system")

    components = [
        {"id": "comp1", "alignment_score": 0.9, "resource_usage": 0.3}
    ]

    assessment = defense.assess_threat(
        complexity=0.5,
        stability=0.6,
        tension=0.55,  # Slightly elevated
        adaptability=0.7,
        coherence=0.4   # Slightly low
    )

    action = defense.execute_defense(components, assessment)

    assert action.mode == DefenseMode.MONITORING
    assert defense.active == True
