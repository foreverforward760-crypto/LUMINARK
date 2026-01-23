"""
Unit tests for SAPCalculator
"""

import pytest
from mycelial_defense.sap import SAPCalculator, SPATVectors


def test_spat_vectors_creation():
    """Test SPAT vectors creation"""
    vectors = SPATVectors(
        complexity=0.7,
        stability=0.6,
        tension=0.5,
        adaptability=0.8,
        coherence=0.9,
        timestamp=0.0
    )

    assert vectors.complexity == 0.7
    assert vectors.stability == 0.6
    assert vectors.tension == 0.5
    assert vectors.adaptability == 0.8
    assert vectors.coherence == 0.9


def test_spat_vectors_to_dict():
    """Test converting SPAT vectors to dictionary"""
    vectors = SPATVectors(
        complexity=0.7,
        stability=0.6,
        tension=0.5,
        adaptability=0.8,
        coherence=0.9,
        timestamp=0.0
    )

    d = vectors.to_dict()

    assert d["complexity"] == 0.7
    assert d["stability"] == 0.6
    assert "timestamp" in d


def test_calculator_initialization():
    """Test SAP calculator initialization"""
    calc = SAPCalculator()

    assert len(calc.history) == 0
    assert calc.baseline is None


def test_calculate_from_metrics():
    """Test calculating from direct metrics"""
    calc = SAPCalculator()

    vectors = calc.calculate_from_metrics(
        complexity=0.7,
        stability=0.6,
        tension=0.5,
        adaptability=0.8,
        coherence=0.9
    )

    assert vectors.complexity == 0.7
    assert vectors.stability == 0.6
    assert len(calc.history) == 1


def test_calculate_from_components():
    """Test calculating from component state"""
    calc = SAPCalculator()

    components = [
        {"id": "comp1"},
        {"id": "comp2"},
        {"id": "comp3"}
    ]

    alignment_scores = {
        "comp1": 0.9,
        "comp2": 0.8,
        "comp3": 0.7
    }

    resource_usage = {
        "comp1": 0.3,
        "comp2": 0.4,
        "comp3": 0.5
    }

    vectors = calc.calculate_from_components(
        components,
        alignment_scores,
        resource_usage
    )

    assert 0.0 <= vectors.complexity <= 1.0
    assert 0.0 <= vectors.stability <= 1.0
    assert 0.0 <= vectors.tension <= 1.0
    assert 0.0 <= vectors.adaptability <= 1.0
    assert 0.0 <= vectors.coherence <= 1.0
    assert len(calc.history) == 1


def test_complexity_calculation():
    """Test complexity calculation"""
    calc = SAPCalculator()

    # More components = higher complexity
    few_components = [{"id": f"comp{i}"} for i in range(5)]
    many_components = [{"id": f"comp{i}"} for i in range(50)]

    alignment = {c["id"]: 0.9 for c in many_components}
    resources = {c["id"]: 0.3 for c in many_components}

    v1 = calc.calculate_from_components(few_components, alignment, resources)
    v2 = calc.calculate_from_components(many_components, alignment, resources)

    assert v2.complexity > v1.complexity


def test_stability_calculation():
    """Test stability calculation"""
    calc = SAPCalculator()

    components = [{"id": f"comp{i}"} for i in range(10)]

    # Consistent alignment = high stability
    consistent_alignment = {c["id"]: 0.9 for c in components}
    # Varying alignment = low stability
    varying_alignment = {c["id"]: 0.1 + i * 0.09 for i, c in enumerate(components)}

    resources = {c["id"]: 0.3 for c in components}

    v1 = calc.calculate_from_components(components, consistent_alignment, resources)
    v2 = calc.calculate_from_components(components, varying_alignment, resources)

    assert v1.stability > v2.stability


def test_tension_calculation():
    """Test tension calculation"""
    calc = SAPCalculator()

    components = [{"id": f"comp{i}"} for i in range(10)]

    # Low misalignment + low resources = low tension
    good_alignment = {c["id"]: 0.9 for c in components}
    low_resources = {c["id"]: 0.2 for c in components}

    # High misalignment + high resources = high tension
    bad_alignment = {c["id"]: 0.2 for c in components}
    high_resources = {c["id"]: 0.9 for c in components}

    v1 = calc.calculate_from_components(components, good_alignment, low_resources)
    v2 = calc.calculate_from_components(components, bad_alignment, high_resources)

    assert v2.tension > v1.tension


def test_adaptability_calculation():
    """Test adaptability calculation"""
    calc = SAPCalculator()

    components = [{"id": f"comp{i}"} for i in range(10)]
    alignment = {c["id"]: 0.9 for c in components}

    # Low resource usage = high adaptability (headroom)
    low_resources = {c["id"]: 0.2 for c in components}
    # High resource usage = low adaptability
    high_resources = {c["id"]: 0.9 for c in components}

    v1 = calc.calculate_from_components(components, alignment, low_resources)
    v2 = calc.calculate_from_components(components, alignment, high_resources)

    assert v1.adaptability > v2.adaptability


def test_coherence_calculation():
    """Test coherence calculation"""
    calc = SAPCalculator()

    components = [{"id": f"comp{i}"} for i in range(10)]
    resources = {c["id"]: 0.3 for c in components}

    # High alignment = high coherence
    high_alignment = {c["id"]: 0.9 for c in components}
    # Low alignment = low coherence
    low_alignment = {c["id"]: 0.2 for c in components}

    v1 = calc.calculate_from_components(components, high_alignment, resources)
    v2 = calc.calculate_from_components(components, low_alignment, resources)

    assert v1.coherence > v2.coherence


def test_set_baseline():
    """Test setting baseline"""
    calc = SAPCalculator()

    vectors = SPATVectors(
        complexity=0.5,
        stability=0.8,
        tension=0.3,
        adaptability=0.7,
        coherence=0.9,
        timestamp=0.0
    )

    calc.set_baseline(vectors)

    assert calc.baseline == vectors


def test_calculate_drift():
    """Test drift calculation"""
    calc = SAPCalculator()

    baseline = SPATVectors(
        complexity=0.5,
        stability=0.8,
        tension=0.3,
        adaptability=0.7,
        coherence=0.9,
        timestamp=0.0
    )
    calc.set_baseline(baseline)

    current = SPATVectors(
        complexity=0.7,
        stability=0.6,
        tension=0.5,
        adaptability=0.5,
        coherence=0.7,
        timestamp=1.0
    )

    drift = calc.calculate_drift(current)

    assert drift["complexity"] == pytest.approx(0.2)
    assert drift["stability"] == pytest.approx(0.2)
    assert drift["tension"] == pytest.approx(0.2)


def test_get_trend():
    """Test trend detection"""
    calc = SAPCalculator()

    # Create increasing tension trend
    for i in range(10):
        calc.calculate_from_metrics(
            complexity=0.5,
            stability=0.5,
            tension=0.3 + i * 0.05,  # Increasing
            adaptability=0.5,
            coherence=0.5
        )

    trend = calc.get_trend("tension", window=10)

    assert trend == "increasing"


def test_get_analysis():
    """Test SPAT analysis"""
    calc = SAPCalculator()

    # Healthy system
    healthy = SPATVectors(
        complexity=0.5,
        stability=0.8,
        tension=0.3,
        adaptability=0.7,
        coherence=0.9,
        timestamp=0.0
    )

    analysis = calc.get_analysis(healthy)

    assert analysis["health_status"] == "healthy"
    assert len(analysis["warnings"]) == 0

    # Critical system
    critical = SPATVectors(
        complexity=0.9,
        stability=0.1,
        tension=0.9,
        adaptability=0.1,
        coherence=0.2,
        timestamp=0.0
    )

    analysis = calc.get_analysis(critical)

    assert analysis["health_status"] == "critical"
    assert len(analysis["warnings"]) > 0
    assert len(analysis["recommendations"]) > 0


def test_empty_components():
    """Test handling empty component list"""
    calc = SAPCalculator()

    vectors = calc.calculate_from_components([], {}, {})

    assert vectors.complexity == 0.0
    assert vectors.stability == 1.0
    assert vectors.coherence == 1.0


def test_connection_map():
    """Test complexity with connection map"""
    calc = SAPCalculator()

    components = [{"id": f"comp{i}"} for i in range(5)]
    alignment = {c["id"]: 0.9 for c in components}
    resources = {c["id"]: 0.3 for c in components}

    # Fully connected
    connections = {c["id"]: [o["id"] for o in components if o["id"] != c["id"]] for c in components}

    vectors = calc.calculate_from_components(
        components,
        alignment,
        resources,
        connection_map=connections
    )

    assert vectors.complexity > 0.0
