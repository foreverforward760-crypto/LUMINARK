"""
Unit tests for OctoCamouflage
"""

import pytest
from mycelial_defense.octo import (
    OctoCamouflage,
    CamouflagePattern,
    CamouflageProfile
)


def test_octo_initialization():
    """Test octo-camouflage initialization"""
    octo = OctoCamouflage()

    assert len(octo.camouflaged) == 0
    assert octo.active == False
    assert octo.void_signature is not None


def test_mimic_void():
    """Test void mimicry (core innovation)"""
    octo = OctoCamouflage()

    profile = octo.mimic_void("comp1", intensity=1.0)

    assert profile.component_id == "comp1"
    assert profile.pattern == CamouflagePattern.MIMIC_VOID
    assert profile.visibility < 0.1  # Nearly invisible
    assert profile.signal_dampening > 0.9  # Heavy dampening
    assert profile.void_mimicry_strength > 0.9
    assert profile.active == True
    assert octo.active == True


def test_mimic_failure():
    """Test failure mimicry"""
    octo = OctoCamouflage()

    profile = octo.mimic_failure("comp1", intensity=0.8)

    assert profile.pattern == CamouflagePattern.MIMIC_FAILURE
    assert profile.active == True
    assert "failure_messages" in profile.metadata


def test_mimic_noise():
    """Test noise mimicry"""
    octo = OctoCamouflage()

    profile = octo.mimic_noise("comp1", intensity=0.7)

    assert profile.pattern == CamouflagePattern.MIMIC_NOISE
    assert profile.active == True


def test_adaptive_camouflage():
    """Test adaptive pattern selection"""
    octo = OctoCamouflage()

    # Scan threat -> void
    profile = octo.adaptive_camouflage("comp1", threat_type="scan")
    assert profile.pattern == CamouflagePattern.MIMIC_VOID

    # Exploit threat -> failure
    profile = octo.adaptive_camouflage("comp2", threat_type="exploit")
    assert profile.pattern == CamouflagePattern.MIMIC_FAILURE

    # DDoS threat -> noise
    profile = octo.adaptive_camouflage("comp3", threat_type="ddos")
    assert profile.pattern == CamouflagePattern.MIMIC_NOISE


def test_deception_score():
    """Test deception score calculation"""
    profile = CamouflageProfile(
        component_id="comp1",
        pattern=CamouflagePattern.MIMIC_VOID,
        visibility=0.05,
        signal_dampening=0.95,
        void_mimicry_strength=0.95,
        active=True,
        created_at=0.0
    )

    score = profile.calculate_deception()

    assert score > 0.8  # Should be highly deceptive
    assert profile.deception_score == score


def test_decloak():
    """Test decloaking component"""
    octo = OctoCamouflage()

    octo.mimic_void("comp1")
    assert "comp1" in octo.camouflaged

    success = octo.decloak("comp1")

    assert success == True
    assert "comp1" not in octo.camouflaged
    assert octo.active == False  # No more camouflaged components


def test_decloak_nonexistent():
    """Test decloaking non-existent component"""
    octo = OctoCamouflage()

    success = octo.decloak("nonexistent")

    assert success == False


def test_get_camouflage_status():
    """Test getting camouflage status"""
    octo = OctoCamouflage()

    octo.mimic_void("comp1", intensity=0.9)

    status = octo.get_camouflage_status("comp1")

    assert status is not None
    assert status["component_id"] == "comp1"
    assert status["pattern"] == CamouflagePattern.MIMIC_VOID.value
    assert status["active"] == True


def test_get_all_statuses():
    """Test getting all statuses"""
    octo = OctoCamouflage()

    octo.mimic_void("comp1")
    octo.mimic_failure("comp2")
    octo.mimic_noise("comp3")

    statuses = octo.get_all_statuses()

    assert len(statuses) == 3
    assert "comp1" in statuses
    assert "comp2" in statuses
    assert "comp3" in statuses


def test_mass_cloak():
    """Test mass cloaking"""
    octo = OctoCamouflage()

    components = ["comp1", "comp2", "comp3"]

    profiles = octo.mass_cloak(
        components,
        pattern=CamouflagePattern.MIMIC_VOID,
        intensity=0.95
    )

    assert len(profiles) == 3
    assert all(p.pattern == CamouflagePattern.MIMIC_VOID for p in profiles.values())
    assert len(octo.camouflaged) == 3


def test_mass_decloak():
    """Test mass decloaking"""
    octo = OctoCamouflage()

    components = ["comp1", "comp2", "comp3"]
    octo.mass_cloak(components)

    results = octo.mass_decloak(components)

    assert len(results) == 3
    assert all(results.values())
    assert len(octo.camouflaged) == 0


def test_adjust_intensity():
    """Test adjusting camouflage intensity"""
    octo = OctoCamouflage()

    profile = octo.mimic_void("comp1", intensity=0.5)
    initial_visibility = profile.visibility

    octo.adjust_intensity("comp1", 1.0)

    assert profile.visibility < initial_visibility
    assert profile.metadata["intensity"] == 1.0


def test_system_status():
    """Test system status"""
    octo = OctoCamouflage()

    octo.mimic_void("comp1")
    octo.mimic_void("comp2")
    octo.mimic_failure("comp3")

    status = octo.get_system_status()

    assert status["active"] == True
    assert status["total_camouflaged"] == 3
    assert status["average_deception"] > 0
    assert "pattern_distribution" in status


def test_intensity_scaling():
    """Test intensity affects visibility"""
    octo = OctoCamouflage()

    low_intensity = octo.mimic_void("comp1", intensity=0.3)
    high_intensity = octo.mimic_void("comp2", intensity=1.0)

    assert high_intensity.visibility < low_intensity.visibility
    assert high_intensity.signal_dampening > low_intensity.signal_dampening


def test_pattern_library():
    """Test pattern library initialization"""
    octo = OctoCamouflage()

    assert len(octo.patterns_library) == 5
    assert CamouflagePattern.MIMIC_VOID in octo.patterns_library
    assert CamouflagePattern.MIMIC_FAILURE in octo.patterns_library
    assert CamouflagePattern.MIMIC_NOISE in octo.patterns_library


def test_void_signature():
    """Test void signature properties"""
    octo = OctoCamouflage()

    sig = octo.void_signature

    assert sig.emptiness_level > 0.9
    assert sig.noise_floor < 0.1
    assert sig.response_delay > 5.0
    assert sig.error_rate > 0.5
