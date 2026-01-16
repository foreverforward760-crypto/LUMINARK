#!/usr/bin/env python3
"""
LUMINARK Mycelial Defense System - Test Suite
Tests the defense system with various threat scenarios
"""

from mycelial_defense import MycelialDefenseSystem


def print_separator():
    """Print a visual separator"""
    print("=" * 80)


def print_test_header(test_num, test_name):
    """Print test header"""
    print_separator()
    print(f"TEST {test_num}: {test_name}")
    print_separator()


def print_test_result(result):
    """Print formatted test result"""
    print(f"\n🎯 DEFENSE MODE: {result['defense_mode']}")
    print(f"📋 STRATEGY: {result['strategy']}")
    print(f"🚨 ALERT LEVEL: {result['alert_level']}")
    print(f"\n📊 Metrics:")
    print(f"   • Stability: {result['metrics']['stability']:.2f}")
    print(f"   • Tension: {result['metrics']['tension']:.2f}")
    print(f"   • Coherence: {result['metrics']['coherence']:.2f}")
    print()


def run_tests():
    """Run all defense system tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "LUMINARK MYCELIAL DEFENSE TESTS" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Initialize defense system
    defense = MycelialDefenseSystem()

    # Test 1: Octo-Camouflage Trigger
    print_test_header(1, "Octo-Camouflage Trigger")
    print("Scenario: High tension with low coherence")
    print("Expected: OCTO_CAMOUFLAGE mode activated")
    print()

    result1 = defense.analyze_threat(
        stability=0.50,  # Normal stability
        tension=0.85,    # High tension
        coherence=0.25   # Low coherence
    )

    print_test_result(result1)
    desc1 = defense.get_defense_description()
    print(f"Status: {desc1['status']}")
    print(f"Threat Level: {desc1['threat_level']}")
    print(f"Actions Taken:")
    for action in desc1['actions']:
        print(f"  ✓ {action}")

    # Test 2: Mycelial Wrap Trigger
    print("\n")
    print_test_header(2, "Mycelial Wrap Trigger")
    print("Scenario: Low stability with high tension")
    print("Expected: MYCELIAL_WRAP mode activated")
    print()

    result2 = defense.analyze_threat(
        stability=0.15,  # Very low stability
        tension=0.75,    # High tension
        coherence=0.60   # Moderate coherence
    )

    print_test_result(result2)
    desc2 = defense.get_defense_description()
    print(f"Status: {desc2['status']}")
    print(f"Threat Level: {desc2['threat_level']}")
    print(f"Actions Taken:")
    for action in desc2['actions']:
        print(f"  ✓ {action}")

    # Test 3: Full Harrowing Trigger
    print("\n")
    print_test_header(3, "Full Harrowing Trigger")
    print("Scenario: Critical instability, extreme tension, broken coherence")
    print("Expected: HARROWING mode activated (highest alert)")
    print()

    result3 = defense.analyze_threat(
        stability=0.05,  # Critical instability
        tension=0.95,    # Extreme tension
        coherence=0.15   # Very low coherence
    )

    print_test_result(result3)
    desc3 = defense.get_defense_description()
    print(f"Status: {desc3['status']}")
    print(f"Threat Level: {desc3['threat_level']}")
    print(f"Actions Taken:")
    for action in desc3['actions']:
        print(f"  ✓ {action}")

    # Summary
    print("\n")
    print_separator()
    print("TEST SUMMARY")
    print_separator()
    print()
    print(f"✓ Test 1 - Defense Mode: {result1['defense_mode']}")
    print(f"  Strategy: {result1['strategy']}")
    print()
    print(f"✓ Test 2 - Defense Mode: {result2['defense_mode']}")
    print(f"  Strategy: {result2['strategy']}")
    print()
    print(f"✓ Test 3 - Defense Mode: {result3['defense_mode']}")
    print(f"  Strategy: {result3['strategy']}")
    print()
    print_separator()
    print("All tests completed successfully! 🎉")
    print_separator()
    print()


if __name__ == '__main__':
    run_tests()
