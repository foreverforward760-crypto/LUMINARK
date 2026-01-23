"""
Interactive Demo - Mycelial Defense System

Demonstrates all three defense modes:
1. Octo-Camouflage
2. Mycelial Wrap
3. Full Harrowing
"""

import time
import random

from mycelial_defense import MycelialDefenseSystem, DefenseMode
from mycelial_defense.utils import generate_mock_components, simulate_attack


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_status(defense: MycelialDefenseSystem):
    """Print current defense status"""
    status = defense.get_status()
    print(f"Mode: {status['mode'].upper()}")
    print(f"Active: {status['active']}")
    print(f"Mycelial Zones: {status['mycelial_network']['total_zones']}")
    print(f"Camouflaged Components: {status['octo_camouflage']['total_camouflaged']}")


def demo_octo_camouflage():
    """Demonstrate Octo-Camouflage defense"""
    print_header("DEMO 1: OCTO-CAMOUFLAGE - Weaponized Emptiness")

    print("Scenario: High-speed system with lost direction")
    print("Trigger: High Tension + Low Coherence")
    print()

    defense = MycelialDefenseSystem("octo_demo")
    components = generate_mock_components(15, aligned_ratio=0.8)

    print("Initial State:")
    print(f"  Components: {len(components)}")
    print(f"  Aligned: {sum(1 for c in components if c['alignment_score'] >= 0.7)}")
    print()

    # Trigger octo-camouflage
    print("Assessing threat...")
    assessment = defense.assess_threat(
        complexity=0.7,
        stability=0.5,
        tension=0.85,  # High tension
        adaptability=0.6,
        coherence=0.25  # Low coherence
    )

    print(f"Threat Level: {assessment.threat_level:.2f}")
    print(f"Recommended Mode: {assessment.recommended_mode.value.upper()}")
    print()

    # Execute defense
    print("Executing defense...")
    action = defense.execute_defense(components, assessment)

    print()
    print("🐙 OCTO-CAMOUFLAGE ACTIVATED!")
    print(f"  Camouflaged: {len(action.components_affected)} components")
    print(f"  Avg Deception: {action.metadata.get('avg_deception', 0):.2f}")
    print(f"  Pattern: MIMIC_VOID (Stage 0)")
    print()
    print("Result: Healthy components now appear as pure emptiness.")
    print("        Attacks pass through harmlessly - nothing to attack!")
    print()

    print_status(defense)


def demo_mycelial_wrap():
    """Demonstrate Mycelial Wrap defense"""
    print_header("DEMO 2: MYCELIAL WRAP - Fungal Containment")

    print("Scenario: System collapse under pressure")
    print("Trigger: Low Stability + High Tension")
    print()

    defense = MycelialDefenseSystem("mycelial_demo")
    components = generate_mock_components(20, aligned_ratio=0.5)

    # Simulate attack
    components = simulate_attack(components, severity=0.6)

    print("Initial State:")
    print(f"  Components: {len(components)}")
    print(f"  Aligned: {sum(1 for c in components if c['alignment_score'] >= 0.7)}")
    print(f"  Misaligned: {sum(1 for c in components if c['alignment_score'] < 0.7)}")
    print()

    # Trigger mycelial wrap
    print("Assessing threat...")
    assessment = defense.assess_threat(
        complexity=0.7,
        stability=0.15,  # Low stability
        tension=0.8,     # High tension
        adaptability=0.4,
        coherence=0.5
    )

    print(f"Threat Level: {assessment.threat_level:.2f}")
    print(f"Recommended Mode: {assessment.recommended_mode.value.upper()}")
    print()

    # Execute defense
    print("Executing defense...")
    action = defense.execute_defense(components, assessment)

    print()
    print("🍄 MYCELIAL WRAP ACTIVATED!")
    print(f"  Zones Created: {action.metadata.get('zones_created', 0)}")
    print(f"  Walls Built: {action.metadata.get('walls_created', 0)}")
    print(f"  Contained: {action.metadata.get('contained_count', 0)} components")
    print()
    print("Result: Misaligned components surrounded by containment walls.")
    print("        Corruption isolated, prevented from spreading!")
    print()

    print_status(defense)


def demo_full_harrowing():
    """Demonstrate Full Harrowing defense"""
    print_header("DEMO 3: FULL HARROWING - Critical Rescue Operation")

    print("Scenario: Total system collapse imminent")
    print("Trigger: Critical Low Stability + Low Coherence + High Tension")
    print()

    defense = MycelialDefenseSystem("harrowing_demo")
    components = generate_mock_components(25, aligned_ratio=0.3)

    # Severe attack
    components = simulate_attack(components, severity=0.9)

    print("Initial State:")
    print(f"  Components: {len(components)}")
    print(f"  Aligned: {sum(1 for c in components if c['alignment_score'] >= 0.7)}")
    print(f"  Misaligned: {sum(1 for c in components if c['alignment_score'] < 0.7)}")
    print()

    # Trigger full harrowing
    print("Assessing threat...")
    assessment = defense.assess_threat(
        complexity=0.8,
        stability=0.05,   # Critical low
        tension=0.95,     # Critical high
        adaptability=0.1,
        coherence=0.15    # Critical low
    )

    print(f"Threat Level: {assessment.threat_level:.2f}")
    print(f"Recommended Mode: {assessment.recommended_mode.value.upper()}")
    print()

    # Execute defense
    print("Executing CRITICAL rescue operation...")
    action = defense.execute_defense(components, assessment)

    print()
    print("⚡ FULL HARROWING ACTIVATED!")
    print()
    print("Operation Steps:")
    print(f"  1. Zones Surrounded: {action.metadata.get('zones_surrounded', 0)}")
    print(f"  2. Components Camouflaged: {action.metadata.get('components_camouflaged', 0)}")
    print(f"  3. Pathways Created: {action.metadata.get('pathways_created', 0)}")
    print(f"  4. Components Extracted: {action.metadata.get('components_extracted', 0)}")
    print()
    print(f"  Rescue Rate: {action.metadata.get('rescue_rate', 0):.1%}")
    print()
    print("Result: Corruption contained, healthy components camouflaged,")
    print("        extraction pathways created, and safe components rescued!")
    print()

    print_status(defense)


def demo_progression():
    """Demonstrate defense mode progression"""
    print_header("DEMO 4: DEFENSE MODE PROGRESSION")

    print("Watching system degrade from healthy to critical...")
    print()

    defense = MycelialDefenseSystem("progression_demo")
    components = generate_mock_components(20)

    scenarios = [
        {
            "name": "Healthy System",
            "complexity": 0.5,
            "stability": 0.8,
            "tension": 0.3,
            "adaptability": 0.7,
            "coherence": 0.9,
            "attack": 0.0
        },
        {
            "name": "Elevated Tension",
            "complexity": 0.6,
            "stability": 0.7,
            "tension": 0.6,
            "adaptability": 0.6,
            "coherence": 0.7,
            "attack": 0.2
        },
        {
            "name": "High Velocity, Lost Direction",
            "complexity": 0.7,
            "stability": 0.5,
            "tension": 0.85,
            "adaptability": 0.5,
            "coherence": 0.25,
            "attack": 0.4
        },
        {
            "name": "System Overload",
            "complexity": 0.8,
            "stability": 0.15,
            "tension": 0.8,
            "adaptability": 0.3,
            "coherence": 0.4,
            "attack": 0.6
        },
        {
            "name": "Critical Collapse",
            "complexity": 0.9,
            "stability": 0.05,
            "tension": 0.95,
            "adaptability": 0.1,
            "coherence": 0.15,
            "attack": 0.9
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print("-" * 50)

        # Simulate attack
        if scenario['attack'] > 0:
            components = simulate_attack(components, severity=scenario['attack'])

        # Assess threat
        assessment = defense.assess_threat(
            complexity=scenario['complexity'],
            stability=scenario['stability'],
            tension=scenario['tension'],
            adaptability=scenario['adaptability'],
            coherence=scenario['coherence']
        )

        # Execute defense
        action = defense.execute_defense(components, assessment)

        # Display results
        print(f"  Threat Level: {assessment.threat_level:.2f}")
        print(f"  Defense Mode: {action.mode.value.upper()}")

        if action.mode == DefenseMode.OCTO_CAMOUFLAGE:
            print(f"  🐙 Camouflaged: {len(action.components_affected)}")
        elif action.mode == DefenseMode.MYCELIAL_WRAP:
            print(f"  🍄 Contained: {action.metadata.get('contained_count', 0)}")
        elif action.mode == DefenseMode.FULL_HARROWING:
            print(f"  ⚡ Rescued: {action.metadata.get('components_extracted', 0)}")

        time.sleep(1)  # Pause for effect

    print()
    print("=" * 70)
    print("PROGRESSION COMPLETE")
    print(f"Total Defense Actions: {len(defense.history)}")
    print("=" * 70)


if __name__ == "__main__":
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  MYCELIAL DEFENSE SYSTEM - Interactive Demo".center(68) + "║")
    print("║" + "  Bio-Inspired Active Defense for AI Systems".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    try:
        demo_octo_camouflage()
        time.sleep(2)

        demo_mycelial_wrap()
        time.sleep(2)

        demo_full_harrowing()
        time.sleep(2)

        demo_progression()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted")

    print()
    print("Demo complete! 🎉")
    print()
