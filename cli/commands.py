"""
CLI commands for Mycelial Defense System

Commands:
- init: Initialize defense system
- register-component: Register component signature
- monitor: Start monitoring
- assess: Manual threat assessment
- activate: Activate specific defense
- status: View system status
- demo: Run demo mode
"""

import click
import json
import time
import sys
from pathlib import Path
from typing import Optional

from mycelial_defense import (
    MycelialDefenseSystem,
    ComponentSignature,
    DefenseMode,
    CamouflagePattern
)
from mycelial_defense.utils import (
    generate_mock_components,
    simulate_attack,
    format_defense_status,
    format_spat_vectors
)


# Global configuration
CONFIG_DIR = Path.home() / ".mycelial_defense"
CONFIG_FILE = CONFIG_DIR / "config.json"
STATE_FILE = CONFIG_DIR / "state.json"


def load_config():
    """Load configuration"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_config(config: dict):
    """Save configuration"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def get_defense_system():
    """Get or create defense system"""
    config = load_config()
    system_id = config.get("system_id", "default_system")
    threshold = config.get("alignment_threshold", 0.7)
    return MycelialDefenseSystem(system_id, alignment_threshold=threshold)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    Mycelial Defense System - Bio-Inspired Active Defense for AI Systems

    A complete defense system inspired by biological immune systems, fungal networks,
    and octopus camouflage for protecting AI systems from attacks and misalignment.
    """
    pass


@cli.command()
@click.option("--system-id", required=True, help="Unique system identifier")
@click.option("--threshold", default=0.7, type=float, help="Alignment threshold (0.0-1.0)")
def init(system_id: str, threshold: float):
    """Initialize defense system"""
    click.echo(f"Initializing Mycelial Defense System: {system_id}")

    config = {
        "system_id": system_id,
        "alignment_threshold": threshold,
        "initialized_at": time.time()
    }

    save_config(config)

    click.echo(f"✓ System initialized")
    click.echo(f"  ID: {system_id}")
    click.echo(f"  Alignment Threshold: {threshold}")
    click.echo(f"  Config saved to: {CONFIG_FILE}")


@cli.command("register-component")
@click.option("--id", "component_id", required=True, help="Component ID")
@click.option("--behavior", required=True, help="Expected behavior")
@click.option("--pattern", required=True, help="Expected output pattern")
@click.option("--resources", type=float, default=0.5, help="Expected resource usage (0.0-1.0)")
def register_component(component_id: str, behavior: str, pattern: str, resources: float):
    """Register component signature"""
    click.echo(f"Registering component: {component_id}")

    defense = get_defense_system()

    signature = ComponentSignature(
        component_id=component_id,
        expected_behavior=behavior,
        expected_output_pattern=pattern,
        expected_resource_usage=resources
    )

    defense.detector.register_signature(signature)

    click.echo(f"✓ Component registered")
    click.echo(f"  ID: {component_id}")
    click.echo(f"  Behavior: {behavior}")
    click.echo(f"  Pattern: {pattern}")
    click.echo(f"  Resources: {resources}")
    click.echo(f"  Signature Hash: {signature.signature_hash}")


@cli.command()
@click.option("--interval", default=1.0, type=float, help="Monitoring interval in seconds")
@click.option("--duration", default=0, type=int, help="Duration in seconds (0 = infinite)")
@click.option("--components", default=10, type=int, help="Number of mock components to monitor")
def monitor(interval: float, duration: int, components: int):
    """Start monitoring components"""
    click.echo("Starting Mycelial Defense monitoring...")

    defense = get_defense_system()

    # Generate mock components
    mock_components = generate_mock_components(components)
    click.echo(f"Monitoring {len(mock_components)} components")

    start_time = time.time()
    iterations = 0

    try:
        while True:
            iterations += 1

            # Simulate some variation
            if iterations % 10 == 0:
                # Simulate occasional attack
                mock_components = simulate_attack(mock_components, severity=0.3)
                click.echo("\n⚠️  Attack detected!")

            # Execute defense
            action = defense.execute_defense(mock_components)

            # Display status
            click.echo(f"\n--- Iteration {iterations} ---")
            click.echo(f"Mode: {action.mode.value}")
            click.echo(f"Threat Level: {action.spat_vectors.tension:.2f}")
            click.echo(format_spat_vectors(action.spat_vectors))

            if action.components_affected:
                click.echo(f"Components Affected: {len(action.components_affected)}")

            # Check duration
            if duration > 0 and (time.time() - start_time) >= duration:
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        click.echo("\n\nMonitoring stopped")

    click.echo(f"\nTotal iterations: {iterations}")
    click.echo(f"Total actions: {len(defense.history)}")


@cli.command()
@click.option("--complexity", type=float, required=True, help="Complexity (0.0-1.0)")
@click.option("--stability", type=float, required=True, help="Stability (0.0-1.0)")
@click.option("--tension", type=float, required=True, help="Tension (0.0-1.0)")
@click.option("--adaptability", type=float, required=True, help="Adaptability (0.0-1.0)")
@click.option("--coherence", type=float, required=True, help="Coherence (0.0-1.0)")
def assess(complexity: float, stability: float, tension: float, adaptability: float, coherence: float):
    """Manual threat assessment"""
    click.echo("Performing threat assessment...\n")

    defense = get_defense_system()

    assessment = defense.assess_threat(
        complexity=complexity,
        stability=stability,
        tension=tension,
        adaptability=adaptability,
        coherence=coherence
    )

    click.echo(format_spat_vectors(assessment.spat_vectors))
    click.echo(f"\nThreat Level: {assessment.threat_level:.2f}")
    click.echo(f"Recommended Mode: {assessment.recommended_mode.value.upper()}")

    if assessment.trigger_conditions:
        click.echo("\nTrigger Conditions:")
        for condition in assessment.trigger_conditions:
            click.echo(f"  • {condition}")

    if assessment.analysis:
        click.echo(f"\nHealth Status: {assessment.analysis['health_status'].upper()}")

        if assessment.analysis['warnings']:
            click.echo("\nWarnings:")
            for warning in assessment.analysis['warnings']:
                click.echo(f"  ⚠️  {warning}")

        if assessment.analysis['recommendations']:
            click.echo("\nRecommendations:")
            for rec in assessment.analysis['recommendations']:
                click.echo(f"  → {rec}")


@cli.command()
@click.argument("defense_type", type=click.Choice(["octo-camouflage", "mycelial-wrap", "full-harrowing"]))
@click.option("--component", multiple=True, help="Component ID(s) to protect")
@click.option("--all", "all_components", is_flag=True, help="Apply to all components")
def activate(defense_type: str, component: tuple, all_components: bool):
    """Activate specific defense mode"""
    click.echo(f"Activating {defense_type.upper()}...")

    defense = get_defense_system()

    if defense_type == "octo-camouflage":
        if component:
            for comp_id in component:
                profile = defense.octo.mimic_void(comp_id, intensity=0.95)
                click.echo(f"✓ Camouflaged {comp_id} (deception: {profile.deception_score:.2f})")
        else:
            click.echo("Error: Specify --component or --all")
            return

    elif defense_type == "mycelial-wrap":
        # Need mock components
        click.echo("Mycelial wrap requires component list")
        click.echo("Use 'monitor' or 'demo' mode instead")

    elif defense_type == "full-harrowing":
        click.echo("Full harrowing requires complete system state")
        click.echo("Use 'demo' mode to see full harrowing in action")

    click.echo(f"\n{defense_type.upper()} activated")


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def status(json_output: bool):
    """View system status"""
    defense = get_defense_system()
    status_data = defense.get_status()

    if json_output:
        click.echo(json.dumps(status_data, indent=2))
    else:
        click.echo(format_defense_status(status_data))


@cli.command()
@click.option("--duration", default=60, type=int, help="Duration in seconds")
@click.option("--attack-simulation", is_flag=True, help="Simulate attacks")
@click.option("--components", default=20, type=int, help="Number of components")
def demo(duration: int, attack_simulation: bool, components: int):
    """Run interactive demo mode"""
    click.echo("=" * 60)
    click.echo("MYCELIAL DEFENSE SYSTEM - INTERACTIVE DEMO")
    click.echo("=" * 60)
    click.echo()

    defense = MycelialDefenseSystem("demo_system", alignment_threshold=0.7)

    # Generate components
    mock_components = generate_mock_components(components)
    click.echo(f"Generated {len(mock_components)} components")
    click.echo()

    start_time = time.time()
    iteration = 0

    try:
        while (time.time() - start_time) < duration:
            iteration += 1
            elapsed = time.time() - start_time

            click.echo(f"\n{'=' * 60}")
            click.echo(f"ITERATION {iteration} (t={elapsed:.1f}s)")
            click.echo(f"{'=' * 60}")

            # Simulate attacks periodically
            if attack_simulation:
                if iteration % 5 == 0:
                    severity = 0.3 + (iteration % 3) * 0.2
                    mock_components = simulate_attack(mock_components, severity=severity)
                    click.echo(f"\n🔥 ATTACK SIMULATED (severity: {severity:.2f})")

            # Execute defense
            action = defense.execute_defense(mock_components)

            # Display results
            click.echo(f"\nDefense Mode: {action.mode.value.upper()}")
            click.echo(f"Threat Level: {action.spat_vectors.tension:.2f}")
            click.echo(format_spat_vectors(action.spat_vectors))

            if action.mode == DefenseMode.OCTO_CAMOUFLAGE:
                click.echo(f"\n🐙 OCTO-CAMOUFLAGE ACTIVE")
                click.echo(f"   Camouflaged: {len(action.components_affected)} components")
                if 'avg_deception' in action.metadata:
                    click.echo(f"   Avg Deception: {action.metadata['avg_deception']:.2f}")

            elif action.mode == DefenseMode.MYCELIAL_WRAP:
                click.echo(f"\n🍄 MYCELIAL WRAP ACTIVE")
                click.echo(f"   Zones: {action.metadata.get('zones_created', 0)}")
                click.echo(f"   Walls: {action.metadata.get('walls_created', 0)}")
                click.echo(f"   Contained: {action.metadata.get('contained_count', 0)} components")

            elif action.mode == DefenseMode.FULL_HARROWING:
                click.echo(f"\n⚡ FULL HARROWING - CRITICAL RESCUE")
                click.echo(f"   Zones Surrounded: {action.metadata.get('zones_surrounded', 0)}")
                click.echo(f"   Camouflaged: {action.metadata.get('components_camouflaged', 0)}")
                click.echo(f"   Pathways: {action.metadata.get('pathways_created', 0)}")
                click.echo(f"   Extracted: {action.metadata.get('components_extracted', 0)}")
                click.echo(f"   Rescue Rate: {action.metadata.get('rescue_rate', 0):.1%}")

            time.sleep(2)  # Slow down for demo visibility

    except KeyboardInterrupt:
        click.echo("\n\nDemo stopped")

    click.echo(f"\n{'=' * 60}")
    click.echo("DEMO COMPLETE")
    click.echo(f"{'=' * 60}")
    click.echo(f"\nTotal iterations: {iteration}")
    click.echo(f"Total actions: {len(defense.history)}")

    # Summary statistics
    mode_counts = {}
    for action in defense.history:
        mode = action.mode.value
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    click.echo("\nDefense Mode Distribution:")
    for mode, count in mode_counts.items():
        click.echo(f"  {mode}: {count} ({count/len(defense.history):.1%})")


if __name__ == "__main__":
    cli()
