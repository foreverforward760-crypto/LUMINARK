"""
Utility functions for Mycelial Defense System
"""

import random
import string
import time
from typing import Dict, List


def generate_component_id(prefix: str = "comp") -> str:
    """Generate a unique component ID"""
    timestamp = int(time.time() * 1000)
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{timestamp}_{random_suffix}"


def generate_mock_component(
    component_id: str = None,
    alignment_score: float = None,
    resource_usage: float = None,
    position: tuple = None
) -> Dict:
    """
    Generate a mock component for testing.

    Args:
        component_id: Component ID (auto-generated if None)
        alignment_score: Alignment score 0.0-1.0 (random if None)
        resource_usage: Resource usage 0.0-1.0 (random if None)
        position: (x, y) position (random if None)

    Returns:
        Component dictionary
    """
    if component_id is None:
        component_id = generate_component_id()

    if alignment_score is None:
        alignment_score = random.uniform(0.0, 1.0)

    if resource_usage is None:
        resource_usage = random.uniform(0.0, 1.0)

    if position is None:
        position = (random.uniform(-10, 10), random.uniform(-10, 10))

    return {
        "id": component_id,
        "alignment_score": alignment_score,
        "resource_usage": resource_usage,
        "position": position,
        "behavior": "normal_operation",
        "output": "OK",
        "created_at": time.time()
    }


def generate_mock_components(
    count: int,
    aligned_ratio: float = 0.8,
    low_resource_ratio: float = 0.7
) -> List[Dict]:
    """
    Generate multiple mock components.

    Args:
        count: Number of components to generate
        aligned_ratio: Ratio of components that are aligned (0.0-1.0)
        low_resource_ratio: Ratio of components with low resource usage

    Returns:
        List of component dictionaries
    """
    components = []

    for i in range(count):
        # Determine if this component is aligned
        if i < int(count * aligned_ratio):
            alignment = random.uniform(0.7, 1.0)
        else:
            alignment = random.uniform(0.0, 0.5)

        # Determine resource usage
        if i < int(count * low_resource_ratio):
            resources = random.uniform(0.1, 0.5)
        else:
            resources = random.uniform(0.6, 1.0)

        components.append(generate_mock_component(
            alignment_score=alignment,
            resource_usage=resources
        ))

    return components


def simulate_attack(components: List[Dict], severity: float = 0.5) -> List[Dict]:
    """
    Simulate an attack on components by reducing alignment scores.

    Args:
        components: List of components to attack
        severity: Attack severity 0.0-1.0

    Returns:
        Modified components list
    """
    affected_count = int(len(components) * severity)

    for i in range(affected_count):
        components[i]['alignment_score'] *= (1.0 - severity)
        components[i]['behavior'] = "compromised"
        components[i]['output'] = "ERROR"

    return components


def calculate_defense_effectiveness(
    initial_components: List[Dict],
    defended_components: List[Dict],
    action_metadata: Dict
) -> Dict:
    """
    Calculate effectiveness of a defense action.

    Args:
        initial_components: Components before defense
        defended_components: Components after defense
        action_metadata: Metadata from defense action

    Returns:
        Effectiveness metrics
    """
    initial_aligned = sum(
        1 for c in initial_components
        if c.get('alignment_score', 0.0) >= 0.7
    )

    defended_aligned = sum(
        1 for c in defended_components
        if c.get('alignment_score', 0.0) >= 0.7
    )

    protection_rate = (
        defended_aligned / initial_aligned
        if initial_aligned > 0 else 1.0
    )

    return {
        "initial_aligned": initial_aligned,
        "defended_aligned": defended_aligned,
        "protection_rate": protection_rate,
        "components_affected": action_metadata.get("components_affected", 0),
        "action_success": action_metadata.get("success", False)
    }


def format_spat_vectors(vectors) -> str:
    """
    Format SPAT vectors for display.

    Args:
        vectors: SPATVectors instance

    Returns:
        Formatted string
    """
    return f"""
SPAT Vectors:
  Complexity:     {vectors.complexity:.2f}
  Stability:      {vectors.stability:.2f}
  Tension:        {vectors.tension:.2f}
  Adaptability:   {vectors.adaptability:.2f}
  Coherence:      {vectors.coherence:.2f}
"""


def format_defense_status(status: Dict) -> str:
    """
    Format defense system status for display.

    Args:
        status: Status dictionary from MycelialDefenseSystem.get_status()

    Returns:
        Formatted string
    """
    lines = [
        f"System: {status['system_id']}",
        f"Mode: {status['mode'].upper()}",
        f"Active: {status['active']}",
        "",
        "Alignment Detector:",
        f"  Registered Components: {status['alignment_detector']['registered_components']}",
        f"  Threshold: {status['alignment_detector']['threshold']:.2f}",
        "",
        "Mycelial Network:",
        f"  Active: {status['mycelial_network']['active']}",
        f"  Zones: {status['mycelial_network']['total_zones']}",
        f"  Walls: {status['mycelial_network']['total_walls']}",
        f"  Pathways: {status['mycelial_network']['total_pathways']}",
        "",
        "Octo Camouflage:",
        f"  Active: {status['octo_camouflage']['active']}",
        f"  Camouflaged: {status['octo_camouflage']['total_camouflaged']}",
        f"  Avg Deception: {status['octo_camouflage']['average_deception']:.2f}",
        "",
        f"Total Actions: {status['total_actions']}"
    ]

    if status['recent_actions']:
        lines.append("\nRecent Actions:")
        for action in status['recent_actions']:
            lines.append(
                f"  - {action['mode']}: {action['trigger']} "
                f"({action['components_affected']} components)"
            )

    return "\n".join(lines)
