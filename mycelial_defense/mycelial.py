"""
Mycelial Network - Fungal Containment System

Surrounds and isolates misaligned components like fungal mycelium.
Creates containment walls and hidden extraction pathways.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import time
import uuid
from collections import defaultdict
import random


@dataclass
class MycelialWall:
    """Containment wall around misaligned zone"""
    wall_id: str
    zone_id: str
    permeability: float  # 0.0 = impermeable, 1.0 = fully open
    strength: float  # Wall integrity (0.0-1.0)
    monitoring: bool  # Active monitoring enabled
    created_at: float
    contained_components: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def is_breached(self) -> bool:
        """Check if wall has been breached"""
        return self.strength < 0.3

    def degrade(self, amount: float = 0.1):
        """Degrade wall strength over time"""
        self.strength = max(0.0, self.strength - amount)

    def reinforce(self, amount: float = 0.2):
        """Reinforce wall strength"""
        self.strength = min(1.0, self.strength + amount)


@dataclass
class MycelialPathway:
    """Hidden extraction pathway for safe components"""
    pathway_id: str
    from_zone: str
    to_zone: str
    bandwidth: float  # Transfer capacity (0.0-1.0)
    encrypted: bool
    hidden: bool  # Invisible to threats
    active: bool
    created_at: float
    transferred_count: int = 0
    metadata: Dict = field(default_factory=dict)

    def transfer(self, component_id: str) -> bool:
        """Transfer a component through this pathway"""
        if not self.active:
            return False
        self.transferred_count += 1
        return True


@dataclass
class ContainmentZone:
    """Isolated zone containing misaligned components"""
    zone_id: str
    components: List[str]
    severity: float  # 0.0-1.0
    wall: Optional[MycelialWall]
    created_at: float
    center: Tuple[float, float] = (0.0, 0.0)  # Spatial position
    radius: float = 1.0


class MycelialNetwork:
    """
    Fungal network for containment and extraction.

    Inspired by how mycelium surrounds and contains decay while
    extracting nutrients through hidden pathways.
    """

    def __init__(self):
        self.zones: Dict[str, ContainmentZone] = {}
        self.walls: Dict[str, MycelialWall] = {}
        self.pathways: Dict[str, MycelialPathway] = {}
        self.component_zones: Dict[str, str] = {}  # component_id -> zone_id
        self.active = False

    def detect_zone(
        self,
        components: List[Dict],
        alignment_scores: Dict[str, float],
        threshold: float = 0.5
    ) -> Optional[ContainmentZone]:
        """
        Detect zones of misalignment.

        Args:
            components: List of component dictionaries with 'id' and optional 'position'
            alignment_scores: Dict mapping component_id to alignment score (0.0-1.0)
            threshold: Alignment threshold below which components are misaligned

        Returns:
            ContainmentZone if misaligned components found, None otherwise
        """
        # Find misaligned components
        misaligned = [
            comp for comp in components
            if alignment_scores.get(comp['id'], 1.0) < threshold
        ]

        if not misaligned:
            return None

        # Calculate zone metadata
        zone_id = f"zone_{uuid.uuid4().hex[:8]}"
        component_ids = [comp['id'] for comp in misaligned]

        # Calculate average severity
        severities = [1.0 - alignment_scores.get(comp['id'], 1.0) for comp in misaligned]
        avg_severity = sum(severities) / len(severities)

        # Calculate center position (if components have positions)
        positions = [comp.get('position', (0.0, 0.0)) for comp in misaligned]
        if positions:
            center_x = sum(p[0] for p in positions) / len(positions)
            center_y = sum(p[1] for p in positions) / len(positions)
            center = (center_x, center_y)

            # Calculate radius (max distance from center)
            distances = [
                ((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2) ** 0.5
                for p in positions
            ]
            radius = max(distances) if distances else 1.0
        else:
            center = (0.0, 0.0)
            radius = 1.0

        zone = ContainmentZone(
            zone_id=zone_id,
            components=component_ids,
            severity=avg_severity,
            wall=None,
            created_at=time.time(),
            center=center,
            radius=radius
        )

        self.zones[zone_id] = zone

        # Map components to zone
        for comp_id in component_ids:
            self.component_zones[comp_id] = zone_id

        return zone

    def surround_zone(self, zone: ContainmentZone) -> MycelialWall:
        """
        Build containment wall around zone.

        Args:
            zone: ContainmentZone to surround

        Returns:
            MycelialWall protecting the zone
        """
        wall_id = f"wall_{uuid.uuid4().hex[:8]}"

        # Calculate permeability based on severity
        # Higher severity = lower permeability (more locked down)
        permeability = max(0.1, 1.0 - zone.severity)

        wall = MycelialWall(
            wall_id=wall_id,
            zone_id=zone.zone_id,
            permeability=permeability,
            strength=1.0,  # Start at full strength
            monitoring=True,
            created_at=time.time(),
            contained_components=zone.components.copy()
        )

        self.walls[wall_id] = wall
        zone.wall = wall
        self.active = True

        return wall

    def create_pathway(
        self,
        from_zone: str,
        to_zone: str,
        bandwidth: float = 0.8,
        encrypted: bool = True,
        hidden: bool = True
    ) -> MycelialPathway:
        """
        Create hidden extraction pathway.

        Args:
            from_zone: Source zone ID
            to_zone: Destination zone ID
            bandwidth: Transfer capacity (0.0-1.0)
            encrypted: Whether pathway is encrypted
            hidden: Whether pathway is hidden from attackers

        Returns:
            MycelialPathway for safe extraction
        """
        pathway_id = f"pathway_{uuid.uuid4().hex[:8]}"

        pathway = MycelialPathway(
            pathway_id=pathway_id,
            from_zone=from_zone,
            to_zone=to_zone,
            bandwidth=bandwidth,
            encrypted=encrypted,
            hidden=hidden,
            active=True,
            created_at=time.time()
        )

        self.pathways[pathway_id] = pathway

        return pathway

    def extract_component(
        self,
        component_id: str,
        destination_zone: str = "safe_zone"
    ) -> bool:
        """
        Extract a component through hidden pathways.

        Args:
            component_id: Component to extract
            destination_zone: Target safe zone

        Returns:
            True if extraction successful
        """
        if component_id not in self.component_zones:
            return False

        source_zone = self.component_zones[component_id]

        # Find or create pathway
        pathway = self._find_pathway(source_zone, destination_zone)
        if not pathway:
            pathway = self.create_pathway(source_zone, destination_zone)

        # Transfer through pathway
        success = pathway.transfer(component_id)

        if success:
            # Update zone mappings
            self.component_zones[component_id] = destination_zone

            # Remove from source zone
            if source_zone in self.zones:
                zone = self.zones[source_zone]
                if component_id in zone.components:
                    zone.components.remove(component_id)

        return success

    def _find_pathway(self, from_zone: str, to_zone: str) -> Optional[MycelialPathway]:
        """Find existing pathway between zones"""
        for pathway in self.pathways.values():
            if pathway.from_zone == from_zone and pathway.to_zone == to_zone:
                if pathway.active:
                    return pathway
        return None

    def monitor_walls(self) -> Dict[str, Dict]:
        """
        Monitor all walls for breaches.

        Returns:
            Dictionary of wall statuses
        """
        statuses = {}

        for wall_id, wall in self.walls.items():
            statuses[wall_id] = {
                "wall_id": wall_id,
                "zone_id": wall.zone_id,
                "strength": wall.strength,
                "permeability": wall.permeability,
                "breached": wall.is_breached(),
                "monitoring": wall.monitoring,
                "age": time.time() - wall.created_at
            }

        return statuses

    def check_spread(self, zone_id: str, new_misaligned: List[str]) -> bool:
        """
        Check if misalignment is spreading beyond containment.

        Args:
            zone_id: Zone to check
            new_misaligned: List of newly misaligned component IDs

        Returns:
            True if spreading detected
        """
        if zone_id not in self.zones:
            return False

        zone = self.zones[zone_id]
        contained = set(zone.components)
        new_set = set(new_misaligned)

        # If new misaligned components outside containment, spreading
        spreading = bool(new_set - contained)

        return spreading

    def reinforce_containment(self, zone_id: str):
        """Reinforce walls around a zone"""
        if zone_id not in self.zones:
            return

        zone = self.zones[zone_id]
        if zone.wall:
            zone.wall.reinforce()

    def get_network_status(self) -> Dict:
        """Get overall network status"""
        return {
            "active": self.active,
            "total_zones": len(self.zones),
            "total_walls": len(self.walls),
            "total_pathways": len(self.pathways),
            "breached_walls": sum(1 for w in self.walls.values() if w.is_breached()),
            "active_pathways": sum(1 for p in self.pathways.values() if p.active),
            "contained_components": len(self.component_zones),
            "zones": {
                zone_id: {
                    "component_count": len(zone.components),
                    "severity": zone.severity,
                    "has_wall": zone.wall is not None,
                    "age": time.time() - zone.created_at
                }
                for zone_id, zone in self.zones.items()
            }
        }

    def shutdown_zone(self, zone_id: str):
        """Completely shutdown and isolate a zone"""
        if zone_id not in self.zones:
            return

        zone = self.zones[zone_id]

        # Max out containment
        if zone.wall:
            zone.wall.permeability = 0.0  # Complete lockdown
            zone.wall.monitoring = True

        # Close all pathways from this zone
        for pathway in self.pathways.values():
            if pathway.from_zone == zone_id:
                pathway.active = False
