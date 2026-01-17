"""
Mycelial Spore Protocol - Tethered Information Defense

Self-defending information that remains connected to the user's mycelial network.
Information "spores" that can track, report, self-destruct, and be remotely controlled.

Capabilities:
1. Track everywhere data goes (audit trail)
2. Report when accessed/copied/moved
3. Self-destruct if compromised
4. Camouflage (corrupt to appear useless)
5. Harrowing Recall (attempt remote deletion)
6. Yunus Sacrifice (data corrupts itself)

Author: Richard Leroy Stanfield Jr. / Meridian Axiom
Integration: LUMINARK Mycelial Defense + SAP V4.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from enum import Enum
import time
import uuid
import hashlib
import json


class SporeState(Enum):
    """State of a mycelial spore"""
    DORMANT = "dormant"          # Created but not deployed
    ACTIVE = "active"            # Deployed and tracking
    ACCESSED = "accessed"        # Someone accessed it
    COPIED = "copied"            # Data was copied
    COMPROMISED = "compromised"  # Detected unauthorized access
    CAMOUFLAGED = "camouflaged"  # Self-corrupted to hide
    DESTROYED = "destroyed"      # Self-destructed
    RECALLED = "recalled"        # Successfully retrieved


class ThreatLevel(Enum):
    """Threat assessment for spore access"""
    SAFE = 0         # Authorized access
    SUSPICIOUS = 1   # Unexpected access pattern
    HOSTILE = 2      # Clear unauthorized access
    CRITICAL = 3     # Active exfiltration attempt


@dataclass
class SporeIdentity:
    """Unique identity for a mycelial spore"""
    spore_id: str
    creator_id: str  # User who created it
    network_id: str  # Mycelial network it belongs to
    created_at: float
    data_hash: str   # Hash of protected data
    classification: str  # "public", "private", "confidential", "secret"

    @classmethod
    def generate(cls, creator_id: str, network_id: str, data: bytes, classification: str) -> "SporeIdentity":
        """Generate new spore identity"""
        spore_id = f"spore_{uuid.uuid4().hex[:12]}"
        data_hash = hashlib.sha256(data).hexdigest()[:16]

        return cls(
            spore_id=spore_id,
            creator_id=creator_id,
            network_id=network_id,
            created_at=time.time(),
            data_hash=data_hash,
            classification=classification
        )


@dataclass
class AccessEvent:
    """Record of spore access"""
    event_id: str
    spore_id: str
    timestamp: float
    accessor_id: Optional[str]  # Who accessed (if known)
    location: Optional[str]  # Where (IP, device, etc.)
    action: str  # "read", "copy", "modify", "delete"
    authorized: bool
    threat_level: ThreatLevel
    metadata: Dict = field(default_factory=dict)


@dataclass
class AuditTrail:
    """Complete audit trail for a spore"""
    spore_id: str
    events: List[AccessEvent] = field(default_factory=list)
    total_accesses: int = 0
    unique_accessors: Set[str] = field(default_factory=set)
    compromised_count: int = 0
    last_seen: Optional[float] = None
    current_locations: List[str] = field(default_factory=list)

    def add_event(self, event: AccessEvent):
        """Add access event to trail"""
        self.events.append(event)
        self.total_accesses += 1
        if event.accessor_id:
            self.unique_accessors.add(event.accessor_id)
        if not event.authorized:
            self.compromised_count += 1
        self.last_seen = event.timestamp

        # Track location
        if event.location and event.location not in self.current_locations:
            self.current_locations.append(event.location)


@dataclass
class MycelialSpore:
    """
    Self-defending information spore

    A piece of data wrapped with defensive mechanisms and network connectivity
    """
    identity: SporeIdentity
    state: SporeState
    audit_trail: AuditTrail
    tether_strength: float  # 0.0-1.0 (connection to network)
    defense_capabilities: List[str] = field(default_factory=list)
    self_destruct_armed: bool = False
    camouflage_active: bool = False
    beacon_active: bool = True  # Phone home capability

    def __post_init__(self):
        # Initialize defense capabilities
        self.defense_capabilities = [
            "tracking",
            "reporting",
            "self_destruct",
            "camouflage",
            "beacon"
        ]

    def report_access(
        self,
        accessor_id: Optional[str],
        location: Optional[str],
        action: str,
        authorized: bool
    ) -> AccessEvent:
        """
        Report access to mycelial network

        Args:
            accessor_id: Who accessed
            location: Where
            action: What they did
            authorized: Whether authorized

        Returns:
            AccessEvent record
        """
        threat_level = self._assess_threat(authorized, action)

        event = AccessEvent(
            event_id=f"event_{uuid.uuid4().hex[:8]}",
            spore_id=self.identity.spore_id,
            timestamp=time.time(),
            accessor_id=accessor_id,
            location=location,
            action=action,
            authorized=authorized,
            threat_level=threat_level
        )

        self.audit_trail.add_event(event)

        # Update state based on event
        if action == "copy":
            self.state = SporeState.COPIED
        elif not authorized:
            self.state = SporeState.COMPROMISED

        return event

    def _assess_threat(self, authorized: bool, action: str) -> ThreatLevel:
        """Assess threat level of access"""
        if not authorized:
            if action in ["copy", "exfiltrate"]:
                return ThreatLevel.CRITICAL
            elif action in ["modify", "delete"]:
                return ThreatLevel.HOSTILE
            else:
                return ThreatLevel.SUSPICIOUS
        return ThreatLevel.SAFE

    def activate_camouflage(self) -> bool:
        """
        Activate octo-camouflage - corrupt data to appear useless

        Returns:
            True if successful
        """
        if "camouflage" not in self.defense_capabilities:
            return False

        self.camouflage_active = True
        self.state = SporeState.CAMOUFLAGED

        # In real implementation:
        # - XOR data with noise
        # - Add corrupted headers
        # - Make file appear damaged

        return True

    def arm_self_destruct(self, trigger_conditions: Optional[Dict] = None):
        """
        Arm self-destruct mechanism

        Args:
            trigger_conditions: When to trigger (unauthorized_access, timeout, etc.)
        """
        if "self_destruct" not in self.defense_capabilities:
            return False

        self.self_destruct_armed = True

        # Trigger conditions could be:
        # - Unauthorized access detected
        # - Copied to untrusted location
        # - Network tether broken
        # - Time-based (expire after X days)

        return True

    def execute_self_destruct(self) -> bool:
        """
        Execute Yunus Protocol - data sacrifices itself

        Returns:
            True if destroyed
        """
        if not self.self_destruct_armed:
            return False

        # In real implementation:
        # - Overwrite data with random bytes
        # - Delete file
        # - Clear memory
        # - Report to network

        self.state = SporeState.DESTROYED
        self.beacon_active = False

        return True

    def phone_home(self, network_endpoint: str) -> Dict:
        """
        Phone home to mycelial network with status update

        Args:
            network_endpoint: Network endpoint to contact

        Returns:
            Network response
        """
        if not self.beacon_active:
            return {"status": "beacon_disabled"}

        # Report current status
        report = {
            "spore_id": self.identity.spore_id,
            "state": self.state.value,
            "tether_strength": self.tether_strength,
            "total_accesses": self.audit_trail.total_accesses,
            "compromised": self.state == SporeState.COMPROMISED,
            "current_locations": self.audit_trail.current_locations,
            "timestamp": time.time()
        }

        # In real implementation: HTTP POST to network
        # For now, simulate success
        return {"status": "received", "commands": []}

    def receive_command(self, command: Dict) -> Any:
        """
        Receive command from mycelial network

        Commands:
        - "self_destruct": Trigger Yunus Protocol
        - "camouflage": Activate concealment
        - "report_status": Send full status
        - "recall": Attempt to return to network

        Args:
            command: Command dictionary

        Returns:
            Command result
        """
        cmd_type = command.get("type")

        if cmd_type == "self_destruct":
            return self.execute_self_destruct()

        elif cmd_type == "camouflage":
            return self.activate_camouflage()

        elif cmd_type == "report_status":
            return self.get_status()

        elif cmd_type == "recall":
            return self._attempt_recall()

        else:
            return {"error": "Unknown command"}

    def _attempt_recall(self) -> Dict:
        """
        Attempt Harrowing Recall - return to network

        Returns:
            Recall result
        """
        if self.tether_strength < 0.3:
            return {
                "success": False,
                "reason": "Tether too weak",
                "recommendation": "Execute self-destruct"
            }

        # In real implementation:
        # - Encrypt and upload data back to network
        # - Delete local copy
        # - Verify retrieval

        self.state = SporeState.RECALLED
        return {"success": True, "retrieved": True}

    def get_status(self) -> Dict:
        """Get comprehensive spore status"""
        return {
            "identity": {
                "spore_id": self.identity.spore_id,
                "network_id": self.identity.network_id,
                "classification": self.identity.classification,
                "age": time.time() - self.identity.created_at
            },
            "state": self.state.value,
            "tether_strength": self.tether_strength,
            "defense": {
                "capabilities": self.defense_capabilities,
                "self_destruct_armed": self.self_destruct_armed,
                "camouflage_active": self.camouflage_active,
                "beacon_active": self.beacon_active
            },
            "audit": {
                "total_accesses": self.audit_trail.total_accesses,
                "unique_accessors": len(self.audit_trail.unique_accessors),
                "compromised_count": self.audit_trail.compromised_count,
                "current_locations": self.audit_trail.current_locations,
                "last_seen": self.audit_trail.last_seen
            }
        }


class MycelialSporeNetwork:
    """
    Network managing all mycelial spores for a user

    Tracks all deployed spores, receives status reports,
    issues commands, executes Harrowing recalls
    """

    def __init__(self, network_id: str, creator_id: str):
        self.network_id = network_id
        self.creator_id = creator_id
        self.spores: Dict[str, MycelialSpore] = {}
        self.active_tracking: bool = True

    def create_spore(
        self,
        data: bytes,
        classification: str = "private",
        enable_self_destruct: bool = True
    ) -> MycelialSpore:
        """
        Create new mycelial spore to protect data

        Args:
            data: Data to protect
            classification: Security classification
            enable_self_destruct: Whether to enable self-destruct

        Returns:
            Created spore
        """
        identity = SporeIdentity.generate(
            self.creator_id,
            self.network_id,
            data,
            classification
        )

        audit_trail = AuditTrail(spore_id=identity.spore_id)

        spore = MycelialSpore(
            identity=identity,
            state=SporeState.ACTIVE,
            audit_trail=audit_trail,
            tether_strength=1.0
        )

        if enable_self_destruct:
            spore.arm_self_destruct({
                "unauthorized_access": True,
                "tether_break": True
            })

        self.spores[identity.spore_id] = spore

        return spore

    def track_spore(self, spore_id: str) -> Optional[Dict]:
        """Get current tracking info for spore"""
        if spore_id not in self.spores:
            return None

        spore = self.spores[spore_id]
        return spore.get_status()

    def list_compromised(self) -> List[MycelialSpore]:
        """List all compromised spores"""
        return [
            spore for spore in self.spores.values()
            if spore.state == SporeState.COMPROMISED
        ]

    def execute_harrowing_recall(self, spore_id: str) -> Dict:
        """
        Execute Harrowing Protocol to recall spore

        Attempt to:
        1. Retrieve data back to network
        2. Delete remote copies
        3. If retrieval fails, execute Yunus (self-destruct)

        Args:
            spore_id: Spore to recall

        Returns:
            Recall result
        """
        if spore_id not in self.spores:
            return {"error": "Spore not found"}

        spore = self.spores[spore_id]

        # Attempt recall
        result = spore.receive_command({"type": "recall"})

        if not result.get("success"):
            # Recall failed - execute Yunus Protocol
            yunus_result = spore.receive_command({"type": "self_destruct"})
            return {
                "recall_failed": True,
                "yunus_executed": yunus_result,
                "data_destroyed": True
            }

        return result

    def mass_recall(self, classification_filter: Optional[str] = None) -> Dict:
        """
        Recall all spores (or filtered by classification)

        Args:
            classification_filter: Only recall spores of this classification

        Returns:
            Summary of recalls
        """
        results = {
            "attempted": 0,
            "successful": 0,
            "failed": 0,
            "destroyed": 0
        }

        for spore in self.spores.values():
            if classification_filter and spore.identity.classification != classification_filter:
                continue

            results["attempted"] += 1
            recall_result = self.execute_harrowing_recall(spore.identity.spore_id)

            if recall_result.get("success"):
                results["successful"] += 1
            elif recall_result.get("yunus_executed"):
                results["destroyed"] += 1
            else:
                results["failed"] += 1

        return results

    def get_network_status(self) -> Dict:
        """Get comprehensive network status"""
        total_spores = len(self.spores)
        active = sum(1 for s in self.spores.values() if s.state == SporeState.ACTIVE)
        compromised = sum(1 for s in self.spores.values() if s.state == SporeState.COMPROMISED)
        destroyed = sum(1 for s in self.spores.values() if s.state == SporeState.DESTROYED)

        return {
            "network_id": self.network_id,
            "total_spores": total_spores,
            "active": active,
            "compromised": compromised,
            "destroyed": destroyed,
            "tracking_active": self.active_tracking,
            "spores_by_classification": self._count_by_classification()
        }

    def _count_by_classification(self) -> Dict[str, int]:
        """Count spores by classification"""
        counts = {}
        for spore in self.spores.values():
            classification = spore.identity.classification
            counts[classification] = counts.get(classification, 0) + 1
        return counts
