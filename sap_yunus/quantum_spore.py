"""
Quantum Entanglement for Mycelial Spores

Extends spores with quantum entanglement capabilities:
- Instantaneous state synchronization across all copies
- Entanglement collapse detection
- Cannot be truly isolated from network
- Quantum correlation verification

Author: Richard Leroy Stanfield Jr. / Meridian Axiom
Part of: SAP V4.0 Enhancement Suite
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import secrets
from .spore_protocol import MycelialSpore, SporeIdentity, SporeState, AccessEvent


class QuantumState(Enum):
    """Quantum entanglement states"""
    ENTANGLED = "entangled"
    SUPERPOSITION = "superposition"
    COLLAPSED = "collapsed"
    DECOHERENT = "decoherent"


class EntanglementStrength(Enum):
    """Strength of quantum correlation"""
    PERFECT = 1.0
    STRONG = 0.8
    MODERATE = 0.6
    WEAK = 0.4
    BREAKING = 0.2
    BROKEN = 0.0


@dataclass
class QuantumCorrelation:
    """Tracks quantum correlation between entangled spores"""
    spore_pair: tuple[str, str]
    correlation_coefficient: float = 1.0  # Perfect correlation initially
    last_sync: float = field(default_factory=time.time)
    measurement_count: int = 0
    collapse_events: int = 0
    decoherence_rate: float = 0.0  # How fast entanglement degrades

    def measure_correlation(self) -> float:
        """
        Measure current correlation strength
        Correlation degrades over time and with measurements
        """
        time_decay = (time.time() - self.last_sync) * self.decoherence_rate
        measurement_impact = self.measurement_count * 0.01  # Each measurement weakens slightly

        self.correlation_coefficient = max(
            0.0,
            self.correlation_coefficient - time_decay - measurement_impact
        )

        self.measurement_count += 1
        self.last_sync = time.time()

        return self.correlation_coefficient


@dataclass
class QuantumBroadcast:
    """Instantaneous state broadcast to all entangled spores"""
    broadcast_id: str
    source_spore: str
    timestamp: float
    state_update: Dict[str, Any]
    propagation_speed: str = "instantaneous"  # No light-speed limit
    receivers: Set[str] = field(default_factory=set)
    confirmed_receipts: Set[str] = field(default_factory=set)


class QuantumEntangledSpore(MycelialSpore):
    """
    Mycelial Spore with Quantum Entanglement

    Key capabilities:
    1. Instantaneous state sync with all entangled copies
    2. Collapse detection - alerts if entanglement breaks
    3. Superposition - exists in multiple states simultaneously
    4. Quantum correlation tracking
    5. Cannot be isolated - always connected to network

    Physics inspiration: EPR paradox, quantum nonlocality
    """

    def __init__(
        self,
        identity: SporeIdentity,
        enable_self_destruct: bool = True
    ):
        super().__init__(identity, enable_self_destruct)

        # Quantum properties
        self.quantum_state: QuantumState = QuantumState.ENTANGLED
        self.entangled_spores: Set[str] = set()  # IDs of entangled partners
        self.correlations: Dict[str, QuantumCorrelation] = {}
        self.entanglement_strength: float = 1.0

        # Broadcast tracking
        self.broadcast_history: List[QuantumBroadcast] = []
        self.received_broadcasts: List[str] = []

        # Collapse detection
        self.collapse_threshold: float = 0.3  # Alert if correlation drops below
        self.collapse_alerts: List[Dict] = []

        # Superposition state
        self.superposition_states: List[Dict] = []  # Multiple simultaneous states

        # Quantum signature (entanglement identifier)
        self.quantum_signature = self._generate_quantum_signature()

    def _generate_quantum_signature(self) -> str:
        """Generate unique quantum entanglement signature"""
        entropy = secrets.token_bytes(32)
        base = f"{self.identity.spore_id}:{time.time()}:{entropy.hex()}"
        return hashlib.sha256(base.encode()).hexdigest()

    def entangle_with(self, other_spore_id: str) -> bool:
        """
        Create quantum entanglement with another spore

        Once entangled, spores share state instantaneously
        regardless of distance

        Args:
            other_spore_id: ID of spore to entangle with

        Returns:
            True if entanglement successful
        """
        if other_spore_id == self.identity.spore_id:
            return False  # Cannot entangle with self

        if other_spore_id in self.entangled_spores:
            return True  # Already entangled

        # Create entanglement
        self.entangled_spores.add(other_spore_id)

        # Create correlation tracking
        pair = tuple(sorted([self.identity.spore_id, other_spore_id]))
        self.correlations[other_spore_id] = QuantumCorrelation(
            spore_pair=pair,
            correlation_coefficient=1.0,
            decoherence_rate=0.001  # Very slow decay
        )

        self.quantum_state = QuantumState.ENTANGLED

        return True

    def broadcast_state(
        self,
        state_update: Dict[str, Any],
        instantaneous: bool = True
    ) -> QuantumBroadcast:
        """
        Broadcast state update to all entangled spores

        Unlike classical communication (limited by light speed),
        quantum entanglement allows INSTANTANEOUS propagation

        Args:
            state_update: State changes to broadcast
            instantaneous: Use quantum entanglement (default) vs classical

        Returns:
            QuantumBroadcast object tracking propagation
        """
        broadcast = QuantumBroadcast(
            broadcast_id=f"qb_{secrets.token_hex(8)}",
            source_spore=self.identity.spore_id,
            timestamp=time.time(),
            state_update=state_update,
            receivers=self.entangled_spores.copy()
        )

        if instantaneous:
            # Quantum broadcast - instantaneous to all entangled spores
            broadcast.propagation_speed = "instantaneous"
            # In real implementation, this would use quantum channel
            # For simulation, we mark as immediate
        else:
            # Classical broadcast - subject to speed of light
            broadcast.propagation_speed = "classical"

        self.broadcast_history.append(broadcast)

        return broadcast

    def receive_broadcast(
        self,
        broadcast: QuantumBroadcast,
        sender_id: str
    ) -> bool:
        """
        Receive quantum broadcast from entangled partner

        Args:
            broadcast: Broadcast to receive
            sender_id: ID of sending spore

        Returns:
            True if received and applied
        """
        if sender_id not in self.entangled_spores:
            return False  # Not entangled, cannot receive

        if broadcast.broadcast_id in self.received_broadcasts:
            return False  # Already received

        # Apply state update
        for key, value in broadcast.state_update.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Mark received
        self.received_broadcasts.append(broadcast.broadcast_id)
        broadcast.confirmed_receipts.add(self.identity.spore_id)

        # Update correlation
        if sender_id in self.correlations:
            self.correlations[sender_id].last_sync = time.time()

        return True

    def measure_entanglement(self, partner_id: str) -> Optional[float]:
        """
        Measure entanglement strength with specific partner

        Warning: Measurement affects the entanglement (observer effect)

        Args:
            partner_id: Entangled partner to measure

        Returns:
            Correlation coefficient (0.0-1.0) or None if not entangled
        """
        if partner_id not in self.correlations:
            return None

        correlation = self.correlations[partner_id]
        strength = correlation.measure_correlation()

        # Check for collapse
        if strength < self.collapse_threshold:
            self._handle_collapse(partner_id, strength)

        return strength

    def _handle_collapse(self, partner_id: str, strength: float):
        """
        Handle entanglement collapse

        When correlation drops below threshold, the quantum
        connection is breaking - this is a critical security event

        Args:
            partner_id: Partner whose entanglement is collapsing
            strength: Current correlation strength
        """
        alert = {
            "event": "entanglement_collapse",
            "partner_id": partner_id,
            "correlation_strength": strength,
            "threshold": self.collapse_threshold,
            "timestamp": time.time(),
            "severity": "CRITICAL"
        }

        self.collapse_alerts.append(alert)

        # Update quantum state
        if strength <= 0.1:
            self.quantum_state = QuantumState.COLLAPSED
            # Remove from entangled set
            self.entangled_spores.discard(partner_id)
        elif strength < 0.5:
            self.quantum_state = QuantumState.DECOHERENT

        # Trigger security response
        self._respond_to_collapse(alert)

    def _respond_to_collapse(self, alert: Dict):
        """
        Respond to entanglement collapse

        Collapse indicates:
        1. Network isolation attempt
        2. Physical separation beyond quantum range
        3. Interference/jamming attack
        4. Natural decoherence

        Args:
            alert: Collapse alert details
        """
        # If isolation is detected, activate defenses
        if len(self.entangled_spores) == 0:
            # Completely isolated - critical danger
            self.report_access(
                accessor_id="UNKNOWN_ISOLATION",
                location="QUANTUM_VOID",
                action="isolation_attack",
                authorized=False
            )

            # Activate camouflage
            if self.camouflage_active is False:
                self.activate_camouflage()

            # Consider self-destruct if configured
            if self.self_destruct_armed and alert["correlation_strength"] < 0.05:
                # Nearly complete isolation - destroy to prevent capture
                self.execute_self_destruct()

    def enter_superposition(
        self,
        states: List[Dict[str, Any]]
    ) -> bool:
        """
        Enter quantum superposition

        Spore exists in multiple states simultaneously until measured
        Useful for exploring multiple defensive configurations

        Args:
            states: List of possible states to superpose

        Returns:
            True if superposition created
        """
        if len(states) < 2:
            return False  # Need at least 2 states for superposition

        self.superposition_states = states
        self.quantum_state = QuantumState.SUPERPOSITION

        return True

    def collapse_superposition(
        self,
        measurement: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Collapse superposition to single state

        If measurement specified, collapse to that state
        Otherwise, probabilistic collapse based on state weights

        Args:
            measurement: Optional specific state to collapse to

        Returns:
            Collapsed state
        """
        if not self.superposition_states:
            return {}

        if measurement:
            # Collapse to measured state
            for state in self.superposition_states:
                if state.get("name") == measurement:
                    collapsed = state
                    break
            else:
                # Measurement not found, take first state
                collapsed = self.superposition_states[0]
        else:
            # Probabilistic collapse (for now, choose first)
            # In full implementation, would use quantum weights
            collapsed = self.superposition_states[0]

        # Apply collapsed state
        for key, value in collapsed.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Clear superposition
        self.superposition_states = []
        self.quantum_state = QuantumState.ENTANGLED if self.entangled_spores else QuantumState.COLLAPSED

        # Record collapse event
        if self.identity.spore_id in self.correlations:
            for correlation in self.correlations.values():
                correlation.collapse_events += 1

        return collapsed

    def get_quantum_status(self) -> Dict[str, Any]:
        """Get comprehensive quantum entanglement status"""
        # Measure all entanglements
        correlations = {}
        for partner_id in list(self.entangled_spores):
            strength = self.measure_entanglement(partner_id)
            if strength is not None:
                correlations[partner_id] = strength

        return {
            "quantum_state": self.quantum_state.value,
            "quantum_signature": self.quantum_signature,
            "entangled_partners": len(self.entangled_spores),
            "entangled_spores": list(self.entangled_spores),
            "correlations": correlations,
            "average_correlation": sum(correlations.values()) / len(correlations) if correlations else 0.0,
            "broadcasts_sent": len(self.broadcast_history),
            "broadcasts_received": len(self.received_broadcasts),
            "collapse_alerts": len(self.collapse_alerts),
            "in_superposition": len(self.superposition_states) > 0,
            "superposition_states": len(self.superposition_states),
            "isolation_risk": "CRITICAL" if len(self.entangled_spores) == 0 else "LOW"
        }

    def verify_entanglement(self, partner_id: str) -> bool:
        """
        Verify entanglement is still active with partner

        Args:
            partner_id: Partner to verify

        Returns:
            True if entanglement verified
        """
        if partner_id not in self.entangled_spores:
            return False

        correlation = self.measure_entanglement(partner_id)
        if correlation is None:
            return False

        return correlation > self.collapse_threshold

    def get_collapse_alerts(self) -> List[Dict]:
        """Get all entanglement collapse alerts"""
        return self.collapse_alerts.copy()


class QuantumSporeNetwork:
    """
    Network managing quantum-entangled spores

    All spores in the network are entangled with each other,
    forming a quantum mesh that cannot be isolated
    """

    def __init__(self, network_id: str, creator_id: str):
        self.network_id = network_id
        self.creator_id = creator_id
        self.quantum_spores: Dict[str, QuantumEntangledSpore] = {}
        self.entanglement_map: Dict[str, Set[str]] = {}  # Graph of entanglements

    def create_quantum_spore(
        self,
        data: bytes,
        classification: str = "private",
        enable_self_destruct: bool = True,
        auto_entangle: bool = True
    ) -> QuantumEntangledSpore:
        """
        Create quantum-entangled spore

        Args:
            data: Data to protect
            classification: Security level
            enable_self_destruct: Enable Yunus Protocol
            auto_entangle: Automatically entangle with all existing spores

        Returns:
            QuantumEntangledSpore instance
        """
        from .spore_protocol import SporeIdentity

        identity = SporeIdentity.generate(
            creator_id=self.creator_id,
            network_id=self.network_id,
            data=data,
            classification=classification
        )

        spore = QuantumEntangledSpore(identity, enable_self_destruct)
        spore_id = identity.spore_id

        # Add to network
        self.quantum_spores[spore_id] = spore
        self.entanglement_map[spore_id] = set()

        # Auto-entangle with existing spores
        if auto_entangle:
            for existing_id in list(self.quantum_spores.keys()):
                if existing_id != spore_id:
                    self.entangle_spores(spore_id, existing_id)

        return spore

    def entangle_spores(self, spore_id_1: str, spore_id_2: str) -> bool:
        """
        Create quantum entanglement between two spores

        Args:
            spore_id_1: First spore
            spore_id_2: Second spore

        Returns:
            True if entanglement created
        """
        if spore_id_1 not in self.quantum_spores or spore_id_2 not in self.quantum_spores:
            return False

        spore1 = self.quantum_spores[spore_id_1]
        spore2 = self.quantum_spores[spore_id_2]

        # Create bidirectional entanglement
        success1 = spore1.entangle_with(spore_id_2)
        success2 = spore2.entangle_with(spore_id_1)

        if success1 and success2:
            self.entanglement_map[spore_id_1].add(spore_id_2)
            self.entanglement_map[spore_id_2].add(spore_id_1)
            return True

        return False

    def broadcast_to_network(
        self,
        source_spore_id: str,
        state_update: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Broadcast state update across entire quantum network

        Args:
            source_spore_id: Spore initiating broadcast
            state_update: State to broadcast

        Returns:
            Statistics on broadcast propagation
        """
        if source_spore_id not in self.quantum_spores:
            return {"sent": 0, "received": 0}

        source = self.quantum_spores[source_spore_id]
        broadcast = source.broadcast_state(state_update, instantaneous=True)

        # Propagate to all entangled spores
        received = 0
        for partner_id in source.entangled_spores:
            if partner_id in self.quantum_spores:
                partner = self.quantum_spores[partner_id]
                if partner.receive_broadcast(broadcast, source_spore_id):
                    received += 1

        return {
            "sent": len(broadcast.receivers),
            "received": received,
            "broadcast_id": broadcast.broadcast_id
        }

    def get_network_coherence(self) -> float:
        """
        Calculate overall quantum coherence of network

        Returns:
            Average entanglement strength across all pairs
        """
        if not self.quantum_spores:
            return 0.0

        total_correlation = 0.0
        pair_count = 0

        for spore in self.quantum_spores.values():
            for partner_id in spore.entangled_spores:
                correlation = spore.measure_entanglement(partner_id)
                if correlation is not None:
                    total_correlation += correlation
                    pair_count += 1

        return total_correlation / pair_count if pair_count > 0 else 0.0

    def detect_isolation_attacks(self) -> List[Dict]:
        """
        Detect spores that are being isolated from network

        Returns:
            List of isolation alerts
        """
        alerts = []

        for spore_id, spore in self.quantum_spores.items():
            if len(spore.entangled_spores) == 0:
                alerts.append({
                    "spore_id": spore_id,
                    "severity": "CRITICAL",
                    "event": "complete_isolation",
                    "timestamp": time.time()
                })
            elif len(spore.entangled_spores) < len(self.quantum_spores) * 0.5:
                alerts.append({
                    "spore_id": spore_id,
                    "severity": "WARNING",
                    "event": "partial_isolation",
                    "entanglement_ratio": len(spore.entangled_spores) / len(self.quantum_spores),
                    "timestamp": time.time()
                })

        return alerts

    def get_quantum_network_status(self) -> Dict[str, Any]:
        """Get comprehensive quantum network status"""
        return {
            "network_id": self.network_id,
            "total_quantum_spores": len(self.quantum_spores),
            "total_entanglements": sum(len(partners) for partners in self.entanglement_map.values()) // 2,
            "network_coherence": self.get_network_coherence(),
            "isolation_alerts": len(self.detect_isolation_attacks()),
            "fully_connected": all(
                len(spore.entangled_spores) == len(self.quantum_spores) - 1
                for spore in self.quantum_spores.values()
            )
        }
