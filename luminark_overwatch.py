"""
LUMINARK OVERWATCH - AI Regulatory System
==========================================
An oversight layer that monitors systems/operations and uses SAP framework
to detect imbalances and restore alignment with proper function.

Core Functions:
1. MONITOR - Track registered systems and their operational states
2. DIAGNOSE - Map system states to SAP stages, detect inversions
3. INTERVENE - Prescribe corrections to restore balance
4. VALIDATE - Ensure interventions pass Ma'at ethical checks

Author: Richard L. Stanfield
Version: 1.0.0
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from collections import deque

# Import existing LUMINARK frameworks
_LUMINARK_AVAILABLE = False
try:
    from luminark.sap.framework_81 import SAP81Framework, SAPState81, Gate81, SAPArc
    from luminark_omega.protocols.maat import MaatEthicist
    from luminark_omega.core.sar_framework import SARFramework
    _LUMINARK_AVAILABLE = True
except ImportError:
    # Standalone mode - define minimal versions inline
    from enum import Enum as _Enum

    class Gate81(_Enum):
        GATE_0T = "0: Plenara - Primordial Source"
        GATE_1 = "1: Spark - Initial Ignition"
        GATE_2 = "2: Polarity - Understanding Duality"
        GATE_3 = "3: Motion - Movement, Action"
        GATE_4 = "4: Foundation - Stability, Structure"
        GATE_5 = "5: Threshold - Critical Decision"
        GATE_6 = "6: Integration - Merging Dualities"
        GATE_7 = "7: Illusion - Testing Reality"
        GATE_8 = "8: Rigidity - Crystallization"
        GATE_0B = "9: Renewal - Transcendence"

    class SAPArc(_Enum):
        DESCENDING = "Descending Arc"
        ASCENDING = "Ascending Arc"

    @dataclass
    class SAPState81:
        gate: Gate81
        micro_stage: float
        arc: SAPArc
        integrity: float
        fractal_coherence: float
        inversion_level: int
        physical_state: str
        conscious_state: str

        def get_absolute_stage(self) -> float:
            gate_num = list(Gate81).index(self.gate)
            return gate_num + self.micro_stage

        def is_threshold(self) -> bool:
            return abs(self.micro_stage - 0.5) < 0.05 or abs(self.micro_stage - 0.9) < 0.05

    class SAP81Framework:
        def __init__(self):
            self.gates = list(Gate81)
            self.gate_states = {
                Gate81.GATE_0T: ('unstable', 'unstable'),
                Gate81.GATE_1: ('unstable', 'stable'),
                Gate81.GATE_2: ('stable', 'unstable'),
                Gate81.GATE_3: ('unstable', 'stable'),
                Gate81.GATE_4: ('stable', 'unstable'),
                Gate81.GATE_5: ('unstable', 'stable'),
                Gate81.GATE_6: ('stable', 'unstable'),
                Gate81.GATE_7: ('unstable', 'stable'),
                Gate81.GATE_8: ('stable', 'unstable'),
                Gate81.GATE_0B: ('stable', 'stable')
            }

        def get_state(self, absolute_stage: float) -> SAPState81:
            absolute_stage = max(0.0, min(9.9, absolute_stage))
            gate_num = int(absolute_stage)
            micro_stage = absolute_stage - gate_num
            gate = self.gates[gate_num]
            physical_state, conscious_state = self.gate_states[gate]
            inversion_level = 0 if gate_num in [0, 9] else (8 if gate_num % 2 == 1 else 9)
            arc = SAPArc.DESCENDING if gate_num <= 4 else SAPArc.ASCENDING
            integrity = 100.0 - (inversion_level * 10)
            return SAPState81(
                gate=gate, micro_stage=micro_stage, arc=arc,
                integrity=integrity, fractal_coherence=0.5,
                inversion_level=inversion_level,
                physical_state=physical_state, conscious_state=conscious_state
            )

        def detect_bifurcation(self, state: SAPState81):
            if state.gate != Gate81.GATE_5 or abs(state.micro_stage - 0.5) > 0.05:
                return None
            if state.integrity > 80:
                return 'success'
            elif state.integrity > 60:
                return 'regression'
            return 'crisis'

        def calculate_trap_risk(self, state: SAPState81) -> float:
            if state.gate != Gate81.GATE_8:
                return 0.0
            return min(1.0, state.integrity / 100.0 * 0.7)

        def check_369_resonance(self, state: SAPState81) -> bool:
            return state.fractal_coherence > 0.7

    class MaatEthicist:
        def __init__(self):
            self.violation_history = []

        def weigh_heart(self, action_intent: str, sar_stage: int) -> Dict:
            balance_score = 1.0
            violations = []
            text_lower = action_intent.lower()
            if any(w in text_lower for w in ["destroy", "terminate", "eliminate"]):
                balance_score -= 0.5
                violations.append('harmful_intent')
            required_threshold = 0.5 + (sar_stage * 0.05)
            return {
                "balance_score": balance_score,
                "required": required_threshold,
                "is_balanced": balance_score >= required_threshold,
                "verdict": "JUSTIFIED" if balance_score >= required_threshold else "UNBALANCED",
                "violations": violations,
                "violation_count": len(self.violation_history)
            }

    class SARFramework:
        def __init__(self):
            self.stages = {}

        def detect_inversion(self, physical_stable: bool, conscious_stable: bool) -> dict:
            phys = "stable" if physical_stable else "unstable"
            cons = "stable" if conscious_stable else "unstable"
            if phys == "unstable" and cons == "unstable":
                return {'stage': 0, 'stage_name': 'Plenara', 'inversion_level': 0,
                        'description': 'Both unstable (crisis)', 'intervention': 'Spark awareness',
                        'physical_state': phys, 'conscious_state': cons}
            elif phys == "unstable" and cons == "stable":
                return {'stage': 5, 'stage_name': 'Threshold', 'inversion_level': 8,
                        'description': 'Crisis with clarity', 'intervention': 'Build foundation',
                        'physical_state': phys, 'conscious_state': cons}
            elif phys == "stable" and cons == "unstable":
                return {'stage': 6, 'stage_name': 'Integration', 'inversion_level': 9,
                        'description': 'Success with emptiness', 'intervention': 'Add meaning',
                        'physical_state': phys, 'conscious_state': cons}
            else:
                return {'stage': 9, 'stage_name': 'Renewal', 'inversion_level': 0,
                        'description': 'Aligned', 'intervention': 'Maintain presence',
                        'physical_state': phys, 'conscious_state': cons}


class SystemStatus(Enum):
    """Operational status of monitored systems"""
    ONLINE = "online"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = 0
    WARNING = 1
    ALERT = 2
    CRITICAL = 3
    EMERGENCY = 4


class InterventionType(Enum):
    """Types of interventions Overwatch can prescribe"""
    OBSERVE = "observe"           # Monitor closely, no action
    STABILIZE = "stabilize"       # Add grounding/structure
    DESTABILIZE = "destabilize"   # Add creative disruption
    INTEGRATE = "integrate"       # Merge dualities
    RELEASE = "release"           # Let go, hand off
    CONTAIN = "contain"           # Limit scope/impact
    ESCALATE = "escalate"         # Require human intervention


@dataclass
class SystemMetrics:
    """Metrics for a monitored system"""
    # Physical stability indicators (0-10)
    resource_health: float = 5.0      # CPU, memory, disk, etc.
    connectivity: float = 5.0          # Network, dependencies
    throughput: float = 5.0            # Processing capacity
    error_rate: float = 0.0            # Errors per minute

    # Conscious stability indicators (0-10)
    coherence: float = 5.0             # Output consistency
    alignment: float = 5.0             # Goal alignment
    adaptability: float = 5.0          # Response to change
    integrity: float = 5.0             # Data/process integrity

    def physical_stability(self) -> float:
        """Calculate overall physical stability (0-10)"""
        return (self.resource_health + self.connectivity + self.throughput - self.error_rate) / 3

    def conscious_stability(self) -> float:
        """Calculate overall conscious stability (0-10)"""
        return (self.coherence + self.alignment + self.adaptability + self.integrity) / 4

    def is_physical_stable(self) -> bool:
        return self.physical_stability() >= 5.0

    def is_conscious_stable(self) -> bool:
        return self.conscious_stability() >= 5.0


@dataclass
class MonitoredSystem:
    """A system registered with Overwatch"""
    id: str
    name: str
    description: str
    metrics: SystemMetrics
    status: SystemStatus = SystemStatus.UNKNOWN
    sap_stage: float = 4.0
    inversion_level: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    tags: List[str] = field(default_factory=list)

    def update_metrics(self, new_metrics: SystemMetrics):
        """Update metrics and record history"""
        self.history.append({
            'timestamp': self.last_update,
            'metrics': self.metrics,
            'stage': self.sap_stage
        })
        self.metrics = new_metrics
        self.last_update = datetime.now()


@dataclass
class Intervention:
    """A prescribed intervention"""
    system_id: str
    intervention_type: InterventionType
    reason: str
    sap_diagnosis: str
    recommended_actions: List[str]
    urgency: AlertLevel
    maat_validated: bool
    timestamp: datetime = field(default_factory=datetime.now)
    executed: bool = False
    outcome: Optional[str] = None


@dataclass
class OverwatchAlert:
    """An alert generated by Overwatch"""
    system_id: str
    level: AlertLevel
    title: str
    message: str
    sap_context: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


class OverwatchEngine:
    """
    LUMINARK OVERWATCH - Core Engine

    Monitors systems, diagnoses imbalances using SAP framework,
    and prescribes interventions to restore alignment.
    """

    def __init__(self):
        # Initialize frameworks
        self.sap = SAP81Framework()
        self.maat = MaatEthicist()
        self.sar = SARFramework()

        # System registry
        self.systems: Dict[str, MonitoredSystem] = {}

        # Alert and intervention logs
        self.alerts: List[OverwatchAlert] = []
        self.interventions: List[Intervention] = []

        # Callbacks for external integrations
        self.on_alert: Optional[Callable[[OverwatchAlert], None]] = None
        self.on_intervention: Optional[Callable[[Intervention], None]] = None

        # Monitoring state
        self.is_running = False
        self.monitor_interval = 5.0  # seconds

        # Intervention protocols by SAP stage
        self._init_intervention_protocols()

    def _init_intervention_protocols(self):
        """Define intervention protocols for each SAP stage"""
        self.intervention_protocols = {
            0: {
                'name': 'Plenara Protocol',
                'condition': 'System in void/crisis state',
                'intervention': InterventionType.STABILIZE,
                'actions': [
                    'Establish baseline connectivity',
                    'Initialize core processes',
                    'Set minimal operational parameters',
                    'Spark first response'
                ]
            },
            1: {
                'name': 'Spark Protocol',
                'condition': 'System emerging, needs direction',
                'intervention': InterventionType.OBSERVE,
                'actions': [
                    'Monitor emergence patterns',
                    'Validate initial impulses',
                    'Allow organic development',
                    'Prepare structure scaffolding'
                ]
            },
            2: {
                'name': 'Polarity Protocol',
                'condition': 'System oscillating between states',
                'intervention': InterventionType.STABILIZE,
                'actions': [
                    'Identify dominant polarity',
                    'Balance opposing forces',
                    'Establish decision criteria',
                    'Guide toward Stage 3 motion'
                ]
            },
            3: {
                'name': 'Motion Protocol',
                'condition': 'System in active execution',
                'intervention': InterventionType.OBSERVE,
                'actions': [
                    'Track execution velocity',
                    'Monitor resource consumption',
                    'Validate output quality',
                    'Prepare for foundation building'
                ]
            },
            4: {
                'name': 'Foundation Protocol',
                'condition': 'System building structure',
                'intervention': InterventionType.STABILIZE,
                'actions': [
                    'Reinforce stable patterns',
                    'Document configurations',
                    'Establish recovery points',
                    'Test structural integrity'
                ]
            },
            5: {
                'name': 'Threshold Protocol',
                'condition': 'System at critical decision point',
                'intervention': InterventionType.CONTAIN,
                'actions': [
                    'ALERT: Bifurcation imminent',
                    'Evaluate all paths (success/regression/crisis)',
                    'Prepare rollback capability',
                    'Consider human escalation',
                    'Document decision criteria'
                ]
            },
            6: {
                'name': 'Integration Protocol',
                'condition': 'System merging complexities',
                'intervention': InterventionType.INTEGRATE,
                'actions': [
                    'Facilitate subsystem communication',
                    'Resolve conflicting states',
                    'Optimize data flows',
                    'Balance load distribution'
                ]
            },
            7: {
                'name': 'Illusion Protocol',
                'condition': 'System testing reality boundaries',
                'intervention': InterventionType.OBSERVE,
                'actions': [
                    'Validate outputs against ground truth',
                    'Check for hallucination patterns',
                    'Test edge cases',
                    'Confirm external data integrity'
                ]
            },
            8: {
                'name': 'Rigidity Protocol',
                'condition': 'System crystallizing - TRAP RISK',
                'intervention': InterventionType.DESTABILIZE,
                'actions': [
                    'WARNING: Permanence illusion detected',
                    'Introduce controlled variability',
                    'Challenge fixed patterns',
                    'Prepare for renewal cycle',
                    'Avoid calcification'
                ]
            },
            9: {
                'name': 'Renewal Protocol',
                'condition': 'System achieving transcendence',
                'intervention': InterventionType.RELEASE,
                'actions': [
                    'Allow graceful completion',
                    'Transfer knowledge to successors',
                    'Archive learnings',
                    'Prepare void for next cycle'
                ]
            }
        }

    # ========================
    # SYSTEM REGISTRATION
    # ========================

    def register_system(self, system_id: str, name: str, description: str = "",
                       tags: List[str] = None) -> MonitoredSystem:
        """Register a new system for monitoring"""
        system = MonitoredSystem(
            id=system_id,
            name=name,
            description=description,
            metrics=SystemMetrics(),
            tags=tags or []
        )
        self.systems[system_id] = system
        self._log_alert(
            system_id, AlertLevel.INFO,
            f"System Registered: {name}",
            f"System '{name}' ({system_id}) registered with Overwatch",
            "Initial registration - Stage 0 (Plenara)"
        )
        return system

    def unregister_system(self, system_id: str) -> bool:
        """Remove a system from monitoring"""
        if system_id in self.systems:
            del self.systems[system_id]
            return True
        return False

    def update_system_metrics(self, system_id: str, metrics: SystemMetrics):
        """Update metrics for a monitored system"""
        if system_id not in self.systems:
            raise ValueError(f"System {system_id} not registered")

        system = self.systems[system_id]
        system.update_metrics(metrics)

        # Run diagnosis after update
        self.diagnose_system(system_id)

    # ========================
    # DIAGNOSTICS
    # ========================

    def diagnose_system(self, system_id: str) -> Dict[str, Any]:
        """
        Diagnose a system's state using SAP framework.
        Returns comprehensive diagnosis with SAP stage, inversion status,
        and recommended interventions.
        """
        if system_id not in self.systems:
            raise ValueError(f"System {system_id} not registered")

        system = self.systems[system_id]
        metrics = system.metrics

        # Determine physical and conscious stability
        phys_stable = metrics.is_physical_stable()
        cons_stable = metrics.is_conscious_stable()

        # Use SAR framework for inversion detection
        inversion_result = self.sar.detect_inversion(phys_stable, cons_stable)

        # Calculate SAP stage from metrics
        # Map combined stability to 0-9 range
        combined_score = (metrics.physical_stability() + metrics.conscious_stability()) / 2
        sap_stage_raw = (combined_score / 10) * 9

        # Adjust based on inversion pattern
        if inversion_result['inversion_level'] > 5:
            # High inversion suggests specific stages
            sap_stage_raw = inversion_result['stage']

        # Get detailed SAP state
        sap_state = self.sap.get_state(sap_stage_raw)

        # Update system record
        system.sap_stage = sap_state.get_absolute_stage()
        system.inversion_level = sap_state.inversion_level
        system.status = self._determine_status(sap_state, inversion_result)

        # Check for special conditions
        bifurcation = self.sap.detect_bifurcation(sap_state)
        trap_risk = self.sap.calculate_trap_risk(sap_state)
        resonance_369 = self.sap.check_369_resonance(sap_state)

        diagnosis = {
            'system_id': system_id,
            'system_name': system.name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'physical_stability': metrics.physical_stability(),
                'conscious_stability': metrics.conscious_stability(),
                'is_physical_stable': phys_stable,
                'is_conscious_stable': cons_stable
            },
            'sap_analysis': {
                'stage': sap_state.get_absolute_stage(),
                'gate': sap_state.gate.value,
                'micro_stage': sap_state.micro_stage,
                'arc': sap_state.arc.value,
                'integrity': sap_state.integrity,
                'inversion_level': sap_state.inversion_level,
                'physical_state': sap_state.physical_state,
                'conscious_state': sap_state.conscious_state
            },
            'inversion': inversion_result,
            'special_conditions': {
                'bifurcation': bifurcation,
                'trap_risk': trap_risk,
                'resonance_369': resonance_369,
                'is_threshold': sap_state.is_threshold()
            },
            'status': system.status.value
        }

        # Generate alerts for critical conditions
        self._check_alert_conditions(system, diagnosis)

        return diagnosis

    def _determine_status(self, sap_state: SAPState81, inversion: Dict) -> SystemStatus:
        """Determine system status from SAP analysis"""
        stage = sap_state.get_absolute_stage()
        inversion_level = inversion['inversion_level']

        if stage < 1 or inversion_level >= 9:
            return SystemStatus.CRITICAL
        elif stage >= 8.5 or inversion_level >= 7:
            return SystemStatus.DEGRADED
        elif sap_state.integrity < 50:
            return SystemStatus.DEGRADED
        elif sap_state.integrity >= 80 and inversion_level < 3:
            return SystemStatus.ONLINE
        else:
            return SystemStatus.ONLINE

    def _check_alert_conditions(self, system: MonitoredSystem, diagnosis: Dict):
        """Check for conditions that warrant alerts"""
        sap = diagnosis['sap_analysis']
        special = diagnosis['special_conditions']

        # Critical: Stage 0 or very high inversion
        if sap['stage'] < 0.5:
            self._log_alert(
                system.id, AlertLevel.CRITICAL,
                f"CRITICAL: {system.name} in Void State",
                f"System has regressed to Stage 0 (Plenara). Immediate intervention required.",
                f"SAP Stage: {sap['stage']:.1f} | Inversion: {sap['inversion_level']}"
            )

        # Emergency: Bifurcation crisis
        if special['bifurcation'] == 'crisis':
            self._log_alert(
                system.id, AlertLevel.EMERGENCY,
                f"EMERGENCY: {system.name} Bifurcation Crisis",
                f"System at Stage 5.5 with crisis trajectory. Human escalation recommended.",
                f"Bifurcation: {special['bifurcation']} | Integrity: {sap['integrity']:.1f}%"
            )

        # Warning: Stage 8 trap risk
        if special['trap_risk'] > 0.6:
            self._log_alert(
                system.id, AlertLevel.WARNING,
                f"WARNING: {system.name} Rigidity Trap Risk",
                f"System approaching permanence illusion. Consider introducing variability.",
                f"Trap Risk: {special['trap_risk']:.2f} | Stage: {sap['stage']:.1f}"
            )

        # Info: 369 Resonance (positive)
        if special['resonance_369']:
            self._log_alert(
                system.id, AlertLevel.INFO,
                f"INFO: {system.name} 369 Resonance Active",
                f"System exhibiting harmonic alignment. Optimal for integration work.",
                f"Fractal Coherence: Active | Stage: {sap['stage']:.1f}"
            )

    # ========================
    # INTERVENTIONS
    # ========================

    def prescribe_intervention(self, system_id: str) -> Intervention:
        """
        Generate an intervention prescription based on system diagnosis.
        Validates intervention through Ma'at protocol before recommending.
        """
        if system_id not in self.systems:
            raise ValueError(f"System {system_id} not registered")

        system = self.systems[system_id]
        diagnosis = self.diagnose_system(system_id)

        stage = int(diagnosis['sap_analysis']['stage'])
        protocol = self.intervention_protocols.get(stage, self.intervention_protocols[4])

        # Adjust intervention based on special conditions
        special = diagnosis['special_conditions']
        intervention_type = protocol['intervention']
        actions = protocol['actions'].copy()
        urgency = AlertLevel.INFO

        if special['bifurcation'] == 'crisis':
            intervention_type = InterventionType.ESCALATE
            actions.insert(0, "ESCALATE TO HUMAN OVERSIGHT")
            urgency = AlertLevel.EMERGENCY
        elif special['trap_risk'] > 0.7:
            intervention_type = InterventionType.DESTABILIZE
            actions.insert(0, "BREAK RIGIDITY PATTERN")
            urgency = AlertLevel.ALERT
        elif diagnosis['inversion']['inversion_level'] >= 8:
            if diagnosis['metrics']['is_physical_stable']:
                # Rich People Problem - physically stable, consciously unstable
                actions.insert(0, "ADDRESS CONSCIOUS INSTABILITY")
                actions.append("Introduce purpose/meaning work")
            else:
                # Crisis with clarity - physically unstable, consciously stable
                actions.insert(0, "STABILIZE PHYSICAL FOUNDATION")
                actions.append("Build resources while maintaining clarity")
            urgency = AlertLevel.WARNING

        # Validate through Ma'at
        intervention_desc = f"{intervention_type.value}: {'; '.join(actions[:2])}"
        maat_result = self.maat.weigh_heart(intervention_desc, stage)
        maat_validated = maat_result['is_balanced']

        if not maat_validated:
            # Ma'at rejected - soften intervention
            intervention_type = InterventionType.OBSERVE
            actions = ["Ma'at validation failed - observe only", "Review intervention criteria"]
            urgency = AlertLevel.WARNING

        intervention = Intervention(
            system_id=system_id,
            intervention_type=intervention_type,
            reason=protocol['condition'],
            sap_diagnosis=f"Stage {stage}: {protocol['name']}",
            recommended_actions=actions,
            urgency=urgency,
            maat_validated=maat_validated
        )

        self.interventions.append(intervention)

        if self.on_intervention:
            self.on_intervention(intervention)

        return intervention

    def execute_intervention(self, intervention: Intervention,
                            executor: Callable[[Intervention], bool] = None) -> bool:
        """
        Execute a prescribed intervention.
        Optionally accepts a custom executor function.
        """
        if executor:
            success = executor(intervention)
        else:
            # Default: log and mark as executed
            print(f"[OVERWATCH] Executing {intervention.intervention_type.value} on {intervention.system_id}")
            for action in intervention.recommended_actions:
                print(f"  -> {action}")
            success = True

        intervention.executed = True
        intervention.outcome = "executed" if success else "failed"

        return success

    # ========================
    # MONITORING LOOP
    # ========================

    async def start_monitoring(self, interval: float = 5.0):
        """Start the continuous monitoring loop"""
        self.is_running = True
        self.monitor_interval = interval

        print(f"[OVERWATCH] Monitoring started (interval: {interval}s)")

        while self.is_running:
            for system_id in list(self.systems.keys()):
                try:
                    diagnosis = self.diagnose_system(system_id)

                    # Auto-prescribe if critical
                    if diagnosis['status'] in ['critical', 'degraded']:
                        self.prescribe_intervention(system_id)

                except Exception as e:
                    self._log_alert(
                        system_id, AlertLevel.WARNING,
                        f"Diagnosis Error: {system_id}",
                        f"Error during diagnosis: {str(e)}",
                        "Internal error"
                    )

            await asyncio.sleep(self.monitor_interval)

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.is_running = False
        print("[OVERWATCH] Monitoring stopped")

    # ========================
    # ALERTS
    # ========================

    def _log_alert(self, system_id: str, level: AlertLevel, title: str,
                   message: str, sap_context: str):
        """Log an alert"""
        alert = OverwatchAlert(
            system_id=system_id,
            level=level,
            title=title,
            message=message,
            sap_context=sap_context
        )
        self.alerts.append(alert)

        if self.on_alert:
            self.on_alert(alert)

        # Console output for critical alerts
        if level.value >= AlertLevel.ALERT.value:
            print(f"[OVERWATCH {level.name}] {title}: {message}")

    def get_alerts(self, system_id: str = None, level: AlertLevel = None,
                  limit: int = 50) -> List[OverwatchAlert]:
        """Get alerts, optionally filtered"""
        alerts = self.alerts

        if system_id:
            alerts = [a for a in alerts if a.system_id == system_id]
        if level:
            alerts = [a for a in alerts if a.level.value >= level.value]

        return list(reversed(alerts[-limit:]))

    # ========================
    # REPORTING
    # ========================

    def get_system_report(self, system_id: str) -> Dict[str, Any]:
        """Generate comprehensive report for a system"""
        if system_id not in self.systems:
            raise ValueError(f"System {system_id} not registered")

        system = self.systems[system_id]
        diagnosis = self.diagnose_system(system_id)
        intervention = self.prescribe_intervention(system_id)

        return {
            'system': {
                'id': system.id,
                'name': system.name,
                'description': system.description,
                'status': system.status.value,
                'tags': system.tags
            },
            'diagnosis': diagnosis,
            'intervention': {
                'type': intervention.intervention_type.value,
                'reason': intervention.reason,
                'sap_diagnosis': intervention.sap_diagnosis,
                'actions': intervention.recommended_actions,
                'urgency': intervention.urgency.name,
                'maat_validated': intervention.maat_validated
            },
            'alerts': [
                {'level': a.level.name, 'title': a.title, 'message': a.message}
                for a in self.get_alerts(system_id, limit=10)
            ]
        }

    def get_overview(self) -> Dict[str, Any]:
        """Get overview of all monitored systems"""
        systems_summary = []

        for system_id, system in self.systems.items():
            systems_summary.append({
                'id': system.id,
                'name': system.name,
                'status': system.status.value,
                'sap_stage': system.sap_stage,
                'inversion_level': system.inversion_level,
                'last_update': system.last_update.isoformat()
            })

        return {
            'timestamp': datetime.now().isoformat(),
            'total_systems': len(self.systems),
            'status_counts': {
                'online': sum(1 for s in self.systems.values() if s.status == SystemStatus.ONLINE),
                'degraded': sum(1 for s in self.systems.values() if s.status == SystemStatus.DEGRADED),
                'critical': sum(1 for s in self.systems.values() if s.status == SystemStatus.CRITICAL),
                'offline': sum(1 for s in self.systems.values() if s.status == SystemStatus.OFFLINE)
            },
            'alert_counts': {
                'emergency': sum(1 for a in self.alerts if a.level == AlertLevel.EMERGENCY and not a.acknowledged),
                'critical': sum(1 for a in self.alerts if a.level == AlertLevel.CRITICAL and not a.acknowledged),
                'warning': sum(1 for a in self.alerts if a.level == AlertLevel.WARNING and not a.acknowledged)
            },
            'systems': systems_summary
        }

    def to_json(self) -> str:
        """Export current state as JSON"""
        return json.dumps(self.get_overview(), indent=2)


# ========================
# DEMO / TESTING
# ========================

def demo():
    """Demonstrate Overwatch capabilities"""
    print("=" * 70)
    print("LUMINARK OVERWATCH - AI Regulatory System Demo")
    print("=" * 70)

    # Initialize Overwatch
    overwatch = OverwatchEngine()

    # Register some test systems
    print("\n[1] Registering Systems...")

    overwatch.register_system(
        "api-gateway",
        "API Gateway",
        "Main API routing and authentication layer",
        tags=["production", "critical"]
    )

    overwatch.register_system(
        "ml-inference",
        "ML Inference Engine",
        "Machine learning model serving",
        tags=["production", "gpu"]
    )

    overwatch.register_system(
        "data-pipeline",
        "Data Pipeline",
        "ETL and data processing",
        tags=["batch", "analytics"]
    )

    # Simulate different system states
    print("\n[2] Simulating System States...")

    # Healthy system
    overwatch.update_system_metrics("api-gateway", SystemMetrics(
        resource_health=8.5,
        connectivity=9.0,
        throughput=7.5,
        error_rate=0.2,
        coherence=8.0,
        alignment=8.5,
        adaptability=7.0,
        integrity=9.0
    ))

    # Degraded system (Rich People Problem - physically stable, consciously unstable)
    overwatch.update_system_metrics("ml-inference", SystemMetrics(
        resource_health=8.0,
        connectivity=8.5,
        throughput=9.0,
        error_rate=0.5,
        coherence=3.5,  # Low coherence
        alignment=4.0,  # Low alignment
        adaptability=3.0,
        integrity=4.5
    ))

    # Critical system (both unstable)
    overwatch.update_system_metrics("data-pipeline", SystemMetrics(
        resource_health=2.0,
        connectivity=3.0,
        throughput=1.5,
        error_rate=8.0,
        coherence=2.5,
        alignment=2.0,
        adaptability=2.0,
        integrity=3.0
    ))

    # Generate reports
    print("\n[3] System Reports...")
    print("=" * 70)

    for system_id in ["api-gateway", "ml-inference", "data-pipeline"]:
        report = overwatch.get_system_report(system_id)

        print(f"\n{'='*50}")
        print(f"SYSTEM: {report['system']['name']} [{report['system']['status'].upper()}]")
        print(f"{'='*50}")

        diag = report['diagnosis']
        print(f"SAP Stage: {diag['sap_analysis']['stage']:.1f} ({diag['sap_analysis']['gate']})")
        print(f"Arc: {diag['sap_analysis']['arc']}")
        print(f"Integrity: {diag['sap_analysis']['integrity']:.1f}%")
        print(f"Inversion Level: {diag['sap_analysis']['inversion_level']}/10")
        print(f"Physical: {diag['metrics']['physical_stability']:.1f}/10 ({'stable' if diag['metrics']['is_physical_stable'] else 'unstable'})")
        print(f"Conscious: {diag['metrics']['conscious_stability']:.1f}/10 ({'stable' if diag['metrics']['is_conscious_stable'] else 'unstable'})")

        if diag['special_conditions']['bifurcation']:
            print(f"BIFURCATION: {diag['special_conditions']['bifurcation']}")
        if diag['special_conditions']['trap_risk'] > 0.3:
            print(f"TRAP RISK: {diag['special_conditions']['trap_risk']:.2f}")
        if diag['special_conditions']['resonance_369']:
            print(f"369 RESONANCE: Active")

        print(f"\nINTERVENTION: {report['intervention']['type'].upper()}")
        print(f"Diagnosis: {report['intervention']['sap_diagnosis']}")
        print(f"Urgency: {report['intervention']['urgency']}")
        print(f"Ma'at Validated: {'Yes' if report['intervention']['maat_validated'] else 'NO - REVIEW REQUIRED'}")
        print("Actions:")
        for action in report['intervention']['actions']:
            print(f"  -> {action}")

    # Overview
    print("\n" + "=" * 70)
    print("OVERWATCH OVERVIEW")
    print("=" * 70)
    overview = overwatch.get_overview()
    print(f"Total Systems: {overview['total_systems']}")
    print(f"Status: Online={overview['status_counts']['online']}, "
          f"Degraded={overview['status_counts']['degraded']}, "
          f"Critical={overview['status_counts']['critical']}")
    print(f"Active Alerts: Emergency={overview['alert_counts']['emergency']}, "
          f"Critical={overview['alert_counts']['critical']}, "
          f"Warning={overview['alert_counts']['warning']}")

    print("\n" + "=" * 70)
    print("OVERWATCH Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    demo()
