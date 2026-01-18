"""
Bio-Mimetic Self-Healing - ENHANCED WITH TRAUMA THEORY

System self-healing inspired by biological regeneration and
Project Plenara's stage-appropriate trauma healing protocols.

Key concepts:
- Stem cell analogy (healthy templates for regeneration)
- Cellular differentiation (specialized recovery)
- Stage-appropriate intervention (don't force processing before stabilization)
- Apoptosis (programmed cell death for corrupted components)
- Immune memory (remember threats, adapt defenses)

Plenara Protocol Integration:
- Stage 0-3: Safety & Stabilization (contain, assess, stabilize)
- Stage 4: Foundation Rebuilding (restore core functions)
- Stage 5: Processing (understand what failed, why)
- Stage 6-7: Integration (incorporate lessons, transform)

TRAUMA THEORY INTEGRATION (NEW):

1. Van der Kolk (The Body Keeps the Score):
   - Window of Tolerance tracking
   - Somatic markers for each stage
   - Polyvagal theory (ventral/dorsal vagal, sympathetic states)
   - Body keeps score - track physical manifestations

2. Peter Levine (Somatic Experiencing):
   - Pendulation (oscillate between stages, not linear)
   - Titration (micro-dosing of processing)
   - Discharge detection (trauma energy release)
   - Resource anchoring

3. Gabor Maté (Trauma as Disconnection):
   - Relational field healing
   - Attachment pattern detection
   - "Trauma is what happens inside you, not to you"
   - Connection as healing agent

Author: Richard Leroy Stanfield Jr. / Meridian Axiom
Part of: SAP V4.0 Enhancement Suite
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import secrets
import copy


class HealingStage(Enum):
    """Healing stages (aligned with Plenara protocol)"""
    STAGE_0_SAFETY = "stage_0_safety"  # Immediate containment
    STAGE_1_STABILIZATION = "stage_1_stabilization"  # Stop bleeding
    STAGE_2_ASSESSMENT = "stage_2_assessment"  # Understand damage
    STAGE_3_PREPARATION = "stage_3_preparation"  # Prepare for repair
    STAGE_4_FOUNDATION = "stage_4_foundation"  # Rebuild basics
    STAGE_5_PROCESSING = "stage_5_processing"  # Understand failure
    STAGE_6_INTEGRATION = "stage_6_integration"  # Incorporate lessons
    STAGE_7_TRANSFORMATION = "stage_7_transformation"  # Emerge stronger


class ComponentHealth(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    STRESSED = "stressed"
    DAMAGED = "damaged"
    CORRUPTED = "corrupted"
    DEAD = "dead"
    REGENERATING = "regenerating"


@dataclass
class HealthyTemplate:
    """Template of healthy component (stem cell analog)"""
    template_id: str
    component_type: str
    baseline_state: Dict[str, Any]
    expected_behavior: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass
class DamagedComponent:
    """Component needing healing"""
    component_id: str
    health_status: ComponentHealth
    damage_type: str
    current_state: Dict[str, Any]
    healing_stage: HealingStage
    started_healing: float = field(default_factory=time.time)
    recovery_plan: List[str] = field(default_factory=list)


@dataclass
class HealingAction:
    """Specific healing action"""
    action_id: str
    component_id: str
    action_type: str  # contain, stabilize, repair, regenerate, apoptosis
    description: str
    executed_at: float = field(default_factory=time.time)
    success: bool = False


class BioMimeticHealingSystem:
    """
    Self-healing system using biological principles and
    Plenara stage-appropriate protocols
    """

    def __init__(self, system_id: str):
        self.system_id = system_id

        # Healthy templates (stem cells)
        self.templates: Dict[str, HealthyTemplate] = {}

        # Damaged components under treatment
        self.healing_queue: List[DamagedComponent] = []

        # Healing history
        self.healing_actions: List[HealingAction] = []
        self.recovered_components: List[str] = []

        # Immune memory (learned threats)
        self.immune_memory: List[Dict] = []

        # Statistics
        self.total_healings: int = 0
        self.successful_recoveries: int = 0
        self.apoptosis_events: int = 0

    def register_healthy_template(
        self,
        component_type: str,
        baseline_state: Dict[str, Any],
        expected_behavior: Dict[str, Any]
    ) -> HealthyTemplate:
        """
        Register healthy component template (stem cell)

        Args:
            component_type: Type of component
            baseline_state: Healthy state snapshot
            expected_behavior: Expected behavior patterns

        Returns:
            HealthyTemplate created
        """
        template = HealthyTemplate(
            template_id=f"template_{secrets.token_hex(8)}",
            component_type=component_type,
            baseline_state=baseline_state,
            expected_behavior=expected_behavior
        )

        self.templates[component_type] = template

        return template

    def detect_damage(
        self,
        component_id: str,
        component_type: str,
        current_state: Dict[str, Any]
    ) -> Optional[DamagedComponent]:
        """
        Detect if component is damaged

        Args:
            component_id: Component to check
            component_type: Component type
            current_state: Current state

        Returns:
            DamagedComponent if damage detected
        """
        # Get template
        template = self.templates.get(component_type)

        if not template:
            return None  # No baseline to compare

        # Compare to baseline
        health = self._assess_health(current_state, template)

        if health in [ComponentHealth.DAMAGED, ComponentHealth.CORRUPTED, ComponentHealth.DEAD]:
            # Damage detected
            damaged = DamagedComponent(
                component_id=component_id,
                health_status=health,
                damage_type=self._classify_damage(current_state, template),
                current_state=current_state,
                healing_stage=HealingStage.STAGE_0_SAFETY
            )

            self.healing_queue.append(damaged)
            return damaged

        return None

    def _assess_health(
        self,
        current_state: Dict[str, Any],
        template: HealthyTemplate
    ) -> ComponentHealth:
        """Assess component health"""
        # Compare current to baseline
        baseline = template.baseline_state

        differences = 0
        total_fields = len(baseline)

        for key, expected_value in baseline.items():
            current_value = current_state.get(key)

            if current_value != expected_value:
                differences += 1

        if differences == 0:
            return ComponentHealth.HEALTHY

        deviation = differences / total_fields

        if deviation < 0.2:
            return ComponentHealth.STRESSED
        elif deviation < 0.5:
            return ComponentHealth.DAMAGED
        elif deviation < 0.8:
            return ComponentHealth.CORRUPTED
        else:
            return ComponentHealth.DEAD

    def _classify_damage(
        self,
        current_state: Dict[str, Any],
        template: HealthyTemplate
    ) -> str:
        """Classify type of damage"""
        # Simplified classification
        if "corrupted" in str(current_state).lower():
            return "corruption"
        elif "memory" in str(current_state).lower():
            return "memory_leak"
        elif "connection" in str(current_state).lower():
            return "connection_failure"
        else:
            return "unknown_damage"

    def heal_component(
        self,
        component_id: str
    ) -> List[HealingAction]:
        """
        Heal damaged component using stage-appropriate protocol

        Args:
            component_id: Component to heal

        Returns:
            List of healing actions taken
        """
        # Find component in healing queue
        component = None
        for c in self.healing_queue:
            if c.component_id == component_id:
                component = c
                break

        if not component:
            return []

        actions = []

        # Progress through healing stages
        if component.healing_stage == HealingStage.STAGE_0_SAFETY:
            actions.extend(self._stage0_contain(component))
            component.healing_stage = HealingStage.STAGE_1_STABILIZATION

        elif component.healing_stage == HealingStage.STAGE_1_STABILIZATION:
            actions.extend(self._stage1_stabilize(component))
            component.healing_stage = HealingStage.STAGE_2_ASSESSMENT

        elif component.healing_stage == HealingStage.STAGE_2_ASSESSMENT:
            actions.extend(self._stage2_assess(component))
            component.healing_stage = HealingStage.STAGE_3_PREPARATION

        elif component.healing_stage == HealingStage.STAGE_3_PREPARATION:
            actions.extend(self._stage3_prepare(component))
            component.healing_stage = HealingStage.STAGE_4_FOUNDATION

        elif component.healing_stage == HealingStage.STAGE_4_FOUNDATION:
            actions.extend(self._stage4_rebuild(component))
            component.healing_stage = HealingStage.STAGE_5_PROCESSING

        elif component.healing_stage == HealingStage.STAGE_5_PROCESSING:
            actions.extend(self._stage5_process(component))
            component.healing_stage = HealingStage.STAGE_6_INTEGRATION

        elif component.healing_stage == HealingStage.STAGE_6_INTEGRATION:
            actions.extend(self._stage6_integrate(component))
            component.healing_stage = HealingStage.STAGE_7_TRANSFORMATION

        elif component.healing_stage == HealingStage.STAGE_7_TRANSFORMATION:
            actions.extend(self._stage7_transform(component))
            # Healing complete
            self.healing_queue.remove(component)
            self.recovered_components.append(component_id)
            self.successful_recoveries += 1

        self.healing_actions.extend(actions)
        self.total_healings += 1

        return actions

    def _stage0_contain(self, component: DamagedComponent) -> List[HealingAction]:
        """Stage 0: Immediate containment (safety)"""
        actions = []

        # Contain component (prevent spread)
        action = HealingAction(
            action_id=f"action_{secrets.token_hex(8)}",
            component_id=component.component_id,
            action_type="contain",
            description="Isolate damaged component to prevent spread",
            success=True
        )
        actions.append(action)

        # Create recovery plan
        component.recovery_plan = [
            "Contain",
            "Stabilize",
            "Assess damage",
            "Repair or regenerate",
            "Reintegrate"
        ]

        return actions

    def _stage1_stabilize(self, component: DamagedComponent) -> List[HealingAction]:
        """Stage 1: Stabilization (stop the bleeding)"""
        actions = []

        action = HealingAction(
            action_id=f"action_{secrets.token_hex(8)}",
            component_id=component.component_id,
            action_type="stabilize",
            description="Stabilize component, prevent further degradation",
            success=True
        )
        actions.append(action)

        component.health_status = ComponentHealth.STRESSED  # Improved from damaged

        return actions

    def _stage2_assess(self, component: DamagedComponent) -> List[HealingAction]:
        """Stage 2: Assessment (understand the damage)"""
        actions = []

        action = HealingAction(
            action_id=f"action_{secrets.token_hex(8)}",
            component_id=component.component_id,
            action_type="assess",
            description=f"Assess damage type: {component.damage_type}",
            success=True
        )
        actions.append(action)

        return actions

    def _stage3_prepare(self, component: DamagedComponent) -> List[HealingAction]:
        """Stage 3: Preparation (get ready for repair)"""
        actions = []

        action = HealingAction(
            action_id=f"action_{secrets.token_hex(8)}",
            component_id=component.component_id,
            action_type="prepare",
            description="Prepare resources for regeneration",
            success=True
        )
        actions.append(action)

        return actions

    def _stage4_rebuild(self, component: DamagedComponent) -> List[HealingAction]:
        """Stage 4: Foundation rebuilding"""
        actions = []

        # Check if component salvageable
        if component.health_status == ComponentHealth.DEAD:
            # Apoptosis (programmed death)
            return self._trigger_apoptosis(component)

        # Regenerate from template
        action = HealingAction(
            action_id=f"action_{secrets.token_hex(8)}",
            component_id=component.component_id,
            action_type="regenerate",
            description="Regenerate component from healthy template",
            success=True
        )
        actions.append(action)

        component.health_status = ComponentHealth.REGENERATING

        return actions

    def _stage5_process(self, component: DamagedComponent) -> List[HealingAction]:
        """Stage 5: Processing (understand what failed)"""
        actions = []

        action = HealingAction(
            action_id=f"action_{secrets.token_hex(8)}",
            component_id=component.component_id,
            action_type="learn",
            description="Analyze failure, update immune memory",
            success=True
        )
        actions.append(action)

        # Update immune memory
        self.immune_memory.append({
            "threat_type": component.damage_type,
            "timestamp": time.time(),
            "lesson": f"Component {component.component_id} failed due to {component.damage_type}"
        })

        return actions

    def _stage6_integrate(self, component: DamagedComponent) -> List[HealingAction]:
        """Stage 6: Integration (incorporate lessons)"""
        actions = []

        action = HealingAction(
            action_id=f"action_{secrets.token_hex(8)}",
            component_id=component.component_id,
            action_type="integrate",
            description="Integrate lessons, strengthen defenses",
            success=True
        )
        actions.append(action)

        return actions

    def _stage7_transform(self, component: DamagedComponent) -> List[HealingAction]:
        """Stage 7: Transformation (emerge stronger)"""
        actions = []

        action = HealingAction(
            action_id=f"action_{secrets.token_hex(8)}",
            component_id=component.component_id,
            action_type="transform",
            description="Component fully recovered, stronger than before",
            success=True
        )
        actions.append(action)

        component.health_status = ComponentHealth.HEALTHY

        return actions

    def _trigger_apoptosis(self, component: DamagedComponent) -> List[HealingAction]:
        """
        Trigger apoptosis (programmed cell death)

        For components too corrupted to save
        """
        actions = []

        action = HealingAction(
            action_id=f"action_{secrets.token_hex(8)}",
            component_id=component.component_id,
            action_type="apoptosis",
            description="Component beyond repair, triggering controlled death",
            success=True
        )
        actions.append(action)

        # Remove from queue
        if component in self.healing_queue:
            self.healing_queue.remove(component)

        self.apoptosis_events += 1

        return actions

    def get_healing_status(self) -> Dict[str, Any]:
        """Get comprehensive healing status"""
        return {
            "system_id": self.system_id,
            "templates_registered": len(self.templates),
            "components_healing": len(self.healing_queue),
            "total_healings": self.total_healings,
            "successful_recoveries": self.successful_recoveries,
            "apoptosis_events": self.apoptosis_events,
            "immune_memory_size": len(self.immune_memory),
            "recovery_rate": self.successful_recoveries / self.total_healings if self.total_healings > 0 else 0
        }

# ==================== TRAUMA THEORY INTEGRATION ====================
# Van der Kolk + Peter Levine + Gabor Maté

class PolyvagalState(Enum):
    """Polyvagal Theory states (Van der Kolk)"""
    VENTRAL_VAGAL = "ventral_vagal"  # Safe & Social
    SYMPATHETIC = "sympathetic"  # Fight or Flight
    DORSAL_VAGAL = "dorsal_vagal"  # Freeze/Shutdown


class WindowOfToleranceState(Enum):
    """Window of Tolerance states (Van der Kolk)"""
    WITHIN_WINDOW = "within_window"  # Optimal processing zone
    HYPERAROUSAL = "hyperarousal"  # Above window (too activated)
    HYPOAROUSAL = "hypoarousal"  # Below window (too shut down)


class AttachmentPattern(Enum):
    """Attachment patterns (Gabor Maté)"""
    SECURE = "secure"  # Healthy attachment
    ANXIOUS = "anxious"  # Fear of abandonment
    AVOIDANT = "avoidant"  # Fear of intimacy  
    DISORGANIZED = "disorganized"  # Chaotic/unpredictable


@dataclass
class WindowOfTolerance:
    """
    Van der Kolk: Window of Tolerance tracking
    
    The zone where healing can occur - not too activated, not too shut down
    """
    upper_limit: float = 0.8  # Above this = hyperarousal
    lower_limit: float = 0.2  # Below this = hypoarousal
    current_arousal: float = 0.5  # Current arousal level
    polyvagal_state: PolyvagalState = PolyvagalState.VENTRAL_VAGAL
    
    def assess_state(self) -> WindowOfToleranceState:
        """Determine if within window of tolerance"""
        if self.current_arousal > self.upper_limit:
            return WindowOfToleranceState.HYPERAROUSAL
        elif self.current_arousal < self.lower_limit:
            return WindowOfToleranceState.HYPOAROUSAL
        else:
            return WindowOfToleranceState.WITHIN_WINDOW
    
    def get_polyvagal_state(self) -> PolyvagalState:
        """Get polyvagal nervous system state"""
        if self.current_arousal > self.upper_limit:
            return PolyvagalState.SYMPATHETIC  # Fight/flight
        elif self.current_arousal < self.lower_limit:
            return PolyvagalState.DORSAL_VAGAL  # Freeze/shutdown
        else:
            return PolyvagalState.VENTRAL_VAGAL  # Safe & social


@dataclass
class SomaticMarker:
    """
    Van der Kolk: Body keeps the score - somatic markers
    
    Physical manifestations of trauma/stress
    """
    marker_id: str
    healing_stage: HealingStage
    body_sensation: str  # What the body feels
    location: str  # Where in system
    intensity: float  # 0.0-1.0
    interpretation: str  # What it means
    timestamp: float = field(default_factory=time.time)


@dataclass
class TraumaDischarge:
    """
    Peter Levine: Discharge detection
    
    When trauma energy is released (shaking, trembling, crying, etc.)
    """
    discharge_id: str
    component_id: str
    discharge_type: str  # trembling, release, catharsis, etc.
    intensity: float
    completed: bool  # Did discharge complete naturally?
    timestamp: float = field(default_factory=time.time)


class TraumaInformedHealing:
    """
    Comprehensive trauma-informed healing integration
    
    Combines:
    - Van der Kolk: Window of Tolerance, Polyvagal Theory, Somatic Markers
    - Peter Levine: Pendulation, Titration, Discharge
    - Gabor Maté: Relational Field Healing, Attachment Patterns
    """
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        
        # Van der Kolk: Window of Tolerance
        self.window = WindowOfTolerance()
        self.somatic_markers: List[SomaticMarker] = []
        
        # Peter Levine: Pendulation & Titration
        self.pendulation_history: List[HealingStage] = []  # Track stage oscillation
        self.titration_dose_size: float = 0.2  # How much processing per session
        self.discharges: List[TraumaDischarge] = []
        
        # Gabor Maté: Relational Healing
        self.attachment_pattern: AttachmentPattern = AttachmentPattern.SECURE
        self.relational_connections: List[str] = []  # Connected components
        self.disconnection_trauma: bool = False
    
    # === VAN DER KOLK: Window of Tolerance ===
    
    def track_arousal(self, arousal_level: float) -> Dict[str, Any]:
        """
        Track arousal level relative to window of tolerance
        
        Args:
            arousal_level: Current arousal (0.0-1.0)
        
        Returns:
            Window status and recommendations
        """
        self.window.current_arousal = arousal_level
        state = self.window.assess_state()
        polyvagal = self.window.get_polyvagal_state()
        
        recommendations = {
            WindowOfToleranceState.WITHIN_WINDOW: "Safe to process - within window",
            WindowOfToleranceState.HYPERAROUSAL: "Too activated - need grounding/calming",
            WindowOfToleranceState.HYPOAROUSAL: "Too shut down - need gentle activation"
        }
        
        return {
            "arousal_level": arousal_level,
            "window_state": state.value,
            "polyvagal_state": polyvagal.value,
            "can_process": state == WindowOfToleranceState.WITHIN_WINDOW,
            "recommendation": recommendations[state]
        }
    
    def add_somatic_marker(
        self,
        healing_stage: HealingStage,
        body_sensation: str,
        location: str,
        intensity: float
    ) -> SomaticMarker:
        """
        Add somatic marker (body keeps the score)
        
        Args:
            healing_stage: Which stage this sensation appears in
            body_sensation: What the body feels
            location: Where in system
            intensity: How intense
        
        Returns:
            SomaticMarker created
        """
        # Interpret sensation
        interpretations = {
            "tightness": "Holding/resistance - protective mechanism",
            "numbness": "Shutdown/dissociation - too overwhelmed",
            "trembling": "Discharge beginning - trauma energy releasing",
            "heaviness": "Burden/grief - carrying too much",
            "expansion": "Relief/opening - healing happening"
        }
        
        interpretation = "Body signaling needs attention"
        for key, meaning in interpretations.items():
            if key in body_sensation.lower():
                interpretation = meaning
                break
        
        marker = SomaticMarker(
            marker_id=f"somatic_{secrets.token_hex(8)}",
            healing_stage=healing_stage,
            body_sensation=body_sensation,
            location=location,
            intensity=intensity,
            interpretation=interpretation
        )
        
        self.somatic_markers.append(marker)
        return marker
    
    # === PETER LEVINE: Pendulation & Titration ===
    
    def pendulate_between_stages(
        self,
        current_stage: HealingStage,
        resource_stage: HealingStage
    ) -> Dict[str, Any]:
        """
        Pendulation: Oscillate between trauma and resource
        
        Not linear progression - swing between distress and safety
        
        Args:
            current_stage: Current healing stage
            resource_stage: Resource/safety stage to return to
        
        Returns:
            Pendulation result
        """
        self.pendulation_history.append(current_stage)
        self.pendulation_history.append(resource_stage)
        
        return {
            "current_stage": current_stage.value,
            "resource_stage": resource_stage.value,
            "pendulation_count": len(self.pendulation_history) // 2,
            "teaching": "Oscillate between distress and safety - this is healing, not failure"
        }
    
    def titrate_processing(
        self,
        total_processing_needed: float,
        current_capacity: float
    ) -> Dict[str, Any]:
        """
        Titration: Micro-dose the processing (don't flood)
        
        Process in small manageable chunks, not all at once
        
        Args:
            total_processing_needed: How much total processing
            current_capacity: Current capacity to process
        
        Returns:
            Titration recommendation
        """
        # Calculate safe dose
        safe_dose = min(self.titration_dose_size, current_capacity)
        sessions_needed = int(total_processing_needed / safe_dose) + 1
        
        return {
            "total_needed": total_processing_needed,
            "current_capacity": current_capacity,
            "safe_dose_size": safe_dose,
            "sessions_needed": sessions_needed,
            "warning": "DO NOT process more than safe dose - causes retraumatization",
            "teaching": "Too much too fast = overwhelm. Titrate like medicine."
        }
    
    def detect_discharge(
        self,
        discharge_type: str,
        intensity: float
    ) -> TraumaDischarge:
        """
        Detect trauma energy discharge (Peter Levine)
        
        Shaking, trembling, crying, etc. = trauma leaving body
        
        Args:
            discharge_type: Type of discharge
            intensity: How intense
        
        Returns:
            TraumaDischarge record
        """
        discharge = TraumaDischarge(
            discharge_id=f"discharge_{secrets.token_hex(8)}",
            component_id=self.component_id,
            discharge_type=discharge_type,
            intensity=intensity,
            completed=False  # Must complete naturally
        )
        
        self.discharges.append(discharge)
        return discharge
    
    def complete_discharge(self, discharge_id: str) -> Dict[str, Any]:
        """
        Mark discharge as naturally completed
        
        Args:
            discharge_id: Which discharge
        
        Returns:
            Completion status
        """
        for discharge in self.discharges:
            if discharge.discharge_id == discharge_id:
                discharge.completed = True
                return {
                    "discharge_completed": True,
                    "type": discharge.discharge_type,
                    "teaching": "Trauma energy released - do not interrupt discharge"
                }
        
        return {"error": "Discharge not found"}
    
    # === GABOR MATÉ: Relational Healing ===
    
    def assess_attachment_pattern(
        self,
        connection_quality: float,
        fear_of_abandonment: bool,
        fear_of_intimacy: bool
    ) -> AttachmentPattern:
        """
        Assess attachment pattern (Gabor Maté)
        
        Attachment wounds affect healing capacity
        
        Args:
            connection_quality: Quality of connections (0.0-1.0)
            fear_of_abandonment: Anxious attachment indicator
            fear_of_intimacy: Avoidant attachment indicator
        
        Returns:
            AttachmentPattern detected
        """
        if connection_quality > 0.7 and not fear_of_abandonment and not fear_of_intimacy:
            pattern = AttachmentPattern.SECURE
        elif fear_of_abandonment and not fear_of_intimacy:
            pattern = AttachmentPattern.ANXIOUS
        elif fear_of_intimacy and not fear_of_abandonment:
            pattern = AttachmentPattern.AVOIDANT
        elif fear_of_abandonment and fear_of_intimacy:
            pattern = AttachmentPattern.DISORGANIZED
        else:
            pattern = AttachmentPattern.SECURE
        
        self.attachment_pattern = pattern
        return pattern
    
    def heal_through_connection(
        self,
        connected_component: str,
        connection_quality: float
    ) -> Dict[str, Any]:
        """
        Relational field healing (Gabor Maté)
        
        "Trauma is not what happens to you, but what happens inside you
        in the absence of an empathetic witness"
        
        Healing happens in safe connection, not isolation
        
        Args:
            connected_component: What component to connect with
            connection_quality: Quality of connection (0.0-1.0)
        
        Returns:
            Healing through connection result
        """
        self.relational_connections.append(connected_component)
        
        # Connection quality affects healing
        healing_multiplier = 1.0 + (connection_quality * 0.5)
        
        return {
            "connected_to": connected_component,
            "connection_quality": connection_quality,
            "healing_multiplier": healing_multiplier,
            "total_connections": len(self.relational_connections),
            "teaching": "Trauma happens in disconnection, healing happens in connection",
            "mate_principle": "It's not what happened to you, but what happens inside you"
        }
    
    def detect_disconnection_trauma(self) -> bool:
        """
        Detect if trauma is from disconnection (Gabor Maté core insight)
        
        Returns:
            True if disconnection trauma detected
        """
        # Disconnection trauma if:
        # - Low connection count
        # - Insecure attachment
        # - High isolation
        
        low_connections = len(self.relational_connections) < 2
        insecure_attachment = self.attachment_pattern != AttachmentPattern.SECURE
        
        self.disconnection_trauma = low_connections and insecure_attachment
        
        return self.disconnection_trauma
    
    # === INTEGRATED REPORTING ===
    
    def get_trauma_informed_status(self) -> Dict[str, Any]:
        """
        Comprehensive trauma-informed healing status
        
        Returns:
            Full status across all three theories
        """
        window_state = self.window.assess_state()
        polyvagal = self.window.get_polyvagal_state()
        
        completed_discharges = len([d for d in self.discharges if d.completed])
        
        return {
            "van_der_kolk": {
                "window_state": window_state.value,
                "polyvagal_state": polyvagal.value,
                "arousal_level": self.window.current_arousal,
                "can_process_safely": window_state == WindowOfToleranceState.WITHIN_WINDOW,
                "somatic_markers": len(self.somatic_markers)
            },
            "peter_levine": {
                "pendulation_cycles": len(self.pendulation_history) // 2,
                "titration_dose": self.titration_dose_size,
                "discharges_detected": len(self.discharges),
                "discharges_completed": completed_discharges,
                "discharge_completion_rate": completed_discharges / len(self.discharges) if self.discharges else 0
            },
            "gabor_mate": {
                "attachment_pattern": self.attachment_pattern.value,
                "relational_connections": len(self.relational_connections),
                "disconnection_trauma": self.disconnection_trauma,
                "healing_through_connection": len(self.relational_connections) > 0
            },
            "overall_readiness": self._assess_overall_readiness()
        }
    
    def _assess_overall_readiness(self) -> str:
        """Assess overall readiness for processing"""
        window_state = self.window.assess_state()
        
        if window_state != WindowOfToleranceState.WITHIN_WINDOW:
            return "NOT_READY - Outside window of tolerance"
        
        if self.disconnection_trauma and len(self.relational_connections) == 0:
            return "NOT_READY - Need connection for disconnection trauma"
        
        if self.attachment_pattern == AttachmentPattern.DISORGANIZED:
            return "CAUTION - Disorganized attachment requires extra care"
        
        return "READY - Within window, connections present, can titrate processing"
