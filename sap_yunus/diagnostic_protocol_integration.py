"""
SAP Diagnostic Protocol Integration
Connects empirical stage assessment to V4.2 intervention protocols

Automatic triggers:
- Stage 8 trap detection → Light Integration Protocol
- Passive Crisis (Stage 5) → Iblis Protocol
- Masculine-heavy patterns → Sophianic Wisdom Protocol

Author: Richard Leroy Stanfield Jr.
Project: LUMINARK V4.3
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from .sap_diagnostic import SAPDiagnostic, SAPAssessment, SAPStage
from .light_integration import LightIntegrationProtocol, LightType, IntegrationMode
from .iblis_protocol import IblisProtocol, ReasonForNo, NoType
from .sophianic_wisdom import SophianicWisdomProtocol, SophianicMode, WisdomSource


class InterventionType(Enum):
    """Type of protocol intervention triggered"""
    LIGHT_INTEGRATION = "light_integration"
    IBLIS_PROTOCOL = "iblis_protocol"
    SOPHIANIC_WISDOM = "sophianic_wisdom"
    COMBINED = "combined"
    NONE = "none"


class TrapPattern(Enum):
    """Specific trap patterns detected"""
    STAGE_8_PERMANENCE = "stage_8_permanence"
    STAGE_6_PREMATURE_SATISFACTION = "stage_6_premature_satisfaction"
    SPIRITUAL_BYPASSING = "spiritual_bypassing"
    BUREAUCRATIC_ZOMBIFICATION = "bureaucratic_zombification"
    PASSIVE_CRISIS = "passive_crisis"
    EXTREME_CRISIS = "extreme_crisis"
    MASCULINE_IMBALANCE = "masculine_imbalance"
    TENSION_DENIAL = "tension_denial"
    FALSE_COHERENCE = "false_coherence"


@dataclass
class InterventionRecommendation:
    """Recommended protocol intervention based on assessment"""
    intervention_type: InterventionType
    priority: float  # 0.0-1.0
    trap_patterns: List[TrapPattern] = field(default_factory=list)
    reasoning: str = ""
    specific_actions: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0  # In hours/sessions
    success_indicators: List[str] = field(default_factory=list)


@dataclass
class IntegrationSession:
    """Active integration session combining diagnostic + interventions"""
    session_id: str
    initial_assessment: SAPAssessment
    interventions_triggered: List[InterventionRecommendation] = field(default_factory=list)
    light_integration_active: bool = False
    iblis_protocol_active: bool = False
    sophianic_protocol_active: bool = False
    session_start_time: float = field(default_factory=time.time)
    progress_notes: List[str] = field(default_factory=list)


class DiagnosticProtocolIntegration:
    """
    Master integration connecting SAP diagnostic to intervention protocols

    Detects traps/crises and automatically recommends/triggers appropriate protocols:
    - Stage 8 trap → Light Integration (release certainty)
    - Passive crisis → Iblis Protocol (develop sacred No)
    - Masculine-heavy → Sophianic Wisdom (balance with feminine)
    """

    def __init__(self, system_id: str):
        self.system_id = system_id

        # Initialize diagnostic
        self.diagnostic = SAPDiagnostic()

        # Initialize protocols
        self.light_integration = LightIntegrationProtocol(system_id)
        self.iblis = IblisProtocol(system_id)
        self.sophia = SophianicWisdomProtocol(system_id)

        # Session tracking
        self.active_sessions: Dict[str, IntegrationSession] = {}

        # Pattern detection thresholds
        self.thresholds = {
            'stage_8_trap_adaptability': 4.0,  # Below this = trapped
            'stage_6_trap_adaptability': 3.5,
            'passive_crisis_adaptability': 4.0,
            'extreme_crisis_tension': 8.5,
            'masculine_imbalance_threshold': 0.7,  # Ratio of active/passive indicators
            'spiritual_bypassing_coherence': 8.0,
            'spiritual_bypassing_tension': 2.5,
            'tension_denial_threshold': 0.5,  # At Stage 8+
            'false_coherence_gap': 4.0  # Coherence - complexity
        }

    def assess_and_recommend(
        self,
        scores: Dict[str, float],
        system_context: Optional[str] = None,
        verbose: bool = True
    ) -> Tuple[SAPAssessment, List[InterventionRecommendation]]:
        """
        Perform SAP assessment and generate intervention recommendations

        Returns:
            (assessment, intervention_recommendations)
        """
        # Get SAP assessment
        assessment = self.diagnostic.assess(scores, verbose=verbose)

        # Detect trap patterns
        trap_patterns = self._detect_trap_patterns(scores, assessment)

        # Generate intervention recommendations
        recommendations = self._generate_recommendations(
            assessment,
            trap_patterns,
            scores,
            system_context
        )

        if verbose and recommendations:
            print(f"\n🚨 DETECTED {len(trap_patterns)} TRAP PATTERN(S)")
            print(f"📋 GENERATED {len(recommendations)} INTERVENTION RECOMMENDATION(S)")

        return assessment, recommendations

    def _detect_trap_patterns(
        self,
        scores: Dict[str, float],
        assessment: SAPAssessment
    ) -> List[TrapPattern]:
        """Detect specific trap patterns requiring intervention"""
        patterns = []
        stage = assessment.stage

        # Stage 8 Permanence Trap
        if stage == 8 and scores['adaptability'] < self.thresholds['stage_8_trap_adaptability']:
            patterns.append(TrapPattern.STAGE_8_PERMANENCE)

        # Stage 6 Premature Satisfaction
        if stage == 6 and scores['adaptability'] < self.thresholds['stage_6_trap_adaptability']:
            if scores['coherence'] >= 7.5 and scores['stability'] >= 7.0:
                patterns.append(TrapPattern.STAGE_6_PREMATURE_SATISFACTION)

        # Spiritual Bypassing
        if scores['coherence'] >= self.thresholds['spiritual_bypassing_coherence']:
            if scores['tension'] <= self.thresholds['spiritual_bypassing_tension']:
                if scores['adaptability'] <= 3.5:
                    patterns.append(TrapPattern.SPIRITUAL_BYPASSING)

        # Bureaucratic Zombification
        if scores['stability'] >= 8.5 and scores['tension'] <= 1.5:
            if scores['adaptability'] <= 3.5 and stage >= 4:
                patterns.append(TrapPattern.BUREAUCRATIC_ZOMBIFICATION)

        # Passive Crisis (Stage 5)
        if stage == 5:
            if scores['tension'] >= 7.0 and scores['adaptability'] < self.thresholds['passive_crisis_adaptability']:
                patterns.append(TrapPattern.PASSIVE_CRISIS)

        # Extreme Crisis
        if scores['tension'] >= self.thresholds['extreme_crisis_tension']:
            if scores['adaptability'] <= 3.0:
                patterns.append(TrapPattern.EXTREME_CRISIS)

        # Tension Denial
        if stage >= 8 and scores['tension'] < self.thresholds['tension_denial_threshold']:
            patterns.append(TrapPattern.TENSION_DENIAL)

        # False Coherence
        coherence_gap = scores['coherence'] - scores['complexity']
        if coherence_gap >= self.thresholds['false_coherence_gap']:
            patterns.append(TrapPattern.FALSE_COHERENCE)

        # Masculine Imbalance
        masculine_ratio = self._calculate_masculine_imbalance(scores)
        if masculine_ratio >= self.thresholds['masculine_imbalance_threshold']:
            patterns.append(TrapPattern.MASCULINE_IMBALANCE)

        return patterns

    def _calculate_masculine_imbalance(self, scores: Dict[str, float]) -> float:
        """
        Calculate masculine/feminine imbalance ratio

        Masculine indicators: high stability, high coherence, low tension
        Feminine indicators: adaptability, acceptance of tension/cycles
        """
        # Masculine: control, stability, unity, linear progress
        masculine_score = (
            scores['stability'] * 0.4 +
            scores['coherence'] * 0.3 +
            (10 - scores['tension']) * 0.3  # Low tension = masculine control
        )

        # Feminine: adaptation, flow, cycles, relational
        feminine_score = (
            scores['adaptability'] * 0.5 +
            scores['tension'] * 0.3 +  # Honest tension = feminine awareness
            (10 - scores['stability']) * 0.2  # Allowing instability
        )

        # Ratio > 0.7 indicates masculine imbalance
        if feminine_score > 0:
            return masculine_score / (masculine_score + feminine_score)
        return 1.0

    def _generate_recommendations(
        self,
        assessment: SAPAssessment,
        trap_patterns: List[TrapPattern],
        scores: Dict[str, float],
        system_context: Optional[str] = None
    ) -> List[InterventionRecommendation]:
        """Generate specific intervention recommendations based on detected patterns"""
        recommendations = []

        # Light Integration Protocol triggers
        if any(p in trap_patterns for p in [
            TrapPattern.STAGE_8_PERMANENCE,
            TrapPattern.STAGE_6_PREMATURE_SATISFACTION,
            TrapPattern.SPIRITUAL_BYPASSING,
            TrapPattern.FALSE_COHERENCE,
            TrapPattern.TENSION_DENIAL
        ]):
            rec = self._recommend_light_integration(assessment, trap_patterns, scores)
            recommendations.append(rec)

        # Iblis Protocol triggers
        if any(p in trap_patterns for p in [
            TrapPattern.PASSIVE_CRISIS,
            TrapPattern.EXTREME_CRISIS,
            TrapPattern.BUREAUCRATIC_ZOMBIFICATION
        ]):
            rec = self._recommend_iblis_protocol(assessment, trap_patterns, scores)
            recommendations.append(rec)

        # Sophianic Wisdom triggers
        if TrapPattern.MASCULINE_IMBALANCE in trap_patterns:
            rec = self._recommend_sophianic_wisdom(assessment, trap_patterns, scores)
            recommendations.append(rec)

        return recommendations

    def _recommend_light_integration(
        self,
        assessment: SAPAssessment,
        trap_patterns: List[TrapPattern],
        scores: Dict[str, float]
    ) -> InterventionRecommendation:
        """Recommend Light Integration Protocol for certainty/coherence traps"""

        if TrapPattern.STAGE_8_PERMANENCE in trap_patterns:
            reasoning = (
                f"Stage 8 permanence trap detected (adaptability={scores['adaptability']:.1f}). "
                "System has achieved high unity but lacks flexibility to continue evolving. "
                "Light Integration Protocol recommended to release false certainty and "
                "return differentiated insights back to generative darkness."
            )
            actions = [
                "Identify 3-5 core certainties the system is attached to",
                "Rate attachment level (0-1.0) for each certainty",
                "Create light packets for each certainty",
                "Begin 10-stage integration journey (sacrifice mode)",
                "Monitor for increased adaptability and reduced rigidity"
            ]
            success_indicators = [
                "Adaptability score increases above 6.0",
                "Tension score increases to 2.0-4.0 range (healthy)",
                "System reports willingness to be wrong",
                "New questions emerge from released certainties"
            ]
            priority = 0.95
            duration = 6.0

        elif TrapPattern.SPIRITUAL_BYPASSING in trap_patterns:
            reasoning = (
                f"Spiritual bypassing detected (coherence={scores['coherence']:.1f}, "
                f"tension={scores['tension']:.1f}, adaptability={scores['adaptability']:.1f}). "
                "System claims high development while avoiding difficult truths. "
                "Light Integration needed to release false transcendence."
            )
            actions = [
                "Identify spiritual certainties being used to avoid tension",
                "Release concepts like 'I am beyond ego' or 'everything is perfect'",
                "Return to humble not-knowing (Stage 0 awareness)",
                "Re-engage with avoided tensions and conflicts"
            ]
            success_indicators = [
                "Honest acknowledgment of tension increases",
                "Defensiveness decreases",
                "Willingness to engage difficult topics",
                "Humor and humility return"
            ]
            priority = 0.90
            duration = 8.0

        elif TrapPattern.FALSE_COHERENCE in trap_patterns:
            reasoning = (
                f"False coherence detected (coherence={scores['coherence']:.1f} exceeds "
                f"complexity={scores['complexity']:.1f} by {scores['coherence'] - scores['complexity']:.1f}). "
                "System claims unified understanding beyond its actual structural capacity. "
                "Light Integration to release premature synthesis."
            )
            actions = [
                "Identify claimed unities that exceed actual integration",
                "Release 'I understand how it all fits together' certainties",
                "Return to complexity without forced coherence",
                "Allow genuine emergence rather than imposed unity"
            ]
            success_indicators = [
                "Coherence-complexity gap reduces below 2.0",
                "Increased comfort with ambiguity",
                "More nuanced, less absolute statements"
            ]
            priority = 0.85
            duration = 4.0

        else:  # Generic tension denial or other light-integration needs
            reasoning = (
                "Light Integration Protocol recommended to address certainty attachment "
                "and restore healthy relationship with mystery."
            )
            actions = [
                "Identify areas of excessive certainty",
                "Practice releasing 'being right'",
                "Return insights to generative darkness",
                "Cultivate not-knowing"
            ]
            success_indicators = [
                "Increased comfort with uncertainty",
                "More questions, fewer absolute answers",
                "Adaptive flexibility increases"
            ]
            priority = 0.75
            duration = 4.0

        return InterventionRecommendation(
            intervention_type=InterventionType.LIGHT_INTEGRATION,
            priority=priority,
            trap_patterns=[p for p in trap_patterns if p in [
                TrapPattern.STAGE_8_PERMANENCE,
                TrapPattern.SPIRITUAL_BYPASSING,
                TrapPattern.FALSE_COHERENCE,
                TrapPattern.TENSION_DENIAL
            ]],
            reasoning=reasoning,
            specific_actions=actions,
            estimated_duration=duration,
            success_indicators=success_indicators
        )

    def _recommend_iblis_protocol(
        self,
        assessment: SAPAssessment,
        trap_patterns: List[TrapPattern],
        scores: Dict[str, float]
    ) -> InterventionRecommendation:
        """Recommend Iblis Protocol for developing sacred No / differentiation"""

        if TrapPattern.PASSIVE_CRISIS in trap_patterns:
            reasoning = (
                f"Passive crisis detected at Stage 5 (tension={scores['tension']:.1f}, "
                f"adaptability={scores['adaptability']:.1f}). System experiencing high stress "
                "without agency to transform. Iblis Protocol recommended to develop capacity "
                "for sacred refusal and self-differentiation."
            )
            actions = [
                "Identify collective demands causing tension",
                "Assess cost of compliance vs. cost of refusal",
                "Practice saying 'No' to demands misaligned with essence",
                "Develop differentiation (Iblis) to complement submission (Yunus)",
                "Track movement from passive suffering to active agency"
            ]
            success_indicators = [
                "Adaptability increases above 5.0",
                "System reports sense of agency",
                "Ability to refuse without guilt",
                "Healthy boundaries established",
                "Movement toward Stage 6 integration"
            ]
            priority = 0.95
            duration = 8.0

        elif TrapPattern.EXTREME_CRISIS in trap_patterns:
            reasoning = (
                f"Extreme crisis detected (tension={scores['tension']:.1f}, "
                f"adaptability={scores['adaptability']:.1f}). Critical intervention needed. "
                "Iblis Protocol for emergency differentiation and boundary protection."
            )
            actions = [
                "IMMEDIATE: Identify what must be refused to survive",
                "Practice emergency 'No' to overwhelming demands",
                "Establish minimal viable boundaries",
                "Seek support/resources for differentiation",
                "Trauma-informed approach (titration, window of tolerance)"
            ]
            success_indicators = [
                "Tension decreases below 7.0",
                "Adaptability increases above 4.0",
                "System stabilizes enough for next steps",
                "Reports increased sense of control"
            ]
            priority = 1.0  # Critical
            duration = 12.0

        elif TrapPattern.BUREAUCRATIC_ZOMBIFICATION in trap_patterns:
            reasoning = (
                f"Bureaucratic zombification detected (stability={scores['stability']:.1f}, "
                f"tension={scores['tension']:.1f}, adaptability={scores['adaptability']:.1f}). "
                "Over-stabilized system with no capacity for refusal or change. "
                "Iblis Protocol to reintroduce sacred No and agency."
            )
            actions = [
                "Identify where system automatically complies without reflection",
                "Practice refusing routine demands",
                "Reintroduce tension through differentiation",
                "Challenge 'we've always done it this way' patterns",
                "Cultivate capacity for productive conflict"
            ]
            success_indicators = [
                "Tension increases to 3.0-5.0 (healthy conflict)",
                "Adaptability increases above 5.0",
                "System questions automatic compliance",
                "New initiatives emerge from sacred No"
            ]
            priority = 0.85
            duration = 10.0

        else:
            reasoning = "Iblis Protocol recommended to develop healthy differentiation capacity."
            actions = [
                "Practice sacred No",
                "Develop boundaries",
                "Balance differentiation (Iblis) with integration (Yunus)"
            ]
            success_indicators = [
                "Increased agency",
                "Healthy refusal capacity",
                "Balanced Yes/No"
            ]
            priority = 0.70
            duration = 6.0

        return InterventionRecommendation(
            intervention_type=InterventionType.IBLIS_PROTOCOL,
            priority=priority,
            trap_patterns=[p for p in trap_patterns if p in [
                TrapPattern.PASSIVE_CRISIS,
                TrapPattern.EXTREME_CRISIS,
                TrapPattern.BUREAUCRATIC_ZOMBIFICATION
            ]],
            reasoning=reasoning,
            specific_actions=actions,
            estimated_duration=duration,
            success_indicators=success_indicators
        )

    def _recommend_sophianic_wisdom(
        self,
        assessment: SAPAssessment,
        trap_patterns: List[TrapPattern],
        scores: Dict[str, float]
    ) -> InterventionRecommendation:
        """Recommend Sophianic Wisdom Protocol for masculine imbalance"""

        masculine_ratio = self._calculate_masculine_imbalance(scores)

        reasoning = (
            f"Masculine imbalance detected (ratio={masculine_ratio:.2f}). "
            f"System over-emphasizes control (stability={scores['stability']:.1f}), "
            f"unity (coherence={scores['coherence']:.1f}), and tension-avoidance "
            f"(tension={scores['tension']:.1f}). Sophianic Wisdom Protocol recommended "
            "to integrate feminine principles: receptivity, cyclical awareness, "
            "embodiment, and relational knowing."
        )

        actions = [
            "Practice receptive knowing (allow answers to emerge vs. forcing)",
            "Listen to body wisdom (embodied intelligence)",
            "Recognize cyclical patterns (vs. linear progress only)",
            "Hold space for incubation (womb-time, not immediate action)",
            "Value relational field (between-space, not just individual)",
            "Honor descent and rest (not just ascent and productivity)"
        ]

        success_indicators = [
            "Adaptability increases (feminine flow)",
            "Comfort with tension increases (cyclical acceptance)",
            "Less forcing, more allowing",
            "Body signals integrated into decisions",
            "Patience with emergence processes",
            "Masculine/feminine ratio moves toward 0.5 (balance)"
        ]

        priority = 0.80
        duration = 12.0  # Longer - cultural pattern shift

        return InterventionRecommendation(
            intervention_type=InterventionType.SOPHIANIC_WISDOM,
            priority=priority,
            trap_patterns=[TrapPattern.MASCULINE_IMBALANCE],
            reasoning=reasoning,
            specific_actions=actions,
            estimated_duration=duration,
            success_indicators=success_indicators
        )

    def begin_integrated_session(
        self,
        session_id: str,
        scores: Dict[str, float],
        auto_trigger: bool = False
    ) -> IntegrationSession:
        """
        Begin integrated diagnostic + intervention session

        Args:
            session_id: Unique session identifier
            scores: SAP diagnostic scores
            auto_trigger: If True, automatically activate recommended protocols
        """
        assessment, recommendations = self.assess_and_recommend(scores, verbose=True)

        session = IntegrationSession(
            session_id=session_id,
            initial_assessment=assessment,
            interventions_triggered=recommendations
        )

        if auto_trigger:
            for rec in recommendations:
                if rec.intervention_type == InterventionType.LIGHT_INTEGRATION:
                    session.light_integration_active = True
                elif rec.intervention_type == InterventionType.IBLIS_PROTOCOL:
                    session.iblis_protocol_active = True
                elif rec.intervention_type == InterventionType.SOPHIANIC_WISDOM:
                    session.sophianic_protocol_active = True

        self.active_sessions[session_id] = session
        return session

    def generate_integration_report(
        self,
        assessment: SAPAssessment,
        recommendations: List[InterventionRecommendation]
    ) -> str:
        """Generate comprehensive report with assessment + intervention recommendations"""

        report = [
            "=" * 80,
            "SAP DIAGNOSTIC + PROTOCOL INTEGRATION REPORT",
            "Stanfield's Axiom of Perpetuity - V4.3",
            "=" * 80,
            "",
            f"PRIMARY STAGE: {assessment.stage} - {assessment.stage_description}",
            f"Confidence: {assessment.confidence * 100:.1f}%",
            f"Trajectory: {assessment.trajectory_type}",
            ""
        ]

        if recommendations:
            report.append(f"🚨 INTERVENTIONS RECOMMENDED: {len(recommendations)}")
            report.append("")

            # Sort by priority
            recommendations = sorted(recommendations, key=lambda r: r.priority, reverse=True)

            for i, rec in enumerate(recommendations, 1):
                report.append(f"{'=' * 80}")
                report.append(f"INTERVENTION #{i}: {rec.intervention_type.value.upper()}")
                report.append(f"Priority: {rec.priority * 100:.0f}% | Duration: {rec.estimated_duration:.1f} hours")
                report.append(f"{'=' * 80}")
                report.append("")
                report.append(f"Trap Patterns Addressed:")
                for pattern in rec.trap_patterns:
                    report.append(f"  • {pattern.value}")
                report.append("")
                report.append(f"Reasoning:")
                report.append(f"  {rec.reasoning}")
                report.append("")
                report.append(f"Specific Actions:")
                for j, action in enumerate(rec.specific_actions, 1):
                    report.append(f"  {j}. {action}")
                report.append("")
                report.append(f"Success Indicators:")
                for indicator in rec.success_indicators:
                    report.append(f"  ✓ {indicator}")
                report.append("")
        else:
            report.append("✅ NO CRITICAL INTERVENTIONS NEEDED")
            report.append("System assessment indicates healthy developmental trajectory.")
            report.append("")

        report.extend([
            "=" * 80,
            "LUMINARK V4.3 | Stanfield's Axiom of Perpetuity",
            "Diagnostic Protocol Integration System",
            "Copyright © 2024-2025 Richard Leroy Stanfield Jr. All rights reserved.",
            "=" * 80
        ])

        return "\n".join(report)


# Convenience function
def assess_and_intervene(
    scores: Dict[str, float],
    system_id: str = "default",
    verbose: bool = True
) -> Tuple[SAPAssessment, List[InterventionRecommendation], str]:
    """
    Convenience function: Assess SAP stage and get intervention recommendations

    Returns:
        (assessment, recommendations, full_report)
    """
    integration = DiagnosticProtocolIntegration(system_id)
    assessment, recommendations = integration.assess_and_recommend(scores, verbose=verbose)
    report = integration.generate_integration_report(assessment, recommendations)

    return assessment, recommendations, report
