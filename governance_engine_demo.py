#!/usr/bin/env python3
"""
SAP Governance Engine - Clean Demonstration
Stanfield's Axiom of Perpetuity v4.4

This demonstrates the production-ready diagnostic system built in V4.4.
It does NOT include:
- Quantum computing (Qiskit) - removed as it has no relation to the diagnostic
- Neural networks - SAP is a diagnostic framework, not a generative model
- Pseudoscientific claims about consciousness

What it DOES include:
- Empirically-calibrated stage assessment
- Trap pattern detection
- Intervention recommendations
- Organizational health diagnostics
"""

import sys
from pathlib import Path

# Add sap_yunus to path
sys.path.insert(0, str(Path(__file__).parent))

from sap_yunus import (
    SAPDiagnostic,
    DiagnosticProtocolIntegration,
    assess_and_intervene
)


class GovernanceEngine:
    """
    SAP Governance Engine v4.4

    Diagnostic tool for assessing organizational/system developmental health.

    NOT a prediction tool. NOT a chatbot. NOT quantum computing.
    This is a diagnostic assessment system based on empirical stage weights.

    Use cases:
    - Organizational health assessment
    - Team dynamics evaluation
    - Project maturity analysis
    - Governance structure review
    """

    def __init__(self, system_id: str = "governance_engine"):
        self.diagnostic = SAPDiagnostic()
        self.integration = DiagnosticProtocolIntegration(system_id)
        self.system_id = system_id

    def assess_organization(
        self,
        complexity: float,
        stability: float,
        tension: float,
        adaptability: float,
        coherence: float,
        qualitative_context: str = ""
    ) -> dict:
        """
        Assess an organization's developmental stage

        Args:
            complexity: Structural complexity (0-10)
            stability: Resilience and predictability (0-10)
            tension: Internal stress/conflict (0-10)
            adaptability: Capacity to change (0-10)
            coherence: Unified purpose (0-10)
            qualitative_context: Optional text description

        Returns:
            Assessment with stage, traps, and interventions
        """
        scores = {
            'complexity': complexity,
            'stability': stability,
            'tension': tension,
            'adaptability': adaptability,
            'coherence': coherence
        }

        # Get core assessment
        assessment = self.diagnostic.assess(scores, verbose=True)

        # Get intervention recommendations
        _, recommendations = self.integration.assess_and_recommend(
            scores,
            system_context=qualitative_context,
            verbose=False
        )

        return {
            'stage': assessment.stage,
            'stage_description': assessment.stage_description,
            'confidence': assessment.confidence,
            'trajectory': assessment.trajectory_type,
            'trap_patterns': [
                pattern.value
                for rec in recommendations
                for pattern in rec.trap_patterns
            ],
            'interventions': [
                {
                    'type': rec.intervention_type.value,
                    'priority': rec.priority,
                    'actions': rec.specific_actions,
                    'success_indicators': rec.success_indicators
                }
                for rec in recommendations
            ],
            'warnings': assessment.warnings,
            'insights': assessment.insights
        }

    def generate_report(
        self,
        complexity: float,
        stability: float,
        tension: float,
        adaptability: float,
        coherence: float
    ) -> str:
        """Generate full text report"""
        scores = {
            'complexity': complexity,
            'stability': stability,
            'tension': tension,
            'adaptability': adaptability,
            'coherence': coherence
        }

        assessment, recommendations = self.integration.assess_and_recommend(
            scores,
            verbose=False
        )

        return self.integration.generate_integration_report(
            assessment,
            recommendations
        )


def demo_healthy_organization():
    """Demo: Healthy Stage 6 organization"""
    print("=" * 80)
    print("DEMO 1: Healthy Organization (Stage 6 Integration)")
    print("=" * 80)

    engine = GovernanceEngine()

    result = engine.assess_organization(
        complexity=7.0,     # Multiple departments
        stability=6.0,      # Stable but flexible
        tension=3.5,        # Healthy disagreement
        adaptability=6.5,   # Responsive to change
        coherence=7.5,      # Clear shared mission
        qualitative_context="Tech startup, 50 employees, product-market fit achieved"
    )

    print(f"\nStage: {result['stage']} - {result['stage_description']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Trajectory: {result['trajectory']}")
    print(f"\nTrap Patterns: {result['trap_patterns'] or 'None - healthy!'}")
    print(f"Interventions Needed: {len(result['interventions'])}")

    if result['insights']:
        print("\nInsights:")
        for insight in result['insights']:
            print(f"  • {insight}")


def demo_trapped_organization():
    """Demo: Stage 8 trap - bureaucratic zombification"""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Trapped Organization (Stage 8 Permanence Trap)")
    print("=" * 80)

    engine = GovernanceEngine()

    result = engine.assess_organization(
        complexity=7.0,     # Established complexity
        stability=9.0,      # Rigidly stable
        tension=1.0,        # No conflict allowed
        adaptability=2.0,   # Resistant to change
        coherence=9.5,      # "Perfect" alignment
        qualitative_context="Fortune 500 company, established processes, 'this is how we've always done it'"
    )

    print(f"\nStage: {result['stage']} - {result['stage_description']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Trajectory: {result['trajectory']}")

    print(f"\n⚠️  Trap Patterns Detected: {len(result['trap_patterns'])}")
    for pattern in result['trap_patterns']:
        print(f"  • {pattern}")

    print(f"\n🔧 Interventions Recommended: {len(result['interventions'])}")
    for i, intervention in enumerate(result['interventions'], 1):
        print(f"\n  Intervention #{i}: {intervention['type'].upper()}")
        print(f"  Priority: {intervention['priority'] * 100:.0f}%")
        print(f"  Actions:")
        for action in intervention['actions'][:3]:  # First 3 actions
            print(f"    - {action}")


def demo_passive_crisis():
    """Demo: Stage 5 passive crisis"""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Passive Crisis (Stage 5 - High Tension, Low Agency)")
    print("=" * 80)

    engine = GovernanceEngine()

    result = engine.assess_organization(
        complexity=5.0,     # Moderate
        stability=3.0,      # Unstable
        tension=8.5,        # Very high stress
        adaptability=3.0,   # Can't respond effectively
        coherence=4.0,      # Unclear direction
        qualitative_context="Nonprofit facing funding crisis, staff burnout, unclear mission"
    )

    print(f"\nStage: {result['stage']} - {result['stage_description']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Trajectory: {result['trajectory']}")

    if result['interventions']:
        intervention = result['interventions'][0]
        print(f"\n🚨 CRITICAL INTERVENTION: {intervention['type'].upper()}")
        print(f"Priority: {intervention['priority'] * 100:.0f}%")
        print("\nSpecific Actions:")
        for action in intervention['actions']:
            print(f"  {action}")


def demo_full_report():
    """Demo: Full text report"""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Full Text Report")
    print("=" * 80)

    engine = GovernanceEngine()

    report = engine.generate_report(
        complexity=8.0,
        stability=8.0,
        tension=1.0,
        adaptability=2.5,
        coherence=9.0
    )

    print(report)


if __name__ == "__main__":
    print("🌿 SAP Governance Engine v4.4 - Demonstration")
    print("Stanfield's Axiom of Perpetuity")
    print()

    demo_healthy_organization()
    demo_trapped_organization()
    demo_passive_crisis()
    demo_full_report()

    print("\n" + "=" * 80)
    print("✅ Demonstrations Complete")
    print("=" * 80)
