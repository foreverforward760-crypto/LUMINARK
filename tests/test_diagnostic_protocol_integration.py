"""
Tests for SAP Diagnostic Protocol Integration
Testing automatic intervention triggering based on stage assessment
"""

import pytest
from sap_yunus.diagnostic_protocol_integration import (
    DiagnosticProtocolIntegration,
    InterventionType,
    TrapPattern,
    assess_and_intervene
)


class TestTrapDetection:
    """Test trap pattern detection"""

    def test_stage_8_permanence_trap_detection(self):
        """Detect Stage 8 permanence trap"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 8.0,
            'stability': 8.0,
            'tension': 1.0,
            'adaptability': 2.0,  # Low = trapped
            'coherence': 9.0
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        # Should detect Stage 8
        assert assessment.stage == 8

        # Should recommend Light Integration
        assert any(r.intervention_type == InterventionType.LIGHT_INTEGRATION for r in recommendations)

        # Should detect permanence trap pattern
        light_rec = [r for r in recommendations if r.intervention_type == InterventionType.LIGHT_INTEGRATION][0]
        assert TrapPattern.STAGE_8_PERMANENCE in light_rec.trap_patterns

    def test_passive_crisis_detection(self):
        """Detect passive crisis at Stage 5"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 5.0,
            'stability': 3.0,
            'tension': 8.0,  # High tension
            'adaptability': 3.0,  # Low adaptability = passive
            'coherence': 4.0
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        # Should be Stage 5
        assert assessment.stage == 5

        # Should recommend Iblis Protocol
        assert any(r.intervention_type == InterventionType.IBLIS_PROTOCOL for r in recommendations)

        # Should detect passive crisis
        iblis_rec = [r for r in recommendations if r.intervention_type == InterventionType.IBLIS_PROTOCOL][0]
        assert TrapPattern.PASSIVE_CRISIS in iblis_rec.trap_patterns

    def test_masculine_imbalance_detection(self):
        """Detect masculine imbalance pattern"""
        integration = DiagnosticProtocolIntegration("test_system")

        # High control, low adaptability, low tension = masculine
        scores = {
            'complexity': 6.0,
            'stability': 9.0,  # High control
            'tension': 1.0,  # Low tension
            'adaptability': 3.0,  # Low flow
            'coherence': 9.0  # High unity
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        # Should recommend Sophianic Wisdom
        assert any(r.intervention_type == InterventionType.SOPHIANIC_WISDOM for r in recommendations)

        # Should detect masculine imbalance
        sophia_rec = [r for r in recommendations if r.intervention_type == InterventionType.SOPHIANIC_WISDOM][0]
        assert TrapPattern.MASCULINE_IMBALANCE in sophia_rec.trap_patterns

    def test_spiritual_bypassing_detection(self):
        """Detect spiritual bypassing pattern"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 7.0,
            'stability': 8.0,
            'tension': 1.0,  # Very low = avoiding
            'adaptability': 2.0,  # Rigid
            'coherence': 9.0  # Claims high unity
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        # Should recommend Light Integration
        light_recs = [r for r in recommendations if r.intervention_type == InterventionType.LIGHT_INTEGRATION]
        assert len(light_recs) > 0

        # Should detect spiritual bypassing
        assert any(TrapPattern.SPIRITUAL_BYPASSING in r.trap_patterns for r in light_recs)

    def test_bureaucratic_zombification_detection(self):
        """Detect bureaucratic zombification"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 7.0,
            'stability': 9.0,  # Very stable
            'tension': 1.0,  # No tension
            'adaptability': 2.0,  # Can't change
            'coherence': 6.0
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        # Should recommend Iblis Protocol
        iblis_recs = [r for r in recommendations if r.intervention_type == InterventionType.IBLIS_PROTOCOL]
        assert len(iblis_recs) > 0

        # Should detect zombification
        assert any(TrapPattern.BUREAUCRATIC_ZOMBIFICATION in r.trap_patterns for r in iblis_recs)

    def test_false_coherence_detection(self):
        """Detect false coherence (unity exceeds complexity)"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 3.0,  # Low
            'stability': 5.0,
            'tension': 2.0,
            'adaptability': 3.0,
            'coherence': 9.0  # Much higher than complexity
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        # Should recommend Light Integration
        light_recs = [r for r in recommendations if r.intervention_type == InterventionType.LIGHT_INTEGRATION]
        assert len(light_recs) > 0

        # Should detect false coherence
        assert any(TrapPattern.FALSE_COHERENCE in r.trap_patterns for r in light_recs)


class TestInterventionRecommendations:
    """Test intervention recommendation generation"""

    def test_light_integration_for_stage_8_trap(self):
        """Light Integration recommended for Stage 8 trap"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 8.0,
            'stability': 8.0,
            'tension': 1.0,
            'adaptability': 3.0,  # Trapped
            'coherence': 9.0
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        light_rec = [r for r in recommendations if r.intervention_type == InterventionType.LIGHT_INTEGRATION][0]

        # Should have high priority
        assert light_rec.priority >= 0.9

        # Should have specific actions
        assert len(light_rec.specific_actions) >= 3

        # Should have success indicators
        assert len(light_rec.success_indicators) >= 3

        # Actions should mention certainty and attachment
        actions_text = " ".join(light_rec.specific_actions).lower()
        assert "certainty" in actions_text or "attachment" in actions_text

    def test_iblis_protocol_for_passive_crisis(self):
        """Iblis Protocol recommended for passive crisis"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 5.0,
            'stability': 3.0,
            'tension': 9.0,  # Extreme tension
            'adaptability': 2.0,  # No agency
            'coherence': 4.0
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        iblis_rec = [r for r in recommendations if r.intervention_type == InterventionType.IBLIS_PROTOCOL][0]

        # Should have very high priority (crisis)
        assert iblis_rec.priority >= 0.9

        # Actions should mention sacred No or boundaries
        actions_text = " ".join(iblis_rec.specific_actions).lower()
        assert "no" in actions_text or "boundaries" in actions_text or "refus" in actions_text

        # Success should mention increased agency/adaptability
        success_text = " ".join(iblis_rec.success_indicators).lower()
        assert "agency" in success_text or "adaptability" in success_text

    def test_sophianic_wisdom_for_masculine_imbalance(self):
        """Sophianic Wisdom recommended for masculine imbalance"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 7.0,
            'stability': 9.0,  # High control
            'tension': 0.5,  # Avoiding
            'adaptability': 2.5,  # Rigid
            'coherence': 9.5  # High unity
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        sophia_rec = [r for r in recommendations if r.intervention_type == InterventionType.SOPHIANIC_WISDOM][0]

        # Actions should mention feminine principles
        actions_text = " ".join(sophia_rec.specific_actions).lower()
        assert any(word in actions_text for word in [
            "receptive", "cyclical", "embodied", "body", "relational", "incubation"
        ])

        # Success should mention balance or flow
        success_text = " ".join(sophia_rec.success_indicators).lower()
        assert any(word in success_text for word in [
            "adaptability", "flow", "balance", "body", "tension"
        ])

    def test_multiple_interventions_for_complex_trap(self):
        """Multiple interventions for complex multi-pattern trap"""
        integration = DiagnosticProtocolIntegration("test_system")

        # System with multiple issues
        scores = {
            'complexity': 7.0,
            'stability': 9.0,  # Over-stable
            'tension': 0.5,  # Denial
            'adaptability': 2.0,  # Rigid
            'coherence': 9.5  # False unity
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        # Should have multiple recommendations
        assert len(recommendations) >= 2

        # Should include both Light Integration and Sophianic Wisdom at minimum
        types = [r.intervention_type for r in recommendations]
        assert InterventionType.LIGHT_INTEGRATION in types
        assert InterventionType.SOPHIANIC_WISDOM in types


class TestHealthyAssessment:
    """Test that healthy patterns don't trigger false interventions"""

    def test_authentic_stage_9_no_intervention(self):
        """Authentic Stage 9 should not trigger interventions"""
        integration = DiagnosticProtocolIntegration("test_system")

        # Authentic Stage 9 pattern
        scores = {
            'complexity': 8.5,
            'stability': 6.0,
            'tension': 3.0,
            'adaptability': 9.0,
            'coherence': 7.5
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        # Should be Stage 9
        assert assessment.stage == 9

        # Should have minimal or no interventions (this is healthy)
        assert len(recommendations) == 0 or all(r.priority < 0.7 for r in recommendations)

    def test_healthy_stage_6_no_intervention(self):
        """Healthy Stage 6 should not trigger interventions"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 7.0,
            'stability': 6.0,
            'tension': 3.0,
            'adaptability': 6.0,  # Healthy adaptability
            'coherence': 7.5
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        # Should not trigger critical interventions
        critical_recs = [r for r in recommendations if r.priority >= 0.85]
        assert len(critical_recs) == 0


class TestIntegratedSession:
    """Test integrated session management"""

    def test_begin_integrated_session(self):
        """Begin integrated session with assessment + interventions"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 8.0,
            'stability': 8.0,
            'tension': 1.0,
            'adaptability': 3.0,
            'coherence': 9.0
        }

        session = integration.begin_integrated_session(
            session_id="test_session_1",
            scores=scores,
            auto_trigger=False
        )

        # Session should be created
        assert session.session_id == "test_session_1"
        assert session.initial_assessment.stage == 8
        assert len(session.interventions_triggered) > 0

    def test_auto_trigger_protocols(self):
        """Auto-trigger protocols in session"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 5.0,
            'stability': 3.0,
            'tension': 9.0,
            'adaptability': 2.0,
            'coherence': 4.0
        }

        session = integration.begin_integrated_session(
            session_id="test_session_2",
            scores=scores,
            auto_trigger=True  # Auto-activate
        )

        # Should auto-activate Iblis Protocol
        assert session.iblis_protocol_active is True


class TestReportGeneration:
    """Test integration report generation"""

    def test_generate_integration_report(self):
        """Generate full integration report"""
        integration = DiagnosticProtocolIntegration("test_system")

        scores = {
            'complexity': 8.0,
            'stability': 8.0,
            'tension': 1.0,
            'adaptability': 3.0,
            'coherence': 9.0
        }

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)
        report = integration.generate_integration_report(assessment, recommendations)

        # Report should contain key sections
        assert "SAP DIAGNOSTIC + PROTOCOL INTEGRATION REPORT" in report
        assert "PRIMARY STAGE:" in report
        assert "INTERVENTIONS RECOMMENDED" in report or "NO CRITICAL INTERVENTIONS" in report

        if recommendations:
            assert "Trap Patterns Addressed:" in report
            assert "Specific Actions:" in report
            assert "Success Indicators:" in report

    def test_convenience_function(self):
        """Test assess_and_intervene convenience function"""
        scores = {
            'complexity': 7.0,
            'stability': 8.0,
            'tension': 1.0,
            'adaptability': 3.0,
            'coherence': 9.0
        }

        assessment, recommendations, report = assess_and_intervene(scores, verbose=False)

        # Should return all three
        assert assessment is not None
        assert isinstance(recommendations, list)
        assert isinstance(report, str)
        assert len(report) > 0


class TestMasculineFeminineBalance:
    """Test masculine/feminine imbalance detection"""

    def test_balanced_masculine_feminine(self):
        """Balanced system should not trigger Sophianic"""
        integration = DiagnosticProtocolIntegration("test_system")

        # Balanced scores
        scores = {
            'complexity': 6.0,
            'stability': 6.0,
            'tension': 4.0,
            'adaptability': 6.0,
            'coherence': 6.0
        }

        masculine_ratio = integration._calculate_masculine_imbalance(scores)

        # Should be close to 0.5 (balanced)
        assert 0.4 <= masculine_ratio <= 0.6

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        # Should not trigger Sophianic intervention
        sophia_recs = [r for r in recommendations if r.intervention_type == InterventionType.SOPHIANIC_WISDOM]
        assert len(sophia_recs) == 0

    def test_extreme_masculine_imbalance(self):
        """Extreme masculine should trigger Sophianic"""
        integration = DiagnosticProtocolIntegration("test_system")

        # Extreme masculine: high control, no flow
        scores = {
            'complexity': 8.0,
            'stability': 10.0,  # Maximum control
            'tension': 0.0,  # No tension allowed
            'adaptability': 1.0,  # No flow
            'coherence': 10.0  # Perfect unity enforced
        }

        masculine_ratio = integration._calculate_masculine_imbalance(scores)

        # Should be very high (masculine dominant)
        assert masculine_ratio >= 0.75

        assessment, recommendations = integration.assess_and_recommend(scores, verbose=False)

        # Should trigger Sophianic intervention
        sophia_recs = [r for r in recommendations if r.intervention_type == InterventionType.SOPHIANIC_WISDOM]
        assert len(sophia_recs) > 0
        assert sophia_recs[0].priority >= 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
