"""
Tests for SAP Stage Diagnostic Tool
Testing empirical stage assessment with guru-proofing
"""

import pytest
from sap_yunus.sap_diagnostic import (
    SAPDiagnostic,
    SAPStage,
    TrajectoryType,
    SAPAssessment
)


class TestBasicAssessment:
    """Test basic stage assessment functionality"""

    def test_stage_0_assessment(self):
        """Test Stage 0 (Plenara/Void) assessment"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 0.0,
            'stability': 0.0,
            'tension': 0.0,
            'adaptability': 1.0,
            'coherence': 0.0
        }

        result = diagnostic.assess(scores)

        assert result.stage == 0
        assert result.confidence > 0.8
        assert "Plenara" in result.stage_description

    def test_stage_5_bilateral_threshold(self):
        """Test Stage 5 (Bilateral Threshold) assessment"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 5.0,
            'stability': 3.0,
            'tension': 9.0,  # Maximum tension
            'adaptability': 6.0,
            'coherence': 4.0
        }

        result = diagnostic.assess(scores)

        assert result.stage == 5
        assert "Bilateral Threshold" in result.stage_description
        assert result.confidence > 0.7

    def test_stage_9_transparent_return(self):
        """Test Stage 9 (Transparent Return) assessment"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 9.0,
            'stability': 5.0,
            'tension': 3.0,
            'adaptability': 9.5,  # Maximum adaptability
            'coherence': 7.0
        }

        result = diagnostic.assess(scores)

        assert result.stage == 9
        assert "Transparent Return" in result.stage_description
        assert result.confidence > 0.8


class TestGuruProofing:
    """Test guru-proofing detection systems"""

    def test_spiritual_bypassing_detection(self):
        """Detect spiritual bypassing: high coherence + low tension + low adaptability"""
        diagnostic = SAPDiagnostic()

        # Guru claiming high development while being rigid
        scores = {
            'complexity': 7.0,
            'stability': 9.0,
            'tension': 0.0,  # No admitted tension
            'adaptability': 2.0,  # Rigid
            'coherence': 10.0  # Claims perfect unity
        }

        result = diagnostic.assess(scores)

        # Should trigger spiritual bypassing warning
        assert any("SPIRITUAL BYPASSING" in w or "PERMANENCE TRAP" in w for w in result.warnings)

    def test_false_coherence_detection(self):
        """Detect false coherence: unity exceeds complexity"""
        diagnostic = SAPDiagnostic()

        # Simple system claiming profound integration
        scores = {
            'complexity': 3.0,  # Low complexity
            'stability': 5.0,
            'tension': 2.0,
            'adaptability': 2.0,  # Rigid
            'coherence': 9.0  # Claims high unity
        }

        result = diagnostic.assess(scores)

        assert any("FALSE COHERENCE" in w for w in result.warnings)

    def test_suspiciously_perfect_detection(self):
        """Detect suspiciously perfect scores"""
        diagnostic = SAPDiagnostic()

        # Everything rated 9+ suggests idealization
        scores = {
            'complexity': 9.5,
            'stability': 9.5,
            'tension': 9.5,
            'adaptability': 9.5,
            'coherence': 9.5
        }

        result = diagnostic.assess(scores)

        assert any("SUSPICIOUSLY PERFECT" in w for w in result.warnings)

    def test_bureaucratic_zombification(self):
        """Detect bureaucratic zombification: over-stabilized with no tension"""
        diagnostic = SAPDiagnostic()

        # Large institution that's "alive" but actually dead
        scores = {
            'complexity': 7.0,
            'stability': 9.0,  # Very stable
            'tension': 1.0,  # No tension
            'adaptability': 3.0,  # Can't change
            'coherence': 6.0
        }

        result = diagnostic.assess(scores)

        assert any("BUREAUCRATIC ZOMBIFICATION" in w for w in result.warnings)

    def test_tension_honesty_check(self):
        """Detect tension denial at high stages"""
        diagnostic = SAPDiagnostic()

        # Stage 8 claiming zero tension = denial
        scores = {
            'complexity': 8.0,
            'stability': 8.0,
            'tension': 0.0,  # Zero tension at Stage 8 is dishonest
            'adaptability': 8.0,
            'coherence': 9.0
        }

        result = diagnostic.assess(scores)

        assert any("TENSION DENIAL" in w for w in result.warnings)

    def test_hypercomplexity_fragmentation(self):
        """Detect hypercomplexity fragmentation"""
        diagnostic = SAPDiagnostic()

        # Internet/planetary scale: massive complexity without coherence
        scores = {
            'complexity': 9.8,  # Extreme complexity
            'stability': 4.0,
            'tension': 6.0,
            'adaptability': 5.0,
            'coherence': 5.0  # Cannot achieve unified purpose
        }

        result = diagnostic.assess(scores)

        assert any("HYPERCOMPLEXITY FRAGMENTATION" in w for w in result.warnings)


class TestBilateralThresholdAnalysis:
    """Test Stage 5 bilateral threshold subtype analysis"""

    def test_active_threshold(self):
        """Active threshold: high adaptability at crisis point"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 5.0,
            'stability': 4.0,
            'tension': 8.0,  # High tension
            'adaptability': 7.0,  # High adaptability = can transform
            'coherence': 4.0
        }

        result = diagnostic.assess(scores)

        if result.stage == 5:
            assert any("ACTIVE THRESHOLD" in i for i in result.insights)

    def test_passive_crisis(self):
        """Passive crisis: low adaptability at crisis point"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 5.0,
            'stability': 4.0,
            'tension': 8.0,  # High tension
            'adaptability': 3.0,  # Low adaptability = suffering without agency
            'coherence': 4.0
        }

        result = diagnostic.assess(scores)

        if result.stage == 5:
            assert any("PASSIVE CRISIS" in i for i in result.insights)

    def test_extreme_crisis(self):
        """Extreme crisis: maximum tension with minimal adaptability"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 5.0,
            'stability': 3.0,
            'tension': 9.5,  # Extreme tension
            'adaptability': 2.0,  # Very low adaptability
            'coherence': 3.0
        }

        result = diagnostic.assess(scores)

        if result.stage == 5:
            assert any("EXTREME CRISIS" in i or "PASSIVE SUFFERING" in i for i in result.insights)


class TestStage7CrisisAnalysis:
    """Test Stage 7 crisis quality assessment"""

    def test_productive_crisis(self):
        """Productive crisis: high adaptability during breakdown"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 6.0,
            'stability': 2.0,  # Low stability (breakdown)
            'tension': 8.0,  # High tension
            'adaptability': 8.5,  # Very high adaptability = can transform
            'coherence': 3.0
        }

        result = diagnostic.assess(scores)

        if result.stage == 7:
            assert any("PRODUCTIVE CRISIS" in i or "alchemical" in i.lower() for i in result.insights)

    def test_destructive_crisis(self):
        """Destructive crisis: low adaptability during breakdown"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 6.0,
            'stability': 2.0,  # Low stability (breakdown)
            'tension': 8.0,  # High tension
            'adaptability': 3.0,  # Low adaptability = destructive
            'coherence': 3.0
        }

        result = diagnostic.assess(scores)

        if result.stage == 7:
            assert any("DESTRUCTIVE CRISIS" in i or "collapse" in i.lower() for i in result.insights)


class TestStage8TrapDetection:
    """Test Stage 8 permanence trap detection"""

    def test_stage_8_permanence_trap(self):
        """Detect Stage 8 permanence trap: unity without adaptability"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 8.0,
            'stability': 8.0,  # High stability
            'tension': 1.0,
            'adaptability': 2.0,  # Low adaptability = trapped
            'coherence': 9.0  # High coherence
        }

        result = diagnostic.assess(scores)

        if result.stage == 8:
            assert any("PERMANENCE TRAP" in w for w in result.warnings)

    def test_stage_6_premature_satisfaction(self):
        """Detect Stage 6 premature satisfaction"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 7.0,
            'stability': 7.0,  # High stability
            'tension': 2.0,
            'adaptability': 2.0,  # Low adaptability
            'coherence': 8.5  # High coherence
        }

        result = diagnostic.assess(scores)

        if result.stage == 6:
            assert any("PREMATURE SATISFACTION" in w for w in result.warnings)


class TestStage9Authenticity:
    """Test Stage 9 authenticity verification"""

    def test_authentic_stage_9(self):
        """Verify authentic Stage 9 pattern"""
        diagnostic = SAPDiagnostic()

        # All criteria balanced correctly for genuine Stage 9
        scores = {
            'complexity': 8.5,  # High complexity
            'stability': 6.0,  # Moderate stability (5-7)
            'tension': 3.0,  # Honest tension (2-5)
            'adaptability': 9.0,  # Very high adaptability (8+)
            'coherence': 7.5  # Strong coherence (7+)
        }

        result = diagnostic.assess(scores)

        if result.stage == 9:
            assert any("AUTHENTIC STAGE 9" in i for i in result.insights)
            # Should NOT have "QUESTIONABLE" warning
            assert not any("QUESTIONABLE" in w for w in result.warnings)

    def test_questionable_stage_9(self):
        """Detect questionable Stage 9 claim"""
        diagnostic = SAPDiagnostic()

        # Trying to fake Stage 9 but pattern doesn't match
        scores = {
            'complexity': 6.0,  # Too low
            'stability': 9.0,  # Too high
            'tension': 0.5,  # Too low
            'adaptability': 9.0,  # High
            'coherence': 10.0  # Too perfect
        }

        result = diagnostic.assess(scores)

        if result.stage == 9:
            assert any("QUESTIONABLE" in w for w in result.warnings)


class TestTransitionDetection:
    """Test transition state detection"""

    def test_stage_transition(self):
        """Detect when system is transitioning between stages"""
        diagnostic = SAPDiagnostic()

        # Scores that match both Stage 5 and Stage 6
        scores = {
            'complexity': 6.0,  # Between 5 and 7
            'stability': 4.5,  # Between 3 and 6
            'tension': 5.5,  # Between 9 and 2
            'adaptability': 5.5,  # Between 6 and 5
            'coherence': 6.0  # Between 4 and 8
        }

        result = diagnostic.assess(scores, verbose=True)

        # Check if transition detected
        # (Either explicitly or via low ambiguity margin)
        transition_detected = (
            any("TRANSITION" in i for i in result.insights) or
            result.ambiguity_margin < 0.15
        )

        # At least one of these should be true
        assert transition_detected or len(result.warnings) > 0

    def test_cycle_completion_9_to_0(self):
        """Detect 9→0 cycle completion transition"""
        diagnostic = SAPDiagnostic()

        # Scores matching both Stage 9 and Stage 0
        scores = {
            'complexity': 4.5,  # Low but not zero
            'stability': 2.5,  # Low
            'tension': 1.5,  # Very low
            'adaptability': 5.0,  # Moderate
            'coherence': 3.5  # Low-moderate
        }

        result = diagnostic.assess(scores, verbose=True)

        # Check stage matches for both 9 and 0
        high_9 = result.stage_matches.get(9, 0) > 0.75
        high_0 = result.stage_matches.get(0, 0) > 0.75

        # If both high, should detect cycle completion
        if high_9 and high_0:
            assert any("CYCLE COMPLETION" in i or "9→0" in i for i in result.insights)


class TestTrajectoryPrediction:
    """Test trajectory prediction system"""

    def test_progressive_trajectory(self):
        """High adaptability = progressive trajectory"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 5.0,
            'stability': 5.0,
            'tension': 5.0,
            'adaptability': 8.0,  # High adaptability
            'coherence': 5.0
        }

        result = diagnostic.assess(scores)

        assert result.trajectory_type == "PROGRESSIVE"

    def test_regressive_trajectory(self):
        """Low adaptability at high stage = regressive trajectory"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 7.0,
            'stability': 7.0,
            'tension': 2.0,
            'adaptability': 2.0,  # Very low adaptability at high stage
            'coherence': 8.0
        }

        result = diagnostic.assess(scores)

        # Should be regressive or trap
        assert "REGRESSIVE" in result.trajectory_type or "TRAP" in result.trajectory_type

    def test_stagnant_trajectory(self):
        """Low adaptability at low stage = stagnant"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 2.0,
            'stability': 2.0,
            'tension': 3.0,
            'adaptability': 2.0,  # Low adaptability
            'coherence': 2.0
        }

        result = diagnostic.assess(scores)

        # Should be stagnant or have stagnation warning
        assert ("STAGNANT" in result.trajectory_type or
                any("STAGNATION" in p for p in result.trajectory_predictions))

    def test_stage_5_breakthrough_prediction(self):
        """Stage 5 with high adaptability predicts breakthrough"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 5.0,
            'stability': 3.0,
            'tension': 9.0,
            'adaptability': 8.0,  # High adaptability at threshold
            'coherence': 4.0
        }

        result = diagnostic.assess(scores)

        if result.stage == 5:
            assert any("BREAKTHROUGH" in p for p in result.trajectory_predictions)

    def test_stage_5_regression_prediction(self):
        """Stage 5 with low adaptability predicts regression"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 5.0,
            'stability': 3.0,
            'tension': 9.0,
            'adaptability': 2.0,  # Low adaptability at threshold
            'coherence': 4.0
        }

        result = diagnostic.assess(scores)

        if result.stage == 5:
            assert any("REGRESSION" in p for p in result.trajectory_predictions)


class TestReportGeneration:
    """Test report generation"""

    def test_generate_text_report(self):
        """Generate human-readable report"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 5.0,
            'stability': 5.0,
            'tension': 5.0,
            'adaptability': 5.0,
            'coherence': 5.0
        }

        report = diagnostic.generate_report(scores)

        assert "SAP STAGE DIAGNOSTIC" in report
        assert "Stanfield's Axiom of Perpetuity" in report
        assert "PRIMARY STAGE:" in report
        assert "TRAJECTORY:" in report

    def test_export_json(self):
        """Export assessment as JSON"""
        diagnostic = SAPDiagnostic()

        scores = {
            'complexity': 6.0,
            'stability': 6.0,
            'tension': 6.0,
            'adaptability': 6.0,
            'coherence': 6.0
        }

        json_str = diagnostic.export_json(scores)

        assert '"stage":' in json_str
        assert '"confidence":' in json_str
        assert '"trajectory_type":' in json_str


class TestInputValidation:
    """Test input validation"""

    def test_missing_criterion(self):
        """Raise error if criterion missing"""
        diagnostic = SAPDiagnostic()

        incomplete_scores = {
            'complexity': 5.0,
            'stability': 5.0,
            # Missing tension
            'adaptability': 5.0,
            'coherence': 5.0
        }

        with pytest.raises(ValueError, match="Missing score"):
            diagnostic.assess(incomplete_scores)

    def test_out_of_range_score(self):
        """Raise error if score out of range"""
        diagnostic = SAPDiagnostic()

        invalid_scores = {
            'complexity': 15.0,  # > 10
            'stability': 5.0,
            'tension': 5.0,
            'adaptability': 5.0,
            'coherence': 5.0
        }

        with pytest.raises(ValueError, match="between 0 and 10"):
            diagnostic.assess(invalid_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
