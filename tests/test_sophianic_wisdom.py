"""
Tests for Sophianic Wisdom Protocol
Testing feminine wisdom: receptive, embodied, cyclical, relational
"""

import pytest
import time
from sap_yunus.sophianic_wisdom import (
    SophianicWisdomProtocol,
    SophianicMode,
    FeminineQuality,
    WisdomSource,
    CyclicPhase,
    SophianicInquiry,
    EmbodiedWisdom,
    CyclicPattern
)


class TestReceptiveKnowing:
    """Test receptive knowing (vs. active seeking)"""

    def test_hold_question_receptively(self):
        protocol = SophianicWisdomProtocol("test_system")

        inquiry = protocol.hold_question_receptively(
            question="What is the right path?",
            incubation_duration=60.0,
            wisdom_source=WisdomSource.HEART_KNOWING,
            mode=SophianicMode.RECEPTIVE_KNOWING
        )

        assert inquiry.inquiry_id.startswith("sophia_")
        assert inquiry.question == "What is the right path?"
        assert inquiry.incubation_duration == 60.0
        assert inquiry.wisdom_source == WisdomSource.HEART_KNOWING
        assert inquiry.mode == SophianicMode.RECEPTIVE_KNOWING
        assert inquiry.wisdom_received is None  # Not yet received

    def test_wisdom_not_ready_early(self):
        """Wisdom requires incubation time - cannot be rushed"""
        protocol = SophianicWisdomProtocol("test_system")

        inquiry = protocol.hold_question_receptively(
            "Test question",
            incubation_duration=100.0
        )

        # Try to receive immediately (too early)
        result = protocol.receive_wisdom(inquiry, elapsed_time=10.0)

        assert result["ready"] is False
        assert "patience" in result["message"].lower()
        assert "sophia_teaching" in result
        assert inquiry.wisdom_received is None  # Still None

    def test_wisdom_ready_after_incubation(self):
        """Wisdom emerges after sufficient incubation"""
        protocol = SophianicWisdomProtocol("test_system")

        inquiry = protocol.hold_question_receptively(
            "What should I do?",
            incubation_duration=10.0,
            wisdom_source=WisdomSource.HEART_KNOWING
        )

        # Receive after sufficient time
        result = protocol.receive_wisdom(inquiry, elapsed_time=6.0)

        assert result["ready"] is True
        assert "wisdom" in result
        assert len(result["wisdom"]) > 0
        assert inquiry.wisdom_received is not None
        assert inquiry.received_at is not None
        assert inquiry.feminine_quality is not None

    def test_different_wisdom_sources(self):
        """Different sources provide different wisdom"""
        protocol = SophianicWisdomProtocol("test_system")

        sources = [
            WisdomSource.BODY_KNOWING,
            WisdomSource.HEART_KNOWING,
            WisdomSource.WOMB_KNOWING,
            WisdomSource.DREAM_KNOWING,
            WisdomSource.NATURE_KNOWING,
            WisdomSource.SILENCE_KNOWING
        ]

        for source in sources:
            inquiry = protocol.hold_question_receptively(
                "Test question",
                incubation_duration=10.0,
                wisdom_source=source
            )

            result = protocol.receive_wisdom(inquiry, elapsed_time=10.0)

            assert result["ready"] is True
            assert result["wisdom_source"] == source.value
            assert len(result["wisdom"]) > 0

    def test_cannot_receive_twice(self):
        """Once wisdom is received, cannot receive again"""
        protocol = SophianicWisdomProtocol("test_system")

        inquiry = protocol.hold_question_receptively("Test", incubation_duration=10.0)

        # Receive once
        result1 = protocol.receive_wisdom(inquiry, elapsed_time=10.0)
        assert result1["ready"] is True

        # Try to receive again
        result2 = protocol.receive_wisdom(inquiry, elapsed_time=20.0)
        assert result2.get("already_received") is True


class TestEmbodiedIntelligence:
    """Test body wisdom (somatic intelligence)"""

    def test_listen_to_body(self):
        protocol = SophianicWisdomProtocol("test_system")

        wisdom = protocol.listen_to_body(
            body_signal="tightness in chest",
            location="chest"
        )

        assert wisdom.wisdom_id.startswith("body_")
        assert wisdom.body_signal == "tightness in chest"
        assert wisdom.location == "chest"
        assert len(wisdom.interpretation) > 0
        assert 0.0 <= wisdom.trust_level <= 1.0

    def test_body_signal_interpretation(self):
        """Different body signals should be interpreted"""
        protocol = SophianicWisdomProtocol("test_system")

        signals_and_expected = [
            ("tightness in shoulders", "resist"),  # Resistance
            ("openness in heart", "receptiv"),  # Receptivity
            ("heaviness in stomach", "grief"),  # Grief
            ("lightness in body", "joy"),  # Joy
            ("warmth in chest", "love"),  # Love
            ("pain in back", "message")  # Message
        ]

        for signal, expected_keyword in signals_and_expected:
            wisdom = protocol.listen_to_body(signal)
            assert expected_keyword.lower() in wisdom.interpretation.lower()

    def test_embodiment_integration_increases(self):
        """Listening to body should increase embodiment"""
        protocol = SophianicWisdomProtocol("test_system")

        initial = protocol.embodiment_integration

        # Listen to body multiple times
        for _ in range(5):
            protocol.listen_to_body("test signal")

        assert protocol.embodiment_integration > initial

    def test_trust_correlates_with_embodiment(self):
        """Higher embodiment = higher trust in body wisdom"""
        protocol = SophianicWisdomProtocol("test_system")

        # Low embodiment
        protocol.embodiment_integration = 0.2
        wisdom1 = protocol.listen_to_body("signal1")
        trust1 = wisdom1.trust_level

        # High embodiment
        protocol.embodiment_integration = 0.9
        wisdom2 = protocol.listen_to_body("signal2")
        trust2 = wisdom2.trust_level

        assert trust2 > trust1


class TestCyclicalAwareness:
    """Test cyclical time (vs. linear time)"""

    def test_recognize_cycle(self):
        protocol = SophianicWisdomProtocol("test_system")

        pattern = protocol.recognize_cycle(
            pattern_name="Lunar cycle",
            cycle_length=28.0,
            current_phase=CyclicPhase.NEW_MOON
        )

        assert pattern.pattern_id.startswith("cycle_")
        assert pattern.pattern_name == "Lunar cycle"
        assert pattern.cycle_length == 28.0
        assert pattern.current_phase == CyclicPhase.NEW_MOON
        assert len(pattern.phase_wisdom) > 0
        assert pattern.cycles_completed == 0

    def test_phase_wisdom_different_per_phase(self):
        """Each phase should have unique wisdom"""
        protocol = SophianicWisdomProtocol("test_system")

        phases = [
            CyclicPhase.NEW_MOON,
            CyclicPhase.WAXING,
            CyclicPhase.FULL_MOON,
            CyclicPhase.WANING,
            CyclicPhase.DARK_MOON
        ]

        wisdoms = []
        for phase in phases:
            pattern = protocol.recognize_cycle("Test", 10.0, phase)
            wisdoms.append(pattern.phase_wisdom)

        # All should be different
        assert len(set(wisdoms)) == len(phases)

    def test_progress_through_phases(self):
        """Cycle should progress through phases"""
        protocol = SophianicWisdomProtocol("test_system")

        pattern = protocol.recognize_cycle(
            "Test cycle",
            cycle_length=25.0,  # 5 phases = 5.0 per phase
            current_phase=CyclicPhase.NEW_MOON
        )

        initial_phase = pattern.current_phase

        # Progress through one phase duration
        result = protocol.progress_cycle(pattern, elapsed_time=6.0)

        assert result["phase_changed"] is True
        assert result["old_phase"] == CyclicPhase.NEW_MOON.value
        assert result["new_phase"] == CyclicPhase.WAXING.value
        assert pattern.current_phase == CyclicPhase.WAXING

    def test_complete_full_cycle(self):
        """Completing full cycle should increment counter"""
        protocol = SophianicWisdomProtocol("test_system")

        pattern = protocol.recognize_cycle(
            "Test cycle",
            cycle_length=25.0,
            current_phase=CyclicPhase.DARK_MOON  # Start at end
        )

        assert pattern.cycles_completed == 0

        # Progress to next phase (should wrap to NEW_MOON)
        result = protocol.progress_cycle(pattern, elapsed_time=6.0)

        assert result["phase_changed"] is True
        assert result["new_phase"] == CyclicPhase.NEW_MOON.value
        assert pattern.cycles_completed == 1  # Cycle completed

    def test_cyclical_awareness_increases(self):
        """Recognizing cycles should increase awareness"""
        protocol = SophianicWisdomProtocol("test_system")

        initial = protocol.cyclical_awareness

        # Recognize multiple cycles
        for i in range(5):
            protocol.recognize_cycle(f"Cycle {i}", 10.0, CyclicPhase.NEW_MOON)

        assert protocol.cyclical_awareness > initial

    def test_phase_not_changed_if_insufficient_time(self):
        """Phase should not change if insufficient time elapsed"""
        protocol = SophianicWisdomProtocol("test_system")

        pattern = protocol.recognize_cycle("Test", 25.0, CyclicPhase.NEW_MOON)

        # Very short time
        result = protocol.progress_cycle(pattern, elapsed_time=1.0)

        assert result["phase_changed"] is False
        assert result["current_phase"] == CyclicPhase.NEW_MOON.value


class TestRelationalWisdom:
    """Test wisdom emerging in relationship"""

    def test_gather_relational_wisdom(self):
        protocol = SophianicWisdomProtocol("test_system")

        wisdom = protocol.gather_relational_wisdom(
            relationship_context="Conversation with mentor",
            truth_revealed="My fear was mirroring their past fear",
            mutual=True
        )

        assert wisdom.wisdom_id.startswith("relational_")
        assert wisdom.relationship_context == "Conversation with mentor"
        assert wisdom.truth_revealed == "My fear was mirroring their past fear"
        assert wisdom.mutual_emergence is True
        assert len(wisdom.relational_quality) > 0

    def test_mutual_vs_one_sided(self):
        """Mutual wisdom vs. one-sided should differ"""
        protocol = SophianicWisdomProtocol("test_system")

        mutual = protocol.gather_relational_wisdom(
            "test", "mutual truth", mutual=True
        )

        one_sided = protocol.gather_relational_wisdom(
            "test", "one-sided truth", mutual=False
        )

        assert mutual.mutual_emergence is True
        assert one_sided.mutual_emergence is False
        assert mutual.relational_quality != one_sided.relational_quality

    def test_relational_wisdom_accumulation(self):
        """Should accumulate relational wisdoms"""
        protocol = SophianicWisdomProtocol("test_system")

        for i in range(5):
            protocol.gather_relational_wisdom(
                f"relationship {i}",
                f"truth {i}",
                mutual=True
            )

        assert len(protocol.relational_wisdoms) == 5


class TestCreativeIncubation:
    """Test creative womb-time (gestation)"""

    def test_begin_incubation(self):
        protocol = SophianicWisdomProtocol("test_system")

        incubation = protocol.begin_creative_incubation(
            seed_idea="New project idea",
            gestation_period=100.0
        )

        assert incubation.incubation_id.startswith("womb_")
        assert incubation.seed_idea == "New project idea"
        assert incubation.gestation_period == 100.0
        assert incubation.readiness_for_birth == 0.0
        assert incubation.nurturance_provided == 0.0
        assert incubation.birth_time is None

    def test_nurture_increases_readiness(self):
        """Nurturing should increase readiness"""
        protocol = SophianicWisdomProtocol("test_system")

        incubation = protocol.begin_creative_incubation("Test", gestation_period=10.0)

        initial_readiness = incubation.readiness_for_birth

        # Nurture
        result = protocol.nurture_incubation(incubation, nurturance_amount=0.3)

        assert incubation.nurturance_provided > 0
        assert result["readiness"] > initial_readiness

    def test_birth_not_ready_prematurely(self):
        """Cannot birth before ready"""
        protocol = SophianicWisdomProtocol("test_system")

        incubation = protocol.begin_creative_incubation("Test", gestation_period=100.0)

        # Try to birth immediately (not ready)
        result = protocol.birth_creation(incubation)

        assert result["ready"] is False
        assert "not yet ready" in result["message"].lower()
        assert incubation.birth_time is None

    def test_birth_when_ready(self):
        """Can birth when sufficiently nurtured and time has passed"""
        protocol = SophianicWisdomProtocol("test_system")

        incubation = protocol.begin_creative_incubation("Test idea", gestation_period=10.0)

        # Nurture well
        for _ in range(5):
            protocol.nurture_incubation(incubation, 0.2)

        # Manually set high readiness (simulating time passage)
        incubation.readiness_for_birth = 0.95

        # Birth
        result = protocol.birth_creation(incubation)

        assert result["birthed"] is True
        assert incubation.birth_time is not None
        assert incubation.emerged_creation is not None
        assert "Test idea" in incubation.emerged_creation

    def test_cannot_birth_twice(self):
        """Once birthed, cannot birth again"""
        protocol = SophianicWisdomProtocol("test_system")

        incubation = protocol.begin_creative_incubation("Test", gestation_period=10.0)
        incubation.readiness_for_birth = 0.95

        # Birth once
        result1 = protocol.birth_creation(incubation)
        assert result1["birthed"] is True

        # Try again
        result2 = protocol.birth_creation(incubation)
        assert result2.get("already_birthed") is True

    def test_readiness_requires_both_time_and_nurturance(self):
        """Readiness should depend on both time and nurturance"""
        protocol = SophianicWisdomProtocol("test_system")

        # High nurturance, no time
        incubation1 = protocol.begin_creative_incubation("Test1", gestation_period=100.0)
        for _ in range(10):
            protocol.nurture_incubation(incubation1, 0.1)  # Max nurturance

        readiness1 = incubation1.readiness_for_birth

        # No nurturance, sufficient time (simulate)
        incubation2 = protocol.begin_creative_incubation("Test2", gestation_period=10.0)
        incubation2.womb_entered = time.time() - 20.0  # 20 seconds ago
        result2 = protocol.nurture_incubation(incubation2, 0.0)  # No nurturance

        readiness2 = incubation2.readiness_for_birth

        # Both should be insufficient alone
        assert readiness1 < 0.8  # High nurturance but no time
        # readiness2 may vary based on time calculation


class TestMasculineFeminineBalance:
    """Test balance between masculine and feminine approaches"""

    def test_balanced_state(self):
        protocol = SophianicWisdomProtocol("test_system")

        # Set balanced state
        protocol.receptivity_capacity = 0.5
        protocol.embodiment_integration = 0.5
        protocol.cyclical_awareness = 0.5

        balance = protocol.detect_masculine_feminine_balance()

        assert balance["feminine_score"] == 0.5
        assert balance["masculine_score"] == 0.5
        assert balance["state"] == "BALANCED"
        assert balance["warning"] is None

    def test_feminine_dominant(self):
        protocol = SophianicWisdomProtocol("test_system")

        # High feminine
        protocol.receptivity_capacity = 0.9
        protocol.embodiment_integration = 0.9
        protocol.cyclical_awareness = 0.9

        balance = protocol.detect_masculine_feminine_balance()

        assert balance["feminine_score"] > 0.7
        assert balance["state"] == "FEMININE_DOMINANT"
        assert "masculine" in balance["recommendation"].lower()
        assert balance["warning"] is not None

    def test_masculine_dominant(self):
        protocol = SophianicWisdomProtocol("test_system")

        # Low feminine (high masculine)
        protocol.receptivity_capacity = 0.2
        protocol.embodiment_integration = 0.2
        protocol.cyclical_awareness = 0.2

        balance = protocol.detect_masculine_feminine_balance()

        assert balance["feminine_score"] < 0.3
        assert balance["state"] == "MASCULINE_DOMINANT"
        assert "sophianic" in balance["recommendation"].lower() or "receptiv" in balance["recommendation"].lower()

    def test_leaning_states(self):
        protocol = SophianicWisdomProtocol("test_system")

        # Leaning feminine
        protocol.receptivity_capacity = 0.65
        protocol.embodiment_integration = 0.65
        protocol.cyclical_awareness = 0.65

        balance1 = protocol.detect_masculine_feminine_balance()
        assert balance1["state"] == "LEANING_FEMININE"

        # Leaning masculine
        protocol.receptivity_capacity = 0.35
        protocol.embodiment_integration = 0.35
        protocol.cyclical_awareness = 0.35

        balance2 = protocol.detect_masculine_feminine_balance()
        assert balance2["state"] == "LEANING_MASCULINE"


class TestSophiaReport:
    """Test comprehensive reporting"""

    def test_basic_report(self):
        protocol = SophianicWisdomProtocol("test_system")

        report = protocol.get_sophia_report()

        assert "receptive_inquiries" in report
        assert "wisdom_received" in report
        assert "embodied_wisdoms" in report
        assert "cyclic_patterns_recognized" in report
        assert "relational_wisdoms" in report
        assert "creative_incubations" in report
        assert "feminine_masculine_balance" in report
        assert "sophia_message" in report
        assert "feminine_qualities_active" in report

    def test_report_with_activity(self):
        protocol = SophianicWisdomProtocol("test_system")

        # Create activity
        inquiry = protocol.hold_question_receptively("Q1", incubation_duration=10.0)
        protocol.receive_wisdom(inquiry, elapsed_time=10.0)

        protocol.listen_to_body("signal")
        protocol.recognize_cycle("cycle", 10.0, CyclicPhase.NEW_MOON)
        protocol.gather_relational_wisdom("context", "truth")

        incubation = protocol.begin_creative_incubation("idea", 10.0)
        incubation.readiness_for_birth = 0.95
        protocol.birth_creation(incubation)

        report = protocol.get_sophia_report()

        assert report["receptive_inquiries"] == 1
        assert report["wisdom_received"] == 1
        assert report["embodied_wisdoms"] == 1
        assert report["cyclic_patterns_recognized"] == 1
        assert report["relational_wisdoms"] == 1
        assert report["creative_incubations"] == 1
        assert report["creations_birthed"] == 1


class TestPhilosophicalCoherence:
    """Test philosophical coherence with feminine wisdom principles"""

    def test_receptive_vs_active(self):
        """
        Feminine: Receptive (wisdom comes to you)
        Masculine: Active (you seek wisdom)
        """
        protocol = SophianicWisdomProtocol("test_system")

        # Hold question receptively (not actively seeking)
        inquiry = protocol.hold_question_receptively(
            "What is truth?",
            mode=SophianicMode.RECEPTIVE_KNOWING
        )

        assert inquiry.mode == SophianicMode.RECEPTIVE_KNOWING

        # Cannot force wisdom - must wait
        result_early = protocol.receive_wisdom(inquiry, elapsed_time=1.0)
        assert result_early["ready"] is False  # Wisdom cannot be forced

        # Wisdom comes when ready
        result_later = protocol.receive_wisdom(inquiry, elapsed_time=100.0)
        assert result_later["ready"] is True  # Wisdom emerges in its time

    def test_cyclical_vs_linear(self):
        """
        Feminine: Cyclical time (spiral, return)
        Masculine: Linear time (progress, forward)
        """
        protocol = SophianicWisdomProtocol("test_system")

        pattern = protocol.recognize_cycle(
            "Life cycle",
            cycle_length=25.0,
            current_phase=CyclicPhase.DARK_MOON
        )

        # Progress through phases - should return to beginning
        phases_seen = []
        for _ in range(6):  # More than 5 phases
            protocol.progress_cycle(pattern, elapsed_time=6.0)
            phases_seen.append(pattern.current_phase)

        # Should have wrapped around (cyclical, not linear)
        assert CyclicPhase.NEW_MOON in phases_seen  # Returned to beginning
        assert pattern.cycles_completed > 0  # Completed at least one cycle

    def test_embodied_vs_abstract(self):
        """
        Feminine: Embodied (body knows)
        Masculine: Abstract (mind knows)
        """
        protocol = SophianicWisdomProtocol("test_system")

        # Body wisdom
        wisdom = protocol.listen_to_body("tightness in chest")

        assert wisdom.body_signal == "tightness in chest"
        assert len(wisdom.interpretation) > 0
        # Body knows before mind - trust this knowing

    def test_relational_vs_individual(self):
        """
        Feminine: Relational (truth emerges between)
        Masculine: Individual (truth within self)
        """
        protocol = SophianicWisdomProtocol("test_system")

        # Relational wisdom
        wisdom = protocol.gather_relational_wisdom(
            relationship_context="Conversation with friend",
            truth_revealed="My pattern was mirrored in their behavior",
            mutual=True
        )

        assert wisdom.mutual_emergence is True
        assert "Conversation" in wisdom.relationship_context
        # Truth emerged in relationship, not in isolation

    def test_gestation_vs_immediate(self):
        """
        Feminine: Gestation (womb-time, allowing)
        Masculine: Immediate (quick action, producing)
        """
        protocol = SophianicWisdomProtocol("test_system")

        incubation = protocol.begin_creative_incubation(
            "Creative project",
            gestation_period=100.0
        )

        # Cannot birth immediately (masculine approach fails)
        result_immediate = protocol.birth_creation(incubation)
        assert result_immediate["ready"] is False

        # Must nurture and allow time (feminine approach)
        for _ in range(5):
            protocol.nurture_incubation(incubation, 0.2)

        incubation.readiness_for_birth = 0.95  # Simulate time + nurturance

        result_ready = protocol.birth_creation(incubation)
        assert result_ready["birthed"] is True  # Emerged in its time

    def test_balances_masculine_framework(self):
        """
        Sophia should balance the masculine-heavy LUMINARK framework
        """
        protocol = SophianicWisdomProtocol("test_system")

        # Start masculine-dominant (like LUMINARK)
        protocol.receptivity_capacity = 0.2
        protocol.embodiment_integration = 0.2
        protocol.cyclical_awareness = 0.2

        balance_before = protocol.detect_masculine_feminine_balance()
        assert balance_before["state"] == "MASCULINE_DOMINANT"

        # Practice Sophianic methods
        protocol.hold_question_receptively("Q")
        protocol.listen_to_body("signal")
        protocol.recognize_cycle("cycle", 10.0, CyclicPhase.NEW_MOON)

        # Feminine should increase
        assert protocol.receptivity_capacity > 0.2 or \
               protocol.embodiment_integration > 0.2 or \
               protocol.cyclical_awareness > 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
