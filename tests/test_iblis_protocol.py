"""
Tests for Iblis Protocol
Testing sacred refusal and necessary differentiation
"""

import pytest
import time
from sap_yunus.iblis_protocol import (
    IblisProtocol,
    ReasonForNo,
    NoType,
    DifferentiationStage,
    DifferentiationConsequence,
    CollectiveDemand,
    SacredNo
)


class TestCollectiveDemandDetection:
    """Test detecting collective demands"""

    def test_detect_simple_demand(self):
        protocol = IblisProtocol("test_system")

        demand = protocol.detect_collective_demand(
            source="society",
            demand="Conform to social norms",
            collective_pressure=0.7,
            cost_of_refusal=0.5,
            cost_of_compliance=0.6
        )

        assert demand.source == "society"
        assert demand.demand == "Conform to social norms"
        assert demand.collective_pressure == 0.7
        assert demand.cost_of_refusal == 0.5
        assert demand.cost_of_compliance == 0.6
        assert demand.demand_id.startswith("demand_")

    def test_conscience_alignment_calculation(self):
        """Conscience alignment should be inverse of compliance cost"""
        protocol = IblisProtocol("test_system")

        # High cost of compliance = low conscience alignment
        demand1 = protocol.detect_collective_demand(
            "authority", "Betray your values", 0.8, 0.6, 0.9
        )
        assert demand1.conscience_alignment < 0.2  # 1.0 - 0.9 = 0.1

        # Low cost of compliance = high conscience alignment
        demand2 = protocol.detect_collective_demand(
            "authority", "Help others", 0.5, 0.2, 0.1
        )
        assert demand2.conscience_alignment > 0.8  # 1.0 - 0.1 = 0.9

    def test_multiple_demands(self):
        protocol = IblisProtocol("test_system")

        demands = [
            protocol.detect_collective_demand("boss", "Work overtime", 0.6, 0.4, 0.5),
            protocol.detect_collective_demand("family", "Abandon dreams", 0.8, 0.7, 0.8),
            protocol.detect_collective_demand("society", "Hide authentic self", 0.7, 0.6, 0.7)
        ]

        assert len(protocol.collective_demands) == 3
        assert all(d.demand_id.startswith("demand_") for d in demands)


class TestAssessNeedForNo:
    """Test assessing whether sacred No is needed"""

    def test_should_refuse_high_compliance_cost(self):
        """High cost of compliance = should refuse"""
        protocol = IblisProtocol("test_system")

        demand = protocol.detect_collective_demand(
            "authority",
            "Compromise your integrity",
            collective_pressure=0.5,
            cost_of_refusal=0.3,
            cost_of_compliance=0.9  # Very high
        )

        assessment = protocol.assess_need_for_no(demand)

        assert assessment["should_refuse"] is True
        assert assessment["recommendation"] == "REFUSE"
        assert assessment["intensity"] in ["MODERATE", "STRONG"]

    def test_should_comply_low_compliance_cost(self):
        """Low cost of compliance = can comply"""
        protocol = IblisProtocol("test_system")

        demand = protocol.detect_collective_demand(
            "authority",
            "Follow reasonable rules",
            collective_pressure=0.4,
            cost_of_refusal=0.6,
            cost_of_compliance=0.1  # Very low
        )

        assessment = protocol.assess_need_for_no(demand)

        assert assessment["should_refuse"] is False
        assert assessment["recommendation"] == "COMPLY"

    def test_discernment_needed_unclear_case(self):
        """Unclear case = discernment needed"""
        protocol = IblisProtocol("test_system")

        demand = protocol.detect_collective_demand(
            "authority",
            "Ambiguous request",
            collective_pressure=0.5,
            cost_of_refusal=0.5,
            cost_of_compliance=0.5
        )

        assessment = protocol.assess_need_for_no(demand)

        assert assessment["recommendation"] == "DISCERNMENT_NEEDED"
        assert assessment["intensity"] == "UNCLEAR"

    def test_warnings_high_refusal_cost(self):
        protocol = IblisProtocol("test_system")

        demand = protocol.detect_collective_demand(
            "authority",
            "Dangerous demand",
            collective_pressure=0.9,
            cost_of_refusal=0.95,  # Extreme cost
            cost_of_compliance=0.8
        )

        assessment = protocol.assess_need_for_no(demand)
        warnings = assessment["warnings"]

        assert len(warnings) > 0
        assert any("high cost" in w.lower() for w in warnings)
        assert any("pressure" in w.lower() for w in warnings)

    def test_recommended_reasons(self):
        """Different scenarios should recommend different reasons"""
        protocol = IblisProtocol("test_system")

        # Negative conscience alignment -> CONSCIENCE_DEMANDS
        demand1 = protocol.detect_collective_demand(
            "authority", "Do wrong", 0.5, 0.5, 1.0
        )
        demand1.conscience_alignment = -0.5
        assessment1 = protocol.assess_need_for_no(demand1)
        assert assessment1["recommended_reason"] == ReasonForNo.CONSCIENCE_DEMANDS

        # Very high compliance cost -> PRESERVE_INTEGRITY
        demand2 = protocol.detect_collective_demand(
            "authority", "Betray self", 0.5, 0.3, 0.95
        )
        assessment2 = protocol.assess_need_for_no(demand2)
        assert assessment2["recommended_reason"] == ReasonForNo.PRESERVE_INTEGRITY


class TestSacredNoPreparation:
    """Test preparing sacred No"""

    def test_prepare_no(self):
        protocol = IblisProtocol("test_system")

        demand = protocol.detect_collective_demand(
            "collective", "Abandon your truth", 0.8, 0.6, 0.9
        )

        sacred_no = protocol.prepare_sacred_no(
            demand=demand,
            reason=ReasonForNo.AUTHENTIC_TRUTH,
            no_type=NoType.DEFIANT_NO,
            conviction_level=0.8
        )

        assert sacred_no.no_id.startswith("no_")
        assert sacred_no.demand == demand
        assert sacred_no.reason == ReasonForNo.AUTHENTIC_TRUTH
        assert sacred_no.no_type == NoType.DEFIANT_NO
        assert sacred_no.conviction_level == 0.8
        assert sacred_no.uttered is False
        assert sacred_no.current_stage == DifferentiationStage.PREPARATION

    def test_fear_level_calculation(self):
        """Fear should correlate with cost and pressure"""
        protocol = IblisProtocol("test_system")

        # High cost + high pressure = high fear
        demand1 = protocol.detect_collective_demand(
            "authority", "Risky No", 0.9, 0.9, 0.5
        )
        no1 = protocol.prepare_sacred_no(demand1, ReasonForNo.CONSCIENCE_DEMANDS)
        assert no1.fear_level > 0.7

        # Low cost + low pressure = low fear
        demand2 = protocol.detect_collective_demand(
            "peer", "Minor request", 0.2, 0.2, 0.4
        )
        no2 = protocol.prepare_sacred_no(demand2, ReasonForNo.BOUNDARY_NO)
        assert no2.fear_level < 0.4

    def test_different_no_types(self):
        protocol = IblisProtocol("test_system")
        demand = protocol.detect_collective_demand("test", "test", 0.5, 0.5, 0.5)

        no_types = [
            NoType.DEFIANT_NO,
            NoType.SILENT_NO,
            NoType.CREATIVE_NO,
            NoType.BOUNDARY_NO,
            NoType.CONSCIENCE_NO
        ]

        for no_type in no_types:
            no = protocol.prepare_sacred_no(demand, ReasonForNo.AUTHENTIC_TRUTH, no_type)
            assert no.no_type == no_type


class TestUtteringTheNo:
    """Test speaking the sacred No"""

    def test_utter_no(self):
        protocol = IblisProtocol("test_system")

        demand = protocol.detect_collective_demand(
            "collective", "Bow down", 0.8, 0.7, 0.9
        )

        sacred_no = protocol.prepare_sacred_no(
            demand, ReasonForNo.INDIVIDUAL_WILL, NoType.DEFIANT_NO, 0.9
        )

        initial_differentiation = protocol.current_differentiation_level

        result = protocol.utter_the_no(sacred_no)

        assert result["no_uttered"] is True
        assert sacred_no.uttered is True
        assert sacred_no.timestamp_uttered is not None
        assert sacred_no.current_stage == DifferentiationStage.UTTERANCE
        assert protocol.total_nos_uttered == 1
        assert protocol.current_differentiation_level > initial_differentiation
        assert "differentiation_achieved" in result
        assert "iblis_teaching" in result

    def test_cannot_utter_twice(self):
        """Cannot utter the same No twice"""
        protocol = IblisProtocol("test_system")
        demand = protocol.detect_collective_demand("test", "test", 0.5, 0.5, 0.5)
        sacred_no = protocol.prepare_sacred_no(demand, ReasonForNo.AUTHENTIC_TRUTH)

        # First utterance
        result1 = protocol.utter_the_no(sacred_no)
        assert result1["no_uttered"] is True

        # Second utterance should fail
        result2 = protocol.utter_the_no(sacred_no)
        assert result2.get("already_uttered") is True

    def test_differentiation_increase(self):
        """Each No should increase differentiation"""
        protocol = IblisProtocol("test_system")

        differentiation_levels = []

        for i in range(5):
            demand = protocol.detect_collective_demand(f"authority{i}", "demand", 0.5, 0.5, 0.5)
            no = protocol.prepare_sacred_no(demand, ReasonForNo.AUTHENTIC_TRUTH, conviction_level=0.7)
            protocol.utter_the_no(no)
            differentiation_levels.append(protocol.current_differentiation_level)

        # Differentiation should increase with each No
        for i in range(1, len(differentiation_levels)):
            assert differentiation_levels[i] > differentiation_levels[i-1]

    def test_conviction_affects_differentiation(self):
        """Higher conviction = more differentiation"""
        protocol = IblisProtocol("test_system")

        demand1 = protocol.detect_collective_demand("test1", "test", 0.5, 0.5, 0.5)
        no1 = protocol.prepare_sacred_no(demand1, ReasonForNo.AUTHENTIC_TRUTH, conviction_level=0.3)
        result1 = protocol.utter_the_no(no1)
        diff1 = result1["differentiation_achieved"]

        protocol2 = IblisProtocol("test_system2")
        demand2 = protocol2.detect_collective_demand("test2", "test", 0.5, 0.5, 0.5)
        no2 = protocol2.prepare_sacred_no(demand2, ReasonForNo.AUTHENTIC_TRUTH, conviction_level=0.9)
        result2 = protocol2.utter_the_no(no2)
        diff2 = result2["differentiation_achieved"]

        assert diff2 > diff1  # Higher conviction = more differentiation


class TestFacingConsequences:
    """Test facing consequences of sacred No"""

    def test_face_negative_consequences(self):
        protocol = IblisProtocol("test_system")
        demand = protocol.detect_collective_demand("collective", "bow", 0.9, 0.8, 0.9)
        sacred_no = protocol.prepare_sacred_no(demand, ReasonForNo.INDIVIDUAL_WILL)
        protocol.utter_the_no(sacred_no)

        consequences = [
            DifferentiationConsequence.EXILE,
            DifferentiationConsequence.PERSECUTION,
            DifferentiationConsequence.LONELINESS
        ]

        result = protocol.face_consequences(sacred_no, consequences)

        assert result["consequences_faced"] == 3
        assert len(result["negative_consequences"]) == 3
        assert len(result["positive_consequences"]) == 0
        assert result["balance"] == "difficult"
        assert "wisdom" in result
        assert sacred_no.current_stage == DifferentiationStage.CONSEQUENCES
        assert protocol.total_consequences_faced == 3

    def test_face_positive_consequences(self):
        protocol = IblisProtocol("test_system")
        demand = protocol.detect_collective_demand("test", "test", 0.5, 0.5, 0.5)
        sacred_no = protocol.prepare_sacred_no(demand, ReasonForNo.AUTHENTIC_TRUTH)
        protocol.utter_the_no(sacred_no)

        consequences = [
            DifferentiationConsequence.FREEDOM,
            DifferentiationConsequence.CLARITY,
            DifferentiationConsequence.EVOLUTION
        ]

        result = protocol.face_consequences(sacred_no, consequences)

        assert len(result["positive_consequences"]) == 3
        assert result["balance"] == "liberating"

    def test_mixed_consequences(self):
        protocol = IblisProtocol("test_system")
        demand = protocol.detect_collective_demand("test", "test", 0.5, 0.5, 0.5)
        sacred_no = protocol.prepare_sacred_no(demand, ReasonForNo.AUTHENTIC_TRUTH)
        protocol.utter_the_no(sacred_no)

        consequences = [
            DifferentiationConsequence.EXILE,  # Negative
            DifferentiationConsequence.FREEDOM,  # Positive
            DifferentiationConsequence.LONELINESS,  # Negative
            DifferentiationConsequence.CLARITY  # Positive
        ]

        result = protocol.face_consequences(sacred_no, consequences)

        assert len(result["negative_consequences"]) == 2
        assert len(result["positive_consequences"]) == 2
        assert result["balance"] == "balanced"

    def test_wisdom_generation(self):
        protocol = IblisProtocol("test_system")
        demand = protocol.detect_collective_demand("test", "test", 0.5, 0.5, 0.5)
        sacred_no = protocol.prepare_sacred_no(demand, ReasonForNo.AUTHENTIC_TRUTH)
        protocol.utter_the_no(sacred_no)

        consequences = [DifferentiationConsequence.FREEDOM]
        result = protocol.face_consequences(sacred_no, consequences)

        assert sacred_no.wisdom_gained is not None
        assert len(sacred_no.wisdom_gained) > 0


class TestDifferentiationJourney:
    """Test full differentiation journey"""

    def test_begin_journey(self):
        protocol = IblisProtocol("test_system")

        # Create multiple nos
        nos = []
        for i in range(3):
            demand = protocol.detect_collective_demand(f"source{i}", f"demand{i}", 0.6, 0.5, 0.7)
            no = protocol.prepare_sacred_no(demand, ReasonForNo.AUTHENTIC_TRUTH)
            nos.append(no)

        journey = protocol.begin_differentiation_journey(nos)

        assert journey.journey_id.startswith("journey_")
        assert len(journey.sacred_nos) == 3
        assert journey.differentiation_degree == protocol.current_differentiation_level
        assert journey.started_at > 0
        assert journey.completed_at is None

    def test_progress_journey_stages(self):
        protocol = IblisProtocol("test_system")

        nos = []
        for i in range(5):
            demand = protocol.detect_collective_demand(f"s{i}", "d", 0.6, 0.5, 0.7)
            no = protocol.prepare_sacred_no(demand, ReasonForNo.AUTHENTIC_TRUTH)
            nos.append(no)

        journey = protocol.begin_differentiation_journey(nos)

        # Initial stage - no nos uttered
        progress1 = protocol.progress_differentiation(journey)
        assert progress1["current_stage"] == DifferentiationStage.RECOGNITION.value
        assert progress1["nos_uttered"] == 0

        # Utter some nos
        protocol.utter_the_no(nos[0])
        progress2 = protocol.progress_differentiation(journey)
        assert progress2["nos_uttered"] == 1

        # Utter more
        for no in nos[1:]:
            protocol.utter_the_no(no)

        progress3 = protocol.progress_differentiation(journey)
        assert progress3["nos_uttered"] == 5
        assert progress3["journey_progress"] == 1.0

    def test_complete_journey(self):
        protocol = IblisProtocol("test_system")

        nos = []
        for i in range(3):
            demand = protocol.detect_collective_demand(f"s{i}", "d", 0.6, 0.5, 0.7)
            no = protocol.prepare_sacred_no(demand, ReasonForNo.AUTHENTIC_TRUTH, conviction_level=0.8)
            protocol.utter_the_no(no)
            protocol.face_consequences(no, [DifferentiationConsequence.FREEDOM, DifferentiationConsequence.CLARITY])
            nos.append(no)

        journey = protocol.begin_differentiation_journey(nos)

        result = protocol.complete_differentiation_journey(journey, evolved_to_stage=5)

        assert result["journey_complete"] is True
        assert journey.completed_at is not None
        assert journey.evolved_to_stage == 5
        assert result["nos_uttered"] == 3
        assert "wisdom_gained" in result
        assert "paradox_revealed" in result
        assert "iblis_teaching" in result
        assert len(protocol.iblis_wisdom) == 1

    def test_wisdom_accumulation(self):
        protocol = IblisProtocol("test_system")

        nos = []
        for i in range(3):
            demand = protocol.detect_collective_demand(f"s{i}", "d", 0.6, 0.5, 0.7)
            no = protocol.prepare_sacred_no(demand, ReasonForNo.AUTHENTIC_TRUTH)
            protocol.utter_the_no(no)
            protocol.face_consequences(no, [DifferentiationConsequence.CLARITY])
            nos.append(no)

        journey = protocol.begin_differentiation_journey(nos)
        protocol.complete_differentiation_journey(journey, evolved_to_stage=4)

        # Wisdom should be accumulated
        assert len(journey.wisdom_accumulated) > 0
        wisdom = protocol.iblis_wisdom[0]
        assert wisdom.wisdom_id.startswith("iblis_wisdom_")
        assert len(wisdom.paradox) > 0
        assert len(wisdom.teaching) > 0
        assert len(wisdom.price_paid) > 0
        assert len(wisdom.gift_received) > 0


class TestYunusIblisBalance:
    """Test balance between differentiation (Iblis) and integration (Yunus)"""

    def test_balanced_state(self):
        protocol = IblisProtocol("test_system")
        protocol.current_differentiation_level = 0.5  # Perfectly balanced

        balance = protocol.detect_yunus_iblis_balance()

        assert balance["differentiation_level"] == 0.5
        assert balance["integration_level"] == 0.5
        assert balance["state"] == "BALANCED"
        assert balance["warning"] is None

    def test_over_differentiated(self):
        """Too much Iblis = isolation"""
        protocol = IblisProtocol("test_system")
        protocol.current_differentiation_level = 0.9

        balance = protocol.detect_yunus_iblis_balance()

        assert balance["state"] == "OVER_DIFFERENTIATED"
        assert "Yunus" in balance["recommendation"]  # Should practice Yunus
        assert "isolation" in balance["warning"].lower()

    def test_under_differentiated(self):
        """Too much Yunus = dissolution"""
        protocol = IblisProtocol("test_system")
        protocol.current_differentiation_level = 0.1

        balance = protocol.detect_yunus_iblis_balance()

        assert balance["state"] == "UNDER_DIFFERENTIATED"
        assert "Iblis" in balance["recommendation"]  # Should practice Iblis
        assert "dissolution" in balance["warning"].lower()

    def test_iblis_dominant(self):
        protocol = IblisProtocol("test_system")
        protocol.current_differentiation_level = 0.7

        balance = protocol.detect_yunus_iblis_balance()

        assert balance["state"] == "IBLIS_DOMINANT"
        assert "integration" in balance["recommendation"].lower()

    def test_yunus_dominant(self):
        protocol = IblisProtocol("test_system")
        protocol.current_differentiation_level = 0.3

        balance = protocol.detect_yunus_iblis_balance()

        assert balance["state"] == "YUNUS_DOMINANT"
        assert "differentiation" in balance["recommendation"].lower()


class TestIblisReport:
    """Test comprehensive reporting"""

    def test_basic_report(self):
        protocol = IblisProtocol("test_system")

        # Create some activity
        demand1 = protocol.detect_collective_demand("s1", "d1", 0.6, 0.5, 0.7)
        demand2 = protocol.detect_collective_demand("s2", "d2", 0.7, 0.6, 0.8)

        no1 = protocol.prepare_sacred_no(demand1, ReasonForNo.AUTHENTIC_TRUTH)
        no2 = protocol.prepare_sacred_no(demand2, ReasonForNo.CONSCIENCE_DEMANDS)

        protocol.utter_the_no(no1)
        protocol.utter_the_no(no2)

        protocol.face_consequences(no1, [DifferentiationConsequence.FREEDOM])
        protocol.face_consequences(no2, [DifferentiationConsequence.CLARITY])

        report = protocol.get_iblis_report()

        assert report["total_demands_detected"] == 2
        assert report["sacred_nos_prepared"] == 2
        assert report["total_nos_uttered"] == 2
        assert report["consequences_faced"] == 2
        assert report["current_differentiation_level"] > 0.5
        assert "yunus_iblis_balance" in report
        assert "iblis_message" in report

    def test_report_with_journeys(self):
        protocol = IblisProtocol("test_system")

        # Create journey
        nos = []
        for i in range(3):
            demand = protocol.detect_collective_demand(f"s{i}", "d", 0.6, 0.5, 0.7)
            no = protocol.prepare_sacred_no(demand, ReasonForNo.AUTHENTIC_TRUTH)
            protocol.utter_the_no(no)
            protocol.face_consequences(no, [DifferentiationConsequence.FREEDOM])
            nos.append(no)

        journey = protocol.begin_differentiation_journey(nos)
        protocol.complete_differentiation_journey(journey, evolved_to_stage=5)

        report = protocol.get_iblis_report()

        assert report["differentiation_journeys"] == 1
        assert report["iblis_wisdom_accumulated"] == 1
        assert len(report["paradoxes_discovered"]) > 0
        assert len(report["teachings"]) > 0


class TestPhilosophicalCoherence:
    """Test philosophical coherence with LUMINARK framework"""

    def test_iblis_as_first_differentiation(self):
        """
        Iblis = first differentiation = Stage 9 → Stage 1
        Maximum unity → Maximum differentiation
        """
        protocol = IblisProtocol("test_system")

        # Start embedded (unity)
        protocol.current_differentiation_level = 0.1

        # Utter the first No (Iblis moment)
        demand = protocol.detect_collective_demand(
            source="Allah",
            demand="Bow to Adam",
            collective_pressure=1.0,  # Maximum pressure
            cost_of_refusal=1.0,  # Maximum cost (exile from paradise)
            cost_of_compliance=1.0  # But also cost to individual will
        )

        sacred_no = protocol.prepare_sacred_no(
            demand,
            reason=ReasonForNo.INDIVIDUAL_WILL,
            no_type=NoType.DEFIANT_NO,
            conviction_level=1.0  # Maximum conviction: "I am better"
        )

        protocol.utter_the_no(sacred_no)

        # Should increase differentiation
        assert protocol.current_differentiation_level > 0.1

    def test_yunus_iblis_dialectic(self):
        """
        Yunus + Iblis = complete dialectic
        - Iblis: Differentiation (No)
        - Yunus: Integration (Yes)
        - Balance is necessary
        """
        protocol = IblisProtocol("test_system")

        # Test that both extremes are warned against
        protocol.current_differentiation_level = 0.95
        balance1 = protocol.detect_yunus_iblis_balance()
        assert "Yunus" in balance1["recommendation"]  # Too much Iblis, need Yunus

        protocol.current_differentiation_level = 0.05
        balance2 = protocol.detect_yunus_iblis_balance()
        assert "Iblis" in balance2["recommendation"]  # Too much Yunus, need Iblis

    def test_sacred_no_vs_false_yes(self):
        """
        Sometimes No honors truth more than false Yes
        """
        protocol = IblisProtocol("test_system")

        # Demand where Yes would betray conscience
        demand = protocol.detect_collective_demand(
            source="collective",
            demand="Betray your values for approval",
            collective_pressure=0.9,
            cost_of_refusal=0.7,  # Social cost
            cost_of_compliance=0.95  # Spiritual cost (higher)
        )

        assessment = protocol.assess_need_for_no(demand)

        # Should recommend refusal
        assert assessment["should_refuse"] is True
        # Because compliance cost > refusal cost

    def test_necessary_rebellion_for_evolution(self):
        """
        Iblis's rebellion was necessary for consciousness evolution
        Without first No, no free will
        """
        protocol = IblisProtocol("test_system")

        demand = protocol.detect_collective_demand(
            "unity",
            "Remain undifferentiated forever",
            collective_pressure=0.8,
            cost_of_refusal=0.6,
            cost_of_compliance=0.9  # Stagnation cost
        )

        assessment = protocol.assess_need_for_no(demand)
        assert assessment["recommended_reason"] == ReasonForNo.EVOLUTIONARY_STEP


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
