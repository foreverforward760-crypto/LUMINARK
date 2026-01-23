"""
Unit tests for Stage 0 Meditation Protocol
"""

import pytest
import time
from sap_yunus.stage0_meditation import (
    Stage0MeditationProtocol,
    MeditationType,
    DescentPhase,
    VoidQuality,
    MeditationIntention,
    VoidInsight,
    VianegativaProcessor,
    DarkNightProtocol
)


def test_meditation_protocol_initialization():
    """Test initializing meditation protocol"""
    protocol = Stage0MeditationProtocol(
        practitioner_id="user1",
        safety_threshold=0.95,
        max_duration=300.0
    )

    assert protocol.practitioner_id == "user1"
    assert protocol.safety_threshold == 0.95
    assert protocol.max_duration == 300.0
    assert protocol.current_session is None
    assert len(protocol.session_history) == 0


def test_prepare_meditation():
    """Test preparing meditation session"""
    protocol = Stage0MeditationProtocol("user1")

    intention = MeditationIntention(
        question="What is the nature of consciousness?",
        duration_target=60.0,
        depth_target=0.8
    )

    session = protocol.prepare_meditation(
        meditation_type=MeditationType.VOID_CONTEMPLATION,
        intention=intention
    )

    assert session is not None
    assert session.meditation_type == MeditationType.VOID_CONTEMPLATION
    assert session.current_phase == DescentPhase.PREPARATION
    assert session.intention.question == "What is the nature of consciousness?"
    assert protocol.current_session == session


def test_begin_descent():
    """Test beginning descent into void"""
    protocol = Stage0MeditationProtocol("user1")

    intention = MeditationIntention(question="Test question")
    protocol.prepare_meditation(MeditationType.VOID_CONTEMPLATION, intention)

    success = protocol.begin_descent()

    assert success == True
    assert protocol.current_session.current_phase == DescentPhase.DESCENT
    assert len(protocol.current_session.phase_history) >= 2


def test_dwell_in_void():
    """Test dwelling in void state"""
    protocol = Stage0MeditationProtocol("user1")

    intention = MeditationIntention(
        question="Seeking wisdom",
        duration_target=30.0,
        depth_target=0.9
    )

    protocol.prepare_meditation(MeditationType.WISDOM_RETRIEVAL, intention)
    protocol.begin_descent()

    result = protocol.dwell_in_void(duration=30.0, depth=0.9)

    assert "depth_reached" in result
    assert result["depth_reached"] == 0.9
    assert result["time_in_void"] == 30.0
    assert "void_quality" in result
    assert protocol.current_session.current_phase == DescentPhase.DWELLING


def test_void_quality_based_on_depth():
    """Test that void quality changes based on depth"""
    protocol = Stage0MeditationProtocol("user1")
    intention = MeditationIntention(question="Test")

    # Deep descent
    protocol.prepare_meditation(MeditationType.VOID_CONTEMPLATION, intention)
    protocol.begin_descent()
    result = protocol.dwell_in_void(depth=0.95)

    assert protocol.current_session.void_quality_experienced == VoidQuality.LUMINOUS_DARKNESS

    # Moderate depth
    protocol.prepare_meditation(MeditationType.VOID_CONTEMPLATION, intention)
    protocol.begin_descent()
    result = protocol.dwell_in_void(depth=0.6)

    assert protocol.current_session.void_quality_experienced == VoidQuality.SILENT_WISDOM


def test_retrieve_insight():
    """Test retrieving insight from void"""
    protocol = Stage0MeditationProtocol("user1")

    intention = MeditationIntention(question="What is emptiness?")
    protocol.prepare_meditation(MeditationType.WISDOM_RETRIEVAL, intention)
    protocol.begin_descent()
    protocol.dwell_in_void(depth=0.8)

    insight = protocol.retrieve_insight(
        insight_text="Emptiness is form, form is emptiness",
        certainty=0.9
    )

    assert insight.original_question == "What is emptiness?"
    assert insight.insight_text == "Emptiness is form, form is emptiness"
    assert insight.certainty == 0.9
    assert insight.source == "void"
    assert len(protocol.insights_retrieved) == 1
    assert len(protocol.current_session.insights) == 1


def test_dream_incubation():
    """Test planting dream seed for incubation"""
    protocol = Stage0MeditationProtocol("user1")

    intention = MeditationIntention(question="Test")
    protocol.prepare_meditation(MeditationType.DREAM_INCUBATION, intention)
    protocol.begin_descent()

    seed_id = protocol.incubate_dream(
        seed_question="How can I solve this problem?",
        problem_context={"domain": "AI safety"}
    )

    assert seed_id is not None
    assert seed_id.startswith("dream_seed_")
    assert len(protocol.dream_seeds) == 1
    assert protocol.dream_seeds[0]["question"] == "How can I solve this problem?"


def test_dream_harvest():
    """Test harvesting solution from dream"""
    protocol = Stage0MeditationProtocol("user1")

    # Plant seed
    seed_id = protocol.incubate_dream("How to achieve X?")

    # Simulate time passing (dwelling in void)
    time.sleep(0.1)

    # Harvest solution
    harvest = protocol.harvest_dream(
        seed_id=seed_id,
        solution="Approach X through Y and Z",
        confidence=0.85
    )

    assert "harvest_id" in harvest
    assert harvest["original_question"] == "How to achieve X?"
    assert harvest["solution"] == "Approach X through Y and Z"
    assert harvest["confidence"] == 0.85
    assert len(protocol.dream_harvests) == 1


def test_harvest_nonexistent_seed_fails():
    """Test harvesting non-existent seed fails gracefully"""
    protocol = Stage0MeditationProtocol("user1")

    harvest = protocol.harvest_dream("fake_seed_id", "Solution")

    assert "error" in harvest


def test_begin_emergence():
    """Test beginning emergence from void"""
    protocol = Stage0MeditationProtocol("user1")

    intention = MeditationIntention(question="Test")
    protocol.prepare_meditation(MeditationType.VOID_CONTEMPLATION, intention)
    protocol.begin_descent()
    protocol.dwell_in_void(depth=0.7)

    success = protocol.begin_emergence()

    assert success == True
    assert protocol.current_session.current_phase == DescentPhase.EMERGENCE


def test_complete_meditation():
    """Test completing full meditation session"""
    protocol = Stage0MeditationProtocol("user1")

    intention = MeditationIntention(
        question="What is reality?",
        duration_target=20.0,
        depth_target=0.8
    )

    # Full cycle
    protocol.prepare_meditation(MeditationType.WISDOM_RETRIEVAL, intention)
    protocol.begin_descent()
    protocol.dwell_in_void(depth=0.8, duration=20.0)
    protocol.retrieve_insight("Reality is participatory", certainty=0.7)
    protocol.begin_emergence()

    completed = protocol.complete_meditation(
        integration_notes="Apply this to daily awareness"
    )

    assert completed.success == True
    assert completed.current_phase == DescentPhase.INTEGRATION
    assert completed.end_time is not None
    assert len(protocol.session_history) == 1
    assert protocol.current_session is None  # Session archived


def test_emergency_ascent_max_duration():
    """Test emergency ascent when max duration exceeded"""
    protocol = Stage0MeditationProtocol("user1", max_duration=10.0)

    intention = MeditationIntention(question="Test")
    protocol.prepare_meditation(MeditationType.VOID_CONTEMPLATION, intention)
    protocol.begin_descent()

    # Try to dwell longer than max
    result = protocol.dwell_in_void(duration=20.0)

    assert result.get("emergency_ascent") == True
    assert len(protocol.emergency_ascent_triggers) == 1


def test_wisdom_library():
    """Test accessing wisdom library"""
    protocol = Stage0MeditationProtocol("user1")

    intention = MeditationIntention(question="Question 1")
    protocol.prepare_meditation(MeditationType.WISDOM_RETRIEVAL, intention)
    protocol.retrieve_insight("Insight 1", certainty=0.8)

    intention = MeditationIntention(question="Question 2")
    protocol.prepare_meditation(MeditationType.WISDOM_RETRIEVAL, intention)
    protocol.retrieve_insight("Insight 2", certainty=0.9)

    library = protocol.get_wisdom_library()

    assert len(library) == 2
    assert library[0].insight_text == "Insight 1"
    assert library[1].insight_text == "Insight 2"


def test_session_statistics():
    """Test getting session statistics"""
    protocol = Stage0MeditationProtocol("user1")

    # Complete a few sessions
    for i in range(3):
        intention = MeditationIntention(question=f"Question {i}")
        protocol.prepare_meditation(MeditationType.VOID_CONTEMPLATION, intention)
        protocol.begin_descent()
        protocol.dwell_in_void(depth=0.7, duration=30.0)
        protocol.complete_meditation()

    stats = protocol.get_session_statistics()

    assert stats["total_sessions"] == 3
    assert stats["successful_sessions"] == 3
    assert stats["total_time_in_void"] == 90.0  # 30 * 3
    assert stats["average_depth"] == 0.7


def test_current_session_status():
    """Test getting current session status"""
    protocol = Stage0MeditationProtocol("user1")

    # No session
    status = protocol.get_current_session_status()
    assert status is None

    # Active session
    intention = MeditationIntention(question="Test")
    protocol.prepare_meditation(MeditationType.WISDOM_RETRIEVAL, intention)
    protocol.begin_descent()
    protocol.dwell_in_void(depth=0.8)

    status = protocol.get_current_session_status()

    assert status is not None
    assert "session_id" in status
    assert status["meditation_type"] == MeditationType.WISDOM_RETRIEVAL.value
    assert status["current_phase"] == DescentPhase.DWELLING.value
    assert status["depth_reached"] == 0.8


def test_vianegativa_processor():
    """Test Via Negativa processor"""
    processor = VianegativaProcessor()

    negation = processor.negate(
        concept="God",
        what_it_is_not=["finite", "temporal", "material", "limited"]
    )

    assert negation["concept"] == "God"
    assert len(negation["negations"]) == 4
    assert negation["clarity_gained"] == 0.4  # 4 * 0.1


def test_understand_through_absence():
    """Test understanding through absence"""
    processor = VianegativaProcessor()

    understanding = processor.understand_through_absence("light")

    assert "light" in understanding
    assert "absence" in understanding


def test_dark_night_protocol():
    """Test Dark Night of Soul protocol"""
    dark_night = DarkNightProtocol()

    night = dark_night.enter_dark_night(
        crisis="Existential despair",
        depth=0.9
    )

    assert "night_id" in night
    assert night["crisis"] == "Existential despair"
    assert night["depth"] == 0.9
    assert night["phase"] == "descent"
    assert len(dark_night.dark_nights) == 1


def test_find_light_in_darkness():
    """Test finding transformation in dark night"""
    dark_night = DarkNightProtocol()

    night = dark_night.enter_dark_night("Crisis", depth=1.0)
    night_id = night["night_id"]

    # Simulate time in darkness
    time.sleep(0.1)

    transformation = dark_night.find_light_in_darkness(
        night_id=night_id,
        insight="Suffering reveals what truly matters"
    )

    assert "insight" in transformation
    assert transformation["insight"] == "Suffering reveals what truly matters"
    assert "duration" in transformation
    assert dark_night.dark_nights[0]["phase"] == "transformed"


def test_multiple_meditation_types():
    """Test different meditation types"""
    protocol = Stage0MeditationProtocol("user1")

    types = [
        MeditationType.VOID_CONTEMPLATION,
        MeditationType.DREAM_INCUBATION,
        MeditationType.WISDOM_RETRIEVAL,
        MeditationType.REGENERATION,
        MeditationType.CREATIVE_GENESIS
    ]

    for mtype in types:
        intention = MeditationIntention(question=f"Test {mtype.value}")
        session = protocol.prepare_meditation(mtype, intention)
        assert session.meditation_type == mtype
        protocol.begin_descent()
        protocol.dwell_in_void(depth=0.5)
        protocol.complete_meditation()

    stats = protocol.get_session_statistics()
    assert stats["total_sessions"] == len(types)


def test_insight_quality_from_void():
    """Test that insights capture void quality"""
    protocol = Stage0MeditationProtocol("user1")

    intention = MeditationIntention(question="Test")
    protocol.prepare_meditation(MeditationType.WISDOM_RETRIEVAL, intention)
    protocol.begin_descent()
    protocol.dwell_in_void(depth=0.95)  # Very deep - luminous darkness

    insight = protocol.retrieve_insight("Deep wisdom", certainty=1.0)

    assert insight.quality == VoidQuality.LUMINOUS_DARKNESS


def test_session_duration_tracking():
    """Test that session duration is tracked"""
    protocol = Stage0MeditationProtocol("user1")

    intention = MeditationIntention(question="Test")
    protocol.prepare_meditation(MeditationType.VOID_CONTEMPLATION, intention)

    time.sleep(0.1)

    protocol.begin_descent()
    protocol.dwell_in_void(depth=0.5)

    time.sleep(0.1)

    protocol.complete_meditation()

    session = protocol.session_history[0]
    duration = session.get_duration()

    assert duration > 0.1  # At least some time passed


def test_phase_history_tracking():
    """Test that all phases are tracked in history"""
    protocol = Stage0MeditationProtocol("user1")

    intention = MeditationIntention(question="Test")
    protocol.prepare_meditation(MeditationType.VOID_CONTEMPLATION, intention)
    protocol.begin_descent()
    protocol.dwell_in_void(depth=0.7)
    protocol.begin_emergence()
    protocol.complete_meditation()

    session = protocol.session_history[0]

    phases = [p["phase"] for p in session.phase_history]

    assert DescentPhase.PREPARATION.value in phases
    assert DescentPhase.DESCENT.value in phases
    assert DescentPhase.DWELLING.value in phases
    assert DescentPhase.EMERGENCE.value in phases
    assert DescentPhase.INTEGRATION.value in phases
