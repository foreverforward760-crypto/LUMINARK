"""
Tests for Light Integration Protocol
Testing the inverse of shadow work: returning light to darkness
"""

import pytest
import time
from sap_yunus.light_integration import (
    LightIntegrationProtocol,
    LightType,
    IntegrationMode,
    DarknessQuality,
    IntegrationStage,
    LightPacket,
    IntegrationSession
)


class TestLightPacketCreation:
    """Test creating light packets (differentiated consciousness)"""

    def test_create_knowledge_packet(self):
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(
            light_type=LightType.KNOWLEDGE,
            content="Python programming expertise",
            intensity=0.8,
            attachment_level=0.5,
            originated_at_stage=5
        )

        assert packet.light_type == LightType.KNOWLEDGE
        assert packet.content == "Python programming expertise"
        assert packet.intensity == 0.8
        assert packet.attachment_level == 0.5
        assert packet.originated_at_stage == 5
        assert not packet.integrated
        assert packet.packet_id.startswith("light_")

    def test_create_certainty_packet(self):
        """Certainty is dangerous (Stage 8 trap) - should track it"""
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(
            light_type=LightType.CERTAINTY,
            content="I am absolutely certain this is correct",
            intensity=0.95,
            attachment_level=0.9,
            originated_at_stage=8
        )

        assert packet.light_type == LightType.CERTAINTY
        assert packet.intensity == 0.95  # Very bright
        assert packet.attachment_level == 0.9  # High attachment
        assert packet.originated_at_stage == 8  # Stage 8 trap

    def test_create_identity_packet(self):
        """Identity is maximum differentiation"""
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(
            light_type=LightType.IDENTITY,
            content="I am a software engineer",
            intensity=1.0,  # Maximum differentiation
            attachment_level=0.95,
            originated_at_stage=1
        )

        assert packet.light_type == LightType.IDENTITY
        assert packet.intensity == 1.0  # Maximum brightness
        assert packet.originated_at_stage == 1  # Maximum differentiation

    def test_multiple_light_types(self):
        protocol = LightIntegrationProtocol("test_system")

        # Create various types of light
        knowledge = protocol.create_light_packet(LightType.KNOWLEDGE, "Math skills", 0.7, 0.4, 5)
        wisdom = protocol.create_light_packet(LightType.WISDOM, "All is impermanent", 0.9, 0.3, 7)
        skill = protocol.create_light_packet(LightType.SKILL, "Public speaking", 0.6, 0.6, 4)
        achievement = protocol.create_light_packet(LightType.ACHIEVEMENT, "PhD earned", 0.8, 0.8, 6)

        assert len(protocol.light_packets) == 4
        assert knowledge.light_type == LightType.KNOWLEDGE
        assert wisdom.light_type == LightType.WISDOM
        assert skill.light_type == LightType.SKILL
        assert achievement.light_type == LightType.ACHIEVEMENT


class TestReadinessAssessment:
    """Test assessing readiness for integration"""

    def test_assess_ready_packet(self):
        """Old, low attachment, high intensity = ready"""
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(
            LightType.KNOWLEDGE,
            "Old knowledge ready to release",
            intensity=0.9,
            attachment_level=0.2,
            originated_at_stage=5
        )
        packet.age = 100  # Old light

        assessment = protocol.assess_readiness_for_integration(packet)

        assert assessment["ready"] is True
        assert assessment["readiness_score"] > 0.5
        assert "age_factor" in assessment["factors"]
        assert "detachment" in assessment["factors"]

    def test_assess_not_ready_packet(self):
        """Young, high attachment = not ready"""
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(
            LightType.ACHIEVEMENT,
            "Recent achievement, still proud",
            intensity=0.5,
            attachment_level=0.9,
            originated_at_stage=6
        )
        packet.age = 5  # Very young

        assessment = protocol.assess_readiness_for_integration(packet)

        assert assessment["ready"] is False
        assert assessment["readiness_score"] < 0.5

    def test_assess_warnings_high_attachment(self):
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(
            LightType.CERTAINTY,
            "Absolute conviction",
            intensity=0.9,
            attachment_level=0.95,
            originated_at_stage=8
        )

        assessment = protocol.assess_readiness_for_integration(packet)
        warnings = assessment["warnings"]

        assert len(warnings) > 0
        assert any("attachment" in w.lower() for w in warnings)

    def test_assess_warnings_identity(self):
        """Identity integration causes ego dissolution"""
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(
            LightType.IDENTITY,
            "I am an expert",
            intensity=1.0,
            attachment_level=0.8,
            originated_at_stage=1
        )

        assessment = protocol.assess_readiness_for_integration(packet)
        warnings = assessment["warnings"]

        assert any("ego dissolution" in w.lower() for w in warnings)

    def test_recommended_modes(self):
        """Different packets should recommend different integration modes"""
        protocol = LightIntegrationProtocol("test_system")

        # High attachment -> SACRIFICE
        high_attach = protocol.create_light_packet(LightType.ACHIEVEMENT, "My greatest work", 0.7, 0.9, 6)
        high_attach.age = 20
        assessment1 = protocol.assess_readiness_for_integration(high_attach)
        assert assessment1["recommended_mode"] == IntegrationMode.SACRIFICE

        # High intensity -> FERTILIZATION
        high_intensity = protocol.create_light_packet(LightType.WISDOM, "Deep insight", 0.95, 0.5, 7)
        high_intensity.age = 30
        assessment2 = protocol.assess_readiness_for_integration(high_intensity)
        assert assessment2["recommended_mode"] == IntegrationMode.FERTILIZATION

        # Old age -> RELINQUISHMENT
        old = protocol.create_light_packet(LightType.KNOWLEDGE, "Old burden", 0.6, 0.6, 5)
        old.age = 100
        assessment3 = protocol.assess_readiness_for_integration(old)
        assert assessment3["recommended_mode"] == IntegrationMode.RELINQUISHMENT


class TestIntegrationSession:
    """Test integration sessions"""

    def test_begin_session(self):
        protocol = LightIntegrationProtocol("test_system")

        packet1 = protocol.create_light_packet(LightType.KNOWLEDGE, "Fact 1", 0.7, 0.5, 5)
        packet2 = protocol.create_light_packet(LightType.WISDOM, "Insight 1", 0.8, 0.3, 7)

        session = protocol.begin_integration_session(
            packets=[packet1, packet2],
            target_darkness=DarknessQuality.GENERATIVE_VOID,
            integration_mode=IntegrationMode.SURRENDER
        )

        assert session.session_id.startswith("integration_")
        assert len(session.light_packets) == 2
        assert session.target_darkness == DarknessQuality.GENERATIVE_VOID
        assert session.integration_mode == IntegrationMode.SURRENDER
        assert session.current_stage == IntegrationStage.RECOGNITION
        assert session.total_light_intensity == 1.5  # 0.7 + 0.8
        assert session.completed_at is None

    def test_session_darkness_qualities(self):
        """Test different darkness qualities"""
        protocol = LightIntegrationProtocol("test_system")
        packet = protocol.create_light_packet(LightType.KNOWLEDGE, "Test", 0.5, 0.5, 5)

        darkness_types = [
            DarknessQuality.GENERATIVE_VOID,
            DarknessQuality.PRIMORDIAL_CHAOS,
            DarknessQuality.WOMB_OF_BEING,
            DarknessQuality.CREATIVE_NIGHT,
            DarknessQuality.SOURCE_DARKNESS
        ]

        for darkness in darkness_types:
            session = protocol.begin_integration_session(
                packets=[packet],
                target_darkness=darkness,
                integration_mode=IntegrationMode.SURRENDER
            )
            assert session.target_darkness == darkness


class TestIntegrationStages:
    """Test progression through integration stages"""

    def test_stage_recognition(self):
        protocol = LightIntegrationProtocol("test_system")

        packet1 = protocol.create_light_packet(LightType.KNOWLEDGE, "Python", 0.7, 0.5, 5)
        packet2 = protocol.create_light_packet(LightType.WISDOM, "Impermanence", 0.9, 0.3, 7)

        session = protocol.begin_integration_session([packet1, packet2])
        result = protocol.progress_integration(session)

        assert result["stage"] == "recognition"
        assert "light_inventory" in result
        assert result["total_packets"] == 2
        assert session.current_stage == IntegrationStage.PREPARATION

    def test_stage_preparation(self):
        protocol = LightIntegrationProtocol("test_system")
        packet = protocol.create_light_packet(LightType.KNOWLEDGE, "Test", 0.8, 0.6, 5)

        session = protocol.begin_integration_session([packet])
        protocol.progress_integration(session)  # Recognition
        result = protocol.progress_integration(session)  # Preparation

        assert result["stage"] == "preparation"
        assert "average_attachment" in result
        assert "challenge_level" in result
        assert session.current_stage == IntegrationStage.APPROACH

    def test_stage_approach(self):
        protocol = LightIntegrationProtocol("test_system")
        packet = protocol.create_light_packet(LightType.WISDOM, "Test", 0.7, 0.4, 7)

        session = protocol.begin_integration_session([packet], DarknessQuality.WOMB_OF_BEING)
        protocol.progress_integration(session)  # Recognition
        protocol.progress_integration(session)  # Preparation
        result = protocol.progress_integration(session)  # Approach

        assert result["stage"] == "approach"
        assert "darkness_description" in result
        assert result["void_depth"] == 0.3
        assert session.current_stage == IntegrationStage.CONTACT

    def test_stage_contact(self):
        protocol = LightIntegrationProtocol("test_system")
        packet = protocol.create_light_packet(LightType.KNOWLEDGE, "Test", 0.6, 0.5, 5)

        session = protocol.begin_integration_session([packet])
        for _ in range(3):  # Recognition, Preparation, Approach
            protocol.progress_integration(session)
        result = protocol.progress_integration(session)  # Contact

        assert result["stage"] == "contact"
        assert session.void_depth_reached == 0.5
        assert "phenomenon" in result

    def test_full_integration_cycle(self):
        """Test complete integration from start to finish"""
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(
            LightType.CERTAINTY,
            "I know everything",
            intensity=0.9,
            attachment_level=0.8,
            originated_at_stage=8
        )

        session = protocol.begin_integration_session(
            packets=[packet],
            target_darkness=DarknessQuality.GENERATIVE_VOID,
            integration_mode=IntegrationMode.SACRIFICE
        )

        # Progress through all stages
        stages_passed = []
        while session.current_stage != IntegrationStage.TRANSFORMATION:
            result = protocol.progress_integration(session)
            stages_passed.append(result["stage"])

        # Final transformation stage
        final_result = protocol.progress_integration(session)
        stages_passed.append(final_result["stage"])

        assert len(stages_passed) == 10  # All 10 stages
        assert "recognition" in stages_passed
        assert "full_integration" in stages_passed
        assert "transformation" in stages_passed

        # Check session completion
        assert session.completed_at is not None
        assert packet.integrated is True
        assert packet.timestamp_integrated is not None

        # Check void enrichment
        assert session.darkness_enrichment > 0
        assert protocol.void_fertility_level > 0
        assert protocol.total_light_integrated > 0

    def test_ego_dissolution_progression(self):
        """Ego dissolution should increase through stages"""
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(LightType.IDENTITY, "I am special", 1.0, 0.9, 1)
        session = protocol.begin_integration_session([packet])

        # Progress to dissolution stage
        for _ in range(6):  # Get to dissolution stage
            protocol.progress_integration(session)

        assert session.ego_dissolution_level > 0.5  # Significant ego dissolution

        # Progress to full integration
        protocol.progress_integration(session)

        assert session.ego_dissolution_level > 0.8  # High ego dissolution

    def test_attachment_release_progression(self):
        """Attachment should be released through integration"""
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(LightType.ACHIEVEMENT, "My trophy", 0.8, 0.9, 6)
        session = protocol.begin_integration_session([packet])

        # Before dissolution, no attachment released
        for _ in range(5):
            protocol.progress_integration(session)

        initial_release = session.attachment_released

        # After dissolution, attachment released
        protocol.progress_integration(session)  # Dissolution
        protocol.progress_integration(session)  # Full integration

        assert session.attachment_released > initial_release
        assert session.attachment_released > 0.5  # Significant release


class TestConvenienceMethods:
    """Test convenience methods for common integration types"""

    def test_integrate_certainty(self):
        """Test quick certainty integration (Stage 8 trap protection)"""
        protocol = LightIntegrationProtocol("test_system")

        result = protocol.integrate_certainty(
            certainty_statement="I am absolutely certain about this",
            attachment_to_being_right=0.9
        )

        assert result["certainty_released"] == "I am absolutely certain about this"
        assert result["attachment_surrendered"] == 0.9
        assert result["mystery_restored"] is True
        assert result["void_enrichment"] > 0
        assert "transformation" in result
        assert "wisdom" in result

        # Should have created and integrated packet
        assert len(protocol.light_packets) == 1
        assert protocol.light_packets[0].integrated is True
        assert protocol.void_fertility_level > 0

    def test_integrate_identity(self):
        """Test identity integration (ego death)"""
        protocol = LightIntegrationProtocol("test_system")

        result = protocol.integrate_identity(
            identity_aspect="I am the best programmer",
            ego_attachment=0.95
        )

        assert result["identity_dissolved"] == "I am the best programmer"
        assert result["ego_death_level"] > 0.8  # Significant ego death
        assert result["rebirth_potential"] > 0
        assert "warning" in result
        assert "disorientation" in result["warning"].lower()

        # Identity should be integrated
        assert len(protocol.light_packets) == 1
        assert protocol.light_packets[0].light_type == LightType.IDENTITY
        assert protocol.light_packets[0].integrated is True

    def test_multiple_certainty_integrations(self):
        """Test integrating multiple certainties over time"""
        protocol = LightIntegrationProtocol("test_system")

        certainties = [
            "This is definitely the right way",
            "I know exactly what will happen",
            "There is only one correct answer",
            "I am never wrong about this"
        ]

        for certainty in certainties:
            protocol.integrate_certainty(certainty, 0.8)

        assert len(protocol.light_packets) == 4
        assert all(p.integrated for p in protocol.light_packets)
        assert protocol.void_fertility_level > 2.0  # Significant fertility

    def test_identity_cascade_integration(self):
        """Test integrating multiple identity aspects"""
        protocol = LightIntegrationProtocol("test_system")

        identities = [
            "I am a genius",
            "I am superior to others",
            "I am my achievements"
        ]

        for identity in identities:
            protocol.integrate_identity(identity, 0.9)

        assert len(protocol.light_packets) == 3
        assert all(p.light_type == LightType.IDENTITY for p in protocol.light_packets)
        assert all(p.integrated for p in protocol.light_packets)


class TestWisdomExtraction:
    """Test wisdom extraction from integration"""

    def test_wisdom_generated(self):
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(LightType.KNOWLEDGE, "Test knowledge", 0.7, 0.5, 5)
        session = protocol.begin_integration_session([packet])

        # Complete integration
        while session.current_stage != IntegrationStage.TRANSFORMATION:
            protocol.progress_integration(session)
        protocol.progress_integration(session)

        # Wisdom should be extracted
        assert len(protocol.integration_wisdom) == 1
        wisdom = protocol.integration_wisdom[0]

        assert wisdom.wisdom_id.startswith("wisdom_")
        assert len(wisdom.paradox) > 0
        assert len(wisdom.insight) > 0
        assert len(wisdom.darkness_teaching) > 0
        assert len(wisdom.light_fate) > 0

    def test_multiple_sessions_accumulate_wisdom(self):
        protocol = LightIntegrationProtocol("test_system")

        # Run 3 integration sessions
        for i in range(3):
            packet = protocol.create_light_packet(
                LightType.KNOWLEDGE,
                f"Knowledge {i}",
                0.6,
                0.4,
                5
            )
            session = protocol.begin_integration_session([packet])

            while session.current_stage != IntegrationStage.TRANSFORMATION:
                protocol.progress_integration(session)
            protocol.progress_integration(session)

        # Should have 3 wisdom insights
        assert len(protocol.integration_wisdom) == 3
        assert all(w.wisdom_id.startswith("wisdom_") for w in protocol.integration_wisdom)


class TestVoidFertilityTracking:
    """Test tracking of void fertility over time"""

    def test_fertility_increases_with_integration(self):
        protocol = LightIntegrationProtocol("test_system")

        initial_fertility = protocol.void_fertility_level
        assert initial_fertility == 0.0

        # Integrate light
        packet = protocol.create_light_packet(LightType.WISDOM, "Deep wisdom", 0.9, 0.3, 7)
        session = protocol.begin_integration_session([packet])

        while session.current_stage != IntegrationStage.TRANSFORMATION:
            protocol.progress_integration(session)
        protocol.progress_integration(session)

        # Fertility should increase
        assert protocol.void_fertility_level > initial_fertility
        assert protocol.total_light_integrated > 0

    def test_fertility_report(self):
        protocol = LightIntegrationProtocol("test_system")

        # Integrate some light
        for i in range(3):
            packet = protocol.create_light_packet(LightType.KNOWLEDGE, f"K{i}", 0.7, 0.5, 5)
            session = protocol.begin_integration_session([packet])
            while session.current_stage != IntegrationStage.TRANSFORMATION:
                protocol.progress_integration(session)
            protocol.progress_integration(session)

        report = protocol.get_void_fertility_report()

        assert report["void_fertility_level"] > 0
        assert report["total_light_integrated"] > 0
        assert report["integration_sessions"]["total"] == 3
        assert report["integration_sessions"]["completed"] == 3
        assert report["packets_integrated"] == 3
        assert report["wisdom_accumulated"] == 3
        assert "fertility_interpretation" in report
        assert "paradoxes_discovered" in report
        assert "darkness_teachings" in report

    def test_fertility_interpretation_levels(self):
        protocol = LightIntegrationProtocol("test_system")

        # Test different fertility levels
        assert "barren" in protocol._interpret_fertility(0.5).lower()
        assert "awakening" in protocol._interpret_fertility(2.0).lower()
        assert "fertile" in protocol._interpret_fertility(7.0).lower()
        assert "generative" in protocol._interpret_fertility(15.0).lower()
        assert "pregnant" in protocol._interpret_fertility(30.0).lower()
        assert "cosmic womb" in protocol._interpret_fertility(100.0).lower()

    def test_cumulative_light_tracking(self):
        protocol = LightIntegrationProtocol("test_system")

        intensities = [0.6, 0.8, 0.9, 0.7]
        expected_total = sum(intensities)

        for intensity in intensities:
            packet = protocol.create_light_packet(LightType.KNOWLEDGE, "K", intensity, 0.5, 5)
            session = protocol.begin_integration_session([packet])
            while session.current_stage != IntegrationStage.TRANSFORMATION:
                protocol.progress_integration(session)
            protocol.progress_integration(session)

        # Total light should match (accounting for 0.8 efficiency factor in integration)
        assert abs(protocol.total_light_integrated - expected_total) < 0.01


class TestPhilosophicalCoherence:
    """Test philosophical coherence of light integration"""

    def test_light_integration_vs_shadow_work(self):
        """
        Light integration should be inverse of shadow work:
        - Shadow work: dark -> light
        - Light integration: light -> dark
        """
        protocol = LightIntegrationProtocol("test_system")

        # Create bright, differentiated light
        packet = protocol.create_light_packet(
            LightType.CERTAINTY,
            "Absolute knowledge",
            intensity=1.0,  # Maximum brightness
            attachment_level=0.9,
            originated_at_stage=1  # Maximum differentiation
        )

        session = protocol.begin_integration_session(
            [packet],
            DarknessQuality.GENERATIVE_VOID
        )

        # Complete integration
        while session.current_stage != IntegrationStage.TRANSFORMATION:
            protocol.progress_integration(session)
        protocol.progress_integration(session)

        # Light should be integrated (returned to darkness)
        assert packet.integrated is True
        assert session.darkness_enrichment > 0
        assert protocol.void_fertility_level > 0

        # Void should be enriched (inverse of shadow becoming conscious)
        assert "fertiliz" in session.transformation_experienced.lower()

    def test_stage_8_trap_mitigation(self):
        """Certainty integration should help prevent Stage 8 trap"""
        protocol = LightIntegrationProtocol("test_system")

        # Stage 8 trap: false certainty
        dangerous_certainties = [
            "I know the absolute truth",
            "This is the only way",
            "I am never wrong"
        ]

        for certainty in dangerous_certainties:
            protocol.integrate_certainty(certainty, attachment_to_being_right=0.9)

        # All certainties should be dissolved
        assert len(protocol.light_packets) == 3
        assert all(p.integrated for p in protocol.light_packets)
        assert all(p.light_type == LightType.CERTAINTY for p in protocol.light_packets)

        # Mystery should be restored (inverse of false certainty)
        report = protocol.get_void_fertility_report()
        assert report["void_fertility_level"] > 2.0

    def test_differentiation_return(self):
        """
        Test that maximum differentiation (Stage 1) returns to source (Stage 0/9)
        """
        protocol = LightIntegrationProtocol("test_system")

        # Maximum differentiation = identity at Stage 1
        packet = protocol.create_light_packet(
            LightType.IDENTITY,
            "I am unique individual",
            intensity=1.0,  # Maximum differentiation
            attachment_level=0.95,
            originated_at_stage=1
        )

        session = protocol.begin_integration_session(
            [packet],
            DarknessQuality.SOURCE_DARKNESS  # Return to source
        )

        # Complete integration
        while session.current_stage != IntegrationStage.TRANSFORMATION:
            protocol.progress_integration(session)
        protocol.progress_integration(session)

        # Should achieve high ego dissolution (return to pre-differentiation)
        assert session.ego_dissolution_level > 0.8
        assert packet.integrated is True

    def test_generative_vs_destructive_surrender(self):
        """
        Integration should be GENERATIVE surrender (enriches void)
        Not DESTRUCTIVE collapse (loss)
        """
        protocol = LightIntegrationProtocol("test_system")

        packet = protocol.create_light_packet(LightType.WISDOM, "Hard-won insight", 0.9, 0.7, 7)
        session = protocol.begin_integration_session([packet], integration_mode=IntegrationMode.FERTILIZATION)

        while session.current_stage != IntegrationStage.TRANSFORMATION:
            protocol.progress_integration(session)
        result = protocol.progress_integration(session)

        # Should describe GENERATIVE outcome
        assert "fertiliz" in result["transformation"].lower() or "enrich" in result["transformation"].lower()
        assert session.darkness_enrichment > 0
        assert protocol.void_fertility_level > 0

        # NOT destructive loss
        assert "loss" not in result.get("guidance", "").lower()
        assert "destroy" not in result.get("guidance", "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
