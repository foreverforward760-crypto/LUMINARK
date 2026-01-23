"""
Unit tests for SAP Cycle Visualizer
"""

import pytest
import json
import math
import os
from sap_yunus.sap_visualizer import (
    SAPCycleVisualizer,
    TorusGeometry,
    CyclePosition,
    SporeParticle,
    ColorScheme,
    VisualizationMode
)


def test_visualizer_initialization():
    """Test creating visualizer"""
    viz = SAPCycleVisualizer()

    assert viz.torus is not None
    assert viz.color_scheme == ColorScheme.CONSCIOUSNESS
    assert viz.current_position is None
    assert len(viz.spore_particles) == 0


def test_custom_torus_geometry():
    """Test creating visualizer with custom torus"""
    torus = TorusGeometry(
        major_radius=15.0,
        minor_radius=4.0,
        radial_segments=48,
        tubular_segments=24
    )

    viz = SAPCycleVisualizer(torus_config=torus)

    assert viz.torus.major_radius == 15.0
    assert viz.torus.minor_radius == 4.0


def test_calculate_position_on_torus():
    """Test calculating 3D position for stage"""
    viz = SAPCycleVisualizer()

    position = viz.calculate_position_on_torus(stage=1, sub_position=0.0)

    assert position.stage == 1
    assert position.sub_position == 0.0
    assert position.x != 0.0 or position.y != 0.0  # Should be on torus


def test_position_for_all_stages():
    """Test calculating positions for all 9 stages"""
    viz = SAPCycleVisualizer()

    positions = []
    for stage in range(1, 10):
        pos = viz.calculate_position_on_torus(stage)
        positions.append(pos)
        assert pos.stage == stage

    # All positions should be different
    coords = [(p.x, p.y, p.z) for p in positions]
    assert len(set(coords)) == 9


def test_update_position():
    """Test updating current position"""
    viz = SAPCycleVisualizer()

    viz.update_position(stage=5, sub_position=0.5, tumble_velocity=0.05)

    assert viz.current_position is not None
    assert viz.current_position.stage == 5
    assert viz.tumble_velocity == 0.05
    assert len(viz.position_history) == 1


def test_position_history_tracking():
    """Test that position history is tracked"""
    viz = SAPCycleVisualizer()

    # Update multiple times
    for stage in range(1, 6):
        viz.update_position(stage)

    assert len(viz.position_history) == 5


def test_position_history_limit():
    """Test that position history respects max limit"""
    viz = SAPCycleVisualizer()
    viz.max_history = 10

    # Add more than max
    for i in range(15):
        viz.update_position(stage=(i % 9) + 1)

    assert len(viz.position_history) == 10


def test_generate_torus_mesh_data():
    """Test generating torus mesh"""
    viz = SAPCycleVisualizer()

    mesh = viz.generate_torus_mesh_data()

    assert mesh["type"] == "torus"
    assert "vertices" in mesh
    assert "faces" in mesh
    assert "colors" in mesh
    assert len(mesh["vertices"]) > 0
    assert len(mesh["faces"]) > 0


def test_torus_mesh_has_9_stage_colors():
    """Test that torus mesh uses 9 stage colors"""
    viz = SAPCycleVisualizer(color_scheme=ColorScheme.CONSCIOUSNESS)

    mesh = viz.generate_torus_mesh_data()

    # Should have colors from stage_colors
    unique_colors = set(mesh["colors"])
    # Should have multiple colors for different stages
    assert len(unique_colors) > 1


def test_generate_position_marker():
    """Test generating current position marker"""
    viz = SAPCycleVisualizer()

    # No position yet
    marker = viz.generate_position_marker()
    assert marker is None

    # Set position
    viz.update_position(stage=3)
    marker = viz.generate_position_marker()

    assert marker is not None
    assert marker["type"] == "sphere"
    assert len(marker["position"]) == 3
    assert marker["radius"] > 0


def test_generate_path_trail():
    """Test generating path trail"""
    viz = SAPCycleVisualizer()

    # No history
    trail = viz.generate_path_trail()
    assert trail["type"] == "line"
    assert len(trail["points"]) == 0

    # Add history
    viz.update_position(stage=1)
    viz.update_position(stage=2)
    viz.update_position(stage=3)

    trail = viz.generate_path_trail()
    assert len(trail["points"]) == 3


def test_generate_stage_labels():
    """Test generating stage labels"""
    viz = SAPCycleVisualizer()

    labels = viz.generate_stage_labels()

    assert len(labels) == 9  # One for each stage
    for i, label in enumerate(labels):
        assert label["type"] == "text"
        assert f"Stage {i+1}" in label["text"]
        assert len(label["position"]) == 3


def test_generate_369_vector_field():
    """Test generating 3-6-9 vector field"""
    viz = SAPCycleVisualizer()

    field = viz.generate_369_vector_field(field_strength=1.5)

    assert field is not None
    assert len(field.pole3_position) == 3
    assert len(field.pole6_position) == 3
    assert len(field.axis9_position) == 3
    assert field.field_strength == 1.5
    assert viz.vector_field == field


def test_add_spore_particle():
    """Test adding spore particle"""
    viz = SAPCycleVisualizer()

    spore = viz.add_spore_particle(
        spore_id="spore1",
        orbit_radius=15.0,
        orbit_speed=0.02,
        color="#FF00FF"
    )

    assert spore.spore_id == "spore1"
    assert spore.color == "#FF00FF"
    assert len(spore.position) == 3
    assert len(spore.velocity) == 3
    assert "spore1" in viz.spore_particles


def test_update_spore_particles():
    """Test updating spore physics"""
    viz = SAPCycleVisualizer()

    spore = viz.add_spore_particle("spore1")
    initial_pos = spore.position

    # Update physics
    viz.update_spore_particles(delta_time=1.0)

    # Position should have changed
    assert spore.position != initial_pos


def test_generate_spore_mesh_data():
    """Test generating spore meshes"""
    viz = SAPCycleVisualizer()

    viz.add_spore_particle("spore1", color="#FF0000")
    viz.add_spore_particle("spore2", color="#00FF00")

    meshes = viz.generate_spore_mesh_data()

    assert len(meshes) == 2
    for mesh in meshes:
        assert mesh["type"] == "sphere"
        assert "id" in mesh
        assert "position" in mesh
        assert "color" in mesh


def test_entangle_spores():
    """Test creating entanglement between spores"""
    viz = SAPCycleVisualizer()

    spore1 = viz.add_spore_particle("spore1")
    spore2 = viz.add_spore_particle("spore2")

    # Create entanglement
    spore1.entangled_with.append("spore2")
    spore2.entangled_with.append("spore1")

    lines = viz.generate_entanglement_lines()

    # Should generate line between entangled spores
    assert len(lines) >= 1


def test_generate_entanglement_lines():
    """Test generating entanglement visualization"""
    viz = SAPCycleVisualizer()

    viz.add_spore_particle("s1")
    viz.add_spore_particle("s2")
    viz.add_spore_particle("s3")

    viz.spore_particles["s1"].entangled_with = ["s2", "s3"]
    viz.spore_particles["s2"].entangled_with = ["s1"]
    viz.spore_particles["s3"].entangled_with = ["s1"]

    lines = viz.generate_entanglement_lines()

    # Should have lines for each entanglement
    assert len(lines) > 0
    for line in lines:
        assert line["type"] == "line"
        assert len(line["points"]) == 2


def test_generate_complete_scene():
    """Test generating complete scene data"""
    viz = SAPCycleVisualizer()

    # Set up scene
    viz.update_position(stage=5, sub_position=0.5)
    viz.add_spore_particle("spore1")
    viz.generate_369_vector_field()

    scene = viz.generate_complete_scene()

    assert "timestamp" in scene
    assert "camera" in scene
    assert "objects" in scene
    assert "metadata" in scene
    assert len(scene["objects"]) > 0


def test_scene_includes_all_elements():
    """Test that complete scene includes all visualization elements"""
    viz = SAPCycleVisualizer()

    viz.update_position(stage=3)
    viz.add_spore_particle("s1")
    viz.add_spore_particle("s2")
    viz.generate_369_vector_field()

    scene = viz.generate_complete_scene()

    # Should have torus, marker, labels, spores, vector field
    object_types = [obj.get("type") for obj in scene["objects"]]

    assert "torus" in object_types
    assert "sphere" in object_types  # Marker and/or spores
    assert "text" in object_types  # Labels


def test_scene_metadata():
    """Test scene metadata"""
    viz = SAPCycleVisualizer()

    viz.update_position(stage=7, tumble_velocity=0.1)
    viz.add_spore_particle("s1")

    scene = viz.generate_complete_scene()

    metadata = scene["metadata"]

    assert metadata["current_stage"] == 7
    assert metadata["tumble_velocity"] == 0.1
    assert metadata["spore_count"] == 1
    assert metadata["color_scheme"] == ColorScheme.CONSCIOUSNESS.value


def test_export_to_json(tmp_path):
    """Test exporting scene to JSON"""
    viz = SAPCycleVisualizer()

    viz.update_position(stage=4)

    output_file = tmp_path / "scene.json"
    viz.export_to_json(str(output_file))

    assert output_file.exists()

    # Load and verify
    with open(output_file) as f:
        data = json.load(f)

    assert "objects" in data
    assert "camera" in data


def test_generate_html_viewer(tmp_path):
    """Test generating HTML viewer"""
    viz = SAPCycleVisualizer()

    output_file = tmp_path / "viewer.html"
    viz.generate_html_viewer(str(output_file))

    assert output_file.exists()

    # Verify HTML content
    with open(output_file) as f:
        html = f.read()

    assert "<!DOCTYPE html>" in html
    assert "three.js" in html.lower()
    assert "SAP" in html


def test_color_schemes():
    """Test different color schemes"""
    schemes = [
        ColorScheme.CONSCIOUSNESS,
        ColorScheme.DEFENSE,
        ColorScheme.POLARITY
    ]

    for scheme in schemes:
        viz = SAPCycleVisualizer(color_scheme=scheme)
        mesh = viz.generate_torus_mesh_data()
        assert len(mesh["colors"]) > 0


def test_camera_configuration():
    """Test camera settings"""
    viz = SAPCycleVisualizer()

    scene = viz.generate_complete_scene()

    camera = scene["camera"]
    assert "position" in camera
    assert "target" in camera
    assert "fov" in camera
    assert len(camera["position"]) == 3
    assert len(camera["target"]) == 3


def test_stage_angle_calculation():
    """Test that stage angles are evenly distributed"""
    viz = SAPCycleVisualizer()

    angles = []
    for stage in range(1, 10):
        pos = viz.calculate_position_on_torus(stage, sub_position=0.0)
        angles.append(pos.angle)

    # Check angular spacing
    # 9 stages should be spaced 40 degrees (2π/9 radians) apart
    expected_spacing = 2 * math.pi / 9

    for i in range(len(angles) - 1):
        spacing = angles[i + 1] - angles[i]
        assert abs(spacing - expected_spacing) < 0.01


def test_torus_parametric_equations():
    """Test that torus uses correct parametric equations"""
    viz = SAPCycleVisualizer()

    pos = viz.calculate_position_on_torus(stage=1, sub_position=0.0)

    # For a torus at stage 1, position 0:
    # angle u = 0
    # x = (R + r) * cos(0) = R + r
    # y = (R + r) * sin(0) = 0
    # z = 0

    R = viz.torus.major_radius
    r = viz.torus.minor_radius

    assert abs(pos.x - (R + r)) < 0.01
    assert abs(pos.y) < 0.01
    assert abs(pos.z) < 0.01


def test_multiple_spores_different_positions():
    """Test that multiple spores get different initial positions"""
    viz = SAPCycleVisualizer()

    spores = []
    for i in range(5):
        spore = viz.add_spore_particle(f"spore{i}")
        spores.append(spore)

    # All positions should be unique
    positions = [s.position for s in spores]
    assert len(set(positions)) == 5


def test_defense_mode_colors():
    """Test defense mode color mapping"""
    viz = SAPCycleVisualizer(color_scheme=ColorScheme.DEFENSE)

    assert "DORMANT" in viz.defense_colors
    assert "OCTO_CAMOUFLAGE" in viz.defense_colors
    assert "MYCELIAL_WRAP" in viz.defense_colors
    assert "FULL_HARROWING" in viz.defense_colors


def test_visualization_with_no_data():
    """Test that visualization handles empty state gracefully"""
    viz = SAPCycleVisualizer()

    # Should still generate valid scene
    scene = viz.generate_complete_scene()

    assert scene is not None
    assert "objects" in scene
    # Should have at least torus and labels
    assert len(scene["objects"]) > 0
