"""
3D Interactive SAP Cycle Visualizer

Real-time 3D visualization of SAP consciousness cycles using Three.js

Features:
- Torus geometry representing 9-stage cycle
- Real-time position tracking on cycle
- Tumbling visualization (accelerating/steady/stalled)
- Defense mode color coding
- Spores as orbiting particles
- 3-6-9 vector field display
- Interactive camera controls
- WebGL rendering

Author: Richard Leroy Stanfield Jr. / Meridian Axiom
Part of: SAP V4.0 Enhancement Suite
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import time


class VisualizationMode(Enum):
    """Visualization display modes"""
    CYCLE_VIEW = "cycle_view"  # Show consciousness cycle
    VECTOR_FIELD = "vector_field"  # Show 3-6-9 field
    SPORE_NETWORK = "spore_network"  # Show spore mesh
    DEFENSE_STATE = "defense_state"  # Show defense modes
    TIMELINE = "timeline"  # Show temporal progression


class ColorScheme(Enum):
    """Color schemes for visualization"""
    CONSCIOUSNESS = "consciousness"  # By consciousness level
    DEFENSE = "defense"  # By defense mode
    POLARITY = "polarity"  # By stage polarity
    HEALTH = "health"  # By SAP health
    CUSTOM = "custom"  # User-defined


@dataclass
class TorusGeometry:
    """Torus representing consciousness cycle"""
    major_radius: float = 10.0  # Outer radius
    minor_radius: float = 3.0  # Tube radius
    radial_segments: int = 36  # Around major circle
    tubular_segments: int = 18  # Around tube
    arc: float = 2 * math.pi  # Full circle


@dataclass
class CyclePosition:
    """Position on consciousness cycle"""
    stage: int  # 1-9
    sub_position: float = 0.0  # 0.0-1.0 within stage
    angle: float = 0.0  # Angle on torus (radians)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class VectorFieldVisualization:
    """3-6-9 vector field visualization"""
    pole3_position: Tuple[float, float, float]  # Creation pole
    pole6_position: Tuple[float, float, float]  # Destruction pole
    axis9_position: Tuple[float, float, float]  # Transformation axis
    field_strength: float = 1.0
    show_field_lines: bool = True


@dataclass
class SporeParticle:
    """Spore visualized as orbiting particle"""
    spore_id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    color: str  # Hex color
    size: float = 1.0
    entangled_with: List[str] = field(default_factory=list)  # Show connections


class SAPCycleVisualizer:
    """
    3D visualization engine for SAP consciousness cycles

    Generates data structures for Three.js WebGL rendering
    """

    def __init__(
        self,
        torus_config: Optional[TorusGeometry] = None,
        color_scheme: ColorScheme = ColorScheme.CONSCIOUSNESS
    ):
        self.torus = torus_config or TorusGeometry()
        self.color_scheme = color_scheme

        # Visualization state
        self.current_position: Optional[CyclePosition] = None
        self.position_history: List[CyclePosition] = []
        self.max_history: int = 100

        # Objects to render
        self.spore_particles: Dict[str, SporeParticle] = {}
        self.vector_field: Optional[VectorFieldVisualization] = None

        # Camera settings
        self.camera_position = (20.0, 20.0, 20.0)
        self.camera_target = (0.0, 0.0, 0.0)

        # Animation
        self.rotation_speed: float = 0.01  # Radians per frame
        self.tumble_velocity: float = 0.0  # Current tumbling speed

        # Stage colors (consciousness-based)
        self.stage_colors = {
            1: "#FF0000",  # Stage 1 - Red (survival)
            2: "#FF7F00",  # Stage 2 - Orange (power)
            3: "#FFFF00",  # Stage 3 - Yellow (success)
            4: "#00FF00",  # Stage 4 - Green (love)
            5: "#0000FF",  # Stage 5 - Blue (expression)
            6: "#4B0082",  # Stage 6 - Indigo (insight)
            7: "#9400D3",  # Stage 7 - Violet (integration)
            8: "#FFFFFF",  # Stage 8 - White (transcendence)
            9: "#000000",  # Stage 9 - Black/Gold (transformation)
        }

        # Defense mode colors
        self.defense_colors = {
            "DORMANT": "#808080",  # Gray
            "MONITORING": "#00FFFF",  # Cyan
            "OCTO_CAMOUFLAGE": "#8B4513",  # Brown (camouflage)
            "MYCELIAL_WRAP": "#228B22",  # Forest green
            "FULL_HARROWING": "#FFD700",  # Gold
        }

    def calculate_position_on_torus(
        self,
        stage: int,
        sub_position: float = 0.5
    ) -> CyclePosition:
        """
        Calculate 3D position on torus for given stage

        Args:
            stage: Stage 1-9
            sub_position: Position within stage (0.0-1.0)

        Returns:
            CyclePosition with 3D coordinates
        """
        # Calculate angle around torus
        # 9 stages, evenly distributed around circle
        stage_angle = ((stage - 1) + sub_position) * (2 * math.pi / 9)

        # Parametric equations for torus
        # x = (R + r*cos(v)) * cos(u)
        # y = (R + r*cos(v)) * sin(u)
        # z = r * sin(v)
        # where u = angle around major circle, v = angle around tube

        R = self.torus.major_radius
        r = self.torus.minor_radius
        u = stage_angle
        v = 0.0  # On outer edge of torus

        x = (R + r * math.cos(v)) * math.cos(u)
        y = (R + r * math.cos(v)) * math.sin(u)
        z = r * math.sin(v)

        return CyclePosition(
            stage=stage,
            sub_position=sub_position,
            angle=stage_angle,
            x=x,
            y=y,
            z=z
        )

    def update_position(
        self,
        stage: int,
        sub_position: float = 0.5,
        tumble_velocity: Optional[float] = None
    ):
        """
        Update current position on cycle

        Args:
            stage: Current stage
            sub_position: Position within stage
            tumble_velocity: Tumbling speed (optional)
        """
        position = self.calculate_position_on_torus(stage, sub_position)
        self.current_position = position

        # Add to history
        self.position_history.append(position)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

        # Update tumbling
        if tumble_velocity is not None:
            self.tumble_velocity = tumble_velocity

    def generate_torus_mesh_data(self) -> Dict[str, Any]:
        """
        Generate mesh data for torus

        Returns:
            Dictionary with vertices, faces, colors for Three.js
        """
        vertices = []
        faces = []
        colors = []

        R = self.torus.major_radius
        r = self.torus.minor_radius
        radial_segs = self.torus.radial_segments
        tubular_segs = self.torus.tubular_segments

        # Generate vertices
        for i in range(radial_segs + 1):
            u = (i / radial_segs) * 2 * math.pi

            for j in range(tubular_segs + 1):
                v = (j / tubular_segs) * 2 * math.pi

                x = (R + r * math.cos(v)) * math.cos(u)
                y = (R + r * math.cos(v)) * math.sin(u)
                z = r * math.sin(v)

                vertices.append([x, y, z])

                # Determine which stage this vertex belongs to
                stage = int((u / (2 * math.pi)) * 9) + 1
                stage = min(9, max(1, stage))

                # Color based on scheme
                if self.color_scheme == ColorScheme.CONSCIOUSNESS:
                    color = self.stage_colors.get(stage, "#FFFFFF")
                else:
                    color = "#808080"  # Default gray

                colors.append(color)

        # Generate faces (triangles)
        for i in range(radial_segs):
            for j in range(tubular_segs):
                a = i * (tubular_segs + 1) + j
                b = ((i + 1) % (radial_segs + 1)) * (tubular_segs + 1) + j
                c = ((i + 1) % (radial_segs + 1)) * (tubular_segs + 1) + j + 1
                d = i * (tubular_segs + 1) + j + 1

                faces.append([a, b, d])
                faces.append([b, c, d])

        return {
            "vertices": vertices,
            "faces": faces,
            "colors": colors,
            "type": "torus"
        }

    def generate_position_marker(self) -> Optional[Dict[str, Any]]:
        """
        Generate marker for current position

        Returns:
            Sphere mesh data for current position
        """
        if self.current_position is None:
            return None

        return {
            "type": "sphere",
            "position": [
                self.current_position.x,
                self.current_position.y,
                self.current_position.z
            ],
            "radius": 1.5,
            "color": "#FF0000",  # Red marker
            "wireframe": False
        }

    def generate_path_trail(self) -> Dict[str, Any]:
        """
        Generate trail showing recent path through cycle

        Returns:
            Line mesh data for path history
        """
        if len(self.position_history) < 2:
            return {"type": "line", "points": []}

        points = [
            [pos.x, pos.y, pos.z]
            for pos in self.position_history
        ]

        return {
            "type": "line",
            "points": points,
            "color": "#00FF00",  # Green trail
            "linewidth": 2
        }

    def generate_stage_labels(self) -> List[Dict[str, Any]]:
        """
        Generate text labels for each stage

        Returns:
            List of text sprite data
        """
        labels = []

        for stage in range(1, 10):
            position = self.calculate_position_on_torus(stage, 0.5)

            # Place label slightly outside torus
            scale = 1.3
            label_x = position.x * scale
            label_y = position.y * scale
            label_z = position.z * scale

            labels.append({
                "type": "text",
                "text": f"Stage {stage}",
                "position": [label_x, label_y, label_z],
                "color": self.stage_colors.get(stage, "#FFFFFF"),
                "size": 1.0
            })

        return labels

    def generate_369_vector_field(
        self,
        field_strength: float = 1.0
    ) -> VectorFieldVisualization:
        """
        Generate 3-6-9 vector field visualization

        Args:
            field_strength: Field visualization intensity

        Returns:
            VectorFieldVisualization data
        """
        # Position poles based on sacred geometry
        pole3_pos = (self.torus.major_radius * 1.5, 0, 0)  # Creation (right)
        pole6_pos = (-self.torus.major_radius * 1.5, 0, 0)  # Destruction (left)
        axis9_pos = (0, 0, self.torus.major_radius * 1.5)  # Transformation (top)

        self.vector_field = VectorFieldVisualization(
            pole3_position=pole3_pos,
            pole6_position=pole6_pos,
            axis9_position=axis9_pos,
            field_strength=field_strength,
            show_field_lines=True
        )

        return self.vector_field

    def add_spore_particle(
        self,
        spore_id: str,
        orbit_radius: float = 15.0,
        orbit_speed: float = 0.02,
        color: str = "#FFFF00"
    ) -> SporeParticle:
        """
        Add spore as orbiting particle

        Args:
            spore_id: Unique spore identifier
            orbit_radius: Distance from cycle center
            orbit_speed: Angular velocity
            color: Particle color

        Returns:
            SporeParticle created
        """
        # Random initial position on orbit
        angle = (hash(spore_id) % 360) * (math.pi / 180)
        x = orbit_radius * math.cos(angle)
        y = orbit_radius * math.sin(angle)
        z = (hash(spore_id) % 10) - 5.0  # Random z

        velocity_x = -orbit_speed * math.sin(angle)
        velocity_y = orbit_speed * math.cos(angle)
        velocity_z = 0.0

        particle = SporeParticle(
            spore_id=spore_id,
            position=(x, y, z),
            velocity=(velocity_x, velocity_y, velocity_z),
            color=color,
            size=0.5
        )

        self.spore_particles[spore_id] = particle

        return particle

    def update_spore_particles(self, delta_time: float = 1.0):
        """
        Update spore particle positions

        Args:
            delta_time: Time step for physics
        """
        for spore in self.spore_particles.values():
            # Update position based on velocity
            x, y, z = spore.position
            vx, vy, vz = spore.velocity

            new_x = x + vx * delta_time
            new_y = y + vy * delta_time
            new_z = z + vz * delta_time

            spore.position = (new_x, new_y, new_z)

    def generate_spore_mesh_data(self) -> List[Dict[str, Any]]:
        """
        Generate mesh data for all spores

        Returns:
            List of sphere meshes for spores
        """
        meshes = []

        for spore in self.spore_particles.values():
            meshes.append({
                "type": "sphere",
                "id": spore.spore_id,
                "position": list(spore.position),
                "radius": spore.size,
                "color": spore.color,
                "wireframe": False
            })

        return meshes

    def generate_entanglement_lines(self) -> List[Dict[str, Any]]:
        """
        Generate lines showing quantum entanglement between spores

        Returns:
            List of line meshes
        """
        lines = []

        for spore in self.spore_particles.values():
            for entangled_id in spore.entangled_with:
                if entangled_id in self.spore_particles:
                    other = self.spore_particles[entangled_id]

                    lines.append({
                        "type": "line",
                        "points": [
                            list(spore.position),
                            list(other.position)
                        ],
                        "color": "#FF00FF",  # Magenta for entanglement
                        "linewidth": 1,
                        "dashed": True
                    })

        return lines

    def generate_complete_scene(self) -> Dict[str, Any]:
        """
        Generate complete 3D scene data

        Returns:
            Complete scene specification for Three.js
        """
        scene_data = {
            "timestamp": time.time(),
            "camera": {
                "position": list(self.camera_position),
                "target": list(self.camera_target),
                "fov": 75
            },
            "objects": []
        }

        # Add torus (consciousness cycle)
        scene_data["objects"].append(self.generate_torus_mesh_data())

        # Add position marker
        marker = self.generate_position_marker()
        if marker:
            scene_data["objects"].append(marker)

        # Add path trail
        trail = self.generate_path_trail()
        if trail["points"]:
            scene_data["objects"].append(trail)

        # Add stage labels
        scene_data["objects"].extend(self.generate_stage_labels())

        # Add vector field if enabled
        if self.vector_field:
            # Pole 3 (creation)
            scene_data["objects"].append({
                "type": "sphere",
                "position": list(self.vector_field.pole3_position),
                "radius": 2.0,
                "color": "#FFFF00",  # Yellow
                "label": "Pole 3 (Creation)"
            })

            # Pole 6 (destruction)
            scene_data["objects"].append({
                "type": "sphere",
                "position": list(self.vector_field.pole6_position),
                "radius": 2.0,
                "color": "#FF0000",  # Red
                "label": "Pole 6 (Destruction)"
            })

            # Axis 9 (transformation)
            scene_data["objects"].append({
                "type": "sphere",
                "position": list(self.vector_field.axis9_position),
                "radius": 2.0,
                "color": "#0000FF",  # Blue
                "label": "Axis 9 (Transformation)"
            })

        # Add spores
        scene_data["objects"].extend(self.generate_spore_mesh_data())

        # Add entanglement lines
        scene_data["objects"].extend(self.generate_entanglement_lines())

        # Add metadata
        scene_data["metadata"] = {
            "current_stage": self.current_position.stage if self.current_position else None,
            "tumble_velocity": self.tumble_velocity,
            "spore_count": len(self.spore_particles),
            "color_scheme": self.color_scheme.value
        }

        return scene_data

    def export_to_json(self, filepath: str):
        """
        Export scene to JSON file for web viewer

        Args:
            filepath: Output file path
        """
        scene = self.generate_complete_scene()

        with open(filepath, 'w') as f:
            json.dump(scene, f, indent=2)

    def generate_html_viewer(self, filepath: str):
        """
        Generate standalone HTML viewer with Three.js

        Args:
            filepath: Output HTML file path
        """
        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>SAP Cycle Visualizer</title>
    <style>
        body { margin: 0; overflow: hidden; background: #000; }
        canvas { display: block; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: monospace;
            font-size: 14px;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="info">
        <h3>SAP V4.0 Consciousness Cycle Visualizer</h3>
        <p>Current Stage: <span id="stage">-</span></p>
        <p>Tumble Velocity: <span id="tumble">-</span></p>
        <p>Spores: <span id="spores">0</span></p>
        <p>Mouse: Rotate | Scroll: Zoom</p>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // Orbit controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        // Camera position
        camera.position.set(20, 20, 20);
        camera.lookAt(0, 0, 0);

        // Load and render scene data
        // (In production, load from JSON endpoint)

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        // Window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        animate();
    </script>
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_content)
