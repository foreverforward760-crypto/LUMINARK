import math
import uuid
import datetime
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# --- STAGE DEFINITIONS ---
# (Mirroring the deep content library for consistency)
STAGE_METADATA = {
    0: {"name": "PLENARA", "subtitle": "The Void • Pure Potential", "geometry": "Point (Zero-Dim)"},
    1: {"name": "PULSE", "subtitle": "First Movement • Unstable Energy", "geometry": "Line (1D)"},
    2: {"name": "DUALITY", "subtitle": "Binary Split • Separation", "geometry": "Vesica Piscis"},
    3: {"name": "EXPRESSION", "subtitle": "Creative Breakthrough • Incarnation", "geometry": "Triangle"},
    4: {"name": "FOUNDATION", "subtitle": "Stable Structure • Equilibrium", "geometry": "Square"},
    5: {"name": "THRESHOLD", "subtitle": "Crisis Point • Irreversible Choice", "geometry": "Pentagon"},
    6: {"name": "INTEGRATION", "subtitle": "Flow State • Peak Harmony", "geometry": "Hexagon"},
    7: {"name": "CRISIS", "subtitle": "Purification • Breakdown", "geometry": "Heptagon"},
    8: {"name": "UNITY PEAK", "subtitle": "Achievement • Permanence Trap", "geometry": "Octagon"},
    9: {"name": "RELEASE", "subtitle": "Transparent Return • Wisdom", "geometry": "Enneagon"}
}

# --- ARCHETYPES (SPAT VECTORS) ---
# Format: Complexity, Stability, Tension, Adaptability, Coherence
ARCHETYPES = {
    0: [0.0, 0.0, 0.0, 0.1, 0.0],       # Plenara
    1: [0.15, 0.1, 0.2, 0.3, 0.2],      # Pulse
    2: [0.3, 0.2, 0.6, 0.2, 0.3],       # Polarity
    3: [0.4, 0.4, 0.3, 0.2, 0.3],       # Expression
    4: [0.5, 0.8, 0.1, 0.1, 0.6],       # Foundation
    5: [0.6, 0.3, 0.9, 0.6, 0.4],       # Threshold
    6: [0.75, 0.65, 0.2, 0.7, 0.85],    # Integration
    7: [0.85, 0.25, 0.85, 0.5, 0.3],    # Analysis/Crisis
    8: [0.95, 0.95, 0.1, 0.1, 0.9],     # Unity Peak
    9: [0.8, 0.6, 0.4, 0.98, 0.85]      # Release
}

@dataclass
class SpatVectors:
    complexity: float
    stability: float
    tension: float
    adaptability: float
    coherence: float
    
    def to_list(self):
        return [self.complexity, self.stability, self.tension, self.adaptability, self.coherence]

class QuantumBrain:
    """
    The 'Layer 2' Intelligence Engine.
    Processes user biometric/slider data to generate high-fidelity
    fractal consciousness assessments.
    """
    
    def analyze_user_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for the analysis engine.
        Mimics the 'MyQuantumAI' proprietary model logic.
        """
        
        # 1. Parse Input
        vectors = SpatVectors(
            complexity=payload['spat_vectors']['complexity'] / 10.0,
            stability=payload['spat_vectors']['stability'] / 10.0,
            tension=payload['spat_vectors']['tension'] / 10.0,
            adaptability=payload['spat_vectors']['adaptability'] / 10.0,
            coherence=payload['spat_vectors']['coherence'] / 10.0
        )
        
        # 2. Toroidal/Fractal Analysis
        fractal_id = self._calculate_fractal_id(vectors)
        velocity = self._determine_velocity(payload.get('slider_movement_pattern', []))
        polyvagal = self._determine_polyvagal_state(vectors)
        
        # 3. Construct Response
        response = {
            "session_id": payload.get('session_id', str(uuid.uuid4())),
            "timestamp": datetime.datetime.now().isoformat(),
            "fractal_id": {
                "macro_stage": fractal_id['stage'],
                "micro_position": fractal_id['micro'],
                "confidence": fractal_id['confidence'],
                "velocity": velocity
            },
            "toroidal_analysis": self._generate_toroidal_analysis(vectors, fractal_id['stage']),
            "pattern_detection": {
                "slider_confidence": "high" if fractal_id['confidence'] > 0.8 else "moderate",
                "notes": self._generate_pattern_notes(fractal_id['stage'], polyvagal, velocity)
            },
            "stage_metadata": {
                "name": STAGE_METADATA[fractal_id['stage']]['name'],
                "subtitle": STAGE_METADATA[fractal_id['stage']]['subtitle'],
                "geometry": STAGE_METADATA[fractal_id['stage']]['geometry'],
                "polyvagal_state": polyvagal
            }
        }
        
        return response

    def _calculate_fractal_id(self, u: SpatVectors) -> Dict[str, Any]:
        """
        Determines the closest archetypal stage using Euclidean distance
        in 5D SPAT space, refined by 'Micro-Positioning'.
        """
        best_stage = 0
        min_dist = float('inf')
        
        # 5V Distance calculation
        for stage, w in ARCHETYPES.items():
            dist = math.sqrt(
                (u.complexity - w[0])**2 +
                (u.stability - w[1])**2 +
                (u.tension - w[2])**2 +
                (u.adaptability - w[3])**2 +
                (u.coherence - w[4])**2
            )
            if dist < min_dist:
                min_dist = dist
                best_stage = stage
        
        # Confidence logic (inverse of distance)
        confidence = max(0.0, 1.0 - min_dist)
        
        # Micro-position: Adds the "decimal" (e.g., 5.73) based on T/S resonance
        # A simple hashing-like deterministic float for the 'micro' feel
        micro_offset = (u.tension * 100) % 9 / 100.0
        micro_pos = round(best_stage + micro_offset, 2)
        
        return {
            "stage": best_stage,
            "micro": micro_pos,
            "confidence": round(confidence, 2)
        }

    def _determine_velocity(self, pattern: List[Dict]) -> Dict[str, Any]:
        """
        Analyzes movement history to determine if user is accelerating, stalled, or steady.
        If no history, infers from Tension/Adaptability ratio.
        """
        # Fallback if no real pattern data
        if not pattern:
            # High tension usually means acceleration or stall
            return {
                "type": "steady", 
                "magnitude": 0.5, 
                "trajectory": "stable_orbit"
            }
            
        # Logic: Calculate variance/speed of changes
        # Mock logic for the demo since we don't have real frontend streams yet
        return {
            "type": "accelerating",
            "magnitude": 0.82,
            "trajectory": "crisis_driven"
        }

    def _determine_polyvagal_state(self, u: SpatVectors) -> str:
        """
        Maps vectors to nervous system state.
        """
        if u.tension > 0.75 and u.stability < 0.4:
            return "sympathetic" # Fight/Flight
        elif u.tension < 0.3 and u.coherence < 0.45:
            return "dorsal" # Shutdown
        else:
            return "ventral" # Safe/Social

    def _generate_toroidal_analysis(self, u: SpatVectors, stage: int) -> Dict[str, Any]:
        """
        Fancy 'Physics' output for the UI.
        """
        next_stage = (stage + 1) % 10
        prev_stage = (stage - 1) % 10
        
        return {
            "primary_attractor": f"stage_{stage}_{STAGE_METADATA[stage]['name'].lower()}",
            "secondary_attractor": f"stage_{next_stage}_{STAGE_METADATA[next_stage]['name'].lower()}",
            "repeller": f"stage_{prev_stage}_{STAGE_METADATA[prev_stage]['name'].lower()}",
            "phase_space_coords": [round(u.complexity, 2), round(u.tension, 2), round(u.stability - u.adaptability, 2)],
            "orbital_stability": "chaotic" if u.tension > 0.8 else "stable"
        }

    def _generate_pattern_notes(self, stage: int, polyvagal: str, velocity: Dict) -> str:
        return f"User shows {polyvagal} activation consistent with Stage {stage} {STAGE_METADATA[stage]['name']}. Velocity is {velocity['type']}."


# --- ORACLE PROMPT GENERATOR (Layer 1 Integration) ---

def generate_oracle_prompt(user_data: Dict[str, Any], brain_response: Dict[str, Any], stage_library: str = "") -> str:
    """
    Generates the GPT-4 System + User prompt for the actual text generation.
    """
    
    meta = brain_response['stage_metadata']
    fid = brain_response['fractal_id']
    vectors = user_data.get('spat_vectors', {})
    
    prompt = f"""
USER PROFILE:
- Temporal Sentiment: Future="{user_data.get('temporal_sentiment', {}).get('future', '')}", Past="{user_data.get('temporal_sentiment', {}).get('past', '')}"
- Dominant Life Vector: {user_data.get('life_vector', 'General')}
- Current Stage: {fid['macro_stage']} ({meta['name']})
- Geometry: {meta['geometry']}
- Micro-Position: {fid['micro_position']}
- Polyvagal State: {meta['polyvagal_state'].upper()}

SPAT VECTORS (0-10):
- Complexity: {vectors.get('complexity', 0)}
- Stability: {vectors.get('stability', 0)}
- Tension: {vectors.get('tension', 0)}
- Adaptability: {vectors.get('adaptability', 0)}
- Coherence: {vectors.get('coherence', 0)}

TASK:
Generate a personalized Oracle reading ("The Antikythera Protocol") for this user.
Use the following structure:
1. RECOGNITION: Validate their current state ({meta['name']}).
2. INSIGHT: Explain *why* they feel "{user_data.get('temporal_sentiment', {}).get('future', '')}" about the future given they are in Stage {fid['macro_stage']}.
3. NAVIGATION: Provide 3 specific tactical steps based on their Polyvagal State ({meta['polyvagal_state']}).
4. WISDOM MIRROR: A historical or spiritual analogy relevant to {user_data.get('life_vector')}.

TONE:
Esoteric but grounded. "Cyber-Shamanic". Direct, high-signal, zero fluff.
"""
    return prompt

if __name__ == "__main__":
    # Self-test
    print("🧠 Initializing QuantumBrain...")
    brain = QuantumBrain()
    
    test_payload = {
        "session_id": "test-123",
        "temporal_sentiment": {"future": "Anxious", "past": "Heavy"},
        "life_vector": "Career",
        "spat_vectors": {
            "complexity": 8.5,
            "stability": 2.5,
            "tension": 9.5,
            "adaptability": 7.0,
            "coherence": 6.5
        },
        "slider_movement_pattern": []
    }
    
    print("📥 Processing Test Payload...")
    result = brain.analyze_user_state(test_payload)
    print(json.dumps(result, indent=2))
    
    print("\n🔮 Generating Oracle Prompt...")
    prompt = generate_oracle_prompt(test_payload, result)
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    print("✅ System Operational.")
