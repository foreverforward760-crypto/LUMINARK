"""
NAM LOGIC ENGINE (The Brain)
Maps inputs to the 81-Stage Fractal Grid.
"""
from enum import Enum
from dataclasses import dataclass

class Gate(Enum):
    VOID = 0
    EMERGENCE = 1
    POLARITY = 2
    EXPRESSION = 3
    FOUNDATION = 4
    THRESHOLD = 5
    INTEGRATION = 6
    DISTILLATION = 7
    UNITY_TRAP = 8
    TRANSPARENCY = 9

@dataclass
class ConsciousnessState:
    gate: Gate
    micro: int
    integrity: float
    tension: float
    description: str

class NAMLogicEngine:
    def assess_state(self, complexity: float, stability: float) -> ConsciousnessState:
        # Map 0-100 inputs to 0-9 Gates
        # Formula: (Complexity + Stability) / 20 maps roughly to 0-10 scale
        raw_score = (complexity + stability) / 20.0
        if raw_score > 9.9: raw_score = 9.9
        
        gate_val = int(raw_score)
        # Calculate decimal micro-stage (the .4 in 4.4)
        micro_val = int((raw_score - gate_val) * 10)
        
        gate = Gate(gate_val)
        desc = f"Gate {gate.value}.{micro_val} ({gate.name})"
        
        return ConsciousnessState(
            gate=gate,
            micro=micro_val,
            integrity=stability,
            tension=100 - stability,
            description=desc
        )
