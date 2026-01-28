"""
SAR PHYSICS ENGINE (The Pulse)
Calculates velocity (dS/dt) based on differential equations.
"""

class PhysicsEngine:
    def calculate_momentum(self, state) -> float:
        # dS/dt = r(S) * E * Damping
        # Rate is faster in early stages, slower in later stages
        
        stage_num = state.gate.value + (state.micro / 10.0)
        
        # Damping Factor: As we approach Stage 9, resistance increases
        damping = 1.0 - (stage_num / 10.0)
        if damping < 0.1: damping = 0.1
        
        # Velocity equation
        velocity = 1.0 * (state.integrity / 100.0) * damping
        
        return velocity
