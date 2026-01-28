"""
YUNUS PROTOCOL (The Heart)
Epistemic Humility & Safety Layer.
"""
from nam_logic import Gate

class YunusProtocol:
    def __init__(self):
        # Words that indicate "False Certainty" (Stage 8 Trap)
        self.arrogance_markers = [
            "absolute", "undeniable", "100%", "impossible to fail", 
            "i know everything", "no doubt", "guaranteed"
        ]
    
    def check_humility(self, thought: str, state) -> bool:
        # Only activate strict checking at Stage 8 (Unity/Trap)
        if state.gate == Gate.UNITY_TRAP:
            # Scan for arrogance markers
            if any(marker in thought.lower() for marker in self.arrogance_markers):
                print(f"🐳 [YUNUS ALERT]: Arrogance detected in Stage 8.")
                print(f"   Violation: '{thought}'")
                print(f"   Action: Swallowing thought. Entering Quarantine.")
                return False
        return True
