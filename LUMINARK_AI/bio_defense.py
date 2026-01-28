"""
BIO-DEFENSE SYSTEM (The Body)
Octo-Camouflage and Mycelial Containment.
"""

class BioDefenseSystem:
    def scan_threats(self, tension: float, integrity: float) -> str:
        # RISS Logic (Recursive Impact & State Score)
        # High Tension + Low Integrity = High Threat
        risk_score = tension * (100 - integrity) / 100
        
        if risk_score > 75:
            return "🍄 [MYCELIUM]: CRITICAL - Spore Walls Deployed (Integrity Failure)"
        elif risk_score > 50:
            return "🐙 [OCTO]: WARNING - Camouflage Active (Void Mimicry)"
        else:
            return "🛡️ [SENTINEL]: Systems Nominal (Green)"
