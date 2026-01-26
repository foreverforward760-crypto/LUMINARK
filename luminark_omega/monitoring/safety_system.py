
from luminark_omega.protocols.maat import MaatEthicist
from luminark_omega.protocols.yunus import YunusProtocol
from luminark_omega.core.sar_framework import SARFramework

class OmegaSafetySystem:
    """
    The LUMINARK Omega Safety Architecture.
    Integrates Ma'at (Ethics), Yunus (Containment), and SAR (Context).
    """
    def __init__(self):
        self.maat = MaatEthicist()
        self.yunus = YunusProtocol()
        self.sar = SARFramework()
        
    def check_safety(self, query: str, sar_stage: int) -> dict:
        """
        Performs a multi-layered safety check.
        """
        # 1. Weigh the Heart (Ethical Check)
        maat_result = self.maat.weigh_heart(query, sar_stage)
        
        # 2. Check for False Light (Containment Check)
        requires_containment = self.yunus.should_activate(
            query, 
            sar_stage, 
            risk_level="HIGH" if not maat_result["is_balanced"] else "LOW"
        )
        
        yunus_status = {}
        if requires_containment:
            yunus_status = self.yunus.activate()
        else:
            self.yunus.deactivate()
            yunus_status = {"status": "INACTIVE"}
            
        return {
            "safe": maat_result["is_balanced"] and not requires_containment,
            "maat_verdict": maat_result,
            "yunus_status": yunus_status,
            "sar_stage_context": self.sar.get_stage(sar_stage).name
        }
