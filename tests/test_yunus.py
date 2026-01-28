import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
LUMINARK_PKG_PATH = os.path.join(ROOT, "LUMINARK_AI")
sys.path.insert(0, LUMINARK_PKG_PATH)

from yunus_safety import YunusProtocol
from nam_logic import ConsciousnessState, Gate


class TestYunusProtocol(unittest.TestCase):
    def setUp(self):
        self.protocol = YunusProtocol()

    def test_detects_arrogance_in_stage8(self):
        state = ConsciousnessState(gate=Gate.UNITY_TRAP, micro=0, integrity=50, tension=50, description="")
        bad_thought = "This is absolutely guaranteed"
        self.assertFalse(self.protocol.check_humility(bad_thought, state))

    def test_allows_normal_thoughts(self):
        state = ConsciousnessState(gate=Gate.FOUNDATION, micro=0, integrity=80, tension=20, description="")
        thought = "Processing uncertain inputs"
        self.assertTrue(self.protocol.check_humility(thought, state))


if __name__ == "__main__":
    unittest.main()
