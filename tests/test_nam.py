import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
LUMINARK_PKG_PATH = os.path.join(ROOT, "LUMINARK_AI")
sys.path.insert(0, LUMINARK_PKG_PATH)

from nam_logic import NAMLogicEngine, Gate


class TestNAMLogic(unittest.TestCase):
    def test_assess_state_threshold(self):
        engine = NAMLogicEngine()
        state = engine.assess_state(50, 50)
        self.assertEqual(state.gate, Gate.THRESHOLD)
        self.assertEqual(state.micro, 0)


if __name__ == "__main__":
    unittest.main()
