import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
LUMINARK_PKG_PATH = os.path.join(ROOT, "LUMINARK_AI")
sys.path.insert(0, LUMINARK_PKG_PATH)

from physics_engine import PhysicsEngine
from nam_logic import ConsciousnessState, Gate


class TestPhysicsEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PhysicsEngine()

    def test_velocity_full_integrity(self):
        state = ConsciousnessState(gate=Gate.VOID, micro=0, integrity=100.0, tension=0.0, description="")
        v = self.engine.calculate_momentum(state)
        self.assertAlmostEqual(v, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
