import os
import sys
import unittest

# Ensure imports in `LUMINARK_AI/main_model.py` (which use plain imports
# like `from nam_logic import ...`) can find modules by adding the
# `LUMINARK_AI` folder to sys.path.
ROOT = os.path.dirname(os.path.dirname(__file__))
LUMINARK_PKG_PATH = os.path.join(ROOT, "LUMINARK_AI")
sys.path.insert(0, LUMINARK_PKG_PATH)

from main_model import LuminarkAI


class SmokeTest(unittest.TestCase):
    def test_process_cycle_runs(self):
        ai = LuminarkAI()
        # Basic run to ensure no exceptions
        ai.process_cycle({'complexity': 10, 'stability': 90}, "smoke")


if __name__ == "__main__":
    unittest.main()
