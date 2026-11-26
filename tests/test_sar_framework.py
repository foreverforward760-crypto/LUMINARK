import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from luminark_omega.core.sar_framework import SARFramework


class SAPStageFrameworkTests(unittest.TestCase):
    def setUp(self):
        self.framework = SARFramework()

    def test_stage_names_match_canonical_0_to_9_sequence(self):
        expected = [
            "PLENARA",
            "SPARK OF NAVIGATION",
            "FORGE OF POLARITY",
            "ENGINE OF EXPRESSION",
            "CRUCIBLE OF EQUILIBRIUM",
            "DYNAMO OF WILL",
            "NEXUS OF HARMONY",
            "LENS OF DISTILLATION",
            "VESSEL OF GROUNDING",
            "TRANSPARENCY OF THE GUIDE",
        ]
        self.assertEqual(
            [self.framework.get_stage(i).name for i in range(10)],
            expected,
        )

    def test_stage_zero_is_both_unstable(self):
        stage = self.framework.get_stage(0)
        self.assertEqual(stage.physical_state, "unstable")
        self.assertEqual(stage.conscious_state, "unstable")
        self.assertFalse(stage.is_inverted)

    def test_stage_nine_is_aligned_and_not_inverted(self):
        stage = self.framework.get_stage(9)
        self.assertEqual(stage.physical_state, "stable")
        self.assertEqual(stage.conscious_state, "stable")
        self.assertFalse(stage.is_inverted)

    def test_invalid_stage_falls_back_to_stage_four(self):
        self.assertEqual(self.framework.get_stage(99).level, 4)

    def test_transition_thresholds(self):
        self.assertEqual(self.framework.assess_transition(4, 0.91), 5)
        self.assertEqual(self.framework.assess_transition(4, 0.29), 3)
        self.assertEqual(self.framework.assess_transition(4, 0.5), 4)
        self.assertEqual(self.framework.assess_transition(9, 0.99), 9)
        self.assertEqual(self.framework.assess_transition(0, 0.01), 0)


if __name__ == "__main__":
    unittest.main()
