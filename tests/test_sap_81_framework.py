import unittest

from luminark.sap.framework_81 import Gate81, SAP81Framework


class SAPGateMappingTests(unittest.TestCase):
    def setUp(self):
        self.framework = SAP81Framework()

    def test_stage_nine_is_not_the_lower_zero_boundary(self):
        state = self.framework.get_state(9.4)
        self.assertEqual(state.gate, Gate81.GATE_9)
        self.assertEqual(state.get_absolute_stage(), 9.4)
        self.assertEqual(state.gate.value, "9: TRANSPARENCY OF THE GUIDE")

    def test_lower_zero_boundary_is_separate_from_numbered_stage_mapping(self):
        self.assertEqual(Gate81.GATE_0B.value, "0ᴮ: lower Stage 0 cycle boundary")
        self.assertNotIn(Gate81.GATE_0B, self.framework.gates)
        self.assertEqual(len(self.framework.gates), 10)

    def test_numbered_mapping_includes_stage_zero_through_stage_nine(self):
        self.assertEqual(self.framework.get_state(0.0).gate, Gate81.GATE_0T)
        self.assertEqual(self.framework.get_state(8.9).gate, Gate81.GATE_8)
        self.assertEqual(self.framework.get_state(9.9).gate, Gate81.GATE_9)


if __name__ == "__main__":
    unittest.main()
