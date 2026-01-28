import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
LUMINARK_PKG_PATH = os.path.join(ROOT, "LUMINARK_AI")
sys.path.insert(0, LUMINARK_PKG_PATH)

from bio_defense import BioDefenseSystem


class TestBioDefense(unittest.TestCase):
    def setUp(self):
        self.bd = BioDefenseSystem()

    def test_critical_threat(self):
        # tension 100, integrity 10 -> risk_score = 100*(90)/100 = 90
        status = self.bd.scan_threats(100, 10)
        self.assertIn("MYCELIUM", status)

    def test_nominal(self):
        status = self.bd.scan_threats(10, 90)
        self.assertIn("SENTINEL", status)


if __name__ == "__main__":
    unittest.main()
