import unittest

from luminark_data_connectors import SECEdgarConnector


class SECTickerResolutionTests(unittest.TestCase):
    def test_resolves_ticker_to_zero_padded_cik(self):
        records = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
        }
        self.assertEqual(
            SECEdgarConnector.resolve_ticker_cik("aapl", records),
            "0000320193",
        )

    def test_returns_none_for_unknown_ticker(self):
        records = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        }
        self.assertIsNone(SECEdgarConnector.resolve_ticker_cik("UNKNOWN", records))

    def test_trims_whitespace(self):
        records = {
            "0": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
        }
        self.assertEqual(
            SECEdgarConnector.resolve_ticker_cik(" MSFT ", records),
            "0000789019",
        )


if __name__ == "__main__":
    unittest.main()
