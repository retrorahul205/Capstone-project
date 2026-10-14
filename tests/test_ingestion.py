import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cslib import convert_to_ts, engineer_features, fetch_data


class IngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fetch_data(str(ROOT / "cs-train"))

    def test_required_columns(self):
        for col in ("country", "invoice", "price", "invoice_date", "stream_id"):
            self.assertIn(col, self.df.columns)

    def test_invoice_ids_are_digits(self):
        sample = self.df["invoice"].head(500).astype(str)
        self.assertTrue(sample.str.fullmatch(r"\d*").all())

    def test_daily_series_has_no_date_gaps(self):
        ts = convert_to_ts(self.df)
        span = (ts["date"].max() - ts["date"].min()).days + 1
        self.assertEqual(len(ts), span)

    def test_feature_matrix_shapes(self):
        ts = convert_to_ts(self.df)
        X, y, dates = engineer_features(ts, training=True)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(dates))
        self.assertGreater(len(X), 100)


if __name__ == "__main__":
    unittest.main()
