import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logger import read_logs, update_predict_log


class LoggerTests(unittest.TestCase):
    def test_read_unknown_kind(self):
        with self.assertRaises(ValueError):
            read_logs(kind="unknown")

    def test_predict_log_roundtrip(self):
        update_predict_log("france", [9.5], None, "2019-01-01", "000:00:01", 0.1, test=True)
        text = read_logs("predict", test=True)
        self.assertIn("france", text)


if __name__ == "__main__":
    unittest.main()
