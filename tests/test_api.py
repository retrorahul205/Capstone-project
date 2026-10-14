import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
from logger import update_predict_log, update_train_log, read_logs


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertIn("endpoints", body)

    def test_predict_missing_params(self):
        res = self.client.get("/predict?country=all")
        self.assertEqual(res.status_code, 400)

    def test_predict_ok(self):
        fake = {
            "country": "all",
            "target_date": "2018-01-05",
            "y_pred": [123.0],
            "y_proba": None,
        }
        with patch("app.model_predict", return_value=fake):
            res = self.client.post(
                "/predict",
                json={"country": "all", "year": "2018", "month": "01", "day": "05"},
            )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["y_pred"], [123.0])

    def test_logs_endpoint(self):
        update_predict_log("all", [1.0], None, "2018-01-05", "000:00:01", 0.1, test=True)
        res = self.client.get("/logs?kind=predict&test=true")
        self.assertEqual(res.status_code, 200)
        self.assertIn("all", res.get_json()["log"])

    def test_train_log_write(self):
        update_train_log(
            "all",
            ("2018-01-01", "2018-06-01"),
            {"rmse": 1},
            "000:00:01",
            0.1,
            "note",
            test=True,
        )
        content = read_logs("train", test=True)
        self.assertIn("all", content)


if __name__ == "__main__":
    unittest.main()
