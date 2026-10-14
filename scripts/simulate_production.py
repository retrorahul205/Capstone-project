#!/usr/bin/env python
"""Part 3: query the trained model through production dates and score vs gold standard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cslib import convert_to_ts, engineer_features, fetch_data
from model import model_load

TRAIN_DIR = ROOT / "cs-train"
PROD_DIR = ROOT / "cs-production"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "reports"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    _, all_models = model_load(prefix="sl", data_dir=TRAIN_DIR, training=False)
    model = all_models["all"]

    train = fetch_data(str(TRAIN_DIR))
    prod = fetch_data(str(PROD_DIR))
    combined = pd.concat([train, prod], ignore_index=True)
    combined.sort_values("invoice_date", inplace=True)
    ts = convert_to_ts(combined)
    X, y, dates = engineer_features(ts, training=False)

    rows = []
    for i, day in enumerate(dates):
        target = str(day)[:10]
        if target < "2019-08-01":
            continue
        pred = float(model.predict(X.iloc[[i]])[0])
        rows.append({"date": target, "predicted": pred, "actual": float(y[i])})

    df = pd.DataFrame(rows)
    scored = df.dropna(subset=["predicted", "actual"])
    metrics = {}
    if not scored.empty:
        metrics = {
            "n_days": int(len(scored)),
            "rmse": float(np.sqrt(mean_squared_error(scored["actual"], scored["predicted"]))),
            "mae": float(mean_absolute_error(scored["actual"], scored["predicted"])),
            "mape": float(
                np.mean(
                    np.abs(scored["actual"] - scored["predicted"])
                    / np.clip(np.abs(scored["actual"]), 1e-6, None)
                )
                * 100
            ),
        }
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(pd.to_datetime(scored["date"]), scored["actual"], label="known 30-day revenue")
        ax.plot(pd.to_datetime(scored["date"]), scored["predicted"], label="predicted 30-day revenue")
        ax.set_title("Model vs gold-standard 30-day revenue")
        ax.set_ylabel("Revenue")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "production_vs_gold.png", dpi=140)
        plt.close()

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(
            pd.to_datetime(scored["date"]),
            scored["predicted"] - scored["actual"],
            color="#c62828",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Prediction residual (predicted - known)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "production_residuals.png", dpi=140)
        plt.close()

    df.to_csv(OUT_DIR / "production_predictions.csv", index=False)
    (OUT_DIR / "part3_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Wrote reports/production_predictions.csv")


if __name__ == "__main__":
    main()
