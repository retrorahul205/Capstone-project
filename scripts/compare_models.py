#!/usr/bin/env python
"""Part 2: compare modeling approaches and train the selected model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cslib import engineer_features, fetch_ts
from model import compare_candidates, model_train

DATA_DIR = ROOT / "cs-train"
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    ts = fetch_ts(str(DATA_DIR))
    comparisons = {}
    for tag in ("all", "united_kingdom"):
        X, y, _ = engineer_features(ts[tag], training=True)
        table = compare_candidates(X, y)
        table.to_csv(OUT_DIR / f"model_comparison_{tag}.csv", index=False)
        comparisons[tag] = table.to_dict(orient="records")
        print(f"\n{tag}")
        print(table.to_string(index=False))

    print("\nRetraining selected RandomForest models on all training data...")
    results = model_train(DATA_DIR, test=False)
    payload = {"comparisons": comparisons, "trained": results}
    (OUT_DIR / "part2_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote reports/part2_metrics.json")


if __name__ == "__main__":
    main()
