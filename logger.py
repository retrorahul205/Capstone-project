#!/usr/bin/env python
"""CSV train/predict logging for the AAVAIL API."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"


def _ensure_log(path: Path, header: list[str]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(header)


def _append(path: Path, header: list[str], row: list) -> None:
    _ensure_log(path, header)
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def update_train_log(
    tag,
    date_range,
    metric,
    runtime,
    model_version,
    model_version_note,
    test: bool = False,
) -> None:
    filename = "train-test-log.csv" if test else "train-log.csv"
    path = LOG_DIR / filename
    header = [
        "timestamp",
        "country",
        "date_start",
        "date_end",
        "eval_metric",
        "runtime",
        "model_version",
        "model_version_note",
    ]
    start, end = date_range if isinstance(date_range, (list, tuple)) else (date_range, date_range)
    _append(
        path,
        header,
        [
            _stamp(),
            tag,
            start,
            end,
            metric,
            runtime,
            model_version,
            model_version_note,
        ],
    )


def update_predict_log(
    country,
    y_pred,
    y_proba,
    target_date,
    runtime,
    model_version,
    test: bool = False,
) -> None:
    filename = "predict-test-log.csv" if test else "predict-log.csv"
    path = LOG_DIR / filename
    header = [
        "timestamp",
        "country",
        "y_pred",
        "y_proba",
        "target_date",
        "runtime",
        "model_version",
    ]
    pred_value = y_pred[0] if hasattr(y_pred, "__len__") else y_pred
    _append(
        path,
        header,
        [_stamp(), country, pred_value, y_proba, target_date, runtime, model_version],
    )


def read_logs(kind: str = "predict", test: bool = False) -> str:
    """Return log file contents. kind is 'predict' or 'train'."""
    if kind not in {"predict", "train"}:
        raise ValueError("kind must be 'predict' or 'train'")
    prefix = f"{kind}-test-log.csv" if test else f"{kind}-log.csv"
    path = LOG_DIR / prefix
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")
