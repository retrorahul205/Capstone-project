#!/usr/bin/env python
"""Train, load, and predict 30-day revenue models for AAVAIL."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cslib import engineer_features, fetch_ts
from logger import update_predict_log, update_train_log

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "trailing 70-day run-rate model for 30-day revenue"

DEFAULT_DATA_DIR = PROJECT_ROOT / "cs-train"


def _runtime_since(time_start: float) -> str:
    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    return "%03d:%02d:%02d" % (h, m, s)


class TrailingRunRateRegressor(BaseEstimator, RegressorMixin):
    """Project next-30-day revenue from the trailing 70-day revenue run rate.

    Selected after time-series holdout comparison: tree models overfit the
    rolling target, while recent run-rate generalizes to later months.
    """

    window_col = "previous_70"
    window_days = 70
    horizon = 30

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(list(X.columns), dtype=object)
        return self

    def predict(self, X):
        return np.asarray(X[self.window_col], dtype=float) * (
            self.horizon / self.window_days
        )


def _build_estimator(test: bool = False):
    return TrailingRunRateRegressor()


def compare_candidates(X, y) -> pd.DataFrame:
    """Compare baseline and candidate models with time-series CV."""
    splitter = TimeSeriesSplit(n_splits=3)
    candidates = {
        "ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=80, max_depth=16, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=80, max_depth=3, random_state=42
                    ),
                ),
            ]
        ),
    }

    rows = []
    mean_rmse, mean_mae, persist_rmse, persist_mae = [], [], [], []
    for train_idx, test_idx in splitter.split(X):
        mean_pred = np.full(len(test_idx), np.mean(y[train_idx]))
        mean_rmse.append(np.sqrt(mean_squared_error(y[test_idx], mean_pred)))
        mean_mae.append(mean_absolute_error(y[test_idx], mean_pred))
        persist = X.iloc[test_idx]["previous_28"].to_numpy() * (30.0 / 28.0)
        persist_rmse.append(np.sqrt(mean_squared_error(y[test_idx], persist)))
        persist_mae.append(mean_absolute_error(y[test_idx], persist))
    rows.append({"model": "mean_baseline", "cv_rmse": float(np.mean(mean_rmse)), "cv_mae": float(np.mean(mean_mae))})
    rows.append(
        {
            "model": "persist_previous_28",
            "cv_rmse": float(np.mean(persist_rmse)),
            "cv_mae": float(np.mean(persist_mae)),
        }
    )
    run_rmse, run_mae = [], []
    for train_idx, test_idx in splitter.split(X):
        pred = X.iloc[test_idx]["previous_70"].to_numpy() * (30.0 / 70.0)
        run_rmse.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        run_mae.append(mean_absolute_error(y[test_idx], pred))
    rows.append(
        {
            "model": "trailing_runrate_70",
            "cv_rmse": float(np.mean(run_rmse)),
            "cv_mae": float(np.mean(run_mae)),
        }
    )
    for name, est in candidates.items():
        rmse = -cross_val_score(
            est, X, y, cv=splitter, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        mae = -cross_val_score(
            est, X, y, cv=splitter, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        rows.append(
            {
                "model": name,
                "cv_rmse": float(np.mean(rmse)),
                "cv_mae": float(np.mean(mae)),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_rmse")


def _model_train(df: pd.DataFrame, tag: str, test: bool = False) -> dict:
    time_start = time.time()
    X, y, dates = engineer_features(df, training=True)
    if X.empty:
        raise ValueError(f"no trainable rows for {tag}")

    if test:
        n_samples = max(40, int(np.round(0.35 * X.shape[0])))
        n_samples = min(n_samples, X.shape[0])
        subset_indices = np.linspace(0, X.shape[0] - 1, n_samples).astype(int)
        X = X.iloc[subset_indices].reset_index(drop=True)
        y = y[subset_indices]
        dates = dates[subset_indices]

    estimator = _build_estimator(test=test)
    split = max(1, int(0.8 * len(y)))
    X_train, X_hold = X.iloc[:split], X.iloc[split:]
    y_train, y_hold = y[:split], y[split:]
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_hold) if len(y_hold) else estimator.predict(X_train)
    y_true = y_hold if len(y_hold) else y_train
    eval_rmse = float(round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))
    eval_mae = float(round(mean_absolute_error(y_true, y_pred), 2))

    estimator.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    version_slug = re.sub(r"\.", "_", str(MODEL_VERSION))
    prefix = "test" if test else "sl"
    saved_model = MODEL_DIR / f"{prefix}-{tag}-{version_slug}.joblib"
    joblib.dump(estimator, saved_model)

    runtime = _runtime_since(time_start)
    update_train_log(
        tag,
        (str(dates[0]), str(dates[-1])),
        {"rmse": eval_rmse, "mae": eval_mae},
        runtime,
        MODEL_VERSION,
        MODEL_VERSION_NOTE,
        test=test,
    )
    print(f"... saved {saved_model} rmse={eval_rmse} mae={eval_mae}")
    return {"country": tag, "rmse": eval_rmse, "mae": eval_mae, "path": str(saved_model)}


def model_train(data_dir=None, test: bool = False) -> list[dict]:
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    ts_data = fetch_ts(str(data_dir))
    results = []
    for country, df in ts_data.items():
        if test and country not in {"all", "united_kingdom"}:
            continue
        results.append(_model_train(df, country, test=test))
    return results


def model_load(prefix: str = "sl", data_dir=None, training: bool = True):
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = [f for f in os.listdir(MODEL_DIR) if f.startswith(prefix) and f.endswith(".joblib")]
    if not models:
        raise FileNotFoundError(
            f"Models with prefix '{prefix}' cannot be found. Did you train?"
        )

    all_models = {}
    for model in models:
        country = model.split("-")[1]
        all_models[country] = joblib.load(MODEL_DIR / model)

    ts_data = fetch_ts(str(data_dir))
    all_data = {}
    for country, df in ts_data.items():
        X, y, dates = engineer_features(df, training=training)
        all_data[country] = {
            "X": X,
            "y": y,
            "dates": np.array([str(d)[:10] for d in dates]),
        }
    return all_data, all_models


def model_predict(
    country,
    year,
    month,
    day,
    all_models=None,
    all_data=None,
    test: bool = False,
    data_dir=None,
):
    time_start = time.time()
    country = country_key(country)

    if all_models is None or all_data is None:
        prefix = "test" if test else "sl"
        all_data, all_models = model_load(
            prefix=prefix, data_dir=data_dir, training=False
        )

    if country not in all_models:
        raise ValueError(f"model for country '{country}' could not be found")

    for value, label in ((year, "year"), (month, "month"), (day, "day")):
        if re.search(r"\D", str(value)):
            raise ValueError(f"invalid {label}: {value}")

    data = all_data[country]
    model = all_models[country]
    target_date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    if target_date not in data["dates"]:
        raise ValueError(
            f"date {target_date} not in range {data['dates'][0]}-{data['dates'][-1]}"
        )

    date_indx = np.where(data["dates"] == target_date)[0][0]
    query = data["X"].iloc[[date_indx]]
    y_pred = model.predict(query)

    runtime = _runtime_since(time_start)
    update_predict_log(
        country,
        y_pred,
        None,
        target_date,
        runtime,
        MODEL_VERSION,
        test=test,
    )
    return {
        "country": country,
        "target_date": target_date,
        "y_pred": [float(y_pred[0])],
        "y_proba": None,
    }


def country_key(name: str) -> str:
    return re.sub(r"\s+", "_", str(name).strip().lower())


if __name__ == "__main__":
    print("TRAINING MODELS")
    model_train(DEFAULT_DATA_DIR, test=True)
    print("LOADING MODELS")
    all_data, all_models = model_load(prefix="test")
    print("... models loaded:", ", ".join(all_models.keys()))
    result = model_predict("all", "2018", "01", "05", test=True)
    print(result)
