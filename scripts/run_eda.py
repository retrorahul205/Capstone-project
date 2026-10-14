#!/usr/bin/env python
"""Part 1: ingest data, run EDA, and write visualizations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cslib import convert_to_ts, fetch_data, fetch_ts

DATA_DIR = ROOT / "cs-train"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "reports"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


def _style():
    plt.rcParams.update(
        {
            "figure.figsize": (11, 5),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_daily_revenue(ts_all: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(ts_all["date"], ts_all["revenue"], color="#1f4e79", linewidth=1)
    ax.set_title("AAVAIL daily revenue (all countries)")
    ax.set_ylabel("Revenue")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "daily_revenue_all.png", dpi=140)
    plt.close()


def plot_rolling_revenue(ts_all: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(ts_all["date"], ts_all["revenue"].rolling(7).mean(), label="7-day", color="#2e7d32")
    ax.plot(ts_all["date"], ts_all["revenue"].rolling(30).mean(), label="30-day", color="#c62828")
    ax.set_title("Smoothed daily revenue")
    ax.set_ylabel("Revenue")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rolling_revenue.png", dpi=140)
    plt.close()


def plot_top_countries(raw: pd.DataFrame) -> pd.DataFrame:
    table = (
        raw.groupby("country")["price"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .rename("revenue")
        .reset_index()
    )
    fig, ax = plt.subplots()
    ax.barh(table["country"][::-1], table["revenue"][::-1], color="#1565c0")
    ax.set_title("Top 10 countries by total revenue")
    ax.set_xlabel("Revenue")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_countries.png", dpi=140)
    plt.close()
    return table


def plot_monthly_revenue(ts_all: pd.DataFrame) -> pd.DataFrame:
    monthly = ts_all.groupby("year_month")["revenue"].sum().reset_index()
    fig, ax = plt.subplots()
    ax.plot(monthly["year_month"], monthly["revenue"], marker="o")
    ax.set_title("Monthly revenue")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_revenue.png", dpi=140)
    plt.close()
    return monthly


def plot_target_relationship(ts_all: pd.DataFrame) -> None:
    from cslib import engineer_features

    X, y, dates = engineer_features(ts_all, training=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(X["previous_28"], y, alpha=0.35, s=12)
    axes[0].set_xlabel("Previous 28-day revenue")
    axes[0].set_ylabel("Next 30-day revenue")
    axes[0].set_title("Target vs recent revenue")
    axes[1].plot(pd.to_datetime(dates), y, color="#6a1b9a")
    axes[1].set_title("Next-30-day revenue over time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_relationship.png", dpi=140)
    plt.close()
    corr = X.assign(target=y).corr(numeric_only=True)["target"].drop("target").sort_values(ascending=False)
    return corr


def main() -> None:
    _style()
    print("Loading invoice JSON...")
    raw = fetch_data(str(DATA_DIR))
    print(f"rows={len(raw):,} countries={raw['country'].nunique()} "
          f"range={raw['invoice_date'].min().date()} to {raw['invoice_date'].max().date()}")

    ts = fetch_ts(str(DATA_DIR), clean=True)
    ts_all = ts["all"]
    plot_daily_revenue(ts_all)
    plot_rolling_revenue(ts_all)
    top = plot_top_countries(raw)
    monthly = plot_monthly_revenue(ts_all)
    corr = plot_target_relationship(ts_all)

    summary = {
        "n_invoices_rows": int(len(raw)),
        "n_countries": int(raw["country"].nunique()),
        "date_min": str(raw["invoice_date"].min().date()),
        "date_max": str(raw["invoice_date"].max().date()),
        "total_revenue": float(raw["price"].sum()),
        "negative_price_rows": int((raw["price"] < 0).sum()),
        "missing_customer_id": int(raw["customer_id"].isna().sum()),
        "top_countries": top.to_dict(orient="records"),
        "uk_share": float(
            raw.loc[raw["country"] == "United Kingdom", "price"].sum() / raw["price"].sum()
        ),
        "feature_target_corr": {k: float(v) for k, v in corr.items()},
        "mean_daily_revenue": float(ts_all["revenue"].mean()),
        "days": int(len(ts_all)),
    }
    (OUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote figures and reports/eda_summary.json")
    print(json.dumps({k: summary[k] for k in ("n_invoices_rows", "total_revenue", "uk_share")}, indent=2))


if __name__ == "__main__":
    main()
