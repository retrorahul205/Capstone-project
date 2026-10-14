#!/usr/bin/env python
"""Data ingestion and feature engineering for the AAVAIL capstone."""

from __future__ import annotations

import os
import re
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

CORRECT_COLUMNS = [
    "country",
    "customer_id",
    "day",
    "invoice",
    "month",
    "price",
    "stream_id",
    "times_viewed",
    "year",
]

COLUMN_ALIASES = {
    "StreamID": "stream_id",
    "TimesViewed": "times_viewed",
    "total_price": "price",
}


def fetch_data(data_dir: str) -> pd.DataFrame:
    """Load all JSON invoice files from a directory into a single DataFrame."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"specified data dir does not exist: {data_dir}")

    file_list = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".json")
    ]
    if not file_list:
        raise FileNotFoundError(f"specified data dir does not contain JSON files: {data_dir}")

    frames = []
    for file_name in file_list:
        try:
            df = pd.read_json(file_name)
        except ValueError as exc:
            raise ValueError(f"could not parse JSON file: {file_name}") from exc

        df = df.rename(columns=COLUMN_ALIASES)
        missing = set(CORRECT_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"columns in {os.path.basename(file_name)} could not be matched: missing {missing}"
            )
        frames.append(df[CORRECT_COLUMNS])

    df = pd.concat(frames, ignore_index=True, sort=False)

    years = df["year"].astype(str).str.zfill(4)
    months = df["month"].astype(str).str.zfill(2)
    days = df["day"].astype(str).str.zfill(2)
    df["invoice_date"] = pd.to_datetime(years + "-" + months + "-" + days, errors="coerce")
    if df["invoice_date"].isna().any():
        bad = int(df["invoice_date"].isna().sum())
        raise ValueError(f"{bad} rows have invalid year/month/day values")

    df["invoice"] = df["invoice"].astype(str).str.replace(r"\D+", "", regex=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce").fillna(0)

    df.sort_values(by="invoice_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def convert_to_ts(df_orig: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Aggregate invoice-level data to a complete daily time series."""
    if country:
        if country not in df_orig["country"].unique():
            raise ValueError(f"country not found: {country}")
        df = df_orig.loc[df_orig["country"] == country]
    else:
        df = df_orig

    start = df["invoice_date"].min()
    end = df["invoice_date"].max()
    full_idx = pd.date_range(start, end, freq="D")

    grouped = df.groupby("invoice_date").agg(
        purchases=("invoice", "size"),
        unique_invoices=("invoice", "nunique"),
        unique_streams=("stream_id", "nunique"),
        total_views=("times_viewed", "sum"),
        revenue=("price", "sum"),
    )
    grouped = grouped.reindex(full_idx, fill_value=0)
    grouped.index.name = "date"
    out = grouped.reset_index()
    out["year_month"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m")
    return out


def country_slug(name: str) -> str:
    return re.sub(r"\s+", "_", str(name).strip().lower())


def fetch_ts(data_dir: str, clean: bool = False) -> dict[str, pd.DataFrame]:
    """Return daily time series for all countries plus the top 10 by revenue."""
    ts_data_dir = os.path.join(data_dir, "ts-data")

    if clean and os.path.exists(ts_data_dir):
        shutil.rmtree(ts_data_dir)
    os.makedirs(ts_data_dir, exist_ok=True)

    existing = [f for f in os.listdir(ts_data_dir) if f.endswith(".csv")]
    if existing:
        return {
            re.sub(r"\.csv$", "", cf)[3:]: pd.read_csv(
                os.path.join(ts_data_dir, cf), parse_dates=["date"]
            )
            for cf in existing
        }

    df = fetch_data(data_dir)
    table = df.pivot_table(index="country", values="price", aggfunc="sum")
    table.columns = ["total_revenue"]
    table.sort_values(by="total_revenue", inplace=True, ascending=False)
    top_ten_countries = list(table.index[:10])

    dfs: dict[str, pd.DataFrame] = {"all": convert_to_ts(df)}
    for country in top_ten_countries:
        dfs[country_slug(country)] = convert_to_ts(df, country=country)

    for key, item in dfs.items():
        item.to_csv(os.path.join(ts_data_dir, f"ts-{key}.csv"), index=False)
    return dfs


def _range_sum(prefix: np.ndarray, dates: np.ndarray, start_delta: int, end_delta: int) -> np.ndarray:
    """Inclusive-exclusive window sums using a prefix array and searchsorted."""
    n = dates.size
    starts = dates + np.timedelta64(start_delta, "D")
    ends = dates + np.timedelta64(end_delta, "D")
    start_idx = np.clip(np.searchsorted(dates, starts), 0, n)
    end_idx = np.clip(np.searchsorted(dates, ends), 0, n)
    return prefix[end_idx] - prefix[start_idx]


def engineer_features(df: pd.DataFrame, training: bool = True):
    """
    Build features for next-30-day revenue.

    For each day the target is revenue over [day, day+30).
    Features are trailing revenue windows, a prior-year window, and recent activity.
    """
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work.sort_values("date", inplace=True)
    work.reset_index(drop=True, inplace=True)

    dates = work["date"].values.astype("datetime64[D]")
    revenue = work["revenue"].to_numpy(dtype=float)
    invoices = work["unique_invoices"].to_numpy(dtype=float)
    views = work["total_views"].to_numpy(dtype=float)
    n = dates.size

    rev_cs = np.concatenate([[0.0], np.cumsum(revenue)])
    inv_cs = np.concatenate([[0.0], np.cumsum(invoices)])
    view_cs = np.concatenate([[0.0], np.cumsum(views)])

    features = defaultdict(list)
    for num in (7, 14, 28, 70):
        features[f"previous_{num}"] = _range_sum(rev_cs, dates, -num, 0)

    features["previous_year"] = _range_sum(rev_cs, dates, -365, -335)
    recent_inv = _range_sum(inv_cs, dates, -30, 0)
    recent_views = _range_sum(view_cs, dates, -30, 0)
    start_idx = np.searchsorted(dates, dates + np.timedelta64(-30, "D"))
    window_len = np.clip(np.arange(n) - start_idx, 1, None)
    features["recent_invoices"] = recent_inv / window_len
    features["recent_views"] = recent_views / window_len

    y = _range_sum(rev_cs, dates, 0, 30)
    X = pd.DataFrame(features)
    X.fillna(0, inplace=True)

    mask = X.sum(axis=1) > 0
    X = X.loc[mask].reset_index(drop=True)
    y = y[mask.to_numpy()]
    dates = dates[mask.to_numpy()]

    if training and X.shape[0] > 30:
        keep = np.arange(X.shape[0]) < (X.shape[0] - 30)
        X = X.loc[keep].reset_index(drop=True)
        y = y[keep]
        dates = dates[keep]

    return X, y, dates
