"""Loading, cleaning and observation/holdout cuts for Online Retail II."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Windows:
    """Observation window [obs_start, obs_end) and holdout window [obs_end, hold_end)."""

    obs_start: pd.Timestamp
    obs_end: pd.Timestamp
    hold_end: pd.Timestamp


def load_online_retail(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        encoding="latin1",
        dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
    )
    return clean_transactions(raw)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep positive sales with a customer ID; drop cancellations and returns."""
    df = raw.dropna(subset=["Customer ID"])
    is_cancellation = df["Invoice"].str.startswith("C")
    df = df.loc[~is_cancellation & (df["Quantity"] > 0) & (df["Price"] > 0)]
    out = pd.DataFrame(
        {
            "customer_id": df["Customer ID"].astype("int64").astype(str),
            "order_id": df["Invoice"],
            "timestamp": pd.to_datetime(df["InvoiceDate"]),
            "revenue": df["Quantity"] * df["Price"],
        }
    ).reset_index(drop=True)
    out["day"] = out["timestamp"].dt.normalize()
    return out


def make_windows(
    first_timestamp: pd.Timestamp, t_obs_months: int = 12, t_hold_months: int = 12
) -> Windows:
    obs_start = pd.Timestamp(first_timestamp).normalize().replace(day=1)
    obs_end = obs_start + pd.DateOffset(months=t_obs_months)
    hold_end = obs_end + pd.DateOffset(months=t_hold_months)
    return Windows(obs_start=obs_start, obs_end=obs_end, hold_end=hold_end)


def split_windows(df: pd.DataFrame, windows: Windows) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = df[(df["day"] >= windows.obs_start) & (df["day"] < windows.obs_end)]
    hold = df[(df["day"] >= windows.obs_end) & (df["day"] < windows.hold_end)]
    return obs, hold
