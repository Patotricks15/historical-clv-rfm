"""RFM summary, value tiers and Pareto share over the observation window."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TIER_LABELS = ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]


def build_rfm(obs: pd.DataFrame, obs_end: pd.Timestamp) -> pd.DataFrame:
    """One row per customer active in the observation window.

    Revenue is summed per customer per day first. ``frequency`` counts repeat
    purchase days (purchase days minus one), ``recency_days`` and ``T_days``
    are measured from the end of the observation window (exclusive).
    """
    daily = obs.groupby(["customer_id", "day"], as_index=False)["revenue"].sum()
    grouped = daily.groupby("customer_id")
    rfm = pd.DataFrame(
        {
            "purchase_days": grouped["day"].nunique(),
            "first_day": grouped["day"].min(),
            "last_day": grouped["day"].max(),
            "historical_revenue": grouped["revenue"].sum(),
        }
    )
    rfm["frequency"] = rfm["purchase_days"] - 1
    rfm["recency_days"] = (obs_end - rfm["last_day"]).dt.days
    rfm["T_days"] = (obs_end - rfm["first_day"]).dt.days
    rfm["monetary_avg"] = rfm["historical_revenue"] / rfm["purchase_days"]
    return rfm


def assign_value_tiers(rfm: pd.DataFrame, n_tiers: int = 5) -> pd.Series:
    """Equal-count tiers by historical revenue; ties are broken by customer order."""
    if n_tiers != len(TIER_LABELS):
        raise ValueError(f"expected {len(TIER_LABELS)} tiers, got {n_tiers}")
    ranks = rfm["historical_revenue"].rank(method="first")
    return pd.qcut(ranks, q=n_tiers, labels=TIER_LABELS).astype(str)


def pareto_share(values: pd.Series | np.ndarray, top_fraction: float = 0.2) -> float:
    """Share of the total held by the top ``top_fraction`` of the values."""
    arr = np.sort(np.asarray(values, dtype=float))[::-1]
    if arr.size == 0 or arr.sum() == 0:
        return float("nan")
    k = max(1, math.ceil(arr.size * top_fraction))
    return float(arr[:k].sum() / arr.sum())
