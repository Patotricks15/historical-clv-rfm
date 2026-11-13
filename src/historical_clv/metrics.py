"""Error and ranking metrics for holdout customer value."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def holdout_revenue(hold: pd.DataFrame, customers: pd.Index) -> pd.Series:
    """Revenue per customer in the holdout window; customers without purchases get 0."""
    totals = hold.groupby("customer_id")["revenue"].sum()
    return totals.reindex(customers, fill_value=0.0).astype(float)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def top_decile_capture(y_true: np.ndarray, y_pred: np.ndarray, fraction: float = 0.1) -> float:
    """Share of realized value held by the customers ranked highest by prediction."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total = y_true.sum()
    if total == 0:
        return float("nan")
    k = max(1, math.ceil(y_true.size * fraction))
    order = np.argsort(-y_pred, kind="stable")[:k]
    return float(y_true[order].sum() / total)
