"""Plots for the historical CLV analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_value_distribution(rfm: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(np.log10(rfm["historical_revenue"].clip(lower=0.01)), bins=60, ax=ax)
    ax.set_xlabel("log10 historical revenue per customer (observation window)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pareto_curve(values: np.ndarray, title: str, path: Path) -> None:
    arr = np.sort(np.asarray(values, dtype=float))[::-1]
    cum_customers = np.arange(1, arr.size + 1) / arr.size
    cum_value = np.cumsum(arr) / arr.sum()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cum_customers, cum_value, label="cumulative revenue share")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="equal split")
    ax.set_xlabel("share of customers (highest revenue first)")
    ax.set_ylabel("share of revenue")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tier_summary(tier_table: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=tier_table, x="tier", y="revenue_share", ax=ax)
    ax.set_ylabel("share of observation revenue")
    ax.set_xlabel("value tier")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
