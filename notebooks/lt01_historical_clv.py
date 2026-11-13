# %% [markdown]
# # LT01 — Historical customer value (RFM, tiers, Pareto)
#
# Question: what did each customer contribute in the past, and how well does that history rank the value they produce next?
#
# Data: Online Retail II (UCI), Kaggle slug `mashlyn/online-retail-ii-uci`, read in place.
# Observation window: first 12 months. Holdout window: the next 12 months. Cancellations and returns are removed.

# %%
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from historical_clv.data import load_online_retail, make_windows, split_windows
from historical_clv.metrics import holdout_revenue, mae, rmse, top_decile_capture
from historical_clv.plots import plot_pareto_curve, plot_tier_summary, plot_value_distribution
from historical_clv.rfm import TIER_LABELS, assign_value_tiers, build_rfm, pareto_share

DATA_PATH = Path(os.environ.get("CLV_DATA_PATH", "/kaggle/input/online-retail-ii-uci/online_retail_II.csv"))
FIG = Path("../reports/figures")
REPORTS = Path("../reports")
FIG.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# %% [markdown]
# ## 1. Load, clean, cut

# %%
tx = load_online_retail(DATA_PATH)
print(f"clean rows: {len(tx):,} | customers: {tx['customer_id'].nunique():,} | dates {tx['day'].min().date()} to {tx['day'].max().date()}")

# %%
windows = make_windows(tx["day"].min(), t_obs_months=12, t_hold_months=12)
obs, hold = split_windows(tx, windows)
title_ctx = f"Online Retail II | obs {windows.obs_start.date()} to {windows.obs_end.date()} | hold to {windows.hold_end.date()} | LT01 historical"
print(f"observation rows: {len(obs):,} | holdout rows: {len(hold):,}")
print(f"obs customers: {obs['customer_id'].nunique():,} | holdout customers: {hold['customer_id'].nunique():,}")

# %% [markdown]
# ## 2. RFM summary and value tiers (observation window only)

# %%
rfm = build_rfm(obs, windows.obs_end)
rfm["tier"] = assign_value_tiers(rfm)
rfm["holdout_revenue"] = holdout_revenue(hold, rfm.index)
rfm["holdout_buyer"] = rfm["holdout_revenue"] > 0
rfm.head()

# %%
total_obs_revenue = rfm["historical_revenue"].sum()
tier_table = (
    rfm.groupby("tier")
    .agg(
        customers=("historical_revenue", "size"),
        revenue=("historical_revenue", "sum"),
        mean_historical_revenue=("historical_revenue", "mean"),
        repeat_rate=("frequency", lambda s: float((s > 0).mean())),
        holdout_buyer_rate=("holdout_buyer", "mean"),
    )
    .reindex(TIER_LABELS)
)
tier_table["revenue_share"] = tier_table["revenue"] / total_obs_revenue
tier_table.reset_index(names="tier")

# %% [markdown]
# ## 3. Pareto share

# %%
pareto_top20 = pareto_share(rfm["historical_revenue"], top_fraction=0.2)
pareto_top10 = pareto_share(rfm["historical_revenue"], top_fraction=0.1)
print(f"top 20% of customers hold {pareto_top20:.1%} of observation revenue")
print(f"top 10% of customers hold {pareto_top10:.1%} of observation revenue")

# %% [markdown]
# ## 4. Does history predict the holdout? Baselines compared
#
# Predictions use observation-window data only. Mean-value baseline: a constant equal to the mean historical revenue of the cohort.
# Historical model: the customer's own observation-window revenue.

# %%
y = rfm["holdout_revenue"].to_numpy()
pred_hist = rfm["historical_revenue"].to_numpy()
pred_mean = np.full_like(y, rfm["historical_revenue"].mean(), dtype=float)

metrics = pd.DataFrame(
    {
        "model": ["mean-value baseline", "historical revenue (LT01)"],
        "MAE": [mae(y, pred_mean), mae(y, pred_hist)],
        "RMSE": [rmse(y, pred_mean), rmse(y, pred_hist)],
        "top10_capture": [top_decile_capture(y, pred_mean), top_decile_capture(y, pred_hist)],
    }
)
metrics

# %% [markdown]
# Calibration of expected purchases does not apply here, since LT01 has no purchase model. It is reported in LT02.

# %%
fig_title = f"Historical revenue vs holdout | {title_ctx}"
plot_value_distribution(rfm, f"Observation revenue per customer | {title_ctx}", FIG / "lt01_value_distribution.png")
plot_pareto_curve(rfm["historical_revenue"].to_numpy(), f"Pareto curve | {title_ctx}", FIG / "lt01_pareto_curve.png")
tier_plot = tier_table.reset_index(names="tier")
plot_tier_summary(tier_plot, f"Revenue share by value tier | {title_ctx}", FIG / "lt01_tier_revenue_share.png")

# %%
plt.figure(figsize=(7, 5))
plt.scatter(rfm["historical_revenue"], rfm["holdout_revenue"], s=6, alpha=0.3)
plt.xscale("symlog")
plt.yscale("symlog")
plt.xlabel("observation revenue (symlog)")
plt.ylabel("holdout revenue (symlog)")
plt.title(fig_title)
plt.tight_layout()
plt.savefig(FIG / "lt01_history_vs_holdout.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Save tables

# %%
tier_table.to_csv(REPORTS / "lt01_tier_table.csv")
metrics.to_csv(REPORTS / "lt01_metrics.csv", index=False)
summary = {
    "dataset": "mashlyn/online-retail-ii-uci",
    "obs_window": [str(windows.obs_start.date()), str(windows.obs_end.date())],
    "holdout_window": [str(windows.obs_end.date()), str(windows.hold_end.date())],
    "customers_observed": int(len(rfm)),
    "holdout_buyer_rate": float(rfm["holdout_buyer"].mean()),
    "pareto_top20_revenue_share": pareto_top20,
    "pareto_top10_revenue_share": pareto_top10,
}
(REPORTS / "lt01_summary.json").write_text(json.dumps(summary, indent=2))
summary
