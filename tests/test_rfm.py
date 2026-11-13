import numpy as np
import pandas as pd
import pytest

from historical_clv.metrics import holdout_revenue, mae, rmse, top_decile_capture
from historical_clv.rfm import assign_value_tiers, build_rfm, pareto_share


def _obs():
    return pd.DataFrame(
        {
            "customer_id": ["a", "a", "a", "b"],
            "day": pd.to_datetime(["2010-01-01", "2010-01-01", "2010-03-01", "2010-02-10"]),
            "revenue": [5.0, 5.0, 20.0, 7.0],
        }
    )


def test_rfm_sums_per_customer_day_and_measures_from_window_end():
    rfm = build_rfm(_obs(), pd.Timestamp("2010-04-01"))
    a = rfm.loc["a"]
    assert a["purchase_days"] == 2
    assert a["historical_revenue"] == pytest.approx(30.0)
    assert a["frequency"] == 1
    assert a["recency_days"] == 31
    assert a["T_days"] == 90
    assert a["monetary_avg"] == pytest.approx(15.0)
    b = rfm.loc["b"]
    assert b["frequency"] == 0
    assert b["recency_days"] == (pd.Timestamp("2010-04-01") - pd.Timestamp("2010-02-10")).days


def test_value_tiers_are_equal_count_and_ordered():
    rfm = pd.DataFrame({"historical_revenue": np.arange(1, 11, dtype=float)})
    tiers = assign_value_tiers(rfm, n_tiers=5)
    assert tiers.value_counts().to_dict() == {t: 2 for t in tiers.unique()}
    assert tiers.iloc[0] == "Q1 lowest"
    assert tiers.iloc[-1] == "Q5 highest"


def test_pareto_share_top_fraction():
    assert pareto_share([1, 1, 1, 1, 96], top_fraction=0.2) == pytest.approx(0.96)


def test_holdout_revenue_fills_non_buyers_with_zero():
    hold = pd.DataFrame({"customer_id": ["a", "a"], "revenue": [3.0, 4.0]})
    y = holdout_revenue(hold, pd.Index(["a", "b"]))
    assert list(y) == [7.0, 0.0]


def test_error_metrics_and_top_decile_capture():
    y_true = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 90.0])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert top_decile_capture(y_true, y_pred) == pytest.approx(1.0)
    assert mae(np.array([1.0, 3.0]), np.array([2.0, 2.0])) == pytest.approx(1.0)
    assert rmse(np.array([0.0, 4.0]), np.array([0.0, 0.0])) == pytest.approx(np.sqrt(8.0))
