import pandas as pd
import pytest

from historical_clv.data import clean_transactions, make_windows, split_windows


def _raw(rows):
    return pd.DataFrame(
        rows,
        columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"],
    )


def test_clean_drops_cancellations_returns_and_missing_customers():
    raw = _raw(
        [
            ["100", "A", "x", 2, "2010-01-05 10:00", 3.0, 1.0, "UK"],
            ["C101", "A", "x", -2, "2010-01-06 10:00", 3.0, 1.0, "UK"],
            ["102", "A", "x", -1, "2010-01-07 10:00", 3.0, 1.0, "UK"],
            ["103", "A", "x", 1, "2010-01-08 10:00", 0.0, 1.0, "UK"],
            ["104", "A", "x", 1, "2010-01-09 10:00", 4.0, None, "UK"],
        ]
    )
    out = clean_transactions(raw)
    assert list(out["order_id"]) == ["100"]
    assert out["revenue"].iloc[0] == pytest.approx(6.0)
    assert out["customer_id"].iloc[0] == "1"


def test_windows_are_month_aligned_and_half_open():
    windows = make_windows(pd.Timestamp("2009-12-09 12:50"), 12, 12)
    assert windows.obs_start == pd.Timestamp("2009-12-01")
    assert windows.obs_end == pd.Timestamp("2010-12-01")
    assert windows.hold_end == pd.Timestamp("2011-12-01")


def test_split_puts_boundary_days_in_the_correct_window():
    df = pd.DataFrame(
        {
            "customer_id": ["a", "a", "a", "a"],
            "day": pd.to_datetime(["2010-11-30", "2010-12-01", "2011-11-30", "2011-12-01"]),
            "revenue": [1.0, 2.0, 4.0, 8.0],
        }
    )
    windows = make_windows(pd.Timestamp("2010-01-01"), 11, 12)
    assert windows.obs_end == pd.Timestamp("2010-12-01")
    obs, hold = split_windows(df, windows)
    assert list(obs["revenue"]) == [1.0]
    assert list(hold["revenue"]) == [2.0, 4.0]
