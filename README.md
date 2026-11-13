# LT01 — Historical customer value (RFM, value tiers, Pareto)

**Question:** What did each customer contribute in the past, and how well does that history rank the value each customer produces next?

## Data

- Dataset: Online Retail II (UCI), Kaggle slug `mashlyn/online-retail-ii-uci`, real transactions from a UK online retailer. Read date and local copy are recorded in [data/README.md](data/README.md).
- Cleaning: rows without a customer ID are dropped. Cancellations (invoices starting with `C`), returns (quantity ≤ 0), and zero-price rows are removed. Revenue = `Quantity × Price`, summed per customer per day.
- Known limit of the cleaning rule: returns are dropped, not netted against the original sale, so revenue is slightly overstated.

## Windows

| Window | Dates | Rule |
|---|---|---|
| Observation (`T_obs`) | 2009-12-01 to 2010-12-01 (12 months) | Features and history use only this window |
| Holdout (`T_hold`) | 2010-12-01 to 2011-12-01 (12 months) | Target = revenue per customer in this window |

Both windows are month-aligned and half-open. The population is the 4,266 customers with at least one purchase in the observation window. Customers who first buy in the holdout are excluded, so the results describe retention of existing customers, not acquisition.

## Method

- RFM summary per customer: `frequency` = purchase days − 1, `recency_days` and `T_days` measured from the end of the observation window, and `monetary_avg` = revenue per purchase day.
- Value tiers: five equal-count tiers by observation revenue (`Q1 lowest` to `Q5 highest`).
- Pareto share of revenue for the top 10% and top 20% of customers.
- Prediction of holdout revenue with the customer's own observation revenue (LT01 model), compared with a constant mean-value baseline.

## Results (test = holdout window)

| Model | MAE | RMSE | Top-10% capture |
|---|---|---|---|
| Mean-value baseline (constant) | 2336.16 | 8618.83 | 0.142 |
| Historical revenue (LT01) | 1165.80 | 4334.73 | 0.598 |

- Top 20% of customers hold 73.3% of observation revenue. Top 10% hold 59.6%.
- Repeat rate and holdout buying rate by tier:

| Tier | Customers | Revenue share | Repeat rate | Holdout buyer rate |
|---|---|---|---|---|
| Q1 lowest | 854 | 1.5% | 0.118 | 0.336 |
| Q2 | 853 | 3.6% | 0.407 | 0.494 |
| Q3 | 853 | 7.1% | 0.804 | 0.672 |
| Q4 | 853 | 14.5% | 0.930 | 0.774 |
| Q5 highest | 853 | 73.2% | 0.974 | 0.905 |

Calibration of expected purchases does not apply here, because LT01 has no purchase model. It is reported in LT02.

## Decision

Use historical revenue as the benchmark for every later model. A future CLV model is worth keeping only if it beats the historical benchmark on holdout MAE, RMSE, and top-decile capture.

## Limitation

Revenue is not margin, and the window is a single 12-month period in a UK B2B-like retailer. The ranking may not transfer to other markets or to subscription businesses.

## Reproduce

```bash
make install
make reproduce
```

The notebook reads `CLV_DATA_PATH` if set, otherwise the Kaggle path. Figures go to `reports/figures/`, and tables and metrics go to `reports/`.
