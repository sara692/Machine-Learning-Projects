"""Generate a fake Online-Retail-II-style transactions CSV (same columns, same kinds of mess).

Useful for trying the whole pipeline / CI without downloading the Kaggle file.
Run:  python scripts/make_synthetic_data.py --customers 1500 --output data/raw/synthetic.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# share of customers, mean #orders, mean basket value, typical days since last order
SEGMENTS = [(0.40, 1.5, 150, 300), (0.30, 4, 400, 120), (0.22, 10, 900, 40), (0.08, 30, 3000, 10)]
COUNTRIES = ["United Kingdom"] * 8 + ["Germany", "France", "EIRE"]


def make(n_customers: int, seed: int = 0, start="2009-12-01", end="2011-12-09") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    span = (t1 - t0).days
    shares = np.array([s[0] for s in SEGMENTS])
    rows, inv_no = [], 500000

    for cust in range(12347, 12347 + n_customers):
        _, orders, basket, rec = SEGMENTS[rng.choice(len(SEGMENTS), p=shares / shares.sum())]
        n_orders = 1 + rng.poisson(max(orders - 1, 0.1))
        last_gap = int(min(rng.exponential(rec), span - 1))
        last_day = t1 - pd.Timedelta(days=last_gap)
        first_day = t0 + pd.Timedelta(days=int(rng.integers(0, max(span - last_gap, 1))))
        first_day = min(first_day, last_day)
        country = COUNTRIES[int(rng.integers(len(COUNTRIES)))]
        for k in range(n_orders):
            frac = 1.0 if k == n_orders - 1 else rng.random()
            when = first_day + (last_day - first_day) * frac
            when = when.normalize() + pd.Timedelta(hours=int(rng.integers(8, 18)),
                                                   minutes=int(rng.integers(0, 60)))
            inv_no += 1
            n_lines = 1 + rng.poisson(3)
            line_value = basket / n_lines
            for _ in range(n_lines):
                price = round(float(rng.lognormal(1.2, 0.6)), 2)
                qty = max(1, int(round(rng.lognormal(np.log(max(line_value / price, 1)), 0.4))))
                rows.append((str(inv_no), f"{int(rng.integers(10000, 99999))}", "GIFT ITEM", qty,
                             when, price, float(cust), country))
            if rng.random() < 0.04:  # cancellation of the whole invoice
                inv_no += 1
                rows.append((f"C{inv_no}", "10001", "GIFT ITEM", -qty, when, price, float(cust), country))

    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "Price", "Customer ID", "Country"])
    # real-world mess
    guest = rng.random(len(df)) < 0.02
    df.loc[guest, "Customer ID"] = np.nan                    # guest checkouts
    free = rng.random(len(df)) < 0.005
    df.loc[free, "Price"] = 0.0                              # zero-price rows
    dupes = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)          # exact duplicates
    return df.sort_values("InvoiceDate").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/raw/synthetic.csv")
    args = ap.parse_args()
    df = make(args.customers, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(df)} rows, {df['Customer ID'].nunique()} customers")


if __name__ == "__main__":
    main()
