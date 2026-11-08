
import argparse
from pathlib import Path

import pandas as pd
import yaml


def build_rfm(df: pd.DataFrame, snapshot_date=None, start=None, end=None) -> pd.DataFrame:

    if start:
        df = df[df["invoice_date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["invoice_date"] < pd.Timestamp(end)]
    if df.empty:
        raise ValueError("No transactions in the selected window.")

    if snapshot_date in (None, "auto"):
        snapshot = df["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    else:
        snapshot = pd.Timestamp(snapshot_date)

    rfm = (df.groupby("customer_id")
             .agg(last_purchase=("invoice_date", "max"),
                  frequency=("invoice", "nunique"),
                  monetary=("revenue", "sum"))
             .reset_index())
    rfm["recency"] = (snapshot - rfm["last_purchase"].dt.normalize()).dt.days.clip(lower=0)
    rfm = rfm[rfm["monetary"] > 0]
    return rfm[["customer_id", "recency", "frequency", "monetary"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="configs/params.yaml")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--snapshot-date", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    p = yaml.safe_load(open(args.params))["features"]
    src = args.input or p["input"]
    dst = args.output or p["output"]
    snap = args.snapshot_date or p.get("snapshot_date", "auto")

    df = pd.read_parquet(src)
    rfm = build_rfm(df, snap, args.start, args.end)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(dst, index=False)
    print(f"Wrote {dst}: {len(rfm)} customers")
    print(rfm[["recency", "frequency", "monetary"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
