"""Clean raw Online Retail II transactions.

Input : data/raw/online_retail_II.csv   (Kaggle: mashlyn/online-retail-ii-uci)
Output: data/processed/clean.parquet

Run:  python src/data/prepare.py [--input PATH] [--output PATH] [--params configs/params.yaml]
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

COLUMN_ALIASES = {
    "invoice": "invoice",
    "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "price",
    "unitprice": "price",
    "customerid": "customer_id",
    "country": "country",
}
REQUIRED = ["invoice", "quantity", "invoice_date", "price", "customer_id"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to snake_case names, e.g. 'Customer ID' -> 'customer_id'."""
    mapping = {}
    for col in df.columns:
        key = re.sub(r"[\s_]+", "", str(col).lower())
        if key in COLUMN_ALIASES:
            mapping[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=mapping)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)}")
    return df


def clean(df: pd.DataFrame, min_quantity=1, min_price=0.01,
          drop_cancelled=True, drop_duplicates=True):
    """Apply cleaning rules. Returns (clean_df, stats_dict)."""
    df = normalize_columns(df)
    stats = {"rows_raw": len(df)}

    if drop_duplicates:
        df = df.drop_duplicates()
    stats["rows_after_dedup"] = len(df)

    df = df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df["invoice"] = df["invoice"].astype(str)

    df = df.dropna(subset=["invoice_date", "quantity", "price", "customer_id"])
    stats["rows_after_dropna"] = len(df)

    if drop_cancelled:
        df = df[~df["invoice"].str.upper().str.startswith("C")]
    stats["rows_after_cancelled"] = len(df)

    df = df[(df["quantity"] >= min_quantity) & (df["price"] >= min_price)]
    stats["rows_after_filters"] = len(df)

    df["customer_id"] = df["customer_id"].astype("int64")
    df["revenue"] = df["quantity"] * df["price"]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows left after cleaning - check the input file and filters.")
    stats["customers"] = int(df["customer_id"].nunique())
    stats["date_min"] = str(df["invoice_date"].min())
    stats["date_max"] = str(df["invoice_date"].max())
    return df, stats


def read_raw(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download it with:\n"
            "  kaggle datasets download -d mashlyn/online-retail-ii-uci -p data/raw --unzip")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, encoding="utf-8", encoding_errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="configs/params.yaml")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    p = yaml.safe_load(open(args.params))["prepare"]
    src = args.input or p["input"]
    dst = args.output or p["output"]

    raw = read_raw(src)
    df, stats = clean(raw, p["min_quantity"], p["min_price"],
                      p["drop_cancelled"], p.get("drop_duplicates", True))

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    print(f"Wrote {dst}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
