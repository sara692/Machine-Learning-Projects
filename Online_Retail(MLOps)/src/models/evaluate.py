import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient


def persona_hint(row, z) -> str:
    """Heuristic label from how a segment's median RFM compares with all customers (z-scores)."""
    r, f, m = -z["recency"], z["frequency"], z["monetary"]  # r high = bought recently
    if f > 0.5 and m > 0.5 and r > 0:
        return "Champions / VIP"
    if r < -0.5 and f < 0:
        return "Lost / Hibernating"
    if r > 0.3 and f <= 0.5:
        return "Recent / Promising"
    if f > 0.3 or m > 0.3:
        return "Loyal / Big spenders"
    return "Regular / Needs attention"


def profile_segments(rfm: pd.DataFrame, labels: np.ndarray, features) -> pd.DataFrame:
    df = rfm.copy()
    df["segment"] = labels
    grp = df.groupby("segment")
    prof = grp.size().rename("customers").to_frame()
    prof["share"] = (prof["customers"] / len(df)).round(4)
    for f in features:
        prof[f"{f}_mean"] = grp[f].mean().round(2)
        prof[f"{f}_median"] = grp[f].median().round(2)
    # z-score of each segment's median (in log space) against the whole population
    logs = np.log1p(df[features])
    mu, sd = logs.mean(), logs.std().replace(0, 1)
    hints = []
    for seg in prof.index:
        med = np.log1p(grp[features].median().loc[seg])
        z = ((med - mu) / sd).to_dict()
        hints.append(persona_hint(None, z))
    prof["persona_hint"] = hints
    return prof.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tuning.yaml")
    ap.add_argument("--tracking-uri", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name = cfg["mlflow"]["registered_model"]
    alias = cfg["mlflow"]["champion_alias"]
    feats = cfg["data"]["features"]

    mlflow.set_tracking_uri(args.tracking_uri or cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    rfm = pd.read_parquet(cfg["data"]["path"])
    pipe = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    version = MlflowClient().get_model_version_by_alias(name, alias).version
    labels = pipe.predict(rfm[feats].astype(float).to_numpy())

    prof = profile_segments(rfm[feats], labels, feats)
    Path("reports").mkdir(exist_ok=True)
    prof.to_csv("reports/cluster_profile.csv", index=False)
    out = rfm[["customer_id"]].copy() if "customer_id" in rfm else pd.DataFrame(index=rfm.index)
    out["segment"] = labels
    out.to_csv("reports/customer_segments.csv", index=False)

    with mlflow.start_run(run_name="evaluate-champion"):
        mlflow.set_tags({"model": name, "model_version": str(version), "alias": alias})
        mlflow.log_metric("n_segments", len(prof))
        mlflow.log_artifact("reports/cluster_profile.csv")
        mlflow.log_artifact("reports/customer_segments.csv")

    print(f"Champion '{name}' v{version}")
    print(prof.to_string(index=False))


if __name__ == "__main__":
    main()
