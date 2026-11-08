import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = ["recency", "frequency", "monetary"]


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index using quantile bins of the reference data."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(reference, edges)[0] / len(reference)
    cur_p = np.histogram(current, edges)[0] / len(current)
    ref_p, cur_p = np.clip(ref_p, eps, None), np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def category_psi(ref_labels, cur_labels, eps: float = 1e-4) -> float:
    """PSI between two label distributions (cluster-share shift)."""
    cats = sorted(set(ref_labels) | set(cur_labels))
    ref_p = np.array([np.mean(ref_labels == c) for c in cats])
    cur_p = np.array([np.mean(cur_labels == c) for c in cats])
    ref_p, cur_p = np.clip(ref_p, eps, None), np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, features=FEATURES, psi_threshold: float = 0.2) -> dict:
    per_feature = {}
    for f in features:
        ks = stats.ks_2samp(ref[f], cur[f])
        p = psi(ref[f], cur[f])
        per_feature[f] = {"psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                          "ks_pvalue": float(ks.pvalue), "drifted": bool(p >= psi_threshold)}
    n_drift = sum(v["drifted"] for v in per_feature.values())
    return {"n_reference": len(ref), "n_current": len(cur), "psi_threshold": psi_threshold,
            "share_of_drifted_features": n_drift / len(features), "features": per_feature}


def push_metrics(report: dict, gateway: str):
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    reg = CollectorRegistry()
    Gauge("seg_data_drift_share", "Share of drifted features", registry=reg).set(
        report["share_of_drifted_features"])
    g = Gauge("seg_feature_psi", "PSI per feature", ["feature"], registry=reg)
    for f, v in report["features"].items():
        g.labels(feature=f).set(v["psi"])
    if "cluster_share_psi" in report:
        Gauge("seg_cluster_share_psi", "PSI of cluster proportions", registry=reg).set(
            report["cluster_share_psi"])
    push_to_gateway(gateway, job="drift-job", registry=reg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/features/rfm.parquet")
    ap.add_argument("--current", default="data/features/rfm_current.parquet")
    ap.add_argument("--psi-threshold", type=float, default=0.2)
    ap.add_argument("--pushgateway", default=None, help="host:port, e.g. localhost:9091")
    ap.add_argument("--model-uri", default=None, help="also compare cluster shares using this model")
    ap.add_argument("--tracking-uri", default="http://localhost:5000")
    ap.add_argument("--output", default="reports/drift.json")
    args = ap.parse_args()

    ref, cur = pd.read_parquet(args.reference), pd.read_parquet(args.current)
    report = drift_report(ref, cur, FEATURES, args.psi_threshold)

    if args.model_uri:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(args.tracking_uri)
        pipe = mlflow.sklearn.load_model(args.model_uri)
        ref_l = pipe.predict(ref[FEATURES].astype(float).to_numpy())
        cur_l = pipe.predict(cur[FEATURES].astype(float).to_numpy())
        report["cluster_share_psi"] = round(category_psi(ref_l, cur_l), 4)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w"), indent=2)
    print(json.dumps(report, indent=2))

    if args.pushgateway:
        push_metrics(report, args.pushgateway)
        print(f"Pushed metrics to {args.pushgateway}")


if __name__ == "__main__":
    main()
