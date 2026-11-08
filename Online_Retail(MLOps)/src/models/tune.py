import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.cluster import DBSCAN, AgglomerativeClustering, Birch, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED_DEFAULT = 42


# ----------------------------------------------------------------------------- models
def build_estimator(algo: str, params: dict, seed: int):
    if algo == "kmeans":
        return KMeans(random_state=seed, **params)
    if algo == "gmm":
        return GaussianMixture(random_state=seed, **params)
    if algo == "birch":
        return Birch(**params)
    if algo == "agglomerative":
        return AgglomerativeClustering(**params)
    if algo == "dbscan":
        return DBSCAN(**params)
    raise ValueError(f"Unknown algorithm: {algo}")


def get_labels(est, X):
    """fit and return labels; hides API differences between estimators."""
    if hasattr(est, "fit_predict"):
        return est.fit_predict(X)
    est.fit(X)
    return est.predict(X)


# ----------------------------------------------------------------------------- scoring
def score(X, labels, cfg, rng_seed):
    """Return metrics dict, or None if the solution is invalid."""
    ev = cfg["evaluation"]
    noise_mask = labels == -1
    noise_share = float(noise_mask.mean())
    keep = ~noise_mask
    kept_labels = labels[keep]
    uniq, counts = np.unique(kept_labels, return_counts=True)
    n_clusters = len(uniq)

    reason = None
    if n_clusters < ev["min_clusters"]:
        reason = "too_few_clusters"
    elif n_clusters > ev["max_clusters"]:
        reason = "too_many_clusters"
    elif noise_share > ev["max_noise_share"]:
        reason = "too_much_noise"
    elif counts.min() / keep.sum() < ev["min_cluster_share"]:
        reason = "tiny_cluster"

    base = {"n_clusters": n_clusters, "noise_share": noise_share,
            "min_cluster_share": float(counts.min() / keep.sum()) if keep.sum() else 0.0}
    if reason:
        return base, reason

    Xk = X[keep]
    sample = min(ev["silhouette_sample_size"], len(Xk))
    sil = silhouette_score(Xk, kept_labels, sample_size=sample, random_state=rng_seed)
    base.update({
        "silhouette": float(sil),
        "davies_bouldin": float(davies_bouldin_score(Xk, kept_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Xk, kept_labels)),
    })
    return base, None


# ----------------------------------------------------------------------------- provenance
def run_provenance(data_path: str) -> dict:
    """Tags that make a run reproducible: git commit + hash of the exact data file used."""
    tags = {}
    try:
        tags["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        tags["git_commit"] = "unknown"
    try:
        tags["data_sha256"] = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()[:16]
    except Exception:
        tags["data_sha256"] = "unknown"
    return tags


# ----------------------------------------------------------------------------- data
def load_features(cfg):
    d = cfg["data"]
    df = pd.read_parquet(d["path"]) if d["path"].endswith(".parquet") else pd.read_csv(d["path"])
    X_raw = df[d["features"]].astype(float).to_numpy()
    X = np.log1p(X_raw) if d["log_transform"] else X_raw
    X = StandardScaler().fit_transform(X)
    return df, X_raw, X


# ----------------------------------------------------------------------------- final model
def make_servable_pipeline(algo, est, X_raw, X_scaled, labels, cfg):

    log = cfg["data"]["log_transform"]
    pre = [("log", FunctionTransformer(np.log1p if log else None, validate=True)),
           ("scaler", StandardScaler())]
    if hasattr(est, "predict"):
        final = est
        note = "native predict"
    else:
        keep = labels != -1
        final = KNeighborsClassifier(n_neighbors=15).fit(X_scaled[keep], labels[keep])
        note = "KNN surrogate trained on cluster labels (algorithm has no predict)"
    pipe = Pipeline(pre + [("model", final)])
    # refit preprocessing on raw data so the pipeline is self-contained
    log_step = pipe.named_steps["log"].fit(X_raw)
    pipe.named_steps["scaler"].fit(log_step.transform(X_raw))
    return pipe, note


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tuning.yaml")
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--data", default=None, help="override data path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.data:
        cfg["data"]["path"] = args.data
    seed = cfg["evaluation"].get("random_state", SEED_DEFAULT)
    random.seed(seed)
    np.random.seed(seed)

    mlflow.set_tracking_uri(args.tracking_uri or cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    df, X_raw, X = load_features(cfg)
    print(f"Loaded {len(df)} customers, features={cfg['data']['features']}")

    sel = cfg["evaluation"]["selection_metric"]
    leaderboard = []  # one row per valid trial
    best = {"score": -np.inf}

    with mlflow.start_run(run_name="model-selection") as parent:
        mlflow.log_artifact(args.config)
        mlflow.set_tags(run_provenance(cfg["data"]["path"]))
        mlflow.log_params({"n_customers": len(df), "selection_metric": sel,
                           "log_transform": cfg["data"]["log_transform"]})

        for algo, space in cfg["search_space"].items():
            grid = list(ParameterGrid(space))
            cap = cfg.get("max_trials_per_algo")
            if cap and len(grid) > cap:
                grid = random.Random(seed).sample(grid, cap)
            print(f"[{algo}] {len(grid)} trials")

            with mlflow.start_run(run_name=algo, nested=True):
                mlflow.set_tag("algorithm", algo)
                algo_best = -np.inf
                for i, params in enumerate(grid):
                    with mlflow.start_run(run_name=f"{algo}-{i:03d}", nested=True):
                        mlflow.set_tag("algorithm", algo)
                        mlflow.log_params({"algorithm": algo, **params})
                        try:
                            est = build_estimator(algo, params, seed)
                            labels = get_labels(est, X)
                            metrics, reason = score(X, labels, cfg, seed)
                        except Exception as e:  # keep sweeping on failures
                            mlflow.set_tag("status_reason", f"error: {e}")
                            continue

                        mlflow.log_metrics({k: v for k, v in metrics.items()})
                        if reason:
                            mlflow.set_tag("status_reason", reason)
                            continue
                        mlflow.set_tag("status_reason", "valid")

                        row = {"algorithm": algo, **params, **metrics}
                        leaderboard.append(row)
                        s = metrics[sel]
                        algo_best = max(algo_best, s)
                        if s > best["score"]:
                            best = {"score": s, "algo": algo, "params": params, "metrics": metrics}

                if np.isfinite(algo_best):
                    mlflow.log_metric(f"best_{sel}", algo_best)

        if not leaderboard:
            raise SystemExit("No valid clustering found. Relax constraints in configs/tuning.yaml.")

        # ---- leaderboard artifacts
        lb = pd.DataFrame(leaderboard).sort_values(sel, ascending=False)
        Path("reports").mkdir(exist_ok=True)
        lb.to_csv("reports/leaderboard.csv", index=False)
        mlflow.log_artifact("reports/leaderboard.csv")
        per_algo = lb.groupby("algorithm")[sel].max().sort_values(ascending=False)
        print("\nBest score per algorithm:\n", per_algo.to_string())

        # ---- refit the winner and register it
        algo, params = best["algo"], best["params"]
        est = build_estimator(algo, params, seed)
        labels = get_labels(est, X)
        pipe, note = make_servable_pipeline(algo, est, X_raw, X, labels, cfg)

        feature_cols = cfg["data"]["features"]
        sample_in = pd.DataFrame(X_raw[:5], columns=feature_cols)
        sig = infer_signature(sample_in, pipe.predict(sample_in))

        mlflow.set_tags({"best_algorithm": algo, "serving_note": note})
        mlflow.log_params({f"best_{k}": v for k, v in params.items()})
        mlflow.log_param("best_algorithm", algo)
        mlflow.log_metrics({f"best_{k}": v for k, v in best["metrics"].items()})

        sizes = pd.Series(labels).value_counts().sort_index()
        sizes.rename_axis("cluster").rename("size").to_csv("reports/cluster_sizes.csv")
        mlflow.log_artifact("reports/cluster_sizes.csv")
        json.dump({"algorithm": algo, "params": params, **best["metrics"]},
                  open("reports/metrics.json", "w"), indent=2)
        mlflow.log_artifact("reports/metrics.json")

        info = mlflow.sklearn.log_model(
            pipe, artifact_path="model", signature=sig, input_example=sample_in,
            serialization_format="cloudpickle",  # skops can't serialize the np.log1p step
            registered_model_name=cfg["mlflow"]["registered_model"])

        client = MlflowClient()
        version = info.registered_model_version
        client.set_registered_model_alias(cfg["mlflow"]["registered_model"],
                                          cfg["mlflow"]["champion_alias"], version)
        print(f"\nWinner: {algo} {params}\n{best['metrics']}")
        print(f"Registered '{cfg['mlflow']['registered_model']}' v{version} "
              f"-> alias '{cfg['mlflow']['champion_alias']}'  ({note})")
        print(f"Parent run id: {parent.info.run_id}")


if __name__ == "__main__":
    main()
