"""Export the champion model from the MLflow registry into a plain folder, ready to bake into a container.

Writes (default folder deploy/model):
  model.pkl               the fitted pipeline (log1p -> scaler -> best model)
  model_info.json         version, algorithm, features, library versions it was trained with
  requirements-model.txt  exact versions of scikit-learn / numpy / pandas / scipy used for training
  segment_profile.csv     segment descriptions (copied from reports/cluster_profile.csv, if it exists)

The deployed API then needs no MLflow server. Re-run this after every new champion.

Run:  python scripts/export_champion.py [--config configs/tuning.yaml] [--tracking-uri URI]
"""
import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

try:
    import cloudpickle as pickler  # ships with MLflow
except ImportError:  # pragma: no cover
    import pickle as pickler

LIBS = ["scikit-learn", "numpy", "pandas", "scipy"]


def export_pipeline(pipe, out_dir, *, model_name, version, features, profile_csv=None,
                    algorithm=None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "model.pkl", "wb") as f:
        pickler.dump(pipe, f)

    libs = {name: pkg_version(name) for name in LIBS}
    info = {
        "model_name": model_name,
        "version": int(version),
        "algorithm": algorithm or type(pipe[-1]).__name__,
        "features": list(features),
        "library_versions": libs,
        "python_version": platform.python_version(),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "model_info.json").write_text(json.dumps(info, indent=2))
    (out / "requirements-model.txt").write_text("".join(f"{k}=={v}\n" for k, v in libs.items()))

    prof = Path(profile_csv) if profile_csv else None
    if prof and prof.exists():
        shutil.copy(prof, out / "segment_profile.csv")
        info["segment_profile"] = True
    else:
        (out / "segment_profile.csv").unlink(missing_ok=True)
        info["segment_profile"] = False
    return info


def main():
    import mlflow
    import mlflow.sklearn
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tuning.yaml")
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--out", default="deploy/model")
    ap.add_argument("--profile", default="reports/cluster_profile.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name, alias = cfg["mlflow"]["registered_model"], cfg["mlflow"]["champion_alias"]
    mlflow.set_tracking_uri(args.tracking_uri or cfg["mlflow"]["tracking_uri"])

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    pipe = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    # the run that produced the model records which algorithm actually won (DBSCAN/Agglomerative are
    # wrapped in a nearest-neighbours classifier for serving, so the pipeline's last step is not informative)
    algorithm = client.get_run(mv.run_id).data.tags.get("best_algorithm")
    info = export_pipeline(pipe, args.out, model_name=name, version=mv.version,
                           features=cfg["data"]["features"], profile_csv=args.profile, algorithm=algorithm)

    print(f"Exported '{name}' v{info['version']} ({info['algorithm']}) to {args.out}/")
    print("  libraries:", ", ".join(f"{k} {v}" for k, v in info["library_versions"].items()))
    if not info["segment_profile"]:
        print(f"  note: {args.profile} not found - run src/models/evaluate.py first if you want "
              "segment descriptions in the UI")


if __name__ == "__main__":
    main()
