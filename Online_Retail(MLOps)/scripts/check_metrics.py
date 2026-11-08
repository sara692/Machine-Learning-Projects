"""Quality gate: fail (exit 1) if the tuned model is not good enough.

Run:  python scripts/check_metrics.py --min-silhouette 0.30 [--min-clusters 2]
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="reports/metrics.json")
    ap.add_argument("--min-silhouette", type=float, default=0.30)
    ap.add_argument("--min-clusters", type=int, default=2)
    args = ap.parse_args()

    m = json.load(open(args.metrics))
    ok = m["silhouette"] >= args.min_silhouette and m["n_clusters"] >= args.min_clusters
    print(f"{m['algorithm']}: silhouette={m['silhouette']:.3f} (min {args.min_silhouette}), "
          f"n_clusters={m['n_clusters']} (min {args.min_clusters}) -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
