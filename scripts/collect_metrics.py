#!/usr/bin/env python
"""
Aggregate every `metrics.json` written by a reproduction run into one CSV + markdown table.

Usage:
    python scripts/collect_metrics.py [--out_dir reports]

Walks both cases' results directories, collects the per-rollout metric dumps produced by
`utils.metrics_dump`, and emits:

    reports/metrics_summary.csv
    reports/metrics_summary.json
    reports/metrics_summary.md

Runs whose `checkpoint_loaded` flag is False are flagged loudly: those numbers come from
randomly initialised weights and are not results.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULT_DIRS = [
    ("Case 04 (homogeneous)", os.path.join(REPO_ROOT, "case_04_dem_simple", "results")),
    ("Case 05 (gravity)", os.path.join(REPO_ROOT, "case_05_dem_hard", "results")),
]


def find_metric_files(root):
    hits = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "metrics" and "metrics.json" in filenames:
            hits.append(os.path.join(dirpath, "metrics.json"))
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "reports"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for case_label, results_dir in RESULT_DIRS:
        if not os.path.isdir(results_dir):
            print(f"[skip] no results directory: {results_dir}")
            continue
        for path in find_metric_files(results_dir):
            with open(path) as fh:
                m = json.load(fh)
            sm = m.get("scaled_mae", {})
            rows.append({
                "case": case_label,
                "experiment": m.get("experiment_name"),
                "mode": m.get("mode"),
                "evaluated_case": m.get("evaluated_case"),
                "rollout_steps": m.get("rollout_steps"),
                "wall_clock_s": m.get("wall_clock_seconds"),
                "checkpoint_loaded": m.get("checkpoint_loaded"),
                "pos_mean": sm.get("position", {}).get("mean"),
                "pos_final": sm.get("position", {}).get("final"),
                "vel_mean": sm.get("velocity", {}).get("mean"),
                "vel_final": sm.get("velocity", {}).get("final"),
                "angvel_mean": sm.get("angular_velocity", {}).get("mean"),
                "angvel_final": sm.get("angular_velocity", {}).get("final"),
                "source": os.path.relpath(path, REPO_ROOT),
            })

    if not rows:
        print("No metrics.json files found. Run the evaluation modes first.")
        return 1

    rows.sort(key=lambda r: (r["case"], r["experiment"] or ""))

    fields = ["case", "experiment", "mode", "evaluated_case", "rollout_steps", "wall_clock_s",
              "checkpoint_loaded", "pos_mean", "pos_final", "vel_mean", "vel_final",
              "angvel_mean", "angvel_final", "source"]

    csv_path = os.path.join(args.out_dir, "metrics_summary.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(fields) + "\n")
        for r in rows:
            fh.write(",".join("" if r[k] is None else str(r[k]) for k in fields) + "\n")

    json_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2)

    def fmt(x):
        return "-" if x is None else f"{x:.4e}"

    md = ["# Rollout metrics summary", "",
          "Scaled MAE is dimensionless: each error is divided by the corresponding training-set",
          "normalisation scale (`edge_feat_max` for position, `node_vel_max` for velocity,",
          "`node_angvel_max` for angular velocity), matching `evaluate_rollout`.", "",
          "| Case | Experiment | Steps | Wall clock (s) | Pos MAE mean | Pos MAE final | "
          "Vel MAE mean | Vel MAE final | AngVel MAE mean | AngVel MAE final |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        flag = "" if r["checkpoint_loaded"] else " **[RANDOM WEIGHTS - NOT A RESULT]**"
        wc = "-" if r["wall_clock_s"] is None else f"{r['wall_clock_s']:.1f}"
        md.append(
            f"| {r['case']} | {r['experiment']}{flag} | {r['rollout_steps']} | {wc} | "
            f"{fmt(r['pos_mean'])} | {fmt(r['pos_final'])} | "
            f"{fmt(r['vel_mean'])} | {fmt(r['vel_final'])} | "
            f"{fmt(r['angvel_mean'])} | {fmt(r['angvel_final'])} |")

    unloaded = [r for r in rows if not r["checkpoint_loaded"]]
    if unloaded:
        md += ["", "> **WARNING:** the following runs evaluated with randomly initialised weights",
               "> because no checkpoint was found. Discard them.", ""]
        md += [f"> - {r['experiment']}" for r in unloaded]

    # Fold in the oblique-wall angular velocity comparison if it exists.
    ow = os.path.join(REPO_ROOT, "case_04_dem_simple", "results",
                      "benchmark_oblique_summary", "metrics",
                      "oblique_wall_angular_velocity.json")
    if os.path.isfile(ow):
        with open(ow) as fh:
            impacts = json.load(fh)["impacts"]
        md += ["", "## Oblique wall impact - final post-collision angular velocity", "",
               "| Angle | Predicted (wx, wy, wz) | DEM ground truth (wx, wy, wz) | L2 error | GT L2 norm |",
               "| ---: | --- | --- | ---: | ---: |"]
        for r in impacts:
            p = ", ".join(f"{v:+.4e}" for v in r["predicted_ang_vel"])
            g = ", ".join(f"{v:+.4e}" for v in r["ground_truth_ang_vel"])
            md.append(f"| {r['angle_deg']:.0f}° | {p} | {g} | "
                      f"{r['l2_error']:.4e} | {r['ground_truth_l2_norm']:.4e} |")

    md_path = os.path.join(args.out_dir, "metrics_summary.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"Wrote:\n  {csv_path}\n  {json_path}\n  {md_path}")
    print(f"{len(rows)} rollout(s) aggregated; {len(unloaded)} with missing checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
