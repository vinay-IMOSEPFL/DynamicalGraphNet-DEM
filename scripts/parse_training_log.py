#!/usr/bin/env python
"""
Turn a training log into loss / validation-score curves.

Usage:
    python scripts/parse_training_log.py logs/train_case04_<stamp>.log --tag case04

Reads the two line formats the training loop emits:

    Epoch 12/500 | Train Loss (MSE): 1.234560e-02          <- every epoch
    Validation Score: 3.4e-03 (Pos: 1.1e-03, Vel: 1.2e-03, AngVel: 1.1e-03)

and writes `<out_dir>/<tag>_training_curve.csv` plus a two-panel PNG of the training loss
and the validation score.
"""

import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+) \| Train Loss \(MSE\): ([0-9.eE+-]+)")
VAL_RE = re.compile(
    r"Validation Score: ([0-9.eE+-]+) \(Pos: ([0-9.eE+-]+), Vel: ([0-9.eE+-]+), AngVel: ([0-9.eE+-]+)\)")
# The validation block is preceded by "Epoch N Train Loss (MSE): ..." (no pipe).
VAL_EPOCH_RE = re.compile(r"Epoch (\d+) Train Loss \(MSE\): ([0-9.eE+-]+)")


def parse(path):
    epochs, losses = [], []
    val_epochs, val_scores, val_pos, val_vel, val_ang = [], [], [], [], []
    last_epoch = None

    with open(path, errors="replace") as fh:
        for raw in fh:
            # tqdm writes carriage-returned progress bars into the same stream.
            for line in raw.split("\r"):
                m = EPOCH_RE.search(line)
                if m:
                    last_epoch = int(m.group(1))
                    epochs.append(last_epoch)
                    losses.append(float(m.group(3)))
                    continue
                # Older logs also carried a separate loss line on validation epochs.
                m = VAL_EPOCH_RE.search(line)
                if m:
                    last_epoch = int(m.group(1))
                    continue
                m = VAL_RE.search(line)
                if m:
                    val_epochs.append(last_epoch if last_epoch else len(val_epochs) * 5)
                    val_scores.append(float(m.group(1)))
                    val_pos.append(float(m.group(2)))
                    val_vel.append(float(m.group(3)))
                    val_ang.append(float(m.group(4)))

    return dict(epochs=epochs, losses=losses, val_epochs=val_epochs, val_scores=val_scores,
                val_pos=val_pos, val_vel=val_vel, val_ang=val_ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = parse(args.log)
    if not d["epochs"]:
        print(f"No per-epoch loss lines found in {args.log}")
        return 1

    csv_path = os.path.join(args.out_dir, f"{args.tag}_training_curve.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_loss_mse"])
        for e, l in zip(d["epochs"], d["losses"]):
            w.writerow([e, f"{l:.10e}"])

    val_csv = os.path.join(args.out_dir, f"{args.tag}_validation_curve.csv")
    with open(val_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "total_val_score", "scaled_mae_pos", "scaled_mae_vel", "scaled_mae_angvel"])
        for e, s, p, v, a in zip(d["val_epochs"], d["val_scores"], d["val_pos"], d["val_vel"], d["val_ang"]):
            w.writerow([e, f"{s:.10e}", f"{p:.10e}", f"{v:.10e}", f"{a:.10e}"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(d["epochs"], d["losses"], lw=1.2, color="#2b6cb0")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss (MSE, standardised)")
    axes[0].set_title(f"{args.tag}: training loss")
    axes[0].grid(alpha=0.3, which="both")

    if d["val_epochs"]:
        axes[1].plot(d["val_epochs"], d["val_scores"], "o-", ms=3, lw=1.2,
                     color="#c53030", label="total score")
        axes[1].plot(d["val_epochs"], d["val_pos"], lw=1, alpha=0.7, label="pos")
        axes[1].plot(d["val_epochs"], d["val_vel"], lw=1, alpha=0.7, label="vel")
        axes[1].plot(d["val_epochs"], d["val_ang"], lw=1, alpha=0.7, label="angvel")
        axes[1].set_yscale("log")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Scaled MAE (200-step rollout)")
    axes[1].set_title(f"{args.tag}: validation rollout")
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    png = os.path.join(args.out_dir, f"{args.tag}_training_curves.png")
    fig.savefig(png, dpi=150)

    best_i = min(range(len(d["val_scores"])), key=lambda i: d["val_scores"][i]) if d["val_scores"] else None
    print(f"epochs parsed: {len(d['epochs'])} (last = {d['epochs'][-1]})")
    print(f"train loss: first {d['losses'][0]:.4e} -> last {d['losses'][-1]:.4e} "
          f"(min {min(d['losses']):.4e})")
    if best_i is not None:
        print(f"best validation score {d['val_scores'][best_i]:.4e} at epoch {d['val_epochs'][best_i]}")
    nan_epochs = [e for e, l in zip(d["epochs"], d["losses"]) if l != l]
    if nan_epochs:
        print(f"*** NaN training loss at epochs: {nan_epochs[:20]} ***")
    print(f"Wrote:\n  {csv_path}\n  {val_csv}\n  {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
