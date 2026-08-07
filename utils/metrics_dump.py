"""
Serialisation of rollout error metrics.

`evaluate_rollout` returns per-step scaled errors for position, velocity and angular
velocity. This module writes those sequences, their summary statistics and the rollout's
wall-clock cost to disk, so each evaluation leaves a durable record that can be cited or
replotted without re-running the rollout.
"""

import json
import os

import numpy as np


def _summarise(seq):
    """Summary statistics for one per-step error sequence."""
    if seq is None or len(seq) == 0:
        return {"mean": None, "final": None, "min": None, "max": None, "steps": 0}
    arr = np.asarray(seq, dtype=float)
    return {
        "mean": float(arr.mean()),
        "final": float(arr[-1]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "steps": int(arr.size),
    }


def dump_rollout_metrics(save_folder, experiment_name, pos_err, vel_err, angvel_err,
                         wall_clock_s, extra=None, write_per_step=True):
    """
    Write metrics for one evaluation to `<save_folder>/<experiment_name>/metrics/`.

    Produces `metrics.json` (summary and provenance) and, optionally, `per_step_errors.csv`
    holding the full sequences so error-growth curves can be replotted later.

    Args:
        save_folder: RESULTS_DIR for the case.
        experiment_name: the value passed to `evaluate_rollout`.
        pos_err, vel_err, angvel_err: scaled per-step error sequences.
        wall_clock_s: duration of the `evaluate_rollout` call, in seconds.
        extra: run-identifying metadata (case, device, checkpoint state).
        write_per_step: also emit the per-step CSV.

    Returns:
        The summary dictionary that was written.
    """
    out_dir = os.path.join(save_folder, experiment_name, "metrics")
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "experiment_name": experiment_name,
        "rollout_steps": len(pos_err) if pos_err is not None else 0,
        "wall_clock_seconds": float(wall_clock_s),
        "scaled_mae": {
            "position": _summarise(pos_err),
            "velocity": _summarise(vel_err),
            "angular_velocity": _summarise(angvel_err),
        },
    }
    if extra:
        summary.update(extra)

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if write_per_step and pos_err:
        with open(os.path.join(out_dir, "per_step_errors.csv"), "w") as fh:
            fh.write("step,scaled_mae_position,scaled_mae_velocity,scaled_mae_angular_velocity\n")
            for i, (p, v, w) in enumerate(zip(pos_err, vel_err, angvel_err)):
                fh.write(f"{i},{p:.10e},{v:.10e},{w:.10e}\n")

    print(f"[metrics] wrote {os.path.join(out_dir, 'metrics.json')} "
          f"({summary['rollout_steps']} steps, {wall_clock_s:.1f}s)")
    return summary


def dump_oblique_wall_summary(save_folder, pred_arr, gt_arr):
    """
    Persist the oblique-wall benchmark's post-collision angular velocity comparison.

    `pred_arr`/`gt_arr` are (n_angles, 4) arrays laid out as [angle_deg, wx, wy, wz],
    taken from the final rollout step at each impact angle.
    """
    out_dir = os.path.join(save_folder, "benchmark_oblique_summary", "metrics")
    os.makedirs(out_dir, exist_ok=True)

    pred_arr = np.asarray(pred_arr, dtype=float)
    gt_arr = np.asarray(gt_arr, dtype=float)

    rows = []
    for pred, gt in zip(pred_arr, gt_arr):
        angle = float(pred[0])
        pw, gw = pred[1:4], gt[1:4]
        rows.append({
            "angle_deg": angle,
            "predicted_ang_vel": [float(x) for x in pw],
            "ground_truth_ang_vel": [float(x) for x in gw],
            "abs_error": [float(x) for x in np.abs(pw - gw)],
            "l2_error": float(np.linalg.norm(pw - gw)),
            "ground_truth_l2_norm": float(np.linalg.norm(gw)),
        })

    with open(os.path.join(out_dir, "oblique_wall_angular_velocity.json"), "w") as fh:
        json.dump({"impacts": rows}, fh, indent=2)

    with open(os.path.join(out_dir, "oblique_wall_angular_velocity.csv"), "w") as fh:
        fh.write("angle_deg,pred_wx,pred_wy,pred_wz,gt_wx,gt_wy,gt_wz,l2_error\n")
        for r in rows:
            p, g = r["predicted_ang_vel"], r["ground_truth_ang_vel"]
            fh.write(f"{r['angle_deg']:.0f},"
                     f"{p[0]:.10e},{p[1]:.10e},{p[2]:.10e},"
                     f"{g[0]:.10e},{g[1]:.10e},{g[2]:.10e},"
                     f"{r['l2_error']:.10e}\n")

    print(f"[metrics] wrote oblique-wall angular velocity comparison to {out_dir}")
    return rows
