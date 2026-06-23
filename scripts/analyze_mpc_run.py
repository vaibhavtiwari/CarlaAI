import argparse
import json
import os
from glob import glob

if __package__ in (None, ""):
    import _bootstrap
else:
    from . import _bootstrap

import matplotlib.pyplot as plt
import numpy as np


def _episode_arrays(episode):
    steps = episode["steps"]
    arrays = {
        "step": np.array([step["step"] for step in steps], dtype=np.int32),
        "speed_kmh": np.array([step["vehicle"]["speed_kmh"] for step in steps], dtype=np.float64),
        "x": np.array([step["vehicle"]["x"] for step in steps], dtype=np.float64),
        "y": np.array([step["vehicle"]["y"] for step in steps], dtype=np.float64),
        "steer": np.array([step["control"]["steer"] for step in steps], dtype=np.float64),
        "throttle": np.array([step["control"]["throttle"] for step in steps], dtype=np.float64),
        "brake": np.array([step["control"]["brake"] for step in steps], dtype=np.float64),
        "distance_from_center": np.array(
            [step["tracking"]["distance_from_center"] for step in steps], dtype=np.float64
        ),
    }

    def maybe_series(field):
        values = []
        for step in steps:
            mpc = step.get("mpc")
            values.append(np.nan if mpc is None else float(mpc[field]))
        return np.array(values, dtype=np.float64)

    arrays["objective_value"] = maybe_series("objective_value")
    arrays["iterations"] = maybe_series("iterations")
    arrays["solve_time_ms"] = maybe_series("solve_time_ms")
    arrays["position_cost"] = maybe_series("position_cost")
    arrays["heading_cost"] = maybe_series("heading_cost")
    arrays["speed_cost"] = maybe_series("speed_cost")
    arrays["control_cost"] = maybe_series("control_cost")
    arrays["smoothness_cost"] = maybe_series("smoothness_cost")
    arrays["terminal_cost"] = maybe_series("terminal_cost")
    arrays["reference_speed_kmh"] = maybe_series("reference_speed_mps") * 3.6
    return arrays


def _save_trajectory_plot(out_dir, episode_idx, arrays):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(arrays["x"], arrays["y"], label="ego trajectory", linewidth=2.0)
    ax.scatter(arrays["x"][0], arrays["y"][0], label="start", s=40)
    ax.scatter(arrays["x"][-1], arrays["y"][-1], label="end", s=40)
    ax.set_title(f"MPC Trajectory - Episode {episode_idx}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"episode_{episode_idx:03d}_trajectory.png"), dpi=150)
    plt.close(fig)


def _save_timeseries_plot(out_dir, episode_idx, arrays):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(arrays["step"], arrays["distance_from_center"], label="cross-track error")
    axes[0].set_ylabel("m")
    axes[0].set_title("Tracking Error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(arrays["step"], arrays["speed_kmh"], label="speed")
    axes[1].plot(arrays["step"], arrays["reference_speed_kmh"], label="target speed", linestyle="--")
    axes[1].set_ylabel("km/h")
    axes[1].set_title("Speed")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(arrays["step"], arrays["steer"], label="steer")
    axes[2].plot(arrays["step"], arrays["throttle"], label="throttle")
    axes[2].plot(arrays["step"], arrays["brake"], label="brake")
    axes[2].set_ylabel("command")
    axes[2].set_title("Controls")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(arrays["step"], arrays["solve_time_ms"], label="solve time")
    axes[3].plot(arrays["step"], arrays["iterations"], label="iterations")
    axes[3].set_ylabel("solver")
    axes[3].set_xlabel("step")
    axes[3].set_title("Solver Behavior")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"episode_{episode_idx:03d}_timeseries.png"), dpi=150)
    plt.close(fig)


def _save_cost_plot(out_dir, episode_idx, arrays):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(arrays["step"], arrays["position_cost"], label="position")
    ax.plot(arrays["step"], arrays["heading_cost"], label="heading")
    ax.plot(arrays["step"], arrays["speed_cost"], label="speed")
    ax.plot(arrays["step"], arrays["control_cost"], label="control")
    ax.plot(arrays["step"], arrays["smoothness_cost"], label="smoothness")
    ax.plot(arrays["step"], arrays["terminal_cost"], label="terminal")
    ax.plot(arrays["step"], arrays["objective_value"], label="total objective", linewidth=2.0, alpha=0.8)
    ax.set_title(f"MPC Cost Breakdown - Episode {episode_idx}")
    ax.set_xlabel("step")
    ax.set_ylabel("cost")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"episode_{episode_idx:03d}_costs.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze MPC debug trace JSON")
    parser.add_argument("trace_path", type=str, help="Path to a debug JSON file or a directory containing per-episode debug files")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for generated figures")
    args = parser.parse_args()

    if os.path.isdir(args.trace_path):
        trace_files = sorted(glob(os.path.join(args.trace_path, "*_episode_*_debug.json")))
        output_dir = args.output_dir or os.path.join(args.trace_path, "plots")
    else:
        trace_files = [args.trace_path]
        output_dir = args.output_dir or os.path.splitext(args.trace_path)[0] + "_plots"

    os.makedirs(output_dir, exist_ok=True)

    analyzed = 0
    for trace_file in trace_files:
        with open(trace_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "steps" not in payload:
            continue
        episode_idx = int(payload["episode_idx"])
        if not payload.get("steps"):
            continue
        arrays = _episode_arrays(payload)
        _save_trajectory_plot(output_dir, episode_idx, arrays)
        _save_timeseries_plot(output_dir, episode_idx, arrays)
        _save_cost_plot(output_dir, episode_idx, arrays)
        analyzed += 1

    print(f"Saved MPC analysis plots to {output_dir} for {analyzed} episode(s)")


if __name__ == "__main__":
    main()
