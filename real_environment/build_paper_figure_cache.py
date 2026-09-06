"""Precompute the small arrays the paper notebook plots.

The grid arms are 95-245 MB pickles each. Reading them inside a notebook is slow and makes the
notebook impossible to rerun on a machine that only has the analysis outputs. This script reduces
the arms the paper needs down to a handful of curves and per-episode scalars, writes them to
results/noise_maneuver_grid/paper_cache, and the notebook reads only that.

Run once after the grid finishes:
    PYTHONPATH=. python -m real_environment.build_paper_figure_cache --seeds 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_DIRECTORY = PROJECT_ROOT / "results/noise_maneuver_grid"
CACHE_DIRECTORY = GRID_DIRECTORY / "paper_cache"
POLICIES = ("advanced_greedy", "frozen_cnnv2_ppo")
# Conditions the paper figures actually draw. The direction-resolved arms are summarised from the
# analysis CSVs instead, so they do not need history curves.
CONDITIONS = ("nullburn", "random", "stress100to2000")
NOISE_TAG = "sigma10"
POINTING_SEED = 0
# Every episode runs about 5400 s of scenario time. The histories are event sampled, so they are
# resampled onto this fixed 10 s grid before being averaged across episodes and policies.
TIME_GRID_SECONDS = np.arange(0.0, 5401.0, 10.0)


def arm_path(policy: str, condition: str, seeds: int) -> Path:
    return GRID_DIRECTORY / f"{policy}_{condition}_{NOISE_TAG}_{seeds}seeds.p"


def load_arm(policy: str, condition: str, seeds: int) -> dict | None:
    path = arm_path(policy, condition, seeds)
    if not path.exists():
        print(f"  missing {path.name}")
        return None
    with path.open("rb") as stream:
        return pickle.load(stream)


def history_curves(result: dict) -> dict[str, np.ndarray]:
    """Average the per-RSO covariance history over episodes, split by tagged and untagged.

    The environment logs a sample per observation event, not on a fixed clock, so the two policies
    produce different numbers of samples over the same 5400 s run and no two episodes line up by
    index. Each episode is reduced to a curve first, then interpolated onto one shared time grid,
    then averaged. Interpolating after the reduction is the same as before it, because every RSO in
    an episode shares that episode's time vector.
    """
    histories = result["trace_covariance_history_by_rso"]
    episodes = len(histories)
    grid = TIME_GRID_SECONDS
    accumulators = {name: np.zeros(grid.size) for name in
                    ("mean_all", "median_all", "mean_tagged", "mean_untagged")}
    for episode in range(episodes):
        times = np.asarray(histories[episode]["time_seconds"], dtype=float)
        traces = np.asarray(histories[episode]["trace_covariance"], dtype=float)
        tagged = np.asarray(result["maneuver_rso_ids"][episode], dtype=int)
        mask = np.zeros(traces.shape[1], dtype=bool)
        mask[tagged] = True
        reductions = {
            "mean_all": traces.mean(axis=1),
            "median_all": np.median(traces, axis=1),
            "mean_tagged": traces[:, mask].mean(axis=1),
            "mean_untagged": traces[:, ~mask].mean(axis=1),
        }
        for name, curve in reductions.items():
            accumulators[name] += np.interp(grid, times, curve)
    output = {name: value / episodes for name, value in accumulators.items()}
    output["time_seconds"] = grid
    output["episodes"] = np.array([episodes])
    return output


def error_history_curves(result: dict) -> dict[str, np.ndarray]:
    """Average the truth-ledger tracking error over episodes, tagged cohort only.

    The truth ledger samples on its own 60 s clock rather than the observation events, and only
    for the tagged objects. Each episode is reduced to its median-over-tagged curve first, then
    interpolated onto the shared time grid, then averaged, mirroring history_curves.
    """
    grid = TIME_GRID_SECONDS
    total = np.zeros(grid.size)
    histories = result["estimation_error_history"]
    for history in histories:
        times = np.asarray(history["time_seconds"], dtype=float)
        errors = np.asarray(history["estimate_truth_error_km"], dtype=float)
        total += np.interp(grid, times, np.median(errors, axis=1))
    return {
        "median_tagged_error_km": total / len(histories),
        "time_seconds": grid,
        "episodes": np.array([len(histories)]),
    }


def episode_rows(result: dict, policy: str, condition: str) -> list[dict]:
    """One row per episode: end-of-run covariance and tracking error, tagged against untagged."""
    rows = []
    episodes = len(result["end_trace_cov_by_episode"])
    for episode in range(episodes):
        end_trace = np.asarray(result["end_trace_cov_by_episode"][episode], dtype=float)
        tagged = np.asarray(result["maneuver_rso_ids"][episode], dtype=int)
        mask = np.zeros(end_trace.size, dtype=bool)
        mask[tagged] = True
        history = result["estimation_error_history"][episode]
        error = np.asarray(history["estimate_truth_error_km"], dtype=float)[-1]
        rso_id = np.asarray(history["rso_id"], dtype=int)
        tagged_error = error[np.isin(rso_id, tagged)]
        reacquired = np.asarray(result["reacquired"][episode], dtype=bool)
        rows.append(
            {
                "policy": policy,
                "condition": condition,
                "episode": episode,
                "mean_end_trace": float(end_trace.mean()),
                "median_end_trace": float(np.median(end_trace)),
                "tagged_mean_end_trace": float(end_trace[mask].mean()),
                "untagged_mean_end_trace": float(end_trace[~mask].mean()),
                "tagged_error_median_km": float(np.median(tagged_error)),
                "tagged_error_p90_km": float(np.percentile(tagged_error, 90)),
                "reacquired_fraction": float(reacquired.mean()),
            }
        )
    return rows


def ceiling_rows(result: dict, policy: str, condition: str) -> list[dict]:
    """Split end-of-run covariance into the three groups that set the metric's dynamic range.

    A maneuver can only reach the covariance by stopping an object from being observed, so the
    largest response the metric can show is the catalog mean when every tagged object goes unseen.
    """
    untagged, seen, unseen = [], [], []
    episodes = len(result["end_trace_cov_by_episode"])
    for episode in range(episodes):
        end_trace = np.asarray(result["end_trace_cov_by_episode"][episode], dtype=float)
        tagged = np.asarray(result["maneuver_rso_ids"][episode], dtype=int)
        mask = np.zeros(end_trace.size, dtype=bool)
        mask[tagged] = True
        counts = np.asarray(result["post_maneuver_observation_count"][episode], dtype=float)
        untagged.extend(end_trace[~mask].tolist())
        for index, rso in enumerate(tagged):
            observed = counts[index] if counts.size == tagged.size else counts[rso]
            (seen if observed > 0 else unseen).append(float(end_trace[rso]))
    tagged_fraction = len(seen) + len(unseen)
    tagged_fraction = tagged_fraction / (tagged_fraction + len(untagged))
    mean_untagged = float(np.mean(untagged))
    mean_unseen = float(np.mean(unseen)) if unseen else float("nan")
    ceiling = (1 - tagged_fraction) * mean_untagged + tagged_fraction * mean_unseen
    return [
        {
            "policy": policy,
            "condition": condition,
            "untagged_mean": mean_untagged,
            "tagged_seen_mean": float(np.mean(seen)) if seen else float("nan"),
            "tagged_unseen_mean": mean_unseen,
            "tagged_fraction": tagged_fraction,
            "observed_catalog_mean": float(np.mean(untagged + seen + unseen)),
            "ceiling_catalog_mean": ceiling,
            "n_untagged": len(untagged),
            "n_tagged_seen": len(seen),
            "n_tagged_unseen": len(unseen),
        }
    ]


def pointing_rows(result: dict, policy: str, condition: str) -> list[dict]:
    times = np.asarray(result["proptime"][POINTING_SEED]["agent_0"], dtype=float)
    azimuth = np.asarray(result["azi"][POINTING_SEED]["agent_0"], dtype=float)
    return [
        {"policy": policy, "condition": condition, "time_seconds": float(t), "azimuth_deg": float(a)}
        for t, a in zip(times, azimuth)
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    arguments = parser.parse_args()

    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    curves: dict[str, np.ndarray] = {}
    error_curves: dict[str, np.ndarray] = {}
    episodes: list[dict] = []
    ceilings: list[dict] = []
    pointing: list[dict] = []

    for policy in POLICIES:
        for condition in CONDITIONS:
            print(f"{policy} / {condition}")
            result = load_arm(policy, condition, arguments.seeds)
            if result is None:
                continue
            for name, array in history_curves(result).items():
                curves[f"{policy}|{condition}|{name}"] = array
            for name, array in error_history_curves(result).items():
                error_curves[f"{policy}|{condition}|{name}"] = array
            episodes.extend(episode_rows(result, policy, condition))
            ceilings.extend(ceiling_rows(result, policy, condition))
            pointing.extend(pointing_rows(result, policy, condition))
            del result

    np.savez_compressed(CACHE_DIRECTORY / "covariance_history.npz", **curves)
    print(f"  wrote covariance_history.npz ({len(curves)} arrays)")
    np.savez_compressed(CACHE_DIRECTORY / "error_history.npz", **error_curves)
    print(f"  wrote error_history.npz ({len(error_curves)} arrays)")
    write_csv(CACHE_DIRECTORY / "episode_end_state.csv", episodes)
    write_csv(CACHE_DIRECTORY / "covariance_ceiling.csv", ceilings)
    write_csv(CACHE_DIRECTORY / "pointing_history.csv", pointing)


if __name__ == "__main__":
    main()
