"""Precompute the per-measurement residual records the residual figure plots.

Smith's filter records one innovation per measurement it processes. The recorder keeps the
normalised innovation squared, the innovation norm and how long after the burn the measurement
arrived, but the arms that hold them are 95-245 MB pickles. This script pulls those records out
of the arms the residual figure needs, concatenates them with an episode index, and writes one
compact table per arm so the notebook never touches a pickle.

Run once after the grid finishes:
    PYTHONPATH=. python -m real_environment.build_residual_cache --seeds 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_DIRECTORY = PROJECT_ROOT / "results/noise_maneuver_grid"
CACHE_DIRECTORY = GRID_DIRECTORY / "residual_cache"
POLICIES = ("advanced_greedy", "frozen_cnnv2_ppo")
# The control plus the two impulsive isotropic arms, which is what the residual figure contrasts.
# The finite arm is included because halved detectability is a residual-shape claim.
CONDITIONS = ("nullburn", "random", "finite1800s", "stress100to2000")
NOISE_TAG = "sigma10"

COLUMNS = (
    ("episode", np.int16),
    ("time_seconds", np.float32),
    ("rso_id", np.int16),
    ("nis", np.float32),
    ("innovation_norm_km", np.float32),
    ("is_maneuvering", np.bool_),
    ("seconds_after_burn", np.float32),
)

# The dedicated residual-trace runs (run_residual_traces.sh) also carry the signed
# innovation, the standard deviation the filter assigns to each of its components, and the
# sensor that took the measurement. The grid arms predate those fields and do not have them.
TRACE_DIRECTORY = GRID_DIRECTORY.parent / "residual_traces"
TRACE_CACHE_DIRECTORY = TRACE_DIRECTORY / "cache"
TRACE_COLUMNS = COLUMNS + (
    ("agent_id", "U8"),
    ("innovation_x_km", np.float32),
    ("innovation_y_km", np.float32),
    ("innovation_z_km", np.float32),
    ("innovation_sigma_x_km", np.float32),
    ("innovation_sigma_y_km", np.float32),
    ("innovation_sigma_z_km", np.float32),
    ("whitened_x", np.float32),
    ("whitened_y", np.float32),
    ("whitened_z", np.float32),
)


def arm_path(policy: str, condition: str, seeds: int) -> Path:
    return GRID_DIRECTORY / f"{policy}_{condition}_{NOISE_TAG}_{seeds}seeds.p"


def residual_table(result: dict, columns=COLUMNS) -> dict[str, np.ndarray]:
    """Concatenate every episode's NIS table, tagging each record with its episode index."""
    tables = result.get("nis_table", [])
    if not tables:
        return {}
    collected: dict[str, list[np.ndarray]] = {name: [] for name, _ in columns}
    for episode, table in enumerate(tables):
        count = len(table["nis"])
        if count == 0:
            continue
        collected["episode"].append(np.full(count, episode, dtype=np.int16))
        for name, dtype in columns[1:]:
            if name not in table:
                raise KeyError(
                    f"{name} is missing from this arm's nis_table. Arms recorded before the "
                    "signed-innovation fields were added do not carry it; rerun the arm."
                )
            collected[name].append(np.asarray(table[name], dtype=dtype))
    if not collected["nis"]:
        return {}
    return {
        name: np.concatenate(parts).astype(dtype)
        for (name, dtype), parts in zip(columns, collected.values())
    }


def build_traces() -> None:
    """Reduce the dedicated residual-trace arms, which carry the signed innovation."""
    paths = sorted(TRACE_DIRECTORY.glob("*.p"))
    if not paths:
        print(f"  no arms in {TRACE_DIRECTORY}")
        return
    TRACE_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with path.open("rb") as stream:
            result = pickle.load(stream)
        table = residual_table(result, TRACE_COLUMNS)
        if not table:
            print(f"  no NIS records in {path.name}")
            continue
        output = TRACE_CACHE_DIRECTORY / f"{path.stem}.npz"
        np.savez_compressed(output, **table)
        sensors = sorted(set(table["agent_id"].tolist()))
        print(
            f"  {path.stem:52s} records={table['nis'].size:7d} "
            f"sensors={len(sensors)} -> {output.name} "
            f"({output.stat().st_size / 1e6:.1f} MB)"
        )
    print(f"trace cache written to {TRACE_CACHE_DIRECTORY}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument(
        "--traces",
        action="store_true",
        help="Reduce results/residual_traces instead of the grid arms.",
    )
    arguments = parser.parse_args()

    if arguments.traces:
        build_traces()
        return

    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for policy in POLICIES:
        for condition in CONDITIONS:
            path = arm_path(policy, condition, arguments.seeds)
            if not path.exists():
                print(f"  missing {path.name}")
                continue
            with path.open("rb") as stream:
                result = pickle.load(stream)
            table = residual_table(result)
            if not table:
                print(f"  no NIS records in {path.name}")
                continue
            output = CACHE_DIRECTORY / f"{policy}_{condition}.npz"
            np.savez_compressed(output, **table)
            finite = np.isfinite(table["seconds_after_burn"])
            print(
                f"  {policy:18s} {condition:16s} "
                f"records={table['nis'].size:7d} tagged={int(finite.sum()):7d} "
                f"-> {output.name} ({output.stat().st_size / 1e6:.1f} MB)"
            )
    print(f"residual cache written to {CACHE_DIRECTORY}")


if __name__ == "__main__":
    main()
