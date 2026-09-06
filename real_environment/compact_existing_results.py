"""Compact evaluator-owned result histories without changing Smith-compatible fields."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np


def compact_covariance_episode(episode) -> dict:
    if "time_seconds" in episode and "trace_covariance" in episode:
        return episode
    rso_ids = sorted(episode)
    if not rso_ids:
        return {
            "time_seconds": np.array([], dtype=np.float32),
            "trace_covariance": np.empty((0, 0), dtype=np.float32),
        }
    times = np.asarray([row[0] for row in episode[rso_ids[0]]], dtype=np.float32)
    values = np.column_stack(
        [np.asarray([row[1] for row in episode[rso_id]], dtype=np.float32) for rso_id in rso_ids]
    )
    return {"time_seconds": times, "trace_covariance": values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        before_bytes = path.stat().st_size
        with path.open("rb") as stream:
            result = pickle.load(stream)
        result["trace_covariance_history_by_rso"] = [
            compact_covariance_episode(episode)
            for episode in result["trace_covariance_history_by_rso"]
        ]
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(result, stream)
        temporary.replace(path)
        after_bytes = path.stat().st_size
        print(
            f"{path}: {before_bytes / 1024**2:.1f} MiB -> {after_bytes / 1024**2:.1f} MiB"
        )


if __name__ == "__main__":
    main()
