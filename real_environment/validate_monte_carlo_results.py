"""Validate the completed paired Monte Carlo result matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "monte_carlo"
REPORT_PATH = RESULT_ROOT / "monte_carlo_validation.json"
POLICIES = ("frozen_cnnv2_ppo", "advanced_greedy")
CONDITIONS = {
    # Coast retains the paired primary experiment's configured 10 m/s delta-v
    # in metadata, but applies it to zero RSOs.
    "coast": (0.0, 10.0),
    "maneuver_rate05_10mps": (0.05, 10.0),
    "maneuver_rate10_10mps": (0.10, 10.0),
    "maneuver_rate20_10mps": (0.20, 10.0),
    "stress_rate20_100mps": (0.20, 100.0),
}
EXPECTED_PAIR_IDS = [f"seed_{seed:03d}" for seed in range(100)]


def result_path(policy: str, condition: str) -> Path:
    return RESULT_ROOT / f"{policy}_{condition}_100seeds.p"


def assert_compact_table(table: dict, expected_columns: set[str], label: str) -> int:
    if set(table) != expected_columns:
        raise AssertionError(f"{label}: columns {set(table)} != {expected_columns}")
    lengths = {column: len(np.asarray(values)) for column, values in table.items()}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"{label}: unequal column lengths {lengths}")
    return next(iter(lengths.values()), 0)


def validate_file(path: Path, policy: str, condition_key: str) -> tuple[dict, dict]:
    expected_fraction, expected_delta_v = CONDITIONS[condition_key]
    with path.open("rb") as stream:
        result = pickle.load(stream)

    if result["pair_ids"] != EXPECTED_PAIR_IDS:
        raise AssertionError(f"{path.name}: missing, duplicate, or unordered pair IDs")
    if result["policy_name"] != policy:
        raise AssertionError(f"{path.name}: policy metadata mismatch")
    if len(result["scenario_manifest_paths"]) != 100:
        raise AssertionError(f"{path.name}: expected 100 manifest paths")
    if len(set(result["scenario_manifest_paths"])) != 100:
        raise AssertionError(f"{path.name}: manifest paths are not unique by seed")
    if not np.isclose(float(result["maneuvering_fraction"]), expected_fraction):
        raise AssertionError(f"{path.name}: maneuver fraction mismatch")
    delta_v = np.asarray(result["delta_v_rtn_mps"], dtype=float)
    np.testing.assert_allclose(delta_v, [0.0, expected_delta_v, 0.0], rtol=0, atol=0)

    start = np.asarray(result["start_trace_cov_by_episode"], dtype=float)
    end = np.asarray(result["end_trace_cov_by_episode"], dtype=float)
    if start.shape != (100, 100) or end.shape != (100, 100):
        raise AssertionError(f"{path.name}: covariance shape mismatch")
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        raise AssertionError(f"{path.name}: non-finite terminal covariance")
    if (start < 0).any() or (end < 0).any():
        raise AssertionError(f"{path.name}: negative trace covariance")

    expected_maneuver_count = int(round(100 * expected_fraction))
    total_truth_events = 0
    total_measurement_events = 0
    total_gate_rejections = 0
    maneuver_ids = []
    for episode in range(100):
        covariance_table = result["trace_covariance_history_by_rso"][episode]
        steps = assert_compact_table(
            covariance_table, {"time_seconds", "trace_covariance"},
            f"{path.name} episode {episode} covariance history",
        )
        covariance_matrix = np.asarray(covariance_table["trace_covariance"])
        if covariance_matrix.shape != (steps, 100) or not np.isfinite(covariance_matrix).all():
            raise AssertionError(f"{path.name} episode {episode}: invalid covariance history")
        times = np.asarray(covariance_table["time_seconds"], dtype=float)
        if steps < 2 or times[0] != 0 or np.any(np.diff(times) < 0):
            raise AssertionError(f"{path.name} episode {episode}: invalid covariance times")

        ids = np.asarray(result["maneuver_rso_ids"][episode], dtype=int)
        if len(ids) != expected_maneuver_count or len(np.unique(ids)) != len(ids):
            raise AssertionError(f"{path.name} episode {episode}: maneuver ID count mismatch")
        if len(ids) and ((ids < 0).any() or (ids >= 100).any()):
            raise AssertionError(f"{path.name} episode {episode}: maneuver ID out of range")
        maneuver_ids.append(tuple(ids.tolist()))

        reacquired = np.asarray(result["reacquired"][episode], dtype=bool)
        reacquisition = np.asarray(result["reacquisition_time"][episode], dtype=float)
        counts = np.asarray(result["post_maneuver_observation_count"][episode], dtype=int)
        if reacquired.shape != (100,) or reacquisition.shape != (100,) or counts.shape != (100,):
            raise AssertionError(f"{path.name} episode {episode}: maneuver outcome shape mismatch")
        outside = np.ones(100, dtype=bool)
        outside[ids] = False
        if reacquired[outside].any() or np.isfinite(reacquisition[outside]).any() or counts[outside].any():
            raise AssertionError(f"{path.name} episode {episode}: outcomes assigned to coast RSOs")
        if not np.array_equal(reacquired[ids], np.isfinite(reacquisition[ids])):
            raise AssertionError(f"{path.name} episode {episode}: reacquisition flag/time mismatch")
        if np.any(reacquisition[ids][np.isfinite(reacquisition[ids])] < 0):
            raise AssertionError(f"{path.name} episode {episode}: negative reacquisition time")

        truth = result["truth_fov_events"][episode]
        measurements = result["measurement_events"][episode]
        if condition_key == "coast" and not truth and not measurements:
            truth_count = measurement_count = 0
        else:
            truth_count = assert_compact_table(
                truth,
                {"time_seconds", "agent_id", "rso_id", "truth_in_fov", "selected_by_smith_estimate",
                 "truth_azimuth_deg", "truth_altitude_deg", "angular_separation_deg"},
                f"{path.name} episode {episode} truth events",
            )
            measurement_count = assert_compact_table(
                measurements,
                {"time_seconds", "agent_id", "rso_id", "selected_by_smith_estimate", "truth_in_fov",
                 "measurement_received"},
                f"{path.name} episode {episode} measurement events",
            )
        total_truth_events += truth_count
        total_measurement_events += measurement_count
        measurement_columns = measurements if isinstance(measurements, dict) else {}
        selected = np.asarray(measurement_columns.get("selected_by_smith_estimate", []), dtype=bool)
        in_fov = np.asarray(measurement_columns.get("truth_in_fov", []), dtype=bool)
        received = np.asarray(measurement_columns.get("measurement_received", []), dtype=bool)
        if np.any(received & (~selected | ~in_fov)):
            raise AssertionError(f"{path.name} episode {episode}: impossible measurement outcome")
        total_gate_rejections += int(np.sum(selected & ~in_fov))

    summary = {
        "episodes": 100,
        "final_mean_trace_covariance": float(np.mean(end, axis=1).mean()),
        "final_median_trace_covariance": float(np.median(end, axis=1).mean()),
        "final_p90_trace_covariance": float(np.quantile(end, 0.90, axis=1).mean()),
        "truth_fov_events": total_truth_events,
        "measurement_events": total_measurement_events,
        "truth_gate_rejections": total_gate_rejections,
    }
    pairing = {"start": start, "maneuver_ids": maneuver_ids}
    return summary, pairing


def main() -> None:
    report = {"status": "passed", "files": {}, "pairing_checks": {}}
    pairing = {}
    for policy in POLICIES:
        for condition_key in CONDITIONS:
            path = result_path(policy, condition_key)
            if not path.is_file():
                raise FileNotFoundError(path)
            summary, pairing[(policy, condition_key)] = validate_file(path, policy, condition_key)
            report["files"][path.name] = summary

    reference_start = pairing[(POLICIES[0], "coast")]["start"]
    start_max_difference = 0.0
    for key, values in pairing.items():
        start_max_difference = max(
            start_max_difference,
            float(np.max(np.abs(values["start"] - reference_start))),
        )
    if start_max_difference != 0.0:
        raise AssertionError("Initial covariance is not exactly paired across all conditions")

    maneuver_id_mismatches = 0
    for condition_key in CONDITIONS:
        left = pairing[(POLICIES[0], condition_key)]["maneuver_ids"]
        right = pairing[(POLICIES[1], condition_key)]["maneuver_ids"]
        maneuver_id_mismatches += sum(a != b for a, b in zip(left, right))
    if maneuver_id_mismatches:
        raise AssertionError("Maneuvering RSO subsets differ between policies")

    report["pairing_checks"] = {
        "initial_covariance_max_absolute_difference": start_max_difference,
        "maneuver_subset_policy_mismatches": maneuver_id_mismatches,
        "pair_ids_exactly_seed_000_through_seed_099": True,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
