"""Validate paired pilot outputs and Smith coast-history reproduction."""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPOSITORY_ROOT / "experiments/csce723_maneuver_eval"
PILOT_ROOT = PROJECT_ROOT / "results/pilot"
SMITH_FIELDS = (
    "proptime",
    "meancovFOR",
    "meancovall",
    "mediancovFOR",
    "mediancovall",
    "rew",
    "nsats",
    "alt",
    "azi",
)


def load_pickle(path: Path) -> dict:
    with path.open("rb") as stream:
        return pickle.load(stream)


def maximum_history_difference(old: dict, new: dict, episode: int = 0) -> dict:
    differences = {}
    for field in SMITH_FIELDS:
        field_differences = []
        for agent_key in old[field][episode]:
            old_values = np.asarray(old[field][episode][agent_key])
            new_values = np.asarray(new[field][episode][agent_key])
            if old_values.shape != new_values.shape:
                raise AssertionError(
                    f"{field}/{agent_key} shape differs: {old_values.shape} != {new_values.shape}"
                )
            field_differences.append(
                float(np.max(np.abs(old_values - new_values))) if old_values.size else 0.0
            )
        differences[field] = max(field_differences, default=0.0)
    return differences


def rejected_measurements(result: dict, episode: int = 0) -> int:
    return sum(
        event["selected_by_smith_estimate"]
        and not event["truth_in_fov"]
        and event["time_seconds"] >= 900.0
        for event in result["measurement_events"][episode]
    )


def main() -> None:
    pilot = {
        "advanced_greedy_coast": load_pickle(PILOT_ROOT / "advanced_greedy_coast.p"),
        "advanced_greedy_maneuver": load_pickle(PILOT_ROOT / "advanced_greedy_maneuver.p"),
        "frozen_cnnv2_ppo_coast": load_pickle(PILOT_ROOT / "frozen_cnnv2_ppo_coast.p"),
        "frozen_cnnv2_ppo_maneuver": load_pickle(PILOT_ROOT / "frozen_cnnv2_ppo_maneuver.p"),
    }
    old = {
        "advanced_greedy": load_pickle(
            REPOSITORY_ROOT / "try_code/MARLSSA/advgreedy/100_sat_3_agents.p"
        ),
        "frozen_cnnv2_ppo": load_pickle(
            REPOSITORY_ROOT / "try_code/MARLSSA/cnn_v2_reward_1/100_sat_3_agents.p"
        ),
    }

    expected_fields = {
        "start_trace_cov_by_episode",
        "end_trace_cov_by_episode",
        "maneuver_rso_ids",
        "reacquisition_time",
        "reacquired",
        "truth_fov_events",
        "measurement_events",
        *SMITH_FIELDS,
    }
    for name, result in pilot.items():
        missing = expected_fields - set(result)
        if missing:
            raise AssertionError(f"{name} missing fields: {sorted(missing)}")
        if len(result["pair_ids"]) != 1:
            raise AssertionError(f"{name} is not a one-episode pilot")
        if np.asarray(result["start_trace_cov_by_episode"]).shape != (1, 100):
            raise AssertionError(f"{name} start covariance has wrong shape")
        if np.asarray(result["end_trace_cov_by_episode"]).shape != (1, 100):
            raise AssertionError(f"{name} end covariance has wrong shape")

    starts = [
        np.asarray(result["start_trace_cov_by_episode"][0]) for result in pilot.values()
    ]
    paired_start_max_difference = max(
        float(np.max(np.abs(starts[0] - other))) for other in starts[1:]
    )
    selected_ids_match = bool(
        np.array_equal(
            pilot["advanced_greedy_maneuver"]["maneuver_rso_ids"][0],
            pilot["frozen_cnnv2_ppo_maneuver"]["maneuver_rso_ids"][0],
        )
    )
    coast_history_differences = {
        "advanced_greedy": maximum_history_difference(
            old["advanced_greedy"], pilot["advanced_greedy_coast"]
        ),
        "frozen_cnnv2_ppo": maximum_history_difference(
            old["frozen_cnnv2_ppo"], pilot["frozen_cnnv2_ppo_coast"]
        ),
    }
    max_coast_history_difference = max(
        value
        for policy_differences in coast_history_differences.values()
        for value in policy_differences.values()
    )

    policy_summary = {}
    for policy_name in ("advanced_greedy", "frozen_cnnv2_ppo"):
        coast = pilot[f"{policy_name}_coast"]
        maneuver = pilot[f"{policy_name}_maneuver"]
        coast_mean = float(np.mean(coast["end_trace_cov_by_episode"][0]))
        maneuver_mean = float(np.mean(maneuver["end_trace_cov_by_episode"][0]))
        maneuver_ids = maneuver["maneuver_rso_ids"][0]
        policy_summary[policy_name] = {
            "coast_final_mean_trace_covariance": coast_mean,
            "maneuver_final_mean_trace_covariance": maneuver_mean,
            "paired_difference": maneuver_mean - coast_mean,
            "maneuvering_rso_count": len(maneuver_ids),
            "reacquired_count": int(np.sum(maneuver["reacquired"][0])),
            "not_reacquired_count": int(
                len(maneuver_ids) - np.sum(maneuver["reacquired"][0])
            ),
            "truth_gate_rejections": rejected_measurements(maneuver),
        }

    payload = {
        "status": "passed",
        "episodes_per_condition": 1,
        "paired_start_covariance_max_absolute_difference": paired_start_max_difference,
        "maneuvering_rso_ids_match_across_policies": selected_ids_match,
        "coast_history_max_absolute_difference_from_smith": max_coast_history_difference,
        "coast_history_differences_by_field": coast_history_differences,
        "policy_summary": policy_summary,
        "interpretation": "Engineering pilot only; no policy-performance inference from one seed.",
    }
    if paired_start_max_difference != 0.0:
        raise AssertionError("Paired pilot initial covariance arrays differ")
    if not selected_ids_match:
        raise AssertionError("Maneuvering RSO subsets differ across policies")
    if max_coast_history_difference > 1e-12:
        raise AssertionError("Coast histories do not reproduce Smith within tolerance")

    output_path = PROJECT_ROOT / "results/pilot/pilot_validation.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
