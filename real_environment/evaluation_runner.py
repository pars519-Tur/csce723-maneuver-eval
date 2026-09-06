"""Inference-only paired evaluator for the external maneuver environment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from real_environment.maneuver_environment import (
    ExternalManeuverEnvironment,
    ManeuverConfiguration,
)
from real_environment.maneuver_plan import (
    ExplicitManeuverConfiguration,
    RandomManeuverConfiguration,
    plan_from_manifest_events,
)
from real_environment.measurement_noise import MeasurementNoiseConfiguration
from real_environment.policy_adapters import AdvancedGreedyAdapter, FrozenCNNv2PPOAdapter
from real_environment.scenario_manifest import validate_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPOSITORY_ROOT / "experiments/csce723_maneuver_eval"
SMITH_SOURCE = REPOSITORY_ROOT / "try_code" / "MARLSSA-main"
if str(SMITH_SOURCE) not in sys.path:
    sys.path.insert(0, str(SMITH_SOURCE))

from Environments.SSAmulti_v2 import SSAmultiv2


SMITH_HISTORY_FIELDS = (
    "proptime",
    "meancovall",
    "mediancovall",
    "meancovFOR",
    "mediancovFOR",
    "rew",
    "nsats",
    "alt",
    "azi",
)

ESTIMATION_ERROR_SAMPLE_INTERVAL_SECONDS = 60.0


def empty_result(
    policy_name: str,
    condition: str,
    run_id: str,
    configuration=None,
    measurement_noise: MeasurementNoiseConfiguration | None = None,
) -> dict:
    measurement_noise = measurement_noise or MeasurementNoiseConfiguration()
    result = {
        "schema_version": 3,
        "run_id": run_id,
        "pair_ids": [],
        "policy_name": policy_name,
        "condition": condition,
        "scenario_manifest_paths": [],
        "immutable_input_manifest": "config/immutable_inputs.yaml",
        "maneuvering_fraction": (
            None if configuration is None else configuration.maneuvering_fraction
        ),
        "maneuver_plan_summary": (
            None if configuration is None else configuration.summary()
        ),
        "measurement_noise": measurement_noise.to_dict(),
        "delta_v_rtn_mps": (
            None
            if configuration is None
            else list(getattr(configuration, "delta_v_rtn_mps", ()) or []) or None
        ),
        "burn_time_seconds": getattr(configuration, "burn_time_seconds", None),
        "start_trace_cov_list": np.array([]),
        "end_trace_cov_list": np.array([]),
        "start_trace_cov_by_episode": [],
        "end_trace_cov_by_episode": [],
        "trace_covariance_history_by_rso": [],
        "maneuver_rso_ids": [],
        "maneuver_events": [],
        "angular_separation_history": [],
        "post_maneuver_observation_count": [],
        "first_post_maneuver_observation_time": [],
        "reacquisition_time": [],
        "reacquired": [],
        "truth_fov_events": [],
        "measurement_events": [],
        "estimation_error_history": [],
        "nis_table": [],
        "nis_summary": [],
        "nis_recorded": False,
    }
    for field in SMITH_HISTORY_FIELDS:
        result[field] = []
    return result


def initialize_agent_histories(agent_keys: list[str]) -> dict:
    return {
        field: {agent_key: [] for agent_key in agent_keys}
        for field in SMITH_HISTORY_FIELDS
    }


def append_agent_event(
    histories: dict,
    environment,
    agent_key: str,
    reward: dict,
    policy_name: str,
) -> None:
    histories["proptime"][agent_key].append(float(environment.time_count[agent_key]))
    histories["meancovall"][agent_key].append(
        float(environment.get_mean_trace_all_post()[agent_key])
    )
    histories["mediancovall"][agent_key].append(
        float(environment.get_median_trace_all_post()[agent_key])
    )
    histories["meancovFOR"][agent_key].append(
        float(environment.get_mean_trace_FOR_post()[agent_key])
    )
    histories["mediancovFOR"][agent_key].append(
        float(environment.get_median_trace_FOR_post()[agent_key])
    )
    histories["rew"][agent_key].append(float(reward[agent_key]))
    histories["nsats"][agent_key].append(int(environment.get_unique_cum_RSO(agent_key)))
    altitude_value = (
        environment.get_alt_action()[agent_key]
        if policy_name == "advanced_greedy"
        else environment.past_point_alt[agent_key]
    )
    histories["alt"][agent_key].append(float(altitude_value))
    histories["azi"][agent_key].append(float(environment.past_point_azi[agent_key]))


def maneuver_arrays(metrics: dict, number_of_rsos: int) -> tuple[np.ndarray, ...]:
    counts = np.zeros(number_of_rsos, dtype=int)
    first = np.full(number_of_rsos, np.nan)
    reacquisition = np.full(number_of_rsos, np.nan)
    reacquired = np.zeros(number_of_rsos, dtype=bool)
    for raw_rso_id, count in metrics["post_maneuver_observation_count"].items():
        rso_id = int(raw_rso_id)
        counts[rso_id] = count
        first_value = metrics["first_post_maneuver_observation_seconds"][raw_rso_id]
        reacquisition_value = metrics["reacquisition_time_seconds"][raw_rso_id]
        if first_value is not None:
            first[rso_id] = first_value
        if reacquisition_value is not None:
            reacquisition[rso_id] = reacquisition_value
        reacquired[rso_id] = metrics["reacquired"][raw_rso_id]
    return counts, first, reacquisition, reacquired


def compact_event_table(events: list[dict], fields: dict[str, object]) -> dict:
    return {
        field: np.asarray([event[field] for event in events], dtype=dtype)
        for field, dtype in fields.items()
    }


def run_episode(
    policy,
    seed: int,
    configuration,
    measurement_noise: MeasurementNoiseConfiguration | None = None,
    record_nis: bool = False,
) -> dict:
    smith_environment = SSAmultiv2(env_config={"scenario_config": "3_agents_eval"})
    environment = ExternalManeuverEnvironment(
        smith_environment, configuration, measurement_noise, record_nis
    )
    environment.seed(seed)
    environment.set_numsat(100)
    observations = environment.reset()
    agent_keys = list(environment.keylist)
    histories = initialize_agent_histories(agent_keys)
    start_covariance = np.asarray(environment.get_RSO_trace_cov(), dtype=float).copy()
    covariance_times = [0.0]
    covariance_values = [start_covariance.copy()]
    estimation_error_snapshots = [environment.estimation_error_snapshot()]

    done = {"__all__": False}
    while not done["__all__"]:
        actions = policy.actions(observations)
        observations, reward, done, _ = environment.step(actions)
        for agent_key in observations:
            append_agent_event(histories, environment, agent_key, reward, policy.name)
            time_seconds = float(environment.time_count[agent_key])
            trace_covariance = np.asarray(environment.get_RSO_trace_cov(), dtype=float)
            covariance_times.append(time_seconds)
            covariance_values.append(trace_covariance.copy())
        current_global_time = float(
            (environment.time_global - environment.time_ini) * 86400.0
        )
        if (
            current_global_time
            - float(estimation_error_snapshots[-1]["time_seconds"])
            >= ESTIMATION_ERROR_SAMPLE_INTERVAL_SECONDS
        ):
            estimation_error_snapshots.append(environment.estimation_error_snapshot())

    end_covariance = np.asarray(environment.get_RSO_trace_cov(), dtype=float).copy()
    final_error_snapshot = environment.estimation_error_snapshot()
    if not np.isclose(
        float(estimation_error_snapshots[-1]["time_seconds"]),
        float(final_error_snapshot["time_seconds"]),
    ):
        estimation_error_snapshots.append(final_error_snapshot)
    metrics = environment.maneuver_metrics()
    counts, first, reacquisition, reacquired = maneuver_arrays(metrics, 100)
    truth_fov_table = compact_event_table(
        metrics["truth_fov_events"],
        {
            "time_seconds": np.float32,
            "agent_id": "U8",
            "rso_id": np.int16,
            "truth_in_fov": bool,
            "selected_by_smith_estimate": bool,
            "truth_azimuth_deg": np.float32,
            "truth_altitude_deg": np.float32,
            "angular_separation_deg": np.float32,
        },
    )
    measurement_table = compact_event_table(
        metrics["measurement_events"],
        {
            "time_seconds": np.float32,
            "agent_id": "U8",
            "rso_id": np.int16,
            "selected_by_smith_estimate": bool,
            "truth_in_fov": bool,
            "measurement_received": bool,
        },
    )
    angular_history = {
        "rso_id": truth_fov_table["rso_id"].copy(),
        "time_seconds": truth_fov_table["time_seconds"].copy(),
        "separation_degrees": truth_fov_table["angular_separation_deg"].copy(),
    }
    maneuver_events = [
        {
            "rso_id": int(event["rso_id"]),
            "time_seconds": float(event["time_seconds"]),
            "frame": "RTN",
            "delta_v_mps": list(event["delta_v_mps"]),
            "delta_v_magnitude_mps": float(
                np.linalg.norm(np.asarray(event["delta_v_mps"], dtype=float))
            ),
        }
        for event in metrics["maneuver_plan_events"]
    ]
    estimation_error_history = {
        "time_seconds": np.asarray(
            [snapshot["time_seconds"] for snapshot in estimation_error_snapshots],
            dtype=np.float32,
        ),
        "rso_id": np.asarray(metrics["maneuver_rso_ids"], dtype=np.int16),
        "estimate_truth_error_km": np.asarray(
            [
                snapshot["estimate_truth_error_km"]
                for snapshot in estimation_error_snapshots
            ],
            dtype=np.float32,
        ),
        "nominal_truth_offset_km": np.asarray(
            [
                snapshot["nominal_truth_offset_km"]
                for snapshot in estimation_error_snapshots
            ],
            dtype=np.float32,
        ),
        "estimate_nominal_error_km": np.asarray(
            [
                snapshot["estimate_nominal_error_km"]
                for snapshot in estimation_error_snapshots
            ],
            dtype=np.float32,
        ),
    }
    return {
        "histories": histories,
        "start_covariance": start_covariance,
        "end_covariance": end_covariance,
        "covariance_history": {
            "time_seconds": np.asarray(covariance_times, dtype=np.float32),
            "trace_covariance": np.asarray(covariance_values, dtype=np.float32),
        },
        "maneuver_rso_ids": metrics["maneuver_rso_ids"],
        "maneuver_events": maneuver_events,
        "angular_separation_history": angular_history,
        "post_maneuver_observation_count": counts,
        "first_post_maneuver_observation_time": first,
        "reacquisition_time": reacquisition,
        "reacquired": reacquired,
        "truth_fov_events": truth_fov_table,
        "measurement_events": measurement_table,
        "estimation_error_history": estimation_error_history,
        "nis_table": metrics["nis_table"],
        "nis_summary": metrics["nis_summary"],
    }


def append_episode(
    result: dict,
    episode: dict,
    seed: int,
    manifest_path: Path | None = None,
) -> None:
    result["pair_ids"].append(f"seed_{seed:03d}")
    result["scenario_manifest_paths"].append(
        None
        if manifest_path is None
        else str(manifest_path.resolve().relative_to(REPOSITORY_ROOT))
    )
    for field in SMITH_HISTORY_FIELDS:
        result[field].append(episode["histories"][field])
    result["start_trace_cov_list"] = episode["start_covariance"]
    result["end_trace_cov_list"] = episode["end_covariance"]
    result["start_trace_cov_by_episode"].append(episode["start_covariance"])
    result["end_trace_cov_by_episode"].append(episode["end_covariance"])
    result["trace_covariance_history_by_rso"].append(episode["covariance_history"])
    for field in (
        "maneuver_rso_ids",
        "maneuver_events",
        "angular_separation_history",
        "post_maneuver_observation_count",
        "first_post_maneuver_observation_time",
        "reacquisition_time",
        "reacquired",
        "truth_fov_events",
        "measurement_events",
        "estimation_error_history",
        "nis_table",
        "nis_summary",
    ):
        result[field].append(episode[field])


def configuration_from_manifest(
    manifest_path: Path, condition: str
) -> tuple[dict, ManeuverConfiguration]:
    schema_path = PROJECT_ROOT / "config/scenario_manifest.schema.yaml"
    manifest = validate_manifest(manifest_path, schema_path, require_runnable=True)
    events = manifest["maneuver_plan"]["events"]
    if condition == "coast":
        return manifest, ExplicitManeuverConfiguration(plan=plan_from_manifest_events([]))

    first_event = events[0]
    shared_burn_time = all(
        event["time_seconds"] == first_event["time_seconds"] for event in events
    )
    shared_delta_v = all(
        event["delta_v_mps"] == first_event["delta_v_mps"] for event in events
    )
    if shared_burn_time and shared_delta_v:
        # Keep the original fixed-plan object so previously generated manifests produce a
        # configuration that compares equal across a batch exactly as they used to.
        configuration = ManeuverConfiguration(
            maneuvering_fraction=len(events) / manifest["scenario"]["rsos"],
            delta_v_rtn_mps=tuple(float(value) for value in first_event["delta_v_mps"]),
            burn_time_seconds=float(first_event["time_seconds"]),
        )
    else:
        configuration = ExplicitManeuverConfiguration(
            plan=plan_from_manifest_events(events)
        )
    return manifest, configuration


def validate_episode_against_manifest(
    episode: dict, manifest: dict, condition: str
) -> None:
    snapshot = manifest["initial_state_snapshot"]
    snapshot_path = REPOSITORY_ROOT / snapshot["path"]
    import hashlib

    digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if digest != snapshot["sha256"]:
        raise AssertionError("Initial-state snapshot SHA256 does not match manifest")
    with np.load(snapshot_path) as stored:
        expected_start_trace = np.trace(stored["covariance"], axis1=0, axis2=1)
    np.testing.assert_allclose(
        episode["start_covariance"], expected_start_trace, rtol=0.0, atol=0.0
    )
    expected_ids = (
        []
        if condition == "coast"
        else sorted(event["rso_id"] for event in manifest["maneuver_plan"]["events"])
    )
    if episode["maneuver_rso_ids"] != expected_ids:
        raise AssertionError(
            f"Maneuvering RSO IDs differ from manifest: {episode['maneuver_rso_ids']} != {expected_ids}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("advanced_greedy", "frozen_cnnv2_ppo"), required=True)
    parser.add_argument("--condition", choices=("coast", "maneuver"), required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--maneuvering-fraction", type=float, default=0.10)
    parser.add_argument("--delta-v-t-mps", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--sigma-km",
        type=float,
        default=0.0,
        help="Per-axis measurement noise. 10.0 matches Smith's assumed R; 0.0 is stock Smith.",
    )
    parser.add_argument("--noise-seed", type=int, default=20260823)
    parser.add_argument(
        "--random-maneuvers",
        action="store_true",
        help="Draw an independent RTN direction, magnitude and burn time per tagged RSO.",
    )
    parser.add_argument("--delta-v-min-mps", type=float, default=1.0)
    parser.add_argument("--delta-v-max-mps", type=float, default=100.0)
    parser.add_argument("--burn-window-start-seconds", type=float, default=600.0)
    parser.add_argument("--burn-window-end-seconds", type=float, default=4800.0)
    parser.add_argument("--plan-seed", type=int, default=20260823)
    parser.add_argument(
        "--delta-v-scale",
        type=float,
        default=1.0,
        help="Scales every sampled magnitude. 0.0 gives the matched null-burn control.",
    )
    parser.add_argument(
        "--burn-duration-seconds",
        type=float,
        default=0.0,
        help="0 is an impulse. Positive spreads the same delta-v over that many seconds.",
    )
    parser.add_argument(
        "--direction-mode",
        choices=("isotropic", "radial", "transverse", "normal"),
        default="isotropic",
        help="RTN direction of the burn. Magnitudes and burn times are unchanged across modes.",
    )
    parser.add_argument(
        "--record-nis",
        action="store_true",
        help="Record normalized innovation squared from Smith's own UKF update.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing output file, skipping seeds it already holds.",
    )
    args = parser.parse_args()

    config_path = REPOSITORY_ROOT / "experiments/csce723_maneuver_eval/config/experiment.yaml"
    policy = (
        AdvancedGreedyAdapter()
        if args.policy == "advanced_greedy"
        else FrozenCNNv2PPOAdapter(config_path)
    )
    measurement_noise = MeasurementNoiseConfiguration(
        sigma_km=args.sigma_km, seed=args.noise_seed
    )
    manifest = None
    if args.manifest is not None:
        manifest, configuration = configuration_from_manifest(
            args.manifest, args.condition
        )
        seeds = [int(manifest["base_seed"])]
    else:
        fraction = 0.0 if args.condition == "coast" else args.maneuvering_fraction
        if args.random_maneuvers:
            configuration = RandomManeuverConfiguration(
                maneuvering_fraction=fraction,
                magnitude_range_mps=(args.delta_v_min_mps, args.delta_v_max_mps),
                burn_time_window_seconds=(
                    args.burn_window_start_seconds,
                    args.burn_window_end_seconds,
                ),
                plan_seed=args.plan_seed,
                magnitude_scale=args.delta_v_scale,
                direction_mode=args.direction_mode,
                burn_duration_seconds=args.burn_duration_seconds,
            )
        else:
            configuration = ManeuverConfiguration(
                maneuvering_fraction=fraction,
                delta_v_rtn_mps=(0.0, args.delta_v_t_mps, 0.0),
            )
        seeds = list(range(args.seeds))
    if args.output is None:
        args.output = (
            REPOSITORY_ROOT
            / "experiments/csce723_maneuver_eval/results/pilot"
            / f"{policy.name}_{args.condition}.p"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.resume and args.output.exists():
        with args.output.open("rb") as stream:
            result = pickle.load(stream)
        if result["policy_name"] != policy.name or result["condition"] != args.condition:
            raise ValueError("Existing checkpoint policy/condition does not match request")
    else:
        result = empty_result(
            policy.name, args.condition, run_id, configuration, measurement_noise
        )
        result["nis_recorded"] = bool(args.record_nis)
    completed_pair_ids = set(result["pair_ids"])

    for seed in seeds:
        if f"seed_{seed:03d}" in completed_pair_ids:
            print(f"skip completed seed={seed}", flush=True)
            continue
        print(f"policy={policy.name} condition={args.condition} seed={seed}", flush=True)
        episode = run_episode(
            policy, seed, configuration, measurement_noise, args.record_nis
        )
        if manifest is not None:
            validate_episode_against_manifest(episode, manifest, args.condition)
        append_episode(result, episode, seed, args.manifest)
        # Checkpoint every episode so a long grid run survives an interruption.
        temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary_path.open("wb") as stream:
            pickle.dump(result, stream)
        temporary_path.replace(args.output)
    summary = {
        "output": str(args.output),
        "policy": policy.name,
        "condition": args.condition,
        "episodes": len(seeds),
        "training_performed": False,
        "maneuvering_fraction": configuration.maneuvering_fraction,
        "maneuver_plan": configuration.summary(),
        "measurement_noise": measurement_noise.to_dict(),
        "nis_recorded": bool(args.record_nis),
        "nis_summary_last_episode": (
            result["nis_summary"][-1] if result["nis_summary"] else None
        ),
        "mean_final_trace_covariance": float(
            np.mean(np.asarray(result["end_trace_cov_by_episode"]))
        ),
        "maneuvering_rsos": [len(ids) for ids in result["maneuver_rso_ids"]],
        "reacquired_maneuvering_rsos": [
            int(np.sum(values)) for values in result["reacquired"]
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
