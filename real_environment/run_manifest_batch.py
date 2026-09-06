"""Checkpointed/resumable batch execution over runnable paired manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import pickle

from real_environment.evaluation_runner import (
    PROJECT_ROOT,
    append_episode,
    configuration_from_manifest,
    empty_result,
    run_episode,
    validate_episode_against_manifest,
)
from real_environment.measurement_noise import MeasurementNoiseConfiguration
from real_environment.policy_adapters import AdvancedGreedyAdapter, FrozenCNNv2PPOAdapter


def save_checkpoint(result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as stream:
        pickle.dump(result, stream)
    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("advanced_greedy", "frozen_cnnv2_ppo"), required=True)
    parser.add_argument("--condition", choices=("coast", "maneuver"), required=True)
    parser.add_argument("--manifest-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sigma-km",
        type=float,
        default=0.0,
        help="Per-axis measurement noise. 10.0 matches Smith's assumed R; 0.0 is stock Smith.",
    )
    parser.add_argument("--noise-seed", type=int, default=20260823)
    args = parser.parse_args()

    manifest_paths = sorted(PROJECT_ROOT.glob(args.manifest_glob))
    if args.limit is not None:
        manifest_paths = manifest_paths[: args.limit]
    if not manifest_paths:
        raise FileNotFoundError(f"No manifests match {args.manifest_glob}")

    measurement_noise = MeasurementNoiseConfiguration(
        sigma_km=args.sigma_km, seed=args.noise_seed
    )
    config_path = PROJECT_ROOT / "config/experiment.yaml"
    policy = (
        AdvancedGreedyAdapter()
        if args.policy == "advanced_greedy"
        else FrozenCNNv2PPOAdapter(config_path)
    )
    first_manifest, first_configuration = configuration_from_manifest(
        manifest_paths[0], args.condition
    )
    if args.output.exists():
        with args.output.open("rb") as stream:
            result = pickle.load(stream)
        if result["policy_name"] != policy.name or result["condition"] != args.condition:
            raise ValueError("Existing checkpoint policy/condition does not match request")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result = empty_result(
            policy.name,
            args.condition,
            run_id,
            first_configuration,
            measurement_noise,
        )

    completed_pair_ids = set(result["pair_ids"])
    for index, manifest_path in enumerate(manifest_paths, start=1):
        manifest, configuration = configuration_from_manifest(
            manifest_path, args.condition
        )
        if configuration != first_configuration:
            raise ValueError("All manifests in one batch must share one evaluation configuration")
        pair_id = manifest["pair_id"]
        if pair_id in completed_pair_ids:
            print(f"skip completed {pair_id}", flush=True)
            continue
        seed = int(manifest["base_seed"])
        print(
            f"[{index}/{len(manifest_paths)}] policy={policy.name} condition={args.condition} seed={seed}",
            flush=True,
        )
        episode = run_episode(policy, seed, configuration, measurement_noise)
        validate_episode_against_manifest(episode, manifest, args.condition)
        append_episode(result, episode, seed, manifest_path)
        completed_pair_ids.add(pair_id)
        save_checkpoint(result, args.output)
    print(f"completed_episodes={len(result['pair_ids'])}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
