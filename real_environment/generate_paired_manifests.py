"""Generate auditable initial snapshots and runnable paired scenario manifests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import ephem
import numpy as np
import yaml

from real_environment.maneuver_environment import select_maneuvering_rso_ids
from real_environment.maneuver_truth import state_teme_km_s
from real_environment.scenario_manifest import validate_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPOSITORY_ROOT / "experiments/csce723_maneuver_eval"
SMITH_SOURCE = REPOSITORY_ROOT / "try_code/MARLSSA-main"
if str(SMITH_SOURCE) not in sys.path:
    sys.path.insert(0, str(SMITH_SOURCE))

from Environments.SSAmulti_v2 import SSAmultiv2


SCENARIO_SHA256 = "c107665e8da659a1b70d458316a7ec374d84be06f3fdedc171db7ea4bd5ee8ca"
SELECTION_SEED_OFFSET = 104729


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_snapshot(seed: int, output_path: Path) -> str:
    environment = SSAmultiv2(env_config={"scenario_config": "3_agents_eval"})
    environment.seed(seed)
    environment.set_numsat(100)
    environment.reset()
    truth_state = np.zeros((100, 6), dtype=float)
    for rso_id, truth_object in enumerate(environment.tle_gt):
        position, velocity = state_teme_km_s(truth_object, environment.time_ini)
        truth_state[rso_id, :3] = position
        truth_state[rso_id, 3:] = velocity
    covariance = environment.state[600:].reshape(6, 6, 100)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        base_seed=np.array(seed, dtype=int),
        truth_tle_records=np.asarray([item.writedb() for item in environment.tle_gt]),
        truth_cartesian_state_km_km_s=truth_state,
        estimated_tle_records=np.asarray([item.writedb() for item in environment.tle_est]),
        estimated_orbital_elements=np.asarray(environment.oe_list, dtype=float),
        covariance=np.asarray(covariance, dtype=float),
        initial_pointing_altitude_index=np.asarray(
            [environment.past_point_alt[key] for key in environment.keylist], dtype=int
        ),
        initial_pointing_azimuth_deg=np.asarray(
            [environment.past_point_azi[key] for key in environment.keylist], dtype=float
        ),
    )
    return file_sha256(output_path)


def manifest_payload(
    seed: int,
    fraction: float,
    delta_v_t_mps: float,
    snapshot_path: Path,
    snapshot_sha256: str,
) -> dict:
    rso_ids = select_maneuvering_rso_ids(
        100, fraction, seed, SELECTION_SEED_OFFSET
    )
    rate_label = f"rate_{int(round(100 * fraction)):02d}pct"
    delta_label = f"dvT_{delta_v_t_mps:g}mps".replace(".", "p")
    return {
        "schema_version": 1,
        "manifest_id": f"seed_{seed:03d}_{rate_label}_{delta_label}",
        "pair_id": f"seed_{seed:03d}",
        "runnable": True,
        "base_seed": seed,
        "scenario": {
            "source": "try_code/MARLSSA-main/Environments/scenario_configs/3_agents_eval.yaml",
            "source_sha256": SCENARIO_SHA256,
            "sensors": 3,
            "rsos": 100,
            "episode_length_seconds": 5400,
        },
        "initial_state_snapshot": {
            "status": "ready",
            "path": str(snapshot_path.relative_to(REPOSITORY_ROOT)),
            "sha256": snapshot_sha256,
            "contents": [
                "truth_tle_records",
                "truth_cartesian_state",
                "estimated_tle_records",
                "estimated_orbital_elements",
                "covariance",
                "initial_sensor_pointing",
            ],
        },
        "randomness": {
            "smith_environment_seed": seed,
            "detection_model": "deterministic_probability_one",
            "policy_exploration": False,
        },
        "conditions": ["coast", "maneuver"],
        "maneuver_plan": {
            "status": "ready",
            "selection_seed": seed + SELECTION_SEED_OFFSET,
            "frame": "RTN",
            "events": [
                {
                    "rso_id": int(rso_id),
                    "time_seconds": 900.0,
                    "delta_v_mps": [0.0, float(delta_v_t_mps), 0.0],
                }
                for rso_id in rso_ids
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--delta-v-t-mps", type=float, default=10.0)
    parser.add_argument("--label", default="primary")
    args = parser.parse_args()

    snapshot_root = PROJECT_ROOT / "results/snapshots"
    manifest_root = PROJECT_ROOT / "config/manifests" / args.label
    manifest_root.mkdir(parents=True, exist_ok=True)
    schema_path = PROJECT_ROOT / "config/scenario_manifest.schema.yaml"
    generated = []
    created_snapshots = 0
    reused_snapshots = 0
    for seed in range(args.seeds):
        snapshot_path = snapshot_root / f"seed_{seed:03d}.npz"
        if snapshot_path.exists():
            snapshot_sha256 = file_sha256(snapshot_path)
            reused_snapshots += 1
        else:
            snapshot_sha256 = generate_snapshot(seed, snapshot_path)
            created_snapshots += 1
        for fraction in args.fractions:
            payload = manifest_payload(
                seed, fraction, args.delta_v_t_mps, snapshot_path, snapshot_sha256
            )
            manifest_path = manifest_root / f"{payload['manifest_id']}.yaml"
            manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))
            validate_manifest(manifest_path, schema_path, require_runnable=True)
            generated.append(manifest_path)
    print(f"created_snapshots={created_snapshots}")
    print(f"reused_snapshots={reused_snapshots}")
    print(f"generated_manifests={len(generated)}")
    for path in generated:
        print(path.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
