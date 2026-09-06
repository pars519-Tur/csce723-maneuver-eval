"""Load Smith's frozen CNNv2 policy without modifying or running his environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "try_code" / "MARLSSA-main").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the thesis-wiki repository root")


def load_config(config_path: Path) -> tuple[Path, dict]:
    repository_root = find_repository_root(config_path.resolve())
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return repository_root, config


def load_frozen_policy(repository_root: Path, config: dict):
    source_root = repository_root / config["smith_inputs"]["source_root"]
    policy_directory = repository_root / config["smith_inputs"]["policy_directory"]

    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    from CNN_v2 import CNNv2
    from ray.rllib.models import ModelCatalog
    from ray.rllib.policy.policy import Policy

    ModelCatalog.register_custom_model("keras_model", CNNv2)
    return Policy.from_checkpoint(str(policy_directory))


def synthetic_valid_observation(shape: tuple[int, ...]) -> np.ndarray:
    observation = np.zeros(shape, dtype=np.float64)
    observation[90:180, :, 0] = 1.0
    observation[90, 0, 10] = 1.0
    return observation


def deterministic_action(policy, observation: np.ndarray) -> tuple[int, np.ndarray]:
    _, _, info = policy.compute_actions(observation[np.newaxis, ...], explore=False)
    logits = np.asarray(info["action_dist_inputs"])
    return int(np.argmax(logits[0])), logits


def run_smoke_test(config_path: Path) -> dict:
    repository_root, config = load_config(config_path)
    policy = load_frozen_policy(repository_root, config)

    configured_shape = tuple(config["policy"]["observation_shape"])
    observation = synthetic_valid_observation(configured_shape)
    action_1, logits_1 = deterministic_action(policy, observation)
    action_2, logits_2 = deterministic_action(policy, observation)

    expected_action_count = int(config["policy"]["action_count"])
    assert tuple(policy.observation_space.shape) == configured_shape
    assert int(policy.action_space.n) == expected_action_count
    assert logits_1.shape == (1, expected_action_count)
    assert 0 <= action_1 < expected_action_count
    assert action_1 == action_2
    np.testing.assert_allclose(logits_1, logits_2, rtol=0.0, atol=0.0)

    return {
        "status": "passed",
        "training_performed": False,
        "environment_rollout_performed": False,
        "policy_class": type(policy).__name__,
        "observation_shape": list(policy.observation_space.shape),
        "action_count": int(policy.action_space.n),
        "logits_shape": list(logits_1.shape),
        "deterministic_action": action_1,
        "repeated_action_matches": action_1 == action_2,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "experiment.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "checkpoint_smoke_test.json",
    )
    args = parser.parse_args()

    summary = run_smoke_test(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
