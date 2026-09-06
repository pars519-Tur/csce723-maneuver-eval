"""Compare unchanged Smith and the external wrapper with maneuver fraction set to zero."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from real_environment.maneuver_environment import (
    ExternalManeuverEnvironment,
    ManeuverConfiguration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SMITH_SOURCE = REPOSITORY_ROOT / "try_code" / "MARLSSA-main"
if str(SMITH_SOURCE) not in sys.path:
    sys.path.insert(0, str(SMITH_SOURCE))

from Environments.SSAmulti_v2 import SSAmultiv2


def maximum_mapping_difference(first: dict, second: dict) -> float:
    if set(first) != set(second):
        raise AssertionError(f"Mapping keys differ: {set(first)} != {set(second)}")
    differences = []
    for key in first:
        first_value = np.asarray(first[key])
        second_value = np.asarray(second[key])
        if first_value.shape != second_value.shape:
            raise AssertionError(
                f"Shape mismatch for {key}: {first_value.shape} != {second_value.shape}"
            )
        differences.append(float(np.max(np.abs(first_value - second_value))))
    return max(differences, default=0.0)


def run(seed: int = 0, steps: int = 12) -> dict:
    smith = SSAmultiv2(env_config={"scenario_config": "3_agents_eval"})
    wrapped_base = SSAmultiv2(env_config={"scenario_config": "3_agents_eval"})
    wrapped = ExternalManeuverEnvironment(
        wrapped_base,
        ManeuverConfiguration(maneuvering_fraction=0.0),
    )

    smith.seed(seed)
    smith.set_numsat(100)
    smith_observation = smith.reset()
    wrapped.seed(seed)
    wrapped.set_numsat(100)
    wrapped_observation = wrapped.reset()

    max_observation_difference = maximum_mapping_difference(
        smith_observation, wrapped_observation
    )
    max_state_difference = float(np.max(np.abs(smith.state - wrapped.state)))
    max_reward_difference = 0.0
    max_time_difference = 0.0

    completed_steps = 0
    for _ in range(steps):
        action = {agent_key: 44 * 19 + 6 for agent_key in smith_observation}
        if set(action) != set(wrapped_observation):
            raise AssertionError("Active-agent keys diverged before step")
        smith_observation, smith_reward, smith_done, _ = smith.step(action)
        wrapped_observation, wrapped_reward, wrapped_done, _ = wrapped.step(action)
        completed_steps += 1

        max_observation_difference = max(
            max_observation_difference,
            maximum_mapping_difference(smith_observation, wrapped_observation),
        )
        max_state_difference = max(
            max_state_difference,
            float(np.max(np.abs(smith.state - wrapped.state))),
        )
        max_reward_difference = max(
            max_reward_difference,
            maximum_mapping_difference(smith_reward, wrapped_reward),
        )
        max_time_difference = max(
            max_time_difference,
            maximum_mapping_difference(smith.time_count, wrapped.time_count),
        )
        if smith_done != wrapped_done:
            raise AssertionError(f"Done mappings differ: {smith_done} != {wrapped_done}")
        if smith_done["__all__"]:
            break

    payload = {
        "seed": seed,
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "maneuvering_fraction": 0.0,
        "max_observation_absolute_difference": max_observation_difference,
        "max_state_absolute_difference": max_state_difference,
        "max_reward_absolute_difference": max_reward_difference,
        "max_time_absolute_difference_seconds": max_time_difference,
        "maneuver_events_recorded": len(wrapped.measurement_events),
        "passed": bool(
            max_observation_difference == 0.0
            and max_state_difference == 0.0
            and max_reward_difference == 0.0
            and max_time_difference == 0.0
            and len(wrapped.measurement_events) == 0
        ),
    }
    if not payload["passed"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    payload = run()
    output_path = (
        REPOSITORY_ROOT
        / "experiments/csce723_maneuver_eval/results/zero_maneuver_regression.json"
    )
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
