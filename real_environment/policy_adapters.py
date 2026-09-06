"""Common deterministic action interface for Smith PPO and advanced-greedy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from real_environment.smith_policy_adapter import load_config, load_frozen_policy


class FrozenCNNv2PPOAdapter:
    name = "frozen_cnnv2_ppo"

    def __init__(self, config_path: Path):
        repository_root, config = load_config(config_path)
        self.policy = load_frozen_policy(repository_root, config)

    def actions(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        actions = {}
        for agent_key, observation in observations.items():
            _, _, info = self.policy.compute_actions(
                np.asarray(observation)[np.newaxis, ...], explore=False
            )
            logits = np.asarray(info["action_dist_inputs"])
            actions[agent_key] = int(np.argmax(logits[0]))
        return actions


@dataclass
class AdvancedGreedyAdapter:
    name: str = "advanced_greedy"
    slew_penalties: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        penalties = np.ones((19, 90, 19), dtype=float)
        for current_altitude_index in range(19):
            for azimuth_index in range(90):
                for altitude_index in range(19):
                    delta_max = max(
                        abs(current_altitude_index - altitude_index),
                        abs(44 - azimuth_index),
                    )
                    delta_time = 9 + max(0, delta_max - 1) * 4.55
                    penalties[
                        current_altitude_index, azimuth_index, altitude_index
                    ] = delta_time ** (1 / 5)
        self.slew_penalties = penalties

    def _action(self, observation: np.ndarray) -> int:
        local_observation = np.asarray(observation)[90:180, :, :]
        current_pointing = np.where(
            local_observation[:, :, 10] == np.amax(local_observation[:, :, 10])
        )
        current_altitude_index = int(current_pointing[1][0])

        score = (
            local_observation[:, :, 9]
            / self.slew_penalties[current_altitude_index]
        )
        candidates = np.where(score == np.amax(score))
        selected_index = 0
        if len(candidates[0]) > 1:
            distance = (candidates[0] - current_pointing[0][0]) ** 2 + (
                candidates[1] - current_pointing[1][0]
            ) ** 2
            selected_index = int(np.argmin(distance))
        azimuth_action = int(candidates[0][selected_index])
        altitude_action = int(candidates[1][selected_index])
        return azimuth_action * 19 + altitude_action

    def actions(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        return {
            agent_key: self._action(observation)
            for agent_key, observation in observations.items()
        }
