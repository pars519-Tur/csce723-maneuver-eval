"""External maneuver wrapper around Smith's unchanged SSAmultiv2 environment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Iterable

import ephem
import numpy as np

from real_environment.maneuver_plan import (
    ExplicitManeuverConfiguration,
    ManeuverConfiguration,
    ManeuverEvent,
    ManeuverPlan,
    RandomManeuverConfiguration,
    plan_from_manifest_events,
    select_maneuvering_rso_ids,
)
from real_environment.maneuver_truth import (
    NominalTruthTrajectory,
    TruthTrajectory,
    create_truth_trajectory,
)
from real_environment.innovation_statistics import NISRecorder, make_update_patch
from real_environment.measurement_noise import (
    MeasurementNoiseConfiguration,
    NoisyMeasurementAdapter,
    rso_noise_generator,
)
from real_environment.truth_geometry import ObserverSite, smith_fov_mask, topocentric_az_alt_deg

__all__ = [
    "CartesianEphemAdapter",
    "ExplicitManeuverConfiguration",
    "ExternalManeuverEnvironment",
    "ManeuverConfiguration",
    "ManeuverEvent",
    "ManeuverPlan",
    "MeasurementNoiseConfiguration",
    "NISRecorder",
    "RandomManeuverConfiguration",
    "filter_maneuver_measurement_indices",
    "plan_from_manifest_events",
    "select_maneuvering_rso_ids",
]


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "try_code" / "MARLSSA-main").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the thesis-wiki repository root")


SMITH_SOURCE = find_repository_root(Path(__file__).resolve()) / "try_code" / "MARLSSA-main"
if str(SMITH_SOURCE) not in sys.path:
    sys.path.insert(0, str(SMITH_SOURCE))

import Environments.SSAmulti_v2 as smith_environment_module
import Environments.SSAMultiUtils as smith_utilities_module
from Environments.SSAMultiUtils import meas as smith_meas, measfun as smith_measfun


def angular_separation_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    unit_a = vector_a / np.linalg.norm(vector_a)
    unit_b = vector_b / np.linalg.norm(vector_b)
    return math.degrees(math.acos(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0)))


class CartesianEphemAdapter:
    """Expose an external Cartesian trajectory to Smith's unchanged measfun_gt API."""

    def __init__(self, trajectory: TruthTrajectory):
        self.trajectory = trajectory
        self.g_ra = 0.0
        self.g_dec = 0.0
        self.elevation = 0.0

    def compute(self, time: ephem.Date) -> None:
        position = self.trajectory.position_km(ephem.Date(time))
        radius_km = float(np.linalg.norm(position))
        self.g_ra = math.atan2(position[1], position[0]) % (2.0 * math.pi)
        self.g_dec = math.asin(np.clip(position[2] / radius_km, -1.0, 1.0))
        self.elevation = (radius_km - 6378.16) * 1000.0


def filter_maneuver_measurement_indices(
    stock_indices: Iterable[int],
    maneuvering_ids: set[int],
    truth_fov_mask_by_rso: dict[int, bool],
) -> np.ndarray:
    """Keep Smith selections, but reject maneuvering objects physically outside truth FOV."""
    return np.asarray(
        [
            int(rso_id)
            for rso_id in stock_indices
            if int(rso_id) not in maneuvering_ids
            or truth_fov_mask_by_rso.get(int(rso_id), False)
        ],
        dtype=int,
    )


class ExternalManeuverEnvironment:
    """Composition wrapper that leaves Smith files and policy observations unchanged."""

    def __init__(
        self,
        smith_environment,
        configuration,
        measurement_noise: MeasurementNoiseConfiguration | None = None,
        record_nis: bool = False,
    ):
        self._environment = smith_environment
        self.configuration = configuration
        self.measurement_noise = measurement_noise or MeasurementNoiseConfiguration()
        self.nis_recorder = NISRecorder(enabled=record_nis)
        self._base_seed = 0
        self.plan = ManeuverPlan()
        self.maneuvering_rso_ids = np.array([], dtype=int)
        self.truth_trajectories: list[TruthTrajectory] = []
        self.measurement_truth_objects: list[object] = []
        self.truth_fov_events: list[dict] = []
        self.measurement_events: list[dict] = []
        self.first_post_maneuver_observation_seconds: dict[int, float | None] = {}
        self.post_maneuver_observation_count: dict[int, int] = {}

    def __getattr__(self, name):
        return getattr(self._environment, name)

    def seed(self, seed=None):
        if seed is not None:
            self._base_seed = int(seed)
        return self._environment.seed(seed)

    def reset(self):
        observation = self._environment.reset()
        self.nis_recorder.reset()
        self.plan = self.configuration.build_plan(
            self._environment.numsat, self._base_seed
        )
        self.maneuvering_rso_ids = self.plan.rso_ids
        maneuvering_set = set(self.maneuvering_rso_ids.tolist())
        self.truth_trajectories = []
        self.measurement_truth_objects = []
        for rso_id, truth_object in enumerate(self._environment.tle_gt):
            if rso_id in maneuvering_set:
                event = self.plan.event_for(rso_id)
                trajectory = create_truth_trajectory(
                    truth_object,
                    ephem.Date(
                        self._environment.time_ini + event.time_seconds * ephem.second
                    ),
                    np.asarray(event.delta_v_rtn_mps, dtype=float),
                    burn_duration_seconds=event.burn_duration_seconds,
                )
                measurement_object = CartesianEphemAdapter(trajectory)
            else:
                trajectory = NominalTruthTrajectory(truth_object)
                measurement_object = truth_object
            if self.measurement_noise.enabled:
                # Every RSO is perturbed, not only the tagged cohort, so the maneuver
                # signal is not confounded with a noise-only difference between groups.
                measurement_object = NoisyMeasurementAdapter(
                    trajectory,
                    self.measurement_noise.sigma_km,
                    rso_noise_generator(
                        self.measurement_noise, self._base_seed, rso_id
                    ),
                )
            self.truth_trajectories.append(trajectory)
            self.measurement_truth_objects.append(measurement_object)

        self.truth_fov_events = []
        self.measurement_events = []
        self.first_post_maneuver_observation_seconds = {
            int(rso_id): None for rso_id in self.maneuvering_rso_ids
        }
        self.post_maneuver_observation_count = {
            int(rso_id): 0 for rso_id in self.maneuvering_rso_ids
        }
        return observation

    def _next_completed_agent_keys(self, action_dict: dict) -> list[str]:
        completion_times = dict(self._environment.obsv_completion_time)
        for agent_key, raw_action in action_dict.items():
            action = int(raw_action) % 1710
            azimuth_action = action // 19
            altitude_action = action % 19
            delta_max = max(
                abs(self._environment.past_point_alt[agent_key] - altitude_action),
                abs(44 - azimuth_action),
            )
            delta_time = 9 + max(0, delta_max - 1) * 4.55
            completion_times[agent_key] = (
                self._environment.time_count[agent_key] + delta_time
            )

        running_keys = [
            key
            for key, done in self._environment.done.items()
            if key != "__all__" and done is False
        ]
        if not running_keys:
            return []
        next_time = min(completion_times[key] for key in running_keys)
        return [
            key
            for key in self._environment.obsv_completion_time
            if key in running_keys and completion_times[key] == next_time
        ]

    @staticmethod
    def _observer_site(agent_key: str, observer) -> ObserverSite:
        return ObserverSite(
            name=agent_key,
            longitude_deg=math.degrees(float(observer.lon)),
            latitude_deg=math.degrees(float(observer.lat)),
            elevation_m=float(observer.elevation),
        )

    def _truth_gate(
        self,
        agent_key: str,
        action: int,
        time_update: ephem.Date,
        stock_indices: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, bool]]:
        azimuth_action = int(action) // 19
        altitude_action = int(action) % 19
        past_azimuth = float(self._environment.past_point_azi[agent_key])
        site = self._observer_site(agent_key, self._environment.agent[agent_key])
        stock_set = {int(value) for value in np.asarray(stock_indices).reshape(-1)}
        truth_masks: dict[int, bool] = {}

        for rso_id in self.maneuvering_rso_ids:
            rso_id = int(rso_id)
            truth_position = self.truth_trajectories[rso_id].position_km(time_update)
            truth_azimuth, truth_altitude = topocentric_az_alt_deg(
                truth_position, site, time_update
            )
            in_truth_fov = bool(
                smith_fov_mask(
                    np.array([truth_azimuth]),
                    np.array([truth_altitude]),
                    past_azimuth,
                    azimuth_action,
                    altitude_action,
                )[0]
            )
            truth_masks[rso_id] = in_truth_fov
            nominal_position = self._environment_position(rso_id, time_update)
            observer_position = self._observer_position(site, time_update)
            event = {
                "agent_id": agent_key,
                "rso_id": rso_id,
                "time_seconds": float(
                    (ephem.Date(time_update) - self._environment.time_ini) * 86400.0
                ),
                "truth_in_fov": in_truth_fov,
                "selected_by_smith_estimate": rso_id in stock_set,
                "truth_azimuth_deg": truth_azimuth,
                "truth_altitude_deg": truth_altitude,
                "angular_separation_deg": angular_separation_deg(
                    nominal_position - observer_position,
                    truth_position - observer_position,
                ),
            }
            self.truth_fov_events.append(event)

        return (
            filter_maneuver_measurement_indices(
                stock_indices, set(self.maneuvering_rso_ids.tolist()), truth_masks
            ),
            truth_masks,
        )

    def _environment_position(self, rso_id: int, time: ephem.Date) -> np.ndarray:
        return NominalTruthTrajectory(self._environment.tle_gt[rso_id]).position_km(time)

    def estimation_error_snapshot(self) -> dict[str, np.ndarray | float]:
        """Compare Smith's filter mean with nominal and maneuvered Cartesian truth.

        Smith stores each UKF mean as TLE-like orbital elements at ``time_ini``.
        Reusing Smith's own ``measfun`` converts that mean to the same TEME Cartesian
        measurement space used by its UKF without changing the filter or policy.
        """
        time_update = ephem.Date(self._environment.time_global)
        state = np.asarray(self._environment.state, dtype=float)
        orbital_elements = state[: 6 * self._environment.numsat].reshape(
            6, 1, self._environment.numsat
        )
        rso_ids = np.asarray(self.maneuvering_rso_ids, dtype=np.int16)
        estimate_truth_error_km = np.empty(len(rso_ids), dtype=np.float32)
        nominal_truth_offset_km = np.empty(len(rso_ids), dtype=np.float32)
        estimate_nominal_error_km = np.empty(len(rso_ids), dtype=np.float32)

        for output_index, raw_rso_id in enumerate(rso_ids):
            rso_id = int(raw_rso_id)
            estimated_position = np.asarray(
                smith_measfun(
                    orbital_elements[:, 0, rso_id],
                    self._environment.time_ini,
                    time_update,
                ),
                dtype=float,
            ).reshape(3)
            nominal_position = self._environment_position(rso_id, time_update)
            truth_position = self.truth_trajectories[rso_id].position_km(time_update)
            estimate_truth_error_km[output_index] = np.linalg.norm(
                estimated_position - truth_position
            )
            nominal_truth_offset_km[output_index] = np.linalg.norm(
                nominal_position - truth_position
            )
            estimate_nominal_error_km[output_index] = np.linalg.norm(
                estimated_position - nominal_position
            )

        return {
            "time_seconds": float(
                (time_update - self._environment.time_ini) * 86400.0
            ),
            "rso_id": rso_ids,
            "estimate_truth_error_km": estimate_truth_error_km,
            "nominal_truth_offset_km": nominal_truth_offset_km,
            "estimate_nominal_error_km": estimate_nominal_error_km,
        }

    @staticmethod
    def _observer_position(site: ObserverSite, time: ephem.Date) -> np.ndarray:
        from real_environment.truth_geometry import observer_position_teme_km

        return observer_position_teme_km(site, time)

    def _record_measurements(
        self,
        agent_key: str,
        time_update: ephem.Date,
        stock_indices: np.ndarray,
        filtered_indices: np.ndarray,
        truth_masks: dict[int, bool],
    ) -> None:
        stock_set = {int(value) for value in np.asarray(stock_indices).reshape(-1)}
        filtered_set = {int(value) for value in np.asarray(filtered_indices).reshape(-1)}
        time_seconds = float(
            (ephem.Date(time_update) - self._environment.time_ini) * 86400.0
        )
        for rso_id in self.maneuvering_rso_ids:
            rso_id = int(rso_id)
            selected = rso_id in stock_set
            received = rso_id in filtered_set
            if selected or truth_masks.get(rso_id, False):
                self.measurement_events.append(
                    {
                        "agent_id": agent_key,
                        "rso_id": rso_id,
                        "time_seconds": time_seconds,
                        "selected_by_smith_estimate": selected,
                        "truth_in_fov": truth_masks.get(rso_id, False),
                        "measurement_received": received,
                    }
                )
            if received and time_seconds >= self.plan.burn_time_for(rso_id):
                self.post_maneuver_observation_count[rso_id] += 1
                if self.first_post_maneuver_observation_seconds[rso_id] is None:
                    self.first_post_maneuver_observation_seconds[rso_id] = time_seconds

    def step(self, action_dict: dict):
        completed_agent_keys = self._next_completed_agent_keys(action_dict)
        original_meas = smith_environment_module.meas
        call_index = 0

        def maneuver_meas(
            dim,
            z,
            p,
            number_of_satellites,
            measurement_noise,
            stock_indices,
            _truth_objects,
            time_initial,
            time_update,
            detection_probability,
            process_noise,
        ):
            nonlocal call_index
            if call_index >= len(completed_agent_keys):
                raise RuntimeError("Unexpected Smith measurement-call count")
            agent_key = completed_agent_keys[call_index]
            call_index += 1
            action = int(self._environment.next_action[agent_key]) % 1710
            stock_indices_array = np.asarray(stock_indices, dtype=int).reshape(-1)
            filtered_indices, truth_masks = self._truth_gate(
                agent_key, action, time_update, stock_indices_array
            )
            self._record_measurements(
                agent_key,
                time_update,
                stock_indices_array,
                filtered_indices,
                truth_masks,
            )
            if self.nis_recorder.enabled:
                # Mirror Smith's own detection test so the recorder sees exactly the RSOs
                # his meas loop will call update for, in the same order.
                detection = self._environment.RSO_pd
                self.nis_recorder.begin_measurement_batch(
                    agent_key,
                    float(
                        (ephem.Date(time_update) - self._environment.time_ini) * 86400.0
                    ),
                    [
                        int(rso_id)
                        for rso_id in filtered_indices
                        if detection[int(rso_id)] > 0
                    ],
                    set(self.maneuvering_rso_ids.tolist()),
                    {
                        int(event.rso_id): float(event.time_seconds)
                        for event in self.plan.events
                    },
                )
            return smith_meas(
                dim,
                z,
                p,
                number_of_satellites,
                measurement_noise,
                filtered_indices,
                self.measurement_truth_objects,
                time_initial,
                time_update,
                detection_probability,
                process_noise,
            )

        smith_environment_module.meas = maneuver_meas
        original_update = smith_utilities_module.update
        if self.nis_recorder.enabled:
            smith_utilities_module.update = make_update_patch(
                original_update, self.nis_recorder
            )
        try:
            result = self._environment.step(action_dict)
        finally:
            smith_environment_module.meas = original_meas
            smith_utilities_module.update = original_update
        if call_index != len(completed_agent_keys):
            raise RuntimeError(
                f"Expected {len(completed_agent_keys)} measurement calls, observed {call_index}"
            )
        return result

    def maneuver_metrics(self) -> dict:
        first_observation = self.first_post_maneuver_observation_seconds
        return {
            "maneuver_rso_ids": self.maneuvering_rso_ids.tolist(),
            "maneuver_plan_events": self.plan.to_dicts(),
            "measurement_noise": self.measurement_noise.to_dict(),
            "nis_table": self.nis_recorder.to_table(),
            "nis_summary": self.nis_recorder.summary(),
            "first_post_maneuver_observation_seconds": first_observation,
            "reacquisition_time_seconds": {
                rso_id: (
                    None if time is None else time - self.plan.burn_time_for(rso_id)
                )
                for rso_id, time in first_observation.items()
            },
            "post_maneuver_observation_count": dict(
                self.post_maneuver_observation_count
            ),
            "reacquired": {
                rso_id: time is not None for rso_id, time in first_observation.items()
            },
            "truth_fov_events": list(self.truth_fov_events),
            "measurement_events": list(self.measurement_events),
        }
