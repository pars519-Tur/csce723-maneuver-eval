"""External impulsive truth maneuvers for Smith-compatible PyEphem trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import sys
from typing import Protocol

import ephem
import numpy as np
from scipy.integrate import solve_ivp


MU_EARTH_KM3_S2 = 398600.4415
SECONDS_PER_DAY = 86400.0


class TruthTrajectory(Protocol):
    """Minimal external truth interface used by the maneuver evaluator."""

    def position_km(self, time: ephem.Date) -> np.ndarray: ...

    def state_km_s(self, time: ephem.Date) -> tuple[np.ndarray, np.ndarray]: ...


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "try_code" / "MARLSSA-main").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the thesis-wiki repository root")


def _smith_utilities():
    source_root = find_repository_root(Path(__file__).resolve()) / "try_code" / "MARLSSA-main"
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from Environments.SSAMultiUtils import create_ephem, measfun_gt

    return create_ephem, measfun_gt


def position_teme_km(satellite, time: ephem.Date) -> np.ndarray:
    """Return the Smith measurement-frame position in km."""
    _, measfun_gt = _smith_utilities()
    return np.asarray(measfun_gt(satellite, ephem.Date(time)), dtype=float).reshape(3)


def state_teme_km_s(
    satellite,
    time: ephem.Date,
    finite_difference_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the PyEphem truth state with a centered position difference."""
    if finite_difference_seconds <= 0:
        raise ValueError("finite_difference_seconds must be positive")

    time = ephem.Date(time)
    offset = finite_difference_seconds * ephem.second
    position_before = position_teme_km(satellite, ephem.Date(time - offset))
    position_after = position_teme_km(satellite, ephem.Date(time + offset))
    position = position_teme_km(satellite, time)
    velocity = (position_after - position_before) / (2.0 * finite_difference_seconds)
    return position, velocity


def rtn_basis(position_km: np.ndarray, velocity_km_s: np.ndarray) -> np.ndarray:
    """Return a 3x3 matrix whose columns are radial, transverse, and normal unit vectors."""
    position = np.asarray(position_km, dtype=float).reshape(3)
    velocity = np.asarray(velocity_km_s, dtype=float).reshape(3)

    radial = position / np.linalg.norm(position)
    angular_momentum = np.cross(position, velocity)
    normal = angular_momentum / np.linalg.norm(angular_momentum)
    transverse = np.cross(normal, radial)
    transverse /= np.linalg.norm(transverse)
    return np.column_stack((radial, transverse, normal))


def apply_rtn_impulse(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    delta_v_rtn_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply an instantaneous RTN-frame velocity change and return the ECI delta-v in km/s."""
    position = np.asarray(position_km, dtype=float).reshape(3)
    velocity = np.asarray(velocity_km_s, dtype=float).reshape(3)
    delta_v_rtn_km_s = np.asarray(delta_v_rtn_mps, dtype=float).reshape(3) / 1000.0
    delta_v_teme_km_s = rtn_basis(position, velocity) @ delta_v_rtn_km_s
    return position.copy(), velocity + delta_v_teme_km_s, delta_v_teme_km_s


@dataclass
class NominalTruthTrajectory:
    """Read-only adapter around Smith's unchanged PyEphem truth object."""

    satellite: object
    finite_difference_seconds: float = 1.0

    def position_km(self, time: ephem.Date) -> np.ndarray:
        return position_teme_km(self.satellite, ephem.Date(time))

    def state_km_s(self, time: ephem.Date) -> tuple[np.ndarray, np.ndarray]:
        return state_teme_km_s(
            self.satellite,
            ephem.Date(time),
            finite_difference_seconds=self.finite_difference_seconds,
        )


@dataclass
class ImpulsiveCartesianTruthTrajectory:
    """Smith truth plus a deterministic differential two-body maneuver perturbation.

    Two Cartesian trajectories start at the same Smith burn state: one coasts and one
    receives the impulse. Their state difference is added to Smith's nominal SGP4
    trajectory. Thus delta-v -> 0 returns exactly to Smith, while no post-burn state is
    converted back to a TLE.
    """

    nominal: NominalTruthTrajectory
    burn_time: ephem.Date
    burn_position_km: np.ndarray
    pre_burn_velocity_km_s: np.ndarray
    post_burn_velocity_km_s: np.ndarray
    delta_v_rtn_mps: np.ndarray
    delta_v_teme_km_s: np.ndarray
    rtol: float = 1e-11
    atol: float = 1e-12
    dense_horizon_seconds: float = 5400.0
    _state_cache: dict[float, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )
    _dense_solutions: dict[str, object] = field(
        default_factory=dict, init=False, repr=False
    )

    def _seconds_after_burn(self, time: ephem.Date) -> float:
        return float((ephem.Date(time) - self.burn_time) * SECONDS_PER_DAY)

    @staticmethod
    def _two_body_dynamics(_time_seconds: float, state: np.ndarray) -> np.ndarray:
        position = state[:3]
        radius = np.linalg.norm(position)
        acceleration = -MU_EARTH_KM3_S2 * position / radius**3
        return np.concatenate((state[3:], acceleration))

    def _initial_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        """Starting velocities for the coast and maneuver legs of the differential pair."""
        return self.pre_burn_velocity_km_s, self.post_burn_velocity_km_s

    def _dynamics_for(self, label: str):
        """RHS for one leg. Subclasses add thrust here rather than editing the integrator."""
        return self._two_body_dynamics

    def _integrated_state(
        self, initial_velocity_km_s: np.ndarray, elapsed_seconds: float, label: str
    ) -> np.ndarray:
        initial_state = np.concatenate((self.burn_position_km, initial_velocity_km_s))
        if elapsed_seconds == 0.0:
            return initial_state
        horizon = max(self.dense_horizon_seconds, float(elapsed_seconds))
        cached = self._dense_solutions.get(label)
        if cached is None or float(cached.t[-1]) < elapsed_seconds:
            solution = solve_ivp(
                self._dynamics_for(label),
                (0.0, horizon),
                initial_state,
                method="DOP853",
                dense_output=True,
                rtol=self.rtol,
                atol=self.atol,
            )
            if not solution.success:
                raise RuntimeError(
                    f"Post-burn truth propagation failed: {solution.message}"
                )
            self._dense_solutions[label] = solution
            cached = solution
        return np.asarray(cached.sol(float(elapsed_seconds)), dtype=float).reshape(6)

    def _post_burn_state(
        self, time: ephem.Date, elapsed_seconds: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if elapsed_seconds < 0.0:
            raise ValueError("Post-burn propagation requires a non-negative elapsed time")
        cache_key = round(float(elapsed_seconds), 9)
        if cache_key in self._state_cache:
            position, velocity = self._state_cache[cache_key]
            return position.copy(), velocity.copy()
        nominal_position, nominal_velocity = self.nominal.state_km_s(time)
        coast_velocity, maneuver_velocity = self._initial_velocities()
        coast_state = self._integrated_state(coast_velocity, elapsed_seconds, "coast")
        maneuver_state = self._integrated_state(
            maneuver_velocity, elapsed_seconds, "maneuver"
        )
        differential_state = maneuver_state - coast_state
        state = (
            nominal_position + differential_state[:3],
            nominal_velocity + differential_state[3:],
        )
        self._state_cache[cache_key] = (state[0].copy(), state[1].copy())
        return state

    def position_km(self, time: ephem.Date) -> np.ndarray:
        elapsed_seconds = self._seconds_after_burn(time)
        if elapsed_seconds < 0.0:
            return self.nominal.position_km(time)
        return self._post_burn_state(time, elapsed_seconds)[0]

    def state_km_s(self, time: ephem.Date) -> tuple[np.ndarray, np.ndarray]:
        elapsed_seconds = self._seconds_after_burn(time)
        if elapsed_seconds < 0.0:
            return self.nominal.state_km_s(time)
        return self._post_burn_state(time, elapsed_seconds)


@dataclass
class FiniteBurnCartesianTruthTrajectory(ImpulsiveCartesianTruthTrajectory):
    """A burn spread over a finite duration rather than applied instantaneously.

    Electric propulsion does not deliver its delta-v in an instant. The same total delta-v
    applied over minutes to hours produces a smaller, later-arriving offset than an impulse, and
    is the harder detection case. Direction is held fixed in the inertial frame at the value the
    impulse would have had, so the two models differ only in how the delta-v is spread in time.

    The burn and coast legs are integrated separately rather than as one run over a
    discontinuous right-hand side, which keeps the solver accurate across thrust cut-off.
    """

    burn_duration_seconds: float = 0.0
    thrust_acceleration_teme_km_s2: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )

    def _initial_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        # Both legs start unburned; the thrust is what separates them.
        return self.pre_burn_velocity_km_s, self.pre_burn_velocity_km_s

    def _thrust_dynamics(self, time_seconds: float, state: np.ndarray) -> np.ndarray:
        derivative = self._two_body_dynamics(time_seconds, state)
        derivative[3:] = derivative[3:] + self.thrust_acceleration_teme_km_s2
        return derivative

    def _integrated_state(
        self, initial_velocity_km_s: np.ndarray, elapsed_seconds: float, label: str
    ) -> np.ndarray:
        if label == "coast" or self.burn_duration_seconds <= 0.0:
            return super()._integrated_state(
                initial_velocity_km_s, elapsed_seconds, label
            )
        if elapsed_seconds == 0.0:
            return np.concatenate((self.burn_position_km, initial_velocity_km_s))

        burn_solution = self._dense_solutions.get("burn")
        if burn_solution is None:
            burn_solution = solve_ivp(
                self._thrust_dynamics,
                (0.0, self.burn_duration_seconds),
                np.concatenate((self.burn_position_km, initial_velocity_km_s)),
                method="DOP853", dense_output=True, rtol=self.rtol, atol=self.atol,
            )
            if not burn_solution.success:
                raise RuntimeError(f"Finite-burn leg failed: {burn_solution.message}")
            self._dense_solutions["burn"] = burn_solution
        if elapsed_seconds <= self.burn_duration_seconds:
            return np.asarray(
                burn_solution.sol(float(elapsed_seconds)), dtype=float
            ).reshape(6)

        horizon = max(self.dense_horizon_seconds, float(elapsed_seconds))
        coast_solution = self._dense_solutions.get("post_burn_coast")
        if coast_solution is None or float(coast_solution.t[-1]) < elapsed_seconds:
            coast_solution = solve_ivp(
                self._two_body_dynamics,
                (self.burn_duration_seconds, horizon),
                np.asarray(
                    burn_solution.sol(self.burn_duration_seconds), dtype=float
                ).reshape(6),
                method="DOP853", dense_output=True, rtol=self.rtol, atol=self.atol,
            )
            if not coast_solution.success:
                raise RuntimeError(f"Post-burn coast failed: {coast_solution.message}")
            self._dense_solutions["post_burn_coast"] = coast_solution
        return np.asarray(
            coast_solution.sol(float(elapsed_seconds)), dtype=float
        ).reshape(6)


def create_truth_trajectory(
    satellite,
    burn_time: ephem.Date | None = None,
    delta_v_rtn_mps: np.ndarray | None = None,
    finite_difference_seconds: float = 1.0,
    burn_duration_seconds: float = 0.0,
) -> TruthTrajectory:
    """Build nominal or maneuvering truth without modifying Smith's truth object."""
    nominal = NominalTruthTrajectory(satellite, finite_difference_seconds)
    if burn_time is None or delta_v_rtn_mps is None:
        return nominal

    delta_v_rtn = np.asarray(delta_v_rtn_mps, dtype=float).reshape(3)
    if np.linalg.norm(delta_v_rtn) == 0.0:
        return nominal

    burn_time = ephem.Date(burn_time)
    position, velocity = nominal.state_km_s(burn_time)
    _, post_velocity, delta_v_teme = apply_rtn_impulse(position, velocity, delta_v_rtn)
    if burn_duration_seconds > 0.0:
        return FiniteBurnCartesianTruthTrajectory(
            nominal=nominal,
            burn_time=burn_time,
            burn_position_km=position,
            pre_burn_velocity_km_s=velocity,
            post_burn_velocity_km_s=post_velocity,
            delta_v_rtn_mps=delta_v_rtn,
            delta_v_teme_km_s=delta_v_teme,
            burn_duration_seconds=float(burn_duration_seconds),
            thrust_acceleration_teme_km_s2=delta_v_teme / float(burn_duration_seconds),
        )
    return ImpulsiveCartesianTruthTrajectory(
        nominal=nominal,
        burn_time=burn_time,
        burn_position_km=position,
        pre_burn_velocity_km_s=velocity,
        post_burn_velocity_km_s=post_velocity,
        delta_v_rtn_mps=delta_v_rtn,
        delta_v_teme_km_s=delta_v_teme,
    )


def _angle_degrees(angle_radians: float) -> float:
    return math.degrees(angle_radians % (2.0 * math.pi))


def rv_to_smith_elements(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
) -> np.ndarray:
    """Convert an elliptic Cartesian state to Smith's [i, RAAN, e, argp, M, n] format."""
    position = np.asarray(position_km, dtype=float).reshape(3)
    velocity = np.asarray(velocity_km_s, dtype=float).reshape(3)
    radius = np.linalg.norm(position)
    speed_squared = float(np.dot(velocity, velocity))

    angular_momentum = np.cross(position, velocity)
    h_norm = np.linalg.norm(angular_momentum)
    node = np.cross(np.array([0.0, 0.0, 1.0]), angular_momentum)
    node_norm = np.linalg.norm(node)
    eccentricity_vector = np.cross(velocity, angular_momentum) / mu_km3_s2 - position / radius
    eccentricity = np.linalg.norm(eccentricity_vector)

    specific_energy = speed_squared / 2.0 - mu_km3_s2 / radius
    if specific_energy >= 0 or eccentricity >= 1:
        raise ValueError("Only bound elliptic post-burn states are supported")
    semi_major_axis = -mu_km3_s2 / (2.0 * specific_energy)

    inclination = math.acos(np.clip(angular_momentum[2] / h_norm, -1.0, 1.0))
    tolerance = 1e-10

    if node_norm > tolerance:
        raan = math.atan2(node[1], node[0]) % (2.0 * math.pi)
    else:
        raan = 0.0

    if eccentricity > tolerance and node_norm > tolerance:
        argument_of_periapsis = math.acos(
            np.clip(np.dot(node, eccentricity_vector) / (node_norm * eccentricity), -1.0, 1.0)
        )
        if eccentricity_vector[2] < 0:
            argument_of_periapsis = 2.0 * math.pi - argument_of_periapsis
    elif eccentricity > tolerance:
        argument_of_periapsis = math.atan2(eccentricity_vector[1], eccentricity_vector[0]) % (
            2.0 * math.pi
        )
    else:
        argument_of_periapsis = 0.0

    if eccentricity > tolerance:
        true_anomaly = math.acos(
            np.clip(np.dot(eccentricity_vector, position) / (eccentricity * radius), -1.0, 1.0)
        )
        if np.dot(position, velocity) < 0:
            true_anomaly = 2.0 * math.pi - true_anomaly
    elif node_norm > tolerance:
        true_anomaly = math.acos(np.clip(np.dot(node, position) / (node_norm * radius), -1.0, 1.0))
        if position[2] < 0:
            true_anomaly = 2.0 * math.pi - true_anomaly
    else:
        true_anomaly = math.atan2(position[1], position[0]) % (2.0 * math.pi)

    eccentric_anomaly = math.atan2(
        math.sqrt(1.0 - eccentricity**2) * math.sin(true_anomaly),
        eccentricity + math.cos(true_anomaly),
    ) % (2.0 * math.pi)
    mean_anomaly = (eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly)) % (
        2.0 * math.pi
    )
    mean_motion_revolutions_per_day = (
        math.sqrt(mu_km3_s2 / semi_major_axis**3) * SECONDS_PER_DAY / (2.0 * math.pi)
    )

    return np.array(
        [
            math.degrees(inclination),
            _angle_degrees(raan),
            eccentricity,
            _angle_degrees(argument_of_periapsis),
            _angle_degrees(mean_anomaly),
            mean_motion_revolutions_per_day,
        ],
        dtype=float,
    )


def smith_elements_to_rv(
    elements: np.ndarray,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Smith elements to a two-body Cartesian state for mathematical validation."""
    inclination_deg, raan_deg, eccentricity, argp_deg, mean_anomaly_deg, mean_motion = (
        np.asarray(elements, dtype=float).reshape(6)
    )
    inclination = math.radians(inclination_deg)
    raan = math.radians(raan_deg)
    argument_of_periapsis = math.radians(argp_deg)
    mean_anomaly = math.radians(mean_anomaly_deg)
    mean_motion_rad_s = mean_motion * 2.0 * math.pi / SECONDS_PER_DAY
    semi_major_axis = (mu_km3_s2 / mean_motion_rad_s**2) ** (1.0 / 3.0)

    eccentric_anomaly = mean_anomaly
    for _ in range(30):
        correction = (
            eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly) - mean_anomaly
        ) / (1.0 - eccentricity * math.cos(eccentric_anomaly))
        eccentric_anomaly -= correction
        if abs(correction) < 1e-13:
            break

    radius_perifocal = np.array(
        [
            semi_major_axis * (math.cos(eccentric_anomaly) - eccentricity),
            semi_major_axis * math.sqrt(1.0 - eccentricity**2) * math.sin(eccentric_anomaly),
            0.0,
        ]
    )
    radius = semi_major_axis * (1.0 - eccentricity * math.cos(eccentric_anomaly))
    velocity_perifocal = (
        math.sqrt(mu_km3_s2 * semi_major_axis) / radius
    ) * np.array(
        [
            -math.sin(eccentric_anomaly),
            math.sqrt(1.0 - eccentricity**2) * math.cos(eccentric_anomaly),
            0.0,
        ]
    )

    cos_raan, sin_raan = math.cos(raan), math.sin(raan)
    cos_inc, sin_inc = math.cos(inclination), math.sin(inclination)
    cos_argp, sin_argp = math.cos(argument_of_periapsis), math.sin(argument_of_periapsis)
    rotation = np.array(
        [
            [
                cos_raan * cos_argp - sin_raan * sin_argp * cos_inc,
                -cos_raan * sin_argp - sin_raan * cos_argp * cos_inc,
                sin_raan * sin_inc,
            ],
            [
                sin_raan * cos_argp + cos_raan * sin_argp * cos_inc,
                -sin_raan * sin_argp + cos_raan * cos_argp * cos_inc,
                -cos_raan * sin_inc,
            ],
            [sin_argp * sin_inc, cos_argp * sin_inc, cos_inc],
        ]
    )
    return rotation @ radius_perifocal, rotation @ velocity_perifocal


@dataclass(frozen=True)
class ManeuverReport:
    burn_time: str
    finite_difference_seconds: float
    delta_v_rtn_mps: list[float]
    delta_v_teme_mps: list[float]
    pre_burn_position_km: list[float]
    pre_burn_velocity_km_s: list[float]
    post_burn_velocity_km_s: list[float]
    post_burn_elements: list[float]
    reconstruction_position_error_km: float
    reconstruction_velocity_error_mps: float
    pyephem_epoch_position_discontinuity_km: float

    def to_dict(self) -> dict:
        return asdict(self)


def create_post_burn_ephem(
    satellite,
    burn_time: ephem.Date,
    delta_v_rtn_mps: np.ndarray,
    finite_difference_seconds: float = 1.0,
):
    """Create a post-burn PyEphem object plus conversion diagnostics."""
    delta_v_rtn = np.asarray(delta_v_rtn_mps, dtype=float).reshape(3)
    position, velocity = state_teme_km_s(satellite, burn_time, finite_difference_seconds)
    post_position, post_velocity, delta_v_teme = apply_rtn_impulse(
        position, velocity, delta_v_rtn
    )
    post_elements = rv_to_smith_elements(post_position, post_velocity)
    reconstructed_position, reconstructed_velocity = smith_elements_to_rv(post_elements)

    if np.linalg.norm(delta_v_rtn) == 0:
        post_burn_satellite = satellite
    else:
        create_ephem, _ = _smith_utilities()
        post_burn_satellite = create_ephem(post_elements, ephem.Date(burn_time))

    epoch_position = position_teme_km(post_burn_satellite, ephem.Date(burn_time))
    report = ManeuverReport(
        burn_time=str(ephem.Date(burn_time)),
        finite_difference_seconds=float(finite_difference_seconds),
        delta_v_rtn_mps=delta_v_rtn.tolist(),
        delta_v_teme_mps=(delta_v_teme * 1000.0).tolist(),
        pre_burn_position_km=position.tolist(),
        pre_burn_velocity_km_s=velocity.tolist(),
        post_burn_velocity_km_s=post_velocity.tolist(),
        post_burn_elements=post_elements.tolist(),
        reconstruction_position_error_km=float(np.linalg.norm(reconstructed_position - post_position)),
        reconstruction_velocity_error_mps=float(
            1000.0 * np.linalg.norm(reconstructed_velocity - post_velocity)
        ),
        pyephem_epoch_position_discontinuity_km=float(np.linalg.norm(epoch_position - position)),
    )
    return post_burn_satellite, report
