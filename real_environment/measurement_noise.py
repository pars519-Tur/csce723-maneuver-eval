"""Additive Cartesian measurement noise for Smith's unchanged UKF measurement path.

Smith's ``measfun_gt`` (``Environments/SSAMultiUtils.py:716``) returns the exact truth
position, while ``SSAmultiv2.step`` (``Environments/SSAmulti_v2.py:306``) hands the UKF
``Rmeas = eye(3) * 10**2``. The filter is therefore told that measurements carry a 10 km
one-sigma error that the released code never generates, which contradicts the thesis text
describing an environment that "generates noisy measurements within the FOV of each
sensor" (Smith 2024, section 2.2).

This module closes that gap from outside Smith's tree. The noise is applied to the truth
object handed to ``meas``, so no Smith file changes and the assumed ``R`` stays untouched.
Setting ``sigma_km`` to 10 makes the UKF statistically consistent with its own ``R``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import ephem
import numpy as np


# Smith's radec2teme reconstructs range as 6378.16 + elevation_m / 1000 (SSAMultiUtils.py:224).
SMITH_RADEC2TEME_RADIUS_OFFSET_KM = 6378.16


@dataclass(frozen=True)
class MeasurementNoiseConfiguration:
    """Zero-mean isotropic Cartesian noise added to every truth measurement.

    ``sigma_km`` is the per-axis one-sigma error in Smith's TEME measurement space.
    ``sigma_km = 0`` reproduces the released Smith behaviour exactly.
    """

    sigma_km: float = 0.0
    seed: int = 20260823

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_km) or self.sigma_km < 0.0:
            raise ValueError("sigma_km must be a finite non-negative value")

    @property
    def enabled(self) -> bool:
        return self.sigma_km > 0.0

    def to_dict(self) -> dict:
        return {"sigma_km": float(self.sigma_km), "seed": int(self.seed)}


def rso_noise_generator(
    configuration: MeasurementNoiseConfiguration, base_seed: int, rso_id: int
) -> np.random.Generator:
    """Return a per-RSO stream that never touches Smith's global ``np.random`` state.

    Smith's ``reset`` draws the initial catalog and covariance from the seeded global
    numpy stream, so consuming from it here would change the initial conditions and break
    the paired coast/maneuver design. ``default_rng`` gives an independent stream.
    """
    return np.random.default_rng(
        [int(configuration.seed), int(base_seed), int(rso_id)]
    )


class NoisyMeasurementAdapter:
    """Expose a truth trajectory to Smith's ``measfun_gt`` with additive Cartesian noise.

    ``measfun_gt`` calls ``compute(time)`` and then reads ``g_dec``, ``g_ra`` and
    ``elevation``. Encoding the perturbed position back into those three fields makes
    Smith's ``radec2teme`` reconstruct it exactly, so the noise reaches the UKF innovation
    without any change to Smith's measurement code.
    """

    def __init__(
        self,
        trajectory,
        sigma_km: float,
        generator: np.random.Generator | None,
    ):
        self._trajectory = trajectory
        self._sigma_km = float(sigma_km)
        self._generator = generator
        if self._sigma_km > 0.0 and generator is None:
            raise ValueError("A generator is required when sigma_km is positive")
        self.g_ra = 0.0
        self.g_dec = 0.0
        self.elevation = 0.0
        self.applied_noise_km: list[tuple[float, np.ndarray]] = []

    def compute(self, time) -> None:
        time = ephem.Date(time)
        position = np.asarray(
            self._trajectory.position_km(time), dtype=float
        ).reshape(3)
        if self._sigma_km > 0.0:
            noise = self._generator.normal(0.0, self._sigma_km, size=3)
            position = position + noise
            self.applied_noise_km.append((float(time), noise))

        radius_km = float(np.linalg.norm(position))
        self.g_ra = math.atan2(position[1], position[0]) % (2.0 * math.pi)
        self.g_dec = math.asin(np.clip(position[2] / radius_km, -1.0, 1.0))
        self.elevation = (radius_km - SMITH_RADEC2TEME_RADIUS_OFFSET_KM) * 1000.0
