"""Normalized innovation squared, recorded from Smith's own UKF update.

The innovation is the only place a measurement value enters Smith's filter. His covariance
update is ``P = P - K S K'`` (``Environments/SSAMultiUtils.py:737``), which depends on the
sigma points and ``R`` but never on the measurement, so the covariance trace that forms the
reward cannot respond to a maneuver. NIS can.

For one measurement Smith forms the innovation ``nu = y - MU`` and the innovation covariance
``S + R``. This module records

    NIS = nu' (S + R)^-1 nu

by temporarily rebinding ``SSAMultiUtils.update`` for the duration of a step, so the numbers
are exactly the ones Smith's filter used. No Smith file is modified.

Under a consistent filter NIS follows a chi-square distribution with three degrees of freedom,
so its mean is 3. That null distribution only means anything once the measurements actually
carry the noise the filter assumes, which is why NIS is recorded together with the injected
measurement noise rather than on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


MEASUREMENT_DIMENSION = 3
# Upper tail of chi-square with three degrees of freedom.
CHI_SQUARE_3_THRESHOLDS = {"p95": 7.814727903251179, "p99": 11.344866730144373}


def _whiten(innovation_covariance: np.ndarray, innovation: np.ndarray) -> np.ndarray:
    """Return L^-1 nu, where L L' = S + R.

    The Cholesky factor is the standard whitening transform. Its squared norm is NIS by
    construction, which makes the whitened vector a check on the NIS value rather than a
    separate quantity. A symmetric eigen decomposition covers the rare case where the
    innovation covariance is not numerically positive definite.
    """
    try:
        factor = np.linalg.cholesky(innovation_covariance)
        return np.linalg.solve(factor, innovation)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(innovation_covariance)
        inverse_root = np.clip(values, 1e-12, None) ** -0.5
        return (vectors * inverse_root) @ (vectors.T @ innovation)


@dataclass
class NISRecord:
    time_seconds: float
    agent_id: str
    rso_id: int
    nis: float
    innovation_norm_km: float
    is_maneuvering: bool
    seconds_after_burn: float
    # The signed innovation and the standard deviation the filter assigns to each of its
    # components. NIS collapses these into one number, which is what a detector wants but
    # loses the sign and the per-axis structure a residual plot needs.
    innovation_km: tuple[float, float, float] = (0.0, 0.0, 0.0)
    innovation_sigma_km: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # The innovation whitened by the full innovation covariance rather than by its diagonal.
    # Under a consistent filter each component is standard normal, and the sum of their squares
    # is exactly NIS. Component-wise division by the diagonal is not equivalent: S + R carries
    # enough correlation that the diagonal version misses the true NIS badly in the tails.
    whitened_innovation: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class NISRecorder:
    """Collects one NIS value per measurement Smith's filter actually processed."""

    enabled: bool = False
    records: list[NISRecord] = field(default_factory=list)
    _pending: list[tuple[str, float, int, bool, float]] = field(
        default_factory=list, init=False, repr=False
    )
    _cursor: int = field(default=0, init=False, repr=False)

    def begin_measurement_batch(
        self,
        agent_id: str,
        time_seconds: float,
        rso_ids: list[int],
        maneuvering_ids: set[int],
        burn_time_by_rso: dict[int, float],
    ) -> None:
        """Declare, in Smith's own update order, which RSOs are about to be updated."""
        self._pending = [
            (
                agent_id,
                time_seconds,
                int(rso_id),
                int(rso_id) in maneuvering_ids,
                (
                    time_seconds - burn_time_by_rso[int(rso_id)]
                    if int(rso_id) in burn_time_by_rso
                    else float("nan")
                ),
            )
            for rso_id in rso_ids
        ]
        self._cursor = 0

    def record_update(
        self,
        measurement_covariance: np.ndarray,
        measurement_noise: np.ndarray,
        measurement: np.ndarray,
        predicted_measurement: np.ndarray,
    ) -> None:
        if not self.enabled or self._cursor >= len(self._pending):
            return
        agent_id, time_seconds, rso_id, is_maneuvering, seconds_after_burn = self._pending[
            self._cursor
        ]
        self._cursor += 1

        predicted = np.asarray(predicted_measurement, dtype=float).reshape(
            MEASUREMENT_DIMENSION, 1
        )
        innovation = (
            np.asarray(measurement, dtype=float).reshape(MEASUREMENT_DIMENSION, 1)
            - predicted
        )
        innovation_covariance = np.asarray(
            measurement_covariance, dtype=float
        ) + np.asarray(measurement_noise, dtype=float)
        try:
            weighted = np.linalg.solve(innovation_covariance, innovation)
        except np.linalg.LinAlgError:
            weighted = np.linalg.pinv(innovation_covariance) @ innovation
        # sqrt of the diagonal, so the band is the standard deviation the filter assigns to
        # that component on its own. NIS uses the full inverse, so a component sitting inside
        # its own band is not by itself evidence that NIS is small.
        component_sigma = np.sqrt(np.clip(np.diag(innovation_covariance), 0.0, None))
        flat_innovation = innovation.reshape(MEASUREMENT_DIMENSION)
        whitened = _whiten(innovation_covariance, flat_innovation)
        self.records.append(
            NISRecord(
                time_seconds=float(time_seconds),
                agent_id=str(agent_id),
                rso_id=int(rso_id),
                nis=float(innovation.T @ weighted),
                innovation_norm_km=float(np.linalg.norm(innovation)),
                is_maneuvering=bool(is_maneuvering),
                seconds_after_burn=float(seconds_after_burn),
                innovation_km=tuple(float(value) for value in flat_innovation),
                innovation_sigma_km=tuple(float(value) for value in component_sigma),
                whitened_innovation=tuple(float(value) for value in whitened),
            )
        )

    def reset(self) -> None:
        self.records = []
        self._pending = []
        self._cursor = 0

    def to_table(self) -> dict[str, np.ndarray]:
        return {
            "time_seconds": np.asarray(
                [record.time_seconds for record in self.records], dtype=np.float32
            ),
            "agent_id": np.asarray(
                [record.agent_id for record in self.records], dtype="U8"
            ),
            "rso_id": np.asarray(
                [record.rso_id for record in self.records], dtype=np.int16
            ),
            "nis": np.asarray([record.nis for record in self.records], dtype=np.float32),
            "innovation_norm_km": np.asarray(
                [record.innovation_norm_km for record in self.records], dtype=np.float32
            ),
            "is_maneuvering": np.asarray(
                [record.is_maneuvering for record in self.records], dtype=bool
            ),
            "seconds_after_burn": np.asarray(
                [record.seconds_after_burn for record in self.records], dtype=np.float32
            ),
            **{
                f"innovation_{axis}_km": np.asarray(
                    [record.innovation_km[index] for record in self.records],
                    dtype=np.float32,
                )
                for index, axis in enumerate("xyz")
            },
            **{
                f"innovation_sigma_{axis}_km": np.asarray(
                    [record.innovation_sigma_km[index] for record in self.records],
                    dtype=np.float32,
                )
                for index, axis in enumerate("xyz")
            },
            **{
                f"whitened_{axis}": np.asarray(
                    [record.whitened_innovation[index] for record in self.records],
                    dtype=np.float32,
                )
                for index, axis in enumerate("xyz")
            },
        }

    def summary(self) -> dict:
        """Consistency and detection summary, split by maneuver status and burn timing."""
        if not self.records:
            return {"measurements": 0}
        values = np.asarray([record.nis for record in self.records], dtype=float)
        maneuvering = np.asarray(
            [record.is_maneuvering for record in self.records], dtype=bool
        )
        after_burn = np.asarray(
            [record.seconds_after_burn for record in self.records], dtype=float
        ) >= 0.0

        def describe(mask: np.ndarray) -> dict:
            selected = values[mask]
            if selected.size == 0:
                return {"measurements": 0}
            return {
                "measurements": int(selected.size),
                "mean_nis": float(np.mean(selected)),
                "median_nis": float(np.median(selected)),
                "fraction_above_chi2_p95": float(
                    np.mean(selected > CHI_SQUARE_3_THRESHOLDS["p95"])
                ),
                "fraction_above_chi2_p99": float(
                    np.mean(selected > CHI_SQUARE_3_THRESHOLDS["p99"])
                ),
            }

        return {
            "measurements": int(values.size),
            "expected_mean_under_consistency": float(MEASUREMENT_DIMENSION),
            "all": describe(np.ones_like(maneuvering)),
            "non_maneuvering": describe(~maneuvering),
            "maneuvering_before_burn": describe(maneuvering & ~after_burn),
            "maneuvering_after_burn": describe(maneuvering & after_burn),
        }


def make_update_patch(original_update, recorder: NISRecorder):
    """Wrap Smith's ``update`` so every call is observed without altering its result."""

    def patched_update(M, P, S, R, C, ymeas, MU):
        recorder.record_update(S, R, ymeas, MU)
        return original_update(M, P, S, R, C, ymeas, MU)

    return patched_update
