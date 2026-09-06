"""Per-RSO impulsive maneuver plans, fixed or randomly sampled.

The original evaluation applied one shared along-track delta-v at one shared burn time to
every tagged RSO. That isolates a single variable but cannot separate observability
effects that depend on burn direction or on how much of the episode remains after the
burn. This module keeps the fixed plan available and adds a randomised plan whose events
carry an independent direction, magnitude and burn time per RSO.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


EPISODE_LENGTH_SECONDS = 5400.0
DEFAULT_SELECTION_SEED_OFFSET = 104729

# RTN axes. A radial or normal burn perturbs the orbit periodically, while a transverse burn
# changes the semi-major axis and so drifts secularly, which is why the three are worth
# separating rather than averaging under one isotropic draw.
DIRECTION_MODES: dict[str, np.ndarray | None] = {
    "isotropic": None,
    "radial": np.array([1.0, 0.0, 0.0]),
    "transverse": np.array([0.0, 1.0, 0.0]),
    "normal": np.array([0.0, 0.0, 1.0]),
}


def select_maneuvering_rso_ids(
    number_of_rsos: int,
    maneuvering_fraction: float,
    base_seed: int,
    selection_seed_offset: int = DEFAULT_SELECTION_SEED_OFFSET,
) -> np.ndarray:
    """Choose the tagged cohort exactly as the fixed-plan evaluation always has.

    Keeping this draw unchanged means a randomised run tags the same RSOs as the existing
    fixed-delta-v runs at the same fraction and seed, so the two remain comparable.
    """
    count = int(round(number_of_rsos * maneuvering_fraction))
    if count == 0:
        return np.array([], dtype=int)
    generator = np.random.RandomState(base_seed + selection_seed_offset)
    return np.sort(generator.choice(number_of_rsos, size=count, replace=False)).astype(int)


@dataclass(frozen=True)
class ManeuverEvent:
    """One impulsive RTN burn applied to Smith's truth trajectory."""

    rso_id: int
    time_seconds: float
    delta_v_rtn_mps: tuple[float, float, float]
    burn_duration_seconds: float = 0.0
    """0 is an impulse. A positive value spreads the same delta-v over that many seconds."""

    @property
    def magnitude_mps(self) -> float:
        return float(np.linalg.norm(np.asarray(self.delta_v_rtn_mps, dtype=float)))

    def to_dict(self) -> dict:
        """Return the manifest-schema representation of this event.

        The duration is only emitted when it is non-zero, so impulsive plans keep producing
        exactly the manifest shape they always did.
        """
        record = {
            "rso_id": int(self.rso_id),
            "time_seconds": float(self.time_seconds),
            "delta_v_mps": [float(value) for value in self.delta_v_rtn_mps],
        }
        if self.burn_duration_seconds > 0.0:
            record["burn_duration_seconds"] = float(self.burn_duration_seconds)
        return record


@dataclass(frozen=True)
class ManeuverPlan:
    """The full set of burns for one episode, indexed by RSO."""

    events: tuple[ManeuverEvent, ...] = ()

    def __post_init__(self) -> None:
        rso_ids = [event.rso_id for event in self.events]
        if len(set(rso_ids)) != len(rso_ids):
            raise ValueError("A maneuver plan may hold at most one event per RSO")

    @property
    def rso_ids(self) -> np.ndarray:
        return np.array(sorted(event.rso_id for event in self.events), dtype=int)

    def event_for(self, rso_id: int) -> ManeuverEvent:
        for event in self.events:
            if event.rso_id == int(rso_id):
                return event
        raise KeyError(f"RSO {rso_id} does not maneuver in this plan")

    def burn_time_for(self, rso_id: int) -> float:
        return self.event_for(rso_id).time_seconds

    def delta_v_for(self, rso_id: int) -> np.ndarray:
        return np.asarray(self.event_for(rso_id).delta_v_rtn_mps, dtype=float)

    def to_dicts(self) -> list[dict]:
        return [event.to_dict() for event in self.events]


def plan_from_manifest_events(events: list[dict]) -> ManeuverPlan:
    """Build a plan from manifest ``maneuver_plan.events`` entries."""
    return ManeuverPlan(
        events=tuple(
            ManeuverEvent(
                rso_id=int(event["rso_id"]),
                time_seconds=float(event["time_seconds"]),
                delta_v_rtn_mps=tuple(float(value) for value in event["delta_v_mps"]),
                burn_duration_seconds=float(event.get("burn_duration_seconds", 0.0)),
            )
            for event in events
        )
    )


def _validate_fraction(maneuvering_fraction: float) -> None:
    if not 0.0 <= maneuvering_fraction <= 1.0:
        raise ValueError("maneuvering_fraction must lie in [0, 1]")


def _validate_burn_time(burn_time_seconds: float) -> None:
    if not 0.0 < burn_time_seconds < EPISODE_LENGTH_SECONDS:
        raise ValueError("burn_time_seconds must lie inside the episode")


@dataclass(frozen=True)
class ManeuverConfiguration:
    """One shared along-track burn applied to every tagged RSO at one shared time."""

    maneuvering_fraction: float
    delta_v_rtn_mps: tuple[float, float, float] = (0.0, 10.0, 0.0)
    burn_time_seconds: float = 900.0
    selection_seed_offset: int = DEFAULT_SELECTION_SEED_OFFSET

    def __post_init__(self) -> None:
        _validate_fraction(self.maneuvering_fraction)
        _validate_burn_time(self.burn_time_seconds)

    def build_plan(self, number_of_rsos: int, base_seed: int) -> ManeuverPlan:
        rso_ids = select_maneuvering_rso_ids(
            number_of_rsos,
            self.maneuvering_fraction,
            base_seed,
            self.selection_seed_offset,
        )
        return ManeuverPlan(
            events=tuple(
                ManeuverEvent(
                    rso_id=int(rso_id),
                    time_seconds=float(self.burn_time_seconds),
                    delta_v_rtn_mps=tuple(
                        float(value) for value in self.delta_v_rtn_mps
                    ),
                )
                for rso_id in rso_ids
            )
        )

    def summary(self) -> dict:
        return {
            "plan_type": "fixed",
            "maneuvering_fraction": float(self.maneuvering_fraction),
            "delta_v_rtn_mps": [float(value) for value in self.delta_v_rtn_mps],
            "burn_time_seconds": float(self.burn_time_seconds),
        }


@dataclass(frozen=True)
class RandomManeuverConfiguration:
    """Independent direction, magnitude and burn time for each tagged RSO.

    The direction is drawn isotropically in the RTN frame, the magnitude log-uniformly so
    that every decade of delta-v is sampled evenly, and the burn time uniformly inside a
    window that leaves enough of the episode for the offset to grow and be observed.
    """

    maneuvering_fraction: float
    magnitude_range_mps: tuple[float, float] = (1.0, 100.0)
    burn_time_window_seconds: tuple[float, float] = (600.0, 4800.0)
    plan_seed: int = 20260823
    selection_seed_offset: int = DEFAULT_SELECTION_SEED_OFFSET
    magnitude_scale: float = 1.0
    burn_duration_seconds: float = 0.0
    direction_mode: str = "isotropic"
    """Which RTN direction the burn takes.

    "isotropic" draws uniformly on the sphere. "radial", "transverse" and "normal" pin the burn
    to one axis, keeping a random sign taken from the isotropic draw's own component on that
    axis. The isotropic draw is consumed either way, so every mode gives the same cohort, the
    same magnitudes and the same burn times, and the arms differ only in direction.
    """
    """Multiplies every sampled magnitude after it is drawn.

    Setting this to 0 produces the matched null-burn control. The random draws are consumed
    in the same order and quantity, so the control tags the same RSOs at the same burn times
    with the same burn directions, and differs from the treatment only in that the burn has
    no magnitude. A control built instead from a fixed burn time would give the tagged objects
    a different post-burn observation window and confound the comparison.
    """

    def __post_init__(self) -> None:
        _validate_fraction(self.maneuvering_fraction)
        if self.magnitude_scale < 0.0:
            raise ValueError("magnitude_scale must be non-negative")
        if self.burn_duration_seconds < 0.0:
            raise ValueError("burn_duration_seconds must be non-negative")
        if self.direction_mode not in DIRECTION_MODES:
            raise ValueError(
                f"direction_mode must be one of {sorted(DIRECTION_MODES)}"
            )
        low_magnitude, high_magnitude = self.magnitude_range_mps
        if not 0.0 < low_magnitude <= high_magnitude:
            raise ValueError("magnitude_range_mps must satisfy 0 < low <= high")
        low_time, high_time = self.burn_time_window_seconds
        if not low_time <= high_time:
            raise ValueError("burn_time_window_seconds must satisfy low <= high")
        _validate_burn_time(low_time)
        _validate_burn_time(high_time)

    def build_plan(self, number_of_rsos: int, base_seed: int) -> ManeuverPlan:
        rso_ids = select_maneuvering_rso_ids(
            number_of_rsos,
            self.maneuvering_fraction,
            base_seed,
            self.selection_seed_offset,
        )
        generator = np.random.default_rng([int(self.plan_seed), int(base_seed)])
        low_magnitude, high_magnitude = self.magnitude_range_mps
        low_time, high_time = self.burn_time_window_seconds
        events = []
        for rso_id in rso_ids:
            direction = generator.normal(size=3)
            norm = float(np.linalg.norm(direction))
            while norm == 0.0:
                direction = generator.normal(size=3)
                norm = float(np.linalg.norm(direction))
            direction = direction / norm
            axis = DIRECTION_MODES[self.direction_mode]
            if axis is not None:
                # Keep the sign the isotropic draw gave this axis, so both senses are sampled.
                sign = 1.0 if float(np.dot(direction, axis)) >= 0.0 else -1.0
                direction = sign * axis
            magnitude = math.exp(
                generator.uniform(math.log(low_magnitude), math.log(high_magnitude))
            )
            burn_time = float(generator.uniform(low_time, high_time))
            velocity_change = direction * magnitude * self.magnitude_scale
            events.append(
                ManeuverEvent(
                    rso_id=int(rso_id),
                    time_seconds=burn_time,
                    delta_v_rtn_mps=tuple(float(value) for value in velocity_change),
                    burn_duration_seconds=float(self.burn_duration_seconds),
                )
            )
        return ManeuverPlan(events=tuple(events))

    def summary(self) -> dict:
        return {
            "plan_type": "random",
            "maneuvering_fraction": float(self.maneuvering_fraction),
            "magnitude_range_mps": [float(value) for value in self.magnitude_range_mps],
            "magnitude_distribution": "log_uniform",
            "direction_distribution": self.direction_mode,
            "burn_time_window_seconds": [
                float(value) for value in self.burn_time_window_seconds
            ],
            "burn_time_distribution": "uniform",
            "plan_seed": int(self.plan_seed),
            "magnitude_scale": float(self.magnitude_scale),
            "burn_duration_seconds": float(self.burn_duration_seconds),
            "burn_model": "impulsive" if self.burn_duration_seconds == 0.0 else "finite",
            "role": "null_burn_control" if self.magnitude_scale == 0.0 else "treatment",
        }


@dataclass(frozen=True)
class ExplicitManeuverConfiguration:
    """A plan fixed ahead of time, used when a manifest already lists every event."""

    plan: ManeuverPlan

    @property
    def maneuvering_fraction(self) -> float:
        return float(len(self.plan.events)) / 100.0

    def build_plan(self, number_of_rsos: int, base_seed: int) -> ManeuverPlan:
        return self.plan

    def summary(self) -> dict:
        magnitudes = [event.magnitude_mps for event in self.plan.events]
        times = [event.time_seconds for event in self.plan.events]
        return {
            "plan_type": "explicit",
            "maneuvering_fraction": self.maneuvering_fraction,
            "event_count": len(self.plan.events),
            "magnitude_range_mps": (
                [min(magnitudes), max(magnitudes)] if magnitudes else None
            ),
            "burn_time_range_seconds": ([min(times), max(times)] if times else None),
        }
