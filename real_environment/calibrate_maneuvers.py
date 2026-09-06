"""Calibrate maneuver magnitudes against Smith's 90-minute evaluation geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import ephem
import numpy as np

from real_environment.maneuver_truth import create_truth_trajectory
from real_environment.truth_geometry import ObserverSite, observer_position_teme_km


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SMITH_SOURCE = REPOSITORY_ROOT / "try_code" / "MARLSSA-main"
if str(SMITH_SOURCE) not in sys.path:
    sys.path.insert(0, str(SMITH_SOURCE))

from Environments.SSAMultiUtils import ephem_initial_random


INITIAL_TIME = ephem.Date("2020/01/01 19:00:00.00")
SENSOR_SITES = (
    ObserverSite("agent_0", -106.6599, 33.8172, 1250.0),
    ObserverSite("agent_1", -71.488247, 42.623311, 131.0),
    ObserverSite("agent_2", 114.1656095, -21.815978, 5.0),
)


def angle_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    unit_a = vector_a / np.linalg.norm(vector_a)
    unit_b = vector_b / np.linalg.norm(vector_b)
    return math.degrees(math.acos(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0)))


def run_calibration(
    seeds: int,
    magnitudes_mps: list[float],
    burn_seconds: float,
    episode_seconds: float,
    sample_step_seconds: float,
) -> list[dict]:
    rows: list[dict] = []
    sample_times = np.arange(
        burn_seconds,
        episode_seconds + 0.5 * sample_step_seconds,
        sample_step_seconds,
    )
    for seed in range(seeds):
        np.random.seed(seed)
        truth_objects, _, _ = ephem_initial_random(1, INITIAL_TIME)
        nominal = create_truth_trajectory(truth_objects[0])
        burn_time = ephem.Date(INITIAL_TIME + burn_seconds * ephem.second)
        for magnitude_mps in magnitudes_mps:
            maneuver = create_truth_trajectory(
                truth_objects[0], burn_time, np.array([0.0, magnitude_mps, 0.0])
            )
            for elapsed_seconds in sample_times:
                time = ephem.Date(INITIAL_TIME + elapsed_seconds * ephem.second)
                nominal_position = nominal.position_km(time)
                maneuver_position = maneuver.position_km(time)
                displacement = maneuver_position - nominal_position
                base = {
                    "seed": seed,
                    "delta_v_t_mps": magnitude_mps,
                    "elapsed_seconds": float(elapsed_seconds),
                    "seconds_since_burn": float(elapsed_seconds - burn_seconds),
                    "position_separation_km": float(np.linalg.norm(displacement)),
                    "geocentric_angle_deg": angle_degrees(
                        nominal_position, maneuver_position
                    ),
                }
                for site in SENSOR_SITES:
                    observer_position = observer_position_teme_km(site, time)
                    row = dict(base)
                    row["sensor"] = site.name
                    row["topocentric_angle_deg"] = angle_degrees(
                        nominal_position - observer_position,
                        maneuver_position - observer_position,
                    )
                    rows.append(row)
    return rows


def summarize(rows: list[dict], episode_seconds: float) -> list[dict]:
    summary: list[dict] = []
    magnitudes = sorted({row["delta_v_t_mps"] for row in rows})
    for magnitude in magnitudes:
        final_rows = [
            row
            for row in rows
            if row["delta_v_t_mps"] == magnitude
            and row["elapsed_seconds"] == episode_seconds
        ]
        topocentric = np.array([row["topocentric_angle_deg"] for row in final_rows])
        displacement = np.array([row["position_separation_km"] for row in final_rows])
        summary.append(
            {
                "delta_v_t_mps": magnitude,
                "final_position_separation_km_median": float(np.median(displacement)),
                "final_position_separation_km_p95": float(np.percentile(displacement, 95)),
                "final_topocentric_angle_deg_median": float(np.median(topocentric)),
                "final_topocentric_angle_deg_p95": float(np.percentile(topocentric, 95)),
                "fraction_sensor_views_above_0_25_deg": float(np.mean(topocentric >= 0.25)),
                "fraction_sensor_views_above_0_5_deg": float(np.mean(topocentric >= 0.5)),
                "fraction_sensor_views_above_1_deg": float(np.mean(topocentric >= 1.0)),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[1, 5, 10, 25, 50, 100])
    parser.add_argument("--burn-seconds", type=float, default=900.0)
    parser.add_argument("--episode-seconds", type=float, default=5400.0)
    parser.add_argument("--sample-step-seconds", type=float, default=300.0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPOSITORY_ROOT / "experiments/csce723_maneuver_eval/results/calibration",
    )
    args = parser.parse_args()

    rows = run_calibration(
        args.seeds,
        args.magnitudes,
        args.burn_seconds,
        args.episode_seconds,
        args.sample_step_seconds,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_directory / "maneuver_observability.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "method": "Smith SGP4 nominal truth plus differential two-body Cartesian maneuver perturbation",
        "seeds": args.seeds,
        "burn_seconds": args.burn_seconds,
        "episode_seconds": args.episode_seconds,
        "sample_step_seconds": args.sample_step_seconds,
        "frame": "RTN",
        "direction": "along_track",
        "summary": summarize(rows, args.episode_seconds),
    }
    json_path = args.output_directory / "maneuver_observability_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
