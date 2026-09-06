"""Summarise the measurement-noise by random-maneuver grid.

The grid crosses measurement noise (0 km, the released Smith behaviour, against 10 km,
which matches the R that Smith's own step function already assumes) with the maneuver
condition (a null burn against an isotropic random burn) for both policies. Every arm tags
the same RSO cohort per seed, so each comparison is paired at the episode level.

Outputs
    cell_summary.csv       one row per policy/noise/maneuver cell
    paired_effects.csv     maneuver and noise effects with bootstrap intervals
    dose_response.csv      per-RSO delta-v against covariance response and tracking error
    action_invariance.json whether a policy repoints when the measurement noise changes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from real_environment.innovation_statistics import CHI_SQUARE_3_THRESHOLDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_DIRECTORY = PROJECT_ROOT / "results/noise_maneuver_grid"
POLICIES = ("advanced_greedy", "frozen_cnnv2_ppo")
NOISE_TAGS = {"sigma00": 0.0, "sigma10": 10.0}
# "nullburn" is the matched control: the treatment's own random plan with zero magnitude,
# so cohort, burn times and burn directions are identical and only the delta-v differs.
# "zerodv" is the earlier fixed-900s control, kept so old runs still analyse.
MANEUVER_TAGS = (
    "nullburn",     # matched control, zero delta-v on the treatment's own plan
    "zerodv",       # superseded fixed-900s control, kept only so old runs still analyse
    "random",       # isotropic impulsive burn
    "radial",       # impulsive, pinned to the radial axis
    "transverse",   # impulsive, pinned to the along-track axis
    "normal",       # impulsive, pinned to the cross-track axis
    "finite1800s",  # isotropic burn spread over 1800 s instead of applied at once
    "stress100to2000",  # isotropic impulsive burn at orbit-transfer scale, 100-2000 m/s
)
CONTROL_TAG = "nullburn"
# Everything the control is contrasted against. "zerodv" is a control, not a treatment.
TREATMENT_TAGS = ("random", "radial", "transverse", "normal", "finite1800s", "stress100to2000")
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260823


def arm_path(policy: str, maneuver: str, noise_tag: str, seeds: int) -> Path:
    return GRID_DIRECTORY / f"{policy}_{maneuver}_{noise_tag}_{seeds}seeds.p"


def load_arm(policy: str, maneuver: str, noise_tag: str, seeds: int) -> dict | None:
    path = arm_path(policy, maneuver, noise_tag, seeds)
    if not path.exists():
        return None
    with path.open("rb") as stream:
        return pickle.load(stream)


def tagged_indices(result: dict, episode: int) -> np.ndarray:
    return np.asarray(result["maneuver_rso_ids"][episode], dtype=int)


def final_errors(result: dict, episode: int) -> dict[str, np.ndarray]:
    history = result["estimation_error_history"][episode]
    return {
        "estimate_truth_error_km": np.asarray(history["estimate_truth_error_km"])[-1],
        "nominal_truth_offset_km": np.asarray(history["nominal_truth_offset_km"])[-1],
        "rso_id": np.asarray(history["rso_id"], dtype=int),
    }


def episode_metrics(result: dict, episode: int) -> dict[str, float]:
    end_trace = np.asarray(result["end_trace_cov_by_episode"][episode], dtype=float)
    tagged = tagged_indices(result, episode)
    errors = final_errors(result, episode)
    rewards = result["rew"][episode]
    reacquired = np.asarray(result["reacquired"][episode], dtype=bool)
    return {
        "final_mean_covariance": float(np.mean(end_trace)),
        "final_median_covariance": float(np.median(end_trace)),
        "final_tagged_covariance": float(np.mean(end_trace[tagged])) if len(tagged) else np.nan,
        "final_tagged_error_median_km": float(
            np.median(errors["estimate_truth_error_km"])
        ),
        "final_tagged_error_p90_km": float(
            np.percentile(errors["estimate_truth_error_km"], 90)
        ),
        "final_tagged_offset_median_km": float(
            np.median(errors["nominal_truth_offset_km"])
        ),
        "mean_reward_per_agent": float(
            np.mean([np.mean(values) for values in rewards.values()])
        ),
        "reacquired_fraction": (
            float(np.sum(reacquired[tagged]) / len(tagged)) if len(tagged) else np.nan
        ),
    }


def arm_frame(result: dict) -> dict[str, np.ndarray]:
    episodes = len(result["pair_ids"])
    rows = [episode_metrics(result, index) for index in range(episodes)]
    return {
        key: np.asarray([row[key] for row in rows], dtype=float) for key in rows[0]
    }


def bootstrap_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    samples = generator.choice(
        values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True
    ).mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                "" if value is None else
                (f"{value:.10g}" if isinstance(value, float) else str(value))
                for value in row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def build_cell_summary(arms: dict, seeds: int) -> list[list]:
    rows = []
    for policy in POLICIES:
        for noise_tag, sigma in NOISE_TAGS.items():
            for maneuver in MANEUVER_TAGS:
                result = arms.get((policy, maneuver, noise_tag))
                if result is None:
                    continue
                frame = arm_frame(result)
                rows.append(
                    [
                        policy,
                        sigma,
                        maneuver,
                        len(result["pair_ids"]),
                        float(np.mean(frame["final_mean_covariance"])),
                        float(np.mean(frame["final_tagged_covariance"])),
                        float(np.mean(frame["final_tagged_error_median_km"])),
                        float(np.mean(frame["final_tagged_error_p90_km"])),
                        float(np.mean(frame["final_tagged_offset_median_km"])),
                        float(np.mean(frame["mean_reward_per_agent"])),
                        float(np.nanmean(frame["reacquired_fraction"])),
                    ]
                )
    return rows


def paired_difference(
    treatment: dict, control: dict, metric: str
) -> tuple[float, float, float]:
    """Difference on the seeds both arms actually completed."""
    treatment_ids = {pair: index for index, pair in enumerate(treatment["pair_ids"])}
    control_ids = {pair: index for index, pair in enumerate(control["pair_ids"])}
    shared = sorted(set(treatment_ids) & set(control_ids))
    if not shared:
        return float("nan"), float("nan"), float("nan")
    treatment_frame = arm_frame(treatment)
    control_frame = arm_frame(control)
    differences = np.asarray(
        [
            treatment_frame[metric][treatment_ids[pair]]
            - control_frame[metric][control_ids[pair]]
            for pair in shared
        ],
        dtype=float,
    )
    return bootstrap_interval(differences)


PAIRED_METRICS = (
    "final_mean_covariance",
    "final_tagged_covariance",
    "final_tagged_error_median_km",
    "reacquired_fraction",
    "mean_reward_per_agent",
)


def build_paired_effects(arms: dict) -> list[list]:
    rows = []
    for policy in POLICIES:
        # Maneuver effect at fixed noise, for every treatment arm.
        for noise_tag, sigma in NOISE_TAGS.items():
            control = arms.get((policy, CONTROL_TAG, noise_tag))
            if control is None:
                continue
            for treatment_tag in TREATMENT_TAGS:
                treatment = arms.get((policy, treatment_tag, noise_tag))
                if treatment is None:
                    continue
                for metric in PAIRED_METRICS:
                    mean, low, high = paired_difference(treatment, control, metric)
                    rows.append(
                        ["maneuver_effect", policy, f"{sigma}|{treatment_tag}",
                         f"{treatment_tag}_minus_control", metric, mean, low, high]
                    )
        # Noise effect at fixed maneuver condition.
        for maneuver in MANEUVER_TAGS:
            treatment = arms.get((policy, maneuver, "sigma10"))
            control = arms.get((policy, maneuver, "sigma00"))
            if treatment is None or control is None:
                continue
            for metric in PAIRED_METRICS:
                mean, low, high = paired_difference(treatment, control, metric)
                rows.append(
                    ["noise_effect", policy, maneuver, "sigma10_minus_sigma00", metric,
                     mean, low, high]
                )
    return rows


def build_dose_response(arms: dict) -> list[list]:
    """Per-RSO delta-v against the covariance response and the tracking error.

    Every treatment arm is included so the delta-v axis spans the primary 1-100 m/s condition and
    the 100-2000 m/s stress condition in one table. The maneuver column says which arm a row is
    from, because the arms are not pooled.
    """
    rows = []
    for policy in POLICIES:
      for maneuver in TREATMENT_TAGS:
        for noise_tag, sigma in NOISE_TAGS.items():
            treatment = arms.get((policy, maneuver, noise_tag))
            control = arms.get((policy, CONTROL_TAG, noise_tag))
            if treatment is None or control is None:
                continue
            control_ids = {pair: index for index, pair in enumerate(control["pair_ids"])}
            for episode, pair in enumerate(treatment["pair_ids"]):
                if pair not in control_ids:
                    continue
                control_episode = control_ids[pair]
                treatment_trace = np.asarray(
                    treatment["end_trace_cov_by_episode"][episode], dtype=float
                )
                control_trace = np.asarray(
                    control["end_trace_cov_by_episode"][control_episode], dtype=float
                )
                errors = final_errors(treatment, episode)
                control_errors = final_errors(control, control_episode)
                error_by_rso = dict(
                    zip(errors["rso_id"].tolist(), errors["estimate_truth_error_km"])
                )
                offset_by_rso = dict(
                    zip(errors["rso_id"].tolist(), errors["nominal_truth_offset_km"])
                )
                control_error_by_rso = dict(
                    zip(
                        control_errors["rso_id"].tolist(),
                        control_errors["estimate_truth_error_km"],
                    )
                )
                reacquired = np.asarray(treatment["reacquired"][episode], dtype=bool)
                reacquisition = np.asarray(
                    treatment["reacquisition_time"][episode], dtype=float
                )
                observation_count = np.asarray(
                    treatment["post_maneuver_observation_count"][episode], dtype=int
                )
                for event in treatment["maneuver_events"][episode]:
                    rso_id = int(event["rso_id"])
                    delta_v = np.asarray(event["delta_v_mps"], dtype=float)
                    rows.append(
                        [
                            policy,
                            sigma,
                            maneuver,
                            pair,
                            rso_id,
                            float(event["delta_v_magnitude_mps"]),
                            float(delta_v[0]),
                            float(delta_v[1]),
                            float(delta_v[2]),
                            float(event["time_seconds"]),
                            float(5400.0 - event["time_seconds"]),
                            float(treatment_trace[rso_id]),
                            float(control_trace[rso_id]),
                            float(treatment_trace[rso_id] - control_trace[rso_id]),
                            float(error_by_rso.get(rso_id, np.nan)),
                            float(control_error_by_rso.get(rso_id, np.nan)),
                            float(offset_by_rso.get(rso_id, np.nan)),
                            bool(reacquired[rso_id]),
                            float(reacquisition[rso_id]),
                            int(observation_count[rso_id]),
                        ]
                    )
    return rows


NIS_GROUPS = ("non_maneuvering", "maneuvering_before_burn", "maneuvering_after_burn")


def pooled_nis(result: dict) -> dict[str, np.ndarray]:
    """Pool every recorded innovation across the arm's episodes."""
    values, maneuvering, after_burn, delay = [], [], [], []
    for table in result.get("nis_table", []):
        if not isinstance(table, dict) or len(table.get("nis", [])) == 0:
            continue
        seconds_after_burn = np.asarray(table["seconds_after_burn"], dtype=float)
        values.append(np.asarray(table["nis"], dtype=float))
        maneuvering.append(np.asarray(table["is_maneuvering"], dtype=bool))
        after_burn.append(seconds_after_burn >= 0.0)
        delay.append(seconds_after_burn)
    if not values:
        return {}
    return {
        "nis": np.concatenate(values),
        "is_maneuvering": np.concatenate(maneuvering),
        "after_burn": np.concatenate(after_burn),
        "seconds_after_burn": np.concatenate(delay),
    }


def build_nis_summary(arms: dict) -> list[list]:
    """Is NIS a valid consistency statistic, and does it separate maneuvers?

    Under a matched filter NIS is chi-square with three degrees of freedom, so the mean is 3
    and one percent of measurements clear the p99 threshold. Departures from those two
    reference values say whether the filter is consistent at each noise level, and the gap
    between the maneuvering-after-burn group and the non-maneuvering group says whether the
    statistic carries a usable detection signal.
    """
    rows = []
    for policy in POLICIES:
        for noise_tag, sigma in NOISE_TAGS.items():
            for maneuver in MANEUVER_TAGS:
                result = arms.get((policy, maneuver, noise_tag))
                if result is None:
                    continue
                pooled = pooled_nis(result)
                if not pooled:
                    continue
                masks = {
                    "all": np.ones_like(pooled["is_maneuvering"]),
                    "non_maneuvering": ~pooled["is_maneuvering"],
                    "maneuvering_before_burn": pooled["is_maneuvering"] & ~pooled["after_burn"],
                    "maneuvering_after_burn": pooled["is_maneuvering"] & pooled["after_burn"],
                }
                for group, mask in masks.items():
                    selected = pooled["nis"][mask]
                    if selected.size == 0:
                        continue
                    rows.append(
                        [
                            policy,
                            sigma,
                            maneuver,
                            group,
                            int(selected.size),
                            float(np.mean(selected)),
                            float(np.median(selected)),
                            float(np.percentile(selected, 95)),
                            float(np.mean(selected > CHI_SQUARE_3_THRESHOLDS["p95"])),
                            float(np.mean(selected > CHI_SQUARE_3_THRESHOLDS["p99"])),
                        ]
                    )
    return rows


def first_post_burn_nis(result: dict) -> list[tuple[int, str, float, float, float]]:
    """The first innovation after a burn, which is where the maneuver signature lives.

    Averaging NIS over every post-burn measurement understates the signal, because once the
    filter has absorbed the first post-burn measurement the object is largely re-acquired and
    later innovations are back near the noise floor.
    """
    rows = []
    events_by_episode = result.get("maneuver_events", [])
    for episode, table in enumerate(result.get("nis_table", [])):
        if not isinstance(table, dict) or len(table.get("nis", [])) == 0:
            continue
        magnitude_by_rso = {
            int(event["rso_id"]): float(event["delta_v_magnitude_mps"])
            for event in events_by_episode[episode]
        }
        rso_ids = np.asarray(table["rso_id"], dtype=int)
        values = np.asarray(table["nis"], dtype=float)
        maneuvering = np.asarray(table["is_maneuvering"], dtype=bool)
        delay = np.asarray(table["seconds_after_burn"], dtype=float)
        eligible = np.where(maneuvering & (delay >= 0.0))[0]
        seen = set()
        for index in eligible[np.argsort(delay[eligible], kind="stable")]:
            rso_id = int(rso_ids[index])
            if rso_id in seen:
                continue
            seen.add(rso_id)
            rows.append(
                (
                    rso_id,
                    result["pair_ids"][episode],
                    magnitude_by_rso.get(rso_id, float("nan")),
                    float(delay[index]),
                    float(values[index]),
                )
            )
    return rows


def build_first_post_burn_nis(arms: dict) -> list[list]:
    rows = []
    for policy in POLICIES:
      for maneuver in (CONTROL_TAG,) + TREATMENT_TAGS:
        for noise_tag, sigma in NOISE_TAGS.items():
            result = arms.get((policy, maneuver, noise_tag))
            if result is None:
                continue
            for rso_id, pair, magnitude, delay, value in first_post_burn_nis(result):
                rows.append(
                    [
                        policy, sigma, maneuver, pair, rso_id, magnitude, delay, value,
                        value > CHI_SQUARE_3_THRESHOLDS["p99"],
                    ]
                )
    return rows


def verify_control_is_matched(arms: dict) -> dict:
    """Confirm the control differs from the treatment only in delta-v magnitude.

    The earlier fixed-900s control gave tagged objects a longer post-burn observation window
    than the randomly timed treatment, which on its own moved reacquisition. This check exists
    so that confound cannot return unnoticed.
    """
    report = {}
    for policy in POLICIES:
        for noise_tag, sigma in NOISE_TAGS.items():
            treatment = arms.get((policy, "random", noise_tag))
            control = arms.get((policy, CONTROL_TAG, noise_tag))
            if treatment is None or control is None:
                continue
            control_index = {pair: i for i, pair in enumerate(control["pair_ids"])}
            burn_time_mismatches = 0
            cohort_mismatches = 0
            control_nonzero_delta_v = 0
            compared = 0
            for episode, pair in enumerate(treatment["pair_ids"]):
                if pair not in control_index:
                    continue
                treated = treatment["maneuver_events"][episode]
                controlled = control["maneuver_events"][control_index[pair]]
                if [e["rso_id"] for e in treated] != [e["rso_id"] for e in controlled]:
                    cohort_mismatches += 1
                    continue
                for a, b in zip(treated, controlled):
                    compared += 1
                    if a["time_seconds"] != b["time_seconds"]:
                        burn_time_mismatches += 1
                    if b["delta_v_magnitude_mps"] != 0.0:
                        control_nonzero_delta_v += 1
            report[f"{policy}|sigma={sigma}"] = {
                "events_compared": compared,
                "cohort_mismatches": cohort_mismatches,
                "burn_time_mismatches": burn_time_mismatches,
                "control_events_with_nonzero_delta_v": control_nonzero_delta_v,
                "matched": (
                    cohort_mismatches == 0
                    and burn_time_mismatches == 0
                    and control_nonzero_delta_v == 0
                    and compared > 0
                ),
            }
    return report


def build_action_invariance(arms: dict) -> dict:
    """Does a policy point anywhere different once the measurements carry noise?"""
    report = {}
    for policy in POLICIES:
        for maneuver in MANEUVER_TAGS:
            quiet = arms.get((policy, maneuver, "sigma00"))
            noisy = arms.get((policy, maneuver, "sigma10"))
            if quiet is None or noisy is None:
                continue
            quiet_ids = {pair: index for index, pair in enumerate(quiet["pair_ids"])}
            noisy_ids = {pair: index for index, pair in enumerate(noisy["pair_ids"])}
            shared = sorted(set(quiet_ids) & set(noisy_ids))
            identical_episodes = 0
            changed_decisions = 0
            total_decisions = 0
            for pair in shared:
                quiet_episode = quiet_ids[pair]
                noisy_episode = noisy_ids[pair]
                episode_identical = True
                for field in ("azi", "alt"):
                    quiet_actions = quiet[field][quiet_episode]
                    noisy_actions = noisy[field][noisy_episode]
                    for agent in quiet_actions:
                        first = np.asarray(quiet_actions[agent], dtype=float)
                        second = np.asarray(noisy_actions[agent], dtype=float)
                        length = min(len(first), len(second))
                        if field == "azi":
                            total_decisions += length
                            changed_decisions += int(
                                np.sum(first[:length] != second[:length])
                            )
                        if len(first) != len(second) or not np.array_equal(first, second):
                            episode_identical = False
                identical_episodes += int(episode_identical)
            report[f"{policy}|{maneuver}"] = {
                "episodes_compared": len(shared),
                "episodes_with_identical_pointing": identical_episodes,
                "fraction_identical": (
                    identical_episodes / len(shared) if shared else None
                ),
                "azimuth_decisions_compared": total_decisions,
                "azimuth_decisions_changed": changed_decisions,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=GRID_DIRECTORY / "analysis")
    args = parser.parse_args()

    arms = {}
    for policy in POLICIES:
        for maneuver in MANEUVER_TAGS:
            for noise_tag in NOISE_TAGS:
                result = load_arm(policy, maneuver, noise_tag, args.seeds)
                if result is not None and result["pair_ids"]:
                    arms[(policy, maneuver, noise_tag)] = result
    if not arms:
        raise FileNotFoundError(f"No grid results found under {GRID_DIRECTORY}")

    write_csv(
        args.output_dir / "cell_summary.csv",
        [
            "policy", "sigma_km", "maneuver", "episodes",
            "mean_final_mean_covariance", "mean_final_tagged_covariance",
            "mean_final_tagged_error_median_km", "mean_final_tagged_error_p90_km",
            "mean_final_tagged_offset_median_km", "mean_reward_per_agent",
            "mean_reacquired_fraction",
        ],
        build_cell_summary(arms, args.seeds),
    )
    write_csv(
        args.output_dir / "paired_effects.csv",
        ["comparison", "policy", "held_fixed", "contrast", "metric",
         "mean_paired_change", "ci_low", "ci_high"],
        build_paired_effects(arms),
    )
    write_csv(
        args.output_dir / "dose_response.csv",
        [
            "policy", "sigma_km", "maneuver", "pair_id", "rso_id", "delta_v_magnitude_mps",
            "delta_v_radial_mps", "delta_v_transverse_mps", "delta_v_normal_mps",
            "burn_time_seconds", "seconds_after_burn",
            "final_trace_maneuver", "final_trace_control", "final_trace_change",
            "final_error_maneuver_km", "final_error_control_km",
            "final_nominal_truth_offset_km", "reacquired", "reacquisition_time_seconds",
            "post_maneuver_observation_count",
        ],
        build_dose_response(arms),
    )
    write_csv(
        args.output_dir / "nis_summary.csv",
        ["policy", "sigma_km", "maneuver", "group", "measurements",
         "mean_nis", "median_nis", "p95_nis",
         "fraction_above_chi2_p95", "fraction_above_chi2_p99"],
        build_nis_summary(arms),
    )
    write_csv(
        args.output_dir / "first_post_burn_nis.csv",
        ["policy", "sigma_km", "maneuver", "pair_id", "rso_id", "delta_v_magnitude_mps",
         "seconds_after_burn", "nis", "exceeds_chi2_p99"],
        build_first_post_burn_nis(arms),
    )
    matched = verify_control_is_matched(arms)
    (args.output_dir / "control_matching_check.json").write_text(
        json.dumps(matched, indent=2) + "\n"
    )
    invariance = build_action_invariance(arms)
    (args.output_dir / "action_invariance.json").write_text(
        json.dumps(invariance, indent=2) + "\n"
    )

    print(f"arms analysed: {len(arms)}")
    for key, result in sorted(arms.items()):
        print(f"  {key} episodes={len(result['pair_ids'])}")
    for key, value in matched.items():
        print(f"  control matched [{key}]: {value['matched']}")
    print(f"outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
