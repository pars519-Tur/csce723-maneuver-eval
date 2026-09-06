from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = NOTEBOOK_DIR / "08_smith_metric_vs_physical_tracking_accuracy.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
notebook["cells"] = [
    markdown(
        r"""
# Smith Metric vs. Physical Tracking Accuracy

## tl;dr

This is the primary maneuver-robustness comparison requested alongside Smith's Chapter 2 figures.

For the **same tagged maneuver cohort**, each main figure places:

1. Smith's reported trace-covariance metric on top; and
2. realized Cartesian filter estimate–truth error underneath.

The result is not that Smith covariance stops decreasing. It continues to decrease normally. The failure is that, after an unmodeled maneuver, this decreasing internal covariance no longer represents physical tracking accuracy.
"""
    ),
    markdown(
        r"""
## Context & Methods

### Key Assumptions

- Frozen inference only; Smith source, checkpoints, policy observations, reward, and UKF implementation are unchanged.
- Policies: Advanced Greedy and frozen CNNv2 Reward 2.9 PPO.
- Scenario: 3 sensors, 100 RSOs, 5400 s.
- Tagged cohort: the same 20 RSOs in every paired 0/10/100 m/s comparison.
- Maneuver: one along-track impulse at 900 s.
- Control: a tagged 0 m/s arm that includes the same physical truth gate and the same RSO cohort.
- Curves are the median across 10 paired episode summaries; ribbons show the 10th–90th percentile.
- Covariance is Smith's trace of the orbital-element UKF covariance and is shown in its native units. Position error is shown separately in km; the two quantities are not overlaid or treated as interchangeable.
- This is a 10-seed diagnostic comparison. Exact effect sizes require the planned final seed count before thesis-level inferential claims.
"""
    ),
    code(
        r"""
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "experiments" / "csce723_maneuver_eval").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate repository root")


REPOSITORY_ROOT = find_repository_root(Path.cwd().resolve())
PROJECT_ROOT = REPOSITORY_ROOT / "experiments" / "csce723_maneuver_eval"
DATA_ROOT = PROJECT_ROOT / "results" / "diagnostic"
FIGURE_ROOT = PROJECT_ROOT / "results" / "smith_metric_vs_tracking_accuracy"
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

POLICIES = {
    "Advanced Greedy": "advanced_greedy",
    "Frozen CNNv2 PPO": "frozen_cnnv2_ppo",
}
DELTA_V_VALUES = [0, 10, 100]
COLORS = {0: "#6b7280", 10: "#d97706", 100: "#be185d"}
LINESTYLES = {0: "--", 10: "-", 100: "-."}
BURN_TIME_SECONDS = 900.0
COMMON_TIME = np.arange(0.0, 5400.1, 60.0)

results = {}
for policy_label, policy_key in POLICIES.items():
    for delta_v in DELTA_V_VALUES:
        path = DATA_ROOT / f"{policy_key}_dvT_{delta_v}mps_rate20_10seeds.p"
        with path.open("rb") as stream:
            results[(policy_label, delta_v)] = pickle.load(stream)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
    "axes.grid": True, "grid.color": "#d1d5db", "grid.alpha": 0.70,
    "axes.spines.top": False, "axes.spines.right": False,
})
"""
    ),
    markdown(
        r"""
## Data

These checks lock the paired comparison and confirm that the 0 m/s truth trajectory is exactly nominal for the tagged cohort.
"""
    ),
    code(
        r"""
checks = []
for policy_label in POLICIES:
    control = results[(policy_label, 0)]
    pair_match = all(
        results[(policy_label, delta_v)]["pair_ids"] == control["pair_ids"]
        for delta_v in DELTA_V_VALUES[1:]
    )
    cohort_match = all(
        np.array_equal(
            control["maneuver_rso_ids"][episode],
            results[(policy_label, delta_v)]["maneuver_rso_ids"][episode],
        )
        for delta_v in DELTA_V_VALUES[1:]
        for episode in range(10)
    )
    zero_offset = max(
        float(np.max(history["nominal_truth_offset_km"]))
        for history in control["estimation_error_history"]
    )
    checks.append({
        "policy": policy_label,
        "paired_seed_ids_match": pair_match,
        "tagged_rso_ids_match": cohort_match,
        "zero_control_truth_offset_max_km": zero_offset,
    })

checks = pd.DataFrame(checks)
assert checks["paired_seed_ids_match"].all()
assert checks["tagged_rso_ids_match"].all()
assert (checks["zero_control_truth_offset_max_km"] == 0.0).all()
checks
"""
    ),
    code(
        r"""
def tagged_covariance_curves(result, reducer):
    curves = []
    for episode, history in enumerate(result["trace_covariance_history_by_rso"]):
        tagged_ids = np.asarray(result["maneuver_rso_ids"][episode], dtype=int)
        tagged_values = np.asarray(history["trace_covariance"], dtype=float)[:, tagged_ids]
        episode_curve = reducer(tagged_values, axis=1)
        curves.append(np.interp(COMMON_TIME, history["time_seconds"], episode_curve))
    return np.asarray(curves)


def tagged_error_curves(result, reducer):
    curves = []
    for history in result["estimation_error_history"]:
        episode_curve = reducer(
            np.asarray(history["estimate_truth_error_km"], dtype=float), axis=1
        )
        curves.append(np.interp(COMMON_TIME, history["time_seconds"], episode_curve))
    return np.asarray(curves)


def plot_curve_with_band(ax, curves, delta_v, label):
    center = np.median(curves, axis=0)
    lower, upper = np.percentile(curves, [10, 90], axis=0)
    ax.plot(
        COMMON_TIME, center, color=COLORS[delta_v], linestyle=LINESTYLES[delta_v],
        linewidth=2.2, label=label,
    )
    ax.fill_between(COMMON_TIME, lower, upper, color=COLORS[delta_v], alpha=0.10)


def smith_metric_and_error_figure(reducer, metric_name, smith_figure_number, filename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for column, policy_label in enumerate(POLICIES):
        for delta_v in DELTA_V_VALUES:
            result = results[(policy_label, delta_v)]
            plot_curve_with_band(
                axes[0, column], tagged_covariance_curves(result, reducer),
                delta_v, f"{delta_v} m/s",
            )
            plot_curve_with_band(
                axes[1, column], tagged_error_curves(result, reducer),
                delta_v, f"{delta_v} m/s",
            )
        axes[0, column].set_title(policy_label)
        axes[0, column].set_ylabel(f"Tagged {metric_name.lower()} trace covariance")
        axes[1, column].set_ylabel(f"Tagged {metric_name.lower()} position error (km)")
        axes[1, column].set_xlabel("Propagation time (s)")
        axes[1, column].set_yscale("symlog", linthresh=0.1)
        axes[1, column].set_ylim(0, 650)
        for row in range(2):
            axes[row, column].axvline(
                BURN_TIME_SECONDS, color="#111827", linestyle=":", linewidth=1.4
            )
            axes[row, column].set_xlim(0, 5400)
            axes[row, column].legend(title="Along-track impulse", fontsize=8)

    fig.suptitle(
        f"Smith Fig. {smith_figure_number}-style tagged {metric_name.lower()} covariance and physical tracking error",
        fontsize=14,
    )
    fig.text(
        0.5, 0.01,
        "Same tagged 20-RSO cohorts; lines are medians across 10 paired episode summaries; bands are 10th–90th percentiles",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(FIGURE_ROOT / filename, dpi=200, bbox_inches="tight")
    plt.show()
"""
    ),
    markdown(
        r"""
## Results

### 1. Smith Fig. 2.10-style tagged mean covariance versus mean physical error

The covariance panels answer whether the agents continue to reduce Smith's uncertainty metric. The error panels answer whether the estimated states remain physically accurate.
"""
    ),
    code(
        r"""
smith_metric_and_error_figure(
    np.mean, "Mean", "2.10", "fig2_10_style_tagged_mean_covariance_vs_tracking_error.png"
)
"""
    ),
    markdown(
        r"""
### 2. Smith Fig. 2.11-style tagged median covariance versus median physical error

The median view confirms that the mismatch is not created only by a few extreme objects.
"""
    ),
    code(
        r"""
smith_metric_and_error_figure(
    np.median, "Median", "2.11", "fig2_11_style_tagged_median_covariance_vs_tracking_error.png"
)
"""
    ),
    markdown(
        r"""
### 3. Sensor-tasking outcome: measurement loss versus post-burn coverage

The left panel reports the proportion of Smith-selected tagged measurements rejected by the physical truth gate. The right panel reports the tagged objects receiving at least one successful post-burn measurement. This distinguishes complete sensor loss from inaccurate state tracking.
"""
    ),
    code(
        r"""
tasking_rows = []
for policy_label in POLICIES:
    for delta_v in DELTA_V_VALUES:
        result = results[(policy_label, delta_v)]
        selected = sum(
            int(np.sum(events["selected_by_smith_estimate"]))
            for events in result["measurement_events"]
        )
        received = sum(
            int(np.sum(events["measurement_received"]))
            for events in result["measurement_events"]
        )
        tagged_total = sum(len(ids) for ids in result["maneuver_rso_ids"])
        post_burn_observed = sum(int(np.sum(values)) for values in result["reacquired"])
        tasking_rows.append({
            "policy": policy_label,
            "delta_v_t_mps": delta_v,
            "truth_gate_rejection_percent": 100.0 * (1.0 - received / selected),
            "post_burn_observed_percent": 100.0 * post_burn_observed / tagged_total,
            "selected_measurements": selected,
        })

tasking = pd.DataFrame(tasking_rows)
tasking.to_csv(FIGURE_ROOT / "tasking_outcome_summary.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
x = np.arange(len(DELTA_V_VALUES))
width = 0.34
policy_colors = {"Advanced Greedy": "#2563eb", "Frozen CNNv2 PPO": "#d97706"}
for index, policy_label in enumerate(POLICIES):
    policy_data = tasking[tasking["policy"] == policy_label].set_index("delta_v_t_mps")
    offset = (index - 0.5) * width
    rejection = policy_data.loc[DELTA_V_VALUES, "truth_gate_rejection_percent"].to_numpy()
    coverage = policy_data.loc[DELTA_V_VALUES, "post_burn_observed_percent"].to_numpy()
    bars_left = axes[0].bar(
        x + offset, rejection, width, color=policy_colors[policy_label], alpha=0.80,
        label=policy_label,
    )
    bars_right = axes[1].bar(
        x + offset, coverage, width, color=policy_colors[policy_label], alpha=0.80,
        label=policy_label,
    )
    axes[0].bar_label(bars_left, fmt="%.1f%%", padding=3, fontsize=8)
    axes[1].bar_label(bars_right, fmt="%.1f%%", padding=3, fontsize=8)

axes[0].set(
    title="Truth-gate measurement rejection",
    xlabel="Along-track impulse (m/s)", ylabel="Rejected selected tagged measurements (%)",
    xticks=x, xticklabels=[str(value) for value in DELTA_V_VALUES], ylim=(0, 12),
)
axes[1].set(
    title="Tagged objects with a post-burn measurement",
    xlabel="Along-track impulse (m/s)", ylabel="Tagged objects observed at least once (%)",
    xticks=x, xticklabels=[str(value) for value in DELTA_V_VALUES], ylim=(0, 100),
)
for ax in axes:
    ax.legend(fontsize=8)
fig.suptitle("Sensor-tasking outcomes for the same tagged maneuver cohorts", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIGURE_ROOT / "tagged_sensor_tasking_outcomes.png", dpi=200, bbox_inches="tight")
plt.show()
tasking.round(2)
"""
    ),
    markdown(
        r"""
### 4. Final comparison table

This table keeps the two conclusions separate: Smith covariance performance and realized state accuracy.
"""
    ),
    code(
        r"""
final_rows = []
for policy_label in POLICIES:
    for delta_v in DELTA_V_VALUES:
        result = results[(policy_label, delta_v)]
        terminal_covariance = np.asarray(result["end_trace_cov_by_episode"], dtype=float)
        tagged_covariance = np.concatenate([
            terminal_covariance[episode, np.asarray(ids, dtype=int)]
            for episode, ids in enumerate(result["maneuver_rso_ids"])
        ])
        final_error = np.concatenate([
            np.asarray(history["estimate_truth_error_km"][-1], dtype=float)
            for history in result["estimation_error_history"]
        ])
        final_rows.append({
            "policy": policy_label,
            "delta_v_t_mps": delta_v,
            "final_tagged_covariance_median": np.median(tagged_covariance),
            "final_position_error_median_km": np.median(final_error),
            "final_position_error_p90_km": np.percentile(final_error, 90),
        })

final_summary = pd.DataFrame(final_rows)
final_summary.to_csv(FIGURE_ROOT / "smith_metric_vs_accuracy_summary.csv", index=False)
final_summary.round({
    "final_tagged_covariance_median": 6,
    "final_position_error_median_km": 2,
    "final_position_error_p90_km": 2,
})
"""
    ),
    markdown(
        r"""
## Takeaways

1. **Yes, both agents continue to reduce Smith's reported covariance under the maneuver conditions.** The tagged covariance histories remain close across 0, 10, and 100 m/s.
2. **No, low covariance does not mean accurate post-maneuver tracking.** The same tagged objects accumulate tens of kilometers of median error at 10 m/s and hundreds at 100 m/s.
3. **The agents do not simply lose every maneuvering object.** Most selected measurements survive the truth gate at 10 m/s, and many tagged objects receive post-burn measurements. The key failure is maneuver-unaware state estimation and covariance–accuracy inconsistency.
4. **Supported research gap:** covariance-based reward and evaluation can report successful uncertainty reduction while physical orbit-estimation accuracy degrades under unmodeled maneuvers.
5. **Required caveat:** this notebook supports the mechanism and direction with 10 paired diagnostic seeds; final thesis effect sizes should use the locked final seed count.
"""
    ),
]

nbf.write(notebook, OUTPUT_PATH)
print(OUTPUT_PATH)
