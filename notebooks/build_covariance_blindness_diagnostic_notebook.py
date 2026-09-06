from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = NOTEBOOK_DIR / "07_covariance_blindness_truth_error_diagnostic.ipynb"


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
# Covariance Blindness Under Unmodeled Maneuvers

## tl;dr

This paired diagnostic shows why the Smith-format covariance plots can remain almost unchanged even when maneuvering-object state errors become operationally large.

- Frozen Smith policies are evaluated without training or source/checkpoint changes.
- The same 10 seeds and the same tagged 20-RSO cohort are used at 0, 10, and 100 m/s along-track impulse magnitude.
- The 0 m/s arm is a tagged physical-gate control, not the untagged coast dataset.
- The primary metric is Cartesian **filter estimate–maneuver truth error** in Smith's own TEME measurement space.

This is a diagnostic companion to Notebook 06, which contains the Smith Fig. 2.10–2.15 coast-versus-maneuver reproductions.
"""
    ),
    markdown(
        r"""
## Context & Methods

### Key Assumptions

- Scenario: 3 sensors, 100 RSOs, 5400 s; 20 tagged RSOs per episode.
- Burn: one impulsive along-track event at 900 s.
- Policies: unchanged Advanced Greedy and frozen CNNv2 Reward 2.9 PPO.
- Conditions differ only in impulse magnitude: 0, 10, or 100 m/s.
- Smith's UKF mean is converted to Cartesian position with Smith's own `measfun`; truth comes from the external maneuver trajectory.
- Error histories are sampled every 60 s and at the final event.
- Results are descriptive because this targeted diagnostic has 10 paired seeds. It is not yet the final 100-seed thesis inference.
- No NIS or NEES is computed. Covariance and realized Cartesian error are intentionally shown as separate quantities.
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
FIGURE_ROOT = PROJECT_ROOT / "results" / "diagnostic_analysis"
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
    "axes.grid": True, "grid.color": "#d1d5db", "grid.alpha": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})
"""
    ),
    markdown(
        r"""
## Data

The following checks prevent a false maneuver comparison caused by different seeds, target cohorts, or a nonzero control trajectory.
"""
    ),
    code(
        r"""
validation_rows = []
for policy_label in POLICIES:
    policy_results = {dv: results[(policy_label, dv)] for dv in DELTA_V_VALUES}
    pair_ids_match = all(
        policy_results[dv]["pair_ids"] == policy_results[0]["pair_ids"]
        for dv in DELTA_V_VALUES[1:]
    )
    cohort_ids_match = all(
        np.array_equal(
            policy_results[0]["maneuver_rso_ids"][episode_index],
            policy_results[dv]["maneuver_rso_ids"][episode_index],
        )
        for dv in DELTA_V_VALUES[1:]
        for episode_index in range(10)
    )
    zero_offset_max = max(
        float(np.max(history["nominal_truth_offset_km"]))
        for history in policy_results[0]["estimation_error_history"]
    )
    zero_error_identity_max = max(
        float(np.max(np.abs(
            history["estimate_truth_error_km"]
            - history["estimate_nominal_error_km"]
        )))
        for history in policy_results[0]["estimation_error_history"]
    )
    validation_rows.append({
        "policy": policy_label,
        "episodes": len(policy_results[0]["pair_ids"]),
        "pair_ids_match": pair_ids_match,
        "tagged_cohorts_match": cohort_ids_match,
        "zero_control_offset_max_km": zero_offset_max,
        "zero_control_error_identity_max_km": zero_error_identity_max,
    })

validation_table = pd.DataFrame(validation_rows)
assert validation_table["pair_ids_match"].all()
assert validation_table["tagged_cohorts_match"].all()
assert (validation_table["zero_control_offset_max_km"] == 0.0).all()
assert (validation_table["zero_control_error_identity_max_km"] == 0.0).all()
validation_table
"""
    ),
    code(
        r"""
def final_tagged_arrays(result):
    final_error = np.concatenate([
        history["estimate_truth_error_km"][-1]
        for history in result["estimation_error_history"]
    ]).astype(float)
    final_offset = np.concatenate([
        history["nominal_truth_offset_km"][-1]
        for history in result["estimation_error_history"]
    ]).astype(float)
    terminal_covariance = np.asarray(result["end_trace_cov_by_episode"], dtype=float)
    tagged_covariance = np.concatenate([
        terminal_covariance[index, np.asarray(rso_ids, dtype=int)]
        for index, rso_ids in enumerate(result["maneuver_rso_ids"])
    ])
    selected = sum(
        int(np.sum(events["selected_by_smith_estimate"]))
        for events in result["measurement_events"]
    )
    received = sum(
        int(np.sum(events["measurement_received"]))
        for events in result["measurement_events"]
    )
    return final_error, final_offset, tagged_covariance, selected, received


summary_rows = []
for policy_label in POLICIES:
    control_episode_medians = np.asarray([
        np.median(history["estimate_truth_error_km"][-1])
        for history in results[(policy_label, 0)]["estimation_error_history"]
    ])
    for delta_v in DELTA_V_VALUES:
        result = results[(policy_label, delta_v)]
        final_error, final_offset, tagged_covariance, selected, received = final_tagged_arrays(result)
        episode_medians = np.asarray([
            np.median(history["estimate_truth_error_km"][-1])
            for history in result["estimation_error_history"]
        ])
        summary_rows.append({
            "policy": policy_label,
            "delta_v_t_mps": delta_v,
            "paired_seeds": len(result["pair_ids"]),
            "tagged_object_episodes": len(final_error),
            "final_error_median_km": np.median(final_error),
            "final_error_p90_km": np.percentile(final_error, 90),
            "final_truth_offset_median_km": np.median(final_offset),
            "final_tagged_covariance_median": np.median(tagged_covariance),
            "paired_episode_error_increase_median_km": np.median(
                episode_medians - control_episode_medians
            ),
            "selected_measurements": selected,
            "truth_gate_rejection_percent": 100.0 * (1.0 - received / selected),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(FIGURE_ROOT / "covariance_blindness_summary.csv", index=False)
summary.round({
    "final_error_median_km": 2,
    "final_error_p90_km": 2,
    "final_truth_offset_median_km": 2,
    "final_tagged_covariance_median": 6,
    "paired_episode_error_increase_median_km": 2,
    "truth_gate_rejection_percent": 2,
})
"""
    ),
    markdown(
        r"""
## Results

### 1. Realized position error grows while covariance barely moves

The upper panels show the median realized position error of the tagged cohort. The lower panels show Smith's internal trace covariance for the same tagged objects. Curves summarize 10 paired seeds; bands are the 10th–90th percentile across episode medians.
"""
    ),
    code(
        r"""
def interpolated_episode_error(result):
    curves = []
    for history in result["estimation_error_history"]:
        episode_median = np.median(history["estimate_truth_error_km"], axis=1)
        curves.append(np.interp(COMMON_TIME, history["time_seconds"], episode_median))
    return np.asarray(curves)


def interpolated_episode_tagged_covariance(result):
    curves = []
    for episode_index, history in enumerate(result["trace_covariance_history_by_rso"]):
        rso_ids = np.asarray(result["maneuver_rso_ids"][episode_index], dtype=int)
        episode_median = np.median(
            np.asarray(history["trace_covariance"], dtype=float)[:, rso_ids], axis=1
        )
        curves.append(np.interp(COMMON_TIME, history["time_seconds"], episode_median))
    return np.asarray(curves)


fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
for column, policy_label in enumerate(POLICIES):
    for delta_v in DELTA_V_VALUES:
        error_curves = interpolated_episode_error(results[(policy_label, delta_v)])
        covariance_curves = interpolated_episode_tagged_covariance(results[(policy_label, delta_v)])
        for ax, curves in [(axes[0, column], error_curves), (axes[1, column], covariance_curves)]:
            median = np.median(curves, axis=0)
            lower, upper = np.percentile(curves, [10, 90], axis=0)
            ax.plot(
                COMMON_TIME, median, color=COLORS[delta_v], linestyle=LINESTYLES[delta_v],
                linewidth=2.1, label=f"{delta_v} m/s",
            )
            ax.fill_between(COMMON_TIME, lower, upper, color=COLORS[delta_v], alpha=0.10)
    axes[0, column].set_title(policy_label)
    axes[0, column].set_ylabel("Estimate–truth position error (km)")
    axes[1, column].set_ylabel("Tagged median trace covariance")
    axes[1, column].set_xlabel("Propagation time (s)")
    axes[0, column].set_yscale("symlog", linthresh=0.1)
    axes[0, column].set_ylim(0, 550)
    for row in range(2):
        axes[row, column].axvline(BURN_TIME_SECONDS, color="#111827", linestyle=":", linewidth=1.4)
        axes[row, column].set_xlim(0, 5400)
        axes[row, column].legend(title="Along-track impulse", fontsize=8)

fig.suptitle("Realized position error and Smith trace covariance for the same tagged RSOs", fontsize=14)
fig.text(0.5, 0.01, "Lines: median across 10 episode medians; bands: 10th–90th percentile; burn at 900 s", ha="center")
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(FIGURE_ROOT / "truth_error_vs_covariance_history.png", dpi=200, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
### 2. Final error distribution and covariance distribution tell different stories

Each box contains 200 tagged object-episodes (20 objects × 10 seeds). The log scale is used only for position error so that the valid near-zero control and maneuver cases remain visible together.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for column, policy_label in enumerate(POLICIES):
    error_groups = []
    covariance_groups = []
    for delta_v in DELTA_V_VALUES:
        final_error, _, tagged_covariance, _, _ = final_tagged_arrays(
            results[(policy_label, delta_v)]
        )
        error_groups.append(final_error)
        covariance_groups.append(tagged_covariance)

    for ax, groups, ylabel in [
        (axes[0, column], error_groups, "Final estimate–truth error (km)"),
        (axes[1, column], covariance_groups, "Final tagged trace covariance"),
    ]:
        parts = ax.boxplot(groups, tick_labels=["0", "10", "100"], patch_artist=True,
                           showfliers=False, widths=0.58)
        for patch, delta_v in zip(parts["boxes"], DELTA_V_VALUES):
            patch.set_facecolor(COLORS[delta_v]); patch.set_alpha(0.28)
            patch.set_edgecolor(COLORS[delta_v])
        for median in parts["medians"]:
            median.set_color("#111827"); median.set_linewidth(2)
        ax.set_xlabel("Along-track impulse (m/s)")
        ax.set_ylabel(ylabel)
    axes[0, column].set_title(policy_label)
    axes[0, column].set_yscale("log")

fig.suptitle("Final realized error grows by orders of magnitude while covariance remains similar", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIGURE_ROOT / "final_error_and_covariance_distributions.png", dpi=200, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
### 3. How much of the maneuver displacement remains in the filter error?

The x-axis is the final displacement between maneuver truth and the nominal coast truth. The y-axis is the final distance between Smith's filter estimate and maneuver truth. Points near the identity line indicate that the filter retained nearly the full unmodeled maneuver displacement as state error. Points below the line indicate partial correction from post-burn measurements.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
for ax, policy_label in zip(axes, POLICIES):
    for delta_v, marker in [(10, "o"), (100, "^")]:
        final_error, final_offset, _, _, _ = final_tagged_arrays(
            results[(policy_label, delta_v)]
        )
        ax.scatter(
            final_offset, final_error, s=24, marker=marker, color=COLORS[delta_v],
            alpha=0.45, edgecolors="none", label=f"{delta_v} m/s",
        )
    limit = 475
    ax.plot([0, limit], [0, limit], color="#111827", linestyle="--", linewidth=1.5,
            label="No correction (error = offset)")
    ax.set(title=policy_label, xlabel="Maneuver truth offset from nominal truth (km)",
           ylabel="Filter estimate–maneuver truth error (km)", xlim=(0, limit), ylim=(0, limit))
    ax.legend(fontsize=8)

fig.suptitle("Final maneuver displacement versus realized filter error — 200 tagged object-episodes per condition", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIGURE_ROOT / "maneuver_offset_vs_filter_error.png", dpi=200, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
### 4. Measurement loss is not required for large state error

Truth-gate rejection is computed only among Smith-estimate-selected measurements for the tagged cohort. A small rejection rate alongside a large state error means that remaining inside the coarse 4° cell does not guarantee an accurate state estimate.
"""
    ),
    code(
        r"""
pivot = summary.pivot(index="delta_v_t_mps", columns="policy", values="truth_gate_rejection_percent")
fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(DELTA_V_VALUES))
width = 0.34
policy_colors = {"Advanced Greedy": "#2563eb", "Frozen CNNv2 PPO": "#d97706"}
for index, policy_label in enumerate(POLICIES):
    values = pivot[policy_label].reindex(DELTA_V_VALUES).to_numpy()
    bars = ax.bar(x + (index - 0.5) * width, values, width, color=policy_colors[policy_label],
                  alpha=0.78, label=policy_label)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
ax.set(xticks=x, xticklabels=[str(value) for value in DELTA_V_VALUES],
       xlabel="Along-track impulse (m/s)", ylabel="Truth-gate rejection among selected measurements (%)",
       title="Tagged measurement rejection rate")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "truth_gate_rejection_rate.png", dpi=200, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
## Takeaways

1. **The near-zero difference in Smith-format covariance plots is not evidence of near-zero maneuver impact.** At 10 m/s, final realized error is tens of kilometers while final tagged covariance stays close to its 0 m/s control.
2. **Advanced Greedy retains almost the entire unmodeled displacement as filter error.** Frozen PPO partially corrects some objects through post-burn measurements, but its median and upper-tail errors remain large.
3. **Large error occurs even when most selected measurements are not rejected by the 4° truth gate.** Staying in the pointing cell is therefore a weak robustness test.
4. **Research gap supported by this diagnostic:** covariance-minimizing tasking can appear successful under its internal uncertainty metric while state accuracy degrades under unmodeled maneuvers.
5. **Scope caveat:** these are 10 paired diagnostic seeds. Promote exact effect sizes to thesis results only after the same locked analysis is run on the planned final seed count.
"""
    ),
]

nbf.write(notebook, OUTPUT_PATH)
print(OUTPUT_PATH)
