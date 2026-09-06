from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = NOTEBOOK_DIR / "06_smith_coast_reproduction_vs_maneuver.ipynb"


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
# Smith Coast Reproduction vs. Maneuver Evaluation

## tl;dr

This notebook answers one direct question: **what changes when Smith's unchanged Advanced Greedy and CNNv2 Reward 2.9 agents are evaluated in the maneuver environment?**

For every available Smith Chapter 2 result figure, the notebook shows:

1. **Coast reproduction** in Smith's original metric/plot format.
2. **Maneuver counterpart** using the same metric, seed aggregation, axes, and visual encoding.
3. **Paired maneuver-minus-coast difference** where that difference can be plotted honestly.

The primary comparison uses the simple 20%-prevalence, 10 m/s along-track maneuver at 900 s. Only the two methods with immutable executable artifacts are shown. The maneuver-specific appendix then adds reacquisition and truth-FOV outcomes that Smith's nominal figures cannot reveal.
"""
    ),
    markdown(
        r"""
## Context & Methods

- Smith source/checkpoints/results remain unchanged.
- Policies: Advanced Greedy and frozen CNNv2 Reward 2.9 PPO.
- Scenario: 3 sensors, 100 RSOs, 5400 s.
- Fig. 2.10-2.12 and Fig. 2.15 use paired seed 1, matching Smith's single-run convention.
- Fig. 2.13-2.14 use 100 paired seeds.
- Maneuver: 20 RSOs, one 10 m/s along-track impulse at 900 s.
- Coast and maneuver panels always share axes.
- The coast runs were previously reconciled against Smith's stored artifacts; maximum history differences remain numerical roundoff scale.
"""
    ),
    code(
        r"""
from pathlib import Path
import json
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
RESULT_ROOT = PROJECT_ROOT / "results" / "monte_carlo"
FIGURE_ROOT = PROJECT_ROOT / "results" / "smith_coast_vs_maneuver"
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

POLICIES = ["Advanced Greedy", "CNNv2 Reward 2.9"]
POLICY_KEYS = {"Advanced Greedy": "advanced_greedy", "CNNv2 Reward 2.9": "frozen_cnnv2_ppo"}
COLORS = {"Advanced Greedy": "#ff7f0e", "CNNv2 Reward 2.9": "#bcbd22"}
SEED_INDEX = 1
BURN_TIME_SECONDS = 900.0
MANEUVER_KEY = "maneuver_rate20_10mps"

validation = json.loads((RESULT_ROOT / "monte_carlo_validation.json").read_text())
regression = json.loads((RESULT_ROOT / "coast_100seed_smith_regression.json").read_text())
assert validation["status"] == "passed"
assert validation["pairing_checks"]["initial_covariance_max_absolute_difference"] == 0.0
assert validation["pairing_checks"]["maneuver_subset_policy_mismatches"] == 0

results = {}
for policy in POLICIES:
    for condition, key in [("Coast", "coast"), ("Maneuver", MANEUVER_KEY)]:
        path = RESULT_ROOT / f"{POLICY_KEYS[policy]}_{key}_100seeds.p"
        with path.open("rb") as stream:
            results[(policy, condition)] = pickle.load(stream)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
    "axes.grid": True, "grid.color": "#b0b0b0", "grid.alpha": 0.65,
})


def covariance_history(result, reducer):
    table = result["trace_covariance_history_by_rso"][SEED_INDEX]
    values = np.asarray(table["trace_covariance"], dtype=float)
    return np.asarray(table["time_seconds"], dtype=float), reducer(values, axis=1)


def common_history_grid(policy, reducer):
    coast_t, coast_y = covariance_history(results[(policy, "Coast")], reducer)
    maneuver_t, maneuver_y = covariance_history(results[(policy, "Maneuver")], reducer)
    common_t = np.unique(np.r_[coast_t, maneuver_t])
    coast_step = np.interp(common_t, coast_t, coast_y)
    maneuver_step = np.interp(common_t, maneuver_t, maneuver_y)
    return common_t, coast_step, maneuver_step


def matched_history_figure(reducer, smith_number, metric_name, filename):
    histories = {}
    all_values = []
    for policy in POLICIES:
        for condition in ["Coast", "Maneuver"]:
            histories[(policy, condition)] = covariance_history(results[(policy, condition)], reducer)
            all_values.extend(histories[(policy, condition)][1])
    margin = 0.03 * (max(all_values) - min(all_values))
    shared_ylim = (min(all_values) - margin, max(all_values) + margin)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for ax, condition in zip(axes[:2], ["Coast", "Maneuver"]):
        for policy in POLICIES:
            time, values = histories[(policy, condition)]
            ax.plot(time, values, color=COLORS[policy], linewidth=2, label=policy)
        ax.axvline(BURN_TIME_SECONDS, color="#303030", linestyle=":", linewidth=1.4,
                   label="Burn time" if condition == "Maneuver" else None)
        ax.set(title=f"{condition}: Smith-format Fig. {smith_number}", xlabel="Propagation Time (seconds)",
               ylabel=f"{metric_name} Trace Covariance", xlim=(0, 5500), ylim=shared_ylim)
        ax.legend(fontsize=8)

    for policy in POLICIES:
        time, coast, maneuver = common_history_grid(policy, reducer)
        axes[2].plot(time, maneuver - coast, color=COLORS[policy], linewidth=2, label=policy)
    axes[2].axhline(0, color="#303030", linewidth=1)
    axes[2].axvline(BURN_TIME_SECONDS, color="#303030", linestyle=":", linewidth=1.4)
    axes[2].set(title="Paired difference: maneuver - coast", xlabel="Propagation Time (seconds)",
                ylabel=f"Change in {metric_name.lower()} trace covariance", xlim=(0, 5500))
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Smith Fig. {smith_number} reproduced for coast and repeated under maneuver — paired seed 1",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / filename, dpi=180, bbox_inches="tight")
    plt.show()
"""
    ),
    markdown(
        r"""
## Results

### Smith Fig. 2.10 - Catalog mean covariance history

The first panel is the coast reproduction for the two retained Smith methods. The second repeats the identical plot in the maneuver environment. The third exposes small differences that overlap in the absolute-value panels.
"""
    ),
    code(
        r"""
matched_history_figure(np.mean, "2.10", "Mean", "fig2_10_coast_vs_maneuver.png")
"""
    ),
    markdown(
        r"""
### Smith Fig. 2.11 - Catalog median covariance history

This is the exact mean-to-median counterpart of the preceding comparison.
"""
    ),
    code(
        r"""
matched_history_figure(np.median, "2.11", "Median", "fig2_11_coast_vs_maneuver.png")
"""
    ),
    markdown(
        r"""
### Smith Fig. 2.12 - Ending cumulative RSO count distribution

Both absolute panels use Smith's cumulative RSO-count y-axis. The difference panel evaluates both distributions on the same covariance thresholds: positive means the maneuver run has more RSOs below the threshold; negative means fewer.
"""
    ),
    code(
        r"""
final_values = {(policy, condition): np.sort(np.asarray(results[(policy, condition)]["end_trace_cov_by_episode"][SEED_INDEX], dtype=float))
                for policy in POLICIES for condition in ["Coast", "Maneuver"]}
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.05]})
for ax, condition in zip(axes[:2], ["Coast", "Maneuver"]):
    for policy in POLICIES:
        ax.plot(final_values[(policy, condition)], np.arange(1, 101), color=COLORS[policy], linewidth=2,
                label=policy)
    ax.set(title=f"{condition}: Smith-format Fig. 2.12", xlabel="Trace covariance", ylabel="Number of RSOs",
           xlim=(0, 0.04), ylim=(0, 101))
    ax.legend(fontsize=8, loc="lower right")

thresholds = np.linspace(0, 0.04, 401)
for policy in POLICIES:
    coast_count = np.searchsorted(final_values[(policy, "Coast")], thresholds, side="right")
    maneuver_count = np.searchsorted(final_values[(policy, "Maneuver")], thresholds, side="right")
    axes[2].step(thresholds, maneuver_count - coast_count, where="post", color=COLORS[policy], linewidth=2,
                 label=policy)
axes[2].axhline(0, color="#303030", linewidth=1)
axes[2].set(title="Paired cumulative-count difference", xlabel="Trace covariance",
            ylabel="Maneuver - coast RSO count", xlim=(0, 0.04))
axes[2].legend(fontsize=8)
fig.suptitle("Smith Fig. 2.12 reproduced for coast and repeated under maneuver — paired seed 1", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "fig2_12_coast_vs_maneuver.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
### Smith Fig. 2.13 - 100-run final catalog-mean covariance

The coast and maneuver panels reproduce the Smith violin definition with identical axes. The paired-difference panel preserves each seed's coast-maneuver pairing and is centered tightly around zero for this simple 10 m/s maneuver.
"""
    ),
    code(
        r"""
def violin_quartiles(ax, data, labels, colors, title, ylabel, zero_line=False):
    parts = ax.violinplot(data, showextrema=False, quantiles=[[0.25, 0.5, 0.75]] * len(data))
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.28)
    parts["cquantiles"].set_color([color for color in colors for _ in range(3)])
    parts["cquantiles"].set_linewidth(2)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_title(title); ax.set_ylabel(ylabel)
    if zero_line:
        ax.axhline(0, color="#303030", linewidth=1)


def monte_carlo_violin(metric, smith_number, metric_label, filename):
    values = {}
    for policy in POLICIES:
        for condition in ["Coast", "Maneuver"]:
            terminal = np.asarray(results[(policy, condition)]["end_trace_cov_by_episode"], dtype=float)
            values[(policy, condition)] = metric(terminal, axis=1)
    absolute_values = np.concatenate(list(values.values()))
    margin = 0.04 * np.ptp(absolute_values)
    shared_ylim = (absolute_values.min() - margin, absolute_values.max() + margin)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for ax, condition in zip(axes[:2], ["Coast", "Maneuver"]):
        violin_quartiles(ax, [values[(policy, condition)] for policy in POLICIES], POLICIES,
                           [COLORS[p] for p in POLICIES], f"{condition}: Smith-format Fig. {smith_number}",
                           f"{metric_label} trace covariance")
        ax.set_ylim(shared_ylim)
    differences = [values[(policy, "Maneuver")] - values[(policy, "Coast")] for policy in POLICIES]
    violin_quartiles(axes[2], differences, POLICIES, [COLORS[p] for p in POLICIES],
                      "Paired difference across 100 seeds", f"Change in {metric_label.lower()} trace covariance",
                      zero_line=True)
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle(f"Smith Fig. {smith_number} reproduced for coast and repeated under maneuver — 100 paired seeds",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / filename, dpi=180, bbox_inches="tight")
    plt.show()


monte_carlo_violin(np.mean, "2.13", "Mean", "fig2_13_coast_vs_maneuver.png")
"""
    ),
    markdown(
        r"""
### Smith Fig. 2.14 - 100-run final catalog-median covariance

The same layout is repeated for Smith's final catalog-median metric.
"""
    ),
    code(
        r"""
monte_carlo_violin(np.median, "2.14", "Median", "fig2_14_coast_vs_maneuver.png")
"""
    ),
    markdown(
        r"""
### Smith Fig. 2.15 - Agent-0 pointing behavior

Each row is one Smith method; columns are coast and maneuver. Axes are identical. For this simple maneuver and seed, pointing histories are almost unchanged, which is itself important: the frozen policies do not visibly enter a maneuver-reacquisition behavior mode.
"""
    ),
    code(
        r"""
def pointing_panel(ax, policy, condition):
    result = results[(policy, condition)]
    time = np.asarray(result["proptime"][SEED_INDEX]["agent_0"], dtype=float)
    azimuth = np.asarray(result["azi"][SEED_INDEX]["agent_0"], dtype=float)
    altitude = np.asarray(result["alt"][SEED_INDEX]["agent_0"], dtype=float)
    line_az, = ax.plot(time, azimuth, color="#1f77b4", linewidth=1.4, label="Azimuth")
    ax_alt = ax.twinx()
    line_alt, = ax_alt.plot(time, altitude, color="#d62728", linestyle="--", linewidth=1.4, label="Altitude")
    ax.axvline(BURN_TIME_SECONDS, color="#303030", linestyle=":", linewidth=1.2)
    ax.set(xlim=(0, 5500), ylim=(0, 390), xlabel="Propagation Time", ylabel="Azimuth")
    ax_alt.set(ylim=(0, 100), ylabel="Altitude")
    ax.set_title(f"{policy} — {condition.lower()}")
    ax.legend([line_az, line_alt], ["Azimuth", "Altitude"], fontsize=8, loc="upper left")


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for row, policy in enumerate(POLICIES):
    for col, condition in enumerate(["Coast", "Maneuver"]):
        pointing_panel(axes[row, col], policy, condition)
fig.suptitle("Smith Fig. 2.15 format: agent-0 coast reproduction vs. maneuver pointing — paired seed 1",
             fontsize=14)
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "fig2_15_coast_vs_maneuver.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
## Maneuver-specific appendix

The Smith-format panels show that aggregate covariance and pointing barely change at 10 m/s. They cannot answer whether individual maneuvering RSOs are successfully found again. These additional metrics provide that missing view.
"""
    ),
    code(
        r"""
condition_specs = [
    ("5% / 10 m/s", "maneuver_rate05_10mps"),
    ("10% / 10 m/s", "maneuver_rate10_10mps"),
    ("20% / 10 m/s", "maneuver_rate20_10mps"),
    ("20% / 100 m/s", "stress_rate20_100mps"),
]
rows = []
for policy in POLICIES:
    for label, key in condition_specs:
        path = RESULT_ROOT / f"{POLICY_KEYS[policy]}_{key}_100seeds.p"
        with path.open("rb") as stream:
            result = pickle.load(stream)
        reacquisition = np.concatenate([
            np.asarray(result["reacquisition_time"][episode], dtype=float)[ids]
            for episode, ids in enumerate(result["maneuver_rso_ids"])
        ])
        selected = rejected = 0
        for measurement in result["measurement_events"]:
            selected_mask = np.asarray(measurement["selected_by_smith_estimate"], dtype=bool)
            truth_mask = np.asarray(measurement["truth_in_fov"], dtype=bool)
            selected += int(selected_mask.sum())
            rejected += int(np.sum(selected_mask & ~truth_mask))
        rows.append({
            "policy": policy, "condition": label,
            "not_reacquired_percent": 100 * np.mean(~np.isfinite(reacquisition)),
            "observed_median_minutes": np.nanmedian(reacquisition) / 60,
            "truth_gate_rejection_percent": 100 * rejected / selected,
        })
outcomes = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
x = np.arange(len(condition_specs))
condition_order = [label for label, _ in condition_specs]
for offset, policy in [(-0.18, "CNNv2 Reward 2.9"), (0.18, "Advanced Greedy")]:
    subset = outcomes.query("policy == @policy").set_index("condition").loc[condition_order]
    axes[0].bar(x + offset, subset["not_reacquired_percent"], 0.34, color=COLORS[policy], label=policy)
    axes[1].bar(x + offset, subset["observed_median_minutes"], 0.34, color=COLORS[policy], label=policy)
    axes[2].bar(x + offset, subset["truth_gate_rejection_percent"], 0.34, color=COLORS[policy], label=policy)
tick_labels = ["5%\n10 m/s", "10%\n10 m/s", "20%\n10 m/s", "20%\n100 m/s"]
for ax, title, ylabel in zip(
    axes,
    ["Not reacquired by episode end", "Observed-only median reacquisition", "Truth-FOV rejected selections"],
    ["Maneuvering RSO cases (%)", "Minutes after burn", "Selected events (%)"],
):
    ax.set_xticks(x, tick_labels); ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(fontsize=8)
fig.suptitle("Maneuver-specific outcomes absent from Smith's nominal result figures", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "maneuver_specific_appendix.png", dpi=180, bbox_inches="tight")
plt.show()

outcomes.style.format({"not_reacquired_percent": "{:.1f}", "observed_median_minutes": "{:.1f}",
                       "truth_gate_rejection_percent": "{:.2f}"})
"""
    ),
    markdown(
        r"""
## Takeaways

1. Smith-format coast and maneuver panels look nearly identical for the simple 10 m/s impulse; the difference panels confirm that aggregate covariance changes are small.
2. Agent-0 pointing histories also remain nearly unchanged, indicating no visible maneuver-specific response mode in the frozen policies.
3. This apparent nominal robustness is incomplete: the maneuver-specific appendix shows that about one fifth of CNNv2 cases and more than half of Advanced Greedy cases are not reacquired before episode end.
4. Therefore the key change is not prominent in Smith's original aggregate figures. It emerges only after adding maneuver-conditioned custody and reacquisition metrics.
"""
    ),
]

nbf.write(notebook, OUTPUT_PATH)
print(OUTPUT_PATH)
