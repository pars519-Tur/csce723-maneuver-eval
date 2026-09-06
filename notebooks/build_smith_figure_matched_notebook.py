from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = NOTEBOOK_DIR / "05_smith_figure_matched_maneuver_comparison.ipynb"


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
# Smith Figure-Matched Coast vs. Maneuver Evaluation

## Purpose

This notebook follows Smith's Chapter 2 result sequence directly. For every relevant Smith figure, the **published thesis figure is shown on the left** and the **matched coast-versus-maneuver evaluation is shown on the right**.

The matched plots use only the two methods available as immutable artifacts in this evaluation:

- Advanced Greedy
- Frozen CNNv2 PPO (Smith's CNNv2 Reward 2.9 checkpoint)

The primary maneuver comparison is a simple 10 m/s along-track impulse applied to 20% of the 100 RSOs at 900 s. The final section adds maneuver-specific metrics that Smith did not report: reacquisition and physical truth-FOV rejection. No training, fine-tuning, NIS, or NEES is used.
"""
    ),
    markdown(
        r"""
## Comparison map

| Smith figure | Smith metric | Matched maneuver comparison |
|---|---|---|
| Fig. 2.10 | Single-seed catalog mean covariance history | Coast vs. 20% / 10 m/s, seed 1 |
| Fig. 2.11 | Single-seed catalog median covariance history | Coast vs. 20% / 10 m/s, seed 1 |
| Fig. 2.12 | Final cumulative RSO covariance count | Coast vs. 20% / 10 m/s, seed 1 |
| Fig. 2.13 | 100-run final catalog-mean violin | Coast vs. 20% / 10 m/s, 100 paired seeds |
| Fig. 2.14 | 100-run final catalog-median violin | Coast vs. 20% / 10 m/s, 100 paired seeds |
| Fig. 2.15 | Agent-0 pointing history | Coast vs. maneuver pointing for both retained methods |
"""
    ),
    code(
        r"""
from pathlib import Path
import json
import pickle

import matplotlib.image as mpimg
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
FIGURE_ROOT = PROJECT_ROOT / "results" / "smith_figure_matched"
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
ASSET_ROOT = REPOSITORY_ROOT / "wiki" / "reading-notes" / "assets"

POLICIES = ["Advanced Greedy", "Frozen CNNv2 PPO"]
POLICY_KEYS = {"Advanced Greedy": "advanced_greedy", "Frozen CNNv2 PPO": "frozen_cnnv2_ppo"}
COLORS = {"Advanced Greedy": "#ff7f0e", "Frozen CNNv2 PPO": "#bcbd22"}
PRIMARY_CONDITION = "maneuver_rate20_10mps"
SEED_INDEX = 1
BURN_TIME_SECONDS = 900

REFERENCE_FIGURES = {
    "2.10": ASSET_ROOT / "smith2024-fig2-10-mean-single-run.png",
    "2.11": ASSET_ROOT / "smith2024-fig2-11-median-single-run.png",
    "2.12": ASSET_ROOT / "smith2024-fig2-12-cdf-single-run.png",
    "2.13": ASSET_ROOT / "smith2024-fig2-13-mean-monte-carlo.png",
    "2.14": ASSET_ROOT / "smith2024-fig2-14-median-monte-carlo.png",
    "2.15_baselines": ASSET_ROOT / "smith2024-fig2-15-greedy-baselines.png",
    "2.15_marl": ASSET_ROOT / "smith2024-fig2-15-marl-sawtooth.png",
}
assert all(path.is_file() for path in REFERENCE_FIGURES.values())

validation = json.loads((RESULT_ROOT / "monte_carlo_validation.json").read_text())
assert validation["status"] == "passed"
assert validation["pairing_checks"]["initial_covariance_max_absolute_difference"] == 0.0
assert validation["pairing_checks"]["maneuver_subset_policy_mismatches"] == 0

results = {}
for policy in POLICIES:
    for condition, key in [("Coast", "coast"), ("Maneuver", PRIMARY_CONDITION)]:
        path = RESULT_ROOT / f"{POLICY_KEYS[policy]}_{key}_100seeds.p"
        with path.open("rb") as stream:
            results[(policy, condition)] = pickle.load(stream)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
    "axes.grid": True, "grid.color": "#b0b0b0", "grid.alpha": 0.65,
})


def show_reference(ax, figure_number):
    ax.imshow(mpimg.imread(REFERENCE_FIGURES[figure_number]))
    ax.axis("off")
    ax.set_title(f"Smith 2024, Fig. {figure_number} — published nominal reference", fontsize=12)


def covariance_history(result, episode, reducer):
    table = result["trace_covariance_history_by_rso"][episode]
    values = np.asarray(table["trace_covariance"], dtype=float)
    return np.asarray(table["time_seconds"], dtype=float), reducer(values, axis=1)


def comparison_label(policy, condition):
    return policy.replace("Frozen ", "") + (" — coast" if condition == "Coast" else " — 20% / 10 m/s")


def style_matched_axis(ax, title, ylabel):
    ax.set_title(title)
    ax.set_xlabel("Propagation Time (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 5500)
    ax.axvline(BURN_TIME_SECONDS, color="#303030", linestyle=":", linewidth=1.5, label="Burn at 900 s")
"""
    ),
    markdown(
        r"""
## Smith Fig. 2.10 - Mean trace covariance over time

The right panel uses the same single-seed, catalog-wide mean aggregation. Solid lines are coast runs; dashed lines are their paired maneuver runs. The 900 s line marks the truth impulse.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(18, 6.3), gridspec_kw={"width_ratios": [1, 1.05]})
show_reference(axes[0], "2.10")
for policy in POLICIES:
    for condition, linestyle in [("Coast", "-"), ("Maneuver", "--")]:
        time, values = covariance_history(results[(policy, condition)], SEED_INDEX, np.mean)
        axes[1].plot(time, values, color=COLORS[policy], linestyle=linestyle, linewidth=2,
                     label=comparison_label(policy, condition))
style_matched_axis(axes[1], "Matched mean trace covariance — paired seed 1", "Mean Trace Covariance")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "fig2_10_matched_mean_history.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
## Smith Fig. 2.11 - Median trace covariance over time

This is the direct median counterpart to Fig. 2.10. As in Smith, the catalog median can remain similar even when a maneuvering subset has worse custody outcomes.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(18, 6.3), gridspec_kw={"width_ratios": [1, 1.05]})
show_reference(axes[0], "2.11")
for policy in POLICIES:
    for condition, linestyle in [("Coast", "-"), ("Maneuver", "--")]:
        time, values = covariance_history(results[(policy, condition)], SEED_INDEX, np.median)
        axes[1].plot(time, values, color=COLORS[policy], linestyle=linestyle, linewidth=2,
                     label=comparison_label(policy, condition))
style_matched_axis(axes[1], "Matched median trace covariance — paired seed 1", "Median Trace Covariance")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "fig2_11_matched_median_history.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
## Smith Fig. 2.12 - Ending cumulative covariance distribution

This is a cumulative **RSO count**, matching Smith's y-axis rather than a normalized ECDF. At each covariance threshold, a higher curve means more of the 100 RSOs finish below that threshold.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(18, 6.3), gridspec_kw={"width_ratios": [1, 1.05]})
show_reference(axes[0], "2.12")
for policy in POLICIES:
    for condition, linestyle in [("Coast", "-"), ("Maneuver", "--")]:
        final_covariance = np.sort(np.asarray(results[(policy, condition)]["end_trace_cov_by_episode"][SEED_INDEX]))
        axes[1].plot(final_covariance, np.arange(1, 101), color=COLORS[policy], linestyle=linestyle,
                     linewidth=2, label=comparison_label(policy, condition))
axes[1].set(title="Matched ending cumulative distribution — paired seed 1",
            xlabel="Trace covariance", ylabel="Number of RSOs", xlim=(0, 0.04), ylim=(0, 101))
axes[1].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "fig2_12_matched_cumulative_count.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
## Smith Fig. 2.13 - Monte Carlo final mean covariance

The right panel uses the same nested aggregation as Smith: first compute the final mean across 100 RSOs for each episode, then show the distribution across 100 episodes. Only the two retained methods and their maneuver counterparts are included.
"""
    ),
    code(
        r"""
def violin_with_quartiles(ax, datasets, labels, colors, title, ylabel):
    parts = ax.violinplot(datasets, showextrema=False, quantiles=[[0.25, 0.5, 0.75]] * len(datasets))
    alphas = [0.18, 0.42, 0.18, 0.42]
    for body, color, alpha in zip(parts["bodies"], colors, alphas):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(alpha)
    parts["cquantiles"].set_color([color for color in colors for _ in range(3)])
    parts["cquantiles"].set_linewidth(2)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=25, ha="right")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("Model and environment")


order = [("Advanced Greedy", "Coast"), ("Advanced Greedy", "Maneuver"),
         ("Frozen CNNv2 PPO", "Coast"), ("Frozen CNNv2 PPO", "Maneuver")]
labels = ["Advanced Greedy\ncoast", "Advanced Greedy\nmaneuver",
          "CNNv2 Reward 2.9\ncoast", "CNNv2 Reward 2.9\nmaneuver"]
violin_colors = [COLORS[policy] for policy, _ in order]
mean_data = [np.asarray(results[key]["end_trace_cov_by_episode"], dtype=float).mean(axis=1) for key in order]

fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={"width_ratios": [1, 1.05]})
show_reference(axes[0], "2.13")
violin_with_quartiles(axes[1], mean_data, labels, violin_colors,
                      "Matched final catalog-mean covariance — 100 paired seeds",
                      "Mean trace covariance")
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "fig2_13_matched_mean_violin.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
## Smith Fig. 2.14 - Monte Carlo final median covariance

This repeats the same comparison using each episode's final catalog median, matching Smith Fig. 2.14.
"""
    ),
    code(
        r"""
median_data = [np.median(np.asarray(results[key]["end_trace_cov_by_episode"], dtype=float), axis=1) for key in order]
fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={"width_ratios": [1, 1.05]})
show_reference(axes[0], "2.14")
violin_with_quartiles(axes[1], median_data, labels, violin_colors,
                      "Matched final catalog-median covariance — 100 paired seeds",
                      "Median trace covariance")
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "fig2_14_matched_median_violin.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
## Smith Fig. 2.15 - Pointing behavior

The left column extracts the relevant published Advanced Greedy and CNNv2 Reward 2.9 panels. The right column shows the corresponding agent-0 coast and maneuver histories from the paired seed. Pointing is a policy behavior diagnostic; it does not by itself prove successful reacquisition.
"""
    ),
    code(
        r"""
baseline_image = mpimg.imread(REFERENCE_FIGURES["2.15_baselines"])
marl_image = mpimg.imread(REFERENCE_FIGURES["2.15_marl"])


def pointing_panel(ax, policy, condition):
    result = results[(policy, condition)]
    times = np.asarray(result["proptime"][SEED_INDEX]["agent_0"], dtype=float)
    azimuth = np.asarray(result["azi"][SEED_INDEX]["agent_0"], dtype=float)
    altitude = np.asarray(result["alt"][SEED_INDEX]["agent_0"], dtype=float)
    ax.plot(times, azimuth, color="#1f77b4", linewidth=1.4, label="Azimuth")
    ax_alt = ax.twinx()
    ax_alt.plot(times, altitude, color="#d62728", linestyle="--", linewidth=1.4, label="Altitude")
    ax.set(xlim=(0, 5500), ylim=(0, 390), xlabel="Propagation Time", ylabel="Azimuth")
    ax_alt.set(ylim=(0, 100), ylabel="Altitude")
    ax.axvline(BURN_TIME_SECONDS, color="#303030", linestyle=":", linewidth=1)
    ax.set_title(f"{policy} — {condition.lower()}")
    lines = ax.get_lines()[:1] + ax_alt.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7, loc="upper left")


fig = plt.figure(figsize=(18, 12))
grid = fig.add_gridspec(2, 3, width_ratios=[1.18, 1, 1], hspace=0.28, wspace=0.28)
ax_ref_ag = fig.add_subplot(grid[0, 0])
ax_ref_ag.imshow(baseline_image[:790, baseline_image.shape[1] // 2:, :]); ax_ref_ag.axis("off")
ax_ref_ag.set_title("Smith Fig. 2.15(b) — Advanced Greedy")
ax_ref_ppo = fig.add_subplot(grid[1, 0])
ax_ref_ppo.imshow(marl_image[:820, marl_image.shape[1] // 2:, :]); ax_ref_ppo.axis("off")
ax_ref_ppo.set_title("Smith Fig. 2.15(d) — CNNv2 Reward 2.9")

pointing_panel(fig.add_subplot(grid[0, 1]), "Advanced Greedy", "Coast")
pointing_panel(fig.add_subplot(grid[0, 2]), "Advanced Greedy", "Maneuver")
pointing_panel(fig.add_subplot(grid[1, 1]), "Frozen CNNv2 PPO", "Coast")
pointing_panel(fig.add_subplot(grid[1, 2]), "Frozen CNNv2 PPO", "Maneuver")
fig.suptitle("Smith pointing references and matched agent-0 coast/maneuver behavior — seed 1", fontsize=15)
fig.savefig(FIGURE_ROOT / "fig2_15_matched_pointing.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
## Added maneuver-specific figures

Smith's covariance and pointing figures are preserved above, but they do not directly measure whether a maneuvering object is found again. The following additions supply that missing operational view.
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
all_maneuver_results = {}
for policy in POLICIES:
    for label, key in condition_specs:
        path = RESULT_ROOT / f"{POLICY_KEYS[policy]}_{key}_100seeds.p"
        with path.open("rb") as stream:
            all_maneuver_results[(policy, label)] = pickle.load(stream)

rows = []
for (policy, label), result in all_maneuver_results.items():
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
for offset, policy in [(-0.18, "Frozen CNNv2 PPO"), (0.18, "Advanced Greedy")]:
    subset = outcomes.query("policy == @policy").set_index("condition").loc[[x[0] for x in condition_specs]]
    axes[0].bar(x + offset, subset["not_reacquired_percent"], 0.34, color=COLORS[policy], label=policy)
    axes[1].bar(x + offset, subset["observed_median_minutes"], 0.34, color=COLORS[policy], label=policy)
    axes[2].bar(x + offset, subset["truth_gate_rejection_percent"], 0.34, color=COLORS[policy], label=policy)
tick_labels = ["5%\n10 m/s", "10%\n10 m/s", "20%\n10 m/s", "20%\n100 m/s"]
titles = ["Not reacquired by episode end", "Observed-only median reacquisition", "Truth-FOV rejected selections"]
ylabels = ["Maneuvering RSO cases (%)", "Minutes after burn", "Selected events (%)"]
for ax, title, ylabel in zip(axes, titles, ylabels):
    ax.set_xticks(x, tick_labels); ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(fontsize=7)
fig.suptitle("Maneuver-specific additions absent from Smith's nominal figure set", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "added_maneuver_metrics.png", dpi=180, bbox_inches="tight")
plt.show()

outcomes.style.format({"not_reacquired_percent": "{:.1f}", "observed_median_minutes": "{:.1f}",
                       "truth_gate_rejection_percent": "{:.2f}"})
"""
    ),
    markdown(
        r"""
## Interpretation

- The Smith-matched mean, median, cumulative-count, and violin panels show why the 10 m/s maneuver can look almost nominal at the full-catalog level.
- The maneuver-specific panels reveal the hidden gap: some maneuvering RSOs are not successfully observed again before the episode ends.
- Frozen CNNv2 PPO retains better zero-shot reacquisition behavior than Advanced Greedy, but neither method provides reliable custody of every maneuvering RSO even under this simple single-impulse test.
- The 100 m/s condition remains a separate stress test and is not mixed into the Smith-matched primary 10 m/s figures.
"""
    ),
]

nbf.write(notebook, OUTPUT_PATH)
print(OUTPUT_PATH)
