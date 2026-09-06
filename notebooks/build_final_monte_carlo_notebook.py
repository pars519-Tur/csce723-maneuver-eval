from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = NOTEBOOK_DIR / "04_final_monte_carlo_comparison.ipynb"


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
# Frozen Smith Policies Under Unseen RSO Maneuvers

## tl;dr

Across **100 paired seeds per condition**, the 10 m/s primary maneuver is weakly observable at the 4° sensor-FOV scale and produces negligible changes in aggregate terminal covariance. At 20% maneuver prevalence, the paired mean change is only **+8.28e-7** for frozen CNNv2 PPO and **-1.26e-7** for advanced-greedy.

The 100 m/s stress test exposes a clearer degradation: mean terminal covariance increases by **+2.18e-5 (0.21%)** for PPO and **+7.62e-5 (0.65%)** for advanced-greedy relative to each policy's paired coast run.

The clearest zero-shot gap is reacquisition. Across the 10 m/s primary rates, **21.3–23.6%** of PPO maneuvering-RSO cases and **54.0–54.6%** of advanced-greedy cases receive no successful post-burn observation before episode end. Among observed cases, median reacquisition is about **8–9 min** for PPO versus **36–39 min** for advanced-greedy. These results evaluate the frozen policies directly; no training, fine-tuning, NIS, or NEES is used.
"""
    ),
    markdown(
        r"""
## Context & Methods

This is a paired, inference-only evaluation of Smith's unchanged frozen CNNv2 PPO checkpoint and unchanged advanced-greedy method. An external wrapper changes only maneuvering-object truth and physically gates their measurements using the 4° FOV. Smith's scheduling, rewards, observations, filters, source files, checkpoints, and nominal artifacts remain untouched.

### Key assumptions

- Scenario: 3 sensors, 100 RSOs, 5400 s; burn at 900 s.
- Primary conditions: 0%, 5%, 10%, and 20% maneuver prevalence; 10 m/s along-track impulse.
- Stress condition: 20% prevalence; 100 m/s along-track impulse.
- Each condition uses seeds 000–099, common initial covariance, and policy-matched maneuver subsets.
- Reacquisition time is first successful post-burn measurement minus burn time. Non-reacquired objects are right-censored at 4500 s.
- Aggregate covariance effects are paired against the same policy's coast episode. Findings are descriptive robustness evidence, not a claim about retrained performance.
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
from IPython.display import display


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "experiments" / "csce723_maneuver_eval").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate repository root")


REPOSITORY_ROOT = find_repository_root(Path.cwd().resolve())
PROJECT_ROOT = REPOSITORY_ROOT / "experiments" / "csce723_maneuver_eval"
RESULT_ROOT = PROJECT_ROOT / "results" / "monte_carlo"
ANALYSIS_ROOT = PROJECT_ROOT / "results" / "final_analysis"
FIGURE_DIR = ANALYSIS_ROOT / "figures"
ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

POLICIES = ["Frozen CNNv2 PPO", "Advanced Greedy"]
POLICY_KEYS = {"Frozen CNNv2 PPO": "frozen_cnnv2_ppo", "Advanced Greedy": "advanced_greedy"}
POLICY_COLORS = {"Frozen CNNv2 PPO": "#2563A6", "Advanced Greedy": "#D17A22"}
CONDITIONS = {
    "Coast": "coast",
    "5% / 10 m/s": "maneuver_rate05_10mps",
    "10% / 10 m/s": "maneuver_rate10_10mps",
    "20% / 10 m/s": "maneuver_rate20_10mps",
    "20% / 100 m/s stress": "stress_rate20_100mps",
}
PRIMARY_CONDITIONS = ["Coast", "5% / 10 m/s", "10% / 10 m/s", "20% / 10 m/s"]
MANEUVER_CONDITIONS = ["5% / 10 m/s", "10% / 10 m/s", "20% / 10 m/s", "20% / 100 m/s stress"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#263238",
    "axes.labelcolor": "#263238", "text.color": "#263238", "font.size": 10,
})

validation = json.loads((RESULT_ROOT / "monte_carlo_validation.json").read_text())
assert validation["status"] == "passed"
assert validation["pairing_checks"]["initial_covariance_max_absolute_difference"] == 0.0
assert validation["pairing_checks"]["maneuver_subset_policy_mismatches"] == 0
validation["pairing_checks"]
"""
    ),
    markdown("## Data"),
    code(
        r"""
results = {}
for policy in POLICIES:
    for condition_label, condition_key in CONDITIONS.items():
        path = RESULT_ROOT / f"{POLICY_KEYS[policy]}_{condition_key}_100seeds.p"
        with path.open("rb") as stream:
            results[(policy, condition_label)] = pickle.load(stream)

assert all(len(result["pair_ids"]) == 100 for result in results.values())
print(f"Loaded {len(results)} result files and {sum(len(x['pair_ids']) for x in results.values()):,} episodes.")
"""
    ),
    code(
        r"""
def episode_total_reward_per_agent(result, episode):
    totals = [np.sum(values) for values in result["rew"][episode].values()]
    return float(np.mean(totals))


episode_rows = []
for (policy, condition), result in results.items():
    terminal = np.asarray(result["end_trace_cov_by_episode"], dtype=float)
    for episode, pair_id in enumerate(result["pair_ids"]):
        episode_rows.append({
            "policy": policy,
            "condition": condition,
            "pair_id": pair_id,
            "final_mean_covariance": terminal[episode].mean(),
            "final_median_covariance": np.median(terminal[episode]),
            "final_p90_covariance": np.quantile(terminal[episode], 0.90),
            "mean_total_reward_per_agent": episode_total_reward_per_agent(result, episode),
        })
episode_metrics = pd.DataFrame(episode_rows)

headline = (episode_metrics.groupby(["policy", "condition"], sort=False)
            .agg(episodes=("pair_id", "count"),
                 mean_final_covariance=("final_mean_covariance", "mean"),
                 mean_final_median_covariance=("final_median_covariance", "mean"),
                 mean_final_p90_covariance=("final_p90_covariance", "mean"),
                 mean_reward_per_agent=("mean_total_reward_per_agent", "mean"))
            .reset_index())
headline.to_csv(ANALYSIS_ROOT / "final_condition_summary.csv", index=False)
display(headline.style.format({
    "mean_final_covariance": "{:.8f}", "mean_final_median_covariance": "{:.8f}",
    "mean_final_p90_covariance": "{:.8f}", "mean_reward_per_agent": "{:.2f}",
}))
"""
    ),
    markdown(
        r"""
## Results

### Smith-compatible terminal covariance distributions

These retain Smith's final mean and median trace-covariance views, now stratified by maneuver condition. The stress test is kept out of this primary panel so it does not visually dominate the 10 m/s comparison.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")
metrics = [("final_mean_covariance", "Final mean trace covariance"),
           ("final_median_covariance", "Final median trace covariance")]
condition_colors = ["#B8C0CC", "#9CB8D6", "#5F8FC4", "#2563A6"]
for row, (metric, title) in enumerate(metrics):
    for col, policy in enumerate(POLICIES):
        ax = axes[row, col]
        data = [episode_metrics.query("policy == @policy and condition == @condition")[metric].to_numpy()
                for condition in PRIMARY_CONDITIONS]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], condition_colors):
            body.set_facecolor(color); body.set_edgecolor("#263238"); body.set_alpha(0.55)
        parts["cmedians"].set_color("#263238")
        ax.set_title(f"{policy} — {title}")
        ax.set_xticks(range(1, 5), ["Coast", "5%", "10%", "20%"])
        ax.set_ylabel("Trace covariance")
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8)
fig.suptitle("Smith-compatible terminal covariance distributions — 100 paired seeds, 10 m/s", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "smith_compatible_terminal_covariance.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("### Smith-compatible reward distribution"),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, policy in zip(axes, POLICIES):
    data = [episode_metrics.query("policy == @policy and condition == @condition")["mean_total_reward_per_agent"].to_numpy()
            for condition in PRIMARY_CONDITIONS]
    boxes = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], condition_colors):
        patch.set_facecolor(color); patch.set_alpha(0.65)
    ax.set_title(policy)
    ax.set_xticks(range(1, 5), ["Coast", "5%", "10%", "20%"])
    ax.set_ylabel("Mean total reward per agent")
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8)
fig.suptitle("Smith-compatible total reward — 100 paired seeds, 10 m/s", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "smith_compatible_reward.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
### Paired change relative to coast

Each dot/interval is the mean paired episode change with a deterministic seed-level bootstrap 95% interval. Pairing removes between-seed initial-condition variation. The 10 m/s effects sit close to zero; this is an observed result, not a plotting failure.
"""
    ),
    code(
        r"""
def bootstrap_mean_ci(values, seed=723, draws=10_000):
    values = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


paired_rows = []
for policy in POLICIES:
    coast = episode_metrics.query("policy == @policy and condition == 'Coast'").set_index("pair_id")
    coast_rso = np.asarray(results[(policy, "Coast")]["end_trace_cov_by_episode"], dtype=float)
    for condition in MANEUVER_CONDITIONS:
        maneuver = episode_metrics.query("policy == @policy and condition == @condition").set_index("pair_id")
        result = results[(policy, condition)]
        maneuver_rso = np.asarray(result["end_trace_cov_by_episode"], dtype=float)
        for metric in ["final_mean_covariance", "final_median_covariance", "final_p90_covariance"]:
            delta = (maneuver.loc[coast.index, metric] - coast[metric]).to_numpy()
            low, high = bootstrap_mean_ci(delta)
            paired_rows.append({"policy": policy, "condition": condition, "metric": metric,
                                "mean_paired_change": delta.mean(), "ci_low": low, "ci_high": high})
        selected_delta = np.array([
            np.mean(maneuver_rso[i, ids] - coast_rso[i, ids])
            for i, ids in enumerate(result["maneuver_rso_ids"])
        ])
        low, high = bootstrap_mean_ci(selected_delta)
        paired_rows.append({"policy": policy, "condition": condition,
                            "metric": "maneuvering_subset_mean_covariance",
                            "mean_paired_change": selected_delta.mean(), "ci_low": low, "ci_high": high})

paired_effects = pd.DataFrame(paired_rows)
paired_effects.to_csv(ANALYSIS_ROOT / "paired_effect_summary.csv", index=False)
display(paired_effects.pivot_table(index=["policy", "condition"], columns="metric",
                                   values="mean_paired_change").style.format("{:+.3e}"))
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
primary_maneuvers = PRIMARY_CONDITIONS[1:]
for ax, policy in zip(axes, POLICIES):
    subset = paired_effects.query("policy == @policy and metric == 'final_mean_covariance' and condition in @primary_maneuvers")
    x = np.array([5, 10, 20])
    y = subset["mean_paired_change"].to_numpy()
    yerr = np.vstack([y - subset["ci_low"].to_numpy(), subset["ci_high"].to_numpy() - y])
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=4,
                color=POLICY_COLORS[policy])
    ax.axhline(0, color="#263238", linewidth=1)
    ax.set(title=policy, xlabel="Maneuvering RSOs (%)", ylabel="Paired change in final mean trace covariance")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(color="#D9DEE5", linewidth=0.8, alpha=0.8)
fig.suptitle("Primary 10 m/s dose response — mean and seed-bootstrap 95% interval", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "primary_paired_covariance_dose_response.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
### Why covariance does not simply decrease

Smith's covariance includes propagation/process growth between measurements. It drops when a valid observation updates the filter, then grows during unobserved propagation. The following paired trajectory uses the maneuvering subset from seed 0; the burn line marks a truth change, not an automatic covariance reset.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, policy in zip(axes, POLICIES):
    coast = results[(policy, "Coast")]["trace_covariance_history_by_rso"][0]
    maneuver_result = results[(policy, "20% / 10 m/s")]
    maneuver = maneuver_result["trace_covariance_history_by_rso"][0]
    ids = np.asarray(maneuver_result["maneuver_rso_ids"][0], dtype=int)
    ax.plot(np.asarray(coast["time_seconds"]) / 60,
            np.asarray(coast["trace_covariance"])[:, ids].mean(axis=1),
            label="Paired coast", color="#6B7280", linewidth=1.8)
    ax.plot(np.asarray(maneuver["time_seconds"]) / 60,
            np.asarray(maneuver["trace_covariance"])[:, ids].mean(axis=1),
            label="20% maneuver / 10 m/s", color=POLICY_COLORS[policy], linewidth=1.8)
    ax.axvline(15, color="#263238", linestyle="--", linewidth=1, label="Burn (15 min)")
    ax.set(title=f"{policy} — seed 0 maneuvering subset", xlabel="Episode time (min)",
           ylabel="Mean trace covariance")
    ax.grid(color="#D9DEE5", linewidth=0.8, alpha=0.8)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "representative_maneuver_subset_covariance_history.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("### Reacquisition and non-observation"),
    code(
        r"""
outcome_rows = []
for policy in POLICIES:
    for condition in MANEUVER_CONDITIONS:
        result = results[(policy, condition)]
        reacquisition = np.concatenate([
            np.asarray(result["reacquisition_time"][episode], dtype=float)[ids]
            for episode, ids in enumerate(result["maneuver_rso_ids"])
        ])
        selected = received = rejected = 0
        selected_separations = []
        for measurement, truth in zip(result["measurement_events"], result["truth_fov_events"]):
            selected_mask = np.asarray(measurement["selected_by_smith_estimate"], dtype=bool)
            truth_mask = np.asarray(measurement["truth_in_fov"], dtype=bool)
            received_mask = np.asarray(measurement["measurement_received"], dtype=bool)
            selected += int(selected_mask.sum())
            received += int(received_mask.sum())
            rejected += int(np.sum(selected_mask & ~truth_mask))
            truth_selected = np.asarray(truth["selected_by_smith_estimate"], dtype=bool)
            selected_separations.extend(np.asarray(truth["angular_separation_deg"], dtype=float)[truth_selected])
        observed = np.isfinite(reacquisition)
        outcome_rows.append({
            "policy": policy, "condition": condition, "maneuver_rso_cases": len(reacquisition),
            "not_reacquired_rate": np.mean(~observed),
            "median_reacquisition_seconds_observed_only": np.nanmedian(reacquisition),
            "selected_events": selected, "received_events": received,
            "truth_gate_rejections": rejected, "truth_gate_rejection_rate": rejected / selected,
            "median_selected_angular_separation_deg": np.median(selected_separations),
            "p95_selected_angular_separation_deg": np.quantile(selected_separations, 0.95),
        })
outcomes = pd.DataFrame(outcome_rows)
outcomes.to_csv(ANALYSIS_ROOT / "maneuver_outcome_summary.csv", index=False)
display(outcomes.style.format({
    "not_reacquired_rate": "{:.1%}", "median_reacquisition_seconds_observed_only": "{:.1f}",
    "truth_gate_rejection_rate": "{:.2%}", "median_selected_angular_separation_deg": "{:.3f}",
    "p95_selected_angular_separation_deg": "{:.3f}",
}))
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(MANEUVER_CONDITIONS))
for offset, policy in zip([-0.18, 0.18], POLICIES):
    subset = outcomes.set_index(["policy", "condition"]).loc[policy].loc[MANEUVER_CONDITIONS]
    axes[0].bar(x + offset, 100 * subset["not_reacquired_rate"], width=0.34,
                color=POLICY_COLORS[policy], alpha=0.82, label=policy)
    axes[1].bar(x + offset, subset["median_reacquisition_seconds_observed_only"] / 60, width=0.34,
                color=POLICY_COLORS[policy], alpha=0.82, label=policy)
labels = ["5%\n10 m/s", "10%\n10 m/s", "20%\n10 m/s", "20%\n100 m/s"]
axes[0].set(title="Maneuvering RSOs not reacquired by episode end", ylabel="Not reacquired (%)")
axes[1].set(title="Observed-only median reacquisition time", ylabel="Minutes after burn")
for ax in axes:
    ax.set_xticks(x, labels); ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "reacquisition_summary.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
The median panel is explicitly **observed-only** and must be read together with the non-reacquired rate. The next plot retains censored cases in a survival-style curve, so the terminal plateau is the fraction still not reacquired.
"""
    ),
    code(
        r"""
def empirical_not_reacquired_curve(times, censor_time=4500.0):
    times = np.asarray(times, dtype=float)
    event_times = np.sort(times[np.isfinite(times)])
    x = np.r_[0.0, event_times, censor_time]
    y = np.r_[1.0, 1.0 - np.arange(1, len(event_times) + 1) / len(times),
              1.0 - len(event_times) / len(times)]
    return x, y


fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, condition in zip(axes, ["20% / 10 m/s", "20% / 100 m/s stress"]):
    for policy in POLICIES:
        result = results[(policy, condition)]
        times = np.concatenate([np.asarray(result["reacquisition_time"][i])[ids]
                                for i, ids in enumerate(result["maneuver_rso_ids"])])
        x_curve, y_curve = empirical_not_reacquired_curve(times)
        ax.step(x_curve / 60, 100 * y_curve, where="post", color=POLICY_COLORS[policy],
                linewidth=2, label=policy)
    ax.set(title=condition, xlabel="Minutes after burn", ylabel="Not yet reacquired (%)")
    ax.set_xlim(0, 75); ax.set_ylim(0, 100)
    ax.grid(color="#D9DEE5", linewidth=0.8, alpha=0.8); ax.legend(fontsize=8)
fig.suptitle("Reacquisition curves with episode-end censoring", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "reacquisition_censored_curves.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("### Physical truth-gate rejections and maneuver observability"),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(MANEUVER_CONDITIONS))
for offset, policy in zip([-0.18, 0.18], POLICIES):
    subset = outcomes.set_index(["policy", "condition"]).loc[policy].loc[MANEUVER_CONDITIONS]
    axes[0].bar(x + offset, 100 * subset["truth_gate_rejection_rate"], width=0.34,
                color=POLICY_COLORS[policy], alpha=0.82, label=policy)
    axes[1].bar(x + offset, subset["p95_selected_angular_separation_deg"], width=0.34,
                color=POLICY_COLORS[policy], alpha=0.82, label=policy)
axes[0].set(title="Estimate-selected measurements rejected by truth FOV", ylabel="Rejected selections (%)")
axes[1].set(title="95th percentile selected-event truth separation", ylabel="Angular separation (deg)")
axes[1].axhline(4.0, color="#263238", linestyle="--", linewidth=1.2, label="4° FOV gate")
for ax in axes:
    ax.set_xticks(x, labels); ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "truth_gate_and_observability.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
### Stress test, shown separately

The 100 m/s condition is a sensitivity/stress test, not part of the primary 10 m/s dose-response. Separating it avoids implying that the two Δv regimes belong to one linear scale.
"""
    ),
    code(
        r"""
stress = paired_effects.query("condition == '20% / 100 m/s stress' and metric in ['final_mean_covariance', 'final_median_covariance', 'final_p90_covariance']")
metric_labels = {"final_mean_covariance": "Mean", "final_median_covariance": "Median", "final_p90_covariance": "90th percentile"}
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(3)
for offset, policy in zip([-0.18, 0.18], POLICIES):
    subset = stress.query("policy == @policy").set_index("metric").loc[list(metric_labels)]
    y = subset["mean_paired_change"].to_numpy()
    yerr = np.vstack([y - subset["ci_low"].to_numpy(), subset["ci_high"].to_numpy() - y])
    ax.bar(x + offset, y, width=0.34, color=POLICY_COLORS[policy], alpha=0.82, label=policy)
    ax.errorbar(x + offset, y, yerr=yerr, fmt="none", color="#263238", capsize=3, linewidth=1)
ax.axhline(0, color="#263238", linewidth=1)
ax.set_xticks(x, list(metric_labels.values()))
ax.set(title="20% maneuver prevalence / 100 m/s stress test", ylabel="Paired terminal covariance change")
ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8); ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "stress_paired_covariance_effects.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        r"""
## Takeaways

1. **The primary 10 m/s maneuver is too subtle to create a large aggregate covariance separation in this geometry and horizon.** The observed paired effects are near zero even at 20% prevalence, consistent with the earlier angular calibration.
2. **The physical truth gate becomes consequential under the 100 m/s stress maneuver.** Rejection rates rise from below 0.6% in all 10 m/s conditions to about 5.4% for PPO and 6.6% for advanced-greedy.
3. **Reacquisition reveals a robustness gap that aggregate covariance obscures.** Advanced-greedy leaves roughly half of maneuvering-object cases unreacquired and has much longer observed-only reacquisition times; PPO leaves roughly one fifth unreacquired.
4. **Stress covariance degradation is larger for advanced-greedy.** At 20% / 100 m/s, its paired mean terminal increase is about 3.5 times the PPO increase, while policy-specific coast levels remain different.
5. **Interpretation boundary:** this experiment measures zero-shot behavior of Smith's frozen methods. It does not show whether retraining, maneuver-aware observations, or a redesigned reward would close the gap. NIS and NEES remain intentionally out of scope.
"""
    ),
]

nbf.write(notebook, OUTPUT_PATH)
print(OUTPUT_PATH)
