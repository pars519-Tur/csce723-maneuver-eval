from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = NOTEBOOK_DIR / "01_smith_baseline_visualization.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}

notebook["cells"] = [
    markdown(
        """
# Smith Baseline Visualization - 3 Sensors / 100 RSOs

## tl;dr

This notebook reads the existing Smith result artifacts; it does **not** retrain a policy or rerun the environment. Across the stored 100 nominal episodes, the frozen CNNv2 PPO result has 9.7% lower mean final trace covariance and 20.2% lower median final trace covariance than advanced-greedy under Smith's aggregation method.

These nominal results are a visualization and data-contract baseline only. They provide no evidence yet about maneuver robustness.
"""
    ),
    markdown(
        """
## Context & Methods

The purpose is to preserve the applicable plots and aggregation logic from Smith's visualization notebooks before building the separate maneuver-aware evaluation environment. All inputs remain read-only.

### Key Assumptions

- Scenario: 3 ground sensors and 100 RSOs.
- Policies: frozen CNNv2 PPO (`cnn_v2_reward_1`) and advanced-greedy.
- Each stored result contains 100 evaluation episodes.
- Final episode-level covariance follows Smith's method: concatenate asynchronous agent events, sort by propagation time, and retain the final state.
- `nsats` is labeled conservatively as the number of unique RSOs entering each sensor's field of regard (FOR), based on the Smith environment implementation.
- `end_trace_cov_list` contains one 100-RSO array per result file, not one array per episode. Its ECDF is therefore shown only as the final stored episode diagnostic and is not used as a 100-episode tail estimate.
"""
    ),
    markdown("## Data\n\n### 1. Load read-only Smith result artifacts"),
    code(
        """
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "try_code" / "MARLSSA").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the thesis-wiki repository root")


REPOSITORY_ROOT = find_repository_root(Path.cwd().resolve())
PROJECT_ROOT = REPOSITORY_ROOT / "experiments" / "csce723_maneuver_eval"
FIGURE_DIR = PROJECT_ROOT / "results" / "baseline_figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATHS = {
    "Frozen CNNv2 PPO": REPOSITORY_ROOT / "try_code" / "MARLSSA" / "cnn_v2_reward_1" / "100_sat_3_agents.p",
    "Advanced Greedy": REPOSITORY_ROOT / "try_code" / "MARLSSA" / "advgreedy" / "100_sat_3_agents.p",
}

results = {}
for policy_name, result_path in RESULT_PATHS.items():
    with result_path.open("rb") as stream:
        results[policy_name] = pickle.load(stream)

pd.DataFrame(
    {
        "policy": RESULT_PATHS.keys(),
        "source": [str(path.relative_to(REPOSITORY_ROOT)) for path in RESULT_PATHS.values()],
        "episodes": [len(results[name]["proptime"]) for name in RESULT_PATHS],
    }
)
"""
    ),
    markdown("### 2. Validate fields, episode counts, and aligned histories"),
    code(
        """
EXPECTED_FIELDS = {
    "proptime",
    "meancovall",
    "mediancovall",
    "rew",
    "nsats",
    "alt",
    "azi",
    "start_trace_cov_list",
    "end_trace_cov_list",
}

validation_rows = []
for policy_name, policy_result in results.items():
    missing_fields = EXPECTED_FIELDS - set(policy_result)
    assert not missing_fields, f"{policy_name} missing fields: {sorted(missing_fields)}"
    assert len(policy_result["proptime"]) == 100
    assert np.asarray(policy_result["end_trace_cov_list"]).shape == (100,)

    aligned = True
    finite = True
    for episode_index in range(100):
        for agent_id, times in policy_result["proptime"][episode_index].items():
            expected_length = len(times)
            for field in ("meancovall", "mediancovall", "rew", "nsats", "alt", "azi"):
                values = np.asarray(policy_result[field][episode_index][agent_id])
                aligned &= len(values) == expected_length
                finite &= bool(np.all(np.isfinite(values)))

    validation_rows.append(
        {
            "policy": policy_name,
            "expected fields present": True,
            "100 episodes": True,
            "history lengths aligned": aligned,
            "history values finite": finite,
            "stored final RSO array length": len(policy_result["end_trace_cov_list"]),
        }
    )

validation_table = pd.DataFrame(validation_rows).set_index("policy")
assert validation_table.drop(columns="stored final RSO array length").to_numpy().all()
validation_table
"""
    ),
    markdown("### 3. Recreate Smith-compatible episode summaries"),
    code(
        """
def merged_episode_history(policy_result, field, episode_index):
    times = []
    values = []
    for agent_id, agent_times in policy_result["proptime"][episode_index].items():
        times.extend(agent_times)
        values.extend(policy_result[field][episode_index][agent_id])
    order = np.argsort(times, kind="stable")
    return np.asarray(times)[order], np.asarray(values)[order]


episode_rows = []
for policy_name, policy_result in results.items():
    for episode_index in range(100):
        _, mean_covariance = merged_episode_history(policy_result, "meancovall", episode_index)
        _, median_covariance = merged_episode_history(policy_result, "mediancovall", episode_index)

        agent_rewards = [
            np.sum(policy_result["rew"][episode_index][agent_id])
            for agent_id in policy_result["rew"][episode_index]
        ]
        terminal_unique_for = [
            policy_result["nsats"][episode_index][agent_id][-1]
            for agent_id in policy_result["nsats"][episode_index]
        ]

        episode_rows.append(
            {
                "policy": policy_name,
                "episode": episode_index,
                "first_recorded_mean_trace_covariance": mean_covariance[0],
                "final_mean_trace_covariance": mean_covariance[-1],
                "first_recorded_median_trace_covariance": median_covariance[0],
                "final_median_trace_covariance": median_covariance[-1],
                "mean_total_reward_per_agent": np.mean(agent_rewards),
                "mean_terminal_unique_rso_in_for_per_agent": np.mean(terminal_unique_for),
            }
        )

episode_summary = pd.DataFrame(episode_rows)
summary_table = episode_summary.groupby("policy").agg(
    episodes=("episode", "count"),
    mean_final_mean_covariance=("final_mean_trace_covariance", "mean"),
    median_final_mean_covariance=("final_mean_trace_covariance", "median"),
    mean_final_median_covariance=("final_median_trace_covariance", "mean"),
    mean_total_reward_per_agent=("mean_total_reward_per_agent", "mean"),
    mean_terminal_unique_rso_in_for_per_agent=("mean_terminal_unique_rso_in_for_per_agent", "mean"),
)
summary_table
"""
    ),
    markdown(
        """
## Results

### 4. First-recorded versus final covariance

The source artifacts do not retain the exact time-zero covariance array for every episode. Accordingly, this paired view uses the first asynchronous action-completion record and the final record in each episode. Thin lines are individual episodes; thick lines are policy averages.
"""
    ),
    code(
        """
POLICY_ORDER = ["Frozen CNNv2 PPO", "Advanced Greedy"]
COLORS = {"Frozen CNNv2 PPO": "#2563A6", "Advanced Greedy": "#D17A22"}

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
metric_pairs = [
    (
        "first_recorded_mean_trace_covariance",
        "final_mean_trace_covariance",
        "Mean trace covariance",
    ),
    (
        "first_recorded_median_trace_covariance",
        "final_median_trace_covariance",
        "Median trace covariance",
    ),
]

for ax, (first_field, final_field, panel_title) in zip(axes, metric_pairs):
    for policy in POLICY_ORDER:
        policy_rows = episode_summary.loc[episode_summary.policy == policy]
        for _, episode_row in policy_rows.iterrows():
            ax.plot(
                [0, 1],
                [episode_row[first_field], episode_row[final_field]],
                color=COLORS[policy],
                linewidth=0.7,
                alpha=0.08,
            )

        policy_average = policy_rows[[first_field, final_field]].mean().to_numpy()
        percent_change = 100 * (policy_average[1] / policy_average[0] - 1)
        ax.plot(
            [0, 1],
            policy_average,
            color=COLORS[policy],
            marker="o",
            markersize=6,
            linewidth=3,
            label=f"{policy} ({percent_change:.1f}%)",
        )

    ax.set_title(panel_title)
    ax.set_xticks([0, 1], ["First recorded", "Final"])
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Trace covariance")
fig.suptitle("First-recorded versus final covariance across 100 nominal episodes", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "smith_first_recorded_vs_final_covariance.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("### 5. Smith-equivalent final mean covariance distribution"),
    code(
        """
POLICY_ORDER = ["Frozen CNNv2 PPO", "Advanced Greedy"]
COLORS = {"Frozen CNNv2 PPO": "#2563A6", "Advanced Greedy": "#D17A22"}


def style_distribution_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Policy")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


fig, ax = plt.subplots(figsize=(8, 5))
data = [
    episode_summary.loc[episode_summary.policy == policy, "final_mean_trace_covariance"].to_numpy()
    for policy in POLICY_ORDER
]
parts = ax.violinplot(data, showmedians=True, showextrema=True)
for body, policy in zip(parts["bodies"], POLICY_ORDER):
    body.set_facecolor(COLORS[policy])
    body.set_edgecolor("#263238")
    body.set_alpha(0.35)
for key in ("cmedians", "cmins", "cmaxes", "cbars"):
    parts[key].set_color("#263238")
ax.set_xticks([1, 2], POLICY_ORDER)
style_distribution_axis(
    ax,
    "Final mean trace covariance across 100 nominal episodes",
    "Mean trace covariance",
)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "smith_final_mean_covariance_violin.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("### 6. Smith-equivalent final median covariance distribution"),
    code(
        """
fig, ax = plt.subplots(figsize=(8, 5))
data = [
    episode_summary.loc[episode_summary.policy == policy, "final_median_trace_covariance"].to_numpy()
    for policy in POLICY_ORDER
]
parts = ax.violinplot(data, showmedians=True, showextrema=True)
for body, policy in zip(parts["bodies"], POLICY_ORDER):
    body.set_facecolor(COLORS[policy])
    body.set_edgecolor("#263238")
    body.set_alpha(0.35)
for key in ("cmedians", "cmins", "cmaxes", "cbars"):
    parts[key].set_color("#263238")
ax.set_xticks([1, 2], POLICY_ORDER)
style_distribution_axis(
    ax,
    "Final median trace covariance across 100 nominal episodes",
    "Median trace covariance",
)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "smith_final_median_covariance_violin.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """
### 7. Smith-equivalent per-RSO final covariance ECDF

**Important:** each source file retains the per-RSO array only for the final stored episode (seed 99 in Smith's 0-99 evaluation loop). This chart is a single-episode diagnostic, not a 100-episode distribution.
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(8, 5))
for policy in POLICY_ORDER:
    values = np.sort(np.asarray(results[policy]["end_trace_cov_list"]))
    cumulative_fraction = np.arange(1, len(values) + 1) / len(values)
    ax.plot(
        values,
        cumulative_fraction,
        label=policy,
        color=COLORS[policy],
        linewidth=2,
    )
ax.set_title("Per-RSO final trace covariance ECDF - final stored nominal episode")
ax.set_xlabel("Trace covariance")
ax.set_ylabel("Cumulative fraction of 100 RSOs")
ax.set_ylim(0, 1.02)
ax.grid(color="#D9DEE5", linewidth=0.8, alpha=0.8)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "smith_final_rso_covariance_ecdf_last_episode.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("### 8. Smith-equivalent covariance histories for one seeded episode"),
    code(
        """
SELECTED_EPISODE = 0
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
for policy in POLICY_ORDER:
    mean_time, mean_values = merged_episode_history(results[policy], "meancovall", SELECTED_EPISODE)
    median_time, median_values = merged_episode_history(results[policy], "mediancovall", SELECTED_EPISODE)
    axes[0].plot(mean_time, mean_values, label=policy, color=COLORS[policy], linewidth=1.8)
    axes[1].plot(median_time, median_values, label=policy, color=COLORS[policy], linewidth=1.8)

axes[0].set_title("Mean trace covariance over time - nominal episode 0")
axes[0].set_ylabel("Mean trace covariance")
axes[1].set_title("Median trace covariance over time - nominal episode 0")
axes[1].set_xlabel("Propagation time (s)")
axes[1].set_ylabel("Median trace covariance")
for ax in axes:
    ax.grid(color="#D9DEE5", linewidth=0.8, alpha=0.8)
    ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "smith_covariance_histories_episode_0.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("### 9. Smith-equivalent unique-RSO-in-FOR history"),
    code(
        """
fig, ax = plt.subplots(figsize=(9, 5))
line_styles = ["-", "--", ":"]
for policy in POLICY_ORDER:
    for agent_index, agent_id in enumerate(sorted(results[policy]["proptime"][SELECTED_EPISODE])):
        ax.plot(
            results[policy]["proptime"][SELECTED_EPISODE][agent_id],
            results[policy]["nsats"][SELECTED_EPISODE][agent_id],
            color=COLORS[policy],
            linestyle=line_styles[agent_index],
            linewidth=1.5,
            label=f"{policy} - {agent_id}",
        )
ax.set_title("Cumulative unique RSOs entering each sensor FOR - nominal episode 0")
ax.set_xlabel("Propagation time (s)")
ax.set_ylabel("Unique RSOs in FOR")
ax.grid(color="#D9DEE5", linewidth=0.8, alpha=0.8)
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "smith_unique_rso_for_history_episode_0.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("### 10. Smith-equivalent pointing history for sensor 0"),
    code(
        """
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for policy in POLICY_ORDER:
    times = results[policy]["proptime"][SELECTED_EPISODE]["agent_0"]
    axes[0].plot(
        times,
        results[policy]["azi"][SELECTED_EPISODE]["agent_0"],
        label=policy,
        color=COLORS[policy],
        linewidth=1.5,
    )
    axes[1].plot(
        times,
        results[policy]["alt"][SELECTED_EPISODE]["agent_0"],
        label=policy,
        color=COLORS[policy],
        linewidth=1.5,
    )
axes[0].set_title("Sensor 0 azimuth pointing history - nominal episode 0")
axes[0].set_ylabel("Azimuth (deg)")
axes[1].set_title("Sensor 0 altitude pointing history - nominal episode 0")
axes[1].set_xlabel("Propagation time (s)")
axes[1].set_ylabel("Altitude (deg)")
for ax in axes:
    ax.grid(color="#D9DEE5", linewidth=0.8, alpha=0.8)
    ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "smith_pointing_history_agent_0_episode_0.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("### 11. Smith-compatible total reward comparison"),
    code(
        """
fig, ax = plt.subplots(figsize=(8, 5))
reward_data = [
    episode_summary.loc[episode_summary.policy == policy, "mean_total_reward_per_agent"].to_numpy()
    for policy in POLICY_ORDER
]
box = ax.boxplot(reward_data, patch_artist=True, medianprops={"color": "#263238", "linewidth": 2})
for patch, policy in zip(box["boxes"], POLICY_ORDER):
    patch.set_facecolor(COLORS[policy])
    patch.set_alpha(0.35)
ax.set_xticks([1, 2], POLICY_ORDER)
ax.set_title("Mean total reward per agent across 100 nominal episodes")
ax.set_xlabel("Policy")
ax.set_ylabel("Mean total episode reward per agent")
ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "smith_total_reward_boxplot.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """
## Takeaways

- Under Smith's stored nominal 3-sensor/100-RSO evaluation, frozen CNNv2 PPO has a mean final mean trace covariance of approximately 0.01059 versus 0.01173 for advanced-greedy (9.7% lower).
- Mean final median trace covariance is approximately 0.00775 for PPO versus 0.00970 for advanced-greedy (20.2% lower).
- These values validate the baseline plotting and aggregation path; they do not establish maneuver robustness.
- The existing artifacts do not preserve a per-episode final covariance vector for every RSO. The maneuver evaluation schema must store that vector for every episode so 90th-percentile and tail claims are supported across Monte Carlo seeds.
- The next notebook revision will retain these baseline views and add maneuver-rate, maneuvered-subset, missed-observation, and reacquisition plots. NIS is intentionally out of scope.
"""
    ),
]

nbf.write(notebook, OUTPUT_PATH)
print(OUTPUT_PATH)
