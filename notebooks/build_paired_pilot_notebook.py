from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = NOTEBOOK_DIR / "03_paired_pilot_visualization.ipynb"


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
        """
# Paired Coast/Maneuver Engineering Pilot

## tl;dr

The external evaluator reproduces Smith's seed-0 coast histories to within 2.3e-15 and uses identical initial covariance and maneuvering-RSO subsets across policies. The 10 m/s, 10%-maneuver pilot runs end-to-end for frozen CNNv2 PPO and advanced-greedy.

This notebook contains **one paired seed only**. Its numerical policy differences are integration diagnostics, not evidence that one method is more maneuver-robust.
"""
    ),
    markdown(
        """
## Data and validation

- Scenario: 3 sensors, 100 RSOs, 5400 s.
- Maneuver: 10 randomly selected RSOs, 10 m/s along-track impulse at t=900 s.
- Policies: unchanged frozen CNNv2 PPO checkpoint and Smith's advanced-greedy logic.
- No training, fine-tuning, NIS, or NEES.
"""
    ),
    code(
        """
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
PILOT_ROOT = PROJECT_ROOT / "results" / "pilot"
FIGURE_DIR = PILOT_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

paths = {
    ("Advanced Greedy", "Coast"): PILOT_ROOT / "advanced_greedy_coast.p",
    ("Advanced Greedy", "Maneuver"): PILOT_ROOT / "advanced_greedy_maneuver.p",
    ("Frozen CNNv2 PPO", "Coast"): PILOT_ROOT / "frozen_cnnv2_ppo_coast.p",
    ("Frozen CNNv2 PPO", "Maneuver"): PILOT_ROOT / "frozen_cnnv2_ppo_maneuver.p",
}
results = {}
for key, path in paths.items():
    with path.open("rb") as stream:
        results[key] = pickle.load(stream)

validation = json.loads((PILOT_ROOT / "pilot_validation.json").read_text())
assert validation["status"] == "passed"
assert validation["paired_start_covariance_max_absolute_difference"] == 0.0
assert validation["coast_history_max_absolute_difference_from_smith"] < 1e-12
pd.DataFrame(validation["policy_summary"]).T
"""
    ),
    markdown(
        """
## Smith-compatible final covariance view

Mean and median are computed from the complete 100-RSO terminal covariance arrays retained by the new evaluator. Values are shown for integration inspection only.
"""
    ),
    code(
        """
rows = []
for (policy, condition), result in results.items():
    terminal = np.asarray(result["end_trace_cov_by_episode"][0])
    rows.extend([
        {"policy": policy, "condition": condition, "metric": "Mean", "value": terminal.mean()},
        {"policy": policy, "condition": condition, "metric": "Median", "value": np.median(terminal)},
    ])
terminal_summary = pd.DataFrame(rows)
display(terminal_summary.pivot_table(index=["policy", "condition"], columns="metric", values="value"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
policy_order = ["Frozen CNNv2 PPO", "Advanced Greedy"]
colors = {"Coast": "#6B7280", "Maneuver": "#D17A22"}
for ax, metric in zip(axes, ["Mean", "Median"]):
    subset = terminal_summary[terminal_summary.metric == metric]
    x = np.arange(len(policy_order))
    for offset, condition in zip((-0.18, 0.18), ("Coast", "Maneuver")):
        values = [subset[(subset.policy == policy) & (subset.condition == condition)].value.iloc[0] for policy in policy_order]
        ax.bar(x + offset, values, width=0.34, color=colors[condition], label=condition)
    ax.set_xticks(x, policy_order)
    ax.set_title(f"Final {metric.lower()} trace covariance")
    ax.set_ylabel("Trace covariance")
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8)
axes[1].legend()
fig.suptitle("Paired seed-0 engineering pilot (not an inferential comparison)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "pilot_final_covariance.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("## Maneuvering-RSO terminal covariance changes"),
    code(
        """
fig, ax = plt.subplots(figsize=(10, 5))
markers = {"Frozen CNNv2 PPO": "o", "Advanced Greedy": "s"}
colors_policy = {"Frozen CNNv2 PPO": "#2563A6", "Advanced Greedy": "#D17A22"}
for policy in policy_order:
    coast = results[(policy, "Coast")]
    maneuver = results[(policy, "Maneuver")]
    ids = np.asarray(maneuver["maneuver_rso_ids"][0], dtype=int)
    delta = np.asarray(maneuver["end_trace_cov_by_episode"][0])[ids] - np.asarray(coast["end_trace_cov_by_episode"][0])[ids]
    ax.scatter(ids, delta, marker=markers[policy], s=55, color=colors_policy[policy], label=policy, alpha=0.85)
ax.axhline(0, color="#111827", linewidth=1)
ax.set(title="Terminal covariance change for the 10 maneuvering RSOs", xlabel="RSO ID", ylabel="Maneuver - coast trace covariance")
ax.grid(color="#D9DEE5", linewidth=0.8, alpha=0.8)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "pilot_maneuver_rso_covariance_delta.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """
## Reacquisition and physical measurement outcomes

Reacquisition time is the first successful post-burn measurement time minus 900 s. A missing marker means the RSO was not successfully reobserved before episode end.
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
offsets = {"Frozen CNNv2 PPO": -0.12, "Advanced Greedy": 0.12}
outcome_rows = []
for policy in policy_order:
    result = results[(policy, "Maneuver")]
    ids = np.asarray(result["maneuver_rso_ids"][0], dtype=int)
    times = np.asarray(result["reacquisition_time"][0])[ids]
    observed = np.isfinite(times)
    axes[0].scatter(ids[observed] + offsets[policy], times[observed] / 60, s=55, marker=markers[policy], color=colors_policy[policy], label=policy)

    after_burn = [event for event in result["measurement_events"][0] if event["time_seconds"] >= 900]
    selected = sum(event["selected_by_smith_estimate"] for event in after_burn)
    received = sum(event["measurement_received"] for event in after_burn)
    rejected = sum(event["selected_by_smith_estimate"] and not event["truth_in_fov"] for event in after_burn)
    outcome_rows.extend([
        {"policy": policy, "outcome": "Selected", "count": selected},
        {"policy": policy, "outcome": "Received", "count": received},
        {"policy": policy, "outcome": "Truth-gate rejected", "count": rejected},
    ])

axes[0].set(title="First post-burn successful measurement", xlabel="Maneuvering RSO ID", ylabel="Reacquisition time (min)")
axes[0].grid(color="#D9DEE5", linewidth=0.8, alpha=0.8)
axes[0].legend()

outcomes = pd.DataFrame(outcome_rows)
x = np.arange(len(policy_order))
for offset, outcome, color in zip((-0.24, 0, 0.24), ["Selected", "Received", "Truth-gate rejected"], ["#6B7280", "#3A9D5D", "#C23B3B"]):
    values = [outcomes[(outcomes.policy == policy) & (outcomes.outcome == outcome)]["count"].iloc[0] for policy in policy_order]
    axes[1].bar(x + offset, values, width=0.22, label=outcome, color=color)
axes[1].set_xticks(x, policy_order)
axes[1].set(title="Post-burn maneuvering-RSO measurement events", ylabel="Event count")
axes[1].grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "pilot_reacquisition_and_measurements.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """
## Pilot conclusion

The integration is ready for paired Monte Carlo execution: coast reproduction passes, both policies use the same initial scenario and maneuver subset, and maneuver-specific records are populated. Seed 0 has no truth-gate rejection at 10 m/s, consistent with the earlier low-observability calibration. Performance conclusions require the planned multi-seed maneuver-rate experiment.
"""
    ),
]

nbf.write(notebook, OUTPUT_PATH)
print(OUTPUT_PATH)
