"""Build the notebook that replays Smith's Chapter 2 figures under maneuvers.

Layout follows the request: within one cell, Smith's published figure and each regenerated
figure are emitted as SEPARATE figures rather than as panels of one figure, so each can be read
at its own size.

The notebook needs only numpy, matplotlib and pickle, so it runs under any kernel that can read
the result files, including the base conda environment.
"""

from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = NOTEBOOK_DIR / "09_smith_figures_under_maneuvers.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


SETUP = r'''
from pathlib import Path
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
GRID = PROJECT_ROOT / "results/noise_maneuver_grid"
ASSETS = REPOSITORY_ROOT / "wiki/reading-notes/assets"

SEED_INDEX = 1          # the single seeded episode Smith's 2.10, 2.11, 2.12 and 2.15 plot
NOISE_TAG = "sigma10"   # measurement noise matched to the R Smith's filter already assumes

POLICIES = {"advanced_greedy": "Advanced Greedy", "frozen_cnnv2_ppo": "Frozen CNNv2 PPO"}

# Conditions are ordered by burn size, so they take one hue as an ordinal ramp rather than
# three unrelated colours. Policy is carried by the figure title, not by hue.
CONDITIONS = [
    ("nullburn",          "no maneuver (control)",      "#86b6ef"),
    ("random",            "1-100 m/s burns",            "#3987e5"),
    ("stress100to2000",   "100-2000 m/s burns",         "#184f95"),
]

SURFACE, INK, INK_2, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDC, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
})

REFERENCE = {
    "2.10": "smith2024-fig2-10-mean-single-run.png",
    "2.11": "smith2024-fig2-11-median-single-run.png",
    "2.12": "smith2024-fig2-12-cdf-single-run.png",
    "2.13": "smith2024-fig2-13-mean-monte-carlo.png",
    "2.14": "smith2024-fig2-14-median-monte-carlo.png",
    "2.15": "smith2024-fig2-15-marl-sawtooth.png",
}

RESULTS = {}
for policy in POLICIES:
    for condition, _, _ in CONDITIONS:
        path = GRID / f"{policy}_{condition}_{NOISE_TAG}_100seeds.p"
        if path.exists():
            with path.open("rb") as stream:
                RESULTS[(policy, condition)] = pickle.load(stream)

available = sorted({condition for _, condition in RESULTS})
print(f"loaded {len(RESULTS)} arms; conditions present: {available}")


def style(axes, xlabel, ylabel, title):
    axes.set_facecolor(SURFACE)
    axes.set_axisbelow(True)
    axes.yaxis.grid(True, color=GRIDC, linewidth=0.8)
    axes.xaxis.grid(False)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    axes.set_xlabel(xlabel, color=INK_2, fontsize=10)
    axes.set_ylabel(ylabel, color=INK_2, fontsize=10)
    axes.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def show_smith(key):
    """Smith's published figure, as its own figure."""
    figure, axes = plt.subplots(figsize=(7.6, 5.0))
    axes.imshow(mpimg.imread(ASSETS / REFERENCE[key]))
    axes.axis("off")
    axes.set_title(f"Smith 2024, Fig. {key} — published, no maneuvers",
                   color=INK, fontsize=12, loc="left", pad=10)
    plt.show()


def new_axes(width=7.6, height=4.4):
    figure, axes = plt.subplots(figsize=(width, height))
    return figure, axes


def present(policy):
    return [c for c in CONDITIONS if (policy, c[0]) in RESULTS]
'''


HISTORY = r'''
def covariance_history(policy, reducer, ylabel, key):
    figure, axes = new_axes()
    for condition, label, colour in present(policy):
        history = RESULTS[(policy, condition)]["trace_covariance_history_by_rso"][SEED_INDEX]
        axes.plot(np.asarray(history["time_seconds"], dtype=float),
                  reducer(np.asarray(history["trace_covariance"], dtype=float), axis=1),
                  color=colour, linewidth=2.0, label=label, zorder=3)
    axes.axvspan(600, 4800, color=INK_MUTED, alpha=0.07, zorder=0)
    style(axes, "Time (s)", ylabel,
          f"{POLICIES[policy]} — Fig. {key} recomputed, seed {SEED_INDEX}")
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    plt.show()
'''


CDF = r'''
def cumulative_distribution(policy):
    figure, axes = new_axes()
    for condition, label, colour in present(policy):
        final = np.asarray(
            RESULTS[(policy, condition)]["end_trace_cov_by_episode"][SEED_INDEX], dtype=float)
        ordered = np.sort(final)
        axes.plot(ordered, np.arange(1, ordered.size + 1), color=colour,
                  linewidth=2.0, label=label, zorder=3)
    axes.set_xlim(0.0, 0.04)   # Smith's own axis range
    style(axes, "Trace covariance at end of episode", "Cumulative number of RSOs",
          f"{POLICIES[policy]} — Fig. 2.12 recomputed, seed {SEED_INDEX}")
    axes.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    plt.show()
'''


VIOLIN = r'''
def monte_carlo_violin(policy, reducer, ylabel, key):
    figure, axes = new_axes(height=4.6)
    labels = []
    for position, (condition, label, colour) in enumerate(present(policy)):
        values = reducer(
            np.asarray(RESULTS[(policy, condition)]["end_trace_cov_by_episode"], dtype=float),
            axis=1)
        parts = axes.violinplot([values], positions=[position], widths=0.7,
                                showmeans=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(colour); body.set_alpha(0.75)
            body.set_edgecolor(colour); body.set_linewidth(1.4)
        axes.plot([position], [np.median(values)], marker="o", markersize=7, color=INK,
                  markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
        labels.append(label.replace(" (control)", "\n(control)").replace(" m/s ", " m/s\n"))
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, color=INK_2, fontsize=9)
    style(axes, "", ylabel, f"{POLICIES[policy]} — Fig. {key} recomputed, 100 seeds")
    plt.show()
'''


POINTING = r'''
def pointing_history(policy):
    figure, axes = new_axes(height=4.4)
    for condition, label, colour in present(policy):
        result = RESULTS[(policy, condition)]
        axes.plot(np.asarray(result["proptime"][SEED_INDEX]["agent_0"], dtype=float),
                  np.asarray(result["azi"][SEED_INDEX]["agent_0"], dtype=float),
                  color=colour, linewidth=1.6, label=label, zorder=3)
    axes.axvspan(600, 4800, color=INK_MUTED, alpha=0.07, zorder=0)
    style(axes, "Time (s)", "Agent 0 azimuth (deg)",
          f"{POLICIES[policy]} — Fig. 2.15 recomputed, seed {SEED_INDEX}")
    axes.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK_2)
    plt.show()
'''


OUTCOME = r'''
def outcome_bars(metric, ylabel, title, formatter):
    """The quantities that DO move, on the same three conditions."""
    for policy in POLICIES:
        figure, axes = new_axes(height=4.2)
        labels, values = [], []
        for condition, label, colour in present(policy):
            result = RESULTS[(policy, condition)]
            episodes = len(result["pair_ids"])
            if metric == "reacquired":
                value = float(np.mean([
                    np.asarray(result["reacquired"][i], dtype=bool)[
                        np.asarray(result["maneuver_rso_ids"][i], dtype=int)].mean()
                    for i in range(episodes)]))
            else:
                value = float(np.mean([
                    np.median(np.asarray(h["estimate_truth_error_km"])[-1])
                    for h in result["estimation_error_history"]]))
            labels.append(label.replace(" (control)", "\n(control)").replace(" m/s ", " m/s\n"))
            values.append(value)
            axes.bar([len(values) - 1], [value], 0.62, color=colour,
                     edgecolor=SURFACE, linewidth=2.0, zorder=3)
        for position, value in enumerate(values):
            axes.annotate(formatter(value), (position, value), textcoords="offset points",
                          xytext=(0, 5), ha="center", fontsize=9.5, color=INK_2)
        axes.set_xticks(range(len(labels)))
        axes.set_xticklabels(labels, color=INK_2, fontsize=9)
        if metric != "reacquired":
            axes.set_yscale("log")
        style(axes, "", ylabel, f"{POLICIES[policy]} — {title}")
        axes.margins(y=0.18)
        plt.show()
'''


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

notebook["cells"] = [
    markdown(
        r"""
# Smith's Chapter 2 figures, replayed under maneuvers

Each cell below emits Smith's published figure first, then the same quantity recomputed as its
own separate figure, one per policy.

Three conditions appear in every recomputed figure, ordered by burn size and drawn as one
sequential blue ramp:

| Condition | What it is |
|---|---|
| **no maneuver (control)** | The tagged cohort with a null burn. Same objects, same burn times, same burn directions as the treatments, only zero delta-v. |
| **1-100 m/s burns** | The primary condition. Isotropic RTN direction, log-uniform magnitude, uniform burn time over 600-4800 s. |
| **100-2000 m/s burns** | The stress condition. Orbit-transfer scale, far outside the regime the policy was trained on. Reported separately from the primary condition. |

All arms use measurement noise of sigma = 10 km, matched to the `R` Smith's filter already
assumes. 100 seeds per arm, 20 tagged RSOs of 100 per episode. Smith's own files are unmodified.

**What to look for.** Smith's figures all plot the covariance trace, which his filter updates as
`P = P - K S K'` without ever reading the measurement. The three curves will sit almost on top of
one another even in the stress condition. The final cell plots the quantities that do move.
"""
    ),
    code(SETUP),
    code(HISTORY + "\n\n" + CDF + "\n\n" + VIOLIN + "\n\n" + POINTING + "\n\n" + OUTCOME),
    markdown("## Fig. 2.10 — mean trace covariance over a single seeded episode"),
    code(
        'show_smith("2.10")\n'
        'for policy in POLICIES:\n'
        '    covariance_history(policy, np.mean, "Mean trace covariance", "2.10")'
    ),
    markdown("## Fig. 2.11 — median trace covariance over a single seeded episode"),
    code(
        'show_smith("2.11")\n'
        'for policy in POLICIES:\n'
        '    covariance_history(policy, np.median, "Median trace covariance", "2.11")'
    ),
    markdown("## Fig. 2.12 — ending cumulative trace covariance distribution"),
    code('show_smith("2.12")\nfor policy in POLICIES:\n    cumulative_distribution(policy)'),
    markdown("## Fig. 2.13 — final mean trace covariance over 100 Monte Carlo runs"),
    code(
        'show_smith("2.13")\n'
        'for policy in POLICIES:\n'
        '    monte_carlo_violin(policy, np.mean, "Final mean trace covariance", "2.13")'
    ),
    markdown("## Fig. 2.14 — final median trace covariance over 100 Monte Carlo runs"),
    code(
        'show_smith("2.14")\n'
        'for policy in POLICIES:\n'
        '    monte_carlo_violin(policy, np.median, "Final median trace covariance", "2.14")'
    ),
    markdown("## Fig. 2.15 — agent pointing history"),
    code('show_smith("2.15")\nfor policy in POLICIES:\n    pointing_history(policy)'),
    markdown(
        r"""
## What Smith's figures cannot show

The covariance panels above barely move. These two quantities, measured on the same episodes and
the same objects, do.
"""
    ),
    code(
        'outcome_bars("reacquired", "Fraction of tagged objects re-observed after their burn",\n'
        '             "custody", lambda v: f"{v:.1%}")\n'
        'outcome_bars("error", "Median estimate-to-truth error (km, log scale)",\n'
        '             "tracking accuracy", lambda v: f"{v:,.1f} km")'
    ),
]

OUTPUT_PATH.write_text(nbf.writes(notebook))
print(f"wrote {OUTPUT_PATH}")
