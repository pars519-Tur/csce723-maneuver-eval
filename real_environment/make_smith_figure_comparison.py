"""Smith's own Chapter 2 result figures, regenerated with random maneuvers added.

Each output puts Smith's published figure on the left and the same quantity recomputed on the
right, for the matched null-burn control against the random-maneuver treatment. This is the
direct answer to "what do Smith's figures look like once the catalog maneuvers".

The right-hand panels use **sigma = 0**, that is Smith's environment exactly as released, so the
only difference from his published run is the maneuvers themselves. Adding the measurement noise
does not change the picture, because the covariance these figures plot is measurement-independent
either way; the noise grid is analysed separately in `make_figures.py`.

Encoding: hue carries the policy, line style carries the condition. Two hues rather than four
keeps the palette inside its validated adjacent-pair gates while the condition rides a second,
non-colour channel.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
GRID = PROJECT_ROOT / "results/noise_maneuver_grid"
ASSETS = REPOSITORY_ROOT / "wiki/reading-notes/assets"
OUTPUT = GRID / "figures_smith_matched"

NOISE_TAG = "sigma00"
SEED_INDEX = 1
EPISODE_SECONDS = 5400.0

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE = "#c3c2b7"

POLICIES = ("advanced_greedy", "frozen_cnnv2_ppo")
SERIES = {"advanced_greedy": "#2a78d6", "frozen_cnnv2_ppo": "#eb6834"}
POLICY_LABEL = {"advanced_greedy": "Advanced Greedy", "frozen_cnnv2_ppo": "Frozen CNNv2 PPO"}
# The two conditions produce identical curves. Drawing the control as a thick pale band with
# the maneuver traced thinly on top makes that coincidence visible, instead of hiding one series
# under the other and looking like a plotting mistake.
CONDITIONS = (
    ("nullburn", "no maneuver", dict(linewidth=5.0, alpha=0.30, linestyle="-")),
    ("random", "random maneuver", dict(linewidth=1.6, alpha=1.0, linestyle="-")),
)

REFERENCE = {
    "2.10": ASSETS / "smith2024-fig2-10-mean-single-run.png",
    "2.11": ASSETS / "smith2024-fig2-11-median-single-run.png",
    "2.12": ASSETS / "smith2024-fig2-12-cdf-single-run.png",
    "2.13": ASSETS / "smith2024-fig2-13-mean-monte-carlo.png",
    "2.14": ASSETS / "smith2024-fig2-14-median-monte-carlo.png",
    "2.15": ASSETS / "smith2024-fig2-15-marl-sawtooth.png",
}


def load(policy: str, condition: str) -> dict:
    path = GRID / f"{policy}_{condition}_{NOISE_TAG}_100seeds.p"
    with path.open("rb") as stream:
        return pickle.load(stream)


RESULTS = {
    (policy, condition): load(policy, condition)
    for policy in POLICIES
    for condition, _, _ in CONDITIONS
}


def style(axes, xlabel: str, ylabel: str, title: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_axisbelow(True)
    axes.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axes.xaxis.grid(False)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    axes.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    axes.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    axes.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def side_by_side(reference_key: str, height: float = 4.6):
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, height), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.0, 1.05]},
    )
    left.imshow(mpimg.imread(REFERENCE[reference_key]))
    left.axis("off")
    left.set_title(
        f"Smith 2024, Fig. {reference_key} — published, no maneuvers",
        color=INK, fontsize=11, loc="left", pad=10,
    )
    return figure, right


def legend(axes, loc: str = "best") -> None:
    handles = [
        plt.Line2D([], [], color=SERIES[policy], label=f"{POLICY_LABEL[policy]}, {label}",
                   **{**style_kwargs, "linewidth": min(style_kwargs["linewidth"], 4.0)})
        for policy in POLICIES
        for _, label, style_kwargs in CONDITIONS
    ]
    axes.legend(handles=handles, frameon=False, fontsize=8.5, loc=loc,
                labelcolor=INK_SECONDARY)


def caption(figure, text: str) -> None:
    figure.text(0.02, -0.04, text, color=INK_MUTED, fontsize=8.5, ha="left")


def save(figure, name: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    figure.savefig(path, dpi=190, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return path


def burn_marker(axes) -> None:
    """Random burn times span 600-4800 s, so mark the window rather than one instant."""
    axes.axvspan(600.0, 4800.0, color=INK_MUTED, alpha=0.08, zorder=0)
    axes.annotate(
        "burn window 600–4800 s", (600.0, axes.get_ylim()[1]),
        textcoords="offset points", xytext=(6, -12),
        ha="left", va="top", fontsize=8.5, color=INK_MUTED,
    )


# --------------------------------------------------------------------------------------
# Fig 2.10 and 2.11: single-seed catalog covariance history.
# --------------------------------------------------------------------------------------
def covariance_history(reference_key: str, reducer, ylabel: str, name: str) -> Path:
    figure, axes = side_by_side(reference_key)
    for policy in POLICIES:
        for condition, _, style_kwargs in CONDITIONS:
            history = RESULTS[(policy, condition)]["trace_covariance_history_by_rso"][SEED_INDEX]
            times = np.asarray(history["time_seconds"], dtype=float)
            traces = np.asarray(history["trace_covariance"], dtype=float)
            axes.plot(times, reducer(traces, axis=1), color=SERIES[policy], zorder=3,
                      **style_kwargs)
    style(axes, "Time (s)", ylabel, f"Same quantity, with random maneuvers (seed {SEED_INDEX})")
    burn_marker(axes)

    # State the measured gap on the figure. Read casually, the two POLICY curves separating can
    # be mistaken for a maneuver effect, so the within-policy difference is spelled out.
    notes = []
    for policy in POLICIES:
        finals = {}
        for condition, _, _ in CONDITIONS:
            history = RESULTS[(policy, condition)]["trace_covariance_history_by_rso"][SEED_INDEX]
            finals[condition] = float(reducer(np.asarray(history["trace_covariance"]), axis=1)[-1])
        gap = finals["random"] - finals["nullburn"]
        relative = 0.0 if finals["nullburn"] == 0 else 100.0 * gap / finals["nullburn"]
        notes.append(
            f"{POLICY_LABEL[policy]}: maneuver minus control = "
            + ("exactly 0" if gap == 0.0 else f"{relative:+.4f}%")
        )
    axes.annotate(
        "Within each colour the two conditions coincide.\n" + "\n".join(notes),
        (0.98, 0.97), xycoords="axes fraction", ha="right", va="top",
        fontsize=8.5, color=INK_SECONDARY,
    )
    legend(axes, loc="lower left")
    caption(
        figure,
        "Right panel: Smith's environment as released, 20 of 100 RSOs given an isotropic burn of "
        "1–100 m/s at a uniformly drawn time.\nThe control shares the cohort, burn times and burn "
        "directions with zero delta-v. The two conditions overlay almost exactly.",
    )
    return save(figure, name)


# --------------------------------------------------------------------------------------
# Fig 2.12: cumulative distribution of final per-RSO trace covariance, single seed.
# --------------------------------------------------------------------------------------
def cumulative_distribution() -> Path:
    figure, axes = side_by_side("2.12")
    for policy in POLICIES:
        for condition, _, style_kwargs in CONDITIONS:
            final = np.asarray(
                RESULTS[(policy, condition)]["end_trace_cov_by_episode"][SEED_INDEX],
                dtype=float,
            )
            ordered = np.sort(final)
            axes.plot(ordered, np.arange(1, ordered.size + 1), color=SERIES[policy],
                      zorder=3, **style_kwargs)
    axes.set_xlim(0.0, 0.04)  # Smith's own axis range, so the panels read at the same scale
    style(axes, "Trace covariance at end of episode", "Cumulative number of RSOs",
          f"Same quantity, with random maneuvers (seed {SEED_INDEX})")
    legend(axes, loc="upper left")
    caption(
        figure,
        "Right panel: the maneuvering and control curves are indistinguishable, because the "
        "covariance update never sees the measurement value.",
    )
    return save(figure, "fig2_12_cumulative_distribution.png")


# --------------------------------------------------------------------------------------
# Fig 2.13 and 2.14: 100-run violin of the final catalog statistic.
# --------------------------------------------------------------------------------------
def monte_carlo_violin(reference_key: str, reducer, ylabel: str, name: str) -> Path:
    figure, axes = side_by_side(reference_key, height=4.8)
    labels, position = [], 0
    for policy in POLICIES:
        for condition, condition_label, _ in CONDITIONS:
            values = reducer(
                np.asarray(
                    RESULTS[(policy, condition)]["end_trace_cov_by_episode"], dtype=float
                ),
                axis=1,
            )
            parts = axes.violinplot([values], positions=[position], widths=0.7,
                                    showmeans=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(SERIES[policy])
                body.set_alpha(0.30 if condition == "nullburn" else 0.65)
                body.set_edgecolor(SERIES[policy])
                body.set_linewidth(1.4)
            axes.plot([position], [np.median(values)], marker="o", markersize=7,
                      color=SERIES[policy], markeredgecolor=SURFACE, markeredgewidth=1.4,
                      zorder=4)
            labels.append(f"{POLICY_LABEL[policy].replace(' ', chr(10), 1)}\n{condition_label}")
            position += 1
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, color=INK_SECONDARY, fontsize=8.5)
    style(axes, "", ylabel, "Same quantity, with random maneuvers (100 seeds)")
    caption(
        figure,
        "Right panel: 100 paired seeds per arm. Filled violins are the maneuvering condition, "
        "pale violins the matched control.\nDots are medians. The distributions sit on top of "
        "one another.",
    )
    return save(figure, name)


# --------------------------------------------------------------------------------------
# Fig 2.15: agent pointing history.
# --------------------------------------------------------------------------------------
def pointing_history() -> Path:
    """Smith's pointing figure, one stacked panel per policy.

    Four sawtooth traces on one axis are unreadable, and the claim being made is per policy:
    does THIS policy point anywhere different when the catalog maneuvers.
    """
    figure = plt.figure(figsize=(13.0, 5.2), facecolor=SURFACE)
    outer = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.16)
    left = figure.add_subplot(outer[0, 0])
    left.imshow(mpimg.imread(REFERENCE["2.15"]))
    left.axis("off")
    left.set_title(
        "Smith 2024, Fig. 2.15 — published, no maneuvers",
        color=INK, fontsize=11, loc="left", pad=10,
    )

    inner = outer[0, 1].subgridspec(len(POLICIES), 1, hspace=0.35)
    for row, policy in enumerate(POLICIES):
        axes = figure.add_subplot(inner[row, 0])
        for condition, label, style_kwargs in CONDITIONS:
            result = RESULTS[(policy, condition)]
            times = np.asarray(result["proptime"][SEED_INDEX]["agent_0"], dtype=float)
            azimuth = np.asarray(result["azi"][SEED_INDEX]["agent_0"], dtype=float)
            axes.plot(times, azimuth, color=SERIES[policy], zorder=3,
                      label=f"{label}", **style_kwargs)
        title = (
            f"{POLICY_LABEL[policy]}, agent 0 azimuth (seed {SEED_INDEX})"
            if row == 0 else f"{POLICY_LABEL[policy]}, agent 0 azimuth"
        )
        style(axes, "Time (s)" if row == len(POLICIES) - 1 else "", "Azimuth (deg)", title)
        axes.axvspan(600.0, 4800.0, color=INK_MUTED, alpha=0.08, zorder=0)
        if row == 0:
            axes.legend(frameon=False, fontsize=8.5, loc="upper right",
                        labelcolor=INK_SECONDARY, ncol=2)

    caption(
        figure,
        "Right panels: the frozen PPO traces for the two conditions coincide exactly, the pale "
        "control band never showing beneath the maneuver\ntrace. Across all 100 seeds the policy "
        "changed zero of 52014 azimuth decisions when the catalog maneuvered; Advanced Greedy "
        "differs\non 0.56% of its decisions. Shaded region is the 600–4800 s burn window.",
    )
    return save(figure, "fig2_15_pointing_history.png")



def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
    })
    outputs = [
        covariance_history("2.10", np.mean, "Mean trace covariance of all RSOs",
                           "fig2_10_mean_covariance_history.png"),
        covariance_history("2.11", np.median, "Median trace covariance of all RSOs",
                           "fig2_11_median_covariance_history.png"),
        cumulative_distribution(),
        monte_carlo_violin("2.13", np.mean, "Final mean trace covariance",
                           "fig2_13_mean_monte_carlo_violin.png"),
        monte_carlo_violin("2.14", np.median, "Final median trace covariance",
                           "fig2_14_median_monte_carlo_violin.png"),
        pointing_history(),
    ]
    for path in outputs:
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
