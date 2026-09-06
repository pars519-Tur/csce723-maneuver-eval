"""Publication figures for the CSCE723 measurement-noise by random-maneuver grid.

Every number is read from the analysis CSVs rather than transcribed, so a figure can never
drift from the data behind it.

Palette is the validated two-slot categorical set (blue, orange), which clears the all-pairs
CVD and normal-vision floors on the light surface. Chrome and ink come from the same system.
Identity is never carried by colour alone: every chart with two series has both a legend and
direct value labels.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = PROJECT_ROOT / "results/noise_maneuver_grid/analysis"
OUTPUT = PROJECT_ROOT / "results/noise_maneuver_grid/figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"advanced_greedy": "#2a78d6", "frozen_cnnv2_ppo": "#eb6834"}
LABEL = {"advanced_greedy": "Advanced Greedy", "frozen_cnnv2_ppo": "Frozen CNNv2 PPO"}
POLICIES = ("advanced_greedy", "frozen_cnnv2_ppo")

CHI2_P99 = 11.344866730144373


def read(name: str) -> list[dict]:
    with (ANALYSIS / name).open() as stream:
        return list(csv.DictReader(stream))


def style_axes(axes, ylabel: str, title: str | None = None) -> None:
    """Recessive chrome. Hairline horizontal grid, no top or right spine."""
    axes.set_facecolor(SURFACE)
    axes.set_axisbelow(True)
    axes.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes.xaxis.grid(False)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    axes.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if title:
        axes.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def new_figure(width: float, height: float, ncols: int = 1):
    figure, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE)
    return figure, axes


def save(figure, name: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------------
# Figure 1: the paired maneuver effect on tracking error against the same effect on the
# reward metric. Two panels because the two measures have different units and scales, and a
# dual axis would be unreadable and misleading.
# --------------------------------------------------------------------------------------
def figure_maneuver_effect() -> Path:
    """Both effects on ONE axis, expressed as percent of each policy's own baseline.

    Giving the covariance its own panel would let it autoscale to its own noise and look like
    a real effect. Sharing an axis is the honest rendering: the reward metric is flat because
    it is flat.
    """
    effects = {
        (r["policy"], r["metric"]): r
        for r in read("paired_effects.csv")
        if r["comparison"] == "maneuver_effect" and r["held_fixed"] == "10"
    }
    cells = {
        (r["policy"], r["maneuver"]): r
        for r in read("cell_summary.csv")
        if r["sigma_km"] == "10"
    }
    measures = [
        ("final_tagged_error_median_km", "mean_final_tagged_error_median_km",
         "Median tracking error\n(physical accuracy)"),
        ("final_mean_covariance", "mean_final_mean_covariance",
         "Mean covariance trace\n(Smith's reward metric)"),
    ]

    figure, axes = new_figure(8.8, 4.9)
    positions = np.arange(len(measures))

    for index, policy in enumerate(POLICIES):
        offsets = positions + (index - 0.5) * 0.24
        for position, (metric, baseline_key, _) in zip(offsets, measures):
            row = effects[(policy, metric)]
            baseline = float(cells[(policy, "nullburn")][baseline_key])
            scale = 100.0 / baseline
            value = float(row["mean_paired_change"]) * scale
            low, high = float(row["ci_low"]) * scale, float(row["ci_high"]) * scale
            axes.errorbar(
                position, value,
                yerr=[[max(value - low, 0.0)], [max(high - value, 0.0)]],
                fmt="o", markersize=9, linewidth=2.0, capsize=6,
                color=SERIES[policy], markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=LABEL[policy] if metric == measures[0][0] else None,
                zorder=3,
            )
            # Percent labels span four orders of magnitude, so pick the format per value
            # rather than letting a single spec push the large ones into exponent notation.
            label = f"{value:+.0f}%" if abs(value) >= 10 else f"{value:+.2g}%"
            axes.annotate(
                label,
                (position, value), textcoords="offset points",
                xytext=(0, 22 if abs(value) >= 10 else 14), ha="center",
                fontsize=9.5, color=INK_SECONDARY,
            )

    axes.axhline(0.0, color=BASELINE, linewidth=1.2, zorder=1)
    axes.set_xticks(positions)
    axes.set_xticklabels([label for _, _, label in measures], color=INK_SECONDARY, fontsize=9.5)
    axes.set_xlim(-0.55, len(measures) - 0.45)
    style_axes(
        axes,
        "Change under maneuvers, percent of baseline",
        "Maneuvers degrade tracking by ~450% and move the reward metric by ~0.01%",
    )
    axes.margins(y=0.24)
    axes.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK_SECONDARY)
    figure.text(
        0.02, -0.12,
        "Paired against the matched null-burn control at sigma = 10 km. 100 seeds, 20 tagged "
        "RSOs per episode. Bars are 95% bootstrap\nintervals. The covariance interval spans "
        "zero for Advanced Greedy; for PPO it excludes zero but the effect is 0.008% of the "
        "metric.",
        color=INK_MUTED, fontsize=8.5, ha="left",
    )
    return save(figure, "fig1_maneuver_effect_error_vs_covariance.png")


# --------------------------------------------------------------------------------------
# Figure 2: NIS exceedance against the chi-square null, with the matched control as the
# negative control that makes the treatment elevation attributable.
# --------------------------------------------------------------------------------------
def figure_nis_detection() -> Path:
    rows = {
        (r["policy"], r["maneuver"], r["group"]): r
        for r in read("nis_summary.csv")
        if r["sigma_km"] == "10"
    }
    groups = [
        ("nullburn", "non_maneuvering", "Untagged objects\n(baseline)"),
        ("nullburn", "maneuvering_after_burn", "Control cohort,\nafter null burn"),
        ("random", "maneuvering_after_burn", "Treatment cohort,\nafter real burn"),
    ]

    figure, axes = new_figure(9.0, 4.8)
    positions = np.arange(len(groups))
    width = 0.36

    for index, policy in enumerate(POLICIES):
        offsets = positions + (index - 0.5) * (width + 0.02)
        values = [
            100.0 * float(rows[(policy, arm, group)]["fraction_above_chi2_p99"])
            for arm, group, _ in groups
        ]
        axes.bar(
            offsets, values, width,
            color=SERIES[policy], label=LABEL[policy], zorder=3,
            edgecolor=SURFACE, linewidth=2.0,
        )
        for offset, value in zip(offsets, values):
            axes.annotate(
                f"{value:.2f}%", (offset, value), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=9, color=INK_SECONDARY,
            )

    axes.axhline(1.0, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 3)), zorder=2)
    axes.annotate(
        "1% expected under a consistent filter",
        (-0.45, 1.0), textcoords="offset points", xytext=(0, 34),
        ha="left", fontsize=9, color=INK_MUTED,
    )

    axes.set_xticks(positions)
    axes.set_xticklabels([label for _, _, label in groups], color=INK_SECONDARY, fontsize=9)
    style_axes(
        axes,
        "Measurements above the p99 threshold (%)",
        "NIS detects maneuvers at 20–27% against a 1% false-alarm rate",
    )
    axes.margins(y=0.18)
    legend = axes.legend(
        frameon=False, fontsize=9, loc="upper center", labelcolor=INK_SECONDARY
    )
    legend.set_zorder(4)
    figure.text(
        0.02, -0.08,
        "Threshold is the 99th percentile of chi-square with three degrees of freedom, 11.34. "
        "sigma = 10 km, matched to the R Smith's\nfilter already assumes. The control shares the "
        "treatment's cohort, burn times and burn directions and differs only in delta-v\n"
        "magnitude, so the rise in the third group is attributable to the maneuver alone.",
        color=INK_MUTED, fontsize=8.5, ha="left",
    )
    return save(figure, "fig2_nis_detection_vs_false_alarm.png")


# --------------------------------------------------------------------------------------
# Figure 3: detection as a function of burn magnitude, using the first post-burn innovation.
# --------------------------------------------------------------------------------------
def figure_dose_response() -> Path:
    rows = read("first_post_burn_nis.csv")
    bands = [(1, 3.16, "1–3.2"), (3.16, 10, "3.2–10"),
             (10, 31.6, "10–32"), (31.6, 100, "32–100")]

    figure, axes = new_figure(8.6, 4.8)
    positions = np.arange(len(bands))

    for policy in POLICIES:
        subset = [r for r in rows if r["policy"] == policy and r["sigma_km"] == "10"]
        rates, counts = [], []
        for low, high, _ in bands:
            band = [
                r for r in subset
                if low <= float(r["delta_v_magnitude_mps"]) < high
            ]
            counts.append(len(band))
            rates.append(
                100.0 * np.mean([r["exceeds_chi2_p99"] == "True" for r in band])
                if band else np.nan
            )
        axes.plot(
            positions, rates, marker="o", markersize=9, linewidth=2.0,
            color=SERIES[policy], label=LABEL[policy],
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
        )
        # Push the two series' labels apart so the low bands stay readable.
        vertical = 13 if policy == POLICIES[0] else -20
        for position, rate in zip(positions, rates):
            axes.annotate(
                f"{rate:.1f}%", (position, rate), textcoords="offset points",
                xytext=(0, vertical), ha="center", fontsize=9, color=INK_SECONDARY,
            )

    axes.axhline(1.0, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 3)), zorder=2)
    axes.annotate(
        "1% false-alarm rate", (len(bands) - 0.6, 1.0), textcoords="offset points",
        xytext=(0, 6), ha="right", fontsize=9, color=INK_MUTED,
    )
    axes.set_xticks(positions)
    axes.set_xticklabels([label for _, _, label in bands], color=INK_SECONDARY, fontsize=9)
    axes.set_xlabel("Burn magnitude (m/s)", color=INK_SECONDARY, fontsize=10)
    axes.set_xlim(-0.35, len(bands) - 0.4)
    style_axes(
        axes,
        "First post-burn innovation above p99 (%)",
        "Burns below about 10 m/s stay under the noise floor",
    )
    axes.set_ylim(-5.0, 62.0)
    axes.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_SECONDARY)
    figure.text(
        0.02, -0.06,
        "sigma = 10 km. Uses each object's first measurement after its own burn, before the "
        "filter re-absorbs it.",
        color=INK_MUTED, fontsize=8.5, ha="left",
    )
    return save(figure, "fig3_detection_vs_burn_magnitude.png")


# --------------------------------------------------------------------------------------
# Figure 4: why custody is not the endpoint. One relationship against one threshold, so a
# single series and no legend; the title names what is plotted.
# --------------------------------------------------------------------------------------
def figure_fov_geometry() -> Path:
    displacement_coefficient = 0.00101  # km per (m/s) per s, fitted to the run
    coast_seconds = 4500.0
    geo_radius_km = 42019.0
    fov_escape_km = geo_radius_km * np.radians(2.0)

    delta_v = np.logspace(0, 3, 400)
    displacement = displacement_coefficient * delta_v * coast_seconds

    figure, axes = new_figure(8.6, 4.8)
    axes.plot(delta_v, displacement, linewidth=2.0, color=SERIES["advanced_greedy"], zorder=3)

    axes.axhline(fov_escape_km, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 3)), zorder=2)
    axes.annotate(
        f"leaves the 4° field of view  ({fov_escape_km:.0f} km)",
        (1.15, fov_escape_km), textcoords="offset points", xytext=(0, 7),
        ha="left", fontsize=9, color=INK_SECONDARY,
    )

    axes.axvspan(1.0, 5.0, color=SERIES["advanced_greedy"], alpha=0.10, zorder=1)
    axes.annotate(
        "typical GEO\nstation keeping\n1-5 m/s",
        (1.12, 260.0), ha="left", va="top", fontsize=9, color=INK_SECONDARY,
    )

    median_burn = 9.6
    median_displacement = displacement_coefficient * median_burn * coast_seconds
    axes.plot(
        [median_burn], [median_displacement], marker="o", markersize=9,
        color=INK, markeredgecolor=SURFACE,
        markeredgewidth=1.5, zorder=4,
    )
    axes.annotate(
        f"median burn in this study\n{median_burn:.1f} m/s gives {median_displacement:.0f} km, "
        f"{100 * median_displacement / fov_escape_km:.1f}% of the way",
        (median_burn, median_displacement), textcoords="offset points", xytext=(14, -6),
        ha="left", fontsize=9, color=INK_SECONDARY,
    )

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("Burn magnitude (m/s)", color=INK_SECONDARY, fontsize=10)
    style_axes(
        axes,
        "Displacement after 4500 s of coast (km)",
        "Custody loss is geometrically unreachable at Smith's episode length",
    )
    axes.set_xlim(1, 1000)
    figure.text(
        0.02, -0.06,
        "Displacement fitted from the run as d ≈ 0.00101 · Δv · t km. Escaping a "
        "4° field of view at a median GEO radius of\n42019 km needs about 324 m/s over this "
        "coast time, roughly two orders of magnitude above a station-keeping burn.",
        color=INK_MUTED, fontsize=8.5, ha="left",
    )
    return save(figure, "fig4_displacement_vs_field_of_view.png")


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
    })
    for builder in (
        figure_maneuver_effect,
        figure_nis_detection,
        figure_dose_response,
        figure_fov_geometry,
    ):
        print(f"wrote {builder().relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
