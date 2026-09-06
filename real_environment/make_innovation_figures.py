"""The innovation view of the same episodes Smith's figures plot.

Smith's Chapter 2 figures all plot the covariance trace, which his filter updates as
``P = P - K S K'`` without ever reading the measurement. The innovation is the only quantity in
that filter a maneuver can reach. These two figures put the two side by side.

    fig_innovation_vs_covariance   one episode, one policy, the same time axis, the reward metric
                                   above and the innovation below
    fig_innovation_null_and_tail   the pooled NIS distribution against its chi-square null, for
                                   the matched control and for the maneuver treatment

Both use sigma = 10 km, the noise the filter already assumes, because a chi-square threshold only
means anything once the filter is consistent. `make_smith_figure_comparison.py` keeps the
covariance panels at sigma = 0 to match Smith's published run exactly; the covariance is flat
either way.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID = PROJECT_ROOT / "results/noise_maneuver_grid"
OUTPUT = GRID / "figures_innovation"

NOISE_TAG = "sigma10"
EPISODE_SEED = 91  # 8 objects re-observed after their burn, 4 of them detected on sight
ILLUSTRATIVE_POLICY = "advanced_greedy"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE = "#c3c2b7"
CONTEXT = "#b9b8b1"  # de-emphasis grey for the non-maneuvering population

SERIES = {"advanced_greedy": "#2a78d6", "frozen_cnnv2_ppo": "#eb6834"}
POLICY_LABEL = {"advanced_greedy": "Advanced Greedy", "frozen_cnnv2_ppo": "Frozen CNNv2 PPO"}
POLICIES = ("advanced_greedy", "frozen_cnnv2_ppo")

CHI2_P99 = chi2.ppf(0.99, 3)
CHI2_MEAN = 3.0


def load(policy: str, condition: str) -> dict:
    with (GRID / f"{policy}_{condition}_{NOISE_TAG}_100seeds.p").open("rb") as stream:
        return pickle.load(stream)


RESULTS = {
    (policy, condition): load(policy, condition)
    for policy in POLICIES
    for condition in ("nullburn", "random")
}


def style(axes, xlabel: str, ylabel: str, title: str | None = None) -> None:
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
    if title:
        axes.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def save(figure, name: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return path


def episode_index(result: dict, seed: int) -> int:
    return result["pair_ids"].index(f"seed_{seed:03d}")


# --------------------------------------------------------------------------------------
# The headline figure: one episode, one policy, two metrics on a shared time axis.
# --------------------------------------------------------------------------------------
def innovation_vs_covariance() -> Path:
    policy = ILLUSTRATIVE_POLICY
    colour = SERIES[policy]
    treatment = RESULTS[(policy, "random")]
    control = RESULTS[(policy, "nullburn")]
    index = episode_index(treatment, EPISODE_SEED)
    control_index = episode_index(control, EPISODE_SEED)

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.6, 7.2), facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.18},
    )

    # Upper panel: Smith's own reward quantity.
    for source, source_index, label, kwargs in (
        (control, control_index, "no maneuver", dict(linewidth=5.0, alpha=0.30)),
        (treatment, index, "random maneuver", dict(linewidth=1.6, alpha=1.0)),
    ):
        history = source["trace_covariance_history_by_rso"][source_index]
        top.plot(
            np.asarray(history["time_seconds"], dtype=float),
            np.mean(np.asarray(history["trace_covariance"], dtype=float), axis=1),
            color=colour, label=label, zorder=3, **kwargs,
        )
    style(top, "", "Mean trace covariance",
          "Smith's reward metric over one episode — the maneuvers are invisible")
    top.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK_SECONDARY)

    # Lower panel: the innovation, same episode, same clock. Three tiers, because the
    # signature is cleanest in the FIRST measurement after a burn, which is the one a detector
    # would actually fire on. Later observations of the same object are not independent of it,
    # so pooling every post-burn measurement into one rate mixes detection with recovery.
    table = treatment["nis_table"][index]
    times = np.asarray(table["time_seconds"], dtype=float)
    values = np.asarray(table["nis"], dtype=float)
    rso_ids = np.asarray(table["rso_id"], dtype=int)
    maneuvering = np.asarray(table["is_maneuvering"], dtype=bool)
    delay = np.asarray(table["seconds_after_burn"], dtype=float)
    post_burn = maneuvering & (delay >= 0.0)

    first_seen: dict[int, int] = {}
    for position in np.where(post_burn)[0][np.argsort(delay[post_burn], kind="stable")]:
        first_seen.setdefault(int(rso_ids[position]), int(position))
    first_mask = np.zeros_like(post_burn)
    first_mask[list(first_seen.values())] = True
    later_mask = post_burn & ~first_mask

    floor = 1e-3
    bottom.scatter(times[~post_burn], np.maximum(values[~post_burn], floor),
                   s=8, color=CONTEXT, alpha=0.5, linewidths=0, zorder=2,
                   label="non-maneuvering objects")
    bottom.scatter(times[later_mask], np.maximum(values[later_mask], floor),
                   s=14, color=colour, alpha=0.30, linewidths=0, zorder=3,
                   label="maneuvering object, later observations")
    bottom.scatter(times[first_mask], np.maximum(values[first_mask], floor),
                   s=95, marker="D", color=colour, edgecolor=SURFACE, linewidths=1.2,
                   zorder=5, label="first observation after the burn")
    bottom.axhline(CHI2_P99, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 3)), zorder=4)
    bottom.annotate(
        f"chi-square (3 dof) p99 = {CHI2_P99:.2f}", (times.min(), CHI2_P99),
        textcoords="offset points", xytext=(4, 6), ha="left", fontsize=9, color=INK_MUTED,
    )
    bottom.set_yscale("log")
    detected_first = int(np.sum(values[first_mask] > CHI2_P99))
    style(bottom, "Time (s)", "Normalized innovation squared",
          f"The innovation over the same episode — {detected_first} of {int(first_mask.sum())} "
          "burns caught on sight")
    bottom.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_SECONDARY)

    false_alarm = float(np.mean(values[~maneuvering] > CHI2_P99))
    figure.suptitle(
        f"{POLICY_LABEL[policy]}, seed {EPISODE_SEED}: the same episode through two metrics",
        color=INK, fontsize=13, x=0.02, ha="left", y=0.965,
    )
    figure.text(
        0.02, -0.06,
        f"sigma = 10 km. {int(first_mask.sum())} of the tagged objects were re-observed after "
        f"their burn and {detected_first} of those first sightings clear the p99\nthreshold, "
        f"against {false_alarm:.1%} of the {int((~maneuvering).sum())} measurements of "
        "non-maneuvering objects. Later observations of the same objects stay elevated while the "
        "filter works\nthe burn out of its estimate. The covariance in the upper panel moves by "
        "less than 0.01% over the same episode. Frozen PPO gives the same picture.",
        color=INK_MUTED, fontsize=8.5, ha="left",
    )
    return save(figure, "fig_innovation_vs_covariance.png")


# --------------------------------------------------------------------------------------
# The distribution figure: is the null right, and where is the tail?
# --------------------------------------------------------------------------------------
def innovation_null_and_tail() -> Path:
    figure, axes_pair = plt.subplots(1, 2, figsize=(12.4, 4.9), facecolor=SURFACE)
    edges = np.logspace(-2, 3, 70)
    centres = np.sqrt(edges[:-1] * edges[1:])

    for axes, policy in zip(axes_pair, POLICIES):
        colour = SERIES[policy]
        pooled = {}
        for condition in ("nullburn", "random"):
            result = RESULTS[(policy, condition)]
            values, maneuvering, after = [], [], []
            for table in result["nis_table"]:
                values.append(np.asarray(table["nis"], dtype=float))
                maneuvering.append(np.asarray(table["is_maneuvering"], dtype=bool))
                after.append(np.asarray(table["seconds_after_burn"], dtype=float) >= 0.0)
            values = np.concatenate(values)
            mask = np.concatenate(maneuvering) & np.concatenate(after)
            pooled[condition] = values[mask]

        for condition, label, kwargs in (
            ("nullburn", "control cohort, after null burn",
             dict(linewidth=5.0, alpha=0.30)),
            ("random", "treatment cohort, after real burn",
             dict(linewidth=1.8, alpha=1.0)),
        ):
            density, _ = np.histogram(pooled[condition], bins=edges, density=True)
            # Empty bins are 0, which a log axis draws as a spike down to the floor. Mask them
            # so the curve breaks where there is no data instead of inventing vertical strokes.
            axes.plot(centres, np.where(density > 0, density, np.nan),
                      color=colour, label=label, zorder=3, **kwargs)

        theoretical = chi2.pdf(centres, 3)
        axes.plot(centres, theoretical, color=INK_MUTED, linewidth=1.4,
                  linestyle=(0, (5, 3)), zorder=2, label="chi-square (3 dof), the null")
        axes.axvline(CHI2_P99, color=INK_MUTED, linewidth=1.0, alpha=0.7, zorder=1)
        axes.annotate("p99", (CHI2_P99, axes.get_ylim()[1]), textcoords="offset points",
                      xytext=(4, -10), ha="left", va="top", fontsize=9, color=INK_MUTED)

        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_ylim(1e-6, 1.0)
        style(axes, "Normalized innovation squared", "Density" if policy == POLICIES[0] else "",
              POLICY_LABEL[policy])
        axes.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK_SECONDARY)

    figure.suptitle(
        "With matched noise the control follows the chi-square null and the maneuvers form the tail",
        color=INK, fontsize=13, x=0.02, ha="left", y=1.02,
    )
    figure.text(
        0.02, -0.08,
        "sigma = 10 km, 100 seeds per arm, pooled over every post-burn measurement. The control "
        "shares the treatment's cohort, burn times\nand burn directions and differs only in "
        "delta-v magnitude, so the separation between the two solid curves is the maneuver.",
        color=INK_MUTED, fontsize=8.5, ha="left",
    )
    return save(figure, "fig_innovation_null_and_tail.png")


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
    })
    for builder in (innovation_vs_covariance, innovation_null_and_tail):
        print(f"wrote {builder().relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
