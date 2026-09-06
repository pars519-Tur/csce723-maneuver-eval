from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = NOTEBOOK_DIR / "02_maneuver_observability_calibration.ipynb"


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
# Maneuver Observability Calibration

## tl;dr

This notebook calibrates the truth perturbation before policy evaluation. A 10 m/s along-track impulse at t=900 s produces a median 0.055 deg sensor-view separation at the end of Smith's 5400 s episode; 100 m/s produces 0.552 deg. We therefore retain 10 m/s as the primary low-observability condition and report 100 m/s separately as a stress condition.

No policy is trained or evaluated here. The calibration uses 25 deterministic Smith catalog seeds and all three fixed sensor sites.
"""
    ),
    markdown(
        """
## Method

- Before the burn, truth is Smith's unchanged PyEphem/SGP4 trajectory.
- After the burn, a differential two-body perturbation is added to the nominal trajectory: maneuvering Cartesian propagation minus coasting Cartesian propagation.
- This construction is continuous at the burn, applies the requested RTN impulse exactly, and returns exactly to Smith's nominal truth as delta-v approaches zero.
- Candidate along-track magnitudes: 1, 5, 10, 25, 50, and 100 m/s.
- Burn time: 900 s; sample interval: 300 s; episode end: 5400 s.
"""
    ),
    code(
        """
from pathlib import Path

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
DATA_PATH = PROJECT_ROOT / "results" / "calibration" / "maneuver_observability.csv"
FIGURE_DIR = PROJECT_ROOT / "results" / "calibration" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

calibration = pd.read_csv(DATA_PATH)
assert len(calibration) == 25 * 6 * 16 * 3
assert calibration.notna().all().all()
calibration.head()
"""
    ),
    markdown("## Results\n\n### Sensor-view separation grows monotonically after the burn"),
    code(
        """
grouped = (
    calibration.groupby(["delta_v_t_mps", "seconds_since_burn"])["topocentric_angle_deg"]
    .agg(median="median", p05=lambda x: np.percentile(x, 5), p95=lambda x: np.percentile(x, 95))
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 5.5))
colors = plt.cm.viridis(np.linspace(0.08, 0.92, grouped.delta_v_t_mps.nunique()))
for color, magnitude in zip(colors, sorted(grouped.delta_v_t_mps.unique())):
    subset = grouped[grouped.delta_v_t_mps == magnitude]
    ax.plot(subset.seconds_since_burn / 60, subset["median"], color=color, linewidth=2.2, label=f"{magnitude:g} m/s")
    ax.fill_between(subset.seconds_since_burn / 60, subset.p05, subset.p95, color=color, alpha=0.10)

ax.set(title="Topocentric divergence from nominal truth", xlabel="Minutes since burn", ylabel="Angular separation (deg)")
ax.grid(color="#D9DEE5", linewidth=0.8, alpha=0.8)
ax.legend(title="Along-track impulse", ncol=2)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "topocentric_separation_vs_time.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("### Final separation and selected evaluation levels"),
    code(
        """
final = calibration[calibration.elapsed_seconds == 5400]
final_summary = final.groupby("delta_v_t_mps").agg(
    median_position_separation_km=("position_separation_km", "median"),
    median_topocentric_angle_deg=("topocentric_angle_deg", "median"),
    p95_topocentric_angle_deg=("topocentric_angle_deg", lambda x: np.percentile(x, 95)),
)
display(final_summary)

fig, ax = plt.subplots(figsize=(8, 5))
x = final_summary.index.to_numpy(dtype=float)
y = final_summary.median_topocentric_angle_deg.to_numpy()
ax.plot(x, y, marker="o", linewidth=2.4, color="#2563A6")
for magnitude, angle in zip(x, y):
    if magnitude in (10, 100):
        label = "Primary" if magnitude == 10 else "Stress"
        offset = (8, -5) if magnitude == 10 else (-105, -5)
        ax.annotate(f"{label}: {magnitude:g} m/s\\n{angle:.3f} deg", (magnitude, angle), xytext=offset, textcoords="offset points")
ax.set(title="End-of-episode maneuver observability", xlabel="Along-track impulse (m/s)", ylabel="Median sensor-view separation (deg)")
ax.set_ylim(-0.01, 0.65)
ax.grid(color="#D9DEE5", linewidth=0.8, alpha=0.8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "final_topocentric_separation_by_delta_v.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """
## Decision and caveat

The main Monte Carlo comparison will vary the fraction of maneuvering RSOs (0%, 5%, 10%, and 20%) using the 10 m/s primary impulse. A separate 100 m/s condition tests whether any apparent robustness is merely caused by the maneuver being too small to affect pointing within a 90-minute episode.

The 100 m/s case is an explicit stress test and must not be described as representative of routine station-keeping. Calibration establishes geometric observability only; it does not establish policy failure or success.
"""
    ),
]

nbf.write(notebook, OUTPUT_PATH)
print(OUTPUT_PATH)
