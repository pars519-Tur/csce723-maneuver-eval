# NotebookLM source set — CSCE 723 maneuver evaluation

Upload everything in this folder as sources to a NotebookLM notebook. Suggested notebook name:
**CSCE 723 — MARL Sensor Tasking Under Unmodelled Maneuvers**.

Refreshed 27 August 2026. This version covers all 22 arms, including the three direction arms, the
100 to 2000 m/s stress arm and the finite-burn arm, none of which were in the earlier package.
The `superseded/` subfolder holds figures the paper set replaced. Do not upload it.

## What each source is for

| File | Role |
|---|---|
| `01_experiment_report.md` | The main narrative. Question, defects found, design, all results, conclusions. Start here. |
| `02_method_and_code_reference.md` | Technical appendix. How noise, maneuvers and NIS were obtained without editing Smith's code, and what each control holds fixed. |
| `table1_condition_summary.csv` | The paper's Table 1. One row per policy and maneuver condition at 10 km noise. |
| `cell_summary.csv` | One row per policy × noise × maneuver cell. Headline means for all 22 arms. |
| `paired_effects.csv` | Every paired contrast with bootstrap confidence intervals. |
| `nis_summary.csv` | Pooled NIS by policy, noise level, arm and group. |
| `first_post_burn_nis.csv` | Per-object first post-burn NIS with its delta-v and delay. The dose-response data. |
| `dose_response.csv` | Per-object delta-v against covariance response, tracking error and reacquisition. Largest file. |
| `action_invariance.json` | How often each policy repoints when the measurement noise changes. |
| `control_matching_check.json` | Proof the control differs from the treatment only in delta-v magnitude. |

## The paper figure set

Produced by `notebooks/10_paper_figures.ipynb` at 600 dpi. Vector PDF versions of all seven live
in `results/noise_maneuver_grid/figures_paper/`.

| File | What it shows |
|---|---|
| `fig1_covariance_history.png` | Covariance trace over the episode under control, 1–100 m/s and 100–2000 m/s. The curves lie on top of each other. |
| `fig2_paired_change_from_control.png` | Percent change from the matched control. The reward metric in panel A, tracking error in panel B, on different scales because they differ by three orders of magnitude. |
| `fig3_innovation_detection.png` | NIS distribution against the chi-square (3 dof) null, plus detection rate per condition. |
| `fig4_dose_response.png` | Dose response across three decades of burn size. NIS climbs into the thousands while covariance stays flat. |
| `fig5_custody_geometry.png` | Displacement against the field of view, plus reacquisition per condition. Why custody cannot be lost below a few hundred m/s. |
| `fig6_covariance_ceiling.png` | The ceiling. What the reward metric would do if every tagged object lost custody, against what it actually did. |
| `fig7_residual_diagnostics.png` | The residual diagnostics. The residual grows after the burn instead of decaying, while the standard deviation the filter assigns to it stays at 10 km in both the control and the treatment. |
| `fig8_measurement_residuals.png` | The signed per-component residual against the filter's own 1 and 3 sigma bands, for a burn above the detection floor and one below it. The detection floor made concrete. |
| `fig9_whitened_innovation.png` | The whitened innovation against the standard normal it should follow. The null and the floor-level burn lie on the diagonal; the detectable burn departs on both tails. |
| `fig_innovation_vs_covariance.png` | One episode, one policy, one time axis. The reward metric above, flat. The innovation below, with burns clearing the p99 threshold on first sight. |

### Smith-matched comparison figures

These are Smith's own Chapter 2 result figures with random maneuvers added. Each puts his
published figure on the left and the same quantity recomputed on the right, control against
maneuver. They use **sigma = 0**, Smith's environment exactly as released, so the only difference
from his published run is the maneuvers themselves.

| File | Smith figure |
|---|---|
| `fig2_10_mean_covariance_history.png` | Fig. 2.10, single-seed mean trace covariance history |
| `fig2_11_median_covariance_history.png` | Fig. 2.11, single-seed median trace covariance history |
| `fig2_12_cumulative_distribution.png` | Fig. 2.12, ending cumulative trace covariance distribution |
| `fig2_13_mean_monte_carlo_violin.png` | Fig. 2.13, 100-run final mean trace covariance |
| `fig2_14_median_monte_carlo_violin.png` | Fig. 2.14, 100-run final median trace covariance |
| `fig2_15_pointing_history.png` | Fig. 2.15, agent pointing history |

In every one of them the maneuver and control curves lie on top of each other. The control is
drawn as a thick pale band and the maneuver as a thin line over it, so the coincidence is
visible rather than looking like a missing series.

NotebookLM reads the text sources for its answers, so the figures are for your slides and the
paper rather than for the notebook's own retrieval. Regenerate the paper set by running
`python -m real_environment.build_paper_figure_cache` and
`python -m real_environment.build_residual_cache`, then executing
`notebooks/10_paper_figures.ipynb`. The Smith-matched set comes from
`python -m real_environment.make_smith_figure_comparison`.

If you want a smaller set, `01`, `02`, `table1_condition_summary.csv`, `cell_summary.csv`,
`paired_effects.csv` and `nis_summary.csv` carry every claim in the report.

## Reading the arm names in the CSVs

| Name | Meaning |
|---|---|
| `random` | Primary treatment. Isotropic direction, log-uniform 1–100 m/s, uniform burn time 600–4800 s. |
| `nullburn` | **The control to use.** Same cohort, burn times and directions as every treatment, zero delta-v. |
| `radial` | Same plan, direction forced to +R. |
| `transverse` | Same plan, direction forced to +T. Called in-track in the report. |
| `normal` | Same plan, direction forced to +N. Called cross-track in the report. |
| `finite1800s` | Same plan, delta-v spread over 1800 s instead of applied at once. |
| `stress100to2000` | Same plan, magnitude widened to 100–2000 m/s. Twenty to four hundred times an operational burn. |
| `zerodv` | **Superseded.** Old control with a fixed 900 s burn time. Confounded by observation window. Do not pair against anything. |
| `sigma_km = 0` | Smith as released, no measurement noise. |
| `sigma_km = 10` | Noise matched to the R the filter already assumes. |

## Questions this source set can answer

- Why can Smith's covariance-trace reward not respond to a maneuver?
- Does the filter absorb the maneuver or diverge from it?
- Is the innovation actually the distribution the filter assumes, and does a burn at the floor change it?
- What does the filter think its own residual should be, and does a maneuver change that?
- How large could that response possibly be, even in the best case for the metric?
- What does adding measurement noise change, and what does it not change?
- Is the filter statistically consistent, and at which noise level?
- How well does NIS detect maneuvers, and at what false-alarm rate?
- Which burn direction is easiest to detect, and by how much?
- What happens at burn magnitudes far outside the operational range?
- Does spreading the burn over half an hour hide it?
- How small a burn becomes undetectable, and why?
- Why is reacquisition the wrong endpoint in this scenario?
- Does the frozen MARL policy behave differently when objects maneuver?
- How was any of this done without modifying Smith's code?

## What this source set does not claim

- It does not claim the MARL policy is more robust to maneuvers than the greedy heuristic. An
  earlier version did; that result was confounded and is retracted in section 3.5 and 4.3 of the
  report.
- It does not report NEES, training metrics or any fine-tuning. The study is inference-only.
- It does not measure the covariance anisotropy that would explain why cross-track burns are the
  most detectable. Section 4.7 offers that as an interpretation, not a result.
- It does not evaluate episodes longer than 5400 s.
