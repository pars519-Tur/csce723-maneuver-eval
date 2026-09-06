# Technical Appendix: How the Interventions Were Made Without Modifying Smith's Code

CSCE 723 project. Companion to the experiment report.

This document answers the mechanism questions. How was noise injected. How was NIS obtained. How
do we know Smith's environment still behaves as released. What exactly does each control arm hold
fixed.

---

## 1. The non-modification constraint and how it is enforced

Smith's source tree at `try_code/MARLSSA-main` is treated as read-only. Two independent checks
back this up.

**Digest verification.** `config/immutable_inputs.yaml` records SHA-256 digests of the whole Smith
source tree, of the CNNv2 reward-1 checkpoint 018000, and of the individual files the study depends
on, namely `Environments/SSAmulti_v2.py`, `CNN_v2.py`, `evaluation_functions.py` and
`Environments/scenario_configs/3_agents_eval.yaml`.

**Behavioural regression.** `real_environment/zero_maneuver_regression.py` instantiates unchanged
Smith and the wrapper side by side on the same seed, steps both, and compares. It reports maximum
absolute differences of exactly 0.0 in observations, environment state, rewards and agent timing.
This test passed before the interventions were added and still passes after.

## 2. The injection mechanism

Smith's `SSAmultiv2.step` calls `meas(...)` resolved through its own module globals. The wrapper
therefore rebinds `Environments.SSAmulti_v2.meas` for the duration of one `step` call and restores
it in a `finally` block. The same pattern is used for `Environments.SSAMultiUtils.update`.

Nothing is written to disk inside Smith's tree, and nothing is patched globally or permanently. If
the wrapper is not in use, imports of Smith behave exactly as shipped.

The wrapper additionally asserts that the number of intercepted measurement calls equals the number
of agents whose action completed in that step, so a silent desynchronisation between the wrapper's
bookkeeping and Smith's internal loop would raise rather than corrupt the record.

## 3. Measurement noise

### 3.1 Where it enters

Smith's `meas` obtains the truth measurement by calling `measfun_gt(tle_truth[i], time_update)`,
which calls `sv_ephem.compute(time)` and then reads three attributes, `g_dec`, `g_ra` and
`elevation`, feeding them to `radec2teme`.

The wrapper already supplies its own truth objects through the `tle_truth` argument. Noise is
therefore applied inside those objects. `NoisyMeasurementAdapter.compute` evaluates the true
position, adds a Gaussian perturbation, and re-encodes the perturbed position into the three
attributes that `measfun_gt` reads:

    r    = |p|
    ra   = atan2(p_y, p_x) mod 2π
    dec  = asin(p_z / r)
    elevation = (r − 6378.16) × 1000

This inverts `radec2teme` exactly, so Smith's measurement function reconstructs the perturbed
Cartesian position with no loss. A unit test confirms that with sigma set to zero the adapter is
numerically invisible, round-tripping a position through `measfun_gt` to within 1e-9 km.

### 3.2 Why the noise is applied to every RSO

Both the tagged cohort and the untagged population are wrapped. If only the tagged objects carried
noise, the difference between the groups would confound the maneuver signal with a noise-only
difference in measurement quality.

### 3.3 Random-stream isolation

Smith's `reset` draws the initial catalog and the initial covariance from the globally seeded numpy
stream. Consuming from that stream would change the initial conditions and destroy the paired
design.

The noise therefore uses `numpy.random.default_rng` seeded per RSO from the triple
(noise seed, base seed, RSO id). This is a separate bit generator that never touches
`numpy.random`'s global state. Two checks confirm the isolation:

- A unit test seeds the global stream, draws 2000 values from twenty per-RSO generators, and
  verifies the global stream's next outputs are unchanged.
- Every episode's starting covariance is compared against the stored per-seed snapshot in
  `results/snapshots/`. A leak would change the catalog and this check would fail.

### 3.4 What was deliberately not changed

The assumed measurement noise `Rmeas = eye(3) × 10²` inside `SSAmultiv2.step` was left exactly as
Smith wrote it. Setting the injected sigma to 10 km makes the realized noise match that assumption
rather than the other way round. This keeps the deviation from Smith minimal and means the
consistency result in the report is a property of Smith's own stated model, not of a retuned one.

## 4. Normalized innovation squared

### 4.1 Definition and source

Smith's `update` receives the sigma-point measurement covariance `S`, the assumed noise `R`, the
measurement `ymeas` and the predicted measurement `MU`, then computes the innovation covariance as
`S + R`. The recorder wraps `update` and computes

    ν   = ymeas − MU
    NIS = ν' (S + R)⁻¹ ν

from those same arrays before delegating to the original function, whose return value it passes
through untouched. A unit test verifies the wrapped call returns Smith's mean and covariance
bit-for-bit.

The solve uses `numpy.linalg.solve` with a pseudo-inverse fallback for the singular case.

### 4.2 Attribution bookkeeping

`meas` loops over the selected RSOs and calls `update` once per object whose detection probability
is positive. Before delegating, the wrapper declares to the recorder the list of RSO ids that will
be updated, in that same order and filtered by the same detection test Smith applies. Each recorded
NIS value therefore carries the correct RSO id, the agent, the time, whether the object is in the
tagged cohort, and how long after its own burn the measurement was taken.

Three assertions guard this. The maneuvering flag on each record must match membership of the
episode's plan. The seconds-after-burn field must equal measurement time minus that object's own
burn time. Update calls beyond the declared batch are ignored rather than misattributed.

### 4.3 The null distribution

For a three-dimensional measurement, a consistent filter gives NIS distributed as chi-square with
three degrees of freedom, so the mean is 3, the 95th percentile is 7.8147 and the 99th percentile
is 11.3449. A Monte Carlo unit test draws 20000 innovations from a known covariance and confirms
the recorder reproduces mean 3 and the correct tail rates.

## 5. The maneuver model

### 5.1 Truth injection

A tagged object's truth is Smith's own SGP4 trajectory plus a differential two-body perturbation.
Two Cartesian trajectories are integrated from the same burn state, one coasting and one receiving
the impulse, and their state difference is added to the nominal SGP4 path.

Two properties follow. A delta-v of zero returns exactly to Smith's trajectory. And no post-burn
state is ever converted back into a two-line element set, which avoids the accuracy loss and
ambiguity of a TLE round trip.

The impulse is applied in the radial-transverse-normal frame built from the object's position and
velocity at the burn epoch.

### 5.2 Cohort selection

The tagged cohort is drawn with `numpy.random.RandomState(base_seed + 104729)` choosing without
replacement, unchanged from the original fixed-delta-v evaluation. A random-maneuver arm therefore
tags exactly the same objects as an earlier fixed-delta-v arm at the same seed and fraction, which
keeps the two comparable.

### 5.3 Random plan

Per tagged object, drawn from `default_rng([plan seed, base seed])` in a fixed order:

- Direction: a standard normal three-vector, normalized. This is Muller's method and gives a
  uniform distribution on the sphere. A unit test confirms the per-axis mean is zero and the mean
  of the squared components is one third, which distinguishes it from an along-track-only plan.
- Magnitude: exponential of a uniform draw between log(1) and log(100), so every decade of
  delta-v is equally represented. A unit test confirms the base-10 logarithm has mean 1 and
  standard deviation 2/sqrt(12).
- Burn time: uniform over 600 to 4800 seconds.

### 5.4 The extension arms

Each extension arm changes one line of the random plan of section 5.3 and leaves every other draw
identical, so all of them pair against the same null-burn control.

| Arm | Flag | Change from the random plan |
|---|---|---|
| Radial | `--direction-mode radial` | Unit vector fixed to +R, magnitudes and burn times unchanged |
| In-track | `--direction-mode transverse` | Unit vector fixed to +T |
| Cross-track | `--direction-mode normal` | Unit vector fixed to +N |
| Stress | `--delta-v-min-mps 100 --delta-v-max-mps 2000` | Log-uniform range widened, direction still isotropic |
| Finite burn | `--burn-duration-seconds 1800` | Same delta-v applied as a constant acceleration over 1800 s |

The direction modes leave the number of random draws unchanged, so the magnitude and burn-time
streams are not shifted relative to the isotropic arm. The stress arm draws from the same
log-uniform generator with different bounds, which preserves draw order but not the values, so it
pairs against the null-burn control rather than against the 1 to 100 m/s arm.

The finite arm resolves the RTN burn direction into TEME at the burn epoch, then integrates the
perturbed trajectory with a constant thrust acceleration of `delta_v_teme / duration` held fixed
in that inertial direction from the start time to the start time plus 1800 s, and coasts
afterwards. The two legs are integrated separately with DOP853 rather than through one
discontinuous right-hand side, so the solver stays accurate across thrust cut-off. Total delta-v
is identical to the impulsive arm. The offset it produces is smaller and arrives later, so it is
the harder detection case rather than an easier one.

### 5.5 The matched null-burn control

The control is the same `RandomManeuverConfiguration` with a `magnitude_scale` of zero. Because the
scale multiplies the magnitude only after it has been drawn, the generator consumes the same draws
in the same order. The control therefore has:

- the same tagged cohort,
- the same burn times,
- the same burn directions,
- zero delta-v.

`verify_control_is_matched` in the analyser re-checks all of this against the saved episodes on
every run and writes the counts to `control_matching_check.json`. All four policy-by-noise cells
report matched with zero mismatches.

### 5.6 The design error this replaced

The first control used `ManeuverConfiguration` with a delta-v of zero, which carries a fixed burn
time of 900 s. Because the treatment drew burn times over 600 to 4800 s, the control's tagged
objects had a median post-burn observation window of 4500 s against 2770 s in the treatment.

Reacquisition counts observations after the burn time, so a longer window raises it mechanically.
The confounded design reported a maneuver-induced reacquisition drop of 15.55 and 8.25 percentage
points. The matched design gives 0.15 and 0.05 points. Roughly ninety-nine percent of the reported
effect was the observation window.

A useful consistency check falls out of this. Because both control variants apply zero delta-v,
they simulate physically identical episodes and differ only in labelling. Their tracking-error and
covariance statistics agree to ten significant figures, while their reacquisition figures differ.
That is exactly the signature of a labelling-only confound.

## 6. Truth-based diagnostics

Smith's filter is never given truth. The diagnostics that compare estimate to truth are computed
outside the filter, in Smith's own TEME measurement space, using his `measfun` to convert the
filter's orbital-element mean to Cartesian. Three quantities are recorded per tagged object at
60-second intervals:

- estimate-to-truth error, the physical tracking error,
- nominal-to-truth offset, the maneuver displacement itself,
- estimate-to-nominal error, the filter's error against the unmaneuvered path.

These use the noise-free trajectories, not the noisy measurement adapters, so the diagnostics
measure real accuracy rather than a noisy sample of it.

## 7. Statistics

All contrasts are paired at the episode level on seeds both arms completed. Confidence intervals
are 10000-resample bootstrap percentile intervals on the paired differences, with a fixed seed so
the numbers reproduce.

## 8. Test coverage

49 unit tests across seven modules, all passing, covering the maneuver plans and their
distributions, the matched-control invariants, the measurement-noise round trip and stream
isolation, the NIS computation against its analytic definition and its chi-square null, the truth
geometry, the policy adapters and the scenario manifests. The zero-maneuver behavioural regression
against unchanged Smith is bit-exact.

## 9. Runtime

Python 3.10.20, numpy 1.23.5, ray 2.2.0, tensorflow 2.13.0, gym 0.23.1, pyephem. Inference only,
no training. 22 arms of 100 seeds, 2200 episodes, covering the core factorial, the corrected
control, the three direction arms, the stress arm and the finite-burn arm. Every run checkpoints
after each episode and resumes with `--resume`.

Figures are built in two stages so they can be regenerated without rereading the episode archives.
`real_environment/build_paper_figure_cache.py` reduces the arm pickles into a small cache, and
`notebooks/10_paper_figures.ipynb` reads that cache plus the analysis CSVs. Because the episode
histories are event sampled and the two policies log different numbers of samples over the same
5400 s, each episode's reduced curve is interpolated onto a shared 10 s grid before averaging.
