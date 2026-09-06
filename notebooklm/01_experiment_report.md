# Robustness of Frozen MARL Sensor Tasking to Unmodelled RSO Maneuvers

CSCE 723 project report. Onur Aytekin. Revised 27 August 2026.

This document is self-contained. Every number in it comes from the run described in section 3.

---

## 1. What this study set out to do, and what it actually found

The starting question was whether Tory D. Smith's trained multi-agent reinforcement learning
sensor-tasking policies degrade when the resident space objects (RSOs) they track perform
unannounced maneuvers.

That question turned out to be unanswerable in Smith's environment as released, for two
independent reasons that the study measured rather than assumed.

**First, the metric cannot move.** Smith's reward is built from the trace of the RSO covariance.
His unscented Kalman filter updates that covariance with `P = P - K S Kᵀ`, an expression that
never contains the measurement value. No measurement realization and no maneuver can raise it.
The one indirect route left open, an object that goes unobserved and so stops being updated, has a
measurable ceiling. Even if every tagged object lost custody the metric would move by 5.5 and 15.0
percent, against a tracking error that moves by 19939 and 22672 percent over the same arm.

**Second, the maneuver cannot threaten custody.** Smith's sensors have a 4° field of view and his
episodes last 5400 seconds. Escaping that field of view at geostationary altitude requires roughly
1467 km of displacement. A realistic geostationary station-keeping burn of 1 to 5 m/s displaces an
object by 4.5 to 23 km, under one percent of the field of view. The object never leaves the beam.

Both conclusions survive being pushed hard. Burns twenty times larger than any operational
maneuver, burns forced into the single most detectable direction, and burns spread over half an
hour instead of applied at once all leave the reward metric within a few percent of its control
value while the innovation moves by three orders of magnitude.

What the study therefore delivers is not a robustness benchmark. It is a measurement-backed
critique of the evaluation metric, plus a working replacement metric, plus the demonstration that
the replacement only becomes statistically valid once a measurement-noise defect in the released
code is corrected.

---

## 2. Two defects in the released Smith environment

### 2.1 The measurement path carries no noise

`measfun_gt` in `Environments/SSAMultiUtils.py` line 716 propagates the ground-truth ephemeris and
returns the position exactly. `radec2teme` is deterministic. Across `SSAMultiUtils.py` and
`SSAmulti_v2.py`, `np.random` appears only in catalog initialization and episode reset, never in
the measurement path.

Meanwhile `SSAmultiv2.step` in `Environments/SSAmulti_v2.py` line 306 hands the filter
`Rmeas = np.eye(3) * np.power(10, 2)`, that is an assumed measurement error of 10 km one sigma per
Cartesian axis.

The filter is therefore told that measurements carry a 10 km error that the code never generates.
This also does not match the thesis text, which describes an environment that "generates noisy
measurements within the FOV of each sensor" (Smith 2024, section 2.2, page 25).

The practical consequence is that the released environment reports a final estimate-to-truth error
of 0.024 to 0.058 km for coasting objects, roughly a thousand times better than the filter's own
stated uncertainty, and it sets the covariance floor from the assumed R rather than from the
measurements actually delivered.

### 2.2 The covariance update is measurement-independent

In `update`, `Environments/SSAMultiUtils.py` line 737, the state mean uses the innovation:

    M = M + K (y - MU)

but the covariance does not:

    S = S + R
    K = C S⁻¹
    P = P - K S Kᵀ

`S`, `K` and `C` depend only on the sigma points and on `R`. The measurement value `y` never
enters `P`. The covariance can therefore only shrink, regardless of how large the residual is. The
only coupling from the measurement to the covariance is second order, through the updated mean
changing the next step's sigma-point spread.

This is a property of the filter formulation, not a bug. But it means the covariance trace, which
is Smith's reward, is structurally decoupled from tracking accuracy. Section 4.11 measures this
directly. The standard deviation the filter assigns to its own residual is the same to three
significant figures whether the object coasts or performs a 2000 m/s burn.

---

## 3. Experimental design

### 3.1 Constraint

No file inside Smith's source tree was modified. Every intervention is applied by an external
wrapper that temporarily rebinds two module-level names for the duration of one environment step
and restores them afterwards:

- `Environments.SSAmulti_v2.meas`, so maneuvered and noise-perturbed truth objects are supplied.
- `Environments.SSAMultiUtils.update`, so the innovation and its covariance can be observed.

A regression test compares the wrapper against unchanged Smith with maneuvers disabled and reports
zero difference in observations, environment state, rewards and timing.

### 3.2 Scenario

Taken unchanged from Smith's three-agent evaluation configuration.

| Parameter | Value |
|---|---|
| RSOs | 100, geostationary regime, median radius 42019 km |
| Sensors | 3 ground sites (Socorro, Haystack, HEH Naval) |
| Field of view | 4° |
| Episode length | 5400 s |
| Assumed measurement noise R | (10 km)² per axis |
| Detection probability | 1.0 |
| Policies | frozen CNNv2 PPO checkpoint 018000, and advanced greedy |

Both policies run inference only. No training or fine-tuning was performed.

### 3.3 Factors

The core design is a full factorial over three factors at 100 seeds per cell.

| Factor | Levels |
|---|---|
| Measurement noise σ | 0 km (released Smith), 10 km (matched to the assumed R) |
| Maneuver | null-burn control, isotropic 1 to 100 m/s |
| Policy | advanced greedy, frozen CNNv2 PPO |

Five further arms were added afterwards, each at σ = 10 km, to answer questions the core design
could not. Every one of them reuses the treatment's own random plan and pairs against the same
null-burn control, so each arm differs from the control in exactly one respect.

| Extension arm | What it isolates |
|---|---|
| Radial | Burn along the position vector |
| In-track | Burn along the velocity vector (`transverse` in the data) |
| Cross-track | Burn along the orbit normal (`normal` in the data) |
| Isotropic 100 to 2000 m/s | Whether a burn far outside the operational range moves the metric |
| Finite, 1800 s | The same delta-v delivered over 1800 s instead of instantaneously |

One arm in the data, `zerodv`, is a superseded control retained only for the retraction in
section 3.5. It must not be paired against anything. Counting it, the run is 22 arms of 100 seeds,
2200 episodes.

Measurement noise is additive zero-mean isotropic Gaussian in Smith's TEME Cartesian measurement
space, applied to every RSO rather than only the tagged cohort, so that the maneuver signal is not
confounded with a noise-only difference between groups. It is drawn from a dedicated per-RSO
generator so that Smith's globally seeded catalog and covariance draw is never consumed; this was
verified by checking each episode's starting covariance against stored per-seed snapshots.

### 3.4 Maneuver model

Twenty of the 100 RSOs are tagged per episode. Each tagged object receives one impulsive burn:

- Direction drawn isotropically in the radial-transverse-normal frame.
- Magnitude drawn log-uniformly over 1 to 100 m/s.
- Burn time drawn uniformly over 600 to 4800 s.

The burn is injected as a differential two-body perturbation added to Smith's nominal SGP4
trajectory, so that a zero delta-v returns exactly to Smith and no post-burn state is converted
back into a two-line element set.

The extension arms change one line of that plan each and leave the rest of the draw untouched.
The direction arms replace the isotropic draw with a fixed radial, in-track or cross-track unit
vector, keeping every magnitude and every burn time. The stress arm widens the magnitude range to
100 to 2000 m/s, keeping the isotropic direction and the burn times. The finite arm keeps
magnitude, direction and start time and spreads the delta-v evenly over 1800 s, which is the
harder detection case because the offset arrives later and smaller.

### 3.5 The control arm, and a corrected design error

The control arm reuses the treatment's own random plan with the magnitude scaled to zero. The
random draws are consumed in the same order and quantity, so the control tags the same RSOs at the
same burn times with the same burn directions, and differs only in that the burn has no magnitude.
Agreement is asserted programmatically on every analysis run.

An earlier version of this study used a control with a fixed 900 s burn time. That gave the tagged
objects a median post-burn observation window of 4500 s against 2770 s in the randomly timed
treatment. The window difference alone moved reacquisition and produced an apparent custody effect
roughly one hundred times larger than the true one. That result is retracted; section 4.3 gives the
corrected numbers. The confounded arm is retained in the data under the name `zerodv` and must not
be paired against the treatment.

---

## 4. Results

### 4.1 Measurement noise raises tracking error by two orders of magnitude and leaves the reward metric untouched

Paired within policy and maneuver condition, σ = 10 km against σ = 0 km:

| Policy | change in median tagged tracking error | 95% CI |
|---|---|---|
| Advanced Greedy | +5.291 km | 4.792 to 5.797 |
| Frozen CNNv2 PPO | +4.377 km | 4.126 to 4.630 |

Over the same contrast the change in final mean covariance is 6.5e-06 for advanced greedy and
8.4e-08 for PPO, with confidence intervals spanning zero in both cases.

In absolute terms the coasting-object error moves from 0.058 km to 5.349 km for advanced greedy
and from 0.024 km to 4.401 km for PPO, while the mean final covariance stays at 0.01173 and
0.01059 respectively.

### 4.2 Maneuvers raise tracking error and leave the reward metric untouched

Paired against the matched null-burn control, at σ = 10 km:

| Policy | metric | change | 95% CI |
|---|---|---|---|
| Advanced Greedy | median tagged error | +23.465 km | 21.237 to 25.818 |
| Advanced Greedy | final mean covariance | +1.51e-06 | −4.56e-06 to +8.86e-06 |
| Advanced Greedy | mean reward per agent | +6.28e-08 | +1.28e-09 to +1.33e-07 |
| Frozen CNNv2 PPO | median tagged error | +20.250 km | 18.143 to 22.500 |
| Frozen CNNv2 PPO | final mean covariance | +8.74e-07 | +4.11e-07 to +1.39e-06 |
| Frozen CNNv2 PPO | mean reward per agent | +1.21e-08 | −1.06e-08 to +5.11e-08 |

The PPO covariance change is statistically distinguishable from zero but amounts to 0.008 percent
of the metric. The reward changes are of order 1e-08.

Per RSO, among the 2002 tagged objects that were observed at least once after their burn, 480 of
them, that is 24.0 percent, show a covariance change of exactly zero. The median absolute change
among observed objects is 5.8e-07. Over the same population the median displacement from the
nominal trajectory is 24.3 km, the 90th percentile is 174.1 km and the 99th percentile is 375.1 km.

### 4.3 Custody is essentially unaffected at operational burn sizes, and geometry explains why

Reacquisition means the object was observed at least once after its burn time. Paired against the
matched control at σ = 10 km:

| Policy | control | treatment | paired change | 95% CI |
|---|---|---|---|---|
| Advanced Greedy | 0.2975 | 0.2960 | −0.0015 | −0.0040 to 0.0000 |
| Frozen CNNv2 PPO | 0.7055 | 0.7050 | −0.0005 | −0.0015 to 0.0000 |

That is 0.15 and 0.05 percentage points across 2000 tagged objects per arm.

The geometric explanation. Measured displacement follows approximately

    d ≈ 0.00101 × Δv[m/s] × t[s]   km

Leaving a 4° field of view at a median geostationary radius of 42019 km requires about 1467 km of
displacement, which needs roughly 324 m/s given 4500 s of coast, or 810 m/s given 1800 s.

| Δv | displacement after 4500 s | fraction of the 4° field of view |
|---|---|---|
| 1 m/s | 4.5 km | 0.15% |
| 5 m/s | 22.7 km | 0.77% |
| 10 m/s | 45.3 km | 1.54% |
| 100 m/s | 453 km | 15.4% |

Typical geostationary station-keeping burns are 1 to 5 m/s. Custody loss through field-of-view
escape is therefore not merely rare in this scenario, it is unreachable for any realistic maneuver
at Smith's episode length and field of view. Reacquisition is the wrong endpoint here, and no
result measured on it can be attributed to maneuver robustness.

The table above says where the endpoint would start working. Section 4.8 pushes the burn into
that range and finds the first measurable custody effect in the study, about 6 percentage
points.

### 4.4 Neither policy responds to maneuvers, and the learned policy barely responds to noise

Comparing the control and treatment arms at σ = 10 km, seed by seed:

| Policy | episodes with identical pointing | azimuth decisions changed |
|---|---|---|
| Frozen CNNv2 PPO | 100 of 100 | 0 of 52014 |
| Advanced Greedy | 50 of 100 | 758 of 135466, 0.56% |

Comparing σ = 0 km against σ = 10 km at fixed maneuver condition:

| Policy | episodes with identical pointing | azimuth decisions changed |
|---|---|---|
| Frozen CNNv2 PPO | 97 to 98 of 100 | 109 to 127 of 52007, about 0.2% |
| Advanced Greedy | 0 to 1 of 100 | about 3500 of 135400, about 2.6% |

The frozen PPO policy issues a bit-identical action sequence whether or not the objects it is
tracking maneuver. This is mechanically expected. A maneuver reaches the policy only through the
filter estimate, and a displacement of a few tens of km at geostationary range is a few hundredths
of a degree, far below the 4° resolution of the observation grid.

A corollary. PPO reacquires far more objects than advanced greedy, 70.6 percent against 29.8
percent. That gap is present in the control arm too. It is a property of PPO's baseline coverage,
not of maneuver handling, and must not be reported as maneuver robustness.

### 4.5 NIS is the responsive metric, and matched noise is what makes it testable

The innovation is the only place a measurement value enters the filter. Normalized innovation
squared is recorded from Smith's own update call:

    NIS = ν' (S + R)⁻¹ ν,   ν = y − MU

Under a consistent filter NIS follows a chi-square distribution with three degrees of freedom, so
its mean is 3 and one percent of measurements exceed the p99 threshold of 11.345.

| Policy | σ | arm | group | measurements | mean NIS | above p99 |
|---|---|---|---|---|---|---|
| Advanced Greedy | 0 | control | non-maneuvering | 109563 | 1.27e-04 | 0.00% |
| Advanced Greedy | 0 | treatment | post-burn | 15112 | 16.33 | 16.44% |
| Advanced Greedy | 10 | control | non-maneuvering | 109733 | 2.897 | 0.97% |
| Advanced Greedy | 10 | control | post-burn label | 15158 | 2.952 | 1.07% |
| Advanced Greedy | 10 | treatment | non-maneuvering | 109867 | 2.897 | 0.96% |
| Advanced Greedy | 10 | treatment | post-burn | 14986 | 19.410 | 20.05% |
| Frozen CNNv2 PPO | 0 | control | non-maneuvering | 42534 | 4.60e-04 | 0.00% |
| Frozen CNNv2 PPO | 0 | treatment | post-burn | 5196 | 23.500 | 23.73% |
| Frozen CNNv2 PPO | 10 | control | non-maneuvering | 42540 | 2.624 | 0.73% |
| Frozen CNNv2 PPO | 10 | control | post-burn label | 5241 | 2.794 | 0.78% |
| Frozen CNNv2 PPO | 10 | treatment | post-burn | 5194 | 26.371 | 27.47% |

Three things follow.

**At σ = 10 km the filter is consistent.** The non-maneuvering mean is 2.897 and 2.624 against an
expectation of 3, with 0.97 and 0.73 percent above p99 against an expected 1 percent. The
chi-square test is therefore calibrated and a threshold has meaning.

**At σ = 0 km the test is unusable, not merely conservative.** The non-maneuvering mean is
1.3e-04 and 4.6e-04 against an expectation of 3. The raw separation between groups is about five
orders of magnitude, so an empirically calibrated threshold would work, but the standard
chi-square theory does not apply and the released environment is unphysical anyway.

**The attribution is airtight.** In the matched control, measurements carrying the post-burn label
show 2.952 and 2.794 with p99 rates of 1.07 and 0.78 percent, matching the non-maneuvering
baselines. Since that control shares the treatment's cohort, burn times and burn directions and
differs only in delta-v magnitude, the elevation to 19.410 and 26.371 in the treatment is
attributable to the maneuver alone.

### 4.6 Detection is strongly dose-dependent

Taking the first post-burn innovation per object, which is where the signature lives before the
filter re-absorbs the object, the p99 detection rate at σ = 10 km:

| Δv band | Advanced Greedy median NIS | detection | PPO median NIS | detection |
|---|---|---|---|---|
| 1 to 3.2 m/s | 1.315 | 0.0% | 1.763 | 0.8% |
| 3.2 to 10 m/s | 1.417 | 2.0% | 2.241 | 1.2% |
| 10 to 32 m/s | 4.048 | 24.1% | 3.439 | 7.5% |
| 32 to 100 m/s | 13.418 | 50.9% | 6.559 | 36.9% |

Burns below roughly 10 m/s remain undetectable because the offset they produce before
re-observation sits under the 10 km noise floor that Smith's assumed R imposes. Section 4.12 shows
this floor directly in the residual rather than inferring it from detection rates. Detecting
station-keeping-scale burns would require an R matched to a realistic sensor rather than to 10 km,
a detector that accumulates innovations across several revisits, or an adaptive process noise that
inflates the covariance when the innovation is large.

### 4.7 Burn direction decides detectability, and the reward metric ignores all of it

Three arms hold the cohort, the magnitudes and the burn times fixed and force the direction to be
purely radial, in-track or cross-track. At σ = 10 km, against the same matched control:

| Policy | direction | post-burn mean NIS | above p99 | first-burn detection | median tagged error | mean covariance |
|---|---|---|---|---|---|---|
| Advanced Greedy | in-track | 4.93 | 7.8% | 8.6% | +19.95 km | −2.1e-06, CI spans zero |
| Advanced Greedy | radial | 6.48 | 10.8% | 16.0% | +21.35 km | −1.9e-06, CI spans zero |
| Advanced Greedy | isotropic | 19.41 | 20.1% | 20.1% | +23.46 km | +1.5e-06, CI spans zero |
| Advanced Greedy | cross-track | 41.42 | 29.9% | 31.2% | +26.19 km | +1.0e-05 |
| Frozen CNNv2 PPO | in-track | 14.64 | 21.8% | 8.2% | +16.65 km | +1.3e-06 |
| Frozen CNNv2 PPO | radial | 19.94 | 25.9% | 11.3% | +19.11 km | +3.0e-08, CI spans zero |
| Frozen CNNv2 PPO | isotropic | 26.37 | 27.5% | 11.8% | +20.25 km | +8.7e-07 |
| Frozen CNNv2 PPO | cross-track | 44.92 | 32.1% | 14.4% | +25.10 km | +1.1e-06 |

The ordering is the same for both policies. Cross-track is the loudest burn and in-track the
quietest, a factor of 8.4 in mean NIS for advanced greedy and 3.1 for PPO. The isotropic arm sits
between them, which is what an average over the sphere should do.

Across that whole spread the covariance change stays at or below 1.0e-05, which is under 0.1
percent of the metric, and for three of the eight rows its confidence interval still contains
zero. Detectability varies by nearly an order of magnitude and the reward metric does not register
the difference.

A plausible mechanism is that the filter's uncertainty in this geostationary catalog is far larger
in-track than cross-track, so an equal displacement is many more standard deviations out of plane
than along it. This study did not measure the covariance anisotropy, so that explanation is an
interpretation of the ordering rather than a result.

### 4.8 A burn twenty times larger than the operational range still leaves the metric flat

The stress arm draws magnitude log-uniformly over 100 to 2000 m/s. Geostationary station-keeping
is 1 to 5 m/s, so the smallest burn in this arm is already twenty times the largest operational
one and the largest is a plane change.

| Quantity | Advanced Greedy control | stress | Frozen CNNv2 PPO control | stress |
|---|---|---|---|---|
| Median tagged tracking error | 5.35 km | 1071.84 km, +19939% | 4.40 km | 1002.28 km, +22672% |
| Tagged covariance | 0.011459 | 0.012187, +6.36% | 0.010328 | 0.010918, +5.71% |
| Catalog mean covariance | 0.011735 | 0.012127, +3.35% | 0.010594 | 0.010719, +1.19% |
| Reacquired fraction | 0.2975 | 0.2335 | 0.7055 | 0.6460 |
| Post-burn mean NIS | 2.95 | 1723.2 | 2.79 | 4137.3 |
| First-burn detection at p99 | 0.0% | 84.8% | 0.6% | 84.9% |

Detection is essentially saturated. Broken out by magnitude, the first-burn detection rate runs
75.2, 88.3, 86.0 and 89.4 percent for advanced greedy across 100 to 200, 200 to 400, 400 to 800
and 800 to 2000 m/s, and 71.3, 83.2, 90.6 and 94.8 percent for PPO.

The covariance still moves by single-digit percent while the tracking error it is supposed to
stand in for moves by four orders of magnitude. Reacquisition finally responds, losing 6.4 and 6.0
percentage points, which is the first arm in the study where custody is measurably affected at
all.

### 4.9 The covariance metric has a ceiling, and it is small

Because `P = P - K S Kᵀ` never reads the measurement, a maneuver can reach the covariance trace by
exactly one route. An object that stops being observed stops having its covariance reduced. That
gives an arithmetic upper bound on the entire effect, which can be evaluated from the run itself.

Take the measured end-of-run covariance of the untagged objects and of the tagged objects that
were never re-observed after their burn, then ask what the catalog mean would be if every tagged
object went unseen. That is the most a maneuver could possibly do to Smith's reward.

| Policy | control catalog mean | measured at 100 to 2000 m/s | ceiling, all 20 tagged unseen |
|---|---|---|---|
| Advanced Greedy | 0.011735 | 0.012127, +3.35% | 0.012381, +5.51% |
| Frozen CNNv2 PPO | 0.010594 | 0.010719, +1.19% | 0.012179, +14.96% |

A maneuver that destroyed custody of every tagged object would move the metric by 5.5 and 15.0
percent. Over the same arm the median tracking error moves by 19939 and 22672 percent. The metric's
entire reachable range against maneuvers is three orders of magnitude smaller than the quantity it
is standing in for, and the study's measured effects use only a fraction of even that.

The PPO ceiling is the larger of the two only because PPO re-observes more objects in the control
and therefore has more to lose. That is the same baseline coverage property as in section 4.4 and
is not evidence of maneuver sensitivity.

### 4.10 Spreading the burn over half an hour halves detectability and changes nothing else

Electric propulsion does not deliver its delta-v in an instant. The finite arm keeps the cohort,
the magnitudes, the directions and the start times of the primary arm and spreads each burn evenly
over 1800 s. The offset it produces is smaller and arrives later, so it is the harder case.

| Quantity | Advanced Greedy impulsive | finite 1800 s | PPO impulsive | finite 1800 s |
|---|---|---|---|---|
| Median displacement from nominal | 28.20 km | 16.20 km | 28.47 km | 16.43 km |
| Change in median tagged error | +23.46 km | +14.58 km | +20.25 km | +11.99 km |
| Post-burn mean NIS | 19.41 | 10.28 | 26.37 | 14.05 |
| Post-burn measurements above p99 | 20.1% | 12.1% | 27.5% | 16.7% |
| First-burn detection at p99 | 20.1% | 11.7% | 11.8% | 3.5% |
| Change in mean covariance | +1.5e-06, CI spans zero | +1.4e-06, CI spans zero | +8.7e-07 | +5.0e-07 |
| Change in reacquired fraction | −0.0015 | −0.0015 | −0.0005 | 0.0000 |

Spreading the burn costs about half the signal on every innovation measure. Mean NIS falls to 53
percent of its impulsive value for both policies, and first-burn detection falls from 20.1 to 11.7
percent and from 11.8 to 3.5 percent.

The observation timing is matched, which rules out the obvious alternative explanation. The median
delay from burn start to first post-burn observation is 1374.8 s in both arms for advanced greedy
and 459.3 against 459.4 s for PPO. The detectability drop is the burn model, not a different look
angle or a different revisit.

None of this reaches the covariance. The advanced greedy change still has a confidence interval
containing zero, and the PPO change of 5.0e-07 is 0.005 percent of the metric. Reacquisition is
unchanged for advanced greedy and exactly zero for PPO.

### 4.11 The residual grows instead of decaying, and the filter's own uncertainty never notices

The residual diagnostics answer a question the pooled statistics of section 4.5 cannot. A maneuver
could raise NIS as a single spike that the filter then absorbs, which would be a well-behaved
filter reacting to a surprise. That is not what happens.

Normalised residual, which is the square root of NIS, binned by how long after the burn the
measurement arrived. Under a consistent filter it should sit at 1.538 at every delay.

| Policy | condition | under 600 s | 3600 s and later |
|---|---|---|---|
| Advanced Greedy | control | 1.52 | 1.49 |
| Advanced Greedy | isotropic 1-100 m/s | 1.62 | 2.38 |
| Advanced Greedy | finite 1800 s | 1.52 | 2.24 |
| Advanced Greedy | isotropic 100-2000 m/s | 7.27 | 30.51 |
| Frozen CNNv2 PPO | control | 1.45 | 1.52 |
| Frozen CNNv2 PPO | isotropic 1-100 m/s | 1.57 | 3.04 |
| Frozen CNNv2 PPO | finite 1800 s | 1.44 | 2.83 |
| Frozen CNNv2 PPO | isotropic 100-2000 m/s | 8.40 | 59.94 |

The control sits at 1.45 to 1.52 at every delay, which is the null. Every treatment grows
monotonically. The filter does not absorb the maneuver, it falls further behind it, because the
object keeps drifting from the trajectory the filter is propagating and the correction never
catches up. By the end of the run the stress arm's measurements are 30 and 60 standard deviations
from the prediction.

Against that, the standard deviation the filter assigns to its own residual. This is the innovation
covariance resolved along the direction the residual actually landed, so it is what the filter
believes the residual should be.

| Policy | control, late in the episode | 100 to 2000 m/s, late in the episode |
|---|---|---|
| Advanced Greedy | 10.069 km | 10.062 km |
| Frozen CNNv2 PPO | 10.765 km | 10.745 km |

The two agree to three significant figures. A burn that drives the residual to 314 km leaves the
filter's estimate of how big that residual should be at 10.7 km. This is section 2.2 measured
rather than derived. `S` and `R` do not contain the measurement, so the innovation covariance
cannot respond, and it settles onto the assumed 10 km within about 1200 s as the state covariance
becomes negligible beside `R`.

That is also why the reward metric and the tracking error can move in opposite directions. The
covariance keeps shrinking on schedule while the estimate diverges, and the reward reads the
shrinking half.

The residual in kilometres explains the detection floor of section 4.6 directly. The median
post-burn residual is 15.57 km for advanced greedy and 16.55 km for PPO in the control, which is
what a 10 km per-axis noise gives. The 1 to 100 m/s arm moves that to 19.62 and 23.48 km. A burn
has to displace an object by more than about 15 km before its residual is distinguishable from
noise at all, and that is the same 10 km assumed `R` that sets the covariance floor.

### 4.12 The residual against the noise band, at a detectable burn and at the floor

Section 4.6 put the detection floor at roughly 10 m/s and attributed it to the 10 km measurement
noise. That was an inference from detection rates. The signed residual shows it directly.

Two dedicated runs hold everything fixed except the burn size. Both are along-track, both use the
primary cohort and burn-time plan, and both run at 10 km noise. One is at 200 m/s, well above the
floor, and one at 5 m/s, well below it. The residual is recorded per component with the standard
deviation the filter assigns to that component, which NIS discards when it collapses the three
into one number.

Standard deviation of the residual before and after the burn, in kilometres:

| Component | 200 m/s before | 200 m/s after | 5 m/s before | 5 m/s after |
|---|---|---|---|---|
| x | 10.24 | 62.56 | 10.24 | 10.43 |
| y | 9.91 | 62.09 | 9.92 | 10.15 |
| z | 10.27 | 39.58 | 10.27 | 10.07 |

At 200 m/s the scatter widens by a factor of six within minutes of the burn and a large share of it
clears the 3 sigma line. At 5 m/s the after-burn scatter is 10.43, 10.15 and 10.07 km against a
before-burn 10.24, 9.92 and 10.27 km. The burn leaves no trace the measurement noise does not
already contain.

This is the detection floor made concrete rather than inferred. It is a measurement-resolution
limit, not a detector-tuning artefact, because no threshold placed on this residual could separate
the 5 m/s column from its own pre-burn half.

The z component widens less than x and y, 39.58 against 62.56 and 62.09. An along-track burn is
in-plane and excites little out-of-plane motion, and for a near-equatorial catalogue the TEME z
axis is close to the orbit normal. That is the expected ordering rather than a measured
decomposition, since the residual was not resolved into the radial-transverse-normal frame.

### 4.13 The whitened innovation against the normal it should follow

NIS answers whether the quadratic form has the right size. It cannot say whether the underlying
innovation is the distribution the filter assumes, because it discards the sign and collapses the
three components into one number. Whitening the innovation by the Cholesky factor of the full
innovation covariance recovers that. Under a consistent filter each whitened component is standard
normal, and the sum of their squares is NIS exactly, so the whitened vector checks the NIS value
rather than replacing it.

Pooling all three components over the objects that never burn gives the null.

| Population | samples | mean | standard deviation |
|---|---|---|---|
| Never maneuvering | 66069 | +0.001 | 0.989 |
| 5 m/s, after the burn | 4326 | +0.021 | 0.995 |
| 200 m/s, after the burn | 3837 | +0.890 | 4.614 |

The null sits at a mean of +0.001 and a standard deviation of 0.989 against the 0 and 1 a
consistent filter requires, and the quantiles lie on the diagonal across the whole plotted range.
Broken out per component the null is 0.976, 0.987 and 1.003 for advanced greedy. For the frozen
PPO policy it is 0.904, 0.917 and 0.981, slightly conservative, meaning that policy's covariance
runs a little larger than the errors it actually makes.

The 5 m/s arm is the result. After the burn its mean is +0.021 and its standard deviation 0.995,
against a null of +0.001 and 0.989. Its quantiles lie on the same diagonal. A burn at that
magnitude does not merely fail a threshold, it leaves the innovation distribution where it was.
That is the strongest available statement of the detection floor, because it does not depend on
which detector or which threshold is chosen.

At 200 m/s the standard deviation reaches 4.614 and the quantiles depart from the diagonal on both
tails. The mean also moves, to +0.890, but that part is not robust. Per component it is −0.590,
+2.389 and +0.870 for advanced greedy and +1.102, −0.625 and +0.897 for the frozen PPO policy,
differing in sign between the two. Repeated looks at a small number of tagged objects with large
individual offsets produce exactly that. The reproducible signature is the spread, not a bias.

One methodological note. Whitening by the diagonal of the innovation covariance rather than by its
Cholesky factor is not equivalent here. The ratio of the diagonal form to the true NIS has a median
of 1.000 but a first percentile of 0.042, so the shortcut agrees in the bulk and fails by more than
an order of magnitude in the tail, which is the part a detection claim rests on.

---

## 5. Conclusions

1. Smith's covariance-trace reward is structurally incapable of responding to a maneuver or to
   measurement quality. Any robustness claim measured on it is measuring a quantity that cannot
   move. This is shown from the filter algebra and from 2200 episodes of measurement.

   The single indirect route the metric does have, losing custody of an object, is bounded. With
   every tagged object left unseen the reward would change by 5.5 and 15.0 percent, three orders
   of magnitude below the tracking error over the same arm.

2. The released environment generates noiseless measurements while assuming a 10 km measurement
   error. Correcting this to match the filter's own assumption raises realized tracking error from
   0.024 to 4.401 km for PPO and from 0.058 to 5.349 km for advanced greedy, and it makes the
   filter statistically consistent.

3. That correction is the precondition for a valid maneuver test. With matched noise, NIS detects
   maneuvers on 20.05 and 27.47 percent of post-burn measurements against a matched-control false
   alarm rate of 1.07 and 0.78 percent.

4. Detection is dose-dependent, reaching about 51 and 37 percent for burns of 32 to 100 m/s,
   falling to near zero below 10 m/s, and saturating near 85 percent once burns exceed 100 m/s.

5. Detectability depends strongly on burn direction. Cross-track burns raise the mean post-burn
   NIS by a factor of 8.4 over in-track burns for advanced greedy and 3.1 for PPO, at identical
   magnitude and burn time. The covariance metric does not distinguish the two cases.

6. Pushing the burn magnitude to 100 to 2000 m/s, twenty to four hundred times an operational
   station-keeping maneuver, still moves the catalog mean covariance by only 3.35 and 1.19
   percent while tracking error moves by 19939 and 22672 percent.

7. The filter does not absorb the maneuver, it diverges from it. The normalised residual grows
   monotonically with time since burn, reaching 30 and 60 standard deviations in the 100 to 2000
   m/s arm, while the standard deviation the filter assigns to that residual stays at 10.07 and
   10.75 km in both the control and the treatment. The estimate diverges while the covariance
   shrinks on schedule, and the reward reads the shrinking half.

8. Spreading the same delta-v over 1800 s halves every innovation measure, with mean post-burn
   NIS falling to 53 percent of its impulsive value for both policies. Observation timing is
   matched between the two arms, so the drop is the burn model rather than a different revisit.
   The covariance and reacquisition are unchanged.

9. Custody loss is geometrically unreachable for operational-scale burns at Smith's episode
   length and field of view, so reacquisition cannot serve as a maneuver-robustness endpoint in
   this environment. It responds only in the 100 to 2000 m/s arm, and then by about 6 percentage
   points.

10. The frozen MARL policy does not respond to maneuvers at all, issuing bit-identical pointing in
   100 of 100 paired episodes. Its higher reacquisition rate relative to the greedy heuristic is a
   baseline coverage property and is not evidence of maneuver handling.

---

## 6. Reproducibility

| Item | Location |
|---|---|
| Design of record | `config/noise_maneuver_grid.yaml` |
| Measurement noise | `real_environment/measurement_noise.py` |
| Maneuver plans | `real_environment/maneuver_plan.py` |
| NIS recording | `real_environment/innovation_statistics.py` |
| Environment wrapper | `real_environment/maneuver_environment.py` |
| Grid driver, core factorial | `real_environment/run_noise_maneuver_grid.sh` |
| Grid driver, direction arms | `real_environment/run_direction_grid.sh` |
| Grid driver, stress arm | `real_environment/run_stress_grid.sh` |
| Grid driver, finite burn arm | `real_environment/run_finite_burn_grid.sh` |
| Analysis | `real_environment/analyze_noise_maneuver_grid.py` |
| Figure data reduction | `real_environment/build_paper_figure_cache.py` |
| Residual record cache | `real_environment/build_residual_cache.py` |
| Paper figures | `notebooks/10_paper_figures.ipynb` |
| Outputs | `results/noise_maneuver_grid/` |
| Input integrity digests | `config/immutable_inputs.yaml` |

Runtime is Python 3.10.20 with numpy 1.23.5, ray 2.2.0 and tensorflow 2.13.0. Runs checkpoint
after every episode and resume with `--resume`. 49 unit tests cover the maneuver plans, the
measurement-noise stream isolation, the NIS computation against its analytic definition, and the
matched-control invariants.

## 7. Source

Smith, T. D. (2024). *Sensor Network Tasking for Space Domain Awareness Using Multi-Agent
Reinforcement Learning*. Master's thesis, Massachusetts Institute of Technology.
Code: github.com/ARCLab-MIT/MARLSSA
