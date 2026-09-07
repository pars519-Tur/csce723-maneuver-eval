# CSCE 723 Maneuver Evaluation

Evaluates Tory Smith's released multi-agent sensor-tasking policies (MARLSSA) under
unmodelled RSO maneuvers, for the paper *Maneuvers the Reward Cannot See* and as the
prototype study for the maneuver-aware MARL thesis.

No file inside `try_code/MARLSSA-main` is modified. Every intervention is external
module-level rebinding, applied for one call and restored in a `finally` block. A SHA-256
manifest of the released tree is verified before every run, and a zero-maneuver regression
compares the wrapper against the stock environment bit-exactly.

The design of record for the experimental arms is `config/noise_maneuver_grid.yaml`.

## Layout

```
real_environment/   wrapper, maneuver model, noise injection, runner, analysis, caches
config/             experiment and grid definitions, manifest schema
tests/              55 unit tests in 8 files
notebooks/          10_paper_figures.ipynb, reads only the cached outputs
results/noise_maneuver_grid/   arm pickles, analysis/, paper_cache/, figures_paper/,
                               figures_discussion/
results/residual_traces/       four short fixed-magnitude arms for the residual figures
```

## Environments

- Arm runs, analysis and caches need the `marlssa` conda environment, which carries the
  released code's TF1-era dependencies. All commands below run from this directory with
  that environment active and `PYTHONPATH=.`.
- The figure notebook runs on a standard scientific Python (matplotlib, pandas, scipy).

## Reproduction

```bash
# 0. tests
python -m unittest discover -s tests

# 1. arms. One evaluation_runner call per arm; the full flag matrix for the 22-arm grid
#    is config/noise_maneuver_grid.yaml. Example, the primary PPO arm:
python -m real_environment.evaluation_runner \
  --policy frozen_cnnv2_ppo --condition maneuver --random-maneuvers \
  --sigma-km 10 --seeds 100 --record-nis \
  --output results/noise_maneuver_grid/frozen_cnnv2_ppo_random_sigma10_100seeds.p
#    The residual-trace arms have their own launcher:
bash real_environment/run_residual_traces.sh 10 4

# 2. analysis and caches
python -m real_environment.analyze_noise_maneuver_grid --seeds 100
python -m real_environment.build_paper_figure_cache --seeds 100
python -m real_environment.build_residual_cache --seeds 100

# 3. figures
jupyter nbconvert --to notebook --execute --inplace notebooks/10_paper_figures.ipynb
```

The paper build reads
`results/noise_maneuver_grid/figures_paper` in place.

## Seeds

- Episode seeds run 0-99 and are paired across arms. A treatment and its matched control
  at the same seed share the catalog draw, the tagged cohort, the burn times and the burn
  directions, and differ only in delta-v.
- Measurement noise draws from `numpy.random.default_rng((noise_seed, base_seed, rso_id))`
  with noise seed 20260823, isolated from the environment's own stream.
- Bootstrap intervals use resampling seed 20260823 with 10,000 resamples; detection-rate
  intervals resample episodes rather than burns.
