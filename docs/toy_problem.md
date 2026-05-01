# Toy Problem: Synthetic Horizon Recovery

## Purpose

`toy_problem.py` verifies the full eigsep_terrain pipeline end-to-end using
a synthetic test where the ground truth is known exactly. Because we
constructed the data ourselves, a failure to recover the true parameters is
unambiguous evidence that something is wrong in the code — not in the real
images or the DEM.

The test answers the question: *given a perfect observer at a known location
looking at the real DEM, can the pipeline recover that location from the
resulting sky mask?*

---

## How It Works

### Step 1 — Choose ground truth

You provide a true camera position (E, N, height above ground), orientation
(elevation angle θ, azimuth φ, tilt), focal length f, and antenna position.
These are the values the pipeline will try to recover.

```bash
--true-e 1760.0 --true-n 2083.0 --true-h 2.0
--true-th 1.47  --true-ph 3.69  --true-ti -0.05  --true-f 5000.0
--true-ant-e 1651.0 --true-ant-n 2025.0 --true-ant-h 100.0
```

The true absolute altitude `u` is computed as `dem.interp_alt(E, N) + h`,
so the camera sits exactly `h` metres above the terrain surface at its
horizontal position.

### Step 2 — Generate the synthetic sky mask

The `SyntheticHorizonImage` class ray-traces the real DEM from the true
camera position at 1/4 resolution (decimation=4), then upsamples to full
resolution using nearest-neighbour interpolation. This produces a binary
`model_sky_true` array: `True` where a ray misses all terrain (sky),
`False` where it hits (ground).

This binary mask is then converted to a smooth probability map `psky` that
mimics a real sky segmentation output:

```
signed_pixel_dist = dist_transform(sky) - dist_transform(ground)
logit = sharpness * signed_pixel_dist / max(H, W)  +  N(0, noise_std)
psky  = 1 / (1 + exp(-logit))
```

`signed_pixel_dist` is the Euclidean distance transform of the binary mask —
positive inside the sky region, negative inside the ground region, zero at
the horizon boundary. Dividing by `max(H, W)` normalises it to image units.

The result is a map that is close to 1.0 deep in the sky, close to 0.0 deep
in the ground, and uncertain (near 0.5) within a band around the horizon.
The width of that band is controlled by `--sharpness` and the amount of
noise by `--noise-std`.

**Why this matters:** in a real image, the segmentation model produces
exactly this kind of smooth probability map. By using the same form here,
the synthetic test exercises the full likelihood pathway including the
behaviour near the horizon boundary.

### Step 3 — Build the PositionSolver

`SyntheticHorizonImage` duck-types `HorizonImage` — it implements the same
interface (`psky`, `horizon_mask`, `horizon_dist`, `get_rays`,
`ray_distance`, `choose_pixels`, `horizon_ray_logL`, `ant_logL`) but is
backed by the synthetic data rather than a real photograph. No segmentation
model is loaded and no `.jpg` or `.npz` files are needed.

A `PositionSolver` is constructed with the synthetic image as a single
camera. The prior centre is set to the true parameters, so the prior is
centred on the answer — this means the prior does not help the optimizer
find the solution, it only regularises it.

### Step 4 — Perturb and optimise

The optimizer starts from the true parameters perturbed by a controlled
amount:

- Restart 0: position perturbed by `--perturb-pos` metres (default 5m),
  angles by `--perturb-ang` radians (default 0.05 rad), all other params
  at truth.
- Subsequent restarts: full random jitter across all parameters, scaled by
  `0.3 * restart_index` of the prior sigmas.

Powell optimisation minimises the negative log posterior:

```
-log p(θ | data) = -logL_rays(θ) - logL_ant(θ) - log_prior(θ)
```

where `log_prior` is a Gaussian centred on the true params with the same
sigmas used in the real MCMC runs.

The best result across all restarts is kept.

### Step 5 — Evaluate recovery

Recovery errors are computed in the internal `log_h` representation (the
same space the optimizer works in) and then converted to physical units for
reporting:

- **Position error** — Euclidean distance `sqrt((E_MAP - E_true)² + (N_MAP - N_true)²)` in metres.
- **Angle error** — `max(|θ_MAP - θ_true|, |φ_MAP - φ_true|)` in radians.

The test **PASS**es if both are within the tolerances set by `--pos-tol`
(default 1m) and `--ang-tol` (default 0.01 rad).

### Step 6 — Optional MCMC

With `--run-mcmc`, a short DEMetropolisZ chain is run starting from the MAP.
The test then checks **posterior coverage**: whether the true parameter
values fall within the 95% credible interval of the posterior. Coverage
below ~90% across parameters indicates the posterior is either too narrow
(prior too tight) or biased away from truth.

---

## Interpreting Results

### logL gap

```
logL gap = logL_MAP - logL_truth
```

This is almost always **negative** (MAP is slightly better than truth) even
when the recovery is correct. This is expected: the synthetic `psky` has
noise, so the optimizer can find parameter combinations that fit the noisy
data slightly better than the true noiseless parameters. A logL gap of
`-50` to `-200` is normal. A large positive gap (MAP is much worse than
truth) means the optimizer is stuck in a local mode.

### The horizon overlay plot

Panel 4 shows the synthetic `psky` with the true horizon (blue) and MAP
horizon (red) overlaid. If recovery is correct they should be
indistinguishable. Systematic offset between the two indicates a biased
solution — look at the recovery error table to identify which parameters are
wrong.

### The error bar chart

Each bar is `|MAP - truth|` in the log_h parameter space. Bars above the
`pos_tol` line are failures. Note that angular parameters (θ, φ, tilt) and
positional parameters (E, N) will have very different scales — the chart
uses raw log_h-space units, so compare each bar to the tolerance line that
applies to its type.

### MCMC coverage

A well-calibrated posterior should cover the truth ~95% of the time for each
parameter. Consistently low coverage (< 80%) for position parameters but
good coverage for angles suggests the E/N posterior is multimodal and the
chain is stuck in one mode. Coverage > 100% (always covered) is also a
warning — it means the posterior is too wide and the MCMC is not actually
constraining those parameters.

---

## Known Limitations

**The synthetic psky is cleaner than reality.** The binary sky mask from
ray-tracing has a perfectly sharp horizon. Real images have trees,
atmospheric haze, and segmentation errors that create ambiguous pixels over
a wider band. The test is therefore easier than the real problem — passing
the toy test is necessary but not sufficient to validate the full pipeline.

**The antenna pixel is approximate.** The synthetic antenna pixel is placed
at `(npix_x // 3, npix_y // 2)` regardless of where the true antenna would
project in the image. The `ant_logL` term is therefore weakly informative in
this test. This is conservative — if anything, it makes the test harder, not
easier.

**Single camera.** The real problem has three cameras whose antenna logL
terms are jointly constrained. The toy problem uses one camera, so it cannot
test cross-camera consistency.

**DEM resolution matters.** The synthetic sky mask is generated at
`decimate=4` resolution (every 4th pixel). At the default image size of
1024x1024 this means 256x256 ray-traced points upsampled to full resolution.
Fine features in the DEM horizon that fall between sampled pixels will be
missing from the synthetic mask. Use a smaller `--decimate` (hardcoded to 4
in `SyntheticHorizonImage.__init__`) for higher-fidelity synthetic data at
the cost of longer generation time.

---

## Usage Examples

```bash
# Quick smoke test (~2 min)
toy_problem.py --seed 42 --npix-y 512 --npix-x 512 \
    --n-rays 500 --n-restarts 3

# Standard test matching real run settings
toy_problem.py --seed 42 \
    --n-rays 1000 --fine-delta 0.25 --n-restarts 5 \
    --pos-err 10.0 --ang-err-deg 5.0

# With MCMC coverage check (~20 min)
toy_problem.py --seed 42 --n-rays 1000 --n-restarts 5 \
    --run-mcmc --mcmc-draws 2000 --mcmc-tune 500

# Stress test: large perturbation, tight tolerances
toy_problem.py --seed 42 \
    --perturb-pos 15.0 --perturb-ang 0.1 \
    --pos-tol 0.5 --ang-tol 0.005 \
    --n-restarts 9
```

---

## Output Files

All outputs are written to `toy_seed{NNN}/` (or `--outdir`):

| File | Contents |
|------|----------|
| `toy_results.png` | 5-panel diagnostic plot |
| `toy_results.json` | Numerical results, errors, all args |

### JSON fields

```json
{
  "seed": 42,
  "overall": "PASS",
  "pos_err_m": 0.312,
  "ang_err_rad": 0.00418,
  "pos_tol": 1.0,
  "ang_tol": 0.01,
  "logL_truth": -1823.4,
  "logL_map": -1801.2,
  "errors": {
    "syn_e":     {"true": ..., "map": ..., "err": ..., "abserr": ...},
    ...
  },
  "args": { ... }
}
```

---

## Relationship to the Real Pipeline

The toy problem exercises exactly the same code paths as the real runs:

| Component | Real run | Toy problem |
|-----------|----------|-------------|
| Ray tracing | `ray_distance_coarse_to_fine_numba` | same |
| logL | `horizon_ray_logL` + `ant_logL` | same |
| Prior | `set_mcmc_sigmas` | same |
| Optimizer | `map_estimate.py` Powell | same (inline) |
| MCMC | `eigsep_terrain_pymc.py` | same (optional) |
| Sky mask | SegFormer segmentation | Synthetic logistic |
| Images | Real `.jpg` + `.npz` | `SyntheticHorizonImage` |

The only substitution is `SyntheticHorizonImage` for `HorizonImage`. All
likelihood and prior code is unchanged.