## description of sanity check tests
### Check 1 — logL at init. Trust: high.

Makes exactly one claim: finite logL is a necessary condition for the chain to move. This is mathematically true — if the init returns -inf, every MH acceptance probability is zero and the chain is frozen. No thresholds, no assumptions about what a "good" logL looks like. The only way this check is misleading is if _safe_logL swallows a bug in total_logL and returns a finite but wrong number, which is always possible since it catches all exceptions.

### Check 2 — DEM bounds. Trust: high.

Verifies the camera's u is between 0 and 10m above the terrain surface. This matters because interp_alt clamps silently at the grid boundary, so a camera positioned below the terrain surface would return a valid-looking logL but the ray tracing geometry would be wrong. The 10m ceiling is arbitrary but any camera more than 10m above local terrain is certainly a misconfigured init.

### Check 3 — Prior predictive. Trust: moderate.

The frac_finite metric is reliable — if most prior draws return -inf, the priors are placing most of their mass in degenerate regions of parameter space and the sampler will waste time there. The rel_spread metric (prior_logL_std / |init_logL|) is noisier because both numerator and denominator are evaluated with only 500 rays (for speed), introducing pixel noise. The thresholds of 10 and 100 are heuristic, not derived. Treat this check as directional: a very high rel_spread is a genuine warning, but the exact PASS/WARN/FAIL boundary should not be taken literally.

### Check 4 — Scaling probe. Trust: low-to-moderate.

The bug where +inf proposals could be accepted is now fixed. The deeper limitation remains: this uses a diagonal Gaussian proposal while DEMetropolisZ uses differential evolution. The acceptance rate from this probe will differ from your actual run. It is useful for catching grossly wrong scaling (orders of magnitude off), but a PASS here does not mean your scaling is well-tuned, and a WARN does not necessarily mean the MCMC run will have poor acceptance. Treat it as "is scaling in the right ballpark" not "is scaling optimal."

### Check 5 — Pixel stability. Trust: high.

The metric (absolute std in logL units) is the right one — it directly quantifies the noise on the log acceptance ratio. The WARN_STD=10 threshold is defensible from first principles: exp(10) ≈ 22000 means pixel noise alone can flip a borderline acceptance decision by 4 orders of magnitude. The implementation relies on setting img._px_choice = None to force pixel redraws, which is private attribute access and would silently break if the caching logic in HorizonImage changes. This is now documented in the docstring. The check correctly reflects what actually happens during MCMC because, as established earlier, _px_choice is cached after the first call and never reset during a run — so this check is measuring the sensitivity to which pixel sample you happen to get at the start of a run, not ongoing noise during sampling.