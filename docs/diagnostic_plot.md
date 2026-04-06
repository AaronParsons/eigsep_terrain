## description of diagnostic plots
### Trace plot

two panels per parameter: the left panel is the chain timeseries (the raw sequence of sampled values draw by draw) and the right is the marginal KDE. The timeseries should look like a fuzzy caterpillar with no trend — if it drifts, sticks in one place for long stretches, or shows sudden jumps to a different value range, the chain hasn't mixed. The KDE should be smooth and unimodal if the posterior is well-behaved.

### Rank plot

a more rigorous version of the trace for multi-chain runs. For each draw, each chain's sample is ranked across all chains combined, then those ranks are histogrammed. If all chains are sampling the same distribution, the rank histogram should be flat (uniform). Chains that systematically get low or high ranks are stuck in different parts of the posterior, which is the clearest sign of non-mixing.

### Autocorrelation

measures how correlated a chain's sample at draw t is with its sample at draw t+k (lag k). If the chain mixes well, this drops to zero within a few lags. Slow decay (still correlated at lag 50, 100, 200) means the chain is moving slowly through the posterior and you're getting far fewer independent samples than the raw draw count suggests. The effective sample size (ESS) is directly controlled by this — ESS ≈ N / (1 + 2 * sum of autocorrelations).

### Posterior marginals

histogram of posterior samples per parameter with the prior Normal overlaid as a dashed red curve. The key question is how much the posterior (blue) differs from the prior (red). If they look almost identical, the data isn't constraining that parameter — either the prior is too tight or the likelihood is insensitive to it. If the posterior is much narrower and shifted from the prior, the data is informative.

### Shrinkage

a single number per parameter: 1 - posterior_std / prior_sigma. Green bars (>0.5) mean the data has meaningfully tightened the prior — the posterior is at least half as wide. Orange (0.1–0.5) means partial information. Red (<0.1) means the posterior is almost as wide as the prior — the data is not constraining that parameter at all. This is a fast summary of which parameters your model actually learns from the observations.

### Pair plot (position / angles)

bivariate scatter of posterior samples for pairs of parameters. The shape of the cloud tells you about correlations. A diagonal elongated cloud between two parameters means they're correlated — moving one requires moving the other to maintain similar logL. This is exactly the structure that makes MH mixing slow, because a proposal needs to move along the ridge rather than across it. For your model, expect to see correlations between camera E/N and θ/φ, and between log_h and f.

### Rolling acceptance rate

the fraction of proposals accepted in a sliding window across draws. Should be roughly flat after tuning and sit in the 0.2–0.4 range. A declining trend means the chain is getting stuck as it moves into lower-density regions. A very spiky or erratic acceptance rate suggests the likelihood surface is rough.

### Sampler stats (scaling & lambda)

both are internal DEMetropolisZ quantities. scaling is the adaptive step-size multiplier — it should decrease and stabilise during tuning, then stay flat. If it's still changing during the sampling draws you need more tuning. lambda is the magnitude of the differential evolution jump vector (the distance between two randomly chosen past chain states used to construct the proposal direction) — it reflects how spread out the chain history is. If lambda collapses to near zero, the chain has degenerated into a tight cluster and proposals become tiny.

### Prior predictive vs posterior

for each parameter, draws n_samples from the prior Normal and shows the distribution of those prior draws alongside the posterior. Similar to the posterior marginals plot but emphasises the full prior distribution rather than just its width, making it easier to see when the prior is placing probability mass in regions the data rules out.

### Step size vs posterior & prior widths

for each parameter, compares three widths side by side: the effective step size (scaling * sigma), the posterior std, and the prior sigma. The step size should be smaller than the posterior std (otherwise proposals jump clean across the posterior) but not orders of magnitude smaller (otherwise the chain barely moves). This plot makes it immediately obvious if any parameter has badly mismatched step size.

### Prior sensitivity (z-score & contraction)

two metrics per parameter. The z-score is (posterior_mean - prior_mean) / prior_sigma — how many prior sigmas has the posterior mean shifted from where you started? Large z means the data strongly pulled the posterior away from the prior. The contraction ratio is 1 - posterior_var / prior_var (similar to shrinkage but in variance units). Together they tell you whether parameters are being moved by the data (high z, high contraction) or just sitting where the prior put them (low z, low contraction).