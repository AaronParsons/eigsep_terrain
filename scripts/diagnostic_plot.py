#!/usr/bin/env python
"""
Plotting script for MCMC runs produced by eigsep_terrain_pymc.py.

Loads a .nc trace and its _meta.json sidecar, then produces any
combination of diagnostic plots controlled by boolean flags.

Usage
-----
  python plot_trace.py trace_seed042.nc [options]

All plots are saved as <stem>_<plot>.png unless --show is given.

Available plots (each a boolean flag, all off by default)
----------------------------------------------------------
  --trace       Chain timeseries + marginal KDE (az.plot_trace)
  --rank        Rank plots for multi-chain mixing (az.plot_rank)
  --autocorr    Autocorrelation per param (az.plot_autocorr)
  --posterior   Marginal posteriors with prior overlay
  --shrinkage   Bar chart: 1 - post_std / prior_sigma per param
  --pair        Bivariate scatter/KDE for a param group
  --acceptance  Rolling acceptance rate over the chain
  --logp        Log-likelihood (lik potential) timeseries
  --prior-predictive   Prior predictive vs posterior histogram per param
  --step-size          Effective step size vs post_std and prior_sigma
  --prior-sensitivity  Z-score and contraction ratio per param
  --all         Enable every plot above
"""
import argparse
import json
import os
import re

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── figure style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.constrained_layout.use": True,
})

PARAM_LABELS = {
    "e": "E [m]", "n": "N [m]", "u": "U [m]",
    "log_h": "log h", "th": "θ [rad]", "ph": "φ [rad]",
    "ti": "tilt [rad]", "f": "f [px]",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(nc_path):
    """Load trace and sidecar metadata. Returns (trace, meta, stem)."""
    stem = re.sub(r"\.nc$", "", nc_path)
    metafile = f"{stem}_meta.json"
    trace = az.from_netcdf(nc_path)
    if os.path.exists(metafile):
        with open(metafile) as f:
            meta = json.load(f)
    else:
        print(f"WARNING: {metafile} not found — prior overlay and shrinkage unavailable.")
        meta = {}
    return trace, meta, stem


def _suptitle(fig, meta, extra=""):
    """Attach a compact run-summary as the figure suptitle."""
    s = meta.get("sampling", {})
    st = meta.get("step", {})
    lk = meta.get("likelihood", {})
    parts = [
        f"seed={meta.get('seed', '?')}",
        f"draws={s.get('draws', '?')}",
        f"tune={s.get('tune', '?')}",
        f"chains={s.get('chains', '?')}",
        f"scaling={st.get('scaling', '?')}",
        f"eps={lk.get('eps', '?')}",
        f"n_rays={lk.get('n_rays', '?')}",
        f"accept={meta.get('accepted_mean', float('nan')):.3f}",
    ]
    if extra:
        parts.append(extra)
    fig.suptitle("  |  ".join(parts), fontsize=7, y=1.01)


def _param_label(name):
    """Turn raw param name (e.g. '0817_log_h') into a readable label."""
    parts = name.split("_", 1)
    if len(parts) == 2:
        key_part = parts[1]
        suffix = PARAM_LABELS.get(key_part, key_part)
        return f"{parts[0]} {suffix}"
    return PARAM_LABELS.get(name, name)


def _save(fig, stem, tag):
    path = f"{stem}_{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved: {path}")
    plt.close(fig)


def _posterior_array(trace, name):
    """Return flattened posterior samples for a param name."""
    return trace.posterior[name].values.flatten()


# ── individual plot functions ─────────────────────────────────────────────────

def plot_trace(trace, meta, stem):
    """Chain timeseries + marginal KDE via ArviZ."""
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    axes = az.plot_trace(trace, var_names=param_names, compact=False,
                         figsize=(12, max(3, len(param_names) * 1.2)))
    fig = axes.ravel()[0].get_figure()
    # relabel y-axes with human-readable names
    for row, name in zip(axes, param_names):
        for ax in row:
            ax.set_title(_param_label(name), fontsize=8)
    _suptitle(fig, meta)
    _save(fig, stem, "trace")


def plot_rank(trace, meta, stem):
    """Rank plots — better than trace for diagnosing multi-chain mixing."""
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    axes = az.plot_rank(trace, var_names=param_names,
                        figsize=(10, max(3, len(param_names) * 0.9)))
    fig = axes.ravel()[0].get_figure()
    _suptitle(fig, meta)
    _save(fig, stem, "rank")


def plot_autocorr(trace, meta, stem):
    """Autocorrelation per param — slow decay = poor ESS."""
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    axes = az.plot_autocorr(trace, var_names=param_names, max_lag=200,
                            figsize=(12, max(3, len(param_names) * 1.0)))
    fig = axes.ravel()[0].get_figure()
    _suptitle(fig, meta)
    _save(fig, stem, "autocorr")


def plot_posterior(trace, meta, stem):
    """
    Marginal posterior KDE per param with prior Normal overlay.
    Prior sigma comes from meta['param_summary'][name]['prior_sigma'].
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    param_summary = meta.get("param_summary", {})

    ncols = 4
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.4))
    axes = np.array(axes).flatten()

    for ax, name in zip(axes, param_names):
        samples = _posterior_array(trace, name)
        ax.hist(samples, bins=50, density=True, color="steelblue",
                alpha=0.6, label="posterior")

        # Prior overlay if available
        if name in param_summary:
            ps = param_summary[name]
            mu = ps["mean"]   # use posterior mean as proxy for prior centre
            # better: use the initial param value — but we don't store it in meta yet,
            # so fall back to posterior mean (conservative; shows width comparison)
            prior_mu = mu
            prior_sigma = ps["prior_sigma"]
            x = np.linspace(samples.min(), samples.max(), 300)
            prior_pdf = (1 / (prior_sigma * np.sqrt(2 * np.pi)) *
                         np.exp(-0.5 * ((x - prior_mu) / prior_sigma) ** 2))
            ax.plot(x, prior_pdf, "r--", lw=1.2, label=f"prior σ={prior_sigma:.3g}")

        ax.set_title(_param_label(name), fontsize=8)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.NullFormatter())
        ax.legend(fontsize=7)

    for ax in axes[len(param_names):]:
        ax.set_visible(False)

    _suptitle(fig, meta)
    _save(fig, stem, "posterior")


def plot_shrinkage(trace, meta, stem):
    """
    Shrinkage = 1 - posterior_std / prior_sigma per param.
    Near 1 → data-dominated. Near 0 → prior not updated (weak likelihood or prior too tight).
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    param_summary = meta.get("param_summary", {})
    if not param_summary:
        print("  shrinkage: no param_summary in metadata, skipping.")
        return

    shrinkages = []
    labels = []
    for name in param_names:
        if name not in param_summary:
            continue
        post_std = _posterior_array(trace, name).std()
        prior_sigma = param_summary[name]["prior_sigma"]
        shrinkages.append(1.0 - post_std / prior_sigma)
        labels.append(_param_label(name))

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.55), 4))
    colors = ["#d9534f" if s < 0.1 else "#5cb85c" if s > 0.5 else "#f0ad4e"
              for s in shrinkages]
    ax.bar(range(len(labels)), shrinkages, color=colors, width=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", label="0.5 reference")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("shrinkage  (1 − post_std / prior_σ)")
    ax.set_ylim(-0.2, 1.1)
    ax.legend(fontsize=8)
    _suptitle(fig, meta)
    _save(fig, stem, "shrinkage")


def plot_pair(trace, meta, stem, group="position"):
    """
    Bivariate scatter/KDE for a param subset.
    group: 'position'  → E, N, log_h params for each camera + antenna
           'angles'    → th, ph, ti params for each camera
           'all'       → all params (can be slow for many params)
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)

    if group == "position":
        selected = [n for n in param_names
                    if any(n.endswith(s) for s in ("_e", "_n", "_log_h"))
                    or n in ("ant_e", "ant_n", "ant_log_h")]
    elif group == "angles":
        selected = [n for n in param_names
                    if any(n.endswith(s) for s in ("_th", "_ph", "_ti"))]
    else:
        selected = param_names

    if len(selected) < 2:
        print(f"  pair ({group}): fewer than 2 params matched, skipping.")
        return

    axes = az.plot_pair(trace, var_names=selected, kind="scatter",
                        marginals=True, divergences=False,
                        figsize=(max(6, len(selected) * 1.8),
                                 max(6, len(selected) * 1.8)))
    fig = axes.ravel()[0].get_figure()
    # relabel axes
    for ax in axes.flatten():
        xl = ax.get_xlabel()
        yl = ax.get_ylabel()
        if xl:
            ax.set_xlabel(_param_label(xl), fontsize=8)
        if yl:
            ax.set_ylabel(_param_label(yl), fontsize=8)
    _suptitle(fig, meta, extra=f"pair={group}")
    _save(fig, stem, f"pair_{group}")


def plot_acceptance(trace, meta, stem, window=100):
    """Rolling acceptance rate over the chain draw index."""
    # sample_stats.accepted is bool (chain, draw)
    accepted = trace.sample_stats.accepted.values  # (chains, draws)
    n_chains, n_draws = accepted.shape

    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(n_draws)
    for c in range(n_chains):
        roll = np.convolve(accepted[c].astype(float),
                           np.ones(window) / window, mode="valid")
        ax.plot(np.arange(len(roll)) + window // 2, roll,
                lw=1.0, label=f"chain {c}")

    overall = float(accepted.mean())
    ax.axhline(overall, color="k", ls="--", lw=0.8,
               label=f"overall {overall:.3f}")
    ax.set_xlabel(f"draw  (rolling window={window})")
    ax.set_ylabel("acceptance rate")
    ax.set_ylim(0, 1)
    ax.legend()
    _suptitle(fig, meta)
    _save(fig, stem, "acceptance")


def plot_logp(trace, meta, stem):
    """
    Log-likelihood (lik potential) timeseries per chain.
    Uses sample_stats; falls back to lp if lik_potential not present.
    """
    ss = trace.sample_stats
    # ArviZ stores the Potential contribution differently depending on PyMC version.
    # Try common attribute names in order.
    for attr in ("lik", "lp", "log_likelihood"):
        if hasattr(ss, attr):
            logp_da = getattr(ss, attr)
            label = attr
            break
    else:
        print("  logp: no recognised logp field in sample_stats, skipping.")
        return

    logp = logp_da.values  # (chains, draws) or (chains, draws, ...)
    if logp.ndim > 2:
        logp = logp.squeeze()

    n_chains = logp.shape[0]
    fig, ax = plt.subplots(figsize=(10, 3))
    for c in range(n_chains):
        ax.plot(logp[c], lw=0.6, alpha=0.8, label=f"chain {c}")
    ax.set_xlabel("draw")
    ax.set_ylabel(f"logL  ({label})")
    ax.legend()
    _suptitle(fig, meta)
    _save(fig, stem, "logp")


def plot_prior_predictive(trace, meta, stem, n_samples=2000):
    """
    Prior predictive distribution vs posterior for every param.

    Draws n_samples from Normal(prior_mu, prior_sigma) for each param and
    overlays the resulting prior predictive histogram against the posterior
    KDE.  This makes it visually clear which params are data-dominated
    (prior and posterior differ markedly) vs prior-dominated (they overlap).

    Requires prior_mu in param_summary (stored by eigsep_terrain_pymc.py).
    Falls back to the posterior mean as prior centre if absent.
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    param_summary = meta.get("param_summary", {})
    if not param_summary:
        print("  prior_predictive: no param_summary in metadata, skipping.")
        return

    rng = np.random.default_rng(meta.get("seed", 0))

    ncols = 4
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.6))
    axes = np.array(axes).flatten()

    for ax, name in zip(axes, param_names):
        post = _posterior_array(trace, name)
        ps = param_summary.get(name, {})
        prior_mu    = ps.get("prior_mu", float(post.mean()))
        prior_sigma = ps.get("prior_sigma", None)

        if prior_sigma is None:
            ax.set_title(_param_label(name) + "\n(no prior info)", fontsize=8)
            ax.hist(post, bins=40, density=True, color="steelblue", alpha=0.7)
            continue

        prior_samples = rng.normal(prior_mu, prior_sigma, size=n_samples)

        # shared x range covering both distributions
        lo = min(post.min(), prior_samples.min())
        hi = max(post.max(), prior_samples.max())
        bins = np.linspace(lo, hi, 50)

        ax.hist(prior_samples, bins=bins, density=True, color="#e07b54",
                alpha=0.45, label="prior")
        ax.hist(post, bins=bins, density=True, color="steelblue",
                alpha=0.65, label="posterior")
        ax.axvline(prior_mu, color="#c0392b", lw=1.0, ls="--", label="prior μ")
        ax.axvline(float(post.mean()), color="#1a5276", lw=1.0, ls="--",
                   label="post μ")

        ax.set_title(_param_label(name), fontsize=8)
        ax.yaxis.set_major_formatter(mticker.NullFormatter())
        ax.legend(fontsize=6, ncol=2)

    for ax in axes[len(param_names):]:
        ax.set_visible(False)

    _suptitle(fig, meta)
    _save(fig, stem, "prior_predictive")


def plot_step_size(trace, meta, stem):
    """
    Proposal step size diagnostics — three panels per param group:

      1. Effective step size vs posterior std
         Bar chart of  step / post_std  per param.
         Well-tuned MH target ≈ 0.3–0.6 (grey band).
         Too small → slow diffusion.  Too large → high rejection.

      2. Absolute effective step size vs prior sigma
         Shows how the proposal compares to the prior width.
         step ≫ prior_sigma means proposals jump outside the prior bulk.

      3. Tuned scaling value (single number from metadata, shown as text
         annotation on panel 1 for reference).

    Requires effective_step and prior_sigma in param_summary.
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    param_summary = meta.get("param_summary", {})
    if not param_summary:
        print("  step_size: no param_summary in metadata, skipping.")
        return

    names, labels, step_over_post, step_over_prior = [], [], [], []
    for name in param_names:
        ps = param_summary.get(name, {})
        eff_step    = ps.get("effective_step")
        prior_sigma = ps.get("prior_sigma")
        post_std    = _posterior_array(trace, name).std()
        if eff_step is None or prior_sigma is None:
            continue
        names.append(name)
        labels.append(_param_label(name))
        step_over_post.append(eff_step / post_std  if post_std  > 0 else np.nan)
        step_over_prior.append(eff_step / prior_sigma if prior_sigma > 0 else np.nan)

    if not names:
        print("  step_size: effective_step missing from all params, skipping.")
        return

    x = np.arange(len(names))
    tuned_scaling = meta.get("step", {}).get("tuned_scaling",
                    meta.get("step", {}).get("scaling", "?"))

    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(max(7, len(names) * 0.55), 7),
                                   sharex=True)

    # Panel 1: step / post_std
    colors1 = []
    for v in step_over_post:
        if np.isnan(v):
            colors1.append("gray")
        elif v < 0.1:
            colors1.append("#d9534f")   # too small — red
        elif v > 1.5:
            colors1.append("#e67e22")   # too large — orange
        else:
            colors1.append("#5cb85c")   # roughly in range — green
    ax1.bar(x, step_over_post, color=colors1, width=0.7)
    ax1.axhspan(0.3, 0.6, color="gray", alpha=0.15, label="target range 0.3–0.6")
    ax1.axhline(0.3, color="gray", lw=0.8, ls="--")
    ax1.axhline(0.6, color="gray", lw=0.8, ls="--")
    ax1.set_ylabel("step / post_std")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8)
    ax1.text(0.99, 0.97, f"tuned scaling = {tuned_scaling}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8,
             color="dimgray")

    # Panel 2: step / prior_sigma
    colors2 = ["#d9534f" if v > 0.5 else "#5cb85c" if v < 0.15 else "#5bc0de"
               for v in step_over_prior]
    ax2.bar(x, step_over_prior, color=colors2, width=0.7)
    ax2.axhline(0.15, color="gray", lw=0.8, ls="--", label="0.15 reference")
    ax2.set_ylabel("step / prior_σ")
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.align_ylabels([ax1, ax2])
    _suptitle(fig, meta)
    _save(fig, stem, "step_size")


def plot_prior_sensitivity(trace, meta, stem):
    """
    Prior sensitivity summary — two panels:

      1. Z-score of posterior mean relative to prior:
            z = (post_mean - prior_mu) / prior_sigma
         Large |z| means the posterior has been pulled far from the prior
         centre, indicating that the likelihood is informative on that param.
         |z| < 0.5 (grey band) suggests the prior is dominating or the
         likelihood has little information there.

      2. Posterior contraction ratio:
            ratio = post_std / prior_sigma
         Near 1 → no contraction (prior = posterior, likelihood uninformative).
         Near 0 → strong contraction (data-dominated).
         Plotted on a log scale so both extremes are visible.

    Together these two panels identify four regimes:
      high |z|, low ratio  → data-dominated, well-identified param
      low  |z|, low ratio  → data tightens posterior near prior (consistent)
      high |z|, high ratio → data shifts but doesn't tighten (weak signal)
      low  |z|, high ratio → prior dominates entirely
    """
    param_names = meta.get("param_names") or list(trace.posterior.data_vars)
    param_summary = meta.get("param_summary", {})
    if not param_summary:
        print("  prior_sensitivity: no param_summary in metadata, skipping.")
        return

    names, labels, zscores, ratios = [], [], [], []
    for name in param_names:
        ps = param_summary.get(name, {})
        prior_mu    = ps.get("prior_mu")
        prior_sigma = ps.get("prior_sigma")
        post_mean   = ps.get("mean")
        post_std    = _posterior_array(trace, name).std()
        if prior_mu is None or prior_sigma is None:
            continue
        names.append(name)
        labels.append(_param_label(name))
        zscores.append((post_mean - prior_mu) / prior_sigma)
        ratios.append(post_std / prior_sigma if prior_sigma > 0 else np.nan)

    if not names:
        print("  prior_sensitivity: prior_mu missing from all params, skipping.")
        return

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(max(7, len(names) * 0.55), 7),
                                   sharex=True)

    # Panel 1: z-score
    colors1 = ["#d9534f" if abs(z) > 2 else "#f0ad4e" if abs(z) > 0.5 else "#5cb85c"
               for z in zscores]
    ax1.bar(x, zscores, color=colors1, width=0.7)
    ax1.axhspan(-0.5, 0.5, color="gray", alpha=0.12, label="|z| < 0.5 (prior-dominated)")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.axhline( 2, color="#c0392b", lw=0.8, ls="--", alpha=0.6)
    ax1.axhline(-2, color="#c0392b", lw=0.8, ls="--", alpha=0.6)
    ax1.set_ylabel("z-score  (post_mean − prior_μ) / prior_σ")
    ax1.legend(fontsize=8)

    # Panel 2: contraction ratio (log scale)
    colors2 = ["#5cb85c" if r < 0.5 else "#f0ad4e" if r < 0.85 else "#d9534f"
               for r in ratios]
    ax2.bar(x, ratios, color=colors2, width=0.7)
    ax2.axhline(1.0, color="k", lw=0.8, ls="--", label="ratio = 1 (no contraction)")
    ax2.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6, label="ratio = 0.5")
    ax2.set_ylabel("contraction  post_std / prior_σ")
    ax2.set_ylim(0, max(1.2, max(r for r in ratios if not np.isnan(r)) * 1.1))
    ax2.legend(fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.align_ylabels([ax1, ax2])
    _suptitle(fig, meta)
    _save(fig, stem, "prior_sensitivity")


# ── main ──────────────────────────────────────────────────────────────────────

def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("nc_file", help="Path to the ArviZ .nc trace file")

    # plot toggles
    ap.add_argument("--trace",      action="store_true", help="Chain timeseries + marginal KDE")
    ap.add_argument("--rank",       action="store_true", help="Rank plots (multi-chain mixing)")
    ap.add_argument("--autocorr",   action="store_true", help="Autocorrelation per param")
    ap.add_argument("--posterior",  action="store_true", help="Marginals with prior overlay")
    ap.add_argument("--shrinkage",  action="store_true", help="Shrinkage bar chart")
    ap.add_argument("--pair",       action="store_true", help="Bivariate scatter (position group)")
    ap.add_argument("--pair-angles",action="store_true", help="Bivariate scatter (angles group)")
    ap.add_argument("--pair-all",   action="store_true", help="Bivariate scatter (all params)")
    ap.add_argument("--acceptance", action="store_true", help="Rolling acceptance rate")
    ap.add_argument("--logp",       action="store_true", help="Log-likelihood timeseries")
    ap.add_argument("--prior-predictive",  action="store_true",
                    help="Prior predictive vs posterior histogram per param")
    ap.add_argument("--step-size",         action="store_true",
                    help="Effective step size vs post_std and prior_sigma")
    ap.add_argument("--prior-sensitivity", action="store_true",
                    help="Z-score and contraction ratio per param")
    ap.add_argument("--all",        action="store_true", help="Enable every plot")

    # options
    ap.add_argument("--show", action="store_true",
                    help="Display figures interactively instead of saving")
    ap.add_argument("--window", type=int, default=100,
                    help="Rolling window size for acceptance plot (default: 100)")

    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    trace, meta, stem = _load(args.nc_file)

    do_all = args.all
    show   = args.show

    print(f"Loaded: {args.nc_file}")
    if meta:
        print(f"  seed={meta.get('seed')}  "
              f"chains={meta.get('sampling',{}).get('chains')}  "
              f"draws={meta.get('sampling',{}).get('draws')}  "
              f"accept={meta.get('accepted_mean', float('nan')):.3f}")

    plots_run = 0

    if do_all or args.trace:
        print("Plotting: trace")
        plot_trace(trace, meta, stem)
        plots_run += 1

    if do_all or args.rank:
        print("Plotting: rank")
        plot_rank(trace, meta, stem)
        plots_run += 1

    if do_all or args.autocorr:
        print("Plotting: autocorr")
        plot_autocorr(trace, meta, stem)
        plots_run += 1

    if do_all or args.posterior:
        print("Plotting: posterior")
        plot_posterior(trace, meta, stem)
        plots_run += 1

    if do_all or args.shrinkage:
        print("Plotting: shrinkage")
        plot_shrinkage(trace, meta, stem)
        plots_run += 1

    if do_all or args.pair:
        print("Plotting: pair (position)")
        plot_pair(trace, meta, stem, group="position")
        plots_run += 1

    if do_all or args.pair_angles:
        print("Plotting: pair (angles)")
        plot_pair(trace, meta, stem, group="angles")
        plots_run += 1

    if args.pair_all:
        print("Plotting: pair (all) — may be slow")
        plot_pair(trace, meta, stem, group="all")
        plots_run += 1

    if do_all or args.acceptance:
        print("Plotting: acceptance")
        plot_acceptance(trace, meta, stem, window=args.window)
        plots_run += 1

    if do_all or args.logp:
        print("Plotting: logp")
        plot_logp(trace, meta, stem)
        plots_run += 1

    if do_all or args.prior_predictive:
        print("Plotting: prior_predictive")
        plot_prior_predictive(trace, meta, stem)
        plots_run += 1

    if do_all or args.step_size:
        print("Plotting: step_size")
        plot_step_size(trace, meta, stem)
        plots_run += 1

    if do_all or args.prior_sensitivity:
        print("Plotting: prior_sensitivity")
        plot_prior_sensitivity(trace, meta, stem)
        plots_run += 1

    if plots_run == 0:
        print("No plots selected. Pass --all or one or more plot flags. See --help.")
        return 1

    if show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())