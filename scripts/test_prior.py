"""
Prior-only sanity check: sample from the PyMC priors defined in
PositionSolver.get_mcmc_prms(), convert log_h -> u, and compare
against dem.interp_alt to verify physical plausibility.
"""
import argparse
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import glob
import os

# ── project imports ──────────────────────────────────────────────────────────
from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r
from eigsep_terrain_pymc import (
    DEFAULT_META, DEFAULT_PRMS, _apply_prms_to_dem_and_meta, BOX_SIZE
)

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--cache-file",       default="marjum_dem.npz")
ap.add_argument("--img-glob",         default="/Users/komalkaur/Desktop/eigsep_stuff/hrzn_mapping/imgs/IMG*.jpg")
ap.add_argument("--seed",             type=int,   default=None)
ap.add_argument("--draws",            type=int,   default=2000)
ap.add_argument("--pos-err",          type=float, default=30.0)
ap.add_argument("--ang-err",          type=float, default=np.deg2rad(5.0))
ap.add_argument("--log-h-sigma",      type=float, default=1.0)
ap.add_argument("--f-err",            type=float, default=0.1)
ap.add_argument("--jitter-scaling",   type=float, default=1.0)
args = ap.parse_args()

# ── seed + output names ───────────────────────────────────────────────────────
SEED    = args.seed if args.seed is not None else int(np.random.randint(1000))
OUT_DIR = f"test_prior_seed{SEED:03d}"
os.makedirs(OUT_DIR, exist_ok=True)
PLOT_OUT  = os.path.join(OUT_DIR, "prior_check.png")
TRACE_OUT = os.path.join(OUT_DIR, "prior_trace.png")
CORNER_OUT= os.path.join(OUT_DIR, "prior_corner.png")
TXT_OUT   = os.path.join(OUT_DIR, "prior_check.txt")
HEIGHT_OUT = os.path.join(OUT_DIR, "height_trace.png")

# ── config ────────────────────────────────────────────────────────────────────
CACHE  = args.cache_file
IMGS   = args.img_glob
N_RAYS = 100   # unused for prior-only, just satisfies PositionSolver
DRAWS  = args.draws

# ── setup ─────────────────────────────────────────────────────────────────────
dem  = DEM(cache_file=CACHE)
meta = {k: dict(v) for k, v in DEFAULT_META.items()}
files = sorted(glob.glob(IMGS))
imgs  = [HorizonImage(f, meta) for f in files]
imgs  = [img for img in imgs if img.key in meta]

fit_imgs, static_imgs = imgs, []
img_keys = [img.key for img in fit_imgs]

prms_u = np.asarray(DEFAULT_PRMS, dtype=dtype_r)
_apply_prms_to_dem_and_meta(dem, meta, img_keys, prms_u, len(PRM_ORDER))

ps = PositionSolver(dem["platform"], fit_imgs, static_imgs, N_RAYS, dem,
                    box_size=BOX_SIZE)
prms_h = ps.prms_u_to_h(prms_u)
ps.set_mcmc_prms(prms_h)
ps.set_mcmc_sigmas(pos_err=args.pos_err, ang_err=args.ang_err,
                   f_err=args.f_err, log_h_sigma=args.log_h_sigma)

# ── prior-only model ──────────────────────────────────────────────────────────
with pm.Model() as prior_model:
    mcmc_prms = ps.get_mcmc_prms()
    prior_trace = pm.sample_prior_predictive(samples=DRAWS, random_seed=SEED)

prior = prior_trace.prior

# ── helper ────────────────────────────────────────────────────────────────────
def compare_logh_u(e_arr, n_arr, logh_arr, label, log_fh):
    h     = np.exp(logh_arr)
    u_dem = np.array([float(dem.interp_alt(e, n)) for e, n in zip(e_arr, n_arr)])
    u_abs = h + u_dem
    lines = [
        f"--- {label} ---",
        f"  dem u (ground)       mean={u_dem.mean():.1f}  [{u_dem.min():.1f}, {u_dem.max():.1f}]",
        f"  h above ground       mean={h.mean():.2f}  [{h.min():.2f}, {h.max():.2f}]",
        f"  u (ground + h)       mean={u_abs.mean():.1f}  [{u_abs.min():.1f}, {u_abs.max():.1f}]",
    ]
    for line in lines:
        print(line)
        log_fh.write(line + "\n")
    return h, u_dem, u_abs

# ── print + save text summary ─────────────────────────────────────────────────
with open(TXT_OUT, "w") as log_fh:
    header = "\n".join([
        f"\n=== Prior-only check  [seed={SEED}] ===",
        f"  draws={DRAWS}  pos_err={args.pos_err}  ang_err={args.ang_err:.4f}",
        f"  log_h_sigma={args.log_h_sigma}  f_err={args.f_err}  jitter_scaling={args.jitter_scaling}",
        "",
    ])
    print(header); log_fh.write(header + "\n")

    # histogram for each image
    fig, axes = plt.subplots(len(fit_imgs) + 1, 2,
                             figsize=(10, 3 * (len(fit_imgs) + 1)))

    # height for each step in trace for each image
    h_fig, h_axes = plt.subplots(len(fit_imgs), 2,
                             figsize=(10, 3 * (len(fit_imgs) + 1)))
    
    for row, img in enumerate(fit_imgs):
        k      = img.key
        e_s    = prior[f"{k}_e"].values.flatten()
        n_s    = prior[f"{k}_n"].values.flatten()
        lh_s   = prior[f"{k}_log_h"].values.flatten()

        h_s, u0_s, u_s = compare_logh_u(e_s, n_s, lh_s, f"img {img.filename}", log_fh)

        init_h = img.prms['u'] - float(dem.interp_alt(img.prms['e'], img.prms['n']))
        rng = np.random.default_rng(SEED)
        jitter_h = rng.normal(0.0, args.log_h_sigma * args.jitter_scaling)
        jittered_h = np.exp(np.log(max(init_h, 1e-3)) + jitter_h)

        # histograms 

        axes[row, 0].hist(h_s, bins=50)
        axes[row, 0].axvline(init_h, color='r', label=f'init h={init_h:.2f}')
        axes[row, 0].axvline(jittered_h, color='orange', linestyle='--',
                             label=f'jittered h={jittered_h:.2f}')
        axes[row, 0].set_title(f"Cam {k}: h above ground [m]")
        axes[row, 0].set_xlabel('height above DEM height [m]')
        axes[row, 0].set_ylabel('counts')
        axes[row, 0].legend()

        axes[row, 1].hist(u_s, bins=50)
        axes[row, 1].axvline(img.prms['u'], color='r', label=f"init u={img.prms['u']:.1f}")
        axes[row, 1].set_title(f"Cam {k}: absolute u [m]")
        axes[row, 1].set_xlabel('DEM height [m]')
        axes[row, 1].set_ylabel('counts')
        axes[row, 1].legend()

        # height plots
        h_axes[row, 0].plot(u0_s, label='DEM U', alpha=0.3)
        h_axes[row, 0].plot(u_s, label='Image U', alpha=0.3)
        h_axes[row, 0].set_title(f'Cam {k}: trace vs U')
        h_axes[row, 0].set_xlabel('Step')
        h_axes[row, 0].set_ylabel('up coord [m]')
        h_axes[row, 0].legend()

        h_axes[row, 1].plot(h_s)
        h_axes[row, 1].set_title(f'Cam {k}: trace vs height')
        h_axes[row, 1].set_xlabel('Step')
        h_axes[row, 1].set_ylabel('height above DEM ground [m]')

        h_fig.suptitle('Trace vs DEM U and height for each image')
        h_fig.savefig(HEIGHT_OUT, dpi=100)
        print(f"Trace vs height plot saved to {HEIGHT_OUT}")

    # antenna
    ae_s  = prior["ant_e"].values.flatten()
    an_s  = prior["ant_n"].values.flatten()
    alh_s = prior["ant_log_h"].values.flatten()

    ah_s, ant_u0_s, au_s = compare_logh_u(ae_s, an_s, alh_s, "antenna", log_fh)

    ant_e0, ant_n0, ant_u_init = ps.ant_pos_prior
    ant_h_init = ant_u_init - float(dem.interp_alt(ant_e0, ant_n0))

    axes[-1, 0].hist(ah_s, bins=50)
    axes[-1, 0].axvline(ant_h_init, color='r', label=f'init h={ant_h_init:.2f}')
    axes[-1, 0].set_title("Antenna: h above ground [m]")
    axes[-1, 0].set_xlabel('height above DEM height [m]')
    axes[-1, 0].set_ylabel('counts')
    axes[-1, 0].legend()

    axes[-1, 1].hist(au_s, bins=50)
    axes[-1, 1].axvline(ant_u_init, color='r', label=f'init u={ant_u_init:.1f}')
    axes[-1, 1].set_title("Antenna: absolute u [m]")
    axes[-1, 1].set_xlabel('DEM height [m]')
    axes[-1, 1].set_ylabel('counts')
    axes[-1, 1].legend()

fig.tight_layout()
fig.savefig(PLOT_OUT, dpi=120)
print(f"\nPlot saved to {PLOT_OUT}")
print(f"Text summary saved to {TXT_OUT}")

# ── trace plot (should look like white noise) ─────────────────────────────────
var_names = [f"{img.key}_{k}" for img in fit_imgs for k in PRM_ORDER
             if k != 'u'] + \
            [f"{img.key}_log_h" for img in fit_imgs] + \
            ["ant_e", "ant_n", "ant_log_h"]

# sample_prior_predictive gives a single "chain"; reshape to fake chain/draw dims
# so az.plot_trace works as expected
axes_trace = az.plot_trace(prior_trace.prior, var_names=var_names, compact=False)
fig_trace = axes_trace.ravel()[0].get_figure()
fig_trace.suptitle(f"Prior trace (should be white noise)  [seed={SEED}]", y=1.01)
fig_trace.savefig(TRACE_OUT, dpi=100, bbox_inches="tight")
print(f"Trace plot saved to {TRACE_OUT}")

# ── corner / pair plot ────────────────────────────────────────────────────────
# focus on the physically interesting params: positions + log_h for each camera + antenna
corner_vars = [f"{img.key}_{k}" for img in fit_imgs for k in ("e", "n", "log_h")] + \
              ["ant_e", "ant_n", "ant_log_h"]

axes_corner = az.plot_pair(
    prior_trace.prior,
    var_names=corner_vars,
    kind="hexbin",
    marginals=True,
    textsize=8,
)
fig_corner = axes_corner.ravel()[0].get_figure()
fig_corner.suptitle(f"Prior pair plot (expect no correlations)  [seed={SEED}]", y=1.01)
fig_corner.savefig(CORNER_OUT, dpi=100, bbox_inches="tight")
print(f"Corner plot saved to {CORNER_OUT}")