"""
Prior-only sanity check: sample from the PyMC priors defined in
PositionSolver.get_mcmc_prms(), convert log_h -> u, and compare
against dem.interp_alt to verify physical plausibility.
"""
import argparse
import os
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from itertools import combinations
import glob

# ── project imports ──────────────────────────────────────────────────────────
from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r
from eigsep_terrain_pymc import (
    DEFAULT_META, DEFAULT_PRMS, _apply_prms_to_dem_and_meta, BOX_SIZE
)

# ── CLI ───────────────────────────────────────────────────────────────────────
# DEFAULT_PRMS = (
#     1734.11, 2069.00, 1760.97, 1.4706, 3.6932, -0.0493, 9830.11,
#     1611.31, 1849.00, 1659.78, 1.2053, 1.2414, -0.0244, 5081.08,
#     1541.90, 1998.96, 1765.06, 1.5412, 0.6147, 0.1585, 2328.64,
#     1651.83, 2024.17, 1781.46,
# )
ap = argparse.ArgumentParser()
ap.add_argument("--cache-file",       default="marjum_dem.npz")
ap.add_argument("--img-glob",         default="/Users/komalkaur/Desktop/eigsep_stuff/hrzn_mapping/imgs/IMG*.jpg")
ap.add_argument("--seed",             type=int,   default=None)
ap.add_argument("--draws",            type=int,   default=2000)
ap.add_argument("--jitter-scaling",   type=float, default=1.0)

ap.add_argument("--img0-e", type=float, default=1734.11)
ap.add_argument("--img0-n", type=float, default=2069.00)
ap.add_argument("--img1-e", type=float, default=1611.31)
ap.add_argument("--img1-n", type=float, default=1849.00)
ap.add_argument("--img2-e", type=float, default=1541.90)
ap.add_argument("--img2-n", type=float, default=1998.96)

ap.add_argument("--cam-height", type=float, default=1.6)

ap.add_argument("--pos-err",          type=float, default=30.0)
ap.add_argument("--ang-err",          type=float, default=np.deg2rad(5.0))
ap.add_argument("--log-h-sigma",      type=float, default=1.0)
ap.add_argument("--f-err",            type=float, default=0.1)

args = ap.parse_args()

# ── seed + output names ───────────────────────────────────────────────────────
SEED    = args.seed if args.seed is not None else int(np.random.randint(1000))
OUT_DIR = f"test_prior_seed{SEED:03d}"
os.makedirs(OUT_DIR, exist_ok=True)
PLOT_OUT   = os.path.join(OUT_DIR, "prior_check.png")
TRACE_OUT  = os.path.join(OUT_DIR, "prior_trace.png")
CORNER_OUT = os.path.join(OUT_DIR, "prior_corner.png")
HEIGHT_OUT = os.path.join(OUT_DIR, "height_trace.png")
TXT_OUT    = os.path.join(OUT_DIR, "prior_check.txt")

# ── shared metadata string (goes on every plot) ───────────────────────────────
META_STR = (
    f"seed={SEED}  |  draws={args.draws}  |  "
    f"cam_height={args.cam_height}m  |  "
    f"pos_err={args.pos_err:.1f}m  |  ang_err={np.rad2deg(args.ang_err):.2f}°  |  "
    f"log_h_sigma={args.log_h_sigma}  |  f_err={args.f_err}  |  "
    f"jitter_scaling={args.jitter_scaling}"
)

# ── config ────────────────────────────────────────────────────────────────────
CACHE  = args.cache_file
IMGS   = args.img_glob
N_RAYS = 100
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
# correct for weird camera heights
prms_u[2] = dem.interp_alt(args.img0_e, args.img0_n) + args.cam_height
prms_u[9] = dem.interp_alt(args.img1_e, args.img1_n) + args.cam_height
prms_u[16] = dem.interp_alt(args.img2_e, args.img2_n) + args.cam_height

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
    print(header)
    log_fh.write(header + "\n")

    # ── prior_check histogram ─────────────────────────────────────────────────
    fig, axes = plt.subplots(len(fit_imgs) + 1, 2,
                             figsize=(10, 3 * (len(fit_imgs) + 1) + 0.8))
    fig.suptitle(f"Prior check: h and u distributions\n{META_STR}",
                 fontsize=7, y=1.01)

    # ── height trace ──────────────────────────────────────────────────────────
    h_fig, h_axes = plt.subplots(len(fit_imgs), 2,
                                 figsize=(10, 3 * len(fit_imgs) + 0.8))
    h_fig.suptitle(f"Trace vs DEM U and height for each image\n{META_STR}",
                   fontsize=7, y=1.01)

    for row, img in enumerate(fit_imgs):
        k   = img.key
        e_s = prior[f"{k}_e"].values.flatten()
        n_s = prior[f"{k}_n"].values.flatten()
        lh_s= prior[f"{k}_log_h"].values.flatten()

        h_s, u0_s, u_s = compare_logh_u(e_s, n_s, lh_s, f"img {img.filename}", log_fh)

        init_h = img.prms['u'] - float(dem.interp_alt(img.prms['e'], img.prms['n']))
        rng_j  = np.random.default_rng(SEED)
        jitter_h   = rng_j.normal(0.0, args.log_h_sigma * args.jitter_scaling)
        jittered_h = np.exp(np.log(max(init_h, 1e-3)) + jitter_h)

        # histograms
        axes[row, 0].hist(h_s, bins=50)
        axes[row, 0].axvline(init_h, color='r', label=f'init h={init_h:.2f}')
        axes[row, 0].axvline(jittered_h, color='orange', linestyle='--',
                             label=f'jittered h={jittered_h:.2f}')
        axes[row, 0].set_title(f"Cam {k}: init - jittered = {(init_h-jittered_h):.2f}")
        axes[row, 0].set_xlabel('height above DEM height [m]')
        axes[row, 0].set_ylabel('counts')
        axes[row, 0].legend()

        axes[row, 1].hist(u_s, bins=50)
        axes[row, 1].axvline(img.prms['u'], color='r', label=f"init u={img.prms['u']:.1f}")
        axes[row, 1].set_title(f"Cam {k}: absolute u [m]")
        axes[row, 1].set_xlabel('DEM height [m]')
        axes[row, 1].set_ylabel('counts')
        axes[row, 1].legend()

        # height traces
        h_axes[row, 0].plot(u0_s, label='DEM U', alpha=0.3)
        h_axes[row, 0].plot(u_s,  label='Image U', alpha=0.3)
        h_axes[row, 0].set_title(f'Cam {k}: trace vs U')
        h_axes[row, 0].set_xlabel('Step')
        h_axes[row, 0].set_ylabel('up coord [m]')
        h_axes[row, 0].legend()

        h_axes[row, 1].plot(h_s)
        h_axes[row, 1].set_title(f'Cam {k}: trace vs height')
        h_axes[row, 1].set_xlabel('Step')
        h_axes[row, 1].set_ylabel('height above DEM ground [m]')

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
fig.savefig(PLOT_OUT, dpi=120, bbox_inches="tight")
print(f"Plot saved to {PLOT_OUT}")

h_fig.tight_layout()
h_fig.savefig(HEIGHT_OUT, dpi=100, bbox_inches="tight")
print(f"Height trace plot saved to {HEIGHT_OUT}")

print(f"Text summary saved to {TXT_OUT}")

# ── trace plot ────────────────────────────────────────────────────────────────
var_names = ([f"{img.key}_{k}" for img in fit_imgs for k in PRM_ORDER if k != 'u'] +
             [f"{img.key}_log_h" for img in fit_imgs] +
             ["ant_e", "ant_n", "ant_log_h"])

axes_trace = az.plot_trace(prior_trace.prior, var_names=var_names, compact=False)
fig_trace  = axes_trace.ravel()[0].get_figure()
fig_trace.suptitle(
    f"Prior trace (should be white noise)\n{META_STR}",
    fontsize=7, y=1.01,
)
fig_trace.savefig(TRACE_OUT, dpi=100, bbox_inches="tight")
print(f"Trace plot saved to {TRACE_OUT}")

# ── corner / pair plot ────────────────────────────────────────────────────────
PARAM_META = {
    "e":      ("East position",     "m"),
    "n":      ("North position",    "m"),
    "log_h":  ("log(height AGL)",   "log m"),
    "th":     ("Elevation angle θ", "rad"),
    "ph":     ("Azimuth angle φ",   "rad"),
    "ti":     ("Camera roll τ",     "rad"),
    "f":      ("Focal length",      "px"),
}

def make_label(pymc_name):
    for prefix in [f"{img.key}_" for img in fit_imgs] + ["ant_"]:
        if pymc_name.startswith(prefix):
            suffix  = pymc_name[len(prefix):]
            src_str = "antenna" if prefix == "ant_" else f"cam {prefix.rstrip('_')}"
            break
    else:
        suffix, src_str = pymc_name, ""
    label, unit = PARAM_META.get(suffix, (suffix, ""))
    return src_str, label, unit

corner_vars = (
    [f"{img.key}_{k}" for img in fit_imgs for k in ("e", "n", "log_h")] +
    ["ant_e", "ant_n", "ant_log_h"]
)

# collect init values + sigmas
init_vals  = {}
sigmas_map = {}
for ji, jimg in enumerate(fit_imgs):
    k    = jimg.key
    base = ji * len(PRM_ORDER)
    u0   = float(dem.interp_alt(jimg.prms["e"], jimg.prms["n"]))
    init_vals[f"{k}_e"]     = jimg.prms["e"]
    init_vals[f"{k}_n"]     = jimg.prms["n"]
    init_vals[f"{k}_log_h"] = np.log(max(jimg.prms["u"] - u0, 1e-3))
    for ki, pk in enumerate(PRM_ORDER):
        pname = f"{k}_log_h" if pk == "u" else f"{k}_{pk}"
        if pname in corner_vars:
            sigmas_map[pname] = ps.sigmas[base + ki]

ant_e0, ant_n0, ant_u_init = ps.ant_pos_prior
ant_u0 = float(dem.interp_alt(ant_e0, ant_n0))
init_vals["ant_e"]     = ant_e0
init_vals["ant_n"]     = ant_n0
init_vals["ant_log_h"] = np.log(max(ant_u_init - ant_u0, 1e-3))
n_cam_prms = len(fit_imgs) * len(PRM_ORDER)
for ki, suffix in enumerate(("e", "n", "log_h")):
    sigmas_map[f"ant_{suffix}"] = ps.sigmas[n_cam_prms + ki]

prior_data = prior_trace.prior
samples    = {v: prior_data[v].values.flatten() for v in corner_vars}

n    = len(corner_vars)
cell = 2.2
fig_corner, axes_c = plt.subplots(n, n, figsize=(cell * n, cell * n + 0.8))
fig_corner.suptitle(
    f"Prior pair plot  —  expect uncorrelated blobs\n{META_STR}",
    fontsize=7, y=1.005,
)

for i in range(n):
    vi    = corner_vars[i]
    src_i, lbl_i, unit_i = make_label(vi)
    xi    = samples[vi]
    iv_i  = init_vals[vi]
    sig_i = sigmas_map.get(vi, None)

    for j in range(n):
        ax   = axes_c[i, j]
        vj   = corner_vars[j]
        xj   = samples[vj]
        iv_j = init_vals[vj]

        if i == j:
            ax.hist(xi, bins=40, color="steelblue", density=True, alpha=0.8)
            ax.axvline(iv_i, color="red", lw=1.5)
            if sig_i is not None:
                ax.axvspan(iv_i - sig_i, iv_i + sig_i, alpha=0.18, color="red")
            ax.set_yticks([])
            src_i2, lbl_i2, unit_i2 = make_label(vi)
            ax.set_title(f"{src_i2}\n{lbl_i2}\n[{unit_i2}]", fontsize=6, pad=2)

        elif i > j:
            src_j, lbl_j, unit_j = make_label(vj)
            ax.hexbin(xj, xi, gridsize=25, cmap="viridis", linewidths=0.2)
            ax.axvline(iv_j, color="red", lw=0.8, alpha=0.7)
            ax.axhline(iv_i, color="red", lw=0.8, alpha=0.7)
            r = float(np.corrcoef(xi, xj)[0, 1])
            ax.text(0.04, 0.96, f"r={r:.2f}", transform=ax.transAxes,
                    fontsize=6, va="top", color="white",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55))
            if j == 0:
                ax.set_ylabel(f"{src_i}\n{lbl_i}\n[{unit_i}]", fontsize=6)
            if i == n - 1:
                src_j2, lbl_j2, unit_j2 = make_label(vj)
                ax.set_xlabel(f"{src_j2}\n{lbl_j2}\n[{unit_j2}]", fontsize=6)

        else:
            ax.axis("off")
            if i == 0 and j == n - 1:
                legend_txt = (
                    "Prior pair plot\n"
                    "───────────────\n"
                    "Diagonal: marginal\n"
                    "  prior distribution\n\n"
                    "Lower tri: hexbin\n"
                    "  joint distribution\n\n"
                    "Upper tri: summary\n"
                    "  statistics\n\n"
                    "─── Red line ───\n"
                    "  init value\n\n"
                    "─── Red band ───\n"
                    "  ±1σ prior width\n\n"
                    "r = Pearson\n"
                    "  correlation\n"
                    "(expect ~0 for\n"
                    "independent priors)"
                )
                ax.text(0.05, 0.97, legend_txt, transform=ax.transAxes,
                        fontsize=6.5, va="top", family="monospace",
                        bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5",
                                  ec="gray", alpha=0.9))
            else:
                mu_i  = xi.mean();  std_i = xi.std()
                mu_j  = xj.mean();  std_j = xj.std()
                r     = float(np.corrcoef(xi, xj)[0, 1])
                src_i2, lbl_i2, _ = make_label(vi)
                src_j2, lbl_j2, _ = make_label(vj)
                txt = (f"{lbl_i2}\n"
                       f"  μ={mu_i:.2f}  σ={std_i:.2f}\n\n"
                       f"{lbl_j2}\n"
                       f"  μ={mu_j:.2f}  σ={std_j:.2f}\n\n"
                       f"r = {r:.3f}")
                ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                        fontsize=6, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0",
                                  ec="lightgray", alpha=0.9))

        ax.tick_params(labelsize=5)

plt.tight_layout(pad=0.4)
fig_corner.savefig(CORNER_OUT, dpi=130, bbox_inches="tight")
print(f"Corner plot saved to {CORNER_OUT}")