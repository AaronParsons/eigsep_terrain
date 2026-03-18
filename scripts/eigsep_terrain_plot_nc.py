#!/usr/bin/env python
"""Plot MCMC traces from eigsep_terrain_pymc.py output .nc files.

Usage:
    eigsep_terrain_plot_nc.py [--cache-file F] [--img-glob G] trace_seed*.nc
"""
import argparse
import glob
import numpy as np
import matplotlib.pylab as plt
from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER
from eigsep_data.plot import terrain_plot
import arviz
import corner

ap = argparse.ArgumentParser()
ap.add_argument("--cache-file", default="marjum_dem.npz")
ap.add_argument(
    "--img-glob", default="/home/aparsons/Downloads/IMG_08*.jpg"
)
ap.add_argument("nc_files", nargs="*")
args = ap.parse_args()

np.random.seed(42)

dem = DEM(cache_file=args.cache_file)

meta = {
    '0817': {'ant_px': (2*1366, 2*1221)},
    '0833': {'ant_px': (1606, 2700)},
    #'0834': {'ant_px': (1622, 2251)},
    '0860': {'ant_px': (2924, 1945)},
}
BOX_SIZE = 0.3  # m

# Default parameter values (e, n, u, th, ph, ti, f) per camera,
# followed by (ant_e, ant_n, ant_u).
DEFAULT_PRMS = (
    1734.11, 2069.00, 1760.97, 1.4706, 3.6932, -0.0493,  9830.11,
    1611.31, 1849.00, 1659.78, 1.2053, 1.2414, -0.0244,  5081.08,
    1541.90, 1998.96, 1765.06, 1.5412, 0.6147,  0.1585,  2328.64,
    1651.83, 2024.17, 1781.46,
)

files = sorted(glob.glob(args.img_glob))
print(files)
imgs = [HorizonImage(f, meta, px_smooth=150, px_dist=30) for f in files]
imgs = [img for img in imgs if img.key in meta]
fit_imgs, static_imgs = imgs, []
n_rays = 4000

# Initialise images and dem markers from DEFAULT_PRMS
default_prms = np.asarray(DEFAULT_PRMS)
for i, img in enumerate(fit_imgs):
    base = i * len(PRM_ORDER)
    img.set_prms(default_prms[base:base + len(PRM_ORDER)])
    dem[img.key] = np.asarray(
        default_prms[base:base + 3], dtype=np.float32
    )
ant_pos_prior = default_prms[-3:].astype(np.float32)
dem['platform'] = ant_pos_prior

ps = PositionSolver(
    ant_pos_prior, fit_imgs, static_imgs, n_rays, dem, box_size=BOX_SIZE
)

trace_files = args.nc_files or sorted(glob.glob("*.nc"))
print(trace_files)

idata = [arviz.from_netcdf(filename) for filename in trace_files]
trc = arviz.concat(*idata, dim="chain")

# Acceptance fraction per chain and overall
if hasattr(trc, 'sample_stats') and hasattr(trc.sample_stats, 'accepted'):
    acc = np.asarray(trc.sample_stats.accepted)  # (chain, draw)
    for c, frac in enumerate(acc.mean(axis=1)):
        print(f"chain {c}: acceptance fraction = {frac:.3f}")
    print(f"overall acceptance fraction = {acc.mean():.3f}")

ordered_names = [
    "0817_e", "0817_n", "0817_log_h", "0817_th", "0817_ph", "0817_ti",
    "0817_f",
    "0833_e", "0833_n", "0833_log_h", "0833_th", "0833_ph", "0833_ti",
    "0833_f",
    "0860_e", "0860_n", "0860_log_h", "0860_th", "0860_ph", "0860_ti",
    "0860_f",
    "ant_e", "ant_n", "ant_log_h",
]

# Update solver and dem markers from trace posterior means (log_h -> u)
trace_means = np.array(
    [float(np.mean(trc.posterior[k])) for k in ordered_names]
)
ps.set_mcmc_prms(trace_means)
for img in ps.fit_imgs:
    dem[img.key] = np.asarray(
        [img.prms[k] for k in 'enu'], dtype=np.float32
    )
dem['platform'] = ps.ant_pos.astype(np.float32)

ps.set_mcmc_sigmas()

fig, axes = plt.subplots(
    nrows=len(trc.posterior), sharex=True, figsize=(8, 12)
)
for i, k in enumerate(trc.posterior.keys()):
    v = np.asarray(trc.posterior[k])
    print(k, np.std(v) / ps.sigmas[i], np.mean(v))
    axes[i].plot(v.T)

for t in range(trc.posterior['ant_e'].shape[0]):
    iprms = [trc.posterior[k][t, 0] for k in ordered_names]
    fprms = [trc.posterior[k][t, -1] for k in ordered_names]
    print(
        t,
        ps.total_logL(np.asarray(iprms)),
        ps.total_logL(np.asarray(fprms)),
    )

vars_to_plot = ["ant_e", "ant_n", "ant_log_h"]

posterior = trc.posterior[vars_to_plot].stack(sample=("chain", "draw"))
samples = np.column_stack(
    [posterior[v].values for v in vars_to_plot]
)
corner.corner(samples, labels=vars_to_plot, show_titles=True)

fig, ax = plt.subplots()
e0, n0, u0 = dem['platform']
rng = 750
alpha = 0.02
terrain_plot(
    dem, ax=ax,
    vmin=u0 - 300, vmax=u0 + 200,
    erng_m=(e0 - rng, e0 + rng),
    nrng_m=(n0 - rng, n0 + rng),
)
ax.plot(
    np.asarray(trc.posterior['ant_e']).flatten(),
    np.asarray(trc.posterior['ant_n']).flatten(),
    'k.', alpha=alpha,
)
for img in imgs:
    try:
        ax.plot(
            np.asarray(trc.posterior[f'{img.key}_e']).flatten(),
            np.asarray(trc.posterior[f'{img.key}_n']).flatten(),
            '.', alpha=alpha,
        )
    except KeyError:
        ax.plot(
            np.asarray(trc.posterior['e']).flatten(),
            np.asarray(trc.posterior['n']).flatten(),
            '.', alpha=alpha,
        )

plt.show()
