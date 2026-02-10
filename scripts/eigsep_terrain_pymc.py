import numpy as np
from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import os, sys, glob
import arviz, corner

seed = np.random.randint(1000)
np.random.seed(seed)
print(f'RANDOM SEED: {seed}')
outfile = f"trace_seed{seed:03d}.nc"
print(f'OUTFILE: {outfile}')

CACHE_FILE = 'marjum_dem.npz'
dem = DEM(cache_file=CACHE_FILE)

meta = {
    '0817': {'ant_px': (2*1366, 2*1221)},
    '0833': {'ant_px': (1606, 2700)},
    #'0834': {'ant_px': (1622, 2251)},
#    'best_prms': ( 1642.45,  1887.80,   1678.94,  1.1787,  1.2417, -0.0310,  2933.66),  #[LOSS= 0.0685]
    '0860': {'ant_px': (2924, 1945)},
}
BOX_SIZE = 0.3  # m
n_rays = 4000

files = sorted(glob.glob('/home/aparsons/Downloads/IMG_08*.jpg'))
imgs = [HorizonImage(f, px_dist=30) for f in files]
imgs = [img for img in imgs if img.key in meta]
fit_imgs, static_imgs = imgs, []
_ps = PositionSolver(np.array([0.0, 0, 0]), fit_imgs, static_imgs, 400, dem)

sols = {
'best_prms': (
 1734.11,  2069.00,  1760.97,  1.4706,  3.6932, -0.0493,  9830.11,
 1611.31,  1849.00,  1659.78,  1.2053,  1.2414, -0.0244,  5081.08,
 1541.90,  1998.96,  1765.06,  1.5412,  0.6147,  0.1585,  2328.64,
 1651.83,  2024.17,  1781.46
),  #[LOGL=-6643.6546]
}
_ps.set_mcmc_prms(sols['best_prms'])

# Propagate best parameters back into original objects
dem['platform'] = _ps.ant_pos
for I in _ps.imgs:
    meta[I.key].update({'best_prms': tuple(I.get_prms())})

for k in meta.keys():
    dem[k] = np.asarray(meta[k]['best_prms'][:3], dtype=dtype_r)
    
imgs = [HorizonImage(f, meta, px_smooth=150, px_dist=30) for f in files]
imgs = [img for img in imgs if img.key in meta]
fit_imgs, static_imgs = imgs, []
ps = PositionSolver(dem['platform'], fit_imgs, static_imgs, n_rays, dem, box_size=BOX_SIZE)
ps.set_mcmc_sigmas()

@as_op(itypes=[pt.fvector], otypes=[pt.fscalar])
def total_logp_op(theta):
    return np.array(ps.total_logL(np.asarray(theta, dtype=dtype_r),
                            eps=1e-2), dtype=dtype_r)
    
with pm.Model() as model:
    prms = ps.get_mcmc_prms()
    theta = pt.cast(pt.stack(prms), "float32")
    logL = total_logp_op(theta)
    pm.Potential("lik", logL)

    step = pm.DEMetropolisZ(
        S=np.array(ps.sigmas),
        scaling=3e-4,
        tune='scaling',   # let it adapt scale
        tune_interval=50, # adapt more often than default 100
    )
    
    trace = pm.sample(
        draws=450,
        tune=50,
        chains=1,
        step=step,
        cores=1,
        random_seed=seed,
        progressbar=True,
    )
    arviz.to_netcdf(trace, outfile);
    print(f"Accepted step fraction = {float(trace.sample_stats.accepted.mean()): 4.3f}")
