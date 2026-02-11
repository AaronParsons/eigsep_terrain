#!/usr/bin/env python
"""
MCMC runner
"""
import argparse
import glob
import os
import sys

import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r 

BOX_SIZE = 0.3  # m

DEFAULT_META = {
    "0817": {"ant_px": (2 * 1366, 2 * 1221)},
    "0833": {"ant_px": (1606, 2700)},
    "0860": {"ant_px": (2924, 1945)},
}

DEFAULT_PRMS = (
    1734.11, 2069.00, 1760.97, 1.4706, 3.6932, -0.0493, 9830.11,
    1611.31, 1849.00, 1659.78, 1.2053, 1.2414, -0.0244, 5081.08,
    1541.90, 1998.96, 1765.06, 1.5412, 0.6147, 0.1585, 2328.64,
    1651.83, 2024.17, 1781.46,
)


def _apply_prms_to_dem_and_meta(
    dem: DEM,
    meta: dict,
    img_keys_in_fit_order: list[str],
    prms: np.ndarray,
    prm_len: int,
) -> None:
    """
    Unpack prms:
      - per-image chunks of length prm_len (e,n,u,th,ph,ti,f)
      - last 3 numbers are platform (ant_e, ant_n, ant_u)
    """
    nimgs = len(img_keys_in_fit_order)
    expected = nimgs * prm_len + 3
    if prms.size != expected:
        raise ValueError(
            f"prms has {prms.size} values; expected {expected} "
            f"({nimgs} images * {prm_len} params + 3 platform)."
        )

    # Platform at end
    platform = prms[-3:].astype(dtype_r)
    dem["platform"] = platform

    # Per-image chunks
    off = 0
    for key in img_keys_in_fit_order:
        chunk = prms[off : off + prm_len]
        off += prm_len
        meta[key]["prms"] = tuple(float(x) for x in chunk)
        dem[key] = np.asarray(chunk[:3], dtype=dtype_r)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default="marjum_dem.npz")
    ap.add_argument("--img-glob", default="/home/aparsons/Downloads/IMG_08*.jpg")
    ap.add_argument("--seed", type=int, default=None, help="Defaults to random [0,999]")

    # HorizonImage params
    ap.add_argument("--px-dist", type=int, default=30)
    ap.add_argument("--px-smooth", type=int, default=150)

    # PositionSolver / ray tracing params
    ap.add_argument("--n-rays", type=int, default=4000)

    # logL op params
    ap.add_argument("--eps", type=float, default=1e-2)

    # Step method params
    ap.add_argument("--scaling", type=float, default=1e-2)
    ap.add_argument("--tune-interval", type=int, default=50)

    # Sampling params
    ap.add_argument("--draws", type=int, default=4500)
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--cores", type=int, default=1)

    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    # Seed / outfile
    seed = args.seed if args.seed is not None else int(np.random.randint(1000))
    np.random.seed(seed)
    outfile = f"trace_seed{seed:03d}.nc"
    print(f"RANDOM SEED: {seed}")
    print(f"OUTFILE: {outfile}")
    assert not os.path.exists(outfile) # make sure file won't be overwritten

    # Load DEM
    dem = DEM(cache_file=args.cache_file)

    # Select images
    files = sorted(glob.glob(args.img_glob))
    if not files:
        raise FileNotFoundError(f"No images matched --img-glob: {args.img_glob}")

    meta = {k: dict(v) for k, v in DEFAULT_META.items()}
    # Build HorizonImage list once (for MCMC)
    imgs = [HorizonImage(f, meta, px_smooth=args.px_smooth, px_dist=args.px_dist) for f in files]
    imgs = [img for img in imgs if img.key in meta]
    if not imgs:
        raise RuntimeError("No images matched keys in meta after loading HorizonImage objects.")

    fit_imgs, static_imgs = imgs, []
    img_keys = [img.key for img in fit_imgs]

    prms = np.asarray(DEFAULT_PRMS, dtype=dtype_r)

    _apply_prms_to_dem_and_meta(
        dem=dem,
        meta=meta,
        img_keys_in_fit_order=img_keys,
        prms=prms,
        prm_len=len(PRM_ORDER),
    )
    platform = dem["platform"]

    ps = PositionSolver(
        dem["platform"],
        fit_imgs,
        static_imgs,
        args.n_rays,
        dem,
        box_size=BOX_SIZE,
    )
    ps.set_mcmc_prms(prms)
    ps.set_mcmc_sigmas()

    eps = dtype_r(args.eps)

    @as_op(itypes=[pt.fvector], otypes=[pt.fscalar])
    def total_logp_op(theta):
        return np.asarray(ps.total_logL(
                          np.asarray(theta, dtype=dtype_r), eps=eps),
                          dtype=dtype_r)

    with pm.Model() as model:
        mcmc_prms = ps.get_mcmc_prms()

        rng = np.random.default_rng(seed)

        initvals = []
        for c in range(args.chains):
            jitter = rng.normal(0.0, np.asarray(ps.sigmas) * args.scaling,
                                size=prms.size)
            start_c = prms + jitter
            initvals.append({p.name: v for p, v in zip(mcmc_prms, start_c)})

        theta = pt.cast(pt.stack(mcmc_prms), "float32")
        logL = total_logp_op(theta)
        pm.Potential("lik", logL)

        step = pm.DEMetropolisZ(
            S=np.asarray(ps.sigmas, dtype=dtype_r),
            scaling=args.scaling,
            tune="scaling",
            tune_interval=args.tune_interval,
        )

        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            step=step,
            initvals=initvals,
            cores=args.cores,
            random_seed=seed,
            progressbar=True,
        )


    az.to_netcdf(trace, outfile)
    print(f"Accepted step fraction = {float(trace.sample_stats.accepted.mean()): 4.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
