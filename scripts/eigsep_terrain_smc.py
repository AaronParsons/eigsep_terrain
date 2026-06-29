#!/usr/bin/env python
"""
SMC (Sequential Monte Carlo) runner using PyMC's built-in pm.sample_smc.

SMC internally tempers the likelihood from 0→1 across stages, which allows
particles to jump between modes — solving the mixing problem seen with
DEMetropolisZ on multimodal log_h posteriors.

Outputs for each run:
  trace_smc_seed{NNN}.nc        ArviZ InferenceData
  trace_smc_seed{NNN}_meta.json Sampling metadata

No new dependencies — uses your existing PyMC install.

Usage (no MAP):
  python eigsep_terrain_smc.py

Usage (with MAP):
  python eigsep_terrain_smc.py --map-file map_seed042.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.compile.ops import as_op, wrap_py

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


def _apply_prms_to_dem_and_meta(dem, meta, img_keys_in_fit_order, prms, prm_len):
    nimgs = len(img_keys_in_fit_order)
    expected = nimgs * prm_len + 3
    if prms.size != expected:
        raise ValueError(
            f"prms has {prms.size} values; expected {expected} "
            f"({nimgs} images * {prm_len} params + 3 platform)."
        )
    platform = prms[-3:].astype(dtype_r)
    dem["platform"] = platform
    off = 0
    for key in img_keys_in_fit_order:
        chunk = prms[off: off + prm_len]
        off += prm_len
        meta[key]["prms"] = tuple(float(x) for x in chunk)
        dem[key] = np.asarray(chunk[:3], dtype=dtype_r)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default="marjum_dem.npz")
    ap.add_argument("--img-glob",
                    default="/Users/komalkaur/Desktop/eigsep_stuff/hrzn_mapping/imgs/IMG*.jpg")
    ap.add_argument("--seed", type=int, default=None)

    # HorizonImage params
    ap.add_argument("--px-dist",   type=int, default=30)
    ap.add_argument("--px-smooth", type=int, default=150)

    # PositionSolver / ray tracing params
    ap.add_argument("--n-rays",      type=int,   default=4000)
    ap.add_argument("--ant-weight",  type=float, default=1.0)
    ap.add_argument("--disable-ant", action="store_true")
    ap.add_argument("--eps",         type=float, default=1e-2)
    ap.add_argument("--fine-delta",  type=float, default=0.25)

    # Camera position / height corrections
    ap.add_argument("--img0-e", type=float, default=1734.11)
    ap.add_argument("--img0-n", type=float, default=2069.00)
    ap.add_argument("--img1-e", type=float, default=1611.31)
    ap.add_argument("--img1-n", type=float, default=1849.00)
    ap.add_argument("--img2-e", type=float, default=1541.90)
    ap.add_argument("--img2-n", type=float, default=1998.96)
    ap.add_argument("--set-cam-height", action="store_true", default=False,
                    help="Override u from DEFAULT_PRMS with DEM + cam_height")
    ap.add_argument("--cam-height", type=float, default=1.6)

    # Prior sigmas
    ap.add_argument("--pos-err",     type=float, default=30.0)
    ap.add_argument("--ang-err-deg", type=float, default=5.0)
    ap.add_argument("--f-err",       type=float, default=0.1)
    ap.add_argument("--log-h-sigma", type=float, default=1.0)

    # MAP file
    ap.add_argument("--map-file", default=None,
                    help="Path to map_seed{NNN}.json. Uses MAP as prior centres.")

    # SMC params
    ap.add_argument("--draws",        type=int,   default=2000,
                    help="Particles (samples) per SMC stage (default: 2000)")
    ap.add_argument("--chains",       type=int,   default=1,
                    help="Independent SMC runs to combine (default: 1)")
    ap.add_argument("--cores",        type=int,   default=12,
                    help="Parallel cores (default: 12)")
    ap.add_argument("--threshold",    type=float, default=0.5,
                    help="ESS threshold for stage adaptation (default: 0.5)")
    ap.add_argument("--correlation-threshold", type=float, default=0.01,
                    help="MH correlation threshold (default: 0.01)")

    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    # ── seed / outfile ────────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else int(np.random.randint(1000))
    np.random.seed(seed)
    stem     = f"trace_smc_seed{seed:03d}"
    outfile  = f"{stem}.nc"
    metafile = f"{stem}_meta.json"
    print(f"RANDOM SEED: {seed}")
    print(f"OUTFILE:     {outfile}")
    print(f"METAFILE:    {metafile}")
    assert not os.path.exists(outfile), \
        f"{outfile} already exists; choose a different seed or move the file."

    # ── load DEM ──────────────────────────────────────────────────────────────
    dem = DEM(cache_file=args.cache_file)

    # ── load images ───────────────────────────────────────────────────────────
    files = sorted(glob.glob(args.img_glob))
    if not files:
        raise FileNotFoundError(f"No images matched --img-glob: {args.img_glob}")

    meta = {k: dict(v) for k, v in DEFAULT_META.items()}
    imgs = [HorizonImage(f, meta, px_smooth=args.px_smooth, px_dist=args.px_dist)
            for f in files]
    imgs = [img for img in imgs if img.key in meta]
    if not imgs:
        raise RuntimeError("No images matched keys in meta.")

    fit_imgs, static_imgs = imgs, []
    img_keys = [img.key for img in fit_imgs]

    # ── build prms_u ──────────────────────────────────────────────────────────
    prms_u = np.asarray(DEFAULT_PRMS, dtype=dtype_r)
    prms_u[0]  = args.img0_e;  prms_u[1]  = args.img0_n
    prms_u[7]  = args.img1_e;  prms_u[8]  = args.img1_n
    prms_u[14] = args.img2_e;  prms_u[15] = args.img2_n

    for idx, (e_arg, n_arg, label) in enumerate([
        (args.img0_e, args.img0_n, "img0"),
        (args.img1_e, args.img1_n, "img1"),
        (args.img2_e, args.img2_n, "img2"),
    ]):
        u_orig = float(DEFAULT_PRMS[2 + idx * 7])
        h_orig = u_orig - float(dem.interp_alt(e_arg, n_arg))
        u_new  = float(dem.interp_alt(e_arg, n_arg)) + args.cam_height
        print(f"{label}: orig u={u_orig:.2f}  orig h={h_orig:.2f}m  "
              f"-> new u={u_new:.2f}  new h={args.cam_height:.2f}m")

    if args.set_cam_height:
        prms_u[2]  = float(dem.interp_alt(args.img0_e, args.img0_n)) + args.cam_height
        prms_u[9]  = float(dem.interp_alt(args.img1_e, args.img1_n)) + args.cam_height
        prms_u[16] = float(dem.interp_alt(args.img2_e, args.img2_n)) + args.cam_height
        print(f"Camera heights set to {args.cam_height}m above terrain.")

    _apply_prms_to_dem_and_meta(
        dem=dem, meta=meta,
        img_keys_in_fit_order=img_keys,
        prms=prms_u, prm_len=len(PRM_ORDER),
    )

    # ── build solver ──────────────────────────────────────────────────────────
    ps = PositionSolver(
        dem["platform"], fit_imgs, static_imgs,
        args.n_rays, dem, box_size=BOX_SIZE,
    )
    prms_h = ps.prms_u_to_h(prms_u)
    ps.set_mcmc_prms(prms_h)
    ps.set_mcmc_sigmas(
        pos_err=args.pos_err,
        ang_err=np.deg2rad(args.ang_err_deg),
        f_err=args.f_err,
        log_h_sigma=args.log_h_sigma,
    )

    # ── load MAP file if provided ─────────────────────────────────────────────
    map_file_meta = None
    if args.map_file is not None:
        print(f"\nLoading MAP from: {args.map_file}")
        with open(args.map_file) as _f:
            _map = json.load(_f)

        map_file_meta = {
            "map_file":      args.map_file,
            "map_seed":      _map.get("seed"),
            "map_method":    _map.get("method"),
            "map_logL":      _map.get("map_logL"),
            "map_converged": _map.get("converged"),
        }

        _param_names = _map["param_names"]
        _map_h       = _map["map_params_h"]
        for i, name in enumerate(_param_names):
            if name in _map_h:
                prms_h[i] = dtype_r(_map_h[name])
        ps.set_mcmc_prms(prms_h)

        if _map.get("hess_stds") is not None:
            _hess      = _map["hess_stds"]
            new_sigmas = list(ps.sigmas)
            for i, name in enumerate(_param_names):
                v = _hess.get(name)
                if v is not None and np.isfinite(v) and v > 0:
                    new_sigmas[i] = dtype_r(v)
            ps.sigmas = new_sigmas
            print(f"  Prior centres: MAP  Prior sigmas: Hessian-derived")
        else:
            print(f"  Prior centres: MAP  Prior sigmas: CLI values")

    # ── likelihood op ─────────────────────────────────────────────────────────
    # wrap_py is the modern replacement for as_op; also avoids pickle issues with SMC
    @wrap_py(itypes=[pt.fvector], otypes=[pt.fscalar])
    def total_logp_op(theta):
        try:
            return np.asarray(
                ps.total_logL(
                    theta=np.asarray(theta, dtype=dtype_r),
                    n_rays=args.n_rays,
                    eps=args.eps,
                    ant_weight=args.ant_weight,
                    disable_ant=args.disable_ant,
                    fine_delta=args.fine_delta,
                ),
                dtype=dtype_r,
            )
        except (ValueError, FloatingPointError):
            return np.asarray(-np.inf, dtype=dtype_r)

    # ── PyMC model ────────────────────────────────────────────────────────────
    with pm.Model() as model:
        mcmc_prms = ps.get_mcmc_prms()
        theta     = pt.cast(pt.stack(mcmc_prms), "float32")
        logL      = total_logp_op(theta)
        pm.Potential("lik", logL)

        # SMC — no step method needed, PyMC handles tempering internally
        trace = pm.sample_smc(
            draws=args.draws,
            chains=args.chains,
            # cores omitted: wrap_py ops cannot be pickled for multiprocessing
            random_seed=seed,
            threshold=args.threshold,
            correlation_threshold=args.correlation_threshold,
            progressbar=True,
        )

    # ── save trace ────────────────────────────────────────────────────────────
    az.to_netcdf(trace, outfile)

    # ── summary stats ─────────────────────────────────────────────────────────
    param_names = [p.name for p in mcmc_prms]

    # SMC doesn't have a single acceptance fraction; use sample_stats if available
    try:
        accepted = float(trace.sample_stats.accepted.mean())
    except Exception:
        accepted = None

    # Number of SMC stages completed
    try:
        n_stages = int(trace.sample_stats.stage.max().values) + 1
    except Exception:
        n_stages = None

    param_summary = {}
    for i, name in enumerate(param_names):
        arr = trace.posterior[name].values.flatten()
        param_summary[name] = {
            "mean":        float(arr.mean()),
            "std":         float(arr.std()),
            "prior_mu":    float(prms_h[i]),
            "prior_sigma": float(ps.sigmas[i]),
        }

    # ── write metadata ────────────────────────────────────────────────────────
    run_meta = {
        "seed":          seed,
        "outfile":       outfile,
        "sampler":       "SMC",
        "img_keys":      img_keys,
        "param_names":   param_names,
        "prm_order":     list(PRM_ORDER),
        "accepted_mean": accepted,
        "n_smc_stages":  n_stages,
        "map_file":      map_file_meta,
        "sampling": {
            "draws":                 args.draws,
            "chains":                args.chains,
            "cores":                 args.cores,
            "threshold":             args.threshold,
            "correlation_threshold": args.correlation_threshold,
        },
        "priors": {
            "pos_err":     args.pos_err,
            "ang_err_deg": args.ang_err_deg,
            "f_err":       args.f_err,
            "log_h_sigma": args.log_h_sigma,
        },
        "likelihood": {
            "eps":         args.eps,
            "n_rays":      args.n_rays,
            "ant_weight":  args.ant_weight,
            "disable_ant": args.disable_ant,
            "fine_delta":  args.fine_delta,
        },
        "image": {
            "px_dist":        args.px_dist,
            "px_smooth":      args.px_smooth,
            "img_glob":       args.img_glob,
            "cam_height":     args.cam_height,
            "set_cam_height": args.set_cam_height,
        },
        "param_summary": param_summary,
    }

    with open(metafile, "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"SMC stages completed:  {n_stages}")
    print(f"Accepted mean:         {accepted}")
    print(f"Trace written to:      {outfile}")
    print(f"Metadata written to:   {metafile}")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())