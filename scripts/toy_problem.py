#!/usr/bin/env python
"""
Toy problem: synthetic horizon recovery on the real DEM.

Workflow
--------
1. Pick a "true" camera position and orientation on the real DEM.
2. Ray-trace a decimated pixel grid (decimate=4) to generate a ground-truth
   binary sky mask, then upsample to full resolution (model_sky_true).
3. Add controlled noise to simulate segmentation uncertainty:
   psky = sigmoid(sharpness * signed_pixel_dist / max(H,W) + noise)
   This gives a smooth probability map that is 1 in the sky, 0 on
   the ground, and uncertain near the horizon.
4. Build a synthetic HorizonImage substitute that satisfies the
   PositionSolver interface without touching real image files.
5. Run MAP estimation from a perturbed starting point.
6. Report recovery error: |MAP - truth| per parameter.
7. Optionally run a short MCMC and check that the posterior covers truth.

The test passes if:
  - MAP recovers position to within pos_tol metres
  - MAP recovers angles to within ang_tol radians
  - (optional) truth lies within the 95% posterior interval

Usage
-----
  toy_problem.py [options]
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import (
    pixels_to_rays, PositionSolver, PRM_ORDER, dtype_r,
)
from eigsep_terrain.ray_numba import ray_distance_coarse_to_fine_numba
from eigsep_terrain.utils import rot_m, mask_near_horizon

BOX_SIZE = 0.3


# ── synthetic HorizonImage substitute ────────────────────────────────────────

class SyntheticHorizonImage:
    """
    Duck-typed substitute for HorizonImage backed by a ray-traced sky mask
    rather than a real photograph and segmentation model.

    The psky field is:
        logit = sharpness * signed_pixel_dist / max(npix_y, npix_x) + N(0, noise_std)
        psky  = σ(logit) = 1 / (1 + exp(-logit))
    where signed_pixel_dist is the Euclidean distance transform of the
    binary sky mask — positive inside sky regions, negative inside ground.
    This mimics a realistic segmentation output: confident sky/ground far
    from the horizon, uncertain near it.
    """

    def __init__(self, key, true_prms, dem, npix_y, npix_x,
                 sharpness=80.0, noise_std=0.5, px_dist=30,
                 fine_delta=0.25, seed=0, ant_px=None):
        self.key        = key
        self.dem        = dem
        self.npix_y     = npix_y
        self.npix_x     = npix_x
        self.px_dist    = px_dist
        self.px_smooth  = px_dist
        self.fine_delta = fine_delta
        self._px_choice = None
        self.meta       = {"ant_px": ant_px or (npix_x // 2, npix_y // 2)}
        self.filename   = f"synthetic_{key}"
        rng             = np.random.default_rng(seed)

        # Set true params and ray-trace full grid at coarse resolution
        self.set_prms(true_prms)
        self._true_prms = dict(self.prms)

        print(f"  [{key}] Ray-tracing {npix_y}x{npix_x} synthetic image "
              f"(decimate=4)...")
        t0 = time.time()
        dec = 4
        xs  = np.arange(0, npix_y, dec)
        ys  = np.arange(0, npix_x, dec)
        yy, xx = np.meshgrid(ys, xs)
        rays_dec = self.get_rays(pixels=(xx.ravel(), yy.ravel()))
        rays_2d  = rays_dec.reshape(3, -1)
        (E, N), U = dem.get_en(), dem.data
        sp = np.array([self.prms[k] for k in 'enu'], dtype=dtype_r)
        r_dec = ray_distance_coarse_to_fine_numba(
            E, N, U, sp, rays_2d, fine_delta=fine_delta
        )
        model_sky_dec = np.isnan(r_dec).reshape(len(xs), len(ys))

        # Upsample to full resolution (nearest neighbour)
        model_sky_full = np.zeros((npix_y, npix_x), dtype=bool)
        row_edges = np.append(xs, npix_y)
        col_edges = np.append(ys, npix_x)
        for i, r0 in enumerate(xs):
            for j, c0 in enumerate(ys):
                model_sky_full[r0:row_edges[i+1], c0:col_edges[j+1]] = \
                    model_sky_dec[i, j]

        # Compute signed distance from horizon (in pixels) using distance transform
        from scipy.ndimage import distance_transform_edt
        dist_sky   = distance_transform_edt( model_sky_full)
        dist_gnd   = distance_transform_edt(~model_sky_full)
        signed_dist = dist_sky - dist_gnd  # +ve = sky, -ve = ground

        # Logistic psky with noise
        logit = sharpness * signed_dist / max(npix_y, npix_x)
        logit += rng.normal(0.0, noise_std, size=logit.shape)
        psky  = 1.0 / (1.0 + np.exp(-logit))
        self.psky       = psky.astype(np.float32)
        self.sky_mask   = model_sky_full
        self._model_sky_true = model_sky_full

        # Horizon mask and distance (same interface as HorizonImage)
        self.horizon_mask, self.horizon_dist = mask_near_horizon(
            self.sky_mask, px_dist
        )
        print(f"  [{key}] done in {time.time()-t0:.1f}s  "
              f"sky_frac={model_sky_full.mean():.2f}")

    def set_prms(self, prms):
        self.prms = dict(zip(PRM_ORDER, prms))

    def get_prms(self):
        return (self.prms[k] for k in PRM_ORDER)

    def get_rays(self, pixels=None, dtype=dtype_r):
        z_rays = pixels_to_rays(self.npix_y, self.npix_x,
                                f=self.prms['f'], uv=pixels, dtype=dtype)
        rm_tilt = rot_m(self.prms['ti'], np.array([0, 0, 1], dtype=dtype))
        rm_th   = rot_m(self.prms['th'], np.array([0, 1, 0], dtype=dtype))
        rm_ph   = rot_m(self.prms['ph'], np.array([0, 0, 1], dtype=dtype))
        rm      = rm_ph @ (rm_th @ rm_tilt)
        return np.einsum('ij,j...->i...', rm, z_rays)

    def choose_pixels(self, N=1000, mask=None, reset=False):
        if reset:
            self._px_choice = None
        if self._px_choice is None:
            if mask is None:
                mask = self.horizon_mask
            x, y = np.where(mask)
            if x.size == 0:
                raise RuntimeError(f"[{self.key}] horizon_mask is empty — "
                                   "try a larger px_dist")
            w    = np.exp(-0.5 * self.horizon_dist[x, y]**2 /
                          (self.px_dist / 2)**2)
            # Use a fixed seed derived from key for reproducibility
            _seed = abs(hash(self.key)) % (2**31)
            rng  = np.random.default_rng(_seed)
            N    = min(N, x.size)
            inds = rng.choice(x.size, size=N, replace=False, p=w / w.sum())
            self._px_choice = (x[inds], y[inds])
        return self._px_choice

    def ray_distance(self, dem, rays, dtype=dtype_r, fine_delta=None):
        if fine_delta is None:
            fine_delta = self.fine_delta
        rays_2d = rays.reshape(rays.shape[0], -1)
        (E, N), U = dem.get_en(), dem.data
        sp = np.array([self.prms[k] for k in 'enu'], dtype=dtype)
        r  = ray_distance_coarse_to_fine_numba(
            E, N, U, sp, rays_2d, fine_delta=fine_delta
        )
        r.shape = rays.shape[1:]
        return r

    def horizon_ray_logL(self, dem, n_rays=1000, dtype=dtype_r,
                         eps=1e-3, fine_delta=None):
        if fine_delta is None:
            fine_delta = self.fine_delta
        x_px, y_px = self.choose_pixels(N=n_rays)
        psky  = self.psky[x_px, y_px].clip(eps, 1 - eps)
        rays  = self.get_rays(pixels=(x_px, y_px), dtype=dtype)
        r     = self.ray_distance(dem, rays, dtype=dtype,
                                  fine_delta=fine_delta)
        model_sky   = np.isnan(r)
        logp_sky    = np.log(psky)
        logp_ground = np.log1p(-psky)
        return float(np.sum(np.where(model_sky, logp_sky, logp_ground)))

    def ant_logL(self, ant_pos, box_size):
        ant_ray = self.get_rays(np.array(self.meta['ant_px'][::-1]))
        r_ant   = ant_pos - np.array(
            [self.prms['e'], self.prms['n'], self.prms['u']]
        )
        cos_pred    = (np.dot(ant_ray, r_ant) /
                       (np.linalg.norm(ant_ray) * np.linalg.norm(r_ant)))
        delta_theta = np.arccos(cos_pred.clip(-1, 1))
        sigma_theta = box_size / np.linalg.norm(r_ant)
        return (np.log(1 / np.sqrt(2 * np.pi * sigma_theta**2)) -
                0.5 * delta_theta**2 / sigma_theta**2)

    @property
    def prms_str(self):
        return (f"{self.prms['e']: 7.2f}, {self.prms['n']: 7.2f}, "
                f"{self.prms['u']: 7.2f}, {self.prms['th']: 6.4f}, "
                f"{self.prms['ph']: 6.4f}, {self.prms['ti']: 5.4f}, "
                f"{self.prms['f']: 7.2f}")


# ── argparse ──────────────────────────────────────────────────────────────────

def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cache-file", default="marjum_dem.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=None)

    # True camera params (ground truth)
    ap.add_argument("--true-e",  type=float, default=1760.0)
    ap.add_argument("--true-n",  type=float, default=2083.0)
    ap.add_argument("--true-h",  type=float, default=2.0,
                    help="True height above ground [m]")
    ap.add_argument("--true-th", type=float, default=1.47)
    ap.add_argument("--true-ph", type=float, default=3.69)
    ap.add_argument("--true-ti", type=float, default=-0.05)
    ap.add_argument("--true-f",  type=float, default=5000.0)

    # True antenna params
    ap.add_argument("--true-ant-e", type=float, default=1651.0)
    ap.add_argument("--true-ant-n", type=float, default=2025.0)
    ap.add_argument("--true-ant-h", type=float, default=100.0,
                    help="True antenna height above ground [m]")

    # Synthetic image settings
    ap.add_argument("--npix-y",    type=int,   default=1024)
    ap.add_argument("--npix-x",    type=int,   default=1024)
    ap.add_argument("--sharpness", type=float, default=80.0,
                    help="Horizon sharpness in synthetic psky (higher = crisper)")
    ap.add_argument("--noise-std", type=float, default=0.5,
                    help="Gaussian noise on logit psky (simulates seg uncertainty)")
    ap.add_argument("--px-dist",   type=int,   default=30)

    # Recovery settings
    ap.add_argument("--n-rays",      type=int,   default=1000)
    ap.add_argument("--eps",         type=float, default=1e-2)
    ap.add_argument("--fine-delta",  type=float, default=0.25)
    ap.add_argument("--pos-err",     type=float, default=10.0)
    ap.add_argument("--ang-err-deg", type=float, default=5.0)
    ap.add_argument("--f-err",       type=float, default=0.1)
    ap.add_argument("--log-h-sigma", type=float, default=1.0)
    ap.add_argument("--perturb-pos", type=float, default=5.0,
                    help="Perturb starting E,N by this many metres")
    ap.add_argument("--perturb-ang", type=float, default=0.05,
                    help="Perturb starting angles by this many radians")
    ap.add_argument("--n-restarts",  type=int,   default=3)
    ap.add_argument("--maxiter",     type=int,   default=50000)
    ap.add_argument("--ftol",        type=float, default=1e-6)

    # MCMC (optional)
    ap.add_argument("--run-mcmc",   action="store_true",
                    help="Run a short MCMC after MAP to check posterior coverage")
    ap.add_argument("--mcmc-draws", type=int, default=2000)
    ap.add_argument("--mcmc-tune",  type=int, default=500)
    ap.add_argument("--scaling",    type=float, default=0.001)
    ap.add_argument("--disable-ant", action="store_true", default=True,
                    help="Disable ant_logL (default True: antenna pixel is "
                         "not projected so ant_logL is uninformative at truth)")

    # Pass/fail thresholds
    ap.add_argument("--pos-tol", type=float, default=1.0,
                    help="Position recovery tolerance [m] (default 1.0)")
    ap.add_argument("--ang-tol", type=float, default=0.01,
                    help="Angle recovery tolerance [rad] (default 0.01)")
    return ap


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    args   = build_argparser().parse_args(argv)
    rng    = np.random.default_rng(args.seed)
    outdir = args.outdir or f"toy_seed{args.seed:03d}"
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TOY PROBLEM — synthetic horizon recovery")
    print(f"{'='*60}")
    print(f"  seed={args.seed}  outdir={outdir}")

    # ── load DEM ──────────────────────────────────────────────────────────────
    dem = DEM(cache_file=args.cache_file)
    E_grid, N_grid = dem.get_en()
    print(f"  DEM: E={E_grid[0]:.0f}..{E_grid[-1]:.0f}  "
          f"N={N_grid[0]:.0f}..{N_grid[-1]:.0f}  "
          f"dE={E_grid[1]-E_grid[0]:.2f}m")

    # ── build true params ─────────────────────────────────────────────────────
    true_u0  = float(dem.interp_alt(args.true_e, args.true_n))
    true_u   = true_u0 + args.true_h
    true_prms = (args.true_e, args.true_n, true_u,
                 args.true_th, args.true_ph, args.true_ti, args.true_f)

    ant_u0  = float(dem.interp_alt(args.true_ant_e, args.true_ant_n))
    true_ant_pos = np.array([args.true_ant_e, args.true_ant_n,
                              ant_u0 + args.true_ant_h], dtype=dtype_r)

    # Synthetic antenna pixel: project true antenna into image
    # (just place it at image centre for simplicity — test still valid)
    ant_px_col = args.npix_x // 3
    ant_px_row = args.npix_y // 2

    print(f"\n  TRUE params:")
    print(f"    E={args.true_e:.2f}  N={args.true_n:.2f}  "
          f"h={args.true_h:.2f}m  u={true_u:.2f}m")
    print(f"    θ={args.true_th:.4f}  φ={args.true_ph:.4f}  "
          f"ti={args.true_ti:.4f}  f={args.true_f:.1f}")
    print(f"    ant: E={args.true_ant_e:.2f}  N={args.true_ant_n:.2f}  "
          f"h={args.true_ant_h:.2f}m")

    # ── build synthetic image ─────────────────────────────────────────────────
    print(f"\n  Building synthetic image ({args.npix_y}x{args.npix_x})...")
    syn_img = SyntheticHorizonImage(
        key="syn",
        true_prms=true_prms,
        dem=dem,
        npix_y=args.npix_y,
        npix_x=args.npix_x,
        sharpness=args.sharpness,
        noise_std=args.noise_std,
        px_dist=args.px_dist,
        fine_delta=args.fine_delta,
        seed=args.seed,
        ant_px=(ant_px_col, ant_px_row),
    )

    # ── build PositionSolver ──────────────────────────────────────────────────
    ps = PositionSolver(
        ant_pos_prior=true_ant_pos,
        fit_imgs=[syn_img],
        static_imgs=[],   # ant_logL uses self.imgs = fit_imgs + static_imgs
        n_rays=args.n_rays,
        dem=dem,
        box_size=BOX_SIZE,
    )

    # prms_u = [e, n, u, th, ph, ti, f,  ant_e, ant_n, ant_u]
    prms_u = np.array(list(true_prms) + list(true_ant_pos), dtype=dtype_r)
    prms_h = ps.prms_u_to_h(prms_u)
    ps.set_mcmc_prms(prms_h)
    ps.set_mcmc_sigmas(
        pos_err=args.pos_err,
        ang_err=np.deg2rad(args.ang_err_deg),
        f_err=args.f_err,
        log_h_sigma=args.log_h_sigma,
    )
    sigmas = np.asarray(ps.sigmas, dtype=np.float64)

    param_names = (
        ["syn_log_h" if k == "u" else f"syn_{k}" for k in PRM_ORDER] +
        ["ant_e", "ant_n", "ant_log_h"]
    )

    # Fix pixel sample
    syn_img.choose_pixels(N=args.n_rays, reset=True)

    eps = dtype_r(args.eps)

    # ── evaluate logL at truth ────────────────────────────────────────────────
    ps.set_mcmc_prms(prms_h)
    logL_truth = ps.total_logL(prms_h, eps=eps, fine_delta=args.fine_delta,
                               disable_ant=args.disable_ant)
    print(f"\n  logL at truth: {logL_truth:.2f}")

    # ── optimizer objective ───────────────────────────────────────────────────
    def neg_log_posterior(theta):
        try:
            logL = float(ps.total_logL(
                np.asarray(theta, dtype=dtype_r), eps=eps,
                fine_delta=args.fine_delta,
                disable_ant=args.disable_ant,
            ))
        except Exception:
            return np.inf
        if not np.isfinite(logL):
            return np.inf
        log_prior = -0.5 * np.sum(((theta - prms_h) / sigmas) ** 2)
        return -(logL + log_prior)

    # ── run MAP with restarts ─────────────────────────────────────────────────
    print(f"\n  Running MAP ({args.n_restarts} restarts)...")
    best_result = None
    best_val    = np.inf

    _ei = list(PRM_ORDER).index('e');  _ni = list(PRM_ORDER).index('n')
    _ti = list(PRM_ORDER).index('th'); _pi = list(PRM_ORDER).index('ph')

    for i in range(args.n_restarts):
        if i == 0:
            # Controlled first restart: only perturb position and angles
            jitter_u = np.zeros_like(prms_h)
            jitter_u[_ei] = rng.normal(0, args.perturb_pos)
            jitter_u[_ni] = rng.normal(0, args.perturb_pos)
            jitter_u[_ti] = rng.normal(0, args.perturb_ang)
            jitter_u[_pi] = rng.normal(0, args.perturb_ang)
            x0 = prms_h + jitter_u
        else:
            scale  = min(0.3 * i, 1.0)
            x0 = prms_h + rng.normal(0.0, sigmas * scale)

        init_logL = -neg_log_posterior(x0)
        print(f"  Restart {i+1}/{args.n_restarts}  init_logL={init_logL:.2f}")

        t0     = time.time()
        result = minimize(neg_log_posterior, x0.astype(np.float64),
                          method="Powell",
                          options={"maxiter": args.maxiter,
                                   "ftol": args.ftol, "disp": False})
        elapsed  = time.time() - t0
        map_logL = -neg_log_posterior(result.x)
        print(f"    converged={result.success}  nfev={result.nfev}  "
              f"map_logL={map_logL:.2f}  time={elapsed:.1f}s")

        if result.fun < best_val:
            best_val, best_result = result.fun, result

    if best_result is None:
        raise RuntimeError(
            "All restarts returned -inf logL. "
            "Check that the true camera position is inside the DEM "
            "and that img.py has been updated with fine_delta support."
        )
    map_theta = best_result.x
    ps.set_mcmc_prms(map_theta)

    # ── compute recovery errors ───────────────────────────────────────────────
    # Build true_vec in log_h representation to match param_names.
    true_log_h = float(np.log(max(args.true_h, 1e-3)))
    ant_log_h  = float(np.log(max(args.true_ant_h, 1e-3)))
    true_vec_h = np.array([
        args.true_e, args.true_n, true_log_h,
        args.true_th, args.true_ph, args.true_ti, args.true_f,
        args.true_ant_e, args.true_ant_n, ant_log_h,
    ], dtype=np.float64)

    errors = {}
    for i, (name, true_val, map_val) in enumerate(
            zip(param_names, true_vec_h, map_theta)):
        errors[name] = {
            "true":   float(true_val),
            "map":    float(map_val),
            "err":    float(map_val - true_val),
            "abserr": float(abs(map_val - true_val)),
        }

    # Position and angle errors
    e_err  = abs(errors["syn_e"]["err"])
    n_err  = abs(errors["syn_n"]["err"])
    th_err = abs(errors["syn_th"]["err"])
    ph_err = abs(errors["syn_ph"]["err"])
    pos_err_m  = np.sqrt(e_err**2 + n_err**2)

    # ── pass / fail ───────────────────────────────────────────────────────────
    pos_pass = pos_err_m  <= args.pos_tol
    ang_pass = max(th_err, ph_err) <= args.ang_tol
    overall  = "PASS" if (pos_pass and ang_pass) else "FAIL"

    print(f"\n{'='*60}")
    print(f"  RECOVERY RESULTS  [{overall}]")
    print(f"{'='*60}")
    print(f"  {'param':<14}  {'true':>10}  {'MAP':>10}  {'error':>10}")
    for name, d in errors.items():
        flag = ""
        if name in ("syn_e", "syn_n"):
            flag = " ✓" if abs(d["err"]) <= args.pos_tol else " ✗"
        elif name in ("syn_th", "syn_ph"):
            flag = " ✓" if abs(d["err"]) <= args.ang_tol else " ✗"
        print(f"  {name:<14}  {d['true']:>10.4f}  {d['map']:>10.4f}  "
              f"{d['err']:>+10.4f}{flag}")
    print(f"\n  Position error: {pos_err_m:.3f}m  "
          f"(tol={args.pos_tol}m)  [{'PASS' if pos_pass else 'FAIL'}]")
    print(f"  Angle error:    {max(th_err, ph_err):.5f}rad  "
          f"(tol={args.ang_tol}rad)  [{'PASS' if ang_pass else 'FAIL'}]")
    print(f"  logL at truth:  {logL_truth:.2f}")
    print(f"  logL at MAP:    {-best_val:.2f}")
    print(f"  logL gap:       {-best_val - logL_truth:+.2f}  "
          f"(negative = MAP better than truth, ok due to noise)")
    print(f"{'='*60}")

    # ── optional MCMC ─────────────────────────────────────────────────────────
    trace = None
    if args.run_mcmc:
        import pymc as pm
        import pytensor.tensor as pt
        from pytensor.compile.ops import as_op
        import arviz as az

        print(f"\n  Running MCMC (draws={args.mcmc_draws}, tune={args.mcmc_tune})...")

        ps.set_mcmc_prms(map_theta)

        @as_op(itypes=[pt.fvector], otypes=[pt.fscalar])
        def total_logp_op(theta):
            try:
                return np.asarray(ps.total_logL(
                    np.asarray(theta, dtype=dtype_r), eps=eps,
                    fine_delta=args.fine_delta,
                    disable_ant=args.disable_ant,
                ), dtype=dtype_r)
            except Exception:
                return np.asarray(-np.inf, dtype=dtype_r)

        with pm.Model():
            mcmc_prms = ps.get_mcmc_prms()
            theta_pt  = pt.cast(pt.stack(mcmc_prms), "float32")
            pm.Potential("lik", total_logp_op(theta_pt))
            step = pm.DEMetropolisZ(
                S=np.asarray(ps.sigmas, dtype=dtype_r),
                scaling=args.scaling,
                tune="scaling",
            )
            trace = pm.sample(
                draws=args.mcmc_draws, tune=args.mcmc_tune,
                chains=2, step=step,
                initvals=[{p.name: v for p, v in zip(mcmc_prms,
                            ps.eval_cur_prms())}
                           for _ in range(2)],
                progressbar=True,
            )

        # Check posterior coverage of truth
        print(f"\n  Posterior coverage check (truth within 95% CI):")
        coverage = {}
        for i, name in enumerate(param_names):
            vals = trace.posterior[name].values.flatten()
            lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
            true_val = true_vec_h[i]
            covered  = lo <= true_val <= hi
            coverage[name] = covered
            flag = "✓" if covered else "✗"
            print(f"    {name:<14}  truth={true_val:.4f}  "
                  f"CI=[{lo:.4f}, {hi:.4f}]  {flag}")

        n_covered = sum(coverage.values())
        print(f"\n  Coverage: {n_covered}/{len(param_names)} params "
              f"({100*n_covered/len(param_names):.0f}%)")

    # ── plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle(
        f"Toy problem — synthetic horizon recovery  [seed={args.seed}]\n"
        f"pos_err={pos_err_m:.3f}m  ang_err={max(th_err,ph_err):.5f}rad  "
        f"overall={overall}",
        fontsize=9, y=1.01
    )

    # Panel 1: synthetic psky
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(syn_img.psky, origin="lower", aspect="auto",
                   cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, label="P(sky)")
    ax.set_title("Synthetic P(sky)", fontsize=8)

    # Panel 2: horizon mask
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(syn_img.horizon_mask, origin="lower", aspect="auto", cmap="gray")
    x_px, y_px = syn_img.choose_pixels(N=args.n_rays)
    ax.scatter(y_px, x_px, s=1, c="red", label=f"sampled ({args.n_rays})",
               rasterized=True)
    ax.set_title("Horizon mask + sampled pixels", fontsize=8)
    ax.legend(fontsize=7)

    # Panel 3: recovery error bar chart
    ax = fig.add_subplot(gs[0, 2])
    names  = list(errors.keys())
    abserrs = [errors[n]["abserr"] for n in names]
    # Use pos_tol for position/focal params, ang_tol for angle params
    _ang_params  = {"syn_th", "syn_ph", "syn_ti"}
    _pos_params  = {"syn_e", "syn_n", "ant_e", "ant_n"}
    # log_h and f are in log/pixel units — colour gray (no simple threshold)
    colors = []
    for n in names:
        if n in _pos_params:
            colors.append("#e02e2e" if errors[n]["abserr"] > args.pos_tol
                          else "#33c733")
        elif n in _ang_params:
            colors.append("#e02e2e" if errors[n]["abserr"] > args.ang_tol
                          else "#33c733")
        else:
            colors.append("#aaaaaa")  # log_h, f — no threshold
    ax.bar(range(len(names)), abserrs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("|MAP - truth|")
    ax.set_title("Absolute recovery error", fontsize=8)
    ax.axhline(args.pos_tol, color="k", ls="--", lw=0.8,
               label=f"pos_tol={args.pos_tol}m")
    ax.axhline(args.ang_tol, color="gray", ls=":", lw=0.8,
               label=f"ang_tol={args.ang_tol}rad")
    ax.legend(fontsize=7)

    # Panel 4: true vs MAP horizon overlay
    ax = fig.add_subplot(gs[1, :2])
    ax.imshow(syn_img.psky, origin="lower", aspect="auto",
              cmap="RdYlGn", vmin=0, vmax=1, alpha=0.6)

    # True horizon
    syn_img.set_prms(true_prms)
    dec = 8
    xs_d = np.arange(0, args.npix_y, dec)
    ys_d = np.arange(0, args.npix_x, dec)
    yy, xx = np.meshgrid(ys_d, xs_d)
    rays_t = syn_img.get_rays(pixels=(xx.ravel(), yy.ravel()))
    r_t    = syn_img.ray_distance(dem, rays_t, fine_delta=args.fine_delta)
    sky_t  = np.isnan(r_t).reshape(len(xs_d), len(ys_d))
    horiz_t = np.zeros_like(sky_t, dtype=bool)
    horiz_t[:, :-1] |= sky_t[:, :-1] != sky_t[:, 1:]
    horiz_t[:-1, :] |= sky_t[:-1, :] != sky_t[1:, :]
    hx_t, hy_t = np.where(horiz_t)
    ax.scatter(ys_d[hy_t], xs_d[hx_t], s=2, c="blue",
               label="true horizon", rasterized=True)

    # MAP horizon
    ps.set_mcmc_prms(map_theta)
    syn_img.set_prms([ps.fit_imgs[0].prms[k] for k in PRM_ORDER])
    rays_m = syn_img.get_rays(pixels=(xx.ravel(), yy.ravel()))
    r_m    = syn_img.ray_distance(dem, rays_m, fine_delta=args.fine_delta)
    sky_m  = np.isnan(r_m).reshape(len(xs_d), len(ys_d))
    horiz_m = np.zeros_like(sky_m, dtype=bool)
    horiz_m[:, :-1] |= sky_m[:, :-1] != sky_m[:, 1:]
    horiz_m[:-1, :] |= sky_m[:-1, :] != sky_m[1:, :]
    hx_m, hy_m = np.where(horiz_m)
    ax.scatter(ys_d[hy_m], xs_d[hx_m], s=2, c="red",
               label="MAP horizon", rasterized=True)
    ax.set_title("True horizon (blue) vs MAP horizon (red)", fontsize=8)
    ax.legend(fontsize=7)

    # Panel 5: MCMC posterior if run
    ax = fig.add_subplot(gs[1, 2])
    if trace is not None:
        for i, name in enumerate(["syn_e", "syn_n"]):
            vals = trace.posterior[name].values.flatten()
            true_val = errors[name]["true"]
            ax.hist(vals, bins=40, alpha=0.6, density=True,
                    label=f"{name} (truth={true_val:.1f})")
            ax.axvline(true_val, color=f"C{i}", lw=2, ls="--")
        ax.set_title("Posterior: E and N", fontsize=8)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "MCMC not run\n(use --run-mcmc)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("Posterior", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(outdir, "toy_results.png")
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot: {plot_path}")

    # ── write JSON ────────────────────────────────────────────────────────────
    out = {
        "seed":       args.seed,
        "overall":    overall,
        "pos_err_m":  pos_err_m,
        "ang_err_rad":max(th_err, ph_err),
        "pos_tol":    args.pos_tol,
        "ang_tol":    args.ang_tol,
        "logL_truth": logL_truth,
        "logL_map":   float(-best_val),
        "errors":     errors,
        "args":       vars(args),
    }
    json_path = os.path.join(outdir, "toy_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  JSON: {json_path}")
    print(f"\n  RESULT: {overall}\n")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())