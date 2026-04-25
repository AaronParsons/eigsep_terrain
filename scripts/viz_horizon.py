#!/usr/bin/env python
"""
Horizon overlay visualization — Test 1 for solution validity.

For each camera, renders three panels side by side:
  1. Raw image with predicted horizon line
  2. P(sky) segmentation mask with predicted horizon overlay
  3. Pixel-wise agreement map (where model agrees/disagrees with psky)

If the MAP solution is physically correct, the predicted horizon line
should align with the visible sky/ground boundary in the raw image.

Usage
-----
  viz_horizon.py --map-file map_seed398.json [options]
  viz_horizon.py --map-file map_seed398.json --trace-file trace_seed634.nc
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, PositionSolver, PRM_ORDER, dtype_r

BOX_SIZE = 0.3

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


def _apply_prms(dem, meta, img_keys, prms, prm_len):
    dem["platform"] = prms[-3:].astype(dtype_r)
    off = 0
    for key in img_keys:
        chunk = prms[off: off + prm_len]
        off += prm_len
        meta[key]["prms"] = tuple(float(x) for x in chunk)
        dem[key] = np.asarray(chunk[:3], dtype=dtype_r)


def _predicted_sky_grid(img, dem, decimate=8):
    """Ray-trace a decimated pixel grid. Returns (row_coords, col_coords, model_sky bool array)."""
    Ny, Nx = img.npix_y, img.npix_x
    xs = np.arange(0, Ny, decimate)
    ys = np.arange(0, Nx, decimate)
    yy, xx = np.meshgrid(ys, xs)
    rays = img.get_rays(pixels=(xx.ravel(), yy.ravel()), dtype=dtype_r)
    r = img.ray_distance(dem, rays, dtype=dtype_r)
    model_sky = np.isnan(r).reshape(xx.shape)
    return xs, ys, model_sky


def _horizon_scatter(xs, ys, model_sky):
    """Return full-pixel (row, col) coords of the sky/ground boundary."""
    horiz = np.zeros_like(model_sky, dtype=bool)
    horiz[:, :-1] |= model_sky[:, :-1] != model_sky[:, 1:]
    horiz[:-1, :] |= model_sky[:-1, :] != model_sky[1:, :]
    hx, hy = np.where(horiz)
    return xs[hx], ys[hy]


def _upsample_sky(model_sky, target_shape, xs, ys):
    """Nearest-neighbour upsample of decimated model_sky to full image shape."""
    Ny, Nx = target_shape
    sky_full = np.zeros((Ny, Nx), dtype=bool)
    # For each decimated row/col, fill the block
    row_edges = np.append(xs, Ny)
    col_edges = np.append(ys, Nx)
    for i, r0 in enumerate(xs):
        r1 = row_edges[i + 1]
        for j, c0 in enumerate(ys):
            c1 = col_edges[j + 1]
            sky_full[r0:r1, c0:c1] = model_sky[i, j]
    return sky_full


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map-file",   required=True)
    ap.add_argument("--trace-file", default=None,
                    help="ArviZ .nc trace; uses posterior mean instead of MAP")
    ap.add_argument("--cache-file", default="marjum_dem.npz")
    ap.add_argument("--img-glob",
                    default="/Users/komalkaur/Desktop/eigsep_stuff/hrzn_mapping/imgs/IMG*.jpg")
    ap.add_argument("--px-dist",    type=int, default=30)
    ap.add_argument("--px-smooth",  type=int, default=150)
    ap.add_argument("--decimate",   type=int, default=8,
                    help="Pixel stride for ray tracing (default 8; use 4 for finer horizon)")
    ap.add_argument("--img0-e", type=float, default=1734.11)
    ap.add_argument("--img0-n", type=float, default=2069.00)
    ap.add_argument("--img1-e", type=float, default=1611.31)
    ap.add_argument("--img1-n", type=float, default=1849.00)
    ap.add_argument("--img2-e", type=float, default=1541.90)
    ap.add_argument("--img2-n", type=float, default=1998.96)
    ap.add_argument("--set-cam-height", action="store_true", default=True)
    ap.add_argument("--cam-height", type=float, default=1.6)
    ap.add_argument("--outdir", default=None)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    stem   = os.path.splitext(os.path.basename(args.map_file))[0]
    outdir = args.outdir or f"{stem}_horizon_viz"
    os.makedirs(outdir, exist_ok=True)

    # ── setup ─────────────────────────────────────────────────────────────────
    dem   = DEM(cache_file=args.cache_file)
    files = sorted(glob.glob(args.img_glob))
    if not files:
        raise FileNotFoundError(f"No images matched: {args.img_glob}")

    meta  = {k: dict(v) for k, v in DEFAULT_META.items()}
    imgs  = [HorizonImage(f, meta, px_smooth=args.px_smooth, px_dist=args.px_dist)
             for f in files]
    imgs  = [img for img in imgs if img.key in meta]
    img_keys = [img.key for img in imgs]

    prms_u = np.asarray(DEFAULT_PRMS, dtype=dtype_r)
    prms_u[0]  = args.img0_e;  prms_u[1]  = args.img0_n
    prms_u[7]  = args.img1_e;  prms_u[8]  = args.img1_n
    prms_u[14] = args.img2_e;  prms_u[15] = args.img2_n

    if args.set_cam_height:
        prms_u[2]  = float(dem.interp_alt(args.img0_e, args.img0_n)) + args.cam_height
        prms_u[9]  = float(dem.interp_alt(args.img1_e, args.img1_n)) + args.cam_height
        prms_u[16] = float(dem.interp_alt(args.img2_e, args.img2_n)) + args.cam_height

    _apply_prms(dem, meta, img_keys, prms_u, len(PRM_ORDER))

    ps = PositionSolver(dem["platform"], imgs, [], 100, dem, box_size=BOX_SIZE)
    prms_h = ps.prms_u_to_h(prms_u)
    param_names = [
        f"{img.key}_log_h" if k == "u" else f"{img.key}_{k}"
        for img in imgs for k in PRM_ORDER
    ] + ["ant_e", "ant_n", "ant_log_h"]

    # ── load params ───────────────────────────────────────────────────────────
    with open(args.map_file) as f:
        map_json = json.load(f)

    theta = prms_h.copy()
    for i, name in enumerate(param_names):
        if name in map_json["map_params_h"]:
            theta[i] = dtype_r(map_json["map_params_h"][name])

    if args.trace_file is not None:
        import arviz as az
        trace = az.from_netcdf(args.trace_file)
        for i, name in enumerate(param_names):
            if name in trace.posterior:
                theta[i] = dtype_r(float(trace.posterior[name].values.mean()))
        param_source = f"posterior mean  ({os.path.basename(args.trace_file)})"
    else:
        param_source = (f"MAP  logL={map_json['map_logL']:.1f}  "
                        f"method={map_json['method']}  "
                        f"converged={map_json['converged']}")

    ps.set_mcmc_prms(theta)
    print(f"Parameters: {param_source}")

    # ── per-camera plots ──────────────────────────────────────────────────────
    for img in imgs:
        key = img.key
        print(f"\nCamera {key}: ray-tracing {img.npix_y}x{img.npix_x} "
              f"at 1/{args.decimate} resolution...")

        xs, ys, model_sky = _predicted_sky_grid(img, dem, decimate=args.decimate)
        hx_full, hy_full  = _horizon_scatter(xs, ys, model_sky)
        sky_up = _upsample_sky(model_sky, (img.npix_y, img.npix_x), xs, ys)

        # agreement map
        obs_sky      = img.psky > 0.5
        agree_sky    =  sky_up &  obs_sky
        agree_gnd    = ~sky_up & ~obs_sky
        wrong_ground = ~sky_up &  obs_sky   # model says ground, image says sky
        wrong_sky    =  sky_up & ~obs_sky   # model says sky,    image says ground

        rgb = np.ones((*img.psky.shape, 3))
        rgb[agree_sky]    = [0.20, 0.78, 0.20]
        rgb[agree_gnd]    = [0.90, 0.90, 0.90]
        rgb[wrong_ground] = [0.88, 0.18, 0.18]
        rgb[wrong_sky]    = [0.18, 0.18, 0.88]

        pct_agree = 100 * (agree_sky | agree_gnd).sum() / img.psky.size

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        title = (f"Camera {key}  —  Horizon overlay\n"
                 f"{param_source}\n"
                 f"E={img.prms['e']:.1f}  N={img.prms['n']:.1f}  "
                 f"h={np.exp(map_json['map_params_h'].get(f'{key}_log_h', 0)):.2f}m  "
                 f"θ={img.prms['th']:.4f}  φ={img.prms['ph']:.4f}")
        fig.suptitle(title, fontsize=8, y=1.01)

        # Panel 1: raw image
        ax = axes[0]
        ax.imshow(img.img, origin="lower", aspect="auto")
        ax.scatter(hy_full, hx_full, s=2, c="red", linewidths=0,
                   label="predicted horizon", rasterized=True)
        if "ant_px" in img.meta:
            apx = img.meta["ant_px"]
            ax.plot(apx[0], apx[1], "y*", ms=14, label="antenna pixel",
                    markeredgecolor="k", markeredgewidth=0.5)
        ax.set_title("Raw image  +  predicted horizon", fontsize=8)
        ax.set_xlabel("pixel x");  ax.set_ylabel("pixel y (0=bottom)")
        ax.legend(fontsize=7)

        # Panel 2: psky
        ax = axes[1]
        im2 = ax.imshow(img.psky, origin="lower", aspect="auto",
                        cmap="RdYlGn", vmin=0, vmax=1)
        ax.scatter(hy_full, hx_full, s=2, c="blue", linewidths=0,
                   label="predicted horizon", rasterized=True)
        plt.colorbar(im2, ax=ax, fraction=0.03, pad=0.02, label="P(sky)")
        ax.set_title("P(sky) mask  +  predicted horizon", fontsize=8)
        ax.set_xlabel("pixel x")
        ax.legend(fontsize=7)

        # Panel 3: agreement
        ax = axes[2]
        ax.imshow(rgb, origin="lower", aspect="auto")
        ax.scatter(hy_full, hx_full, s=2, c="black", linewidths=0,
                   label="predicted horizon", rasterized=True)
        legend_elements = [
            Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="#33c733", ms=10,
                   label=f"agree sky    ({agree_sky.sum():,})"),
            Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="#e5e5e5", ms=10,
                   label=f"agree ground ({agree_gnd.sum():,})"),
            Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="#e02e2e", ms=10,
                   label=f"wrong ground ({wrong_ground.sum():,})"),
            Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="#2e2ee0", ms=10,
                   label=f"wrong sky    ({wrong_sky.sum():,})"),
            Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="black", ms=10,
                   label="predicted horizon"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="upper right")
        ax.set_title(f"Agreement map  ({pct_agree:.1f}% pixels agree)", fontsize=8)
        ax.set_xlabel("pixel x")

        plt.tight_layout()
        outpath = os.path.join(outdir, f"horizon_{key}.png")
        fig.savefig(outpath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {outpath}")
        print(f"  pixel agreement: {pct_agree:.1f}%  "
              f"(wrong_ground={wrong_ground.sum():,}  wrong_sky={wrong_sky.sum():,})")

    print(f"\nDone. Output in: {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())