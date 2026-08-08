#!/usr/bin/env python
"""
Interactive matplotlib sliders for e, n, u, th, ph, ti, f — live-updates the
ray-traced horizon overlay on the image. Use this to hand-tune a starting
point, then feed the printed params into fit_image.py's --e/--n or a
--map-file, or just eyeball the fit directly.

Usage:
  python tune_image.py --which 0
"""
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from eigsep_terrain.img import HorizonImage, dtype_r

DEFAULT_META = {
    '2209' : {"ant_px": (2146, 232)},
    '2210' : {"ant_px": (1362, 137)},
    '2211' : {"ant_px": (1785, 505)},
    '2213' : {"ant_px": (1117, 549)},
    '2214' : {"ant_px": (1206, 300)},
    '2215' : {"ant_px": (2469, 1411)},
    '2216' : {"ant_px": (2606, 719)},
    '2217' : {"ant_px": (2228, 912)},
    '2218' : {"ant_px": (2711, 919)},
    '2219' : {"ant_px": (1626, 1082)},
    '2220' : {"ant_px": (1580, 166)},
    '2221' : {"ant_px": (2278, 790)},
    '2222' : {"ant_px": (1020, 720)},
    '2223' : {"ant_px": (1439, 758)},
    '2224' : {"ant_px": (799, 744)},
    '2225' : {"ant_px": (1959, 1116)},
    '2226' : {"ant_px": (3207, 364)},
    '2227' : {"ant_px": (2719, 930)},
    '2228' : {"ant_px": (1693, 786)},
    '2229' : {"ant_px": (2759, 706)},
    '2230' : {"ant_px": (3295, 744)},
    '2231' : {"ant_px": (3476, 338)},
    '2232' : {"ant_px": (2318, 454)},
    '2233' : {"ant_px": (3092, 982)},
    '2234' : {"ant_px": (2405, 1161)},
    '2235' : {"ant_px": (2234, 464)},
    '2236' : {"ant_px": (2562, 1208)},
    '2237' : {"ant_px": (1935, 646)},
    '2238' : {"ant_px": (2131, 1032)},
    '2239' : {"ant_px": (2436, 271)},
    '2241' : {"ant_px": (1652, 877)},
    '2242' : {"ant_px": (1917, 483)},
    '2243' : {"ant_px": (2087, 528)},
    '2245' : {"ant_px": (2294, 902)}
}
IMG_KEYS = list(DEFAULT_META.keys())
IMG_GLOB = "/Users/komalkaur/Desktop/eigsep_stuff/eigsep_terrain/2026_imgs"
# DEFAULT_PRMS_U_BY_KEY = {
#     "0817": (1734.11, 2069.00, 1760.97, 1.4706, 3.6932, -0.0493, 9830.11),
#     "0833": (1611.31, 1849.00, 1659.78, 1.2053, 1.2414, -0.0244, 5081.08),
#     "0860": (1541.90, 1998.96, 1765.06, 1.5412, 0.6147, 0.1585, 2328.64),
# }

DEFAULT_PRMS_U_BY_KEY = {
    '2209' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2210' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2211' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2213' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2214' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2215' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2216' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2217' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2218' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2219' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2220' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2221' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2222' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2223' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2224' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2225' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2226' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2227' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2228' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2229' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2230' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2231' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2232' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2233' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2234' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2235' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2236' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2237' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2238' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2239' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2241' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2242' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2243' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0),
    '2245' : (1600.0, 2000.0, 1600.0, 1.5, 1.0, 0.0, 5000.0)
}

SLIDER_RANGES = {  # name -> (delta_minus, delta_plus) around start value
    "e": (-30, 30), "n": (-30, 30), "u": (-15, 15),
    "th": (-0.1, 0.1), "ph": (-0.1, 0.1), "ti": (-0.1, 0.1),
    "f": (-1000, 1000),
}


def find_img_file(which, img_glob):
    key = IMG_KEYS[which]
    files = sorted(glob.glob(img_glob))
    matches = [f for f in files if os.path.basename(f).split("_")[-1].split(".")[0] == key]
    if not matches:
        raise FileNotFoundError(f"No file matching key {key!r} via {img_glob!r}")
    return matches[0]


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--img-glob", default=IMG_GLOB)
    ap.add_argument("--cache-file", default="marjum_dem.npz")
    ap.add_argument("--stride", type=int, default=10,
                    help="Pixel stride for the live ray grid (bigger = faster).")
    ap.add_argument("--fine-delta", type=float, default=0.25)
    ap.add_argument("--n-rays", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=1e-2)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    key = IMG_KEYS[args.which]

    dem = DEM(cache_file=args.cache_file)
    meta = {k: dict(v) for k, v in DEFAULT_META.items()}
    img_file = find_img_file(args.which, args.img_glob)
    img = HorizonImage(img_file, meta, px_smooth=150, px_dist=30)

    start = dict(zip(["e", "n", "u", "th", "ph", "ti", "f"],
                     DEFAULT_PRMS_U_BY_KEY[key]))

    ys = np.arange(0, img.npix_y, args.stride)
    xs = np.arange(0, img.npix_x, args.stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    x_px, y_px = yy.ravel(), xx.ravel()
    actual_sky = img.sky_mask[np.ix_(ys, xs)] > 0.5

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.42)
    ax.imshow(img.img, origin="lower")
    ax.contour(xx, yy, actual_sky.astype(float), levels=[0.5],
               colors="cyan", linewidths=1.5)
    ax.set_title(f"Image {key} — drag sliders, fit horizon updates live")

    contour_holder = {"cs": None}

    def compute_and_draw(prms):
        img.set_prms(tuple(prms[k] for k in ["e", "n", "u", "th", "ph", "ti", "f"]))
        rays = img.get_rays(pixels=(x_px, y_px), dtype=dtype_r)
        r = img.ray_distance(dem, rays, dtype=dtype_r, fine_delta=args.fine_delta)
        model_sky = np.isnan(r).reshape(yy.shape)
        if contour_holder["cs"] is not None:
            contour_holder["cs"].remove()
        contour_holder["cs"] = ax.contour(
            xx, yy, model_sky.astype(float), levels=[0.5],
            colors="red", linewidths=1.5,
        )
        fig.canvas.draw_idle()

    sliders = {}
    reset_btns = {}
    textboxes = {}
    for i, (name, (lo, hi)) in enumerate(SLIDER_RANGES.items()):
        s_ax = plt.axes([0.13, 0.35 - i * 0.045, 0.55, 0.03])
        s0 = start[name]
        sliders[name] = Slider(s_ax, name, s0 + lo, s0 + hi, valinit=s0)
        t_ax = plt.axes([0.71, 0.35 - i * 0.045, 0.08, 0.03])
        textboxes[name] = TextBox(t_ax, "", initial=f"{s0:.4f}")
        r_ax = plt.axes([0.80, 0.35 - i * 0.045, 0.06, 0.03])
        reset_btns[name] = Button(r_ax, "reset")

    def on_change(_):
        prms = {name: s.val for name, s in sliders.items()}
        compute_and_draw(prms)
        logL = img.horizon_ray_logL(dem, n_rays=args.n_rays, eps=args.eps,
                                    fine_delta=args.fine_delta)
        ax.set_title(f"Image {key} — logL={logL:.2f}")
        print("prms_u =", tuple(round(prms[k], 4)
                                for k in ["e", "n", "u", "th", "ph", "ti", "f"]),
              f" logL={logL:.2f}")

    _updating = {"flag": False}

    def on_slider_change(name):
        def cb(val):
            if _updating["flag"]:
                return
            _updating["flag"] = True
            textboxes[name].set_val(f"{val:.4f}")
            _updating["flag"] = False
            on_change(val)
        return cb

    for name, s in sliders.items():
        s.on_changed(on_slider_change(name))

    def on_text_submit(name):
        def cb(text):
            if _updating["flag"]:
                return
            try:
                v = float(text)
            except ValueError:
                return
            v = min(max(v, sliders[name].valmin), sliders[name].valmax)
            _updating["flag"] = True
            sliders[name].set_val(v)
            _updating["flag"] = False
            textboxes[name].set_val(f"{v:.4f}")
            on_change(v)
        return cb

    for name, t in textboxes.items():
        t.on_submit(on_text_submit(name))

    for name, b in reset_btns.items():
        b.on_clicked(lambda event, n=name: sliders[n].reset())

    reset_ax = plt.axes([0.85, 0.90, 0.1, 0.04])
    reset_btn = Button(reset_ax, "Reset")

    def on_reset(_):
        for name, s in sliders.items():
            s.reset()

    reset_btn.on_clicked(on_reset)

    xlim0, ylim0 = ax.get_xlim(), ax.get_ylim()
    reset_zoom_ax = plt.axes([0.85, 0.85, 0.1, 0.04])
    reset_zoom_btn = Button(reset_zoom_ax, "Reset zoom")

    def on_reset_zoom(_):
        ax.set_xlim(xlim0)
        ax.set_ylim(ylim0)
        fig.canvas.draw_idle()

    reset_zoom_btn.on_clicked(on_reset_zoom)

    active = {"name": None}
    for name, s in sliders.items():
        s.ax.figure.canvas.mpl_connect("button_press_event",
            lambda event, n=name, a=s.ax: active.__setitem__("name", n) if event.inaxes == a else None)

    def on_key(event):
        n = active["name"]
        if n is None:
            return
        s = sliders[n]
        step = (s.valmax - s.valmin) / 100
        if event.key == "right":
            s.set_val(min(s.val + step, s.valmax))
        elif event.key == "left":
            s.set_val(max(s.val - step, s.valmin))

    fig.canvas.mpl_connect("key_press_event", on_key)

    def on_scroll(event):
        if event.inaxes != ax or event.xdata is None:
            return
        factor = 0.9 if event.button == "up" else 1.1
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        ax.set_xlim(xd - (xd - xlo) * factor, xd + (xhi - xd) * factor)
        ax.set_ylim(yd - (yd - ylo) * factor, yd + (yhi - yd) * factor)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    compute_and_draw(start)
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())