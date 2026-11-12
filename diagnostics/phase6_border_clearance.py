"""[STAGE 6 / 0b, direction 2] How much border clearance does each spatial-
frequency channel need?

The hard border discard of perception/contour_field.py removes every peak
within k // 2 px of the frame, k being the LARGEST kernel of the bank (56 px
on a 256 px image).  This script measures, before anything is changed:

 1. per channel, the half-support its OWN kernel has (bank geometry);
 2. per channel, the clearance it NEEDS: on a 144 x 144 window cut out of the
    middle of a 256 px image, the distance from the window's border beyond
    which the channel's response agrees with the response computed on the
    whole image (true surround), relative error < 5 % of the channel's
    maximum.  Three stages: raw complex energy (kernel only), normalised
    complex (divisive pool), integration (association field; what the peak
    extraction reads).  Same bank for both (scale forced);
 3. the share of the silhouette perimeter inside the interior under the
    per-channel rule.

    python diagnostics/phase6_border_clearance.py [--out NAME]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import parameters as P  # noqa: E402
import eth80_frontend as fe  # noqa: E402
import eth80_split as sp  # noqa: E402
import perception.contour_field as ctf  # noqa: E402
from perception.observation_2d import Observation2D  # noqa: E402
from phase6_closure import perimeter_pixels, silhouette  # noqa: E402

TOL = 0.05
WIN0, WIN1 = 56, 200          # the window: every pixel is clear of the full frame's band


def own_half_supports(scale: float):
    """Half the size of each channel's OWN kernel (before it is zero-padded to
    the largest one), from the same formula GaborBank.build_default uses."""
    s = max(float(scale), 1e-6)
    sfs = np.geomspace(P.GABOR_SF_MIN / s, P.GABOR_SF_MAX / s, P.GABOR_N_SF)
    out = []
    for sf in sfs:
        sx = P.GABOR_SIGMA_X_FRAC / float(sf)
        sy = sx / P.GABOR_ASPECT
        sig = max(sx, sy) if P.GABOR_KERNEL_SIZE_FROM_MAX_SIGMA else sx
        size = int(2 * math.ceil(3.0 * sig)) + 1
        cap = int(round(float(P.GABOR_KERNEL_MAX) * max(1.0, s)))
        cap += (cap % 2 == 0)
        size = min(size, cap)
        size += (size % 2 == 0)
        out.append((float(sf), float(sx), float(sy), int(size), int(size // 2)))
    return out


def border_distance(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.minimum(np.minimum(yy, h - 1 - yy), np.minimum(xx, w - 1 - xx))


def clearance(win: np.ndarray, full: np.ndarray, dist: np.ndarray):
    """Per channel: smallest d* with rel. error < TOL at every distance >= d*."""
    n_sf = win.shape[1]
    dmax = int(dist.max())
    out, prof = [], []
    for c in range(n_sf):
        ref = float(np.abs(full[:, c]).max()) or 1.0
        err = np.abs(win[:, c] - full[:, c]).max(axis=0) / ref
        e_d = np.array([float(err[dist == d].max()) for d in range(dmax + 1)])
        bad = np.nonzero(e_d >= TOL)[0]
        out.append(int(bad.max()) + 1 if bad.size else 0)
        prof.append([round(float(v), 4) for v in e_d])
    return out, prof


def stages(lum: np.ndarray, scale: float):
    f = ctf.field_from_luminance(lum, scale=scale)
    _simple, raw, _phase = ctf.v1_responses(lum, scale=scale)
    return {"raw complex": np.asarray(raw), "normalised complex": np.asarray(f["complex"]),
            "integration": np.asarray(f["integration"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase6_border_clearance")
    ap.add_argument("--per-class", type=int, default=1)
    a = ap.parse_args(argv)
    split = sp.load_split()
    sample = list(split["references"].items())
    for c, ps in split["eval"].items():
        sample += [(c, p) for p in ps[: a.per_class]]
    every = list(split["references"].items()) + [
        (c, p) for c, ps in split["eval"].items() for p in ps]

    scale = ctf.front_end_scale(256, 256)
    own = own_half_supports(scale)
    lines = ["[STAGE 6 / 0b-2] border clearance per spatial-frequency channel", "",
             "1. the bank (256 px frame, scale %.3f): the discard in force is %d px for every channel"
             % (scale, ctf._interior_margin(ctf._border_margin_px(ctf._get_bank(scale)), 256, 256)),
             "   channel   sf (c/px)  wavelength  sigma_x  sigma_y  own kernel  own half-support"]
    for i, (sf, sx, sy, size, half) in enumerate(own):
        lines.append("   %d         %.4f     %5.1f px    %5.2f    %5.2f    %3d px      %2d px" % (
            i, sf, 1.0 / sf, sx, sy, size, half))

    dist = border_distance(WIN1 - WIN0, WIN1 - WIN0)
    need, profiles = {}, {}
    for cls, rel in sample:
        img = fe._load_rgb(fe.image_path(cls, rel))
        obs = Observation2D.from_image(img, rich=False)
        lum = np.asarray(obs.get_luminance(), dtype=np.float32)
        if lum.ndim == 3:
            from perception.filters import luminance_adapt, opponent_channels
            lum = luminance_adapt(opponent_channels(lum.astype(np.float32))[..., 0]).astype(np.float32)
        full = stages(lum, scale)
        win = stages(np.ascontiguousarray(lum[WIN0:WIN1, WIN0:WIN1]), scale)
        for name in full:
            d, prof = clearance(win[name], full[name][:, :, WIN0:WIN1, WIN0:WIN1], dist)
            need.setdefault(name, []).append(d)
            profiles.setdefault(name, []).append(prof)
    lines += ["", "2. the clearance each channel NEEDS (px from the border; relative error < %.0f %% of the"
              % (100 * TOL),
              "   channel's maximum at every larger distance; window 144 x 144 with reflect padding",
              "   against the same pixels computed inside the whole 256 px image; %d images:"
              % len(sample),
              "   median / maximum over images)",
              "   stage                 " + "".join("  ch%d      " % i for i in range(len(own)))]
    for name, rows_ in need.items():
        arr = np.asarray(rows_)
        lines.append("   %-20s  " % name + "".join(
            "  %2.0f / %2d  " % (np.median(arr[:, c]), arr[:, c].max()) for c in range(arr.shape[1])))
    lines.append("   own half-support      " + "".join("  %2d       " % h for *_x, h in own))
    lines.append("   (the window is 72 px deep: a value of 72 means 'never clears inside this window')")

    margins_own = [max(int(ctf._PEAK_BORDER_MARGIN_PX), h) for *_x, h in own]
    rows = []
    for cls, rel in every:
        img = fe._load_rgb(fe.image_path(cls, rel))
        px, py = perimeter_pixels(silhouette(img))
        H, W = img.shape[:2]
        row = {}
        for m in sorted(set(margins_own + [56])):
            inside = (px >= m) & (px < W - m) & (py >= m) & (py < H - m)
            row[m] = float(inside.mean()) if len(px) else float("nan")
        rows.append(row)
    lines += ["", "3. share of the silhouette perimeter inside the interior, 79 graphs",
              "   (per-channel rule: margin = max(%d px floor, the channel's own half-support))"
              % ctf._PEAK_BORDER_MARGIN_PX,
              "   margin   channels it applies to        median   p10    min    whole silhouette inside"]
    for m in sorted(set(margins_own + [56]), reverse=True):
        v = np.array([r[m] for r in rows])
        chans = [i for i, mm in enumerate(margins_own) if mm == m]
        label = ("ch " + ",".join(str(c) for c in chans)) if chans else "-"
        if m == 56:
            label += "  (today: every channel)"
        lines.append("   %2d px    %-28s  %.2f     %.2f   %.2f   %d of %d" % (
            m, label, np.median(v), np.percentile(v, 10), v.min(), int((v >= 0.999).sum()), len(v)))
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w") as fh:
        json.dump({"own": own, "need": need, "margins_own": margins_own,
                   "profiles": profiles, "visibility": rows}, fh, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
