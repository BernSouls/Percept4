"""U12.0 -- FRESH HELD-OUT SCENES.

Why this module exists
----------------------
The U11 corpus is 8 scenes with 12 renderer-true junctions, and the two scenes
labelled TEST (`box_over_box_test`, `cylinder_over_box_shift`) were looked at
repeatedly while deriving R9 (the 2.06 contact bound) and R10 (the truth
oracle's owner-normal guard).  A scene whose junction statistics have been read
is no longer a held-out set: `box_over_box_test` contributed the 7.07 px / 1.16x
measurement that the contact bound was checked against, and both TEST scenes
were in every tolerance scan.  So from U12 on:

    CAL   -- the old corpus minus its TEST pair, used to DERIVE
    FRESH -- this module, used to VERIFY.  Not read until the rule is frozen.

These are the same four families as the old corpus (flat box over box, the same
layout hazed + blurred, disk over box, shaded cylinder over box) with **new
geometry drawn from explicit seeds** and new grey levels, plus two fresh
no-occlusion controls.  8 occlusion scenes x 2 junctions = 16 held-out truth
junctions, against 12 in the whole old corpus.

Two deliberate choices, recorded so they are not read as omissions:

* **No pixel noise.**  The old corpus is noise-free, and adding seeded noise
  would change the front end's peak population rather than the geometry, so a
  fresh-scene failure could not be attributed.  "New seeds" here means the
  geometry draw, which is what the cue is supposed to generalise over.
* **Junctions exist by construction.**  The sampler constrains the near object to
  straddle the far object's top edge with margin on both sides, so every scene
  carries exactly two renderer-true T-junctions and a zero-junction result is
  always the cue's doing, never the renderer's.

Truth is analytic, from the same geometry the renderer drew with: a box/disk/
cylinder crossing a straight top edge at `y = y0` gives contacts at
`x = nx0`, `nx1` (near box sides), `cx +/- sqrt(r^2 - (y0-cy)^2)` (disk) and
`cx +/- rx` (cylinder limb).
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from scenes import SIZE, _cylinder_shade, _disk_mask, _rect_mask, _to_scene

#: the seeds every fresh geometry is drawn from.  Recorded in the freeze file so
#: a rebuild is bit-identical; changing one of these invalidates the hold-out.
SEEDS = dict(
    f_box_a=1201, f_box_b=1202,
    f_haze_a=1203, f_haze_b=1204,
    f_disk_a=1205, f_disk_b=1206,
    f_cyl_a=1207, f_cyl_b=1208,
    f_ctrl_disk=1209, f_ctrl_box=1210,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _draw(rng, lo: float, hi: float) -> int:
    """An integer parameter in [lo, hi] -- geometry stays on the pixel grid."""
    return int(rng.integers(int(lo), int(hi) + 1))


def _drawf(rng, lo: float, hi: float) -> float:
    return float(rng.uniform(float(lo), float(hi)))


# --------------------------------------------------------------------------
# family: flat box over box (optionally hazed + blurred)
# --------------------------------------------------------------------------

def _box_params(seed: int) -> Dict[str, Any]:
    rng = _rng(seed)
    fx0 = _draw(rng, 16, 34)
    fy0 = _draw(rng, 92, 114)
    fw = _draw(rng, 112, 140)
    fh = _draw(rng, 44, 62)
    fx1, fy1 = fx0 + fw, fy0 + fh
    # the near box straddles the far box's top edge, with margin on both sides
    nx0 = _draw(rng, fx0 + 22, fx0 + 48)
    nw = _draw(rng, 34, 56)
    nx1 = min(nx0 + nw, fx1 - 18)
    ny0 = fy0 - _draw(rng, 36, 56)
    ny1 = fy0 + _draw(rng, 28, 50)
    return dict(far=(fx0, fy0, fx1, fy1), near=(nx0, ny0, nx1, ny1),
                bg=_drawf(rng, 0.22, 0.33), far_val=_drawf(rng, 0.50, 0.64),
                near_val=_drawf(rng, 0.80, 0.92))


def fresh_box_over_box(seed: int, name: str, haze_far: float = 0.0,
                       blur_far: float = 0.0, size: int = SIZE):
    p = _box_params(seed)
    h = w = size
    s = size / float(SIZE)
    fx0, fy0, fx1, fy1 = (int(v * s) for v in p["far"])
    nx0, ny0, nx1, ny1 = (int(v * s) for v in p["near"])
    img = np.full((h, w), p["bg"], dtype=np.float64)
    far_mask = _rect_mask(h, w, fx0, fy0, fx1, fy1)
    near_mask = _rect_mask(h, w, nx0, ny0, nx1, ny1)
    img[far_mask] = p["far_val"]
    img[near_mask] = p["near_val"]
    if blur_far > 0.0:
        from scipy.ndimage import gaussian_filter
        far_only = np.where(far_mask & ~near_mask, img, np.nan)
        filled = np.where(np.isnan(far_only), img, far_only)
        blurred = gaussian_filter(filled, sigma=blur_far * s, mode="nearest")
        region = far_mask & ~near_mask
        img[region] = blurred[region]
    if haze_far > 0.0:
        region = far_mask & ~near_mask
        img[region] = (1.0 - haze_far) * img[region] + haze_far * p["bg"]
    junctions = [
        dict(id="J_left", x=float(nx0), y=float(fy0),
             near="near_box", far="far_box",
             near_contour="near_box_side", far_contour="far_box_top_edge"),
        dict(id="J_right", x=float(nx1), y=float(fy0),
             near="near_box", far="far_box",
             near_contour="near_box_side", far_contour="far_box_top_edge"),
    ]
    sc = _to_scene(name, img, masks=dict(far_box=far_mask, near_box=near_mask),
                   depth=dict(near_box=0, far_box=1, background=2),
                   junctions=junctions,
                   note="FRESH held-out flat box over box (seed %d)" % seed)
    sc.params = dict(p, seed=seed, haze_far=haze_far, blur_far=blur_far,
                     family="box_over_box")
    return sc


# --------------------------------------------------------------------------
# family: disk over box
# --------------------------------------------------------------------------

def _disk_params(seed: int) -> Dict[str, Any]:
    rng = _rng(seed)
    bx0 = _draw(rng, 28, 44)
    ytop = _draw(rng, 94, 112)
    bx1 = _draw(rng, 148, 164)
    by1 = ytop + _draw(rng, 46, 62)
    cx = _draw(rng, 86, 106)
    dy = _drawf(rng, 2.0, 12.0)          # how far the disk's centre is above ytop
    half = _drawf(rng, 18.0, 32.0)       # half-distance between the junctions
    cy = ytop - dy
    r = math.sqrt(half * half + dy * dy)
    return dict(box=(bx0, ytop, bx1, by1), cx=float(cx), cy=float(cy),
                r=float(r), half=float(half), dy=float(dy),
                bg=_drawf(rng, 0.22, 0.33), box_val=_drawf(rng, 0.52, 0.66),
                disk_val=_drawf(rng, 0.82, 0.93))


def fresh_disk_on_box(seed: int, name: str, size: int = SIZE):
    p = _disk_params(seed)
    h = w = size
    s = size / float(SIZE)
    bx0, ytop, bx1, by1 = (int(v * s) for v in p["box"])
    img = np.full((h, w), p["bg"], dtype=np.float64)
    box_mask = _rect_mask(h, w, bx0, ytop, bx1, by1)
    img[box_mask] = p["box_val"]
    cx, cy, r = p["cx"] * s, p["cy"] * s, p["r"] * s
    disk_mask = _disk_mask(h, w, cx, cy, r)
    img[disk_mask] = p["disk_val"]
    yj = float(ytop)
    half = math.sqrt(max(0.0, r * r - (yj - cy) ** 2))
    junctions = [
        dict(id="J_left", x=cx - half, y=yj, near="disk", far="box",
             near_contour="disk_silhouette", far_contour="box_top_edge"),
        dict(id="J_right", x=cx + half, y=yj, near="disk", far="box",
             near_contour="disk_silhouette", far_contour="box_top_edge"),
    ]
    sc = _to_scene(name, img, masks=dict(box=box_mask, disk=disk_mask),
                   depth=dict(disk=0, box=1, background=2), junctions=junctions,
                   note="FRESH held-out disk over box (seed %d)" % seed)
    sc.params = dict(p, seed=seed, family="disk_on_box")
    return sc


# --------------------------------------------------------------------------
# family: shaded cylinder over box
# --------------------------------------------------------------------------

def _cyl_params(seed: int) -> Dict[str, Any]:
    rng = _rng(seed)
    bx0 = _draw(rng, 28, 44)
    ytop = _draw(rng, 96, 114)
    bx1 = _draw(rng, 148, 164)
    by1 = ytop + _draw(rng, 44, 60)
    cx = float(_draw(rng, 86, 106))
    rx = _drawf(rng, 20.0, 31.0)
    # the cylinder must span the box's top edge with margin above and below
    cy = ytop - _drawf(rng, 26.0, 44.0)
    ry = (ytop - cy) + _drawf(rng, 16.0, 30.0)
    return dict(box=(bx0, ytop, bx1, by1), cx=cx, rx=float(rx), cy=float(cy),
                ry=float(ry), bg=_drawf(rng, 0.24, 0.34),
                box_val=_drawf(rng, 0.54, 0.68), base=_drawf(rng, 0.88, 0.98))


def fresh_cylinder_over_box(seed: int, name: str, haze_far: float = 0.0,
                            size: int = SIZE):
    p = _cyl_params(seed)
    h = w = size
    s = size / float(SIZE)
    bx0, ytop, bx1, by1 = (int(v * s) for v in p["box"])
    img = np.full((h, w), p["bg"], dtype=np.float64)
    box_mask = _rect_mask(h, w, bx0, ytop, bx1, by1)
    img[box_mask] = p["box_val"]
    cx, rx = p["cx"] * s, p["rx"] * s
    cy, ry = p["cy"] * s, p["ry"] * s
    cyl_img, cyl_mask = _cylinder_shade(h, w, cx, cy, rx, ry, base=p["base"])
    img[cyl_mask] = cyl_img[cyl_mask]
    if haze_far > 0.0:
        img[box_mask & ~cyl_mask] = ((1.0 - haze_far) * p["box_val"]
                                     + haze_far * p["bg"])
    yj = float(ytop)
    junctions = [
        dict(id="J_left", x=cx - rx, y=yj, near="cylinder", far="box",
             near_contour="cylinder_silhouette", far_contour="box_top_edge"),
        dict(id="J_right", x=cx + rx, y=yj, near="cylinder", far="box",
             near_contour="cylinder_silhouette", far_contour="box_top_edge"),
    ]
    sc = _to_scene(name, img,
                   masks=dict(box=box_mask, cylinder=cyl_mask),
                   depth=dict(cylinder=0, box=1, background=2),
                   junctions=junctions,
                   note="FRESH held-out cylinder over box (seed %d)" % seed)
    sc.params = dict(p, seed=seed, haze_far=haze_far,
                     family="cylinder_over_box")
    return sc


# --------------------------------------------------------------------------
# fresh controls: no occluder, no junction, must stay silent
# --------------------------------------------------------------------------

def fresh_single_disk(seed: int, name: str, size: int = SIZE):
    rng = _rng(seed)
    h = w = size
    s = size / float(SIZE)
    cx, cy = float(_draw(rng, 84, 108)), float(_draw(rng, 84, 108))
    r = _drawf(rng, 42.0, 56.0)
    bg = _drawf(rng, 0.22, 0.34)
    val = _drawf(rng, 0.74, 0.90)
    img = np.full((h, w), bg, dtype=np.float64)
    mask = _disk_mask(h, w, cx * s, cy * s, r * s)
    img[mask] = val
    sc = _to_scene(name, img, masks=dict(disk=mask),
                   depth=dict(disk=0, background=1), junctions=[],
                   note="FRESH held-out control: one closed contour (seed %d)"
                        % seed)
    sc.params = dict(cx=cx, cy=cy, r=r, bg=bg, val=val, seed=seed,
                     family="control_disk")
    return sc


def fresh_single_box(seed: int, name: str, size: int = SIZE):
    rng = _rng(seed)
    h = w = size
    s = size / float(SIZE)
    x0 = _draw(rng, 38, 56)
    y0 = _draw(rng, 46, 64)
    x1 = x0 + _draw(rng, 84, 104)
    y1 = y0 + _draw(rng, 74, 92)
    bg = _drawf(rng, 0.22, 0.34)
    val = _drawf(rng, 0.70, 0.86)
    img = np.full((h, w), bg, dtype=np.float64)
    mask = _rect_mask(h, w, int(x0 * s), int(y0 * s), int(x1 * s), int(y1 * s))
    img[mask] = val
    sc = _to_scene(name, img, masks=dict(box=mask),
                   depth=dict(box=0, background=1), junctions=[],
                   note="FRESH held-out control: one rectangle (seed %d)" % seed)
    sc.params = dict(box=(x0, y0, x1, y1), bg=bg, val=val, seed=seed,
                     family="control_box")
    return sc


# --------------------------------------------------------------------------
# the frozen list
# --------------------------------------------------------------------------

#: name -> builder.  Order is the reporting order: families, then controls.
BUILDERS: Dict[str, Callable[[], Any]] = {
    "f_box_a":   lambda: fresh_box_over_box(SEEDS["f_box_a"], "f_box_a"),
    "f_box_b":   lambda: fresh_box_over_box(SEEDS["f_box_b"], "f_box_b"),
    "f_haze_a":  lambda: fresh_box_over_box(SEEDS["f_haze_a"], "f_haze_a",
                                            haze_far=0.42, blur_far=1.1),
    "f_haze_b":  lambda: fresh_box_over_box(SEEDS["f_haze_b"], "f_haze_b",
                                            haze_far=0.55, blur_far=1.6),
    "f_disk_a":  lambda: fresh_disk_on_box(SEEDS["f_disk_a"], "f_disk_a"),
    "f_disk_b":  lambda: fresh_disk_on_box(SEEDS["f_disk_b"], "f_disk_b"),
    "f_cyl_a":   lambda: fresh_cylinder_over_box(SEEDS["f_cyl_a"], "f_cyl_a"),
    "f_cyl_b":   lambda: fresh_cylinder_over_box(SEEDS["f_cyl_b"], "f_cyl_b",
                                                 haze_far=0.35),
    "f_ctrl_disk": lambda: fresh_single_disk(SEEDS["f_ctrl_disk"], "f_ctrl_disk"),
    "f_ctrl_box":  lambda: fresh_single_box(SEEDS["f_ctrl_box"], "f_ctrl_box"),
}

FRESH_SCENES: Tuple[str, ...] = tuple(BUILDERS)
FRESH_OCCLUSION: Tuple[str, ...] = tuple(
    n for n in FRESH_SCENES if not n.startswith("f_ctrl"))
FRESH_CONTROL: Tuple[str, ...] = tuple(
    n for n in FRESH_SCENES if n.startswith("f_ctrl"))


def build_fresh(name: str, size: int = SIZE):
    return BUILDERS[name]()


def build_fresh_all(size: int = SIZE) -> Dict[str, Any]:
    return {n: BUILDERS[n]() for n in FRESH_SCENES}


def freeze_record(size: int = SIZE) -> Dict[str, Any]:
    """The frozen hold-out record: geometry, rendering and TRUTH ONLY.

    Deliberately contains no pipeline output of any kind -- no peaks, chains,
    sites or edges -- because U12.0 requires the fresh scenes' junction
    statistics to stay unread until the rules under test are frozen.  Everything
    here is a renderer INPUT or a renderer TRUTH.
    """
    out: Dict[str, Any] = dict(
        purpose="U12.0 fresh held-out scenes: verify, never derive",
        renderer=dict(cylinder_silhouette=(
            "B37 fixed.  The silhouette mask is |x - cx| <= rx over the row "
            "band |y - cy| <= ry.  Before the fix a clipped normal coordinate "
            "made the mask span the full frame width, its paint overwrote the "
            "far object's contour inside that band, and truth attribution "
            "credited every fragment in the band to the cylinder.  The seeds, "
            "the drawn parameters and the junction geometry are UNCHANGED by "
            "the fix; only the rendered pixels and the masks are.  Re-frozen "
            "before any fresh-scene rule was re-derived, and the reason is a "
            "renderer defect found by the stage census, not a statistic.")),
        size=size, seeds=dict(SEEDS), scenes={},
    )
    total = 0
    for n in FRESH_SCENES:
        sc = BUILDERS[n]()
        prm = {k: (list(v) if isinstance(v, tuple) else v)
               for k, v in getattr(sc, "params", {}).items()}
        lum = np.asarray(sc.raw_luminance, dtype=np.float64)
        out["scenes"][n] = dict(
            family=prm.get("family"), seed=prm.get("seed"), params=prm,
            n_true_junctions=len(sc.junctions),
            junctions=[dict(id=j["id"], x=round(float(j["x"]), 4),
                            y=round(float(j["y"]), 4), near=j["near"],
                            far=j["far"]) for j in sc.junctions],
            objects=sorted(sc.masks), depth=dict(sc.depth),
            img_min=round(float(lum.min()), 6),
            img_max=round(float(lum.max()), 6),
            img_mean=round(float(lum.mean()), 6),
            note=sc.note,
        )
        total += len(sc.junctions)
    out["n_scenes"] = len(FRESH_SCENES)
    out["n_occlusion_scenes"] = len(FRESH_OCCLUSION)
    out["n_control_scenes"] = len(FRESH_CONTROL)
    out["n_true_junctions"] = total
    body = json.dumps(out, sort_keys=True, default=str)
    out["_sha256_of_record"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return out


if __name__ == "__main__":
    import os
    import sys
    rec = freeze_record()
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "diagnostics", "_u12_fresh_scenes.json")
    dest = os.path.normpath(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True, default=str)
    print("FROZEN %d fresh scenes, %d true junctions -> %s"
          % (rec["n_scenes"], rec["n_true_junctions"], dest))
    print("sha256(record) = %s" % rec["_sha256_of_record"])
    for n in FRESH_SCENES:
        s = rec["scenes"][n]
        print("  %-12s %-18s seed=%-5d junctions=%d  %s"
              % (n, s["family"], s["seed"], s["n_true_junctions"],
                 [(j["id"], j["x"], j["y"]) for j in s["junctions"]]))
