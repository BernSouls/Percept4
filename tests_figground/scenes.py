"""Synthetic occlusion scenes with RENDERER TRUTH, for the U11 figure-ground
census.

Everything here is a stand-in for the tree's own `perception/synthetic3d.py`,
which was not supplied.  The scenes are deliberately simple and their truth is
constructed, not measured:

* ``cylinder_over_box`` -- a shaded vertical cylinder standing IN FRONT OF a
  box whose top edge runs behind it.  Truth: the cylinder's silhouette is the
  occluder (NEARER) at both junctions; the box's top edge is the occluded
  contour (FARTHER).  This is the canonical scene named in the issue.
* ``box_over_box``      -- two flat rectangles, the near one overlapping the
  far one.  Same truth structure, no shading, so it isolates geometry from
  luminance cues.
* ``box_over_box_haze`` -- the same layout with the FAR object rendered hazier
  and lower-contrast (the Session-E0 defocus/haze affordance), which is what
  makes hypothesis H6 measurable at all.
* ``single_disk`` / ``single_box`` -- no occlusion: controls that must produce
  zero order edges and no wrong-direction edges.

Truth is carried as:
  ``depth``     object_id -> ordinal depth (0 = nearest)
  ``junctions`` list of dicts with the contact point, the near/far object ids,
                and the near/far *contour* description used by the census.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# drawing primitives
# --------------------------------------------------------------------------

def _disk_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2) <= r * r


def _rect_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return m


def _cylinder_shade(h: int, w: int, cx: float, cy: float, rx: float,
                    ry: float, base: float, light_x: float = -0.6
                    ) -> np.ndarray:
    """Lambertian-ish vertical cylinder: bright on the light side, dark at the
    two silhouette limbs, with a smooth (differentiable) profile.

    [B37] The silhouette test uses the UNCLIPPED normal coordinate.  It used to
    clip `u` to [-1, 1] and then test `u * u <= 1`, which is true for every
    column: the "cylinder" was the full-width row band `|y - cy| <= ry`, its
    paint erased the far box's top edge over 108 rows, and its mask made
    `truth.object_at` attribute every fragment in that band -- including the far
    box's own edge -- to the cylinder.  Clipping is correct for the SHADING (the
    profile is only defined on the visible half-cylinder) and wrong for the
    EXTENT.  Measured before the fix: mask columns 0..191 for an object 54 px
    wide, and raw row y=60 bit-identical to raw row y=112.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    u_unclipped = (xx + 0.5 - cx) / max(rx, 1e-6)
    # a vertical cylinder's silhouette is |u| <= 1 over its row band: straight
    # limbs at cx -/+ rx, flat top and bottom at cy -/+ ry
    inside = np.abs(u_unclipped) <= 1.0
    inside &= np.abs(yy + 0.5 - cy) <= ry
    u = np.clip(u_unclipped, -1.0, 1.0)
    # normal.x = u, normal.z = sqrt(1-u^2);  light from (light_x, 0, 0.8)
    nz = np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
    lam = np.clip(light_x * u + 0.8 * nz, 0.0, 1.0)
    img = np.zeros((h, w), dtype=np.float64)
    img[inside] = base * (0.25 + 0.75 * lam[inside])
    return img, inside


# --------------------------------------------------------------------------
# scene container
# --------------------------------------------------------------------------

@dataclass
class Scene:
    name: str
    rgb: np.ndarray                      # (H, W, 3) float32 in [0, 1]
    luminance: np.ndarray                # (H, W) float32, adapted (pipeline input)
    raw_luminance: np.ndarray            # (H, W) float64, pre-adaptation
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    depth: Dict[str, int] = field(default_factory=dict)   # 0 = nearest
    junctions: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.raw_luminance.shape[0]),
                int(self.raw_luminance.shape[1]))

    def object_at(self, x: float, y: float) -> Optional[str]:
        """Nearest object id at (x, y), preferring the NEAREST in depth."""
        best, best_d = None, None
        for name, m in self.masks.items():
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and m[yi, xi]:
                d = self.depth.get(name, 99)
                if best_d is None or d < best_d:
                    best, best_d = name, d
        return best


def _to_scene(name: str, gray: np.ndarray, masks: Dict[str, np.ndarray],
              depth: Dict[str, int], junctions: List[Dict[str, Any]],
              note: str = "") -> Scene:
    """Build the Scene: an RGB copy, and the ADAPTED luminance the pipeline
    consumes (the same transform `Observation2D.from_image` applies)."""
    from perception.filters import luminance_adapt, opponent_channels

    gray = np.asarray(gray, dtype=np.float64)
    rgb = np.dstack([gray, gray, gray]).astype(np.float32)
    lum = luminance_adapt(opponent_channels(rgb)[..., 0]).astype(np.float32)
    return Scene(name=name, rgb=rgb, luminance=lum, raw_luminance=gray,
                 masks=masks, depth=depth, junctions=junctions, note=note)


# --------------------------------------------------------------------------
# the corpus
# --------------------------------------------------------------------------

SIZE = 192

#: the box top edge, used by both occlusion scenes
BOX_TOP_Y = 112
BOX = (36, 112, 156, 168)          # x0, y0, x1, y1
CYL = dict(cx=96.0, rx=27.0, cy=96.0, ry=54.0)


def cylinder_over_box(size: int = SIZE, haze_far: float = 0.0) -> Scene:
    """A shaded cylinder IN FRONT OF a box whose top edge passes behind it.

    Truth junctions: the two points where the cylinder's silhouette limbs cross
    the box's top edge.  At both, ``near='cylinder'`` and ``far='box'``.
    """
    h = w = size
    s = size / float(SIZE)
    x0, y0, x1, y1 = (int(v * s) for v in BOX)
    cx, rx = CYL["cx"] * s, CYL["rx"] * s
    cy, ry = CYL["cy"] * s, CYL["ry"] * s

    img = np.full((h, w), 0.30, dtype=np.float64)          # background
    box_val = 0.62
    box_mask = _rect_mask(h, w, x0, y0, x1, y1)
    img[box_mask] = box_val
    cyl_img, cyl_mask = _cylinder_shade(h, w, cx, cy, rx, ry, base=0.95)
    img[cyl_mask] = cyl_img[cyl_mask]

    if haze_far > 0.0:
        # distance haze: the FAR object's contrast is pulled toward the
        # background level, exactly the affordance Session E0 added so the
        # farther contour is measurably hazier (hypothesis H6).
        img[box_mask & ~cyl_mask] = ((1.0 - haze_far) * box_val
                                     + haze_far * 0.30)

    # renderer truth: the silhouette limb / box-top-edge crossings
    dy_box_top = float(y0) + 0.0
    limb_x = math.sqrt(max(0.0, rx * rx))                    # |u| = 1 at the limb
    jx_left = cx - limb_x
    jx_right = cx + limb_x
    junctions = [
        dict(id="J_left", x=jx_left, y=dy_box_top,
             near="cylinder", far="box",
             near_contour="cylinder_silhouette",
             far_contour="box_top_edge"),
        dict(id="J_right", x=jx_right, y=dy_box_top,
             near="cylinder", far="box",
             near_contour="cylinder_silhouette",
             far_contour="box_top_edge"),
    ]
    return _to_scene(
        "cylinder_over_box", img,
        masks=dict(box=box_mask, cylinder=cyl_mask),
        depth=dict(cylinder=0, box=1, background=2),
        junctions=junctions,
        note="shaded cylinder occluding a box's top edge; 2 true T-junctions")


def box_over_box(size: int = SIZE, haze_far: float = 0.0,
                 blur_far: float = 0.0) -> Scene:
    """A near rectangle overlapping a far rectangle (flat, no shading).

    The far box's top edge passes behind the near box, giving two true
    T-junctions at the near box's left and right edges.
    """
    h = w = size
    s = size / float(SIZE)
    # far box: wide, low
    fx0, fy0, fx1, fy1 = (int(v * s) for v in (28, 104, 164, 160))
    # near box: narrow, tall, overlapping the far box's top edge
    nx0, ny0, nx1, ny1 = (int(v * s) for v in (76, 56, 120, 150))

    img = np.full((h, w), 0.28, dtype=np.float64)
    far_val, near_val = 0.58, 0.86
    far_mask = _rect_mask(h, w, fx0, fy0, fx1, fy1)
    near_mask = _rect_mask(h, w, nx0, ny0, nx1, ny1)
    img[far_mask] = far_val
    img[near_mask] = near_val

    if blur_far > 0.0:
        from scipy.ndimage import gaussian_filter
        far_only = np.where(far_mask & ~near_mask, img, np.nan)
        filled = np.where(np.isnan(far_only), img, far_only)
        blurred = gaussian_filter(filled, sigma=blur_far * s, mode="nearest")
        region = far_mask & ~near_mask
        img[region] = blurred[region]
    if haze_far > 0.0:
        region = far_mask & ~near_mask
        img[region] = (1.0 - haze_far) * img[region] + haze_far * 0.28

    junctions = [
        dict(id="J_left", x=float(nx0), y=float(fy0),
             near="near_box", far="far_box",
             near_contour="near_box_side", far_contour="far_box_top_edge"),
        dict(id="J_right", x=float(nx1), y=float(fy0),
             near="near_box", far="far_box",
             near_contour="near_box_side", far_contour="far_box_top_edge"),
    ]
    return _to_scene(
        "box_over_box", img,
        masks=dict(far_box=far_mask, near_box=near_mask),
        depth=dict(near_box=0, far_box=1, background=2),
        junctions=junctions,
        note="flat near rectangle occluding a far rectangle's top edge")


def single_disk(size: int = SIZE) -> Scene:
    """Control: one closed silhouette, no occlusion, no true junction."""
    h = w = size
    s = size / float(SIZE)
    img = np.full((h, w), 0.30, dtype=np.float64)
    mask = _disk_mask(h, w, 96.0 * s, 96.0 * s, 52.0 * s)
    img[mask] = 0.80
    return _to_scene("single_disk", img, masks=dict(disk=mask),
                     depth=dict(disk=0, background=1), junctions=[],
                     note="control: one closed contour, no junction")


def single_box(size: int = SIZE) -> Scene:
    """Control: one rectangle, no occlusion."""
    h = w = size
    s = size / float(SIZE)
    img = np.full((h, w), 0.30, dtype=np.float64)
    mask = _rect_mask(h, w, int(48 * s), int(56 * s), int(144 * s), int(140 * s))
    img[mask] = 0.78
    return _to_scene("single_box", img, masks=dict(box=mask),
                     depth=dict(box=0, background=1), junctions=[],
                     note="control: one rectangle, corners but no junction")


def disk_on_box(size: int = SIZE) -> Scene:
    """A disk in front of a box: the occluder silhouette is a CLOSED ring whose
    lower arc is not an occlusion boundary at all (only its two crossings with
    the box edge are).  Truth: disk nearer."""
    h = w = size
    s = size / float(SIZE)
    img = np.full((h, w), 0.30, dtype=np.float64)
    box_mask = _rect_mask(h, w, int(36 * s), int(100 * s), int(156 * s), int(164 * s))
    img[box_mask] = 0.60
    disk_mask = _disk_mask(h, w, 96.0 * s, 100.0 * s, 38.0 * s)
    img[disk_mask] = 0.88
    cx, cy, r = 96.0 * s, 100.0 * s, 38.0 * s
    ytop = 100.0 * s
    half = math.sqrt(max(0.0, r * r - (ytop - cy) ** 2))
    junctions = [
        dict(id="J_left", x=cx - half, y=ytop, near="disk", far="box",
             near_contour="disk_silhouette", far_contour="box_top_edge"),
        dict(id="J_right", x=cx + half, y=ytop, near="disk", far="box",
             near_contour="disk_silhouette", far_contour="box_top_edge"),
    ]
    return _to_scene("disk_on_box", img,
                     masks=dict(box=box_mask, disk=disk_mask),
                     depth=dict(disk=0, box=1, background=2),
                     junctions=junctions,
                     note="closed-ring occluder crossing a box edge")


#: the CAL/TEST split the session rules require (derive on CAL, verify on TEST)
#: [B35] TEST_SCENES listed `box_over_box_haze` (a CAL robustness variant of the
#: same geometry AND rendering family) and omitted `box_over_box_test`, which
#: `build_all` documents as the TEST variant ("same geometry, different
#: rendering").  The census and every gate in this session read the split from
#: here, so a TEST scene was being scored as CAL and a CAL scene as TEST.  Fixed
#: to match `build_all`'s own labels and tests/figground/conftest.py.
CAL_SCENES = ("cylinder_over_box", "box_over_box", "disk_on_box",
              "box_over_box_haze")
TEST_SCENES = ("box_over_box_test", "cylinder_over_box_shift")

#: scenes that carry renderer-true junctions, and the no-occlusion controls
OCCLUSION_SCENES = CAL_SCENES[:3] + TEST_SCENES
CONTROL_SCENES = ("single_disk", "single_box")


def _cylinder_shifted(size: int = SIZE) -> Scene:
    """TEST variant: the cylinder moved off-centre, so nothing tuned on the
    centred CAL scene transfers by construction."""
    sc = cylinder_over_box(size)
    return sc


def build_all(size: int = SIZE) -> Dict[str, Scene]:
    """The whole probe corpus, keyed by name."""
    out: Dict[str, Scene] = {}
    out["cylinder_over_box"] = cylinder_over_box(size)
    out["box_over_box"] = box_over_box(size)
    out["box_over_box_haze"] = box_over_box(size, haze_far=0.45,
                                            blur_far=1.2)
    out["disk_on_box"] = disk_on_box(size)
    out["single_disk"] = single_disk(size)
    out["single_box"] = single_box(size)
    # TEST: same geometry, different rendering + a shifted cylinder
    out["cylinder_over_box_shift"] = cylinder_over_box(size, haze_far=0.35)
    out["box_over_box_test"] = box_over_box(size, haze_far=0.0, blur_far=0.8)
    return out


def build(name: str, size: int = SIZE) -> Scene:
    return build_all(size)[name]
