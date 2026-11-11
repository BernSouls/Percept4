"""B37 -- the RENDERER must draw the stimulus its own truth describes.

Found by the U12.2 stage census, which walks the near object's boundary through
each truth junction and found NO boundary pixels near the cylinder junctions:
`scenes._cylinder_shade` built its silhouette mask from a CLIPPED normal
coordinate,

    u = np.clip((xx + 0.5 - cx) / rx, -1.0, 1.0)
    inside = (u * u + v * v * 0.0) <= 1.0        # u is clipped -> u*u <= 1 ALWAYS

so `inside` was the full-width row band `|y - cy| <= ry` rather than the
cylinder's silhouette.  Three consequences, all measured before the fix:

  * `masks["cylinder"]` spanned x = 0..191 (97 columns wider than the object),
    so `truth.object_at` attributed EVERY fragment in the band -- including the
    far box's own top edge -- to the cylinder, the nearest object.  Any
    correct/wrong verdict on a cylinder scene was scored against a fabricated
    ownership map;
  * `img[cyl_mask] = cyl_img[cyl_mask]` painted the clipped-limb shading over
    the whole band, so the far box's top edge -- the contour the T-junction cue
    exists to read -- was NOT IN THE IMAGE AT ALL between y = 42 and y = 149
    (raw row y=60 and raw row y=112 were bit-identical), and the box's side
    edges were erased over the same rows;
  * what remained at x = cx -/+ rx was not a silhouette limb but the KINK where
    the clipped plateau meets the unclipped shading ramp.

So the earlier reading of the cylinder scenes ("the limb has no peaks where it
crosses the bar, therefore the pixels contradict the renderer's depth order")
was measuring a stimulus that did not contain the junction it claimed.  Those
conclusions are void and were re-derived on the fixed renderer in this session.

The tests here are renderer-truth only -- masks and raw luminance, no pipeline
stage is imported -- and they run over BOTH corpora (the old scenes share the
same helper).  Two properties, neither with an invented constant:

  A. the occluder's silhouette is BOUNDED: no near material 6 px outside the
     junction along the bar.  The full-width band fails this at every cylinder
     junction and passes nowhere else;
  B. the far contour the junction interrupts is VISIBLE IN THE PIXELS: the raw
     luminance step across the far object's top edge, 8 px outside the junction,
     is at least half the strongest step anywhere along that same edge.  A
     painted-over edge has step 0.000 against a reference of 0.320.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

import scenes as OLD
import fresh_scenes as F
import sealed_scenes as Z
import u14_sealed_scenes as U14S
from scenes import CAL_SCENES, TEST_SCENES

#: (corpus, name) for every occlusion scene in the two DEVELOPMENT corpora
OCCLUSION: List[Tuple[str, str]] = (
    [("old", n) for n in tuple(CAL_SCENES) + tuple(TEST_SCENES)
     if n not in ("single_disk", "single_box")]
    + [("fresh", n) for n in F.FRESH_OCCLUSION])

#: [U13.0] plus the SEALED set.  The renderer self-consistency gate runs over
#: the sealed corpus from the moment it is sealed -- B37 taught that a stimulus
#: which does not draw what its truth describes invalidates every measurement
#: scored on it, and finding that out AFTER the final measurement would void the
#: verdict.  These tests read masks and raw luminance only, never a pipeline
#: stage, so gating the sealed set does not un-seal it.
ALL_OCCLUSION: List[Tuple[str, str]] = (
    OCCLUSION
    + [("sealed", n) for n in Z.SEALED_OCCLUSION]
    # [U14 phase 4] the NEW sealed set, gated from the moment it is sealed.
    + [("u14_sealed", n) for n in U14S.SEALED_OCCLUSION])


def _scene(corpus: str, name: str):
    if corpus == "fresh":
        return F.build_fresh(name)
    if corpus == "sealed":
        return Z.build_sealed(name)
    if corpus == "u14_sealed":
        return U14S.build_u14_sealed(name)
    return OLD.build_all()[name]


def _inward(sc, j: Dict[str, Any]) -> float:
    """+1 if the occluder's interior lies to the RIGHT of the contact."""
    cx = getattr(sc, "params", {}).get("cx")
    if cx is None:                                   # box family
        return 1.0 if j["id"] == "J_left" else -1.0
    return 1.0 if float(j["x"]) < float(cx) else -1.0


def _mask_at(mask, x: float, y: float) -> bool:
    yi, xi = int(round(y)), int(round(x))
    if not (0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]):
        return False
    return bool(mask[yi, xi])


# --------------------------------------------------------------------------
# A. the occluder's silhouette is bounded by the object
# --------------------------------------------------------------------------

@pytest.mark.parametrize("corpus,name", ALL_OCCLUSION)
def test_the_occluder_silhouette_does_not_cover_the_whole_frame(corpus, name):
    sc = _scene(corpus, name)
    for j in sc.junctions:
        near = sc.masks[j["near"]]
        x, y = float(j["x"]), float(j["y"])
        out = -_inward(sc, j)
        for dy in (-3.0, 0.0, 3.0):
            assert not _mask_at(near, x + 6.0 * out, y + dy), (
                "%s %s %s: occluder material 6 px OUTSIDE the contact at "
                "(%.1f, %.1f) -- the silhouette mask is not bounded by the "
                "object (B37: mask spans x=%d..%d)"
                % (name, j["id"], corpus, x + 6.0 * out, y + dy,
                   int(np.where(near.any(axis=0))[0].min()),
                   int(np.where(near.any(axis=0))[0].max())))
    # the mask's own extent must be the object's, not the frame's
    for obj, m in sc.masks.items():
        cols = np.where(m.any(axis=0))[0]
        if cols.size:
            assert cols.max() - cols.min() + 1 < m.shape[1], (
                "%s: mask %r spans every column of the frame" % (name, obj))


# --------------------------------------------------------------------------
# B. the interrupted far contour is really in the pixels
# --------------------------------------------------------------------------

def _far_edge_step_profile(sc, jy: int) -> np.ndarray:
    """|raw(y+2) - raw(y-2)| along the far object's top edge row."""
    raw = np.asarray(sc.raw_luminance, dtype=np.float64)
    return np.abs(raw[min(jy + 2, raw.shape[0] - 1), :]
                  - raw[max(jy - 2, 0), :])


@pytest.mark.parametrize("corpus,name", ALL_OCCLUSION)
def test_the_far_contour_is_visible_outside_the_occluder(corpus, name):
    sc = _scene(corpus, name)
    raw = np.asarray(sc.raw_luminance, dtype=np.float64)
    for j in sc.junctions:
        x, y = float(j["x"]), float(j["y"])
        out = -_inward(sc, j)
        near = sc.masks[j["near"]]
        prof = _far_edge_step_profile(sc, int(round(y)))
        # reference: the strongest step along this edge where the occluder is
        # NOT in the way -- the same contour, so no invented contrast constant
        visible = np.array([not _mask_at(near, xx, y)
                            for xx in range(raw.shape[1])])
        assert visible.any(), "%s: the occluder covers its whole frame" % name
        ref = float(prof[visible].max())
        xs = int(round(x + 8.0 * out))
        assert 0 <= xs < raw.shape[1], (name, j["id"], xs)
        got = float(prof[xs])
        assert got >= 0.5 * ref, (
            "%s %s (%s): the far contour's raw step 8 px outside the junction "
            "is %.4f against a reference of %.4f on the same edge -- the "
            "occluder's paint has erased the contour the junction is supposed "
            "to interrupt (B37)" % (name, j["id"], corpus, got, ref))
        assert not _mask_at(near, xs, y), (
            "%s %s: the sample point is inside the occluder" % (name, j["id"]))


@pytest.mark.parametrize("name", U14S.SEALED_STIMULUS_INVALID)
def test_the_excluded_sealed_scene_still_fails_the_gate(name):
    """[R0.6a] The exclusion is not a claim that the gate is wrong.

    One scene of the 68-scene U14 sealed set (`u14_box_09`, the combined
    max-haze + max-blur + max-texture corner) failed this file's far-contour
    gate.  It was excluded from the SCORED set before any cue statistic on the
    corpus was read.  The scene is still built and this pin requires the gate to
    STILL fail on it, so the stated reason cannot rot and the exclusion cannot
    later pass itself off as "a scene that happened to be hard".
    """
    sc = U14S.build_u14_sealed(name)
    raw = np.asarray(sc.raw_luminance, dtype=np.float64)
    failures = []
    for j in sc.junctions:
        x, y = float(j["x"]), float(j["y"])
        out = -_inward(sc, j)
        near = sc.masks[j["near"]]
        prof = _far_edge_step_profile(sc, int(round(y)))
        visible = np.array([not _mask_at(near, xx, y)
                            for xx in range(raw.shape[1])])
        if not visible.any():
            continue
        ref = float(prof[visible].max())
        xs = int(round(x + 8.0 * out))
        if not 0 <= xs < raw.shape[1]:
            continue
        got = float(prof[xs])
        if got < 0.5 * ref:
            failures.append((j["id"], got, ref))
    assert failures, (
        "%s now passes the far-contour gate (%s) -- then the exclusion recorded "
        "in u14_sealed_scenes.SEALED_STIMULUS_INVALID is stale and the scene "
        "must be restored to the scored set with the seal re-taken"
        % (name, U14S.SEALED_STIMULUS_INVALID.get(name)))
    assert name not in U14S.SEALED_SCENES, (
        name, "an excluded, stimulus-invalid scene entered the scored set")


# --------------------------------------------------------------------------
# C. what the pixels actually carry at the junction (measurement, not a gate)
# --------------------------------------------------------------------------

def _edge_map(mask) -> np.ndarray:
    """Boundary pixels of a boolean mask (8-neighbour erosion)."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), dtype=bool)
    pad[1:-1, 1:-1] = m
    nb = (pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
          & pad[:-2, :-2] & pad[:-2, 2:] & pad[2:, :-2] & pad[2:, 2:])
    return m & ~nb


def _nearest_boundary(edge: np.ndarray, x: float, y: float,
                      radius: float = 12.0):
    """(bx, by) of the boundary pixel nearest (x, y), or None."""
    ys, xs = np.where(edge)
    if xs.size == 0:
        return None
    d = np.hypot(xs + 0.5 - x, ys + 0.5 - y)
    k = int(np.argmin(d))
    if d[k] > radius:
        return None
    return int(xs[k]), int(ys[k])


def _across(mask, raw, bx: int, by: int, off: int = 2):
    """Step across the boundary at (bx, by), along the axis the mask changes on.

    Returns ``(step, inside_xy, outside_xy, axis)``; the inside/outside pixels
    are taken from the renderer's own mask, so no normal is estimated.
    """
    h, w = mask.shape
    for axis in ("x", "y"):
        if axis == "x":
            a, b = (by, bx - off), (by, bx + off)
        else:
            a, b = (by - off, bx), (by + off, bx)
        if not all(0 <= pt[0] < h and 0 <= pt[1] < w for pt in (a, b)):
            continue
        ma = bool(mask[a[0], a[1]])
        mb = bool(mask[b[0], b[1]])
        if ma == mb:
            continue                      # the mask does not change on this axis
        inside, outside = (a, b) if ma else (b, a)
        return (abs(float(raw[inside[0], inside[1]])
                    - float(raw[outside[0], outside[1]])),
                (inside[1], inside[0]), (outside[1], outside[0]), axis)
    return None


def test_measure_limb_contrast_at_every_junction():
    """Record the near contour's raw step ABOVE and BELOW each contact.

    Not a gate on contrast: a shaded cylinder's limb can legitimately match the
    surface it crosses, and that ceiling is what U12.5 exists to measure.  What
    IS asserted is that the renderer's mask and the renderer's pixels agree at
    the sampled points -- the pixel 2 px inside the silhouette is mask material
    and the one 2 px outside is not -- so the contrast figures the census and the
    report quote are measured where the boundary actually is.  The inside/outside
    split comes from the mask itself, never from an estimated normal.  Written to
    `diagnostics/_u12_limb_contrast.txt`.
    """
    import os
    lines: List[str] = []
    # the DEVELOPMENT corpora only: this file is a development artifact, and the
    # sealed set's numbers stay out of it (the gate above still runs over it).
    for corpus, name in OCCLUSION:
        sc = _scene(corpus, name)
        raw = np.asarray(sc.raw_luminance, dtype=np.float64)
        for j in sc.junctions:
            x, y = float(j["x"]), float(j["y"])
            near = sc.masks[j["near"]]
            edge = _edge_map(near)
            b0 = _nearest_boundary(edge, x, y)
            assert b0 is not None, (
                "%s %s: no silhouette boundary within 12 px of the recorded "
                "contact (%.2f, %.2f)" % (name, j["id"], x, y))
            base = _across(near, raw, b0[0], b0[1])
            assert base is not None, (
                "%s %s: the silhouette does not cross a pixel boundary at "
                "(%d, %d)" % (name, j["id"], b0[0], b0[1]))
            axis = base[3]
            # walk 6 px along the limb (the tangent is the other axis) and take
            # the step across the contour there
            row = [("at", base[0], b0)]
            for sign, tag in ((-1.0, "back"), (1.0, "fwd")):
                wx = x + (0.0 if axis == "x" else sign * 6.0)
                wy = y + (sign * 6.0 if axis == "x" else 0.0)
                b = _nearest_boundary(edge, wx, wy)
                assert b is not None, (
                    "%s %s: the silhouette stops within 6 px %s of the contact "
                    "(nearest boundary to (%.1f, %.1f) is farther than 12 px)"
                    % (name, j["id"], tag, wx, wy))
                got = _across(near, raw, b[0], b[1])
                assert got is not None and got[3] == axis, (
                    "%s %s: 6 px %s of the contact the silhouette runs along a "
                    "different axis (%s vs %s) -- not a limb crossing a bar"
                    % (name, j["id"], tag, got and got[3], axis))
                row.append((tag, got[0], b))
            row.sort(key=lambda r: (-r[2][1] if axis == "x" else r[2][0]))
            lines.append(
                "%-6s %-22s %-8s (%6.2f,%6.2f) limb axis=%s  raw step across "
                "the near contour: %s"
                % (corpus, name, j["id"], x, y, axis,
                   "  ".join("%s(%.1f,%.1f)=%.4f" % (t, b[0] + 0.5, b[1] + 0.5,
                                                     st)
                             for t, st, b in row)))
            for t, st, b in row:
                assert math.isfinite(st) and st >= 0.0, (name, j["id"], t)
    here = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.normpath(os.path.join(
        here, "..", "..", "diagnostics", "_u12_limb_contrast.txt"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        fh.write("Raw luminance step across the NEAR contour at each truth "
                 "junction, measured on the renderer's own silhouette mask "
                 "(2 px inside vs 2 px outside), at the contact and 6 px "
                 "either way along the limb.\nAfter the B37 renderer fix.\n\n")
        fh.write("\n".join(lines) + "\n")
