"""memory/aspects.py -- extracted from consolidator.py (Move 5a).

MOVE ONLY: whole top-level statements, verbatim.

[P4 classifier of record] This module IS the category scorer of record.
`AspectGraph.category_scores` is the only path that turns a view descriptor
into per-category scores; nothing in the default scoring path reads the
consolidator's core machinery, which stays a memory/operator store (core
growth, containment, margins). The demoted virtual-view branch is off by
default (see category_scores) and no other scorer is consulted anywhere in
the pipeline.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Sequence
import numpy as np
import parameters as P
from core.registries import (  # noqa: F401
    CONSTITUTIONAL_RANGES,
    POPULATION_AXES,
)
from core.population import (  # noqa: F401
    aligned_population_similarity,
    axis_similarity,
    population_similarity,
)
from memory.relational import RELATIONAL_GROUP_WEIGHT  # noqa: F401


# [Stage 5 rewire] Slot ranges for the PEAK_CHAIN descriptor source. A
# PEAK_CHAIN carries six slots (n_peaks, mean_theta, theta_var, closure,
# curvature, length); slot 1 is circular over 12, which under the tree's N/2
# convention declares range 6.0 while the bin index runs 0..11 -- so it is
# looked up explicitly rather than reused verbatim from the registry.
_CHAIN_R = (15.0, 6.0, 7.0, 1.0, 7.0, 7.0)


def _ranges_for(n_slots: int) -> Tuple[float, ...]:
    """Slot ranges by descriptor arity, so both sources share one matcher.

    Length 6 is a PEAK_CHAIN; anything else is a shock RIBBON. Dispatching on
    arity (rather than on a global mode) means a view built from chains and a
    view built from ribbons stay distinguishable to the matcher instead of
    silently mismatching.
    """
    return _CHAIN_R

@dataclass
class SurfaceDesc:
    """[W2/O7] Graded surface slots around one contour fragment.

    Every slot is represented on BOTH sides of the fragment -- the enclosed
    ("interior") side and the flank -- because a boundary fragment is a figure/
    ground statement, not a single sample. Sampling uses the continuous
    interior field as a WEIGHT (`interior_weighted_sample` / its vectorised
    twin): weights are `interior` for the enclosed side and `1 - interior` for
    the flank. No mask is ever formed.

    Invariants:
      * hue_* in radians [0, 2*pi); undefined when sat_* is ~0 (the hue
        similarity is down-weighted by saturation, never read as a value).
      * sat_*, lum_* in [0, 1].
      * theta_* are L1-normalised distributions over the bank's orientation
        channels; sf_* over its spatial-frequency channels (empty when no
        texture field was supplied).
    """
    lum_i: float = 0.0
    lum_f: float = 0.0
    hue_i: float = 0.0
    hue_f: float = 0.0
    sat_i: float = 0.0
    sat_f: float = 0.0
    theta_i: Tuple[float, ...] = ()
    theta_f: Tuple[float, ...] = ()
    sf_i: Tuple[float, ...] = ()
    sf_f: Tuple[float, ...] = ()


@dataclass
class PartDesc:
    bins: Tuple[float, ...]
    pos: Tuple[float, float]      # Object-centroid relative, RMS-spread normalized
    ang: float                    # Axial axis orientation in radians [0, pi)
    # [W2/O7] The graded truth the bins index. `population` is the PEAK_CHAIN
    # population carried on the detection (theta_dist circular 12, curvature 8,
    # length 8). Ignored by the legacy matcher, so attaching it is behaviour-
    # preserving; the V2 matcher reads it.
    population: Tuple[Tuple[float, ...], ...] = ()
    # [W2/O7] Object-centred UPRIGHT polar address: (r_norm, theta_radians),
    # theta measured from image-up (0 = above the object centre, +pi/2 = right).
    # `pos` stays the Cartesian canonical form; `polar` is derived from it.
    polar: Optional[Tuple[float, float]] = None
    # [W2/O7] Graded surface slots (interior + flank), or None when no image
    # planes were supplied.
    surface: Optional[SurfaceDesc] = None
    # [P2] The RELATIONAL CONTEXT: the mixture of the edge populations
    # incident to this part's node, tagged by relation family
    # (memory.relational.RelationalContext). None when the view was built
    # without the graph.
    relations: Optional[Any] = None


@dataclass(eq=False)
class ViewRecord:
    vid: int
    t: int
    instance: str
    label: Any
    az: float
    el: float
    parts: List[PartDesc]
    proto: int = -1
    # [SIZE SLOT] The view's absolute size (object RMS member spread, in
    # pixels) as its OWN slot, strictly apart from the normalized shape: the
    # part `pos` values are RMS-normalized (shape without size), so the size
    # must be stored separately or it is lost. Consumed by `_size_similarity`
    # with its own kernel and switch, never folded into the shape slot vector.
    size: float = 1.0
    # [W2/O7] View-level soft angular mass histogram over the upright polar
    # axis (circular, L1-normalised). Adds the GLOBAL arrangement that a
    # per-part kernel cannot see. None for legacy records.
    # [I] The polar histogram was written by the builders and never read
    # (the scorer recomputes it from the parts), so it is retired.
    # [V3] The GLOBAL-LAYOUT group: the soft coarse grid of oriented contour
    # energy, the energy's global moments, and its radial / projection
    # profiles (memory.gist.GistDesc). None for legacy records. It carries
    # the layout the shape groups were missing (the cheap layout baselines
    # beat them), and its energy centroid is the arrangement group's origin.
    gist: Optional[Any] = None
    # [PEAK-FRAGMENT GROUP] The soft 3-D population over the peak
    # fragments (angular position x signed curvature x orientation relative
    # to the radial direction). None when the view was built without the
    # peaks.
    peak_map: Optional[Tuple[float, ...]] = None
    # [TIER + DEPTH] The lazily-built per-view measurements of the middle
    # tier (memory/assemblies.py) and of the static depth cues
    # (memory/depth_cues.py). They are cached HERE because a sweep pairs the
    # same views thousands of times, and both are pure functions of the view.
    tier: Optional[Any] = None
    depth: Optional[Any] = None
    # [P2] The VIEW-LEVEL RELATIONAL CONTEXT: the same mixture over the
    # view's whole graph (memory.relational.RelationalContext). None when
    # the view was built without the graph.
    relational: Optional[Any] = None

def _chain_members(det: Any) -> List[Any]:
    """Member anchors of a PEAK_CHAIN detection, in the chain's own order."""
    return [r[1] for r in getattr(det, "rho", ()) or ()
            if len(r) > 1 and r[0] == "member"]


def _chain_axis_angle(pts: Sequence[Tuple[float, float]]) -> float:
    """Circular-mean direction of a chain's member steps, on [0, pi).

    Each member contributes its step direction toward the next member, and
    the directions are combined on the pi-periodic axis (a contour axis has
    no head/tail), so a chain traversed in either direction yields the same
    axis. Falls back to the first step when the mean is degenerate.

    [RESOLUTION AGREEMENT] The combination is the DOUBLED-angle mean
    (2*angle), which is the correct reading for an AXIAL quantity: the
    directions are combined as unit vectors on the doubled circle, so a
    direction and its opposite reinforce instead of cancelling. The plain
    mean -- which the docstring described but the code did not do -- is
    head/tail-sensitive and collapses toward zero whenever the steps
    disagree, so the axis was unstable: measured over the same chains
    captured at 256 and 512 px, the plain mean disagreed by a MEDIAN 13.6
    degrees and a p90 of 75.1, while the doubled-angle mean disagrees by a
    median 0.13 degrees and a p90 of 2.0. The 75-degree tail was this
    measure, not the front end.
    """
    if len(pts) < 2:
        return 0.0
    sx = sy = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        m = math.hypot(dx, dy)
        if m < 1e-9:
            continue
        # The doubled angle: the pi-periodic axis as a circle.
        a2 = 2.0 * math.atan2(dy, dx)
        sx += math.cos(a2)
        sy += math.sin(a2)
    if abs(sx) < 1e-12 and abs(sy) < 1e-12:
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return 0.0
        return math.atan2(dy, dx) % math.pi
    return (0.5 * math.atan2(sy, sx)) % math.pi


def _upright_polar(pos: Sequence[float]) -> Tuple[float, float]:
    """[W2/O7] Object-centred polar address in the UPRIGHT frame.

    `pos` is the RMS-normalised Cartesian address (x right, y down). The
    radius is its length; the angle is measured from image-up (0 = above the
    object centre, +pi/2 = to the right), so the frame is the viewer's upright,
    not a principal-axis frame. Returns (r, theta) with theta in (-pi, pi].
    """
    x, y = float(pos[0]), float(pos[1])
    return (math.hypot(x, y), math.atan2(x, -y))


def _soft_polar_hist(
    polars: Sequence[Tuple[float, float]], bins: int,
) -> Tuple[float, ...]:
    """[W2/O7] Soft circular histogram of the view's upright polar addresses.

    Each part contributes a wrapped Gaussian bump (width 1 bin, the angular
    resolution) weighted by its radius, so a part near the centre -- whose
    direction is genuinely ambiguous -- contributes little, and mass far from
    the centre dominates. L1-normalised; uniform when there is no evidence.
    """
    n = max(1, int(bins))
    acc = [0.0] * n
    for r, th in polars:
        w = max(0.0, float(r))
        if w <= 0.0:
            continue
        centre = (float(th) + math.pi) / (2.0 * math.pi) * n
        for j in range(n):
            d = abs(j - centre)
            d = min(d, n - d)
            acc[j] += w * math.exp(-0.5 * (d / 1.0) ** 2)
    total = sum(acc)
    if total <= 1e-12:
        return tuple(1.0 / n for _ in range(n))
    return tuple(a / total for a in acc)


def _hue_sat_planes(color: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Hue (radians, circular) and saturation (0..1) planes from an RGB image.

    Hue is the standard colour-wheel angle; saturation is (max-min)/max. A
    grey pixel has saturation 0, and its hue is NOT evidence -- the matcher
    down-weights hue by saturation, so a fabricated hue never scores.
    """
    rgb = np.asarray(color, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return None, None
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb[..., :3], axis=-1)
    mn = np.min(rgb[..., :3], axis=-1)
    d = mx - mn
    sat = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    dd = np.maximum(d, 1e-12)
    h = np.select(
        [mx == r, mx == g, mx == b],
        [((g - b) / dd) % 6.0, (b - r) / dd + 2.0, (r - g) / dd + 4.0],
        default=0.0,
    )
    hue = np.where(d > 1e-12, (h * (math.pi / 3.0)) % (2.0 * math.pi), 0.0)
    return hue, sat


def _weighted_window(
    plane: Any, interior: Any, x: float, y: float, radius: float,
) -> Optional[np.ndarray]:
    """[W2/O7] Vectorised interior-weighted mean over a window.

    Same contract as `perception.interior_field.interior_weighted_sample` but
    works on a stack of planes: `plane` may be (H, W) or (..., H, W), and the
    result is a scalar or an array over the leading axes. The weight of each
    pixel is `gaussian * weight_plane`; passing `1 - interior` as
    `weight_plane` samples the FLANK. Never a mask: every pixel contributes
    proportionally to its continuous enclosure value.
    """
    p = np.asarray(plane, dtype=np.float64)
    enc = np.asarray(interior, dtype=np.float64)
    if p.ndim < 2 or enc.ndim != 2 or p.shape[-2:] != enc.shape:
        return None
    h, w = enc.shape
    r = max(1, int(math.ceil(float(radius))))
    s = max(float(radius), 1e-6)
    x0, x1 = max(0, int(math.floor(x - r))), min(w - 1, int(math.ceil(x + r)))
    y0, y1 = max(0, int(math.floor(y - r))), min(h - 1, int(math.ceil(y + r)))
    if x1 < x0 or y1 < y0:
        return None
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    gw = np.exp(-0.5 * (((xx - x) / s) ** 2 + ((yy - y) / s) ** 2))
    wgt = gw * enc[y0:y1 + 1, x0:x1 + 1]
    den = float(wgt.sum())
    if den <= 1e-12:
        return None
    win = p[..., y0:y1 + 1, x0:x1 + 1]
    return (win * wgt).sum(axis=(-1, -2)) / den


def _l1(v: Any) -> Tuple[float, ...]:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    t = float(a.sum())
    if not np.isfinite(t) or t <= 1e-12:
        return ()
    return tuple(float(x / t) for x in a)


def _sample_surface(
    x: float,
    y: float,
    radius: float,
    luminance: Any,
    color: Any,
    interior: Any,
    gabor_complex: Any,
) -> Optional["SurfaceDesc"]:
    """[W2/O7] Sample the graded surface slots on BOTH sides of a fragment.

    Interior weights = the continuous interior field; flank weights =
    1 - interior. A slot that the field cannot support (all weights vanish,
    e.g. a fragment deep in the object has no flank) is stored as -1.0 for
    scalars, or an empty tuple for populations, and the matcher SKIPS it --
    it is never compared as a fabricated 0.
    """
    if interior is None:
        return None
    enc = np.asarray(interior, dtype=np.float64)
    if enc.ndim != 2:
        return None
    flank = 1.0 - enc
    if luminance is None:
        return None
    lum = np.asarray(luminance, dtype=np.float64)
    if lum.ndim == 3:
        lum = lum[..., 0]
    if lum.shape != enc.shape:
        return None

    def _s(plane: Any, weights: np.ndarray) -> float:
        v = _weighted_window(plane, weights, x, y, radius)
        return -1.0 if v is None else float(v)

    lum_i = _s(lum, enc)
    lum_f = _s(lum, flank)

    hue_i = hue_f = sat_i = sat_f = -1.0
    if color is not None:
        hue_p, sat_p = _hue_sat_planes(color)
        if hue_p is not None:
            hue_i = _s(hue_p, enc)
            hue_f = _s(hue_p, flank)
            sat_i = _s(sat_p, enc)
            sat_f = _s(sat_p, flank)

    theta_i: Tuple[float, ...] = ()
    theta_f: Tuple[float, ...] = ()
    sf_i: Tuple[float, ...] = ()
    sf_f: Tuple[float, ...] = ()
    if gabor_complex is not None:
        gc = np.asarray(gabor_complex, dtype=np.float64)
        if gc.ndim == 4 and gc.shape[-2:] == enc.shape:
            ti = _weighted_window(gc, enc, x, y, radius)
            tf = _weighted_window(gc, flank, x, y, radius)
            if ti is not None and ti.size:
                theta_i = _l1(ti.sum(axis=1))
                sf_i = _l1(ti.sum(axis=0))
            if tf is not None and tf.size:
                theta_f = _l1(tf.sum(axis=1))
                sf_f = _l1(tf.sum(axis=0))

    return SurfaceDesc(
        lum_i=lum_i, lum_f=lum_f,
        hue_i=hue_i, hue_f=hue_f,
        sat_i=sat_i, sat_f=sat_f,
        theta_i=theta_i, theta_f=theta_f,
        sf_i=sf_i, sf_f=sf_f,
    )


def _peak_fragment_map(peaks, center: Optional[Tuple[float, float]],
                       anchor_positions: Optional[Dict[Any, Any]] = None,
                       angle_bins: int = 12, curvature_bins: int = 8,
                       relative_bins: int = 8) -> Optional[Tuple[float, ...]]:
    """The PEAK-FRAGMENT map: a soft 3-D population over
    (angular position around the energy centre) x (signed curvature) x
    (orientation relative to the radial direction).

    WHY PEAKS AND NOT CHAINS. A chain's MEMBERS change with the view (the
    field's peaks move, so a chain is cut differently); measured across 256
    and 512 px, only 72% of a chain's members correspond. A PEAK is a local
    maximum of the same field, so it recurs. The map is therefore built from
    the peak detections, each contributing its GRADED curvature and its
    GRADED orientation (read from its population), placed at its angular
    position around the energy centre and rotated into the frame of the
    radial direction (so "radial" and "tangential" are the same quantity for
    every object).

    The map is a soft histogram: every peak's activity is added with a
    Gaussian over the angle, so the comparison is an overlap, not a bin
    match. It is a field statistic of the front end -- no mask, no category.
    """
    if not peaks or center is None:
        return None
    cx, cy = center
    acc = [0.0] * (angle_bins * curvature_bins * relative_bins)
    for det in peaks:
        sig = getattr(det, "sigma", None)
        pop = getattr(sig, "population", ()) or ()
        if len(pop) < 3:
            continue
        theta_axis, _sf_axis, curv_axis = pop[0], pop[1], pop[2]
        if not theta_axis or not curv_axis:
            continue
        anchor = getattr(det, "alpha", None)
        pos = None
        if anchor_positions is not None:
            pos = anchor_positions.get(anchor)
        if pos is None:
            pos = getattr(det, "_abs_pos", None)
        if pos is None or len(pos) < 2:
            continue
        px, py = float(pos[0]), float(pos[1])
        dx = px - cx
        dy = py - cy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        radial = math.atan2(dy, dx)
        # The peak's orientation: the circular mean of its theta activity.
        # [C] The orientation axis is PI-PERIODIC, so the phase argument
        # is the DOUBLED angle (2*pi*(i+0.5)/n is 2*theta_i) and the
        # result is halved once. (The argument was previously FOUR times
        # the true angle, so the channel carried 2*theta.)
        sx = sum(math.sin(2.0 * math.pi * (i + 0.5) / len(theta_axis))
                 * w for i, w in enumerate(theta_axis))
        sy = sum(math.cos(2.0 * math.pi * (i + 0.5) / len(theta_axis))
                 * w for i, w in enumerate(theta_axis))
        if abs(sx) < 1e-12 and abs(sy) < 1e-12:
            continue
        peak_theta = 0.5 * math.atan2(sx, sy)
        rel = (peak_theta - radial) % math.pi
        # The angle around the energy centre (the peak's own position).
        ang = math.atan2(dy, dx) % (2.0 * math.pi)
        # The soft placement: a Gaussian over the angle bin.
        centre_bin = ang / (2.0 * math.pi) * angle_bins
        for ab in range(angle_bins):
            d = abs(ab - centre_bin)
            d = min(d, angle_bins - d)
            gw = math.exp(-0.5 * (d / 1.0) ** 2)
            if gw < 1e-4:
                continue
            base = ab * curvature_bins * relative_bins
            # The signed curvature: its graded activity, placed in the bins.
            for ci, cw in enumerate(curv_axis):
                if cw < 1e-4:
                    continue
                cbase = base + ci * relative_bins
                # The orientation relative to the radial direction: the
                # theta activity ROTATED into the radial frame.
                for ri in range(relative_bins):
                    r_centre = rel / math.pi * relative_bins
                    dr = abs(ri - r_centre)
                    dr = min(dr, relative_bins - dr)
                    rw = math.exp(-0.5 * (dr / 1.0) ** 2) * cw
                    if rw < 1e-5:
                        continue
                    acc[cbase + ri] += gw * rw
    total = sum(acc)
    if total <= 0.0:
        return None
    return tuple(a / total for a in acc)


def _peak_fragment_similarity(a, b) -> float:
    """The peak-fragment group's similarity: the chance-corrected overlap of
    the two soft maps (a Pearson correlation of the activity vectors)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return axis_similarity(a, b)


def build_view_descriptor_from_chains(
    chain_dets: Any,
    anchor_positions: Optional[Dict[Any, Any]],
    R: float,
    luminance: Any = None,
    color: Any = None,
    interior: Any = None,
    gabor_complex: Any = None,
    peaks: Optional[List[Any]] = None,
    relations: Optional[Dict[Any, Any]] = None,
    relational: Optional[Any] = None,
) -> Optional[Tuple[List[PartDesc], List[Tuple[float, ...]], float]]:
    """[Stage 5 rewire] View descriptor from the live field's PEAK_CHAINs.

    Replaces the shock-structure source, which is unreachable on this tree
    (the segmenter region path is empty, so no segments are ever produced).
    A PEAK_CHAIN is the same OBJECT CLASS a shock ribbon was -- a bounded
    contour fragment with an internal axis -- so it drops straight into the
    descriptor, matching, resonance, clustering and link machinery without
    changing any of it.

    Each chain becomes one part:
      bins  its six quantized slots
      pos   its member centroid, object-centre relative, RMS-spread normalized
      ang   the circular mean of its member step directions

    Ensembles are GONE: they were a shock-graph grouping (a hand-built
    category), and the chain level never had one. A view is a set of
    contour FRAGMENTS; grouping them into parts is the assembly tier's job,
    and it is driven by repetition, not by a declared vocabulary.

    [Q2 bug fix] `pos` used to be divided by the median WAVELENGTH (R), which
    made every position ~15 units wide against a 0.35-wide position kernel:
    the position term was identically zero for every part pair, so matching
    degenerated into a pure bag-of-slots comparison (measured: 10226
    same-class and 21785 cross-class part pairs with slot similarity >= 0.90
    and median position distance 14.8 / 15.5). The normaliser is now the
    view's own RMS member spread about the object centroid, weighted by each
    chain's peak count, so a part position is O(1) and the position kernel can
    do its job.

    [W2/O7] The graded truth now rides on every part:
      * `population` is the chain's own PEAK_CHAIN population (theta_dist
        circular 12, curvature 8, length 8) -- the full vector the bins index.
      * `polar` is the same address in the object-centred UPRIGHT frame.
      * when image planes are supplied, `surface` carries graded hue,
        saturation, luminance and texture-by-orientation/scale sampled on the
        interior and flank sides with the soft interior field as the weight.
    The return is a 3-tuple for legacy callers; it becomes a 4-tuple ending in
    the view-level soft polar histogram only when V2 planes were supplied.
    """
    if not chain_dets or not R or R <= 0.0:
        return None
    ordered = sorted(chain_dets, key=lambda d: getattr(d, "alpha", 0))
    raw: List[Tuple[Tuple[float, ...], List[Tuple[float, float]],
                     Tuple[Tuple[float, ...], ...]]] = []
    for det in ordered:
        q = getattr(getattr(det, "sigma", None), "quantized", None)
        if not q or len(q) < 6:
            continue
        pts: List[Tuple[float, float]] = []
        for a in _chain_members(det):
            if anchor_positions is None:
                continue
            p = anchor_positions.get(a)
            if p is not None and len(p) >= 2:
                try:
                    pts.append((float(p[0]), float(p[1])))
                except (TypeError, ValueError):
                    continue
        if len(pts) < 2:
            continue
        pop = tuple(
            tuple(float(v) for v in vec)
            for vec in (getattr(getattr(det, "sigma", None), "population", ()) or ())
            if vec
        )
        raw.append((tuple(float(v) for v in q[:6]), pts, pop))
    if not raw:
        return None

    # global object centroid across all chains, matching the ribbon source's
    # "all axis points" convention, so pos is comparable between the two.
    all_pts = [p for _b, pts, _pop in raw for p in pts]
    cx = sum(p[0] for p in all_pts) / len(all_pts)
    cy = sum(p[1] for p in all_pts) / len(all_pts)

    # RMS member spread about that centroid, weighted by chain peak count
    # (bins[0]). Fallback to the wavelength when the view has no spread, so
    # the normaliser is always finite and positive.
    num = 0.0
    den = 0.0
    for bins, pts, _pop in raw:
        w = max(1.0, float(bins[0]))
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        num += w * ((mx - cx) ** 2 + (my - cy) ** 2)
        den += w
    rms = math.sqrt(num / den) if den > 0.0 else 0.0
    if not (rms > 1e-6):
        rms = float(R)

    v2 = any(v is not None for v in
             (luminance, color, interior, gabor_complex))
    surf_radius = 0.0
    if v2:
        surf_radius = min(
            float(P.ASPECT_V2_SURFACE_RADIUS_MAX),
            max(float(P.ASPECT_V2_SURFACE_RADIUS_MIN),
                float(P.ASPECT_V2_SURFACE_RADIUS_FRAC) * float(R)),
        )

    # [V3] The GLOBAL-LAYOUT group and the ARRANGEMENT's origin.
    #
    # The arrangement used to be centred on the parts' own centroid and
    # scaled by the parts' own RMS spread. That centre is a property of the
    # DETECTED PARTS, so it moves with the clutter and it is not the object's
    # centre: the measured arrangement term was a net loss (full 0.4625 <
    # no_pos 0.6625) and the least stable group under perturbation
    # (pos_only retained 0.3272 under a 0.9 scale change).
    #
    # The energy centroid and the energy radius are the object's own centre
    # and scale, read off the contour ENERGY the front end already computed:
    # they are weighted by the evidence rather than by the part list, so
    # texture clutter moves them far less than a part list does.
    gist = None
    if v2:
        from memory.gist import build_gist, oriented_energy
        energy = oriented_energy(gabor_complex)
        gist = build_gist(energy)
    if gist is not None and gist.radius_px > 1e-6:
        org_x, org_y = gist.center_px
        org_r = gist.radius_px
    else:
        org_x, org_y = cx, cy
        org_r = rms if rms > 1e-6 else float(R)

    parts: List[PartDesc] = []
    polars: List[Tuple[float, float]] = []
    for bins, pts, pop in raw:
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        pos = ((mx - cx) / rms, (my - cy) / rms)
        # [V3] The upright polar address is now taken from the ENERGY
        # centre: r is the distance in ENERGY RADII (scale-free), theta is
        # still measured from image-up (the viewer's upright, not the
        # principal axis -- the stability probe showed the upright frame
        # beats the principal-axis frame in every class).
        pol = _upright_polar(((mx - org_x) / org_r, (my - org_y) / org_r))
        polars.append(pol)
        surface = None
        if v2:
            surface = _sample_surface(
                mx, my, surf_radius, luminance, color, interior, gabor_complex)
        parts.append(PartDesc(
            bins=bins,
            pos=pos,
            ang=_chain_axis_angle(pts),
            population=pop,
            polar=pol,
            surface=surface,
            # [P2] The RELATIONAL CONTEXT of this part, when the caller
            # resolved it (the graph's incident edges, tagged by relation
            # family).  Keyed by the chain's ANCHOR, which is stable across
            # the builder's own ordering.
            relations=(relations.get(det.alpha)
                       if relations is not None else None),
        ))
    # [SIZE SLOT] The un-normalized object spread (pixels) is the view's size;
    # the RMS normaliser above removed it from `pos`, so it is returned as its
    # own slot rather than discarded.
    if v2:
        hist = _soft_polar_hist(polars, int(P.ASPECT_V2_POLAR_BINS))
        # [PEAK-FRAGMENT GROUP] The soft 3-D population over the peak
        # fragments (the peaks recur across views; the chains' members do
        # not). It is built around the energy centre, which the layout group
        # measured.
        frag = None
        if peaks and gist is not None:
            frag = _peak_fragment_map(peaks, gist.center_px, anchor_positions)
        if relational is not None:
            # [P2] The VIEW-LEVEL relational context rides in a sixth slot.
            return parts, rms, hist, gist, frag, relational
        return parts, rms, hist, gist, frag
    return parts, rms


def _awrap(x: float) -> float:
    """Wrap angular differences into [-pi/2, pi/2)."""
    return (x + math.pi / 2.0) % math.pi - math.pi / 2.0


def _size_similarity(A: "ViewRecord", B: "ViewRecord") -> float:
    """[SIZE SLOT] Graded similarity of two views' absolute sizes.

    A log-ratio kernel (size is scale-free, so equal RATIOS mean equal
    similarity): two views of the same spread score 1.0, a factor-e mismatch
    falls off with its own sigma. Its own switch (`ASPECT_SIZE_SLOT_ENABLED`)
    keeps size APART from the normalized shape terms -- it is never folded
    into the part slot vectors or the position normaliser.
    """
    if not bool(getattr(P, "ASPECT_SIZE_SLOT_ENABLED", True)):
        return 1.0
    sa = max(float(getattr(A, "size", 1.0)), 1e-9)
    sb = max(float(getattr(B, "size", 1.0)), 1e-9)
    sigma = max(float(getattr(P, "ASPECT_SIZE_SIGMA", 0.7)), 1e-6)
    return math.exp(-((math.log(sa / sb) / sigma) ** 2))

def _bins_sim(a: Sequence[float], b: Sequence[float], ranges: Sequence[float]) -> float:
    n = min(len(a), len(b), len(ranges))
    if n == 0:
        return 0.0
    return math.exp(-sum(((a[i] - b[i]) / (max(ranges[i], 1e-6) / 3.0)) ** 2 for i in range(n)) / n)

def _greedy_match(sim: Dict[Tuple[int, int], float], min_sim: float) -> List[Tuple[int, int, float]]:
    out: List[Tuple[int, int, float]] = []
    ui: Set[int] = set()
    uj: Set[int] = set()
    for (i, j), v in sorted(sim.items(), key=lambda kv: (-kv[1], kv[0])):
        if v >= min_sim and i not in ui and j not in uj:
            out.append((i, j, v))
            ui.add(i)
            uj.add(j)
    return out

def match_views(
    A: ViewRecord, B: ViewRecord, mode: Optional[str] = None
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float, int]]]:
    """Hierarchical part persistence matcher:
    1. Match macro-ensembles via greedy bipartite alignment.
    2. Match parts conditioned on matched parent ensembles.
    3. Match residual unassigned parts using stricter similarity thresholds.

    [Q2] When `P.ASPECT_GRADED_PARTS` is on, part similarity is computed on
    the coarsened slot set `P.ASPECT_PART_SLOTS` and the exact slot-0
    equality gate is replaced by the graded slot distance (which already
    carries slot 0). The flag ships DARK (off): this is a scoring-facing
    change, so per the operating policy it is not enabled until second
    instances per class confirm it. The legacy path (all six slots, exact
    slot-0 gate x ASPECT_ORDER_MISMATCH) stays the default.

    [W2/O7] `mode` selects the population-coded matcher (see
    `match_views_v2`); "legacy" forces the legacy matcher, which stays
    available as the ablation arm and for the legacy machinery that has no
    V3 equivalent. None keeps the legacy matcher unless
    `ASPECT_V3_ENABLED` is switched on.
    """
    if mode == "legacy":
        mode = None
    elif mode is None and not bool(getattr(P, "ASPECT_V3_ENABLED", False)):
        mode = None
    else:
        return [], match_views_v2(A, B, mode or "full")
    ms = P.ASPECT_MATCH_MIN
    sp = P.ASPECT_SIGMA_POS

    slots = tuple(P.ASPECT_PART_SLOTS) if P.ASPECT_GRADED_PARTS else ()
    v2_rng = tuple(_CHAIN_R[k] for k in slots if k < len(_CHAIN_R))

    def psim(i: int, j: int) -> float:
        pa, pb = A.parts[i], B.parts[j]
        if P.ASPECT_GRADED_PARTS:
            # [Q2] coarsened slot set + graded slot-0 (no exact identity gate)
            qa = tuple(pa.bins[k] for k in slots if k < len(pa.bins))
            qb = tuple(pb.bins[k] for k in slots if k < len(pb.bins))
            s = _bins_sim(qa, qb, v2_rng) if len(qa) == len(v2_rng) else 0.0
        else:
            # legacy: exact slot-0 identity gate ("shock order"), ranges by
            # descriptor arity (6 slots = PEAK_CHAIN, 5 = shock RIBBON)
            order_penalty = 1.0 if pa.bins[0] == pb.bins[0] else P.ASPECT_ORDER_MISMATCH
            s = _bins_sim(pa.bins, pb.bins, _ranges_for(len(pa.bins))) * order_penalty
        dist_sq = (pa.pos[0] - pb.pos[0]) ** 2 + (pa.pos[1] - pb.pos[1]) ** 2
        return s * math.exp(-(math.sqrt(dist_sq) / sp) ** 2)

    # One level: every part of A against every part of B. The threshold is
    # the residual level's 1.25*ms, which is what this matcher effectively
    # used before (the ensemble level never fired: every chain carried
    # ens=-1 and ens_bins was empty).
    pm: List[Tuple[int, int, float, int]] = []
    s = {
        (i, j): psim(i, j)
        for i in range(len(A.parts))
        for j in range(len(B.parts))
    }
    for i, j, v in _greedy_match(s, 1.25 * ms):
        pm.append((i, j, v, 1))

    return [], sorted(pm, key=lambda x: (x[3], -x[2]))


# =============================================================================
# [W2/O7] Population-coded descriptor V2
# =============================================================================
# The V2 matcher reads the full graded vector carried on each part. The six
# bins are the LOOKUP INDEX (identity/coarse graph); they enter only as a
# small coarse term inside the shape group. The groups are explicit so the
# O10 ablations drop whole groups and renormalise rather than retune weights:
#
#   shape    fragment population overlap (+ coarse bins)
#   pos      soft object-centred UPRIGHT polar kernel (r, theta)
#   surface  interior/flank hue, saturation, luminance, texture by
#            orientation and scale
#   size     the separate ViewRecord.size slot (unchanged)
#
# `bins_only` reproduces the legacy representation (six-slot Gaussian +
# exact slot-0 order gate, no position) so the comparison against look #1 is
# apples to apples.

_V2_MODES = ("full", "no_pos", "bins_only", "surface_only", "shape_only")
# [V3] The arms of the calibrated fusion. `no_layout` drops the global-layout
# group, `layout_only` keeps it alone, `no_pos` drops the arrangement.
_V3_MODES = ("full", "no_pos", "no_layout", "no_shape", "no_size",
             "no_fragments", "layout_only", "bins_only", "surface_only",
             "shape_only", "no_relational", "relational_only")


def _fragment_similarity(pa: PartDesc, pb: PartDesc) -> float:
    """Graded fragment shape: PEAK_CHAIN population overlap + coarse bins."""
    axes = POPULATION_AXES.get("PEAK_CHAIN", ())
    frag: Optional[float] = None
    if (
        pa.population and pb.population
        and len(axes) == len(pa.population) == len(pb.population)
        and all(len(v) == ax.n_bins for ax, v in zip(axes, pa.population))
        and all(len(v) == ax.n_bins for ax, v in zip(axes, pb.population))
    ):
        frag = population_similarity(pa.population, pb.population, axes)
    elif pa.population and pb.population and axes:
        # [P2] The layout may declare one more graded axis than the writer
        # fills (the PEAK_CHAIN CLOSURE, which is a 0/1 activity).  The
        # strict comparison above needs an exact arity match; the aligned
        # comparison skips the axes neither side carries, so the fragment
        # keeps its graded reading and the closure starts to contribute as
        # soon as the writer fills it.
        frag = aligned_population_similarity(pa.population, pb.population,
                                             axes)
    coarse = _bins_sim(pa.bins, pb.bins, _ranges_for(len(pa.bins)))
    if frag is None:
        return coarse
    w = max(0.0, min(1.0, float(P.ASPECT_V2_COARSE_BIN_WEIGHT)))
    return (1.0 - w) * frag + w * coarse


def _polar_similarity(pa: PartDesc, pb: PartDesc) -> float:
    """Soft object-centred UPRIGHT polar kernel: radial x circular angular."""
    ra, ta = pa.polar if pa.polar is not None else _upright_polar(pa.pos)
    rb, tb = pb.polar if pb.polar is not None else _upright_polar(pb.pos)
    sr = max(1e-6, float(P.ASPECT_V2_POS_SIGMA_R))
    st = math.radians(max(1e-6, float(P.ASPECT_V2_POS_SIGMA_THETA_DEG)))
    dth = (ta - tb + math.pi) % (2.0 * math.pi) - math.pi
    return math.exp(-((ra - rb) / sr) ** 2) * math.exp(-(dth / st) ** 2)


def _circ_sim(a: float, b: float, sigma: float) -> float:
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return math.exp(-(d / max(sigma, 1e-6)) ** 2)


def _lin_sim(a: float, b: float, sigma: float) -> float:
    return math.exp(-((a - b) / max(sigma, 1e-6)) ** 2)


def _surface_similarity(
    sa: Optional[SurfaceDesc], sb: Optional[SurfaceDesc],
) -> Optional[float]:
    """Graded surface overlap over both sides of the fragment.

    Every PRESENT slot contributes equally (one vote); a slot unavailable on
    either side (stored as -1.0 / empty) is skipped, never compared as 0.
    Hue is down-weighted by the smaller saturation, so a grey pixel's
    fabricated hue cannot score.
    """
    if sa is None or sb is None:
        return None
    sig_hue = math.radians(max(1e-6, float(P.ASPECT_V2_SURFACE_SIGMA_HUE_DEG)))
    sig_sat = max(1e-6, float(P.ASPECT_V2_SURFACE_SIGMA_SAT))
    sig_lum = max(1e-6, float(P.ASPECT_V2_SURFACE_SIGMA_LUM))
    num = 0.0
    den = 0.0
    for hi_a, hi_b, si_a, si_b in (
        (sa.hue_i, sb.hue_i, sa.sat_i, sb.sat_i),
        (sa.hue_f, sb.hue_f, sa.sat_f, sb.sat_f),
    ):
        if hi_a >= 0.0 and hi_b >= 0.0 and si_a > 1e-9 and si_b > 1e-9:
            conf = min(si_a, si_b)
            num += conf * _circ_sim(hi_a, hi_b, sig_hue)
            den += conf
    for a, b in ((sa.sat_i, sb.sat_i), (sa.sat_f, sb.sat_f)):
        if a >= 0.0 and b >= 0.0:
            num += _lin_sim(a, b, sig_sat)
            den += 1.0
    for a, b in ((sa.lum_i, sb.lum_i), (sa.lum_f, sb.lum_f)):
        if a >= 0.0 and b >= 0.0:
            num += _lin_sim(a, b, sig_lum)
            den += 1.0
    for va, vb in (
        (sa.theta_i, sb.theta_i), (sa.theta_f, sb.theta_f),
        (sa.sf_i, sb.sf_i), (sa.sf_f, sb.sf_f),
    ):
        if va and vb and len(va) == len(vb):
            num += axis_similarity(va, vb)
            den += 1.0
    if den <= 1e-12:
        return None
    return num / den


def _part_similarity_v2(pa: PartDesc, pb: PartDesc, mode: str) -> float:
    if mode == "bins_only":
        order = (
            1.0 if (pa.bins and pb.bins and pa.bins[0] == pb.bins[0])
            else float(P.ASPECT_ORDER_MISMATCH)
        )
        return _bins_sim(pa.bins, pb.bins, _ranges_for(len(pa.bins))) * order
    shape = _fragment_similarity(pa, pb)
    if mode == "shape_only":
        return shape
    surface = _surface_similarity(pa.surface, pb.surface)
    if mode == "surface_only":
        return surface if surface is not None else 0.0
    if mode == "no_pos":
        terms = [(float(P.ASPECT_V2_W_SHAPE), shape)]
        if surface is not None:
            terms.append((float(P.ASPECT_V2_W_SURFACE), surface))
    else:  # "full"
        terms = [
            (float(P.ASPECT_V2_W_SHAPE), shape),
            (float(P.ASPECT_V2_W_POS), _polar_similarity(pa, pb)),
        ]
        if surface is not None:
            terms.append((float(P.ASPECT_V2_W_SURFACE), surface))
    wsum = sum(w for w, _v in terms)
    if wsum <= 1e-12:
        return 0.0
    return sum(w * v for w, v in terms) / wsum


# ---------------------------------------------------------------------------
# [V3] The label-free fusion.
#
# Each group gets a weight derived from TWO MEASURED, LABEL-FREE quantities:
#   reliability    -- how well the group matches an image to its OWN
#                     perturbed version (measured on the dev split);
#   distinctiveness-- how far the group's similarity separates DIFFERENT
#                     images (1 - the mean off-diagonal similarity relative
#                     to the mean diagonal one).
# The weight is their product, normalised. A noisy group therefore cannot
# dilute an informative one, and NO LABEL is used anywhere in the
# derivation. The values live in parameters (with their provenance) and
# are produced by diagnostics/_calibrate.py.
# ---------------------------------------------------------------------------

def fusion_weights() -> Dict[str, float]:
    """The calibrated group weights (label-free)."""
    raw = {
        "shape": float(P.ASPECT_V3_W_SHAPE),
        "arrangement": float(P.ASPECT_V3_W_ARRANGEMENT),
        "surface": float(P.ASPECT_V3_W_SURFACE),
        "layout": float(P.ASPECT_V3_W_LAYOUT),
        "size": float(P.ASPECT_V3_W_SIZE),
        "fragments": float(P.ASPECT_V3_W_FRAGMENTS),
        "pairs": float(P.ASPECT_V3_W_PAIRS),
        "texture2": float(P.ASPECT_V3_W_TEXTURE2),
        # [TIER + DEPTH] The two NEW groups (the middle tier and the static
        # depth cues). Their weights come from the SAME label-free
        # calibration as the others; a group that did not earn its weight
        # is retired to 0 exactly like the pair histogram was.
        "tier": float(getattr(P, "ASPECT_V3_W_TIER", 0.0)),
        "depth": float(getattr(P, "ASPECT_V3_W_DEPTH", 0.0)),
        # [P2] The RELATIONAL CONTEXT group.  Its weight lives in
        # memory/relational.py (the calibration that produced it is
        # diagnostics/_p2_calibrate.py), because parameters/_text.py is
        # owned by another mission.  It is LABEL-FREE, like every other
        # weight here: reliability x distinctiveness, both measured on the
        # dev split without a label.
        "relational": float(RELATIONAL_GROUP_WEIGHT),
    }
    raw = {k: max(0.0, v) for k, v in raw.items()}
    total = sum(raw.values())
    if total <= 1e-12:
        return {"shape": 0.25, "arrangement": 0.25, "surface": 0.25,
                "layout": 0.25}
    return {k: v / total for k, v in raw.items()}


def _relational_similarity(pa: PartDesc, pb: PartDesc) -> Optional[float]:
    """The RELATIONAL CONTEXT similarity of two parts, or None when either
    side carries no relational evidence."""
    from memory.relational import context_similarity
    return context_similarity(getattr(pa, "relations", None),
                              getattr(pb, "relations", None))


def _part_similarity_v3(pa: PartDesc, pb: PartDesc,
                        mode: str, w: Dict[str, float]) -> float:
    """The per-part similarity under a V3 arm, with the calibrated weights."""
    if mode == "bins_only":
        order = (
            1.0 if (pa.bins and pb.bins and pa.bins[0] == pb.bins[0])
            else float(P.ASPECT_ORDER_MISMATCH)
        )
        return _bins_sim(pa.bins, pb.bins, _ranges_for(len(pa.bins))) * order
    if mode == "relational_only":
        # [P2] The arm that keeps ONLY the relational context: the part's
        # similarity IS the overlap of the two relational contexts.
        rel = _relational_similarity(pa, pb)
        return 0.0 if rel is None else rel
    shape = _fragment_similarity(pa, pb)
    if mode == "shape_only":
        return shape
    surface = _surface_similarity(pa.surface, pb.surface)
    if mode == "surface_only":
        return surface if surface is not None else 0.0
    terms: list = []
    if mode in ("full", "no_layout"):
        terms.append((w.get("shape", 0.0), shape))
    if mode in ("full", "no_pos"):
        terms.append((w.get("arrangement", 0.0), _polar_similarity(pa, pb)))
    if mode == "full":
        if surface is not None:
            terms.append((w.get("surface", 0.0), surface))
    else:
        if surface is not None:
            terms.append((w.get("surface", 0.0), surface))
    # [P2] The RELATIONAL CONTEXT: the part's own mixture of the relation
    # families it sits in.  Absent evidence is ABSENT, not a zero: a part
    # with no relational context simply does not contribute this term.
    rel = _relational_similarity(pa, pb)
    if rel is not None and mode != "no_relational":
        terms.append((w.get("relational", 0.0), rel))
    wsum = sum(x for x, _v in terms)
    if wsum <= 1e-12:
        return 0.0
    return sum(x * v for x, v in terms) / wsum


def match_views_v3(
    A: ViewRecord, B: ViewRecord, mode: str = "full"
) -> List[Tuple[int, int, float]]:
    """Greedy bipartite part match under an explicit V3 ablation arm."""
    w = fusion_weights()
    sim = {
        (i, j): _part_similarity_v3(pa, pb, mode, w)
        for i, pa in enumerate(A.parts)
        for j, pb in enumerate(B.parts)
    }
    return _greedy_match(sim, float(P.ASPECT_MATCH_MIN))


def layout_similarity(A: ViewRecord, B: ViewRecord) -> float:
    """The GLOBAL-LAYOUT group's similarity (the soft grid + the moments +
    the radial and projection profiles)."""
    from memory.gist import gist_similarity
    return gist_similarity(getattr(A, "gist", None), getattr(B, "gist", None),
                           center_sigma=float(P.ASPECT_V3_LAYOUT_CENTER_SIGMA),
                           radius_sigma=float(P.ASPECT_V3_LAYOUT_RADIUS_SIGMA))


# ---------------------------------------------------------------------------
# [W2/D] The MIDDLE TIER: fragment PAIRS and second-order TEXTURE.
# ---------------------------------------------------------------------------
# Every group above is first-order and local. These two are not:
#
#   PAIRS   the arrangement RELATION BETWEEN fragments. A quadruped's four
#           legs make a characteristic pair pattern; a cup's rim and wall do
#           not. Measured as two L1-normalised histograms over the pairs:
#           the distance between the fragments and their relative bearing.
#
#   TEXTURE2 the SECOND-ORDER texture of a fragment: the circular
#           autocorrelation of its orientation energy at lags 1 and 2. A
#           first-order orientation histogram cannot tell a smooth hide from
#           a speckled coat; the autocorrelation can.
#
# Both are measured from what the view already carries. No category, no
# segmentation, no new sensor.
_PAIR_DIST_BINS = 8
_PAIR_ANG_BINS = 12
# The positions are RMS-normalised, so the object's own extent is ~1 and a
# pair rarely exceeds 2; the distance axis is scaled by that.
_PAIR_DIST_MAX = 2.0


def _spread(bins: List[float], pos: float, weight: float = 1.0) -> None:
    """Spread `weight` over the two bins straddling `pos` (linear).

    A circular axis wraps: the bins are indexed modulo their count, so the
    spread across the 0/2pi seam is continuous.
    """
    n = len(bins)
    if n == 0:
        return
    x = pos - 0.5
    i0 = int(math.floor(x))
    frac = x - i0
    bins[i0 % n] += weight * (1.0 - frac)
    bins[(i0 + 1) % n] += weight * frac


def _pair_histograms(parts: Sequence[PartDesc]
                     ) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    """The fragment-PAIR histogram: distance and relative bearing.

    Two L1-normalised distributions over the pairs. None when the view has
    fewer than two fragments: the pair histogram of a single point is not
    evidence.
    """
    pts: List[Tuple[float, float]] = []
    for p in parts:
        pol = p.polar if p.polar is not None else _upright_polar(p.pos)
        pts.append((float(pol[0]), float(pol[1])))
    n = len(pts)
    if n < 2:
        return None
    dist = [0.0] * _PAIR_DIST_BINS
    ang = [0.0] * _PAIR_ANG_BINS
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            d = math.hypot(dx, dy)
            a = math.atan2(dy, dx) % (2.0 * math.pi)
            dpos = min(float(_PAIR_DIST_BINS - 1),
                       d / _PAIR_DIST_MAX * _PAIR_DIST_BINS)
            apos = a / (2.0 * math.pi) * _PAIR_ANG_BINS
            _spread(dist, dpos)
            _spread(ang, apos)
    sd = sum(dist)
    sa = sum(ang)
    if sd <= 1e-12 or sa <= 1e-12:
        return None
    return (tuple(v / sd for v in dist), tuple(v / sa for v in ang))


def _hist_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Chance-corrected overlap of two L1-normalised histograms.

    The chance level of the intersection of two uniform distributions over n
    bins is 1/n; the excess over that is the evidence, rescaled to [0, 1].
    """
    if not a or len(a) != len(b):
        return 0.0
    m = sum(min(float(x), float(y)) for x, y in zip(a, b))
    chance = 1.0 / len(a)
    if chance >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (m - chance) / (1.0 - chance)))


def _pair_similarity(A: ViewRecord, B: ViewRecord) -> Optional[float]:
    ha = _pair_histograms(A.parts)
    hb = _pair_histograms(B.parts)
    if ha is None or hb is None:
        return None
    return 0.5 * (_hist_overlap(ha[0], hb[0]) + _hist_overlap(ha[1], hb[1]))


def _texture_autocorr(parts: Sequence[PartDesc]
                      ) -> Optional[Tuple[float, float]]:
    """Second-order texture: the circular autocorrelation of each fragment's
    orientation energy at lags 1 and 2, averaged over the fragments.

    None when no fragment carries an orientation distribution.
    """
    l1 = 0.0
    l2 = 0.0
    n = 0
    for p in parts:
        s = p.surface
        if s is None or not s.theta_i:
            continue
        th = list(s.theta_i)
        m = len(th)
        if m < 5:
            continue
        num1 = den1 = num2 = den2 = 0.0
        for k, v in enumerate(th):
            v1 = float(v)
            v2 = float(th[(k + 1) % m])
            v3 = float(th[(k + 2) % m])
            num1 += v1 * v2
            den1 += v1 * v1
            num2 += v1 * v3
            den2 += v1 * v1
        if den1 <= 1e-12 or den2 <= 1e-12:
            continue
        l1 += num1 / den1
        l2 += num2 / den2
        n += 1
    if n == 0:
        return None
    return (l1 / n, l2 / n)


def _texture2_similarity(A: ViewRecord, B: ViewRecord) -> Optional[float]:
    ta = _texture_autocorr(A.parts)
    tb = _texture_autocorr(B.parts)
    if ta is None or tb is None:
        return None
    d = 0.5 * (abs(ta[0] - tb[0]) + abs(ta[1] - tb[1]))
    return max(0.0, min(1.0, 1.0 - d))


def _group_nulls() -> Dict[str, float]:
    """Each group's own corpus null (measured, label-free)."""
    raw = getattr(P, "ASPECT_V3_GROUP_NULL", None)
    if not raw:
        return {}
    try:
        return {str(k): float(v) for k, v in dict(raw).items()}
    except Exception:
        return {}


def _null_corrected(value: float, group: str) -> float:
    """Chance-correct a group's similarity against ITS OWN corpus null.

    The groups have very different dynamic ranges, and a fusion over the RAW
    similarities lets the widest one dominate. The null is the mean
    similarity of that group over the corpus pairs (measured; no class label
    enters), and the corrected score is (value - null) / (1 - null).
    """
    null = _group_nulls().get(group)
    if null is None:
        return value
    null = max(0.0, min(0.999999, float(null)))
    return max(0.0, min(1.0, (float(value) - null) / (1.0 - null)))


def group_similarities(A: ViewRecord, B: ViewRecord,
                       pm: Optional[List[Tuple[int, int, float]]] = None
                       ) -> Dict[str, float]:
    """The similarity of EACH group between two views, separately.

    This is what the label-free calibration measures: a group's reliability
    (how well it survives a perturbation) and its distinctiveness (how far it
    separates different images) are read per group, so the fusion can weight
    them without any label.

    Each group is read on its OWN evidence, Dice-normalised over the two
    views' part counts (so an unmatched part still dilutes that group, as it
    should): 2 * sum(group similarity over the matched parts) / (|A|+|B|).
    A group with no matched evidence is ABSENT from the result rather than
    scored as 0.
    """
    out: Dict[str, float] = {}
    if pm is None:
        pm = match_views_v3(A, B, "full")
    total = len(A.parts) + len(B.parts)
    if pm and total > 0:
        shape_vals, arr_vals, surf_vals = [], [], []
        for i, j, _s in pm:
            pa, pb = A.parts[i], B.parts[j]
            shape_vals.append(_fragment_similarity(pa, pb))
            arr_vals.append(_polar_similarity(pa, pb))
            sv = _surface_similarity(pa.surface, pb.surface)
            if sv is not None:
                surf_vals.append(sv)
        if shape_vals:
            out["shape"] = (2.0 * sum(shape_vals)) / float(total)
        if arr_vals:
            out["arrangement"] = (2.0 * sum(arr_vals)) / float(total)
        if surf_vals:
            out["surface"] = (2.0 * sum(surf_vals)) / float(total)
    lay = layout_similarity(A, B)
    if lay > 0.0:
        out["layout"] = lay
    # The SIZE group: the object's extent, on its own kernel.
    out["size"] = _size_similarity(A, B)
    # The PEAK-FRAGMENT group: the soft 3-D population over the peak
    # fragments (the peaks recur across views; the chains' members do not).
    frag = _peak_fragment_similarity(getattr(A, "peak_map", None),
                                     getattr(B, "peak_map", None))
    if frag > 0.0:
        out["fragments"] = frag
    # [W2/D] The middle tier: the fragment-PAIR relation and the
    # second-order texture.
    pair = _pair_similarity(A, B)
    if pair is not None:
        out["pairs"] = pair
    tex2 = _texture2_similarity(A, B)
    if tex2 is not None:
        out["texture2"] = tex2
    # [TIER] The MIDDLE TIER: the assemblies over fragment populations
    # (the formations, their readout and their cohesion), the second-order
    # texture statistics, the multi-scale fragment map, and the pair
    # histogram re-tested INSIDE the tier.
    tier = None
    try:
        from memory.assemblies import tier_similarity
        tier = tier_similarity(A, B)
    except Exception:
        tier = None
    if tier is not None:
        out["tier"] = tier
    # [DEPTH] The STATIC DEPTH CUES: the texture-gradient vector, the
    # shading field, the blur level and the contrast/saturation fall-off
    # (each admitted only after its own repeatability gate).
    depth = None
    try:
        from memory.depth_cues import depth_similarity
        depth = depth_similarity(A, B)
    except Exception:
        depth = None
    if depth is not None:
        out["depth"] = depth
    # [P2] The RELATIONAL CONTEXT group: the VIEW-LEVEL relational
    # mixture (the whole graph's relation families), and the same group's
    # per-part reading folded into the part similarity.  Absent evidence is
    # absent, not a zero.
    rel = None
    try:
        from memory.relational import context_similarity
        rel = context_similarity(getattr(A, "relational", None),
                                 getattr(B, "relational", None))
    except Exception:
        rel = None
    if rel is not None:
        out["relational"] = rel
    return out


def match_views_v2(
    A: ViewRecord, B: ViewRecord, mode: str = "full"
) -> List[Tuple[int, int, float]]:
    """Greedy bipartite part match under an explicit V2 ablation arm."""
    if mode not in _V2_MODES:
        raise ValueError(f"unknown descriptor mode {mode!r}")
    sim = {
        (i, j): _part_similarity_v2(pa, pb, mode)
        for i, pa in enumerate(A.parts)
        for j, pb in enumerate(B.parts)
    }
    return _greedy_match(sim, float(P.ASPECT_MATCH_MIN))


def _polar_hist_of(parts: Sequence[PartDesc]) -> Tuple[float, ...]:
    return _soft_polar_hist(
        [(p.polar if p.polar is not None else _upright_polar(p.pos)) for p in parts],
        int(P.ASPECT_V2_POLAR_BINS),
    )


def resonance_v2(A: ViewRecord, B: ViewRecord, mode: str = "full") -> float:
    """Sorensen-Dice resonance under a V2 arm, with the separate size slot.

    The view-level soft polar histogram is a GLOBAL arrangement term and is
    applied only to `full`; the `no_pos` ablation must not silently regain
    position information through it.
    """
    pm = match_views_v2(A, B, mode)
    total = len(A.parts) + len(B.parts)
    if total == 0 or not pm:
        return 0.0
    base = (2.0 * sum(s for _i, _j, s in pm)) / float(total)
    base *= _size_similarity(A, B)
    if mode == "full":
        w = max(0.0, min(1.0, float(P.ASPECT_V2_W_POLAR_HIST)))
        base *= (1.0 - w) + w * axis_similarity(
            _polar_hist_of(A.parts), _polar_hist_of(B.parts))
    return base


def resonance_v3(A: ViewRecord, B: ViewRecord, mode: str = "full",
                 pm: Optional[List[Tuple[int, int, float]]] = None,
                 gs: Optional[Dict[str, float]] = None) -> float:
    """The V3 resonance: the calibrated fusion of the descriptor's groups.

    Each group contributes ITS OWN similarity, weighted by its calibrated
    weight, and the groups are never mixed before they are weighted: a
    group that is noisy (or uninformative) therefore cannot dilute an
    informative one, which is the whole point of the label-free weights.
    The part-level groups are Dice-normalised over the two views' part
    counts, so an unmatched part still costs that group its share.

    `pm` is the part correspondence; when it is not given it is computed
    here. Passing it lets a caller share ONE correspondence across the
    ablation arms, so the arms differ only in their scoring.

    The arms drop whole groups: `no_pos` drops the arrangement, `no_layout`
    drops the global layout, `shape_only` / `surface_only` / `layout_only` /
    `bins_only` keep a single group.
    """
    if mode == "legacy":
        # The reference arm must be the LEGACY matcher, not the V3: routing
        # it through `_resonance(..., None)` would pick the V3 up (the
        # V3 switch is on) and the arm would silently duplicate "full".
        return AspectGraph._legacy_resonance(A, B)
    w = fusion_weights()
    if mode == "layout_only":
        return _null_corrected(layout_similarity(A, B), "layout")
    if pm is None:
        pm = match_views_v3(A, B, mode)
    if not pm:
        return 0.0
    if mode == "bins_only":
        # The coarse index alone: the six quantised slots, with the
        # ensemble-order gate the legacy reader used.
        total = len(A.parts) + len(B.parts)
        vals = []
        for i, j, _s in pm:
            pa, pb = A.parts[i], B.parts[j]
            order = (1.0 if (pa.bins and pb.bins and pa.bins[0] == pb.bins[0])
                     else float(P.ASPECT_ORDER_MISMATCH))
            vals.append(_bins_sim(pa.bins, pb.bins,
                                  _ranges_for(len(pa.bins))) * order)
        if not vals or total <= 0:
            return 0.0
        return (2.0 * sum(vals)) / float(total)
    if gs is None:
        gs = group_similarities(A, B, pm)
    drop = {"no_pos": "arrangement", "no_layout": "layout",
            "no_shape": "shape", "no_size": "size",
            "no_fragments": "fragments",
            "no_pairs": "pairs",
            "no_texture2": "texture2",
            "no_tier": "tier",
            "no_depth": "depth",
            "no_relational": "relational"}.get(mode)
    if mode == "no_new":
        # the baseline arm: the fusion WITHOUT the two new groups (the
        # middle tier and the depth cues), i.e. the tree's shipped fusion.
        drop = ("tier", "depth")
    only = {"shape_only": "shape", "surface_only": "surface",
            "layout_only": "layout", "pairs_only": "pairs",
            "texture2_only": "texture2",
            "tier_only": "tier", "depth_only": "depth",
            "relational_only": "relational"}.get(mode)
    num = 0.0
    den = 0.0
    for g, v in gs.items():
        if drop is not None and g == drop:
            continue
        if isinstance(drop, tuple) and g in drop:
            continue
        if only is not None and g != only:
            continue
        weight = w.get(g, 0.0)
        if weight <= 0.0:
            continue
        # [W2/F] Every group is chance-corrected against its OWN corpus
        # null before it is weighted, so the group with the widest
        # dynamic range cannot dominate the fusion.
        num += weight * _null_corrected(v, g)
        den += weight
    if den <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, num / den))


def _dOmega(a_az: float, a_el: float, b_az: float, b_el: float) -> Tuple[float, float, float]:
    """Calculate (delta_azimuth in [-180, 180), delta_elevation, great_circle_distance_deg)."""
    daz = (b_az - a_az + 180.0) % 360.0 - 180.0
    ea, eb = math.radians(a_el), math.radians(b_el)
    c = math.sin(ea) * math.sin(eb) + math.cos(ea) * math.cos(eb) * math.cos(math.radians(daz))
    dist = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    return daz, b_el - a_el, dist

class AspectGraph:
    """Aspect Graph Memory Engine.
    Clusters views into canonical Aspect Prototypes via ART resonance,
    links adjacent viewpoints with persistent transformation traces,
    and synthesizes virtual halfway views to support unseen view classification.
    """

    def __init__(self) -> None:
        self.views: List[ViewRecord] = []
        self.protos: Dict[int, Dict[str, Any]] = {}
        self.links: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.diag: Counter = Counter()
        self._by_instance: Dict[str, List[int]] = defaultdict(list)
        # [O7] The view ids the horizon evicted, for the caller's log.
        self.evicted_views: List[int] = []
        # [O7] The eviction is FROZEN during an evaluation window: the
        # store must not change under a measurement.
        self.eviction_frozen: bool = False

    @staticmethod
    def make_view(
        desc: Tuple[List[PartDesc], List[Tuple[float, ...]]],
        instance: str = "",
        label: Any = None,
        az: float = 0.0,
        el: float = 0.0,
        vid: int = -1,
        t: int = 0,
    ) -> ViewRecord:
        size = float(desc[1]) if len(desc) > 1 else 1.0
        return ViewRecord(vid, t, instance, label, az, el, desc[0],
                          size=size,
                             gist=(desc[3] if len(desc) > 3 else None),
                          peak_map=(desc[4] if len(desc) > 4 else None),
                          # [P2] the view-level relational context
                          relational=(desc[5] if len(desc) > 5 else None))

    @staticmethod
    def _resonance(A: ViewRecord, B: ViewRecord, mode: Optional[str] = None) -> float:
        """Balanced Sørensen–Dice resonance over matched parts:
        Resonance = (2 * sum(matched_sim)) / (|A.parts| + |B.parts|)
        Tolerates oversegmentation noise while strictly penalizing missing structures.

        [SIZE SLOT] The result is scaled by the SEPARATE size-similarity kernel
        (`_size_similarity`): size is its own slot with its own switch, never
        merged into the normalized shape/position terms.

        [W2/O7] `mode` selects the population-coded arm; None keeps the
        legacy matcher unless `ASPECT_V3_ENABLED` is switched on -- and then
        the LIVE arm is the V3 fusion (the calibrated groups), not the V2 one.
        """
        if mode == "legacy":
            return AspectGraph._legacy_resonance(A, B)
        if mode is not None or bool(getattr(P, "ASPECT_V3_ENABLED", False)):
            arm = mode or "full"
            if arm in _V3_MODES:
                return resonance_v3(A, B, arm)
            return resonance_v2(A, B, arm)
        return AspectGraph._legacy_resonance(A, B)

    @staticmethod
    def _legacy_resonance(A: ViewRecord, B: ViewRecord) -> float:
        """The legacy matcher: the ensemble-aware part matching and the
        multiplicative size slot. It stays available as the `legacy` arm (the
        ablation baseline) and for the legacy machinery that has no V3
        equivalent (the virtual view).
        """
        _em, pm = match_views(A, B, mode="legacy")
        total_parts = len(A.parts) + len(B.parts)
        if total_parts == 0 or not pm:
            return 0.0
        base = (2.0 * sum(s for _i, _j, s, _l in pm)) / float(total_parts)
        return base * _size_similarity(A, B)

    def observe(
        self, t: int, instance: str, label: Any, az: float, el: float, desc: Optional[Tuple[List[PartDesc], List[Tuple[float, ...]]]]
    ) -> Optional[ViewRecord]:
        if not P.ASPECT_ENABLED or desc is None:
            return None
        # [LIVE] The GLOBAL-LAYOUT group (desc[4]) and the polar
        # histogram (desc[3]) are part of the view: the readout scores
        # with them, so dropping them here made the live scores blind to
        # the layout the system measures.
        v = ViewRecord(len(self.views), int(t), str(instance), label,
                       float(az), float(el), desc[0],
                       size=(float(desc[1]) if len(desc) > 1 else 1.0),
                       gist=(desc[3] if len(desc) > 3 else None),
                       peak_map=(desc[4] if len(desc) > 4 else None),
                       # [P2] the view-level relational context
                       relational=(desc[5] if len(desc) > 5 else None))
        v.proto = self._assign(v)
        self._decay()
        for uid in self._link_candidates(v):
            self._trace_link(self.views[uid], v)
        self.views.append(v)
        self._by_instance[v.instance].append(v.vid)
        # [O7] THE VIEW HORIZON.  The view store was UNBOUNDED: every
        # view ever observed was retained in full (parts, surface, gist,
        # peak map), so the system's largest memory sink grew without
        # limit.  The horizon evicts the OLDEST views, and it protects a
        # RARE prototype: a prototype with a single member is the one
        # the store can least afford to lose, because losing its member
        # loses the prototype.
        self._evict_to_horizon()
        return v

    def _evict_to_horizon(self) -> None:
        """[O7] The view store's capacity valve.  Logged through the diag
        counter; nothing ever leaves the store unlogged."""
        cap = int(getattr(P, "ASPECT_MAX_VIEWS", 4096))
        if self.eviction_frozen or cap <= 0 or len(self.views) <= cap:
            return
        # The members of each prototype: a prototype with one member is
        # RARE and its member is protected.
        members: Dict[int, int] = defaultdict(int)
        for v in self.views:
            if v.proto >= 0:
                members[v.proto] += 1
        victims: List[int] = []
        for v in self.views:
            if len(victims) >= (len(self.views) - cap):
                break
            if v.proto >= 0 and members.get(v.proto, 0) <= 1:
                continue
            victims.append(v.vid)
        if not victims:
            self.diag["view_horizon_all_protected"] += 1
            return
        victim_set = set(victims)
        self.views = [v for v in self.views if v.vid not in victim_set]
        for inst, vids in list(self._by_instance.items()):
            kept = [x for x in vids if x not in victim_set]
            if kept:
                self._by_instance[inst] = kept
            else:
                del self._by_instance[inst]
        # The links that referenced an evicted view go with it.
        for key in [k for k in self.links
                    if k[0] in victim_set or k[1] in victim_set]:
            del self.links[key]
        self.diag["views_evicted"] += len(victims)
        self.diag["view_horizon_evictions"] += 1
        self.evicted_views = list(victims)

    def _link_candidates(self, v: ViewRecord) -> List[int]:
        if P.ASPECT_PAIR_MODE == "temporal":
            return [self.views[-1].vid] if self.views and v.t - self.views[-1].t <= 1 else []
        return [
            u
            for u in self._by_instance[v.instance]
            if _dOmega(self.views[u].az, self.views[u].el, v.az, v.el)[2] <= P.ASPECT_LINK_MAX_DEG
        ]

    def _assign(self, v: ViewRecord) -> int:
        best_pid: Optional[int] = None
        best_r = -1.0
        for pid in sorted(self.protos):
            p = self.protos[pid]
            if _dOmega(p["az"], p["el"], v.az, v.el)[2] > P.ASPECT_ASSIGN_MAX_DEG:
                continue
            if v.label is not None and p["labels"]:
                majority_label = sorted(p["labels"].items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]
                if majority_label != v.label:
                    continue  # Match tracking: prevent cross-class prototype pollution
            r = self._resonance(v, p["rep"])
            if r > best_r:
                best_pid = pid
                best_r = r

        if best_pid is not None and best_r >= P.ASPECT_VIGILANCE:
            p = self.protos[best_pid]
            n = p["n"]
            # Circular mean update for prototype viewpoint
            rad_p = math.radians(p["az"])
            rad_v = math.radians(v.az)
            # Spherical EMA: treats existing mean as n unit vectors.
            # Inflates weight of established prototypes to resist outlier drift.
            x = math.cos(rad_p) * n + math.cos(rad_v)
            y = math.sin(rad_p) * n + math.sin(rad_v)
            p["az"] = math.degrees(math.atan2(y, x)) % 360.0
            # Re-wrap: a tiny-negative atan2 result rounds to exactly 360.0
            # under the float modulo above; keep azimuth in [0, 360).
            p["az"] %= 360.0
            p["el"] = (p["el"] * n + v.el) / (n + 1)
            p["n"] = n + 1
            if v.label is not None:
                p["labels"][v.label] += 1
            p["views"].append(v.vid)
            # [O5d] The prototype's MEAN, VARIABILITY and RELIABILITY are
            # updated from the evidence of this recall.
            self._update_prototype_stats(p, v, best_r)
            self.diag["assign_existing"] += 1
            return best_pid

        new_pid = len(self.protos)
        self.protos[new_pid] = {
            "rep": v,
            "az": v.az,
            "el": v.el,
            "n": 1,
            "labels": Counter([v.label] if v.label is not None else []),
            "views": [v.vid],
            # [O5d] The prototype's own statistics start from its first
            # member, so the mean, the variability and the reliability are
            # defined from the first recall onwards.
            "score_n": 0,
            "score_sum": 0.0,
            "score_sq": 0.0,
            "score_mean": 0.0,
            "score_var": 0.0,
            "reliability": 0.0,
            "size_mean": float(v.size),
            "size_var": 0.0,
            "size_n": 0,
            "part_mean": {},
            "part_n": {},
        }
        self.diag["assign_new"] += 1
        return new_pid

    def _update_prototype_stats(self, p: Dict[str, Any], v: ViewRecord,
                                score: float, mode: Optional[str] = None) -> None:
        """[O5d] THE PROTOTYPE'S MEAN, VARIABILITY AND RELIABILITY.

        A prototype that is only its FIRST exemplar plus a moving camera
        angle is not a prototype: it cannot say how typical a member is,
        nor how much its members disagree with each other.  On every
        successful recall (a view that resonated with the prototype at or
        above the vigilance) three statistics are updated from the
        EVIDENCE of that recall:

          * mean: the running mean of the members' resonance scores;
          * variability: the running variance of those scores;
          * reliability: an EMA that rises with consistent support and
            falls when a member's evidence contradicts the prototype.

        Plus the prototype's per-part MEAN, over a REP-ANCHORED frame:
        every member is matched against the SAME exemplar, so the
        exemplar's part index is a consistent index for the whole
        prototype, and the mean of each part's position and orientation is
        well defined.  The mean is recorded, not yet used for matching
        (see the note in the mission log: the matcher does not yet expose
        a per-part similarity against the MEAN, only against the
        exemplar).

        Deterministic: no randomness, and the same recalls in the same
        order give the same numbers.  LOGGED through the diag counter.
        The BINS stay first-write-wins: the prototype's identity -- its
        exemplar -- never changes.
        """
        try:
            n = int(p.get("score_n", 0))
            s = float(p.get("score_sum", 0.0))
            sq = float(p.get("score_sq", 0.0))
            n += 1
            s += float(score)
            sq += float(score) * float(score)
            p["score_n"] = n
            p["score_sum"] = s
            p["score_sq"] = sq
            p["score_mean"] = s / n
            var = max(0.0, (sq - (s * s) / n) / n)
            p["score_var"] = var
            # Reliability: an EMA of the evidence, discounted by the
            # VARIABILITY -- a prototype whose members disagree is less
            # reliable than one whose members agree at the same mean.
            kappa = float(getattr(P, "ASPECT_PROTO_RELIABILITY_KAPPA", 0.2))
            rel = float(p.get("reliability", 0.0))
            p["reliability"] = float(min(1.0, max(0.0,
                rel + kappa * (float(score) - var - rel))))

            # The SIZE slot's own statistics: the prototype's typical
            # size and how much its members differ in size.
            sn = int(p.get("size_n", 0))
            if sn == 0:
                p["size_mean"] = float(v.size)
                p["size_var"] = 0.0
                p["size_n"] = 1
            else:
                prev_mean = float(p.get("size_mean", v.size))
                prev_var = float(p.get("size_var", 0.0))
                new_mean = prev_mean + (float(v.size) - prev_mean) / (sn + 1)
                new_var = (prev_var * sn
                           + (float(v.size) - prev_mean) * (float(v.size) - new_mean)) / (sn + 1)
                p["size_mean"] = new_mean
                p["size_var"] = max(0.0, new_var)
                p["size_n"] = sn + 1

            # The per-part MEAN, over the REP-ANCHORED frame.
            part_mean: Dict[int, List[float]] = p.setdefault("part_mean", {})
            part_n: Dict[int, int] = p.setdefault("part_n", {})
            try:
                m1, m2 = match_views(v, p["rep"], mode)
                matched = m1 if m1 else m2
                for pair in matched:
                    try:
                        _i, j = int(pair[0]), int(pair[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if not (0 <= j < len(v.parts) and 0 <= j < len(p["rep"].parts)):
                        continue
                    src = v.parts[j]
                    cur = part_mean.get(j)
                    if cur is None:
                        part_mean[j] = [float(src.pos[0]), float(src.pos[1]),
                                        float(src.ang)]
                        part_n[j] = 1
                        continue
                    k = part_n.get(j, 1)
                    k += 1
                    part_n[j] = k
                    # The orientation is AXIAL: its mean is a circular mean.
                    th = float(src.ang)
                    cur_t = cur[2]
                    d = (th - cur_t + math.pi / 2.0) % math.pi - math.pi / 2.0
                    cur[0] = cur[0] + (float(src.pos[0]) - cur[0]) / k
                    cur[1] = cur[1] + (float(src.pos[1]) - cur[1]) / k
                    cur[2] = (cur_t + d / k) % math.pi
            except Exception:
                # The part-level mean is a diagnostic refinement; a matcher
                # that cannot be called must not break the assignment.
                self.diag["proto_part_mean_skipped"] += 1
            self.diag["proto_stats_update"] += 1
        except Exception:
            self.diag["proto_stats_failed"] += 1

    def _decay(self) -> None:
        for L in self.links.values():
            L["w"] *= P.ASPECT_TRACE_DECAY

    def _trace_link(self, U: ViewRecord, V: ViewRecord) -> None:
        pu, pv = U.proto, V.proto
        if pu == pv:
            self.diag["link_same_proto"] += 1
            return
        daz, dele, ang = _dOmega(U.az, U.el, V.az, V.el)
        # [F6] The link machinery must use the CALIBRATED matcher: with
        # no mode, `match_views` routes to the uncalibrated V2 arm
        # whenever the V3 switch is on.
        pm = match_views_v3(U, V, "full")
        a, b, flip = (pu, pv, False) if pu < pv else (pv, pu, True)
        if flip:
            daz, dele = -daz, -dele
        L = self.links.setdefault(
            (a, b),
            {
                "w": 0.0,
                "n": 0,
                "daz": 0.0,
                "del": 0.0,
                "m2az": 0.0,
                "m2el": 0.0,
                "persist": {},
                "virt": None,
            },
        )
        prox = math.exp(-(ang / P.ASPECT_LINK_SIGMA_DEG) ** 2)
        L["w"] += P.ASPECT_TRACE_ETA * prox * (1.0 - L["w"] / P.ASPECT_W_MAX)
        L["n"] += 1

        d1 = daz - L["daz"]
        L["daz"] += d1 / L["n"]
        L["m2az"] += d1 * (daz - L["daz"])
        d2 = dele - L["del"]
        L["del"] += d2 / L["n"]
        L["m2el"] += d2 * (dele - L["del"])

        # The match tuple is (i, j, similarity) in the graded matcher and
        # (i, j, similarity, level) in the legacy one; the level is not read
        # here, so both shapes are accepted.
        for entry in pm:
            i, j, _s = entry[0], entry[1], entry[2]
            pa, pb = U.parts[i], V.parts[j]
            if flip:
                pa, pb = pb, pa
            e = L["persist"].setdefault((pa.bins, pb.bins), {"n": 0, "dx": 0.0, "dy": 0.0, "da": 0.0})
            e["n"] += 1
            k = 1.0 / e["n"]
            e["dx"] += k * ((pb.pos[0] - pa.pos[0]) - e["dx"])
            e["dy"] += k * ((pb.pos[1] - pa.pos[1]) - e["dy"])
            e["da"] += k * (_awrap(pb.ang - pa.ang) - e["da"])

        while len(L["persist"]) > P.ASPECT_PERSIST_CAP:
            victim_key = min(L["persist"], key=lambda kk: (L["persist"][kk]["n"], repr(kk)))
            L["persist"].pop(victim_key)

        L["virt"] = None  # Invalidate cached interpolated view
        self.diag["links_updated"] += 1

    def _virtual(self, key: Tuple[int, int], L: Dict[str, Any]) -> ViewRecord:
        """Synthesize intermediate virtual view from the persistence map across prototypes."""
        if L["virt"] is not None:
            return L["virt"]
        A = self.protos[key[0]]["rep"]
        B = self.protos[key[1]]["rep"]
        # The virtual view is the LEGACY machinery (the demoted
        # interpolation): it interpolates the legacy matcher's part
        # correspondence, so it reads the legacy matcher explicitly.
        _em, pm = match_views(A, B, mode="legacy")

        parts: List[PartDesc] = []
        ua: Set[int] = set()
        ub: Set[int] = set()

        # (i, j, similarity) in the graded matcher, (i, j, similarity, level)
        # in the legacy one; the level is not read here.
        for entry in pm:
            i, j = entry[0], entry[1]
            pa, pb = A.parts[i], B.parts[j]
            parts.append(
                PartDesc(
                    bins=pa.bins,
                    pos=(0.5 * (pa.pos[0] + pb.pos[0]), 0.5 * (pa.pos[1] + pb.pos[1])),
                    ang=(pa.ang + 0.5 * _awrap(pb.ang - pa.ang)) % math.pi,
                )
            )
            ua.add(i)
            ub.add(j)

        # Append unmapped residual parts from both viewpoints
        parts.extend([PartDesc(p.bins, p.pos, p.ang) for i, p in enumerate(A.parts) if i not in ua])
        parts.extend([PartDesc(p.bins, p.pos, p.ang) for j, p in enumerate(B.parts) if j not in ub])

        # Exact circular mean for virtual view azimuth
        rad_a = math.radians(A.az)
        rad_b = math.radians(B.az)
        ax = math.cos(rad_a) + math.cos(rad_b)
        ay = math.sin(rad_a) + math.sin(rad_b)
        virt_az = math.degrees(math.atan2(ay, ax)) % 360.0
        # Re-wrap: a tiny-negative atan2 result rounds to exactly 360.0 under
        # the float modulo above; keep azimuth in [0, 360).
        virt_az %= 360.0

        L["virt"] = ViewRecord(
            vid=-1,
            t=0,
            instance="",
            label=None,
            az=virt_az,
            el=0.5 * (A.el + B.el),
            parts=parts,
            size=0.5 * (float(getattr(A, "size", 1.0))
                        + float(getattr(B, "size", 1.0))),
        )
        return L["virt"]

    def category_scores(
        self,
        desc: Tuple[List[PartDesc], List[Tuple[float, ...]]],
        use_links: bool = True,
        use_virtual: bool = False,
        descriptor_mode: Optional[str] = None,
    ) -> Dict[Any, float]:
        """Evaluate category recognition score using prototypes and (optionally)
        virtual views.

        [Aspect-graph demotion, 2026-10-10] `use_virtual` defaults to False:
        the midpoint-interpolated virtual view is no longer part of the scoring
        path. Evidence: on the ETH-80 equator ring the virtual view did not
        beat the better flank on F (-0.017 [-0.026, -0.009]); its apparent
        coverage win was a part-set-size effect (the mismatched-pair control
        absorbed ~80% of it); the true interpolation gain shrank as the
        position kernel tightened; and matched-part "motion" was null-sized
        against a same-viewpoint cross-instance control (0.159R within vs
        0.193R across), i.e. the stored displacement is matcher noise.

        So the claim supported by the data is about THIS construction
        (midpoint interpolation of shock-ribbon descriptors at this
        resolution), not "views cannot be interpolated" in general.

        Links and their persistence maps remain maintained as a
        transition-statistics store (part-type persistence and per-type mean
        motion); nothing in the default scoring path reads them.
        """
        q = ViewRecord(-1, 0, "", None, 0.0, 0.0, desc[0],
                       size=(float(desc[1]) if len(desc) > 1 else 1.0),
                       gist=(desc[3] if len(desc) > 3 else None),
                       peak_map=(desc[4] if len(desc) > 4 else None),
                       # [P2] the view-level relational context
                       relational=(desc[5] if len(desc) > 5 else None))
        scores: Dict[Any, float] = defaultdict(float)

        def add(labels: Counter, r: float) -> None:
            tot = sum(labels.values())
            if tot <= 0:
                return
            for lab, c in labels.items():
                scores[lab] = max(scores[lab], r * (c / tot))

        # Direct prototype resonance
        for pid in sorted(self.protos):
            if self.protos[pid]["labels"]:
                add(self.protos[pid]["labels"],
                    self._resonance(q, self.protos[pid]["rep"], descriptor_mode))

        # Link-induced virtual view resonance (demoted: off by default)
        if use_links and use_virtual:
            for key in sorted(self.links):
                L = self.links[key]
                if L["w"] < P.ASPECT_LINK_MIN_W:
                    continue
                labs = self.protos[key[0]]["labels"] + self.protos[key[1]]["labels"]
                if labs:
                    weight_scale = math.sqrt(min(1.0, L["w"] / P.ASPECT_W_MAX))
                    virt_resonance = self._resonance(
                    q, self._virtual(key, L), descriptor_mode)
                    add(labs, weight_scale * virt_resonance)

        return dict(scores)

    def summary(self) -> Dict[str, Any]:
        ws = [L["w"] for L in self.links.values()]
        return {
            "views": len(self.views),
            "protos": len(self.protos),
            "links": len(self.links),
            "mean_w": (sum(ws) / len(ws)) if ws else 0.0,
            **dict(self.diag),
        }

