"""perception/interior_field.py -- bounded-diffusion interior/exterior field.

THE GAP THIS FILLS. The representation had no per-pixel figure-ground signal:
regions carried a single hard `interior=(x,y)` point from polygon containment
(`perception/chain_regions.py`), and `BO_WEIGHTS` in `parameters/_text.py`
(reserved keys `enclosure`, `convexity`, `symmetry`, ...) was DEFINED but never
consumed by any lens. Enclosure -- "which side of this boundary is the object we
own" -- therefore had no source.

WHAT IT COMPUTES. A SOFT, signed interior(+)/exterior(-) scalar field. For every
pixel we ask how much EXTRA path cost the edges impose on getting to the image
frame, against the same trip through an edge-FREE image:

    geo_barrier = geodesic distance from the frame where crossing an edge costs
                  (1 + INTERIOR_EDGE_PENALTY * edge) per pixel entered,
    geo_free    = the same geodesic distance when every pixel costs 1,
    interior    = clip((geo_barrier - geo_free) / INTERIOR_ENCLOSURE_SCALE, 0, 1)
    signed      = 2*interior - 1                     (+ inside, - outside)

This is why it fills: any point sealed inside a closed high-edge boundary must
pay the extra edge cost to escape, so its interiorness is > 0 across the WHOLE
enclosed area. It is why a blank image reads 0 EVERYWHERE (the two trips are
identical -- enclosure must be EARNED by edges, and a finite-A artifact of a
naive bounded diffusion is impossible). And it is why the field is SOFT: the
geodesic is computed on a coarse working grid and bilinearly upsampled, so the
boundary transition is graded rather than a hard mask.

WHY A BOUNDED DIFFUSION, NOT A HARMONIC SOLVE. The enclosure signal lives in the
FINITE-TIME spread of the frame value; the steady state of `div(g grad u)=0` with
u=1 on the frame is u==1 everywhere for ANY positive conductance g, i.e. it
carries no enclosure at all. So the field is a bounded (fixed-sweep) min-plus
relaxation, the discrete analogue of a bounded diffusion.

CONTRACT (the mission's requirements for every mechanism):
  * deterministic; no randomness, no per-pixel ODE.
  * bounded compute: O((h+w) * hw) sweeps on a grid of longest side
    INTERIOR_WORK_SIZE, with early exit when the relaxation stops changing.
  * pure function of (luminance, parameters); memoised by content hash +
    front-end fingerprint (mirrors lenses/peak_lens.py::_field_key).
  * scale-aware BY CONSTRUCTION: the working grid is a fixed size, so the same
    INTERIOR_ENCLOSURE_SCALE means the same thing at every image resolution.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import parameters as P

_INF = 1.0e18


def _params() -> Tuple[float, float, int, int, float]:
    return (
        float(P.INTERIOR_EDGE_PENALTY),
        float(P.INTERIOR_ENCLOSURE_SCALE),
        int(P.INTERIOR_WORK_SIZE),
        int(P.INTERIOR_FIELD_CACHE_SIZE),
        float(P.INTERIOR_EDGE_FLOOR_PCT),
    )


def interior_fingerprint() -> str:
    """Stable digest of everything that changes the field, so a cache keyed on
    content hash can never hand back a field built under other parameters."""
    extra = (getattr(P, "INTERIOR_EDGE_NOISE_K", 2.0),
             getattr(P, "INTERIOR_EDGE_SCALE_PCT", 90.0), "v2-maxpool-cornercut")
    return "|".join(str(v) for v in _params() + extra)


def _edge_magnitude(lum: np.ndarray) -> np.ndarray:
    """Gradient magnitude in [0, 1] with a MEASURED noise floor.

    floor = median + k * 1.4826 * MAD of the gradient magnitude -- the frame's own
    noise level at k-sigma significance (k=2: the largest k whose
    boundary floor still admits a contour of ~3x the texture amplitude; k=3
    put the floor ON such a boundary and the enclosure leaked) (INTERIOR_EDGE_NOISE_K, default 2).
    scale = INTERIOR_EDGE_SCALE_PCT (default 90) percentile of the magnitudes ABOVE
    the floor, i.e. what real edges look like in THIS frame.

    The previous pair (floor = p95, scale = p95) had a span of exactly zero, so the
    map collapsed to a binary 'top 5% of pixels by rank' whatever the image held:
    0.5% luminance noise put barriers on 5% of the background. Here nothing is
    forced to occupy a fixed fraction of the frame, and the output is continuous.
    """
    from scipy.ndimage import gaussian_filter

    sm = gaussian_filter(lum.astype(np.float64), sigma=1.0, mode="nearest")
    gy, gx = np.gradient(sm)
    mag = np.hypot(gx, gy)
    if mag.size == 0:
        return np.zeros_like(mag)
    k = float(getattr(P, "INTERIOR_EDGE_NOISE_K", 2.0))
    top = float(getattr(P, "INTERIOR_EDGE_SCALE_PCT", 90.0))
    med = float(np.median(mag))
    mad = float(np.median(np.abs(mag - med)))
    floor = med + k * 1.4826 * mad
    above = mag[mag > floor]
    if above.size == 0:
        return np.zeros_like(mag)
    scale = float(np.percentile(above, top))
    return np.clip((mag - floor) / max(scale - floor, 1e-12), 0.0, 1.0)


def _frame_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m


def _chamfer(cost: np.ndarray) -> np.ndarray:
    """Geodesic distance from the frame through per-pixel traversal `cost`
    (8-neighbour chamfer / min-plus relaxation, diagonal step sqrt(2)).

    A DIAGONAL step may not slip between two diagonally-adjacent barrier pixels:
    it pays max(cost[target], min(cost[a], cost[b])) where a, b are the two
    4-neighbours shared by source and target. Without this rule a 1-px, 8-connected
    diagonal barrier is crossable for free (measured: extra path cost 0.41 vs 8.0
    for the same barrier axis-aligned). In an edge-free image every cost is 1 and
    the result is bit-identical to the plain chamfer, so geo_barrier - geo_free
    remains a like-for-like comparison.
    """
    h, w = cost.shape
    d = np.full((h, w), _INF, dtype=np.float64)
    frame = _frame_mask(h, w)
    d[frame] = 0.0
    sqrt2 = math.sqrt(2.0)
    pc = np.pad(cost, 1, constant_values=_INF)
    dcost = {}
    for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        ca = pc[1 + dy:1 + dy + h, 1:1 + w]
        cb = pc[1:1 + h, 1 + dx:1 + dx + w]
        dcost[(dy, dx)] = np.maximum(cost, np.minimum(ca, cb))

    def shifted(dy: int, dx: int) -> np.ndarray:
        out = np.full_like(d, _INF)
        ys = slice(max(0, -dy), h - max(0, dy))
        xs = slice(max(0, -dx), w - max(0, dx))
        yd = slice(max(0, dy), h - max(0, -dy))
        xd = slice(max(0, dx), w - max(0, -dx))
        out[ys, xs] = d[yd, xd]            # out[y, x] = d[y + dy, x + dx]
        return out

    cap = h + w + 2
    for _ in range(cap):
        axial = np.minimum(np.minimum(shifted(-1, 0), shifted(1, 0)),
                           np.minimum(shifted(0, -1), shifted(0, 1)))
        best = axial + cost
        for (dy, dx), dc in dcost.items():
            best = np.minimum(best, shifted(dy, dx) + sqrt2 * dc)
        nd = np.minimum(d, best)
        nd[frame] = 0.0
        if np.array_equal(nd, d):
            break
        d = nd
    return d


def _downsample(arr: np.ndarray, work: int):
    """Block MAX over valid pixels to a grid whose longest side is <= work.

    Returns (coarse, y_centres, x_centres, f), centres in fine-pixel units.
    MAX, not mean: a barrier anywhere inside a block bars the block, so the
    crossing cost does not shrink as 1/f when the image is larger than the work
    grid (block-mean made the same scene read enclosure 1.00 / 0.78 / 0.39 at
    256 / 512 / 1024 px). No zero padding enters partial edge blocks.
    """
    h, w = arr.shape
    f = max(1, int(np.ceil(max(h, w) / max(1, work))))
    if f == 1:
        return (arr.astype(np.float64), np.arange(h, dtype=np.float64),
                np.arange(w, dtype=np.float64), 1)
    ys = np.arange(0, h, f)
    xs = np.arange(0, w, f)
    mx = np.maximum.reduceat(
        np.maximum.reduceat(arr.astype(np.float64), ys, axis=0), xs, axis=1)
    cy = np.diff(np.append(ys, h))
    cx = np.diff(np.append(xs, w))
    return mx, ys + (cy - 1) / 2.0, xs + (cx - 1) / 2.0, f


def _upsample(coarse: np.ndarray, ycen: np.ndarray, xcen: np.ndarray,
              h: int, w: int) -> np.ndarray:
    """Bilinear sample of a coarse grid whose node (i, j) sits at
    (ycen[i], xcen[j]) fine pixels (scipy.zoom aligned corners instead, which
    shifted the field by up to ~f/2 px and more at partial blocks)."""
    def axis(c: np.ndarray, n: int):
        p = np.arange(n, dtype=np.float64)
        if len(c) < 2:
            z = np.zeros(n, dtype=np.int64)
            return z, z, np.zeros(n)
        i1 = np.clip(np.searchsorted(c, p, side="right"), 1, len(c) - 1)
        i0 = i1 - 1
        den = np.where(c[i1] - c[i0] == 0, 1.0, c[i1] - c[i0])
        return i0, i1, np.clip((p - c[i0]) / den, 0.0, 1.0)
    y0, y1, ty = axis(ycen, h)
    x0, x1, tx = axis(xcen, w)
    r = coarse[y0, :] * (1 - ty)[:, None] + coarse[y1, :] * ty[:, None]
    return r[:, x0] * (1 - tx)[None, :] + r[:, x1] * tx[None, :]


def interior_field(lum: np.ndarray,
                   edge: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Bounded-diffusion interior/exterior field.

    Returns {"interior": [0,1] HxW, "signed": [-1,1] HxW, "edge": [0,1] HxW}.
    """
    arr = np.asarray(lum, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    h, w = arr.shape
    if h < 3 or w < 3:
        z = np.zeros((h, w), dtype=np.float32)
        return {"interior": z, "signed": z, "edge": z}

    penalty, enclose_scale, work, _cache, _floor_pct = _params()
    e_full = _edge_magnitude(arr) if edge is None else np.clip(
        np.asarray(edge, dtype=np.float64), 0.0, 1.0)

    e, ycen, xcen, _f = _downsample(e_full, work)
    cost = 1.0 + penalty * e
    geo_barrier = _chamfer(cost)
    geo_free = _chamfer(np.ones_like(cost))
    enc = np.clip((geo_barrier - geo_free) / max(enclose_scale, 1e-6), 0.0, 1.0)

    if enc.shape != (h, w):
        enc = np.clip(_upsample(enc, ycen, xcen, h, w), 0.0, 1.0)

    interior = enc.astype(np.float32)
    signed = (2.0 * interior - 1.0).astype(np.float32)
    return {"interior": interior, "signed": signed,
            "edge": e_full.astype(np.float32)}


def _bilinear(field: np.ndarray, y: float, x: float) -> float:
    h, w = field.shape
    if not (0.0 <= y <= h - 1 and 0.0 <= x <= w - 1):
        return 0.0
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1, x1 = min(y0 + 1, h - 1), min(x0 + 1, w - 1)
    fy, fx = y - y0, x - x0
    return float(
        field[y0, x0] * (1 - fy) * (1 - fx)
        + field[y1, x0] * fy * (1 - fx)
        + field[y0, x1] * (1 - fy) * fx
        + field[y1, x1] * fy * fx)


def ownership_from_enclosure(field: Dict[str, np.ndarray],
                             point: Tuple[float, float],
                             normal: Tuple[float, float],
                             probe: Optional[float] = None) -> float:
    """Enclosure cue at a boundary point: positive when the +normal side is the
    enclosed (interior) side, negative when the -normal side is, 0 when the
    field is flat there (no enclosure evidence). `point` is (x, y); `normal` is
    a (nx, ny) whose sign is the candidate figure direction."""
    interior = field.get("interior")
    if interior is None:
        return 0.0
    p = float(P.INTERIOR_PROBE_PX) if probe is None else float(probe)
    x, y = float(point[0]), float(point[1])
    nx, ny = float(normal[0]), float(normal[1])
    nrm = (nx * nx + ny * ny) ** 0.5
    if nrm <= 1e-9:
        return 0.0
    nx, ny = nx / nrm, ny / nrm
    plus = _bilinear(interior, y + ny * p, x + nx * p)
    minus = _bilinear(interior, y - ny * p, x - nx * p)
    return float(max(-1.0, min(1.0, (plus - minus) * 2.0)))


def interior_weighted_sample(
    plane: np.ndarray,
    interior: np.ndarray,
    x: float,
    y: float,
    radius: float,
    sigma: Optional[float] = None,
) -> Optional[float]:
    """Soft interior-weighted mean of `plane` around (x, y).

    The weight of each pixel is the product of the CONTINUOUS interior field
    (an enclosure strength in [0, 1], never thresholded) and a Gaussian falloff
    of radius `sigma` (default `radius`). No hard mask is ever formed: a pixel
    on the exterior simply contributes proportionally less, so a surface sample
    is pulled toward the enclosed material without a boundary cut. Returns None
    when the window falls entirely outside the plane or the weights vanish, so
    the caller can keep its own default rather than read a fabricated 0.
    """
    if plane is None or interior is None:
        return None
    p = np.asarray(plane, dtype=np.float64)
    enc = np.asarray(interior, dtype=np.float64)
    if p.ndim != 2 or enc.ndim != 2 or p.shape != enc.shape:
        return None
    h, w = p.shape
    r = max(1, int(math.ceil(float(radius))))
    s = float(sigma) if sigma and sigma > 0.0 else max(float(radius), 1e-6)
    x0, x1 = int(math.floor(x - r)), int(math.ceil(x + r))
    y0, y1 = int(math.floor(y - r)), int(math.ceil(y + r))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 < x0 or y1 < y0:
        return None
    num = 0.0
    den = 0.0
    for yy in range(y0, y1 + 1):
        dy = (yy - y) / s
        for xx in range(x0, x1 + 1):
            dx = (xx - x) / s
            gw = math.exp(-0.5 * (dx * dx + dy * dy))
            iw = float(enc[yy, xx])
            if not math.isfinite(iw):
                continue
            wgt = gw * iw
            if wgt <= 0.0:
                continue
            v = float(p[yy, xx])
            if not math.isfinite(v):
                continue
            num += wgt * v
            den += wgt
    if den <= 1e-12:
        return None
    return float(num / den)


_CACHE: Dict[Any, Dict[str, np.ndarray]] = {}


def _cache_key(lum: np.ndarray) -> Tuple:
    h = hashlib.blake2b(
        np.ascontiguousarray(np.asarray(lum, dtype=np.float32)).tobytes(),
        digest_size=16).digest()
    return (np.asarray(lum).shape, h, interior_fingerprint())


def interior_field_for(observation: Any = None,
                       luminance: Any = None) -> Optional[Dict[str, np.ndarray]]:
    """Share one interior field per observation, memoised by content hash +
    fingerprint (module-level, bounded). Returns None when no luminance is
    reachable."""
    cached = None
    if observation is not None:
        c = getattr(observation, "_interior_field_cache", None)
        if isinstance(c, dict) and "interior" in c:
            cached = c
    lum = luminance
    if lum is None and observation is not None:
        getter = getattr(observation, "get_luminance", None)
        if callable(getter):
            lum = getter()
    if lum is None:
        # [U30 D7] The observation's cached field is only handed back when
        # the fingerprint it was computed under is still the current one.
        # Without this check the observation cache silently outlived
        # changes to any INTERIOR_* constant that moved enclosure — the
        # cached field was handed back with no parameters verification at
        # all.  Unverifiable -> abstain (the caller keeps its own
        # default), never a stale field.
        fp = getattr(observation, "_interior_field_fingerprint", None)
        if (cached is not None
                and fp is not None
                and fp == interior_fingerprint()):
            return cached
        return None
    arr = np.asarray(lum)
    if arr.ndim == 3:
        from perception.filters import luminance_adapt, opponent_channels
        arr = luminance_adapt(opponent_channels(arr.astype(np.float32))[..., 0])
    if arr.ndim != 2:
        return cached
    arr = arr.astype(np.float32, copy=False)
    key = _cache_key(arr)
    if cached is not None and getattr(
            observation, "_interior_field_cache_key", None) == key:
        return cached
    hit = _CACHE.get(key)
    if hit is None:
        hit = interior_field(arr)
        size = int(P.INTERIOR_FIELD_CACHE_SIZE)
        if size > 0:
            if len(_CACHE) >= size:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = hit
    if observation is not None:
        try:
            observation._interior_field_cache = hit
            observation._interior_field_cache_key = key
            observation._interior_field_fingerprint = interior_fingerprint()
        except (AttributeError, TypeError):
            pass
    return hit
