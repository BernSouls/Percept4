"""memory/depth_cues.py -- the STATIC DEPTH CUES (outcome B).

WHAT THIS IS. Four graded measurements of the static depth cues the tree was
missing, each computed ONLY from quantities the view descriptor already
carries (the per-part surface samples and the global-layout gist):

  TEXTURE_GRADIENT  the local spatial GRADIENT of the texture: how the
                    texture's orientation and its spatial-frequency scale
                    change as one walks across the object. The classic
                    Gibsonian texture gradient, measured here on the
                    descriptor's own graded texture distributions.
  SHADING           the low-frequency LUMINANCE structure: the object's
                    interior luminance level, its radial fall-off, and the
                    soft angular profile of the shading.
  BLUR              the local loss of HIGH-FREQUENCY energy: the mass the
                    surface's spatial-frequency distribution still carries
                    in the finest band.
  FALLOFF           the fall-off of CONTRAST and of SATURATION from the
                    object's centre to its rim (the inner/outer split is
                    geometric: the area bisector of the energy radius).

NO CUE IS A CONSTANT, AND NOTHING IS HARD-CODED. Every prior the cues need
(the shading field's smoothing scale, each cue's tolerance) is MEASURED on
the dev corpus by diagnostics/_depth_gate.py and written into
parameters/_text.py with its provenance. A cue that cannot survive the
pre-registered perturbations (an INTEGER SHIFT and a 0.9 SCALE) is not
allowed into the fusion: it is logged with its measured repeatability and
stays out (see `ADMITTED_CUES`).

THE GROUP. The surviving cues form ONE label-free group ("depth") in the
fusion. Its internal weights are the SAME product rule the top-level groups
use (reliability x distinctiveness, both label-free, both measured on the dev
corpus), and the group reports its own ablation.

DETERMINISM. Every function here is a pure function of the view record: no
randomness, no mask, no category, no label.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import parameters as P


# ---------------------------------------------------------------------------
# The cue names. The order is the report's order.
# ---------------------------------------------------------------------------
CUE_TEXTURE_GRADIENT = "texture_gradient"
CUE_SHADING = "shading"
CUE_BLUR = "blur"
CUE_FALLOFF = "falloff"

ALL_CUES: Tuple[str, ...] = (CUE_TEXTURE_GRADIENT, CUE_SHADING,
                              CUE_BLUR, CUE_FALLOFF)

#: The angular resolution of the shading field's profile (a population axis).
SHADING_ANGLE_BINS = 12


@dataclass(frozen=True)
class DepthMeasures:
    """The four cues' measurements of one view.

    texture_gradient : (dtau_x, dtau_y, dsigma_x, dsigma_y, |grad tau|,
                        |grad sigma|) -- the texture orientation's and the
                        texture scale's gradients in the object's upright
                        Cartesian frame (energy radii), and their mean
                        magnitudes.
    shading          : (level, radial slope, *angular profile) -- the
                        interior luminance's level, its slope against the
                        energy radius, and the soft angular profile.
    blur             : the mean high-frequency loss (1 - the finest band's
                        mass) over the parts. None when no part carries a
                        spatial-frequency distribution (the cue is ABSENT,
                        not zero).
    falloff          : (contrast fall-off, saturation fall-off). None when
                        no part carries a luminance or a saturation sample.
    """

    texture_gradient: Tuple[float, ...] = ()
    shading: Tuple[float, ...] = ()
    blur: Optional[float] = None
    falloff: Optional[Tuple[float, float]] = None

    def cue_vector(self, cue: str) -> Optional[Tuple[float, ...]]:
        if cue == CUE_TEXTURE_GRADIENT:
            return self.texture_gradient or None
        if cue == CUE_SHADING:
            return self.shading or None
        if cue == CUE_BLUR:
            if self.blur is None:
                return None
            return (float(self.blur),)
        if cue == CUE_FALLOFF:
            if self.falloff is None:
                return None
            return tuple(self.falloff)
        return None


# ---------------------------------------------------------------------------
# The small statistics the cues need.
# ---------------------------------------------------------------------------

def _axial_mean(vec: Sequence[float]) -> Optional[float]:
    """The mean direction of an AXIAL (pi-periodic) distribution.

    The orientation channel is pi-periodic, so the mean direction is the
    argument of the DOUBLED angle, halved once. Returns None when the
    distribution carries no directional content (a uniform one).
    """
    v = np.asarray(vec, dtype=np.float64)
    if v.size < 3:
        return None
    total = float(v.sum())
    if total <= 1e-12:
        return None
    ang = np.pi * (np.arange(v.size) + 0.5) / v.size
    c = float((v * np.cos(2.0 * ang)).sum())
    s = float((v * np.sin(2.0 * ang)).sum())
    if abs(c) < 1e-12 and abs(s) < 1e-12:
        return None
    return float((0.5 * math.atan2(s, c)) % math.pi)


def _band_centroid(vec: Sequence[float]) -> Optional[float]:
    """The centroid of a spatial-frequency distribution, in band units.

    The Gabor bank's scale axis is LOG-spaced (geometric), so the band's index
    IS the log-scale: the centroid of the index is the log-scale centroid.
    """
    v = np.asarray(vec, dtype=np.float64)
    if v.size < 2:
        return None
    total = float(v.sum())
    if total <= 1e-12:
        return None
    idx = np.arange(v.size, dtype=np.float64)
    return float((v * idx).sum() / total) / float(v.size - 1)


def _fine_mass(vec: Sequence[float], n_bands: int = 2) -> Optional[float]:
    """The mass the distribution still carries in its FINEST band(s).

    The blur level is the loss of that mass: a blurred surface keeps its
    energy in the coarse bands and loses the fine one.
    """
    v = np.asarray(vec, dtype=np.float64)
    if v.size < 3:
        return None
    total = float(v.sum())
    if total <= 1e-12:
        return None
    k = max(1, min(int(n_bands), v.size))
    return float(v[-k:].sum() / total)


def _safe(value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        return 0.0
    return v


# ---------------------------------------------------------------------------
# The measurements.
# ---------------------------------------------------------------------------

def _part_texture(parts: Sequence[Any]) -> List[Tuple[float, float, float, float]]:
    """Each part's (x, y, tau, sigma) in the object's upright Cartesian frame.

    x, y are the part's polar address in ENERGY RADII (the descriptor's own
    frame: y is UP, so the frame is the viewer's upright). tau is the texture
    orientation (the axial mean of the interior orientation distribution) and
    sigma the texture scale (the centroid of the interior spatial-frequency
    distribution), both in [0, 1] (sigma) and radians (tau).

    The FLANK is used when the interior side has no evidence (a fragment deep
    inside the object): the cue asks for the texture AT the part, not for a
    particular side of it.
    """
    out: List[Tuple[float, float, float, float]] = []
    for p in parts:
        s = getattr(p, "surface", None)
        if s is None:
            continue
        theta = s.theta_i or ()
        sf = s.sf_i or ()
        if not theta and s.theta_f:
            theta = s.theta_f
            sf = s.sf_f or ()
        if not theta:
            continue
        tau = _axial_mean(theta)
        if tau is None:
            continue
        sigma = _band_centroid(sf) if sf else None
        if sigma is None:
            continue
        pol = getattr(p, "polar", None)
        if pol is None:
            pos = getattr(p, "pos", (0.0, 0.0))
            r = math.hypot(float(pos[0]), float(pos[1]))
            th = math.atan2(float(pos[0]), -float(pos[1]))
        else:
            r = float(pol[0])
            th = float(pol[1])
        x = r * math.sin(th)
        y = -r * math.cos(th)
        out.append((x, y, tau, sigma))
    return out


def _plane_gradient(points: Sequence[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
    """The least-squares plane through (x, y, value): (gx, gy, mean |grad|).

    The gradient is the field's local derivative in the object's upright
    Cartesian frame (per energy radius). None when the fit is not supported
    (fewer than three points, or a degenerate design).
    """
    if len(points) < 3:
        return None
    x = np.asarray([p[0] for p in points], dtype=np.float64)
    y = np.asarray([p[1] for p in points], dtype=np.float64)
    v = np.asarray([p[2] for p in points], dtype=np.float64)
    A = np.stack([np.ones_like(x), x, y], axis=1)
    try:
        sol, *_rest = np.linalg.lstsq(A, v, rcond=None)
    except Exception:
        return None
    gx, gy = float(sol[1]), float(sol[2])
    mag = float(np.mean(np.sqrt((A[:, 1] * gx) ** 2 + (A[:, 2] * gy) ** 2)))
    return (gx, gy, mag)


def _shading_profile(parts: Sequence[Any], scale: float) -> Tuple[List[float], List[float]]:
    """The soft angular profile of the interior luminance, and the radii.

    Each part contributes its interior luminance at its own angular position,
    spread with a Gaussian whose ANGULAR width is the spatial scale `scale`
    (in energy radii) divided by the part's radius -- so the smoothing is
    uniform in SPACE, not in angle.
    """
    n = SHADING_ANGLE_BINS
    acc = [0.0] * n
    radii: List[float] = []
    for p in parts:
        s = getattr(p, "surface", None)
        if s is None:
            continue
        lum = float(s.lum_i)
        if lum < 0.0:
            continue
        pol = getattr(p, "polar", None)
        if pol is None:
            pos = getattr(p, "pos", (0.0, 0.0))
            r = math.hypot(float(pos[0]), float(pos[1]))
            th = math.atan2(float(pos[0]), -float(pos[1]))
        else:
            r = float(pol[0])
            th = float(pol[1])
        radii.append(r)
        width = max(1e-3, float(scale) / max(r, 0.25))
        centre = (th + math.pi) / (2.0 * math.pi) * n
        for j in range(n):
            d = abs(j - centre)
            d = min(d, n - d)
            acc[j] += lum * math.exp(-0.5 * (d / width) ** 2)
    total = sum(acc)
    if total <= 1e-12:
        return [1.0 / n for _ in range(n)], radii
    return [a / total for a in acc], radii


def measure(view: Any) -> DepthMeasures:
    """Every cue's measurement of one view."""
    parts = list(getattr(view, "parts", ()) or ())
    scale = float(P.ASPECT_DEPTH_SHADING_SCALE)

    # -- the texture gradient ------------------------------------------------
    tex = _part_texture(parts)
    grad_tau = _plane_gradient([(x, y, t) for x, y, t, _s in tex])
    grad_sig = _plane_gradient([(x, y, s) for x, y, _t, s in tex])
    if grad_tau is not None and grad_sig is not None:
        texture_gradient = (
            _safe(grad_tau[0]), _safe(grad_tau[1]),
            _safe(grad_sig[0]), _safe(grad_sig[1]),
            _safe(grad_tau[2]), _safe(grad_sig[2]),
        )
    else:
        texture_gradient = ()

    # -- the shading field ----------------------------------------------------
    profile, radii = _shading_profile(parts, scale)
    lums: List[Tuple[float, float]] = []
    sats: List[Tuple[float, float]] = []
    for p in parts:
        s = getattr(p, "surface", None)
        if s is None:
            continue
        pol = getattr(p, "polar", None)
        if pol is None:
            pos = getattr(p, "pos", (0.0, 0.0))
            r = math.hypot(float(pos[0]), float(pos[1]))
        else:
            r = float(pol[0])
        if float(s.lum_i) >= 0.0:
            lums.append((r, float(s.lum_i)))
        if float(s.sat_i) >= 0.0:
            sats.append((r, float(s.sat_i)))
    shading: Tuple[float, ...] = ()
    if lums:
        level = float(np.mean([v for _r, v in lums]))
        rs = np.asarray([r for r, _v in lums], dtype=np.float64)
        vs = np.asarray([v for _r, v in lums], dtype=np.float64)
        if rs.size >= 3 and float(rs.std()) > 1e-9:
            slope = float(np.polyfit(rs, vs, 1)[0])
        else:
            slope = 0.0
        shading = (_safe(level), _safe(slope)) + tuple(_safe(v) for v in profile)

    # -- the blur level -------------------------------------------------------
    fines: List[float] = []
    n_bands = int(P.ASPECT_DEPTH_BLUR_BANDS)
    for p in parts:
        s = getattr(p, "surface", None)
        if s is None:
            continue
        sf = s.sf_i or s.sf_f or ()
        fm = _fine_mass(sf, n_bands)
        if fm is None:
            continue
        fines.append(1.0 - fm)
    blur: Optional[float] = None
    if fines:
        blur = float(np.mean(fines))

    # -- the contrast and saturation fall-off ---------------------------------
    # The split is GEOMETRIC, not tuned: the inner bin is the disc of the
    # energy radius that holds half the area (r < R/sqrt(2)).
    inner = 1.0 / math.sqrt(2.0)
    contrast_fall = 0.0
    saturation_fall = 0.0
    have_fall = False
    if lums:
        li = [v for r, v in lums if r <= inner]
        lo = [v for r, v in lums if r > inner]
        if li and lo:
            contrast_fall = float(np.mean(li) - np.mean(lo))
            have_fall = True
    if sats:
        si = [v for r, v in sats if r <= inner]
        so = [v for r, v in sats if r > inner]
        if si and so:
            saturation_fall = float(np.mean(si) - np.mean(so))
            have_fall = True
    falloff = (contrast_fall, saturation_fall) if have_fall else None

    return DepthMeasures(
        texture_gradient=texture_gradient,
        shading=shading,
        blur=_safe(blur) if blur is not None else None,
        falloff=falloff,
    )


# ---------------------------------------------------------------------------
# The similarities.
# ---------------------------------------------------------------------------

def _tol(cue: str) -> float:
    """The cue's tolerance (corpus-derived; see parameters)."""
    table = getattr(P, "ASPECT_DEPTH_TOLERANCE", None) or {}
    try:
        return float(dict(table).get(cue, 0.0))
    except Exception:
        return 0.0


def cue_similarity(a: DepthMeasures, b: DepthMeasures, cue: str) -> Optional[float]:
    """One cue's similarity between two views, in [0, 1].

    The kernel is a Gaussian on the cue's OWN difference, with the tolerance
    MEASURED on the corpus (the p90 of the difference over the corpus's
    random pairs): a typical difference scores ~0.6, an identical one 1.0 and
    a far one ~0.

    None when the cue is ABSENT on either side (no evidence): an absent cue
    is not a zero, it is a missing measurement.
    """
    va = a.cue_vector(cue)
    vb = b.cue_vector(cue)
    if va is None or vb is None:
        return None
    if len(va) != len(vb):
        return None
    tol = _tol(cue)
    if tol <= 0.0:
        return None
    d = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(va, vb)))
    return float(max(0.0, min(1.0, math.exp(-0.5 * (d / tol) ** 2))))


def admitted_cues() -> Tuple[str, ...]:
    """The cues that passed their own repeatability gate (parameters)."""
    raw = getattr(P, "ASPECT_DEPTH_CUES", None) or ()
    try:
        return tuple(str(c) for c in raw)
    except Exception:
        return ()


def depth_similarity(A: Any, B: Any) -> Optional[float]:
    """The depth group's similarity between two views.

    The weighted mean of the ADMITTED cues' similarities; the weights are
    the label-free product rule (reliability x distinctiveness), measured on
    the dev corpus and written into the parameters.
    """
    ma = _measures(A)
    mb = _measures(B)
    if ma is None or mb is None:
        return None
    weights = getattr(P, "ASPECT_DEPTH_WEIGHTS", None) or {}
    num = 0.0
    den = 0.0
    for cue in admitted_cues():
        s = cue_similarity(ma, mb, cue)
        if s is None:
            continue
        w = float(dict(weights).get(cue, 0.0))
        if w <= 0.0:
            continue
        num += w * s
        den += w
    if den <= 1e-12:
        return None
    return float(max(0.0, min(1.0, num / den)))


def _measures(view: Any) -> Optional[DepthMeasures]:
    """The view's measures, cached on the view (a sweep pairs the same views
    thousands of times, and the measures are a pure function of the view)."""
    cached = getattr(view, "depth", None)
    if cached is not None:
        return cached
    m = measure(view)
    try:
        view.depth = m
    except Exception:
        pass
    return m
