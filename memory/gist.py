"""memory/gist.py -- the GLOBAL-LAYOUT group of the view descriptor.

THE GAP THIS FILLS. The measured shape groups (surface 0.70-0.74, shape-only
0.39-0.46) LOST to the cheap layout baselines (bounding-box aspect 0.40,
bounding-box area 0.50) and to a colour histogram (0.66). What was missing was
not more shape detail: it was the LAYOUT -- where the object's contour energy
sits in the frame, how far it is spread, and how it is oriented. A bounding
box approximates that with a rectangle; this module measures it from the
field the front end already computed.

WHAT IT COMPUTES. From the oriented contour energy (the magnitude of the V1
complex response, summed over the scale channels):

  * a SOFT COARSE GRID of that energy (overlapping Gaussian cell support, not
    a box, and not a mask: every cell is a weighted mean of the field);
  * the GLOBAL MOMENTS of the energy: its centroid, its RMS radius about that
    centroid, and its second-moment matrix (so the layout's principal axis and
    anisotropy are read off the energy, not off a bounding box);
  * the RADIAL PROFILE of the energy about the centroid, and the two
    PROJECTION PROFILES onto the frame axes.

Every vector is graded (soft pooling) and L1-normalised, and the energy is
normalised by its own total, so the whole group is invariant to a
brightness/contrast gain. The profiles are population-coded in the tree's
sense: an activity vector over bins, compared by the chance-corrected
overlap (`core.population.axis_similarity`), never by an argmax index.

CONTRACT
  * deterministic; pure function of (energy map, fixed parameters);
  * no randomness, no mask, no region, no hand-built category;
  * the same energy map always gives the same gist.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from core.population import axis_similarity

#: The soft coarse grid: CELLS x CELLS cells over the frame.
GIST_GRID = 8
#: The orientation channels are grouped into this many axial groups
#: (0-45, 45-90, 90-135, 135-180 degrees) before pooling.
GIST_ORIENT_GROUPS = 4
#: The radial profile's bins.
GIST_RADIAL_BINS = 8
#: Each projection profile's bins.
GIST_PROJECTION_BINS = 8
#: The cell support half-width, in cell units. 0.5 gives the cells a
#: 50% overlap -- soft pooling, no hard cell edges.
GIST_CELL_SIGMA = 0.5


@dataclass(frozen=True)
class GistDesc:
    """The global layout of one view.

    grid      : (n_groups, CELLS*CELLS) -- the soft-pooled oriented energy,
                L1-normalised per orientation group.
    orient_frac : (n_groups,) -- the RELATIVE energy of each orientation
                group. The per-group L1 normalisation above throws the
                anisotropy away (each group's map is compared by its SHAPE),
                and the anisotropy is signal: a horse's contour energy is
                far more horizontal than a cow's, and no shape comparison
                can see that. These fractions carry it.
    moments   : (cx, cy, radius, mxx, myy, mxy) -- the energy centroid in
                FRAME units [0, 1], the RMS energy radius in frame units, and
                the normalised second moments.
    radial    : the energy's radial profile about the centroid (L1-normalised).
    proj_x    : the energy projected on the frame's x axis (L1-normalised).
    proj_y    : the energy projected on the frame's y axis (L1-normalised).
    center_px : the energy centroid in pixels (the arrangement group's origin).
    radius_px : the RMS energy radius in pixels (the arrangement group's scale).
    """
    grid: Tuple[Tuple[float, ...], ...] = ()
    orient_frac: Tuple[float, ...] = ()
    moments: Tuple[float, ...] = ()
    radial: Tuple[float, ...] = ()
    proj_x: Tuple[float, ...] = ()
    proj_y: Tuple[float, ...] = ()
    center_px: Tuple[float, float] = (0.0, 0.0)
    radius_px: float = 0.0


def oriented_energy(gabor_complex: Any) -> Optional[np.ndarray]:
    """The oriented contour energy: |complex| summed over the scale channels.

    Returns an array of shape (n_theta, H, W), or None when there is no
    texture field. The magnitude is used, so the result is non-negative --
    a field statistic, never a signed response.
    """
    if gabor_complex is None:
        return None
    arr = np.asarray(gabor_complex, dtype=np.float32)
    if arr.ndim == 3:
        # (theta, H, W): already per-orientation
        mag = np.abs(arr)
    elif arr.ndim == 4:
        # (theta, sf, H, W): sum the scale channels
        mag = np.abs(arr).sum(axis=1)
    else:
        return None
    if mag.ndim != 3 or mag.size == 0:
        return None
    return mag


def _l1(v: np.ndarray) -> np.ndarray:
    total = float(v.sum())
    if total <= 0.0:
        return np.zeros_like(v)
    return v / total


def _soft_grid(energy: np.ndarray, cells: int = GIST_GRID,
               sigma_cells: float = GIST_CELL_SIGMA) -> np.ndarray:
    """The energy on a soft coarse grid.

    Each cell is the Gaussian-weighted mean of the WHOLE map around its
    centre (support sigma_cells in cell units), so neighbouring cells overlap
    and no cell edge is a hard boundary. Returns (cells, cells).
    """
    h, w = energy.shape
    cy = (np.arange(cells) + 0.5) * (h / cells)
    cx = (np.arange(cells) + 0.5) * (w / cells)
    yy, xx = np.mgrid[0:h, 0:w]
    sy = max(1e-6, sigma_cells * (h / cells))
    sx = max(1e-6, sigma_cells * (w / cells))
    out = np.zeros((cells, cells), dtype=np.float64)
    for j, gy in enumerate(cy):
        wy = np.exp(-0.5 * ((yy - gy) / sy) ** 2)
        for i, gx in enumerate(cx):
            wt = wy * np.exp(-0.5 * ((xx - gx) / sx) ** 2)
            den = float(wt.sum())
            out[j, i] = float((wt * energy).sum()) / den if den > 0.0 else 0.0
    return out


def build_gist(energy: Any, cells: int = GIST_GRID,
               orient_groups: int = GIST_ORIENT_GROUPS,
               radial_bins: int = GIST_RADIAL_BINS,
               projection_bins: int = GIST_PROJECTION_BINS) -> Optional[GistDesc]:
    """The global-layout descriptor of one view, from its oriented energy.

    `energy` is the (n_theta, H, W) oriented energy map.
    """
    if energy is None:
        return None
    e = np.asarray(energy, dtype=np.float64)
    if e.ndim != 3 or e.size == 0:
        return None
    n_theta, h, w = e.shape
    total = float(e.sum())
    if total <= 0.0:
        return None
    # One scalar normalisation for the whole field: a brightness/contrast gain
    # scales every orientation equally, so this removes it while PRESERVING
    # the anisotropy between orientations (which is signal).
    e = e / total

    # -- the soft coarse grid, per orientation group -------------------------
    n_groups = max(1, int(orient_groups))
    per_theta = max(1, int(math.ceil(n_theta / n_groups)))
    groups: list = []
    group_energy: list = []
    for g in range(n_groups):
        lo = g * per_theta
        hi = min(n_theta, lo + per_theta)
        acc = np.zeros((h, w), dtype=np.float64)
        for t in range(lo, hi):
            acc += e[t]
        group_energy.append(float(acc.sum()))
        cell = _soft_grid(acc, cells)
        groups.append(_l1(cell).reshape(-1))
    grid = tuple(tuple(float(v) for v in g) for g in groups)
    # The RELATIVE energy of each orientation group: the anisotropy the
    # per-group L1 normalisation throws away.
    ge = np.asarray(group_energy, dtype=np.float64)
    tot_ge = float(ge.sum())
    orient_frac = tuple(float(v) for v in (ge / tot_ge)) if tot_ge > 0 else ()

    # -- the global moments --------------------------------------------------
    total_e = e.sum(axis=0)
    mass = float(total_e.sum())
    if mass <= 0.0:
        return None
    ys, xs = np.mgrid[0:h, 0:w]
    cx_px = float((total_e * xs).sum() / mass)
    cy_px = float((total_e * ys).sum() / mass)
    rad = np.hypot(xs - cx_px, ys - cy_px)
    radius_px = float(math.sqrt((total_e * rad ** 2).sum() / mass))
    if radius_px <= 1e-9:
        radius_px = 1.0
    dx = xs - cx_px
    dy = ys - cy_px
    mxx = float((total_e * dx * dx).sum() / (mass * radius_px ** 2))
    myy = float((total_e * dy * dy).sum() / (mass * radius_px ** 2))
    mxy = float((total_e * dx * dy).sum() / (mass * radius_px ** 2))
    moments = (cx_px / max(1.0, w - 1.0), cy_px / max(1.0, h - 1.0),
               radius_px / max(1.0, 0.5 * (h + w) / 2.0),
               mxx, myy, mxy)

    # -- the radial profile --------------------------------------------------
    rmax = float(rad.max()) if rad.size else 1.0
    rbin = np.clip((rad / max(rmax, 1e-9) * radial_bins).astype(int),
                    0, radial_bins - 1)
    radial = np.zeros(radial_bins, dtype=np.float64)
    for k in range(radial_bins):
        radial[k] = float(total_e[rbin == k].sum())
    radial = _l1(radial)

    # -- the projection profiles --------------------------------------------
    px = np.zeros(projection_bins, dtype=np.float64)
    py = np.zeros(projection_bins, dtype=np.float64)
    xb = np.clip((xs / max(1.0, w) * projection_bins).astype(int),
                 0, projection_bins - 1)
    yb = np.clip((ys / max(1.0, h) * projection_bins).astype(int),
                 0, projection_bins - 1)
    for k in range(projection_bins):
        px[k] = float(total_e[xb == k].sum())
        py[k] = float(total_e[yb == k].sum())
    px = _l1(px)
    py = _l1(py)

    return GistDesc(
        grid=grid,
        orient_frac=orient_frac,
        moments=moments,
        radial=tuple(float(v) for v in radial),
        proj_x=tuple(float(v) for v in px),
        proj_y=tuple(float(v) for v in py),
        center_px=(cx_px, cy_px),
        radius_px=radius_px,
    )


def gist_similarity(a: Optional[GistDesc], b: Optional[GistDesc],
                    center_sigma: float = 0.15,
                    radius_sigma: float = 0.35) -> float:
    """The gist group's similarity, in [0, 1].

    * the soft grid: the chance-corrected overlap per orientation group;
    * the moments: a Gaussian on the centroid distance (in units of the
      energy radius, so the reading is scale-free), a Gaussian on the log
      radius ratio, and a Gaussian on the second-moment difference;
    * the radial and projection profiles: the chance-corrected overlap.

    A missing gist on either side scores 0.0 -- the group contributes
    nothing rather than a fabricated value.
    """
    if a is None or b is None:
        return 0.0
    parts: list = []
    weights: list = []

    if a.grid and b.grid and len(a.grid) == len(b.grid):
        for ga, gb in zip(a.grid, b.grid):
            if len(ga) == len(gb) and len(ga) > 0:
                parts.append(axis_similarity(ga, gb))
                weights.append(1.0)

    # The orientation ANISOTROPY is measured but NOT part of the similarity.
    # It was tried as a similarity term (the relative energy of each
    # orientation group, read on a log scale) and it measurably HURT the
    # fused arm: the layout group alone rose from 0.8333 to 0.8385, but the
    # fused arm fell from 0.8854 to 0.8490 on the dev split -- the term adds
    # more noise to the layout's evidence than the ranking it adds. The
    # fractions stay on the descriptor (they are a field statistic and a
    # later session may read them differently), but they do not score.

    if a.moments and b.moments and len(a.moments) >= 6 and len(b.moments) >= 6:
        # [F7] The energy-centre term is GONE. It compared the two
        # views' energy centroids in ABSOLUTE pixels (normalised by the
        # radius), so it measured WHERE the object sits in the frame --
        # a property of the frame, not of the object. The layout's shape
        # evidence is the frame-relative radial and projection profiles
        # below, which are already measured from the energy centre.
        # (The two centre sigmas remain parameters: the CENTRE term is
        # what they scaled.)
        del center_sigma
        ra = max(1e-6, a.radius_px)
        rb = max(1e-6, b.radius_px)
        lr = math.log(ra / rb)
        parts.append(math.exp(-0.5 * (lr / radius_sigma) ** 2))
        weights.append(1.0)
        dm = math.sqrt((a.moments[3] - b.moments[3]) ** 2
                       + (a.moments[4] - b.moments[4]) ** 2
                       + (a.moments[5] - b.moments[5]) ** 2)
        parts.append(math.exp(-0.5 * (dm / 0.75) ** 2))
        weights.append(1.0)

    for va, vb in ((a.radial, b.radial), (a.proj_x, b.proj_x),
                   (a.proj_y, b.proj_y)):
        if va and vb and len(va) == len(vb):
            parts.append(axis_similarity(va, vb))
            weights.append(1.0)

    if not parts:
        return 0.0
    wsum = sum(weights)
    if wsum <= 0.0:
        return 0.0
    return max(0.0, min(1.0, sum(p * w for p, w in zip(parts, weights)) / wsum))
