"""[STAGE 6 / R1] A local feature's place on an extended boundary.

One measured relation, written once for every consumer:

    host     an ordered polyline with at least two positioned vertices
             (a PEAK_CHAIN in chain order, a Contour in member order)
    feature  a positioned local detection (Orientation, CONTOUR_PEAK,
             EndStopped, Corner, ContourFragment, ...)

    foot           the closest point of the host polyline to the feature
    arc_pos        arc length from the host's first vertex to the foot, as a
                   fraction of the host's length.  ORDINAL: the host's member
                   order is the chain former's choice, so this number means
                   something only inside its own host (the order constraint
                   reads it; nothing compares it across graphs)
    arc_end_dist   min(arc_pos, 1 - arc_pos), the distance from the nearer
                   end.  The same whichever way the host is walked.  ABSENT on
                   a closed host: a ring has no end and no origin
    perp_dist      |feature - foot| in extraction scales
    tangent_angle  angle between the feature's own tangent and the host's
                   tangent at the foot, folded to [0, pi/2]; ABSENT when the
                   feature carries no orientation

THE AMBIGUITY RULE (pre-registered, diagnostics/results/
phase6_preregistration.md, section 2).  A feature is attached to a host of a
given kind only when that host is the feature's unambiguous nearest of that
kind: d1 <= D_MAX and d2 - d1 >= AMBIGUITY_MARGIN, d1 <= d2 being the two
smallest distances to DISTINCT hosts of the kind.  Otherwise nothing is
written.  Two near-equally plausible hosts are not a measurement.

A host that the data structure itself names (a chain's listed member peak, a
Corner's own chain, a Contour's listed member) is KNOWN: no test, it is a
recorded fact.  The membership flag IDENTIFIES the host; the attachment
supplies the MEASUREMENT (where on it, how far, at what angle).  That is not
circular: the flag carries no geometry.

Pure geometry: no graph, no parameters, no state.  The binder writer
(`Binder._run_boundary_attachment`) and the census
(`diagnostics/phase6_attachment_census.py`) both call `attach`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: admission radius, in extraction scales.  A cell-anchored feature can sit up
#: to 0.88 scales from a boundary crossing its own cell (half a cell diagonal);
#: the next cell centre is 1.25 scales away.
D_MAX: float = 1.0
#: two hosts closer to each other (as seen from the feature) than this are
#: "near-equally plausible": half a grid cell, the position uncertainty of a
#: cell-anchored feature.
AMBIGUITY_MARGIN: float = 0.5

#: outcomes of one (feature, host kind) attempt
KNOWN, UNAMBIGUOUS, AMBIGUOUS, NO_HOST = "known", "unambiguous", "ambiguous", "no_host"


@dataclass(frozen=True)
class Host:
    """An extended boundary: an ordered polyline."""
    kind: str
    key: Any                      # canonical identity of the detection (its anchor)
    node: Optional[int]
    pts: Tuple[Tuple[float, float], ...]
    closed: bool = False
    #: the host's OWN measured orientation (axial, radians), when its polyline
    #: is too coarse to give a local tangent.  None -> the segment at the foot.
    tangent: Optional[float] = None


@dataclass(frozen=True)
class Feature:
    """A local feature with a position and, when it has one, a tangent."""
    kind: str
    key: Any
    node: Optional[int]
    pos: Tuple[float, float]
    tangent: Optional[float] = None          # axial, radians, in [0, pi)
    known: Tuple[Tuple[str, Any], ...] = ()  # (host kind, host key) the data names
    only_known: bool = False                 # never attached by geometry


@dataclass(frozen=True)
class Attachment:
    feature: Feature
    host: Host
    arc_pos: float
    arc_end_dist: Optional[float]
    perp_dist: float
    tangent_angle: Optional[float]
    known: bool


class _Poly:
    """A host polyline prepared for projection."""

    __slots__ = ("a", "seg", "len", "cum", "total", "closed")

    def __init__(self, pts: Sequence[Tuple[float, float]], closed: bool) -> None:
        p = np.asarray(pts, dtype=np.float64).reshape((-1, 2))
        if closed and len(p) >= 3 and bool(np.any(p[0] != p[-1])):
            p = np.vstack([p, p[:1]])
        self.a = p[:-1]
        self.seg = p[1:] - p[:-1]
        self.len = np.hypot(self.seg[:, 0], self.seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.len)])
        self.total = float(self.cum[-1])
        self.closed = bool(closed)

    def project(self, pos: Tuple[float, float]) -> Tuple[float, float, int]:
        """(distance in px, arc length at the foot, segment index).

        Ties go to the lowest segment index, so the answer does not depend on
        float noise or on iteration order."""
        px, py = float(pos[0]), float(pos[1])
        dx, dy = px - self.a[:, 0], py - self.a[:, 1]
        l2 = self.len * self.len
        t = np.where(l2 > 0.0,
                     (dx * self.seg[:, 0] + dy * self.seg[:, 1])
                     / np.where(l2 > 0.0, l2, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)
        fx = self.a[:, 0] + t * self.seg[:, 0]
        fy = self.a[:, 1] + t * self.seg[:, 1]
        d = np.hypot(px - fx, py - fy)
        i = int(np.argmin(d))            # first minimum
        return float(d[i]), float(self.cum[i] + t[i] * self.len[i]), i

    def segment_angle(self, i: int) -> Optional[float]:
        if not (self.len[i] > 0.0):
            return None
        return math.atan2(float(self.seg[i, 1]), float(self.seg[i, 0])) % math.pi


def is_host_polyline(pts: Sequence[Tuple[float, float]]) -> bool:
    """At least two vertices and a positive length."""
    if pts is None or len(pts) < 2:
        return False
    p = np.asarray(pts, dtype=np.float64).reshape((-1, 2))
    return bool(np.hypot(*(p[1:] - p[:-1]).T).sum() > 0.0)


def axial_difference(a: float, b: float) -> float:
    """Angle between two AXIAL directions, in [0, pi/2]."""
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def _host_tangent(h: Host, poly: _Poly, i: int) -> Optional[float]:
    return h.tangent if h.tangent is not None else poly.segment_angle(i)


def _measure(f: Feature, h: Host, poly: _Poly, scale: float, known: bool) -> Attachment:
    d, s, i = poly.project(f.pos)
    arc = s / poly.total if poly.total > 0.0 else 0.0
    arc = min(1.0, max(0.0, arc))
    host_angle = _host_tangent(h, poly, i)
    ang = None
    if f.tangent is not None and host_angle is not None:
        ang = axial_difference(float(f.tangent), host_angle)
    return Attachment(
        feature=f, host=h, arc_pos=arc,
        arc_end_dist=None if poly.closed else min(arc, 1.0 - arc),
        perp_dist=d / scale, tangent_angle=ang, known=known)


def attach(
    features: Iterable[Feature],
    hosts: Iterable[Host],
    scale: float,
    d_max: float = D_MAX,
    margin: float = AMBIGUITY_MARGIN,
) -> Tuple[List[Attachment], Dict[Tuple[str, str], Dict[str, int]]]:
    """Attach every feature to its host of each kind, or to none.

    Returns `(attachments, stats)`; `stats[(feature kind, host kind)]` counts
    the outcomes KNOWN / UNAMBIGUOUS / AMBIGUOUS / NO_HOST.  Deterministic:
    hosts and features are processed in the order given (the callers sort
    canonically) and every tie is broken by that order.
    """
    scale = float(scale)
    if not (scale > 0.0):
        return [], {}
    by_kind: Dict[str, List[Tuple[Host, _Poly]]] = {}
    by_key: Dict[Tuple[str, Any], Tuple[Host, _Poly]] = {}
    for h in hosts:
        if not is_host_polyline(h.pts):
            continue
        poly = _Poly(h.pts, h.closed)
        by_kind.setdefault(h.kind, []).append((h, poly))
        by_key.setdefault((h.kind, h.key), (h, poly))
    out: List[Attachment] = []
    stats: Dict[Tuple[str, str], Dict[str, int]] = {}
    kinds = sorted(by_kind)
    for f in features:
        named: Dict[str, List[Any]] = {}
        for hk, key in f.known:
            named.setdefault(hk, []).append(key)
        for hk in kinds:
            if f.only_known and hk not in named:
                continue
            st = stats.setdefault((f.kind, hk), {KNOWN: 0, UNAMBIGUOUS: 0, AMBIGUOUS: 0, NO_HOST: 0})
            keys = [k for k in named.get(hk, ()) if (hk, k) in by_key]
            if len(keys) == 1:
                h, poly = by_key[(hk, keys[0])]
                out.append(_measure(f, h, poly, scale, True))
                st[KNOWN] += 1
                continue
            if len(keys) > 1:
                # the data names several hosts of one kind: not one fact
                st[AMBIGUOUS] += 1
                continue
            if f.only_known:
                st[NO_HOST] += 1
                continue
            best = second = None
            best_i = -1
            for i, (h, poly) in enumerate(by_kind[hk]):
                if h.node is not None and h.node == f.node:
                    continue
                d = poly.project(f.pos)[0] / scale
                if best is None or d < best:
                    second, best, best_i = best, d, i
                elif second is None or d < second:
                    second = d
            if best is None or best > d_max:
                st[NO_HOST] += 1
                continue
            if second is not None and (second - best) < margin:
                st[AMBIGUOUS] += 1
                continue
            h, poly = by_kind[hk][best_i]
            out.append(_measure(f, h, poly, scale, False))
            st[UNAMBIGUOUS] += 1
    return out, stats


# ---------------------------------------------------------------------------
# From detections to hosts and features
# ---------------------------------------------------------------------------
#: the kinds R1 serves.  One table, read by the binder writer and the census.
HOST_KINDS: Tuple[str, ...] = ("PEAK_CHAIN", "Contour")
FEATURE_KINDS: Tuple[str, ...] = (
    "Orientation", "CONTOUR_PEAK", "EndStopped", "Corner", "ContourFragment")
#: [R3] a RegionShape is the shape of ONE chain (its anchor IS the chain's):
#: it is placed on that chain and on nothing else.
KNOWN_ONLY_KINDS: Tuple[str, ...] = ("RegionShape",)


def _members(det: Any) -> List[Any]:
    """Member anchors of a detection, in the ordinal order its `rho` gives."""
    pairs = []
    for role in (getattr(det, "rho", None) or ()):
        if len(role) >= 3 and role[0] == "member" and isinstance(role[2], int):
            pairs.append((role[2], role[1]))
    pairs.sort(key=lambda p: p[0])
    return [a for _o, a in pairs]


def _role(det: Any, name: str) -> Optional[Any]:
    for role in (getattr(det, "rho", None) or ()):
        if len(role) >= 2 and role[0] == name:
            return role[1]
    return None


def _xy(positions: Dict[Any, Any], anchor: Any) -> Optional[Tuple[float, float]]:
    try:
        p = positions.get(anchor)
    except TypeError:
        return None
    if p is None or len(p) < 2:
        return None
    x, y = float(p[0]), float(p[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


def host_of(det: Any, positions: Dict[Any, Any], node: Optional[int],
            n_theta: int = 12) -> Optional[Host]:
    """The host polyline a PEAK_CHAIN or Contour detection carries, or None.

    A PEAK_CHAIN's vertices are peaks (pixel positions): its tangent at the
    foot is the segment there.  A Contour's vertices are region-grid cells:
    its segments only take the grid's eight directions (measured: every
    feature kind reads a 45-degree median against them), so its tangent is
    the Contour's own measured `mean_theta` slot instead.
    """
    if det.k not in HOST_KINDS:
        return None
    pts = [p for p in (_xy(positions, a) for a in _members(det)) if p is not None]
    if not is_host_polyline(pts):
        return None
    q = det.sigma.quantized
    # both classes keep their closure flag in slot 3 (PEAK_CHAIN: the chain
    # lens' ring test; CONTOUR: the `closure` slot)
    closed = bool(len(q) > 3 and float(q[3]) >= 0.5 and len(pts) >= 3)
    tangent = None
    if det.k == "Contour" and len(q) > 1:
        tangent = ((float(q[1]) + 0.5) * math.pi / float(n_theta)) % math.pi
    return Host(kind=det.k, key=det.alpha, node=node, pts=tuple(pts),
                closed=closed, tangent=tangent)


def feature_tangent(det: Any, n_theta: int) -> Optional[float]:
    """The feature's own tangent (axial, radians), or None when it has none.

    Conventions (each to be verified against renderer truth, see
    diagnostics/phase6_tangent_truth.py):
      CONTOUR_PEAK  slot 0 is the field's theta index over 12 channels; the
                    field's theta is the ridge TANGENT
      Orientation   slot 0 is the orientation bin over `n_theta` (bin
                    centre); it is the edge TANGENT
      EndStopped    slot 0 is the carrier bin over `n_theta`; tangent + pi/2
    """
    q = det.sigma.quantized
    if not q:
        return None
    if det.k == "CONTOUR_PEAK":
        return (float(q[0]) * math.pi / 12.0) % math.pi
    if det.k == "Orientation":
        return ((float(q[0]) + 0.5) * math.pi / float(n_theta)) % math.pi
    if det.k == "EndStopped":
        return (float(q[0]) * math.pi / float(n_theta) + math.pi / 2.0) % math.pi
    return None


def hosts_and_features(
    detections: Sequence[Tuple[Any, Optional[int]]],
    positions: Dict[Any, Any],
    n_theta: int,
) -> Tuple[List[Host], List[Feature]]:
    """Build the hosts and the features of one step.

    `detections` is `[(detection, node id)]` in canonical order.  A host the
    data structure names is recorded on the feature as KNOWN:
      * a peak listed as a member of a chain             -> that chain
      * a Corner whose `rho` names its chain             -> that chain
      * an Orientation anchor listed as a Contour member -> that Contour
      * a RegionShape (its anchor is its chain's anchor) -> that chain
    """
    hosts: List[Host] = []
    member_of: Dict[Tuple[str, Any], List[Any]] = {}
    for det, node in detections:
        h = host_of(det, positions, node, n_theta)
        if h is None:
            continue
        hosts.append(h)
        for a in _members(det):
            member_of.setdefault((det.k, a), []).append(det.alpha)
    feats: List[Feature] = []
    for det, node in detections:
        if det.k not in FEATURE_KINDS and det.k not in KNOWN_ONLY_KINDS:
            continue
        pos = _xy(positions, det.alpha)
        known: List[Tuple[str, Any]] = []
        if det.k == "RegionShape":
            # its anchor IS its chain's anchor; its position is the one the
            # chain registered (the chain's own centre)
            if pos is None:
                continue
            feats.append(Feature(kind=det.k, key=det.alpha, node=node, pos=pos,
                                 tangent=None,
                                 known=(("PEAK_CHAIN", det.alpha),),
                                 only_known=True))
            continue
        if det.k == "CONTOUR_PEAK":
            known = [("PEAK_CHAIN", k) for k in member_of.get(("PEAK_CHAIN", det.alpha), ())]
        elif det.k == "Orientation":
            known = [("Contour", k) for k in member_of.get(("Contour", det.alpha), ())]
        elif det.k == "Corner":
            chain = _role(det, "chain")
            vertex = _role(det, "vertex")
            if chain is not None:
                known = [("PEAK_CHAIN", chain)]
                vpos = _xy(positions, vertex) if vertex is not None else None
                if vpos is not None:
                    pos = vpos          # the corner IS that vertex of its chain
        if pos is None:
            continue
        feats.append(Feature(kind=det.k, key=det.alpha, node=node, pos=pos,
                             tangent=feature_tangent(det, n_theta),
                             known=tuple(known)))
    return hosts, feats
