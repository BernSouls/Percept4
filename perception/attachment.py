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
smallest distances to DISTINCT hosts of the kind.  Two near-equally plausible
hosts are not a measurement.

TANGENT AGREEMENT (Amendment 2, section 2.3) is the one thing that can still
decide between near-equidistant hosts: a feature that has a tangent attaches
to the contender it PARALLELS (angle <= PARALLEL_MAX) when no other contender
is nearly as parallel (the next angle is at least PARALLEL_MARGIN larger).
Otherwise nothing is written.

A host that the data structure itself names (a chain's listed member peak, a
Corner's own chain, a Contour's listed member) is KNOWN: no test, it is a
recorded fact.  The membership flag IDENTIFIES the host; the attachment
supplies the MEASUREMENT (where on it, how far, at what angle).  That is not
circular: the flag carries no geometry.

SEVERAL NAMED HOSTS ARE SEVERAL FACTS (Amendment 3.1).  An Orientation cell
where the cell graph branches is listed by every Contour that ends on it
(three, four, five of them).  That is not an ambiguity: the ambiguity rule is
about a guess between candidates, and nothing here is guessed.  The feature
is attached to EACH host that names it and measured on each.

AN ORIENTATION MUST PASS TRUTH (Amendment 3.2).  A feature kind's own
orientation enters `tangent_angle` and tangent agreement only if renderer
truth supports it (`TANGENT_KINDS`); for the others the angle is ABSENT and
only the place is written.

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

#: "parallel": within one step of the peak's twelve orientation channels
PARALLEL_MAX: float = math.pi / 12.0
#: ... and the next contender at least one step less parallel
PARALLEL_MARGIN: float = math.pi / 12.0

#: outcomes of one (feature, host kind) attempt
KNOWN, UNAMBIGUOUS, AMBIGUOUS, NO_HOST = "known", "unambiguous", "ambiguous", "no_host"
#: the data names several hosts of the kind; the feature is attached to each
KNOWN_SEVERAL = "known_several"
#: the host was near-equidistant with another and was chosen by its tangent
BY_TANGENT = "by_tangent"


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
    by_tangent: bool = False     # chosen among near-equidistant hosts by its tangent


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


def _by_tangent(f: Feature, contenders: Sequence[Tuple[Host, _Poly]],
                scale: float, parallel_max: float, parallel_margin: float,
                known: bool) -> Optional[Attachment]:
    """TANGENT AGREEMENT among near-equidistant contenders, or None.

    The feature attaches to the contender it parallels (angle <= parallel_max)
    when the next most parallel contender is at least `parallel_margin` less
    so.  A contender whose angle cannot be measured cannot be ruled out, so it
    blocks the decision."""
    if f.tangent is None or not contenders:
        return None
    scored = []
    for h, poly in contenders:
        a = _measure(f, h, poly, scale, known)
        if a.tangent_angle is None:
            return None
        scored.append((a.tangent_angle, a))
    scored.sort(key=lambda t: t[0])
    if scored[0][0] > parallel_max + 1e-12:
        return None
    if len(scored) > 1 and (scored[1][0] - scored[0][0]) < parallel_margin - 1e-12:
        return None
    best = scored[0][1]
    return Attachment(feature=best.feature, host=best.host, arc_pos=best.arc_pos,
                      arc_end_dist=best.arc_end_dist, perp_dist=best.perp_dist,
                      tangent_angle=best.tangent_angle, known=known,
                      by_tangent=True)


def attach(
    features: Iterable[Feature],
    hosts: Iterable[Host],
    scale: float,
    d_max: float = D_MAX,
    margin: float = AMBIGUITY_MARGIN,
    parallel_max: float = PARALLEL_MAX,
    parallel_margin: float = PARALLEL_MARGIN,
    use_tangent: bool = True,
    every_named: bool = True,
) -> Tuple[List[Attachment], Dict[Tuple[str, str], Dict[str, int]]]:
    """Attach every feature to its host of each kind, or to none.

    Returns `(attachments, stats)`; `stats[(feature kind, host kind)]` counts
    the outcomes KNOWN / KNOWN_SEVERAL / UNAMBIGUOUS / BY_TANGENT / AMBIGUOUS /
    NO_HOST, one per feature and host kind (a KNOWN_SEVERAL feature yields one
    attachment per host that names it).
    `use_tangent=False` is the distance-only rule and `every_named=False` the
    Amendment-2 treatment of several named hosts (tangent agreement or
    nothing); both exist for the before/after reports.  Deterministic: hosts
    and features are processed in the order given (the callers sort
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
            st = stats.setdefault((f.kind, hk), {
                KNOWN: 0, KNOWN_SEVERAL: 0, UNAMBIGUOUS: 0, BY_TANGENT: 0,
                AMBIGUOUS: 0, NO_HOST: 0})
            keys = []
            for k in named.get(hk, ()):
                if (hk, k) in by_key and k not in keys:
                    keys.append(k)
            if len(keys) == 1:
                h, poly = by_key[(hk, keys[0])]
                out.append(_measure(f, h, poly, scale, True))
                st[KNOWN] += 1
                continue
            if len(keys) > 1 and every_named:
                # several recorded memberships are several facts: the
                # feature is on every one of these hosts, measured on each
                for k in keys:
                    h, poly = by_key[(hk, k)]
                    out.append(_measure(f, h, poly, scale, True))
                st[KNOWN_SEVERAL] += 1
                continue
            if len(keys) > 1:
                # [Amendment 2 behaviour, kept for the before/after report]
                # one of them, if the feature parallels exactly one
                got = (_by_tangent(f, [by_key[(hk, k)] for k in keys], scale,
                                   parallel_max, parallel_margin, True)
                       if use_tangent else None)
                if got is None:
                    st[AMBIGUOUS] += 1
                else:
                    out.append(got)
                    st[BY_TANGENT] += 1
                continue
            if f.only_known:
                st[NO_HOST] += 1
                continue
            dists: List[Tuple[float, int]] = []
            for i, (h, poly) in enumerate(by_kind[hk]):
                if h.node is not None and h.node == f.node:
                    continue
                dists.append((poly.project(f.pos)[0] / scale, i))
            if not dists:
                st[NO_HOST] += 1
                continue
            best, best_i = min(dists)
            if best > d_max:
                st[NO_HOST] += 1
                continue
            rivals = [d for d, i in dists if i != best_i and (d - best) < margin]
            if not rivals:
                h, poly = by_kind[hk][best_i]
                out.append(_measure(f, h, poly, scale, False))
                st[UNAMBIGUOUS] += 1
                continue
            # near-equidistant hosts: the contenders are the admissible ones
            contenders = [by_kind[hk][i] for d, i in sorted(dists, key=lambda t: t[1])
                          if d <= d_max and (d - best) < margin]
            got = (_by_tangent(f, contenders, scale, parallel_max,
                               parallel_margin, False) if use_tangent else None)
            if got is None:
                st[AMBIGUOUS] += 1
            else:
                out.append(got)
                st[BY_TANGENT] += 1
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


#: [Amendment 3.2] the kinds whose own orientation renderer truth supports
#: (diagnostics/results/phase6_tangent_truth.txt: share of a kind's
#: detections on smooth synthetic boundary within 15 degrees of truth; the
#: bar is one half):
#:   CONTOUR_PEAK 0.97, Orientation 0.99 of the true TANGENT -> used
#:   EndStopped 0.09.  On a filled shape's boundary the winning channel is
#:       the locally dominant one in 3 % of the detections and holds a median
#:       6 % of its energy: the end-stop score is a relative suppression, so
#:       an off-axis leak wins along a smooth edge.  (On an isolated bar the
#:       slot IS the bar's orientation.) -> ABSENT
#:   ContourFragment 0.00, median error 90.0: its chord is the boundary
#:       NORMAL by construction, because the stream-2 tracer runs across
#:       edges -> ABSENT (a constant is not a measurement)
#:   Corner 0.24 of the true (internal) bisector, 33 detections -> ABSENT
TANGENT_KINDS: Tuple[str, ...] = ("CONTOUR_PEAK", "Orientation")


def feature_tangent(det: Any, n_theta: int) -> Optional[float]:
    """The orientation the attachment may use: `raw_orientation` for the kinds
    truth supports, None (ABSENT) for every other kind."""
    if det.k not in TANGENT_KINDS:
        return None
    return raw_orientation(det, n_theta)


def raw_orientation(det: Any, n_theta: int) -> Optional[float]:
    """What the detection itself says its orientation is (axial, radians), or
    None when it says nothing.  Read by `feature_tangent` and by the truth
    diagnostic, which is how a kind gets into `TANGENT_KINDS` or stays out.

    Sources:
      CONTOUR_PEAK     slot 0, the field's theta index over 12 channels: the
                       ridge tangent
      Orientation      slot 0, the orientation bin over `n_theta` (bin
                       centre): the edge tangent
      EndStopped       slot 0, the bank channel whose response ends, over
                       `n_theta` ('round' quantiser).  It is NOT a carrier: on
                       isolated bars the slot sits a median 7.5 degrees from
                       the bar, so no 90-degree turn is applied.  Not in
                       `TANGENT_KINDS`: on a boundary it is a leak channel
      ContourFragment  the chord direction of the fragment, carried on the
                       detection by the lens (`_frag_tangent`).  The signature
                       alone leaves two candidates (theta +/- tvr).  Not in
                       `TANGENT_KINDS`: the chord crosses the boundary
      Corner           the BISECTOR: the CORNER class' own `orient` slot (12
                       bins over [0, pi), bin centre).  A junction-class
                       Corner with two arms: ref_theta + gap0 / 2; with any
                       other arm count it has no single direction.  Not in
                       `TANGENT_KINDS`: not within reach of the true bisector
    """
    q = det.sigma.quantized
    if det.k == "ContourFragment":
        t = getattr(det, "_frag_tangent", None)
        return None if t is None else float(t) % math.pi
    if not q:
        return None
    if det.k == "CONTOUR_PEAK":
        return (float(q[0]) * math.pi / 12.0) % math.pi
    if det.k == "Orientation":
        return ((float(q[0]) + 0.5) * math.pi / float(n_theta)) % math.pi
    if det.k == "EndStopped":
        return (float(q[0]) * math.pi / float(n_theta)) % math.pi
    if det.k == "Corner":
        cls = getattr(det.sigma, "class_", "")
        if cls == "CORNER" and len(q) > 1:
            return ((float(q[1]) + 0.5) * math.pi / 12.0) % math.pi
        if cls == "JUNCTION" and len(q) > 4 and int(round(float(q[0]))) == 2:
            ref = (float(q[1]) + 0.5) * 2.0 * math.pi / 24.0
            gap = (float(q[4]) + 0.5) * 2.0 * math.pi / 8.0
            return (ref + gap / 2.0) % math.pi
        return None
    return None


def feature_position(det: Any, positions: Dict[Any, Any]) -> Optional[Tuple[float, float]]:
    """Where the feature is.  A ContourFragment's anchor is its parent
    contour's region (one position for all of that contour's fragments), so
    its own midpoint, carried by the lens, is used; everything else is at its
    anchor."""
    if det.k == "ContourFragment":
        mid = getattr(det, "_frag_mid", None)
        if mid is not None and len(mid) >= 2:
            return (float(mid[0]), float(mid[1]))
    return _xy(positions, det.alpha)


def feature_key(det: Any) -> Any:
    """Canonical identity of a feature detection.  Its anchor, except for a
    ContourFragment: every fragment of a contour carries that contour's
    region as its anchor, so the fragment index is part of the identity."""
    if det.k == "ContourFragment":
        return (det.alpha, _role(det, "fragment"))
    return det.alpha


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
        pos = feature_position(det, positions)
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
        feats.append(Feature(kind=det.k, key=feature_key(det), node=node, pos=pos,
                             tangent=feature_tangent(det, n_theta),
                             known=tuple(known)))
    return hosts, feats
