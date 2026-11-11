"""[STAGE 6 / R5] The CurveComplex link rule: one step of geometry.

Which Contours belong to one complex?  The old rule read a temporal EMA
(`Edge.weight` of CONTOUR_CONTINUITY / JUNCTION_CONTOUR).  On the first
presentation that weight is 0.0, so every complex was one Contour; with the
weight available every pair within reach linked and one complex swallowed the
graph (stage 5, A2 trial; reproduced on the multi-step route in stage 6, 0a).

This rule (pre-registered, diagnostics/results/phase6_preregistration.md,
section 4) reads no weight at all:

  * evidence is geometry between contour ENDS in this step: the gap between
    the two end vertices, and the bend, i.e. how far the step from one end to
    the other departs from the direction each contour is travelling in there;
  * a link is possible iff gap <= GAP_MAX and bend <= BEND_MAX;
  * each END takes at most one link (best first), so a Contour has degree
    <= 2 and a complex is a simple path or a cycle;
  * a Junction within GAP_MAX of an end takes that end (the nearest one,
    unless a second junction is nearly as near); an end a junction holds is
    not available for a contour link.  A junction may hold any number of ends.

Pure geometry: no graph, no state.  `lenses/v4.py` (CurveComplexLens) and
`diagnostics/phase6_complex_rule.py` both call `group`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: adjacent cells of the region grid, the diagonal included (1.77 scales)
GAP_MAX: float = 1.8
#: one step of direction on that grid
BEND_MAX: float = math.pi / 4.0
#: two junctions this close in distance (as seen from an end) are not told apart
JUNCTION_MARGIN: float = 0.5
_EPS = 1e-9


@dataclass(frozen=True)
class ContourEnds:
    """A Contour as the rule sees it: its two ends and where each one heads."""
    key: Any                                   # canonical identity (sortable)
    ends: Tuple[Tuple[float, float], Tuple[float, float]]
    out: Tuple[Tuple[float, float], Tuple[float, float]]   # unit, pointing away


def contour_ends(key: Any, pts: Sequence[Tuple[float, float]],
                 mean_theta: Optional[float]) -> Optional[ContourEnds]:
    """Ends and outward directions of a contour polyline (member order).

    With two or more distinct vertices the outward direction of an end is the
    last segment, pointing away from the contour.  A one-vertex contour has no
    segment: its two "ends" are the same point, heading the two ways along its
    own mean orientation.  None when neither is available."""
    p = [(float(x), float(y)) for x, y in pts]
    if not p:
        return None

    def unit(dx, dy):
        n = math.hypot(dx, dy)
        return (dx / n, dy / n) if n > 0.0 else None

    first = last = None
    if len(p) >= 2:
        for q in p[1:]:
            first = unit(p[0][0] - q[0], p[0][1] - q[1])
            if first is not None:
                break
        for q in reversed(p[:-1]):
            last = unit(p[-1][0] - q[0], p[-1][1] - q[1])
            if last is not None:
                break
    if first is None or last is None:
        if mean_theta is None:
            return None
        t = (math.cos(mean_theta), math.sin(mean_theta))
        return ContourEnds(key=key, ends=(p[0], p[-1]),
                           out=((-t[0], -t[1]), t))
    return ContourEnds(key=key, ends=(p[0], p[-1]), out=(first, last))


def _angle(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    d = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
    return math.acos(d)


def link_evidence(a: ContourEnds, ea: int, b: ContourEnds, eb: int,
                  scale: float) -> Tuple[float, float]:
    """(gap in scales, bend in radians) between end `ea` of a and `eb` of b."""
    pa, pb = a.ends[ea], b.ends[eb]
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    gap_px = math.hypot(dx, dy)
    into_b = (-b.out[eb][0], -b.out[eb][1])
    if gap_px <= 0.0:
        return 0.0, _angle(a.out[ea], into_b)
    c = (dx / gap_px, dy / gap_px)
    return gap_px / scale, max(_angle(a.out[ea], c), _angle(c, into_b))


def group(
    contours: Sequence[ContourEnds],
    junctions: Sequence[Tuple[Any, Tuple[float, float]]],
    scale: float,
    gap_max: float = GAP_MAX,
    bend_max: float = BEND_MAX,
    junction_margin: float = JUNCTION_MARGIN,
) -> Tuple[List[Tuple[List[Any], List[Any]]], Dict[str, int]]:
    """Group contours (and the junctions that hold their ends) into complexes.

    Returns `(complexes, stats)`; a complex is `(contour keys, junction keys)`,
    both sorted, and the complexes are sorted by their first contour key.
    Junctions that hold no end are not in any complex.  Deterministic: every
    order is the canonical key order and every tie is broken by it.
    `stats["max_contour_degree"]` is the largest number of contour-to-contour
    links any one Contour took (never above 2).
    """
    scale = float(scale)
    cs = sorted(contours, key=lambda c: c.key)
    js = sorted(junctions, key=lambda j: j[0])
    stats = {"candidates": 0, "links": 0, "ends_on_junctions": 0,
             "ends_junction_ambiguous": 0, "max_contour_degree": 0}
    taken: Dict[Tuple[int, int], Any] = {}        # (contour index, end) -> owner
    j_links: List[Tuple[int, int]] = []           # (contour index, junction index)
    cands: List[Tuple[float, int, int, int, int]] = []
    if scale > 0.0:
        for ci, c in enumerate(cs):
            one_place = c.ends[0] == c.ends[1]
            # a one-vertex contour has ONE place: one junction at most
            for e in ((0,) if one_place else (0, 1)):
                best = second = None
                best_j = -1
                for ji, (_k, pj) in enumerate(js):
                    d = math.hypot(pj[0] - c.ends[e][0], pj[1] - c.ends[e][1]) / scale
                    if best is None or d < best:
                        second, best, best_j = best, d, ji
                    elif second is None or d < second:
                        second = d
                if best is None or best > gap_max + _EPS:
                    continue
                if second is not None and (second - best) < junction_margin:
                    stats["ends_junction_ambiguous"] += 1
                    continue
                taken[(ci, e)] = ("J", best_j)
                if one_place:
                    taken[(ci, 1)] = ("J", best_j)
                j_links.append((ci, best_j))
                stats["ends_on_junctions"] += 1
        for ai in range(len(cs)):
            for bi in range(ai + 1, len(cs)):
                for ea in (0, 1):
                    if (ai, ea) in taken:
                        continue
                    for eb in (0, 1):
                        if (bi, eb) in taken:
                            continue
                        gap, bend = link_evidence(cs[ai], ea, cs[bi], eb, scale)
                        if gap <= gap_max + _EPS and bend <= bend_max + _EPS:
                            cands.append((gap / gap_max + bend / bend_max,
                                          ai, ea, bi, eb))
        stats["candidates"] = len(cands)
        cands.sort()
    parent = list(range(len(cs) + len(js)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    degree = [0] * len(cs)
    for _cost, ai, ea, bi, eb in cands:
        if (ai, ea) in taken or (bi, eb) in taken:
            continue
        if find(ai) == find(bi) and (cs[ai].ends[0] == cs[ai].ends[1]
                                     or cs[bi].ends[0] == cs[bi].ends[1]):
            # a one-vertex contour has ONE place: linking both of its "ends"
            # to the same partner would count that place twice
            continue
        taken[(ai, ea)] = ("C", bi)
        taken[(bi, eb)] = ("C", ai)
        degree[ai] += 1
        degree[bi] += 1
        union(ai, bi)
        stats["links"] += 1
    stats["max_contour_degree"] = max(degree) if degree else 0
    for ci, ji in j_links:
        union(ci, len(cs) + ji)
    comps: Dict[int, Tuple[List[Any], List[Any]]] = {}
    for ci, c in enumerate(cs):
        comps.setdefault(find(ci), ([], []))[0].append(c.key)
    held = {ji for _ci, ji in j_links}
    for ji, (k, _p) in enumerate(js):
        if ji in held:
            comps.setdefault(find(len(cs) + ji), ([], []))[1].append(k)
    out = [(sorted(cc), sorted(jj)) for cc, jj in comps.values() if cc]
    out.sort(key=lambda t: t[0][0])
    return out, stats
