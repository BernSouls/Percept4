"""matching/relation_walker.py -- [STAGE 6 / TRACK M] a correspondence grown
along measured relations.

THE CONTRACT (the stage-6 brief, M2; constants pre-registered in
diagnostics/results/phase6_preregistration.md, section 6 and Amendment 4).

  * A pair of nodes is admitted BECAUSE a relation around it agrees: it is
    reached from an already admitted pair along two edges of ONE relation
    class whose measured axes agree.  Signature similarity is a GATE (same
    lens type, graded similarity >= SIGNATURE_GATE); it adds to no score.
  * Only the FRAME (matching/relation_frame.py) is grown.  ATTRIBUTES are
    attached afterwards through same-place CO_LOCATED edges.  There is no
    position prior, and CO_LOCATED / proximity edges are never growth edges.
  * An axis that is not measured on both edges weighs nothing either way; an
    edge with no measured axis is not a relation this walker can test.
  * SEEDS are a small fixed number of pairs that share rare relation tokens.
  * ORDER: features admitted on an admitted pair of boundaries keep their arc
    order (up to the direction the boundary is walked in, which is a gauge).
  * One-to-one, deterministic (every tie is broken by a canonical node key
    that does not depend on node ids or on the hash seed), budgeted.
  * SCORE: Jaccard on preserved relations over the mapped part.  Below the
    minimum support the pair of graphs is UNSUPPORTED and has no score.

THE SWITCH.  `match_evidence` is the one dispatch point:
P.MATCH_RELATION_WALKER off (default) -> `memory.cores.evidence_score`,
called with the same arguments and returning what it returns (the old path,
untouched); on -> this walker.  `evidence_score` itself gains no term.

Nothing here reads a position, an activation or a weight.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import parameters as P
from core.schema import (
    CIRCULAR_EDGE_CYCLE_LENGTHS,
    EdgeParamsMixture,
    reverse_edge_params,
)
from core.types import LEGACY_TO_CATEGORY, EdgeCategory, EdgeType, is_undirected_edge
from matching.relation_frame import is_frame, measured_axes

# ---------------------------------------------------------------------------
# the pre-registered constants
# ---------------------------------------------------------------------------
#: the signature gate: same lens type and graded similarity at least this
SIGNATURE_GATE: float = 0.5
#: number of seeds
N_SEEDS: int = 8
#: a relation token is RARE when at most this many nodes carry it, in each graph
RARE_TOKEN_MAX_NODES: int = 2
#: a seed shares at least this many rare tokens
SEED_MIN_SHARED_TOKENS: int = 2
#: two measured values agree when their bins differ by at most this
BIN_TOLERANCE: float = 1.0
#: pairs examined (signature similarities computed) <= this share of
#: |FRAME_q| x |FRAME_r|
BUDGET_FRACTION: float = 0.05
#: minimum support: this many frame pairs AND this share of the smaller frame
MIN_SUPPORT_PAIRS: int = 8
MIN_SUPPORT_FRACTION: float = 0.10


def is_growth_edge_type(et: Any) -> bool:
    """A relation the walker may grow along: any declared edge type that is
    not PROXIMITY in the repository's own category table (CO_LOCATED, CONTACT,
    TOPOLOGY, DEGREE, CROSS_SCALE_IDENTITY are).  An undeclared type is not."""
    cat = et if isinstance(et, EdgeCategory) else LEGACY_TO_CATEGORY.get(et)
    return cat is not None and cat is not EdgeCategory.PROXIMITY


def _etype_name(et: Any) -> str:
    return str(getattr(et, "name", et))


def _node_key(lens: str, sig: Any) -> Tuple:
    """Canonical identity of a node: what it IS, never its id."""
    return (lens, str(getattr(sig, "class_", "")), str(getattr(sig, "relation", "")),
            tuple(float(v) for v in (getattr(sig, "quantized", ()) or ())))


@dataclass(frozen=True)
class _Half:
    """One relation as seen from one of its two ends."""
    cls: Tuple[str, str, str]          # (edge type, 'o' / 'i' / 'u', lens type of the far node)
    far: int
    bins: Tuple[Tuple[int, float], ...]     # (axis, bin) of every measured axis
    cycles: Tuple[Tuple[int, int], ...]     # (axis, cycle) of the circular ones
    ordinal: Optional[float]
    etype: Any

    def token(self) -> Tuple:
        return (self.cls, self.bins)


def _slack(a: _Half, b: _Half) -> Optional[int]:
    """None when the two relations do NOT agree; otherwise the number of
    shared measured axes on which they differ by one bin (0 = identical).

    Agreement: at least one axis measured on BOTH, and every such axis within
    BIN_TOLERANCE bins (circular axes wrapped).  An axis measured on one side
    only weighs nothing."""
    da = dict(a.bins)
    cyc = dict(a.cycles)
    shared = 0
    slack = 0
    for ax, vb in b.bins:
        va = da.get(ax)
        if va is None:
            continue
        shared += 1
        d = abs(va - vb)
        c = cyc.get(ax)
        if c:
            d = min(d, c - d)
        if d > BIN_TOLERANCE + 1e-9:
            return None
        if d > 1e-9:
            slack += 1
    return slack if shared else None


def _dominant(ep: Any) -> Any:
    if isinstance(ep, EdgeParamsMixture):
        return ep.dominant if ep.protos else None
    return ep


class _Index:
    """One graph, prepared: the frame, the relations seen from each node, the
    attributes at each node.  Every list is in canonical order."""

    def __init__(self, sg: Any, sigs: Dict[int, Tuple[str, Any]]) -> None:
        self.lens = {n: lt for n, (lt, _s) in sigs.items()}
        self.sig = {n: s for n, (_lt, s) in sigs.items()}
        self.key = {n: _node_key(self.lens[n], self.sig[n]) for n in sigs}
        self.frame = sorted((n for n in sigs if is_frame(self.lens[n])),
                            key=lambda n: (self.key[n], n))
        frame = set(self.frame)
        self.half: Dict[int, Dict[Tuple, List[_Half]]] = {n: {} for n in self.frame}
        #: (a, b, edge type name) -> the relation as seen from a
        self.pair: Dict[Tuple[int, int, str], _Half] = {}
        #: measured growth edges, once each: (s, d, edge type name)
        self.edges: List[Tuple[int, int, str]] = []
        #: feature -> {host: arc position}, host -> {feature: arc position}
        self.on_host: Dict[int, Dict[int, float]] = {}
        self.hosts: Dict[int, Dict[int, float]] = {}
        #: frame node -> {attribute lens type: [attribute nodes]}
        self.attrs: Dict[int, Dict[str, List[int]]] = {}
        self.unmeasured_edges = 0
        params = getattr(sg, "edge_params", None) or {}
        for s, d, et in sorted(sg.edges, key=lambda e: (e[0], e[1], _etype_name(e[2]))):
            if s == d or s not in self.lens or d not in self.lens:
                continue
            if et == EdgeType.CO_LOCATED:
                for a, b in ((s, d), (d, s)):
                    if a in frame and b not in frame:
                        self.attrs.setdefault(a, {}).setdefault(self.lens[b], []).append(b)
                continue
            if s not in frame or d not in frame or not is_growth_edge_type(et):
                continue
            ep = _dominant(params.get((s, d, et)))
            axes = measured_axes(ep)
            if not axes:
                self.unmeasured_edges += 1
                continue
            name = _etype_name(et)
            und = is_undirected_edge(et)
            cyc = CIRCULAR_EDGE_CYCLE_LENGTHS.get(getattr(ep, "class_", ""), {})

            def half(ep_seen, far, direction):
                ax = measured_axes(ep_seen)
                return _Half(
                    cls=(name, direction, self.lens[far]), far=far,
                    bins=tuple((i, float(ep_seen.quantized[i])) for i in ax),
                    cycles=tuple(sorted((i, int(c)) for i, c in cyc.items() if i in ax)),
                    ordinal=getattr(ep_seen, "ordinal", None), etype=et)

            hs = half(ep, d, "u" if und else "o")
            hd = half(reverse_edge_params(ep) if und else ep, s, "u" if und else "i")
            if (s, d, name) in self.pair:
                continue                      # an undirected edge stored both ways
            self.half[s].setdefault(hs.cls, []).append(hs)
            self.half[d].setdefault(hd.cls, []).append(hd)
            self.pair[(s, d, name)] = hs
            if und:
                self.pair[(d, s, name)] = hd
            self.edges.append((s, d, name))
            if et == EdgeType.BOUNDARY_ATTACHMENT and hs.ordinal is not None:
                self.on_host.setdefault(d, {})[s] = float(hs.ordinal)
                self.hosts.setdefault(s, {})[d] = float(hs.ordinal)
        for n in self.frame:
            for cls in self.half[n]:
                self.half[n][cls].sort(key=lambda h: (self.key[h.far], h.bins, h.far))
        for n in self.attrs:
            for lt in self.attrs[n]:
                self.attrs[n][lt] = sorted(set(self.attrs[n][lt]),
                                           key=lambda a: (self.key[a], a))

    def closed(self, host: int) -> bool:
        """A PEAK_CHAIN and a Contour keep their closure flag in slot 3."""
        q = getattr(self.sig.get(host), "quantized", ()) or ()
        return (self.lens.get(host) in ("PEAK_CHAIN", "Contour")
                and len(q) > 3 and float(q[3]) >= 0.5)


def _order_consistent(pairs: List[Tuple[float, float]], cyclic: bool) -> bool:
    """Is one arc order kept on both boundaries, up to the walking direction?

    `pairs` = (arc position on the query boundary, on the reference boundary)
    of the features admitted on one pair of boundaries.
    OPEN boundary: every two features with distinct positions on both sides
    must be in the same relative order on both (all pairs one way), or in the
    opposite order on both (all pairs the other way: the boundary is walked
    backwards, which is a gauge, not a difference).  A mix is a reversal.
    CLOSED boundary: the same up to a rotation of the origin.
    Features at one position on either side constrain nothing."""
    if len(pairs) < 3:
        return True
    if not cyclic:
        sign = 0
        n = len(pairs)
        for i in range(n):
            a0, b0 = pairs[i]
            for j in range(i + 1, n):
                a1, b1 = pairs[j]
                if a0 == a1 or b0 == b1:
                    continue
                s = 1 if (a1 - a0) * (b1 - b0) > 0.0 else -1
                if sign == 0:
                    sign = s
                elif s != sign:
                    return False
        return True
    # closed: drop repeated positions, then the reference positions read in
    # query order must be a rotation of a rising or of a falling sequence
    seen_a, seen_b, pts = set(), set(), []
    for a0, b0 in sorted(pairs):
        if a0 in seen_a or b0 in seen_b:
            continue
        seen_a.add(a0)
        seen_b.add(b0)
        pts.append(b0)
    n = len(pts)
    if n < 4:
        return True
    falls = sum(1 for i in range(n) if pts[(i + 1) % n] < pts[i])
    rises = n - falls
    return falls <= 1 or rises <= 1


@dataclass
class WalkResult:
    mapping: Dict[int, int] = field(default_factory=dict)       # frame: query -> reference
    seeds: List[Tuple[int, int]] = field(default_factory=list)
    supported: bool = False
    score: Optional[float] = None
    preserved: int = 0
    relations_q: int = 0
    relations_r: int = 0
    attributes_shared: int = 0
    frame_q: int = 0
    frame_r: int = 0
    examined: int = 0            # signature similarities computed
    edge_checks: int = 0         # relation pairs compared
    budget: int = 0
    budget_exhausted: bool = False
    order_rejections: int = 0
    gate_rejections: int = 0
    dense: int = 0               # |nodes_q| x |nodes_r|, the old matcher's matrix


def walk(Q: Any, R: Any, q_sigs: Dict[int, Tuple[str, Any]],
         r_sigs: Dict[int, Tuple[str, Any]]) -> WalkResult:
    """Grow one correspondence between the frames of two graphs."""
    from matching.structural_similarity import _sig_sim

    gq, gr = _Index(Q, q_sigs), _Index(R, r_sigs)
    res = WalkResult(frame_q=len(gq.frame), frame_r=len(gr.frame),
                     dense=len(q_sigs) * len(r_sigs))
    res.budget = int(BUDGET_FRACTION * len(gq.frame) * len(gr.frame))
    sim_cache: Dict[Tuple[int, int], float] = {}

    def similarity(u: int, v: int) -> Optional[float]:
        """Graded signature similarity, or None once the budget is spent."""
        got = sim_cache.get((u, v))
        if got is not None:
            return got
        if gq.lens[u] != gr.lens[v]:
            sim_cache[(u, v)] = 0.0
            return 0.0
        if res.examined >= res.budget:
            res.budget_exhausted = True
            return None
        res.examined += 1
        got = float(_sig_sim(gq.sig[u], gr.sig[v]))
        sim_cache[(u, v)] = got
        return got

    map_q: Dict[int, int] = {}
    map_r: Dict[int, int] = {}
    rejected: set = set()            # pairs the order constraint has refused
    #: (u, v) -> {admitted pair: slack of its best agreeing relation}
    support: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
    heap: List[Tuple] = []

    def order_ok(u: int, v: int) -> bool:
        # (a) the new pair as a FEATURE on already admitted boundaries
        for hq, aq in sorted(gq.on_host.get(u, {}).items()):
            hr = map_q.get(hq)
            if hr is None:
                continue
            ar = gr.on_host.get(v, {}).get(hr)
            if ar is None:
                continue
            pts = [(aq, ar)]
            for f, af in gq.hosts.get(hq, {}).items():
                g = map_q.get(f)
                if g is None or f == u:
                    continue
                ag = gr.hosts.get(hr, {}).get(g)
                if ag is not None:
                    pts.append((af, ag))
            if not _order_consistent(pts, gq.closed(hq) and gr.closed(hr)):
                return False
        # (b) the new pair as a BOUNDARY under already admitted features
        pts = []
        for f, af in gq.hosts.get(u, {}).items():
            g = map_q.get(f)
            if g is None:
                continue
            ag = gr.hosts.get(v, {}).get(g)
            if ag is not None:
                pts.append((af, ag))
        return _order_consistent(pts, gq.closed(u) and gr.closed(v))

    def admit(u: int, v: int) -> None:
        map_q[u] = v
        map_r[v] = u
        for cls in sorted(gq.half[u]):
            theirs = gr.half[v].get(cls)
            if not theirs:
                continue
            for hu in gq.half[u][cls]:
                if hu.far in map_q:
                    continue
                for hv in theirs:
                    if hv.far in map_r:
                        continue
                    if (hu.far, hv.far) in rejected:
                        continue
                    res.edge_checks += 1
                    s = _slack(hu, hv)
                    if s is None:
                        continue
                    known = (hu.far, hv.far) in sim_cache
                    sim = similarity(hu.far, hv.far)
                    if sim is None:
                        continue                 # the budget is spent
                    if sim < SIGNATURE_GATE:
                        res.gate_rejections += int(not known)
                        continue
                    sup = support.setdefault((hu.far, hv.far), {})
                    old = sup.get((u, v))
                    if old is None or s < old:
                        sup[(u, v)] = s
                    heapq.heappush(heap, (
                        -len(sup), sum(sup.values()), -sim,
                        gq.key[hu.far], gr.key[hv.far], hu.far, hv.far))

    # ---- seeds: pairs that share rare relation tokens --------------------
    def carriers(g: _Index) -> Dict[Tuple, List[int]]:
        out: Dict[Tuple, List[int]] = {}
        for n in g.frame:
            seen = set()
            for cls in g.half[n]:
                for h in g.half[n][cls]:
                    t = h.token()
                    if t not in seen:
                        seen.add(t)
                        out.setdefault(t, []).append(n)
        return out

    cq, cr = carriers(gq), carriers(gr)
    shared: Dict[Tuple[int, int], int] = {}
    for t in cq:
        if t not in cr:
            continue
        if len(cq[t]) > RARE_TOKEN_MAX_NODES or len(cr[t]) > RARE_TOKEN_MAX_NODES:
            continue
        for u in cq[t]:
            for v in cr[t]:
                shared[(u, v)] = shared.get((u, v), 0) + 1
    cands = []
    for (u, v), n in sorted(shared.items(), key=lambda kv: (
            -kv[1], gq.key[kv[0][0]], gr.key[kv[0][1]], kv[0])):
        if n < SEED_MIN_SHARED_TOKENS:
            continue
        sim = similarity(u, v)
        if sim is None:
            break
        if sim >= SIGNATURE_GATE:
            cands.append((-n, -sim, gq.key[u], gr.key[v], u, v))
    cands.sort()
    for _n, _s, _ku, _kv, u, v in cands:
        if len(res.seeds) >= N_SEEDS:
            break
        if u in map_q or v in map_r:
            continue
        if not order_ok(u, v):
            res.order_rejections += 1
            continue
        res.seeds.append((u, v))
        admit(u, v)

    # ---- growth: best supported first -------------------------------------
    while heap:
        neg, slack, _nsim, _ku, _kv, u, v = heapq.heappop(heap)
        if u in map_q or v in map_r:
            continue
        sup = support.get((u, v))
        if not sup or -neg != len(sup) or slack != sum(sup.values()):
            continue                         # stale: a fresher entry exists
        if not order_ok(u, v):
            res.order_rejections += 1
            rejected.add((u, v))
            continue
        admit(u, v)

    res.mapping = dict(sorted(map_q.items()))
    _score(res, gq, gr, similarity)
    return res


def _score(res: WalkResult, gq: _Index, gr: _Index, similarity: Any) -> None:
    """Jaccard on preserved relations over the mapped part, and the support.

    A relation is (i) a measured growth edge between two mapped frame nodes:
    preserved when its image exists in the same class and agrees; a pair of
    edges with no axis measured on BOTH weighs nothing (it is counted on
    neither side); or (ii) an attribute type attached to a mapped frame node
    through CO_LOCATED: preserved when both nodes carry that type and two of
    the attributes pass the signature gate."""
    m = res.mapping
    inv = {v: u for u, v in m.items()}

    def edge_id(g: _Index, s: int, d: int, name: str) -> Tuple:
        und = g.pair[(s, d, name)].cls[1] == "u"
        return (min(s, d), max(s, d), name) if und else (s, d, name)

    preserved = n_q = n_r = 0
    void_r = set()                     # reference edges whose pair shares no measured axis
    for s, d, name in gq.edges:
        if s not in m or d not in m:
            continue
        hq = gq.pair[(s, d, name)]
        hr = gr.pair.get((m[s], m[d], name))
        if hr is None:
            n_q += 1
            continue
        sl = _slack(hq, hr)
        if sl is None and not ({i for i, _v in hq.bins} & {i for i, _v in hr.bins}):
            void_r.add(edge_id(gr, m[s], m[d], name))
            continue                   # nothing measured on both: weighs nothing
        n_q += 1
        if sl is not None:
            preserved += 1
    for s, d, name in gr.edges:
        if s not in inv or d not in inv:
            continue
        if edge_id(gr, s, d, name) in void_r:
            continue
        n_r += 1
    shared_attr = 0
    for u, v in m.items():
        aq, ar = gq.attrs.get(u, {}), gr.attrs.get(v, {})
        for lt in sorted(set(aq) | set(ar)):
            n_q += int(lt in aq)
            n_r += int(lt in ar)
            if lt in aq and lt in ar and any(
                    _attr_similarity(gq, gr, a, b) >= SIGNATURE_GATE
                    for a in aq[lt] for b in ar[lt]):
                shared_attr += 1
    preserved += shared_attr
    res.preserved, res.relations_q, res.relations_r = preserved, n_q, n_r
    res.attributes_shared = shared_attr
    smaller = min(res.frame_q, res.frame_r)
    res.supported = (len(m) >= MIN_SUPPORT_PAIRS
                     and len(m) >= MIN_SUPPORT_FRACTION * smaller)
    denom = n_q + n_r - preserved
    res.score = (preserved / denom if denom > 0 else 0.0) if res.supported else None


def _attr_similarity(gq: _Index, gr: _Index, a: int, b: int) -> float:
    from matching.structural_similarity import _sig_sim
    if gq.lens[a] != gr.lens[b]:
        return 0.0
    return float(_sig_sim(gq.sig[a], gr.sig[b]))


def match_evidence(Q: Any, C: Any, q_sigs: Dict[int, Tuple[str, Any]],
                   c_sigs: Dict[int, Tuple[str, Any]],
                   trace: Optional[Dict[str, Any]] = None
                   ) -> Tuple[Optional[float], Dict[int, int]]:
    """THE ONE SWITCH.  Off (default): `memory.cores.evidence_score`, the old
    matcher, with the same arguments and its own return value.  On: the
    relation walker; the score is None when the pair is UNSUPPORTED."""
    if not bool(getattr(P, "MATCH_RELATION_WALKER", False)):
        from memory.cores import evidence_score
        return evidence_score(Q, C, q_sigs, c_sigs, trace=trace)
    res = walk(Q, C, q_sigs, c_sigs)
    if trace is not None:
        trace["relation_walk"] = res
    return res.score, dict(res.mapping)
