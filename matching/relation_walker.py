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
#: ---------------------------------------------------------------------------
#: [Amendment 7 / D0-D5] the constants fixed BEFORE any of them was measured
#: ---------------------------------------------------------------------------
#: D0: the weight of each half of the objective (equal; see the pre-registration)
W_NODE: float = 1.0
W_RELATION: float = 1.0
#: D2: a shared axis further apart than this (in bins) contradicts; less only
#: lowers the agreement.  A quarter of a turn on an 8-bin axis.
CONTRADICTION_MARGIN: float = 4.0
#: D3: the node gate is relative: above an absolute floor, and within a margin
#: of the node's best similarity in the other graph
NODE_FLOOR: float = 0.30
NODE_MARGIN: float = 0.25
#: D4: how many already admitted features on a pair of boundaries must keep
#: their order with a pair before the order itself proposes it
ORDER_PROPOSAL_MIN: int = 3
#: D5: beam width, in seed sets
BEAM_WIDTH: int = 4
#: pairs examined (signature similarities computed) <= this share of
#: |FRAME_q| x |FRAME_r| ...
BUDGET_FRACTION: float = 0.05
#: ... but never fewer than this many looks per node of the smaller frame (on
#: a small frame 5 % of the product is less than one look per node: 13 nodes
#: would get 8 pairs), and never more than the product itself
BUDGET_FLOOR_PER_NODE: int = 8
#: minimum support: this many frame pairs AND this share of the smaller frame
MIN_SUPPORT_PAIRS: int = 8
MIN_SUPPORT_FRACTION: float = 0.10


#: below this many pairs the whole matrix is cheaper than any fraction of it
SMALL_FRAME_PRODUCT: int = 1000


def budget(frame_q: int, frame_r: int) -> int:
    """How many pairs one walk may examine.

    Amendment 7.13: a SMALL frame (product <= 1000, i.e. fewer than ~32 nodes
    a side) may examine the whole matrix -- the dense matrix there is cheaper
    than the bookkeeping, and 5 % of it is less than one look per node, which
    left the frontier starving (measured: a 13 x 13 pin graph, 104 looks,
    growth stopped at 9 of 13 nodes).  Above it the pre-registered rule
    stands: 5 % of the product, floored at 8 looks per node of the smaller
    frame, never more than the product."""
    product = frame_q * frame_r
    if product <= SMALL_FRAME_PRODUCT:
        return product
    return min(product, max(int(BUDGET_FRACTION * product),
                            BUDGET_FLOOR_PER_NODE * min(frame_q, frame_r)))


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


def _agreement(a: _Half, b: _Half) -> Optional[float]:
    """Graded agreement of two relations, or None when no axis is shared.

    Agreement of ONE shared axis = max(0, 1 - |difference in bins| /
    CONTRADICTION_MARGIN), circular axes wrapped.  The relation's value is the
    mean over its shared axes.  (D0 / 7.0.)"""
    da = dict(a.bins)
    cyc = dict(a.cycles)
    vals = []
    for ax, vb in b.bins:
        va = da.get(ax)
        if va is None:
            continue
        d = abs(va - vb)
        c = cyc.get(ax)
        if c:
            d = min(d, c - d)
        vals.append(max(0.0, 1.0 - d / CONTRADICTION_MARGIN))
    return (sum(vals) / len(vals)) if vals else None


def _contradicts(a: _Half, b: _Half) -> bool:
    """D2: does any axis measured on BOTH disagree by more than the margin?"""
    da = dict(a.bins)
    cyc = dict(a.cycles)
    for ax, vb in b.bins:
        va = da.get(ax)
        if va is None:
            continue
        d = abs(va - vb)
        c = cyc.get(ax)
        if c:
            d = min(d, c - d)
        if d > CONTRADICTION_MARGIN + 1e-9:
            return True
    return False


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
        # [Amendment 7 / D1] THE TOPOLOGICAL TOKEN of every relation as seen
        # from this node: no bins, no similarity.
        self.tok: Dict[int, Dict[Tuple, List[_Half]]] = {n: {} for n in self.frame}
        for n in self.frame:
            for cls in self.half[n]:
                for h in self.half[n][cls]:
                    kind, direction, far_lens = cls
                    t = (self.lens[n], kind, direction, far_lens,
                         len(self.half.get(h.far, {})), self.closed(h.far))
                    self.tok[n].setdefault(t, []).append(h)
        self.degree = {n: sum(len(v) for v in self.half[n].values()) for n in self.frame}
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
    contradicts: int = 0
    #: the terms of the Amendment-7 objective of this mapping
    terms: Dict[str, Any] = field(default_factory=dict)
    #: the hypotheses of the beam: seed count, mapped, Q
    beam: List[Dict[str, Any]] = field(default_factory=list)
    dense: int = 0               # |nodes_q| x |nodes_r|, the old matcher's matrix
    #: the admissions in order: (query node, reference node, "seed" or the
    #: relation classes that supported it).  For diagnostics; decides nothing.
    log: List[Tuple[int, int, Any]] = field(default_factory=list)


def _record_support(sup_d: Dict[Any, Tuple[float, bool]], sup: Any,
                    slack: float, measured: bool) -> None:
    """Record ONE proposal's evidence for a pair (Amendment 7.19).

    `sup_d` maps a support (the admitted pair the proposal came through) to
    (slack, measured).  An ORDER proposal carries NO measured slack: it may
    open a support that no measurement opened (then it records (0.0, False)),
    it may never lower a measured slack, and a measurement arriving later
    REPLACES a weak entry instead of being minimized into it.  Two measured
    proposals through the same support keep the smaller slack.
    """
    prev = sup_d.get(sup)
    if prev is None:
        sup_d[sup] = (slack, measured)
        return
    prev_slack, prev_measured = prev
    if measured and not prev_measured:
        sup_d[sup] = (slack, True)          # the measurement is the evidence now
    elif measured and prev_measured:
        sup_d[sup] = (min(prev_slack, slack), True)
    elif not prev_measured:
        sup_d[sup] = (min(prev_slack, slack), False)   # two order backings: no slack either way


def _measured_slack(sup_d: Dict[Any, Tuple[float, bool]]) -> float:
    """The slack the MEASUREMENTS give a pair: 0 when there are none.  The
    serving order must not read that 0 as perfect agreement -- that is what
    the tier in `_serve_key` is for."""
    return sum(sl for sl, measured in sup_d.values() if measured)


def _serve_key(how: Any, n_support: int, slack: float, sim: float,
               key_q: Tuple, key_r: Tuple, u: int, v: int) -> Tuple:
    """The order proposals are served in (Amendment 4.1 + 7.18).

    TIER first: an order proposal is weaker evidence than a measured relation
    and is served only when no measured proposal is pending -- otherwise a
    proposal with no slack of its own would outrank every measurement.
    Then, within a tier, the 4.1 rank: more supporting pairs first; less total
    slack first; the more similar signature; the canonical keys (which are
    arbitrary with respect to the measurement); the node ids last of all.
    """
    tier = 1 if (isinstance(how, tuple) and how and how[0] == "order") else 0
    return (tier, -n_support, slack, -sim, key_q, key_r, u, v)


def _choose(hypotheses: List["WalkResult"]) -> "WalkResult":
    """[D5 / 7.11] which hypothesis is the walk's answer.

    Q alone prefers a small mapping that is very similar to a large one that
    is slightly less similar (it is a mean, not a sum).  Among hypotheses that
    both pass the support floor the larger mapping wins; Q decides between
    hypotheses that do not both pass it.  Ties go to the earlier seed set.
    """
    supported = [h for h in hypotheses if len(h.mapping) >= MIN_SUPPORT_PAIRS
                 and len(h.mapping) >= MIN_SUPPORT_FRACTION * min(h.frame_q, h.frame_r)]
    pool = supported or hypotheses
    best = pool[0]
    for h in pool[1:]:
        if supported:
            if (len(h.mapping), h.terms["q"]) > (len(best.mapping), best.terms["q"] + 1e-12):
                best = h
        elif h.terms["q"] > best.terms["q"] + 1e-12:
            best = h
    return best


class _Sim:
    """Graded signature similarity with ONE budget for the whole walk."""

    def __init__(self, gq: "_Index", gr: "_Index", res: WalkResult) -> None:
        from matching.structural_similarity import _sig_sim
        self._f = _sig_sim
        self.gq, self.gr, self.res = gq, gr, res
        self.cache: Dict[Tuple[int, int], float] = {}

    def __call__(self, u: int, v: int) -> Optional[float]:
        got = self.cache.get((u, v))
        if got is not None:
            return got
        if self.gq.lens[u] != self.gr.lens[v]:
            self.cache[(u, v)] = 0.0
            return 0.0
        if self.res.examined >= self.res.budget:
            self.res.budget_exhausted = True
            return None
        self.res.examined += 1
        got = float(self._f(self.gq.sig[u], self.gr.sig[v]))
        self.cache[(u, v)] = got
        return got


def _node_gate(sim: Optional[float], best: float) -> bool:
    """[D3] relative node gate: above the floor and within a margin of the
    node's best similarity in the other graph."""
    return sim is not None and sim > NODE_FLOOR and sim >= best - NODE_MARGIN


def _seeds(gq: "_Index", gr: "_Index", sim: _Sim,
           res: WalkResult) -> List[Tuple[int, int]]:
    """[D1] SEEDS from TOPOLOGICAL tokens only, ranked by node similarity.

    A token is (lens type, edge type, direction, far lens type, the far node's
    degree in the growth graph, whether the far node is closed): no bins, no
    similarity.  A rare token is carried by at most RARE_TOKEN_MAX_NODES
    nodes in each graph; a pair shares at least SEED_MIN_SHARED_TOKENS of
    them.  Seeds are one-to-one, ranked by (rare tokens shared, tokens
    shared, node similarity, canonical keys) -- Amendment 7.9."""
    def carriers(g: "_Index") -> Dict[Tuple, List[int]]:
        out: Dict[Tuple, List[int]] = {}
        for n in g.frame:
            for t in g.tok[n]:
                out.setdefault(t, []).append(n)
        return out

    cq, cr = carriers(gq), carriers(gr)
    shared: Dict[Tuple[int, int], int] = {}
    rare: Dict[Tuple[int, int], int] = {}
    for t in sorted(cq):
        if t not in cr:
            continue
        is_rare = (len(cq[t]) <= RARE_TOKEN_MAX_NODES and len(cr[t]) <= RARE_TOKEN_MAX_NODES)
        for u in cq[t]:
            for v in cr[t]:
                shared[(u, v)] = shared.get((u, v), 0) + 1
                if is_rare:
                    rare[(u, v)] = rare.get((u, v), 0) + 1
    cands = []
    for (u, v), n in sorted(shared.items()):
        if n < SEED_MIN_SHARED_TOKENS:
            continue
        s = sim(u, v)
        if s is None or s <= NODE_FLOOR:
            continue
        cands.append((-rare.get((u, v), 0), -n, -s, gq.key[u], gr.key[v], u, v))
    cands.sort()
    out: List[Tuple[int, int]] = []
    used_q, used_r = set(), set()
    for _nr, _n, _s, _ku, _kv, u, v in cands:
        if len(out) >= N_SEEDS:
            break
        if u in used_q or v in used_r:
            continue
        out.append((u, v))
        used_q.add(u)
        used_r.add(v)
    return out


def _grow(gq: "_Index", gr: "_Index", sim: _Sim, res: WalkResult,
          seeds: List[Tuple[int, int]]) -> WalkResult:
    """One hypothesis: one-to-one growth from `seeds` (D0-D4).

    Admission is proposed by a relation (D2) or by the order of an admitted
    boundary pair (D4); a proposal is admitted when the node gate (D3) passes
    and no shared axis contradicts (D2).  Order still holds on every admitted
    boundary pair (D4's own test, 4.7)."""
    out = WalkResult(frame_q=res.frame_q, frame_r=res.frame_r, dense=res.dense,
                     budget=res.budget, seeds=list(seeds))
    map_q: Dict[int, int] = {}
    map_r: Dict[int, int] = {}
    rejected: set = set()
    #: the best similarity the reference frame offers each query node
    best: Dict[int, float] = {}

    def note(u: int, v: int, s: float) -> None:
        if s > best.get(u, -1.0):
            best[u] = s

    def order_ok(u: int, v: int) -> bool:
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
        pts = []
        for f, af in gq.hosts.get(u, {}).items():
            g = map_q.get(f)
            if g is None:
                continue
            ag = gr.hosts.get(v, {}).get(g)
            if ag is not None:
                pts.append((af, ag))
        return _order_consistent(pts, gq.closed(u) and gr.closed(v))

    support: Dict[Tuple[int, int], Dict[Tuple[int, int], Tuple[float, bool]]] = {}
    via: Dict[Tuple[int, int], List[Tuple]] = {}
    heap: List[Tuple] = []

    def propose(u: int, v: int, sup: Tuple[int, int], slack: float, how: Tuple,
                measured: bool = True) -> None:
        if u in map_q or v in map_r or (u, v) in rejected:
            return
        s = sim(u, v)
        if s is None:
            return
        note(u, v, s)
        sup_d = support.setdefault((u, v), {})
        _record_support(sup_d, sup, slack, measured)          # 7.19
        via.setdefault((u, v), []).append(how)
        heapq.heappush(heap, _serve_key(how, len(sup_d), _measured_slack(sup_d), s,
                                        gq.key[u], gr.key[v], u, v))

    def order_propose(hq: int, hr: int) -> None:
        """[D4] two admitted boundaries with enough admitted features in order
        propose the features attached to both."""
        fa, fb = gq.hosts.get(hq, {}), gr.hosts.get(hr, {})
        if not fa or not fb:
            return
        pts = [(af, fb[map_q[f]]) for f, af in fa.items()
               if f in map_q and map_q[f] in fb]
        if len(pts) < ORDER_PROPOSAL_MIN:
            return
        for f, af in sorted(fa.items()):
            if f in map_q:
                continue
            for g, ag in sorted(fb.items(), key=lambda kv: (gr.key[kv[0]], kv[0])):
                if g in map_r:
                    continue
                if not _order_consistent(pts + [(af, ag)],
                                         gq.closed(hq) and gr.closed(hr)):
                    continue
                propose(f, g, (hq, hr), 0.0, ("order", hq, hr), measured=False)

    def contradicted(u: int, v: int) -> bool:
        """[D2 / 7.16-7.17] the veto is on the PAIR, not on the one relation
        that proposed it: a node whose attachment axis is seven bins away is
        not admitted because its successor gap happens to agree.

        Two ways to contradict, and no others:
          * a relation whose FAR END is already mapped to the other relation's
            far end contradicts (the correspondence there is established);
          * a class the two nodes share in which EVERY pair of relations
            contradicts (no reading of the class agrees at all)."""
        for cls in sorted(gq.half[u]):
            theirs = gr.half[v].get(cls)
            if not theirs:
                continue
            any_agrees = False
            for hu in gq.half[u][cls]:
                for hv in theirs:
                    bad = _contradicts(hu, hv)
                    if map_q.get(hu.far) == hv.far and bad:
                        return True
                    if not bad:
                        any_agrees = True
            if not any_agrees:
                return True
        return False

    def admit(u: int, v: int, how: Any) -> None:
        map_q[u] = v
        map_r[v] = u
        out.mapping = dict(sorted(map_q.items()))
        out.log.append((u, v, how))
        for cls in sorted(gq.half[u]):
            theirs = gr.half[v].get(cls)
            if not theirs:
                continue
            for hu in gq.half[u][cls]:
                if hu.far in map_q:
                    continue
                for hv in theirs:
                    if hv.far in map_r or (hu.far, hv.far) in rejected:
                        continue
                    s = sim(hu.far, hv.far)
                    if s is None:
                        continue
                    note(hu.far, hv.far, s)
                    if s <= NODE_FLOOR:
                        out.gate_rejections += 1
                        continue
                    out.edge_checks += 1
                    agree = _agreement(hu, hv) or 0.0
                    propose(hu.far, hv.far, (u, v), 1.0 - agree, cls)
        # [D4] the boundary the new pair may have completed
        for hq in (u,):
            hr = map_q.get(hq)
            if hr is not None:
                order_propose(hq, hr)
        for fq, hq in gq.hosts.get(u, {}).items():
            hr = map_q.get(hq)
            if hr is not None and fq in map_q:
                order_propose(hq, hr)

    for u, v in seeds:
        if contradicted(u, v):
            out.contradicts += 1
            rejected.add((u, v))
            continue
        if not order_ok(u, v):
            out.order_rejections += 1
            rejected.add((u, v))
            continue
        admit(u, v, "seed")

    while heap:
        _tier, _neg, _slack_sum, _nsim, _ku, _kv, u, v = heapq.heappop(heap)
        if u in map_q or v in map_r or (u, v) in rejected:
            continue
        sup_d = support.get((u, v))
        if not sup_d or _neg != -len(sup_d) or _slack_sum != _measured_slack(sup_d):
            continue
        s = sim(u, v)
        if not _node_gate(s, best.get(u, 0.0)):
            out.gate_rejections += 1
            continue
        if contradicted(u, v):
            out.contradicts += 1
            rejected.add((u, v))
            continue
        if not order_ok(u, v):
            out.order_rejections += 1
            rejected.add((u, v))
            continue
        admit(u, v, tuple(sorted(set(x if not isinstance(x, tuple) else x[0]
                                     for x in via.get((u, v), ())))))
    out.mapping = dict(sorted(map_q.items()))
    return out


def walk(Q: Any, R: Any, q_sigs: Dict[int, Tuple[str, Any]],
         r_sigs: Dict[int, Tuple[str, Any]]) -> WalkResult:
    """[Amendment 7] One walk: seeds (D1), a beam of hypotheses over seed
    prefixes (D5), the objective (D0) choosing among them."""
    gq, gr = _Index(Q, q_sigs), _Index(R, r_sigs)
    res = WalkResult(frame_q=len(gq.frame), frame_r=len(gr.frame),
                     dense=len(q_sigs) * len(r_sigs))
    res.budget = budget(len(gq.frame), len(gr.frame))
    sim = _Sim(gq, gr, res)
    seeds = _seeds(gq, gr, sim, res)
    hypotheses: List[WalkResult] = []
    seen_prefix = set()
    # [Amendment 7.10] the beam's four seed sets: the first seed, two
    # intermediate prefixes and the whole seed list (fewer when fewer seeds).
    # They are grown LARGEST FIRST: the budget is shared by the beam, and the
    # hypothesis the seed list was built for must not be starved by its own
    # alternatives.
    n_seed = len(seeds)
    ks = sorted({1, max(1, n_seed // 3), max(1, (2 * n_seed) // 3), n_seed},
                reverse=True) if n_seed else [0]
    for k in ks[:BEAM_WIDTH]:
        prefix = tuple(seeds[:k])
        if prefix in seen_prefix:
            continue
        seen_prefix.add(prefix)
        h = _grow(gq, gr, sim, res, list(prefix))
        h.terms = objective(h.mapping, gq, gr, lambda u, v: sim.cache.get((u, v), 0.0))
        hypotheses.append(h)
    if not hypotheses:
        hypotheses = [_grow(gq, gr, sim, res, [])]
        hypotheses[0].terms = objective({}, gq, gr, lambda u, v: 0.0)
    # [Amendment 7.11] the choice among the beam's hypotheses
    best = _choose(hypotheses)
    # the reported walk is the best hypothesis, with the shared cost counts
    out = WalkResult(frame_q=best.frame_q, frame_r=best.frame_r, dense=best.dense,
                     budget=res.budget, seeds=best.seeds, mapping=best.mapping,
                     log=best.log, examined=res.examined, edge_checks=res.edge_checks,
                     budget_exhausted=res.budget_exhausted,
                     order_rejections=best.order_rejections,
                     gate_rejections=best.gate_rejections,
                     contradicts=best.contradicts, terms=best.terms)
    _score(out, gq, gr, lambda u, v: sim.cache.get((u, v), 0.0))
    out.beam = [{"seeds": len(h.seeds), "mapped": len(h.mapping), "q": h.terms["q"],
                 "order_rejections": h.order_rejections} for h in hypotheses]
    return out


def objective(mapping: Dict[int, int], gq: "_Index", gr: "_Index",
              similarity: Any) -> Dict[str, Any]:
    """The Amendment-7 objective of ANY one-to-one mapping of frame nodes.

    Q = (W_NODE * S + W_RELATION * (P + A))
        / (W_NODE * |M| + W_RELATION * (R + G)),
    the terms as stated in 7.0, G being the number of (mapped pair, attribute
    lens type) units A is summed over (Amendment 7.8: without it a mapping
    with many attributes could score above 1).  `similarity(u, v)` supplies the graded
    signature similarity of a pair (it may be cached, budgeted or unbounded;
    the OBJECTIVE never gates)."""
    m = dict(mapping)
    inv = {v: u for u, v in m.items()}
    s_node = 0.0
    for u, v in sorted(m.items()):
        if gq.lens[u] != gr.lens[v]:
            return {"q": 0.0, "mapped": len(m), "S": 0.0, "P": 0.0, "A": 0.0, "R": 0,
                    "jaccard": 0.0}
        s_node += float(similarity(u, v))

    def edge_id(g: "_Index", s_: int, d_: int, name: str) -> Tuple:
        und = g.pair[(s_, d_, name)].cls[1] == "u"
        return (min(s_, d_), max(s_, d_), name) if und else (s_, d_, name)

    p_mass = 0.0
    p_candidates = 0
    # Amendment 7.14: the two graphs' node ids are DIFFERENT spaces, so the
    # union of 7.0 cannot be a set union of edge keys.  It is counted: (edges
    # with at least one end in M) + (edges with at least one end in the image
    # of M) - (edges with both ends mapped, the P candidates).
    n_q_any = n_r_any = 0
    for s_, d_, name in gq.edges:
        if s_ in m and d_ in m:
            p_candidates += 1
            n_q_any += 1
            hr = gr.pair.get((m[s_], m[d_], name))
            if hr is None:
                continue
            g = _agreement(gq.pair[(s_, d_, name)], hr)
            if g is not None:
                p_mass += g
        elif s_ in m or d_ in m:
            n_q_any += 1
    for s_, d_, name in gr.edges:
        if s_ in inv and d_ in inv:
            n_r_any += 1
        elif s_ in inv or d_ in inv:
            n_r_any += 1
    r_union = n_q_any + n_r_any - p_candidates

    a_mass = 0.0
    a_units = 0
    for u, v in sorted(m.items()):
        aq, ar = gq.attrs.get(u, {}), gr.attrs.get(v, {})
        for lt in sorted(set(aq) | set(ar)):
            a_units += 1
            mine, theirs = aq.get(lt, []), ar.get(lt, [])
            if not mine or not theirs:
                continue
            finer = mine if len(mine) <= len(theirs) else theirs
            coarse = theirs if finer is mine else mine
            hits = 0
            for x in finer:
                best = 0.0
                for y in coarse:
                    sim = _attr_similarity(gq if finer is mine else gr,
                                           gr if finer is mine else gq, x, y)
                    best = max(best, sim)
                if best >= NODE_FLOOR:
                    hits += 1
            a_mass += hits / len(finer)

    num = W_NODE * s_node + W_RELATION * (p_mass + a_mass)
    # Amendment 7.8: the attribute units are in the denominator too, the same
    # (mapped pair, lens type) units the attribute mass is summed over, so the
    # objective stays in [0, 1].
    den = W_NODE * len(m) + W_RELATION * (r_union + a_units)
    jac = (p_mass / r_union) if r_union else 0.0
    return {"q": (num / den) if den > 0 else 0.0, "mapped": len(m),
            "S": s_node, "P": p_mass, "A": a_mass, "R": r_union,
            "A_units": a_units, "p_candidates": p_candidates, "jaccard": jac}


def _score(res: WalkResult, gq: "_Index", gr: "_Index", similarity: Any) -> None:
    """The walk's own score: the 7.0 objective of the mapping it produced, and
    the support floor.  (Kept beside the walker's terms for the report: the
    preserved mass, the relation counts it was computed from.)"""
    m = res.mapping
    inv = {v: u for u, v in m.items()}
    preserved = n_q = n_r = 0
    void_r = set()
    for s_, d_, name in gq.edges:
        if s_ not in m or d_ not in m:
            continue
        hq = gq.pair[(s_, d_, name)]
        hr = gr.pair.get((m[s_], m[d_], name))
        if hr is None:
            n_q += 1
            continue
        sl = _slack(hq, hr)
        if sl is None and not ({i for i, _v in hq.bins} & {i for i, _v in hr.bins}):
            und = hq.cls[1] == "u"
            void_r.add((min(m[s_], m[d_]), max(m[s_], m[d_]), name) if und
                       else (m[s_], m[d_], name))
            continue
        n_q += 1
        if sl is not None:
            preserved += 1
    for s_, d_, name in gr.edges:
        if s_ not in inv or d_ not in inv:
            continue
        und = gr.pair[(s_, d_, name)].cls[1] == "u"
        key = (min(s_, d_), max(s_, d_), name) if und else (s_, d_, name)
        if key in void_r:
            continue
        n_r += 1
    shared_attr = 0
    for u, v in m.items():
        aq, ar = gq.attrs.get(u, {}), gr.attrs.get(v, {})
        for lt in sorted(set(aq) | set(ar)):
            if lt in aq and lt in ar and any(
                    _attr_similarity(gq, gr, a, b) >= SIGNATURE_GATE
                    for a in aq[lt] for b in ar[lt]):
                shared_attr += 1
    res.preserved, res.relations_q, res.relations_r = preserved, n_q, n_r
    res.attributes_shared = shared_attr
    smaller = min(res.frame_q, res.frame_r)
    res.supported = (len(m) >= MIN_SUPPORT_PAIRS
                     and len(m) >= MIN_SUPPORT_FRACTION * smaller)
    terms = objective(m, gq, gr, similarity)
    res.terms = terms
    res.score = float(terms["q"]) if res.supported else None


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
