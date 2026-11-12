"""memory/assemblies.py -- the MIDDLE TIER (outcome A).

THE TIER THE MISSION LEFT OPEN. Every group in the fusion is FIRST-ORDER and
LOCAL: the shape of a fragment, where it sits, what its surface looks like,
how the object's energy is laid out. What is missing is the tier BETWEEN a
fragment and the whole object: the REPETITION that says which fragments
travel together. This module builds it, on the designs the mission report
left on the table (section G) and audit #3:

  1. FORMATIONS over fragment populations. A fragment's TYPE is the cell of
     its own graded population (the orientation it points at, the length it
     spans). Two types CO-OCCUR when the same view carries both. The
     co-occurrence is NULL-CALIBRATED: the observed rate minus the corpus's
     own independence null for that pair (label-free, the same shape as the
     fusion's per-group null correction). A formation is a set of types whose
     links are all above that null.

  2. ADMISSION by the EXISTING MDL shadow machinery
     (memory/assembly_readout.py::shadow_mdl) -- the tier adds NO second
     admission rule. The corpus's fragment-type graph is presented to
     shadow_mdl as the cluster store it already knows how to read, and the
     length argument decides.

  3. READOUT. Every admitted formation carries a graded readout: its
     co-occurrence profile, the vector of what it habitually travels with.
     A view's readout is that profile weighted by how much of the formation
     the view actually carries, so the vector says which fragments the view
     COMPLETES to.

  4. COHESION by leave-one-out completion. Hold out one fragment of a
     formation: can the rest still predict it? The measure is the graded
     strength of that prediction, averaged over the formation's members.

  5. SECOND-ORDER TEXTURE: the correlation between NEIGHBOURING orientation
     and scale channels of a part's surface (the orientation axis and the
     spatial-frequency axis).

  6. A MULTI-SCALE FRAGMENT MAP: the fragment map at a finer
     curvature/angular resolution with a scale axis, built from the parts'
     graded populations.

  7. The RETIRED PAIR HISTOGRAM is re-tested INSIDE the tier (on its own it
     measured to dilute the fusion; inside the tier it may earn its weight)
     and its contribution is reported.

WHAT THE TIER CANNOT DO (stated, not hidden). A true MULTI-SCALE map in the
IMAGE's scale is not reachable from this descriptor: the Gabor bank is fixed
at the frame's scale and the view carries ONE scale (its size slot). The
scale axis of the map below is therefore the FRAGMENT's own scale (the
chain's graded length axis): the map is multi-scale in the fragment's
sense, not in the image's.

NOTHING HERE IS A CONSTANT THAT WAS NOT MEASURED. The corpus's co-occurrence
structure and the admitted formations live in parameters/_text.py, measured on
the dev corpus by diagnostics/_tier_corpus.py; the tier's internal weights
live there too (the label-free product rule). Every function here is a pure
function of the view records and of those constants.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import parameters as P
from core.population import axis_similarity


# ---------------------------------------------------------------------------
# The fragment TYPE space.
#
# A fragment's type is the cell of its OWN graded population: the orientation
# it points at (the population's axial mean, in 12 cells of 15 deg -- the
# population's own resolution) and the length it spans (the population's
# graded mean, in 8 cells). The cell id is orientation * LENGTH_BINS + length.
#
# The mean is used rather than the argmax because the mean is the stable
# statistic of a graded population (the argmax flips at a bin edge); the
# resolution is the population's own, so no new resolution is invented.
# ---------------------------------------------------------------------------
TYPE_ORIENT_BINS = 12
TYPE_LENGTH_BINS = 8
TYPE_CELLS = TYPE_ORIENT_BINS * TYPE_LENGTH_BINS


def _axial_mean(vec: Sequence[float]) -> Optional[float]:
    """The mean direction of an axial (pi-periodic) distribution, in radians."""
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


def type_cell(part: Any) -> Optional[int]:
    """The fragment's type cell, or None when it carries no population."""
    pop = getattr(part, "population", ()) or ()
    if len(pop) < 3:
        return None
    theta, _curv, length = pop[0], pop[1], pop[2]
    m = _axial_mean(theta)
    if m is None:
        return None
    w = np.asarray(length, dtype=np.float64)
    if w.size < 2 or float(w.sum()) <= 1e-12:
        return None
    idx = np.arange(w.size, dtype=np.float64)
    mean_len = float((w * idx).sum() / float(w.sum()))
    o = int(round(m / (math.pi / TYPE_ORIENT_BINS))) % TYPE_ORIENT_BINS
    l = int(round(mean_len))
    l = max(0, min(TYPE_LENGTH_BINS - 1, l))
    return o * TYPE_LENGTH_BINS + l


# ---------------------------------------------------------------------------
# The corpus.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierCorpus:
    """The corpus's fragment-type structure (label-free).

    links      cell -> {cell: the null-calibrated co-occurrence}. Only the
               links that are ABOVE the corpus's independence null are kept.
    formations the admitted formations, each a tuple of type cells.
    null      the corpus's own null (the mean co-occurrence over the type
               pairs) -- the tier's corpus null, like the fusion's.
    """

    links: Tuple[Tuple[Tuple[int, float], ...], ...] = ()
    formations: Tuple[Tuple[int, ...], ...] = ()
    null: float = 0.0

    # -- derived views ------------------------------------------------------
    def link_table(self) -> Dict[int, Dict[int, float]]:
        out: Dict[int, Dict[int, float]] = {}
        for i, row in enumerate(self.links):
            out[i] = {int(c): float(v) for c, v in row}
        return out

    def link(self, a: int, b: int) -> float:
        if not self.links:
            return 0.0
        if a < 0 or b < 0 or a >= len(self.links) or b >= len(self.links):
            return 0.0
        row = self.links[a]
        for c, v in row:
            if int(c) == int(b):
                return float(v)
        return 0.0

    def formation_links(self, members: Sequence[int]) -> List[Tuple[int, int, float]]:
        """The links inside one formation."""
        out = []
        ms = list(members)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                v = self.link(ms[i], ms[j])
                if v > 0.0:
                    out.append((ms[i], ms[j], v))
        return out


# ---------------------------------------------------------------------------
# The MDL adapter: the corpus's fragment-type graph, presented to the
# EXISTING shadow_mdl. It adds no admission rule of its own -- it only gives
# shadow_mdl the three things it reads.
# ---------------------------------------------------------------------------

class _CorpusParams:
    """The graded parameters of one relation (the co-occurrence profile)."""

    __slots__ = ("class_", "quantized")

    def __init__(self, quantized: Sequence[int]) -> None:
        self.class_ = "CO_LOCATED"
        self.quantized = tuple(int(x) for x in quantized)


class _CorpusEdgeType:
    """The relation's type. The name is the REGISTERED edge type the
    relation is read as: the two fragments of a formation are CO-LOCATED
    (they travel together in the same view)."""

    __slots__ = ("name", "value")

    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value


class _CorpusEdge:
    __slots__ = ("parent", "child", "edge_type", "params")

    def __init__(self, parent: int, child: int, edge_type: Any,
                 params: Any) -> None:
        self.parent = int(parent)
        self.child = int(child)
        self.edge_type = edge_type
        self.params = params


class CorpusGraph:
    """The corpus's fragment-type graph, as the cluster store shadow_mdl
    reads: outgoing edges with their graded parameters, and the live node
    ids. Nothing else is invented -- the admission itself is shadow_mdl's."""

    def __init__(self, cells: Sequence[int],
                 links: Dict[Tuple[int, int], float],
                 support: Dict[Tuple[int, int], int],
                 max_support: int) -> None:
        self._nodes = tuple(sorted(int(c) for c in cells))
        self._index = {c: i for i, c in enumerate(self._nodes)}
        self._edges: Dict[int, List[_CorpusEdge]] = {c: [] for c in self._nodes}
        et = _CorpusEdgeType("CO_LOCATED", "CO_LOCATED")
        for (a, b), v in sorted(links.items()):
            if a not in self._index or b not in self._index:
                continue
            sup = int(support.get((a, b), 0))
            # The relation's graded parameters: the null-calibrated
            # co-occurrence and the support, each in the registered 0..9
            # range of the edge type they are read as.
            q = (int(round(9.0 * float(v))), int(round(9.0 * sup / max(1, max_support))))
            self._edges[a].append(_CorpusEdge(a, b, et, _CorpusParams(q)))
            self._edges[b].append(_CorpusEdge(b, a, et, _CorpusParams(q)))

    def get_outgoing_edges(self, nid: int) -> Tuple[_CorpusEdge, ...]:
        return tuple(self._edges.get(int(nid), ()))

    def get_live_pattern_node_ids(self) -> Tuple[int, ...]:
        return self._nodes


class FormationCandidate:
    """A candidate formation: the set of fragment types it would cluster."""

    def __init__(self, node_ids: Sequence[int]) -> None:
        self.node_ids = tuple(int(n) for n in node_ids)


# ---------------------------------------------------------------------------
# The corpus builder (label-free). It is what diagnostics/_tier_corpus.py
# runs on the dev corpus; the SHIPPED constants are its output.
# ---------------------------------------------------------------------------

def build_corpus(views: Sequence[Any],
                 min_support: int = 1,
                 ) -> Tuple[TierCorpus, Dict[str, Any]]:
    """Build the fragment-type corpus from a list of views.

    The steps (each one label-free):

      1. the type census (the fraction of views carrying each type);
      2. the co-occurrence rate of every type pair;
      3. the corpus's INDEPENDENCE null (the product of the marginals) and
         the null-calibrated co-occurrence (observed - null) / (1 - null);
      4. the candidate formations: the greedy maximal-clique growth of the
         above-null link graph, seeded from every type;
      5. the ADMISSION of every candidate by the EXISTING shadow_mdl;
      6. the readout of every admitted formation.

    Returns the corpus and a report (the numbers the report quotes).
    """
    from memory.assembly_readout import shadow_mdl

    n_views = len(views)
    present_sets: List[set] = []
    for v in views:
        cells = set()
        for p in getattr(v, "parts", ()) or ():
            c = type_cell(p)
            if c is not None:
                cells.add(c)
        present_sets.append(cells)
    cells = sorted({c for s in present_sets for c in s})
    index = {c: i for i, c in enumerate(cells)}
    T = len(cells)
    if T == 0 or n_views == 0:
        return TierCorpus(), {"n_views": n_views, "n_types": 0}

    pres = np.zeros((n_views, T), dtype=bool)
    for vi, s in enumerate(present_sets):
        for c in s:
            pres[vi, index[c]] = True
    p = pres.mean(axis=0)

    co = np.zeros((T, T))
    sup = np.zeros((T, T), dtype=int)
    for i in range(T):
        for j in range(i + 1, T):
            both = int((pres[:, i] & pres[:, j]).sum())
            sup[i, j] = sup[j, i] = both
            co[i, j] = co[j, i] = both / float(n_views)
    null_mat = np.outer(p, p)
    S = np.clip((co - null_mat) / np.maximum(1.0 - null_mat, 1e-9), 0.0, 1.0)
    np.fill_diagonal(S, 0.0)
    corpus_null = float(co[np.triu_indices(T, 1)].mean()) if T > 1 else 0.0
    null_mean_indep = (float(null_mat[np.triu_indices(T, 1)].mean())
                       if T > 1 else 0.0)

    # -- the link set: above the null AND observed at least min_support times
    links: Dict[Tuple[int, int], float] = {}
    supports: Dict[Tuple[int, int], int] = {}
    for i in range(T):
        for j in range(i + 1, T):
            if S[i, j] > 0.0 and int(sup[i, j]) >= int(min_support):
                links[(i, j)] = float(S[i, j])
                links[(j, i)] = float(S[i, j])
                supports[(i, j)] = int(sup[i, j])
                supports[(j, i)] = int(sup[i, j])

    max_support = max(supports.values()) if supports else 1

    # -- the candidate formations: the greedy maximal-clique growth
    candidates: List[Tuple[int, ...]] = []
    seen = set()
    for i in range(T):
        members = [i]
        while True:
            best_j, best = -1, -1.0
            for j in range(T):
                if j in members:
                    continue
                m = min(S[j, m2] for m2 in members)
                if m > best:
                    best, best_j = m, j
            if best_j < 0 or best <= 0.0:
                break
            members.append(best_j)
        key = tuple(sorted(members))
        if key not in seen:
            seen.add(key)
            candidates.append(key)

    # -- the ADMISSION: the existing shadow_mdl, over the corpus graph
    graph = CorpusGraph(cells, links, supports, max_support)
    admitted: List[Tuple[int, ...]] = []
    mdl_reports: List[Dict[str, Any]] = []
    for cand in candidates:
        rep = shadow_mdl(graph, FormationCandidate([cells[i] for i in cand]))
        if rep is None:
            continue
        rec = {"members": [int(cells[i]) for i in cand],
               "size": len(cand),
               "admits": bool(rep.get("admits_by_length")),
               "cost_bits": float(rep.get("cost_bits", 0.0)),
               "null_bits": float(rep.get("baseline_bits", 0.0)),
               "relations": int(rep.get("relations", 0)),
               "deviation_bits": float(rep.get("deviation_bits", 0.0)),
               "context_bits": float(rep.get("context_bits", 0.0))}
        mdl_reports.append(rec)
        if rec["admits"]:
            admitted.append(tuple(sorted(cells[i] for i in cand)))

    # -- the corpus's link table, in the shipped form
    link_rows: List[Tuple[Tuple[int, float], ...]] = []
    for i, c in enumerate(cells):
        row = []
        for j, d in enumerate(cells):
            if i == j:
                continue
            v = links.get((c, d))
            if v is not None and v > 0.0:
                row.append((int(d), float(v)))
        link_rows.append(tuple(row))

    corpus = TierCorpus(links=tuple(link_rows), formations=tuple(admitted),
                         null=corpus_null)
    report = {
        "n_views": n_views,
        "n_types": T,
        "types_per_view_mean": float(np.mean([len(s) for s in present_sets])),
        "corpus_null": corpus_null,
        "null_mean_independence": null_mean_indep,
        "n_links": len(links) // 2,
        "min_support": int(min_support),
        "n_candidates": len(candidates),
        "n_admitted": len(admitted),
        "candidate_sizes": sorted({len(c) for c in candidates}),
        "admitted_sizes": sorted({len(c) for c in admitted}),
        "mdl": mdl_reports,
    }
    return corpus, report


def corpus_from_parameters() -> TierCorpus:
    """The shipped corpus: the constants measured on the dev corpus."""
    raw = P.ASPECT_TIER_CORPUS or {}
    try:
        d = dict(raw)
    except Exception:
        return TierCorpus()
    links = tuple(tuple((int(c), float(v)) for c, v in row)
                  for row in d.get("links", ()))
    formations = tuple(tuple(int(c) for c in f)
                       for f in d.get("formations", ()))
    return TierCorpus(links=links, formations=formations,
                      null=float(d.get("null", 0.0)))


# ---------------------------------------------------------------------------
# The per-view measures.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierMeasures:
    """The tier's measurements of one view."""

    activation: Tuple[float, ...] = ()    # per admitted formation
    assembly: Tuple[float, ...] = ()      # the completed readout over the types
    cohesion: float = 0.0
    texture2: Tuple[float, ...] = ()
    msmap: Tuple[float, ...] = ()


def _cells_of(view: Any) -> set:
    out = set()
    for p in getattr(view, "parts", ()) or ():
        c = type_cell(p)
        if c is not None:
            out.add(c)
    return out


def _formation_measures(view: Any, corpus: TierCorpus
                        ) -> Tuple[Tuple[float, ...], Tuple[float, ...], float]:
    """The activation vector, the completed readout and the cohesion.

    The activation of a formation = the fraction of its members the view
    carries (graded, no threshold).

    The completed readout: every admitted formation contributes its own
    co-occurrence profile (its readout) weighted by its activation, so the
    vector says which fragments the view COMPLETES to.

    The cohesion: for every formation the view carries, hold out one of its
    members and ask whether the REST of the formation still predicts it;
    the measure is the mean of that prediction's strength.
    """
    cells = _cells_of(view)
    forms = corpus.formations
    if not forms:
        return (), (), 0.0
    activation: List[float] = []
    assembly: Dict[int, float] = {}
    coh_num = 0.0
    coh_den = 0.0
    for members in forms:
        present = [m for m in members if m in cells]
        act = len(present) / float(len(members)) if members else 0.0
        activation.append(act)
        if act <= 0.0:
            continue
        # the formation's readout: its co-occurrence profile over the types
        for m in present:
            for other, v in _links_from(corpus, m):
                val = act * float(v)
                if val > assembly.get(other, 0.0):
                    assembly[other] = val
        # the leave-one-out completion INSIDE the formation
        if len(present) >= 2:
            preds = []
            for held in present:
                best = 0.0
                for g in present:
                    if g == held:
                        continue
                    best = max(best, corpus.link(g, held))
                preds.append(best)
            if preds:
                c = float(sum(preds) / len(preds))
                coh_num += act * c
                coh_den += act
    vec = tuple(assembly.get(c, 0.0) for c in range(TYPE_CELLS))
    cohesion = (coh_num / coh_den) if coh_den > 0.0 else 0.0
    return tuple(activation), vec, cohesion


def _links_from(corpus: TierCorpus, cell: int) -> Sequence[Tuple[int, float]]:
    """The links that leave one type cell."""
    if not corpus.links or cell < 0 or cell >= len(corpus.links):
        return ()
    return corpus.links[cell]


def _lag_autocorr(vec: Sequence[float], lag: int) -> Optional[float]:
    """The circular autocorrelation of a distribution at a lag."""
    v = np.asarray(vec, dtype=np.float64)
    n = v.size
    if n < 4:
        return None
    num = 0.0
    den = 0.0
    for k in range(n):
        num += float(v[k]) * float(v[(k + lag) % n])
        den += float(v[k]) * float(v[k])
    if den <= 1e-12:
        return None
    return num / den


def _second_order(parts: Sequence[Any]) -> Tuple[float, ...]:
    """The second-order texture statistics of a view.

    For every part: the circular autocorrelation of its ORIENTATION
    distribution at lags 1 and 2, and of its SPATIAL-FREQUENCY
    distribution at lags 1 and 2. The view's statistics are their means.
    """
    acc = [0.0, 0.0, 0.0, 0.0]
    n = 0
    for p in parts:
        s = getattr(p, "surface", None)
        if s is None:
            continue
        theta = s.theta_i or s.theta_f or ()
        sf = s.sf_i or s.sf_f or ()
        if not theta or not sf:
            continue
        t1 = _lag_autocorr(theta, 1)
        t2 = _lag_autocorr(theta, 2)
        s1 = _lag_autocorr(sf, 1)
        s2 = _lag_autocorr(sf, 2)
        if t1 is None or t2 is None or s1 is None or s2 is None:
            continue
        acc[0] += t1
        acc[1] += t2
        acc[2] += s1
        acc[3] += s2
        n += 1
    if n == 0:
        return ()
    return tuple(a / n for a in acc)


#: The multi-scale fragment map's axes: a finer angular and curvature
#: resolution than the peak map's, plus the scale axis.
MS_ANGLE_BINS = 24
MS_CURVATURE_BINS = 16
MS_SCALE_BINS = 8


def _multiscale_map(parts: Sequence[Any]) -> Tuple[float, ...]:
    """The multi-scale fragment map of a view.

    The axes: the part's ANGULAR position around the object's centre (24
    bins, finer than the peak map's 12), the chain's graded CURVATURE (8
    bins, spread onto 16 by linear interpolation), and the chain's graded
    LENGTH -- the SCALE axis (8 bins).

    THE LIMIT: the scale axis is the FRAGMENT's scale, not the IMAGE's: the
    descriptor carries one image scale per view (the size slot), so a truly
    multi-scale (image-scale) map is not reachable from it.
    """
    acc = np.zeros((MS_ANGLE_BINS, MS_CURVATURE_BINS, MS_SCALE_BINS),
                   dtype=np.float64)
    for p in parts:
        pop = getattr(p, "population", ()) or ()
        if len(pop) < 3:
            continue
        _theta, curv, length = pop[0], pop[1], pop[2]
        c = np.asarray(curv, dtype=np.float64)
        l = np.asarray(length, dtype=np.float64)
        if c.size < 2 or l.size < 2:
            continue
        pol = getattr(p, "polar", None)
        if pol is None:
            pos = getattr(p, "pos", (0.0, 0.0))
            r = math.hypot(float(pos[0]), float(pos[1]))
            th = math.atan2(float(pos[0]), -float(pos[1]))
        else:
            r = float(pol[0])
            th = float(pol[1])
        # the curvature axis, spread onto the finer bins
        fine = np.zeros(MS_CURVATURE_BINS, dtype=np.float64)
        scale = np.zeros(MS_SCALE_BINS, dtype=np.float64)
        for k, w in enumerate(c):
            if w <= 0.0:
                continue
            f0 = 2 * k
            fine[f0] += 0.5 * float(w)
            fine[min(f0 + 1, MS_CURVATURE_BINS - 1)] += 0.5 * float(w)
        for k, w in enumerate(l):
            if w <= 0.0:
                continue
            scale[min(k, MS_SCALE_BINS - 1)] += float(w)
        ctot = float(fine.sum())
        ltot = float(scale.sum())
        if ctot <= 1e-12 or ltot <= 1e-12:
            continue
        fine /= ctot
        scale /= ltot
        centre = (th + math.pi) / (2.0 * math.pi) * MS_ANGLE_BINS
        ab = np.arange(MS_ANGLE_BINS, dtype=np.float64)
        d = np.abs(ab - centre)
        d = np.minimum(d, MS_ANGLE_BINS - d)
        gauss = np.exp(-0.5 * d * d)
        acc += (gauss[:, None, None] * fine[None, :, None] * scale[None, None, :])
    total = float(acc.sum())
    if total <= 0.0:
        return ()
    return tuple((acc / total).reshape(-1))


def measures(view: Any, corpus: Optional[TierCorpus] = None
             ) -> TierMeasures:
    """Every tier measure of one view."""
    if corpus is None:
        corpus = corpus_from_parameters()
    parts = list(getattr(view, "parts", ()) or ())
    activation, assembly, cohesion = _formation_measures(view, corpus)
    tex2 = _second_order(parts)
    msmap = _multiscale_map(parts)
    return TierMeasures(activation=activation, assembly=assembly,
                        cohesion=cohesion, texture2=tex2, msmap=msmap)


def _cached(view: Any, corpus: TierCorpus) -> TierMeasures:
    """The view's measures, cached on the view (a sweep pairs the same views
    thousands of times and the measures are a pure function of the view)."""
    cached = getattr(view, "tier", None)
    if cached is not None:
        return cached
    m = measures(view, corpus)
    try:
        view.tier = m
    except Exception:
        pass
    return m


# ---------------------------------------------------------------------------
# The sub-terms and their similarities.
# ---------------------------------------------------------------------------

SUB_FORMATION = "formation"
SUB_READOUT = "readout"
SUB_COHESION = "cohesion"
SUB_TEXTURE2 = "texture2"
SUB_MSMAP = "msmap"
SUB_PAIRS = "pairs"

SUB_TERMS: Tuple[str, ...] = (SUB_FORMATION, SUB_READOUT, SUB_COHESION,
                               SUB_TEXTURE2, SUB_MSMAP, SUB_PAIRS)


def _tol(sub: str) -> float:
    table = P.ASPECT_TIER_TOLERANCE or {}
    try:
        return float(dict(table).get(sub, 0.0))
    except Exception:
        return 0.0


def sub_difference(a: TierMeasures, b: TierMeasures, sub: str) -> Optional[float]:
    """The RAW difference of one difference-like sub-term between two views.

    The tolerance of such a sub-term is MEASURED from these differences (the
    p90 over the corpus's random pairs), so the kernel's width is never
    hand-set. None for the overlap-like sub-terms, which need no kernel.
    """
    if sub == SUB_COHESION:
        return abs(float(a.cohesion) - float(b.cohesion))
    if sub == SUB_TEXTURE2:
        if not a.texture2 or not b.texture2 or len(a.texture2) != len(b.texture2):
            return None
        return math.sqrt(sum((float(x) - float(y)) ** 2
                             for x, y in zip(a.texture2, b.texture2)))
    return None


def sub_similarity(a: TierMeasures, b: TierMeasures, sub: str,
                   pair_sim: Optional[float] = None) -> Optional[float]:
    """One sub-term's similarity between two views, in [0, 1].

    The overlap-like sub-terms (the formation activation, the completed
    readout, the multi-scale map) are CHANCE-CORRECTED OVERLAPS, so they need
    no kernel. The difference-like ones (the cohesion, the second-order
    texture) use a Gaussian on the difference with the tolerance MEASURED on
    the corpus (the p90 of the difference over the corpus's random pairs).
    """
    if sub == SUB_FORMATION:
        if not a.activation or not b.activation:
            return None
        if len(a.activation) != len(b.activation):
            return None
        return float(axis_similarity(a.activation, b.activation))
    if sub == SUB_READOUT:
        if not a.assembly or not b.assembly:
            return None
        return float(axis_similarity(a.assembly, b.assembly))
    if sub == SUB_COHESION:
        tol = _tol(SUB_COHESION)
        if tol <= 0.0:
            return None
        d = abs(float(a.cohesion) - float(b.cohesion))
        return float(max(0.0, min(1.0, math.exp(-0.5 * (d / tol) ** 2))))
    if sub == SUB_TEXTURE2:
        if not a.texture2 or not b.texture2:
            return None
        if len(a.texture2) != len(b.texture2):
            return None
        tol = _tol(SUB_TEXTURE2)
        if tol <= 0.0:
            return None
        d = math.sqrt(sum((float(x) - float(y)) ** 2
                           for x, y in zip(a.texture2, b.texture2)))
        return float(max(0.0, min(1.0, math.exp(-0.5 * (d / tol) ** 2))))
    if sub == SUB_MSMAP:
        if not a.msmap or not b.msmap:
            return None
        if len(a.msmap) != len(b.msmap):
            return None
        return float(axis_similarity(a.msmap, b.msmap))
    if sub == SUB_PAIRS:
        return pair_sim
    return None


def tier_similarity(A: Any, B: Any, corpus: Optional[TierCorpus] = None,
                    include_pairs: bool = True) -> Optional[float]:
    """The tier's similarity between two views, in [0, 1].

    The weighted mean of the sub-terms' similarities. The weights are the
    LABEL-FREE product rule (reliability x distinctiveness) measured on the
    dev corpus; the sub-term that has no evidence on either side is skipped
    (an absent measurement is not a zero).
    """
    if corpus is None:
        corpus = corpus_from_parameters()
    if not corpus.formations:
        return None
    ma = _cached(A, corpus)
    mb = _cached(B, corpus)
    weights = P.ASPECT_TIER_WEIGHTS or {}
    try:
        wtab = dict(weights)
    except Exception:
        return None
    pair_sim = None
    if include_pairs:
        from memory.aspects import _pair_similarity
        pair_sim = _pair_similarity(A, B)
    num = 0.0
    den = 0.0
    for sub in SUB_TERMS:
        if sub == SUB_PAIRS and not include_pairs:
            continue
        s = sub_similarity(ma, mb, sub, pair_sim)
        if s is None:
            continue
        w = float(wtab.get(sub, 0.0))
        if w <= 0.0:
            continue
        num += w * s
        den += w
    if den <= 1e-12:
        return None
    return float(max(0.0, min(1.0, num / den)))
