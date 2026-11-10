"""memory/assembly_readout.py -- DERIVED READOUTS over the live cluster store.

The single cluster mechanism is `concept_graph.ConceptGraph` + `memory.Builder`.
This module computes the assembly contract's quantities as PURE FUNCTIONS over
that live store; it owns no state and defines no second store. `memory/assembly`
holds the contract shapes and the reference fixtures (never imported by runtime
code); this module reads the real `Concept`s and the real pattern-node / edge
stores and returns the contract quantities on demand.

Every function is deterministic (canonical sorted order, no randomness) and
bounded (member/edge iteration only). Population widths are read from the
node's own `TypedSignature.population`, so a tolerance is the representation's
own dispersion, never a tuned number.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import parameters as P
from core.registries import get_population_axes
from core.schema import CONSTITUTIONAL_EDGE_RANGES
from memory.assembly import (
    ASSEMBLY_HEADER_BITS,
    admits_by_length,
    description_length_bits,
    parameter_precision_cost_bits,
    unclustered_baseline_bits,
)


def _signature(nid: int, cg: Any):
    getter = getattr(cg, "get_pattern_node_full_signature", None)
    if getter is None:
        return None
    return getter(int(nid))


def _axis_width(vec: Sequence[float], circular: bool) -> float:
    """Dispersion (bins) of a normalised activity vector along one axis."""
    n = len(vec)
    if n <= 1:
        return 0.0
    mass = float(sum(vec))
    if mass <= 0.0:
        return 0.0
    if circular:
        two_pi = 2.0 * math.pi
        c = 0.0
        s = 0.0
        for j, v in enumerate(vec):
            ang = two_pi * j / n
            c += v * math.cos(ang)
            s += v * math.sin(ang)
        r = math.hypot(c, s) / mass
        if r <= 0.0:
            return float(n)
        circ_std_rad = math.sqrt(max(0.0, -2.0 * math.log(min(1.0, r))))
        return circ_std_rad * n / two_pi
    mean = 0.0
    for j, v in enumerate(vec):
        mean += j * v
    mean /= mass
    var = 0.0
    for j, v in enumerate(vec):
        var += v * (j - mean) ** 2
    var /= mass
    return math.sqrt(max(0.0, var))


def overlap_links(cg: Any, threshold: float = None,
                  budget: int = None) -> Dict[int, Tuple[Tuple[int, float], ...]]:
    """Competing-assembly links: graded member-set overlap between live concepts.

    Overlap is the Jaccard index of the two member sets. Only pairs at or above
    `threshold` are recorded. Computed in bounded deterministic order; when the
    number of live concepts exceeds `budget`, the cache is left empty (logged by
    the caller) rather than spending unbounded compute.
    """
    if threshold is None:
        threshold = float(P.ASSEMBLY_OVERLAP_MIN)
    if budget is None:
        budget = int(P.N_NODE_MAX)
    items = []
    for cid in sorted(cg.concepts):
        c = cg.concepts[cid]
        if c.members:
            items.append((cid, frozenset(c.members)))
    if len(items) > budget:
        return {}
    out: Dict[int, list] = {}
    for a in range(len(items)):
        cid_a, set_a = items[a]
        for b in range(a + 1, len(items)):
            cid_b, set_b = items[b]
            union = len(set_a | set_b)
            if union == 0:
                continue
            j = len(set_a & set_b) / float(union)
            if j >= threshold:
                out.setdefault(cid_a, []).append((cid_b, j))
                out.setdefault(cid_b, []).append((cid_a, j))
    return {cid: tuple(sorted(v)) for cid, v in out.items()}


def _relation_parameter_widths(relations: Sequence[Tuple]) -> Tuple[Tuple[float, float], ...]:
    """One `(range, width)` per relation, from the edge schema's own bins.

    A relation's graded parameters are described to the front-end resolution:
    width = one bin, range = the largest constitutional range of its registered
    edge schema (`EDGE_<type>`, the same name the similarity stack uses). When
    the schema is unregistered the range falls back to one bin, so the cost is
    never fabricated below the representation's own resolution.
    """
    widths = []
    for rel in relations:
        et = rel[2]
        name = getattr(et, "name", None)
        cls = f"EDGE_{name}" if name else None
        ranges = CONSTITUTIONAL_EDGE_RANGES.get(cls, ()) if cls else ()
        rng = max(ranges) if ranges else 1.0
        # [ROUND 10 / PSYCHOPHYSICS] width = the representation's OWN resolution,
        # measured from the V1 orientation tuning (parameters.
        # RELATION_PARAMETER_WIDTH_BINS), not a hand-chosen 1.0 bin. A finer
        # representation resolves more levels, so the MDL charges the true cost.
        widths.append((float(rng), float(P.RELATION_PARAMETER_WIDTH_BINS)))
    return tuple(widths)


def _relation_coherence(
    cg: Any, rels: Sequence[Tuple[int, int, Any]],
) -> Tuple[float, float, float]:
    """The relations' SHARED CONTEXT and their DEPARTURE from it.

    The coding scheme describes a cluster's relations' graded parameters at
    the front end's resolution, one statement per relation. A cluster whose
    relations share a context does not need that: the context is stated
    once and each relation only states how far it departs from it. The
    saving is what pays for the cluster, and it is what a SHUFFLED structure
    does not have -- so this is the quantity that lets the length test
    reject a null.

    Returns `(context_bits, deviation_bits, full_bits)`:
      context_bits   the cost of stating the shared context (one parameter
                     statement per relation class present);
      deviation_bits the cost of the relations' departures from it (the
                     number of choices within the observed departure, per
                     relation);
      full_bits      the cost of stating every relation's parameters
                     individually -- what the null pays.
    """
    # The relations' graded parameters, grouped by their schema class.
    by_class: Dict[str, List[Tuple[float, ...]]] = {}
    ranges: Dict[str, float] = {}
    widths: Dict[str, float] = {}
    for parent, child, etype in rels:
        params = None
        for edge in cg.get_outgoing_edges(parent):
            if int(edge.child) == int(child) and edge.edge_type == etype:
                params = edge.params
                break
        if params is None:
            continue
        cls = getattr(params, "class_", None) or ""
        q = tuple(float(v) for v in (getattr(params, "quantized", ()) or ()))
        if not q:
            continue
        by_class.setdefault(cls, []).append(q)
        name = getattr(etype, "name", None)
        key = f"EDGE_{name}" if name else ""
        rng = CONSTITUTIONAL_EDGE_RANGES.get(key, ()) if key else ()
        ranges[cls] = max(ranges.get(cls, 0.0), max(rng) if rng else 1.0)
        widths[cls] = float(P.RELATION_PARAMETER_WIDTH_BINS)
    if not by_class:
        return 0.0, 0.0, 0.0
    context_bits = 0.0
    deviation_bits = 0.0
    full_bits = 0.0
    for cls, rows in by_class.items():
        rng = ranges.get(cls, 1.0)
        width = widths.get(cls, 1.0)
        pc = parameter_precision_cost_bits(rng, width)
        full_bits += pc * len(rows)
        context_bits += pc
        # The context is the per-slot median of the relations' parameters.
        nslots = min(len(r) for r in rows)
        med = []
        for s in range(nslots):
            vals = sorted(r[s] for r in rows)
            med.append(vals[len(vals) // 2])
        for r in rows:
            d = 0
            for s in range(min(len(r), len(med))):
                d += abs(r[s] - med[s])
            if d > 0:
                # The number of choices within +-d bins of the context.
                deviation_bits += math.log2(2.0 * d + 1.0)
    return context_bits, deviation_bits, full_bits


def shadow_mdl(cg: Any, candidate: Any) -> Optional[Dict[str, Any]]:
    """Description length of a candidate beside its threshold decision.

    SHADOW ONLY: computes `description_length_bits` and the null it is
    measured against, and reports `admits_by_length`. Nothing here gates
    admission -- the threshold predicate remains the decision of record until
    a parity-gated session switches it.

    THE NULL. The comparison used to be against the "unclustered" baseline,
    which pays a full assembly header PER MEMBER while a cluster pays it
    once -- so every cluster of two or more members was shorter and the
    length rule admitted everything (measured: 48 of 48 clusters admitted,
    an agreement rate bounded by the threshold's own admit rate). The null
    is now the SAME structure WITHOUT the cluster's shared context: the same
    members, the same relations, one header -- but every relation's graded
    parameters stated individually. The member headers cancel, and what is
    left is the question the length argument was always meant to ask: does
    the cluster's shared context pay for itself?

    A structure whose relations share a context states the context once and
    each relation only its departure from it, so it is SHORTER than the null.
    A SHUFFLED structure has no context to share, so its departures cost as
    much as stating every parameter outright, and it is LONGER: the length
    rule rejects it.
    """
    node_ids = getattr(candidate, "node_ids", None)
    if not node_ids:
        return None
    members = set(int(n) for n in node_ids)
    n = len(members)
    if n < 1:
        return None
    rels = []
    seen = set()
    for nid in sorted(members):
        for edge in cg.get_outgoing_edges(nid):
            if edge.child in members:
                key = (int(edge.parent), int(edge.child), edge.edge_type.value)
                if key in seen:
                    continue
                seen.add(key)
                rels.append((int(edge.parent), int(edge.child), edge.edge_type))
    n_pairs = n * (n - 1) // 2
    n_nodes = max(1, len(cg.get_live_pattern_node_ids()))
    widths = _relation_parameter_widths(rels)
    cost = description_length_bits(
        n_members=n, member_ref_ids=tuple(sorted(members)),
        n_relations=len(rels), n_candidate_pairs=n_pairs,
        n_memory_nodes=n_nodes, parameter_widths=widths)
    base = unclustered_baseline_bits(
        n_members=n, n_relations=len(rels), n_candidate_pairs=n_pairs,
        n_memory_nodes=n_nodes, parameter_widths=widths)
    ctx_bits, dev_bits, full_bits = _relation_coherence(cg, rels)
    # The null: the SAME structure -- the same members, the same relations,
    # one header -- but without the cluster's shared context, so every
    # relation's graded parameters are stated individually. The unclustered
    # baseline's per-member headers are dropped (the null has the same
    # number of clusters as the candidate, so it pays the same headers).
    null_bits = base.total_bits - n * ASSEMBLY_HEADER_BITS
    # The candidate: the shared context once, plus each relation's departure.
    candidate_bits = cost.total_bits - full_bits + ctx_bits + dev_bits
    return {
        "members": n,
        "relations": len(rels),
        "cost_bits": candidate_bits,
        "baseline_bits": null_bits,
        "context_bits": ctx_bits,
        "deviation_bits": dev_bits,
        "full_parameter_bits": full_bits,
        "admits_by_length": candidate_bits < null_bits,
    }


def shadow_decision(cg: Any, candidate: Any, threshold: bool) -> Optional[Dict[str, Any]]:
    """Shadow MDL BESIDE one crystallization decision, for every candidate.

    ``shadow_mdl`` reports the length argument alone; this wrapper attaches the
    threshold predicate's own decision and whether the two AGREE, so the
    shadow-vs-threshold agreement rate can be measured without either side
    gating the other. Admission stays the threshold predicate's job here; the
    length argument is recorded only.
    """
    rep = shadow_mdl(cg, candidate)
    if rep is None:
        return None
    out = dict(rep)
    out["threshold"] = bool(threshold)
    out["agree"] = (bool(rep["admits_by_length"]) == bool(threshold))
    return out


def agreement_rate(decisions: Sequence[Optional[Dict[str, Any]]]
                   ) -> Optional[Dict[str, Any]]:
    """The shadow-vs-threshold agreement over a sequence of decisions.

    Reports how often the MDL length rule and the threshold predicate agree
    -- the number the W1/O4 report quotes. It is a REPORT, never a gate: the
    threshold predicate remains the decision of record.
    """
    items = [d for d in decisions if d is not None]
    if not items:
        return None
    agree = sum(1 for d in items if d.get("agree"))
    return {"n": len(items), "agree": agree, "rate": agree / float(len(items))}
