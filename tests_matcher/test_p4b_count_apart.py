"""PIN -- [P4b] the count kept apart from the physics.

The spec's second line: "Topological count separated from physical relation;
the count may not masquerade as physics in the score."

Where it masqueraded.  ``EDGE_UNIT_TABLE`` declares 12 edge slots categorical
lookup keys.  ``EDGE_SCHEMAS`` -- the table the similarity reads -- agrees on
eight (identity slots) and declares four linear or ungated: the membership
flag of CONTOUR_ORIENTATION / JUNCTION_ORIENTATION / REGION_CONTENT (a
literal ``containment = 1.0``; ``EDGE_INDEX_ONLY``) and OCCLUSION.source.  A
flag pair therefore read 1.0 as a SIMILARITY in ``evidence_score``'s
relational mean, and counted as a MEASURED AXIS in the edge mass of
``mcs_score``.

Pinned here:
  * the four slots, by name -- if the tables are reconciled this reddens and
    the switches below become no-ops
  * ``edge_graded_axes``: 0 for a flag pair, a parameter-less pair, a
    categorical agreement; the graded slots otherwise; cross-schema pairs
    and mixtures; symmetric
  * P.EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS: a flag stops earning f(1) = 2
    of edge mass; nothing else moves
  * P.EVIDENCE_RELATION_COUNT_APART: the relational term is
    presence x physics, by hand; it is unchanged when there is no count-only
    pair, when the physics is perfect, when nothing physical was matched;
    both sides under the query-side relation term; RULE B16's branch
    untouched; off = the historical mean, bit for bit
  * it never rises WHERE EVERY COUNT-ONLY PAIR READS 1.0 -- flags,
    parameter-less pairs, categorical agreement; and it does rise with the
    Phase-4 floors restored, where an all-UNKNOWN pair reads 0.2 as a
    similarity and is a full count once the count is kept apart

Every pin below states the switches its numbers depend on.
"""
from __future__ import annotations

import random

import pytest

from core.schema import (EDGE_IDENTITY_SLOTS, EDGE_INDEX_ONLY,
                         EDGE_UNIT_TABLE, UNKNOWN, EdgeParams,
                         EdgeParamsMixture, edge_graded_axes,
                         edge_measured_axes, polymorphic_edge_similarity)
from core.types import EdgeType
from memory.cores import evidence_score, mcs_score

from _graphs import graph, tex

U = UNKNOWN
RC = EdgeType.REGION_CONTENT
CO = EdgeType.CONTOUR_ORIENTATION
JO = EdgeType.JUNCTION_ORIENTATION
JC = EdgeType.JUNCTION_CONTOUR
TOPO = EdgeType.TOPOLOGY
ALL = ("node", "relation", "absence")


def flag(cls):
    return EdgeParams(class_="EDGE_" + cls, quantized=(1.0,))


def jc(dist, incidence, topology):
    return EdgeParams(class_="EDGE_JUNCTION_CONTOUR",
                      quantized=(dist, incidence, topology))


def esim(a, b, et=JC):
    return polymorphic_edge_similarity(a, b, use_confidence=False,
                                       edge_types=(et, et))


# ------------------------------------------------------------------ tables

def test_the_two_slot_tables_disagree_on_exactly_these_four():
    not_gated = sorted(
        (cls, s.name)
        for cls, specs in EDGE_UNIT_TABLE.items()
        for i, s in enumerate(specs)
        if s.quantiser == "lookup_key"
        and i not in EDGE_IDENTITY_SLOTS.get(cls, ()))
    assert not_gated == [
        ("EDGE_CONTOUR_ORIENTATION", "containment"),
        ("EDGE_JUNCTION_ORIENTATION", "containment"),
        ("EDGE_OCCLUSION", "source"),
        ("EDGE_REGION_CONTENT", "containment"),
    ]
    assert set(EDGE_INDEX_ONLY) == {
        "EDGE_CONTOUR_ORIENTATION", "EDGE_JUNCTION_ORIENTATION",
        "EDGE_REGION_CONTENT"}


# ---------------------------------------------------------- graded axes

@pytest.mark.parametrize("cls", ["REGION_CONTENT", "CONTOUR_ORIENTATION",
                                 "JUNCTION_ORIENTATION"])
def test_a_membership_flag_is_a_perfect_match_with_nothing_measured(cls):
    a, b = flag(cls), flag(cls)
    assert esim(a, b, getattr(EdgeType, cls)) == 1.0
    assert edge_graded_axes(a, b) == 0


def test_graded_axes_of_plain_pairs():
    assert edge_graded_axes(None, None) == 0
    assert edge_graded_axes(None, jc(1.0, 2.0, 3.0)) == 0
    assert edge_graded_axes(jc(1.0, 2.0, 3.0), None) == 0
    # dist and incidence are graded, topology_class is an identity slot
    assert edge_graded_axes(jc(1.0, 2.0, 3.0), jc(2.0, 5.0, 3.0)) == 2
    assert edge_graded_axes(jc(1.0, U, 3.0), jc(2.0, 5.0, 3.0)) == 1
    assert edge_graded_axes(jc(1.0, U, 3.0), jc(U, 5.0, 3.0)) == 0
    assert edge_graded_axes(jc(U, U, 3.0), jc(U, U, 3.0)) == 0   # categorical agreement
    assert edge_graded_axes(jc(1.0, 2.0, 3.0), flag("REGION_CONTENT")) == 0


def test_graded_axes_agree_with_measured_axes_where_there_is_no_lookup_key():
    rng = random.Random(2)
    for _ in range(200):
        a = jc(*(rng.choice((U, 0.0, 1.0, 2.0)) for _ in range(3)))
        b = jc(*(rng.choice((U, 0.0, 1.0, 2.0)) for _ in range(3)))
        assert edge_graded_axes(a, b) == edge_measured_axes(a, b)
        assert edge_graded_axes(a, b) == edge_graded_axes(b, a)


def test_graded_axes_of_a_cross_schema_pair():
    oc = EdgeParams(class_="EDGE_ORIENTATION_CONTINUITY",
                    quantized=(1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0))
    cc = EdgeParams(class_="EDGE_CONTOUR_CONTINUITY",
                    quantized=(1.0, 2.0, 0.0))
    s = polymorphic_edge_similarity(
        oc, cc, use_confidence=False,
        edge_types=(EdgeType.ORIENTATION_CONTINUITY,
                    EdgeType.CONTOUR_CONTINUITY))
    n = edge_graded_axes(oc, cc)
    assert n == edge_graded_axes(cc, oc)
    assert (s > 0.0) == (n > 0)
    assert n >= 1
    blank = EdgeParams(class_="EDGE_CONTOUR_CONTINUITY", quantized=(U, U, U))
    assert edge_graded_axes(oc, blank) == 0
    # an incompatible family has nothing in common
    assert edge_graded_axes(oc, jc(1.0, 2.0, 3.0)) == 0


def test_graded_axes_of_a_mixture_is_the_best_prototype():
    mix = EdgeParamsMixture([jc(U, U, 3.0), jc(1.0, 2.0, 3.0)], [1.0, 1.0], 0)
    assert edge_graded_axes(mix, jc(1.0, 5.0, 3.0)) == 2
    assert edge_graded_axes(jc(1.0, 5.0, 3.0), mix) == 2
    flags = EdgeParamsMixture([flag("REGION_CONTENT")], [1.0], 0)
    assert edge_graded_axes(flags, flag("REGION_CONTENT")) == 0


# ------------------------------------------------- the measured-axis count

def test_a_flag_stops_counting_as_a_measured_axis(params):
    a, b = flag("REGION_CONTENT"), flag("REGION_CONTENT")
    params(EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS=False)
    assert edge_measured_axes(a, b) == 1
    assert edge_measured_axes(jc(1.0, 2.0, 3.0), jc(2.0, 5.0, 3.0)) == 2
    params(EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS=True)
    assert edge_measured_axes(a, b) == 0
    assert edge_measured_axes(jc(1.0, 2.0, 3.0), jc(2.0, 5.0, 3.0)) == 2


def test_the_edge_mass_of_a_flag_loses_its_facilitation(params):
    nodes = {1: ("Tex", tex(1, 1)), 2: ("Tex", tex(8, 8))}
    Q, qs = graph(nodes, [(1, 2, RC)], {(1, 2, RC): flag("REGION_CONTENT")})
    C, cs = graph(nodes, [(1, 2, RC)], {(1, 2, RC): flag("REGION_CONTENT")})
    params(EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS=False)
    tr = {}
    mcs_score(Q, C, qs, cs, trace=tr)
    assert tr["c_matched_em"] == 2.0            # 1 edge x sim 1.0 x f(1)
    params(EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS=True)
    tr = {}
    mcs_score(Q, C, qs, cs, trace=tr)
    assert tr["c_matched_em"] == 1.0            # f(0): presence, no physics


def test_the_measured_axis_switch_does_not_reach_the_evidence_score(params):
    Q, qs, C, cs = scenario()
    out = []
    for v in (False, True):
        params(EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS=v)
        out.append(evidence_score(Q, C, qs, cs, trace={})[0])
    assert out[0] == out[1]


# -------------------------------------------------- the relational term

#: five nodes; every node has its twin on the other side, so the
#: correspondence is the identity
NODES = {i: ("Tex", tex(2 * i, 9 - i)) for i in range(1, 6)}
J1, J2, J3 = jc(1.0, 2.0, 3.0), jc(4.0, 2.0, 3.0), jc(1.0, 6.0, 3.0)
Q1, Q2 = jc(2.0, 2.0, 3.0), jc(4.0, 5.0, 3.0)


def scenario(flags=True, query_extra=False):
    """Core: two membership flags, three measured relations.  Query: both
    flags, two of the measured relations (each one bin off), nothing where
    the third is."""
    c_edges = {(1, 2, JC): J1, (2, 3, JC): J2, (3, 4, JC): J3}
    q_edges = {(1, 2, JC): Q1, (2, 3, JC): Q2}
    if flags:
        for e in ((1, 3, RC), (2, 4, RC)):
            c_edges[e] = flag("REGION_CONTENT")
            q_edges[e] = flag("REGION_CONTENT")
    if query_extra:
        q_edges[(4, 5, JC)] = jc(0.0, 0.0, 3.0)
        q_edges[(1, 5, RC)] = flag("REGION_CONTENT")
    Q, qs = graph(NODES, list(q_edges), q_edges)
    C, cs = graph(NODES, list(c_edges), c_edges)
    return Q, qs, C, cs


S1, S2 = esim(Q1, J1), esim(Q2, J2)


def test_the_scenario_is_what_the_arithmetic_below_assumes():
    assert 0.0 < S1 < 1.0 and 0.0 < S2 < 1.0 and S1 != S2
    Q, qs, C, cs = scenario()
    _s, corr = evidence_score(Q, C, qs, cs, trace={})
    assert corr == {i: i for i in range(1, 6)}


def test_presence_times_physics_by_hand(params):
    Q, qs, C, cs = scenario()
    params(EVIDENCE_QUERY_SIDE_TERMS=(), EVIDENCE_RELATION_COUNT_APART=False)
    off = {}
    evidence_score(Q, C, qs, cs, trace=off)
    # five verifiable edges: 2 flags at 1.0, two measured, one with no partner
    assert off["relational_consistency"] == pytest.approx((2 + S1 + S2) / 5)
    assert off["relation_presence"] == pytest.approx(4 / 5)
    assert off["relation_physics"] == pytest.approx((S1 + S2) / 2)
    assert off["relation_count_apart"] is False

    params(EVIDENCE_QUERY_SIDE_TERMS=(), EVIDENCE_RELATION_COUNT_APART=True)
    on = {}
    evidence_score(Q, C, qs, cs, trace=on)
    assert on["relational_consistency"] == pytest.approx(
        (4 / 5) * (S1 + S2) / 2)
    assert on["relation_count_apart"] is True
    assert on["relational_consistency"] < off["relational_consistency"]
    # (t + p m)/v  ->  (t + p) m / v
    t, p, v, m = 2, 2, 5, (S1 + S2) / 2
    assert off["relational_consistency"] == pytest.approx((t + p * m) / v)
    assert on["relational_consistency"] == pytest.approx((t + p) * m / v)
    # nothing but the relational term and what is built on it moves
    for k in ("node_coverage", "edge_absence", "matched_rel_sim_sum",
              "verifiable_core_edges_count", "rel_present", "rel_physical"):
        assert on[k] == off[k]


def test_unchanged_without_a_count_only_pair(params):
    Q, qs, C, cs = scenario(flags=False)
    out = {}
    for v in (False, True):
        params(EVIDENCE_RELATION_COUNT_APART=v)
        tr = {}
        out[v] = (evidence_score(Q, C, qs, cs, trace=tr)[0],
                  tr["relational_consistency"])
    assert out[True][1] == pytest.approx(out[False][1], abs=1e-15)
    assert out[True][0] == pytest.approx(out[False][0], abs=1e-15)


def test_unchanged_when_nothing_physical_was_matched(params):
    """Flags only: the term is the presence, as it always was."""
    edges = {(1, 3, RC): flag("REGION_CONTENT"),
             (2, 4, RC): flag("REGION_CONTENT")}
    c_edges = dict(edges)
    c_edges[(1, 5, RC)] = flag("REGION_CONTENT")       # absent in the query
    Q, qs = graph(NODES, list(edges), edges)
    C, cs = graph(NODES, list(c_edges), c_edges)
    for v in (False, True):
        params(EVIDENCE_QUERY_SIDE_TERMS=(), EVIDENCE_RELATION_COUNT_APART=v)
        tr = {}
        evidence_score(Q, C, qs, cs, trace=tr)
        assert tr["relational_consistency"] == pytest.approx(2 / 3)
        assert tr["relation_physics"] == 1.0


def test_unchanged_when_the_physics_is_perfect(params):
    edges = {(1, 2, JC): J1, (1, 3, RC): flag("REGION_CONTENT")}
    c_edges = dict(edges)
    c_edges[(3, 4, JC)] = J3
    Q, qs = graph(NODES, list(edges), edges)
    C, cs = graph(NODES, list(c_edges), c_edges)
    for v in (False, True):
        params(EVIDENCE_QUERY_SIDE_TERMS=(), EVIDENCE_RELATION_COUNT_APART=v)
        tr = {}
        evidence_score(Q, C, qs, cs, trace=tr)
        assert tr["relational_consistency"] == pytest.approx(2 / 3)


def test_a_parameter_less_pair_is_count_only(params):
    edges = [(1, 2, TOPO), (2, 3, JC)]
    Q, qs = graph(NODES, edges, {(2, 3, JC): Q2})
    C, cs = graph(NODES, edges, {(2, 3, JC): J2})
    f6 = dict(EDGE_UNPARAMETERISED_STRUCTURAL_MATCH=True,
              EVIDENCE_QUERY_SIDE_TERMS=())
    params(EVIDENCE_RELATION_COUNT_APART=False, **f6)
    off = {}
    evidence_score(Q, C, qs, cs, trace=off)
    assert off["relational_consistency"] == pytest.approx((1 + S2) / 2)
    params(EVIDENCE_RELATION_COUNT_APART=True, **f6)
    on = {}
    evidence_score(Q, C, qs, cs, trace=on)
    assert on["relation_presence"] == 1.0
    assert on["relational_consistency"] == pytest.approx(S2)


def test_both_sides_under_the_query_side_relation_term(params):
    Q, qs, C, cs = scenario(query_extra=True)
    params(EVIDENCE_QUERY_SIDE_TERMS=ALL, EVIDENCE_RELATION_COUNT_APART=True)
    tr = {}
    evidence_score(Q, C, qs, cs, trace=tr)
    # core side: 5 verifiable, 4 present, 2 physical (S1, S2)
    # query side: 6 verifiable, 4 present (the two extra edges have no
    #             counterpart), 2 physical (S1, S2)
    assert tr["relation_presence"] == pytest.approx(8 / 11)
    assert tr["relation_physics"] == pytest.approx((S1 + S2) / 2)
    assert tr["relational_consistency"] == pytest.approx(
        (8 / 11) * (S1 + S2) / 2)
    params(EVIDENCE_QUERY_SIDE_TERMS=ALL, EVIDENCE_RELATION_COUNT_APART=False)
    off = {}
    evidence_score(Q, C, qs, cs, trace=off)
    assert off["relational_consistency"] == pytest.approx(
        (2 * 2 + 2 * (S1 + S2)) / 11)


def test_the_only_edges_branch_is_untouched(params):
    Q, qs, C, cs = scenario()
    out = {}
    for v in (False, True):
        params(EVIDENCE_ONLY_EDGES=True, EVIDENCE_RELATION_COUNT_APART=v)
        tr = {}
        score, _ = evidence_score(Q, C, qs, cs, trace=tr)
        out[v] = (score, tr["relational_consistency"],
                  tr["relational_multiplier"], tr["relation_count_apart"])
    assert out[True] == out[False]
    assert out[True][3] is False


def test_an_edgeless_core_stays_vacuous(params):
    Q, qs, _C, _cs = scenario()
    C, cs = graph(NODES)
    for v in (False, True):
        params(EVIDENCE_RELATION_COUNT_APART=v)
        tr = {}
        evidence_score(Q, C, qs, cs, trace=tr)
        assert tr["relational_consistency"] == 1.0


def random_pair(rng):
    """Twin nodes on both sides (the correspondence is the identity); the
    query's edges are the core's, some dropped, some one bin off, plus a
    few of its own."""
    ids = list(range(1, 9))
    nodes = {i: ("Tex", tex(i, 9 - i)) for i in ids}

    def one(a, b):
        k = rng.randrange(4)
        if k == 0:
            return (a, b, RC), flag("REGION_CONTENT")
        if k == 1:
            return (a, b, TOPO), None
        return (a, b, JC), jc(float(rng.randrange(4)),
                              float(rng.randrange(6)), 3.0)

    ce = dict(one(*sorted(rng.sample(ids, 2)))
              for _ in range(rng.randrange(6, 14)))
    qe = {}
    for e, prm in ce.items():
        if rng.random() < 0.25:
            continue
        if prm is not None and e[2] is JC and rng.random() < 0.6:
            d, inc, topo = prm.quantized
            prm = jc(d + 1.0, inc, topo)
        qe[e] = prm
    for _ in range(rng.randrange(0, 4)):
        e, prm = one(*sorted(rng.sample(ids, 2)))
        qe.setdefault(e, prm)
    Q, qs = graph(nodes, list(qe), {e: x for e, x in qe.items() if x})
    C, cs = graph(nodes, list(ce), {e: x for e, x in ce.items() if x})
    return Q, qs, C, cs


@pytest.mark.parametrize("terms", [(), ALL])
def test_the_term_and_the_score_never_rise(params, terms):
    """Every count-only pair of these graphs reads exactly 1.0 (a flag, a
    parameter-less pair under F6): (t + p m)/v -> (t + p) m/v, m <= 1."""
    rng = random.Random(11 + len(terms))
    moved = 0
    for _ in range(60):
        Q, qs, C, cs = random_pair(rng)
        out = {}
        for v in (False, True):
            params(EVIDENCE_QUERY_SIDE_TERMS=terms,
                   EDGE_UNPARAMETERISED_STRUCTURAL_MATCH=True,
                   EDGE_UNMEASURED_IS_NOT_EVIDENCE=True,
                   EVIDENCE_RELATION_COUNT_APART=v)
            tr = {}
            score, corr = evidence_score(Q, C, qs, cs, trace=tr)
            out[v] = (score, tr["relational_consistency"], corr,
                      tr["node_coverage"], tr["edge_absence"])
        assert out[True][1] <= out[False][1] + 1e-15
        assert out[True][0] <= out[False][0] + 1e-15
        assert out[True][2:] == out[False][2:]
        moved += out[True][1] < out[False][1] - 1e-12
    assert moved > 10


def test_off_is_the_historical_mean_bit_for_bit(params):
    """With the switch off the score is the expression it always was."""
    rng = random.Random(5)
    params(EVIDENCE_QUERY_SIDE_TERMS=(), EVIDENCE_RELATION_COUNT_APART=False)
    for _ in range(40):
        Q, qs, C, cs = random_pair(rng)
        tr = {}
        evidence_score(Q, C, qs, cs, trace=tr)
        v = tr["verifiable_core_edges_count"]
        if len(C.edges) == 0:
            want = 1.0
        elif v == 0:
            want = 0.0
        else:
            want = min(1.0, max(0.0, tr["matched_rel_sim_sum"] / float(v)))
        assert tr["relational_consistency"] == want



def test_with_the_floors_restored_an_unmeasured_pair_is_a_full_count(params):
    """The exception to "never rises", found by the pin triage.  With the
    Phase-4 floors back (EDGE_UNMEASURED_IS_NOT_EVIDENCE = False) an
    all-UNKNOWN pair reads 0.2 -- a similarity of nothing.  It is a
    counterpart with nothing measured: once the count is kept apart it is
    one present relation, and the term rises from 0.2 to 1.0.  Under Phase 4
    the same pair reads 0.0 and is absent under either reading."""
    blank = jc(U, U, U)
    edges = {(1, 2, JC): blank}
    Q, qs = graph(NODES, list(edges), edges)
    C, cs = graph(NODES, list(edges), edges)
    assert edge_graded_axes(blank, blank) == 0
    got = {}
    for p4 in (False, True):
        for apart in (False, True):
            params(EVIDENCE_QUERY_SIDE_TERMS=(),
                   EDGE_UNMEASURED_IS_NOT_EVIDENCE=p4,
                   EVIDENCE_RELATION_COUNT_APART=apart)
            tr = {}
            evidence_score(Q, C, qs, cs, trace=tr)
            got[(p4, apart)] = tr["relational_consistency"]
    assert got[(False, False)] == pytest.approx(0.2)
    assert got[(False, True)] == 1.0
    assert got[(True, False)] == 0.0 and got[(True, True)] == 0.0
