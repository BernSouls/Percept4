"""PIN -- the [P2] specificity cap: documented as the matching-time cap on
the generic proximity edges, unreachable as built.

``polymorphic_edge_similarity`` applies ``edge_match_weight`` at its end, in
the cross-schema branch.  A same-schema pair has returned before that, and
no generic class has a cross-schema partner -- so the factor is 1.0 wherever
it is applied.  Pinned here:

  * the fact: every class with a cross-schema partner has weight 1.0, and
    every class with a weight below 1.0 has no partner but itself
  * as built, a same-schema pair of generic edges reads its full similarity
  * under P.EDGE_SPECIFICITY_CAP_SAME_SCHEMA (default False, dark) it reads
    similarity x specificity; a specific relation is untouched; 0 stays 0;
    the result stays symmetric
  * what that costs: under the cap a graph of generic edges no longer
    scores 1.0 against itself
  * the cap in combination with P4b: a capped flag reads its specificity
    in the mean and a full count once the count is kept apart, so the two
    together can RAISE the relational term
"""
from __future__ import annotations

import pytest

from core.schema import (EDGE_FAMILY_COMPATIBILITY, EDGE_SPECIFICITY,
                         GENERIC_PROXIMITY_EDGES, EdgeParams,
                         edge_match_weight, get_compatible_types,
                         polymorphic_edge_similarity)
from core.types import EdgeType


def test_the_cap_is_unreachable_as_built():
    capped = sorted(c for c in EDGE_SPECIFICITY if edge_match_weight(c) < 1.0)
    assert capped, "no class is capped at all"
    assert set(capped) <= set(GENERIC_PROXIMITY_EDGES)
    # a capped class has no cross-schema partner ...
    for c in capped:
        assert get_compatible_types(c) == frozenset({c})
    # ... and a class that has one is not capped
    for c, partners in EDGE_FAMILY_COMPATIBILITY.items():
        for other in partners:
            assert edge_match_weight(c) == 1.0
            assert edge_match_weight(other) == 1.0


def co(a, b):
    return EdgeParams(class_="EDGE_CO_LOCATED", quantized=(a, b))


def sim(a, b, et):
    return polymorphic_edge_similarity(a, b, use_confidence=False,
                                       edge_types=(et, et))


def test_the_switch_caps_a_same_schema_pair_of_generic_edges(params):
    CL, CO_ = EdgeType.CO_LOCATED, EdgeType.CONTOUR_ORIENTATION
    flag = EdgeParams(class_="EDGE_CONTOUR_ORIENTATION", quantized=(1.0,))
    a, b, far = co(-1.0, 3.0), co(-1.0, 4.0), co(-1.0, 7.0)
    params(EDGE_SPECIFICITY_CAP_SAME_SCHEMA=False)
    full_same, full_near = sim(a, a, CL), sim(a, b, CL)
    assert full_same == 1.0 and 0.0 < full_near < 1.0
    assert sim(flag, flag, CO_) == 1.0
    params(EDGE_SPECIFICITY_CAP_SAME_SCHEMA=True)
    w = edge_match_weight("EDGE_CO_LOCATED")
    assert w == pytest.approx(0.282)
    assert sim(a, a, CL) == pytest.approx(w)
    assert sim(a, b, CL) == pytest.approx(w * full_near)
    assert sim(a, b, CL) == sim(b, a, CL)
    assert sim(flag, flag, CO_) == pytest.approx(
        edge_match_weight("EDGE_CONTOUR_ORIENTATION"))
    assert sim(a, far, CL) == pytest.approx(w * sim_off(a, far, CL, params))


def sim_off(a, b, et, params):
    params(EDGE_SPECIFICITY_CAP_SAME_SCHEMA=False)
    v = sim(a, b, et)
    params(EDGE_SPECIFICITY_CAP_SAME_SCHEMA=True)
    return v


def test_a_specific_relation_is_untouched(params):
    JC = EdgeType.JUNCTION_CONTOUR
    a = EdgeParams(class_="EDGE_JUNCTION_CONTOUR", quantized=(1.0, 2.0, 3.0))
    b = EdgeParams(class_="EDGE_JUNCTION_CONTOUR", quantized=(2.0, 2.0, 3.0))
    c = EdgeParams(class_="EDGE_JUNCTION_CONTOUR", quantized=(2.0, 2.0, 4.0))
    out = []
    for v in (False, True):
        # the parameter-less pair reads 1.0 by F6: stated, not inherited
        params(EDGE_SPECIFICITY_CAP_SAME_SCHEMA=v,
               EDGE_UNPARAMETERISED_STRUCTURAL_MATCH=True)
        out.append((sim(a, a, JC), sim(a, b, JC), sim(a, c, JC),
                    sim(None, None, JC)))
    assert out[0] == out[1]
    assert out[0][2] == 0.0 and out[0][3] == 1.0


def test_under_the_cap_a_graph_no_longer_scores_one_against_itself(params):
    """What the cap costs, stated: the relational term is a mean similarity,
    so a graph whose relations are generic proximity edges does not reach
    1.0 against ITSELF once those edges read their specificity.  Every pin
    that expects 1.0 for an identical generic edge states the uncapped
    value (pin triage, setting `capon`)."""
    from _graphs import graph, tex
    from memory.cores import evidence_score

    CL, JC = EdgeType.CO_LOCATED, EdgeType.JUNCTION_CONTOUR
    nodes = {1: ("L", tex(1, 1)), 2: ("L", tex(5, 5)), 3: ("L", tex(9, 2))}

    def build(et, cls, q):
        return graph(nodes, [(1, 2, et), (2, 3, et)],
                     {(1, 2, et): EdgeParams(class_=cls, quantized=q),
                      (2, 3, et): EdgeParams(class_=cls, quantized=q)})

    generic = build(CL, "EDGE_CO_LOCATED", (-1.0, 3.0))
    specific = build(JC, "EDGE_JUNCTION_CONTOUR", (1.0, 2.0, 3.0))
    w = edge_match_weight("EDGE_CO_LOCATED")
    for cap, want_generic in ((False, 1.0), (True, w)):
        params(EDGE_SPECIFICITY_CAP_SAME_SCHEMA=cap,
               EVIDENCE_QUERY_SIDE_TERMS=(), EVIDENCE_GRADED_NODE_CREDIT=True,
               EVIDENCE_NODE_NULL_CALIBRATED=False,
               EVIDENCE_RELATION_COUNT_APART=False,
               EDGE_UNMEASURED_IS_NOT_EVIDENCE=True)
        tr = {}
        s, corr = evidence_score(generic[0], generic[0], generic[1],
                                 generic[1], trace=tr)
        assert corr == {1: 1, 2: 2, 3: 3}
        assert tr["relational_consistency"] == pytest.approx(want_generic)
        assert s == pytest.approx(want_generic, abs=1e-6)
        s2, _ = evidence_score(specific[0], specific[0], specific[1],
                               specific[1])
        assert s2 == pytest.approx(1.0, abs=1e-6)


def test_under_the_cap_keeping_the_count_apart_can_raise_the_term(params):
    """The second exception to P4b's "never rises" (the first is the
    Phase-4 floors restored).  A membership flag is a count-only pair.  As
    built it reads 1.0, so the mean and presence x physics agree on a lone
    flag.  Under the cap it reads its class's specificity in the MEAN, while
    with the count kept apart it is one present relation with nothing
    measured: presence 1, physics 1 (vacuous).  The two dark switches
    together move the term from the specificity to 1.0.  Found by the
    stored-record ledger (bug hunt, section A) on 5 EVAL pairs measured
    under the cap."""
    from _graphs import graph, tex
    from memory.cores import evidence_score

    CO_ = EdgeType.CONTOUR_ORIENTATION
    nodes = {1: ("L", tex(1, 1)), 2: ("L", tex(5, 5))}
    flag = EdgeParams(class_="EDGE_CONTOUR_ORIENTATION", quantized=(1.0,))
    Q, qs = graph(nodes, [(1, 2, CO_)], {(1, 2, CO_): flag})
    w = edge_match_weight("EDGE_CONTOUR_ORIENTATION")
    assert 0.0 < w < 1.0
    got = {}
    for cap in (False, True):
        for apart in (False, True):
            params(EDGE_SPECIFICITY_CAP_SAME_SCHEMA=cap,
                   EVIDENCE_RELATION_COUNT_APART=apart,
                   EVIDENCE_QUERY_SIDE_TERMS=(),
                   EVIDENCE_GRADED_NODE_CREDIT=True,
                   EVIDENCE_NODE_NULL_CALIBRATED=False,
                   EDGE_UNMEASURED_IS_NOT_EVIDENCE=True)
            tr = {}
            _s, corr = evidence_score(Q, Q, qs, qs, trace=tr)
            assert corr == {1: 1, 2: 2}
            got[(cap, apart)] = tr["relational_consistency"]
    assert got[(False, False)] == 1.0 and got[(False, True)] == 1.0
    assert got[(True, False)] == pytest.approx(w)
    assert got[(True, True)] == 1.0
