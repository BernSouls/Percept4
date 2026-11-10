"""PIN -- [F6 / PHASE 1] a topology-only edge can be matched.

``polymorphic_edge_similarity`` documented, in its own BLACK-HOLE comment,
that two corresponding edges of the same relational predicate that carry no
``EdgeParams`` are a structural match -- and then returned 0.0 on both
branches.  A core edge of a type that never receives params (TOPOLOGY, DEGREE)
could therefore never be matched, whatever the query held.

Pinned here:
  * same predicate, no params on EITHER side  -> 1.0
  * different predicates                      -> 0.0
  * predicate pair not supplied               -> 0.0 (the strict default that
                                                 every other caller relies on)
  * params on ONE side only                   -> 0.0 (a missing measurement is
                                                 not a perfect one)
  * parameterised pairs                       -> untouched
  * the structural match carries zero measured axes (no coverage facilitation)
  * P.EDGE_UNPARAMETERISED_STRUCTURAL_MATCH = False restores the strict 0.0
"""
from __future__ import annotations

import pytest

from core.schema import (EdgeParams, edge_measured_axes,
                         polymorphic_edge_similarity)
from core.types import EdgeType

from _graphs import graph, tex

T, D, CO = EdgeType.TOPOLOGY, EdgeType.DEGREE, EdgeType.CO_LOCATED


def _co(scale: float, salience: float) -> EdgeParams:
    return EdgeParams(class_="EDGE_CO_LOCATED", quantized=(scale, salience))


# ---------------------------------------------------------------- function

def test_same_predicate_without_params_is_a_structural_match():
    assert polymorphic_edge_similarity(None, None, edge_types=(T, T)) == 1.0
    assert polymorphic_edge_similarity(None, None, edge_types=(D, D)) == 1.0


def test_different_predicates_without_params_do_not_match():
    assert polymorphic_edge_similarity(None, None, edge_types=(T, D)) == 0.0
    assert polymorphic_edge_similarity(None, None, edge_types=(D, T)) == 0.0


def test_without_the_predicate_pair_the_strict_zero_is_kept():
    # matching/structural_similarity.py calls without `edge_types`
    assert polymorphic_edge_similarity(None, None) == 0.0
    assert polymorphic_edge_similarity(None, None, edge_types=None) == 0.0
    assert polymorphic_edge_similarity(None, None,
                                       edge_types=(None, None)) == 0.0


def test_params_on_one_side_only_stay_zero_in_both_directions():
    ep = _co(-1.0, 5.0)
    assert polymorphic_edge_similarity(ep, None, edge_types=(CO, CO)) == 0.0
    assert polymorphic_edge_similarity(None, ep, edge_types=(CO, CO)) == 0.0
    assert ep.similarity(None) == 0.0          # the convention it mirrors


def test_parameterised_pairs_are_untouched_by_the_switch(params):
    pairs = [(_co(-1.0, 5.0), _co(-1.0, 5.0)),
             (_co(-1.0, 5.0), _co(-1.0, 6.0)),
             (_co(-1.0, 0.0), _co(-1.0, 7.0)),
             (_co(-1.0, -1.0), _co(-1.0, -1.0))]
    on = [polymorphic_edge_similarity(a, b, edge_types=(CO, CO))
          for a, b in pairs]
    params(EDGE_UNPARAMETERISED_STRUCTURAL_MATCH=False)
    off = [polymorphic_edge_similarity(a, b, edge_types=(CO, CO))
           for a, b in pairs]
    assert on == off
    assert on[0] == 1.0                        # equal measured bins


def test_the_switch_restores_the_historical_strict_zero(params):
    params(EDGE_UNPARAMETERISED_STRUCTURAL_MATCH=False)
    assert polymorphic_edge_similarity(None, None, edge_types=(T, T)) == 0.0


def test_a_structural_match_measures_no_axis():
    # edge mass = weight * similarity * (1 + log2(axes + 1)); zero axes means
    # the structural match earns no coverage facilitation
    assert edge_measured_axes(None, None) == 0


# ----------------------------------------------------------------- matcher

def _pair(q_edges, c_edges):
    nodes = {1: ("Texture", tex(1, 1)), 2: ("Texture", tex(8, 8))}
    return graph(nodes, q_edges), graph(nodes, c_edges)


def _evidence(q_edges, c_edges):
    from memory.cores import evidence_score

    (Q, qs), (C, cs) = _pair(q_edges, c_edges)
    trace: dict = {}
    score, corr = evidence_score(Q, C, qs, cs, trace=trace)
    assert corr == {1: 1, 2: 2}
    return score, trace


def test_matcher_credits_a_topology_edge_present_on_both_sides():
    score, tr = _evidence([(1, 2, T)], [(1, 2, T)])
    assert tr["verifiable_core_edges_count"] == 1
    assert tr["matched_rel_sim_sum"] == 1.0
    assert tr["relational_consistency"] == 1.0
    assert tr["coverage_axes"] == [0]
    assert score == pytest.approx(1.0, abs=1e-6)


def test_matcher_gave_that_edge_nothing_before_the_fix(params):
    params(EDGE_UNPARAMETERISED_STRUCTURAL_MATCH=False)
    score, tr = _evidence([(1, 2, T)], [(1, 2, T)])
    assert tr["verifiable_core_edges_count"] == 1
    assert tr["matched_rel_sim_sum"] == 0.0
    assert tr["relational_consistency"] == 0.0
    assert score == 0.0


def test_matcher_does_not_credit_a_topology_edge_the_query_lacks():
    score, tr = _evidence([], [(1, 2, T)])
    assert tr["verifiable_core_edges_count"] == 1
    assert tr["matched_rel_sim_sum"] == 0.0
    assert score == 0.0


def test_matcher_does_not_match_it_to_another_predicate_on_the_same_nodes():
    score, tr = _evidence([(1, 2, D)], [(1, 2, T)])
    assert tr["matched_rel_sim_sum"] == 0.0
    assert score == 0.0
