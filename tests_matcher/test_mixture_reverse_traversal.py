"""PIN -- [MIXTURE REVERSAL] a mixture read against its stored direction is
reversed, like the plain parameters it holds.

The matcher's edge loops read an undirected edge from either end.  Read
against its stored direction, its parameters are reversed
(``reverse_edge_params``: roles swap, a signed turn reflects).  That function
returns anything that is not an ``EdgeParams`` unchanged, so an
``EdgeParamsMixture`` was read backwards with its prototypes NOT reversed:

  * a single-prototype mixture and the plain prototype it holds -- the same
    edge -- compared differently;
  * the same edge pair got two different similarities from its two ends.

``EdgeParamsMixture.flipped_copy`` is the library's own operation for it and
``matching/structural_similarity.py`` uses it; ``memory/cores.py`` did not.
Found by ``diagnostics/phase2_bug_hunt.py`` (sections C, D) on the EVAL
graphs: 198 mixtures a reversal changes, 45 | 44 edge pairs read two ways.

Pinned here (P.EDGE_REVERSE_FLIPS_MIXTURE, default True):
  * a single-prototype mixture reads exactly as its plain prototype, on
    either side, in the historical loop and in the mirror loop
  * the same edge pair reads the same from both ends
  * a mixture with several prototypes is reversed prototype by prototype
  * the switch restores the historical reading
  * untouched: aligned pairs, mixtures a reversal does not change, plain
    parameters, parameter-less edges
  * the loop is shared: mcs_score / mcs_similarity move with the switch too
  * the mirror loop stays the matcher's own loop under either value
"""
from __future__ import annotations

import pytest

from core.schema import (EdgeParams, EdgeParamsMixture,
                         polymorphic_edge_similarity, reverse_edge_params)
from core.types import EdgeType
from memory.cores import (_edge_side_evidence, _reverse_for_traversal,
                          evidence_score, mcs_score, mcs_similarity)
from orchestration.binder import UNDIRECTED_EDGE_TYPES

from _graphs import graph, tex

OC, CO = EdgeType.ORIENTATION_CONTINUITY, EdgeType.CO_LOCATED
ALL = ("node", "relation", "absence")


def oc(turn: float, tail: float, head: float) -> EdgeParams:
    """(dist, perp, cocirc_asym, turn, tail_role, head_role, polarity_rel):
    reversal swaps the two roles and reflects the turn."""
    return EdgeParams(class_="EDGE_ORIENTATION_CONTINUITY",
                      quantized=(2.0, 1.0, 3.0, turn, tail, head, 1.0))


def mix(*protos: EdgeParams) -> EdgeParamsMixture:
    return EdgeParamsMixture(list(protos), [1.0] * len(protos), 0)


P_ROLES = oc(1.0, 0.0, 2.0)        # reversed: roles swapped -> hard gate 0.0
P_TURN = oc(3.0, 1.0, 1.0)         # reversed: turn 3 -> 4, a graded change


def _pair(q_params, c_params, aligned: bool = False):
    """One undirected edge on each side.  ``aligned=False``: the query edge
    1-2 corresponds to the core edge 101-102 END FOR END (1 -> 102, 2 -> 101),
    so each loop reads the other side's edge against its stored direction."""
    q = graph({1: ("Texture", tex(1, 1)), 2: ("Texture", tex(8, 8))},
              [(1, 2, OC)], edge_params={(1, 2, OC): q_params})
    lo, hi = (tex(1, 1), tex(8, 8)) if aligned else (tex(8, 8), tex(1, 1))
    c = graph({101: ("Texture", lo), 102: ("Texture", hi)},
              [(101, 102, OC)], edge_params={(101, 102, OC): c_params})
    return q, c


def _ev(Q, C):
    trace: dict = {}
    score, corr = evidence_score(Q[0], C[0], Q[1], C[1], trace=trace)
    return score, corr, trace


@pytest.fixture
def switches(params):
    params(EDGE_UNPARAMETERISED_STRUCTURAL_MATCH=True,
           EVIDENCE_GRADED_NODE_CREDIT=True, EVIDENCE_QUERY_SIDE_TERMS=ALL,
           EDGE_UNMEASURED_IS_NOT_EVIDENCE=True,
           EVIDENCE_NODE_NULL_CALIBRATED=False,
           EDGE_REVERSE_FLIPS_MIXTURE=True)
    return params


def test_the_scenario_is_what_it_says(switches):
    assert OC in UNDIRECTED_EDGE_TYPES
    for p in (P_ROLES, P_TURN):
        r = reverse_edge_params(p)
        assert r.quantized != p.quantized
        assert reverse_edge_params(r).quantized == p.quantized   # involution
    f = lambda a, b: polymorphic_edge_similarity(      # noqa: E731
        a, b, use_confidence=False, edge_types=(OC, OC))
    assert f(P_ROLES, reverse_edge_params(P_ROLES)) == 0.0
    assert 0.5 < f(P_TURN, reverse_edge_params(P_TURN)) < 1.0
    # the defect: the function leaves a mixture as it is
    m = mix(P_ROLES)
    assert reverse_edge_params(m) is m
    # plain parameters, end for end: the edge and its reverse are one relation
    q, c = _pair(P_ROLES, reverse_edge_params(P_ROLES))
    score, corr, tr = _ev(q, c)
    assert corr == {1: 102, 2: 101}
    assert tr["matched_rel_sim_sum"] == 1.0
    assert tr["q_matched_rel_sim_sum"] == 1.0
    assert score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("p", [P_ROLES, P_TURN])
@pytest.mark.parametrize("q_mixture,c_mixture",
                         [(True, False), (False, True), (True, True)])
def test_a_single_prototype_mixture_reads_as_its_plain_prototype(
        switches, p, q_mixture, c_mixture):
    r = reverse_edge_params(p)
    q, c = _pair(mix(p) if q_mixture else p, mix(r) if c_mixture else r)
    score, corr, tr = _ev(q, c)
    assert corr == {1: 102, 2: 101}
    assert tr["matched_rel_sim_sum"] == 1.0          # the historical loop
    assert tr["q_matched_rel_sim_sum"] == 1.0        # the mirror loop
    assert score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("p,wrong", [
    (P_ROLES, 0.0),
    (P_TURN, polymorphic_edge_similarity(
        P_TURN, reverse_edge_params(P_TURN), use_confidence=False,
        edge_types=(OC, OC)))])
def test_the_switch_restores_the_historical_reading(switches, p, wrong):
    r = reverse_edge_params(p)
    switches(EDGE_REVERSE_FLIPS_MIXTURE=False)
    # the mixture on the QUERY side: the historical loop reads it backwards
    _s, _c, tr = _ev(*_pair(mix(p), r))
    assert tr["matched_rel_sim_sum"] == pytest.approx(wrong)
    assert tr["q_matched_rel_sim_sum"] == 1.0
    # the mixture on the CORE side: the mirror loop does
    _s, _c, tr = _ev(*_pair(p, mix(r)))
    assert tr["matched_rel_sim_sum"] == 1.0
    assert tr["q_matched_rel_sim_sum"] == pytest.approx(wrong)
    # on both: both loops
    _s, _c, tr = _ev(*_pair(mix(p), mix(r)))
    assert tr["matched_rel_sim_sum"] == pytest.approx(wrong)
    assert tr["q_matched_rel_sim_sum"] == pytest.approx(wrong)


def test_the_same_edge_pair_reads_the_same_from_both_ends(switches):
    for q_params, c_params in ((mix(P_ROLES), reverse_edge_params(P_ROLES)),
                               (P_TURN, mix(reverse_edge_params(P_TURN))),
                               (mix(P_TURN), mix(oc(5.0, 1.0, 1.0)))):
        q, c = _pair(q_params, c_params)
        switches(EDGE_REVERSE_FLIPS_MIXTURE=True)
        _s, corr, _t = _ev(q, c)
        inv = {cid: qid for qid, cid in corr.items()}
        from_core = _edge_side_evidence(c[0], q[0], inv)
        from_query = _edge_side_evidence(q[0], c[0], corr)
        assert from_core == from_query
    # historically the two ends disagreed
    q, c = _pair(mix(P_ROLES), reverse_edge_params(P_ROLES))
    switches(EDGE_REVERSE_FLIPS_MIXTURE=False)
    assert _edge_side_evidence(c[0], q[0], {102: 1, 101: 2})[2] == 0.0
    assert _edge_side_evidence(q[0], c[0], {1: 102, 2: 101})[2] == 1.0


def test_several_prototypes_are_reversed_prototype_by_prototype(switches):
    m = mix(P_ROLES, P_TURN)
    flipped = _reverse_for_traversal(m)
    assert isinstance(flipped, EdgeParamsMixture) and flipped is not m
    assert [p.quantized for p in flipped.protos] == [
        reverse_edge_params(P_ROLES).quantized,
        reverse_edge_params(P_TURN).quantized]
    assert flipped.mass == m.mass and m.protos[0] is P_ROLES   # m untouched
    c_params = reverse_edge_params(P_TURN)
    want = polymorphic_edge_similarity(
        m.flipped_copy(), c_params, use_confidence=False, edge_types=(OC, OC))
    assert 0.0 < want < 1.0
    _s, _c, tr = _ev(*_pair(m, c_params))
    assert tr["matched_rel_sim_sum"] == pytest.approx(want)
    switches(EDGE_REVERSE_FLIPS_MIXTURE=False)
    _s, _c, tr = _ev(*_pair(m, c_params))
    assert tr["matched_rel_sim_sum"] != pytest.approx(want)


def test_what_the_switch_does_not_touch(switches):
    cases = {
        # read along the stored direction: nothing is reversed
        "aligned": _pair(mix(P_ROLES), mix(P_ROLES), aligned=True),
        # plain parameters on both sides
        "plain": _pair(P_TURN, reverse_edge_params(P_TURN)),
    }
    # a mixture a reversal does not change (EDGE_CO_LOCATED)
    co = EdgeParams(class_="EDGE_CO_LOCATED", quantized=(-1.0, 5.0))
    assert reverse_edge_params(co).quantized == co.quantized
    q = graph({1: ("Texture", tex(1, 1)), 2: ("Texture", tex(8, 8))},
              [(1, 2, CO)], edge_params={(1, 2, CO): mix(co)})
    c = graph({101: ("Texture", tex(8, 8)), 102: ("Texture", tex(1, 1))},
              [(101, 102, CO)], edge_params={(101, 102, CO): mix(co)})
    cases["symmetric mixture"] = (q, c)
    for name, (q, c) in cases.items():
        switches(EDGE_REVERSE_FLIPS_MIXTURE=True)
        on = _ev(q, c)
        switches(EDGE_REVERSE_FLIPS_MIXTURE=False)
        off = _ev(q, c)
        assert on[0] == off[0] == pytest.approx(1.0, abs=1e-6), name
        for k in ("matched_rel_sim_sum", "q_matched_rel_sim_sum"):
            assert on[2][k] == off[2][k] == 1.0, (name, k)
    # the helper itself
    assert _reverse_for_traversal(None) is None
    assert _reverse_for_traversal(P_TURN).quantized == \
        reverse_edge_params(P_TURN).quantized
    switches(EDGE_REVERSE_FLIPS_MIXTURE=False)
    m = mix(P_TURN)
    assert _reverse_for_traversal(m) is m


def test_the_containment_scores_share_the_loop(switches):
    """mcs_score / mcs_similarity read the same edge similarities: they move
    with the switch as well.  Their gates were set on the historical values."""
    q, c = _pair(mix(P_ROLES), reverse_edge_params(P_ROLES))
    switches(EDGE_REVERSE_FLIPS_MIXTURE=True)
    on = (mcs_score(q[0], c[0], q[1], c[1])[0],
          mcs_similarity(q[0], c[0], q[1], c[1])[0])
    switches(EDGE_REVERSE_FLIPS_MIXTURE=False)
    off = (mcs_score(q[0], c[0], q[1], c[1])[0],
           mcs_similarity(q[0], c[0], q[1], c[1])[0])
    assert on[0] == pytest.approx(1.0) and on[1] == pytest.approx(1.0)
    assert off[0] < on[0] and off[1] < on[1]


@pytest.mark.parametrize("flip", [True, False])
def test_the_mirror_loop_is_the_matchers_loop_under_either_value(switches,
                                                                 flip):
    switches(EDGE_REVERSE_FLIPS_MIXTURE=flip)
    for q_params, c_params in ((mix(P_ROLES), reverse_edge_params(P_ROLES)),
                               (mix(P_TURN), mix(oc(5.0, 1.0, 1.0))),
                               (mix(P_ROLES, P_TURN), P_TURN)):
        q, c = _pair(q_params, c_params)
        _s, corr, tr = _ev(q, c)
        inv = {cid: qid for qid, cid in corr.items()}
        n_edges, n_ver, sim_sum = _edge_side_evidence(c[0], q[0], inv)
        assert (n_edges, n_ver) == (tr["n_core_edges"],
                                    tr["verifiable_core_edges_count"])
        assert sim_sum == tr["matched_rel_sim_sum"]
