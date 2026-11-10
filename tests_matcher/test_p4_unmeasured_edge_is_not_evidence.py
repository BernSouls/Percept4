"""PIN -- [PHASE 4 / EDGE HONESTY] an unmeasured edge pair is not evidence.

The EDGE-EVIDENCE spec: "edge similarity depends on measured axes only; an
all-UNKNOWN pair contributes 0; an unmeasured axis contributes no mass (no
UNKNOWN floor 1.0 / 0.2 acting as evidence)".  The ``[U32]`` comment in
``EdgeParams.similarity`` already said so; the code returned

  * 0.2 for two all-UNKNOWN tuples,
  * 1.0 for two tuples that DIFFER but share no measured slot (one measured
    the first axis, the other the second -- nothing was compared), and
  * in the cross-schema path, a mean over ALL shared axes, so every
    unmeasured shared axis counted as a perfect agreement.

Pinned here (P.EDGE_UNMEASURED_IS_NOT_EVIDENCE, default True):
  * no slot measured on both sides            -> 0.0
  * any slot measured on both sides           -> exactly the historical value
  * cross-schema: no measured shared axis     -> 0.0; the mean runs over the
    shared axes measured on both sides
  * in the matcher, such an edge pair adds nothing to the similarity sum
  * the switch restores the floors
"""
from __future__ import annotations

import math

import pytest

from core.schema import UNKNOWN as U
from core.schema import (EdgeParams, edge_measured_axes,
                         polymorphic_edge_similarity)
from core.types import EdgeType

from _graphs import graph, tex

CO = EdgeType.CO_LOCATED


def co(scale: float, salience: float) -> EdgeParams:
    """EDGE_CO_LOCATED: two graded slots, no identity slot."""
    return EdgeParams(class_="EDGE_CO_LOCATED", quantized=(scale, salience))


def sim(a: EdgeParams, b: EdgeParams) -> float:
    return polymorphic_edge_similarity(a, b, edge_types=(CO, CO))


# ---------------------------------------------------------- same schema

def test_two_all_unknown_tuples_are_not_evidence(params):
    assert sim(co(U, U), co(U, U)) == 0.0
    params(EDGE_UNMEASURED_IS_NOT_EVIDENCE=False)
    assert sim(co(U, U), co(U, U)) == pytest.approx(0.2)       # the old floor


def test_tuples_measured_on_different_axes_are_not_a_perfect_match(params):
    a, b = co(U, 5.0), co(3.0, U)       # nothing measured on both sides
    assert edge_measured_axes(a, b) == 0
    assert sim(a, b) == 0.0
    assert sim(b, a) == 0.0
    params(EDGE_UNMEASURED_IS_NOT_EVIDENCE=False)
    assert sim(a, b) == pytest.approx(1.0)                     # the old floor


def test_one_side_all_unknown_is_not_evidence(params):
    assert sim(co(U, U), co(3.0, 5.0)) == 0.0
    assert sim(co(3.0, 5.0), co(U, U)) == 0.0


def test_a_slot_measured_on_both_sides_keeps_the_historical_value(params):
    pairs = [(co(U, 5.0), co(U, 5.0)),      # equal, one axis measured
             (co(U, 5.0), co(U, 6.0)),      # graded, one axis
             (co(2.0, 5.0), co(U, 6.0)),    # graded, one axis in common
             (co(2.0, 5.0), co(2.0, 5.0)),  # equal, two axes
             (co(2.0, 5.0), co(4.0, 7.0))]  # graded, two axes
    on = [sim(a, b) for a, b in pairs]
    params(EDGE_UNMEASURED_IS_NOT_EVIDENCE=False)
    off = [sim(a, b) for a, b in pairs]
    assert on == off
    assert on[0] == 1.0 and on[3] == 1.0
    assert 0.0 < on[1] < 1.0 and on[1] == on[2]
    assert 0.0 < on[4] < on[1]


# --------------------------------------------------------- cross schema

def _cocircular(cocirc: float, dist: float) -> EdgeParams:
    return EdgeParams(class_="EDGE_COCIRCULAR", quantized=(cocirc, dist))


def _contour_continuity(dist: float, turn: float, curv: float) -> EdgeParams:
    return EdgeParams(class_="EDGE_CONTOUR_CONTINUITY",
                      quantized=(dist, turn, curv))


def _orientation_continuity(dist: float, cocirc: float) -> EdgeParams:
    # (dist, perp, cocirc_asym, turn, tail_role, head_role, polarity_rel)
    return EdgeParams(class_="EDGE_ORIENTATION_CONTINUITY",
                      quantized=(dist, U, cocirc, U, U, U, U))


def test_cross_schema_with_no_measured_shared_axis_is_not_evidence(params):
    # the one shared axis ('dist') is unmeasured on both sides
    a, b = _cocircular(3.0, U), _contour_continuity(U, 2.0, 4.0)
    assert polymorphic_edge_similarity(a, b) == 0.0
    assert polymorphic_edge_similarity(b, a) == 0.0
    params(EDGE_UNMEASURED_IS_NOT_EVIDENCE=False)
    assert polymorphic_edge_similarity(a, b) > 0.0             # the old floor


def test_cross_schema_mean_runs_over_the_measured_shared_axes(params):
    # shared: 'cocirc_asym' (measured on both, 3 bins apart) and 'dist'
    # (unmeasured on one side)
    a, b = _cocircular(3.0, U), _orientation_continuity(5.0, 6.0)
    honest = polymorphic_edge_similarity(a, b)
    same = polymorphic_edge_similarity(_cocircular(3.0, U),
                                       _orientation_continuity(5.0, 3.0))
    params(EDGE_UNMEASURED_IS_NOT_EVIDENCE=False)
    floor = polymorphic_edge_similarity(a, b)
    same_floor = polymorphic_edge_similarity(_cocircular(3.0, U),
                                             _orientation_continuity(5.0, 3.0))
    assert 0.0 < honest < floor         # the unmeasured axis no longer dilutes
    # exp(-x/1) against exp(-x/2): the honest value is the floor value squared,
    # up to the common discount factors
    assert honest / same == pytest.approx((floor / same_floor) ** 2, rel=1e-9)
    assert same == same_floor           # agreement on the measured axis: same


def test_cross_schema_fully_measured_pairs_are_untouched(params):
    a, b = _cocircular(3.0, 2.0), _contour_continuity(4.0, 1.0, 1.0)
    on = polymorphic_edge_similarity(a, b)
    params(EDGE_UNMEASURED_IS_NOT_EVIDENCE=False)
    assert on == polymorphic_edge_similarity(a, b)
    assert 0.0 < on < 1.0


# -------------------------------------------------------------- matcher

def _evidence(q_params, c_params):
    from memory.cores import evidence_score

    nodes = {1: ("Texture", tex(1, 1)), 2: ("Texture", tex(8, 8))}
    Q, qs = graph(nodes, [(1, 2, CO)], edge_params={(1, 2, CO): q_params})
    C, cs = graph(nodes, [(1, 2, CO)], edge_params={(1, 2, CO): c_params})
    trace: dict = {}
    score, corr = evidence_score(Q, C, qs, cs, trace=trace)
    assert corr == {1: 1, 2: 2}
    return score, trace


def test_matcher_gives_an_unmeasured_edge_pair_no_similarity(params):
    score, tr = _evidence(co(U, U), co(U, U))
    assert tr["verifiable_core_edges_count"] == 1
    assert tr["matched_rel_sim_sum"] == 0.0
    assert score == 0.0
    params(EDGE_UNMEASURED_IS_NOT_EVIDENCE=False)
    score, tr = _evidence(co(U, U), co(U, U))
    assert tr["matched_rel_sim_sum"] == pytest.approx(0.2)
    assert score == pytest.approx(0.2, abs=1e-6)


def test_matcher_still_credits_a_measured_edge_pair(params):
    score, tr = _evidence(co(U, 5.0), co(U, 5.0))
    assert tr["matched_rel_sim_sum"] == 1.0
    assert tr["coverage_axes"] == [1]
    assert score == pytest.approx(1.0, abs=1e-6)
    score, tr = _evidence(co(U, 5.0), co(U, 6.0))
    assert 0.0 < tr["matched_rel_sim_sum"] < 1.0
    assert math.isclose(score, tr["matched_rel_sim_sum"], abs_tol=1e-6)
