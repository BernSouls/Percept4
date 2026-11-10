"""PIN -- [NULL CALIBRATION] node coverage as evidence above chance.

``evidence_score`` credits every pair the correspondence keeps.  When any
pairing of the two graphs would earn nearly the same credit, that credit says
nothing about THIS correspondence: it is what the two graphs' compositions
give for free.  ``matching/calibration.py`` already holds the answer (built,
never called by the live score):

    calibrate_fraction(observed, chance) = clamp((observed - chance) / (1 - chance))
    chance = the coverage of a RANDOM PAIRING = the mean of the candidate matrix

Under ``P.EVIDENCE_NODE_NULL_CALIBRATED`` the node term is that calibrated
value.  The null is formed where the matcher forms pairs (inside each lens
type) and with the masses of the observed term:

    expected credit of a node = sum of the credit over its candidates
                                / max(|Q_l|, |C_l|)

Pinned here:
  * off (the default) the score and the returned masses are untouched
  * the chance level is the hand-computed mean of the matrix -- with lens
    boosts, with node weights, one-sided and balanced (D3)
  * a coverage any pairing reaches is worth 0; a specific one keeps its
    credit; in between, the value is (N - N0) / (1 - N0)
  * binary credit calibrates to nothing: a count is what a random pairing gets
  * a homogeneous graph scores 0 against itself (flagged behaviour change)
  * mcs_score / mcs_similarity do not read the switch
"""
from __future__ import annotations

import pytest

from core.types import EdgeType
from matching.calibration import calibrate_fraction
from matching.structural_similarity import GESTALT_LENS_BOOST
from memory.cores import evidence_score, mcs_score, mcs_similarity

from _graphs import graph, tex

T = EdgeType.TOPOLOGY
ALL = ("node", "relation", "absence")


def _ev(Q, C):
    trace: dict = {}
    score, corr = evidence_score(Q[0], C[0], Q[1], C[1], trace=trace)
    return score, corr, trace


def _g(bins, lens="Texture", base=0, **kw):
    """One node per entry of ``bins``; node k carries tex(b, b)."""
    return graph({base + k: (lens, tex(b, b)) for k, b in enumerate(bins)},
                 **kw)


def _sim(a, b):
    return tex(a, a).similarity(tex(b, b))


# ------------------------------------------------------------------- off

def test_off_is_the_uncalibrated_score(params):
    q, c = _g((1, 6, 12)), _g((2, 7), base=100)
    off, corr_off, tr_off = _ev(q, c)
    assert tr_off["node_null_calibrated"] is False
    assert tr_off["node_coverage"] == tr_off["node_coverage_raw"]
    assert "node_coverage_chance" not in tr_off
    assert "c_chance_nm" not in tr_off and "q_chance_nm" not in tr_off

    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    on, corr_on, tr_on = _ev(q, c)
    assert corr_on == corr_off
    # the calibration only re-reads the node term: everything the matcher
    # accumulated is identical, bit for bit
    for k in ("q_matched_nm", "c_matched_nm", "q_node_total", "c_node_total",
              "matched_rel_sim_sum", "verifiable_core_edges_count",
              "core_node_coverage", "query_node_coverage"):
        assert tr_on[k] == tr_off[k], k
    assert tr_on["node_coverage_raw"] == tr_off["node_coverage"]
    assert on < off


# ------------------------------------------------------------ the null

def test_chance_is_the_mean_of_the_candidate_matrix(params):
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    qb, cb = (1, 6), (2, 7, 12)
    q, c = _g(qb), _g(cb, base=100)
    _s, corr, tr = _ev(q, c)
    assert corr == {0: 100, 1: 101}
    total = sum(_sim(a, b) for a in qb for b in cb)
    # a random maximum pairing of 2 with 3 holds a given pair with
    # probability 1/3; every node's expected credit is its row/column sum / 3
    assert tr["c_chance_nm"] == pytest.approx(total / 3.0, abs=1e-12)
    assert tr["q_chance_nm"] == pytest.approx(total / 3.0, abs=1e-12)
    n0 = (total / 3.0) / 3.0                      # over the 3 core nodes
    assert tr["node_coverage_chance"] == pytest.approx(n0, abs=1e-9)
    n = (_sim(1, 2) + _sim(6, 7)) / 3.0
    assert tr["node_coverage_raw"] == pytest.approx(n, abs=1e-9)
    assert tr["node_coverage"] == pytest.approx(
        calibrate_fraction(n, n0), abs=1e-9)
    assert 0.0 < tr["node_coverage"] < tr["node_coverage_raw"]


def test_chance_carries_the_lens_boost(params):
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    boost = GESTALT_LENS_BOOST["Contour"]
    assert boost != 1.0
    nodes_q = {0: ("Texture", tex(1, 1)), 1: ("Texture", tex(6, 6)),
               2: ("Contour", tex(3, 3)), 3: ("Contour", tex(9, 9))}
    nodes_c = {100: ("Texture", tex(2, 2)), 101: ("Texture", tex(7, 7)),
               102: ("Contour", tex(4, 4))}
    _s, corr, tr = _ev(graph(nodes_q), graph(nodes_c))
    assert corr == {0: 100, 1: 101, 2: 102}
    tex_sum = sum(_sim(a, b) for a in (1, 6) for b in (2, 7)) / 2.0
    con_sum = sum(_sim(a, 4) for a in (3, 9)) / 2.0
    c_total = 2.0 + boost
    assert tr["c_node_total"] == pytest.approx(c_total)
    assert tr["c_chance_nm"] == pytest.approx(tex_sum + boost * con_sum,
                                              abs=1e-12)
    assert tr["node_coverage_chance"] == pytest.approx(
        (tex_sum + boost * con_sum) / c_total, abs=1e-9)
    # a lens type only one side has earns no chance credit (and no credit)
    nodes_c[103] = ("Junction", tex(5, 5))
    _s, _c, tr2 = _ev(graph(nodes_q), graph(nodes_c))
    assert tr2["c_chance_nm"] == tr["c_chance_nm"]
    assert tr2["node_coverage_chance"] < tr["node_coverage_chance"]


def test_chance_carries_the_node_weights(params):
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    qb, cb = (1, 6), (2, 7)
    wq, wc = {0: 1.0, 1: 3.0}, {100: 2.0, 101: 0.5}
    q = _g(qb, node_weights=wq)
    c = _g(cb, base=100, node_weights=wc)
    _s, corr, tr = _ev(q, c)
    assert corr == {0: 100, 1: 101}
    c_ch = sum(wc[100 + j] * sum(_sim(a, b) for a in qb) / 2.0
               for j, b in enumerate(cb))
    q_ch = sum(wq[i] * sum(_sim(a, b) for b in cb) / 2.0
               for i, a in enumerate(qb))
    assert tr["c_chance_nm"] == pytest.approx(c_ch, abs=1e-12)
    assert tr["q_chance_nm"] == pytest.approx(q_ch, abs=1e-12)
    assert tr["node_coverage_chance"] == pytest.approx(c_ch / 2.5, abs=1e-9)


# ------------------------------------------------------- what it is worth

def test_a_coverage_any_pairing_reaches_is_worth_nothing(params):
    # three interchangeable nodes on each side: every pairing scores the same
    q, c = _g((1, 1, 1)), _g((2, 2, 2), base=100)
    off, _c, tr_off = _ev(q, c)
    assert off == pytest.approx(_sim(1, 2), abs=1e-6)
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    on, _c, tr = _ev(q, c)
    assert tr["node_coverage_chance"] == pytest.approx(
        tr["node_coverage_raw"], abs=1e-9)
    assert tr["node_coverage"] == pytest.approx(0.0, abs=1e-9)
    assert on == pytest.approx(0.0, abs=1e-9)


def test_a_specific_correspondence_keeps_its_credit(params):
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    q, c = _g((1, 6, 12)), _g((1, 6, 12), base=100)
    score, corr, tr = _ev(q, c)
    assert corr == {0: 100, 1: 101, 2: 102}
    assert 0.3 < tr["node_coverage_chance"] < 0.5     # the matrix mean
    assert tr["node_coverage"] == pytest.approx(1.0, abs=1e-6)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_between_the_two_the_value_is_the_excess_over_chance(params):
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    qb, cb = (1, 6), (2, 7)
    score, corr, tr = _ev(_g(qb), _g(cb, base=100))
    assert corr == {0: 100, 1: 101}
    n = (_sim(1, 2) + _sim(6, 7)) / 2.0
    n0 = sum(_sim(a, b) for a in qb for b in cb) / 4.0
    assert tr["node_coverage"] == pytest.approx((n - n0) / (1.0 - n0),
                                                abs=1e-9)
    assert score == pytest.approx((n - n0) / (1.0 - n0), abs=1e-6)
    assert score < n                                # never above the raw term


def test_binary_credit_calibrates_to_nothing(params):
    # a count is exactly what a random pairing gets
    params(EVIDENCE_GRADED_NODE_CREDIT=False)
    q, c = _g((1, 6, 12)), _g((2, 7, 11), base=100)
    off, _c, _t = _ev(q, c)
    assert off == pytest.approx(1.0, abs=1e-6)
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    on, _c, tr = _ev(q, c)
    assert tr["node_coverage_raw"] == pytest.approx(1.0, abs=1e-6)
    assert tr["node_coverage_chance"] == pytest.approx(1.0, abs=1e-6)
    assert on == 0.0


def test_a_homogeneous_graph_scores_nothing_against_itself(params):
    """FLAGGED BEHAVIOUR CHANGE: with interchangeable nodes the null explains
    the whole coverage, so a self-match carries no evidence."""
    q = _g((4, 4, 4))
    assert _ev(q, q)[0] == pytest.approx(1.0, abs=1e-6)
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    score, _c, tr = _ev(q, q)
    assert tr["node_coverage_chance"] == pytest.approx(1.0, abs=1e-6)
    assert score == 0.0
    # one distinguishable node is enough to bring the evidence back
    q2 = _g((4, 4, 9))
    assert _ev(q2, q2)[0] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------ with D3

def test_balanced_node_term_is_calibrated_over_both_sides(params):
    params(EVIDENCE_NODE_NULL_CALIBRATED=True,
           EVIDENCE_QUERY_SIDE_TERMS=("node",))
    qb, cb = (1, 6, 12), (2, 7)
    _s, corr, tr = _ev(_g(qb), _g(cb, base=100))
    assert corr == {0: 100, 1: 101}
    total = sum(_sim(a, b) for a in qb for b in cb)
    n0 = (total / 3.0 + total / 3.0) / (3.0 + 2.0)
    n = 2.0 * (_sim(1, 2) + _sim(6, 7)) / (3.0 + 2.0)
    assert tr["node_coverage_chance"] == pytest.approx(n0, abs=1e-9)
    assert tr["node_coverage_raw"] == pytest.approx(n, abs=1e-9)
    assert tr["node_coverage"] == pytest.approx(
        calibrate_fraction(n, n0), abs=1e-9)


def test_calibrated_balanced_score_is_symmetric(params):
    params(EVIDENCE_NODE_NULL_CALIBRATED=True, EVIDENCE_QUERY_SIDE_TERMS=ALL)
    q = _g((1, 6, 12, 3), edges=[(0, 1, T), (1, 2, T), (2, 3, T)])
    c = _g((2, 7, 11), base=100, edges=[(100, 101, T)])
    s_qc, _c1, tr_qc = _ev(q, c)
    s_cq, _c2, tr_cq = _ev(c, q)
    assert s_qc == pytest.approx(s_cq, abs=1e-9)
    assert tr_qc["node_coverage_chance"] == pytest.approx(
        tr_cq["node_coverage_chance"], abs=1e-9)
    assert 0.0 < s_qc < 1.0


def test_calibration_keeps_the_balanced_preference_for_the_whole(params):
    """D3 x null: the balanced node term still ranks the core that explains
    the whole query above a fragment of it; only the size of the gap moves."""
    q = _g((1, 4, 7, 10, 13, 16))
    fragment = _g((1, 4, 7), base=100)
    whole = _g((1, 4, 7, 10, 13, 16), base=200)
    params(EVIDENCE_QUERY_SIDE_TERMS=("node",))
    raw_frag, raw_whole = _ev(q, fragment)[0], _ev(q, whole)[0]
    assert raw_whole == pytest.approx(1.0, abs=1e-6)
    assert raw_frag == pytest.approx((3 + 3) / (6 + 3), abs=1e-6)
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    cal_frag, cal_whole = _ev(q, fragment)[0], _ev(q, whole)[0]
    assert cal_whole == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < cal_frag < raw_frag
    assert cal_whole > cal_frag


# ------------------------------------------------------ other consumers

def test_the_other_scores_do_not_read_the_switch(params):
    q = _g((1, 6, 12), edges=[(0, 1, T)])
    c = _g((2, 7), base=100, edges=[(100, 101, T)])
    before = (mcs_score(q[0], c[0], q[1], c[1]),
              mcs_similarity(q[0], c[0], q[1], c[1]))
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    assert (mcs_score(q[0], c[0], q[1], c[1]),
            mcs_similarity(q[0], c[0], q[1], c[1])) == before


def test_an_untraced_call_returns_the_same_score(params):
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    q, c = _g((1, 6, 12)), _g((2, 7), base=100)
    traced = _ev(q, c)[0]
    assert evidence_score(q[0], c[0], q[1], c[1])[0] == traced


def test_global_consistency_correspondence_reports_the_same_null(params):
    q, c = _g((1, 6)), _g((2, 7, 12), base=100)
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    _s, corr, tr_local = _ev(q, c)
    params(MATCH_GLOBAL_CONSISTENCY_ENABLED=True)
    _s, corr_g, tr_global = _ev(q, c)
    assert corr_g == corr
    assert tr_global["c_chance_nm"] == pytest.approx(tr_local["c_chance_nm"],
                                                     abs=1e-12)
    assert tr_global["node_coverage_chance"] == pytest.approx(
        tr_local["node_coverage_chance"], abs=1e-12)
