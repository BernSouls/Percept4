"""PIN -- [D1] node credit is graded by signature similarity.

Before D1 a node pair the correspondence kept earned its FULL mass
(``GESTALT_LENS_BOOST * node_weight``) whenever its similarity was merely
positive, so ``node_coverage`` was, to a close approximation, the per-lens
node-count overlap of the two graphs: two cores with the same lens census tied
on the node term however alike or unlike their nodes were.

Pinned here, on ``evidence_score``:
  * credit of a kept pair = boost * weight * TypedSignature.similarity
  * identical graphs still score a full node term
  * with equal lens census, the core whose nodes are more alike now wins
    (binary credit tied them exactly)
  * graded coverage never exceeds the binary coverage of the same
    correspondence, and the correspondence itself does not change
  * P.EVIDENCE_GRADED_NODE_CREDIT = False restores binary credit
and on the containment scores:
  * mcs_score / mcs_similarity keep BINARY mass under either setting (their
    gate thresholds are calibrated on it)
"""
from __future__ import annotations

import random

import pytest

from core.types import TypedSignature
from matching.structural_similarity import GESTALT_LENS_BOOST
from memory.cores import evidence_score, mcs_score, mcs_similarity

from _graphs import graph, tex


def _ev(Q, C):
    trace: dict = {}
    score, corr = evidence_score(Q[0], C[0], Q[1], C[1], trace=trace)
    return score, corr, trace


# Query: three Texture nodes.  Two cores with the SAME lens census (three
# Texture nodes each): `near` is one bin off on every node, `far` is several.
QUERY = graph({1: ("Texture", tex(1, 1)), 2: ("Texture", tex(4, 4)),
               3: ("Texture", tex(8, 8))})
NEAR = graph({11: ("Texture", tex(1, 2)), 12: ("Texture", tex(4, 5)),
              13: ("Texture", tex(8, 9))})
FAR = graph({21: ("Texture", tex(2, 3)), 22: ("Texture", tex(5, 6)),
             23: ("Texture", tex(9, 6))})


def _expected_cov(Q, C, corr):
    """sum boost*w*sim over kept pairs / sum boost*w over the core."""
    num = 0.0
    for q, c in corr.items():
        lt, sig_c = C[1][c]
        w = C[0].node_weights.get(c, 0.0) if C[0].node_weights else 1.0
        num += GESTALT_LENS_BOOST.get(lt, 1.0) * w * Q[1][q][1].similarity(sig_c)
    den = 0.0
    for c, (lt, _s) in C[1].items():
        w = C[0].node_weights.get(c, 0.0) if C[0].node_weights else 1.0
        den += GESTALT_LENS_BOOST.get(lt, 1.0) * w
    return num / den


def test_identical_graphs_keep_a_full_node_term():
    score, corr, tr = _ev(QUERY, QUERY)
    assert corr == {1: 1, 2: 2, 3: 3}
    assert tr["node_credit"] == "graded"
    assert tr["node_coverage"] == pytest.approx(1.0, abs=1e-6)
    assert tr["node_coverage_binary"] == pytest.approx(1.0, abs=1e-6)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_credit_is_boost_times_weight_times_similarity():
    for core in (NEAR, FAR):
        _score, corr, tr = _ev(QUERY, core)
        assert len(corr) == 3
        sims = [QUERY[1][q][1].similarity(core[1][c][1])
                for q, c in corr.items()]
        assert all(0.0 < s < 1.0 for s in sims), sims     # a graded case
        assert tr["node_coverage"] == pytest.approx(
            _expected_cov(QUERY, core, corr), abs=1e-9)
        assert tr["node_coverage"] == pytest.approx(sum(sims) / 3, abs=1e-9)
        # every node found a partner: the binary rule reads full coverage
        assert tr["node_coverage_binary"] == pytest.approx(1.0, abs=1e-6)


def test_equal_census_the_more_alike_core_now_wins():
    s_near, _c, tr_near = _ev(QUERY, NEAR)
    s_far, _c, tr_far = _ev(QUERY, FAR)
    assert tr_near["node_coverage"] > tr_far["node_coverage"] + 0.05
    assert s_near > s_far + 0.05
    # ... where binary credit could not tell them apart
    assert tr_near["node_coverage_binary"] == tr_far["node_coverage_binary"]


def test_binary_credit_tied_them_and_the_switch_restores_it(params):
    params(EVIDENCE_GRADED_NODE_CREDIT=False)
    s_near, c_near, tr_near = _ev(QUERY, NEAR)
    s_far, c_far, tr_far = _ev(QUERY, FAR)
    assert tr_near["node_credit"] == tr_far["node_credit"] == "binary"
    assert tr_near["node_coverage"] == tr_far["node_coverage"]
    assert tr_near["node_coverage"] == tr_near["node_coverage_binary"]
    assert s_near == s_far == pytest.approx(1.0, abs=1e-6)
    assert len(c_near) == len(c_far) == 3


def test_node_weight_and_lens_boost_enter_the_credit():
    assert GESTALT_LENS_BOOST.get("Contour", 1.0) != 1.0   # a boosted lens
    q = graph({1: ("Texture", tex(1, 1)), 2: ("Contour", tex(4, 4))})
    c = graph({11: ("Texture", tex(1, 3)), 12: ("Contour", tex(4, 5))},
              node_weights={11: 0.25, 12: 2.0})
    _score, corr, tr = _ev(q, c)
    assert corr == {1: 11, 2: 12}
    b = GESTALT_LENS_BOOST["Contour"]
    s1 = tex(1, 1).similarity(tex(1, 3))
    s2 = tex(4, 4).similarity(tex(4, 5))
    want = (1.0 * 0.25 * s1 + b * 2.0 * s2) / (1.0 * 0.25 + b * 2.0)
    assert tr["node_coverage"] == pytest.approx(want, abs=1e-9)
    assert tr["node_coverage"] == pytest.approx(_expected_cov(q, c, corr),
                                                abs=1e-9)


def test_the_correspondence_does_not_depend_on_the_credit_rule(params):
    graded = [_ev(QUERY, core)[1] for core in (NEAR, FAR, QUERY)]
    params(EVIDENCE_GRADED_NODE_CREDIT=False)
    binary = [_ev(QUERY, core)[1] for core in (NEAR, FAR, QUERY)]
    assert graded == binary


def _random_graph(rng, n, base):
    return graph({base + i: ("Texture", tex(rng.randrange(10),
                                             rng.randrange(10)))
                  for i in range(n)})


def test_graded_coverage_never_exceeds_binary_coverage():
    rng = random.Random(20261110)
    for _ in range(40):
        q = _random_graph(rng, rng.randrange(1, 9), 0)
        c = _random_graph(rng, rng.randrange(1, 9), 100)
        _s, corr, tr = _ev(q, c)
        assert tr["node_coverage"] <= tr["node_coverage_binary"] + 1e-12
        assert 0.0 <= tr["node_coverage"] <= 1.0
        if corr:
            assert tr["node_coverage"] == pytest.approx(
                _expected_cov(q, c, corr), abs=1e-9)


def test_containment_scores_keep_binary_mass(params):
    """mcs_score / mcs_similarity: every node of the smaller side found a
    partner, so the binary containment is 1.0 -- under either setting."""
    for core in (NEAR, FAR):
        assert mcs_score(QUERY[0], core[0], QUERY[1], core[1])[0] == \
            pytest.approx(1.0)
        assert mcs_similarity(QUERY[0], core[0], QUERY[1], core[1])[0] == \
            pytest.approx(1.0)
    on = [mcs_score(QUERY[0], c[0], QUERY[1], c[1])[0] for c in (NEAR, FAR)]
    params(EVIDENCE_GRADED_NODE_CREDIT=False)
    off = [mcs_score(QUERY[0], c[0], QUERY[1], c[1])[0] for c in (NEAR, FAR)]
    assert on == off


def test_an_unmatched_core_node_earns_nothing_under_either_rule(params):
    # the POLARITY node has no partner of its lens type in the query
    pol = TypedSignature(kind="CANONICAL", class_="POLARITY", quantized=(1.0,))
    core = graph({11: ("Texture", tex(1, 1)), 12: ("Polarity", pol)})
    query = graph({1: ("Texture", tex(1, 1))})
    _s, corr, tr = _ev(query, core)
    assert corr == {1: 11}
    assert tr["node_coverage"] == pytest.approx(0.5, abs=1e-6)
    params(EVIDENCE_GRADED_NODE_CREDIT=False)
    assert _ev(query, core)[2]["node_coverage"] == pytest.approx(0.5, abs=1e-6)
