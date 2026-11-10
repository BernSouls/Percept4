"""PIN -- ``trace["node_coverage_binary"]`` is the BINARY TWIN of the node term.

D1 added the key as "the binary coverage of the same correspondence (== the
node_coverage when graded credit is off)".  D3 and the null calibration then
changed what ``node_coverage`` is -- balanced over both sides, read above
chance -- and the key stayed the one-sided, core-only fraction.  Under the
balanced node term it described a different quantity from the one beside it:
with graded credit off the two were no longer equal, and "graded never
exceeds binary" was false (measured on the stored EVAL records: 3,576 of
3,600 balanced pairs carried a twin of the wrong form; in 727 the node term
exceeded it).  No score reads the key; the harness had worked round it.

Pinned here:
  * the twin has the FORM of the term in use: one-sided, or balanced under
    D3 "node" -- (q_binary + c_binary) / (q_total + c_total)
  * it is never calibrated
  * under binary credit ``node_coverage_raw`` IS the twin, bit for bit, in
    every setting
  * node_coverage <= node_coverage_raw <= node_coverage_binary <= 1 in every
    setting
  * with the query-side terms off the value is the historical one
"""
from __future__ import annotations

import random

import pytest

from core.types import TypedSignature
from memory.cores import evidence_score

from _graphs import graph, tex

ALL = ("node", "relation", "absence")
SETTINGS = [(terms, cal) for terms in ((), ("node",), ALL)
            for cal in (False, True)]


def _ev(Q, C):
    trace: dict = {}
    score, corr = evidence_score(Q[0], C[0], Q[1], C[1], trace=trace)
    return score, corr, trace


POL = TypedSignature(kind="CANONICAL", class_="POLARITY", quantized=(1.0,))
# one query node, two core nodes, one of them without a partner
QUERY = graph({1: ("Texture", tex(1, 1))})
CORE = graph({11: ("Texture", tex(1, 3)), 12: ("Polarity", POL)})


def _random_graph(rng, n, base):
    return graph({base + i: ("Texture", tex(rng.randrange(10),
                                            rng.randrange(10)))
                  for i in range(n)})


def test_the_twin_has_the_form_of_the_term_in_use(params):
    params(EVIDENCE_GRADED_NODE_CREDIT=True,
           EVIDENCE_NODE_NULL_CALIBRATED=False)
    s = tex(1, 1).similarity(tex(1, 3))
    assert 0.0 < s < 1.0

    params(EVIDENCE_QUERY_SIDE_TERMS=())
    _s, corr, tr = _ev(QUERY, CORE)
    assert corr == {1: 11}
    assert tr["node_coverage_raw"] == pytest.approx(s / 2, abs=1e-9)
    assert tr["node_coverage_binary"] == pytest.approx(1 / 2, abs=1e-9)

    params(EVIDENCE_QUERY_SIDE_TERMS=("node",))
    _s, corr, tr = _ev(QUERY, CORE)
    assert tr["node_coverage_raw"] == pytest.approx((s + s) / (1 + 2),
                                                    abs=1e-9)
    # the same form, the pair credited in full: (1 + 1) / (1 + 2), not 1 / 2
    assert tr["node_coverage_binary"] == pytest.approx((1 + 1) / (1 + 2),
                                                       abs=1e-9)
    assert tr["node_coverage_binary"] == pytest.approx(
        (tr["q_matched_nm_binary"] + tr["c_matched_nm_binary"])
        / (tr["q_node_total"] + tr["c_node_total"]), abs=1e-9)


@pytest.mark.parametrize("terms,calibrated", SETTINGS)
def test_under_binary_credit_the_raw_term_is_the_twin(params, terms,
                                                      calibrated):
    params(EVIDENCE_GRADED_NODE_CREDIT=False, EVIDENCE_QUERY_SIDE_TERMS=terms,
           EVIDENCE_NODE_NULL_CALIBRATED=calibrated)
    rng = random.Random(3)
    for _ in range(25):
        q = _random_graph(rng, rng.randrange(1, 9), 0)
        c = _random_graph(rng, rng.randrange(1, 9), 100)
        _s, _c, tr = _ev(q, c)
        assert tr["node_credit"] == "binary"
        assert tr["node_coverage_raw"] == tr["node_coverage_binary"]


@pytest.mark.parametrize("terms", [(), ("node",), ALL])
def test_the_twin_is_never_calibrated(params, terms):
    params(EVIDENCE_GRADED_NODE_CREDIT=True, EVIDENCE_QUERY_SIDE_TERMS=terms)
    rng = random.Random(5)
    pairs = [(_random_graph(rng, rng.randrange(2, 9), 0),
              _random_graph(rng, rng.randrange(2, 9), 100))
             for _ in range(15)]
    params(EVIDENCE_NODE_NULL_CALIBRATED=False)
    off = [_ev(q, c)[2]["node_coverage_binary"] for q, c in pairs]
    params(EVIDENCE_NODE_NULL_CALIBRATED=True)
    on = [_ev(q, c)[2] for q, c in pairs]
    assert [t["node_coverage_binary"] for t in on] == off
    # ... while the term beside it is: some pair must have moved
    assert any(t["node_coverage"] < t["node_coverage_raw"] for t in on)


@pytest.mark.parametrize("graded", [True, False])
@pytest.mark.parametrize("terms,calibrated", SETTINGS)
def test_calibrated_le_raw_le_twin_in_every_setting(params, graded, terms,
                                                    calibrated):
    params(EVIDENCE_GRADED_NODE_CREDIT=graded, EVIDENCE_QUERY_SIDE_TERMS=terms,
           EVIDENCE_NODE_NULL_CALIBRATED=calibrated)
    rng = random.Random(20261110)
    for _ in range(40):
        q = _random_graph(rng, rng.randrange(1, 9), 0)
        c = _random_graph(rng, rng.randrange(1, 9), 100)
        _s, _c, tr = _ev(q, c)
        assert 0.0 <= tr["node_coverage"] <= tr["node_coverage_raw"] + 1e-12
        assert tr["node_coverage_raw"] <= tr["node_coverage_binary"] + 1e-12
        assert tr["node_coverage_binary"] <= 1.0


def test_with_the_query_side_terms_off_the_twin_is_the_historical_value(
        params):
    params(EVIDENCE_GRADED_NODE_CREDIT=True, EVIDENCE_QUERY_SIDE_TERMS=(),
           EVIDENCE_NODE_NULL_CALIBRATED=False)
    import parameters as P

    rng = random.Random(11)
    for _ in range(20):
        q = _random_graph(rng, rng.randrange(1, 9), 0)
        c = _random_graph(rng, rng.randrange(1, 9), 100)
        _s, _c, tr = _ev(q, c)
        assert tr["node_coverage_binary"] == (
            tr["c_matched_nm_binary"] / (tr["c_node_total"] + P.EPSILON))
