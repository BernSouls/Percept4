"""PIN -- [D3] the evidence score can read the QUERY side.

Historically every term of ``evidence_score`` is a fraction of the CORE:
how much of the core found a partner, how alike the core's verifiable edges
are, how many core edges went unverified.  Nothing asks how much of the query
the core explains, so a small core is "covered" by almost any query and two
cores that are both fully covered tie, whatever their size.

``P.EVIDENCE_QUERY_SIDE_TERMS`` names the terms that read both sides (the
balanced form ``mcs_similarity`` and matching/fuzzy.py already use):

    node      (q_matched + c_matched) / (q_total + c_total)
    relation  (core-side sim sum + query-side sim sum)
              / (verifiable core edges + verifiable query edges)
    absence   (uncovered core edges + uncovered query edges)
              / (core edges + query edges)

Pinned here:
  * empty setting == the historical one-sided score
  * node: a core that explains the whole query beats a fully covered fragment
    (they tied before); identical graphs still score 1.0
  * relation: a query relation between mapped nodes that the core lacks costs
  * absence: a query edge the correspondence cannot reach costs
  * an edgeless core keeps the vacuous relational reading in every setting
  * the mirror loop, called in the core direction, IS the matcher's own loop
  * with all three terms the score is symmetric in its two arguments
  * an unknown term name is an error, not a silent no-op
"""
from __future__ import annotations

import pytest

from core.schema import EdgeParams
from core.types import EdgeType, TypedSignature
from memory.cores import _edge_side_evidence, evidence_score

from _graphs import graph, tex

T, CO = EdgeType.TOPOLOGY, EdgeType.CO_LOCATED
ALL = ("node", "relation", "absence")


def _ev(Q, C):
    trace: dict = {}
    score, corr = evidence_score(Q[0], C[0], Q[1], C[1], trace=trace)
    return score, corr, trace


def _nodes(ids, base=0):
    """Texture nodes; node k carries bins (k, k) on either side, so the
    identical-signature pairing is the unique best correspondence."""
    return {base + k: ("Texture", tex(k, k)) for k in ids}


def _co(salience: float) -> EdgeParams:
    return EdgeParams(class_="EDGE_CO_LOCATED", quantized=(-1.0, salience))


# ------------------------------------------------------------------ off

def test_empty_setting_is_the_historical_one_sided_score(params):
    params(EVIDENCE_QUERY_SIDE_TERMS=())
    q = graph(_nodes(range(6)), [(0, 1, T), (1, 2, T), (4, 5, T)])
    c = graph(_nodes(range(3), base=100), [(100, 101, T)])
    score, corr, tr = _ev(q, c)
    assert corr == {0: 100, 1: 101, 2: 102}
    assert tr["query_side_terms"] == ()
    assert tr["node_coverage"] == tr["core_node_coverage"]
    assert "q_verifiable_edges_count" not in tr        # mirror loop not run
    want = (tr["core_node_coverage"] * tr["relational_consistency"]
            - 0.15 * tr["edge_absence"])
    assert score == pytest.approx(want, abs=1e-12)
    assert score == pytest.approx(1.0, abs=1e-6)       # fully covered core


# ------------------------------------------------------------------ node

def test_node_term_prefers_the_core_that_explains_the_whole_query(params):
    q = graph(_nodes(range(6)))
    fragment = graph(_nodes(range(3), base=100))
    whole = graph(_nodes(range(6), base=200))

    params(EVIDENCE_QUERY_SIDE_TERMS=())
    assert _ev(q, fragment)[0] == pytest.approx(1.0, abs=1e-6)
    assert _ev(q, whole)[0] == pytest.approx(1.0, abs=1e-6)     # a tie

    params(EVIDENCE_QUERY_SIDE_TERMS=("node",))
    s_frag, _c, tr = _ev(q, fragment)
    s_whole, _c, _t = _ev(q, whole)
    assert s_whole == pytest.approx(1.0, abs=1e-6)
    assert s_frag == pytest.approx((3 + 3) / (6 + 3), abs=1e-6)
    assert s_whole > s_frag + 0.3
    assert tr["core_node_coverage"] == pytest.approx(1.0, abs=1e-6)
    assert tr["query_node_coverage"] == pytest.approx(0.5, abs=1e-6)
    assert tr["node_coverage"] == pytest.approx(2 / 3, abs=1e-6)


def test_node_term_carries_the_graded_credit(params):
    params(EVIDENCE_QUERY_SIDE_TERMS=("node",))
    q = graph({1: ("Texture", tex(1, 1)), 2: ("Texture", tex(5, 5))})
    c = graph({11: ("Texture", tex(1, 2)), 12: ("Texture", tex(5, 6)),
               13: ("Texture", tex(9, 9))})
    _s, corr, tr = _ev(q, c)
    assert corr == {1: 11, 2: 12}
    sims = [tex(1, 1).similarity(tex(1, 2)), tex(5, 5).similarity(tex(5, 6))]
    assert tr["node_coverage"] == pytest.approx(
        2 * sum(sims) / (2 + 3), abs=1e-9)


# -------------------------------------------------------------- relation

def test_relation_term_counts_a_query_relation_the_core_lacks(params):
    q = graph(_nodes(range(3)), [(0, 1, T), (1, 2, T)])
    c = graph(_nodes(range(3), base=100), [(100, 101, T)])

    params(EVIDENCE_QUERY_SIDE_TERMS=())
    assert _ev(q, c)[0] == pytest.approx(1.0, abs=1e-6)

    params(EVIDENCE_QUERY_SIDE_TERMS=("relation",))
    score, _c, tr = _ev(q, c)
    assert tr["verifiable_core_edges_count"] == 1
    assert tr["matched_rel_sim_sum"] == 1.0
    assert tr["q_verifiable_edges_count"] == 2
    assert tr["q_matched_rel_sim_sum"] == 1.0
    assert tr["relational_consistency"] == pytest.approx(2 / 3)
    assert score == pytest.approx(2 / 3, abs=1e-6)


# --------------------------------------------------------------- absence

def test_absence_term_counts_a_query_edge_the_map_cannot_reach(params):
    pol = TypedSignature(kind="CANONICAL", class_="POLARITY", quantized=(1.0,))
    q_nodes = _nodes(range(3))
    q_nodes[9] = ("Polarity", pol)          # no partner of its lens in core
    q = graph(q_nodes, [(0, 1, T), (2, 9, T)])
    c = graph(_nodes(range(3), base=100), [(100, 101, T)])

    params(EVIDENCE_QUERY_SIDE_TERMS=())
    s0, corr, tr0 = _ev(q, c)
    assert 9 not in corr
    assert tr0["edge_absence"] == 0.0
    assert s0 == pytest.approx(1.0, abs=1e-6)

    params(EVIDENCE_QUERY_SIDE_TERMS=("absence",))
    s1, _c, tr1 = _ev(q, c)
    assert tr1["n_query_edges"] == 2
    assert tr1["q_verifiable_edges_count"] == 1
    assert tr1["edge_absence"] == pytest.approx(1 / 3)
    assert s1 == pytest.approx(1.0 - 0.15 / 3, abs=1e-6)


# ---------------------------------------------------------- edgeless core

def test_an_edgeless_core_keeps_the_vacuous_relational_reading(params):
    q = graph(_nodes(range(3)), [(0, 1, T), (1, 2, T)])
    c = graph(_nodes(range(3), base=100))                 # no edges
    for terms in ((), ("relation",), ("absence",), ALL):
        params(EVIDENCE_QUERY_SIDE_TERMS=terms)
        score, _c, tr = _ev(q, c)
        assert tr["relational_consistency"] == 1.0, terms
        assert tr["edge_absence"] == 0.0, terms
        assert score == pytest.approx(1.0, abs=1e-6), terms


# ------------------------------------------------------------ mirror loop

def _mixed_pair():
    """Param-less and parameterised edges, some matched, some not."""
    q = graph(_nodes(range(5)),
              [(0, 1, T), (1, 2, CO), (2, 3, CO), (3, 4, T), (0, 4, CO)],
              edge_params={(1, 2, CO): _co(5.0), (2, 3, CO): _co(2.0),
                           (0, 4, CO): _co(7.0)})
    c = graph(_nodes(range(4), base=100),
              [(100, 101, T), (101, 102, CO), (102, 103, CO), (100, 103, T)],
              edge_params={(101, 102, CO): _co(5.0), (102, 103, CO): _co(4.0)})
    return q, c


def test_mirror_loop_in_the_core_direction_is_the_matchers_own_loop(params):
    params(EVIDENCE_QUERY_SIDE_TERMS=())
    q, c = _mixed_pair()
    _s, corr, tr = _ev(q, c)
    assert corr == {0: 100, 1: 101, 2: 102, 3: 103}
    inv = {cid: qid for qid, cid in corr.items()}
    n_edges, n_ver, sim_sum = _edge_side_evidence(c[0], q[0], inv)
    assert n_edges == tr["n_core_edges"] == 4
    assert n_ver == tr["verifiable_core_edges_count"] == 4
    assert sim_sum == tr["matched_rel_sim_sum"]            # bit-identical
    # the case is not degenerate: one exact, one graded, one topological,
    # one core edge with no query partner
    assert 2.0 < sim_sum < 3.0


def test_query_side_reads_the_same_pair_from_the_other_end(params):
    params(EVIDENCE_QUERY_SIDE_TERMS=ALL)
    q, c = _mixed_pair()
    _s, corr, tr = _ev(q, c)
    n_edges, n_ver, sim_sum = _edge_side_evidence(q[0], c[0], corr)
    assert (n_edges, n_ver) == (5, 3)       # (3,4,T) and (0,4,CO) reach node 4
    assert tr["n_query_edges"] == 5
    assert tr["q_verifiable_edges_count"] == 3
    assert tr["q_matched_rel_sim_sum"] == sim_sum
    # every matched pair is seen from both ends with the same similarity
    assert sim_sum == pytest.approx(tr["matched_rel_sim_sum"], abs=1e-12)


def test_all_three_terms_make_the_score_symmetric(params):
    params(EVIDENCE_QUERY_SIDE_TERMS=ALL)
    q, c = _mixed_pair()
    s_qc, corr_qc, tr_qc = _ev(q, c)
    s_cq, corr_cq, tr_cq = _ev(c, q)
    assert corr_cq == {v: k for k, v in corr_qc.items()}
    assert s_qc == pytest.approx(s_cq, abs=1e-9)
    for k in ("node_coverage", "relational_consistency", "edge_absence"):
        assert tr_qc[k] == pytest.approx(tr_cq[k], abs=1e-9), k
    # ... which the one-sided score is not
    params(EVIDENCE_QUERY_SIDE_TERMS=())
    assert abs(_ev(q, c)[0] - _ev(c, q)[0]) > 0.01


def test_identical_graphs_score_one_under_every_setting(params):
    q, _c = _mixed_pair()
    for terms in ((), ("node",), ("node", "absence"), ALL):
        params(EVIDENCE_QUERY_SIDE_TERMS=terms)
        assert _ev(q, q)[0] == pytest.approx(1.0, abs=1e-6), terms


# ------------------------------------------------------------- the switch

def test_an_unknown_term_is_an_error(params):
    params(EVIDENCE_QUERY_SIDE_TERMS=("nodes",))
    q = graph(_nodes(range(2)))
    with pytest.raises(ValueError):
        _ev(q, q)


def test_a_single_name_may_be_given_as_a_string(params):
    params(EVIDENCE_QUERY_SIDE_TERMS="node")
    q = graph(_nodes(range(2)))
    assert _ev(q, q)[2]["query_side_terms"] == ("node",)
