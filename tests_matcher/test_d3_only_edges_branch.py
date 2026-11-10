"""PIN -- [D3] under ``EVIDENCE_ONLY_EDGES`` only the node term reads both sides.

The D3 pre-registration addendum fixed this before any D3 number was
computed, and said it was pinned; it was not.  The SCORE behaved as stated.
The TRACE did not: with "relation" or "absence" named, the mirror loop ran
for nothing and wrote a balanced ``relational_consistency`` that the branch
never reads, and ``query_side_terms`` listed terms that had not acted.

``EVIDENCE_ONLY_EDGES`` (RULE B16) keeps its own relational reading: the
similarity sum over ALL core edges, no absence term.  Pinned here:
  * the score with all three terms named equals the score with "node" alone,
    and differs from the one-sided score
  * the names that cannot act are dropped before any work: the mirror loop
    is not run, the traced relation is the one-sided one, and the trace lists
    the terms the score used
  * outside that branch nothing changes
  * a mistyped name is an error under either branch, and on the early exit
    as well (it used to be read only after the matching, so two graphs with
    no lens type in common hid it)
"""
from __future__ import annotations

import pytest

from core.schema import EdgeParams
from core.types import EdgeType
from memory.cores import evidence_score

from _graphs import graph, tex

T, CO = EdgeType.TOPOLOGY, EdgeType.CO_LOCATED
ALL = ("node", "relation", "absence")


def _ev(Q, C):
    trace: dict = {}
    score, corr = evidence_score(Q[0], C[0], Q[1], C[1], trace=trace)
    return score, corr, trace


def _nodes(ids, base=0):
    return {base + k: ("Texture", tex(k, k)) for k in ids}


def _co(salience: float) -> EdgeParams:
    return EdgeParams(class_="EDGE_CO_LOCATED", quantized=(-1.0, salience))


def _pair():
    """5 query nodes and edges against 4 core nodes and edges: every term
    differs between the one-sided and the balanced reading."""
    q = graph(_nodes(range(5)),
              [(0, 1, T), (1, 2, CO), (2, 3, CO), (3, 4, T), (0, 4, CO)],
              edge_params={(1, 2, CO): _co(5.0), (2, 3, CO): _co(2.0),
                           (0, 4, CO): _co(7.0)})
    c = graph(_nodes(range(4), base=100),
              [(100, 101, T), (101, 102, CO), (102, 103, CO), (100, 103, T)],
              edge_params={(101, 102, CO): _co(5.0), (102, 103, CO): _co(4.0)})
    return q, c


@pytest.fixture
def switches(params):
    params(EDGE_UNPARAMETERISED_STRUCTURAL_MATCH=True,
           EVIDENCE_GRADED_NODE_CREDIT=True,
           EDGE_UNMEASURED_IS_NOT_EVIDENCE=True,
           EVIDENCE_NODE_NULL_CALIBRATED=False)
    return params


def test_only_the_node_term_acts(switches):
    q, c = _pair()
    switches(EVIDENCE_ONLY_EDGES=True)
    by_terms = {}
    for terms in ((), ("node",), ("node", "absence"), ("relation",), ALL):
        switches(EVIDENCE_QUERY_SIDE_TERMS=terms)
        by_terms[terms] = _ev(q, c)
    s_one = by_terms[()][0]
    s_node = by_terms[("node",)][0]
    assert s_node < s_one - 0.05                  # the node term does act
    assert by_terms[ALL][0] == s_node
    assert by_terms[("node", "absence")][0] == s_node
    assert by_terms[("relation",)][0] == s_one    # ... and nothing else does
    # RULE B16's own reading: node term x (similarity sum / ALL core edges)
    tr = by_terms[ALL][2]
    assert tr["relational_multiplier"] == pytest.approx(
        tr["matched_rel_sim_sum"] / tr["n_core_edges"])
    assert s_node == pytest.approx(
        tr["node_coverage"] * tr["relational_multiplier"], abs=1e-12)


def test_the_names_that_cannot_act_are_dropped_before_any_work(switches):
    q, c = _pair()
    switches(EVIDENCE_ONLY_EDGES=True, EVIDENCE_QUERY_SIDE_TERMS=())
    one_sided = _ev(q, c)[2]
    for terms, used in ((ALL, ("node",)), (("node", "absence"), ("node",)),
                        (("relation", "absence"), ())):
        switches(EVIDENCE_QUERY_SIDE_TERMS=terms)
        _s, _c, tr = _ev(q, c)
        assert tr["query_side_terms"] == used
        for key in ("n_query_edges", "q_verifiable_edges_count",
                    "q_matched_rel_sim_sum"):
            assert key not in tr                  # the mirror loop did not run
        for key in ("relational_consistency", "relational_multiplier",
                    "edge_absence"):
            assert tr[key] == one_sided[key], key


def test_outside_that_branch_every_named_term_acts(switches):
    q, c = _pair()
    switches(EVIDENCE_ONLY_EDGES=False, EVIDENCE_QUERY_SIDE_TERMS=ALL)
    s_all, _c, tr = _ev(q, c)
    assert tr["query_side_terms"] == ("absence", "node", "relation")
    assert tr["q_verifiable_edges_count"] == 3
    switches(EVIDENCE_QUERY_SIDE_TERMS=("node",))
    s_node, _c, tr_node = _ev(q, c)
    assert tr["relational_consistency"] != tr_node["relational_consistency"]
    assert tr["edge_absence"] > tr_node["edge_absence"] == 0.0
    assert s_all != s_node


@pytest.mark.parametrize("only_edges", [False, True])
def test_a_mistyped_name_is_an_error_on_every_path(switches, only_edges):
    switches(EVIDENCE_ONLY_EDGES=only_edges,
             EVIDENCE_QUERY_SIDE_TERMS=("node", "relations"))
    q, c = _pair()
    with pytest.raises(ValueError):
        _ev(q, c)
    # two graphs with no lens type in common leave through the early exit
    apart = graph({1: ("Contour", tex(1, 1))})
    with pytest.raises(ValueError):
        _ev(q, apart)
    switches(EVIDENCE_QUERY_SIDE_TERMS=("node",))
    assert _ev(q, apart)[0] == 0.0
