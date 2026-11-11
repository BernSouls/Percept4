"""PIN -- [EXACT ASSIGNMENT] the node correspondence is a maximum-weight
matching, not an approximation of one.

``_max_weight_bipartite_matching`` pruned the candidates per ROW (top 8),
assigned greedily and polished with 2-opt -- a speed concession from the time
the exact solver was a pure-Python Hungarian.  Measured on the ETH-80 EVAL
pairs: 7.3 % / 5.4 % of the pairable node slots left unpaired although a
partner with a positive weight was free, in every pair; and a different
correspondence for (Q, C) than for (C, Q) in every probed pair.

Under ``P.MATCH_ASSIGNMENT_EXACT`` (default True) the same weight matrix is
solved exactly (``scipy.optimize.linear_sum_assignment``).  Pinned here:
  * the total weight is the brute-force maximum; the approximation is never
    above it and is sometimes below
  * nothing is stranded: no unmatched row and unmatched column share a
    positive weight
  * matching(W transposed) is matching(W) transposed -- ties included
  * only strictly positive, finite weights are paired; empty inputs give []
  * same input, same output; pairs sorted by row
  * the per-row pruning stranded nodes; the exact solver pairs them
  * in the matcher: the correspondence of (Q, C) is that of (C, Q), and the
    balanced score of a pair the approximation scored two ways is one number
  * the switch restores the approximation, bit for bit
"""
from __future__ import annotations

import itertools
import random

import pytest

from matching.structural_similarity import (_exact_max_weight_matching,
                                            _max_weight_bipartite_matching)
from memory.cores import evidence_score

from _graphs import graph, tex

ALL = ("node", "relation", "absence")


def total(w, pairs):
    return sum(w[i][j] for i, j in pairs)


def brute_force(w):
    n, m = len(w), len(w[0])
    best = 0.0
    if n <= m:
        for perm in itertools.permutations(range(m), n):
            best = max(best, sum(w[i][j] for i, j in enumerate(perm)))
    else:
        for perm in itertools.permutations(range(n), m):
            best = max(best, sum(w[i][j] for j, i in enumerate(perm)))
    return best


def random_matrix(rng, n, m, zeros=True):
    return [[(0.0 if zeros and rng.random() < 0.3 else rng.random())
             for _ in range(m)] for _ in range(n)]


def transpose(w):
    return [list(col) for col in zip(*w)]


def stranded(w, pairs):
    """Unmatched (row, column) pairs with a positive weight."""
    rows = {i for i, _j in pairs}
    cols = {j for _i, j in pairs}
    return [(i, j) for i in range(len(w)) for j in range(len(w[0]))
            if i not in rows and j not in cols and w[i][j] > 0.0]


def test_the_total_weight_is_the_maximum(params):
    rng = random.Random(20261110)
    below = 0
    for _ in range(200):
        w = random_matrix(rng, rng.randrange(1, 7), rng.randrange(1, 7))
        best = brute_force(w)
        params(MATCH_ASSIGNMENT_EXACT=True)
        exact = _max_weight_bipartite_matching(w)
        assert total(w, exact) == pytest.approx(best, abs=1e-12)
        assert len(set(i for i, _j in exact)) == len(exact)       # a matching
        assert len(set(j for _i, j in exact)) == len(exact)
        params(MATCH_ASSIGNMENT_EXACT=False)
        approx = _max_weight_bipartite_matching(w)
        assert total(w, approx) <= best + 1e-12
        below += int(total(w, approx) < best - 1e-12)
    assert below > 0          # the approximation does miss the optimum


def test_nothing_is_stranded(params):
    rng = random.Random(4)
    for _ in range(60):
        w = random_matrix(rng, rng.randrange(1, 40), rng.randrange(1, 40))
        params(MATCH_ASSIGNMENT_EXACT=True)
        assert stranded(w, _max_weight_bipartite_matching(w)) == []


def test_the_per_row_pruning_stranded_nodes_and_the_exact_solver_pairs_them(
        params):
    # ten rows that all prefer the same eight columns (weights below
    # THETA_SIM, so the noise-floor union adds nothing); two more columns
    # every row could still use
    w = [[0.60 - 0.01 * i - 0.001 * j for j in range(8)] + [0.10, 0.09]
         for i in range(10)]
    params(MATCH_ASSIGNMENT_EXACT=False)
    approx = _max_weight_bipartite_matching(w)
    assert len(approx) == 8
    assert len(stranded(w, approx)) == 4       # 2 rows x 2 free columns
    params(MATCH_ASSIGNMENT_EXACT=True)
    exact = _max_weight_bipartite_matching(w)
    assert len(exact) == 10 and stranded(w, exact) == []
    assert total(w, exact) > total(w, approx) + 0.15


def test_the_transposed_problem_has_the_transposed_solution(params):
    params(MATCH_ASSIGNMENT_EXACT=True)
    rng = random.Random(9)
    cases = [random_matrix(rng, rng.randrange(1, 25), rng.randrange(1, 25))
             for _ in range(60)]
    # ties: interchangeable rows and columns, square and not
    cases += [[[0.5] * 4 for _ in range(4)], [[0.5] * 3 for _ in range(5)],
              [[0.7, 0.7, 0.2], [0.7, 0.7, 0.2], [0.2, 0.2, 0.9]],
              [[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]]]
    for w in cases:
        direct = _max_weight_bipartite_matching(w)
        swapped = _max_weight_bipartite_matching(transpose(w))
        assert sorted((j, i) for i, j in swapped) == direct
    # the approximation is not symmetric
    params(MATCH_ASSIGNMENT_EXACT=False)
    differ = sum(
        sorted((j, i) for i, j in
               _max_weight_bipartite_matching(transpose(w)))
        != _max_weight_bipartite_matching(w) for w in cases)
    assert differ > 0


def test_only_positive_finite_weights_are_paired(params):
    params(MATCH_ASSIGNMENT_EXACT=True)
    nan, inf = float("nan"), float("inf")
    assert _max_weight_bipartite_matching([]) == []
    assert _max_weight_bipartite_matching([[]]) == []
    assert _max_weight_bipartite_matching([[0.0, 0.0], [0.0, 0.0]]) == []
    assert _max_weight_bipartite_matching([[-1.0, nan], [inf, -0.5]]) == []
    w = [[0.9, 0.0, -1.0], [nan, 0.0, 0.0], [0.0, 0.0, 0.4]]
    assert _max_weight_bipartite_matching(w) == [(0, 0), (2, 2)]
    # a row with no positive weight stays unmatched; nothing is forced
    w = [[0.8, 0.2], [0.0, 0.0], [0.7, 0.0]]
    assert _max_weight_bipartite_matching(w) == [(0, 1), (2, 0)]


def test_same_input_same_output_sorted_by_row(params):
    params(MATCH_ASSIGNMENT_EXACT=True)
    rng = random.Random(2)
    w = random_matrix(rng, 30, 45)
    first = _max_weight_bipartite_matching(w)
    assert first == sorted(first)
    assert all(_max_weight_bipartite_matching(w) == first for _ in range(3))
    assert _exact_max_weight_matching(w) == first
    # the approximation's own arguments do not reach the exact solver
    assert _max_weight_bipartite_matching(w, top_k=1,
                                          max_local_opt_passes=0) == first


def test_the_switch_restores_the_approximation(params):
    params(MATCH_ASSIGNMENT_EXACT=False)
    # the sparse greedy takes the heaviest pair first and cannot undo it:
    # rows 0 and 1 both want column 0; greedy gives it to row 0 (0.9), row 1
    # falls back to column 1 (0.1): 1.0.  The optimum is 0.8 + 0.85 = 1.65
    # -- which 2-opt does find here, so use a 3-cycle 2-opt cannot fix.
    w = [[0.90, 0.80, 0.00],
         [0.85, 0.00, 0.10],
         [0.00, 0.75, 0.70]]
    approx = _max_weight_bipartite_matching(w)
    params(MATCH_ASSIGNMENT_EXACT=True)
    exact = _max_weight_bipartite_matching(w)
    assert total(w, exact) == pytest.approx(brute_force(w))
    assert total(w, approx) <= total(w, exact) + 1e-12
    # whatever the approximation returns, it is the historical function's
    # value: greedy order (weight desc, row, col), then 2-opt
    params(MATCH_ASSIGNMENT_EXACT=False)
    assert _max_weight_bipartite_matching(w) == approx
    assert approx == [(0, 0), (1, 2), (2, 1)] or total(w, approx) == \
        pytest.approx(total(w, exact))


# ------------------------------------------------------------- the matcher

def _ev(Q, C):
    trace: dict = {}
    score, corr = evidence_score(Q[0], C[0], Q[1], C[1], trace=trace)
    return score, corr, trace


def _pruned_pair():
    """Ten Texture nodes a side.  Eight are identical on both sides.  The
    query's last two (q8, q9) resemble those eight more than they resemble
    the core's last two (c8, c9) -- so c8, c9 are beyond their top 8 -- while
    for c8, c9 the query's last two are the best partners there are."""
    q = graph({k: ("Texture", tex(2, 2) if k < 8 else tex(2, 4))
               for k in range(10)})
    c = graph({100 + k: ("Texture", tex(2, 2) if k < 8 else tex(2, 7))
               for k in range(10)})
    return q, c


def test_in_the_matcher_the_two_directions_share_one_correspondence(params):
    params(EDGE_UNPARAMETERISED_STRUCTURAL_MATCH=True,
           EVIDENCE_GRADED_NODE_CREDIT=True, EVIDENCE_QUERY_SIDE_TERMS=ALL,
           EVIDENCE_NODE_NULL_CALIBRATED=False)
    import parameters as P

    a = tex(2, 4).similarity(tex(2, 2))      # q8 with the eight
    b = tex(2, 4).similarity(tex(2, 7))      # q8 with c8
    c_ = tex(2, 2).similarity(tex(2, 7))     # the eight with c8
    assert a > b > c_ > 0.0 and b < P.THETA_SIM   # the scenario is as said
    q, c = _pruned_pair()
    params(MATCH_ASSIGNMENT_EXACT=True)
    s_qc, corr_qc, _t = _ev(q, c)
    s_cq, corr_cq, _t = _ev(c, q)
    assert len(corr_qc) == 10                      # every node is placed
    assert corr_cq == {v: k for k, v in corr_qc.items()}
    assert s_qc == pytest.approx(s_cq, abs=1e-9)
    assert {corr_qc[8], corr_qc[9]} == {108, 109}
    # the approximation: q8, q9 are stranded one way and paired the other
    params(MATCH_ASSIGNMENT_EXACT=False)
    a_qc, acorr_qc, _t = _ev(q, c)
    a_cq, acorr_cq, _t = _ev(c, q)
    assert len(acorr_qc) == 8 and len(acorr_cq) == 10
    assert abs(a_qc - a_cq) > 0.01
    assert a_cq == pytest.approx(s_cq, abs=1e-9)   # the complete one agrees


def test_in_the_matcher_no_pairable_node_is_left_out(params):
    params(EVIDENCE_GRADED_NODE_CREDIT=False, EVIDENCE_QUERY_SIDE_TERMS=(),
           EVIDENCE_NODE_NULL_CALIBRATED=False)
    q, c = _pruned_pair()
    params(MATCH_ASSIGNMENT_EXACT=False)
    _s, corr_a, tr_a = _ev(q, c)
    assert len(corr_a) == 8
    assert tr_a["node_coverage"] == pytest.approx(0.8, abs=1e-6)
    params(MATCH_ASSIGNMENT_EXACT=True)
    _s, corr_e, tr_e = _ev(q, c)
    assert len(corr_e) == 10
    assert tr_e["node_coverage"] == pytest.approx(1.0, abs=1e-6)
    # the eight identical nodes are still paired among themselves
    assert all(corr_e[k] < 108 for k in range(8))
