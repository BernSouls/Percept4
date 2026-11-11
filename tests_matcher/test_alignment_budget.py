"""PIN -- [D2 PREREQUISITES] the dark global alignment does what its
parameters say.

``MATCH_ALIGN_MAX_PAIR_EVALS`` is "the pair-evaluation budget for the
refinement".  ``global_consistency_alignment`` charged the seed's similarity
matrix to it when it built the matrix itself (``sim_fn``) and not when the
caller supplied one (``sim``).  The same inputs gave two answers by entry
point, and the route the matcher uses (``_global_correspondence``) returned
the UNREFINED seed on any pair with more candidates than the budget -- every
ETH-80 pair.

Pinned here:
  * the defect, reproduced under the historical accounting: with
    |A| x |B| >= budget the ``sim_fn`` route refines nothing while the
    ``sim`` route refines
  * ``budget_refinement_only``: the two routes return the same alignment,
    and the refinement runs
  * a candidate of similarity 0 is never evaluated, never charged and never
    matched; the weights are what they were
  * ``seed_evals_charged`` reproduces the historical ``sim_fn`` charge for a
    caller that supplies the matrix
  * ``_global_correspondence``: blocks or pairs, the same answer; with the
    anchor prior and a single round it IS the live correspondence when
    the lens types carry different signature classes; when two lens types
    share a class it pairs across them ("cross-type", its design)
  * the default path does not go through any of this
"""
from __future__ import annotations

import random

import pytest

import matching.alignment as AL
from core.graph_types import Subgraph
from core.types import EdgeType, TypedSignature
from matching.alignment import global_consistency_alignment
from memory.cores import _global_correspondence, _mcs_accumulate

from _graphs import tex

TOPO = EdgeType.TOPOLOGY


def scene(rng, n_a=7, n_b=8, zero=0.35):
    """Two small graphs, a sparse similarity matrix and typed edges."""
    A = list(range(n_a))
    B = list(range(100, 100 + n_b))
    sim = [[(0.0 if rng.random() < zero else round(rng.uniform(0.2, 1.0), 3))
            for _ in B] for _ in A]
    ea = sorted({tuple(sorted(rng.sample(A, 2))) + (TOPO,)
                 for _ in range(2 * n_a)})
    eb = sorted({tuple(sorted(rng.sample(B, 2))) + (TOPO,)
                 for _ in range(2 * n_b)})
    return A, B, sim, ea, eb


def fn_of(A, B, sim):
    ia = {u: i for i, u in enumerate(A)}
    ib = {v: j for j, v in enumerate(B)}
    return lambda u, v: sim[ia[u]][ib[v]]


def key(al):
    return (al.mapping, al.quality, al.rounds_used, al.budget_exhausted,
            al.systematicity, al.node_coverage)


def test_the_defect_under_the_historical_accounting():
    rng = random.Random(1)
    A, B, sim, ea, eb = scene(rng)
    budget = len(A) * len(B)                # the seed alone spends it
    by_fn = global_consistency_alignment(
        A, B, fn_of(A, B, sim), ea, eb, rounds=4, max_pair_evals=budget)
    by_matrix = global_consistency_alignment(
        A, B, None, ea, eb, rounds=4, max_pair_evals=budget, sim=sim)
    assert by_fn.budget_exhausted and by_fn.rounds_used == 0
    assert by_matrix.rounds_used >= 1
    # ... and a supplied matrix can be told to reproduce the sim_fn charge
    charged = global_consistency_alignment(
        A, B, None, ea, eb, rounds=4, max_pair_evals=budget, sim=sim,
        seed_evals_charged=budget)
    assert key(charged) == key(by_fn)


@pytest.mark.parametrize("seed", range(8))
def test_refinement_only_the_two_routes_agree_and_refine(seed):
    rng = random.Random(10 + seed)
    A, B, sim, ea, eb = scene(rng)
    budget = len(A) * len(B)
    by_fn = global_consistency_alignment(
        A, B, fn_of(A, B, sim), ea, eb, rounds=4, max_pair_evals=budget,
        budget_refinement_only=True)
    by_matrix = global_consistency_alignment(
        A, B, None, ea, eb, rounds=4, max_pair_evals=budget, sim=sim,
        budget_refinement_only=True)
    assert key(by_fn) == key(by_matrix)
    assert by_fn.rounds_used >= 1
    for u, v in by_fn.mapping.items():
        assert sim[A.index(u)][B.index(v)] > 0.0


def test_a_zero_similarity_candidate_is_never_evaluated(monkeypatch):
    rng = random.Random(3)
    A, B, sim, ea, eb = scene(rng)
    seen = []
    real = AL._relational_agreement

    def counting(u, v, *a, **k):
        seen.append((u, v))
        return real(u, v, *a, **k)

    monkeypatch.setattr(AL, "_relational_agreement", counting)
    al = global_consistency_alignment(
        A, B, None, ea, eb, rounds=2, sim=sim, budget_refinement_only=True)
    assert al.rounds_used == 1
    nonzero = sum(1 for row in sim for s in row if s != 0.0)
    assert all(sim[A.index(u)][B.index(v)] != 0.0 for u, v in seen)
    # one sweep over the positive candidates, then one call per kept pair
    assert len(seen) == nonzero + len(al.mapping)

    seen.clear()
    old = global_consistency_alignment(
        A, B, None, ea, eb, rounds=2, sim=sim, budget_refinement_only=False)
    assert len(seen) == len(A) * len(B) + len(old.mapping)
    # the weights are the same, so the alignment is
    assert key(old)[:2] == key(al)[:2]


def test_refinement_only_charges_the_positive_candidates_only():
    rng = random.Random(4)
    A, B, sim, ea, eb = scene(rng, zero=0.6)
    nonzero = sum(1 for row in sim for s in row if s != 0.0)
    # a budget one sweep of the positive candidates fits in, a dense sweep
    # does not: the second round is allowed only under the new accounting
    budget = nonzero + 1
    assert budget < len(A) * len(B)
    new = global_consistency_alignment(
        A, B, None, ea, eb, rounds=6, max_pair_evals=budget, sim=sim,
        budget_refinement_only=True)
    unbounded = global_consistency_alignment(
        A, B, None, ea, eb, rounds=3, max_pair_evals=None, sim=sim,
        budget_refinement_only=True)
    assert new.rounds_used <= 2
    assert key(new)[:2] == key(unbounded)[:2] or new.budget_exhausted


# ---------------------------------------------------------- the matcher

def ori(a, b, c) -> TypedSignature:
    """A bins-only ORIENTATION signature (another class than TEXTURE)."""
    return TypedSignature(kind="CANONICAL", class_="ORIENTATION",
                          quantized=(float(a), float(b), float(c)))


def matcher_pair(rng, n=9, one_class=False):
    """Two lens types on both sides, random signatures, anchors, edges.
    The two lens types carry different signature classes unless
    `one_class`."""
    def sig(k):
        if one_class or k % 2 == 0:
            return tex(rng.randrange(10), rng.randrange(10))
        return ori(rng.randrange(8), rng.randrange(8), rng.randrange(8))

    def side(base):
        ids = [base + i for i in range(n)]
        sigs = {nid: ("L%d" % (k % 2), sig(k)) for k, nid in enumerate(ids)}
        edges = {tuple(sorted(rng.sample(ids, 2))) + (TOPO,)
                 for _ in range(2 * n)}
        sg = Subgraph(nodes=frozenset(ids), edges=frozenset(edges))
        sg.anchor_of_node = {nid: (rng.uniform(0, 100), rng.uniform(0, 100))
                             for nid in ids}
        return sg, sigs
    (Q, qs), (C, cs) = side(0), side(1000)
    return Q, qs, C, cs


@pytest.mark.parametrize("refinement_only", [True, False])
@pytest.mark.parametrize("anchor", [True, False])
def test_global_correspondence_blocks_or_pairs(params, refinement_only,
                                               anchor):
    rng = random.Random(21)
    for _ in range(6):
        Q, qs, C, cs = matcher_pair(rng)
        out = []
        for blocks in (True, False):
            params(MATCH_SIGNATURE_MATRIX_BLOCKS=blocks,
                   MATCH_ALIGN_BUDGET_REFINEMENT_ONLY=refinement_only,
                   MATCH_GLOBAL_ANCHOR_PRIOR=anchor)
            out.append(_global_correspondence(Q, C, qs, cs))
        assert out[0] == out[1]
        assert out[0]


def test_one_round_with_the_anchor_prior_is_the_live_correspondence(params):
    rng = random.Random(22)
    same = 0
    for _ in range(12):
        Q, qs, C, cs = matcher_pair(rng)
        params(MATCH_GLOBAL_CONSISTENCY_ENABLED=False)
        live = _mcs_accumulate(Q, C, qs, cs)[6]
        params(MATCH_GLOBAL_ANCHOR_PRIOR=True, MATCH_ALIGN_ROUNDS=1)
        dark = _global_correspondence(Q, C, qs, cs)
        assert dark == live
        params(MATCH_GLOBAL_ANCHOR_PRIOR=False)
        same += _global_correspondence(Q, C, qs, cs) == live
    assert same < 12, "the anchor prior made no difference in any scenario"


def test_the_dark_path_pairs_across_lens_types_of_one_class(params):
    """"Cross-type" is its design: the candidates are every pair of positive
    signature similarity, whatever the lens type.  The live correspondence
    never pairs across lens types.  A difference to know, not a defect."""
    rng = random.Random(24)
    crossed = 0
    for _ in range(12):
        Q, qs, C, cs = matcher_pair(rng, one_class=True)
        params(MATCH_GLOBAL_CONSISTENCY_ENABLED=False)
        live = _mcs_accumulate(Q, C, qs, cs)[6]
        assert all(qs[u][0] == cs[v][0] for u, v in live.items())
        params(MATCH_GLOBAL_ANCHOR_PRIOR=True, MATCH_ALIGN_ROUNDS=1)
        dark = _global_correspondence(Q, C, qs, cs)
        crossed += any(qs[u][0] != cs[v][0] for u, v in dark.items())
    assert crossed > 0


def test_the_default_path_does_not_use_the_alignment(params, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the global alignment ran on the default path")

    monkeypatch.setattr(AL, "global_consistency_alignment", boom)
    Q, qs, C, cs = matcher_pair(random.Random(23))
    assert _mcs_accumulate(Q, C, qs, cs)[6]
