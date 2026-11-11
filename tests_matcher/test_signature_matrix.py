"""PIN -- [SIGNATURE MATRIX IN BLOCKS] the block computation of the
node-similarity matrix returns the per-pair values BIT FOR BIT.

``matching/signature_matrix.py`` evaluates the population branch of
``TypedSignature.similarity``, and the bins path of its Gaussian kernel, for
a whole lens type with numpy.  It is a performance change only, and the
contract is equality, not closeness: every ``==`` below is on the floats
themselves.  Pinned here:

  * the block equals ``population_similarity`` on every pair of random
    populations of the layouts the tree registers, the degenerate vectors
    included (no variance, all zero, one hot, a variance at the 1e-18 guard)
  * the Gaussian block equals ``_similarity_gaussian`` on bins-only
    signatures of every class the tree registers ranges for: circular
    slots, zero-range slots, equal tuples, values far outside the range
  * the matrix equals ``_sig_similarity`` on mixed lists: several classes
    and relations, bins-only signatures, a population of the wrong layout,
    ABSENT slots, integer slots, other arities, a class with its own slot
    rule, a class without ranges, OPAQUE signatures, ``None``
  * registered slot weights are followed; a weight tuple of the wrong
    length raises from both paths
  * it follows the per-pair function when the population path is switched
    off, under the linear kernel, and when the similarity implementation is
    not one the tree ships
  * a block too small to pay for its numpy calls is left to the per-pair
    function, and the result is the same either way
  * the anchor weights equal the per-pair expression of the correspondence,
    missing anchors and missing boxes included
  * in the matcher, P.MATCH_SIGNATURE_MATRIX_BLOCKS changes nothing: same
    score, same correspondence, same trace -- under graded and binary
    credit, the null calibration, the query-side terms and both assignments
"""
from __future__ import annotations

import math
import random
import zlib

import pytest

import core.registries as R
import core.types as T
import matching.signature_matrix as SM
from core.graph_types import Subgraph
from core.population import population_similarity
from core.registries import (CIRCULAR_CYCLE_LENGTHS, CONSTITUTIONAL_RANGES,
                             POPULATION_AXES, get_population_axes)
from core.types import EdgeType, TypedSignature
from matching.signature_matrix import (_gaussian_plan, gaussian_bins_block,
                                       population_similarity_block,
                                       signature_similarity_matrix)
from memory.cores import (_anchor_weights, _normalised_anchors,
                          _sig_similarity, evidence_score, mcs_score)

CLASSES = sorted(c for c, ax in POPULATION_AXES.items() if ax)[:12]

#: every class whose bins path the Gaussian block evaluates
GAUSS_CLASSES = sorted(c for c, r in CONSTITUTIONAL_RANGES.items()
                       if _gaussian_plan(c, len(r)) is not None)


def seed_of(name):
    """A seed that does not depend on PYTHONHASHSEED."""
    return zlib.crc32(name.encode()) % 100000


@pytest.fixture(params=["default sizes", "every block"])
def block_sizes(request, monkeypatch):
    """Run a pin twice: with the size thresholds of the tree, and with every
    block evaluated in numpy however small."""
    if request.param == "every block":
        monkeypatch.setattr(SM, "_MIN_PAIRS_POPULATION", 0)
        monkeypatch.setattr(SM, "_MIN_PAIRS_GAUSSIAN", 0)
    return request.param


def rand_vector(rng, n):
    kind = rng.randrange(8)
    if kind == 0:
        return tuple([1.0 / n] * n)                     # no variance
    if kind == 1:
        return tuple([0.0] * n)                         # all zero
    if kind == 2:
        v = [0.0] * n
        v[rng.randrange(n)] = 1.0                       # one hot
        return tuple(v)
    if kind == 3:                                       # variance near the guard
        return tuple(0.25 + 1e-10 * rng.random() for _ in range(n))
    v = [rng.random() ** rng.choice((1, 3)) for _ in range(n)]
    s = sum(v)
    return tuple(x / s for x in v) if kind < 7 else tuple(v)


def rand_population(rng, axes):
    return tuple(rand_vector(rng, ax.n_bins) for ax in axes)


def rand_signature(rng, cls=None, relation=None):
    cls = cls or rng.choice(CLASSES)
    axes = get_population_axes(cls)
    return TypedSignature(
        kind="CANONICAL", class_=cls,
        relation=rng.choice(("", "", "r")) if relation is None else relation,
        quantized=tuple(float(rng.randrange(4)) for _ in range(3)),
        population=rand_population(rng, axes))


def odd_signature(rng):
    """A signature the block must leave to the per-pair function."""
    kind = rng.randrange(6)
    cls = rng.choice(CLASSES)
    axes = get_population_axes(cls)
    q = tuple(float(rng.randrange(4)) for _ in range(3))
    if kind == 0:
        return None
    if kind == 1:
        return TypedSignature.opaque("tag%d" % rng.randrange(2), rng.randrange(3))
    if kind == 2:                                        # bins only
        return TypedSignature(kind="CANONICAL", class_=cls, quantized=q)
    if kind == 3:                                        # one axis short
        return TypedSignature(kind="CANONICAL", class_=cls, quantized=q,
                              population=rand_population(rng, axes)[:-1])
    if kind == 4:                                        # a vector too long
        pop = list(rand_population(rng, axes))
        pop[0] = pop[0] + (0.5,)
        return TypedSignature(kind="CANONICAL", class_=cls, quantized=q,
                              population=tuple(pop))
    return TypedSignature(kind="CANONICAL", class_="TEXTURE",   # no layout
                          quantized=(float(rng.randrange(4)),
                                     float(rng.randrange(4))))


def scalar_matrix(A, B):
    return [[_sig_similarity(a, b) for b in B] for a in A]


def assert_same(got, want):
    assert len(got) == len(want)
    for rg, rw in zip(got, want):
        assert len(rg) == len(rw)
        for x, y in zip(rg, rw):
            assert type(x) is float
            assert x == y and math.copysign(1.0, x) == math.copysign(1.0, y)


def test_the_tree_registers_population_layouts():
    assert len(CLASSES) >= 6


@pytest.mark.parametrize("cls", CLASSES)
def test_block_equals_population_similarity(cls):
    rng = random.Random(seed_of(cls) + 7)
    axes = get_population_axes(cls)
    pa = [rand_population(rng, axes) for _ in range(23)]
    pb = [rand_population(rng, axes) for _ in range(19)]
    block = population_similarity_block(pa, pb, axes).tolist()
    assert_same(block, [[float(population_similarity(a, b, axes)) for b in pb]
                        for a in pa])


@pytest.mark.parametrize("seed", range(6))
def test_matrix_equals_the_per_pair_function_on_mixed_lists(seed, block_sizes):
    rng = random.Random(seed)
    A = [rand_signature(rng) if rng.random() < 0.7 else odd_signature(rng)
         for _ in range(40)]
    B = [rand_signature(rng) if rng.random() < 0.7 else odd_signature(rng)
         for _ in range(37)]
    assert_same(signature_similarity_matrix(A, B, _sig_similarity),
                scalar_matrix(A, B))


def test_one_class_one_relation_is_the_block_itself():
    rng = random.Random(3)
    cls = CLASSES[0]
    A = [rand_signature(rng, cls, "") for _ in range(30)]
    B = [rand_signature(rng, cls, "") for _ in range(28)]
    got = signature_similarity_matrix(A, B, _sig_similarity)
    assert_same(got, scalar_matrix(A, B))
    assert any(0.0 < x < 1.0 for row in got for x in row)


def test_empty_sides():
    s = rand_signature(random.Random(1))
    assert signature_similarity_matrix([], [s], _sig_similarity) == []
    assert signature_similarity_matrix([s], [], _sig_similarity) == [[]]


def test_it_follows_the_per_pair_function_when_populations_are_off(params):
    rng = random.Random(5)
    A = [rand_signature(rng) for _ in range(12)]
    B = [rand_signature(rng) for _ in range(12)]
    params(POPULATION_CODING_ENABLED=False)
    assert_same(signature_similarity_matrix(A, B, _sig_similarity),
                scalar_matrix(A, B))


def test_an_unknown_similarity_implementation_is_left_alone(monkeypatch):
    rng = random.Random(6)
    A = [rand_signature(rng, CLASSES[0], "") for _ in range(8)]
    B = [rand_signature(rng, CLASSES[0], "") for _ in range(8)]

    def half(self, other, context=None):
        return 0.5 * T._similarity_gaussian(self, other, context=context)

    monkeypatch.setattr(T, "_SIMILARITY_IMPL", half)
    want = scalar_matrix(A, B)
    assert_same(signature_similarity_matrix(A, B, _sig_similarity), want)
    assert any(x != 0.0 for row in want for x in row)


# ------------------------------------------------- the Gaussian bins block

def rand_bins(rng, cls, arity=None):
    """A quantized tuple for `cls`: bin indices, fractional values, values
    beyond the range and the cycle, negative values, signed zeros, values
    near the largest a signature can hold."""
    ranges = CONSTITUTIONAL_RANGES[cls]
    cyc = CIRCULAR_CYCLE_LENGTHS.get(cls) or {}
    out = []
    for i in range(len(ranges) if arity is None else arity):
        top = float(cyc.get(i) or (ranges[i] if i < len(ranges) else 0) or 6)
        k = rng.randrange(8)
        if k <= 1:
            x = float(rng.randrange(int(top) + 1))
        elif k == 2:
            x = rng.uniform(0.0, top)
        elif k == 3:
            x = float(rng.randrange(2))
        elif k == 4:
            x = -rng.uniform(0.0, 2.0 * top)
        elif k == 5:
            x = rng.uniform(1.0, 3.0) * top
        elif k == 6:
            x = float(rng.randrange(int(top) + 1)) + 0.5
        else:
            # (a signature cannot hold more: its serialisation is 64-bit)
            x = rng.choice((0.0, -0.0, 1e-300, 3079196513.0, -9.0e11, 9.0e11))
        out.append(x)
    return tuple(out)


def bins_signature(rng, cls, relation="", **kw):
    return TypedSignature(kind="CANONICAL", class_=cls, relation=relation,
                          quantized=rand_bins(rng, cls), **kw)


def bins_list(rng, cls, n):
    """`n` bins-only signatures of one class; about a third repeat an
    earlier tuple, so equal pairs occur."""
    out = []
    for _ in range(n):
        if out and rng.random() < 0.35:
            out.append(TypedSignature(kind="CANONICAL", class_=cls,
                                      quantized=rng.choice(out).quantized))
        else:
            out.append(bins_signature(rng, cls))
    return out


def test_the_tree_registers_ranges_for_the_gaussian_block():
    assert len(GAUSS_CLASSES) >= 40
    # the classes the ETH-80 graphs carry without a population
    for cls in ("POSITION", "ORIENTATION", "BOUNDARY_ANCHORED", "TEXTURE",
                "COLOR_LEGACY", "ORIENTATION_SEGMENT", "ONSET"):
        assert cls in GAUSS_CLASSES
    # a class with its own slot rule is left to the per-pair function
    assert "REGION_SHAPE" not in GAUSS_CLASSES
    assert _gaussian_plan("NO_SUCH_CLASS", 3) is None
    assert _gaussian_plan("POSITION", 4) is None            # another arity


@pytest.mark.parametrize("cls", GAUSS_CLASSES)
def test_gaussian_block_equals_the_scalar_kernel(cls):
    rng = random.Random(seed_of(cls) + 11)
    A = bins_list(rng, cls, 13)
    B = bins_list(rng, cls, 9) + [
        TypedSignature(kind="CANONICAL", class_=cls, quantized=a.quantized)
        for a in A[:4]]
    plan = _gaussian_plan(cls, len(CONSTITUTIONAL_RANGES[cls]))
    block = gaussian_bins_block([a.quantized for a in A],
                                [b.quantized for b in B], plan).tolist()
    want = [[float(T._similarity_gaussian(a, b)) for b in B] for a in A]
    assert_same(block, want)
    assert any(x == 1.0 for row in want for x in row)       # equal tuples
    # and through the matrix, which must choose the block by itself
    assert_same(signature_similarity_matrix(A, B, _sig_similarity), want)


def odd_bins_signature(rng, cls):
    """A bins-only signature the Gaussian block must leave alone, or a
    partner that makes the pair one the per-pair function decides."""
    kind = rng.randrange(12)
    q = list(rand_bins(rng, cls))
    if kind == 0:                                       # one slot ABSENT
        q[rng.randrange(len(q))] = T.ABSENT
    elif kind == 1:                                     # every slot ABSENT
        q = [T.ABSENT] * len(q)
    elif kind == 2:                                     # an integer slot
        q[rng.randrange(len(q))] = rng.randrange(5)
    elif kind == 3:                                     # one slot more
        q = q + [1.0]
    elif kind == 4:                                     # one slot fewer
        q = q[:-1]
    elif kind == 5:                                     # another relation
        return TypedSignature(kind="CANONICAL", class_=cls, relation="r",
                              quantized=tuple(q))
    elif kind == 6:                                     # a class with a rule
        return bins_signature(rng, "REGION_SHAPE")
    elif kind == 7:                                     # no ranges at all
        return TypedSignature(kind="CANONICAL", class_="NO_SUCH_CLASS",
                              quantized=tuple(q))
    elif kind == 8:                                     # a boolean slot
        q[rng.randrange(len(q))] = True
    elif kind == 9:
        return rng.choice((None, TypedSignature.opaque("t", rng.randrange(2))))
    elif kind == 10:
        # a population one axis short of the layout (PEAK_CHAIN on ETH-80):
        # with a populated partner the similarity is the ALIGNED overlap,
        # not the bins
        axes = get_population_axes(cls)
        pop = (rand_population(rng, axes)[:-1] if axes is not None
               else ((0.25, 0.75),))
        return TypedSignature(kind="CANONICAL", class_=cls,
                              quantized=tuple(q), population=pop)
    else:                                               # a population
        axes = get_population_axes(cls)
        pop = (rand_population(rng, axes) if axes is not None
               else ((0.25, 0.75), (0.5, 0.5)))
        return TypedSignature(kind="CANONICAL", class_=cls,
                              quantized=tuple(q), population=pop)
    return TypedSignature(kind="CANONICAL", class_=cls, quantized=tuple(q))


@pytest.mark.parametrize("seed", range(8))
def test_matrix_equals_the_per_pair_function_on_bins_only_lists(seed,
                                                                block_sizes):
    rng = random.Random(100 + seed)
    # two classes without a population layout, two with one
    classes = ("POSITION", "ORIENTATION", "CORNER", "CONTOUR_PEAK")
    assert get_population_axes("POSITION") is None
    assert get_population_axes("CORNER") is not None

    def one():
        cls = rng.choice(classes)
        if rng.random() < 0.7:
            return bins_signature(rng, cls, rng.choice(("", "", "r")))
        return odd_bins_signature(rng, cls)

    A = [one() for _ in range(45)]
    B = [one() for _ in range(41)] + [a for a in A[:6]]
    assert_same(signature_similarity_matrix(A, B, _sig_similarity),
                scalar_matrix(A, B))


@pytest.mark.parametrize("weights", [(0.5, 2.0, 1.0), (3, 1, 1),
                                     (0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                     (-1.0, 1.0, 1.0), (1e60, 1.0, 1.0)])
def test_registered_slot_weights_are_followed(monkeypatch, weights):
    rng = random.Random(21)
    A = bins_list(rng, "POSITION", 12)
    B = bins_list(rng, "POSITION", 11)
    uniform = scalar_matrix(A, B)
    monkeypatch.setitem(R.SLOT_WEIGHT_PRIORS, "POSITION", weights)
    if weights[0] < 0.0:
        # a negative weight makes the per-pair kernel itself overflow on a
        # distant pair; the matrix leaves such a class to it and raises the
        # same error
        with pytest.raises(OverflowError):
            scalar_matrix(A, B)
        with pytest.raises(OverflowError):
            signature_similarity_matrix(A, B, _sig_similarity)
        assert _gaussian_plan("POSITION", 3) is None
        return
    want = scalar_matrix(A, B)
    assert_same(signature_similarity_matrix(A, B, _sig_similarity), want)
    if weights == (0.5, 2.0, 1.0):
        assert want != uniform, "the weights were not read"
        assert _gaussian_plan("POSITION", 3) is not None
    if weights == (1e60, 1.0, 1.0):
        assert _gaussian_plan("POSITION", 3) is not None   # w / W_sum <= 1


def test_a_weight_tuple_of_the_wrong_length_raises_from_both_paths(monkeypatch):
    rng = random.Random(22)
    A = bins_list(rng, "POSITION", 8)
    B = [bins_signature(rng, "POSITION") for _ in range(8)]
    monkeypatch.setitem(R.SLOT_WEIGHT_PRIORS, "POSITION", (1.0, 1.0))
    with pytest.raises(ValueError):
        scalar_matrix(A, B)
    with pytest.raises(ValueError):
        signature_similarity_matrix(A, B, _sig_similarity)


def test_the_linear_kernel_is_left_to_the_per_pair_function(monkeypatch):
    rng = random.Random(23)
    A = bins_list(rng, "POSITION", 10) + [rand_signature(rng, CLASSES[0], "")
                                          for _ in range(8)]
    B = bins_list(rng, "POSITION", 10) + [rand_signature(rng, CLASSES[0], "")
                                          for _ in range(8)]
    gaussian = scalar_matrix(A, B)
    monkeypatch.setattr(T, "_SIMILARITY_IMPL", T._similarity_linear)
    monkeypatch.setattr(SM, "gaussian_bins_block", None)   # must not be called
    want = scalar_matrix(A, B)
    assert want != gaussian
    assert_same(signature_similarity_matrix(A, B, _sig_similarity), want)


def test_a_small_block_is_left_to_the_per_pair_function(monkeypatch):
    """Speed only: under the thresholds no numpy block is built; from them
    on it is.  The values are the same either way (pinned above)."""
    calls = []
    real_g, real_p = SM.gaussian_bins_block, SM.population_similarity_block
    monkeypatch.setattr(SM, "gaussian_bins_block",
                        lambda *a: calls.append("G") or real_g(*a))
    monkeypatch.setattr(SM, "population_similarity_block",
                        lambda *a: calls.append("P") or real_p(*a))
    rng = random.Random(24)
    g = bins_list(rng, "POSITION", 8)
    p = [rand_signature(rng, CLASSES[0], "") for _ in range(8)]

    def run(A, B):
        del calls[:]
        assert_same(signature_similarity_matrix(A, B, _sig_similarity),
                    scalar_matrix(A, B))
        return list(calls)

    assert SM._MIN_PAIRS_GAUSSIAN == 16 and SM._MIN_PAIRS_POPULATION == 48
    assert run(g[:3], g[:5]) == []                # 15 pairs
    assert run(g[:4], g[:4]) == ["G"]             # 16 pairs
    assert run(p[:6], p[:7]) == []                # 42 pairs
    assert run(p[:6], p[:8]) == ["P"]             # 48 pairs
    # a mixed list: each group is judged by its own size
    assert run(g[:4] + p[:6], g[:4] + p[:7]) == ["G"]
    assert run(g[:4] + p[:6], g[:4] + p[:8]) == ["G", "P"]
    assert run(g[:3] + p[:6], g[:5] + p[:8]) == ["P"]


# ------------------------------------------------------------------ anchors

def closure_weights(sims, q_ids, c_ids, Q, C):
    """The per-pair expression of `_mcs_accumulate`, written out."""
    qa = getattr(Q, "anchor_of_node", None) or {}
    ca = getattr(C, "anchor_of_node", None) or {}

    def bbox(m):
        if not m:
            return None
        xs = [float(p[0]) for p in m.values() if p and len(p) >= 2]
        ys = [float(p[1]) for p in m.values() if p and len(p) >= 2]
        if len(xs) < 2:
            return None
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        return (x0, y0, max(x1 - x0, 1e-9), max(y1 - y0, 1e-9))

    qb, cb = bbox(qa), bbox(ca)

    def anchor(q, c):
        if qb is None or cb is None:
            return 1.0
        pa, pb = qa.get(q), ca.get(c)
        if pa is None or pb is None or len(pa) < 2 or len(pb) < 2:
            return 1.0
        qx = (float(pa[0]) - qb[0]) / qb[2]
        qy = (float(pa[1]) - qb[1]) / qb[3]
        cx = (float(pb[0]) - cb[0]) / cb[2]
        cy = (float(pb[1]) - cb[1]) / cb[3]
        return math.exp(-2.5 * math.hypot(qx - cx, qy - cy))

    return [[sims[i][j] * anchor(q, c) for j, c in enumerate(c_ids)]
            for i, q in enumerate(q_ids)]


def anchored(rng, ids, mode):
    sg = Subgraph(nodes=frozenset(ids))
    if mode == "none":
        return sg
    anchors = {}
    for n in ids:
        r = rng.random()
        if mode == "holes" and r < 0.2:
            continue
        if mode == "holes" and r < 0.3:
            anchors[n] = (rng.uniform(0, 200),)          # too short
        elif mode == "one":
            anchors[n] = (5.0, 5.0)
        else:
            anchors[n] = (rng.uniform(0, 200), rng.uniform(0, 150))
    if mode == "one":
        anchors = {ids[0]: (5.0, 5.0)}
    sg.anchor_of_node = anchors
    return sg


@pytest.mark.parametrize("qmode,cmode", [
    ("full", "full"), ("holes", "full"), ("full", "holes"),
    ("none", "full"), ("full", "one"), ("holes", "holes")])
def test_anchor_weights_equal_the_per_pair_expression(qmode, cmode):
    rng = random.Random(len(qmode) * 7 + len(cmode))
    q_ids = list(range(10, 31))
    c_ids = list(range(100, 118))
    Q, C = anchored(rng, q_ids, qmode), anchored(rng, c_ids, cmode)
    sims = [[rng.choice((0.0, rng.random())) for _ in c_ids] for _ in q_ids]
    got = _anchor_weights(sims, q_ids, c_ids,
                          _normalised_anchors(Q), _normalised_anchors(C))
    assert_same(got, closure_weights(sims, q_ids, c_ids, Q, C))


# -------------------------------------------------------------- the matcher

LENSES = ("L0", "L1", "L2")
ET = (EdgeType.TOPOLOGY, EdgeType.CONTACT)


def random_graph(rng, n, base):
    ids = [base + i for i in range(n)]
    sigs = {}
    for k, nid in enumerate(ids):
        lens = LENSES[k % len(LENSES)]
        cls = CLASSES[LENSES.index(lens)]
        sig = (rand_signature(rng, cls, "") if rng.random() < 0.85
               else odd_signature(rng) or rand_signature(rng, cls, ""))
        sigs[nid] = (lens, sig)
    edges = set()
    for _ in range(2 * n):
        a, b = rng.sample(ids, 2)
        edges.add((min(a, b), max(a, b), rng.choice(ET)))
    sg = Subgraph(nodes=frozenset(ids), edges=frozenset(edges))
    sg.anchor_of_node = {nid: (rng.uniform(0, 100), rng.uniform(0, 100))
                         for nid in ids if rng.random() < 0.9}
    return sg, sigs


SETTINGS = [
    dict(),
    dict(EVIDENCE_GRADED_NODE_CREDIT=False),
    dict(EVIDENCE_NODE_NULL_CALIBRATED=True),
    dict(EVIDENCE_QUERY_SIDE_TERMS=("node", "relation", "absence")),
    dict(EVIDENCE_QUERY_SIDE_TERMS=("node",), EVIDENCE_NODE_NULL_CALIBRATED=True),
    dict(MATCH_ASSIGNMENT_EXACT=False),
]


@pytest.mark.parametrize("setting", range(len(SETTINGS)))
def test_the_switch_changes_nothing_in_the_matcher(params, setting,
                                                   block_sizes):
    rng = random.Random(40 + setting)
    for _ in range(6):
        Q, qs = random_graph(rng, rng.randrange(9, 22), 0)
        C, cs = random_graph(rng, rng.randrange(9, 22), 1000)
        out = {}
        for blocks in (True, False):
            params(MATCH_SIGNATURE_MATRIX_BLOCKS=blocks, **SETTINGS[setting])
            tr = {}
            score, corr = evidence_score(Q, C, qs, cs, trace=tr)
            m_tr = {}
            m_score, m_corr = mcs_score(Q, C, qs, cs, trace=m_tr)
            out[blocks] = (score, corr, tr, m_score, m_corr, m_tr)
        assert out[True] == out[False]
        assert out[True][1], "the scenario matched nothing"
