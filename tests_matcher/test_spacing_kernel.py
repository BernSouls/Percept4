"""PIN -- [SIGNATURE MATRIX IN BLOCKS] the anchor spacing of a subgraph
(``memory.cores._median_positive_nn``, the bucket size of the redundancy-
aware edge mass) is the same float whether its nearest neighbours are found
by the double loop or with numpy.

The kernel is governed by the same switch as the signature matrix,
``P.MATCH_SIGNATURE_MATRIX_BLOCKS``; it is a performance change only and the
contract is equality on the floats.  The reference below is the function as
it stood before the change, copied verbatim.  Pinned here:

  * the spacing equals the reference on random points, on integer grids
    (many exact ties), with repeated and co-located points, with entries
    that are ``None``, too short or not finite, below and above the size at
    which numpy is used, and across more than one chunk of rows
  * the kernel equals the reference loop point by point when points lie
    closer than the 1e-12 guard, at it, and just above it
  * coordinates too large for a square are left to the loop
  * the redundancy-aware edge weights of a subgraph are the same dict under
    both values of the switch
"""
from __future__ import annotations

import math
import random

import pytest

import memory.cores as MC
from core.graph_types import Subgraph
from core.types import EdgeType
from memory.cores import (_effective_edge_weights, _median_positive_nn,
                          _nearest_positive_distance_of,
                          _nearest_positive_distances)


def reference_spacing(points):
    """`_median_positive_nn` as it stood before the kernel, verbatim."""
    pts = sorted({
        (round(float(p[0]), 6), round(float(p[1]), 6))
        for p in points
        if p is not None and len(p) >= 2
        and math.isfinite(float(p[0])) and math.isfinite(float(p[1]))
    })
    if len(pts) < 2:
        return 0.0
    nn = []
    for i, (x, y) in enumerate(pts):
        best = math.inf
        for j, (u, v) in enumerate(pts):
            if i == j:
                continue
            d = math.hypot(x - u, y - v)
            if 1e-12 < d < best:
                best = d
        if math.isfinite(best):
            nn.append(best)
    if not nn:
        return 0.0
    nn.sort()
    mid = len(nn) // 2
    return nn[mid] if len(nn) % 2 else 0.5 * (nn[mid - 1] + nn[mid])


def reference_distances(pts):
    out = []
    for i, (x, y) in enumerate(pts):
        best = math.inf
        for j, (u, v) in enumerate(pts):
            if i == j:
                continue
            d = math.hypot(x - u, y - v)
            if 1e-12 < d < best:
                best = d
        if math.isfinite(best):
            out.append(best)
    return out


def same(x, y):
    return (type(x) is float and x == y
            and math.copysign(1.0, x) == math.copysign(1.0, y))


def points(rng, n, kind):
    if kind == "uniform":
        return [(rng.uniform(0, 300), rng.uniform(0, 200)) for _ in range(n)]
    if kind == "grid":                      # pixel centres: exact ties
        return [(float(rng.randrange(40)), float(rng.randrange(30)))
                for _ in range(n)]
    if kind == "half-grid":
        return [(rng.randrange(60) * 0.5, rng.randrange(60) * 0.5)
                for _ in range(n)]
    if kind == "clustered":                 # co-located and nearly so
        base = [(rng.uniform(0, 50), rng.uniform(0, 50)) for _ in range(6)]
        out = []
        for _ in range(n):
            bx, by = rng.choice(base)
            k = rng.randrange(4)
            if k == 0:
                out.append((bx, by))
            elif k == 1:
                out.append((bx + 1e-7, by))             # rounds onto the base
            elif k == 2:
                out.append((bx + 2e-6, by - 3e-6))      # a micro-step away
            else:
                out.append((bx + rng.uniform(-2, 2), by + rng.uniform(-2, 2)))
        return out
    if kind == "line":                      # one coordinate constant
        return [(float(rng.randrange(500)), 7.0) for _ in range(n)]
    if kind == "wide":                      # very different scales
        return [(rng.choice((1e-3, 1.0, 1e3, 1e6)) * rng.uniform(-1, 1),
                 rng.choice((1e-3, 1.0, 1e3, 1e6)) * rng.uniform(-1, 1))
                for _ in range(n)]
    raise AssertionError(kind)


KINDS = ("uniform", "grid", "half-grid", "clustered", "line", "wide")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n", [2, 3, 23, 24, 25, 60, 257])
def test_spacing_equals_the_reference(params, kind, n):
    rng = random.Random(1000 * KINDS.index(kind) + n)
    for _ in range(4):
        pts = points(rng, n, kind)
        want = reference_spacing(pts)
        params(MATCH_SIGNATURE_MATRIX_BLOCKS=True)
        assert same(_median_positive_nn(pts), want)
        params(MATCH_SIGNATURE_MATRIX_BLOCKS=False)
        assert same(_median_positive_nn(pts), want)


@pytest.mark.parametrize("kind", KINDS)
def test_every_nearest_distance_equals_the_loop(kind):
    """Not only the median: the distance of every point, float for float."""
    rng = random.Random(77 + KINDS.index(kind))
    for n in (30, 200):
        pts = sorted({(round(x, 6), round(y, 6))
                      for x, y in points(rng, n, kind)})
        got = _nearest_positive_distances(pts)
        want = reference_distances(pts)
        assert len(got) == len(want)
        assert all(same(a, b) for a, b in zip(got, want))


def test_the_kernel_is_used_from_the_threshold_on(params, monkeypatch):
    calls = []
    real = MC._nearest_positive_distances
    monkeypatch.setattr(MC, "_nearest_positive_distances",
                        lambda pts: calls.append(len(pts)) or real(pts))
    small = [(float(i), float(i * i % 7)) for i in range(23)]
    large = small + [(100.0, 100.0)]
    assert MC._NN_BLOCK_MIN_POINTS == 24
    params(MATCH_SIGNATURE_MATRIX_BLOCKS=True)
    _median_positive_nn(small)
    assert calls == []
    _median_positive_nn(large)
    assert calls == [24]
    params(MATCH_SIGNATURE_MATRIX_BLOCKS=False)
    _median_positive_nn(large)
    assert calls == [24]


def test_entries_that_are_not_points_are_skipped_as_before(params):
    rng = random.Random(6)
    pts = points(rng, 80, "uniform")
    pts[3] = None
    pts[10] = (5.0,)
    pts[20] = (math.nan, 1.0)
    pts[30] = (1.0, math.inf)
    pts[40] = (2.0, 3.0, 9.0)                # a third coordinate is ignored
    pts[50] = [4.0, 5.0]
    want = reference_spacing(pts)
    for blocks in (True, False):
        params(MATCH_SIGNATURE_MATRIX_BLOCKS=blocks)
        assert same(_median_positive_nn(pts), want)
        assert _median_positive_nn([]) == 0.0
        assert _median_positive_nn([(1.0, 1.0)] * 40) == 0.0


def test_more_than_one_chunk_of_rows(params):
    rng = random.Random(7)
    pts = points(rng, 1500, "uniform") + points(rng, 300, "grid")
    params(MATCH_SIGNATURE_MATRIX_BLOCKS=True)
    assert 2_000_000 // len({(round(x, 6), round(y, 6)) for x, y in pts}) \
        < len(pts), "the scenario fits one chunk"
    assert same(_median_positive_nn(pts), reference_spacing(pts))


@pytest.mark.parametrize("gap", [0.0, 1e-13, 9.99e-13, 1e-12, 1.0000001e-12,
                                 2e-12, 1e-9])
def test_points_at_the_guard(gap):
    """The kernel itself, on points the rounding of the caller would have
    merged: a neighbour at 1e-12 or closer is not a neighbour, and the
    nearest admissible one must still be found."""
    rng = random.Random(8)
    pts = [(float(rng.randrange(50)), float(rng.randrange(50)))
           for _ in range(40)]
    pts = sorted(set(pts))
    twins = [(x + gap, y) for x, y in pts[:10]] + \
            [(x, y - gap) for x, y in pts[10:20]]
    pts = sorted(set(pts + twins))
    got = _nearest_positive_distances(pts)
    want = reference_distances(pts)
    assert len(got) == len(want)
    assert all(same(a, b) for a, b in zip(got, want))
    for i in range(len(pts)):
        ref = reference_distances_of(pts, i)
        assert _nearest_positive_distance_of(pts, i) == ref


def reference_distances_of(pts, i):
    x, y = pts[i]
    best = math.inf
    for j, (u, v) in enumerate(pts):
        if i != j:
            d = math.hypot(x - u, y - v)
            if 1e-12 < d < best:
                best = d
    return best


def test_a_neighbour_exactly_at_the_guard_is_not_a_neighbour():
    """`1e-12 < d` is strict.  From the origin the twin sits at exactly
    1e-12 -- inside the kernel's candidate margin, and not admissible -- so
    the nearest neighbour is the next point, as in the loop."""
    pts = sorted([(0.0, 0.0), (1e-12, 0.0), (0.0, 3.0), (5.0, 0.0),
                  (7.0, 7.0), (7.0, 7.0 + 1e-12), (7.0, 9.5),
                  (20.0, 1.0), (20.0 + 2e-12, 1.0)])
    assert math.hypot(1e-12, 0.0) == 1e-12
    got = _nearest_positive_distances(pts)
    want = reference_distances(pts)
    assert got == want
    by_point = dict(zip(pts, want))
    assert by_point[(0.0, 0.0)] == 3.0
    assert by_point[(1e-12, 0.0)] == 3.0
    # 2e-12 is above the guard: that twin IS the nearest neighbour
    assert by_point[(20.0, 1.0)] < 1e-11


def test_every_point_closer_than_the_guard_gives_no_distance():
    pts = sorted((1.0 + k * 1e-14, 2.0) for k in range(30))
    assert reference_distances(pts) == []
    assert _nearest_positive_distances(pts) == []


@pytest.mark.parametrize("scale", [1e149, 1e151, 1e200, 1e300])
def test_coordinates_too_large_for_a_square(scale):
    rng = random.Random(9)
    pts = sorted({(scale * rng.uniform(-1, 1), scale * rng.uniform(-1, 1))
                  for _ in range(40)})
    got = _nearest_positive_distances(pts)
    want = reference_distances(pts)
    assert len(got) == len(want)
    assert all(same(a, b) for a, b in zip(got, want))


ET = (EdgeType.CONTACT, EdgeType.CO_LOCATED, EdgeType.TOPOLOGY)


@pytest.mark.parametrize("seed", range(5))
def test_effective_edge_weights_do_not_depend_on_the_switch(params, seed):
    rng = random.Random(50 + seed)
    n = rng.randrange(30, 120)
    ids = list(range(n))
    edges = set()
    for _ in range(4 * n):
        a, b = rng.sample(ids, 2)
        edges.add((min(a, b), max(a, b), rng.choice(ET)))
    sg = Subgraph(nodes=frozenset(ids), edges=frozenset(edges))
    sg.anchor_of_node = {i: (float(rng.randrange(25)), float(rng.randrange(25)))
                         for i in ids if rng.random() < 0.95}
    out = {}
    for blocks in (True, False):
        params(MATCH_SIGNATURE_MATRIX_BLOCKS=blocks)
        out[blocks] = _effective_edge_weights(sg)
    assert out[True] == out[False]
    assert list(out[True].items()) == list(out[False].items())
    assert any(w != 1.0 for w in out[True].values()), \
        "no edge was discounted: the spacing was not used"
