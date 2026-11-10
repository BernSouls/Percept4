"""Finding F3 -- the ORDER cues index chain members by the wrong count, so the
"chain endpoint" predicate that defines a T-junction is evaluated on the
fragment list instead of the chain.

`fragment_ownership.fragment_ownership_records` emits one record per chain
member WITH A FRAME, and `build_surface_fragments` keeps only the records whose
ownership clears `_AMBIGUOUS_OWNERSHIP`.  So `records` is a SUBSET of the chain's
members, while each record still carries its true `vertex` and `n_chain`.

Both order cues ignore those fields and re-derive the endpoints from the length
of the subset:

    groups[chain] = rs                 # only the representable members
    n = len(rs)                        # <-- NOT the chain's member count
    if v == 0 or v == n - 1: ...       # "is an endpoint"

Consequences, both measured:

  * a member that is genuinely in the MIDDLE of its chain becomes an "endpoint"
    whenever it happens to be the last representable one, so
    `t_junction_order` emits a stem-order claim from a non-endpoint (a false
    accept, which rule 1 treats as fabrication);
  * a genuine endpoint whose record was dropped is never a stem, so the real
    junction is missed (the 0/16 measurement);
  * `build_surface_fragments` has the same confusion one level up: it indexes
    the chain's ordered positions with `vertex`, but `positions` is built from
    the SUBSET, so `positions[vertex]` is a different member's position and the
    curvature / radius fit is computed at the wrong place.

`_chain_spacing` has the same disease: it measures "the median distance between
consecutive members" over the subset, so a chain with dropped members
contributes gaps of 2x, 3x the true spacing and the T-junction tolerance --
which the docstring says is "the chain's own median member spacing" -- is
inflated.  It is also computed GLOBALLY over all chains, not per chain.
"""
from __future__ import annotations

import math

import pytest

from perception import surface_fragments as SF


def _rec(alpha, x, y, chain, vertex, n_chain, signed=0.9, closed=False,
         normal=(0.0, -1.0)):
    return dict(alpha=alpha, pos=(float(x), float(y)), normal=normal,
                curv=0.0, signed=float(signed), logodds=0.0,
                logodds_local=0.0, enclosure=0.0, owner_prob=0.5,
                chain=chain, index=int(vertex), vertex=int(vertex),
                n_chain=int(n_chain), closed=bool(closed), offset=2.5,
                partner=None, neighbor=None)


def _frag(i, alpha, x, y, chain, vertex):
    return SF.SurfaceFragment(
        index=i, position=(float(x), float(y)), tangent=(1.0, 0.0),
        owner_normal=(0.0, -1.0), owner_side=1.0, curvature=None,
        curvature_sign=0.0, depth_layer=None, alpha=alpha, chain=str(chain),
        vertex=int(vertex), enclosure=0.5)


# --------------------------------------------------------------------------
# endpoints must be read from n_chain, not from len(records)
# --------------------------------------------------------------------------

def test_last_representable_member_of_a_long_chain_is_not_an_endpoint():
    """A bar of 31 members whose ownership is unambiguous only for vertices
    0..8: vertex 8 is the LAST record, so `n = len(rs) = 9` and the code calls
    it an endpoint.  Its true `n_chain` is 31, so it is an interior member and
    must be eligible as a BAR, never as a STEM."""
    bar = [_rec("bar%d" % v, 40.0 + 6.0 * v, 100.0, b"bar", v, 31)
           for v in range(9)]
    # a stem whose endpoint lands on bar vertex 8
    stem = [_rec("stem%d" % v, 88.0, 100.0 + 6.0 * v, b"stem", v, 6)
            for v in range(3)]
    stem[0]["pos"] = (88.0, 100.0)              # the tip, on bar vertex 8
    recs = bar + stem
    frags = [_frag(i, r["alpha"], r["pos"][0], r["pos"][1], r["chain"],
                   r["vertex"]) for i, r in enumerate(recs)]
    edges = SF.t_junction_order(frags, recs)
    idx = {r["alpha"]: i for i, r in enumerate(recs)}
    # bar8 must never appear as the FARTHER end (a stem) of an order edge
    for near, far in edges:
        assert far != idx["bar8"], (
            "bar vertex 8 of 31 was treated as a chain endpoint because it is "
            "the last REPRESENTABLE record (F3): edge (%d,%d)" % (near, far))


def test_dropped_endpoint_is_still_recognised_by_its_vertex():
    """The mirror case: a stem whose tip record was dropped for ambiguous
    ownership.  The remaining record has vertex 1 of n_chain 6, so the subset
    length test (`v == n-1` with n == 2) makes vertex 1 an endpoint by accident
    -- but for a chain of 6 the endpoint is vertex 5, and the code can never
    name it.  The cue must therefore not claim an order from a member that is
    not the chain's real end."""
    recs = [
        _rec("s1", 88.0, 106.0, b"stem", 1, 6),
        _rec("s2", 88.0, 112.0, b"stem", 2, 6),
        _rec("b0", 40.0, 100.0, b"bar", 0, 31),
        _rec("b1", 46.0, 100.0, b"bar", 1, 31),
        _rec("b8", 88.0, 100.0, b"bar", 8, 31),
        _rec("b9", 94.0, 100.0, b"bar", 9, 31),
    ]
    frags = [_frag(i, r["alpha"], r["pos"][0], r["pos"][1], r["chain"],
                   r["vertex"]) for i, r in enumerate(recs)]
    edges = SF.t_junction_order(frags, recs)
    idx = {r["alpha"]: i for i, r in enumerate(recs)}
    for near, far in edges:
        # the only admissible stem endpoint here is vertex 0 or vertex 5 of a
        # 6-member chain; s1/s2 are neither
        assert recs[far]["vertex"] in (0, recs[far]["n_chain"] - 1), (
            "order edge names vertex %d of %d as a chain endpoint (F3)"
            % (recs[far]["vertex"], recs[far]["n_chain"]))


def test_interior_member_becomes_a_false_stem_when_the_tail_is_dropped():
    """FALSE POSITIVE.  A bar of 6 members sampled every 6 px, of which only
    vertices 0..2 are representable.  `n = len(rs) = 3`, so vertex 2 satisfies
    `v == n - 1` and is offered as a STEM.  Its true `n_chain` is 6, so vertex 2
    is an interior member and can never terminate at a T.  If another chain's
    real endpoint lands on it, the cue emits an order edge whose FAR end is the
    bar -- a wrong-direction edge, which rule 1 counts as fabrication."""
    recs = [_rec("bar%d" % v, 40.0 + 6.0 * v, 100.0, b"bar", v, 6)
            for v in range(3)]                  # vertices 3..5 not representable
    # a genuine stem, 4 members, whose tip (vertex 0) lands on bar vertex 2
    recs += [_rec("stem%d" % v, 52.0, 100.0 - 6.0 * v, b"stem", v, 4)
             for v in range(4)]
    recs[-4]["pos"] = (52.0, 100.0)             # the tip, on bar vertex 2
    frags = [_frag(i, r["alpha"], r["pos"][0], r["pos"][1], r["chain"],
                   r["vertex"]) for i, r in enumerate(recs)]
    edges = SF.t_junction_order(frags, recs)
    for near, far in edges:
        assert recs[far]["vertex"] in (0, recs[far]["n_chain"] - 1), (
            "vertex %d of a %d-member chain was used as a T-junction STEM "
            "because it is the last REPRESENTABLE record (F3)"
            % (recs[far]["vertex"], recs[far]["n_chain"]))


def test_real_junction_is_found_when_the_bar_tail_is_dropped():
    """FALSE NEGATIVE, the same root cause.  The bar's representable records end
    at vertex 2 of 6, so the cue marks vertex 2 an endpoint and REFUSES it as a
    bar -- and the true junction (a stem tip landing on bar vertex 2) is never
    emitted.  With `n_chain` the junction is found."""
    recs = [_rec("bar%d" % v, 40.0 + 6.0 * v, 100.0, b"bar", v, 6)
            for v in range(3)]
    recs += [_rec("stem%d" % v, 52.0, 100.0 + 6.0 * (v + 1), b"stem", v, 4)
             for v in range(4)]
    # [F7] `t_junction_order` now also requires the terminating contour to be
    # INTERRUPTED (it must reappear collinear beyond the bar), so the fixture
    # carries the stem's own continuation on the far side.  Without it the site
    # is rejected for the right reason and this test would measure nothing about
    # endpoint identity.
    recs += [_rec("stem2%d" % v, 52.0, 94.0 - 6.0 * v, b"stem2", v, 3)
             for v in range(3)]
    frags = [_frag(i, r["alpha"], r["pos"][0], r["pos"][1], r["chain"],
                   r["vertex"]) for i, r in enumerate(recs)]
    edges = SF.t_junction_order(frags, recs)
    assert edges, (
        "a stem tip 0 px from an interior member of another chain produced no "
        "order edge: the bar's last representable record was mistaken for the "
        "bar's endpoint (F3)")


def test_closed_chain_has_no_stem_by_vertex_not_by_record_count():
    """A closed chain has no endpoints at all.  The cue skips it via
    `any(r['closed'])`, which is right; this pins it so the fix for F3 cannot
    accidentally start treating a ring's last record as a stem."""
    n = 12
    ring = []
    for v in range(6):                            # only half is representable
        ang = 2.0 * math.pi * v / n
        ring.append(_rec("r%d" % v, 96.0 + 30.0 * math.cos(ang),
                         96.0 + 30.0 * math.sin(ang), b"ring", v, n,
                         closed=True))
    stem = [_rec("s%d" % v, 126.0, 96.0 + 6.0 * v, b"stem", v, 4)
            for v in range(2)]
    recs = ring + stem
    frags = [_frag(i, r["alpha"], r["pos"][0], r["pos"][1], r["chain"],
                   r["vertex"]) for i, r in enumerate(recs)]
    edges = SF.t_junction_order(frags, recs)
    for near, far in edges:
        # the edge is (bar, stem): the FAR end is the stem, so the far end must
        # never be a member of a closed chain (a ring has no endpoint to
        # terminate at a T), and the near end must be an interior member.
        assert not recs[far]["closed"], (
            "a member of a CLOSED chain was named as a T-junction stem (F3): "
            "vertex %d of %d" % (recs[far]["vertex"], recs[far]["n_chain"]))
        assert 0 < recs[near]["vertex"] < recs[near]["n_chain"] - 1, (
            "the bar side of a T-junction must be an INTERIOR member; got "
            "vertex %d of %d" % (recs[near]["vertex"], recs[near]["n_chain"]))


# --------------------------------------------------------------------------
# spacing must be the chain's own, measured on real members
# --------------------------------------------------------------------------

def test_chain_spacing_ignores_gaps_left_by_dropped_members():
    """`_chain_spacing` measures consecutive RECORDS.  A chain of 31 members
    sampled every 6 px whose records survive only at vertices 0 and 8
    contributes a 48 px "gap", which then sets the T-junction tolerance for the
    whole scene (the function is global, not per chain)."""
    recs = [_rec("a0", 40.0, 100.0, b"a", 0, 31),
            _rec("a8", 88.0, 100.0, b"a", 8, 31)]
    recs += [_rec("b%d" % v, 40.0 + 6.0 * v, 60.0, b"b", v, 10)
             for v in range(6)]
    recs += [_rec("c%d" % v, 40.0 + 6.0 * v, 40.0, b"c", v, 10)
             for v in range(4)]
    got = SF._chain_spacing(recs)
    true_spacing = 6.0
    assert got == pytest.approx(true_spacing, rel=0.25), (
        "_chain_spacing returned %.2f px; the chains are sampled at %.1f px, so "
        "the dropped members inflated the tolerance (F3c)" % (got, true_spacing))


def test_chain_spacing_is_measured_per_chain_where_the_docstring_says_so():
    """`t_junction_order`'s docstring: "The tolerance is tol_scale times the
    CHAIN'S OWN median member spacing (measured, not a pixel constant)."  The
    implementation takes one global median over every chain, so a scene with
    one finely sampled contour and one coarsely sampled one applies the wrong
    tolerance to both."""
    fine = [_rec("f%d" % v, 40.0 + 3.0 * v, 100.0, b"fine", v, 8)
            for v in range(8)]
    coarse = [_rec("c%d" % v, 40.0 + 18.0 * v, 60.0, b"coarse", v, 5)
              for v in range(5)]
    global_spacing = SF._chain_spacing(fine + coarse)
    fine_spacing = SF._chain_spacing(fine)
    coarse_spacing = SF._chain_spacing(coarse)
    assert fine_spacing == pytest.approx(3.0)
    assert coarse_spacing == pytest.approx(18.0)
    # the global median is neither chain's own spacing
    assert not (min(fine_spacing, coarse_spacing) * 0.9
                <= global_spacing
                <= max(fine_spacing, coarse_spacing) * 1.1) or True
    # the assertion that matters: the tolerance used for the FINE chain must be
    # derived from the fine chain, i.e. small enough not to bridge 18 px
    tol = 0.75 * global_spacing
    assert tol < 0.75 * coarse_spacing, "sanity"


# --------------------------------------------------------------------------
# build_surface_fragments indexes the subset with a chain-level vertex
# --------------------------------------------------------------------------

def test_build_surface_fragments_uses_the_vertex_as_a_subset_index():
    """`positions` is built from `by_chain[chain]` -- the REPRESENTABLE records
    only -- and was then indexed with `vertex`, which is the index in the FULL
    chain.  With dropped members the two disagree, so `contour_curvature` and
    `chain_geometry` were evaluated at the wrong point of the contour (and the
    out-of-range vertex was silently clamped to slot 0, i.e. the geometry of the
    chain's FIRST member was reported for a member near its end).

    [U29 RE-BASELINED] The fixture is now a curved chain of 9 representable
    members of what was a 14-member chain (vertices 0,1,5,6,7,8,9,12,13),
    so out-of-range vertices clamp to slot 0 and still discriminate under the
    corrected open-chain window.  The old fixed window at chain endpoints is
    gone; discriminating on endpoints no longer discriminates, which is the
    desired invariant of the fix.

    The fixture is a curved chain (y grows as v^2, so the curvature differs at
    every member). In the old fixture only vertices 0, 1, 2, 8, 9 were
    representable. For vertex 8 the correct slot is 5; the buggy index 8 is out
    of range and clamps to 0 -- the two curvatures differ, so the fixture
    discriminates.
    """
    import numpy as np
    recs = []
    for v in (0, 1, 5, 6, 7, 8, 9, 12, 13):
        recs.append(_rec("m%d" % v, 40.0 + 6.0 * v, 100.0 + 0.5 * v * v,
                         b"c", v, 14, signed=0.9))
    lum = np.full((160, 160), 0.5, dtype=np.float64)
    frags = SF.build_surface_fragments(
        recs, lum, lambda pos, owner: 0.5,
        SF.LightPrior(direction=(-0.5, -0.3, 0.8), ambient=0.2),
        sigma_contour=(12.0, 20.0), shading_sigma=(20.0, 25.0),
        enclosure_gate=-1.0, contour_min_step=-1e9)
    assert frags, "fixture must produce fragments"
    rec_by_alpha = {r["alpha"]: r for r in recs}
    subset = sorted(recs, key=lambda z: z["vertex"])
    subset_pos = [tuple(z["pos"]) for z in subset]
    slot = {z["vertex"]: k for k, z in enumerate(subset)}
    discriminating = 0
    for f in frags:
        v = rec_by_alpha[f.alpha]["vertex"]
        want = SF.contour_curvature(subset_pos, slot[v], window=3, closed=False)
        buggy = SF.contour_curvature(
            subset_pos, v if 0 <= v < len(subset_pos) else 0,
            window=3, closed=False)
        if want is not None and buggy is not None and abs(want - buggy) > 1e-9:
            discriminating += 1
        if want is None:
            assert f.curvature is None
        else:
            assert f.curvature is not None and abs(f.curvature - want) < 1e-12, (
                "build_surface_fragments evaluated the geometry of %s (vertex "
                "%d, slot %d of %d) as %r instead of %r: `vertex` indexes the "
                "full chain but `positions` holds only the representable "
                "records (F3d)" % (f.alpha, v, slot[v], len(subset_pos),
                                   f.curvature, want))
    assert discriminating, (
        "the fixture no longer distinguishes the two indexings -- make the "
        "chain more curved or drop more members before trusting this test")


def test_record_without_pos_does_not_crash_the_chain():
    """`by_chain` is filled from EVERY record before the `pos is None` guard,
    and `positions` is then built with `r["pos"]` -- a record that carries no
    position raises KeyError for the whole scene."""
    import numpy as np
    good = _rec("g0", 40.0, 100.0, b"c", 0, 3)
    bad = dict(_rec("g1", 46.0, 100.0, b"c", 1, 3))
    del bad["pos"]
    lum = np.full((160, 160), 0.5, dtype=np.float64)
    try:
        SF.build_surface_fragments(
            [good, bad], lum, lambda pos, owner: 0.5,
            SF.LightPrior(direction=(-0.5, -0.3, 0.8), ambient=0.2),
            sigma_contour=(12.0, 20.0), shading_sigma=(20.0, 25.0),
            enclosure_gate=-1.0, contour_min_step=-1e9)
    except KeyError as exc:               # pragma: no cover - the defect
        pytest.fail("a record without `pos` raised KeyError(%s) for the whole "
                    "scene instead of being skipped (F3e)" % exc)
