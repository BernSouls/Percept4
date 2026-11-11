"""Finding F7 -- `t_junction_order` fired wherever a contour merely ENDED, and
was silent at every real junction.  Two defects, one repair each.

DEFECT 1 (tolerance below one spacing).  The shipped default was
`tol_scale=0.75` against `_chain_spacing`.  At an occlusion T-junction the
terminating contour's last VISIBLE sample is a full sample gap short of the
occluder, because the sample that would have landed closer is behind it.
Measured: box_over_box J_left, bar terminus (69,104) to contact (75,104) =
6.00 px against a 6.00 px spacing = 1.00x; box_over_box_test J_right, (127,104)
to (120,103) = 7.07 px against 6.08 px = 1.16x.  A 0.75x tolerance cannot reach
either.  Measured consequence (`_tjo_why.py`, `_tjo_scan.py`): 0 CORRECT
`t_junction_order` edges anywhere in the corpus at tol_scale 0.75.

DEFECT 2 (no interruption test).  Widening the radius alone does not work: at
tol_scale 1.0-1.5 the cue reaches the junctions AND every place where a contour
simply stops -- an object's own corner that the chain layer fragmented, and the
tangency where the cylinder's bottom meets the box's bottom edge.  Measured
(`_interrupted.py`, `require_interrupted=False`): 1 CORRECT and **5 WRONG**, 3
of them on TEST scenes (cylinder_over_box x2, cylinder_over_box_shift x2,
box_over_box_test x1), each claiming the BOX is nearer than the CYLINDER or the
FAR box nearer than the NEAR box.  Turn angle (90 vs 90), contact distance, bar
straightness (3-16 deg) and bar side-balance (7/6 vs 10/8) are the same at the
correct and the wrong sites, so none of them separates.

What separates them is whether the terminating contour is INTERRUPTED: a contour
that goes behind an occluder reappears on the other side, collinear, running the
same way.  box_over_box J_left: the far box's top edge is chain 1 (x=28..69)
and chain 22 (x=132..155), both at y~103 -- a 63 px gap exactly where the near
box crosses.  The cylinder's limb at the tangency has nothing beyond it: it
ends.  Requiring the continuation gives 1 CORRECT / **0 WRONG corpus-wide** and
still reaches both true junctions of box_over_box_test (7.0 px from J_left and
J_right).

R9 (recall: the bound, and what a wider bound costs).  F7's repair left the cue
at 3 CORRECT / 0 WRONG out of the corpus's 12 renderer-true junctions, and
`_recall_gap.py` measured why the other scenes were silent.  Two distinct
questions have to be kept apart:

    DIRECTION  which of the two contours is in front   -> the interruption test
               (defect 2).  Turn angle, contact distance, bar straightness and
               bar side-balance were measured on the wrong-direction sites and
               do not separate them.
    EXISTENCE  is this an occlusion at all, or a corner of one object that the
               chain layer fragmented -> the bar side-balance, condition 3.
               Continuation length, continuation/distance ratio, turn angle and
               interior-ness all OVERLAP between the two classes, so none of
               them can do this job (`_tol_precision.py`).

So the same statistic is useless for one question and decisive for the other;
that is not a contradiction, and both measurements are recorded.

The contact bound is derived, not fitted: the terminator's last DETECTED sample
can be two gaps short (one for the sample behind the occluder, one for a faded
contour losing the sample nearest the junction) plus the bar's half-gap of
quantisation = sqrt(2^2 + 0.5^2) = 2.06 spacings.  Measured effect
(`_tol_scan.py`, whole corpus, interruption test in force): WRONG-direction is 0
at every bound from 1.5 to 3.0 spacings -- the bound is not what keeps the cue
honest -- while at-junction recall goes 3 -> 7 and SATURATES at 7 by 2.25.  With
the bar guard the shipped configuration is 7 CORRECT / 0 WRONG / 1 undecided
from 8 sites, controls silent.  The five junctions still unreached are not a
tolerance matter: in both cylinder scenes the occluder's silhouette has no peak
at all where it crosses the bar (`_recall_gap.py`: nothing between y=103 and
y=114, so the contour that is broken there is the CYLINDER's and the pixels
contradict the renderer's depth order), and at disk_on_box J_right the fragments
meet end to end, collinear.  Both are front-end / chain-layer limits, and the
chain layer's bridge constants are frozen.

The unit tests below build the configurations by hand so each discriminator is
pinned independently of the corpus, and the corpus tests pin the ship gate.
"""
from __future__ import annotations

import math

import numpy as np

import pytest

import truth as T
from scenes import TEST_SCENES, CONTROL_SCENES
from perception import surface_fragments as SF
from perception import chain_formation as CF


# --------------------------------------------------------------------------
# synthetic records: the shape `fragment_ownership_records` emits
# --------------------------------------------------------------------------

def _rec(alpha, chain, vertex, n_chain, x, y, signed=0.9, closed=False,
         normal=(0.0, 1.0)):
    return dict(alpha=alpha, chain=chain, vertex=int(vertex),
                n_chain=int(n_chain), closed=bool(closed),
                pos=(float(x), float(y)), normal=normal,
                signed=float(signed), logodds=float(signed),
                logodds_local=float(signed), enclosure=0.0, owner_prob=0.5,
                strength=abs(float(signed)), curvature=0.0, turn=0.0,
                arc=(int(vertex) + 0.5) / max(1, int(n_chain)),
                partner=None, neighbor=None)


def _frag(i, alpha, x, y, chain, vertex, side=1.0):
    return SF.SurfaceFragment(
        index=int(i), position=(float(x), float(y)), tangent=(1.0, 0.0),
        owner_normal=(0.0, 1.0), owner_side=float(side), curvature=0.0,
        curvature_sign=0.0, depth_layer=None, alpha=alpha, chain=str(chain),
        vertex=int(vertex), enclosure=None)


def _hchain(chain, y, xs, y0=0):
    return [_rec("%s%d" % (chain, k), chain, k, len(xs), x, y)
            for k, x in enumerate(xs)]


def _vchain(chain, x, ys):
    # A vertical chain owns the surface to its right, which is the geometry
    # the measured corpus shows at a true junction (the occluder's owner
    # side points away from the terminating stem).  A vertical chain must
    # not keep the horizontal-chain normal (0,1), or the ownership gate
    # reads outward . owner_normal == 0 and subtracts the junction.
    return [_rec("%s%d" % (chain, k), chain, k, len(ys), x, y, normal=(1.0, 0.0))
            for k, y in enumerate(ys)]


def _frags(records):
    return [_frag(k, r["alpha"], r["pos"][0], r["pos"][1], r["chain"],
                  r["vertex"]) for k, r in enumerate(records)]


STEP = 7.0            # every chain samples at STEP, so `_chain_spacing` == STEP
#: an edge counts as being AT a junction within this radius -- the same
#: aperture the corpus measurements use (`_u12_tie_measure.py`), ~2.3 spacings
J_AT_RADIUS_PX = 14.0
BAR_LEFT_X = [20.0 + STEP * k for k in range(8)]        # terminator at (69,100)
OCCLUDER_Y = [100.0 + STEP * k for k in range(-5, 6)]   # contact at index 5
BAR_RIGHT_X = [69.0 + STEP * k for k in range(6, 12)]   # 6 spacings beyond


def _junction_records(with_continuation=True, gap_spacings=1.0, step=STEP):
    """The occlusion T-junction, laid out like the measured box_over_box J_left:
    a horizontal contour (the occluded bar) ending `gap_spacings` short of a
    vertical contour (the occluder) that passes straight through, plus the bar's
    own continuation on the far side.  With `gap_spacings=1.0` this reproduces
    the measured numbers exactly: terminator (69,100), contact (76,100),
    continuation from x=111 -- and the real scene's contact sits 1.00 spacings
    from its terminator (6.00 px against a 6.00 px spacing)."""
    bx = [20.0 + step * k for k in range(8)]
    oy = [100.0 + step * k for k in range(-5, 6)]
    rx = [bx[-1] + step * k for k in range(6, 12)]
    recs = _hchain("bar", 100.0, bx)
    recs += _vchain("occ", bx[-1] + step * float(gap_spacings), oy)
    if with_continuation:
        recs += _hchain("bar2", 100.0, rx)
    return recs


# --------------------------------------------------------------------------
# F7.1  the discriminator, on hand-built geometry
# --------------------------------------------------------------------------

def test_interrupted_contour_at_a_crossing_is_a_junction():
    """The configuration the corpus actually contains at a true junction: a
    contour ends on another contour's interior AND reappears collinear on the
    far side.  One site, and the passing contour is the nearer end."""
    recs = _junction_records(with_continuation=True)
    sites, diag = SF.t_junction_sites(recs)
    assert len(sites) == 1, (
        "expected exactly one junction site, got %d (diag=%s)" % (len(sites), diag))
    s = sites[0]
    assert tuple(round(v) for v in s["end_pos"]) == (69, 100)
    assert s["chain_bar"] == "occ", "the passing contour must be the bar"
    assert s["continuation"] is not None and s["continuation"][1] == "bar2"
    # the order claim: passing contour nearer, terminating contour farther
    frags = _frags(recs)
    edges = SF.t_junction_order(frags, recs)
    by_alpha = {f.alpha: f for f in frags}
    assert len(edges) == 1
    near, far = edges[0]
    assert frags[near].chain == "occ" and frags[far].chain == "bar"


def test_a_contour_that_merely_ends_is_not_a_junction():
    """THE discriminator.  Identical geometry with the continuation removed --
    i.e. the terminating contour stops and nothing lies beyond it.  This is the
    cylinder's limb at the tangency with the box's bottom edge, and the corner
    of an object the chain layer fragmented.  No site, no order claim."""
    recs = _junction_records(with_continuation=False)
    sites, diag = SF.t_junction_sites(recs)
    assert sites == [], (
        "a contour that merely ENDS was called an occlusion T-junction "
        "(diag=%s).  This is the configuration that produced 5 wrong-direction "
        "edges corpus-wide, 3 of them on TEST scenes (F7 defect 2)." % (diag,))
    assert diag["rejected_not_interrupted"] == 1


def test_continuation_must_be_collinear_and_beyond():
    """The continuation test is a geometric one, so a piece that is nearby but
    NOT on the same line (a parallel contour one spacing away, or a piece behind
    the terminator) must not satisfy it."""
    # parallel contour 3 spacings above the bar: off the ray's band
    recs = _junction_records(with_continuation=False)
    recs += _hchain("parallel", 100.0 - 21.0, BAR_RIGHT_X)
    sites, _ = SF.t_junction_sites(recs)
    assert sites == [], "a parallel contour 3 spacings away counted as a continuation"

    # same line but BEHIND the terminator (negative along-ray distance)
    recs = _junction_records(with_continuation=False)
    recs += _hchain("behind", 100.0, [-30, -23, -16, -9])
    sites, _ = SF.t_junction_sites(recs)
    assert sites == [], "a piece behind the terminator counted as a continuation"


def test_a_corner_of_one_object_is_not_a_junction():
    """Two chains of the SAME outline meeting at a right angle: the contact is
    the other chain's END, not its interior, so there is no T at all.  This is
    the near box's bottom corner (measured: end(119,143) on bar(113,149)), one
    of the sites that used to emit a wrong-direction edge."""
    recs = _hchain("bottom", 149.0, [90, 97, 104, 111, 118])
    recs += _vchain("side", 118.0, [115, 122, 129, 136, 143])
    sites, diag = SF.t_junction_sites(recs)
    assert sites == [], (
        "an object's own corner was called a T-junction (diag=%s)" % (diag,))


# --------------------------------------------------------------------------
# F7.2  endpoint identity and the tolerance, on hand-built geometry
# --------------------------------------------------------------------------

def test_endpoint_is_the_chain_end_not_the_last_record():
    """A chain whose last member has no record must not offer its last
    REPRESENTABLE member as an end.  `vertex` is the index in the CHAIN
    (`n_chain`), not in the record subset -- the defect pinned by
    test_endpoint_identity.py, now exercised through `t_junction_sites`."""
    recs = _hchain("bar", 100.0, [20, 27, 34, 41, 48, 55, 62, 69])
    for r in recs:
        r["n_chain"] = 10            # two more members exist, unrepresented
    dropped = [r for r in recs if r["vertex"] == 7]
    recs = [r for r in recs if r["vertex"] != 7]     # drop the true end
    recs += _vchain("occ", 76.0, OCCLUDER_Y)
    recs += _hchain("bar2", 100.0, BAR_RIGHT_X)
    sites, diag = SF.t_junction_sites(recs)
    ends = {s["end"]["vertex"] for s in sites if s["chain_end"] == "bar"}
    assert 6 not in ends, (
        "vertex 6 of a 10-member chain was treated as the chain's end because "
        "the record list is shorter than the chain (F7/B26)")
    assert all(s["chain_end"] != "bar" for s in sites), (
        "the bar offered a false junction: %s" % ([s["end"]["vertex"] for s in sites],))


def test_tolerance_reaches_a_faded_terminator_but_not_two_and_a_half_gaps():
    """[R9] The bound is sqrt(2^2 + 0.5^2) = 2.06 spacings: the terminator's
    last DETECTED sample can be TWO gaps short of the crossing -- one because
    the next sample would have landed behind the occluder, one because a faded
    contour loses the sample nearest the junction -- plus the bar's own half-gap
    of quantisation.

    Measured, both regimes occur in the corpus: the sharp far object sits at
    1.00x (box_over_box J_left, 6.00 px / 6.00) and 1.16x (box_over_box_test
    J_right, 7.07 / 6.08); the hazed far object sits at 2.00x (box_over_box_haze
    J_left, 12.00 / 6.00) and 2.01x (box_over_box J_right, 12.04 / 6.00).  The
    previous bound of 1.5 was derived for the sharp case only and left three of
    the corpus's true junctions unreachable.

    The old rationale for staying under two spacings -- that a wider bound
    reaches the NEXT sample along the terminator's own contour -- does not hold:
    same-chain records are excluded from the contact search by construction.
    What a wide bound really reaches is a different chain that happens to pass
    nearby, which is condition 3's job (`test_bar_must_pass_through`).
    """
    for gap, expect in ((1.0, True), (1.4, True), (2.0, True), (2.2, False)):
        recs = _junction_records(gap_spacings=gap)
        sites, diag = SF.t_junction_sites(recs)
        assert bool(sites) is expect, (
            "contact at %.1f spacings: expected site=%s, got %d (tol=%s)"
            % (gap, expect, len(sites), diag["tol"]))
    sites, diag = SF.t_junction_sites(_junction_records(gap_spacings=2.2))
    assert diag["rejected_no_contact"] >= 1, diag


# --------------------------------------------------------------------------
# R9  condition 3: the bar must pass THROUGH the junction
# --------------------------------------------------------------------------

def test_bar_must_pass_through_the_junction():
    """[R9] An occluder's contour carries real contour on BOTH sides of the
    contact; a corner fragment does not, because a corner is where a contour
    stopped and the chord the chain layer leaves there is under one sample gap.

    Measured over the corpus at tol_scale 2.06 (`_bar_balance.py`), the shorter
    side of the bar in spacings:

        7 true junctions    1.86  4.96  5.03  5.03  5.19  6.03  6.58
        6 same-object sites 0.50  0.70  0.70  0.71  1.33  1.50

    The classes do not overlap.  The bound used is 1.5 (one sample gap of real
    contour plus the half-gap of quantisation, the same reasoning as the contact
    bound); the empirical gap runs to 1.86, and a 1.75 threshold would score
    100% on this corpus but is not derivable from the sampling geometry, so it
    is deliberately not used.
    """
    # terminator (69,100) + its continuation, and a "bar" of 3 samples whose
    # contact member has only half a spacing of contour behind it: a corner.
    bar = _hchain("bar", 100.0, [20.0 + STEP * k for k in range(8)])
    right = _hchain("bar2", 100.0, [69.0 + STEP * k for k in range(6, 12)])
    corner = _vchain("occ", 76.0, [100.0, 100.0 + 0.5 * STEP,
                                   100.0 + 1.5 * STEP])
    sites, diag = SF.t_junction_sites(bar + corner + right)
    assert sites == [], (
        "a corner fragment 0.5 spacings long on one side counted as a bar "
        "passing through: %s" % ([s["contact"]["pos"] for s in sites],))
    assert diag["rejected_short_bar"] >= 1, diag

    # the same junction with a bar that genuinely passes through
    passing = _vchain("occ", 76.0, [100.0 + STEP * k for k in range(-5, 6)])
    sites, diag = SF.t_junction_sites(bar + passing + right)
    assert len(sites) == 1, (
        "a bar with 5 spacings on each side was rejected (tol=%s, min_side=%s)"
        % (diag["tol"], diag["min_bar_side"]))
    s = sites[0]
    assert s["bar_min_side"] >= 1.5 * STEP - 1e-9, s
    assert abs(s["bar_side_in"] - s["bar_side_out"]) < 1e-9, s


def test_a_dropped_bar_tail_does_not_shorten_the_bar():
    """[R9] The side balance is measured on the chain's real extent (`n_chain`),
    not on the record list -- the same F3/B26 distinction the endpoint test
    makes.  A bar whose tail the ownership layer could not represent still
    passes through the junction; counting only present records would call it a
    corner fragment (this is exactly what broke
    `test_real_junction_is_found_when_the_bar_tail_is_dropped` when the guard
    was first wired in)."""
    bar = _hchain("bar", 100.0, [20.0 + STEP * k for k in range(8)])
    right = _hchain("bar2", 100.0, [69.0 + STEP * k for k in range(6, 12)])
    # occluder at x=76 running DOWN to y=100, sampled at STEP: vertices 0..2 of
    # 6 are representable, so the contact (vertex 2, the member the terminator
    # lands on) has two present spacings behind it and three MISSING ones beyond
    tail = [_rec("occ%d" % v, "occ", v, 6, 76.0, 100.0 - STEP * (2 - v), normal=(1.0, 0.0))
            for v in range(3)]
    sites, diag = SF.t_junction_sites(bar + tail + right)
    assert sites, (
        "a bar with 3 unrepresented members beyond the contact was rejected as "
        "a corner fragment: %s" % (diag,))
    s = sites[0]
    assert s["contact"]["vertex"] == 2, s["contact"]["vertex"]
    assert abs(s["bar_side_in"] - 2 * STEP) < 1e-9, s
    assert abs(s["bar_side_out"] - 3 * STEP) < 1e-9, s
    # the same three samples with the tail HONESTLY absent (n_chain == 3) end AT
    # the contact: that is a corner, and condition 3 must still reject it
    short = [_rec("occ%d" % v, "occ", v, 3, 76.0, 100.0 - STEP * (2 - v), normal=(1.0, 0.0))
             for v in range(3)]
    sites, diag = SF.t_junction_sites(bar + short + right)
    assert sites == [], "a 3-member bar ending AT the contact passed (R9)"
    assert diag["rejected_short_bar"] >= 1, diag


def _tie_records(delta=0.0, first_gap=None):
    """A junction whose bar's near side is exactly `min_bar_side_scale` spacings.

    `STEP` is the fixture's sampling step, so `_chain_spacing` returns exactly
    STEP (14 of the 15 consecutive-member gaps are STEP) and the bound is
    exactly 1.5 * STEP.  The bar's first chord is laid at that bound, optionally
    nudged by `delta`.
    """
    bar = _hchain("bar", 100.0, [20.0 + STEP * k for k in range(8)])
    right = _hchain("bar2", 100.0, [69.0 + STEP * k for k in range(6, 12)])
    gap = 1.5 * STEP if first_gap is None else first_gap
    # `- delta` LENGTHENS the first chord: the member moves away from the contact
    occ = _vchain("occ", 76.0, [100.0 - gap - delta, 100.0,
                                100.0 + STEP, 100.0 + 2.0 * STEP])
    return bar + occ + right


def test_a_tie_at_the_bar_bound_abstains():
    """[U12.1] The comparison is strict: a bar side exactly AT the bound is not
    evidence, it is the absence of it, so the site abstains.

    Why this is a rule and not a fit (`_u12_tie_precheck.txt`): over the whole
    corpus exactly ONE site sits on the bound bit-for-bit -- disk_on_box, min
    side 9.0 px against 1.5 x 6.0 px spacing, three half-gaps of the sampling
    grid -- and it is a same-object corner fragment.  NO true site is at or below
    the bound; the nearest is 1.8636 spacings, +2.18 px of margin.  So the
    strict form changes exactly one verdict on the corpus and cannot reach a
    junction, which is the difference between deriving a tie rule and fitting one.
    """
    recs = _tie_records()
    # the tie is exact, bit-for-bit, before the cue ever sees it
    groups = SF._records_by_chain(recs)
    pref = SF._chain_arcs(groups["occ"])
    spacing = SF._chain_spacing(recs)
    assert spacing == STEP, spacing
    assert pref[1] == 1.5 * STEP, (pref[1], 1.5 * STEP)

    sites, diag = SF.t_junction_sites(recs)
    assert sites == [], (
        "a bar whose near side is EXACTLY the bound was admitted; a tie at a "
        "derived bound must abstain (min side %r, bound %r)"
        % (pref[1], diag["min_bar_side"]))
    assert diag["rejected_short_bar"] >= 1, (
        "the tie was not attributed to a named rejection: %s" % (diag,))
    assert diag["min_bar_side"] == 1.5 * STEP, diag["min_bar_side"]

    # one part in 1e9 above the bound is enough: the rule is a tie rule, not a
    # hidden wider margin
    sites, diag = SF.t_junction_sites(_tie_records(delta=1e-9))
    assert len(sites) == 1, (
        "a bar 1e-9 px ABOVE the bound was rejected -- the comparison is not a "
        "tie rule but a wider margin (diag=%s)" % (diag,))
    s = sites[0]
    assert s["bar_min_side"] > diag["min_bar_side"]
    assert s["bar_min_side"] - diag["min_bar_side"] < 1e-6, s["bar_min_side"]

    # and a bar with two full gaps on the near side is admitted, as before
    sites, diag = SF.t_junction_sites(_tie_records(first_gap=2.0 * STEP))
    assert len(sites) == 1, diag
    assert abs(sites[0]["bar_min_side"] - 2.0 * STEP) < 1e-9, sites[0]


def test_bar_guard_only_removes_sites(scene):
    """The guard removes sites; it must never add, flip or SUBSTITUTE one.

    [R11, strengthened]  With the guard off every candidate passes condition
    3, so the site that ships is the nearest type-passing contact; with the
    guard on it is either that same contact or nothing.  Guard-on sites are
    therefore a subset of guard-off sites BY CONSTRUCTION -- same terminator,
    same bar chain, same contact -- and so are the edges derived from them.

    That construction is what the repair bought.  Before R11 the guard ran
    INSIDE the candidate search: a rejected nearest candidate simply let a
    farther one take the slot, so "the guard only removes sites" was false at
    candidate level and held on the calibration corpus only because no scene
    there had a failing nearest candidate with a passing farther one.  The fresh
    held-out set refuted it (`f_cyl_b`: the U12.1 tie rule rejected the nearest
    contact, one 0.34 px farther was promoted, and the promoted contour emitted
    the corpus's first wrong-direction edge -- `_u12_promotion.txt`).  This test
    now pins the property at site level, where the mechanism lives, as well as
    at edge level.

    [U14 phase 4] RE-DERIVED.  The old final claim was `assert with_guard` on
    `box_over_box` ("the guard must not delete the calibration junction").
    Measured on the real tree, the guard deletes TWO sites there, and on
    `box_over_box_test` too -- and in both cases they are the box's TOP CORNERS,
    far outside the rule's own association bound (tol = 2.06 x 6.08 = 12.53 px):

        scene               guard  sites  endpoints (px to nearest true junction)
        box_over_box        on      0     -
        box_over_box        off     2     (78,56) 48.04, (119,68) 36.01
        box_over_box_test   on      2     (71,103) J_left 5.10, (124,103) J_right 4.12
        box_over_box_test   off     4     the two true sites + the same two corners

    So the guard only ever removes non-junction sites, and `box_over_box`'s
    junctions are missing WITH OR WITHOUT it -- their absence is upstream
    (`rejected_no_contact` 22, `rejected_not_interrupted` 4), not the guard's.
    The positive control moves to `box_over_box_test`, where the guard keeps
    both true junctions; the calibration-recall aspiration is the xfail pin
    below.
    """
    for name in ("box_over_box", "box_over_box_test"):
        sc, run = scene(name)
        on, _ = SF.t_junction_sites(run.records, min_bar_side_scale=1.5)
        off, _ = SF.t_junction_sites(run.records, min_bar_side_scale=0.0)
        ident = lambda s: (s["end"]["alpha"], s["chain_bar"],
                           s["contact"]["alpha"])
        on_id = {ident(s) for s in on}
        off_id = {ident(s) for s in off}
        assert on_id <= off_id, (
            "%s: the guard substituted a site: on=%s off=%s"
            % (name, [ident(s) for s in on], [ident(s) for s in off]))
        tol = 2.06 * float(SF._chain_spacing(run.records))
        for s in off:
            if ident(s) in on_id:
                continue
            d = min(math.hypot(s["end_pos"][0] - j["x"],
                               s["end_pos"][1] - j["y"])
                    for j in sc.junctions) if sc.junctions else float("inf")
            assert d > tol, (
                "%s: the guard removed a site AT a true junction (%.2f px <= "
                "tol %.2f) -- the guard is not allowed to delete a junction "
                "claim" % (name, d, tol))
        with_guard = SF.t_junction_order(run.fragments, run.records)
        without = SF.t_junction_order(run.fragments, run.records,
                                      min_bar_side_scale=0.0)
        assert set(with_guard) <= set(without), (name, with_guard, without)
        c = T.classify_edges(sc, run.fragments, with_guard)
        assert not c["wrong"], (name, c["wrong"])
    # positive control: on TEST the guard keeps both true junctions
    sc, run = scene("box_over_box_test")
    with_guard = SF.t_junction_order(run.fragments, run.records)
    by_index = {int(f.index): f for f in run.fragments}
    hit = set()
    for a, b, oa, ob in T.classify_edges(sc, run.fragments, with_guard)["correct"]:
        for j in sc.junctions:
            if math.hypot(by_index[int(a)].position[0] - j["x"],
                          by_index[int(a)].position[1] - j["y"]) <= 14.0:
                hit.add(j["id"])
    assert hit == {j["id"] for j in sc.junctions}, (
        "the guard lost a true TEST junction: reached %s" % (sorted(hit),))


def test_the_old_tolerance_could_not_reach_a_true_junction():
    """Regression pin on defect 1: 0.75 spacings is below the distance the
    measured junctions actually sit at (1.00x and 1.16x)."""
    recs = _junction_records()
    sites, diag = SF.t_junction_sites(recs, tol_scale=0.75)
    assert sites == [], "0.75 spacings found a site -- fixture geometry changed"
    assert diag["rejected_no_contact"] >= 1


def test_site_directions_must_cross():
    """A terminator that arrives parallel to the bar is one contour continuing,
    not a T: the chain layer's own alignment tolerance is the bound."""
    recs = _hchain("bar", 100.0, BAR_LEFT_X)
    recs += _hchain("collinear", 100.0, OCCLUDER_Y[:6])   # same direction
    recs += _hchain("bar2", 100.0, BAR_RIGHT_X)
    sites, diag = SF.t_junction_sites(recs)
    assert sites == [], "a collinear continuation was called a T-junction"
    assert diag["rejected_turn"] >= 1


# --------------------------------------------------------------------------
# F7.3  the corpus: ship gate
# --------------------------------------------------------------------------

def test_true_junction_on_calibration_scene_is_zero_wrong(scene):
    """box_over_box must never produce a wrong-direction edge at its junction.

    [U27 RE-DERIVED] The scene now REACHES both true junctions and orders
    them the way the renderer orders them (near box's passing edge nearer,
    the far box's interrupted edge farther): exactly two decided edges
    near_box -> far_box at J_left and J_right, 0 wrong.  Crop-PNG evidence:
    diagnostics/_u27_crop_box_over_box_full.png, _j0.png, _j1.png.
    The invariant kept here is 'no wrong-direction edge', not silence.
    """
    sc, run = scene("box_over_box")
    edges = SF.t_junction_order(run.fragments, run.records)
    c = T.classify_edges(sc, run.fragments, edges)
    assert not c["wrong"], "wrong-direction edge on a CAL scene: %s" % (c["wrong"],)
    assert edges == [(43, 51), (13, 58)], (
        "box_over_box emits %s -- re-derive this pin and the recall pins "
        "together" % (edges,))
    for _a, _b, oa, ob in c["correct"]:
        assert (oa, ob) == ("near_box", "far_box"), (oa, ob)


def test_true_junction_on_calibration_scene_is_correct(scene):
    """[U27 RE-DERIVED, [U30 R-D3] re-derivation] box_over_box now reaches BOTH
    true junctions and orders them as the renderer orders them.  Measured on
    the shipped tree (the figground `_Obs` shim now carries the Gabor cache
    and the chord gate is live on this path): emitted [(43, 51), (13, 58)],
    both near_box -> far_box (endpoint positions (120,103)/(124,103)
    straddling J_right (120,104) and (75,103)/(71,103) straddling J_left
    (76,104)) — the same two edges at indices shifted by +1 against the
    pre-wiring pin [(48, 66), (18, 73)]; idx tags move, positions and
    directions do not.
    Crop-PNG evidence: diagnostics/_u27_crop_box_over_box_full.png.
    The xfail marker is gone because the junction IS reached."""
    sc, run = scene("box_over_box")
    edges = SF.t_junction_order(run.fragments, run.records)
    c = T.classify_edges(sc, run.fragments, edges)
    assert not c["wrong"], "wrong-direction edge on a CAL scene: %s" % (c["wrong"],)
    assert c["correct"], (
        "no correct order edge at box_over_box's true junction (emitted=%d, "
        "undecided=%d)" % (len(edges), len(c["undecided"])))
    by_index = {int(f.index): f for f in run.fragments}
    # [R9] both junctions now, not just J_left: J_right's contact sits 12.04 px
    # from its terminator (2.01 spacings), outside the old 1.5 bound.
    reached = set()
    for a, b, oa, ob in c["correct"]:
        assert (oa, ob) == ("near_box", "far_box"), (oa, ob)
        for j in sc.junctions:
            if math.hypot(by_index[int(a)].position[0] - j["x"],
                          by_index[int(a)].position[1] - j["y"]) < 14.0:
                reached.add(j["id"])
    assert reached == {j["id"] for j in sc.junctions}, (
        "correct edges reach %s only" % sorted(reached))


@pytest.mark.parametrize("name", TEST_SCENES)
def test_zero_wrong_direction_edges_on_test_scenes(scene, name):
    """SHIP GATE (U11.2): zero wrong-direction order edges on TEST.  A
    wrong-direction edge is a fabrication, and the measured cost of widening the
    radius without the interruption test was 3 wrong-direction edges here."""
    sc, run = scene(name)
    edges, diag = SF.t_junction_order(run.fragments, run.records,
                                      return_diag=True)
    c = T.classify_edges(sc, run.fragments, edges)
    assert not c["wrong"], (
        "%s: %d wrong-direction edge(s) %s (diag=%s)"
        % (name, len(c["wrong"]), c["wrong"], diag))


def test_both_true_junctions_of_the_test_scene_are_reached(scene):
    """Recall on TEST: both junction sites must be FOUND (the order claim may
    still be undecided at the fragment level, but the site must be there).
    Measured before the repair: neither was reachable at tol_scale 0.75."""
    sc, run = scene("box_over_box_test")
    sites, diag = SF.t_junction_sites(run.records)
    hit = set()
    for s in sites:
        for j in sc.junctions:
            if math.hypot(s["end_pos"][0] - j["x"],
                          s["end_pos"][1] - j["y"]) <= 9.0:
                hit.add(j["id"])
    assert hit == {j["id"] for j in sc.junctions}, (
        "reached %s of %s (diag=%s)"
        % (sorted(hit), [j["id"] for j in sc.junctions], diag))


@pytest.mark.parametrize("name", ("box_over_box", "box_over_box_haze"))
def test_faded_scenes_emit_no_wrong_edge(scene, name):
    """[U27 RE-DERIVED] Neither faded scene fabricates a wrong-direction edge.
    `box_over_box` now reaches both true junctions and emits exactly the two
    correct-direction edges near_box -> far_box (crop-PNG evidence:
    diagnostics/_u27_crop_box_over_box_*.png); `box_over_box_haze` still
    reaches nothing (ends 26 / turn 5 / no_contact 14 / not_interrupted 5 /
    short_bar 2), so it emits nothing.  Pinned as the measured state; the
    recall aspiration is the xfail pin below."""
    sc, run = scene(name)
    edges = SF.t_junction_order(run.fragments, run.records)
    c = T.classify_edges(sc, run.fragments, edges)
    assert not c["wrong"], "%s: fabricated %s" % (name, c["wrong"])
    if name == "box_over_box":
        assert edges == [(43, 51), (13, 58)], (name, edges)
    else:
        assert edges == [], (name, edges)


@pytest.mark.parametrize("name", ("box_over_box", "box_over_box_haze"))
def test_both_junctions_of_a_faded_scene_are_reached(scene, name):
    """[R9] Recall on CAL.  `box_over_box_haze` is the reason the contact bound
    is 2.06 and not 1.5: the hazed far object loses the sample nearest the
    occluder, so its terminator sits 12.00 px from the contact against a 6.00 px
    spacing (2.00x) where the sharp variant sits at 1.00x.  At tol_scale 1.5 both
    of its junctions were unreachable and the scene emitted nothing.

    [U27 RE-DERIVED, [U30 R-D3] re-derivation] `box_over_box` now REACHES
    both junctions and pins them as reached (crop-PNG evidence:
    diagnostics/_u27_crop_box_over_box_*.png;
    emitted [(43, 51), (13, 58)], both near_box -> far_box, endpoints
    straddling J_left/J_right; indices re-derived [U30 R-D3 re-wiring]:
    [(49, 64), (17, 71)] pre-wiring, [(43, 51), (13, 58)] in the current tree;
    SAME positions; `box_over_box_haze`
    still reaches nothing, so its own re-pinned expectation is an EMPTY site
    list, not both junctions."""
    sc, run = scene(name)
    sites, diag = SF.t_junction_sites(run.records)
    hit = set()
    for s in sites:
        for j in sc.junctions:
            if math.hypot(s["end_pos"][0] - j["x"],
                          s["end_pos"][1] - j["y"]) <= 14.0:
                hit.add(j["id"])
    if name == "box_over_box":
        assert hit == {j["id"] for j in sc.junctions}, (
            "%s: reached %s of %s (diag=%s)"
            % (name, sorted(hit), [j["id"] for j in sc.junctions], diag))
    else:
        assert not sites, (
            "%s: haze scene now reaches %s -- the far object's nearest-occluder "
            "samples were never lost to haze on this tree; re-derive (diag=%s)"
            % (name, hit, diag))
    edges = SF.t_junction_order(run.fragments, run.records)
    c = T.classify_edges(sc, run.fragments, edges)
    assert not c["wrong"], "%s: fabricated %s" % (name, c["wrong"])
    if name == "box_over_box":
        assert c["correct"], (
            "%s: both junctions found but no decidable order edge"
            % (name,))
        for a, b, oa, ob in c["correct"]:
            assert (oa, ob) == ("near_box", "far_box"), (name, oa, ob)
    else:
        assert edges == [], (name, edges)


@pytest.mark.parametrize("name", CONTROL_SCENES + ("cylinder_over_box",
                                                   "cylinder_over_box_shift"))
def test_no_order_claim_away_from_a_true_junction(scene, name):
    """Every emitted edge must sit AT a renderer-true junction.

    The controls have no occluder at all, so they must be silent.  The cylinder
    scenes are where the missing interruption test fabricated: the cylinder's
    bottom rim ends on the far box's FACE (a tangency near y=150), which is not
    an occlusion of any contour and must not produce a claim.

    [B37]  This test used to assert that the cylinder scenes are silent
    altogether.  That was a property of a BROKEN STIMULUS, not of the cue:
    `_cylinder_shade` built its silhouette mask from a clipped normal
    coordinate, so the "cylinder" was a full-width painted band, the far box's
    top edge -- the very contour the junction interrupts -- was not in the image
    between y=42 and y=149, and the mask credited every fragment in the band to
    the cylinder.  With the renderer fixed the scene really does carry a T at the
    left limb and the cue reads it correctly: terminator = the box's top edge
    ending at (61,112), 8 px short of the limb at x=69; bar = the cylinder's
    limb, min side 23.18 px = 3.86 spacings against a 9.0 px bound; edge
    (cylinder, box) against truth cylinder=0 / box=1.  So the honest invariant is
    LOCALITY, not silence -- and locality is what still excludes the tangency.
    `_u12_promotion_postB37.txt`.

    Locality is asserted on the DECIDED edges, because those are the only ones
    that make an order claim: a same-object site is a false claim of OCCLUSION
    (the existence question) and is counted in the false-site census the bar
    guard is measured against, not here.  Measured post-fix on the two scenes
    this test covers: `cylinder_over_box` ships 1 site and 1 decided edge, 8.0 px
    from J_left; `cylinder_over_box_shift` (haze 0.35 on the far box) ships 3
    sites -- (53,124) box/box, (117,132) cylinder/cylinder, (132,104)
    unattributed/cylinder -- and 0 decided edges, so it makes no order claim at
    all.  Neither scene ships a site at the bottom-rim tangency near y=150.
    """
    sc, run = scene(name)
    edges = SF.t_junction_order(run.fragments, run.records)
    c = T.classify_edges(sc, run.fragments, edges)
    assert not c["wrong"], "%s: fabricated %s" % (name, c["wrong"])
    if not sc.junctions:
        assert not edges, "%s: a control scene emitted %s" % (name, edges)
        return
    by_index = {int(f.index): f for f in run.fragments}
    for a, b, _oa, _ob in c["correct"] + c["wrong"]:
        pts = [by_index[int(a)].position, by_index[int(b)].position]
        d = min(math.hypot(px - float(j["x"]), py - float(j["y"]))
                for j in sc.junctions for px, py in pts)
        assert d <= J_AT_RADIUS_PX, (
            "%s: a decided order edge %.1f px from the nearest true junction %s "
            "-- a claim where the geometry carries none"
            % (name, d, [(j["id"], j["x"], j["y"]) for j in sc.junctions]))
    sites, _diag = SF.t_junction_sites(run.records)
    rim_y = None
    if "cylinder" in sc.masks:
        ys = np.where(sc.masks["cylinder"].any(axis=1))[0]
        rim_y = float(ys.max())
    for st in sites:
        ex, ey = st["end_pos"]
        if rim_y is not None:
            assert abs(ey - rim_y) > 3.0, (
                "%s: a site ON the cylinder's bottom rim (y=%.0f vs rim %.0f) "
                "-- the tangency is not an occlusion" % (name, ey, rim_y))


def test_diag_attributes_every_rejection(scene):
    """The cue must say why it stayed silent -- the session's attribution rule.
    Every chain end is accounted for as a site or by a named rejection."""
    sc, run = scene("box_over_box")
    sites, diag = SF.t_junction_sites(run.records)
    for key in ("ends", "sites", "rejected_turn", "rejected_no_contact",
                "rejected_not_interrupted", "rejected_short_bar",
                "min_bar_side", "min_bar_side_scale", "spacing", "tol"):
        assert key in diag, "diag is missing %r" % key
    assert diag["sites"] == len(sites)
    assert diag["ends"] >= diag["sites"]
    assert diag["rejected_not_interrupted"] > 0, (
        "on box_over_box the interruption test rejects nothing -- the corpus or "
        "the test changed; re-measure before trusting the gate")


def test_a_failed_bar_guard_never_promotes_a_farther_candidate():
    """[R11] The bar guard is SUBTRACTIVE OVER SITES, not over candidates.

    Condition 3 is an evidence test about the contour the terminator actually
    runs into.  If that contour does not pass through, there is no T at the
    site, and the site must abstain -- it must not go looking for some other
    contour within the contact bound that WOULD support a T, because that
    answers a different question ("is there any nearby contour whose chain
    looks like a bar?") with the same confident voice as a real junction.

    Found on the fresh held-out set, not derived there: `f_cyl_b` emitted
    `(33,141,'box','cylinder')`, the corpus's first WRONG-direction edge, on a
    scene whose junction statistics no rule had seen (`_u12_promotion.txt`).
    Mechanism: the strict tie rule (U12.1) rejected the nearest crossing
    candidate -- a same-object corner whose bar side sits exactly on the bound
    -- and the loop kept searching, so a candidate 0.34 px farther won the slot
    and shipped its direction.  Promotion was latent under `>=` as well; the
    tie rule is what exposed it.

    This fixture is the same shape: the nearest candidate fails the guard, a
    farther one would pass it, and the site must stay silent."""
    base = _junction_records()
    sites, _ = SF.t_junction_sites(base)
    assert len(sites) == 1 and sites[0]["chain_bar"] == "occ", sites
    assert sites[0]["end_pos"] == (69.0, 100.0), sites[0]["end_pos"]
    # a corner fragment 2 px nearer the terminator's end than the real bar:
    # three members, so it carries one spacing (7 px) either side of the
    # contact against a bound of 1.5 * 7 = 10.5 px
    frag = _vchain("frag", 71.0, [93.0, 100.0, 107.0])
    sites, diag = SF.t_junction_sites(base + frag)
    assert [s for s in sites if s["end_pos"] == (69.0, 100.0)] == [], (
        "the nearest crossing candidate failed the bar guard and a farther one "
        "(%s) was promoted into its place"
        % ([s["chain_bar"] for s in sites if s["end_pos"] == (69.0, 100.0)],))
    assert diag["rejected_short_bar"] >= 1, diag
    # the farther candidate is still reachable as a bar -- what changed is that
    # it is no longer reachable THROUGH a rejected nearer one
    assert diag["spacing"] == STEP, diag["spacing"]
