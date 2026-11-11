"""Finding F5 -- `_corner_links`' distance gate rejects a true corner at the
measured sampling, which is why no occlusion scene produces a closed ring.

`_corner_links` exists for exactly this case ("used so that a sharp-cornered
outline closes as one contour instead of breaking into arcs at every corner").
Its five conditions are

    1. sqrt(d2) <= spacing            # spacing = _median_nn_spacing(ALL peaks)
    2. dth > ALIGNMENT_TOL
    3. dth <= CORNER_MAX_TURN
    4. kappa <= curvature_max * CORNER_CURVATURE_FACTOR
    5. shared smooth neighbour, or both degree-1

Measured on the corpus (`_corner_links` instrumented):

    scene              smooth links -> kept   corner links -> kept
    box_over_box       175 -> 171 (98%)       1 -> 1
    single_box         170 -> 170 (100%)      0 -> 0
    disk_on_box        166 -> 165 (99%)       1 -> 1
    cylinder_over_box  171 -> 170 (99%)       2 -> 2

The end-stop veto is therefore NOT what kills corner links -- every candidate
that exists survives it.  Condition 1 is: at the four corners of `single_box`
the cross-edge peak pairs measure

    d = 6.08, 7.81, 9.43, 10.00, 11.66, 12.00 px   against spacing = 6.000 px

with dth = 1.57 rad (90 deg, so conditions 2-4 pass) -- condition 1 alone
rejects every one of them.

Why that is structural, not a tuning miss: the peak detector puts no peak inside
a corner (the Gabor bank has no channel tuned to "two orientations at once"
there), so the pair that has to cross the corner spans two straight-run gaps
plus the corner's own geometry -- for a 90 deg turn sampled at `s` that is
s*sqrt(2) = 1.414*s at best, and more when the runs are sampled at s but the
corner point is skipped.  Condition 1 compares that chord against the STRAIGHT
RUN's spacing, so it rejects the corner at the sampling the contour actually
has.  Measured: 6.08 px against a 6.00 px gate is 1.01x -- a hair.

Downstream: no corner link -> the rectangle's four sides stay four components ->
no component contains a cycle -> `_principal_cycle`, which can only return a
FUNDAMENTAL cycle of the spanning tree, has nothing to return -> `is_ring` is
False for every chain.  Measured rings per scene: cylinder_over_box 0,
box_over_box 0, disk_on_box 0, box_over_box_test 0, cylinder_over_box_shift 0,
box_over_box_haze 0 -- while the two NO-occlusion controls close
(single_disk 1, single_box 1).

REPAIR STATUS -- the two obvious repairs are REFUTED BY MEASUREMENT, and the
order signal no longer depends on this (see test_junction_representability.py
and test_t_junction_interrupted.py, which get the order from the chain topology
directly).  Both variants were run over the whole corpus:

  FIX1  condition 1 -> `d < max_gap` (the module's own link budget):
        corner links 1 -> 40 on box_over_box, chains 28 -> 18, and
        **rings still 0**.  The extra links merge the two boxes into one
        115-node component spanning x[25,166] y[33,162]; the ring test (D5)
        needs the spanning-tree spine's two ENDS to be adjacent, which a
        component of that shape does not have.
  FIX2  FIX1 plus eligibility judged on the ORIGINAL smooth adjacency (the
        greedy cascade in condition 5 is what let one accepted link make the
        next one eligible):
        chains 28 -> 17, biggest chain 97 members (both boxes), **rings still
        0**, and the junction cue LOSES its sites: t_junction_order goes 1 -> 0
        on box_over_box and 2 -> 0 on box_over_box_test, because the bar and the
        stem are now members of ONE chain and the end-on-interior configuration
        no longer exists.

  And the bound cannot be widened far enough anyway: at the near box's two
  BOTTOM corners the peaks nearest the corner are (75,140) and (82,149), a chord
  of 11.4 px against a max_gap of 10.35 px, so no pair there is even a
  candidate.  A closed ring for that occluder needs the corner itself to carry a
  peak, which the front end does not produce.

So these three tests stay as xfail pins on an OPEN defect (B30/B29): the
mechanism is real and measured, but "close the occluder into one ring" is not a
route to the order signal -- it is measured to destroy it.  If a future change
makes them pass they will report XPASS.

That is the issue's candidate (d), and it is also what starves
`t_junction_arms`: measured on box_over_box it finds 7 two-chain clusters with
`through_pairs = 0`, `corners = 7`, `stems = 0`, and not one of the 7 cluster
tips is at either true junction ((76,104) and (120,104)) -- the nearest arm to
J_left is 7 px away and J_right has no arm at all within 12 px.  The occluder's
silhouette never becomes a contour that PASSES the junction, so there is no
through-pair to find.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception import chain_formation as CF


SPACING = 6.0
MEASURED_CORNER_CHORDS = (6.08, 7.81, 9.43, 10.00, 11.66, 12.00)


def _rectangle_peaks(w=96.0, h=60.0, x0=20.0, y0=20.0, spacing=SPACING):
    """A rectangle sampled at `spacing` along each edge, EXCLUDING the corner
    points themselves -- the peak detector does not place a peak inside a
    corner, so the pair that must cross it spans two straight-run gaps.

    Per edge: samples at t = k/n, k = 1 .. n-1 (the two ends are the corners and
    are left out).  Consecutive same-edge samples are exactly `spacing` apart.
    """
    corners = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
    peaks = []
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(3, int(round(L / spacing)))
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0]) % math.pi
        for k in range(1, n):
            t = k / float(n)
            peaks.append(dict(x=int(round(p0[0] + (p1[0] - p0[0]) * t)),
                              y=int(round(p0[1] + (p1[1] - p0[1]) * t)),
                              theta=ang))
    return peaks


def _link_state(peaks):
    xs = np.array([p["x"] for p in peaks], dtype=np.int64)
    ys = np.array([p["y"] for p in peaks], dtype=np.int64)
    th = np.array([p["theta"] for p in peaks], dtype=np.float64)
    spacing = CF._median_nn_spacing(xs, ys)
    max_gap = 2.0 * spacing
    curv_max = CF.ARC_TURN_PER_SAMPLE / spacing
    half_sag = CF.SAGITTA_FRAC_OF_GAP * max_gap
    I, J, D2 = CF._candidate_pairs(xs, ys, th, max_gap ** 2, curv_max,
                                   CF.ALIGNMENT_TOL, half_sag)
    ci, cj, cd2 = CF._corner_links(I, J, D2, xs, ys, th, spacing, curv_max,
                                   CF.ALIGNMENT_TOL, max_gap)
    return dict(spacing=spacing, max_gap=max_gap, smooth=I.size,
                corner=ci.size, corner_pairs=list(zip(ci.tolist(),
                                                      cj.tolist())),
                xs=xs, ys=ys, th=th, I=I, J=J)


def test_corner_chord_exceeds_the_spacing_gate_at_the_measured_sampling():
    """The measured numbers, against the two budgets in force.

    `_corner_links` compares the corner chord against `spacing` =
    `_median_nn_spacing(all peaks)`; every other link rule in this module
    compares against `max_gap = chain_max_gap_factor * median_wavelength`.
    On `box_over_box` (spacing 5.385, median_wavelength 5.175, max_gap 10.351)
    the near rectangle's four corners measure:

        corner      straddling pair        d      dth    <= spacing?  < max_gap?
        (76, 56)    (76,55)-(75,61)      6.08     90 deg    no         yes
        (119,56)    (120,56)-(114,56)    6.00     90 deg    no         yes
        (76,149)    (74,149)-(75,140)    9.06     75 deg    no         yes
        (119,149)   (119,149)-(119,143)  6.00     75 deg    no         yes

    All four are inside the module's own link budget and all four are rejected
    by condition 1.  The analytic reason is that a corner chord is *longer*
    than a straight run's spacing: with no peak inside the corner, the pair
    that has to cross it spans at least s*sqrt(2) = 1.414*s for a 90 deg turn
    (the fixture below measures exactly 8.485 = 6*sqrt(2)).
    """
    for d, dth in ((6.00, 90.0), (6.08, 90.0), (9.06, 75.0), (6.00, 75.0)):
        assert d > 5.385, "corner chord %g is not beyond spacing" % d
        assert d < 10.351, "corner chord %g is not inside max_gap" % d
        assert dth > 35.0
    assert 6.0 * math.sqrt(2) > 6.0, "a 90 deg corner chord is s*sqrt(2) > s"


@pytest.mark.xfail(reason="B30 open: condition 1 rejects every true corner (chord >= spacing*sqrt(2)); both candidate repairs are refuted by measurement -- see this module's docstring", strict=False)
def test_corner_links_are_offered_and_survive_the_endstop_veto():
    """Control for the main test below: `_corner_links` DOES offer the corner
    pairs (conditions 2-5 pass) and the end-stop veto keeps them.  So the loss
    is not in the veto."""
    peaks = _rectangle_peaks()
    got = _link_state(peaks)
    assert got["corner"] >= 4, (
        "expected >= 4 corner candidates on a 4-corner rectangle, got %d "
        "(spacing %.2f, max_gap %.2f).  If 0, the pairs were never offered: "
        "check `_candidate_pairs` reach before blaming condition 1."
        % (got["corner"], got["spacing"], got["max_gap"]))


@pytest.mark.xfail(reason="B30 open: condition 1 rejects every true corner (chord >= spacing*sqrt(2)); both candidate repairs are refuted by measurement -- see this module's docstring", strict=False)
def test_corner_link_gate_admits_a_chord_that_max_gap_admits():
    """THE gate, called directly with the measured geometry.

    Two peaks straddling a 90 deg corner: (110,20) pointing along +x and
    (116,26) pointing along +y, 8.485 px apart (= 6*sqrt(2), the analytic
    corner chord at 6 px sampling).  spacing 6.0, max_gap 12.0, and both peaks
    have exactly one smooth neighbour each (condition 5's "two runs meeting at
    a corner" case).

    Conditions 2, 3, 4 and 5 all pass.  Condition 1 (`d <= spacing`) rejects
    it, so `_corner_links` returns nothing and the corner stays open.
    """
    xs = np.array([110, 116], dtype=np.int64)
    ys = np.array([20, 26], dtype=np.int64)
    th = np.array([0.0, math.pi / 2.0], dtype=np.float64)
    d = math.hypot(6.0, 6.0)
    spacing = SPACING
    max_gap = 2.0 * spacing
    curvature_max = CF.ARC_TURN_PER_SAMPLE / spacing
    # sanity: the other four conditions hold for this pair
    dth = math.pi / 2.0
    assert dth > CF.ALIGNMENT_TOL and dth <= CF.CORNER_MAX_TURN
    assert 2.0 * math.sin(dth / 2.0) / d <= curvature_max * CF.CORNER_CURVATURE_FACTOR
    assert d < max_gap and d > spacing
    ci, cj, _cd2 = CF._corner_links(np.array([0], dtype=np.int64),
                                    np.array([1], dtype=np.int64),
                                    np.array([36.0]), xs, ys, th,
                                    spacing, curvature_max, CF.ALIGNMENT_TOL,
                                    max_gap)
    assert ci.size == 1, (
        "corner link at d=%.3f px (= 6*sqrt(2)), spacing=%.1f, max_gap=%.1f was "
        "rejected by condition 1 (`d <= spacing`) although conditions 2-5 all "
        "pass: a 90 deg corner cannot close (F5)" % (d, spacing, max_gap))


@pytest.mark.xfail(reason="B30 open: condition 1 rejects every true corner (chord >= spacing*sqrt(2)); both candidate repairs are refuted by measurement -- see this module's docstring", strict=False)
def test_rectangle_closes_as_one_ring_through_form_chains():
    """End-to-end at the chain layer, with the measured corner geometry: the
    outline must come out of `form_chains` as ONE closed chain holding most of
    the peaks.  This is the property the whole figure-ground thread needs of an
    occluder silhouette."""
    peaks = _rectangle_peaks()
    endstop = np.zeros((12, 5, 160, 160), dtype=np.float64)
    chains = CF.form_chains(peaks, endstop, median_wavelength=SPACING)
    assert chains, "form_chains emitted nothing"
    biggest = max(chains, key=lambda c: len(c[0]))
    assert biggest[1], (
        "the largest chain (%d of %d peaks) is NOT a ring: the rectangle's "
        "outline came out as %d open arcs (F5)"
        % (len(biggest[0]), len(peaks), len(chains)))
    assert len(biggest[0]) >= 0.8 * len(peaks)


def test_smooth_rule_cannot_cross_a_90_degree_corner():
    """Control: `_candidate_pairs` (rules 1-4) must NOT link across the corner,
    otherwise the corner link is doing no work and the tests above prove
    nothing.  Rule 3 (ALIGNMENT_TOL = 35 deg, mutual pointing) rejects a 90 deg
    turn by design."""
    peaks = _rectangle_peaks()
    got = _link_state(peaks)
    th = got["th"]
    cross = 0
    for i, j in zip(got["I"].tolist(), got["J"].tolist()):
        dth = abs(th[i] - th[j]) % math.pi
        dth = min(dth, math.pi - dth)
        if dth > CF.ALIGNMENT_TOL:
            cross += 1
    assert cross == 0, "the smooth rule already crosses corners; F5 is moot"
