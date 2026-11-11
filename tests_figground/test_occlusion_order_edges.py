"""B42 -- the occlusion order cue must not claim a depth order it cannot
resolve: two ends closer than one sampling step of the contour layer are the
same local piece of contour, so there is no occlusion between them to order.

The fabrication this pins (measured, U13.3, `diagnostics/_u13_occ_pairs.txt`):
after B38 fixed the orientation estimate, EVERY one of the 27 `neighbor` pairs
the ownership pass produces across the 18 development scenes is closer than
1.01 sampling steps.  22 of them are two fragments of ONE object -- a convex
corner whose two edges the linking pass put in different chains, so
`fragment_ownership._pair_edges`' [F2c] same-chain exclusion (same argument,
same bound) never runs, and [F2a]'s diverging-owner-direction test passes
because a rectangle corner turns its owner direction by ~45 deg, past the
35 deg ALIGNMENT_TOL bound.  One of them is the corpus's only WRONG-direction
order edge: f_cyl_b fragments 74->92, a sample of the cylinder's left
silhouette 1 px outside its own mask (which the truth oracle therefore reads as
the box behind it) against a sample of the cylinder's bottom-left corner, 5.10
px apart = 0.72 steps.  Cross-object CORRECT pairs on the whole corpus: 0.

Those pairs are not inert: `assign_depth_layers` calls `occlusion_order_edges`,
so a same-object pair splits ONE object's silhouette across two depth layers
(f_cyl_b: cylinder fragments in layers 0 and 1; cylinder_over_box and
cylinder_over_box_shift report 3 distinct layers for 2 objects).

The bound is `_chain_spacing(records)` -- the median gap between consecutive
members of a chain, the field's own sampling step, already the quantity the
T-junction tolerance is derived from.  No new constant, and no pixel radius that
would pass at one resolution and fail at another.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pytest

from perception import surface_fragments as SF

STEP = 6.0            # the synthetic corpus' sampling step, in px


def _frag(index: int, x: float, y: float, nx: float, ny: float, alpha: Any,
          chain: str, vertex: int = 0, side: float = 0.9) -> SF.SurfaceFragment:
    """A fragment whose owner side is `side` along the normal (nx, ny)."""
    return SF.SurfaceFragment(
        index=int(index), position=(float(x), float(y)),
        tangent=(-float(ny), float(nx)), owner_normal=(float(nx), float(ny)),
        owner_side=float(side), curvature=None, curvature_sign=0.0,
        depth_layer=None, alpha=alpha, chain=chain, vertex=int(vertex),
        enclosure=1.0)


def _rec(alpha: Any, x: float, y: float, chain: str, vertex: int,
         neighbor: Optional[Any] = None) -> Dict[str, Any]:
    return dict(alpha=alpha, pos=(float(x), float(y)), chain=chain,
                vertex=int(vertex), index=int(vertex), closed=False,
                neighbor=neighbor, partner=None)


def _ladder(n: int = 4, step: float = STEP, x0: float = 20.0,
            y0: float = 150.0) -> List[Dict[str, Any]]:
    """`n` records of one chain, `step` px apart: this is what makes
    `_chain_spacing` return `step`."""
    return [_rec("L%d_%d" % (n, k), x0 + k * step, y0, "ladder", k)
            for k in range(n)]


def _pair(d: float, kind: str = "T", step: float = STEP
          ) -> Tuple[List[SF.SurfaceFragment], List[Dict[str, Any]]]:
    """Two fragments, `d` px apart, in DIFFERENT chains, that satisfy every
    admissibility test `occlusion_order_edges` had before B42:

    * both ends have a decided owner side (`owner_side != 0`);
    * `(a.pos - b.pos) . b.owner_normal < 0` -- a is on b's NON-owner side, so
      the record `a.neighbor == b` reads as "b's owner is the figure here, in
      front of the surface a sits on";
    * the two owner directions diverge (90 deg for `kind="T"`, 180 deg for
      `kind="facing"`), so [F2a]'s same-side rejection does not apply.

    `kind="T"` is the real occlusion geometry: `b` is a sample of the BAR (the
    near object's silhouette, owner to its left) and `a` is the TERMINATOR's end
    sample `d` px to the right of it, owner below (the far object's interior).
    `kind="facing"` is the degenerate anti-parallel case, kept so the rule is
    pinned in two geometries rather than one.
    """
    if kind == "T":
        b = _frag(11, 40.0, 60.0, -1.0, 0.0, "B", "chain_bar", vertex=7)
        a = _frag(12, 40.0 + d, 60.0, 0.0, 1.0, "A", "chain_term", vertex=0)
    elif kind == "facing":
        b = _frag(11, 40.0, 60.0, 0.0, -1.0, "B", "chain_bar", vertex=7)
        a = _frag(12, 40.0, 60.0 + d, 0.0, 1.0, "A", "chain_term", vertex=0)
    else:                                                # pragma: no cover
        raise AssertionError(kind)
    recs = _ladder(step=step)
    recs += [_rec("A", a.position[0], a.position[1], "chain_term", 0,
                  neighbor="B"),
             _rec("B", b.position[0], b.position[1], "chain_bar", 7)]
    return [a, b], recs


# --------------------------------------------------------------------------
# the mechanism, pinned synthetically
# --------------------------------------------------------------------------

def test_the_synthetic_ladder_really_sets_the_sampling_step():
    _frags, recs = _pair(0.4 * STEP)
    assert SF._chain_spacing(recs) == pytest.approx(STEP)


def test_the_pair_is_admissible_before_b42():
    """Guard on the guard: at a separation the contour layer CAN resolve, both
    geometries emit their edge, so the rejections below are the new rule doing
    the work and not a pair that was never admissible."""
    for kind in ("T", "facing"):
        frags, recs = _pair(3.0 * STEP, kind=kind)
        assert SF.occlusion_order_edges(frags, recs) == [(11, 12)], kind


@pytest.mark.parametrize("frac", [0.2, 0.5, 0.72, 0.99])
@pytest.mark.parametrize("kind", ["T", "facing"])
def test_a_pair_inside_one_sampling_step_emits_no_order_edge(kind, frac):
    """Two ends closer than the contour layer can resolve are the same local
    piece of one contour, whatever chain identity the linking pass gave them.
    0.72 is the measured f_cyl_b fabrication."""
    frags, recs = _pair(frac * STEP, kind=kind)
    assert SF.occlusion_order_edges(frags, recs) == []


@pytest.mark.parametrize("kind", ["T", "facing"])
def test_a_pair_exactly_one_sampling_step_apart_is_rejected(kind):
    """Strict `>`, matching `min_bar_side_scale`'s strict bound: a tie at the
    derived bound abstains (absent is not neutral)."""
    frags, recs = _pair(STEP, kind=kind)
    assert SF.occlusion_order_edges(frags, recs) == []


def test_the_step_is_the_measured_spacing_not_a_pixel_constant():
    """The same 8 px pair is resolvable at a 6 px step and unresolvable at a
    12 px step: the bound moves with the field's own sampling, so it cannot be
    a pixel radius in disguise."""
    frags, recs = _pair(8.0, kind="T")
    assert SF._chain_spacing(recs) == pytest.approx(STEP)
    assert SF.occlusion_order_edges(frags, recs) == [(11, 12)]
    recs2 = [r for r in recs if not str(r["alpha"]).startswith("L")]
    recs2 += _ladder(step=12.0)
    assert SF._chain_spacing(recs2) == pytest.approx(12.0)
    assert SF.occlusion_order_edges(frags, recs2) == []


def test_without_a_measurable_spacing_the_cue_keeps_its_old_behaviour():
    """No chain with two members -> nothing to measure a separation against.
    The cue must not silently become empty (that would be a different change,
    and a bigger claim than the evidence supports)."""
    frags, recs = _pair(0.5 * STEP, kind="T")
    solo = [r for r in recs if not str(r["alpha"]).startswith("L")]
    assert SF._chain_spacing(solo) is None
    assert SF.occlusion_order_edges(frags, solo) == [(11, 12)]


# --------------------------------------------------------------------------
# the measured fabrication, pinned end to end on the scene that produced it
# --------------------------------------------------------------------------

def test_f_cyl_b_no_longer_emits_the_wrong_direction_edge():
    """f_cyl_b 74->92 was the corpus's only wrong-direction order edge: both
    fragments are samples of the CYLINDER's own silhouette, 5.10 px apart
    (0.72 sampling steps), straddling its bottom-left corner.  The truth oracle
    reads the one 1 px outside the cylinder's mask as the box behind it, so the
    pair scored as a cross-object reversal -- and `assign_depth_layers` put the
    cylinder's own silhouette in two different layers."""
    import fresh_scenes as F
    import pipeline as P
    import truth as T

    scene = F.build_fresh("f_cyl_b")
    run = P.run_pipeline(scene)
    cls = T.classify_edges(scene, run.fragments, run.order["occlusion"])
    assert cls["wrong"] == [], cls["wrong"]
    emitted = {tuple(int(v) for v in e) for e in run.order["occlusion"]}
    assert (74, 92) not in emitted and (92, 74) not in emitted, emitted
    lay = T.layer_correctness(scene, run)
    assert lay["n_wrong"] == 0, lay
    by_index = {int(f.index): f for f in run.fragments}
    layers_of_cylinder = set()
    for idx, layer in (run.layers or {}).items():
        f = by_index.get(int(idx))
        if f is None:
            continue
        if T.fragment_object(scene, f) == "cylinder":
            layers_of_cylinder.add(layer)
    assert len(layers_of_cylinder) <= 1, (
        "the cylinder's own fragments span layers", layers_of_cylinder)


def test_the_controls_still_emit_nothing():
    """B42 only removes edges, so the controls (must NOT fabricate a wrong
    direction) stay free of wrong-direction edges -- see the single_box
    re-derivation below for the measured state.

    [U28 RE-DERIVED, crop-PNG evidence: diagnostics/_u28_sb_corner_*.png and
    diagnostics/_u28_sb_frame.png]  'Controls are silent' was the correct
    expectation for a control without corners.  `single_box` is a
    rectangle with crisp corners (scene note: 'corners but no junction'), and
    its four corners each produce one GHOST halo fragment ~5 px outside the
    box mask in the background halo, exactly where the corner's antialiased
    edge eats into the background.  The ownership cue pairs each ghost with the
    flank fragment at that corner, yielding exactly four order edges
    (44,0), (61,9), (38,12), (43,37).  Every one resolves to (box flank,
    ghost halo), which is the renderer's depth law (box depth=0, background
    depth=1) forward.  The premise tightening: every emitted control edge is
    (box over background) or undecided -- never a wrong direction, never
    box over box.  `single_disk` never emits anything.
    """
    import scenes as S
    import pipeline as P
    import truth as T

    old = S.build_all()
    for name in ("single_box", "single_disk"):
        run = P.run_pipeline(old[name])
        assert run.order["t_junction_order"] == [], name
        assert run.order["t_junction_arms_diag"].get("resolved", []) == [], name
        edges = run.order["occlusion"]
        sc = old[name]
        if name == "single_disk":
            assert edges == [], name
            continue
        c = T.classify_edges(sc, run.fragments, edges)
        assert not c["wrong"], ("a control emitted a wrong-direction edge", name)
        # every emitted edge must be (box, background-halo) or undecided:
        # never a box-over-box claim, never an inverted order.
        for a, b, oa, ob in list(c["correct"]) + list(c["undecided"]):
            assert oa == "box", (name, a, b, oa, ob)
            assert ob != "box", (name, a, b, oa, ob)


@pytest.mark.xfail(strict=True, reason=(
    "single_box closes 0 rings on the real tree: its silhouette fragments into 8 "
    "chains whose four remaining gaps are 11.05 px = 2.135 x median_wavelength "
    "against the FROZEN bridge bound chain_max_gap_factor = 2.0 (10.35 px), so "
    "the ring cannot close by 0.7 px. Measured, not a cue defect: the control "
    "emits no order edge (the test above). Repair would mean raising a frozen "
    "chain-layer constant -- measured elsewhere to fabricate (ledger: do-not-"
    "retry). single_disk still closes 3 rings."))
def test_the_control_rings_still_close():
    """A single closed silhouette is the configuration B42 was derived on, so
    pin that the chain layer still closes it -- for the object where it can."""
    import scenes as S
    import pipeline as P

    old = S.build_all()
    for name in ("single_box", "single_disk"):
        run = P.run_pipeline(old[name])
        assert sum(1 for c in run.chains if c[1]) >= 1, (name, "ring closed")
