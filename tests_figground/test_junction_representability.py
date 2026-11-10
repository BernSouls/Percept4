"""Finding F6 (PRIMARY, supersedes the corner-gate story) -- at a true
T-junction the contour point is DELETED from the representation, so no order
cue can see it.

The chain, every link measured on `box_over_box` / `box_over_box_test`:

1. `fragment_ownership.ownership_fragments` computes `signed` as
   `tanh(_consensus(s_local, frames, closed, positions, tjs, radius))` and
   `logodds` as `_consensus(lo_local, ...)`.  `tjs` comes from
   `_t_junctions(observation, ...)`, which is EMPTY on every occlusion scene
   (census: liveTJ = 0 for all 5, because the observation's
   `get_t_junctions` needs a closed ring and rings = 0).  With no junction
   mask, the consensus smoothing averages the two opposing votes at a junction
   to exactly zero:

       pos        vertex  signed        logodds      logodds_local
       (120,103)  6/12    2.8e-17       1.9e-17      -0.2527
       (119, 98)  7/12    2.8e-17       1.9e-17      +0.2527
       (120,109)  5/12    2.1e-08       2.2e-08       0.0
       ( 69,104)  6/ 7    0.0           0.0           0.0      <- bar terminus at J_left

   26 of 182 records have `signed` exactly 0.0; 38 have |signed| < 0.02.

2. `surface_fragments._owner_normal` returns None when
   `abs(signed) < _AMBIGUOUS_OWNERSHIP` (0.02), and `build_surface_fragments`
   does `if on is None: continue` -- so NO FRAGMENT IS CREATED.  Measured:
   box_over_box 182 records -> 144 fragments, and all 38 drops are
   `ambiguous_ownership` (0 for missing normal, 0 for missing pos).

3. `t_junction_order` looks up BOTH ends in `by_alpha` and skips the site when
   either is missing.  At J_left the bar's terminus (69,104) is one of the 38;
   at J_right in `box_over_box_test` the bar's terminus (127,104) AND the stem
   contact (120,103) are both among the 32.  So the cue cannot express
   precisely the sites it exists to find.  Measured with the site search alone
   (`_tjo_why.py`): at tol_scale 1.5 box_over_box emits 16 alpha edges of which
   9 are unrepresentable, and 0 are CORRECT.

4. Therefore no order edge, no depth layer (census: layers 2/144 on
   box_over_box, from an occ/tjo 2-cycle that abstains), and the four
   figure-ground mechanisms all degenerate at the junction -- which is the
   reported symptom.

The repair this file pins is REPRESENTABILITY, not ownership: a record with a
position and a finite normal denotes a contour point whether or not its
ownership is decidable, and the order layer needs position + chain topology,
not ownership.  Per the session rule, ownership keeps answering only "which
side is my object" -- every ownership-dependent cue stays gated on a decided
owner, so no slant/tilt/owner claim is invented for an undecided point.

Measured payoff (`_besi_measure.py`, the same end-on-interior rule evaluated on
chains so representability does not intervene): 3 CORRECT / 0 WRONG /
10 UNDECIDED(same object) / 5 unattributable over the whole corpus, including
BOTH true junctions of `box_over_box_test`.
"""
from __future__ import annotations

import math

import pytest

import truth as T
from scenes import (TEST_SCENES as TEST, OCCLUSION_SCENES,
                    CONTROL_SCENES)
from perception import surface_fragments as SF


JUNCTION_TOL_PX = 9.0


def _frag_alphas(run):
    return {f.alpha for f in run.fragments}


def _records_at_junctions(scene, run, radius=JUNCTION_TOL_PX):
    """Records lying within `radius` of a renderer-true junction."""
    out = []
    for rec in run.records:
        pos = rec.get("pos")
        if pos is None:
            continue
        for j in scene.junctions:
            if math.hypot(float(pos[0]) - j["x"], float(pos[1]) - j["y"]) <= radius:
                out.append((j, rec))
                break
    return out


# --------------------------------------------------------------------------
# F6.1  existence is not ownership
# --------------------------------------------------------------------------

def test_every_placed_record_gets_a_fragment(scene):
    """A record with a position and a finite normal denotes a contour point.
    It must exist in the fragment list even when its ownership is undecided --
    otherwise the order layer cannot see the junction at all.

    Measured now: box_over_box 182 records -> 144 fragments, all 38 drops
    caused by |signed| < _AMBIGUOUS_OWNERSHIP."""
    sc, run = scene("box_over_box")
    have = _frag_alphas(run)
    placed = [r for r in run.records
              if r.get("pos") is not None and r.get("normal") is not None
              and math.hypot(float(r["normal"][0]), float(r["normal"][1])) > 1e-9]
    missing = [r for r in placed if r.get("alpha") not in have]
    assert not missing, (
        "%d of %d placed records have no surface fragment (first: pos=%s "
        "signed=%r) -- an undecided owner deletes the contour point (F6)"
        % (len(missing), len(placed),
           tuple(round(float(v)) for v in missing[0]["pos"]),
           missing[0].get("signed")))


def test_junction_records_are_represented_and_decided(scene):
    """Regression pin for the repair: the records at a true junction must EXIST
    as fragments.

    [U14 phase 4] RE-DERIVED on the real tree.  The stand-in measurement this
    pin used to quote ("every record within 9 px of J_left except (75,110) has
    |signed| below the 0.02 floor, so it has no fragment") is VOID (U14.-1.7):
    with the real parameter set the junction records are DECIDED.  Measured at
    a 9 px radius:

        scene               junction  records  decided  undecided  no-fragment
        box_over_box        J_left         4        4          0            0
        box_over_box        J_right        4        4          0            0
        box_over_box_test   J_left         4        4          0            0
        box_over_box_test   J_right        4        4          0            0

    (box_over_box J_left: (76,97) signed -0.6169, (71,108) +1.0000,
    (75,103) -0.5006, (75,109) -0.4707.)  The requirement the repair bought is
    REPRESENTATION at any |signed|; the old "all undecided at junctions"
    expectation belongs to the stand-in.  What is still asserted here is both
    halves of that: every junction record is a fragment, a DECIDED record
    reports a decided owner side, and an undecided one never does.
    """
    sc, run = scene("box_over_box")
    have = _frag_alphas(run)
    at_j = _records_at_junctions(sc, run)
    assert at_j, "no records near the true junctions -- fixture is broken"
    missing = [(j["id"], tuple(round(float(v)) for v in r["pos"]))
               for j, r in at_j if r.get("alpha") not in have]
    assert not missing, (
        "records at true junctions still have no fragment: %s (F6)" % (missing,))
    by_alpha = {f.alpha: f for f in run.fragments}
    decided = undecided = 0
    for j, r in at_j:
        f = by_alpha.get(r.get("alpha"))
        assert f is not None
        if abs(float(r.get("signed", 0.0) or 0.0)) >= SF._AMBIGUOUS_OWNERSHIP:
            decided += 1
            assert f.owner_is_figure(), (
                "a junction record with |signed| >= the ambiguity floor is not "
                "reported as having a decided owner side at %s" % (j["id"],))
        else:
            undecided += 1
            assert not f.owner_is_figure(), (
                "a record with |signed| below the ambiguity floor is being "
                "reported as having a decided owner side at %s" % (j["id"],))
    assert decided, (
        "no junction record is DECIDED any more -- if the tree really went back "
        "to cancelling ownership at junctions, re-derive this pin from the new "
        "measurement instead of restoring the stand-in numbers")
    assert decided + undecided == len(at_j)


def test_undecided_ownership_is_marked_not_silently_dropped(scene):
    """When a fragment exists for an undecided point it must SAY so: the
    owner-side claim is what downstream cues read, and an undecided point must
    not look like a decided one."""
    sc, run = scene("box_over_box")
    undecided = [f for f in run.fragments if not f.owner_is_figure()]
    have = _frag_alphas(run)
    n_expected = sum(1 for r in run.records
                     if r.get("alpha") not in have
                     and r.get("pos") is not None and r.get("normal") is not None)
    assert undecided, (
        "no fragment reports undecided ownership although %d records were "
        "dropped for it (F6)" % n_expected)
    for f in undecided:
        assert f.owner_side == 0.0


# --------------------------------------------------------------------------
# F6.2  the repair must not invent an ownership answer
# --------------------------------------------------------------------------

def test_undecided_fragments_carry_no_owner_dependent_cue(scene):
    """Guard on the repair: an undecided point has no owner side, so the
    contour cue and the shading cue (both of which need the inward normal) must
    not be attached to it.  Nothing may invent a slant/tilt there."""
    sc, run = scene("box_over_box")
    for f in run.fragments:
        if f.owner_is_figure():
            continue
        assert "contour" not in f.cues, (
            "undecided fragment at %s carries a contour cue" % (f.position,))
        assert "shading" not in f.cues, (
            "undecided fragment at %s carries a shading cue" % (f.position,))
        assert f.enclosure is None or f.owner_side == 0.0


def test_occlusion_order_edges_use_only_decided_owners(scene):
    """`occlusion_order_edges` reads `b.owner_normal` to test which side A is
    on.  With undecided fragments now present, that test would be reading a
    normal whose SIGN is undefined -- i.e. it could fabricate an order claim
    out of an ownership abstention.  Every emitted edge must have decided
    ownership at both ends."""
    for nm in OCCLUSION_SCENES + CONTROL_SCENES:
        scene_obj, run = scene(nm)
        by_index = {int(f.index): f for f in run.fragments}
        for a, b in SF.occlusion_order_edges(run.fragments, run.records):
            fa, fb = by_index.get(int(a)), by_index.get(int(b))
            assert fa is not None and fb is not None
            assert fa.owner_is_figure() and fb.owner_is_figure(), (
                "%s: occlusion edge (%d,%d) uses an undecided owner (F6 guard)"
                % (nm, a, b))


# --------------------------------------------------------------------------
# F6.3  the payoff: the junction becomes visible to the order layer
# --------------------------------------------------------------------------

def test_t_junction_order_reaches_a_true_junction(scene):
    """On a TEST scene the cue must emit at least one edge AT a true junction.
    Measured now: 0 of 4 emitted edges is within 12 px of either junction (the
    nearest is 36.9 px away), because both junction sites are unrepresentable."""
    sc, run = scene("box_over_box_test")
    by_index = {int(f.index): f for f in run.fragments}
    edges = SF.t_junction_order(run.fragments, run.records)
    hits = []
    for a, b in edges:
        fa, fb = by_index.get(int(a)), by_index.get(int(b))
        if fa is None or fb is None:
            continue
        for j in sc.junctions:
            d_near = math.hypot(fa.position[0] - j["x"], fa.position[1] - j["y"])
            d_far = math.hypot(fb.position[0] - j["x"], fb.position[1] - j["y"])
            if max(d_near, d_far) <= JUNCTION_TOL_PX:
                hits.append((j["id"], a, b, round(d_near, 1), round(d_far, 1)))
    assert hits, (
        "t_junction_order emitted %d edges on box_over_box_test and NONE is at "
        "a true junction (%s): the sites exist but their records have no "
        "fragment (F6)"
        % (len(edges), [(j["id"], j["x"], j["y"]) for j in sc.junctions]))


def test_zero_wrong_direction_junction_edges_on_test(scene):
    """SHIP GATE: on the TEST scenes, every DECIDED t_junction_order edge must
    agree with the renderer's depth order.  A wrong-direction edge is a
    fabrication, and widening the cue's radius alone produces them (measured:
    1 WRONG on box_over_box_test at tol_scale 1.5)."""
    for nm in TEST:
        scene_obj, run = scene(nm)
        edges = SF.t_junction_order(run.fragments, run.records)
        c = T.classify_edges(scene_obj, run.fragments, edges)
        assert not c["wrong"], (
            "%s: t_junction_order fabricated %d wrong-direction edge(s): %s"
            % (nm, len(c["wrong"]), c["wrong"]))


def test_at_least_one_correct_junction_edge_over_the_corpus(scene):
    """Recall must be > 0 somewhere in the corpus (U11.2).  Measured now: 0
    CORRECT t_junction_order edges on every scene at the shipped tolerance."""
    total_correct = 0
    detail = []
    for nm in OCCLUSION_SCENES:
        scene_obj, run = scene(nm)
        edges = SF.t_junction_order(run.fragments, run.records)
        c = T.classify_edges(scene_obj, run.fragments, edges)
        total_correct += len(c["correct"])
        detail.append("%s:%d/%d" % (nm, len(c["correct"]), len(c["wrong"])))
    assert total_correct > 0, (
        "no correct order edge anywhere in the occlusion corpus (%s)"
        % " ".join(detail))


def test_controls_stay_silent(scene):
    """A control scene has no occluder, so it must produce no DECIDED order
    claim (an intra-object edge is undecided and harmless; a decided one is a
    fabrication)."""
    for nm in CONTROL_SCENES:
        scene_obj, run = scene(nm)
        edges = SF.t_junction_order(run.fragments, run.records)
        c = T.classify_edges(scene_obj, run.fragments, edges)
        assert not c["correct"] and not c["wrong"], (
            "%s: control scene produced a decided order claim %s"
            % (nm, (c["correct"], c["wrong"])))


# --------------------------------------------------------------------------
# F6.4  diagnosis pin: WHY the ownership cancels at the junction
# --------------------------------------------------------------------------

def test_junction_records_keep_a_decided_consensus_on_the_real_tree(scene):
    """Measurement pin, RE-DERIVED [U14 phase 4] -- the stand-in version of this
    pin asserted the OPPOSITE ("the consensus erases a decisive local switch")
    and is VOID (U14.-1.7).

    With the real parameter set the consensus does NOT cancel at these
    junctions: the smoothed `signed`/`logodds` are decided and agree in sign
    with the member's own evidence.  Measured on `box_over_box`, 9 px radius:

        pos        vertex   signed    logodds   logodds_local
        (76, 97)   3/15     -0.6169    -3.9      -3.9860
        (71,108)   4/ 6     +1.0000   +15.1     +15.0771
        (75,103)   4/15     -0.5006    -3.4      -1.7122
        (75,109)   5/15     -0.4707    -3.0      -0.4600

    and on `box_over_box_test` the one record whose own evidence is exactly
    silent ((75,109), logodds_local +0.0000) is still DECIDED by the consensus
    (signed -0.4194).  So this pin now records the real measurement: at least
    half the junction records are decided, and no junction record shows the
    stand-in's erasure pattern (|logodds| < 1e-9 while |logodds_local| > 0.1).
    """
    sc, run = scene("box_over_box")
    at_j = _records_at_junctions(sc, run)
    assert len(at_j) >= 4, len(at_j)
    decided = [r for _j, r in at_j
               if abs(float(r.get("signed", 0.0) or 0.0))
               >= SF._AMBIGUOUS_OWNERSHIP]
    assert len(decided) >= len(at_j) // 2, (
        "only %d of %d junction records are decided -- the consensus is "
        "cancelling again, so the real-tree measurement moved"
        % (len(decided), len(at_j)))
    erased = [r for _j, r in at_j
              if abs(float(r.get("logodds", 0.0) or 0.0)) < 1e-9
              and abs(float(r.get("logodds_local", 0.0) or 0.0)) > 0.1]
    assert not erased, (
        "the stand-in's consensus erasure pattern is back at %d junction "
        "record(s) -- if the junction mask got wired, state that; do not restore "
        "the stand-in numbers (U14.-1.7 VOID)" % (len(erased),))
