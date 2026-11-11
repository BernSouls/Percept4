"""[STAGE 5 / PHASE A] Writer truth: a slot holds a measurement or is ABSENT.

Each pin names the physical rule it holds, then shows the rule on the real
writer.  None of them needs an image: the writers are called on small
hand-built detections whose geometry is known in closed form.

  A1  A size ratio needs two independently measured sizes.
  A2  A whole is located at the centroid of its footprint.
      A quantity compared with itself is not a measurement.
      The bearing of a zero separation is undefined.
  A3  Absence is never a neutral number.
  A4  An index is an index in both tables.
  A5  A population axis the writer can measure is written.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import core.schema as S
import parameters as P
from core.schema import UNKNOWN
from core.types import Detection, EdgeType, TypedSignature
from orchestration.binder import Binder

R = 10.0   # one wavelength (the extraction scale), in position units


# --------------------------------------------------------------------------
# hand-built detections
# --------------------------------------------------------------------------

def _sig(tag: float) -> TypedSignature:
    return TypedSignature(kind="CANONICAL", class_="TEST", relation="T",
                          quantized=(float(tag),), population=())


def _contour(tag, anchors, scale=0):
    """A member: a detection whose footprint is `anchors`."""
    return Detection(
        k="Contour", source_layer=2, scale=scale, sigma=_sig(tag),
        alpha=anchors[0], z=1.0, t=0,
        rho=tuple(("member", a, i, "+") for i, a in enumerate(anchors)))


def _whole(tag, members, scale=0):
    """A whole over `members` = [(node id, detection)], built the way the
    CurveComplex lens builds it: alpha is the FIRST member's first anchor,
    the footprint is the union of the members' anchors."""
    footprint = []
    for _nid, det in members:
        for role in det.rho:
            if role[0] == "member" and role[1] not in footprint:
                footprint.append(role[1])
    return Detection(
        k="CurveComplex", source_layer=2, scale=scale, sigma=_sig(tag),
        alpha=footprint[0], z=1.0, t=0,
        rho=(("entity", footprint[0], 0, "+"),)
        + tuple(("member", a, i, "+") for i, a in enumerate(footprint))
        + tuple(("member_node", (nid,), i, "+")
                for i, (nid, _d) in enumerate(members)))


def _values(whole, child_nid, members, positions):
    return Binder._containment_values(
        whole, child_nid, dict(members), positions, R)


#: two members.  A lies on the x axis from 0 to 40 (centre 20, extent 20);
#: B is a short vertical stroke far to the right (centre (100, 0), extent 5).
POS = {
    "a0": (0.0, 0.0, 0.0), "a1": (40.0, 0.0, 0.0),
    "b0": (100.0, -5.0, 0.0), "b1": (100.0, 5.0, 0.0),
}
A = _contour(1, ["a0", "a1"])
B = _contour(2, ["b0", "b1"])
MEMBERS = [(11, A), (12, B)]
WHOLE = _whole(9, MEMBERS)
#: footprint centroid of the whole: mean of the four anchors
CENTRE = (60.0, 0.0)


# --------------------------------------------------------------------------
# A2 -- where the whole is
# --------------------------------------------------------------------------

def test_a_whole_is_located_at_its_footprint_centroid_not_at_its_first_member():
    """RULE: the address of a part is measured from the centre of the whole.

    The CurveComplex lens sets `alpha` to its first member's first anchor,
    here (0, 0): a point on the rim.  Measured from there, member A would
    sit 2 wavelengths east and B 10 wavelengths east.  Measured from the
    footprint centroid (60, 0) -- the centre the part writers use -- A is 4
    wavelengths WEST and B 4 wavelengths EAST: the two parts are placed
    symmetrically about their whole, which is what the scene is.
    """
    assert WHOLE.alpha == "a0"
    va = _values(WHOLE, 11, MEMBERS, POS)
    vb = _values(WHOLE, 12, MEMBERS, POS)
    assert va["rel_dist"] == pytest.approx(40.0 / R)
    assert vb["rel_dist"] == pytest.approx(40.0 / R)
    assert va["polar_angle"] == pytest.approx(math.pi)       # west
    assert vb["polar_angle"] == pytest.approx(0.0)           # east
    # and it does not depend on which member the lens happened to list first
    flipped = _whole(9, [(12, B), (11, A)])
    assert flipped.alpha == "b0"
    assert _values(flipped, 11, MEMBERS, POS) == va
    assert _values(flipped, 12, MEMBERS, POS) == vb


def test_a_whole_with_one_member_has_no_internal_geometry():
    """RULE: a quantity compared with itself is not a measurement.

    A whole whose only member is M has M's footprint: separation 0 and size
    ratio 1 hold by construction, for any image.  This is the state of every
    CurveComplex on the single-step path (1,090 of 1,090 containment edges
    of the 79 graphs), and written as measurements the two zeros made every
    such edge agree with every other on "2 measured axes".  They are ABSENT.
    """
    solo = _whole(9, [(11, A)])
    v = _values(solo, 11, [(11, A)], POS)
    assert v == {"rel_dist": None, "polar_angle": None, "scale_ratio": None}

    ep = S.edge_params("EDGE_CURVE_CONTAINMENT", v, sign=1, confidence=1.0)
    assert ep.quantized == (UNKNOWN, UNKNOWN, UNKNOWN)
    # no information: nothing measured on both sides, so nothing to agree on
    assert S.edge_measured_axes(ep, ep) == 0
    assert S.edge_graded_axes(ep, ep) == 0
    assert ep.similarity(ep) == 0.0

    # what the same edge used to say, and what it earned for saying it
    old = S.edge_params("EDGE_CURVE_CONTAINMENT",
                        {"rel_dist": 0.0, "polar_angle": 0.0,
                         "scale_ratio": None})
    assert S.edge_measured_axes(old, old) == 2
    assert old.similarity(old) == 1.0


def test_the_bearing_of_a_zero_separation_is_absent():
    """RULE: a direction needs a displacement.

    atan2(0, 0) returns 0, which reads as "due east".  A member that sits
    exactly at the centre of its whole has a measured separation (0) and no
    bearing.
    """
    centre_piece = _contour(3, ["c0", "c1"])
    pos = dict(POS, c0=(60.0, -8.0, 0.0), c1=(60.0, 8.0, 0.0))
    members = MEMBERS + [(13, centre_piece)]
    whole = _whole(9, members)
    v = _values(whole, 13, members, pos)
    assert v["rel_dist"] == 0.0            # measured: it is at the centre
    assert v["polar_angle"] is None        # undefined, so absent
    assert v["scale_ratio"] is not None
    ep = S.edge_params("EDGE_CURVE_CONTAINMENT", v)
    assert ep.quantized[0] == 0.0 and ep.quantized[1] == UNKNOWN


# --------------------------------------------------------------------------
# A1 -- size
# --------------------------------------------------------------------------

def test_the_size_ratio_of_a_part_is_the_ratio_of_the_two_footprint_extents():
    """RULE: a size ratio needs two independently measured sizes.

    No lens on the region path declares a size (`Detection.scale` is 0 on
    every detection), so the ratio was never measured.  A whole and its
    member do have sizes the step can measure with one ruler: the extent
    (half-diameter) of each footprint.  Whole: anchors span 0..100 in x and
    -5..5 in y, diameter |(0,0)-(100,+-5)| = sqrt(10025).  A: diameter 40.
    B: diameter 10.
    """
    whole_extent = 0.5 * math.hypot(100.0, 5.0)
    va = _values(WHOLE, 11, MEMBERS, POS)
    vb = _values(WHOLE, 12, MEMBERS, POS)
    assert va["scale_ratio"] == pytest.approx(math.log2(20.0 / whole_extent))
    assert vb["scale_ratio"] == pytest.approx(math.log2(5.0 / whole_extent))
    # a member is never larger than its whole: half-diameter is monotone
    assert va["scale_ratio"] < 0.0 and vb["scale_ratio"] < 0.0
    assert vb["scale_ratio"] < va["scale_ratio"]


def test_a_part_with_one_position_has_no_size_and_the_ratio_is_absent():
    """RULE: one position has no extent; nothing is put in its place."""
    dot = _contour(4, ["d0"])
    pos = dict(POS, d0=(60.0, 30.0, 0.0))
    members = MEMBERS + [(14, dot)]
    whole = _whole(9, members)
    v = _values(whole, 14, members, pos)
    assert v["scale_ratio"] is None
    assert v["rel_dist"] is not None and v["polar_angle"] is not None
    centre, extent = Binder._footprint(dot, pos)
    assert centre == (60.0, 30.0, 0.0) and extent is None


def test_a_declared_size_is_read_when_both_ends_declare_one():
    """RULE: a size a lens measured outranks a size inferred from positions.
    (The legacy reading, kept: log2(child scale / parent scale).)"""
    a = _contour(1, ["a0", "a1"], scale=2)
    b = _contour(2, ["b0", "b1"], scale=2)
    members = [(11, a), (12, b)]
    whole = _whole(9, members, scale=8)
    assert _values(whole, 11, members, POS)["scale_ratio"] == pytest.approx(-2.0)
    # one end only: not two sizes, so the footprints are used instead
    whole0 = _whole(9, members, scale=0)
    assert _values(whole0, 11, members, POS)["scale_ratio"] == pytest.approx(
        math.log2(20.0 / (0.5 * math.hypot(100.0, 5.0))))


class _Capture(Binder):
    """A Binder whose channels record their candidates instead of writing."""

    def __init__(self):            # no concept graph: the writer is all we run
        self.seen = {}
        self._step_extraction_scale = R

    def _process_channel(self, edge_type, candidates):
        self.seen[edge_type] = list(candidates)


def _v1(k, anchor, tag, scale=0, z=1.0):
    return Detection(k=k, source_layer=0, scale=scale, sigma=_sig(tag),
                     alpha=anchor, z=z, t=0, rho=())


def test_co_located_size_ratio_is_absent_when_no_lens_declares_a_size():
    """RULE: one ruler read twice is not a measurement.

    CO_LOCATED joins detections pooled over the SAME anchor.  Their
    footprints are that anchor's footprint, so a ratio taken from it is 0
    for every pair of every image.  The slot is ABSENT unless both lenses
    declare a size of their own.
    """
    color, texture = _v1("Color", 7, 1), _v1("Texture", 7, 2)
    b = _Capture()
    b._run_co_located((color, texture), {color: 1, texture: 2})
    (cand,) = b.seen[EdgeType.CO_LOCATED]
    ep = cand[4]
    assert ep.class_ == "EDGE_CO_LOCATED"
    assert ep.quantized[0] == UNKNOWN             # scale_ratio
    assert ep.quantized[1] != UNKNOWN             # salience_ratio is measured

    # two declared sizes are two measurements: |log2(2 / 8)| = 2
    color, texture = _v1("Color", 7, 1, scale=2), _v1("Texture", 7, 2, scale=8)
    b = _Capture()
    b._run_co_located((color, texture), {color: 1, texture: 2})
    (cand,) = b.seen[EdgeType.CO_LOCATED]
    spec = S.edge_slot_spec("EDGE_CO_LOCATED", "scale_ratio")
    assert cand[4].quantized[0] == float(S.quantise(spec, 2.0))


# --------------------------------------------------------------------------
# A3 -- absence on the bins path
# --------------------------------------------------------------------------

def test_missing_geometry_is_absent_on_the_bins_path_not_a_neutral_triple():
    """RULE: absence is never a neutral number.

    `_containment_geometry` returned (0, 0, 4) -- "at the centre, bearing 0,
    equal size" -- when a position was missing.  It now returns the unit
    table's quantisation of the same values the edge is built from.
    """
    no_positions = {}
    bins, cont = Binder._containment_geometry(
        WHOLE, 11, dict(MEMBERS), no_positions, R)
    assert bins == (UNKNOWN, UNKNOWN, UNKNOWN)
    assert cont == (None, None, None)
    assert Binder._containment_bins(
        WHOLE, 11, dict(MEMBERS), no_positions, R) == bins
    # an unknown child is absent too
    bins, _ = Binder._containment_geometry(WHOLE, 99, dict(MEMBERS), POS, R)
    assert bins == (UNKNOWN, UNKNOWN, UNKNOWN)

    # with geometry, the two paths are one path
    values = _values(WHOLE, 12, MEMBERS, POS)
    bins, cont = Binder._containment_geometry(WHOLE, 12, dict(MEMBERS), POS, R)
    assert bins == S.edge_quantise("EDGE_CURVE_CONTAINMENT", values)
    assert cont == (values["rel_dist"], values["polar_angle"],
                    values["scale_ratio"])


def test_an_edge_with_no_geometry_is_still_emitted():
    """RULE: the membership is a fact even when its geometry is not measured.

    No gate drops the edge: the writer hands the channel one candidate per
    member, at full signal, whether or not its slots are measured.
    """
    solo = _whole(9, [(11, A)])
    b = _Capture()
    b._run_curve_containment((solo, A), {solo: 50, A: 11}, POS)
    (cand,) = b.seen[EdgeType.CURVE_CONTAINMENT]
    assert cand[:3] == (50, 11, 1.0)
    assert cand[4].quantized == (UNKNOWN, UNKNOWN, UNKNOWN)

    b = _Capture()
    b._run_curve_containment((WHOLE, A, B), {WHOLE: 50, A: 11, B: 12}, POS)
    got = {c[1]: c[4].quantized for c in b.seen[EdgeType.CURVE_CONTAINMENT]}
    assert set(got) == {11, 12}
    assert all(UNKNOWN not in q for q in got.values())


# --------------------------------------------------------------------------
# A4 -- the two tables
# --------------------------------------------------------------------------

FLAG_CLASSES = ("EDGE_CONTOUR_ORIENTATION", "EDGE_JUNCTION_ORIENTATION",
                "EDGE_REGION_CONTENT")


def test_the_two_edge_tables_describe_every_slot_as_the_same_kind_of_thing():
    """RULE: an index is an index in both tables.

    EDGE_SCHEMAS said "linear over 8 bins" of three membership flags the
    unit table calls a two-level lookup key, and "ordinal" of a polarity
    class the unit table calls a lookup key.
    """
    assert S.edge_table_disagreements() == []
    S.verify_schema_consistency()
    for cls in FLAG_CLASSES:
        (slot,), (spec,) = S.EDGE_SCHEMAS[cls], S.EDGE_UNIT_TABLE[cls]
        assert (slot.tag, slot.domain, slot.n_bins) == ("containment", "categorical", 2.0)
        assert (spec.quantiser, spec.n_bins) == ("lookup_key", 2)
        assert not slot.is_identity          # nothing to veto: it cannot differ
        assert S.CONSTITUTIONAL_EDGE_RANGES[cls] == (2.0,)
    by_tag = {s.tag: s for s in S.EDGE_SCHEMAS["EDGE_ORIENTATION_CONTINUITY"]}
    assert by_tag["polarity_rel"].domain == "categorical"
    by_tag = {s.tag: s for s in S.EDGE_SCHEMAS["EDGE_OCCLUSION"]}
    assert by_tag["source"].domain == "categorical"
    assert S.edge_slot_spec("EDGE_OCCLUSION", "source").quantiser == "lookup_key"


@pytest.mark.parametrize("cls", FLAG_CLASSES)
def test_a_membership_flag_has_no_graded_population(cls):
    """RULE: an index carries no graded evidence.

    The flag's population is the uniform activity (the overlap reads it as
    0), and it is not a graded axis.  ONE READER STILL COUNTS IT AS A
    MEASURED AXIS: `edge_measured_axes`, while
    EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS is off.  That switch is pinned
    but could not be measured in the ETH-80 harness (it moves `mcs_score` /
    `mcs_similarity`, which the harness does not read: REPORT.md, stage 4)
    and was left off as the owner's decision.  A4 does not change it, so A4
    does NOT stop a bare flag from being counted as a measured axis.
    """
    ep = S.edge_params(cls, {"containment": 1.0})
    assert ep.quantized == (1.0,)
    assert ep.population == ((0.5, 0.5),)
    assert S.edge_graded_axes(ep, ep) == 0
    assert P.EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS is False
    assert S.edge_measured_axes(ep, ep) == 1          # unchanged by A4


@pytest.mark.parametrize("cls", FLAG_CLASSES)
def test_the_existing_switch_removes_the_flag_from_the_measured_axes(cls, monkeypatch):
    """The existing switch still does what it did; A4 leaves it alone."""
    monkeypatch.setattr(P, "EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS", True)
    ep = S.edge_params(cls, {"containment": 1.0})
    assert S.edge_measured_axes(ep, ep) == 0


def test_occlusion_source_is_an_index_with_no_graded_population():
    """RULE: the same, for the occlusion cue class (T-junction / depth)."""
    values = {"depth": 1.0, "stack_depth": 1.0, "source": 1.0,
              "depth_units": 0.5}
    ep = S.edge_params("EDGE_OCCLUSION", values)
    names = [s.name for s in S.EDGE_UNIT_TABLE["EDGE_OCCLUSION"]]
    i = names.index("source")
    assert ep.quantized[i] == 1.0
    assert len(set(ep.population[i])) == 1            # uniform: no information
    assert S.edge_graded_axes(ep, ep) == len(names) - 1


# --------------------------------------------------------------------------
# A5 -- the PEAK_CHAIN closure axis
# --------------------------------------------------------------------------

def _chains():
    from lenses.chain_lens import ChainLens
    yy, xx = np.mgrid[0:96, 0:96]
    disc = np.where((xx - 48) ** 2 + (yy - 48) ** 2 < 28 ** 2, 0.9, 0.1)
    return ChainLens().step(0, None, None, {}, None, {},
                            extraction_scale=8.0,
                            luminance=disc.astype(np.float64))


def test_peak_chain_writes_every_axis_of_its_population_layout():
    """RULE: a population axis the writer can measure is written.

    The layout declares four axes (theta_dist, curvature, length, closure);
    the lens wrote three, so every chain comparison took the aligned path
    that skips the missing axis.  Closure is topological -- a chain returns
    to its start or it does not -- so its activity is one-hot on the level
    the bins already name.
    """
    from core.population import (aligned_population_similarity,
                                 population_similarity)
    from core.registries import POPULATION_AXES

    axes = POPULATION_AXES["PEAK_CHAIN"]
    assert [a.name for a in axes] == ["theta_dist", "curvature", "length",
                                      "closure"]
    chains = _chains()
    assert len(chains) >= 2
    for d in chains:
        pop = d.sigma.population
        assert [len(v) for v in pop] == [a.n_bins for a in axes]
        level = int(d.sigma.quantized[3])
        assert pop[3] == tuple(1.0 if j == level else 0.0 for j in range(2))
    a, b = chains[0].sigma.population, chains[1].sigma.population
    # the strict comparison is now defined (it returned 0.0 on a 3-axis
    # vector) and the closure axis is part of it
    strict = population_similarity(a, b, axes)
    assert strict > 0.0
    assert strict == pytest.approx(aligned_population_similarity(a, b, axes))
    three = population_similarity(a[:3], b[:3], axes)
    assert three == 0.0


def test_a_closed_and_an_open_chain_do_not_overlap_on_the_closure_axis():
    """RULE: two levels of a topological property share nothing."""
    from core.population import axis_similarity
    open_, closed = (1.0, 0.0), (0.0, 1.0)
    assert axis_similarity(open_, open_) == 1.0
    assert axis_similarity(closed, closed) == 1.0
    assert axis_similarity(open_, closed) == 0.0


# --------------------------------------------------------------------------
# the same two rules, in three more writers (scope extended: see REPORT.md)
# --------------------------------------------------------------------------

def _oriented(tag, anchor, theta_bin):
    """A Contour whose mean orientation is bin `theta_bin` of 24 over pi."""
    return Detection(
        k="Contour", source_layer=2, scale=0,
        sigma=TypedSignature(kind="CANONICAL", class_="TEST", relation="T",
                             quantized=(float(tag), float(theta_bin)),
                             population=()),
        alpha=anchor, z=1.0, t=0, rho=())


def _slots(ep):
    names = [s.name for s in S.EDGE_UNIT_TABLE[ep.class_]]
    return dict(zip(names, ep.quantized))


def test_radiality_alignment_is_absent_because_the_hub_is_defined_by_it():
    """RULE: a quantity compared with itself is not a measurement.

    The RADIALITY writer defines the hub as the intersection of the two
    elements' own orientation lines and then asks how well each element
    points at the hub.  It points at it exactly, by construction: the
    deviation was 0 (to 1e-14) on every one of the 1,691 RADIALITY edges of
    the 79 graphs.  The slot is ABSENT; the hub's bearing, its distance
    ratio and the spoke separation are real measurements and stay.
    """
    a = _oriented(1, "p", 5)       # 41.25 deg, through (0, 0)
    b = _oriented(2, "q", 17)      # 131.25 deg, through (20, 0)
    pos = {"p": (0.0, 0.0, 0.0), "q": (20.0, 0.0, 0.0)}
    bd = _Capture()
    bd._run_radiality((a, b), {a: 1, b: 2}, pos)
    (cand,) = bd.seen[EdgeType.RADIALITY]
    got = _slots(cand[4])
    assert got["alignment"] == UNKNOWN
    assert all(v != UNKNOWN for k, v in got.items() if k != "alignment")


def test_spirality_expansion_is_absent_because_the_hub_is_the_midpoint():
    """RULE: a quantity compared with itself is not a measurement.

    The SPIRALITY writer puts the hub at the MIDPOINT of the pair, so the
    two radii are equal and the log radius ratio -- the expansion -- is 0
    for every pair of every image (939 of 939 edges).  ABSENT.
    """
    a = _oriented(1, "p", 5)
    b = _oriented(2, "q", 5)
    pos = {"p": (0.0, 0.0, 0.0), "q": (20.0, 0.0, 0.0)}
    bd = _Capture()
    bd._run_spirality((a, b), {a: 1, b: 2}, pos)
    (cand,) = bd.seen[EdgeType.SPIRALITY]
    got = _slots(cand[4])
    assert got["expansion"] == UNKNOWN
    assert all(v != UNKNOWN for k, v in got.items() if k != "expansion")


def _segment(tag, anchor, theta_bin, idx, n=2):
    return Detection(
        k="Orientation", source_layer=0, scale=0,
        sigma=TypedSignature(kind="CANONICAL", class_="ORIENTATION_SEGMENT",
                             relation="SEGMENT",
                             # slot 2 is the chain bin the writer buckets on
                             quantized=(float(theta_bin), 1.0, 0.0, float(tag)),
                             population=()),
        alpha=anchor, z=1.0, t=0,
        rho=(("entity", "E", 0, "+"), ("segment", idx, 0, "+"),
             ("num_segments", n, 0, "+"), ("closed", False, 0, "+")))


def test_curvature_chord_is_measured_between_the_two_segments_or_is_absent():
    """RULE: absence is never a neutral number.

    The chord between two boundary segments was looked up through their
    MEMBER anchors; a segment has none, so the lookup found nothing and the
    writer fell back to chord 0.0 and rate 0.0 -- on all 87 CURVATURE edges
    of the 79 graphs.  A segment is located at its own anchor: two segments
    30 apart (3 wavelengths at R = 10) have a chord of 3 and a rate of
    |turn| / 3.  With no position, both are ABSENT.
    """
    a, b = _segment(1, "s0", 2, 0), _segment(2, "s1", 10, 1)
    ids = {a: 1, b: 2}
    pos = {"s0": (0.0, 0.0, 0.0), "s1": (30.0, 0.0, 0.0)}
    bd = _Capture()
    bd._run_curvature((a, b), ids, pos)
    (cand,) = bd.seen[EdgeType.CURVATURE]
    ep = cand[4]
    got = _slots(ep)
    spec = S.edge_slot_spec("EDGE_CURVATURE", "chord_len")
    assert got["chord_len"] == float(S.quantise(spec, 3.0))
    assert got["curv_rate"] != UNKNOWN and got["signed_turn"] != UNKNOWN

    bd = _Capture()
    bd._run_curvature((a, b), ids, {"elsewhere": (1.0, 1.0, 0.0)})
    (cand,) = bd.seen[EdgeType.CURVATURE]
    got = _slots(cand[4])
    assert got["chord_len"] == UNKNOWN and got["curv_rate"] == UNKNOWN
    assert got["signed_turn"] != UNKNOWN          # the turn needs no position
