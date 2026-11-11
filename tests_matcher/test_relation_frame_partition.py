"""[STAGE 5 / C1] The FRAME / ATTRIBUTE partition is an explicit list.

The partition decides which nodes may ever carry a correspondence, so it is
pinned literally: a lens type added to the front end, or moved from one side
to the other, reddens this file until someone decides it.
"""
from __future__ import annotations

import core.schema as S
from core.schema import UNKNOWN, EdgeParamsMixture
from matching.relation_frame import (ATTRIBUTE_LENS_TYPES, FRAME_LENS_TYPES,
                                     has_measured_axis, is_frame,
                                     measured_axes)

#: the lens types of the two tiers, as the ETH-80 front end emits them
#: (diagnostics/eth80_frontend.py, `det_counts`): the V1 floor and the higher
#: lenses.  Node lens types of the higher tier, not lens names: the Peak lens
#: emits CONTOUR_PEAK nodes, the Chain lens PEAK_CHAIN nodes, the Junction
#: lens Junction / T / Y nodes.
V1_TIER = {"Boundary", "Color", "ContourFragment", "Onset", "Orientation",
           "Position", "Texture"}
HIGHER_TIER = {"BorderOwnership", "PEAK_CHAIN", "Contour", "Corner",
               "CurveComplex", "EndStopped", "IllusoryContour", "Junction",
               "T", "Y", "CONTOUR_PEAK", "Polarity", "RegionShape", "Surface",
               "TextureBoundary"}


def test_the_partition_is_the_pinned_list():
    assert FRAME_LENS_TYPES == {
        "Contour", "ContourFragment", "Corner", "Junction", "T", "Y",
        "CurveComplex", "Orientation", "CONTOUR_PEAK", "PEAK_CHAIN",
        "EndStopped", "IllusoryContour", "RegionShape"}
    assert ATTRIBUTE_LENS_TYPES == {
        "Color", "Texture", "Position", "Boundary", "Polarity",
        "TextureBoundary", "BorderOwnership", "Surface", "Onset"}
    assert not (FRAME_LENS_TYPES & ATTRIBUTE_LENS_TYPES)
    # every lens type the front end emits is on exactly one side
    assert V1_TIER | HIGHER_TIER == FRAME_LENS_TYPES | ATTRIBUTE_LENS_TYPES


def test_the_tier_does_not_decide_the_side():
    """WHY NOT `source_layer`: the producing tier cuts across the partition
    in both directions.  Splitting on it would move these six lens types to
    the wrong side."""
    assert V1_TIER & FRAME_LENS_TYPES == {"Orientation", "ContourFragment"}
    assert HIGHER_TIER & ATTRIBUTE_LENS_TYPES == {
        "Polarity", "Surface", "TextureBoundary", "BorderOwnership"}


def test_an_unknown_lens_type_is_not_frame():
    """The safe side to be wrong on: it can then never drive growth."""
    assert not is_frame("SomethingNew")
    assert is_frame("Contour") and not is_frame("Color")


def test_measured_axes_are_the_graded_slots_that_were_measured():
    """A slot counts when it is not UNKNOWN, not an identity slot and not a
    lookup key."""
    # a membership flag: an index, nothing graded
    flag = S.edge_params("EDGE_CONTOUR_ORIENTATION", {"containment": 1.0})
    assert measured_axes(flag) == () and not has_measured_axis(flag)
    # containment with no geometry: nothing measured
    absent = S.edge_params("EDGE_CURVE_CONTAINMENT", {
        "rel_dist": None, "polar_angle": None, "scale_ratio": None})
    assert measured_axes(absent) == ()
    # two of three measured
    part = S.edge_params("EDGE_CURVE_CONTAINMENT", {
        "rel_dist": 1.0, "polar_angle": None, "scale_ratio": -1.0})
    assert measured_axes(part) == (0, 2)
    # identity slots and lookup keys are never graded axes
    names = [s.name for s in S.EDGE_UNIT_TABLE["EDGE_JUNCTION_CONTOUR"]]
    jc = S.edge_params("EDGE_JUNCTION_CONTOUR", {
        "dist": 1.0, "incidence": 0.5, "topology_class": 2.0})
    assert measured_axes(jc) == (names.index("dist"), names.index("incidence"))
    assert measured_axes(None) == ()


def test_a_mixture_is_read_through_its_dominant_prototype():
    light = S.edge_params("EDGE_CURVE_CONTAINMENT", {
        "rel_dist": 1.0, "polar_angle": 1.0, "scale_ratio": -1.0})
    heavy = S.edge_params("EDGE_CURVE_CONTAINMENT", {
        "rel_dist": None, "polar_angle": None, "scale_ratio": None})
    mix = EdgeParamsMixture([light, heavy], [1.0, 3.0], 0)
    assert mix.dominant is heavy
    assert measured_axes(mix) == ()
    assert measured_axes(EdgeParamsMixture([light, heavy], [3.0, 1.0], 0)) == (0, 1, 2)
    assert heavy.quantized == (UNKNOWN,) * 3
