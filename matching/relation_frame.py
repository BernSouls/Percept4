"""matching/relation_frame.py -- the FRAME / ATTRIBUTE partition of a graph.

[STAGE 5 / C1]  A graph is read in two parts.

  FRAME      nodes that are pieces of SHAPE: a contour, a corner, a junction,
             an oriented edge element, a curvature peak.  Their relations
             join two different places (continuity, co-circularity, a
             junction and its arms, a chain and its peaks).  In the design
             this partition was written for, correspondence would be
             decided here, by relations.

  ATTRIBUTE  nodes that say what a place IS LIKE: its colour, its texture,
             its position, its boundary statistics, its polarity.  In that
             design they would attach to the frame after it is matched and
             never decide a correspondence.

WHY AN EXPLICIT LIST AND NOT `source_layer`.  `Detection.source_layer` is the
tier that PRODUCED a detection (0 = the V1 floor, 2 = a higher lens), not the
kind of thing it describes, and the two cut across each other on both sides:

  * Orientation and ContourFragment are tier 0 and are shape: Orientation
    carries the ORIENTATION_CONTINUITY / BEND / JUNCTION relations, which are
    the densest structural relations in a graph.
  * Polarity, Surface, TextureBoundary and BorderOwnership are higher-tier
    lenses and are attributes: a contrast sign, a surface's colour and
    texture, a texture step, a side assignment.

Splitting on the tier would put the first group among the attributes and the
second in the frame.  The list below is therefore written out, by lens type,
and pinned (tests_matcher/test_relation_frame_partition.py): adding a lens
type to the front end reddens the pin until someone decides which side it is
on.  A lens type in neither list is treated as ATTRIBUTE -- it can then never
drive growth, which is the safe side to be wrong on.

Nothing here reads a switch or a matcher; the module is a declaration plus
three small readers.  Its ONLY user is the Phase B feasibility census
(diagnostics/phase5_feasibility.py).  NO RELATION-GROWN MATCHER EXISTS: the
census fired its pre-registered stop rule (the reachable set is 16 % of the
frame at the median, against a floor of 25 %), so none was built.  See
diagnostics/REPORT.md, stage 5.
"""
from __future__ import annotations

from typing import Any, FrozenSet, Tuple

from core.schema import (
    EDGE_IDENTITY_SLOTS,
    EDGE_UNIT_TABLE,
    UNKNOWN,
    U30_EDGE_IDENTITY_SLOTS,
    U30_EDGE_UNIT_TABLE,
    EdgeParamsMixture,
    _is_u30_edge_params,
)

#: lens types whose nodes are pieces of shape
FRAME_LENS_TYPES: FrozenSet[str] = frozenset({
    "Contour",            # a chain of rim elements, with its curvature profile
    "ContourFragment",    # a curved fragment: curvature, tangent, radius
    "Corner",             # a turn between two arms
    "Junction",           # a meeting of arms (also the L / T / Y aliases)
    "T",
    "Y",
    "CurveComplex",       # a bundle of contours and junctions
    "Orientation",        # an oriented edge element
    "CONTOUR_PEAK",       # a peak of the contour field
    "PEAK_CHAIN",         # a chain of field peaks
    "EndStopped",         # a line end
    "IllusoryContour",    # a completed contour between two inducers
    "RegionShape",        # the outline shape of a region
})

#: lens types whose nodes say what a place is like
ATTRIBUTE_LENS_TYPES: FrozenSet[str] = frozenset({
    "Color",
    "Texture",
    "Position",
    "Boundary",
    "Polarity",
    "TextureBoundary",    # a texture step: an attribute contrast, not a shape
    "BorderOwnership",    # which side owns a border
    "Surface",            # a surface patch's colour / texture summary
    "Onset",              # a temporal event
})


def is_frame(lens_type: str) -> bool:
    """True when `lens_type` is a piece of shape.  Unknown types are not."""
    return lens_type in FRAME_LENS_TYPES


def measured_axes(ep: Any) -> Tuple[int, ...]:
    """Slot indices of ONE edge's parameters that hold a graded measurement.

    A slot counts when it is not UNKNOWN, not an identity (NAP) slot and not
    a lookup key: the same three exclusions `core.schema.edge_graded_axes`
    applies to a pair, applied to one side.  A mixture is read through its
    DOMINANT prototype (`EdgeParamsMixture.dominant`: the heaviest, ties to
    the earliest).
    """
    if ep is None:
        return ()
    if isinstance(ep, EdgeParamsMixture):
        if not ep.protos:
            return ()
        ep = ep.dominant
    quantized = getattr(ep, "quantized", None)
    if not quantized:
        return ()
    u30 = _is_u30_edge_params(ep)
    if u30:
        categorical = set(U30_EDGE_IDENTITY_SLOTS.get(ep.class_, ()))
        specs = U30_EDGE_UNIT_TABLE.get(ep.class_, ())
    else:
        ids = getattr(ep, "identity_slots", None)
        categorical = set(ids if ids is not None
                          else EDGE_IDENTITY_SLOTS.get(ep.class_, ()))
        specs = EDGE_UNIT_TABLE.get(ep.class_, ())
    if len(specs) == len(quantized):
        categorical.update(
            i for i, s in enumerate(specs) if s.quantiser == "lookup_key")
    return tuple(
        i for i, v in enumerate(quantized)
        if v != UNKNOWN and i not in categorical)


def has_measured_axis(ep: Any) -> bool:
    """True when the edge carries at least one graded measurement."""
    return bool(measured_axes(ep))
