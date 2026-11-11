"""core/schema.py -- edge schemas, edge params, polymorphic matching.

MOVE ONLY: statements extracted verbatim from primitives.py by
_reorg_move2_schema.py, preserving their order (module-level registration order
is part of the persisted-memory contract). primitives.py re-exports from here.

Zero project imports except core.hashing and parameters (a leaf module).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from dataclasses import replace as _dc_replace  # noqa: F401  (moved import alias)
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from core.hashing import clamp, hash_128, hash_64, serialize_bytes, serialize_tuple
from core.population import (  # noqa: F401
    PopAxis,
    population_similarity,
    soft_bump_bins,
)
from core.registries import (  # noqa: F401
    SlotSpec,
    bin_at_centre,
    population_centre,
    quantise,
)

import parameters as P
from parameters import EPSILON

CONSTITUTIONAL_EDGE_RANGES: Dict[str, Tuple[float, ...]] = {
    # (gap_len, perp_offset, collinearity_error, signed_curv_delta, src_arc, dst_arc)
    # [Pass 21] (gap, perp, cocircular_asym, signed_turn, tail_role,
    #  head_role, polarity_relation). Slot 3 is circular on cycle 8,
    #  so its range is 4.0 under the N_HALF convention -- NOT 8.0.
    "EDGE_ORIENTATION_CONTINUITY": (8.0, 8.0, 8.0, 4.0, 3.0, 3.0, 3.0),
    # (normal_angle, overlap_len, curv_src, curv_dst)
    "EDGE_CONTACT": (12.0, 9.0, 9.0),
    # (color_dist, texture_dist, orient_delta, contrast_step)
    "EDGE_SURFACE_GROUPING": (9.0, 9.0, 9.0, 9.0),
    # (signed_turn, chord_len, curv_rate)
    "EDGE_CURVATURE": (8.0, 8.0, 8.0),
    # (phase_diff, polarity_congruency, salience_ratio)
    # [Pass 21] (scale_ratio, salience_ratio)
    "EDGE_CO_LOCATED": (9.0, 9.0),
    # (angle_delta, bisector_theta, spatial_dist)
    "EDGE_ORIENTATION_BEND": (8.0, 12.0, 8.0),
    "EDGE_ORIENTATION_JUNCTION": (8.0, 12.0, 8.0),
    # [Pass 12] Relational edge types E7–E11
    "EDGE_PARALLEL": (9.0, 9.0, 9.0),
    "EDGE_SYMMETRY": (8.0, 12.0, 8.0),
    "EDGE_BORDER_OWNERSHIP": (8.0, 12.0, 8.0),
    # [Pass 15] High-tier binder edge parameter schemas
    "EDGE_CONTOUR_CONTINUITY": (8.0, 4.0, 8.0),
    "EDGE_JUNCTION_CONTOUR": (8.0, 12.0, 7.0),
    # [Pass 22] CO_MOTION: (mean_speed, dir_theta, speed_diff, dir_diff)
    "EDGE_CO_MOTION": (8.0, 12.0, 8.0, 8.0),
    "EDGE_CONTOUR_ORIENTATION": (8.0,),
    "EDGE_JUNCTION_ORIENTATION": (8.0,),
    # [Pass 18 / C4+C6] (rel_dist, polar_angle, log_scale_ratio). The polar
    # slot is 24 bins on a circle, hence range 12.0 under N_HALF; registered
    # in CIRCULAR_EDGE_CYCLE_LENGTHS below.
    "EDGE_CURVE_CONTAINMENT": (8.0, 12.0, 8.0),
    "EDGE_PART_CONTAINMENT": (8.0, 12.0, 8.0),
    "EDGE_REGION_CONTENT": (8.0,),
    # [Stage 3] Occlusion: (rank_gap | depth_step, stack_depth | 8 = none, source
    # 0 = T-junction, 1 = depth). Was arity 2 (dz_bin, conf_bin); the depth-DAG
    # path now carries the longest-path rank gap and the weakly-connected stack
    # depth. No circular slot, so no CIRCULAR_EDGE_CYCLE_LENGTHS entry.
    "EDGE_OCCLUSION": (8.0, 9.0, 2.0),
    # [Pass 16] Relational edge types
    "EDGE_CURVATURE_SIGN_CONTINUITY": (8.0, 8.0),
    "EDGE_COCIRCULAR": (9.0, 9.0),
    "EDGE_AMODAL_LINK": (8.0, 8.0),
    "EDGE_SHARED_BOUNDARY": (8.0, 8.0),
    # [Pass 23] values 42-51. Circular slots satisfy range == cycle/2;
    # _verify_edge_circular_ranges enforces this at import.
    "EDGE_CONCENTRICITY": (8.0, 8.0, 8.0, 8.0),
    "EDGE_TEXTURE_BOUNDARY": (8.0, 8.0, 8.0, 8.0),
    "EDGE_SURFACE_FOLD": (12.0, 6.0, 3.0, 8.0),
    "EDGE_RADIALITY": (12.0, 8.0, 8.0, 12.0),
    "EDGE_SPIRALITY": (6.0, 8.0, 3.0, 8.0),
    "EDGE_COLOR_OPPONENT_BORDER": (12.0, 8.0, 8.0, 8.0),
    "EDGE_MOTION_OPPOSITION": (12.0, 8.0, 12.0, 8.0),
    "EDGE_CROSS_SCALE_IDENTITY": (8.0, 12.0, 8.0, 8.0),
    "EDGE_BINOCULAR_DISPARITY": (8.0, 8.0, 4.0, 8.0),
    # [Pass 23 audit] slot 3 is the AXIAL (mod-pi) parent orientation, so it
    # takes a 12-cycle and range 6.0 -- not a 24-cycle. The draft declared a
    # 24-cycle here, which made adjacent axial values score as maximally
    # different under the min(d, cycle-d) wrap.
    "EDGE_LINE_TERMINATION": (5.0, 8.0, 8.0, 6.0),
}

CIRCULAR_EDGE_CYCLE_LENGTHS: Dict[str, Dict[int, int]] = {
    "EDGE_CONTACT": {0: 24},
    "EDGE_CONTOUR_CONTINUITY": {1: 8},
    "EDGE_JUNCTION_CONTOUR": {1: 24},
    "EDGE_CO_MOTION": {1: 24},
    "EDGE_ORIENTATION_CONTINUITY": {3: 8},
    "EDGE_ORIENTATION_BEND": {1: 24},
    "EDGE_ORIENTATION_JUNCTION": {1: 24},
    "EDGE_SYMMETRY": {1: 24},
    # [P2] EDGE_BORDER_OWNERSHIP's 24-cycle entry is GONE: the slot that
    # carried it (the "bearing") never held a bearing -- the writer put a
    # distance there.  The class has no circular slot any more.
    # [Pass 18 / C4+C6] index 1 is the polar_angle slot of the containment
    # edges: 24 bins on a circle, range 12.0 under the N_HALF convention.
    "EDGE_CURVE_CONTAINMENT": {1: 24},
    "EDGE_PART_CONTAINMENT": {1: 24},
    # [Pass 23] slot 1 is the AXIAL (mod-pi) fold axis -> 12-cycle.
    "EDGE_SURFACE_FOLD": {1: 12},
    "EDGE_RADIALITY": {0: 24},
    "EDGE_COLOR_OPPONENT_BORDER": {0: 24},
    "EDGE_MOTION_OPPOSITION": {0: 24, 2: 24},
    # [Pass 23 audit] axial (mod-pi) parent orientation -> 12-cycle.
    "EDGE_LINE_TERMINATION": {3: 12},
    # [Pass 23] slot 1 is phase_alignment: 24 bins on a circle (phase_bin
    # is computed mod 24 in _run_cross_scale_identity), range 12.0 under
    # the N_HALF convention — satisfies _verify_edge_circular_ranges.
    "EDGE_CROSS_SCALE_IDENTITY": {1: 24},
}

def get_edge_similarity_ranges(class_name: str) -> Tuple[float, ...] | None:
    """Return constitutional ranges for edge parameter schemas, or None if unregistered."""
    return CONSTITUTIONAL_EDGE_RANGES.get(class_name)

UNKNOWN: float = -1.0

EDGE_IDENTITY_SLOTS: Dict[str, Tuple[int, ...]] = {
    # (gap, perp, cocircular_asym, signed_turn, tail_role, head_role, polarity_relation)
    # Slots 4 & 5: bipole tail/head roles (0=terminal, 1=weak, 2=strong interior)
    # Slot 6: polarity relation (0=flipped, 1=none, 2=same)
    "EDGE_ORIENTATION_CONTINUITY": (4, 5, 6),
    # (dist, incidence, topo) -> Slot 2: topology class (0..6: L, T, Y, X, Star, etc.)
    "EDGE_JUNCTION_CONTOUR": (2,),
    # (dihedral, fold_axis, convexity, sharpness) -> Slot 2: convexity (0=concave, 1=flat, 2=convex)
    "EDGE_SURFACE_FOLD": (2,),
    # (pitch, consistency, chirality, expansion) -> Slot 2: chirality (0=cw, 1=ambiguous, 2=ccw)
    "EDGE_SPIRALITY": (2,),
    # (junction, sharpness, curvature, parent) -> Slot 0: junction type (0=end, 1=corner, 2=collinear, 3=T, 4=Y)
    "EDGE_LINE_TERMINATION": (0,),
    # (disp, grad, cell_type, conf) -> Slot 2: near/far arrangement (0..3)
    "EDGE_BINOCULAR_DISPARITY": (2,),
    # (dist, congruence) -> Slot 1: congruence (0=congruent, 7=opposite)
    "EDGE_SHARED_BOUNDARY": (1,),
}

EDGE_REVERSAL: Dict[str, Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]] = {
    # (gap, perp, cocircular_asym, signed_turn, tail_role, head_role, polarity_relation)
    # Slots 4 & 5 swap roles. Slot 3 (signed turn on cycle 8) reflects: (8 - b) % 8.
    "EDGE_ORIENTATION_CONTINUITY": (((4, 5),), (3,)),
    # Slot 0 is normal_angle on cycle 24: reversing endpoints rotates normal by 180 deg (+12 bins).
    "EDGE_CONTACT": (((),), (0,)),
    # Slot 1 is signed turn on cycle 8: reflects across center.
    "EDGE_CONTOUR_CONTINUITY": (((),), (1,)),
    # Slot 0 is signed turn on cycle 8: reflects across center.
    "EDGE_CURVATURE": (((),), (0,)),
}

@dataclass(frozen=True)
class Slot:
    """Single relational parameter slot in an EdgeParams schema.

    domain:
        "linear"      — metric distance, ratio, or intensity; Gaussian similarity, no wrap.
        "circular"    — angular; wraps at n_bins (axial mod pi or full 2pi).
        "ordinal"     — discrete ordered steps.
        "categorical" — unordered discrete classes (exact match only).

    reversal: transformation applied when edge endpoints are swapped:
        "same"             — symmetric slot; unchanged.
        "reflect_linear"   — reflects across linear range: (n_bins - 1.0) - v.
        "reflect_circular" — reflects across circular cycle: (cycle - v) % cycle.
        ("swap", target)   — swaps value with slot at index `target`.
        ("rotate", offset) — adds `offset` bins modulo cycle length.

    is_identity: True -> Non-Accidental Property (NAP) hard gate. Differences >= 0.5 yield similarity 0.0.
    """
    tag: str
    domain: Literal["linear", "circular", "ordinal", "categorical"]
    n_bins: float
    reversal: Union[
        Literal["same", "reflect_linear", "reflect_circular"],
        Tuple[Literal["swap"], int],
        Tuple[Literal["rotate"], int],
    ]
    is_identity: bool = False

EDGE_SCHEMAS: Dict[str, Tuple[Slot, ...]] = {
    # 7 slots: dist, perp, cocirc_asym, turn, tail_role, head_role, polarity_relation
    # [Stage 2 / Note 3] Slot 0 renamed gap -> dist so the along-axis
    # bipole gap shares its tag with CC's endpoint distance. Intentional
    # semantic approximation: OC's slot 0 is the along-axis gap between
    # bipole anchors; CC's slot 0 is endpoint distance. Both bin a
    # physical separation to 0..7 with similar normalization, so the
    # cross-schema comparison is approximately meaningful (damped by the
    # cross-family prior). Positional matching by binder.py is unaffected
    # (it emits by position, never by tag).
    "EDGE_ORIENTATION_CONTINUITY": (
        Slot("dist",            "linear",      8.0, "same"),
        Slot("perp",            "linear",      8.0, "same"),
        Slot("cocirc_asym",     "linear",      8.0, "same"),
        Slot("turn",            "circular",    8.0, "reflect_circular"),
        Slot("tail_role",       "categorical", 3.0, ("swap", 5), is_identity=True),
        Slot("head_role",       "categorical", 3.0, ("swap", 4), is_identity=True),
        Slot("polarity_rel",    "ordinal",     3.0, "same",      is_identity=True),
    ),
    # 3 slots: normal_angle (24-cycle), dist, depth_step
    # [P2 / MIS-READ FIX] The second slot was called "overlap_len" but the
    # writer never measured an overlap length: it measured the SEPARATION
    # of the two entities.  The slot is renamed to say so.
    "EDGE_CONTACT": (
        Slot("normal_angle",    "circular",   24.0, ("rotate", 12)),
        Slot("dist",            "linear",      9.0, "same"),
        Slot("depth_step",      "ordinal",     9.0, "reflect_linear"),
    ),
    # 4 slots: color_dist, tex_dist, contrast_step, spatial_dist
    "EDGE_SURFACE_GROUPING": (
        Slot("color_dist",      "linear",      9.0, "same"),
        Slot("tex_dist",        "linear",      9.0, "same"),
        Slot("contrast_step",   "linear",      9.0, "same"),
        Slot("spatial_dist",    "linear",      9.0, "same"),
    ),
    # 3 slots: signed_turn_bin (8-cycle over 2*pi), chord_len_bin, curv_rate_bin
    # [P2] The signed turn is a DIRECTED quantity with a 2*pi period, so its
    # slot wraps: it was declared "linear", which put a +179 deg turn and a
    # -179 deg turn seven bins apart when they are one bin apart.
    "EDGE_CURVATURE": (
        Slot("signed_turn",     "circular",    8.0, "reflect_circular"),
        Slot("chord_len",       "linear",      8.0, "same"),
        Slot("curv_rate",       "linear",      8.0, "same"),
    ),
    # 2 slots: scale_bin, salience_bin
    "EDGE_CO_LOCATED": (
        Slot("scale_ratio",     "linear",      9.0, "same"),
        Slot("salience_ratio",  "linear",      9.0, "same"),
    ),
    # 3 slots: angle_delta_bin, bisector_bin (24-cycle), spatial_dist_bin
    "EDGE_ORIENTATION_BEND": (
        Slot("angle_delta",     "linear",      8.0, "same"),
        Slot("bisector_theta",  "circular",   24.0, "same"),
        Slot("spatial_dist",    "linear",      8.0, "same"),
    ),
    "EDGE_ORIENTATION_JUNCTION": (
        Slot("angle_delta",     "linear",      8.0, "same"),
        Slot("bisector_theta",  "circular",   24.0, "same"),
        Slot("spatial_dist",    "linear",      8.0, "same"),
    ),
    # 3 slots: perp_bin, stag_bin, conv_bin
    "EDGE_PARALLEL": (
        Slot("perp_dist",       "linear",      9.0, "same"),
        Slot("stagger_dist",    "linear",      9.0, "reflect_linear"),
        Slot("convergence",     "linear",      9.0, "reflect_linear"),
    ),
    # 3 slots: sym_bin, axis_bin (24-cycle), dist_bin
    "EDGE_SYMMETRY": (
        Slot("sym_score",       "linear",      8.0, "same"),
        Slot("axis_angle",      "circular",   24.0, "same"),
        Slot("dist",            "linear",      8.0, "same"),
    ),
    # 3 slots: owner_signed (the SIGNED figure-ground scalar), dist,
    # curvature.
    #
    # [P2 / MIS-READ FIX] The old layout was (ang_diff, bearing, curvature):
    # slot 0 was declared an ANGLE but the writer put the SIGNED ownership
    # scalar (a value in [-1, 1], perception.ownership.ownership_signed)
    # there, and slot 1 was declared a 24-cycle BEARING but the writer put a
    # DISTANCE there.  The reader therefore compared a signed scalar and a
    # distance as if both were angles over 2*pi -- the category error
    # ownership.py documents.  The slots now say what they carry:
    #   owner_signed  the signed figure-ground scalar, [-1, 1]
    #   dist          the endpoint separation, in wavelengths
    #   curvature     the fragment's signed curvature, [-1, 1]
    # No slot is circular any more, so the class has no circular cycle.
    "EDGE_BORDER_OWNERSHIP": (
        Slot("owner_signed",    "linear",      8.0, "same"),
        Slot("dist",            "linear",      8.0, "same"),
        Slot("curvature",       "linear",      8.0, "same"),
    ),
    # 3 slots: dist_bin, turn_bin (8-cycle), curv_agree_bin
    # [P2] Slot 2 was called "curvature" but never carried a curvature: the
    # writer put the AGREEMENT of the two contours' signed chirality there
    # (1 = the same handedness, 0 = opposite).  The slot is renamed to say
    # so.
    "EDGE_CONTOUR_CONTINUITY": (
        Slot("dist",            "linear",      8.0, "same"),
        Slot("turn",            "circular",    8.0, "reflect_circular"),
        Slot("curv_agree",      "linear",      8.0, "same"),
    ),
    # 3 slots: dist_bin, incidence_bin (24-cycle), topo_bin
    "EDGE_JUNCTION_CONTOUR": (
        Slot("dist",            "linear",      8.0, "same"),
        Slot("incidence",       "circular",   24.0, "same"),
        Slot("topology_class",  "categorical", 7.0, "same", is_identity=True),
    ),
    # 4 slots: speed_bin, heading_bin (24-cycle), ratio_bin, ang_bin
    "EDGE_CO_MOTION": (
        Slot("mean_speed",      "linear",      8.0, "same"),
        Slot("heading",         "circular",   24.0, "same"),
        Slot("speed_ratio",     "linear",      8.0, "same"),
        Slot("angle_diff",      "linear",      8.0, "same"),
    ),
    "EDGE_CONTOUR_ORIENTATION": (
        Slot("containment",     "linear",      8.0, "same"),
    ),
    "EDGE_JUNCTION_ORIENTATION": (
        Slot("containment",     "linear",      8.0, "same"),
    ),
    # 3 slots: rel_dist_bin, polar_angle_bin (24-cycle), scale_bin
    "EDGE_CURVE_CONTAINMENT": (
        Slot("rel_dist",        "linear",      8.0, "same"),
        Slot("polar_angle",     "circular",   24.0, "same"),
        Slot("scale_ratio",     "linear",      8.0, "reflect_linear"),
    ),
    "EDGE_PART_CONTAINMENT": (
        Slot("rel_dist",        "linear",      8.0, "same"),
        Slot("polar_angle",     "circular",   24.0, "same"),
        Slot("scale_ratio",     "linear",      8.0, "reflect_linear"),
    ),
    # [P2 / MIS-READ FIX] The binder wrote EdgeParams with class
    # "EDGE_REGION_CONTENT", but the registered schema was named
    # "EDGE_PART_CONTENT": the writer's class had NO schema, so its
    # similarity read the bins path with no ranges, and the schema that
    # existed described a relation nothing emitted.  The schema now carries
    # the name the writer uses (EDGE_REGION_CONTENT is also the name the
    # constitutional range table already carried); the dead
    # EDGE_PART_CONTENT schema is removed rather than left as a second name
    # for the same relation.
    "EDGE_REGION_CONTENT": (
        Slot("containment",     "linear",      8.0, "same"),
    ),
    # 3 slots: depth (rank_gap|depth_step), stack_depth (|8=none), source (0=T-junction, 1=depth)
    # 4 slots: depth (the LAYER gap), stack_depth (the layer COUNT),
    # source (which cue fired) and depth_units (the 3-D depth difference).
    # [P2] The 3-D depth difference is a graded quantity in its OWN unit,
    # and it is not a layer count, so it gets its own slot instead of
    # overloading the layer count.
    "EDGE_OCCLUSION": (
        Slot("depth",           "linear",      8.0, "reflect_linear"),
        Slot("stack_depth",     "linear",      9.0, "same"),
        Slot("source",          "categorical", 2.0, "same"),
        Slot("depth_units",     "linear",      8.0, "same"),
    ),
    "EDGE_CURVATURE_SIGN_CONTINUITY": (
        Slot("curv_diff",       "linear",      8.0, "same"),
        Slot("dist",            "linear",      8.0, "same"),
    ),
    # 2 slots: cocirc_asym_bin, dist_bin
    # [Stage 2 / Note 2] Slot 0 tag is "cocirc_asym" (shared with OC's
    # slot 2) so cross-schema projection matches invariant to invariant.
    # Both are unsigned magnitudes of co-circular deviation
    # (binder err = abs(axial_wrap(al_t + al_h)); OC asym = abs(...) of
    # the same expression), binned to 0..7: metrically comparable, not
    # merely ordinally valid. Bins are 8.0 (was 9.0) so the two schemas
    # share both invariant tag AND bin count.
    "EDGE_COCIRCULAR": (
        Slot("cocirc_asym",     "linear",      8.0, "same"),
        Slot("dist",            "linear",      8.0, "same"),
    ),
    # 2 slots: gap_bin, err_bin
    "EDGE_AMODAL_LINK": (
        Slot("gap",             "linear",      8.0, "same"),
        Slot("error",           "linear",      8.0, "same"),
    ),
    # 2 slots: dist_bin, congruence_bin
    "EDGE_SHARED_BOUNDARY": (
        Slot("dist",            "linear",      8.0, "same"),
        Slot("congruence",      "categorical", 8.0, "same", is_identity=True),
    ),
    # [P2] The fourth slot is the ECCENTRICITY DIFFERENCE, not an
    # "agreement" (the writer measures |ecc_a - ecc_b|).
    "EDGE_CONCENTRICITY": (
        Slot("radial_ratio",    "linear",      8.0, "reflect_linear"),
        Slot("offset",          "linear",      8.0, "same"),
        Slot("extent",          "linear",      8.0, "same"),
        Slot("ecc_difference",  "linear",      8.0, "same"),
    ),
    "EDGE_TEXTURE_BOUNDARY": (
        Slot("tex_diff",        "linear",      8.0, "same"),
        Slot("lum_diff",        "linear",      8.0, "same"),
        Slot("hue_diff",        "linear",      8.0, "same"),
        Slot("chroma_diff",     "linear",      8.0, "same"),
    ),
    # 4 slots: dihedral, fold_axis (12-cycle), convexity, sharpness
    "EDGE_SURFACE_FOLD": (
        Slot("dihedral",        "linear",     12.0, "same"),
        Slot("fold_axis",       "circular",   12.0, "same"),
        Slot("convexity",       "categorical", 3.0, "reflect_linear", is_identity=True),
        Slot("sharpness",       "linear",      8.0, "same"),
    ),
    "EDGE_RADIALITY": (
        Slot("hub_bearing",     "circular",   24.0, "same"),
        Slot("alignment",       "linear",      8.0, "same"),
        Slot("hub_ratio",       "linear",      8.0, "same"),
        # [P2] The spoke separation is AXIAL and folds (it does not
        # wrap): the writer bins it LINEARLY over [0, pi].
        Slot("spoke_separation", "linear",     12.0, "same"),
    ),
    # [P2] The pitch is AXIAL and FOLDS (it does not wrap): the writer
    # bins it LINEARLY over [0, pi/2], so the slot is linear, not circular.
    "EDGE_SPIRALITY": (
        Slot("pitch",           "linear",      12.0, "same"),
        Slot("consistency",     "linear",      8.0, "same"),
        Slot("chirality",       "categorical", 3.0, "reflect_linear", is_identity=True),
        Slot("expansion",       "linear",      8.0, "reflect_linear"),
    ),
    "EDGE_COLOR_OPPONENT_BORDER": (
        Slot("opp_angle",       "circular",   24.0, "same"),
        Slot("contrast",        "linear",      8.0, "same"),
        Slot("lum_indep",       "linear",      8.0, "same"),
        Slot("extent",          "linear",      8.0, "same"),
    ),
    "EDGE_MOTION_OPPOSITION": (
        Slot("dir_contrast",    "circular",   24.0, "same"),
        Slot("speed_contrast",  "linear",      8.0, "same"),
        Slot("boundary_angle",  "circular",   24.0, "same"),
        Slot("speed",           "linear",      8.0, "same"),
    ),
    "EDGE_CROSS_SCALE_IDENTITY": (
        Slot("octave_ratio",    "linear",      8.0, "reflect_linear"),
        Slot("phase_alignment", "circular",   24.0, "same"),
        Slot("drift",           "linear",      8.0, "same"),
        Slot("similarity",      "linear",      8.0, "same"),
    ),
    "EDGE_BINOCULAR_DISPARITY": (
        Slot("disparity",       "linear",      8.0, "reflect_linear"),
        Slot("gradient",        "linear",      8.0, "same"),
        Slot("depth_straddle",  "categorical", 4.0, "same", is_identity=True),
        Slot("confidence",      "linear",      8.0, "same"),
    ),
    "EDGE_LINE_TERMINATION": (
        Slot("junction_type",   "categorical", 5.0, "same", is_identity=True),
        Slot("sharpness",       "linear",      8.0, "same"),
        Slot("curvature",       "linear",      8.0, "same"),
        Slot("parent_theta",    "circular",   12.0, "same"),
    ),
}


# ============================================================================
# [P2] THE EDGE UNIT TABLE -- ONE per EDGE class
# ============================================================================
# The node classes have their unit table in core/registries.py (SlotSpec).
# The EDGE classes get the SAME treatment here, for the same reason: the
# bins are only a lookup index, and the GRADED population is the
# representation.  That only holds if the two readings of an edge are
# quantisations of the SAME measurement, in the SAME unit, over the SAME
# span -- so every edge slot declares:
#
#   name      the stable identifier (== the schema's tag)
#   unit      the UNIT of the CONTINUOUS quantity the writer measured
#   lo, hi    the SPAN of that quantity (the writer's own normalisation)
#   n_bins    the BIN COUNT (the writer's own level count)
#   circular  whether the slot wraps
#   quantiser the bins-path convention (the same vocabulary as SlotSpec)
#
# THE WRITER'S UNITS (not re-binned bins):
#   * distances are in WAVELENGTHS -- the front end's extraction scale, so
#     the value is scale-free and comparable across images;
#   * angles are in RADIANS (an AXIAL angle on a pi-period where the
#     relation is orientation-like, a full 2*pi-period where it is not);
#   * ratios are RATIOS or LOG-RATIOS;
#   * signed strengths are SIGNED SCALARS in [-1, 1];
#   * a slot whose quantity is genuinely CATEGORICAL is declared
#     "lookup_key" and its bin is the class, not a quantisation -- those
#     slots are INTENTIONALLY INDEX-ONLY and carry NO population.
#
# The table is the SINGLE READER of an edge's slots: `edge_slot(class, name)`
# resolves a slot by NAME, and `edge_params(class, **values)` builds the
# EdgeParams from NAMED continuous values (a wrong name is a KeyError, never
# a silent mis-read).
# ---------------------------------------------------------------------------


def _eslot(name: str, unit: str, lo: float, hi: float, n_bins: int,
           circular: bool = False, quantiser: str = "floor",
           keys: Tuple[float, ...] = ()) -> SlotSpec:
    """One entry of an edge class' unit table."""
    return SlotSpec(name=name, unit=unit, lo=float(lo), hi=float(hi),
                    n_bins=int(n_bins), circular=bool(circular),
                    quantiser=str(quantiser), keys=tuple(float(k) for k in keys))


#: The extraction-scale-normalised spans the writers use.  A wavelength is
#: the front end's extraction scale, so "3 wavelengths" is the same physical
#: relation at any image resolution.
_WAVELENGTH_SPANS: Dict[str, float] = {
    # edge class                 -> the span in wavelengths
    "EDGE_ORIENTATION_CONTINUITY": 3.0,
    "EDGE_CONTACT": 3.0,
    "EDGE_CONTOUR_CONTINUITY": 3.0,
    "EDGE_CURVATURE_SIGN_CONTINUITY": 4.0,
    "EDGE_COCIRCULAR": 5.0,
    "EDGE_AMODAL_LINK": 5.0,
    "EDGE_SHARED_BOUNDARY": 2.5,
    "EDGE_BORDER_OWNERSHIP": 2.0,
    "EDGE_ORIENTATION_JUNCTION": 1.0,
    "EDGE_ORIENTATION_BEND": 1.0,
    "EDGE_CURVATURE": 3.0,
    "EDGE_JUNCTION_CONTOUR": 2.5,
}

EDGE_UNIT_TABLE: Dict[str, Tuple[SlotSpec, ...]] = {
    # --- ORIENTATION CONTINUITY (V1 association field) --------------------
    # The V1 association field's length unit is the MEDIAN NEAREST-NEIGHBOUR
    # SPACING L (the field's own scale), not the extraction scale, so the
    # two distances are expressed in median spacings.
    # dist        the along-axis gap between the two segments, in median
    #             spacings
    # perp        the perpendicular offset of the pair, in median spacings
    # cocirc_asym the co-circularity asymmetry, in radians (0 = co-circular)
    # turn        the SIGNED axial turn between the two tangents, in radians
    # tail/head   the bipole ROLES: a categorical lookup key, NOT a
    #             quantisation of a graded value (INTENTIONALLY INDEX-ONLY)
    # polarity_rel the contrast-polarity relation: also a categorical
    #             lookup key (INTENTIONALLY INDEX-ONLY)
    "EDGE_ORIENTATION_CONTINUITY": (
        _eslot("dist", "median_spacings", 0.0, 4.0, 8),
        _eslot("perp", "median_spacings", 0.0, 2.0, 8),
        _eslot("cocirc_asym", "radians", 0.0, 0.5 * math.pi, 8),
        _eslot("turn", "radians", -0.5 * math.pi, 0.5 * math.pi, 8,
               circular=True),
        _eslot("tail_role", "class", 0.0, 3.0, 3, quantiser="lookup_key",
               keys=(0.0, 1.0, 2.0)),
        _eslot("head_role", "class", 0.0, 3.0, 3, quantiser="lookup_key",
               keys=(0.0, 1.0, 2.0)),
        _eslot("polarity_rel", "class", 0.0, 3.0, 3, quantiser="lookup_key",
               keys=(0.0, 1.0, 2.0)),
    ),
    # --- CONTACT (the generic proximity relation) -------------------------
    # normal_angle the contact NORMAL (the perpendicular of the joining
    #             line): an AXIAL angle over pi, 24 levels
    # dist         the separation of the two entities, in MEDIAN SPACINGS
    #             (the contact radius r_med is the V1 field's own scale)
    # depth_step   the depth difference, in the 3-D depth units (ordinal)
    "EDGE_CONTACT": (
        _eslot("normal_angle", "radians", 0.0, math.pi, 24, circular=True,
               quantiser="round"),
        _eslot("dist", "median_spacings", 0.0, 1.0, 8),
        _eslot("depth_step", "depth_units", 0.0, 3.0, 8),
    ),
    # --- SURFACE GROUPING -------------------------------------------------
    # The three surface distances are measured on the LIVERY's own planes:
    # the opponent colour distance, the texture distance and the Weber
    # contrast step, each in the plane's own units (a fraction of the full
    # range).  spatial_dist is the endpoint separation as a LOG2 SPACING.
    "EDGE_SURFACE_GROUPING": (
        _eslot("color_dist", "colour_sigma_distance", 0.0, 3.0, 8),
        _eslot("tex_dist", "texture_sigma_distance", 0.0, 3.0, 8),
        _eslot("contrast_step", "weber_contrast", 0.0, 1.0, 8),
        _eslot("spatial_dist", "log2_spacings", 0.0, 4.0, 8),
    ),
    # --- CURVATURE (between adjacent chain segments) ----------------------
    # signed_turn the signed turn between the two segment tangents, radians
    #            (a FULL 2*pi-period: the turn is a directed quantity)
    # chord_len   the chord between the two segments, in wavelengths
    # curv_rate   the turn PER wavelength: the curvature rate
    "EDGE_CURVATURE": (
        _eslot("signed_turn", "radians", -math.pi, math.pi, 8, circular=True,
               quantiser="round"),
        _eslot("chord_len", "wavelengths", 0.0, 3.0, 8),
        _eslot("curv_rate", "radians_per_wavelength", 0.0, 1.0, 8),
    ),
    # --- CO-LOCATED -------------------------------------------------------
    # scale_ratio  the ABSOLUTE log2 SIZE RATIO of the two entities (the
    #              edge is undirected, so the ratio is order-invariant)
    # salience_ratio the ratio of their saliences
    "EDGE_CO_LOCATED": (
        _eslot("scale_ratio", "abs_log2_ratio", 0.0, 4.0, 8),
        _eslot("salience_ratio", "fraction", 0.0, 1.0, 8),
    ),
    # --- ORIENTATION BEND / JUNCTION (the angle relations) ---------------
    # angle_delta    the axial angle difference, in radians
    # bisector_theta the bisector orientation, in radians (axial, pi-period)
    # spatial_dist   the endpoint separation, in wavelengths
    "EDGE_ORIENTATION_BEND": (
        _eslot("angle_delta", "radians", 0.0, 0.5 * math.pi, 8),
        _eslot("bisector_theta", "radians", 0.0, math.pi, 24, circular=True,
               quantiser="round"),
        _eslot("spatial_dist", "wavelengths", 0.0, 1.0, 8),
    ),
    "EDGE_ORIENTATION_JUNCTION": (
        _eslot("angle_delta", "radians", 0.0, 0.5 * math.pi, 8),
        _eslot("bisector_theta", "radians", 0.0, math.pi, 24, circular=True,
               quantiser="round"),
        _eslot("spatial_dist", "wavelengths", 0.0, 1.0, 8),
    ),
    # --- PARALLEL ---------------------------------------------------------
    # perp_dist    the perpendicular separation, in wavelengths (the
    #              relation's reach is 3 * R = 6 wavelengths)
    # stagger_dist the stagger ALONG the mean axis, in wavelengths
    #              (SIGNED: a negative stagger is a reversed order)
    # convergence  the signed convergence angle, in radians
    "EDGE_PARALLEL": (
        _eslot("perp_dist", "wavelengths", 0.0, 6.0, 8),
        _eslot("stagger_dist", "wavelengths", -6.0, 6.0, 8),
        _eslot("convergence", "radians", -math.pi / 12.0, math.pi / 12.0, 8),
    ),
    # --- SYMMETRY ---------------------------------------------------------
    # sym_score  the symmetry score in [0, 1]
    # axis_angle the symmetry AXIS, in radians (axial, pi-period)
    # dist       the separation of the two entities, in wavelengths
    "EDGE_SYMMETRY": (
        _eslot("sym_score", "fraction", 0.0, 1.0, 8),
        _eslot("axis_angle", "radians", 0.0, math.pi, 24, circular=True,
               quantiser="round"),
        _eslot("dist", "wavelengths", 0.0, 4.0, 8),
    ),
    # --- BORDER OWNERSHIP (the figure-ground edge) ------------------------
    # owner_signed the SIGNED figure-ground scalar in [-1, 1]: + = the
    #             detection's entity is the figure, - = its neighbour is
    # dist          the endpoint separation, in wavelengths
    # curvature     the fragment's signed curvature, in [-1, 1]
    # (See the MIS-READ FIX note on the schema above.)
    "EDGE_BORDER_OWNERSHIP": (
        _eslot("owner_signed", "signed_scalar", -1.0, 1.0, 8,
               quantiser="signed_centered"),
        _eslot("dist", "wavelengths", 0.0, 2.0, 8),
        _eslot("curvature", "signed_scalar", -1.0, 1.0, 8,
               quantiser="signed_centered"),
    ),
    # --- CONTOUR CONTINUITY (the V2 chain relation) -----------------------
    # dist        the separation of the two contours, in wavelengths (the
    #             relation's reach is R = 2 wavelengths)
    # turn        the SIGNED axial turn between them, in radians
    # curv_agree  the AGREEMENT of the two contours' signed curvature
    #             (a fraction in [0, 1]; 1 = the same handedness)
    "EDGE_CONTOUR_CONTINUITY": (
        _eslot("dist", "wavelengths", 0.0, 2.0, 8),
        _eslot("turn", "radians", -0.5 * math.pi, 0.5 * math.pi, 8,
               circular=True),
        _eslot("curv_agree", "fraction", 0.0, 1.0, 8),
    ),
    # --- JUNCTION <-> CONTOUR --------------------------------------------
    # dist          the separation, in wavelengths
    # incidence     the incidence angle between the junction's chord and
    #               the contour's tangent, in radians (axial, pi-period)
    # topology_class the junction's topology class: a categorical LOOKUP
    #               KEY (INTENTIONALLY INDEX-ONLY)
    "EDGE_JUNCTION_CONTOUR": (
        _eslot("dist", "wavelengths", 0.0, 2.5, 8),
        _eslot("incidence", "radians", 0.0, math.pi, 24, circular=True,
               quantiser="round"),
        _eslot("topology_class", "class", 0.0, 7.0, 7,
               quantiser="lookup_key", keys=tuple(float(i) for i in range(7))),
    ),
    # --- CO-MOTION --------------------------------------------------------
    "EDGE_CO_MOTION": (
        _eslot("mean_speed", "log2_speed_ratio", -6.0, 0.0, 8),
        _eslot("heading", "radians", 0.0, 2.0 * math.pi, 24, circular=True,
               quantiser="round"),
        _eslot("speed_ratio", "abs_log2_ratio", 0.0, 3.0, 8),
        _eslot("angle_diff", "radians", 0.0, math.pi, 8),
    ),
    # --- CONTAINMENT (curve / part) --------------------------------------
    # rel_dist    the separation relative to the container's own reach
    #             (2 * R = 5 wavelengths)
    # polar_angle the polar address of the contained entity, in radians
    # scale_ratio the SIZE RATIO (log2), so equal and reciprocal are +-1
    "EDGE_CURVE_CONTAINMENT": (
        _eslot("rel_dist", "wavelengths", 0.0, 5.0, 8),
        _eslot("polar_angle", "radians", 0.0, 2.0 * math.pi, 24,
               circular=True, quantiser="round"),
        _eslot("scale_ratio", "log2_ratio", -2.0, 2.0, 8),
    ),
    "EDGE_PART_CONTAINMENT": (
        _eslot("rel_dist", "wavelengths", 0.0, 5.0, 8),
        _eslot("polar_angle", "radians", 0.0, 2.0 * math.pi, 24,
               circular=True, quantiser="round"),
        _eslot("scale_ratio", "log2_ratio", -2.0, 2.0, 8),
    ),
    # --- CONTAINMENT FLAGS (the membership relations) --------------------
    # The membership itself is a 0/1 FLAG: there is no graded quantity
    # behind "this segment is summarised by that contour" beyond the flag
    # and the containment strength, which the edge's CONFIDENCE carries.
    # INTENTIONALLY INDEX-ONLY.
    "EDGE_CONTOUR_ORIENTATION": (
        _eslot("containment", "flag", 0.0, 1.0, 2, quantiser="lookup_key",
               keys=(0.0, 1.0)),
    ),
    "EDGE_JUNCTION_ORIENTATION": (
        _eslot("containment", "flag", 0.0, 1.0, 2, quantiser="lookup_key",
               keys=(0.0, 1.0)),
    ),
    "EDGE_REGION_CONTENT": (
        _eslot("containment", "flag", 0.0, 1.0, 2, quantiser="lookup_key",
               keys=(0.0, 1.0)),
    ),
    # --- OCCLUSION --------------------------------------------------------
    # depth       the DEPTH STEP: the number of layers between the occluder
    #             and the occluded (a layer COUNT)
    # stack_depth the depth of the occluder's stack (a layer COUNT)
    # source      WHICH cue fired: a categorical LOOKUP KEY
    #             (INTENTIONALLY INDEX-ONLY)
    "EDGE_OCCLUSION": (
        _eslot("depth", "layers", 0.0, 8.0, 8),
        _eslot("stack_depth", "layers", 0.0, 9.0, 9),
        _eslot("source", "class", 0.0, 2.0, 2, quantiser="lookup_key",
               keys=(0.0, 1.0)),
        _eslot("depth_units", "depth_units", 0.0, 1.0, 8),
    ),
    # --- CURVATURE SIGN CONTINUITY ---------------------------------------
    # curv_diff the difference of the two contours' NET SIGNED TURNS, in
    #            radians (the chirality difference)
    # dist      the separation, in wavelengths
    "EDGE_CURVATURE_SIGN_CONTINUITY": (
        _eslot("curv_diff", "radians", 0.0, 2.0 * math.pi, 8),
        _eslot("dist", "wavelengths", 0.0, 4.0, 8),
    ),
    # --- CO-CIRCULAR ------------------------------------------------------
    # cocirc_asym the co-circular deviation, in radians (0 = co-circular)
    # dist        the separation, in wavelengths
    "EDGE_COCIRCULAR": (
        _eslot("cocirc_asym", "radians", 0.0, math.pi / 6.0, 8),
        _eslot("dist", "wavelengths", 0.0, 5.0, 8),
    ),
    # --- AMODAL LINK ------------------------------------------------------
    # gap   the amodal GAP the two collinear segments span, in wavelengths
    # error the mean collinearity error against the chord, in radians
    "EDGE_AMODAL_LINK": (
        _eslot("gap", "wavelengths", 0.0, 5.0, 8),
        _eslot("error", "radians", 0.0, math.pi / 8.0, 8),
    ),
    # --- SHARED BOUNDARY --------------------------------------------------
    # dist       the separation of the two surface patches, in wavelengths
    # congruence whether the two patches AGREE on the field: a categorical
    #            LOOKUP KEY (INTENTIONALLY INDEX-ONLY)
    "EDGE_SHARED_BOUNDARY": (
        _eslot("dist", "wavelengths", 0.0, 2.5, 8),
        _eslot("congruence", "class", 0.0, 8.0, 8, quantiser="lookup_key",
               keys=(0.0, 7.0)),
    ),
    # --- CONCENTRICITY ----------------------------------------------------
    "EDGE_CONCENTRICITY": (
        _eslot("radial_ratio", "log2_ratio", 0.0, 4.0, 8),
        _eslot("offset", "fraction", 0.0, 1.0, 8),
        _eslot("extent", "fraction", 0.0, 1.0, 8),
        _eslot("ecc_difference", "fraction", 0.0, 1.0, 8),
    ),
    # --- TEXTURE BOUNDARY -------------------------------------------------
    "EDGE_TEXTURE_BOUNDARY": (
        _eslot("tex_diff", "fraction", 0.0, 1.0, 8),
        _eslot("lum_diff", "fraction", 0.0, 1.0, 8),
        _eslot("hue_diff", "radians", 0.0, math.pi, 8),
        _eslot("chroma_diff", "fraction", 0.0, 1.0, 8),
    ),
    # --- SURFACE FOLD -----------------------------------------------------
    "EDGE_SURFACE_FOLD": (
        _eslot("dihedral", "radians", 0.0, math.pi, 12),
        _eslot("fold_axis", "radians", 0.0, math.pi, 12, circular=True,
               quantiser="round"),
        _eslot("convexity", "class", 0.0, 3.0, 3, quantiser="lookup_key",
               keys=(0.0, 1.0, 2.0)),
        _eslot("sharpness", "fraction", 0.0, 1.0, 8),
    ),
    # --- RADIALITY --------------------------------------------------------
    "EDGE_RADIALITY": (
        _eslot("hub_bearing", "radians", 0.0, 2.0 * math.pi, 24,
               circular=True, quantiser="round"),
        _eslot("alignment", "radians", 0.0, 0.5 * math.pi, 8,
               quantiser="round"),
        _eslot("hub_ratio", "ratio", 0.0, 8.0, 8, quantiser="round"),
        _eslot("spoke_separation", "radians", 0.0, math.pi, 12,
               quantiser="round"),
    ),
    # --- SPIRALITY --------------------------------------------------------
    "EDGE_SPIRALITY": (
        _eslot("pitch", "radians", 0.0, 0.5 * math.pi, 12, quantiser="round"),
        _eslot("consistency", "fraction", 0.0, 1.0, 8),
        _eslot("chirality", "class", 0.0, 3.0, 3, quantiser="lookup_key",
               keys=(0.0, 1.0, 2.0)),
        _eslot("expansion", "expansion_rate", 0.0, 8.0 / 3.0, 8),
    ),
    # --- COLOUR OPPONENT BORDER -------------------------------------------
    "EDGE_COLOR_OPPONENT_BORDER": (
        _eslot("opp_angle", "radians", 0.0, 2.0 * math.pi, 24, circular=True,
               quantiser="round"),
        _eslot("contrast", "fraction", 0.0, 1.0, 8),
        _eslot("lum_indep", "fraction", 0.0, 1.0, 8),
        _eslot("extent", "fraction", 0.0, 1.0, 8),
    ),
    # --- MOTION OPPOSITION ------------------------------------------------
    "EDGE_MOTION_OPPOSITION": (
        _eslot("dir_contrast", "radians", 0.0, 2.0 * math.pi, 24,
               circular=True, quantiser="round"),
        _eslot("speed_contrast", "fraction", 0.0, 1.0, 8),
        _eslot("boundary_angle", "radians", 0.0, 2.0 * math.pi, 24,
               circular=True, quantiser="round"),
        _eslot("speed", "speed_units", 0.0, 1.0, 8),
    ),
    # --- CROSS-SCALE IDENTITY ---------------------------------------------
    "EDGE_CROSS_SCALE_IDENTITY": (
        _eslot("octave_ratio", "log2_ratio", -2.0, 2.0, 8),
        _eslot("phase_alignment", "radians", 0.0, 2.0 * math.pi, 24,
               circular=True, quantiser="round"),
        _eslot("drift", "fraction", 0.0, 1.0, 8),
        _eslot("similarity", "fraction", 0.0, 1.0, 8),
    ),
    # --- BINOCULAR DISPARITY ----------------------------------------------
    "EDGE_BINOCULAR_DISPARITY": (
        _eslot("disparity", "depth_units", 0.0, 1.0, 8),
        _eslot("gradient", "depth_units_per_pixel", 0.0, 1.4, 8),
        _eslot("depth_straddle", "class", 0.0, 4.0, 4, quantiser="lookup_key",
               keys=(0.0, 1.0, 2.0, 3.0)),
        _eslot("confidence", "fraction", 0.0, 1.0, 8),
    ),
    # --- LINE TERMINATION -------------------------------------------------
    # junction_type the junction's TYPE (end / corner / collinear / T / Y):
    #              a categorical LOOKUP KEY (INTENTIONALLY INDEX-ONLY)
    # sharpness    the end-stop sharpness, a fraction in [0, 1]
    # curvature     the corner's angle, a fraction of pi/2
    # parent_theta  the parent line's orientation, in radians (axial)
    "EDGE_LINE_TERMINATION": (
        _eslot("junction_type", "class", 0.0, 5.0, 5, quantiser="lookup_key",
               keys=(0.0, 1.0, 2.0, 3.0, 4.0)),
        _eslot("sharpness", "fraction", 0.0, 1.0, 8),
        _eslot("curvature", "fraction", 0.0, 1.0, 8),
        _eslot("parent_theta", "radians", 0.0, math.pi, 12, circular=True,
               quantiser="round"),
    ),
}
#: [U30] EdgeParams schemas for the 8 canonical physical edge categories.
#: Additive-only registry trip for Phases 1-3; _edge_params() is switched
#: over to this table at Phase 4 cutover.  Take25 remains routed to
#: EDGE_UNIT_TABLE until then.
#:
#: Units match the WRITERS, not the names: distances are in MEDIAN
#: SPACINGS (the scene median nearest-neighbour distance, the v8 scale-
#: adaptive convention) except the containment relation, which the legacy
#: writers already measure in WAVELENGTHS; turn/angle slots are radians
#: (axial angles are radians_axial, the mod-pi convention); separations the
#: writers divide by the extraction scale go in wavelengths.  ABSENT=None
#: is explicit ("no measurement") on every slot; a writer never fabricates
#: a quantity it did not compute.
U30_EDGE_UNIT_TABLE: Dict[str, Tuple[SlotSpec, ...]] = {
    "EDGE_PROXIMITY": (
        _eslot("distance", "spacings", 0.0, 8.0, 8),
        _eslot("bearing", "radians_axial", 0.0, math.pi, 12,
               circular=True),
        _eslot("scale_ratio", "log2_ratio", -2.0, 2.0, 8),
    ),
    "EDGE_CONTINUITY": (
        _eslot("gap", "spacings", 0.0, 4.0, 8),
        _eslot("turn_angle", "radians", -0.5 * math.pi, 0.5 * math.pi, 12,
               circular=True),
        _eslot("perp_offset", "spacings", 0.0, 3.0, 8),
        _eslot("curv_agree", "agreement", -1.0, 1.0, 8),
    ),
    "EDGE_CONTAINMENT": (
        _eslot("rel_dist", "wavelengths", 0.0, 5.0, 8),
        _eslot("polar_angle", "radians", 0.0, 2.0 * math.pi, 24,
               circular=True, quantiser="round"),
        _eslot("scale_ratio", "log2_ratio", -2.0, 2.0, 8),
        _eslot("ownership_strength", "fraction", 0.0, 1.0, 8),
        _eslot("side", "sign", -1.0, 1.0, 2,
               quantiser="lookup_key", keys=(-1.0, 1.0)),
    ),
    "EDGE_OCCLUSION": (
        _eslot("depth_diff", "rank_units", 0.0, 4.0, 8),
        _eslot("source_cue", "cue_id", 0.0, 3.0, 4,
               quantiser="lookup_key", keys=(0.0, 1.0, 2.0, 3.0)),
        _eslot("strength", "fraction", 0.0, 1.0, 8),
        _eslot("relation", "kind", 0.0, 1.0, 2,
               quantiser="lookup_key", keys=(0.0, 1.0)),
    ),
    "EDGE_JUNCTION": (
        _eslot("angle_delta", "radians_axial", 0.0, 0.5 * math.pi, 12),
        _eslot("topo_class", "junction_type", 0.0, 6.0, 7,
               quantiser="lookup_key", keys=tuple(float(i) for i in range(7))),
        _eslot("arm_count", "arms", 2.0, 7.0, 6),
    ),
    "EDGE_SURFACE": (
        _eslot("normal_angle", "radians", 0.0, math.pi, 12),
        _eslot("material_diff", "fraction", 0.0, 1.0, 8),
        _eslot("slant_diff", "radians_axial", 0.0, 0.5 * math.pi, 8),
        _eslot("motion_kin", "fraction", 0.0, 1.0, 8),
    ),
    "EDGE_SHAPE": (
        _eslot("sym_score", "fraction", 0.0, 1.0, 8),
        _eslot("radial_ratio", "log2_ratio", -2.0, 2.0, 8),
        _eslot("curvature", "asinh_curvature", -1.0, 1.0, 9,
               quantiser="signed_centered"),
    ),
    "EDGE_MATERIAL": (
        _eslot("chrom_diff", "fraction", 0.0, 1.0, 8),
        _eslot("lum_diff", "fraction", 0.0, 1.0, 8),
        _eslot("tex_diff", "fraction", 0.0, 1.0, 8),
    ),
}

#: [U30] The source_cue vocabulary, by cue id (documented here so the
#: binder's writes are not magic): 0 = disparity (A4 binocular / 3-D depth),
#: 1 = line_termination (A3 end-stopping, including the 2-D T-junction cue),
#: 2 = kinetic (R3 motion opposition; dormant on static frames),
#: 3 = figural_separation (anti-phase border-ownership / closed-chain pairs;
#: the contested boundary's figurial silhouette).
#: The relation vocabulary: 0 = depth_order (a rank gap between layers),
#: 1 = figural_separation (an occluded-by-silhouette boundary, no rank claim).
#: depth_diff is ABSENT=None for every figural_separation edge.





#: The edge classes whose parameters are INTENTIONALLY INDEX-ONLY: every
#: slot is a categorical lookup key, so there is NO graded quantity to
#: populate and none is invented.  The reason is recorded per class.
EDGE_INDEX_ONLY: Dict[str, str] = {
    "EDGE_CONTOUR_ORIENTATION":
        "membership is a 0/1 flag; the containment strength is the edge's "
        "confidence, not a slot",
    "EDGE_JUNCTION_ORIENTATION":
        "membership is a 0/1 flag; the containment strength is the edge's "
        "confidence, not a slot",
    "EDGE_REGION_CONTENT":
        "membership is a 0/1 flag; the containment strength is the edge's "
        "confidence, not a slot",
}


def edge_slot_specs(class_name: str) -> Tuple[SlotSpec, ...]:
    """The unit table of one edge class, in slot order."""
    return EDGE_UNIT_TABLE.get(class_name, ())


def edge_slot_spec(class_name: str, slot_name: str) -> Optional[SlotSpec]:
    """The unit-table entry of ONE slot of one edge class, by NAME."""
    for s in EDGE_UNIT_TABLE.get(class_name, ()):
        if s.name == slot_name:
            return s
    return None


def edge_slot(class_name: str, slot_name: str) -> int:
    """The INDEX of one slot of one edge class, resolved BY NAME.

    The single reader of an edge's slot layout.  A wrong name raises
    KeyError, so a mis-read cannot be written silently: the old emitters
    addressed slots by POSITION (`quantized=(a, b, c)`), which made a
    re-ordered or re-named slot indistinguishable from a correct one.
    """
    for i, s in enumerate(EDGE_UNIT_TABLE.get(class_name, ())):
        if s.name == slot_name:
            return i
    raise KeyError(
        f"edge class {class_name!r} has no slot named {slot_name!r} "
        f"(known: {[s.name for s in EDGE_UNIT_TABLE.get(class_name, ())]})"
    )


def edge_quantise(class_name: str,
                  values: Dict[str, Optional[float]],
                  *,
                  _table: Optional[Dict[str, Tuple[SlotSpec, ...]]] = None,
                  ) -> Tuple[float, ...]:
    """The BINS path: the lookup index of one edge, from NAMED values.

    The values are the writer's CONTINUOUS measurements in the slot's own
    unit; the table quantises them.  `None` means "no measurement for this
    slot" and is written as UNKNOWN (the reader skips it), never as 0.
    """
    specs = (_table or EDGE_UNIT_TABLE).get(class_name)
    if not specs:
        raise KeyError(f"edge class {class_name!r} has no unit table")
    unknown = set(values) - {s.name for s in specs}
    if unknown:
        raise KeyError(
            f"edge class {class_name!r}: unknown slot(s) {sorted(unknown)}; "
            f"known: {[s.name for s in specs]}")
    missing = {s.name for s in specs} - set(values)
    if missing:
        # A slot that is not named at all is a MIS-READ, not "no
        # measurement": "no measurement" is spelled None, explicitly.
        raise KeyError(
            f"edge class {class_name!r}: slot(s) {sorted(missing)} were not "
            f"named; every slot must be named (use None for 'no "
            f"measurement')")
    out: List[float] = []
    for s in specs:
        v = values.get(s.name)
        if v is None:
            out.append(UNKNOWN)
            continue
        if s.quantiser == "lookup_key":
            out.append(float(v))
            continue
        out.append(float(quantise(s, float(v))))
    return tuple(out)


def edge_population(class_name: str,
                    values: Dict[str, Optional[float]],
                    *,
                    _table: Optional[Dict[str, Tuple[SlotSpec, ...]]] = None,
                    _axes: Optional[Dict[str, Tuple[PopAxis, ...]]] = None,
                    ) -> Tuple[Tuple[float, ...], ...]:
    """The GRADED path: one tuning bump per slot, from the SAME values.

    Derived from the unit table, so the graded vector and the lookup index
    name the same measurement of the same quantity.
    """
    specs = (_table or EDGE_UNIT_TABLE).get(class_name)
    axes = (_axes if _axes is not None
            else EDGE_POPULATION_AXES).get(class_name)
    if not specs or axes is None or len(specs) != len(axes):
        return ()
    out: List[Tuple[float, ...]] = []
    axes_by_name = {ax.name: ax for ax in axes}
    for v, s in zip([values.get(sp.name) for sp in specs], specs):
        ax = axes_by_name.get(s.name)
        if ax is None:
            return ()
        if v is None or s.quantiser == "lookup_key":
            # NO MEASUREMENT (or a categorical key): the axis carries the
            # uniform (uninformative) activity, which the overlap reads as
            # 0.0 -- never a fabricated value.
            out.append(tuple([1.0 / ax.n_bins] * ax.n_bins))
            continue
        x = float(v)
        if not math.isfinite(x):
            return ()
        out.append(soft_bump_bins(population_centre(s, x), ax))
    return tuple(out)


def edge_slot_value(class_name: str, slot_name: str, b: float) -> float:
    """The bin's CENTRE in the slot's own unit -- the DE-BINNING.

    The graph stores the LOOKUP INDEX of an edge's parameters, but a
    relational context needs the measurement those bins name.  This is the
    inverse of the table's quantiser: the value whose quantisation is the
    bin, in the slot's own unit (radians, wavelengths, a signed scalar).

    A slot whose bin is a CATEGORICAL lookup key returns the key itself.
    """
    spec = edge_slot_spec(class_name, slot_name)
    if spec is None:
        raise KeyError(f"edge class {class_name!r} has no slot {slot_name!r}")
    if spec.quantiser == "lookup_key":
        return float(b)
    if b == UNKNOWN or not math.isfinite(b):
        return float("nan")
    span = spec.hi - spec.lo
    n = float(spec.n_bins)
    x = float(b)
    if spec.quantiser == "signed_centered":
        t = (x / max(n - 1.0, 1.0))
    elif spec.quantiser == "signed_log":
        half = (n - 1.0) / 2.0
        t = (x - half) / max(n - 1.0, 1.0)
    elif spec.quantiser == "round":
        t = x / n
    else:  # "floor"
        t = (x + 0.5) / n
    return float(spec.lo + t * span)


def edge_params(class_name: str,
                values: Dict[str, Optional[float]],
                sign: int = 1,
                confidence: float = 1.0,
                ontology: str = "",
                provenance: str = "",
                *,
                _table: Optional[Dict[str, Tuple[SlotSpec, ...]]] = None,
                _axes: Optional[Dict[str, Tuple[PopAxis, ...]]] = None,
                ) -> "EdgeParams":
    """Build an EdgeParams from NAMED continuous values.

    The one way an edge's parameters are constructed: the slot layout is
    resolved by NAME through the unit table, the bins are the table's
    quantisation of the values, and the graded population is built from the
    same values.  A wrong name raises; a wrong order cannot be written.
    """
    return EdgeParams(
        class_=class_name,
        quantized=edge_quantise(class_name, values, _table=_table),
        sign=int(sign),
        confidence=float(confidence),
        population=edge_population(class_name, values,
                                   _table=_table, _axes=_axes),
        ontology=str(ontology),
        provenance=str(provenance),
    )


def u30_edge_params(class_name: str,
                    values: Dict[str, Optional[float]],
                    sign: int = 1,
                    confidence: float = 1.0,
                    ) -> "EdgeParams":
    """[U30] The same writer, routed at the 8 canonical categories.

    Takes the U30 unit table (U30_EDGE_UNIT_TABLE) and the U30 graded axes
    (U30_EDGE_POPULATION_AXES) instead of the legacy ones.  Every U30 slot
    must be NAMED; a slot the writer cannot measure is None ("ABSENT"),
    never a fabricated number.
    """
    return edge_params(class_name, values, sign, confidence,
                       ontology="U30",
                       _table=U30_EDGE_UNIT_TABLE,
                       _axes=U30_EDGE_POPULATION_AXES)


# ---------------------------------------------------------------------------
# [P2] EDGE SPECIFICITY -- the MATCHING-TIME cap on the generic edges.
# ---------------------------------------------------------------------------
# SPECIFICITY (label-free). An edge type's specificity is how CONCENTRATED
# its endpoints are in a node class: the share of the type's edges whose
# endpoints fall in the most frequent LENS-TYPE pair, measured on the dev
# corpus by diagnostics/_p2_edge_census.py.  It uses the node's own lens
# type (the graph's "class"), never a semantic label, so nothing
# label-based enters the mechanism.
#
# WHY IT EXISTS. The generic proximity edges (TOPOLOGY, DEGREE, CONTACT)
# connect ARBITRARY structural anchors within a radius, so their endpoints
# are spread over the whole node-type population: their specificity is low
# and their parameters are near-constant (a distance, a rank).  When two
# graphs are compared, those edges contribute a similarity that carries
# little information about the relation, and they can out-vote the
# specific relations.  So the specificity CAPS their influence AT MATCHING
# TIME ONLY: the edges are still EMITTED, with their own parameters, and
# the graph's edge counts do not change.  Nothing about the emission is
# touched.
#
# The measured values live in EDGE_SPECIFICITY (below); the census writes
# the JSON they are derived from.
EDGE_SPECIFICITY: Dict[str, float] = {
    # edge class -> endpoint concentration in [0, 1]
    #
    # MEASURED by diagnostics/_p2_edge_census.py on the dev corpus (144
    # images) and reproducible from diagnostics/_p2_edge_census.json.
    # The value is the CONCENTRATION EXCESS over chance (the share of the
    # type's edges whose endpoint pair is the most frequent one, minus the
    # share that pair has by chance under the corpus' own node-type
    # distribution), DISCOUNTED by the relation's SELF-CONCENTRATION (the
    # share of its endpoints' incident edges that are of this type: a
    # relation that dominates its neighbourhood is a mere proximity
    # graph).  Both factors are LABEL-FREE: the node class is the
    # endpoint's LENS TYPE, the graph's own vocabulary.
    "EDGE_AMODAL_LINK": 0.996,
    "EDGE_BORDER_OWNERSHIP": 0.837,
    "EDGE_COCIRCULAR": 0.993,
    "EDGE_CONCENTRICITY": 0.607,
    "EDGE_CONTACT": 0.597,
    "EDGE_CONTOUR_CONTINUITY": 0.980,
    "EDGE_CONTOUR_ORIENTATION": 0.576,
    "EDGE_CO_LOCATED": 0.282,
    "EDGE_CURVATURE": 0.991,
    "EDGE_CURVATURE_SIGN_CONTINUITY": 0.984,
    "EDGE_CURVE_CONTAINMENT": 0.974,
    "EDGE_JUNCTION_CONTOUR": 0.996,
    "EDGE_JUNCTION_ORIENTATION": 1.000,
    "EDGE_LINE_TERMINATION": 1.000,
    "EDGE_ORIENTATION_BEND": 0.978,
    "EDGE_ORIENTATION_CONTINUITY": 0.972,
    "EDGE_ORIENTATION_JUNCTION": 0.984,
    "EDGE_PARALLEL": 0.993,
    "EDGE_PART_CONTAINMENT": 0.992,
    "EDGE_RADIALITY": 0.987,
    "EDGE_REGION_CONTENT": 0.997,
    "EDGE_SHARED_BOUNDARY": 0.999,
    "EDGE_SPIRALITY": 0.993,
    "EDGE_SYMMETRY": 0.992,
    "EDGE_TEXTURE_BOUNDARY": 0.408,
}

#: The generic proximity edges: the ones whose endpoints are NOT
#: concentrated in any node class.  They are the ONLY classes the
#: specificity cap acts on (a specific relation is already at 1.0).
#:
#: TOPOLOGY and DEGREE are listed here for the record, but they carry NO
#: EdgeParams at all: the binder emits them as bare (src, dst) edges, so
#: they never reach a parametric comparison.  They are INTENTIONALLY
#: INDEX-ONLY (their evidence IS the neighbourhood, and the neighbourhood
#: is the relation).
GENERIC_PROXIMITY_EDGES: FrozenSet[str] = frozenset({
    "EDGE_CONTACT",
    "EDGE_SURFACE_GROUPING",
    "EDGE_CO_LOCATED",
    "EDGE_TEXTURE_BOUNDARY",
    "EDGE_CONTOUR_ORIENTATION",
})

#: The floor of the matching-time cap: a generic edge never loses more
#: than this share of its influence, so the cap can never silence a
#: relation outright.
EDGE_SPECIFICITY_FLOOR: float = 0.25


def edge_match_weight(class_name: str) -> float:
    """The MATCHING-TIME weight of an edge class: its specificity, capped.

    1.0 for a specific relation (the cap never bites), the measured
    specificity (never below EDGE_SPECIFICITY_FLOOR) for a GENERIC one
    (the set in GENERIC_PROXIMITY_EDGES).  This affects the SIMILARITY of
    two edges when they are COMPARED, never their emission.
    """
    if class_name not in GENERIC_PROXIMITY_EDGES:
        return 1.0
    spec = EDGE_SPECIFICITY.get(class_name)
    if spec is None:
        return 1.0
    return float(max(EDGE_SPECIFICITY_FLOOR, min(1.0, spec)))


EDGE_FAMILY_COMPATIBILITY: Dict[str, FrozenSet[str]] = {
    "EDGE_ORIENTATION_CONTINUITY": frozenset({
        "EDGE_ORIENTATION_CONTINUITY",
        "EDGE_CONTOUR_CONTINUITY",
        "EDGE_COCIRCULAR",
    }),
    "EDGE_CONTOUR_CONTINUITY": frozenset({
        "EDGE_CONTOUR_CONTINUITY",
        "EDGE_ORIENTATION_CONTINUITY",
        "EDGE_COCIRCULAR",
    }),
    "EDGE_COCIRCULAR": frozenset({
        "EDGE_COCIRCULAR",
        "EDGE_ORIENTATION_CONTINUITY",
        "EDGE_CONTOUR_CONTINUITY",
    }),
    "EDGE_PARALLEL": frozenset({
        "EDGE_PARALLEL",
    }),
    "EDGE_ORIENTATION_BEND": frozenset({
        "EDGE_ORIENTATION_BEND",
        "EDGE_ORIENTATION_JUNCTION",
        "EDGE_JUNCTION_CONTOUR",
    }),
    "EDGE_ORIENTATION_JUNCTION": frozenset({
        "EDGE_ORIENTATION_JUNCTION",
        "EDGE_ORIENTATION_BEND",
        "EDGE_JUNCTION_CONTOUR",
    }),
    "EDGE_JUNCTION_CONTOUR": frozenset({
        "EDGE_JUNCTION_CONTOUR",
        "EDGE_ORIENTATION_JUNCTION",
        "EDGE_ORIENTATION_BEND",
    }),
}

def derive_edge_ranges() -> Dict[str, Tuple[float, ...]]:
    """Derive constitutional ranges adhering to the N/2 circular rule."""
    ranges: Dict[str, Tuple[float, ...]] = {}
    for class_name, slots in EDGE_SCHEMAS.items():
        slot_ranges = []
        for s in slots:
            if s.domain == "circular":
                slot_ranges.append(s.n_bins / 2.0)
            else:
                slot_ranges.append(s.n_bins)
        ranges[class_name] = tuple(slot_ranges)
    return ranges

def derive_circular_cycles() -> Dict[str, Dict[int, int]]:
    """Derive circular cycle lengths strictly for circular slots."""
    cycles: Dict[str, Dict[int, int]] = {}
    for class_name, slots in EDGE_SCHEMAS.items():
        slot_cycles: Dict[int, int] = {}
        for i, s in enumerate(slots):
            if s.domain == "circular":
                slot_cycles[i] = int(round(s.n_bins))
        if slot_cycles:
            cycles[class_name] = slot_cycles
    return cycles

def derive_identity_slots() -> Dict[str, Tuple[int, ...]]:
    """Derive NAP categorical gate slot indices."""
    identity: Dict[str, Tuple[int, ...]] = {}
    for class_name, slots in EDGE_SCHEMAS.items():
        id_slots = tuple(i for i, s in enumerate(slots) if s.is_identity)
        if id_slots:
            identity[class_name] = id_slots
    return identity

def derive_reversal_specs() -> Tuple[
    Dict[str, Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]],
    Dict[str, Dict[int, int]],
]:
    """Derive endpoint-swap reversal specifications.

    [Stage 1.2 fix] Reciprocal swap declarations (slot 4 says swap->5 AND
    slot 5 says swap->4) collapse to ONE canonical pair — emitting both
    would apply the swap twice and cancel it, leaving tail/head roles
    unreversed on backward traversal.
    """
    reversal_table: Dict[str, Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]] = {}
    rotation_table: Dict[str, Dict[int, int]] = {}
    for class_name, slots in EDGE_SCHEMAS.items():
        swaps: List[Tuple[int, int]] = []
        reflects: List[int] = []
        rotates: Dict[int, int] = {}
        for i, s in enumerate(slots):
            if s.reversal in ("reflect_linear", "reflect_circular"):
                reflects.append(i)
            elif isinstance(s.reversal, tuple) and s.reversal[0] == "swap":
                j = s.reversal[1]
                if i == j:
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair not in swaps:
                    swaps.append(pair)
            elif isinstance(s.reversal, tuple) and s.reversal[0] == "rotate":
                rotates[i] = s.reversal[1]
        if swaps or reflects:
            reversal_table[class_name] = (tuple(swaps), tuple(reflects))
        if rotates:
            rotation_table[class_name] = rotates
    return reversal_table, rotation_table

CONSTITUTIONAL_EDGE_RANGES.update(derive_edge_ranges())

CIRCULAR_EDGE_CYCLE_LENGTHS.update(derive_circular_cycles())

EDGE_IDENTITY_SLOTS.update(derive_identity_slots())

#: [U31 / M4] A U30 slot whose quantiser is `lookup_key` names a TOPOLOGY
#: (junction class, occlusion cue, containment side), not a graded quantity:
#: a measured disagreement is a contradiction, not a penalty.  Deriving the
#: gate from the unit table keeps it from drifting away from the axes it
#: gates, and UNKNOWN stays a wildcard exactly as in the legacy NAP gate.
#: Kept SEPARATE from EDGE_IDENTITY_SLOTS, which is keyed by class name:
#: `EDGE_OCCLUSION` is both a legacy 3-slot class and a 4-slot category, so a
#: merged table would gate the legacy class' slot 1.  Resolution is by ARITY
#: in EdgeParams.similarity, mirroring `_is_u30_edge_params`.
U30_EDGE_IDENTITY_SLOTS: Dict[str, Tuple[int, ...]] = {
    cls: tuple(i for i, s in enumerate(specs) if s.quantiser == "lookup_key")
    for cls, specs in U30_EDGE_UNIT_TABLE.items()
    if any(s.quantiser == "lookup_key" for s in specs)
}

#: [U31 / M4] The U30 bins fallback compares BIN indices, so its Gaussian
#: needs a BIN-unit width (n_bins/2), which is the legacy convention.  The
#: registered CONSTITUTIONAL_EDGE_RANGES entry for these classes is in
#: physical units (radians, wavelengths), which is right for the cross-schema
#: projection and wrong here.
U30_EDGE_BIN_RANGES: Dict[str, Tuple[float, ...]] = {
    cls: tuple(float(s.n_bins) / 2.0 for s in specs)
    for cls, specs in U30_EDGE_UNIT_TABLE.items()
}

_derived_reversal, _derived_rotation = derive_reversal_specs()

for _cls in list(EDGE_REVERSAL):
    if _cls in EDGE_SCHEMAS:
        del EDGE_REVERSAL[_cls]

EDGE_REVERSAL.update(_derived_reversal)

_EDGE_ROTATION_TABLE: Dict[str, Dict[int, int]] = {}

_EDGE_ROTATION_TABLE.update(_derived_rotation)


# --- [ROUND 10 / A] Edge-parameter populations ---------------------------
# The graded companion of the edge-parameter bins, for the relational
# quantities the directive names: relative angle (circular), distance and
# scale ratio. One tuning axis per slot, n_bins = the slot's bin count (or its
# circular cycle). Attached ONLY to these classes, so the change is scoped to
# the contained/parallel/symmetric relations that carry those quantities; every
# other edge keeps the exact bins path. EdgeParams.similarity reads the graded
# overlap whenever a class has axes and both sides' bins are fully defined,
# else it falls back to the bins Gaussian (never a silent drop).
def _edge_axes(cls: str,
               table: Optional[Dict[str, Tuple[SlotSpec, ...]]] = None,
               ) -> Tuple[PopAxis, ...]:
    """The graded axes of one edge class, FROM THE UNIT TABLE.

    The axis layout is the unit table's: one axis per slot, the slot's own
    bin count, and the slot's own circularity.  Deriving the axes from the
    schema instead (the previous behaviour) let the two disagree whenever a
    schema's range was not the writer's bin count (9.0 and 12.0 are
    similarity RANGES, not level counts), so the graded vector could not
    name the bin the lookup key named.
    """
    specs = (table or EDGE_UNIT_TABLE).get(cls, ())
    return tuple(PopAxis(s.name, s.n_bins, circular=s.circular) for s in specs)


#: Every edge class with a unit table has a graded population layout.  The
#: classes whose slots are all lookup keys (INTENTIONALLY INDEX-ONLY) get
#: their categorical axis anyway: the axis carries the UNIFORM activity
#: (the category is the lookup key, not a graded quantity), so it is
#: uninformative and the overlap reads it as 0.0.
EDGE_POPULATION_CLASSES: Tuple[str, ...] = tuple(sorted(EDGE_UNIT_TABLE))


EDGE_POPULATION_AXES: Dict[str, Tuple[PopAxis, ...]] = {
    _cls: _edge_axes(_cls) for _cls in EDGE_POPULATION_CLASSES
}

#: [U30] The same graded-axis derivation over the 8 canonical categories.
#: Additive; EDGE_POPULATION_AXES keeps serving Take25 until Phase 4.
U30_EDGE_POPULATION_CLASSES: Tuple[str, ...] = tuple(sorted(U30_EDGE_UNIT_TABLE))

U30_EDGE_POPULATION_AXES: Dict[str, Tuple[PopAxis, ...]] = {
    _cls: _edge_axes(_cls, U30_EDGE_UNIT_TABLE) for _cls in U30_EDGE_POPULATION_CLASSES
}


def edge_population_for_bins(
    class_name: str, quantized: Tuple[float, ...]
) -> Tuple[Tuple[float, ...], ...]:
    """Graded population of an edge-parameter tuple: one soft tuning bump per
    slot axis, centred at the bin value. Returns () when the class has no axes,
    the arity disagrees, or any slot is UNKNOWN/non-finite (the caller then
    keeps the bins path).

    A slot whose bin is a CATEGORICAL LOOKUP KEY gets the UNIFORM activity,
    exactly as the value-derived path does: the category is an index, not a
    graded quantity, and a bump on it would make the two paths disagree.
    """
    axes = EDGE_POPULATION_AXES.get(class_name)
    specs = EDGE_UNIT_TABLE.get(class_name)
    if axes is None or specs is None or len(axes) != len(quantized):
        return ()
    out: List[Tuple[float, ...]] = []
    for v, ax, s in zip(quantized, axes, specs):
        if v == UNKNOWN or not math.isfinite(v) or v < 0.0 or v >= ax.n_bins:
            return ()
        if s.quantiser == "lookup_key":
            out.append(tuple([1.0 / ax.n_bins] * ax.n_bins))
            continue
        out.append(soft_bump_bins(float(v), ax))
    return tuple(out)


from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=8192)
def _edge_pop_cached(
    class_name: str, quantized: Tuple[float, ...]
) -> Tuple[Tuple[float, ...], ...]:
    return edge_population_for_bins(class_name, quantized)


@dataclass(frozen=True)
class EdgeParams:
    """[C1 / PASS 25 REPLACEMENT] Per-edge structural arrangement and relational parameter container.
    Implements biological Non-Accidental Property (NAP) categorical gating and graded metric similarity.
    """
    class_: str                     # Registered schema name
    quantized: Tuple[float, ...]    # Quantized integer bins (as floats)
    sign: int = 1                   # +1 = Excitatory/Positive evidence, -1 = Gating/Occlusion
    confidence: float = 1.0         # Certainty in [0, 1]
    identity_slots: Optional[Tuple[int, ...]] = None
    # [GRADED POPULATION] The edge's PARAMETER POPULATION: one graded
    # activity vector per registered slot axis (EDGE_POPULATION_AXES). The
    # bins stay the coarse index; the population is the graded truth, and it
    # ACCUMULATES over the writers (the posterior), so a relation that many
    # observations agree on reads differently from one that a single writer
    # asserted. Empty => the bins path (the legacy reading).
    population: Tuple[Tuple[float, ...], ...] = ()
    #: [U31] Which ontology's slot layout `quantized`/`population` use.
    #: "" = the legacy EDGE_SCHEMAS layout; "U30" = U30_EDGE_UNIT_TABLE.
    #: Needed because `EDGE_OCCLUSION` names BOTH a legacy class (4 slots)
    #: and a category (4 slots), so name and arity are both ambiguous.
    #: Appended last so every positional construction stays valid.
    ontology: str = ""
    #: [U31 / Ruling 1] "" or "native" for a v2 edge, "legacy" for a
    #: projection.  A native edge wins a (src, dst, category) collision.
    provenance: str = ""
    #: [U32 / Layer 3] The PRODUCER's class before U30 projection.  A legacy
    #: EDGE_CONTACT projects onto EDGE_PROXIMITY, so `class_` alone cannot
    #: tell a proximity flood from true occlusion evidence.  "" means the
    #: producer did not record one; readers fall back to `class_`.
    origin: str = ""

    def similarity(self, other: Any, use_confidence: bool = True) -> float:
        """Graded Gaussian similarity respecting categorical NAP gates and circular metric slots.

        [S5] Now dispatches to EdgeParamsMixture.similarity when other is a mixture,
        and supports use_confidence to bypass confidence weighting during MCS.
        """
        if other is None:
            return 0.0
        if isinstance(other, EdgeParamsMixture):
            return other.similarity(self, use_confidence=use_confidence)
        if not isinstance(other, EdgeParams):
            return 0.0
        if self.class_ != other.class_:
            # [U31 / Ruling 1] Cross-representation cutover: a legacy edge is
            # projected onto the U30 axes and compared there.  Native/native
            # and legacy/legacy pairs never enter this branch, so a homogeneous
            # graph is untouched.  Lazy import keeps core -> matching acyclic.
            from matching.category_bridge import bridge_similarity
            return bridge_similarity(self, other, use_confidence=use_confidence)
        if len(self.quantized) != len(other.quantized):
            return 0.0
        if self.sign != other.sign:
            return 0.0

        id_slots = (
            set(self.identity_slots) if self.identity_slots is not None
            else set(other.identity_slots) if other.identity_slots is not None
            else set(EDGE_IDENTITY_SLOTS.get(self.class_, ()))
        )
        # [U31 / M4] U30 lookup-key slots are NAPs.  Resolved by ARITY so the
        # legacy EDGE_OCCLUSION class (3 slots) is not gated by the category
        # layout (4 slots) that shares its name.
        if _is_u30_edge_params(self):
            id_slots = set(U30_EDGE_IDENTITY_SLOTS.get(self.class_, ()))

        # 1. Non-Accidental Properties (Identity Slots) Hard Gate
        for idx in id_slots:
            if idx < len(self.quantized):
                v_self = self.quantized[idx]
                v_other = other.quantized[idx]
                # Wildcards (UNKNOWN) pass unconditionally
                if v_self != UNKNOWN and v_other != UNKNOWN:
                    if abs(v_self - v_other) >= 0.5:
                        return 0.0

        # [U32] UNKNOWN slots are not measurements. Sharing an all-UNKNOWN
        # tuple is not evidence of ANYTHING — if every value is unmeasured,
        # sim = 0 rather than 1. A single measured slot is enough to put the
        # pair through the Gaussian instead of the flat 1.0 fallback.
        #
        # [PHASE 4 / EDGE HONESTY] The rule above was only half in the code:
        # an all-UNKNOWN pair read 0.2 (the branch just below), and a pair
        # whose tuples DIFFER but share no measured slot read 1.0 (the
        # `evaluated_graded_count == 0` fallback further down, or whatever
        # the population path made of two unmeasured axes). Both are a
        # comparison in which nothing was compared. With no slot measured on
        # BOTH sides the similarity is 0.0: no physical evidence. Whether the
        # two edges are the same relation at all is a topological fact the
        # caller already holds (the pair exists); it is not physics.
        # A slot measured on both sides -- categorical or graded -- keeps the
        # pair on the historical path, so categorical agreement that passed
        # the gate above still reads 1.0.
        # P.EDGE_UNMEASURED_IS_NOT_EVIDENCE = False restores the floors.
        if getattr(P, "EDGE_UNMEASURED_IS_NOT_EVIDENCE", True) and not any(
                a != UNKNOWN and b != UNKNOWN
                for a, b in zip(self.quantized, other.quantized)):
            return 0.0
        _n_measured_self = sum(1 for v in self.quantized if v != UNKNOWN)
        _n_measured_other = sum(1 for v in other.quantized if v != UNKNOWN)
        if self.quantized == other.quantized:
            base = 1.0 if (_n_measured_self > 0 or _n_measured_other > 0) else 0.2
        else:
            # [ROUND 10 / A] Graded population overlap for the relational
            # quantities (relative angle / distance / scale ratio). Read when
            # BOTH sides carry a population for this class and no categorical
            # identity slot is present; otherwise the bins Gaussian below is
            # the fallback (never a silent zero).
            _axes = EDGE_POPULATION_AXES.get(self.class_)
            if _axes is None:
                _axes = U30_EDGE_POPULATION_AXES.get(self.class_)
            _base = None
            if (getattr(P, "MATCHER_USE_POPULATION_DISTANCE", False)
                    and self.class_ in U30_EDGE_UNIT_TABLE
                    and self.population and other.population):
                # [U31 / S2b] The graded-distance metric.  Identity slots are
                # already handled by the hard gate above; this reads only the
                # axes both sides measured and reports coverage separately.
                from matching.population_distance import (
                    category_axis_units,
                    population_similarity_masked,
                )
                _units = category_axis_units(self.class_)
                _dist_sim, _dist_cov = population_similarity_masked(
                    self.population, other.population, _units,
                    agg=str(getattr(P, "MATCHER_POP_DIST_AGG", "mean")),
                    coverage_floor=float(getattr(
                        P, "MATCHER_POP_DIST_COVERAGE_FLOOR", 0.5)),
                )
                if _dist_sim is not None:
                    _base = _dist_sim
            if (_base is None and _axes is not None
                    and getattr(P, "EDGE_POPULATION_CODING_ENABLED", True)
                    and not id_slots):
                # [GRADED POPULATION] The STORED population is the posterior
                # over the writers of this relation, so it is read first; the
                # bins-derived reading is the fallback for an edge that has
                # no accumulated population yet.
                _pa = self.population or _edge_pop_cached(self.class_,
                                                           self.quantized)
                _pb = other.population or _edge_pop_cached(other.class_,
                                                            other.quantized)
                if _pa and _pb:
                    _base = population_similarity(_pa, _pb, _axes)

            if _base is not None:
                base = _base
            else:
                ranges = CONSTITUTIONAL_EDGE_RANGES.get(self.class_)
                if _is_u30_edge_params(self):
                    ranges = U30_EDGE_BIN_RANGES.get(self.class_)
                if ranges is None or len(ranges) != len(self.quantized):
                    return 0.0
                cycle_lengths = CIRCULAR_EDGE_CYCLE_LENGTHS.get(self.class_, {})

                total_sq = 0.0
                evaluated_graded_count = 0
                for i, (a, b) in enumerate(zip(self.quantized, other.quantized)):
                    if i in id_slots:
                        continue  # Categorical slots already passed hard gate
                    if a == UNKNOWN or b == UNKNOWN:
                        continue  # Wildcards contribute no distance penalty
                    evaluated_graded_count += 1
                    d = abs(a - b)
                    if cycle_lengths and i in cycle_lengths:
                        n = cycle_lengths[i]
                        d = min(d, n - d)
                    sigma = ranges[i] / 3.0 if ranges[i] > 0.0 else 1.0
                    total_sq += (d / sigma) ** 2

                if evaluated_graded_count > 0:
                    base = math.exp(-total_sq / float(evaluated_graded_count))
                else:
                    base = 1.0

        conf_factor = min(self.confidence, other.confidence) if use_confidence else 1.0
        # [P2] The SPECIFICITY CAP lives at the COMPARISON site
        # (polymorphic_edge_similarity), not here: this primitive keeps its
        # contract ("identical parameters -> 1.0"), and the cap acts on the
        # GENERIC relations only, where it is supposed to act.
        return base * conf_factor

def _is_u30_edge_params(ep: "EdgeParams") -> bool:
    """[U31] True when `ep` uses a U30 category layout.

    ARITY decides, not the class name: `EDGE_OCCLUSION` exists in both
    EDGE_SCHEMAS (3 slots) and U30_EDGE_UNIT_TABLE (4 slots), and a name test
    captured the legacy class.  A params tuple whose length matches the U30
    table is the category; otherwise the legacy schema owns the name.
    """
    if getattr(ep, "ontology", "") == "U30":
        return True
    if getattr(ep, "ontology", ""):
        return False
    # Unmarked params: only a value tuple that is VALID for the U30 layout
    # (every bin inside its axis) can be a category.  A legacy EDGE_OCCLUSION
    # tuple fails this (its slot 1 has 9 bins, the U30 slot 1 has 4).
    specs = U30_EDGE_UNIT_TABLE.get(ep.class_)
    if specs is None or len(specs) != len(ep.quantized):
        return False
    for v, s in zip(ep.quantized, specs):
        if v == UNKNOWN:
            continue
        if not math.isfinite(v) or v < 0.0 or v >= float(s.n_bins):
            return False
    return True


def edge_measured_axes(qp: "EdgeParams", cp: "EdgeParams") -> int:
    """[U32 / Part B.2] Number of axes MEASURED ON BOTH SIDES of an edge
    pair.

    Counts slots `i` such that both params carry a non-UNKNOWN bin AND
    `i` is not a categorical gate (identity/NAP slot: those pass or kill
    through the hard gate and are not graded evidence). This is the exact
    set of axes that can carry graded similarity for the pair — the same
    re-normalised slot set the Gaussians in `similarity` read.

    Used by the matcher to scale an edge's evidence mass by coverage
    ([Part B.2 / Ruling 2]). An unmeasured axis NEVER penalizes; coverage
    only ever ADDS mass (statistical facilitation, not a gate).
    """
    if qp is None or cp is None:
        return 0
    if qp.class_ != cp.class_ or len(qp.quantized) != len(cp.quantized):
        return 0
    # [mirrors similarity's slot resolution exactly: params' pins first, U30
    # category layouts override via arity, else the legacy table]
    if qp.identity_slots is not None:
        id_slots = set(qp.identity_slots)
    elif cp.identity_slots is not None:
        id_slots = set(cp.identity_slots)
    else:
        id_slots = set(EDGE_IDENTITY_SLOTS.get(qp.class_, ()))
    if _is_u30_edge_params(qp):
        id_slots = set(U30_EDGE_IDENTITY_SLOTS.get(qp.class_, ()))
    # [P4b] A slot the unit table declares a categorical LOOKUP KEY is not a
    # graded axis either. EDGE_SCHEMAS (which derives the identity slots
    # above) agrees with the unit table on 8 of its 12 lookup keys and
    # declares the other four linear or ungated: the membership flag of
    # CONTOUR_ORIENTATION / JUNCTION_ORIENTATION / REGION_CONTENT (a literal
    # containment = 1.0, EDGE_INDEX_ONLY) and OCCLUSION.source. Those four
    # were counted as a measured axis, so a bare membership flag earned
    # f(1) = 2 of edge mass. P.EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS
    # (default False = the historical count) leaves them out.
    key_slots: FrozenSet[int] = frozenset()
    if getattr(P, "EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS", False):
        key_slots = _edge_lookup_key_slots(qp) | _edge_lookup_key_slots(cp)
    n = 0
    for i, (a, b) in enumerate(zip(qp.quantized, cp.quantized)):
        if a == UNKNOWN or b == UNKNOWN:
            continue                      # unmeasured axes never penalize
        if i in id_slots:
            continue                      # categorical gates are not graded axes
        if i in key_slots:
            continue                      # nor are lookup keys [P4b]
        n += 1
    return n


def _edge_lookup_key_slots(ep: Any) -> FrozenSet[int]:
    """[P4b] Slot indices of `ep` that its unit table declares a categorical
    LOOKUP KEY (`quantiser == "lookup_key"`): an index, not a graded
    quantity. The table is chosen as `_is_u30_edge_params` chooses the
    layout; an arity the table does not describe yields none."""
    table = U30_EDGE_UNIT_TABLE if _is_u30_edge_params(ep) else EDGE_UNIT_TABLE
    specs = table.get(ep.class_, ())
    if len(specs) != len(ep.quantized):
        return frozenset()
    return frozenset(
        i for i, s in enumerate(specs) if s.quantiser == "lookup_key")


def _edge_categorical_slots(ep: "EdgeParams") -> FrozenSet[int]:
    """[P4b] Every slot of `ep` that is categorical: its identity (NAP)
    slots, resolved as `EdgeParams.similarity` resolves them, and its lookup
    keys."""
    if ep.identity_slots is not None:
        ids = set(ep.identity_slots)
    else:
        ids = set(EDGE_IDENTITY_SLOTS.get(ep.class_, ()))
    if _is_u30_edge_params(ep):
        ids = set(U30_EDGE_IDENTITY_SLOTS.get(ep.class_, ()))
    ids.update(_edge_lookup_key_slots(ep))
    return frozenset(ids)


def _graded_axes_plain(a: "EdgeParams", b: "EdgeParams") -> int:
    if not isinstance(a, EdgeParams) or not isinstance(b, EdgeParams):
        return 0
    cat_a = _edge_categorical_slots(a)
    cat_b = _edge_categorical_slots(b)
    if a.class_ == b.class_:
        if len(a.quantized) != len(b.quantized):
            return 0
        n = 0
        for i, (x, y) in enumerate(zip(a.quantized, b.quantized)):
            if x == UNKNOWN or y == UNKNOWN:
                continue
            if i in cat_a or i in cat_b:
                continue
            n += 1
        return n
    # cross-schema: the shared axes polymorphic_edge_similarity compares
    if b.class_ not in get_compatible_types(a.class_):
        return 0
    n = 0
    for idx_a, idx_b, _tag in get_shared_subspace(a.class_, b.class_):
        if idx_a >= len(a.quantized) or idx_b >= len(b.quantized):
            continue
        if a.quantized[idx_a] == UNKNOWN or b.quantized[idx_b] == UNKNOWN:
            continue
        if idx_a in cat_a or idx_b in cat_b:
            continue
        n += 1
    return n


def edge_graded_axes(ep1: Any, ep2: Any) -> int:
    """[P4b] Number of GRADED axes measured on BOTH sides of an edge pair:
    the quantities a similarity of the two can be a PHYSICAL reading of.

    A slot counts when neither side holds UNKNOWN in it and it is neither an
    identity (NAP) slot nor a lookup key of the unit table. Same schema: the
    slots themselves. Cross-schema (a compatible family): the shared axes
    `polymorphic_edge_similarity` compares, under the same conditions. A
    mixture: the best over its prototypes. No parameters on either side: 0.

    0 means the pair has no physics to read: whatever similarity it earned
    is the TOPOLOGICAL fact that the relation exists on both sides (a
    membership flag, a parameter-less predicate, a categorical agreement) --
    a count. `evidence_score` uses this to keep the count apart from the
    physics (P.EVIDENCE_RELATION_COUNT_APART). Unlike `edge_measured_axes`
    it does not depend on any switch and it reads cross-schema pairs.
    """
    if ep1 is None or ep2 is None:
        return 0
    protos1 = ep1.protos if isinstance(ep1, EdgeParamsMixture) else (ep1,)
    protos2 = ep2.protos if isinstance(ep2, EdgeParamsMixture) else (ep2,)
    best = 0
    for a in protos1:
        for b in protos2:
            n = _graded_axes_plain(a, b)
            if n > best:
                best = n
    return best


#: [U31 / M2] Endpoint reversal for the U30 categories.  The legacy spec
#: table is derived from EDGE_SCHEMAS, which has no U30 entry, so every
#: category edge used to come back UNREVERSED on backward traversal.  A slot
#: is listed only when swapping the endpoints changes its measured value:
#:   CONTINUITY.turn_angle    +theta -> -theta          (reflection about 0)
#:   CONTAINMENT.side         -1 <-> +1                (reflection about 0)
#:   CONTAINMENT.polar_angle  +pi (half the 24-cycle)  (rotation)
#:   PROXIMITY.scale_ratio    log2(a/b) -> -log2(a/b)  (reflection)
#: Bearings are AXIAL over [0, pi): reversing the pair leaves the axis
#: unchanged, so they are deliberately absent (no reflection and no rotation).
_U30_REFLECT_SLOTS = {
    "EDGE_PROXIMITY": (2,),
    "EDGE_CONTINUITY": (1,),
    "EDGE_CONTAINMENT": (4,),
}
_U30_ROTATE_SLOTS = {
    "EDGE_CONTAINMENT": (1,),
}


def _reverse_u30_params(ep: "EdgeParams") -> "EdgeParams":
    """[U31 / M2] Schema-driven reversal over U30_EDGE_UNIT_TABLE.

    Same conventions as the legacy branch: reflection about zero of a slot's
    own span is (n_bins - 1 - b), and a rotation moves by half the axis.
    """
    specs = U30_EDGE_UNIT_TABLE.get(ep.class_, ())
    q = list(ep.quantized)
    for idx in _U30_REFLECT_SLOTS.get(ep.class_, ()):
        if idx >= len(q) or idx >= len(specs):
            continue
        if q[idx] == UNKNOWN or not math.isfinite(q[idx]):
            continue
        n_levels = int(specs[idx].n_bins)
        if n_levels <= 1:
            continue
        max_bin = float(n_levels - 1)
        q[idx] = float(clamp(max_bin - q[idx], 0.0, max_bin))
    for idx in _U30_ROTATE_SLOTS.get(ep.class_, ()):
        if idx >= len(q) or idx >= len(specs):
            continue
        if q[idx] == UNKNOWN or not math.isfinite(q[idx]):
            continue
        n_levels = int(specs[idx].n_bins)
        if n_levels <= 0:
            continue
        q[idx] = float((int(round(q[idx])) + n_levels // 2) % n_levels)
    return _dc_replace(ep, quantized=tuple(q))


def reverse_edge_params(ep: "EdgeParams") -> "EdgeParams":
    """Reverse relational edge parameters on backward traversal.
    Schema-driven (Stage 1.2): swaps, domain-aware reflections, and
    rotations derived from EDGE_SCHEMAS. Maintains clean unsigned
    non-negative bin bounds across all domains."""
    if not isinstance(ep, EdgeParams):
        return ep
    if _is_u30_edge_params(ep):
        return _reverse_u30_params(ep)
    schema = EDGE_SCHEMAS.get(ep.class_)
    if schema is None:
        return ep

    spec = EDGE_REVERSAL.get(ep.class_)
    swaps = spec[0] if spec else ()
    reflects = spec[1] if spec else ()
    rotates = _EDGE_ROTATION_TABLE.get(ep.class_, {})

    q = list(ep.quantized)

    # 1. Swap asymmetric pairs (canonical, deduped at derivation time)
    for pair in swaps:
        if len(pair) == 2:
            i, j = pair
            if i < len(q) and j < len(q):
                q[i], q[j] = q[j], q[i]

    # 2. Reflect slots (domain-aware, never emits negative bins)
    #
    # [P2] The reflection is now read from the UNIT TABLE: a slot's level
    # count is the table's n_bins, and the reflection of a bin about the
    # zero of its own quantity is (n_bins - 1 - b).  The previous circular
    # branch used (cycle - b) % cycle, which is off by one at b == 0 (the
    # reflection of the bin CENTRE 0.5 about zero is the bin 7.5, i.e. bin
    # 7, not bin 0) -- so reversing an edge whose turn was the lowest bin
    # left the turn unchanged.
    cycles = CIRCULAR_EDGE_CYCLE_LENGTHS.get(ep.class_, {})
    specs = EDGE_UNIT_TABLE.get(ep.class_, ())
    for idx in reflects:
        if idx < len(q) and idx < len(schema) and q[idx] != UNKNOWN and math.isfinite(q[idx]):
            n_levels = None
            if idx < len(specs):
                n_levels = specs[idx].n_bins
            elif idx in cycles:
                n_levels = int(cycles[idx])
            else:
                n_levels = int(round(schema[idx].n_bins))
            if n_levels <= 1:
                continue
            max_bin = float(n_levels - 1)
            q[idx] = float(clamp(max_bin - q[idx], 0.0, max_bin))

    # 3. Rotate circular slots (e.g. EDGE_CONTACT normal +12 on cycle 24)
    for idx, offset in rotates.items():
        if idx < len(q) and q[idx] != UNKNOWN and math.isfinite(q[idx]):
            c = cycles.get(idx, 24)
            q[idx] = float((int(round(q[idx])) + offset) % c)

    return _dc_replace(ep, quantized=tuple(q))

def get_shared_subspace(class_a: str, class_b: str) -> List[Tuple[int, int, str]]:
    """Returns list of (slot_idx_a, slot_idx_b, common_tag) for polymorphic projection.

    [Stage 2 bug fix] The draft required sa.n_bins == sb.n_bins for a
    shared slot. The draft's own OC/CC worked example pairs slots with
    IDENTICAL bin counts (8.0/8.0 after the COCIRCULAR realignment), so
    the guard never fires there — but as a general rule it is wrong: the
    graded Gaussian metric compares slot VALUES (each normalized by its
    own schema range), not raw bin counts. Two slots measuring the same
    invariant at different resolutions are still comparable. Bin-count
    equality (and arity differences) are already absorbed by the
    Szymkiewicz-Simpson dim_discount in polymorphic_edge_similarity.
    """
    schema_a = EDGE_SCHEMAS.get(class_a, ())
    schema_b = EDGE_SCHEMAS.get(class_b, ())
    if not schema_a or not schema_b:
        return []
    shared = []
    for i, sa in enumerate(schema_a):
        for j, sb in enumerate(schema_b):
            if sa.tag == sb.tag and sa.domain == sb.domain:
                shared.append((i, j, sa.tag))
    return shared

def get_compatible_types(class_a: str) -> FrozenSet[str]:
    """Returns compatible edge types for polymorphic matching."""
    return EDGE_FAMILY_COMPATIBILITY.get(class_a, frozenset({class_a}))

def polymorphic_edge_similarity(
    ep1: "EdgeParams",
    ep2: "EdgeParams",
    use_confidence: bool = False,
    edge_types: "Optional[Tuple[Any, Any]]" = None,
) -> float:
    """Relational edge similarity with cross-schema polymorphic support.

    Strictly symmetric: polymorphic_edge_similarity(A, B) ==
    polymorphic_edge_similarity(B, A) — the shared-subspace tags match
    commutatively, the graded metric uses each side's own schema ranges,
    and the confidence factor is a min().

    Same schema: full multi-slot Gaussian fidelity via EdgeParams.similarity.
    Compatible cross-schema pair: graded Gaussian over the shared tag
    subspace, discounted by the Szymkiewicz-Simpson overlap coefficient
    n_shared / min(len(s1), len(s2)) (no abstraction penalty) and the
    EDGE_CROSS_FAMILY_PRIOR (default 0.90).
    Incompatible pair: 0.0.
    """
    if ep1 is None or ep2 is None:
        # [BLACK-HOLE FIX] Two edges of the SAME relational predicate that
        # correspond to each other but carry no EdgeParams are a STRUCTURAL
        # MATCH, not a mismatch. Returning 0.0 here zeroed every
        # unparameterised edge, driving relational_consistency to 0 and
        # capping evidence_score at EVIDENCE_ALPHA_CONJUNCTION. Many emitted
        # EdgeTypes are topology-only and never receive params, so this is
        # the common case rather than an edge case. Missing the edge-type
        # pair (default) keeps the old strict-0.0 behaviour.
        #
        # [F6 / PHASE 1] The branch below used to `return 0.0` as well, so
        # the fix described above was inert: a topology-only core edge could
        # never be matched, whatever the query held. It now returns 1.0 when
        #   * the caller supplied the predicate pair and the two are EQUAL, and
        #   * BOTH sides carry no params (the "carry no EdgeParams" case).
        # A pair with params on ONE side only stays 0.0: a missing
        # measurement must not read as a perfect one, and it keeps
        # EdgeParams.similarity(None) == 0.0 true from either direction.
        # The result is a structural (topological) match with zero measured
        # axes -- edge_measured_axes(None, None) == 0 -- so it earns no
        # coverage facilitation in the matcher's edge mass.
        # P.EDGE_UNPARAMETERISED_STRUCTURAL_MATCH = False restores the
        # historical strict 0.0.
        if (
            ep1 is None
            and ep2 is None
            and edge_types is not None
            and edge_types[0] is not None
            and edge_types[0] == edge_types[1]
            and getattr(P, "EDGE_UNPARAMETERISED_STRUCTURAL_MATCH", True)
        ):
            return 1.0
        return 0.0

    # 1. Same schema: full multi-slot Gaussian fidelity
    if ep1.class_ == ep2.class_:
        return float(ep1.similarity(ep2, use_confidence=use_confidence))

    # 2. Family compatibility gate (symmetric — enforced by
    # verify_schema_consistency)
    if ep2.class_ not in get_compatible_types(ep1.class_):
        return 0.0

    shared = get_shared_subspace(ep1.class_, ep2.class_)
    if not shared:
        return 0.0

    schema1 = EDGE_SCHEMAS.get(ep1.class_, ())
    schema2 = EDGE_SCHEMAS.get(ep2.class_, ())

    n_shared = len(shared)
    dim_discount = n_shared / float(max(1, min(len(schema1), len(schema2))))
    cross_family_prior = float(getattr(P, "EDGE_CROSS_FAMILY_PRIOR", 0.90))

    ranges1 = CONSTITUTIONAL_EDGE_RANGES.get(ep1.class_, ())
    cycles1 = CIRCULAR_EDGE_CYCLE_LENGTHS.get(ep1.class_, {})

    total_sq = 0.0
    n_compared = 0
    for idx_a, idx_b, _tag in shared:
        if idx_a >= len(ep1.quantized) or idx_b >= len(ep2.quantized):
            continue
        va = ep1.quantized[idx_a]
        vb = ep2.quantized[idx_b]
        if va == UNKNOWN or vb == UNKNOWN:
            continue
        n_compared += 1
        d = abs(va - vb)
        if idx_a in cycles1:
            n = cycles1[idx_a]
            d = min(d, n - d)
        sigma = (ranges1[idx_a] / 3.0) if idx_a < len(ranges1) and ranges1[idx_a] > 0 else 1.0
        total_sq += (d / sigma) ** 2

    if getattr(P, "EDGE_UNMEASURED_IS_NOT_EVIDENCE", True):
        # [PHASE 4 / EDGE HONESTY] The mean runs over the shared axes that
        # were MEASURED ON BOTH SIDES, as the same-schema Gaussian already
        # does. Dividing by n_shared counted every unmeasured shared axis as
        # a perfect agreement, and a pair with no measured shared axis read
        # exp(0) = 1.0. The dimensional discount below is structure, not
        # measurement, and is unchanged.
        if n_compared == 0:
            return 0.0
        base_sim = math.exp(-total_sq / float(n_compared))
    else:
        base_sim = math.exp(-total_sq / float(max(1, n_shared)))
    conf_factor = min(ep1.confidence, ep2.confidence) if use_confidence else 1.0
    # [P2] The matching-time specificity cap (see EDGE_SPECIFICITY).
    spec_w = min(edge_match_weight(ep1.class_),
                 edge_match_weight(ep2.class_))
    return float(base_sim * dim_discount * cross_family_prior * conf_factor
                 * spec_w)

def verify_schema_consistency() -> None:
    """Diagnostic verification: asserts exact agreement across derived values
    and the N/2 circular rule. Safe for testing; NOT called at module import
    time (per Stage 1.2 directive).

    [Stage 2] Also validates cross-type compatibility symmetry: every
    (A -> B) mapping must have a reciprocal (B -> A) mapping, with every
    referenced class present in the table.
    """
    derived_ranges = derive_edge_ranges()
    derived_cycles = derive_circular_cycles()
    for class_name, slots in EDGE_SCHEMAS.items():
        assert len(slots) == len(derived_ranges[class_name]), f"Arity mismatch in {class_name}"
        if class_name in derived_cycles:
            for slot_idx, cycle_len in derived_cycles[class_name].items():
                expected = cycle_len / 2.0
                actual = derived_ranges[class_name][slot_idx]
                assert abs(actual - expected) < 1e-9, (
                    f"{class_name} slot {slot_idx}: range {actual} != cycle {cycle_len} / 2 = {expected}"
                )
    for c_a, compatible_set in EDGE_FAMILY_COMPATIBILITY.items():
        for c_b in compatible_set:
            assert c_b in EDGE_FAMILY_COMPATIBILITY, f"Missing compatibility entry for {c_b}"
            assert c_a in EDGE_FAMILY_COMPATIBILITY[c_b], (
                f"Asymmetric compatibility: {c_b} in {c_a}, but {c_a} not in {c_b}"
            )

@dataclass(eq=False)
class EdgeParamsMixture:
    """Bounded, time-decayed mixture of joint EdgeParams prototypes.
    Preserves true multimodal relational geometry across view changes without averaging.
    """
    protos: List[EdgeParams]
    mass: List[float]
    t_last: int

    class_ = property(lambda s: s.protos[0].class_)
    quantized = property(lambda s: s.protos[0].quantized)  # Founder prototype: stable for WL hashing
    identity_slots = property(lambda s: s.protos[0].identity_slots)
    sign = property(lambda s: s.dominant.sign)

    @property
    def dominant(self) -> EdgeParams:
        k = max(range(len(self.mass)), key=lambda i: (self.mass[i], -i))
        return self.protos[k]

    @property
    def confidence(self) -> float:
        m = sum(self.mass)
        return sum(w * p.confidence for w, p in zip(self.mass, self.protos)) / m if m > 0.0 else 0.0

    @classmethod
    def from_point(cls, ep: EdgeParams, t: int, mass: float = 1.0) -> "EdgeParamsMixture":
        return cls([ep], [float(mass)], int(t))

    @staticmethod
    def _close(a: EdgeParams, b: EdgeParams, tol: float) -> bool:
        if a.class_ != b.class_ or a.sign != b.sign or len(a.quantized) != len(b.quantized):
            return False
        cyc = CIRCULAR_EDGE_CYCLE_LENGTHS.get(a.class_, {})
        id_slots = set(EDGE_IDENTITY_SLOTS.get(a.class_, ()))
        for i, (x, y) in enumerate(zip(a.quantized, b.quantized)):
            fx, fy = math.isfinite(x), math.isfinite(y)
            if fx != fy:
                return False
            if not fx or x == UNKNOWN or y == UNKNOWN:
                continue
            if i in id_slots:
                if abs(x - y) >= 0.5:
                    return False
                continue
            d = abs(x - y)
            if i in cyc:
                d = min(d, cyc[i] - d)
            if d > tol:
                return False
        return True

    def observe(self, ep: EdgeParams, t: int, weight: float = 1.0) -> None:
        """Accumulate an observation decaying strictly by elapsed time steps."""
        gamma = getattr(P, "EDGE_MIX_GAMMA", 0.98)
        tol = getattr(P, "EDGE_MIX_MERGE_TOL", 1.0)
        cap = getattr(P, "EDGE_MIX_MAX_MODES", 4)

        dt = max(0, int(t) - self.t_last)
        if dt > 0:
            f = gamma ** dt
            self.mass = [m * f for m in self.mass]
            self.t_last = int(t)

        w = max(float(weight), 1e-6)
        for k, p in enumerate(self.protos):
            if self._close(p, ep, tol):
                self.mass[k] += w
                return

        self.protos.append(ep)
        self.mass.append(w)
        if len(self.protos) > cap:
            drop_idx = min(range(len(self.mass)), key=lambda i: (self.mass[i], i))
            del self.protos[drop_idx]
            del self.mass[drop_idx]

    def flipped_copy(self) -> "EdgeParamsMixture":
        """Return a copy with all joint prototypes reversed for inverted endpoint traversals."""
        return EdgeParamsMixture(
            [reverse_edge_params(p) for p in self.protos],
            list(self.mass),
            self.t_last,
        )

    def _inner(self, other: "EdgeParamsMixture", use_confidence: bool = False) -> float:
        ma, mb = sum(self.mass), sum(other.mass)
        if ma <= 0.0 or mb <= 0.0:
            return 0.0
        return sum(
            (wp / ma) * (wq / mb) * p.similarity(q, use_confidence=use_confidence)
            for p, wp in zip(self.protos, self.mass)
            for q, wq in zip(other.protos, other.mass)
        )

    def similarity(self, other: Any, use_confidence: bool = True) -> float:
        if other is None:
            return 0.0
        if isinstance(other, EdgeParams):
            other = EdgeParamsMixture([other], [1.0], self.t_last)
        if not isinstance(other, EdgeParamsMixture):
            return 0.0
        if len(self.protos) == 1 and len(other.protos) == 1:
            return self.protos[0].similarity(other.protos[0], use_confidence=use_confidence)

        aa = self._inner(self, use_confidence=False)
        bb = other._inner(other, use_confidence=False)
        ab = self._inner(other, use_confidence=False)

        geo = ab / math.sqrt(aa * bb) if (aa > 0.0 and bb > 0.0) else 0.0
        conf = min(self.confidence, other.confidence) if use_confidence else 1.0
        return clamp(geo, 0.0, 1.0) * conf

def coarsen_edge_params_for_wl(ep: Optional[EdgeParams]) -> Tuple[float, ...]:
    """Coarsens graded slots for WL canonical hashing to absorb 1-bin metric jitter.
    Identity slots (NAPs) stay at full resolution; UNKNOWN slots remain -1.0.
    """
    if ep is None or not hasattr(ep, "quantized") or not ep.quantized:
        return ()
    id_slots = (
        set(ep.identity_slots) if getattr(ep, "identity_slots", None) is not None
        else set(EDGE_IDENTITY_SLOTS.get(ep.class_, ()))
    )
    out: List[float] = []
    for i, v in enumerate(ep.quantized):
        if v == UNKNOWN:
            out.append(-1.0)
        elif i in id_slots:
            out.append(float(int(round(v))))
        else:
            out.append(float(math.floor(v / 2.0)))
    return tuple(out)

def _verify_edge_circular_ranges() -> None:
    """[Pass 22] Machine check: every circular slot in
    CIRCULAR_EDGE_CYCLE_LENGTHS for an EDGE_* class must declare a range
    exactly equal to cycle / 2, the same N_HALF convention the node-level
    _verify_circular_ranges enforces. Guards the edge registries against
    the silent arity/cycle drift that Pass 22 exists to fix.
    """
    for class_name, cycles in CIRCULAR_EDGE_CYCLE_LENGTHS.items():
        if not class_name.startswith("EDGE_"):
            continue
        ranges = CONSTITUTIONAL_EDGE_RANGES.get(class_name)
        if ranges is None:
            continue
        for slot_idx, cycle_len in cycles.items():
            if slot_idx >= len(ranges):
                raise RuntimeError(
                    f"{class_name}: circular slot {slot_idx} out of range "
                    f"(arity {len(ranges)})"
                )
            expected = cycle_len / 2.0
            actual = ranges[slot_idx]
            if abs(actual - expected) > 1e-9:
                raise RuntimeError(
                    f"{class_name} slot {slot_idx}: range {actual} != "
                    f"cycle {cycle_len} / 2 = {expected}"
                )

_verify_edge_circular_ranges()

