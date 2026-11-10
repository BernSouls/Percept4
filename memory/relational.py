"""memory/relational.py -- the RELATIONAL CONTEXT group.

WHAT THIS IS.  Every other group of the descriptor reads a part's OWN
appearance: its shape population, its position, its surface, its size.  The
relational context reads what the part is RELATED TO: the mixture of the
edge populations incident to the part's node, tagged by RELATION FAMILY
(continuation, containment, symmetry/parallel, occlusion/depth,
surface/boundary, figure-ground).

So a part is described by two things: what it looks like, and the web of
relations it sits in.  A quadruped's leg is a fragment like any other; what
makes it a leg is that it is CONTINUOUS with the body, OCCLUDED by it, and
in a FIGURE-GROUND relation with it.

THE DESIGN, IN THREE PARTS.

  1. THE FAMILIES.  Every edge class belongs to exactly one relation
     family (RELATION_FAMILY below, with the reason recorded).  The
     families are the COARSE vocabulary of the relational context; the
     edge classes are the fine one.

  2. THE CHANNELS.  A family reads a small set of GRADED CHANNELS -- the
     quantities its edges actually measure, in the WRITER'S units (radians
     for an angle, wavelengths for a distance, a signed scalar for a
     figure-ground statement, a layer count for a depth step).  Each
     channel is (name, unit, span, bins).  An edge class fills a channel by
     naming the SLOT that carries the quantity (EDGE_CHANNELS below), and
     the value is de-binned from the graph's lookup index through the edge
     unit table (core.schema.edge_slot_value), so the channel carries the
     MEASUREMENT, not the bin.

  3. THE CONTEXT.  A part's context is, per family, the MIXTURE of the
     incident edges' channel activity -- the family's own graded vector --
     plus the family's MASS (its share of the part's relational evidence).
     The view's context is the same mixture over the view's whole graph.

WHAT THIS IS NOT.  No label, no category, no mask, no segmentation: the
context is built from the graph the binders already produced.

THE HONEST LIMITS.  A family whose edges carry only a categorical flag
(the membership relations) contributes its MASS and no graded channel: the
context says "this part is contained in something" and not how much, because
the graph does not measure that.  No channel is fabricated to fill the gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.population import PopAxis, soft_bump_bins
from core.schema import (
    UNKNOWN,
    edge_match_weight,
    edge_slot_value,
)

#: THE GROUP'S FUSION WEIGHT (label-free).
#:
#: Every other group's weight is calibrated by diagnostics/_calibrate.py and
#: written into parameters/_text.py -- a file another mission owns, so this
#: mission does not edit it.  The relational group's weight therefore lives
#: HERE, with the SAME derivation the other weights use: reliability (how
#: well the group survives a perturbation) times distinctiveness (how far
#: the group's similarity separates DIFFERENT images), both measured on the
#: dev split WITHOUT a label.  The calibration that produced the value
#: below is diagnostics/_p2_calibrate.py, and the value is replaced there
#: when the measurement changes.
#:
#: The default is 0.0 (the group earns no influence until it is measured),
#: and the measured value is written into this file by the calibration.
#:
#: MEASURED (diagnostics/_p2_calibrate.py, the dev split, 144 images):
#:   the group's own similarity against a perturbed view of the same image
#:   0.923 (a 7 px shift) and 0.923 (a 0.9x scale)   -> RELIABILITY 0.923
#:   the group's mean off-diagonal similarity 0.913 against its diagonal
#:   1.000                                          -> DISTINCTIVENESS 0.087
#:   weight = reliability x distinctiveness          -> 0.0804
#: (the fusion renormalises, so the group takes ~7% of the total weight).
RELATIONAL_GROUP_WEIGHT: float = 0.080430

# ---------------------------------------------------------------------------
# THE RELATION FAMILIES
# ---------------------------------------------------------------------------
RELATION_FAMILIES: Tuple[str, ...] = (
    "continuation",
    "containment",
    "symmetry_parallel",
    "occlusion_depth",
    "surface_boundary",
    "figure_ground",
)

#: Edge class -> relation family, with the reason the class belongs there.
RELATION_FAMILY: Dict[str, str] = {
    # --- continuation: "this continues into that" -----------------------
    "EDGE_CONTOUR_CONTINUITY": "continuation",
    "EDGE_ORIENTATION_CONTINUITY": "continuation",
    "EDGE_CURVATURE": "continuation",
    "EDGE_CURVATURE_SIGN_CONTINUITY": "continuation",
    # --- containment: "this is inside that" -----------------------------
    "EDGE_CURVE_CONTAINMENT": "containment",
    "EDGE_PART_CONTAINMENT": "containment",
    "EDGE_CONTOUR_ORIENTATION": "containment",
    "EDGE_JUNCTION_ORIENTATION": "containment",
    "EDGE_REGION_CONTENT": "containment",
    "EDGE_JUNCTION_CONTOUR": "containment",
    # --- symmetry / parallel: "this mirrors that" -----------------------
    "EDGE_SYMMETRY": "symmetry_parallel",
    "EDGE_PARALLEL": "symmetry_parallel",
    "EDGE_ORIENTATION_BEND": "symmetry_parallel",
    "EDGE_ORIENTATION_JUNCTION": "symmetry_parallel",
    "EDGE_CONCENTRICITY": "symmetry_parallel",
    "EDGE_RADIALITY": "symmetry_parallel",
    "EDGE_SPIRALITY": "symmetry_parallel",
    "EDGE_COCIRCULAR": "symmetry_parallel",
    # --- occlusion / depth: "that is in front of this" -----------------
    "EDGE_OCCLUSION": "occlusion_depth",
    "EDGE_BINOCULAR_DISPARITY": "occlusion_depth",
    "EDGE_AMODAL_LINK": "occlusion_depth",
    # --- surface / boundary: "these two share a surface" ---------------
    "EDGE_SHARED_BOUNDARY": "surface_boundary",
    "EDGE_TEXTURE_BOUNDARY": "surface_boundary",
    "EDGE_SURFACE_GROUPING": "surface_boundary",
    "EDGE_SURFACE_FOLD": "surface_boundary",
    "EDGE_COLOR_OPPONENT_BORDER": "surface_boundary",
    # --- figure-ground: "this side owns that border" --------------------
    "EDGE_BORDER_OWNERSHIP": "figure_ground",
    "EDGE_LINE_TERMINATION": "figure_ground",
    # --- the generic proximity relations: no family of their own --------
    "EDGE_CONTACT": "surface_boundary",
    "EDGE_CO_LOCATED": "surface_boundary",
    "EDGE_CURVE_CONTAINMENT_X": "surface_boundary",
}

#: The families each edge class's REASON, for the report.
RELATION_FAMILY_REASON: Dict[str, str] = {
    "EDGE_CONTOUR_CONTINUITY":
        "a contour continuing into another is a CONTINUATION relation",
    "EDGE_ORIENTATION_CONTINUITY":
        "the V1 association field's collinear continuation",
    "EDGE_CURVATURE":
        "the bend between two adjacent segments is a continuation",
    "EDGE_CURVATURE_SIGN_CONTINUITY":
        "the agreement of the curvature SIGN is a continuation",
    "EDGE_CURVE_CONTAINMENT":
        "a curve complex bundling a member is a CONTAINMENT relation",
    "EDGE_PART_CONTAINMENT":
        "a region shape enclosing a curve complex is a CONTAINMENT relation",
    "EDGE_CONTOUR_ORIENTATION":
        "a contour summarising a segment is a CONTAINMENT relation",
    "EDGE_JUNCTION_ORIENTATION":
        "a junction summarising a segment is a CONTAINMENT relation",
    "EDGE_REGION_CONTENT":
        "a content node inside a region shape is a CONTAINMENT relation",
    "EDGE_JUNCTION_CONTOUR":
        "the junction's incidence in a contour is a CONTAINMENT relation",
    "EDGE_SYMMETRY":
        "a mirror relation is a SYMMETRY relation",
    "EDGE_PARALLEL":
        "a parallel relation is a SYMMETRY/PARALLEL relation",
    "EDGE_ORIENTATION_BEND":
        "the bend between two orientations is an ALIGNMENT relation",
    "EDGE_ORIENTATION_JUNCTION":
        "the junction between two orientations is an ALIGNMENT relation",
    "EDGE_CONCENTRICITY":
        "a concentric relation is a SYMMETRY relation",
    "EDGE_RADIALITY":
        "a radial relation is a SYMMETRY relation",
    "EDGE_SPIRALITY":
        "a spiral relation is a SYMMETRY relation",
    "EDGE_COCIRCULAR":
        "a co-circular relation is a SHAPE-ALIGNMENT relation",
    "EDGE_OCCLUSION":
        "an occlusion order is an OCCLUSION/DEPTH relation",
    "EDGE_BINOCULAR_DISPARITY":
        "a disparity is an OCCLUSION/DEPTH relation",
    "EDGE_AMODAL_LINK":
        "an amodal completion across an occluder is an OCCLUSION relation",
    "EDGE_SHARED_BOUNDARY":
        "two patches sharing an interface is a SURFACE/BOUNDARY relation",
    "EDGE_TEXTURE_BOUNDARY":
        "a texture border is a SURFACE/BOUNDARY relation",
    "EDGE_SURFACE_GROUPING":
        "similar surfaces group into a SURFACE relation",
    "EDGE_SURFACE_FOLD":
        "a fold is a SURFACE relation",
    "EDGE_COLOR_OPPONENT_BORDER":
        "an opponent border is a SURFACE/BOUNDARY relation",
    "EDGE_BORDER_OWNERSHIP":
        "a border's owner is the FIGURE-GROUND relation",
    "EDGE_LINE_TERMINATION":
        "a line ending at a boundary is a FIGURE-GROUND (occlusion) cue",
    "EDGE_CONTACT":
        "a generic proximity contact: SURFACE, downweighted at matching",
    "EDGE_CO_LOCATED":
        "a generic co-location: SURFACE, downweighted at matching",
}


# ---------------------------------------------------------------------------
# THE GRADED CHANNELS
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Channel:
    """One graded channel of a relation family.

    name    the stable identifier
    unit    the unit of the continuous quantity
    lo, hi  the span of that quantity
    n_bins  the number of levels of the channel
    """
    name: str
    unit: str
    lo: float
    hi: float
    n_bins: int

    def axis(self) -> PopAxis:
        return PopAxis(self.name, int(self.n_bins), circular=False)

    def activity(self, value: float) -> Tuple[float, ...]:
        """The soft bump of one measured value on this channel."""
        span = self.hi - self.lo
        if span <= 0.0 or not math.isfinite(value):
            raise ValueError(f"channel {self.name!r}: bad value {value!r}")
        t = (min(max(float(value), self.lo), self.hi) - self.lo) / span
        return soft_bump_bins(t * self.n_bins, self.axis())


#: The channels each family reads, in the writer's units.
FAMILY_CHANNELS: Dict[str, Tuple[Channel, ...]] = {
    "continuation": (
        Channel("turn", "radians", -0.5 * math.pi, 0.5 * math.pi, 8),
        Channel("dist", "wavelengths", 0.0, 4.0, 8),
    ),
    "containment": (
        Channel("polar_angle", "radians", 0.0, 2.0 * math.pi, 12),
        Channel("scale_ratio", "log2_ratio", -2.0, 2.0, 8),
    ),
    "symmetry_parallel": (
        Channel("axis_angle", "radians", 0.0, math.pi, 12),
        Channel("dist", "wavelengths", 0.0, 4.0, 8),
    ),
    "occlusion_depth": (
        Channel("depth", "layers", 0.0, 8.0, 8),
        Channel("depth_units", "depth_units", 0.0, 1.0, 8),
    ),
    "surface_boundary": (
        Channel("dist", "wavelengths", 0.0, 3.0, 8),
        Channel("contrast", "fraction", 0.0, 1.0, 8),
    ),
    "figure_ground": (
        Channel("owner_signed", "signed_scalar", -1.0, 1.0, 8),
        Channel("dist", "wavelengths", 0.0, 2.0, 8),
    ),
}

#: Which SLOT of which edge class fills which channel.  A class that does
#: not measure a channel simply does not fill it: the channel stays empty
#: and nothing is fabricated.
EDGE_CHANNELS: Dict[str, Dict[str, str]] = {
    "EDGE_CONTOUR_CONTINUITY": {"turn": "turn", "dist": "dist"},
    "EDGE_ORIENTATION_CONTINUITY": {"turn": "turn", "dist": "dist"},
    "EDGE_CURVATURE": {"turn": "signed_turn", "dist": "chord_len"},
    "EDGE_CURVATURE_SIGN_CONTINUITY": {"dist": "dist"},
    "EDGE_CURVE_CONTAINMENT": {"polar_angle": "polar_angle",
                               "scale_ratio": "scale_ratio"},
    "EDGE_PART_CONTAINMENT": {"polar_angle": "polar_angle",
                               "scale_ratio": "scale_ratio"},
    "EDGE_SYMMETRY": {"axis_angle": "axis_angle", "dist": "dist"},
    "EDGE_PARALLEL": {"dist": "perp_dist"},
    "EDGE_ORIENTATION_BEND": {"axis_angle": "bisector_theta",
                              "dist": "spatial_dist"},
    "EDGE_ORIENTATION_JUNCTION": {"axis_angle": "bisector_theta",
                                  "dist": "spatial_dist"},
    "EDGE_CONCENTRICITY": {"dist": "offset"},
    "EDGE_COCIRCULAR": {"dist": "dist"},
    "EDGE_OCCLUSION": {"depth": "depth", "depth_units": "depth_units"},
    "EDGE_BINOCULAR_DISPARITY": {"depth_units": "disparity"},
    "EDGE_AMODAL_LINK": {"dist": "gap"},
    "EDGE_SHARED_BOUNDARY": {"dist": "dist"},
    "EDGE_TEXTURE_BOUNDARY": {"contrast": "tex_diff"},
    "EDGE_SURFACE_GROUPING": {"dist": "spatial_dist", "contrast": "tex_dist"},
    "EDGE_COLOR_OPPONENT_BORDER": {"contrast": "contrast"},
    "EDGE_BORDER_OWNERSHIP": {"owner_signed": "owner_signed",
                              "dist": "dist"},
    "EDGE_LINE_TERMINATION": {"dist": "curvature"},
    "EDGE_CONTACT": {"dist": "dist"},
    "EDGE_CO_LOCATED": {},
}


# ---------------------------------------------------------------------------
# THE CONTEXT
# ---------------------------------------------------------------------------
@dataclass
class RelationalContext:
    """A relational context: per family, a graded mixture and a mass.

    mixtures  one tuple of activity vectors per family, in the order of
              FAMILY_CHANNELS[family] (an empty tuple when the family has
              no graded evidence)
    masses    the family's share of the relational evidence, in the order
              of RELATION_FAMILIES
    """
    mixtures: Dict[str, Tuple[Tuple[float, ...], ...]] = field(
        default_factory=dict)
    masses: Dict[str, float] = field(default_factory=dict)

    def families(self) -> Tuple[str, ...]:
        return tuple(f for f in RELATION_FAMILIES if self.masses.get(f, 0.0) > 0.0)

    def is_empty(self) -> bool:
        return not any(self.masses.get(f, 0.0) > 0.0 for f in RELATION_FAMILIES)


def _empty_context() -> RelationalContext:
    return RelationalContext({}, {f: 0.0 for f in RELATION_FAMILIES})


def edge_channels(edge_class: str,
                  quantized: Sequence[float]
                  ) -> List[Tuple[str, float]]:
    """The (channel, value) pairs one edge contributes.

    The values are DE-BINNED through the edge unit table, so the channel
    carries the measurement (radians, wavelengths, a signed scalar) and not
    the lookup index.  An edge whose class has no channel map, or whose
    slot is UNKNOWN, contributes nothing.
    """
    out: List[Tuple[str, float]] = []
    for ch_name, slot_name in EDGE_CHANNELS.get(edge_class, {}).items():
        idx = _slot_index(edge_class, slot_name)
        if idx is None or idx >= len(quantized):
            continue
        b = float(quantized[idx])
        if b == UNKNOWN or not math.isfinite(b):
            continue
        try:
            v = edge_slot_value(edge_class, slot_name, b)
        except KeyError:
            continue
        if not math.isfinite(v):
            continue
        out.append((ch_name, v))
    return out


def _slot_index(edge_class: str, slot_name: str) -> Optional[int]:
    from core.schema import edge_slot_specs
    for i, spec in enumerate(edge_slot_specs(edge_class)):
        if spec.name == slot_name:
            return i
    return None


def _add_edge(acc: Dict[str, Dict[str, List[float]]],
              masses: Dict[str, float],
              edge_class: str,
              quantized: Sequence[float],
              confidence: float,
              ) -> None:
    """Accumulate one edge into the accumulator."""
    family = RELATION_FAMILY.get(edge_class)
    if family is None:
        return
    # The edge's mass: its CONFIDENCE times the MATCHING-TIME specificity
    # of its class (a generic relation carries less of the context).
    mass = max(0.0, float(confidence)) * float(edge_match_weight(edge_class))
    if mass <= 0.0:
        return
    masses[family] = masses.get(family, 0.0) + mass
    for ch_name, value in edge_channels(edge_class, quantized):
        ch = _channel(family, ch_name)
        if ch is None:
            continue
        bucket = acc.setdefault(family, {}).setdefault(ch_name, [])
        vec = ch.activity(value)
        if len(bucket) == 0:
            bucket.extend([0.0] * len(vec))
        for i, v in enumerate(vec):
            bucket[i] += mass * v


def _channel(family: str, name: str) -> Optional[Channel]:
    for ch in FAMILY_CHANNELS.get(family, ()):
        if ch.name == name:
            return ch
    return None


def build_context(edges: Iterable[Tuple[str, Sequence[float], float]]
                   ) -> RelationalContext:
    """The relational context of a set of edges.

    `edges` is an iterable of (edge_class, quantized, confidence).  The
    families' mixtures are the L1-normalised weighted sum of the incident
    edges' channel activity; the families' masses are normalised to sum to
    one over the families that carry evidence.
    """
    acc: Dict[str, Dict[str, List[float]]] = {}
    masses: Dict[str, float] = {}
    for edge_class, quantized, confidence in edges:
        _add_edge(acc, masses, edge_class, quantized, confidence)
    mixtures: Dict[str, Tuple[Tuple[float, ...], ...]] = {}
    for family, chans in acc.items():
        fam_mix: List[Tuple[float, ...]] = []
        for ch in FAMILY_CHANNELS.get(family, ()):
            vec = chans.get(ch.name)
            if not vec:
                fam_mix.append(())
                continue
            total = sum(vec)
            if total <= 0.0:
                fam_mix.append(())
                continue
            fam_mix.append(tuple(v / total for v in vec))
        mixtures[family] = tuple(fam_mix)
    total_mass = sum(masses.values())
    norm = {f: (masses.get(f, 0.0) / total_mass if total_mass > 0.0 else 0.0)
            for f in RELATION_FAMILIES}
    return RelationalContext(mixtures, norm)


# ---------------------------------------------------------------------------
# THE COMPARISON
# ---------------------------------------------------------------------------
def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """The chance-corrected overlap of two activity vectors."""
    from core.population import axis_similarity
    if not a or not b or len(a) != len(b):
        return 0.0
    return axis_similarity(a, b)


def family_overlap(a: RelationalContext, b: RelationalContext) -> Optional[float]:
    """The overlap of two contexts' FAMILY MIXTURES.

    The mixture over the families is a distribution, so the comparison is
    the chance-corrected overlap of the two distributions -- the same
    reading the pair-histogram group uses.
    """
    va = [a.masses.get(f, 0.0) for f in RELATION_FAMILIES]
    vb = [b.masses.get(f, 0.0) for f in RELATION_FAMILIES]
    if sum(va) <= 0.0 or sum(vb) <= 0.0:
        return None
    return _overlap(tuple(va), tuple(vb))


def channel_overlap(a: RelationalContext, b: RelationalContext
                    ) -> Optional[float]:
    """The overlap of two contexts' per-family CHANNEL mixtures.

    Only the families that carry graded evidence on BOTH sides count; a
    family with no graded evidence is not a mismatch, it is an absence.
    """
    vals: List[float] = []
    for family in RELATION_FAMILIES:
        ma = a.mixtures.get(family)
        mb = b.mixtures.get(family)
        if not ma or not mb:
            continue
        ch_vals: List[float] = []
        for va, vb in zip(ma, mb):
            if not va or not vb:
                continue
            ch_vals.append(_overlap(va, vb))
        if ch_vals:
            vals.append(sum(ch_vals) / len(ch_vals))
    if not vals:
        return None
    return sum(vals) / len(vals)


def context_similarity(a: Optional[RelationalContext],
                       b: Optional[RelationalContext]
                       ) -> Optional[float]:
    """The relational similarity of two contexts, or None when either side
    carries no relational evidence at all.

    The family mixture (the coarse structure) and the channel mixtures (the
    parameter detail) each contribute half.
    """
    if a is None or b is None or a.is_empty() or b.is_empty():
        return None
    fo = family_overlap(a, b)
    co = channel_overlap(a, b)
    if fo is None and co is None:
        return None
    fo = 0.0 if fo is None else fo
    co = 0.0 if co is None else co
    return max(0.0, min(1.0, 0.5 * fo + 0.5 * co))


def relational_context_for_view(
    cg: Any,
    chains: Sequence[Any],
    anchor_positions: Optional[Dict[Any, Any]] = None,
    extraction_scale: float = 1.0,
) -> Tuple[Dict[Any, RelationalContext], RelationalContext]:
    """The relational context of a view, read off the CONCEPT GRAPH.

    This is the API the live pipeline needs: it takes the graph and the
    view's chains and returns

      * a mapping from each chain's ANCHOR to that part's relational
        context (the key the descriptor builder expects), and
      * the VIEW-LEVEL relational context (the same mixture over the whole
        graph).

    The relational HOST.  The descriptor's parts are the field's
    PEAK_CHAINs, and the graph's GRADED relations live on the V2 tier (the
    Contour nodes); the two tiers describe the same field structure at two
    resolutions and are NOT connected in the graph (a chain's only
    incident edge is the containment of its member peaks, a 0/1 flag).  So
    the part's relations are read from the node that describes the SAME
    structure: the Contour node nearest the part's own centroid.  The
    join is deterministic and label-free (positions only), and a part with
    no Contour host keeps its own (containment-only) context.
    """
    from core.hashing import _node_signature_hash
    from core.schema import EdgeParams, EdgeParamsMixture

    inc: Dict[int, List[Tuple[str, List[float], float]]] = {}
    all_edges: List[Tuple[str, List[float], float]] = []
    for (src, dst, _et), edge in getattr(cg, "edges", {}).items():
        params = getattr(edge, "params", None)
        if params is None:
            continue
        if isinstance(params, EdgeParamsMixture):
            params = params.dominant
        if not isinstance(params, EdgeParams):
            continue
        triple = (params.class_, [float(v) for v in params.quantized],
                  float(params.confidence))
        inc.setdefault(src, []).append(triple)
        inc.setdefault(dst, []).append(triple)
        all_edges.append(triple)

    tol = 2.0 * max(float(extraction_scale or 1.0), 1.0)
    contour_nodes: List[Tuple[int, float, float]] = []
    seen = set()
    for (src, dst, _et) in getattr(cg, "edges", {}).keys():
        for nid in (src, dst):
            if nid in seen:
                continue
            seen.add(nid)
            try:
                if cg.get_node_lens_type(nid) != "Contour":
                    continue
                pos = cg.pattern_node_position(nid)
            except Exception:
                continue
            if pos is None:
                continue
            contour_nodes.append((nid, float(pos[0]), float(pos[1])))

    out: Dict[Any, RelationalContext] = {}
    for det in chains:
        try:
            nid = cg.get_or_create_pattern_node(
                "PEAK_CHAIN", _node_signature_hash(det.sigma))
        except Exception:
            continue
        host = nid
        pts: List[Tuple[float, float]] = []
        if anchor_positions:
            for a in (getattr(det, "rho", ()) or ()):
                if len(a) > 1 and a[0] == "member":
                    p = anchor_positions.get(a[1])
                    if p is not None and len(p) >= 2:
                        pts.append((float(p[0]), float(p[1])))
        if pts and contour_nodes:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            best: Optional[Tuple[float, int]] = None
            for cnid, cxx, cyy in contour_nodes:
                d = math.hypot(cxx - cx, cyy - cy)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, cnid)
            if best is not None:
                host = best[1]
        out[getattr(det, "alpha", None)] = build_context(inc.get(host, ()))
    return out, build_context(all_edges)


def shuffle_contexts(contexts: Sequence[RelationalContext],
                     seed: int = 20261020) -> List[RelationalContext]:
    """The STRUCTURAL null: the contexts are PERMUTED among their owners.

    This is the null the directive asks for: the RELATIONSHIP STRUCTURE is
    shuffled, not the labels.  A descriptor whose relational context has
    been permuted keeps its shape, its position and its surface, and loses
    only the relational web -- so the arm measures what the relations add
    and nothing else.  The permutation is deterministic (a fixed seed), so
    the null is reproducible.
    """
    import random

    n = len(contexts)
    if n < 2:
        return list(contexts)
    order = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(order)
    out: List[RelationalContext] = []
    for i in order:
        out.append(contexts[i])
    return out
