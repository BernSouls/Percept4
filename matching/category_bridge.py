"""matching/category_bridge.py -- legacy -> U30 projection (Ruling 1).

A legacy core stores bins in the EDGE_SCHEMAS layout.  A v2 edge stores bins
in the U30 layout of the SAME name but a DIFFERENT arity and different slot
units.  Matching therefore cannot compare the two bin tuples directly: the
legacy edge is DE-BINNED to the physical values its bins name
(`edge_slot_value`) and re-quantised through `u30_edge_params`, which puts it
on the category's axes.

Provenance travels with the projected params (`provenance="legacy"`).  When a
legacy projection and a native v2 edge describe the same `(src, dst,
category)`, the NATIVE edge wins and the projection is dropped, so a
half-migrated graph cannot double-count an edge.

Deliberately NOT wired into `_mcs_accumulate`: the live matcher still uses the
legacy layout.  `project_edges` is the cutover seam; see
`diagnostics/_matcher_e2e_test.py` for the legacy-core vs v2-query gate.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.schema import (
    EDGE_SCHEMAS,
    U30_EDGE_UNIT_TABLE,
    UNKNOWN,
    EdgeParams,
    edge_params,
    edge_slot_value,
    u30_edge_params,
)

#: Every legacy class has an EXPLICIT disposition: either it projects onto a
#: U30 category here, or it is listed in NO_U30_EQUIVALENT with a reason.  A
#: class in neither set is a bug, and `_selftest` refuses it, so a newly added
#: legacy class cannot silently fall through to 0.0.
LEGACY_TO_CATEGORY: Dict[str, str] = {
    # continuity family
    "EDGE_CONTOUR_CONTINUITY": "CONTINUITY",
    "EDGE_ORIENTATION_CONTINUITY": "CONTINUITY",
    "EDGE_ORIENTATION_BEND": "CONTINUITY",
    "EDGE_CURVATURE_SIGN_CONTINUITY": "CONTINUITY",
    "EDGE_COCIRCULAR": "CONTINUITY",
    "EDGE_AMODAL_LINK": "CONTINUITY",
    "EDGE_SHARED_BOUNDARY": "CONTINUITY",
    # containment family
    "EDGE_CURVE_CONTAINMENT": "CONTAINMENT",
    "EDGE_PART_CONTAINMENT": "CONTAINMENT",
    "EDGE_REGION_CONTENT": "CONTAINMENT",
    "EDGE_BORDER_OWNERSHIP": "CONTAINMENT",
    # junction family
    "EDGE_JUNCTION_CONTOUR": "JUNCTION",
    "EDGE_ORIENTATION_JUNCTION": "JUNCTION",
    # proximity family
    "EDGE_CO_LOCATED": "PROXIMITY",
    # occlusion / depth-cue family
    "EDGE_OCCLUSION": "OCCLUSION",
    # [R-D1-alignment] CONTACT is a pure spatial-proximity class (failed-owner,
    # front-line, valley between mated edges etc.), never a depth order. The
    # U30 ruling (core/types.py EdgeType.CONTACT -> EdgeCategory.PROXIMITY)
    # puts it in PROXIMITY; the legacy bridge previously sent it to OCCLUSION
    # and the matcher's denominator learned to read 268 flat proximity edges
    # as structural occlusion evidence.
    "EDGE_CONTACT": "PROXIMITY",
    "EDGE_BINOCULAR_DISPARITY": "OCCLUSION",
    "EDGE_LINE_TERMINATION": "OCCLUSION",
    # surface / material family
    "EDGE_SURFACE_FOLD": "SURFACE",
    "EDGE_MOTION_OPPOSITION": "SURFACE",
    "EDGE_SURFACE_GROUPING": "MATERIAL",
    "EDGE_TEXTURE_BOUNDARY": "MATERIAL",
    "EDGE_COLOR_OPPONENT_BORDER": "MATERIAL",
    # shape family
    "EDGE_SYMMETRY": "SHAPE",
    "EDGE_CURVATURE": "SHAPE",
    "EDGE_CONCENTRICITY": "SHAPE",
    "EDGE_RADIALITY": "SHAPE",
    "EDGE_SPIRALITY": "SHAPE",
    "EDGE_CROSS_SCALE_IDENTITY": "SHAPE",
}

#: Classes with NO U30 equivalent, and why.  They are still bridged in the
#: sense that the decision is explicit: a cross-ontology comparison against
#: one of these is a genuine contradiction, not an accident.
NO_U30_EQUIVALENT: Dict[str, str] = {
    "EDGE_PARALLEL": "no shared axis with PROXIMITY (perp/stagger/convergence)",
    "EDGE_CO_MOTION": "no U30 motion axis",
    "EDGE_CONTOUR_ORIENTATION": "single orientation scalar; no CONTAINMENT axis",
    "EDGE_JUNCTION_ORIENTATION": "single orientation scalar; no JUNCTION axis",
}

#: Only slots whose NAMES differ need an entry; everything else projects by
#: the identical name.  A name absent from both sides is passed as None, which
#: `u30_edge_params` stores as UNKNOWN ("no measurement"), never as a value.
_SLOT_ALIASES: Dict[str, Dict[str, str]] = {
    "EDGE_CONTOUR_CONTINUITY": {
        "gap": "dist", "turn_angle": "turn", "curv_agree": "curv_agree"},
    "EDGE_ORIENTATION_CONTINUITY": {
        "gap": "dist", "turn_angle": "turn", "perp_offset": "perp"},
    "EDGE_CURVE_CONTAINMENT": {},
    "EDGE_JUNCTION_CONTOUR": {"topo_class": "topology_class"},
    "EDGE_CO_LOCATED": {"scale_ratio": "scale_ratio"},
    # [R-D1-alignment] CONTACT lands in PROXIMITY: `dist` (the separation of
    #  the two entities, median-spacings) -> `distance`, and `normal_angle`
    #  (the contact normal, axial pi) -> `bearing`.  The legacy `depth_step`
    #  slot is an ordinal depth-difference rank, not a log2 size ratio, so it
    #  has no honest slot on these axes and stays unmapped (UNKNOWN in the
    #  projected params).
    "EDGE_CONTACT": {"distance": "dist", "bearing": "normal_angle"},
    "EDGE_ORIENTATION_BEND": {"turn_angle": "angle_delta"},
    "EDGE_BORDER_OWNERSHIP": {"ownership_strength": "owner_signed"},
    "EDGE_SURFACE_GROUPING": {"chrom_diff": "color_dist",
                              "tex_diff": "tex_dist",
                              "lum_diff": "contrast_step"},
}


def _with_provenance(ep: EdgeParams, label: str,
                     like: Optional[EdgeParams] = None) -> EdgeParams:
    """Copy the evidence fields (sign, confidence) from the original edge.

    A projection keeps the physics: an occlusion edge with sign=-1 must stay
    gating after it lands on the U30 axes.
    """
    src = like or ep
    return EdgeParams(
        class_=ep.class_, quantized=ep.quantized, sign=src.sign,
        confidence=src.confidence, identity_slots=ep.identity_slots,
        population=ep.population, ontology=ep.ontology, provenance=label,
        # The producer's class is preserved across the U30 projection.  Without
        # it a proximity flood and true occlusion evidence would be
        # indistinguishable after both land on their U30 category, and the
        # matcher could not keep a proximity flood from non-ignorable
        # structural relations.
        origin=(getattr(src, "origin", "") or getattr(src, "class_", "")))


def _slot_name(slot: Any) -> str:
    """The slot's tag, whichever attribute the Slot dataclass exposes."""
    for attr in ("name", "tag", "label"):
        value = getattr(slot, attr, None)
        if isinstance(value, str):
            return value
    return ""


def project_edge(ep: Optional[EdgeParams]) -> Optional[EdgeParams]:
    """One legacy `EdgeParams` -> its U30 category projection, or None."""
    if ep is None:
        return None
    cat = LEGACY_TO_CATEGORY.get(ep.class_)
    if cat is None:
        return None
    target = "EDGE_" + cat
    specs = U30_EDGE_UNIT_TABLE.get(target)
    legacy = EDGE_SCHEMAS.get(ep.class_)
    if not specs or not legacy:
        return None
    alias = _SLOT_ALIASES.get(ep.class_, {})
    by_name = {_slot_name(s): i for i, s in enumerate(legacy)}
    values: Dict[str, Optional[float]] = {}
    for slot in specs:
        src = alias.get(slot.name, slot.name)
        idx = by_name.get(src)
        if idx is None or idx >= len(ep.quantized):
            values[slot.name] = None
            continue
        b = ep.quantized[idx]
        if b == UNKNOWN or not math.isfinite(float(b)):
            values[slot.name] = None
        else:
            values[slot.name] = edge_slot_value(ep.class_, src, float(b))
    return _with_provenance(u30_edge_params(target, values), "legacy", like=ep)


def _category_name(key: Any) -> str:
    name = getattr(key, "name", None) or str(key)
    return name if name.startswith("EDGE_") else "EDGE_" + name


def merge_projected(
    params: Dict[Any, EdgeParams],
) -> Tuple[Dict[Any, EdgeParams], Dict[str, int]]:
    """Project legacy edges, then let a native edge win a key collision.

    `params` keys are opaque edge keys; the returned dict is keyed by
    `(src, dst, category_name)` because the category IS the identity of a v2
    edge.  Native params are those already on the U30 ontology.
    """
    merged: Dict[Any, EdgeParams] = {}
    stats = Counter()
    for key, ep in params.items():
        if len(key) < 3:
            continue
        src, dst, et = key[0], key[1], key[2]
        native = getattr(ep, "ontology", "") == "U30"
        cat = (_category_name(et) if native
               else ("EDGE_" + LEGACY_TO_CATEGORY.get(ep.class_, "") or None))
        if not cat or cat == "EDGE_":
            stats["unmapped"] += 1
            continue
        if not native:
            ep = project_edge(ep)
            if ep is None:
                stats["unmapped"] += 1
                continue
        mk = (src, dst, cat)
        if mk in merged and merged[mk].provenance != "legacy":
            stats["native_wins"] += 1
            continue
        if mk in merged:
            stats["legacy_replaced"] += 1
        merged[mk] = ep
        stats["native" if native else "legacy"] += 1
    return merged, dict(stats)


def bridge_similarity(a: EdgeParams, b: EdgeParams,
                      use_confidence: bool = True) -> float:
    """Compare across ontologies by projecting the legacy side(s) to U30.

    Called from `EdgeParams.similarity` ONLY when the class names differ.  A
    pair that maps to different categories, or has no mapping at all, is a
    genuine contradiction (0.0); a pair that lands on the same U30 category is
    compared on that category's axes with the graded metric.
    """
    pa = a if getattr(a, "ontology", "") == "U30" else project_edge(a)
    pb = b if getattr(b, "ontology", "") == "U30" else project_edge(b)
    if pa is None or pb is None or pa.class_ != pb.class_:
        return 0.0
    if len(pa.quantized) != len(pb.quantized):
        return 0.0
    return float(pa.similarity(pb, use_confidence=use_confidence))


def category_counts(keys: Iterable[Any]) -> Counter:
    return Counter(_category_name(k if not isinstance(k, tuple) else k[2])
                   for k in keys)


def category_count_can_contain(q_keys: Iterable[Any],
                               c_keys: Iterable[Any]) -> bool:
    """Naive upper bound: the core must have at least as many edges of every
    category as the query.  Necessary (not sufficient) for full containment.
    """
    qc = category_counts(q_keys)
    cc = category_counts(c_keys)
    return all(n <= cc.get(cat, 0) for cat, n in qc.items())


def _selftest() -> None:
    # Gate: every legacy class has an explicit disposition.
    missing = set(EDGE_SCHEMAS) - set(LEGACY_TO_CATEGORY) - set(NO_U30_EQUIVALENT)
    decided = set(LEGACY_TO_CATEGORY) & set(NO_U30_EQUIVALENT)
    assert not missing and not decided, (sorted(missing), sorted(decided))
    for cat in LEGACY_TO_CATEGORY.values():
        assert ("EDGE_" + cat) in U30_EDGE_UNIT_TABLE, cat
    legacy = edge_params("EDGE_CONTOUR_CONTINUITY",
                         {"dist": 2.0, "turn": 3.0, "curv_agree": 1.0})
    proj = project_edge(legacy)
    assert proj is not None and proj.class_ == "EDGE_CONTINUITY"
    assert proj.ontology == "U30" and proj.provenance == "legacy"
    assert len(proj.quantized) == 4 and proj.population
    native = u30_edge_params("EDGE_CONTINUITY",
                             {"gap": 2.0, "turn_angle": 3.0,
                              "perp_offset": 1.0, "curv_agree": 1.0})
    merged, stats = merge_projected({
        ("a", "b", "EDGE_CONTOUR_CONTINUITY"): legacy,
        ("a", "b", "EDGE_CONTINUITY"): native,
    })
    assert len(merged) == 1, stats
    assert stats.get("native_wins", 0) + stats.get("legacy_replaced", 0) == 1, stats
    only = next(iter(merged.values()))
    assert only.provenance != "legacy" and only.class_ == "EDGE_CONTINUITY"
    # A legacy-only core survives on its own.
    merged2, _ = merge_projected({("a", "b", "EDGE_CONTOUR_CONTINUITY"): legacy})
    assert next(iter(merged2.values())).provenance == "legacy"
    # Cutover: the legacy edge and the v2 edge of the derived category meet
    # through `similarity` even though their class names differ.
    assert 0.0 < legacy.similarity(native) <= 1.0, legacy.similarity(native)
    other_cat = u30_edge_params("EDGE_JUNCTION",
                                {"angle_delta": None, "topo_class": None,
                                 "arm_count": None})
    assert legacy.similarity(other_cat) == 0.0
    # Count bound: 5 query JUNCTIONs cannot fit a core with 3.
    assert category_count_can_contain(
        ["EDGE_JUNCTION"] * 3, ["EDGE_JUNCTION"] * 3 + ["EDGE_SURFACE"])
    assert not category_count_can_contain(
        ["EDGE_JUNCTION"] * 5, ["EDGE_JUNCTION"] * 3)
    print("category_bridge self-test: OK")


if __name__ == "__main__":
    _selftest()
