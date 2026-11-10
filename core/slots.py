"""core/slots.py -- the ONE registry-driven accessor for signature slots.

WHY THIS EXISTS.  Every signature class' slots used to be addressed by
POSITION: `sig.quantized[3]`.  A reader could not say WHAT it was reading,
the writer could not say what it was writing, and a silent slot reorder
inside an emitter changed the meaning of every stored memory with no
error anywhere.  The codebase already had the cure for the analogous
problem in the EDGE domain (core/schema.py's edge unit table) and in the
POPULATION_AXES layout (which names the graded axes of 22 classes), but
the node-side bins had no name table at all.

THE CONTRACT.

  * A class' slot names live in SLOT_NAMES, keyed by the class' `class_`
    string, in SLOT ORDER.
  * A reader asks for a NAME.  `slot_value(sig, "COLOR", "hue")`.
  * A writer asks for a NAME too, and the emitter registry test
    (tests/core/test_named_slots_contract.py) fails the build if a
    registered class is read by a raw integer index anywhere outside this
    module.

THE TABLE is assembled from three sources, in priority order:

  1. POPULATION_AXES -- the graded layout, which for most classes names
     every slot (22 classes).
  2. The explicit tables below, for the classes whose graded layout is a
     strict SUBSET of the bins (PEAK_CHAIN) or which have no graded
     layout at all (BOUNDARY, COLOR_LEGACY, MOTION_POPULATION, ...).
  3. The rich classes' unit tables (core.registries.b1_slot_layout /
     b4_slot_layout), which the rich emitters also derive their spans
     from -- so a rich slot's name, span and period all come from one
     place.

NOTHING here invents a value.  The accessor only renames positions; the
measurement, the quantiser and the writer are unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import math

from core.registries import (
    CIRCULAR_CYCLE_LENGTHS,
    CONSTITUTIONAL_RANGES,
    GABOR_NSF,
    GABOR_NT,
    POPULATION_AXES,
    b1_slot_layout,
    b4_slot_layout,
)

# ---------------------------------------------------------------------------
# THE EXPLICIT TABLES.
#
# Only for classes whose slots are NOT already named by POPULATION_AXES, or
# whose graded layout names a strict subset of the bins.  The names are the
# writer's own semantics, read off the emitter that fills the slot.
# ---------------------------------------------------------------------------

_EXPLICIT: Dict[str, Tuple[str, ...]] = {
    # --- V1 lens classes ---
    "BOUNDARY": ("circularity", "complexity"),
    "BOUNDARY_ANCHORED": ("circularity", "complexity", "anchor_class"),
    "BOUNDARY_CONTENT": ("circularity", "complexity"),
    "COLOR_LEGACY": ("r", "g", "b"),
    "COLOR_LEGACY_CONTENT": ("hue", "sat"),
    "COLOR_CONTENT": ("hue", "sat"),
    "TEXTURE": ("roughness", "isotropy"),
    "SIZE": ("radius", "aspect"),
    "POSITION": ("dx", "dy", "dz"),
    "RELATIVE_POSITION": ("dx", "dy"),
    "ONSET": ("recency",),
    "ORIENTATION": ("theta", "elongation", "anchor_class"),
    "ORIENTATION_CONTENT": ("theta", "elongation"),
    "ORIENTATION_SEGMENT": ("theta", "arc_len", "chain_id", "chain_pos"),
    "ORIENTATION_GABOR": (
        "theta_idx", "sf_idx", "phase", "contrast", "elongation",
        "is_complex", "asymmetry",
    ),
    "ORIENTATION_GABOR_CONTENT": ("theta_idx", "sf_idx"),
    "MOTION": ("vx", "vy", "vz", "speed"),
    "MOTION_POPULATION": (
        "dir_energy_0", "dir_energy_1", "dir_energy_2", "dir_energy_3",
        "dir_energy_4", "dir_energy_5", "dir_energy_6", "dir_energy_7",
        "coherence", "speed", "transparency", "dir_primary", "dir_secondary",
        "divergence", "curl", "shear",
    ),
    "MOTION_CONTENT": ("dir_primary", "dir_secondary", "transparency"),
    # --- the singular-point vocabulary ---
    # PEAK_CHAIN's graded layout names only 4 of its 6 bins (the graded
    # part); the two leading bins are the member count and the mean
    # member orientation.
    "PEAK_CHAIN": (
        "n_peaks", "mean_theta", "theta_var", "closure", "curvature", "length",
    ),
    # --- the content projections ---
    "CONTOUR_CONTENT": (
        "length", "closure", "total_turn", "corner_count", "chirality",
    ),
    "ILLUSORY_CONTOUR_CONTENT": ("theta",),
    "REGION_SHAPE_CONTENT": (
        "curv0", "curv1", "curv2", "curv3", "curv4", "curv5", "curv6",
        "curv7", "curv8", "curv9", "curv10", "curv11", "curv12", "curv13",
        "curv14", "curv15", "elong", "sym", "fold",
    ),
    "ENDSTOPPED_CONTENT": ("theta", "sf"),
}


def _build_slot_names() -> Dict[str, Tuple[str, ...]]:
    names: Dict[str, Tuple[str, ...]] = {}

    # 1. the graded layout, where it names every bin.
    for cls, axes in POPULATION_AXES.items():
        arity = len(CONSTITUTIONAL_RANGES.get(cls, ()))
        axis_names = tuple(a.name for a in axes)
        if arity and len(axis_names) == arity:
            names[cls] = axis_names
        elif arity and cls not in _EXPLICIT:
            # A graded layout that names a strict SUBSET of the bins: the
            # names it gives are still true of the slots it covers, but they
            # do not enumerate the class, so the class is left to an
            # explicit table rather than being mis-described.
            pass

    # 2. the explicit tables win.
    names.update(_EXPLICIT)

    # 3. the rich classes: the unit table is the single source for their
    #    names, spans and periods.
    for cls, layout in (
        ("ORIENTATION_RICH", b1_slot_layout(GABOR_NT, GABOR_NSF)),
        ("TEXTURE_RICH", b4_slot_layout(GABOR_NT, GABOR_NSF)),
    ):
        names[cls] = tuple(n for (n, *_rest) in layout)
    # the content projections of the rich classes are slices of them.
    b1_names = names["ORIENTATION_RICH"]
    names["ORIENTATION_RICH_CONTENT"] = b1_names[:60]
    b4_names = names["TEXTURE_RICH"]
    names["TEXTURE_RICH_CONTENT"] = b4_names[-4:]

    # 4. the rich classes' CONTENT projections of the OTHER classes are
    #    slices of the source class' names; name them here so a reader of
    #    the content class can also ask by name.
    names.setdefault("SURFACE_PATCH_CONTENT",
                     names.get("SURFACE_PATCH", ())[:6])
    names.setdefault("CURVE_COMPLEX_CONTENT",
                     names.get("CURVE_COMPLEX", ())[2:])
    names.setdefault("TEXTURE_BOUNDARY_CONTENT",
                     names.get("TEXTURE_BOUNDARY", ())[1:])
    names.setdefault("JUNCTION_CONTENT", names.get("JUNCTION", ())[:12])
    return names


SLOT_NAMES: Dict[str, Tuple[str, ...]] = _build_slot_names()


class SlotNameError(KeyError):
    """Raised when a slot name is not part of a class' registered layout."""


def slot_names(class_name: str) -> Tuple[str, ...]:
    """The slot names of a class, in slot order.  Empty when the class has
    no registered name table (the caller must then use positional access,
    which the static contract test flags)."""
    names = SLOT_NAMES.get(class_name)
    if names is not None:
        return names
    # The two RICH classes are shape-dependent (ORIENTATION_RICH_12_5),
    # but their layout is fixed by the layout function, so resolve the
    # base name and rebuild it for the class' own bank shape.
    base = _rich_base_name(class_name)
    if base is not None:
        nt, nsf = _rich_shape(class_name)
        layout = (b1_slot_layout(nt, nsf) if base == "ORIENTATION_RICH"
                  else b4_slot_layout(nt, nsf))
        return tuple(n for (n, *_rest) in layout)
    return ()


def _rich_base_name(class_name: str) -> Optional[str]:
    for base in ("ORIENTATION_RICH", "TEXTURE_RICH"):
        if class_name == base or class_name.startswith(base + "_"):
            if class_name.endswith("_CONTENT"):
                continue
            return base
    return None


def _rich_shape(class_name: str) -> Tuple[int, int]:
    parts = class_name.rsplit("_", 2)
    return int(parts[-2]), int(parts[-1])



def has_slot_names(class_name: str) -> bool:
    return bool(slot_names(class_name))


def slot_index(class_name: str, name: str) -> int:
    """The SLOT POSITION of a named slot.  Raises SlotNameError when the
    class has no name table or the name is not one of its slots -- a
    misspelled slot name must fail loudly, not read a neighbour."""
    names = slot_names(class_name)
    if not names:
        raise SlotNameError(
            f"class {class_name!r} has no registered slot-name table")
    try:
        return names.index(name)
    except ValueError:
        raise SlotNameError(
            f"class {class_name!r} has no slot named {name!r}; "
            f"its slots are {list(names)}") from None



def slot_value(sig: Any, class_name: str, name: str,
               default: Optional[float] = None) -> float:
    """Read one slot of a signature BY NAME.

    `default` is returned when the signature is too short for the slot
    (a stale arity), which is the only tolerated failure: a short
    signature is a real possibility in a live graph and must not crash a
    reader.  A MISSING NAME is always an error.
    """
    idx = slot_index(class_name, name)
    q = getattr(sig, "quantized", ())
    if idx >= len(q):
        if default is None:
            raise IndexError(
                f"{class_name}.{name} is slot {idx} but the signature has "
                f"{len(q)} slots")
        return float(default)
    return float(q[idx])


def slot_values(sig: Any, class_name: str,
                names: Sequence[str]) -> Tuple[float, ...]:
    """Read several slots of a signature BY NAME, in the order asked."""
    return tuple(slot_value(sig, class_name, n) for n in names)


def slot_slice(sig: Any, class_name: str,
               names: Sequence[str]) -> Tuple[float, ...]:
    """The slots of a signature named `names`, IN THE SIGNATURE'S OWN SLOT
    ORDER (not the order asked for).  Used by the content projections,
    which slice a source class' tail or head."""
    want = {slot_index(class_name, n) for n in names}
    q = getattr(sig, "quantized", ())
    return tuple(float(q[i]) for i in sorted(want) if i < len(q))


# ---------------------------------------------------------------------------
# CLASS-LOCAL SLOT RULES.
#
# A few slots carry a rule that is only meaningful for their own class: the
# value of one slot decides whether another slot's value is a measurement at
# all.  These rules used to live INSIDE the generic similarity kernel, with
# the class name, the slot index and the magic threshold written into it --
# so every class in the system passed through a branch written for one
# other class, and a re-registration of that class silently misfired.
#
# A comparator returns the slot's DISTANCE when the class declares a rule
# for it, or None to let the generic distance stand.
# ---------------------------------------------------------------------------

# The REGION_SHAPE symmetry AXIS is only defined when a symmetry was
# detected: the symmetry SCORE slot below 2 means "no symmetry", and the
# axis bin is then an arbitrary direction rather than a measurement, so it
# must not be read as a disagreement.  The rule is preserved EXACTLY as it
# was written inside the generic kernel -- this is a RELOCATION, not a
# re-specification, so the comparison's values do not change.
_REGION_SHAPE_AXIS_SLOT = 19
_REGION_SHAPE_SYMMETRY_SLOT = 18
_REGION_SHAPE_UNDEFINED_AXIS = 12.0
_REGION_SHAPE_NO_SYMMETRY = 2.0


def _region_shape_slot_distance(index: int, a: float, b: float,
                                sig_a: Any, sig_b: Any) -> Optional[float]:
    if index != _REGION_SHAPE_AXIS_SLOT:
        return None
    if a >= _REGION_SHAPE_UNDEFINED_AXIS or b >= _REGION_SHAPE_UNDEFINED_AXIS:
        return 0.0
    qa = getattr(sig_a, "quantized", ())
    qb = getattr(sig_b, "quantized", ())
    if len(qa) > _REGION_SHAPE_SYMMETRY_SLOT and len(qb) > _REGION_SHAPE_SYMMETRY_SLOT:
        if (qa[_REGION_SHAPE_SYMMETRY_SLOT] < _REGION_SHAPE_NO_SYMMETRY
                or qb[_REGION_SHAPE_SYMMETRY_SLOT] < _REGION_SHAPE_NO_SYMMETRY):
            return 0.0
    return None


_CLASS_SLOT_RULES = {
    "REGION_SHAPE": _region_shape_slot_distance,
}


def class_slot_distance(class_name: str, index: int, a: float, b: float,
                        sig_a: Any, sig_b: Any) -> Optional[float]:
    """The class' OWN rule for one slot's distance, or None for the
    generic distance."""
    rule = _CLASS_SLOT_RULES.get(class_name)
    if rule is None:
        return None
    return rule(index, a, b, sig_a, sig_b)


# ---------------------------------------------------------------------------
# THE ORIENTATION CLASSES.
#
# The Orientation lens emits THREE different classes depending on what the
# observation could offer (a boundary segment, a whole region's major
# axis, or the polarised major axis), and all three carry a THETA in slot
# 0.  A reader that wants "the orientation" must not have to know which
# class it is holding.
# ---------------------------------------------------------------------------

ORIENTATION_CLASSES: Tuple[str, ...] = (
    "ORIENTATION", "ORIENTATION_POL", "ORIENTATION_SEGMENT",
)


def orientation_class(sig: Any) -> str:
    """The orientation class of a signature, defaulting to ORIENTATION."""
    cls = getattr(sig, "class_", "") or ""
    if cls in ORIENTATION_CLASSES:
        return cls
    return "ORIENTATION"


def orientation_theta(sig: Any, default: float = 0.0) -> float:
    """The THETA slot of whichever orientation class a signature is."""
    return slot_value(sig, orientation_class(sig), "theta", default=default)


def orientation_theta_radians(sig: Any, default: float = 0.0) -> float:
    """The AXIAL orientation in RADIANS, for every orientation family.

    The families do not share a convention, and reading one with another's
    constant silently halves or doubles the angle:

      * ORIENTATION / ORIENTATION_POL / ORIENTATION_SEGMENT: a BIN over
        [0, pi) whose declared cycle length is 24, read at the bin CENTRE
        ((q + 0.5) * pi / 24) -- the convention every geometry consumer in
        this tree already uses.
      * ORIENTATION_GABOR: a filter-BANK CHANNEL index, one channel of a
        pi-wide bank of GABOR_NT channels, at the channel CENTRE
        i * pi / GABOR_NT.  base.py records the historical bug of reading
        this index with ORIENTATION_N_THETA (=24) instead of the bank's 12:
        "measured in the read-back as a 90-degree pair scoring 45 degrees".

    This is the reader the ORIENTATION SPINE uses, so both of its sides (the
    binder that lays the continuity edges and the lens that chains them) read
    the same angle from the same signature.
    """
    cls = getattr(sig, "class_", "") or ""
    if cls in ("ORIENTATION_GABOR", "ORIENTATION_GABOR_CONTENT"):
        idx = slot_value(sig, cls, "theta_idx", default=default)
        return float(idx) * math.pi / GABOR_NT
    if cls in ("ORIENTATION_CONTENT",):
        q = slot_value(sig, "ORIENTATION_CONTENT", "theta", default=default)
        return (float(q) + 0.5) * math.pi / 24.0
    if cls in ORIENTATION_CLASSES:
        q = slot_value(sig, cls, "theta", default=default)
        n = CIRCULAR_CYCLE_LENGTHS.get(cls, {}).get(0, 24)
        return (float(q) + 0.5) * math.pi / float(n)
    return float(default)


