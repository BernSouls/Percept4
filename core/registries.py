"""core/registries.py -- lens-level constitutional ranges and cycle lengths,
slot weights, arities and action-name helpers.

MOVE ONLY: statements extracted verbatim from primitives.py by
_reorg_move2_schema.py, preserving their order. primitives.py re-exports from
here. Zero project imports except core.hashing and parameters (a leaf module).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.hashing import clamp, hash_64, hash_64_signed, serialize_bytes
from core.population import PopAxis, validate_population

import parameters as P

ACTION_NULL: str = "null"

ACTION_NOOP: str = "stay"

PROBE_ACTION_PREFIX: str = "probe_"

def is_noop_action(action: Any) -> bool:
    """Constitutional null check (§2.6): action is None or explicit no-op ⊘."""
    return action is None or action == ACTION_NULL or action == ACTION_NOOP

def is_probe_action(action: Any, env: Any = None) -> bool:
    """Check if action is an exploratory sensorimotor probe.
    Prefers env-declared classification; falls back to protocol prefix."""
    if env is not None and hasattr(env, "is_probe_action"):
        try:
            return bool(env.is_probe_action(action))
        except Exception:
            pass
    return isinstance(action, str) and action.startswith(PROBE_ACTION_PREFIX)

def is_teleological_action(action: Any) -> bool:
    """Check if action is a goal reference rather than an executable motor command."""
    return isinstance(action, tuple) and len(action) >= 2 and action[0] == "goal"

def is_action_lens_eligible(action: Any) -> bool:
    """Single eligibility predicate for the first-class Action lens (B2).

    PURE by construction — the lens (inside SensoryFloor, which has no env
    reference) and the pipeline's execution-time onset pre-registration
    MUST reach the identical verdict for the same action, or the pattern's
    onset timing desynchronizes from its emission. Consulting an
    env-declared probe classifier here would reintroduce exactly that
    divergence, so this predicate uses only the constitutional checks:
    real, executable motor/manipulation actions are eligible; no-ops,
    probe-prefix actions, and teleological goal references are not.
    """
    if is_noop_action(action):
        return False
    if is_teleological_action(action):
        return False
    if isinstance(action, str) and action.startswith(PROBE_ACTION_PREFIX):
        return False
    return True

# ---------------------------------------------------------------------------
# [O1] THE ABSENT MARKER.
#
# A slot the writer could NOT MEASURE must be ABSENT, never the slot's
# minimum.  "not measured" and "measured as the minimum" are different
# facts, and conflating them makes an unmeasurable slot assert a
# measurement that was never made: the similarity then treats two
# unrelated regions as maximally different (or, worse, as agreeing at the
# bottom of the axis).
#
# The marker is a FINITE float, because core/hashing.serialize_float
# REJECTS non-finite values (a NaN in a signature would make the node's
# identity unserialisable).  It is far outside every registered span: the
# node classes' slots hold either bin indices (0 .. n_bins-1) or
# continuous measurements inside the unit table's span (at most a few
# radians), so no real value can collide with it.
#
# The reader skips an absent slot on EITHER side: it contributes no
# distance and no weight.  A comparison with NO comparable slot is 0.0 --
# the house convention for a missing group ("absent is not zero").
# ---------------------------------------------------------------------------

ABSENT: float = -1.0e9


def is_absent(value: Any) -> bool:
    """True when a slot carries the ABSENT marker (no measurement)."""
    try:
        return float(value) == ABSENT
    except (TypeError, ValueError):
        return False


CONSTITUTIONAL_RANGES: dict[str, tuple[int, ...]] = {
    # --- Lens classes (§2.2) ---
    # Boundary, collapsed form: Ch 0 = circularity (10 bins), Ch 1 =
    # boundary complexity (10 bins). Emitted by the Boundary lens when the
    # observation exposes NO per-entity shape channels (symbolic envs), where
    # same-shape entities deliberately share one pattern node.
    "BOUNDARY": (10, 10),
    # Boundary, anchored form: [Arm C / ANCHOR-IN-SIGNATURE FIX] Arity-3:
    # (circularity_range, complexity_range, anchor_range). The anchor
    # component is a CATEGORICAL discriminator; its range is 0.0. In the
    # sum-form similarity (total_diff / sum(ranges)), a nonzero anchor
    # difference saturates total_diff → clamps to 0.0 for cross-anchor pairs;
    # same-anchor pairs contribute 0 on this slot, preserving graded shape
    # similarity on slots 0/1 exactly as before. Emitted by the Boundary lens
    # when the observation DOES expose shape channels (Observation2D/3D).
    "BOUNDARY_ANCHORED": (10, 10, 0.0),
    # B.8 Color: Opponent channels (Lum, RG, BY, Sat, Hue).
    "COLOR": (8.0, 16.0, 12.0, 8.0, 8.0),
    "COLOR_LEGACY": (12.0, 12.0, 12.0),
    "COLOR_CONTENT": (8.0, 8.0),
    "COLOR_LEGACY_CONTENT": (8.0, 8.0),
    # Motion: speed (20 bins), 3D direction (8 bins/axis)
    "MOTION": (20, 8, 8, 8),
    # Onset: age-since-onset normalized (20 bins)
    "ONSET": (20,),
    # Polarity: single categorical slot, 3 classes (OFF/balanced/ON).
    # Value is the draft's declared 3.0; no pair of classes can differ
    # by 3.0 (max diff 2), noted for the first POLARITY matching
    # calibration (register section 39).
    "POLARITY": (3.0,),
    # [Option (b)] Polarized orientation: the single-region ORIENTATION
    # slots (theta bin, elongation bin, 4x4 position class) plus the
    # 3-class polarity slot. Ranges mirror ORIENTATION_N_THETA = 24 and
    # ORIENTATION_N_ELONG = 4. NOTE: the adjacent "ORIENTATION" entry
    # still reads (12, 4, 0.0) and the same class also carries arity-6
    # GABOR_RESPONSE signatures -- pre-existing, recorded, not changed.
    "ORIENTATION_POL": (24, 4, 16.0, 3.0),
    # Position: Object-centric relative dx, relative dy, elevation
    # (16 bins per axis — fallback ladder refinement for layout resolution)
    "POSITION": (16, 16, 16),
    # RelativePosition: position relative to macro envelope centroid,
    # normalized by envelope extent. 16 bins per axis (fallback ladder:
    # 10 bins measured too coarse for layout discrimination). View-stable.
    "RELATIVE_POSITION": (16, 16),
    # Size: Ch 0 = radial scale (10 bins), Ch 1 = inertia aspect ratio / roundness (10 bins)
    "SIZE": (10, 10),
    # Texture: Ch 0 = surface roughness (10 bins), Ch 1 = directional isotropy (10 bins)
    "TEXTURE": (10, 10),
    # --- Binder classes (§2.3) ---
    "TOPOLOGY": (4, 20, 20, 20, 20),
    "DEGREE": (10,),
    "EDGE": (10,),
    "CONTACT": (10,),
    "CO_MOTION": (10,),
    "CONTRAST": (10,),
    "PERIODICITY": (20, 20),
    "PERSISTENCE": (10,),
    # --- Structure binder classes ---
    "STRUCTURE": (4, 4, 4, 4, 4, 4, 4, 4),
    # --- Order-2 binder classes (§2.3) ---
    "SEQUENCE": (20, 1, 1, 1, 1, 1, 1, 1, 1),
    # [v8] ORIENTATION signature (arity 3).
    # Slot 0 (angle): circular; range = N_theta / 2 = 12.
    # Slot 1 (elongation): N_ELONG - 1 = 4.
    # Slot 2 (anchor trunk): 0.0 categorical discriminator.
    # NOTE: hard-coded to mirror parameters.ORIENTATION_N_THETA and
    # parameters.ORIENTATION_N_ELONG; primitives.py does not import
    # parameters.py for layering reasons.
    "ORIENTATION": (12, 4, 0.0),
    # [v9] ORIENTATION_SEGMENT signature (arity 4).
    # Slot 0 (theta): circular; range = N_theta / 2 = 12.
    # Slot 1 (arc_len): ORIENTATION_SEGMENT_N_ARC - 1 = 7.
    # Slot 2 (chain_id): 0.0 categorical discriminator.
    # Slot 3 (chain_pos): ORIENTATION_SEGMENT_N_POS - 1 = 7.
    "ORIENTATION_SEGMENT": (12, 7, 0.0, 7),
    # [v32 CONTENT TIER] Position-invariant content classes. Registered
    # here (primitives) rather than parameters.py because parameters.py
    # does not import these dicts — primitives does not import parameters
    # for layering reasons, so the registries must live in primitives.
    # Content signatures are the anchor-stripped forms produced by
    # content.content_signature().
    # BOUNDARY_CONTENT: quantized[:2] of BOUNDARY/BOUNDARY_ANCHORED.
    "BOUNDARY_CONTENT": (10.0, 10.0),
    # ORIENTATION_CONTENT: (theta, e_b). Slot 0 circular; slot 1 elongation.
    "ORIENTATION_CONTENT": (12.0, 5.0),
        # --- [PHASE B] higher-lens classes (futurework3.md B.5–B.10) ---
    # Values mirror parameters.py for layering reasons (primitives does
    # not import parameters): ORIENTATION_N_THETA = 24,
    # COLOR_QUANT_BINS = 12.
    #
    # A CIRCULAR slot's range must be exactly N/2 (see the
    # CIRCULAR_CYCLE_LENGTHS note below): the wrap-around formula
    # min(d, N-d) caps that slot's contribution at N/2, so a larger
    # range would under-penalize opposite angles.
    #
    # [B6 FIX v2 / ROUND 10 C] BorderOwnership: PER-FRAGMENT signed direction.
    # Arity 4: (arc_pos_bin, signed_dir_bin, curvature_bin, strength_bin).
    # Slot 1 is a SIGNED LINEAR scalar, NOT circular: + means the owner (figure)
    # is on the LEFT of the fragment's TRAVEL direction, magnitude is the
    # confidence, near zero is ambiguous. Slot 2 is signed linear too; no slot
    # is circular. One detection is emitted per oriented edge fragment (a chain
    # member), so a boundary is a field of fragments. The population axis for
    # slot 1 is therefore linear (see POPULATION_AXES derivation below), and
    # every reader (perception/ownership.py, anti_phase, binder) decodes it
    # through the ONE signed convention.
    "BORDER_OWNERSHIP": (8.0, 8.0, 8.0, 8.0),
    # [Stage 3] ContourFragment: Pasupathy & Connor object-centred polar
    # fragment. Arity 4: (kappa_bin 0..8, theta_bin 0..23 circular, r_bin 0..7,
    # tangent_vs_radial_bin 0..7). Slot 1 is the polar angle on a 24-cycle, so
    # its range is 24/2 = 12.0 under the N/2 convention (registered in
    # CIRCULAR_CYCLE_LENGTHS below).
    "CONTOUR_FRAGMENT": (9.0, 12.0, 8.0, 8.0),
    # [B5] Contour: arity 27 (7 base slots + 20 curvature sequence).
    #   0 length_bin, 1 mean_theta_bin (circular on 24 -> range 12),
    #   2 circ_var_bin (dispersion), 3 closure_bin,
    #   4 total_abs_turn_bin, 5 corner_count_bin,
    #   6 chirality_bin (net signed turn / 2pi),
    #   7..26 canonical signed curvature sequence (CONTOUR_N_SEQ = 20 bins, range 8.0).
    "CONTOUR": (8.0, 12.0, 8.0, 2.0, 8.0, 8.0, 8.0) + (8.0,) * 20,
    # [Stage A2] CONTOUR_PEAK: a peak of the continuous V1/V2 integration field
    # (perception/contour_field.py), NOT a shock segment. Arity 8:
    #   0 theta_bin              circular, 12 bins over [0, pi)
    #   1 sf_bin                 graded, 5 bins
    #   2 signed_curvature_bin   graded, 8 bins, clipped to +-0.6
    #   3 bend_angle_bin         graded, 8 bins, clipped to 1.5
    #   4 phase_bin              circular, 4 bins
    #   5 end_stop_flag          binary
    #   6 integration_strength_bin graded, 8 bins, clipped to 20
    #   7 coherence_bin          graded, 8 bins, clipped to 0.15
    # Bin edges are fixed in perception/contour_peak.py and documented there.
    # Circular slots obey the N/2 convention: range[slot] == cycle/2, AND the
    # cycle equals the emitter's bin count (contour_peak.py: theta_idx %12,
    # phase_bin %4) so the wrap actually fires. A cycle larger than the bin
    # count never wraps and makes adjacent bins maximally far apart.
    #   slot 0: theta_bin, contour_peak.py theta_idx in [0,11] -> cycle 12, range 6
    #   slot 4: phase_bin, contour_peak.py phase_bin in [0,3] -> cycle 4, range 2
    "CONTOUR_PEAK": (6.0, 4.0, 7.0, 7.0, 2.0, 1.0, 7.0, 7.0),
    # [A5] PEAK_CHAIN, arity 6. A chain of CONTOUR_PEAK detections grouped by
    # perception/chain_formation.form_chains (proximity + axial orientation +
    # end-stop clearance). Slots:
    #   0 n_peaks_bin          graded, 0..15
    #   1 mean_theta_bin       circular over 12 (see N/2 note below)
    #   2 theta_var_bin        graded, 0..7
    #   3 closure_bin          binary
    #   4 curvature_summary_bin graded, 0..7
    #   5 length_bin           graded, 0..7
    # Bin edges are fixed in lenses/chain_lens.py.
    "PEAK_CHAIN": (15.0, 6.0, 7.0, 1.0, 7.0, 7.0),
    "CONTOUR_CONTENT": (8.0, 2.0, 8.0, 8.0, 8.0),
    # B.7+B21 Junction, arity 19:
    #   0-7 as B7 (degree, ref_theta, topo, spread, gap_0..3)
    #   8  arm_support: mean #oriented segments per arm, capped (linear 8)
    #   9  arm_sharp_mean, 10 arm_sharp_min: doubled-angle resultant of the arm's
    #      segment tangents (8 bins). Only meaningful when arm_support > 1.
    #   11-18 pairwise arm-angle counts over (0, pi], 8 bins of 22.5 deg, capped at 7.
    "JUNCTION": (6.0, 12.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                 8.0, 8.0, 8.0,
                 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
    # Content: rotation-invariant. degree, topo, spread, arm_sharp_mean,
    # pair histogram. (arm_sharp_min and ref_theta are dropped: ref_theta is
    # already 0 for symmetric topologies, and min is a duplicate of mean.)
    "JUNCTION_CONTENT": (6.0, 7.0, 8.0, 8.0,
                         8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
    # B12 Surface: (shape, lum, hue, sat, tex, grad_mag, grad_side), arity 7.
    # Slot 2 (hue) is circular on 16 -> range 8.0.
    "SURFACE_PATCH": (8.0,) * 7,
    "SURFACE_PATCH_CONTENT": (8.0,) * 6,
    # [Session B / B3] SURFACE_FRAGMENT: the 2.5D orientation of ONE surface
    # point, estimated from the pictorial cues on a single (slant, tilt) grid.
    # Arity 4: (slant_bin, tilt_bin, curvature_sign, confidence_bin).
    #   slot 0 slant      0 .. pi/2, 18 bins (5 deg), linear
    #   slot 1 tilt       normal azimuth, 24 bins CIRCULAR on 2pi ->
    #                     range 12.0 under the N/2 convention (cycle below)
    #   slot 2 curvature  owner-relative SIGN, 3 levels (-1 / 0 / +1,
    #                     signed_centered), i.e. a signed linear scalar
    #   slot 3 confidence the population's resultant length, 8 bins
    # The class is NOT yet in LENS_DECLARATION_BY_NAME: the emitter ships
    # standalone and the pipeline wiring is a separate contract change.
    "SURFACE_FRAGMENT": (18.0, 12.0, 3.0, 8.0),
    # B.8/B13 CurveComplex: arity 24.
    # 0 n_contours, 1 n_junctions, 2 extent, 3 closure, 4-11 signed spatial bins,
    # 12-23 V4 template population.
    "CURVE_COMPLEX": (8.0,) * 44,
    "CURVE_COMPLEX_CONTENT": (8.0,) * 42,
    # B14+B22 RegionShape: arity 21 (was 18).
    # 0-15 curvature histogram, 16 area, 17 elongation,
    # 18 sym_score (8 bins), 19 sym_axis (circular 12 -> 6.0),
    # 20 sym_fold (k-1, 0..5).
    "REGION_SHAPE": (8.0,) * 18 + (8.0, 6.0, 6.0),
    "REGION_SHAPE_CONTENT": (8.0,) * 18 + (6.0,),
    # B.9/B15 Part: arity 23.
    # 0 n_complexes, 1-8 spatial layout, 9-20 V4 templates (12), 21 hue (circ 16 -> 8.0), 22 sat.
    # B.10/B16 ObjectShape: arity 33 (was 37/17).
    # 0-7 boundary sym, 8-15 part sym, 16 n_parts, 17-24 ring presence, 25-32 ring template.
    # B17 Illusory Contour: V2 modal completion. Arity 3.
    # 0: orientation_bin (circular 24 -> 12.0)
    # 1: length_bin (linear 8, gap distance normalized)
    # 2: strength_bin (linear 8, completion confidence)
    "ILLUSORY_CONTOUR": (12.0, 8.0, 8.0),
    "ILLUSORY_CONTOUR_CONTENT": (12.0,),
    # [D1] End-Stopped (Hypercomplex): line terminations and short bars/dots.
    # (theta_bin, sf_bin, ends_bin, strength_bin).
    "ENDSTOPPED": (12.0, 8.0, 3.0, 8.0),
    "ENDSTOPPED_CONTENT": (12.0, 8.0),
    # [FIGURE-GROUND / singular points] CORNER: a boundary vertex where the
    # tangent turns sharply, derived from the CONTINUOUS FIELD's chain relations
    # (lenses/corner_lens.py reads PEAK_CHAIN member order and the
    # member peak positions), NOT from the dead shock graph. One typed node per
    # local turn-angle extremum. Arity 4:
    #   0 turn_mag_bin    graded, |signed turn| over [0, pi] -> 0..7
    #   1 orient_bin      circular, corner-bisector orientation over [0, pi),
    #                     cycle 12 -> range 6.0 (N/2 convention)
    #   2 prom_bin        graded, local prominence (|turn| - max neighbour
    #                     |turn|) over [0, pi/2] -> 0..7
    #   3 scale_bin       graded, corner arm length / (8 * extraction_scale)
    "CORNER": (8.0, 6.0, 8.0, 8.0),
    "CORNER_CONTENT": (8.0, 6.0, 8.0),
    # [B3a/b] ORIENTATION_GABOR: (theta, sf, phase, contrast, elong, simple_flag, asym)
    # [Pass 17.2] phase slot (slot 2) holds a 0..23 bin (ORIENTATION_PHASE_BINS
    # = 24): range must be N/2 = 12.0 under the circular convention, and slot 2
    # must be listed in CIRCULAR_CYCLE_LENGTHS, or cross-image simple-cell
    # similarity collapses to exp(-400) ~ 0.
    "ORIENTATION_GABOR": (6.0, 4.0, 12.0, 15.0, 4.0, 1.0, 8.0),
    "ORIENTATION_GABOR_CONTENT": (6.0, 4.0),
    # [B11] Motion Population & Content. Arity 16, matching the emitter
    # (lenses/base.py motion emitter). Slots: 0-7 dir_bins (8-way energy
    # fraction, linear 8), 8 coherence, 9 speed, 10 transparency,
    # 11 dir1 / 12 dir2 (circular 8 -> range 4.0), 13 div / 14 curl
    # (signed 8), 15 shear (8). The registry previously carried only the
    # first 13 slots, so every emitted 16-slot motion signature failed the
    # arity length guard and silently scored 0.0.
    "MOTION_POPULATION": (8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                          8.0, 4.0, 4.0, 8.0, 8.0, 8.0),
    "MOTION_CONTENT": (4.0, 4.0, 8.0),
    # [B20] Texture-Defined Boundary
    "TEXTURE_BOUNDARY": (8.0, 12.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
    "TEXTURE_BOUNDARY_CONTENT": (12.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
    # [U30] The 8 canonical physical edge-categories (see _u30_inventory.md).
    "EDGE_PROXIMITY": (6.0, 6.0, 4.0),
    "EDGE_CONTINUITY": (4.0, 6.0, 3.0, 2.0),
    "EDGE_CONTAINMENT": (5.0, 12.0, 4.0, 1.0, 2.0),
    "EDGE_OCCLUSION": (4.0, 4.0, 1.0, 2.0),
    "EDGE_JUNCTION": (math.pi / 2.0, 6.0, 5.0),
    "EDGE_SURFACE": (math.pi, 1.0, math.pi / 2.0, 1.0),
    "EDGE_SHAPE": (1.0, 4.0, 2.0),
    "EDGE_MATERIAL": (1.0, 1.0, 1.0),
    # [aIT Holistic Shape] Arity 6 — global boundary-shape summary emitted by
    # HolisticShapeLens, from the same polyline_shape_descriptor as RegionShape:
    # 0 area_bin (8.0), 1 elong_bin (8.0), 2 sym_score_bin (8.0),
    # 3 sym_axis (circular 12 -> 6.0, 12 = undefined),
    # 4 fold (0..5), 5 curvature-histogram summary bin (8.0).
}

CIRCULAR_CYCLE_LENGTHS: Dict[str, Dict[int, int]] = {
    "ORIENTATION": {0: 24},
    "ORIENTATION_SEGMENT": {0: 24},
    # [v32 CONTENT TIER] theta slot of the anchor-stripped Orientation.
    "ORIENTATION_CONTENT": {0: 24},
    # [PHASE B] futurework3.md B.5–B.7. CONTOUR/JUNCTION cycle is
    # ORIENTATION_N_THETA (24); mirrored here for the same layering
    # reason as the ORIENTATION entry above.
    "CONTOUR": {1: 24},
    # [Stage 3] ContourFragment slot 1 is the polar angle (24-cycle -> range 12).
    "CONTOUR_FRAGMENT": {1: 24},
    # [Stage A2 / A5c] CONTOUR_PEAK: slot 0 runs 0..11 (contour_peak.py
    # theta_idx %12) -> cycle 12, range 6; slot 4 runs 0..3 (phase_bin %4)
    # -> cycle 4, range 2. Cycle must equal the emitted bin count, or the
    # wrap never fires and adjacent bins score maximally far apart.
    "CONTOUR_PEAK": {0: 12, 4: 4},
    # [A5] PEAK_CHAIN: slot 1 is the circular-mean member-theta bin. 12
    # bins over a 24-cycle odometer: the range is 6.0 per the N/2
    # convention, and the emitted bin index is taken mod 12.
    "PEAK_CHAIN": {1: 12},
    "JUNCTION": {1: 24},
    "ENDSTOPPED": {0: 24},
    "ENDSTOPPED_CONTENT": {0: 24},
    # [singular points] CORNER slot 1 is the corner-bisector orientation over a
    # 12-cycle (range 6.0 under the N/2 convention).
    "CORNER": {1: 12},
    "CORNER_CONTENT": {1: 12},
    "ORIENTATION_GABOR": {0: 12, 2: 24},
    "ORIENTATION_GABOR_CONTENT": {0: 12},
    "MOTION_POPULATION": {11: 8, 12: 8},
    "MOTION_CONTENT": {0: 8, 1: 8},
    "TEXTURE_BOUNDARY": {1: 24},
    "TEXTURE_BOUNDARY_CONTENT": {0: 24},
    "COLOR": {4: 16},
    "COLOR_CONTENT": {0: 16},
    "COLOR_LEGACY_CONTENT": {0: 16},
    "SURFACE_PATCH": {2: 16},
    "SURFACE_PATCH_CONTENT": {2: 16},
    # [Session B / B3] SURFACE_FRAGMENT slot 1 (tilt) is a 24-cycle over 2pi:
    # 24 bins, range 12.0 (N/2).  The emitter's graded axis uses the same
    # 24 bins, so the wrap the similarity applies and the bump the emitter
    # writes agree.
    "SURFACE_FRAGMENT": {1: 24},
    "REGION_SHAPE": {19: 12},
    "ILLUSORY_CONTOUR": {0: 24},
    "ILLUSORY_CONTOUR_CONTENT": {0: 24},
    # [U30] The 8 canonical physical edge-categories: bearing (axial over pi),
    # gap-angle (axial) and polar_angle (24 over 2pi) define 12-/12-/24-cycles.
    "EDGE_PROXIMITY": {1: 12},
    "EDGE_CONTINUITY": {1: 12},
    "EDGE_CONTAINMENT": {1: 24},
    # [aIT Holistic Shape] slot 3 sym_axis is a 12-cycle (12 = undefined),
    # same convention as REGION_SHAPE slot 19.
}

#: [ROUND 10 / ITEM F] Rich classes registered by `perception.rich.b1_register`
#: store BOTH their range and their cycle in the slot's OWN units (radians),
#: not integer bin counts: each such slot spans exactly ONE full period, so the
#: convention is `cycle == range`, NOT the integer-bin N/2 convention
#: (`range == cycle/2`) used by the quantized classes. The verifier below
#: branches on this so the rich classes are not falsely flagged.
_RADIAN_PERIOD_CLASS_PREFIXES = ("ORIENTATION_RICH", "TEXTURE_RICH")


def _is_radian_period_class(cls: str) -> bool:
    return any(cls.startswith(p) for p in _RADIAN_PERIOD_CLASS_PREFIXES)


def _verify_circular_ranges(ranges: dict, cycles: dict, enforce_classes: set) -> None:
    """Verifies the N/2 circular range convention.
    Raises RuntimeError for classes in `enforce_classes`.
    Issues a warning for other classes that violate the convention.
    """
    for _cls, _cycles in cycles.items():
        _ranges = ranges.get(_cls)
        if _ranges is None:
            if _cls in enforce_classes:
                raise RuntimeError(
                    f"CIRCULAR_CYCLE_LENGTHS lists {_cls} but it is missing "
                    f"from CONSTITUTIONAL_RANGES"
                )
            continue

        for _slot_idx, _cycle_len in _cycles.items():
            if _slot_idx >= len(_ranges):
                if _cls in enforce_classes:
                    raise RuntimeError(
                        f"CIRCULAR_CYCLE_LENGTHS[{_cls}][{_slot_idx}] out of bounds "
                        f"for arity {len(_ranges)}"
                    )
                continue

            _actual_range = _ranges[_slot_idx]
            if _is_radian_period_class(_cls):
                # Radian-period convention: range spans one full period.
                if _actual_range == _cycle_len:
                    continue
                msg = (
                    f"Radian-period convention violated: {_cls} slot {_slot_idx} "
                    f"has cycle {_cycle_len} but range {_actual_range}. "
                    f"Expected cycle == range (one full period)."
                )
                if _cls in enforce_classes:
                    raise RuntimeError(msg)
                warnings.warn(f"{msg} (Legacy class, not enforced at boot)")
                continue
            _expected_range = _cycle_len / 2.0
            if _actual_range != _expected_range:
                msg = (
                    f"Circular range convention violated: {_cls} slot {_slot_idx} "
                    f"has cycle {_cycle_len} but range {_actual_range}. "
                    f"Expected N/2 = {_expected_range}."
                )
                if _cls in enforce_classes:
                    raise RuntimeError(msg)
                else:
                    warnings.warn(f"{msg} (Legacy class, not enforced at boot)")

_ENFORCE_N_HALF_CONVENTION = {
    "JUNCTION", "COLOR", "COLOR_CONTENT", "COLOR_LEGACY_CONTENT",
    "SURFACE_PATCH", "SURFACE_PATCH_CONTENT", "REGION_SHAPE",
    "SURFACE_FRAGMENT",
    "ILLUSORY_CONTOUR", "ILLUSORY_CONTOUR_CONTENT",
    "TEXTURE_BOUNDARY", "TEXTURE_BOUNDARY_CONTENT",
    # [Pass 17.2] simple-cell phase slot is circular 24 -> range must be 12.
    "ORIENTATION_GABOR", "ORIENTATION_GABOR_CONTENT",
}

_verify_circular_ranges(
    CONSTITUTIONAL_RANGES, CIRCULAR_CYCLE_LENGTHS, _ENFORCE_N_HALF_CONVENTION
)

# --- [POPULATION CODING] tuning-curve axes -------------------------------
# The graded companion of each class's bins. One PopAxis per graded quantity;
# a signature's `population` is a tuple of L1-normalised activity vectors, one
# per axis, in the order listed here. Circular axes use n_bins == the class's
# CIRCULAR_CYCLE_LENGTHS entry for that quantity (checked below). This is the
# "full vector on the detection" the representation thesis requires; the bins
# remain as lookup keys only.
#
# [CONFIDENCE SCALE] The scale the CONTOUR_PEAK confidence is normalised by.
# The field's integration_strength is NOT a [0, 1] quantity: it was measured
# over 2643 peaks from 5 real ETH-80 images in [0.029, 25.446] (p50 17.015)
# (see perception/contour_peak.py). Clamping it to [0, 1] therefore reported
# z = 1.0 for EVERY peak. The confidence is the measured strength normalised
# by the range the field actually produces: the scale is 25.0, the measured
# upper bound rounded down, so a peak at or above the observed maximum
# saturates at z = 1.0 and the median peak (17.0) reports z ~ 0.68.
# Declared here (not in parameters/_text.py) as a local default.
CONTOUR_PEAK_STRENGTH_Z_SCALE: float = 25.0

POPULATION_AXES: Dict[str, Tuple[PopAxis, ...]] = {
    # The CONTOUR_PEAK and PEAK_CHAIN layouts are declared BELOW, next to
    # their comments: the end-stop magnitude and the closure axes are
    # appended to them, and keeping each class' layout in one place is what
    # makes the append visible.
    # PEAK_CHAIN: an ensemble. The orientation axis holds the DISTRIBUTION of
    # member orientations (a break-stable mixture), not the circular mean, so a
    # chain is described by its spread of evidence and not one collapsed number.
    "PEAK_CHAIN": (
        PopAxis("theta_dist", 12, circular=True,
                sigma_bins=float(P.POP_ORIENTATION_SIGMA_BINS)),
        PopAxis("curvature", 8, circular=False),
        PopAxis("length", 8, circular=False),
        # [P2] THE CLOSURE, AS A 0/1 ACTIVITY.  Closure is TOPOLOGICAL, not
        # a gradient: the chain either returns to its start or it does not,
        # and there is no continuous quantity between the two answers (a
        # "slightly open" chain is still open).  The axis therefore carries
        # TWO levels -- closed and open -- and the activity is the bump on
        # the level the chain actually has.  Two closed chains overlap at
        # 1.0, a closed and an open chain at 0.0, which is the whole
        # statement the relation makes.
        #
        # The axis is APPENDED, so the writer side's existing vectors keep
        # their positions: until the writer fills this axis, the comparison
        # SKIPS it (core.population.aligned_population_similarity), so the
        # graded comparison of the class keeps working.
        PopAxis("closure", 2, circular=False),
    ),
    # CONTOUR_PEAK: the field peak. The axes are in the WRITER's order
    # (theta, sf, curvature, bend, phase, strength, coherence), which is
    # NOT the bins' order (the bins carry the end-stop FLAG at slot 5).
    # [P2] THE END-STOP MAGNITUDE.  The field's end-stop array is GRADED
    # (the normalised |forward - backward| of the terminator's channel);
    # the FLAG (bins' slot 5) says only "this terminator is above half its
    # channel's asymmetry", so the flag is the INDEX of the measurement and
    # the magnitude is the measurement itself.  The magnitude therefore gets
    # its OWN axis -- APPENDED, so the writer side's existing vectors keep
    # their positions; until the writer fills it, the comparison skips it.
    "CONTOUR_PEAK": (
        PopAxis("theta", 12, circular=True,
                sigma_bins=float(P.POP_ORIENTATION_SIGMA_BINS)),
        PopAxis("sf", 5, circular=False),
        PopAxis("curvature", 8, circular=False),
        PopAxis("bend", 8, circular=False),
        PopAxis("phase", 4, circular=True),
        PopAxis("strength", 8, circular=False),
        PopAxis("coherence", 8, circular=False),
        PopAxis("endstop", 8, circular=False),
    ),
}


# --- [POPULATION CODING] THE UNIT TABLE -----------------------------------
# ONE table per class. For every slot it declares:
#   name      the stable identifier
#   unit      the UNIT of the CONTINUOUS quantity the emitter measured
#   lo, hi    the SPAN of that quantity (the emitter's own span)
#   n_bins    the BIN COUNT (the emitter's own level count)
#   circular  whether the slot wraps
#   quantiser the bins-path convention (see SlotSpec)
#   keys      for a "lookup_key" slot, the categorical bin values
#
# BOTH readings of a detection are derived from THIS table:
#   * the BINS path   -> `quantise(spec, value)`
#   * the GRADED path -> `population_centre(spec, value)`, the continuous
#     BIN position of the same value (the sub-bin position the bin rounds
#     off), from which the very same quantiser derives the bin.
# Because the bin is DEFINED as the quantiser applied to the population
# centre, the two readings cannot disagree: there is one number, one span
# and one map.  The earlier layouts declared a span the emitter never used
# (JUNCTION's gaps in radians over a [0, 8] span, TEXTURE_BOUNDARY's
# contrast drop under a "log2_energy" name), so the graded bump landed on
# a different bin than the lookup key -- two quantisations of one
# measurement.  The table removes that class of defect by construction.
#
# The bin COUNT belongs to the emitter (never to the constitutional RANGE,
# which is a similarity range, not a level count), and the SPAN belongs to
# the quantity's unit.
@dataclass(frozen=True)
class SlotSpec:
    """One entry of a class' UNIT TABLE.

    name      stable identifier of the slot
    unit      the unit of the continuous quantity (the emitter's measurement)
    lo, hi    the continuous span of that quantity
    n_bins    the number of levels of the axis (the emitter's bin count)
    circular  True when the slot wraps (the axis is a cycle)
    quantiser the bins-path convention:
        "floor"           floor((v - lo) / (hi - lo) * n)      (q_linear)
        "round"           round((v - lo) / (hi - lo) * n)      (mod n)
        "signed_centered" round((v - lo) / (hi - lo) * (n - 1)) (odd level
                          count, zero at the middle level)
        "signed_log"      the signed log-compressed level (the CONTOUR
                          curvature sequence)
        "lookup_key"      the bin is a CATEGORICAL lookup key, NOT a
                          quantisation of the graded value (the POLARITY
                          class, the end-stop open side)
    keys      for a "lookup_key" slot: the categorical bin values
    """
    name: str
    unit: str
    lo: float
    hi: float
    n_bins: int
    circular: bool = False
    quantiser: str = "floor"
    keys: Tuple[float, ...] = ()


#: Back-compatible alias: the unit table entry used to be called PopSlot.
PopSlot = SlotSpec


def _slot(name: str, unit: str, lo: float, hi: float, n_bins: int,
          circular: bool = False, quantiser: str = "floor",
          keys: Tuple[float, ...] = ()) -> SlotSpec:
    return SlotSpec(name=name, unit=unit, lo=float(lo), hi=float(hi),
                    n_bins=int(n_bins), circular=bool(circular),
                    quantiser=str(quantiser),
                    keys=tuple(float(k) for k in keys))


def _normalised(spec: SlotSpec, value: float) -> float:
    """The value mapped onto [0, 1] over the slot's span: saturating on a
    linear slot (the emitter's clipping) and wrapping on a circular one."""
    span = spec.hi - spec.lo
    if span <= 0.0:
        return 0.0
    v = float(value)
    if spec.circular:
        return ((v - spec.lo) % span) / span
    return (min(max(v, spec.lo), spec.hi) - spec.lo) / span


def population_centre(spec: SlotSpec, value: float) -> float:
    """The GRADED path: the continuous BIN position of a measured value.

    The centre of the tuning bump, in the SAME unit and span as the bins
    path, so the bin the two readings name is the same bin.  This is the
    ONLY place a value is turned into a position.
    """
    n = spec.n_bins
    t = _normalised(spec, value)
    if spec.quantiser == "signed_centered":
        # an ODD level count whose zero sits on the middle level: the
        # position runs over (n - 1) intervals, not n.
        pos = t * (n - 1)
    elif spec.quantiser == "signed_log":
        # the signed log-compressed level: half = (n - 1) / 2 levels to
        # either side of zero.
        half = (n - 1) / 2.0
        pos = half * (1.0 + (2.0 * t - 1.0))
    else:
        pos = t * n
    if not spec.circular:
        # A linear axis has no room past its last level: the bump is centred
        # ON the last bin (the bin index is the bump's centre), never half a
        # bin outside the axis.
        pos = min(max(pos, 0.0), float(n - 1))
    return pos


def bin_at_centre(spec: SlotSpec, centre: float) -> int:
    """The BINS path: the bin a continuous centre falls in.

    This is the SINGLE rounding convention of the table; the emitters that
    read the table call `quantise`, which is this function applied to the
    population centre of the measured value.
    """
    n = spec.n_bins
    if spec.quantiser == "lookup_key":
        raise ValueError(
            f"slot {spec.name!r} is a lookup key: its bin is a categorical "
            f"class, not a quantisation of the graded value")
    if spec.quantiser == "signed_log":
        half = (n - 1) / 2.0
        d = float(centre) - half
        if abs(d) < 1e-12:
            return int(half)
        level = half + math.copysign(math.floor(abs(d) + 0.5), d)
        return int(min(n - 1, max(0, level)))
    if spec.quantiser in ("round", "signed_centered"):
        q = int(round(float(centre)))
    else:
        q = int(math.floor(float(centre)))
    if spec.circular:
        return q % n
    return int(min(n - 1, max(0, q)))


def quantise(spec: SlotSpec, value: float) -> int:
    """The BINS path for one slot: the bin index the emitter writes.

    Defined as the quantiser applied to the population centre of the
    measured value, so the lookup index and the graded bump can never name
    different bins of the same measurement.
    """
    return bin_at_centre(spec, population_centre(spec, value))


# The spans the emitters use. They are the emitters' own quantiser calls:
#   q_linear(v, n, lo, hi)      -> n bins over [lo, hi]
#   _q_unsigned(v, step, max)    -> max+1 bins over [0, step*(max+1))
#   _q_signed(v, max)            -> max bins over [-1, 1]
#   qi_circ(v, period, n)        -> n bins over [0, period)
POPULATION_SLOTS: Dict[str, Tuple[SlotSpec, ...]] = {
    # CORNER (corner_lens.py): the turn magnitude, the bisector orientation
    # (12 bins, axial), the turn prominence and the arm length fraction.
    "CORNER": (
        _slot("turn", "radians", 0.0, math.pi, 8),
        _slot("orient", "radians", 0.0, math.pi, 12, circular=True),
        _slot("prominence", "radians", 0.0, 0.5 * math.pi, 8),
        _slot("arm", "fraction_of_full_arm", 0.0, 1.0, 8),
    ),
    # CONTOUR_FRAGMENT (perception/polar.py): the signed curvature (asinh
    # compressed, 9 LEVELS with zero at the middle level), the polar angle
    # (24, circular over 2pi), the normalised radius and the tangent-vs-
    # radial angle.
    "CONTOUR_FRAGMENT": (
        _slot("kappa", "asinh_curvature", -1.0, 1.0, 9,
              quantiser="signed_centered"),
        _slot("theta", "radians", 0.0, 2.0 * math.pi, 24, circular=True,
              quantiser="round"),
        _slot("r", "radius_fraction", 0.0, 1.0, 8),
        _slot("tangent_vs_radial", "radians", 0.0, 0.5 * math.pi, 8),
    ),
    # COLOR (lenses/base.py): the RETINEX-OPPONENT measurements -- the
    # luminance component, the two opponent differences (r-g and b-y), the
    # saturation and the hue.  [O1] The class had a graded layout
    # (POPULATION_AXES) but NO unit table, so a reader could not ask for a
    # slot by name and the Surface emitter read the colour's bins by
    # position across a class boundary.  The spans and bin counts are the
    # ones the emitter already used (parameters.COLOR_*_BINS).
    "COLOR": (
        _slot("lum", "retinex_luminance", -1.0, 1.0, 8),
        _slot("rg", "red_green_opponent", -1.0, 1.0, 16),
        _slot("by", "blue_yellow_opponent", -1.0, 1.0, 12),
        _slot("sat", "saturation_level", 0.0, 1.0, 8),
        _slot("hue", "radians", 0.0, 2.0 * math.pi, 16, circular=True),
    ),
    # POLARITY (lenses/base.py): the SIGNED polarity (the mean/rms of the
    # even Gabor channel), continuous over [-1, 1]. The class' single slot is
    # a 3-class categorical today; the graded code carries the signed value.
    # The bin is the LOOKUP KEY (the class), the graded axis the measurement.
    "POLARITY": (
        _slot("signed", "signed_polarity", -1.0, 1.0, 8,
              quantiser="lookup_key", keys=(0.0, 1.0, 2.0)),
    ),
    # ENDSTOPPED (lenses/derived_base.py): the terminator orientation (24,
    # circular over pi), the scale index (the emitter bins the INDEX over
    # [0, nsf - 1]; the graded axis carries the same fraction of the range),
    # the OPEN-SIDE (signed: -1 open toward -a, +1 open toward +a, 0 no end;
    # its bin is the 3-class LOOKUP KEY) and the end-stop strength.
    "ENDSTOPPED": (
        _slot("theta", "radians", 0.0, math.pi, 24, circular=True,
              quantiser="round"),
        _slot("sf", "scale_index_fraction", 0.0, 1.0, 8),
        _slot("ends", "signed_open_side", -1.0, 1.0, 3,
              quantiser="lookup_key", keys=(0.0, 1.0, 2.0)),
        _slot("strength", "endstop_strength", 0.0, 1.0, 8),
    ),
    # BORDER_OWNERSHIP (lenses/v2.py, the fragment path): the arc position,
    # the SIGNED ownership direction, the signed curvature and the strength.
    "BORDER_OWNERSHIP": (
        _slot("arc", "arc_fraction", 0.0, 1.0, 8),
        _slot("signed_dir", "signed_direction", -1.0, 1.0, 8),
        _slot("curvature", "signed_curvature", -1.0, 1.0, 8),
        _slot("strength", "confidence", 0.0, 1.0, 8),
    ),
    # TEXTURE_BOUNDARY (lenses/v2.py, the FIELD arm -- the live one): the
    # arc position, the AXIAL orientation difference (24, circular over pi),
    # the log-scale-factor difference, the OPPONENT-CONTRAST DROP (the slot
    # the old layout mislabelled "log2_energy" over a [0, 3] span, while the
    # emitter measures a contrast drop in [0, 1]), the opponent-vector
    # distance, the luminance distance and the two flank concentrations.
    "TEXTURE_BOUNDARY": (
        _slot("arc_pos", "arc_fraction", 0.0, 1.0, 8),
        _slot("dtheta", "radians", 0.0, math.pi, 24, circular=True,
              quantiser="round"),
        _slot("dlogsf", "log2_scale_factor_ratio", -1.5, 1.5, 8),
        _slot("d_contrast", "opponent_contrast_drop", 0.0, 1.0, 8),
        _slot("d_ori", "opponent_vector_distance", 0.0, 1.0, 8),
        _slot("lum_dist", "luminance_distance", 0.0, 0.5, 8),
        _slot("conc_a", "opponent_concentration", 0.0, 1.0, 8),
        _slot("conc_b", "opponent_concentration", 0.0, 1.0, 8),
    ),
    # SURFACE_PATCH (lenses/v4.py): the shape (the curvature-variance
    # fraction), the luminance LEVEL, the hue (16, circular over 2pi), the
    # saturation LEVEL, the texture richness, the gradient magnitude (the
    # ownership field's strength) and the gradient side (its signed
    # direction).  The lum/hue/sat slots carry the SOURCE detection's own
    # level: their bin is that level's index and their graded axis is that
    # level's axis, so the two readings name the same level.
    "SURFACE_PATCH": (
        _slot("shape", "curvature_variance_fraction", 0.0, 1.0, 8),
        _slot("lum", "luminance_level", 0.0, 1.0, 8),
        _slot("hue", "radians", 0.0, 2.0 * math.pi, 16, circular=True),
        _slot("sat", "saturation_level", 0.0, 1.0, 8),
        _slot("tex", "texture_richness", 0.0, 1.0, 8),
        _slot("grad_mag", "ownership_strength", 0.0, 1.0, 8),
        _slot("grad_side", "signed_direction", 0.0, 1.0, 8),
    ),
    # SURFACE_FRAGMENT (lenses/surface_fragment_lens.py, Session B):
    # ONE surface point's graded 2.5D orientation, in the SAME order as the
    # class' arity.  Every slot is a MEASUREMENT the cue populations already
    # carry; none is a bin index echoed back.
    #   slant       the angle between the visible normal and the view
    #               direction, 0 .. pi/2 radians, 18 bins (5 deg);
    #   tilt        the normal's image-plane azimuth, 24 bins circular on
    #               2pi (the class' registered cycle), quantiser "round" so
    #               an angle sits at a bin centre;
    #   curvature   the owner-relative curvature SIGN as a signed scalar in
    #               [-1, +1] on 3 levels (signed_centered: -1, 0, +1).  The
    #               continuous value is welcome; the cue resolves the sign,
    #               which is what the emitter writes;
    #   confidence  the population's resultant length in [0, 1], 8 bins.
    "SURFACE_FRAGMENT": (
        _slot("slant", "radians", 0.0, 0.5 * math.pi, 18),
        _slot("tilt", "radians", 0.0, 2.0 * math.pi, 24, circular=True,
              quantiser="round"),
        _slot("curvature", "signed_curvature_sign", -1.0, 1.0, 3,
              quantiser="signed_centered"),
        _slot("confidence", "confidence", 0.0, 1.0, 8),
    ),
    # JUNCTION (lenses/v2.py): the degree (the arm count), the reference
    # orientation, the topology class, the NORMALISED gap spread, the four
    # arm gaps IN RADIANS, the arm support (the mean number of oriented
    # segments per arm), the arm sharpness (mean and min, both a doubled-
    # angle resultant in [0, 1]) and the eight pairwise arm-angle COUNTS.
    "JUNCTION": (
        _slot("degree", "arm_count", 0.0, 6.0, 6),
        _slot("ref_theta", "radians", 0.0, 2.0 * math.pi, 24, circular=True),
        _slot("topo", "topology_class", 0.0, 8.0, 8),
        _slot("spread", "normalised_gap_spread", 0.0, 1.0, 8),
        _slot("gap0", "radians", 0.0, 2.0 * math.pi, 8),
        _slot("gap1", "radians", 0.0, 2.0 * math.pi, 8),
        _slot("gap2", "radians", 0.0, 2.0 * math.pi, 8),
        _slot("gap3", "radians", 0.0, 2.0 * math.pi, 8),
        _slot("arm_support", "segments_per_arm", 0.0, 7.0, 8),
        _slot("arm_sharp_mean", "fraction", 0.0, 1.0, 8),
        _slot("arm_sharp_min", "fraction", 0.0, 1.0, 8),
        _slot("arm_angle0", "pair_count", 0.0, 8.0, 8),
        _slot("arm_angle1", "pair_count", 0.0, 8.0, 8),
        _slot("arm_angle2", "pair_count", 0.0, 8.0, 8),
        _slot("arm_angle3", "pair_count", 0.0, 8.0, 8),
        _slot("arm_angle4", "pair_count", 0.0, 8.0, 8),
        _slot("arm_angle5", "pair_count", 0.0, 8.0, 8),
        _slot("arm_angle6", "pair_count", 0.0, 8.0, 8),
        _slot("arm_angle7", "pair_count", 0.0, 8.0, 8),
    ),
    # CONTOUR (lenses/v2.py, derived_common._contour_slots): the arc length
    # (in members), the mean AXIAL orientation (24, circular over pi), the
    # circular variance, the closure flag, the total turn (log compressed),
    # the corner count, the chirality (signed) and the 20-bin canonical
    # signed curvature sequence (9 LEVELS -- the level count is the axis' bin
    # count; the constitutional range of 8.0 is a RANGE, not a bin count,
    # and deriving the axis from it aliased the top level).
    "CONTOUR": (
        _slot("length", "member_count", 0.0, 32.0, 8),
        _slot("mean_theta", "radians", 0.0, math.pi, 24, circular=True),
        _slot("circ_var", "fraction", 0.0, 1.0, 8),
        _slot("closure", "flag", 0.0, 1.0, 2),
        _slot("total_turn", "fraction", 0.0, 1.0, 8),
        _slot("corner_count", "count", 0.0, 8.0, 8),
        _slot("chirality", "signed_turn_over_2pi", -1.0, 1.0, 8),
    ) + tuple(_slot(f"curv{i}", "signed_log_curvature", -1.0, 1.0, 9,
                    quantiser="signed_log") for i in range(20)),
    # ORIENTATION_POL (lenses/base.py): the polarised single-region
    # orientation. Slot 0 is an AXIAL orientation, but the class' registered
    # range (24) does NOT follow the N/2 circular convention and the class
    # has no CIRCULAR_CYCLE_LENGTHS entry -- so the bins path reads it as a
    # LINEAR quantity. A circular axis here would need a registry change
    # (the range and the cycle), which is out of scope, so the graded axis
    # is declared LINEAR: the same reading the bins path uses, with the
    # sub-bin position the bin rounds off kept.
    # Slot 1 is the elongation (5 levels, the emitter's own count), slot 2
    # the 4x4 position class (16), and slot 3 the POLARITY: the 3-class
    # categorical is the LOOKUP KEY, the SIGNED value is the measurement
    # (the same contract as the POLARITY class).
    "ORIENTATION_POL": (
        _slot("theta", "radians", 0.0, math.pi, 24),
        _slot("elong", "fraction", 0.0, 1.0, 5),
        _slot("anchor", "position_class", 0.0, 16.0, 16),
        _slot("polarity", "signed_polarity", -1.0, 1.0, 8,
              quantiser="lookup_key", keys=(0.0, 1.0, 2.0)),
    ),
    # REGION_SHAPE (lenses/v4.py, B14 + B22): the 16 curvature-histogram
    # slots (each the square-root of the normalised arc mass in that turn
    # bin, quantised to 8 levels), the area (the log10 of the region's area
    # as a fraction of the image), the elongation (the eigenvalue ratio,
    # normalised), the symmetry score, the symmetry AXIS (the axis angle in
    # radians over the 12 registered axes) and the fold count.
    "REGION_SHAPE": (
    ) + tuple(_slot(f"curv{i}", "sqrt_curvature_mass", 0.0, 1.0, 8)
             for i in range(16)) + (
        _slot("area", "log10_area_fraction", -6.0, 0.0, 8),
        _slot("elong", "elongation_fraction", 0.0, 1.0, 8),
        _slot("sym", "symmetry_score", 0.0, 1.0, 8),
        _slot("axis", "radians", 0.0, math.pi, 12, circular=True,
              quantiser="round"),
        _slot("fold", "fold_count", 0.0, 6.0, 6),
    ),
    # CURVE_COMPLEX (lenses/v4.py, B.8): the contour count, the junction
    # count, the spatial extent (the normalised span), the closure (the mean
    # closure flag), the eight curvature-template ENERGIES (each normalised
    # to [-1, 1] over the bundle's strongest template) and the twelve
    # curvature-template POPULATIONS (the softmax weight of each template).
    "CURVE_COMPLEX": (
        _slot("n_contours", "count", 0.0, 8.0, 8),
        _slot("n_junctions", "count", 0.0, 8.0, 8),
        _slot("extent", "extent_fraction", 0.0, 1.0, 8),
        _slot("closure", "closure_fraction", 0.0, 1.0, 8),
    ) + tuple(_slot(f"template{i}", "template_energy", -1.0, 1.0, 8)
             for i in range(8)) + tuple(
        _slot(f"shape{i}", "template_population", 0.0, 1.0, 8)
        for i in range(12)) + tuple(
        # [PART 1 / BOUNDARY'S SPATIAL SIGNATURE] the boundary's curvature
        # AS A FUNCTION OF ARC LENGTH, composed from the member Contours'
        # own 20 signed_log_curvature levels: the per-slot extreme across
        # the bundle's members (max by |curvature|), preserving arc order
        # from arc start to arc end. The child stays OUT of the matcher;
        # the part carries the child's spatial signature.
        _slot(f"arc_curv{i}", "signed_log_curvature", -1.0, 1.0, 9,
              quantiser="signed_log")
        for i in range(20)),
    # ILLUSORY_CONTOUR (lenses/v4.py, B17): the completed orientation (the
    # axial angle in radians), the gap length as a FRACTION of the maximum
    # illusory gap, and the completion strength.
    "ILLUSORY_CONTOUR": (
        _slot("theta", "radians", 0.0, math.pi, 24, circular=True),
        _slot("length", "length_fraction", 0.0, 1.0, 8),
        _slot("strength", "completion_strength", 0.0, 1.0, 8),
    ),
    # [U30] The 8 canonical physical edge-categories (see _u30_inventory.md).
    # OCCLUSION carries relation: 0 = depth_order, 1 = figural_separation
    # (anti_phase pairs); those leave depth_diff at ABSENT=None.
    # SURFACE_FRAGMENT's kinematic axis (motion_kin) is dormant on static
    # frames; MATERIAL carries only the two chromatic border members.
    "EDGE_PROXIMITY": (
        _slot("distance", "spacings", 0.0, 8.0, 8),
        _slot("bearing", "radians_axial", 0.0, math.pi, 12, circular=True),
        _slot("scale_ratio", "log2_ratio", -2.0, 2.0, 8),
    ),
    "EDGE_CONTINUITY": (
        _slot("gap", "spacings", 0.0, 4.0, 8),
        _slot("turn_angle", "radians", -0.5 * math.pi, 0.5 * math.pi, 12,
              circular=True),
        _slot("perp_offset", "spacings", 0.0, 3.0, 8),
        _slot("curv_agree", "agreement", -1.0, 1.0, 8),
    ),
    "EDGE_CONTAINMENT": (
        _slot("rel_dist", "wavelengths", 0.0, 5.0, 8),
        _slot("polar_angle", "radians", 0.0, 2.0 * math.pi, 24, circular=True,
              quantiser="round"),
        _slot("scale_ratio", "log2_ratio", -2.0, 2.0, 8),
        _slot("ownership_strength", "fraction_of_max", 0.0, 1.0, 8),
        _slot("side", "sign", -1.0, 1.0, 2,
              quantiser="lookup_key", keys=(-1.0, 1.0)),
    ),
    "EDGE_OCCLUSION": (
        _slot("depth_diff", "rank_units", 0.0, 4.0, 8),
        _slot("source_cue", "cue_id", 0.0, 3.0, 4,
              quantiser="lookup_key", keys=(0.0, 1.0, 2.0, 3.0)),
        _slot("strength", "fraction_of_max", 0.0, 1.0, 8),
        _slot("relation", "kind", 0.0, 1.0, 2,
              quantiser="lookup_key", keys=(0.0, 1.0)),
    ),
    "EDGE_JUNCTION": (
        _slot("angle_delta", "radians_axial", 0.0, 0.5 * math.pi, 12),
        _slot("topo_class", "junction_type", 0.0, 6.0, 7,
              quantiser="lookup_key", keys=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
        _slot("arm_count", "arms", 2.0, 7.0, 6),
    ),
    "EDGE_SURFACE": (
        _slot("normal_angle", "radians", 0.0, math.pi, 12),
        _slot("material_diff", "fraction_of_max", 0.0, 1.0, 8),
        _slot("slant_diff", "radians_axial", 0.0, 0.5 * math.pi, 8),
        _slot("motion_kin", "fraction_of_max", 0.0, 1.0, 8),
    ),
    "EDGE_SHAPE": (
        _slot("sym_score", "fraction_of_max", 0.0, 1.0, 8),
        _slot("radial_ratio", "log2_ratio", -2.0, 2.0, 8),
        _slot("curvature", "asinh_curvature", -1.0, 1.0, 9,
              quantiser="signed_centered"),
    ),
    "EDGE_MATERIAL": (
        _slot("chrom_diff", "fraction_of_max", 0.0, 1.0, 8),
        _slot("lum_diff", "fraction_of_max", 0.0, 1.0, 8),
        _slot("tex_diff", "fraction_of_max", 0.0, 1.0, 8),
    ),
}

# The classes whose population is built from the CONTINUOUS values.


def _axes_from_slots(cls: str) -> Optional[Tuple[PopAxis, ...]]:
    """The axes of an EXPLICIT population layout (the emitter's bin counts)."""
    slots = POPULATION_SLOTS.get(cls)
    if not slots:
        return None
    return tuple(PopAxis(s.name, s.n_bins, circular=s.circular) for s in slots)


# The explicit layouts first: they carry the emitter's own bin counts.
for _c in POPULATION_SLOTS:
    _ax = _axes_from_slots(_c)
    if _ax is not None and _c not in POPULATION_AXES:
        POPULATION_AXES[_c] = _ax



# --- [ROUND 10 / A] Colour population -------------------------------------
# Hue on a CIRCULAR axis plus the luminance and opponent levels, one axis per
# COLOR slot (so the graded overlap replaces the bins comparison without
# dropping a channel). Emitted from the CONTINUOUS opponent values (sub-bin
# centre), not from the bins, so the vector carries the pre-quantization
# evidence. Circular hue n_bins == the registered COLOR cycle (16).
POPULATION_AXES["COLOR"] = (
    PopAxis("lum", 8, circular=False),
    PopAxis("rg", 16, circular=False),
    PopAxis("by", 12, circular=False),
    PopAxis("sat", 8, circular=False),
    PopAxis("hue", 16, circular=True),
)


def _select_axes(cls: str, idxs: Tuple[int, ...]) -> Tuple[PopAxis, ...]:
    """Content-tier population axes: the SOURCE class's axes for the selected
    slots, in the content projection's slot order. Same axis objects => the
    projected activity vectors have the length each axis declares, so the
    overlap is well-defined by construction."""
    src = POPULATION_AXES.get(cls, ())
    return tuple(src[i] for i in idxs if i < len(src))


# [ROUND 10 / A] Content populations are the SOURCE population PROJECTED onto
# the content slots (a selection, never an average), so a content signature
# carries the same graded evidence its source detection holds. Slot maps mirror
# content.content_signature() exactly.
POPULATION_AXES["JUNCTION_CONTENT"] = _select_axes(
    "JUNCTION", (0, 2, 3, 9, 11, 12, 13, 14, 15, 16, 17, 18))
POPULATION_AXES["ENDSTOPPED_CONTENT"] = _select_axes("ENDSTOPPED", (0, 1))
POPULATION_AXES["SURFACE_PATCH_CONTENT"] = _select_axes(
    "SURFACE_PATCH", (0, 1, 2, 3, 4, 5))
POPULATION_AXES["TEXTURE_BOUNDARY_CONTENT"] = _select_axes(
    "TEXTURE_BOUNDARY", (1, 2, 3, 4, 5, 6, 7))
# Colour content: hue (circular) + saturation, matching the COLOR source axes
# 4 and 3. COLOR_LEGACY_CONTENT has no source population (legacy RGB path), so
# its population is derived from its own hue/sat bins with the same layout.
POPULATION_AXES["COLOR_CONTENT"] = (
    PopAxis("hue", 16, circular=True),
    PopAxis("sat", 8, circular=False),
)
POPULATION_AXES["COLOR_LEGACY_CONTENT"] = POPULATION_AXES["COLOR_CONTENT"]


def population_for_bins(class_name: str,
                        bins: Sequence[float]) -> Tuple[Tuple[float, ...], ...]:
    """The graded population of one node from its quantized bins: one soft
    tuning bump per registered axis, centred at the bin index. Returns () when
    the class has no registered population or the arity disagrees, so the
    signature falls back to the bins path in similarity (never a silent drop).

    THIS IS THE FALLBACK, not the graded code: it can only echo the bin, so
    it adds NO sub-bin information. A class that declares a layout in
    POPULATION_SLOTS must build its population with
    `population_from_values` instead, from the pre-quantisation value.
    """
    axes = POPULATION_AXES.get(class_name)
    if axes is None or len(axes) != len(bins):
        return ()
    from core.population import soft_bump_bins
    return tuple(soft_bump_bins(float(b), ax) for ax, b in zip(axes, bins))


# ---------------------------------------------------------------------------
# [U22] NAME-BASED POPULATION ADDRESSING.
#
# A graded population is a tuple of activity vectors, one per axis, and the
# axis has a NAME.  Until U22 every writer addressed an axis by its POSITION
# in POPULATION_SLOTS / POPULATION_AXES (`_bump(1, lum)`, `axes[0]`), so a
# reordering of the registry would silently rewrite a measurement onto a
# different quantity -- the same defect the O1 note records for a class whose
# bins were read by position across a class boundary.
#
# The four helpers below are the ONLY supported way to reach an axis:
#   slot_axes                   the (slot, axis) pairs of a class, BY NAME
#   population_axis             one axis of a class, by name
#   population_for_slot_value    one axis' bump for one measured value, by name
#   population_from_named_vectors  a whole population from per-name vectors
#   population_slot_vector      one vector of a BUILT population, by name
# The honesty gate is tests/perception/test_population_scramble_u22.py.
# ---------------------------------------------------------------------------
def slot_axes(class_name: str) -> Tuple[Tuple[SlotSpec, PopAxis], ...]:
    """The class' (slot, axis) pairs, MATCHED BY SLOT NAME.

    Both tables are keyed by name, so the pairing is independent of the order
    either table was written in.  Returns () when a slot has no same-named
    axis (a layout disagreement is never mixed into a population).
    """
    slots = POPULATION_SLOTS.get(class_name) or ()
    axes = POPULATION_AXES.get(class_name) or ()
    by_name = {a.name: a for a in axes}
    pairs: List[Tuple[SlotSpec, PopAxis]] = []
    for s in slots:
        ax = by_name.get(s.name)
        if ax is None:
            return ()
        pairs.append((s, ax))
    return tuple(pairs)


def population_axis(class_name: str, axis_name: str) -> Optional[PopAxis]:
    """ONE axis of a class, by NAME (None when the class declares none)."""
    for ax in POPULATION_AXES.get(class_name) or ():
        if ax.name == axis_name:
            return ax
    return None


def population_for_slot_value(class_name: str,
                              slot_name: str,
                              value: Optional[float]
                              ) -> Tuple[float, ...]:
    """ONE slot's activity vector for one measured value, addressed BY NAME.

    The graded companion of `quantise(slot_spec(cls, name), value)`: the same
    unit-table entry, the same centre, so the vector and the bin name the
    same measurement.  Returns () when the slot or its axis is undeclared, or
    when the value is not finite.
    """
    spec = slot_spec(class_name, slot_name)
    axis = population_axis(class_name, slot_name)
    if spec is None or axis is None:
        return ()
    if value is None:
        # NO MEASUREMENT: the uninformative (uniform) activity, never a
        # fabricated value.
        return tuple([1.0 / axis.n_bins] * axis.n_bins)
    x = float(value)
    if not math.isfinite(x):
        return ()
    from core.population import soft_bump_bins
    return soft_bump_bins(population_centre(spec, x), axis)


def population_from_named_vectors(class_name: str,
                                  vectors: Dict[str, Sequence[float]]
                                  ) -> Tuple[Tuple[float, ...], ...]:
    """A whole population from per-slot vectors addressed BY NAME.

    The result is in the class' CURRENT slot order, so a reader can zip it
    with `slot_axes(class_name)` safely.  Returns () on any missing vector or
    a vector whose length is not its axis' bin count (never a mis-shaped
    population).
    """
    pairs = slot_axes(class_name)
    if not pairs:
        return ()
    out: List[Tuple[float, ...]] = []
    for spec, axis in pairs:
        vec = vectors.get(spec.name)
        if vec is None or len(vec) != axis.n_bins:
            return ()
        out.append(tuple(vec))
    return tuple(out)


def population_slot_vector(population: Sequence[Sequence[float]],
                           class_name: str,
                           slot_name: str
                           ) -> Tuple[float, ...]:
    """ONE activity vector of a BUILT population, by the axis' NAME.

    Used when a population is projected ACROSS classes (the surface patch's
    material is the colour detection's own graded vector): addressing the
    source vector by name, never by its position in the source tuple.

    [U22b] A BUILT population is ordered by POPULATION_SLOTS, not by
    POPULATION_AXES: `population_from_values` / `population_for_slot_value` /
    `population_from_named_vectors` assemble in slot order.  Reading the
    position from the AXIS table (the pre-U22b form) was name-based only while
    the two tables happened to have the same order; a reversed axis table sent
    this reader to the wrong vector.

    `population_for_bins` is the exception: it returns one vector per AXIS in
    axis order (its caller supplies anonymous bins).  Do not pass its result
    here unless the two tables are known to share an order; use the bins
    directly instead.
    """
    names = [s.name for s in (POPULATION_SLOTS.get(class_name) or ())]
    if not population or slot_name not in names:
        return ()
    pos = names.index(slot_name)
    if pos >= len(population):
        return ()
    return tuple(population[pos])


def population_from_values(class_name: str,
                           values: Sequence[float]
                           ) -> Tuple[Tuple[float, ...], ...]:
    """The graded population of one node from its PRE-QUANTISATION values.

    Each axis' tuning bump is centred at the value's sub-bin position, so
    the activity vector carries the information the bin throws away -- the
    representation thesis: the full vector on the detection, the bin as a
    lookup key only.

    `values` are in the SAME ORDER and SPAN as the class' POPULATION_SLOTS
    layout. Returns () when the class has no declared layout or the arity
    disagrees (never a silent, mis-shaped population).

    [U22] The slot and its axis are paired BY NAME (`slot_axes`), not by
    position, so a reordered registry cannot move a measurement onto another
    quantity.
    """
    pairs = slot_axes(class_name)
    if not pairs or len(values) != len(pairs):
        return ()
    from core.population import soft_bump_bins
    out = []
    for v, (s, ax) in zip(values, pairs):
        if v is None:
            # NO MEASUREMENT for this slot: the axis carries the uniform
            # (uninformative) activity, which the overlap reads as 0.0 --
            # never a fabricated value.
            out.append(tuple([1.0 / ax.n_bins] * ax.n_bins))
            continue
        x = float(v)
        if not math.isfinite(x):
            return ()
        # The bump is centred on the UNIT TABLE's centre for this slot: the
        # same number the bins path quantises, so the graded reading and the
        # lookup index name the same measurement.
        out.append(soft_bump_bins(population_centre(s, x), ax))
    return tuple(out)


def population_values_in_span(class_name: str,
                              values: Sequence[float]
                              ) -> Tuple[float, ...]:
    """The continuous values clamped into the class' declared spans.

    A value outside its slot's span is SATURATED the way the emitter's
    quantiser saturates it (the last bin of a linear slot) and WRAPS on a
    circular slot, so the graded bump lands on the same bin the coarse
    bin names: the two readings never disagree about the measurement.

    A None entry means "NO MEASUREMENT for this slot" and is carried
    through as such (the population then holds the uniform activity for
    that axis).

    Returns () when the class has no declared layout or the arity
    disagrees (never a silent, mis-shaped population).
    """
    slots = POPULATION_SLOTS.get(class_name)
    if not slots or len(slots) != len(values):
        return ()
    out: List[float] = []
    for v, s in zip(values, slots):
        if v is None:
            # no measurement for this slot: carried through as such.
            out.append(None)  # type: ignore[arg-type]
            continue
        x = float(v)
        if not math.isfinite(x):
            return ()
        span = s.hi - s.lo
        if span <= 0.0:
            out.append(float(s.lo))
            continue
        if s.circular:
            x = s.lo + ((x - s.lo) % span)
        else:
            # Saturate just below the top of the span: the last bin.
            x = min(max(x, s.lo), s.hi - 1e-9 * max(1.0, abs(span)))
        out.append(float(x))
    return tuple(out)


def slot_specs(class_name: str) -> Tuple[SlotSpec, ...]:
    """The UNIT TABLE of one class: one entry per slot, in slot order."""
    return POPULATION_SLOTS.get(class_name, ())


def slot_spec(class_name: str, slot_name: str) -> Optional[SlotSpec]:
    """The unit-table entry of ONE slot of one class, by name.

    The lookup is by NAME, so it never depends on the order the table was
    written in.
    """
    for s in POPULATION_SLOTS.get(class_name, ()):
        if s.name == slot_name:
            return s
    return None


def get_population_axes(class_name: str) -> Optional[Tuple[PopAxis, ...]]:
    """The declared population layout for a class, or None (bins only)."""
    return POPULATION_AXES.get(class_name)


def _verify_population_axes(ranges: dict, cycles: dict, pop: dict) -> None:
    """Each circular axis's n_bins must equal the class's registered cycle for
    the matching quantity; each axis n_bins must equal the class range length
    where the axis maps 1:1 onto slots is NOT assumed (populations are their own
    layout), so only the circular/cycle agreement is enforced."""
    for _cls, _axes in pop.items():
        _cycles = cycles.get(_cls, {})
        _names_cycle = {}
        for _slot, _cyc in _cycles.items():
            _names_cycle[_cyc] = _names_cycle.get(_cyc, 0) + 1
        for _ax in _axes:
            if _ax.circular and _ax.n_bins not in _names_cycle:
                # A circular population axis must correspond to a registered
                # circular slot with the same bin count, or the wrap convention
                # the similarity uses would diverge from the class's.
                import warnings as _w
                _w.warn(
                    f"POPULATION_AXES[{_cls}].{_ax.name}: circular axis with "
                    f"n_bins={_ax.n_bins} has no matching CIRCULAR_CYCLE_LENGTHS "
                    f"entry {_cycles}")


_verify_population_axes(CONSTITUTIONAL_RANGES, CIRCULAR_CYCLE_LENGTHS,
                        POPULATION_AXES)

def b1_class_name(nt: int, nsf: int) -> str:
    """Canonical identifier for B1 hypercolumn population schema."""
    return f"ORIENTATION_RICH_{nt}_{nsf}"

def b4_class_name(nt: int, nsf: int) -> str:
    """Canonical identifier for B4 second-order texture schema."""
    return f"TEXTURE_RICH_{nt}_{nsf}"


# ---------------------------------------------------------------------------
# [O1] THE RICH CLASS UNIT TABLE.
#
# ORIENTATION_RICH_{nt}_{nsf} and TEXTURE_RICH_{nt}_{nsf} are the two
# signature classes whose slots carry CONTINUOUS MEASUREMENTS (not bin
# indices), so their slots' spans are the measurements' own spans.  Until
# now they had NO unit table: their similarity ranges were a flat
# CONSTITUTIONAL_RANGES entry of 1.0 per slot, so a phase difference of
# half a radian was compared as if it were half a slot and drove the
# similarity to ~0.1 on its own, and the ORIENTATION_RICH_CONTENT
# projection (the first 60 slots) inherited the same flat ranges even
# though several of those slots span 2*pi.
#
# The layouts below are the SINGLE SOURCE for the rich classes' slot
# names, spans and periods.  perception/rich.py attaches the WEIGHTS (a
# separate concern) and derives the class' similarity ranges and
# circular cycles from these layouts, so the two readings of a rich slot
# cannot disagree -- the same property the POPULATION_SLOTS unit table
# gives the other classes.
# ---------------------------------------------------------------------------

B1_BASE_SLOTS: Tuple[Tuple[str, float, float, Optional[float]], ...] = (
    ("E_peak", 0.0, 1.0, None),
    ("energy_conc", 0.0, 1.0, None),
    ("sf_peak_ratio", 0.0, 2.0, None),
    ("is_2d_peak", 0.0, 1.0, None),
    ("region_lum_mean", 0.0, 1.0, None),
    ("region_lum_rms", 0.0, 1.0, None),
    ("surround_lum_mean", 0.0, 1.0, None),
    ("lum_mean_diff", -1.0, 1.0, None),
    ("boundary_interior_ratio", 0.0, 1.0, None),
    ("simple_flag", 0.0, 1.0, None),
    ("on_simple_flag", 0.0, 1.0, None),
    ("off_simple_flag", 0.0, 1.0, None),
    ("cross_scale_phase", -math.pi, math.pi, 2 * math.pi),
    ("dtheta", -math.pi / 2, math.pi / 2, math.pi),
    ("dlogsf", -1.0, 1.0, None),
)

B1_PHASE_DIFF_SLOTS: Tuple[str, ...] = (
    "phase_diff_theta_plus", "phase_diff_theta_minus",
    "phase_diff_theta_plus2", "phase_diff_theta_minus2",
    "phase_diff_sf_plus", "phase_diff_sf_minus",
    "phase_diff_sf_plus2", "phase_diff_sf_minus2",
    "phase_grad_theta", "phase_grad_sf",
    "phase_diff_tp_sp", "phase_diff_tp_sm",
    "phase_diff_tm_sp", "phase_diff_tm_sm",
)

B1_PHASE_R_SLOTS: Tuple[str, ...] = (
    "phase_R_theta_plus", "phase_R_theta_minus",
    "phase_R_theta_plus2", "phase_R_theta_minus2",
    "phase_R_sf_plus", "phase_R_sf_minus",
    "phase_R_sf_plus2", "phase_R_sf_minus2",
    "on_phase_R", "off_phase_R", "on_phase_skew", "off_phase_skew",
)

B1_ONOFF_PHASE_DIFF_SLOTS: Tuple[str, ...] = (
    "on_phase_diff_theta_plus", "on_phase_diff_theta_minus",
    "off_phase_diff_theta_plus", "off_phase_diff_theta_minus",
    "on_off_phase_diff",
)


def b1_slot_layout(nt: int, nsf: int) -> Tuple[Tuple[str, float, float, Optional[float], str], ...]:
    """The unit table of ORIENTATION_RICH_{nt}_{nsf}:
    (name, lo, hi, period, weight_block).

    The order IS the slot order.  The weight BLOCK names the group a slot
    belongs to; perception/rich.py turns a block into a weight (the weights
    are the writer's business, the layout is the registry's), so the two
    cannot drift apart.
    """
    out: List[Tuple[str, float, float, Optional[float], str]] = [
        (n, lo, hi, p, "base") for (n, lo, hi, p) in B1_BASE_SLOTS]
    for prefix in ("hist", "on_hist", "off_hist"):
        for k in range(nt * nsf):
            out.append((f"{prefix}_{k}", 0.0, 1.0, None, "hist"))
    out.append(("phase_R", 0.0, 1.0, None, "phase"))
    out.append(("phase_skew", 0.0, 1.0, None, "phase"))
    for n in B1_PHASE_DIFF_SLOTS:
        out.append((n, -math.pi, math.pi, 2 * math.pi, "phase"))
    for n in B1_PHASE_R_SLOTS:
        out.append((n, 0.0, 1.0, None, "phase"))
    for n in B1_ONOFF_PHASE_DIFF_SLOTS:
        out.append((n, -math.pi, math.pi, 2 * math.pi, "phase"))
    out.append(("surround_theta", 0.0, math.pi, math.pi, "surround"))
    out.append(("surround_theta_conc", 0.0, 1.0, None, "surround"))
    out.append(("surround_logsf_mean", -5.0, 0.0, None, "surround"))
    out.append(("surround_logsf_std", 0.0, 2.0, None, "surround"))
    out.append(("surround_on_mean", 0.0, 1.0, None, "surround"))
    out.append(("surround_off_mean", 0.0, 1.0, None, "surround"))
    for k in range(nt * 2):
        out.append((f"surround_hist_{k}", 0.0, 1.0, None, "surround"))
    for q in range(4):
        for k in range(nt * 3):
            out.append((f"sp_hist_{q}_{k}", 0.0, 1.0, None, "sp_hist"))
        out.append((f"sp_theta_rel_{q}", -math.pi / 2, math.pi / 2, math.pi,
                    "sp_nonhist"))
        out.append((f"sp_nonempty_{q}", 0.0, 1.0, None, "sp_nonhist"))
        out.append((f"sp_R_{q}", 0.0, 1.0, None, "sp_nonhist"))
        out.append((f"sp_dtheta_{q}", -math.pi / 2, math.pi / 2, math.pi,
                    "sp_nonhist"))
        out.append((f"sp_dlogsf_{q}", -1.0, 1.0, None, "sp_nonhist"))
    for n in ("sp_phase_diff_01", "sp_phase_diff_23",
              "sp_phase_diff_02", "sp_phase_diff_13"):
        out.append((n, -math.pi, math.pi, 2 * math.pi, "q_phase_diff"))
    for n in ("sp_theta_diff_01", "sp_theta_diff_23"):
        out.append((n, -math.pi / 2, math.pi / 2, math.pi, "q_theta_diff"))
    out.append(("on_dtheta", -math.pi / 2, math.pi / 2, math.pi, "onoff_subbin"))
    out.append(("on_dlogsf", -1.0, 1.0, None, "onoff_subbin"))
    out.append(("off_dtheta", -math.pi / 2, math.pi / 2, math.pi, "onoff_subbin"))
    out.append(("off_dlogsf", -1.0, 1.0, None, "onoff_subbin"))
    out.append(("sf_curvature", -1.0, 1.0, None, "sf_curv"))
    return tuple(out)


def b4_slot_layout(nt: int, nsf: int) -> Tuple[Tuple[str, float, float, Optional[float]], ...]:
    """The unit table of TEXTURE_RICH_{nt}_{nsf}: (name, lo, hi, period).

    Three second-order statistics per (theta, sf) cell (the
    autocorrelation peak, its normalised period, and its direction
    relative to the carrier's long axis), one cross-channel covariance
    per (theta, sf) cell, and four invariant globals.
    """
    out: List[Tuple[str, float, float, Optional[float]]] = []
    for it in range(nt):
        for isf in range(nsf):
            out.append((f"ac_peak_{it}_{isf}", 0.0, 1.0, None))
            out.append((f"ac_period_{it}_{isf}", 0.0, 1.0, None))
            out.append((f"ac_dir_{it}_{isf}", 0.0, 1.0, None))
    for it in range(nt):
        for isf in range(nsf):
            out.append((f"xchan_{it}_{isf}", 0.0, 1.0, None))
    out.append(("orient_conc", 0.0, 1.0, None))
    out.append(("sf_centroid", 0.0, 1.0, None))
    out.append(("roughness", 0.0, 1.0, None))
    out.append(("contrast_energy", 0.0, 1.0, None))
    return tuple(out)

def compute_b1_arity(nt: int, nsf: int) -> int:
    """Closed-form arity formula for B1 hypercolumn population signature:
    3*(nt*nsf) + 14*nt + 85 = 433 slots for a 12x5 bank."""
    return 3 * (nt * nsf) + 14 * nt + 85

def compute_b4_arity(nt: int, nsf: int) -> int:
    """Closed-form arity formula for B4 second-order texture signature:
    4*(nt*nsf) + 4 globals = 244 slots for a 12x5 bank."""
    return 4 * (nt * nsf) + 4

_GABOR_NT = getattr(P, "GABOR_N_THETA", 12)

_GABOR_NSF = getattr(P, "GABOR_N_SF", 5)

# [O1] The live bank shape, published so the slot-name table (core/slots.py)
# describes the SAME bank the emitters fill.
GABOR_NT: int = _GABOR_NT

GABOR_NSF: int = _GABOR_NSF

_B1_DERIVED_ARITY = compute_b1_arity(_GABOR_NT, _GABOR_NSF)

_B4_DERIVED_ARITY = compute_b4_arity(_GABOR_NT, _GABOR_NSF)

# [O1] The rich classes' unit tables (the single source for their slot
# names, spans and periods).
_B1_LAYOUT = b1_slot_layout(_GABOR_NT, _GABOR_NSF)

_B4_LAYOUT = b4_slot_layout(_GABOR_NT, _GABOR_NSF)

_ORIENTATION_MARGINALS = getattr(P, "ORIENTATION_RICH_MARGINAL_BINS", 60)

if len(_B1_LAYOUT) != _B1_DERIVED_ARITY:
    raise AssertionError(
        f"B1 unit table has {len(_B1_LAYOUT)} slots, the arity formula says "
        f"{_B1_DERIVED_ARITY}")
if len(_B4_LAYOUT) != _B4_DERIVED_ARITY:
    raise AssertionError(
        f"B4 unit table has {len(_B4_LAYOUT)} slots, the arity formula says "
        f"{_B4_DERIVED_ARITY}")

CONSTITUTIONAL_RANGES["ORIENTATION_RICH_CONTENT"] = tuple(
    hi - lo for (_n, lo, hi, _p, _b) in _B1_LAYOUT[:_ORIENTATION_MARGINALS])

CONSTITUTIONAL_RANGES["TEXTURE_RICH_CONTENT"] = tuple(
    hi - lo for (_n, lo, hi, _p) in _B4_LAYOUT[-4:])

CONSTITUTIONAL_RANGES[b1_class_name(_GABOR_NT, _GABOR_NSF)] = tuple(
    hi - lo for (_n, lo, hi, _p, _b) in _B1_LAYOUT)

CONSTITUTIONAL_RANGES[b4_class_name(_GABOR_NT, _GABOR_NSF)] = tuple(
    hi - lo for (_n, lo, hi, _p) in _B4_LAYOUT)

# [O1] The rich classes' CIRCULAR cycles are declared at BOOT as well as at
# runtime (perception/rich.py's b1_register writes them again per image).
# Without the boot-time entry, a comparison made before the first image
# wrapped a phase slot as if it were a linear one.
_B1_CYCLES = {i: p for i, (_n, _lo, _hi, p, _b) in enumerate(_B1_LAYOUT) if p}
if _B1_CYCLES:
    CIRCULAR_CYCLE_LENGTHS[b1_class_name(_GABOR_NT, _GABOR_NSF)] = _B1_CYCLES


def get_similarity_ranges(class_name: str) -> tuple[int, ...] | None:
    """Return the constitutional R_j tuple for a CANONICAL signature class,
    or None if unregistered (which forces sim = 0.0)."""
    return CONSTITUTIONAL_RANGES.get(class_name)

SLOT_WEIGHTS: Dict[str, Tuple[float, ...]] = {}

SLOT_WEIGHTS_BY_CONTEXT: Dict[Tuple[str, Any], Tuple[float, ...]] = {}

SLOT_WEIGHT_PRIORS: Dict[str, Tuple[float, ...]] = {}

def resolve_slot_weights(
    class_name: str,
    n_slots: int,
    context: Any = None,
) -> Tuple[float, ...]:
    """context-specific -> class prior -> legacy SLOT_WEIGHTS -> uniform.

    [B23] The arity check RAISES rather than warns. A length mismatch here
    silently misaligns every weight against the wrong slot, degrading
    similarity in a way no harness will attribute to this function.
    """
    candidates = []
    if context is not None:
        candidates.append(SLOT_WEIGHTS_BY_CONTEXT.get((class_name, context)))
    candidates.append(SLOT_WEIGHT_PRIORS.get(class_name))
    candidates.append(globals().get("SLOT_WEIGHTS", {}).get(class_name))  # legacy registry
    for w in candidates:
        if w is not None:
            if len(w) != n_slots:
                raise ValueError(
                    f"slot weights for {class_name}: {len(w)} != {n_slots}"
                )
            return w
    return (1.0,) * n_slots

def get_slot_weights(
    class_name: str,
    n_slots: Optional[int] = None,
    context: Any = None,
) -> Tuple[float, ...]:
    """Back-compatible entry point.

    [B23] n_slots is OPTIONAL here, not positional-required. Existing call
    sites pass only the class name; making it mandatory would break every one
    of them at import.
    """
    if n_slots is None:
        rng = CONSTITUTIONAL_RANGES.get(class_name)
        if not rng:
            raise KeyError(
                f"unknown class {class_name!r} and no n_slots given"
            )
        n_slots = len(rng)
    return resolve_slot_weights(class_name, n_slots, context)

# --- moved from binder.py (Move 3c): edge-layer map + the
# self-binder / order-2 registry constants --------------------------

# --- moved from binder.py (Move 3c): registry constants ------------
SELF_BINDER_TYPES: FrozenSet[EdgeType] = frozenset()

# §2.3: order-2 edge types. Empty in v31 — structural similarity was removed
# from the binder (a comparison emits no relation; see structural_similarity.py)
# and the set is retained only as the single extension point for a future
# order-2 binder. edge_type_layer() therefore contributes 0 for every type.

