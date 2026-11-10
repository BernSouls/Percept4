"""The figure-ground pipeline driver: one call, luminance -> every cue the
U11 census asks about.

This is a TEST HARNESS, not production code.  It reproduces the wiring the
real tree does across PeakLens -> ChainLens -> BorderOwnershipLens ->
SurfaceFragmentLens -> binder, using the supplied modules directly, so a
regression in any of them shows up here.

Every stage's output is kept on the returned ``Run`` so a test can assert on
the intermediate quantity that broke, not just on the final answer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------

@dataclass
class Run:
    scene_name: str
    luminance: np.ndarray
    field: Dict[str, Any]
    peaks: List[Dict[str, Any]]
    anchor_positions: Dict[Any, Tuple[float, ...]]
    peak_dets: List[Any]
    chain_dets: List[Any]
    chains: List[Tuple[List[int], bool]]
    records: List[Dict[str, Any]]
    bo_dets: List[Any]
    interior: Optional[Dict[str, np.ndarray]]
    fragments: List[Any]
    order: Dict[str, Any] = field(default_factory=dict)
    layers: Dict[int, int] = field(default_factory=dict)
    tjs_live: List[Tuple[float, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # -- convenience views -------------------------------------------------
    def median_wavelength(self) -> Optional[float]:
        sfs = self.field.get("sfs")
        if sfs is None:
            return None
        arr = np.asarray(sfs, dtype=np.float64).ravel()
        if arr.size == 0:
            return None
        return float(np.median(1.0 / np.maximum(arr, 1e-6)))

    def record_by_alpha(self) -> Dict[Any, Dict[str, Any]]:
        return {r["alpha"]: r for r in self.records}

    def frag_by_alpha(self) -> Dict[Any, Any]:
        return {f.alpha: f for f in self.fragments}

    def chain_of(self, alpha: Any) -> Optional[Any]:
        for r in self.records:
            if r["alpha"] == alpha:
                return r["chain"]
        return None

    def records_near(self, x: float, y: float, radius: float
                     ) -> List[Dict[str, Any]]:
        out = []
        for r in self.records:
            if math.hypot(r["pos"][0] - x, r["pos"][1] - y) <= radius:
                out.append(r)
        return out

    def fragments_near(self, x: float, y: float, radius: float) -> List[Any]:
        return [f for f in self.fragments
                if math.hypot(f.position[0] - x, f.position[1] - y) <= radius]


# --------------------------------------------------------------------------

def run_pipeline(scene, *,
                 with_live_junctions: bool = False,
                 junction_radius_scale: float = 1.5,
                 bridge: bool = False,
                 size: Optional[int] = None,
                 keep_field: bool = False,
                 ) -> Run:
    """Luminance -> field -> peaks -> chains -> ownership -> fragments -> order.

    ``with_live_junctions`` is the H5 switch: False reproduces the SHIPPED tree
    (Observation2D exposes no ``get_t_junctions``), True injects the junction
    sites a live detector would supply, through the very parameter
    ``fragment_ownership_records`` already has for them.
    """
    from lenses.peak_lens import PeakLens, field_for_observation
    from lenses.chain_lens import ChainLens
    from perception.fragment_ownership import fragment_ownership_records
    from perception.interior_field import interior_field_for
    from perception import surface_fragments as SF

    class _Obs:
        """The observation view this harness needs: luminance + field caches.

        The declared tree's `Observation2D` is a full region-graph container
        with a different constructor; every consumer here reads only these
        members, so this keeps the harness independent of that surface.
        """

        def __init__(self, luminance, rgb=None, t_junctions=()):
            self._luminance = luminance
            self.rgb = rgb
            self._t_junctions = list(t_junctions)
            self._contour_field_cache = None
            self._contour_field_cache_key = None
            self._interior_field_cache = None
            self._interior_field_cache_key = None

        def get_luminance(self):
            return self._luminance

        def get_t_junctions(self):
            return list(self._t_junctions)

    lum = np.asarray(scene.luminance, dtype=np.float32)
    obs: Any = _Obs(luminance=lum, rgb=scene.rgb)

    # -- A0/A1: the field ---------------------------------------------------
    fld = field_for_observation(obs, lum)

    # [U30 R-D3] The observation the chord gate reads must carry the Gabor
    # complex cache or the gate abstains wherever it cannot see the map. The
    # shim used to skip it entirely, silencing the gate on every figground
    # scene (pre-D3, the gate still saw `field["endstop"]` here, so the old
    # ratio rule WAS active on this path). `v1_responses` is content-cached,
    # so this costs ~nothing: same (lum, scale, fingerprint) key.
    try:
        from perception.contour_field import v1_responses, front_end_scale
        _scale = front_end_scale(lum.shape[0], lum.shape[1])
        _simple, _complex_raw, _phase = v1_responses(lum, scale=_scale)
        obs._gabor_complex = np.hypot(
            _simple[:, :, 0], _simple[:, :, 1]).astype(np.float32)
        from perception.contour_field import _get_bank
        _bank = _get_bank(_scale)
        obs._gabor_thetas = list(_bank.theta)
        obs._gabor_sfs = list(_bank.sf)
    except Exception as _exc:  # evidence unavailable -> gate abstains
        print("[figground] Gabor cache wiring failed: %s" % (_exc,), flush=True)

    peaks = list(fld.get("peaks") or [])

    anchor_positions: Dict[Any, Tuple[float, ...]] = {}

    # -- A2: CONTOUR_PEAK ---------------------------------------------------
    peak_dets = PeakLens().step(0, None, None, anchor_positions, None, {},
                                0.0, None, observation=obs, luminance=lum)

    # -- A5: PEAK_CHAIN -----------------------------------------------------
    chain_lens = ChainLens()
    chain_dets = chain_lens.step(0, None, None, anchor_positions, None,
                                 {"CONTOUR_PEAK": peak_dets}, 0.0, None,
                                 observation=obs, luminance=lum)

    # the raw chain structure, for the "is the occluder one closed ring?" column
    from perception.chain_formation import form_chains, form_chains_detailed
    median_wl = None
    sfs = fld.get("sfs")
    if sfs is not None:
        arr = np.asarray(sfs, dtype=np.float64).ravel()
        if arr.size:
            median_wl = float(np.median(1.0 / np.maximum(arr, 1e-6)))
    chains: List[Tuple[List[int], bool]] = []
    if peaks and median_wl:
        endstop = np.asarray(fld["endstop"])
        support = fld.get("integration_driving")
        if support is None:
            support = fld["integration"]
        orientations = None
        try:
            from perception.chain_formation import continuous_orientations
            cplx = fld.get("complex")
            if cplx is not None:
                orientations = continuous_orientations(
                    np.asarray(cplx, dtype=np.float64), peaks,
                    sfs=np.asarray(fld["sfs"], dtype=np.float64))
        except Exception:
            orientations = None
        fn = form_chains_detailed if bridge else form_chains
        out = fn(peaks, endstop, median_wl,
                 integration=np.asarray(support),
                 sfs=np.asarray(fld["sfs"], dtype=np.float64),
                 orientations=orientations)
        chains = out[0] if bridge else out

    # -- H5: the live T-junction sites -------------------------------------
    tjs_live: List[Tuple[float, float]] = []
    if with_live_junctions:
        tjs_live = live_t_junctions(scene, anchor_positions, chain_dets,
                                    peaks, median_wl)
        obs = _Obs(luminance=lum, rgb=scene.rgb,
                   t_junctions=[(0, 1, (x, y), 1.0)
                                for (x, y) in tjs_live])
        obs._contour_field_cache = fld
        obs._contour_field_cache_key = getattr(
            obs, "_contour_field_cache_key", None)

    # -- Round 10/C: fragment ownership ------------------------------------
    records = fragment_ownership_records(
        chain_dets, anchor_positions, observation=obs, extraction_scale=0.0,
        t_junction_positions=(tjs_live and [(0, 1, p, 1.0) for p in tjs_live])
        or None)

    # -- the BorderOwnership detections the binder would read ---------------
    bo_dets = border_ownership_detections(records)

    # -- interior field (enclosure) ----------------------------------------
    interior = interior_field_for(obs, luminance=lum)

    # -- Session A/B: surface fragments ------------------------------------
    def enc_lookup(pos, owner):
        if interior is None:
            return None
        from perception.interior_field import ownership_from_enclosure
        return ownership_from_enclosure(interior, pos, owner)

    light = SF.LightPrior(direction=(-0.55, -0.35, 0.75), ambient=0.2)
    fragments = SF.build_surface_fragments(
        records, lum, enc_lookup, light,
        interior=(interior or {}).get("interior"),
        sigma_contour=(12.0, 20.0),
        shading_sigma=(20.0, 25.0),
        enclosure_gate=0.0,
        contour_min_step=0.01,
        # [U20 LEAD 4] the RGB plane the material payload's hue/saturation
        # slots sample (a single grep-visible input; the corpus is achromatic,
        # so hue abstains by the tree's own achromatic rule).
        rgb=getattr(scene, "rgb", None),
    )

    # -- the three ORDER cues + depth --------------------------------------
    order: Dict[str, Any] = {}
    order["occlusion"] = SF.occlusion_order_edges(fragments, records)
    tjo = SF.t_junction_order(fragments, records)
    order["t_junction_order"] = tjo
    arms_edges, arms_diag = SF.t_junction_arms(fragments, records)
    order["t_junction_arms_alpha"] = arms_edges
    order["t_junction_arms_diag"] = arms_diag

    # [U15 lead 2] Trust = rule-of-three lower bound from the merged pool
    # (dev-28 + spent-seal-67 = 95 scenes). Read from the trust table JSON
    # (rule 11), not a literal, so the recomputing test can pin it.
    import json as _json
    import os as _os
    _trust_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        "diagnostics", "_u16_trust_table.json")
    if not _os.path.exists(_trust_path):
        _trust_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
            "diagnostics", "_u15_trust_table.json")
    try:
        with open(_trust_path) as _f:
            _tt = _json.load(_f)
        TRUST: Dict[str, float] = {k: float(v["trust"]) for k, v in _tt["per_source"].items()}
    except Exception:
        TRUST = {"occ": 0.2, "tjo": 0.8909, "arms": 0.4286}
    constraints: List[Tuple[int, int, float, str]] = []
    for a, b in order["occlusion"]:
        constraints.append((int(a), int(b), 1.0, "occ"))
    for a, b in order["t_junction_order"]:
        constraints.append((int(a), int(b), 1.0, "tjo"))
    for a, b in arms_diag.get("resolved", []):
        constraints.append((int(a), int(b), 1.0, "arms"))

    from perception.depth import resolve_depth_constraints, depth_layout
    kept, removed = resolve_depth_constraints(constraints, trust=TRUST)

    # FACT floor: degenerate on dev (0 wrongs) -> floor = min effective weight
    # among kept. Edges in `kept` with eff >= floor -> FACT, else HYPOTHESIS;
    # dropped edges (removed) survive as HYPOTHESIS (tagged, not deleted).
    if kept:
        effs = [float(k[2]) * float(TRUST.get(k[3], 1.0)) for k in kept]
        floor = float(min(effs)) if effs else 0.0
    else:
        floor = 0.0
    # [U17 existence gate] |contour curvature| * spacing > 1/3 -> HYPOTHESIS-only.
    # Unmeasured (None/NaN) curvature or spacing demotes the edge (rule 1).
    from perception.surface_fragments import _chain_spacing as _cs
    _spacing = _cs(records)
    def _bar_curvature_fact_eligible(key):
        curv = None
        for _f in fragments:
            if getattr(_f, "index", None) == key[0]:
                curv = getattr(_f, "curvature", None)
                break
        if curv is None:
            return False
        try:
            c = float(curv)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(c):
            return False
        if _spacing is None:
            return False
        try:
            sp = float(_spacing)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(sp) or sp <= 0:
            return False
        return abs(c) * sp <= (1.0 / 3.0)

    kept_set = set((k[0], k[1]) for k in kept)
    edge_tier: Dict[Tuple[int, int], str] = {}
    for (a, b, w, src) in constraints:
        key = (int(a), int(b))
        if key in kept_set:
            eff = float(w) * float(TRUST.get(src, 1.0))
            if not _bar_curvature_fact_eligible(key):
                edge_tier[key] = "HYPOTHESIS"
            else:
                edge_tier[key] = "FACT" if eff >= floor else "HYPOTHESIS"
        else:
            edge_tier[key] = "HYPOTHESIS"

    order["extra_edges"] = [(k[0], k[1]) for k in kept]
    rank, _comp = depth_layout(kept)
    layers = rank
    order["depth_layers"] = layers
    order["resolver_kept"] = kept
    order["resolver_removed"] = removed
    order["resolver_floor"] = floor
    order["resolver_trust"] = dict(TRUST)
    order["edge_tier"] = edge_tier

    for f in fragments:
        f.depth_layer = layers.get(f.index, None)
        ft = None
        for (a, b), t in edge_tier.items():
            if f.index in (a, b):
                if ft is None or t == "FACT":
                    ft = t if ft is None else ("FACT" if (ft == "FACT" or t == "FACT") else ft)
        f.tier = ft

    # [B36] The Gabor field is 62 MB per scene (six 12x5x192x192 float32
    # planes).  Keeping it on every cached Run OOMs the box (1 GB total) long
    # before the corpus finishes, and nothing downstream of this point reads it
    # -- `Run.median_wavelength()` needs only `sfs`, which is 5 floats.  So the
    # retained field is slimmed to the keys a test actually reads and the
    # observation's field cache is dropped.  `keep_field=True` restores the full
    # field for a diagnostic that needs the planes.
    if not keep_field:
        fld = {"sfs": fld.get("sfs"), "scale": fld.get("scale"),
               "peaks": fld.get("peaks"),
               "curvature_signed": fld.get("curvature_signed"),
               "bend_angle": fld.get("bend_angle")}
        obs._contour_field_cache = None
        interior = None if interior is None else {
            k: v for k, v in interior.items()
            if not hasattr(v, "nbytes") or getattr(v, "nbytes", 0) < 1 << 20}

    return Run(scene_name=scene.name, luminance=lum, field=fld, peaks=peaks,
               anchor_positions=anchor_positions, peak_dets=peak_dets,
               chain_dets=chain_dets, chains=chains, records=records,
               bo_dets=bo_dets, interior=interior, fragments=fragments,
               order=order, layers=layers, tjs_live=tjs_live)


# --------------------------------------------------------------------------

def border_ownership_detections(records: Sequence[Dict[str, Any]]
                                ) -> List[Any]:
    """Reproduce lenses/v2.py::BorderOwnershipLens.step's slot
    construction (the lens itself needs derived_common, which was not
    supplied).  The signed slot is `_q_signed(signed, max_bins=8)`, the decode
    under audit is `ownership.ownership_signed`."""
    from core.quantize import _q_signed, _q_unsigned
    from core.registries import population_from_values
    from core.types import Detection, TypedSignature

    out = []
    for r in records:
        signed = float(r["signed"])
        slots = (
            float(_q_unsigned(float(r["arc"]), 1.0 / 8.0, max_bins=7)),
            float(_q_signed(signed, max_bins=8)),
            float(_q_signed(float(r["curvature"]), max_bins=8)),
            float(_q_unsigned(float(r["strength"]), 1.0 / 8.0, max_bins=7)),
        )
        pop = population_from_values(
            "BORDER_OWNERSHIP",
            (float(r["arc"]), signed, float(r["curvature"]),
             float(r["strength"])))
        rho: Tuple[Tuple[str, Any, int, str], ...] = (
            ("entity", r["alpha"], 0, "+"),
            ("vertex", int(r["vertex"]), 0, "+"),
        )
        if r.get("neighbor") is not None:
            rho += (("neighbor", r["neighbor"], 0, "+"),)
        if r.get("partner") is not None:
            rho += (("partner", r["partner"], 0, "+"),)
        out.append(Detection(k="BorderOwnership", source_layer=1, scale=0,
                             sigma=TypedSignature(
                                 kind="CANONICAL", class_="BORDER_OWNERSHIP",
                                 relation="FIGURE_SIDE_FIELD",
                                 quantized=slots, population=pop),
                             alpha=r["alpha"], z=float(r["strength"]), t=0,
                             rho=rho))
    return out


# --------------------------------------------------------------------------

def live_t_junctions(scene, anchor_positions: Dict[Any, Any],
                     chain_dets: List[Any], peaks: List[Dict[str, Any]],
                     median_wl: Optional[float]) -> List[Tuple[float, float]]:
    """A LIVE junction detector built only from what the new front end has:
    open-chain ENDPOINTS that land on another chain's INTERIOR.

    This is the plumbing candidate (c) in the issue: the same predicate
    `surface_fragments.t_junction_order` uses, but evaluated on the chain
    geometry directly (no fragments, no ownership floor), which is what makes
    it usable as an ownership-consensus mask and as a census column.
    """
    from perception.chain_polylines import member_entries, chain_is_closed

    if not median_wl or median_wl <= 0:
        return []
    tol = 0.75 * float(median_wl) * 0.5      # half a wavelength, chain scale
    open_chains: List[Tuple[Any, List[Tuple[int, Tuple[float, float]]]]] = []
    closed_members: List[Tuple[Any, Tuple[float, float]]] = []
    for det in chain_dets:
        entries = member_entries(det, anchor_positions)
        if len(entries) < 3:
            continue
        if chain_is_closed(det):
            for _i, p in entries:
                closed_members.append((det.alpha, p))
        else:
            open_chains.append((det.alpha, entries))

    sites: List[Tuple[float, float]] = []
    seen: List[Tuple[float, float]] = []
    for _anchor, entries in open_chains:
        for tip in (entries[0], entries[-1]):
            tx, ty = tip[1]
            for _ca, (mx, my) in closed_members:
                if math.hypot(mx - tx, my - ty) <= tol:
                    cand = (round(tx, 3), round(ty, 3))
                    if any(math.hypot(cand[0] - s[0], cand[1] - s[1]) < tol
                           for s in seen):
                        continue
                    seen.append(cand)
                    sites.append(cand)
                    break
    return sites


# --------------------------------------------------------------------------

def depth_from_layers(run: Run) -> Dict[Any, int]:
    """Map each fragment's alpha -> assigned depth layer (or None)."""
    out: Dict[Any, int] = {}
    for f in run.fragments:
        out[f.alpha] = run.layers.get(f.index)
    return out
