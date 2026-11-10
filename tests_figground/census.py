"""U11.0 CENSUS -- per true junction, every column the session asks for.

Writes ``diagnostics/_u11_census.json`` and prints the table.  Run with:

    PYTHONPATH=src:.pylibs:. python3 tests/figground/census.py

Columns (one row per renderer-true junction, per scene):
  1 chains_separate   are the stem's and the bar's peaks in different chains
                      after the U4 ring-safe junction split?
  2 bar_closed        is the occluder's contour held as ONE closed ring?
  3 tjo_fires         does surface_fragments.t_junction_order emit an edge here?
  4 tjo_direction     is that edge (near, far) == renderer truth?
  5 arms_site         did t_junction_arms find a cluster here?
  6 arms_through      did it find a through-pair (not a corner)?
  7 arms_edge         alpha edge emitted?
  8 arms_resolved     resolved to index pair, or dropped (stems_dropped)?
  9 occ_edge          does occlusion_order_edges emit (bar, stem) here?
 10 occ_direction     correct?
 11 depth_rank        does the assigned layer rank the bar nearer?
 12 enc_bar/stem      enclosure magnitude on each side's fragments
 13 contrast_bar/stem edge-contrast asymmetry (H6)
 14 get_t_junctions   does the observation expose the live detector? (H5)
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from scenes import build_all, Scene                      # noqa: E402
from pipeline import run_pipeline, Run                    # noqa: E402


def _frag_at(run: Run, x: float, y: float, radius: float) -> List[Any]:
    return run.fragments_near(x, y, radius)


def _rec_at(run: Run, x: float, y: float, radius: float) -> List[Dict[str, Any]]:
    return run.records_near(x, y, radius)


def _chains_of(run: Run, x: float, y: float, radius: float) -> List[Any]:
    return sorted({r["chain"] for r in _rec_at(run, x, y, radius)})


def edge_contrast(lum: np.ndarray, x: float, y: float,
                  radius: float = 3.0) -> float:
    """Local edge contrast: max |gradient| in a disk around (x, y).

    The H6 observable.  A hazier / defocused contour has a strictly lower
    peak gradient magnitude than a sharp one at the same albedo step.
    """
    h, w = lum.shape
    xi, yi = int(round(x)), int(round(y))
    r = int(math.ceil(radius))
    x0, x1 = max(1, xi - r), min(w - 1, xi + r + 1)
    y0, y1 = max(1, yi - r), min(h - 1, yi + r + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    patch = lum[y0:y1, x0:x1].astype(np.float64)
    gy, gx = np.gradient(patch)
    mag = np.hypot(gx, gy)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = ((xx + 0.0 - x) ** 2 + (yy + 0.0 - y) ** 2) <= radius * radius
    if not inside.any():
        return float(mag.max())
    return float(mag[inside].max())


def _layer_rank(run: Run, alpha: Any) -> Optional[int]:
    for f in run.fragments:
        if f.alpha == alpha:
            return run.layers.get(f.index)
    return None


def _edge_here(edges: List[Tuple[int, int]], run: Run,
               x: float, y: float, radius: float) -> List[Tuple[int, int]]:
    """Edges whose NEARER endpoint sits within `radius` of the junction."""
    out = []
    for a, b in edges:
        fa = fb = None
        for f in run.fragments:
            if f.index == a:
                fa = f
            if f.index == b:
                fb = f
        if fa is None:
            continue
        if math.hypot(fa.position[0] - x, fa.position[1] - y) <= radius:
            out.append((a, b))
    return out


def census_scene(scene: Scene, run: Run, radius: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = {f.alpha: f for f in run.fragments}
    recs = run.record_by_alpha()
    occ = run.order.get("occlusion", [])
    tjo = run.order.get("t_junction_order", [])
    arms_alpha = run.order.get("t_junction_arms_alpha", [])
    arms_diag = run.order.get("t_junction_arms_diag", {})
    resolved = set(map(tuple, arms_diag.get("resolved", [])))

    for j in scene.junctions:
        x, y = float(j["x"]), float(j["y"])
        near_recs = [r for r in _rec_at(run, x, y, radius)]
        near_chains = sorted({r["chain"] for r in near_recs})
        # the near/far side by truth object mask, at a small offset either side
        bar_frags = _frag_at(run, x, y, radius)
        row: Dict[str, Any] = dict(
            scene=scene.name, junction=j["id"], x=round(x, 2), y=round(y, 2),
            near=j["near"], far=j["far"],
            records_in_radius=len(near_recs),
            fragments_in_radius=len(bar_frags),
            chains_at_junction=len(near_chains),
            chains_separate=len(near_chains) >= 2,
            bar_closed=any(bool(r.get("closed")) for r in near_recs),
            rings_in_scene=sum(1 for _m, ring in run.chains if ring),
            tjo_edges_here=len(_edge_here(tjo, run, x, y, radius)),
            tjo_fires=bool(_edge_here(tjo, run, x, y, radius)),
            occ_edges_here=len(_edge_here(occ, run, x, y, radius)),
            occ_fires=bool(_edge_here(occ, run, x, y, radius)),
            arms_sites=arms_diag.get("sites", 0),
            arms_through_pairs=arms_diag.get("through_pairs", 0),
            arms_corners=arms_diag.get("corners", 0),
            arms_edges=len(arms_alpha),
            arms_resolved=len(resolved),
            arms_stems_dropped=arms_diag.get("stems_dropped", 0),
            depth_layers_assigned=len(run.layers),
            get_t_junctions=hasattr(scene, "_tjs_probe"),
        )
        # enclosure + contrast on each side of the junction
        enc_near = [abs(float(r.get("enclosure", 0.0))) for r in near_recs]
        row["enclosure_max"] = round(max(enc_near), 4) if enc_near else None
        row["enclosure_median"] = (round(float(np.median(enc_near)), 4)
                                   if enc_near else None)
        row["edge_contrast"] = round(edge_contrast(run.luminance, x, y,
                                                   radius), 5)
        row["signed_abs_median"] = (
            round(float(np.median([abs(float(r["signed"])) for r in near_recs])), 4)
            if near_recs else None)
        rows.append(row)

    # H6 scene-level: contrast on the NEAR contour vs the FAR contour, measured
    # away from the junction itself (each object's own boundary).
    return rows


def h6_contrast(scene: Scene, run: Run) -> Dict[str, Any]:
    """Bar-vs-stem edge contrast, measured on each object's OWN boundary away
    from the junction, which is the H6 observable (the far contour should read
    hazier when the renderer applies haze/defocus)."""
    from perception.chain_flanks import planes_from_rgb

    out: Dict[str, Any] = {}
    lum = run.luminance
    per_object: Dict[str, List[float]] = {}
    for name, mask in scene.masks.items():
        if name == "background":
            continue
        # boundary pixels of this object that are NOT inside a nearer object
        nearer = np.zeros_like(mask)
        for other, om in scene.masks.items():
            if other == name or scene.depth.get(other, 9) >= scene.depth.get(name, 9):
                continue
            nearer |= om
        own = mask & ~nearer
        ys, xs = np.nonzero(own)
        if len(xs) < 8:
            continue
        # keep only true boundary pixels (a 4-neighbour of the object's outside)
        vals = []
        step = max(1, len(xs) // 200)
        for k in range(0, len(xs), step):
            xi, yi = int(xs[k]), int(ys[k])
            vals.append(edge_contrast(lum, float(xi), float(yi), 2.5))
        per_object[name] = [round(float(np.median(vals)), 5),
                            round(float(np.percentile(vals, 90)), 5),
                            len(vals)]
    out["per_object_median_p90_n"] = per_object
    if "near_box" in per_object and "far_box" in per_object:
        out["near_minus_far_median"] = round(
            per_object["near_box"][0] - per_object["far_box"][0], 5)
    if "cylinder" in per_object and "box" in per_object:
        out["near_minus_far_median"] = round(
            per_object["cylinder"][0] - per_object["box"][0], 5)
    return out


def main() -> int:
    scenes = build_all()
    all_rows: Dict[str, Any] = {}
    for name, scene in scenes.items():
        if not scene.junctions:
            continue
        for wired in (False, True):
            run = run_pipeline(scene, with_live_junctions=wired)
            median_wl = run.median_wavelength() or 6.0
            radius = max(3.0, 0.9 * median_wl)
            rows = census_scene(scene, run, radius)
            key = "%s%s" % (name, "_WIRED" if wired else "")
            all_rows[key] = dict(
                rows=rows,
                h6=h6_contrast(scene, run),
                totals=dict(
                    peaks=len(run.peaks),
                    chains=len(run.chain_dets),
                    rings=sum(1 for _m, r in run.chains if r),
                    records=len(run.records),
                    fragments=len(run.fragments),
                    occ_edges=len(run.order.get("occlusion", [])),
                    tjo_edges=len(run.order.get("t_junction_order", [])),
                    arms_edges=len(run.order.get("t_junction_arms_alpha", [])),
                    arms_diag={k: v for k, v in
                               run.order.get("t_junction_arms_diag", {}).items()
                               if k != "resolved"},
                    depth_layers=len(run.layers),
                    median_wavelength=round(float(median_wl), 4),
                    live_tj_sites=len(run.tjs_live),
                ),
            )
            print("== %-32s peaks=%d chains=%d rings=%d records=%d frags=%d "
                  "occ=%d tjo=%d arms=%d layers=%d liveTJ=%d" % (
                      key, len(run.peaks), len(run.chain_dets),
                      all_rows[key]["totals"]["rings"], len(run.records),
                      len(run.fragments),
                      len(run.order.get("occlusion", [])),
                      len(run.order.get("t_junction_order", [])),
                      len(run.order.get("t_junction_arms_alpha", [])),
                      len(run.layers), len(run.tjs_live)))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "diagnostics", "_u11_census.json")
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, indent=2, sort_keys=True)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
