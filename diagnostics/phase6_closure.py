"""[STAGE 6 / 0b] CLOSURE: why is every chain of the 79 graphs open?

Three candidate causes (the brief's): (A) the contour extraction does not
deliver a closed boundary, (B) the chain former breaks rings, (C) there is no
closed boundary in the images.  This script measures each, with no scoring:

  C. ground truth per image: the object's silhouette (pixels far from the
     backdrop colour, largest component, holes filled) is ONE closed curve.
  A. how much of that closed curve the peak field can see at all: the share of
     the silhouette perimeter inside the peak-extraction interior (the hard
     border discard of perception/contour_field.py), and of that visible part
     the share with a peak within one chain spacing.
  B. the chain former on the peaks that exist: cycles in the geometric
     adjacency (cycle rank per component), what the junction split cuts, and
     every fundamental cycle the ring test could consider with the reason it
     is accepted or rejected.

    python diagnostics/phase6_closure.py [--out NAME] [--figground] [--reuse]
    PERCEPT_SET=FIELD_BORDER_DISCARD_PER_CHANNEL=1 python diagnostics/phase6_closure.py \
        --figground --out phase6_closure_per_channel
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import eth80_frontend as fe  # noqa: E402
import eth80_split as sp  # noqa: E402
import perception.chain_formation as cf  # noqa: E402
import perception.contour_field as ctf  # noqa: E402
from lenses.chain_lens import ChainLens  # noqa: E402
from lenses.derived_common import endstopped_map_for  # noqa: E402
from lenses.peak_lens import field_for_observation  # noqa: E402
from perception.observation_2d import Observation2D  # noqa: E402


def silhouette(img: np.ndarray):
    """Largest blob of pixels far from the border's median colour, holes filled."""
    a = img.astype(float)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    m = np.sqrt(((a - bg) ** 2).sum(axis=2)) > 60.0
    m = ndimage.binary_opening(m, iterations=1)
    lab, k = ndimage.label(m)
    if k == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, k + 1))
    return ndimage.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))


def perimeter_pixels(mask: np.ndarray):
    edge = mask & ~ndimage.binary_erosion(mask)
    ys, xs = np.nonzero(edge)
    return xs.astype(float), ys.astype(float)


def _pct(v, q):
    return float(np.percentile(np.asarray(v, dtype=float), q)) if len(v) else float("nan")


def cycle_verdict(path, ck, n, xl, yl):
    """Why `_total_turning_closes` accepts or rejects one fundamental cycle
    (the same tests in the same order, with the reason returned)."""
    m = len(path)
    if m < 3:
        return "shorter_than_3"
    for k in range(m):
        a, b = path[k], path[(k + 1) % m]
        key = a * n + b if a < b else b * n + a
        if key not in ck:
            return "cycle_edge_not_chainable"
    pts = [(xl[v], yl[v]) for v in path]
    area2 = 0.0
    for i in range(m):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % m]
        area2 += x0 * y1 - x1 * y0
    if abs(area2) <= 0.0:
        return "zero_area"
    total, sharpest = 0.0, 0.0
    for i in range(m):
        a, b, c = pts[(i - 1) % m], pts[i], pts[(i + 1) % m]
        v1x, v1y = b[0] - a[0], b[1] - a[1]
        v2x, v2y = c[0] - b[0], c[1] - b[1]
        turn = math.atan2(v1x * v2y - v1y * v2x, v1x * v2x + v1y * v2y)
        total += turn
        sharpest = max(sharpest, abs(turn))
    if sharpest > 2.0 * math.pi / 3.0 + 1e-9:
        return "turn_sharper_than_120deg"
    if abs(abs(total) - 2.0 * math.pi) > cf._RING_TURN_TOLERANCE:
        return "total_turning_not_2pi"
    return "closes"


def cycle_turning(path, xl, yl):
    """(sharpest turn in degrees, vertices turning more than 120 degrees,
    |total turning| / 2 pi) of a cycle walked in path order."""
    m = len(path)
    if m < 3:
        return (0.0, 0, 0.0)
    pts = [(xl[v], yl[v]) for v in path]
    total, sharpest, n_sharp = 0.0, 0.0, 0
    for i in range(m):
        a, b, c = pts[(i - 1) % m], pts[i], pts[(i + 1) % m]
        v1x, v1y = b[0] - a[0], b[1] - a[1]
        v2x, v2y = c[0] - b[0], c[1] - b[1]
        turn = math.atan2(v1x * v2y - v1y * v2x, v1x * v2x + v1y * v2y)
        total += turn
        sharpest = max(sharpest, abs(turn))
        n_sharp += int(abs(turn) > 2.0 * math.pi / 3.0 + 1e-9)
    return (round(math.degrees(sharpest), 1), n_sharp,
            round(abs(total) / (2.0 * math.pi), 3))


def fundamental_cycles(g, tree, cut_keys):
    """Every fundamental cycle the ring test could consider, with its verdict.

    Mirrors `_component_paths` -> `_principal_cycle`: per component of the
    junction-split forest, each geometric link that is not a tree edge and not
    a cut, plus the tree path between its ends."""
    n, xl, yl = g["n"], g["xl"], g["yl"]
    ck = set(g.get("geo_keys", ())) - cut_keys
    comps = cf._components_of(tree, n, keep_isolated=True)
    out = []
    for root in sorted(comps):
        nodes = comps[root]
        if len(nodes) < 3:
            continue
        member = set(nodes)
        adj = {v: [w for w in tree[v] if w in member] for v in nodes}
        tree_edges = {v * n + w for v in nodes for w in adj[v] if v < w}
        for key in sorted(ck):
            a, b = divmod(key, n)
            if a >= b or a not in member or b not in member or key in tree_edges:
                continue
            path = cf._tree_path(a, b, adj)
            if path is None:
                out.append((0, "no_tree_path", 0.0, 0, 0.0))
                continue
            out.append((len(path), cycle_verdict(path, ck, n, xl, yl))
                       + cycle_turning(path, xl, yl))
    return out


def chain_former_census(obs):
    field = field_for_observation(obs)
    peaks = list(field["peaks"])
    out = {"peaks": len(peaks)}
    if not peaks:
        return out, peaks, field
    wl = ChainLens._median_wavelength(field)
    sfs = np.asarray(field["sfs"], dtype=np.float64)
    support = field.get("integration_driving")
    support = field["integration"] if support is None else support
    orient = cf.continuous_orientations(
        np.asarray(field["complex"], dtype=np.float64), peaks, sfs=sfs)
    emap = endstopped_map_for(obs)
    g = cf._graph_components(peaks, emap, wl, 2.0, cf.CHAIN_ENDSTOP_MAX, orient,
                             np.asarray(support), sfs, cf.LINK_SUPPORT_MIN)
    n = g["n"]
    geo = set(g.get("geo_keys", ()))
    links = sorted({(min(k // n, k % n), max(k // n, k % n)) for k in geo})
    comp_of = {}
    for cid, members in g["comps"].items():
        for i in members:
            comp_of[i] = cid
    n_links = {}
    for i, j in links:
        if comp_of.get(i) == comp_of.get(j):
            n_links[comp_of[i]] = n_links.get(comp_of[i], 0) + 1
    rank = {cid: n_links.get(cid, 0) - (len(m) - 1) for cid, m in g["comps"].items()}
    cyclic = [cid for cid, r in rank.items() if r > 0]
    tree, _keys, cuts, cut_keys = cf._apply_junction_split(g)
    chains = cf.form_chains(peaks, emap, wl, integration=np.asarray(support),
                            sfs=sfs, orientations=orient)
    cycles = fundamental_cycles(g, tree, cut_keys)
    comp_after = {}
    for root, members in cf._components_of(tree, n, keep_isolated=True).items():
        for i in members:
            comp_after[i] = root
    severed = sum(1 for i, j in links if comp_after.get(i) != comp_after.get(j))
    deg = {}
    for i, j in links:
        deg[i] = deg.get(i, 0) + 1
        deg[j] = deg.get(j, 0) + 1
    out.update({
        "wavelength": float(wl), "spacing": float(g["spacing"]),
        "max_gap": float(g["max_gap"]),
        "geo_links": len(links), "components": len(g["comps"]),
        "largest_component": max((len(m) for m in g["comps"].values()), default=0),
        "cyclic_components": len(cyclic),
        "cycle_rank_total": int(sum(r for r in rank.values() if r > 0)),
        "cyclic_component_sizes": sorted((len(g["comps"][c]) for c in cyclic), reverse=True)[:6],
        "junction_cuts": len(cuts), "cut_links": len(cut_keys) // 2,
        "degree3plus_peaks": sum(1 for v in deg.values() if v >= 3),
        "chains": len(chains), "rings": sum(1 for _c, r in chains if r),
        "longest_chain": max((len(c) for c, _r in chains), default=0),
        "fundamental_cycles": [[int(c[0]), c[1]] for c in cycles],
        "cycle_turning": [[int(c[0]), c[1], c[2], c[3], c[4]] for c in cycles],
        "links_between_trees_after_split": int(severed),
    })
    return out, peaks, field


def eth_row(cls, rel):
    img = fe._load_rgb(fe.image_path(cls, rel))
    obs = Observation2D.from_image(img, rich=False)
    row, peaks, _field = chain_former_census(obs)
    H, W = img.shape[:2]
    mask = silhouette(img)
    px, py = perimeter_pixels(mask)
    row["silhouette_perimeter_px"] = int(len(px))
    row["silhouette_touches_border"] = bool(
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    xs = ys = None
    if peaks:
        xs = np.array([p["x"] for p in peaks], float)
        ys = np.array([p["y"] for p in peaks], float)
        row["peak_x_range"] = [float(xs.min()), float(xs.max())]
        row["peak_y_range"] = [float(ys.min()), float(ys.max())]
    bank = ctf._get_bank(ctf.front_end_scale(H, W))
    m = ctf._interior_margin(ctf._border_margin_px(bank), H, W)
    row["single_margin_px"] = m
    if ctf._per_channel_discard():
        row["channel_margins_px"] = [int(v) for v in ctf._channel_margins(bank, H, W)]
        m = min(row["channel_margins_px"])
    row["extraction_margin_px"] = m
    if peaks:
        d_border = np.minimum(np.minimum(xs, W - 1 - xs), np.minimum(ys, H - 1 - ys))
        row["peaks_in_old_band"] = int((d_border < row["single_margin_px"]).sum())
        by = {}
        for p_ in peaks:
            by[int(p_["sf_idx"])] = by.get(int(p_["sf_idx"]), 0) + 1
        row["peaks_by_channel"] = by
    if len(px):
        inside = (px >= m) & (px < W - m) & (py >= m) & (py < H - m)
        row["perimeter_inside_interior"] = float(inside.mean())
        if peaks and inside.any():
            d = np.sqrt((px[inside, None] - xs[None, :]) ** 2
                        + (py[inside, None] - ys[None, :]) ** 2).min(axis=1)
            row["visible_perimeter_with_peak_within_spacing"] = float(
                (d <= row["spacing"]).mean())
            row["visible_perimeter_with_peak_within_max_gap"] = float(
                (d <= row["max_gap"]).mean())
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase6_closure")
    ap.add_argument("--figground", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the ETH-80 rows of an earlier run (the JSON)")
    a = ap.parse_args(argv)
    split = sp.load_split()
    entries = list(split["references"].items()) + [
        (c, p) for c, ps in split["eval"].items() for p in ps]
    rows_path = os.path.join(RESULTS_DIR, a.out + ".json")
    rows = []
    if a.reuse and os.path.isfile(rows_path):
        with open(rows_path) as fh:
            rows = json.load(fh)["eth80"]
    else:
        for cls, rel in entries:
            r = eth_row(cls, rel)
            r["class"], r["image"] = cls, os.path.basename(rel)
            rows.append(r)
        with open(rows_path, "w") as fh:
            json.dump({"eth80": rows, "figground": []}, fh, indent=1, sort_keys=True)
    lines = ["[STAGE 6 / 0b] closure census, %d graphs (4 references + 75 EVAL)" % len(rows), ""]

    def stat(key, fmt="%.2f"):
        v = [r[key] for r in rows if r.get(key) is not None]
        return ("median " + fmt + "  p10 " + fmt + "  min " + fmt + "  max " + fmt) % (
            _pct(v, 50), _pct(v, 10), min(v), max(v))

    lines.append("C. ground truth: one closed silhouette per image")
    lines.append("   silhouette touches the image border in %d of %d images" % (
        sum(1 for r in rows if r["silhouette_touches_border"]), len(rows)))
    lines.append("   silhouette perimeter (px): " + stat("silhouette_perimeter_px", "%.0f"))
    lines.append("")
    lines.append("A. what the peak field can see")
    lines.append("   per-channel discard: %s" % ("ON, channel margins (px) %s" % rows[0].get(
        "channel_margins_px") if rows[0].get("channel_margins_px") else "OFF (one margin for every channel)"))
    lines.append("   extraction margin (px): %s" % sorted({r["extraction_margin_px"] for r in rows}))
    tot = {}
    for r in rows:
        for k, v in (r.get("peaks_by_channel") or {}).items():
            tot[int(k)] = tot.get(int(k), 0) + v
    n_all = sum(tot.values()) or 1
    lines.append("   peaks by dominant channel: %s" % ", ".join(
        "ch%d %d (%.0f %%)" % (k, v, 100.0 * v / n_all) for k, v in sorted(tot.items())))
    lines.append("   peaks inside the band the single margin discards (within %s px of the frame): %d of %d" % (
        rows[0].get("single_margin_px"), sum(r.get("peaks_in_old_band", 0) for r in rows), n_all))
    lines.append("   share of the silhouette perimeter inside the peak interior: "
                 + stat("perimeter_inside_interior"))
    lines.append("   images whose WHOLE silhouette is inside the interior: %d of %d" % (
        sum(1 for r in rows if r.get("perimeter_inside_interior", 0) >= 0.999), len(rows)))
    lines.append("   of the visible perimeter, share with a peak within one chain spacing: "
                 + stat("visible_perimeter_with_peak_within_spacing"))
    lines.append("   ... within the chain former's max gap: "
                 + stat("visible_perimeter_with_peak_within_max_gap"))
    lines.append("")
    lines.append("B. the chain former on the peaks that exist")
    for key, fmt in (("peaks", "%.0f"), ("geo_links", "%.0f"), ("components", "%.0f"),
                     ("largest_component", "%.0f"), ("cyclic_components", "%.0f"),
                     ("cycle_rank_total", "%.0f"), ("degree3plus_peaks", "%.0f"),
                     ("junction_cuts", "%.0f"), ("cut_links", "%.0f"),
                     ("chains", "%.0f"), ("longest_chain", "%.0f"), ("rings", "%.0f")):
        lines.append("   %-22s %s" % (key, stat(key, fmt)))
    lines.append("   graphs with at least one cycle in the geometric adjacency: %d of %d" % (
        sum(1 for r in rows if r.get("cyclic_components", 0) > 0), len(rows)))
    lines.append("   total cyclic components %d; their sizes (peaks): %s" % (
        sum(r.get("cyclic_components", 0) for r in rows),
        sorted((s for r in rows for s in r.get("cyclic_component_sizes", [])), reverse=True)[:20]))
    lines.append("   total rings %d of %d chains, in %d graphs" % (
        sum(r.get("rings", 0) for r in rows), sum(r.get("chains", 0) for r in rows),
        sum(1 for r in rows if r.get("rings", 0) > 0)))
    lines += ["", "B2. every fundamental cycle the ring test could consider, and its verdict",
              "    (a geometric link that is not a tree edge and not a junction cut, plus the",
              "     tree path between its ends; the tests of `_total_turning_closes`, in order)"]
    allc = [c for r in rows for c in r.get("fundamental_cycles", [])]
    for lo, hi, name in ((3, 3, "3 peaks"), (4, 6, "4-6 peaks"), (7, 11, "7-11 peaks"),
                         (12, 10 ** 6, ">= 12 peaks")):
        sub = [v for L, v in allc if lo <= L <= hi]
        cnt = {}
        for v in sub:
            cnt[v] = cnt.get(v, 0) + 1
        lines.append("    %-12s %5d cycles: %s" % (name, len(sub), ", ".join(
            "%s %d" % kv for kv in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])))))
    big = [c for r in rows for c in r.get("cycle_turning", []) if c[0] >= 12]
    if big:
        tol = cf._RING_TURN_TOLERANCE / (2.0 * math.pi)
        one_sharp = [c for c in big if c[3] == 1]
        near_2pi = [c for c in big if abs(c[4] - 1.0) <= tol]
        both = [c for c in big if c[3] <= 2 and abs(c[4] - 1.0) <= tol]
        lines.append("    the %d cycles of >= 12 peaks: sharpest turn median %.0f deg (min %.0f); vertices turning"
                     % (len(big), _pct([c[2] for c in big], 50), min(c[2] for c in big)))
        lines.append("      more than 120 deg per cycle: median %.0f (min %d, max %d); cycles with exactly one such"
                     % (_pct([c[3] for c in big], 50), min(c[3] for c in big), max(c[3] for c in big)))
        lines.append("      vertex: %d; cycles whose total turning is 2 pi within the tolerance: %d; cycles with at"
                     % (len(one_sharp), len(near_2pi)))
        lines.append("      most two such vertices AND a 2 pi total: %d" % len(both))
    lines.append("    geometric links whose ends the junction split left in different trees: "
                 + stat("links_between_trees_after_split", "%.0f"))
    whole = [r for r in rows if r.get("perimeter_inside_interior", 0) >= 0.999]
    need = [r["silhouette_perimeter_px"] / r["spacing"] for r in whole]
    if whole:
        lines.append("    the %d images whose whole silhouette is inside the peak interior:" % len(whole))
        lines.append("      peaks a closed silhouette chain would need (perimeter / spacing): median %.0f;"
                     " largest component median %.0f; longest chain median %.0f" % (
                         _pct(need, 50), _pct([r["largest_component"] for r in whole], 50),
                         _pct([r["longest_chain"] for r in whole], 50)))
        lines.append("      with a ring of >= 12 peaks: %d; with any cycle of at least half the needed length: %d" % (
            sum(1 for r in whole if any(c[1] == "closes" and c[0] >= 12 for c in r["cycle_turning"])),
            sum(1 for r, n_ in zip(whole, need) if any(c[0] >= 0.5 * n_ for c in r["cycle_turning"]))))
    fig = []
    if a.figground:
        sys.path.insert(0, os.path.dirname(HERE))
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests_figground"))
        from tests_figground.scenes import build_all
        lines += ["", "D. the synthetic figground scenes (same census; image size in px)"]
        scenes = build_all()
        scenes = list(scenes.items()) if isinstance(scenes, dict) else [(s.name, s) for s in scenes]
        for sname, sc in scenes:
            img = np.clip(np.asarray(sc.rgb, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            obs = Observation2D.from_image(img, rich=False)
            row, _peaks, _field = chain_former_census(obs)
            H, W = img.shape[:2]
            bank = ctf._get_bank(ctf.front_end_scale(H, W))
            m = ctf._interior_margin(ctf._border_margin_px(bank), H, W)
            row.update({"scene": sname, "size": [H, W], "extraction_margin_px": m})
            fig.append(row)
            lines.append("   %-24s %dx%d margin %s peaks %3d links %3d cyclic comps %d (sizes %s) "
                         "cuts %d chains %2d rings %d longest %d; cycles %s" % (
                             sname, H, W, m, row["peaks"], row.get("geo_links", 0),
                             row.get("cyclic_components", 0), row.get("cyclic_component_sizes", []),
                             row.get("junction_cuts", 0), row.get("chains", 0),
                             row.get("rings", 0), row.get("longest_chain", 0),
                             sorted(row.get("cycle_turning", []), reverse=True)[:2]))
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(rows_path, "w") as fh:
        json.dump({"eth80": rows, "figground": fig}, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
