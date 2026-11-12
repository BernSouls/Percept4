"""[STAGE 6 / R5] The CurveComplex link rule, measured OFFLINE before it is
built into the lens (pre-registration, section 4).

Per image (real front end, single step, trace): the Contours and Junctions the
CurveComplex lens would see (one detection per pattern node, as the lens does),
grouped by `perception.complex_links.group`.  Reported: complex sizes and the
three pre-registered failure criteria.

    python diagnostics/phase6_complex_rule.py [--workers 2] [--out NAME]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import eth80_frontend as fe  # noqa: E402
import eth80_split as sp  # noqa: E402


def one(entry):
    import parameters as P
    from core.hashing import _detection_sort_key, _node_signature_hash
    from perception import attachment as A
    from perception import complex_links as CL
    cls, rel = entry
    tr = {}
    fe.build_graph(fe._load_rgb(fe.image_path(cls, rel)), name=os.path.basename(rel), trace=tr)
    positions = dict(tr["anchor_positions"])
    n_theta = float(P.ORIENTATION_N_THETA)
    seen = set()
    contours, junctions = [], []
    for d in sorted(tr["tier_dets"].get("Contour", []), key=_detection_sort_key):
        pat = (d.k, _node_signature_hash(d.sigma))
        if pat in seen or len(d.sigma.quantized) < 4:
            continue
        seen.add(pat)
        pts = [p for p in (A._xy(positions, a) for a in A._members(d)) if p is not None]
        theta = ((float(d.sigma.quantized[1]) + 0.5) * math.pi / n_theta) % math.pi
        ce = CL.contour_ends(len(contours), pts, theta)
        if ce is not None:
            contours.append(ce)
    for d in sorted(tr["tier_dets"].get("Junction", []), key=_detection_sort_key):
        pat = (d.k, _node_signature_hash(d.sigma))
        if pat in seen or len(d.sigma.quantized) < 3:
            continue
        seen.add(pat)
        p = A._xy(positions, d.alpha)
        if p is not None:
            junctions.append((len(junctions), p))
    comps, stats = CL.group(contours, junctions, float(tr["extraction_scale"]))
    sizes = sorted((len(c) for c, _j in comps), reverse=True)
    return {"class": cls, "image": os.path.basename(rel), "contours": len(contours),
            "junctions": len(junctions), "sizes": sizes, "stats": stats,
            "with_junction": sum(1 for _c, j in comps if j)}


def _pct(v, q):
    return float(np.percentile(np.asarray(v, dtype=float), q)) if len(v) else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="phase6_complex_rule")
    ap.add_argument("--split", default="eval")
    a = ap.parse_args(argv)
    split = sp.load_split()
    entries = list(split["references"].items()) + [
        (c, p) for c, ps in split[a.split].items() for p in ps]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(one, entries, chunksize=1)
    else:
        rows = [one(e) for e in entries]
    from perception import complex_links as CL
    L = ["[STAGE 6 / R5] the CurveComplex link rule, measured offline, %d graphs" % len(rows),
         "gap <= %.1f scales, bend <= %.0f degrees, each end linked at most once" % (
             CL.GAP_MAX, math.degrees(CL.BEND_MAX)), ""]

    def stat(name, f, fmt="%.2f"):
        v = [f(r) for r in rows]
        L.append(("  %-58s median " + fmt + "   p10 " + fmt + "   min " + fmt + "   max " + fmt) % (
            name, _pct(v, 50), _pct(v, 10), min(v), max(v)))

    stat("Contours (one per node, as the lens sees them)", lambda r: r["contours"], "%.0f")
    stat("Junctions", lambda r: r["junctions"], "%.0f")
    stat("link candidates (gap and bend within bounds)", lambda r: r["stats"]["candidates"], "%.0f")
    stat("links accepted", lambda r: r["stats"]["links"], "%.0f")
    stat("largest contour-to-contour degree of any Contour", lambda r: r["stats"]["max_contour_degree"], "%.0f")
    stat("contour ends held by a junction", lambda r: r["stats"]["ends_on_junctions"], "%.0f")
    stat("complexes", lambda r: len(r["sizes"]), "%.0f")
    stat("complexes with more than one Contour", lambda r: sum(1 for s in r["sizes"] if s > 1), "%.0f")
    stat("share of Contours in a multi-member complex",
         lambda r: sum(s for s in r["sizes"] if s > 1) / max(1, r["contours"]))
    stat("largest complex, Contours", lambda r: r["sizes"][0] if r["sizes"] else 0, "%.0f")
    stat("largest complex / Contours", lambda r: (r["sizes"][0] if r["sizes"] else 0) / max(1, r["contours"]))
    allsizes = [s for r in rows for s in r["sizes"]]
    hist = {}
    for s in allsizes:
        hist[min(s, 8)] = hist.get(min(s, 8), 0) + 1
    L += ["", "complex sizes, all graphs pooled (Contours per complex; 8 = 8 or more): "
          + ", ".join("%d: %d" % kv for kv in sorted(hist.items()))]
    a_fail = sum(1 for r in rows if r["sizes"] and r["sizes"][0] > 0.5 * r["contours"])
    b_fail = sum(1 for r in rows if r["contours"] >= 4 and len(r["sizes"]) == 1)
    c_fail = sum(1 for r in rows if r["sizes"] and r["sizes"][0] == 1)
    L += ["", "PRE-REGISTERED FAILURE CRITERIA (the rule fails if any holds in 20 or more of the %d graphs)" % len(rows),
          "  (a) one complex holds more than half of the graph's Contours:   %d graphs" % a_fail,
          "  (b) every Contour in one complex (graphs with >= 4 Contours):   %d graphs" % b_fail,
          "  (c) every complex has one member:                               %d graphs" % c_fail,
          "  VERDICT: %s" % ("FAILS" if max(a_fail, b_fail, c_fail) >= 20 else "does not fail")]
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w") as fh:
        json.dump(rows, fh, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
