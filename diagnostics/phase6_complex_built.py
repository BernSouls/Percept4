"""[STAGE 6 / R5, Amendment 2.7] The CurveComplex link rule THROUGH THE REAL LENS.

`phase6_complex_rule.py` measured the rule offline (on the Contour detections
of a trace).  This runs the front end with GRAPH_STAGE6_RELATIONS on and
records what `CurveComplexLens._geometric_components` returns inside the real
step: the components the lens actually builds its CurveComplex detections
from.  (The lens groups one Contour per pattern node, so its units are nodes;
the offline census grouped detections.)  The three pre-registered failure
criteria are recomputed on those components.

    python diagnostics/phase6_complex_built.py [--split eval] [--workers 2] [--out NAME]
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PERCEPT_SET"] = ",".join(
    [s for s in os.environ.get("PERCEPT_SET", "").split(",")
     if s.strip() and not s.strip().startswith("GRAPH_STAGE6_RELATIONS=")]
    + ["GRAPH_STAGE6_RELATIONS=1"])
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import eth80_frontend as fe  # noqa: E402
import eth80_split as sp  # noqa: E402


def one(entry):
    import parameters as P
    import lenses.v4 as v4
    from perception import complex_links as cl
    assert bool(P.GRAPH_STAGE6_RELATIONS)
    cls, rel = entry
    seen, stats_seen = [], []
    raw = v4.CurveComplexLens.__dict__["_geometric_components"]
    fn = raw.__func__ if isinstance(raw, staticmethod) else raw
    group = cl.group

    def wrapped(node_id_to_detection, node_id_to_kind, anchor_positions, extraction_scale):
        out = fn(node_id_to_detection, node_id_to_kind, anchor_positions, extraction_scale)
        seen.append(([sorted(c) for c in out], dict(node_id_to_kind)))
        return out

    def group_wrapped(*a, **k):
        comps, st = group(*a, **k)
        stats_seen.append(dict(st))
        return comps, st

    v4.CurveComplexLens._geometric_components = staticmethod(wrapped)
    cl.group = group_wrapped
    try:
        tr = {}
        fe.build_graph(fe._load_rgb(fe.image_path(cls, rel)), name=os.path.basename(rel), trace=tr)
    finally:
        v4.CurveComplexLens._geometric_components = raw
        cl.group = group
    if not seen:
        return {"class": cls, "image": os.path.basename(rel), "calls": 0}
    comps, kind = seen[-1]
    sizes = sorted((sum(1 for n in c if kind[n] == "Contour") for c in comps), reverse=True)
    sizes = [s for s in sizes if s > 0]
    n_contours = sum(sizes)
    st = stats_seen[-1] if stats_seen else {}
    dets = [d for v in tr["tier_dets"].values() for d in v if d.k == "CurveComplex"]
    return {"class": cls, "image": os.path.basename(rel), "calls": len(seen),
            "contours": n_contours, "sizes": sizes,
            "junctions": sum(1 for k in kind.values() if k != "Contour"),
            "max_contour_degree": int(st.get("max_contour_degree", -1)),
            "stats": {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                      for k, v in st.items() if isinstance(v, (int, float, np.integer, np.floating))},
            "curvecomplex_detections": len(dets)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", default="phase6_complex_built")
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
    rows = [r for r in rows if r.get("calls")]
    n = len(rows)
    L = ["[STAGE 6 / R5] the link rule through the real lens (GRAPH_STAGE6_RELATIONS on), %d graphs "
         "(4 references + %d %s)" % (n, n - 4, a.split.upper()),
         "units: Contour pattern nodes, as the lens groups them", ""]

    def stat(name, f, fmt="%.2f"):
        v = np.asarray([f(r) for r in rows], dtype=float)
        L.append(("  %-62s median " + fmt + "   p10 " + fmt + "   p90 " + fmt + "   max " + fmt) % (
            name, np.median(v), np.percentile(v, 10), np.percentile(v, 90), v.max()))

    stat("Contour nodes entering the lens", lambda r: r["contours"], "%.0f")
    stat("complexes", lambda r: len(r["sizes"]), "%.0f")
    stat("complexes with more than one Contour", lambda r: sum(1 for s in r["sizes"] if s > 1), "%.0f")
    stat("share of Contours in a multi-member complex",
         lambda r: sum(s for s in r["sizes"] if s > 1) / max(r["contours"], 1))
    stat("largest complex, Contours", lambda r: r["sizes"][0] if r["sizes"] else 0, "%.0f")
    stat("largest complex / Contours", lambda r: (r["sizes"][0] if r["sizes"] else 0) / max(r["contours"], 1))
    stat("CurveComplex detections emitted", lambda r: r["curvecomplex_detections"], "%.0f")
    a_fail = sum(1 for r in rows if r["sizes"] and r["sizes"][0] > 0.5 * r["contours"])
    b_fail = sum(1 for r in rows if r["contours"] >= 4 and len(r["sizes"]) == 1)
    c_fail = sum(1 for r in rows if r["sizes"] and r["sizes"][0] == 1)
    bound = 20
    L += ["", "PRE-REGISTERED FAILURE CRITERIA (the rule fails if any holds in %d or more of the %d graphs)" % (bound, n),
          "  (a) one complex holds more than half of the graph's Contours:   %d graphs" % a_fail,
          "  (b) every Contour in one complex (graphs with >= 4 Contours):   %d graphs" % b_fail,
          "  (c) every complex has one member:                               %d graphs" % c_fail,
          "  VERDICT: %s" % ("FAILS" if max(a_fail, b_fail, c_fail) >= bound else "does not fail"),
          "", "  a Contour's degree in the link graph: max over all graphs %d (the rule allows 2)" % max(
              r["max_contour_degree"] for r in rows)]
    allsz = [s for r in rows for s in r["sizes"]]
    hist = {k: sum(1 for s in allsz if s == k) for k in range(1, 7)}
    L.append("  complex sizes over all graphs: " + ", ".join("%d: %d" % kv for kv in hist.items())
             + ", 7+: %d" % sum(1 for s in allsz if s >= 7))
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w") as fh:
        json.dump(rows, fh, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
