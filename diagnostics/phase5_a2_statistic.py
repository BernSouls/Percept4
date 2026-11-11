"""PHASE 5 / A2 -- the pre-registered CurveComplex statistic (read-only).

Per graph: how many CurveComplex nodes there are, how many of them hold more
than one Contour, and the share of the graph's Contour nodes held by its
LARGEST complex (membership read from the CURVE_CONTAINMENT edges).

The pre-registration (results/phase5_preregistration.md) names the failure
of the A2 lens fix as: "the median graph has one complex holding more than
half of its Contour nodes".

    python diagnostics/phase5_a2_statistic.py [--out NAME]

Run in the committed tree it describes the lens as it is.  Run in a tree
with results/phase5_a2_trial.patch applied (and the cache rebuilt) it
evaluates the fix that was NOT kept.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import os
import statistics
from collections import defaultdict

from _bootstrap import RESULTS_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", default="phase5_a2_statistic")
    args = ap.parse_args()

    import eth80_frontend as fe
    import eth80_split as sp
    from phase5_feasibility import _pct
    from phase5_graph_census import entries_of

    rows = []
    for cls, rel in entries_of(sp.load_split(), args.split):
        g = fe.graph_for(cls, rel)
        lens = {n: lt for n, (lt, _s) in g.sigs.items()}
        contours = {n for n, lt in lens.items() if lt == "Contour"}
        members = defaultdict(set)
        for e in g.subgraph.edges:
            if (e[2].name == "CURVE_CONTAINMENT"
                    and lens.get(e[0]) == "CurveComplex" and e[1] in contours):
                members[e[0]].add(e[1])
        largest = max((len(v) for v in members.values()), default=0)
        rows.append({
            "contours": len(contours),
            "complexes": sum(1 for lt in lens.values() if lt == "CurveComplex"),
            "multi": sum(1 for v in members.values() if len(v) > 1),
            "share": largest / max(1, len(contours)),
        })

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s, flush=True)

    def col(name, key):
        xs = [r[key] for r in rows]
        emit("  %-52s median %6.3f   p10 %6.3f   min %6.3f   max %6.3f"
             % (name, statistics.median(xs), _pct(xs, 0.10), min(xs), max(xs)))

    emit("PHASE 5 / A2 -- CurveComplex membership, %d %s graphs, front-end digest %s"
         % (len(rows), args.split.upper(), fe.frontend_source_digest()[:12]))
    emit()
    col("Contour nodes", "contours")
    col("CurveComplex nodes", "complexes")
    col("complexes with more than one Contour", "multi")
    col("largest complex / Contour nodes", "share")
    emit()
    over = sum(1 for r in rows if r["share"] > 0.5)
    full = sum(1 for r in rows if r["share"] >= 1.0)
    emit("graphs whose largest complex holds more than half of the Contour nodes: %d of %d"
         % (over, len(rows)))
    emit("graphs whose largest complex holds every Contour node: %d of %d"
         % (full, len(rows)))
    med = statistics.median(r["share"] for r in rows)
    emit()
    n_multi = sum(1 for r in rows if r["multi"] > 0)
    emit("PRE-REGISTERED A2 EXPECTED EFFECT (complexes with more than one member appear): %s (in %d of %d graphs)"
         % (n_multi > 0, n_multi, len(rows)))
    emit("PRE-REGISTERED A2 FAILURE (median graph: one complex holds > 1/2 of the Contour nodes): %s (median %.3f)"
         % (med > 0.5, med))
    with open(os.path.join(RESULTS_DIR, args.out + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
