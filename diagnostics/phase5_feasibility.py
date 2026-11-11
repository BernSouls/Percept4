"""PHASE 5 / B -- can relations carry a correspondence at all?  (static)

No matcher runs.  Per graph, on the FRAME / ATTRIBUTE partition of
matching/relation_frame.py:

  (1) FRAME node count
  (2) FRAME nodes with >= 1 and >= 2 structural edges
  (3) the largest connected component over structural edges
  (4) the share of structural edges with a measured graded axis
  (5) the REACHABLE set: the largest connected component over structural
      edges that carry a measured graded axis -- the most the growth rule
      can ever admit from its best seed, since admission needs a structural
      relation that agrees on a measured axis.

The definitions and the stop rule were fixed in
results/phase5_preregistration.md before this script was run:
STOP if median (3) < 8, or median (5)/(1) < 0.25.

    python diagnostics/phase5_feasibility.py [--split eval] [--out NAME]
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import os
import statistics
from collections import Counter, defaultdict

from _bootstrap import RESULTS_DIR


def _components(nodes, adj):
    seen, out = set(), []
    for s in sorted(nodes):
        if s in seen:
            continue
        comp, stack = set(), [s]
        seen.add(s)
        while stack:
            x = stack.pop()
            comp.add(x)
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        out.append(comp)
    return out


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def frame_stats(g):
    """The five quantities for one ImageGraph, plus per-type tallies."""
    from matching.relation_frame import (ATTRIBUTE_LENS_TYPES, has_measured_axis,
                                         is_frame)
    sg = g.subgraph
    lens = {n: lt for n, (lt, _s) in g.sigs.items()}
    frame = {n for n, lt in lens.items() if is_frame(lt)}
    unknown = Counter(lt for lt in lens.values()
                      if not is_frame(lt) and lt not in ATTRIBUTE_LENS_TYPES)
    adj_all, adj_meas = defaultdict(set), defaultdict(set)
    by_type = defaultdict(lambda: [0, 0])     # etype -> [structural, measured]
    n_struct = n_meas = 0
    for e in sorted(sg.edges, key=lambda e: (e[0], e[1], e[2].name)):
        u, v, et = e[0], e[1], e[2]
        if u == v or u not in frame or v not in frame:
            continue
        n_struct += 1
        by_type[et.name][0] += 1
        adj_all[u].add(v)
        adj_all[v].add(u)
        if has_measured_axis(sg.edge_params.get(e)):
            n_meas += 1
            by_type[et.name][1] += 1
            adj_meas[u].add(v)
            adj_meas[v].add(u)
    comps_all = _components(frame, adj_all)
    comps_meas = _components(frame, adj_meas)
    return {
        "frame": len(frame),
        "attribute": len(lens) - len(frame),
        "deg1": sum(1 for n in frame if len(adj_all.get(n, ())) >= 1),
        "deg2": sum(1 for n in frame if len(adj_all.get(n, ())) >= 2),
        "largest": max((len(c) for c in comps_all), default=0),
        "struct_edges": n_struct,
        "meas_edges": n_meas,
        "reachable": max((len(c) for c in comps_meas), default=0),
        "meas_deg": sorted(len(adj_meas.get(n, ())) for n in frame),
        "by_type": {k: tuple(v) for k, v in by_type.items()},
        "frame_by_lens": Counter(lens[n] for n in frame),
        "reach_by_lens": Counter(
            lens[n] for n in max(comps_meas, key=len, default=())),
        "unknown": unknown,
        # descriptive (computed after the verdict; decides nothing)
        "meas_sizes": sorted((len(c) for c in comps_meas), reverse=True),
        "all_sizes": sorted((len(c) for c in comps_all), reverse=True),
        "second_by_lens": Counter(
            lens[n] for n in (sorted(comps_meas, key=lambda c: (-len(c), min(c)))[1]
                              if len(comps_meas) > 1 else ())),
        "isolated_by_lens": Counter(
            lens[n] for n in frame if not adj_all.get(n)),
        "meas_lens_nodes": Counter(
            lens[n] for n in frame if adj_meas.get(n)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", default="phase5_feasibility")
    args = ap.parse_args()

    import eth80_frontend as fe
    import eth80_split as sp
    from phase5_graph_census import entries_of

    split = sp.load_split()
    entries = entries_of(split, args.split)
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s, flush=True)

    emit("PHASE 5 / B -- static feasibility of relation growth, %d %s graphs"
         % (len(entries), args.split.upper()))
    emit("front-end digest %s; no matcher is run"
         % fe.frontend_source_digest()[:12])
    emit()
    rows, by_type, frame_by_lens, reach_by_lens = [], defaultdict(lambda: [0, 0]), Counter(), Counter()
    unknown = Counter()
    per_class = defaultdict(list)
    for cls, rel in entries:
        st = frame_stats(fe.graph_for(cls, rel))
        rows.append(st)
        per_class[cls].append(st)
        for k, (a, b) in st["by_type"].items():
            by_type[k][0] += a
            by_type[k][1] += b
        frame_by_lens.update(st["frame_by_lens"])
        reach_by_lens.update(st["reach_by_lens"])
        unknown.update(st["unknown"])

    def col(name, f):
        xs = [f(r) for r in rows]
        emit("  %-58s median %7.2f   p10 %7.2f   min %7.2f   max %7.2f"
             % (name, statistics.median(xs), _pct(xs, 0.10), min(xs), max(xs)))
        return statistics.median(xs)

    emit("per graph (n = %d)" % len(rows))
    col("(1) FRAME nodes", lambda r: r["frame"])
    col("    ATTRIBUTE nodes", lambda r: r["attribute"])
    col("(2) FRAME nodes with >= 1 structural edge, share", lambda r: r["deg1"] / max(1, r["frame"]))
    col("    FRAME nodes with >= 2 structural edges, share", lambda r: r["deg2"] / max(1, r["frame"]))
    med_largest = col("(3) largest structural component, nodes", lambda r: r["largest"])
    col("    largest structural component / FRAME", lambda r: r["largest"] / max(1, r["frame"]))
    col("    structural edges", lambda r: r["struct_edges"])
    col("(4) structural edges with a measured graded axis, share", lambda r: r["meas_edges"] / max(1, r["struct_edges"]))
    col("(5) reachable set (measured component), nodes", lambda r: r["reachable"])
    med_reach = col("    reachable set / FRAME", lambda r: r["reachable"] / max(1, r["frame"]))
    col("    measured structural degree, median over FRAME nodes", lambda r: statistics.median(r["meas_deg"]) if r["meas_deg"] else 0)
    col("    FRAME nodes with measured degree 0, share", lambda r: sum(1 for d in r["meas_deg"] if d == 0) / max(1, r["frame"]))
    emit()
    emit("per class: median (1) | (3) | (5) | (5)/(1)")
    for cls in sorted(per_class):
        rs = per_class[cls]
        emit("  %-8s n=%2d  %6.0f | %6.0f | %6.0f | %.2f" % (
            cls, len(rs), statistics.median(r["frame"] for r in rs),
            statistics.median(r["largest"] for r in rs),
            statistics.median(r["reachable"] for r in rs),
            statistics.median(r["reachable"] / max(1, r["frame"]) for r in rs)))
    emit()
    emit("structural edges by type (all graphs): count, with a measured graded axis")
    for k, (a, b) in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
        emit("  %-28s %7d   measured %7d  (%5.1f%%)" % (k, a, b, 100.0 * b / max(1, a)))
    emit()
    emit("FRAME nodes by lens type (all graphs): count, in the reachable set")
    for k, a in sorted(frame_by_lens.items(), key=lambda kv: -kv[1]):
        emit("  %-18s %7d   reachable %7d  (%5.1f%%)" % (k, a, reach_by_lens.get(k, 0), 100.0 * reach_by_lens.get(k, 0) / max(1, a)))
    if unknown:
        emit()
        emit("lens types in NEITHER list (counted as ATTRIBUTE): %s" % dict(unknown))
    emit()
    fire = med_largest < 8 or med_reach < 0.25
    emit("STOP RULE (pre-registered): median (3) < 8 -> %s (%.0f);  median (5)/(1) < 0.25 -> %s (%.3f)"
         % (med_largest < 8, med_largest, med_reach < 0.25, med_reach))
    emit("VERDICT: %s" % ("STOP -- do not build the matcher" if fire else "the rule does not fire -- Phase C is built"))
    # ------------------------------------------------------------------
    # descriptive: what the frame is made of.  Computed after the verdict;
    # none of it is a criterion.
    # ------------------------------------------------------------------
    emit()
    emit("DESCRIPTIVE (after the verdict; decides nothing)")

    def nth(sizes, k):
        return sizes[k] if len(sizes) > k else 0

    for k, label in ((0, "largest"), (1, "second"), (2, "third")):
        col("measured component, %s, nodes" % label, lambda r, k=k: nth(r["meas_sizes"], k))
    col("measured components of size >= 2, count", lambda r: sum(1 for x in r["meas_sizes"] if x >= 2))
    col("FRAME share in measured components of size >= 8 (any number of seeds)",
        lambda r: sum(x for x in r["meas_sizes"] if x >= 8) / max(1, r["frame"]))
    col("FRAME share in measured components of size >= 2",
        lambda r: sum(x for x in r["meas_sizes"] if x >= 2) / max(1, r["frame"]))
    col("what-if every structural edge were measured: (3)/(1)",
        lambda r: r["largest"] / max(1, r["frame"]))
    col("structural components of size >= 2, count", lambda r: sum(1 for x in r["all_sizes"] if x >= 2))
    second = Counter()
    isolated = Counter()
    meas_lens = Counter()
    for r in rows:
        second.update(r["second_by_lens"])
        isolated.update(r["isolated_by_lens"])
        meas_lens.update(r["meas_lens_nodes"])
    emit()
    emit("second-largest measured component, by lens type (all graphs): %s"
         % dict(second.most_common()))
    emit()
    emit("FRAME nodes by lens type: total | with a measured structural edge | with NO structural edge at all")
    for k, a in sorted(frame_by_lens.items(), key=lambda kv: -kv[1]):
        emit("  %-18s %7d | %7d (%5.1f%%) | %7d (%5.1f%%)" % (
            k, a, meas_lens.get(k, 0), 100.0 * meas_lens.get(k, 0) / max(1, a),
            isolated.get(k, 0), 100.0 * isolated.get(k, 0) / max(1, a)))
    related = {k for k, v in meas_lens.items() if v > 0}
    emit()
    emit("lens types with at least one measured structural edge anywhere: %s" % sorted(related))
    xs = []
    for r in rows:
        denom = sum(v for k, v in r["frame_by_lens"].items() if k in related)
        xs.append(r["reachable"] / max(1, denom))
    emit("  (5) / FRAME restricted to those lens types: median %.3f, p10 %.3f"
         "   [not the pre-registered frame: it leaves out %.0f%% of the FRAME nodes]"
         % (statistics.median(xs), _pct(xs, 0.10),
            100.0 * sum(v for k, v in frame_by_lens.items() if k not in related)
            / max(1, sum(frame_by_lens.values()))))
    with open(os.path.join(RESULTS_DIR, args.out + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
