"""PHASE 5 / B, AFTER THE VERDICT -- what would move the reachable set?

DESCRIPTIVE AND POST HOC.  Written after the pre-registered stop rule had
fired (results/phase5_feasibility.txt) and after its numbers had been seen,
on the same 79 EVAL graphs.  Nothing here is a criterion, and none of it
can be used to re-decide the stop on EVAL.

It recomputes the reachable set / FRAME of phase5_feasibility.py while
pretending that some of the EXISTING unmeasured structural edges carried a
measurement:

  W1  the flags that join an Orientation node to another lens type
      (CONTOUR_ORIENTATION, JUNCTION_ORIENTATION, REGION_CONTENT)
  W2  the flag that joins a PEAK_CHAIN to its CONTOUR_PEAKs
  W3  every structural edge (the ceiling: the largest structural component)

and lists the lens types of the largest structural component.

    python diagnostics/phase5_whatif.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import os
import statistics
from collections import Counter, defaultdict

from _bootstrap import RESULTS_DIR


def main() -> int:
    import eth80_frontend as fe
    import eth80_split as sp
    from matching.relation_frame import has_measured_axis, is_frame
    from phase5_feasibility import _components, _pct
    from phase5_graph_census import entries_of

    def largest(g, pretend):
        lens = {n: lt for n, (lt, _s) in g.sigs.items()}
        frame = {n for n, lt in lens.items() if is_frame(lt)}
        adj = defaultdict(set)
        for e in g.subgraph.edges:
            u, v, et = e[0], e[1], e[2]
            if u == v or u not in frame or v not in frame:
                continue
            if (has_measured_axis(g.subgraph.edge_params.get(e))
                    or pretend(et.name, lens[u], lens[v])):
                adj[u].add(v)
                adj[v].add(u)
        big = max(_components(frame, adj), key=len)
        return len(big) / max(1, len(frame)), Counter(lens[n] for n in big)

    cases = (
        ("registered definition", lambda t, a, b: False),
        ("W1 Orientation flags measured", lambda t, a, b: (
            t in ("CONTOUR_ORIENTATION", "JUNCTION_ORIENTATION",
                  "REGION_CONTENT") and "Orientation" in (a, b))),
        ("W2 chain-peak flag measured", lambda t, a, b: (
            t == "CONTOUR_ORIENTATION" and "PEAK_CHAIN" in (a, b))),
        ("W3 every structural edge measured (ceiling)", lambda t, a, b: True),
    )
    rows = {name: [] for name, _f in cases}
    ceiling_lens = Counter()
    frame_lens = Counter()
    for cls, rel in entries_of(sp.load_split(), "eval"):
        g = fe.graph_for(cls, rel)
        frame_lens.update(lt for _n, (lt, _s) in g.sigs.items() if is_frame(lt))
        for name, f in cases:
            r, comp = largest(g, f)
            rows[name].append(r)
            if name.startswith("W3"):
                ceiling_lens.update(comp)

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s, flush=True)

    emit("PHASE 5 / B, AFTER THE VERDICT -- what-if, %d EVAL graphs, front-end digest %s"
         % (len(rows["registered definition"]), fe.frontend_source_digest()[:12]))
    emit("DESCRIPTIVE AND POST HOC: computed after the stop rule fired; decides nothing.")
    emit()
    emit("largest component over (measured or pretended-measured) structural edges, as a share of FRAME")
    for name, _f in cases:
        xs = rows[name]
        emit("  %-46s median %.3f   p10 %.3f   graphs at 0.25 or more: %2d of %d"
             % (name, statistics.median(xs), _pct(xs, 0.10),
                sum(1 for x in xs if x >= 0.25), len(xs)))
    emit()
    emit("the largest STRUCTURAL component, by lens type (all graphs): nodes in it / FRAME nodes of the type")
    for lt, n in sorted(frame_lens.items(), key=lambda kv: -kv[1]):
        emit("  %-18s %6d / %6d" % (lt, ceiling_lens.get(lt, 0), n))
    with open(os.path.join(RESULTS_DIR, "phase5_whatif.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
