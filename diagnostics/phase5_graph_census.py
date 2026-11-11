"""PHASE 5 step 0 -- what the graphs are made of (read-only, no matching).

Per lens type: nodes per graph.  Per edge type: how many, which lens-type
pairs they connect, whether the two ends sit on the SAME observation anchor,
and per parameter slot the measured share and the number of distinct values.

    python diagnostics/phase5_graph_census.py [--split eval] [--out NAME]

Diagnostics only: reads the cached graphs, changes nothing.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import os
import statistics
from collections import Counter, defaultdict

from _bootstrap import RESULTS_DIR


def entries_of(split, which: str):
    refs = list(split["references"].items())
    if which not in ("eval", "heldout"):
        raise ValueError(which)
    return refs + [(c, p) for c in sorted(split[which])
                   for p in split[which][c]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", default="phase5_graph_census")
    args = ap.parse_args()

    import eth80_frontend as fe
    import eth80_split as sp
    from core.schema import EDGE_SCHEMAS, UNKNOWN

    split = sp.load_split()
    entries = entries_of(split, args.split)
    lines = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s, flush=True)

    emit("PHASE 5 step 0 -- graph census, %d %s graphs (read-only)"
         % (len(entries), args.split.upper()))
    emit("front-end digest %s" % fe.frontend_source_digest()[:12])

    nodes_per = defaultdict(list)          # lens -> [count per graph]
    n_nodes, n_edges = [], []
    et_count = Counter()
    et_pairs = defaultdict(Counter)        # etype -> Counter((lensA, lensB))
    et_same_anchor = defaultdict(lambda: [0, 0, 0])   # same, differ, unknown
    et_slot = defaultdict(lambda: defaultdict(lambda: [0, Counter()]))
    et_noparams = Counter()
    for cls, rel in entries:
        g = fe.graph_for(cls, rel)
        sg = g.subgraph
        c = Counter(lt for (lt, _s) in g.sigs.values())
        for lt in c:
            nodes_per[lt].append(c[lt])
        n_nodes.append(len(sg.nodes))
        n_edges.append(len(sg.edges))
        for (u, v, et) in sg.edges:
            name = et.name
            et_count[name] += 1
            la = g.sigs.get(u, ("?", None))[0]
            lb = g.sigs.get(v, ("?", None))[0]
            et_pairs[name][(la, lb)] += 1
            au, av = g.node_anchor.get(u), g.node_anchor.get(v)
            if au is None or av is None:
                et_same_anchor[name][2] += 1
            elif au == av:
                et_same_anchor[name][0] += 1
            else:
                et_same_anchor[name][1] += 1
            ep = sg.edge_params.get((u, v, et))
            if ep is None or not getattr(ep, "quantized", None):
                et_noparams[name] += 1
                continue
            schema = EDGE_SCHEMAS.get(ep.class_, ())
            for i, b in enumerate(ep.quantized):
                tag = schema[i].tag if i < len(schema) else "slot%d" % i
                rec = et_slot[name][(i, tag)]
                if b != UNKNOWN:
                    rec[0] += 1
                    rec[1][b] += 1

    emit("")
    emit("nodes per graph: median %d (min %d, max %d);  edges per graph: "
         "median %d (min %d, max %d)" % (
             statistics.median(n_nodes), min(n_nodes), max(n_nodes),
             statistics.median(n_edges), min(n_edges), max(n_edges)))
    emit("")
    emit("  %-22s %7s %8s %8s %8s" % ("lens type", "graphs", "median",
                                       "min", "max"))
    for lt in sorted(nodes_per, key=lambda k: -sum(nodes_per[k])):
        v = nodes_per[lt]
        emit("  %-22s %7d %8d %8d %8d" % (
            lt, len(v), statistics.median(v), min(v), max(v)))

    emit("")
    emit("edges: type, count, same-anchor share, endpoint lens pairs "
         "(top 4), slots (measured%, distinct values)")
    for name, n in et_count.most_common():
        same, diff, unk = et_same_anchor[name]
        pairs = ", ".join("%s>%s %d" % (a, b, k)
                          for (a, b), k in et_pairs[name].most_common(4))
        more = len(et_pairs[name]) - 4
        if more > 0:
            pairs += ", +%d more" % more
        emit("  %-26s %6d  same-anchor %5.1f%% (unknown %d)" % (
            name, n, 100.0 * same / max(n, 1), unk))
        emit("      pairs: %s" % pairs)
        if et_noparams[name] == n:
            emit("      slots: (no parameters)")
            continue
        slots = []
        for (i, tag), (m, vals) in sorted(et_slot[name].items()):
            slots.append("%s %d%% %s" % (
                tag, round(100.0 * m / max(n, 1)),
                ("%d" % len(vals)) if m else "-"))
        emit("      slots: %s" % " | ".join(slots))

    out = os.path.join(RESULTS_DIR, args.out + ".txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
