"""[STAGE 6 / direction 1] Is the binder's position map a stale copy?

Hypothesis (to verify, not assume): `Pipeline.step` hands the binder a COPY of
the anchor-position map made before the higher lens tier runs.  The tier then
registers positions for its own anchors (EndStoppedLens writes
`anchor_positions[syn] = (x, y)`; peaks, chains and corners likewise) in the
lens tier's map only.  So every binder writer that asks the binder's map for
such an anchor gets nothing, and skips.

Per image (real front end, single step, with a trace):

 A. per detection kind: how many anchors are in the binder's map, how many in
    the lens tier's map;
 B. `_run_line_termination` replayed twice on the same detections: with the
    binder's map as it was, and with the map completed from the lens tier's
    (nothing else changed);
 C. `_run_part_containment` (RegionShape -> CurveComplex), which the pipeline
    never emits ("lo_pit" is not a level any caller asks for): the three gates
    the writer has, counted with each map;
 D. the edges EndStopped and RegionShape nodes actually have.

    python diagnostics/phase6_position_map.py [--workers 2] [--out NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import eth80_frontend as fe  # noqa: E402
import eth80_split as sp  # noqa: E402


def _replay(tr, positions):
    """Run the two writers on the step's own detections with `positions`."""
    from core.hashing import _detection_sort_key
    from core.types import EdgeType
    from lenses.derived_common import is_junction_kind
    from orchestration.binder import Binder
    from perception.depth import _extract_member_anchors, _mean_position

    class _Capture(Binder):
        def __init__(self, scale):
            self.seen = {}
            self._step_extraction_scale = scale

        def _process_channel(self, edge_type, candidates):
            self.seen[edge_type] = list(candidates)

    dets = tuple(sorted((d for v in tr["tier_dets"].values() for d in v),
                        key=_detection_sort_key))
    node_ids = dict(tr["higher_node_ids"])
    b = _Capture(float(tr["extraction_scale"]))
    b._run_line_termination(dets, node_ids, positions)
    lt = b.seen.get(EdgeType.LINE_TERMINATION, [])
    kind_of = {}
    for d in dets:
        if d in node_ids:
            kind_of.setdefault(node_ids[d], d.k)
    term_pos = Counter()
    for d in dets:
        if (d.k == "EndStopped" or is_junction_kind(d.k)) and d in node_ids:
            term_pos["EndStopped" if d.k == "EndStopped" else "junction"] += int(
                positions.get(d.alpha) is not None)
    out = {
        "terminators_positioned": dict(term_pos),
        "lt_candidates": len(lt),
        "lt_candidates_from": dict(Counter(
            "EndStopped" if kind_of.get(c[0]) == "EndStopped" else "junction" for c in lt)),
        "lt_parent_kind": dict(Counter(kind_of.get(c[1], "?") for c in lt)),
        "lt_endstopped_nodes_bound": len({c[0] for c in lt if kind_of.get(c[0]) == "EndStopped"}),
    }
    rs = [d for d in dets if d.k == "RegionShape"]
    cc = [d for d in dets if d.k == "CurveComplex"]
    out["rs_total"] = len(rs)
    out["rs_gate_position"] = sum(1 for d in rs if positions.get(d.alpha) is not None)
    out["rs_gate_arity5"] = sum(1 for d in rs if len(d.sigma.quantized) >= 5)
    out["cc_total"] = len(cc)
    out["cc_gate_member_positions"] = sum(
        1 for d in cc if _mean_position(_extract_member_anchors(d), positions) is not None)
    b2 = _Capture(float(tr["extraction_scale"]))
    b2._run_part_containment(dets, node_ids, positions)
    pc = b2.seen.get(EdgeType.PART_CONTAINMENT, [])
    out["pc_candidates"] = len(pc)
    out["pc_regionshape_nodes_bound"] = len({c[0] for c in pc})
    return out


def one(entry):
    cls, rel = entry
    tr = {}
    g = fe.build_graph(fe._load_rgb(fe.image_path(cls, rel)), name=os.path.basename(rel), trace=tr)
    binder_map = dict(tr["step_positions"])
    tier_map = dict(tr["anchor_positions"])
    row = {"class": cls, "image": os.path.basename(rel),
           "binder_map_keys": len(binder_map), "tier_map_keys": len(tier_map),
           "binder_keys_missing_from_tier": sum(1 for k in binder_map if k not in tier_map)}
    kinds = {}
    for d in list(tr["D_v1"]) + [d for v in tr["tier_dets"].values() for d in v]:
        k = kinds.setdefault(d.k, [0, 0, 0])
        k[0] += 1
        k[1] += int(d.alpha in binder_map)
        k[2] += int(d.alpha in tier_map)
    row["kinds"] = kinds
    row["as_run"] = _replay(tr, binder_map)
    completed = dict(tier_map)
    completed.update(binder_map)          # the binder's own entries win
    row["completed"] = _replay(tr, completed)
    row["real_lt_candidates"] = int(g.binder_diag["candidates_by_type"].get("LINE_TERMINATION", 0))
    lens = {n: lt for n, (lt, _s) in g.sigs.items()}
    deg = Counter()
    for e in g.subgraph.edges:
        for n in (e[0], e[1]):
            if lens.get(n) in ("EndStopped", "RegionShape"):
                deg[lens[n]] += 1
    row["edges_touching"] = dict(deg)
    row["nodes"] = dict(Counter(v for v in lens.values() if v in ("EndStopped", "RegionShape")))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="phase6_position_map")
    a = ap.parse_args(argv)
    split = sp.load_split()
    entries = list(split["references"].items()) + [
        (c, p) for c, ps in split["eval"].items() for p in ps]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(one, entries, chunksize=1)
    else:
        rows = [one(e) for e in entries]

    L = ["[STAGE 6 / direction 1] the binder's position map, %d graphs" % len(rows), ""]
    L.append("keys in the binder's map: median %d; in the lens tier's map: median %d; "
             "binder keys missing from the tier's map: %d" % (
                 sorted(r["binder_map_keys"] for r in rows)[len(rows) // 2],
                 sorted(r["tier_map_keys"] for r in rows)[len(rows) // 2],
                 sum(r["binder_keys_missing_from_tier"] for r in rows)))
    L += ["", "A. anchors with a position, per detection kind (all graphs pooled)",
          "   kind               detections   in the binder's map   in the lens tier's map"]
    tot = {}
    for r in rows:
        for k, (n, nb, nt) in r["kinds"].items():
            t = tot.setdefault(k, [0, 0, 0])
            t[0] += n
            t[1] += nb
            t[2] += nt
    for k in sorted(tot, key=lambda k: (tot[k][1] / max(tot[k][0], 1), k)):
        n, nb, nt = tot[k]
        L.append("   %-18s %8d      %7d (%.3f)       %7d (%.3f)" % (k, n, nb, nb / max(n, 1), nt, nt / max(n, 1)))

    def s(key, which):
        return sum(r[which].get(key, 0) for r in rows)

    def sd(key, which):
        c = Counter()
        for r in rows:
            c.update(r[which].get(key, {}))
        return dict(sorted(c.items()))

    L += ["", "B. `_run_line_termination`, same detections, two maps",
          "                                            binder's map (as run)    map completed from the tier"]
    L.append("   terminators with a position             %-24s %s" % (sd("terminators_positioned", "as_run"), sd("terminators_positioned", "completed")))
    L.append("   candidates the writer builds            %-24d %d" % (s("lt_candidates", "as_run"), s("lt_candidates", "completed")))
    L.append("   ... by terminator kind                  %-24s %s" % (sd("lt_candidates_from", "as_run"), sd("lt_candidates_from", "completed")))
    L.append("   ... by parent kind                      %-24s %s" % (sd("lt_parent_kind", "as_run"), sd("lt_parent_kind", "completed")))
    L.append("   EndStopped nodes that get an edge       %-24d %d" % (s("lt_endstopped_nodes_bound", "as_run"), s("lt_endstopped_nodes_bound", "completed")))
    L.append("   (check: candidates in the real run's diagnostics: %d)" % sum(r["real_lt_candidates"] for r in rows))
    L += ["", "C. `_run_part_containment` (RegionShape -> CurveComplex); the pipeline never calls its level",
          "                                            binder's map             map completed from the tier"]
    L.append("   RegionShape detections                  %-24d %d" % (s("rs_total", "as_run"), s("rs_total", "completed")))
    L.append("   gate 1: own anchor has a position       %-24d %d" % (s("rs_gate_position", "as_run"), s("rs_gate_position", "completed")))
    L.append("   gate 2: len(q) >= 5                     %-24d %d" % (s("rs_gate_arity5", "as_run"), s("rs_gate_arity5", "completed")))
    L.append("   CurveComplex detections                 %-24d %d" % (s("cc_total", "as_run"), s("cc_total", "completed")))
    L.append("   gate 3: member positions resolve        %-24d %d" % (s("cc_gate_member_positions", "as_run"), s("cc_gate_member_positions", "completed")))
    L.append("   candidates if the level were emitted    %-24d %d" % (s("pc_candidates", "as_run"), s("pc_candidates", "completed")))
    L.append("   RegionShape nodes that would get one    %-24d %d" % (s("pc_regionshape_nodes_bound", "as_run"), s("pc_regionshape_nodes_bound", "completed")))
    nodes = Counter()
    edges = Counter()
    for r in rows:
        nodes.update(r["nodes"])
        edges.update(r["edges_touching"])
    L += ["", "D. in the graphs as built: EndStopped %d nodes, %d edges touching them; RegionShape %d nodes, %d edges" % (
        nodes.get("EndStopped", 0), edges.get("EndStopped", 0), nodes.get("RegionShape", 0), edges.get("RegionShape", 0))]
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w") as fh:
        json.dump(rows, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
