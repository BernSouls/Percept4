"""PHASE 0 / AUDIT C (read-only) -- anchor-shuffle control (F5).

Handoff finding under test:
  F5  position is a hidden prior that CO_LOCATED edges restate.

The matcher weighs every candidate node pair by
``signature_similarity * exp(-2.5 * distance)``, the distance taken between
the two nodes' anchor positions in each graph's own bounding box
(``memory/cores.py``, ``_anchor_similarity``).  Position therefore enters the
correspondence without being an edge and without being in a signature.

This control changes ONLY the query's occupancy map (``anchor_of_node``).
Every node, every signature, every edge and every edge parameter is left
exactly as the front end emitted it.

  no_anchor       the map is removed: the position factor is 1.0 everywhere
  shuffle/anchor  the DISTINCT positions are permuted; nodes that shared a
                  patch still share one (co-location is preserved)
  shuffle/node    positions are permuted over the nodes individually; nodes
                  that shared a patch are scattered

A matcher that read the object's STRUCTURE would be unmoved by all three: the
relations between primitives are in the edges, and the edges are untouched.

Under the shuffle the score clamps to exactly 0.0 for every reference on some
queries.  The benchmark gives such a tie to the first class in the order
apple, car, cup, horse, which hands apple queries a free hit, so C1 reports
three accuracies: the benchmark's rule, ties counted wrong, and ties split
uniformly.  Every arm is also compared PAIRED with the baseline.

    python diagnostics/phase0_audit_C_anchor_shuffle.py [--seeds 5]
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import os
from typing import Any, Dict, List

from _bootstrap import RESULTS_DIR


def _own_class_means(res: Dict[str, Any]) -> Dict[str, float]:
    """Means over the 75 (query, own-class reference) pairs."""
    keys = ("score", "node_coverage", "relational_consistency",
            "edge_absence")
    acc = {k: 0.0 for k in keys}
    for q in res["queries"]:
        pr = q["pairs"][q["true"]]
        for k in keys:
            acc[k] += pr[k]
    n = len(res["queries"])
    out = {k: v / n for k, v in acc.items()}
    e = res["agg"]["edge_true"]
    out["verifiable"] = sum(r["verifiable"] for r in e.values())
    out["no_q_edge"] = sum(r["no_query_edge"] for r in e.values())
    out["matched"] = sum(r["matched"] for r in e.values())
    out["sim_sum"] = sum(r["sim_sum"] for r in e.values())
    co = e.get("CO_LOCATED", {})
    out["co_verifiable"] = co.get("verifiable", 0)
    out["co_matched"] = co.get("matched", 0)
    out["co_sim_sum"] = co.get("sim_sum", 0.0)
    a = res["agg"]["node_true"]
    c = sum(r["credited"] for r in a.values())
    out["credited"] = c
    out["mean_sig_sim"] = (sum(r["sim_sum"] for r in a.values()) / c) if c else 0.0
    return out


def main() -> int:
    import _stats as st
    import eth80_benchmark as bm
    import eth80_frontend as fe
    from eth80_split import load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    seeds = list(range(a.seeds))

    out: List[str] = []
    P = out.append
    P("PHASE 0 / AUDIT C -- anchor-shuffle control (EVAL-75, in-sample)")
    P("only the QUERY's anchor_of_node map is changed; nodes, signatures,"
      " edges and edge parameters are untouched")
    P("")

    for window in bm.WINDOWS:
        base = bm.run_cached("eval", window, "baseline", breakdown=True)
        s_base = bm.summarize(base)
        P("=" * 78)
        P("window = %s" % window)
        P("=" * 78)
        rows = [("baseline", base, s_base)]
        arms = (["no_anchor"]
                + ["shuffle_anchor_%d" % s for s in seeds]
                + ["shuffle_node_%d" % s for s in seeds])
        for arm in arms:
            res = bm.run_cached("eval", window, arm, breakdown=True)
            rows.append((arm, res, bm.summarize(res)))

        P("C1. accuracy")
        for arm, res, s in rows:
            P("  " + bm.fmt_row(arm, s))
        P("")
        P("  ties and paired comparison (all %d tied decisions below have a"
          " best score of exactly 0.0)" % sum(
              int(st.tie_report(res)["ties_zero"]) for _a, res, _s in rows))
        P("  %-18s %8s %10s %10s %6s   %s" % (
            "arm", "as rule", "ties wrong", "ties split", "ties",
            "paired with baseline (first-in-order rule)"))
        reports = {}
        for arm, res, s in rows:
            tr = st.tie_report(res)
            assert tr["ties"] == tr["ties_zero"], arm
            reports[arm] = tr
            if arm == "baseline":
                tail = ""
            else:
                g, l, pv = st.paired(base, res)
                churn = sum(int(x != y) for x, y in zip(s["preds"],
                                                        s_base["preds"]))
                tail = "+%d/-%d  p=%.3f   %d predictions changed" % (
                    g, l, pv, churn)
            P("  %-18s %8d %10d %10.2f %6d   %s" % (
                arm, tr["first"], tr["strict"], tr["split"], tr["ties"],
                tail))
        b_tr = reports["baseline"]
        for level in ("anchor", "node"):
            trs = [reports[arm] for arm, _r, _s in rows
                   if arm.startswith("shuffle_%s_" % level)]
            if not trs:
                continue
            m_first = st.mean([t["first"] for t in trs])
            m_strict = st.mean([t["strict"] for t in trs])
            m_split = st.mean([t["split"] for t in trs])
            P("  shuffle/%-6s mean of %d seeds: as rule %.1f (min %d, max %d)"
              " | ties wrong %.1f | ties split %.2f | ties %.1f" % (
                  level, len(trs), m_first,
                  min(t["first"] for t in trs), max(t["first"] for t in trs),
                  m_strict, m_split, st.mean([t["ties"] for t in trs])))
            P("     baseline %d  ->  drop %.1f queries as scored,"
              " %.1f with ties split (%.1f points)" % (
                  b_tr["first"], b_tr["first"] - m_first,
                  b_tr["split"] - m_split,
                  100.0 * (b_tr["split"] - m_split) / b_tr["n"]))
        P("  chance for a 4-way decision on this split: 25.0%% uniform;"
          " always answering the largest class: %d/%d = %.1f%%"
          % (max(v[1] for v in s_base["per_class"].values()), s_base["n"],
             100.0 * max(v[1] for v in s_base["per_class"].values())
             / s_base["n"]))
        P("")

        P("C2. what moved, on the 75 (query, own-class reference) pairs")
        P("  %-18s %7s %7s %7s %7s | %8s %9s %8s %8s | %8s %8s %8s | %7s" % (
            "arm", "score", "nodecov", "rel.con", "absence", "verif",
            "no-q-edge", "matched", "sim_sum", "CO verif", "CO match",
            "CO sim", "sig sim"))
        shown = ["baseline", "no_anchor"]
        groups: Dict[str, List[Dict[str, float]]] = {}
        table = {arm: _own_class_means(res) for arm, res, _s in rows}
        for arm in table:
            if arm.startswith("shuffle_"):
                groups.setdefault(arm.rsplit("_", 1)[0], []).append(table[arm])
        lines = [(arm, table[arm]) for arm in shown]
        for gname, members in groups.items():
            mean = {k: sum(m[k] for m in members) / len(members)
                    for k in members[0]}
            lines.append((gname + " (mean)", mean))
        for arm, m in lines:
            P("  %-18s %7.4f %7.4f %7.4f %7.4f | %8.0f %9.0f %8.0f %8.1f |"
              " %8.0f %8.0f %8.1f | %7.3f" % (
                  arm, m["score"], m["node_coverage"],
                  m["relational_consistency"], m["edge_absence"],
                  m["verifiable"], m["no_q_edge"], m["matched"], m["sim_sum"],
                  m["co_verifiable"], m["co_matched"], m["co_sim_sum"],
                  m["mean_sig_sim"]))
        P("")

    # ---- C3: position that the shuffle does NOT touch ---------------------
    split = load_split()
    entries = list(split["references"].items()) + [
        (c, p) for c in bm.CLASSES for p in split["eval"][c]]
    n_nodes = n_pos = n_bnd = n_anch = 0
    for c, p in entries:
        g = fe.graph_for(c, p)
        n_nodes += len(g.sigs)
        n_anch += len(g.subgraph.anchor_of_node)
        for lt, sig in g.sigs.values():
            n_pos += int(sig.class_ == "POSITION")
            n_bnd += int(sig.class_ == "BOUNDARY_ANCHORED")
    P("C3. position the shuffle does not touch (it lives in SIGNATURES)")
    P("  nodes over %d images: %d;  with an anchor position: %d (%.1f%%)"
      % (len(entries), n_nodes, n_anch, 100.0 * n_anch / n_nodes))
    P("  POSITION nodes (signature = object-centric coordinates): %d (%.1f%%)"
      % (n_pos, 100.0 * n_pos / n_nodes))
    P("  BOUNDARY_ANCHORED nodes (slot 2 = 4x4 position class): %d (%.1f%%)"
      % (n_bnd, 100.0 * n_bnd / n_nodes))
    P("  => %.1f%% of all nodes carry image position inside their identity;"
      " the control above leaves that channel intact."
      % (100.0 * (n_pos + n_bnd) / n_nodes))
    P("")

    # ---- C4: what an anchor position is ----------------------------------
    from collections import Counter

    from memory.cores import lens_tier

    kept, nodes, anch = Counter(), Counter(), Counter()
    for c, p in entries:
        g = fe.graph_for(c, p)
        for lt, v in g.det_counts.get("v1_kept", {}).items():
            kept[lt] += v
        for nid, (lt, _sig) in g.sigs.items():
            if nid in g.subgraph.nodes:
                nodes[lt] += 1
                anch[lt] += int(nid in g.subgraph.anchor_of_node)
    P("C4. what the position factor compares")
    P("  A node is a signature, not a place: detections with the same"
      " signature share ONE node,")
    P("  and the node keeps the position of the detection that minted it"
      " (ConceptGraph.note_pattern_node_position, first write wins).")
    P("  %-18s %5s %9s %8s %9s %10s" % (
        "lens", "tier", "kept det", "nodes", "det/node", "anchored"))
    for lt in sorted(nodes, key=lambda k: -nodes[k]):
        P("  %-18s %5d %9s %8d %9s %9.0f%%" % (
            lt, lens_tier(lt),
            ("%d" % kept[lt]) if lt in kept else "-",
            nodes[lt],
            ("%.2f" % (kept[lt] / nodes[lt])) if lt in kept else "-",
            100.0 * anch[lt] / nodes[lt]))
    tot_n = sum(nodes.values())
    tot_a = sum(anch.values())
    v1_n = sum(v for k, v in nodes.items() if lens_tier(k) == 0)
    v1_a = sum(v for k, v in anch.items() if lens_tier(k) == 0)
    P("  anchored: %d of %d nodes (%.1f%%) in the full window; %d of %d"
      " (%.1f%%) in the v1 window." % (
          tot_a, tot_n, 100.0 * tot_a / tot_n, v1_a, v1_n,
          100.0 * v1_a / v1_n))
    P("  An unanchored node gets a position factor of 1.0, so the shuffle"
      " cannot reach it.")
    P("")

    text = "\n".join(out)
    print(text)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase0_audit_C.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
