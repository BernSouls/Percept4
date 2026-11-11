"""STAGE 2 addendum -- descriptive re-reads of the stored stage-2 records.

Read-only: nothing is scored again.  NONE of these tables was a
pre-registered criterion; they explain what the criteria tables show and are
labelled as such in REPORT.md.

  A  the three terms decided alone, per configuration (Audit B1 re-read)
  B  per-class paired movement of the pre-registered comparisons
  C  sizes: each reference against the queries of its own class
  D  what the correspondence leaves unpaired (binary coverage vs the count
     a maximum pairing reaches)
  E  the queries the calibrated score cannot decide
  F  is the chance level class-neutral?  (it is built from the two graphs
     being compared)

    python diagnostics/phase2_addendum.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import json
import os
import sys
from typing import Any, Dict, List

import _stats as st
from _bootstrap import RESULTS_DIR
from _stats import CLASSES

TAGS = [("F6+D1", "p2_R0"), ("D3:node", "p2_D3n"), ("D3:node+abs", "p2_D3na"),
        ("D3", "p2_D3"), ("D3,D1off", "p2_D3b"), ("NULL", "p2_NULL"),
        ("NULL+D3", "p2_NULLD3")]


def load(tag: str, w: str) -> Dict[str, Any]:
    with open(os.path.join(RESULTS_DIR, "%s_%s.json" % (tag, w)), "r",
              encoding="utf-8") as fh:
        return json.load(fh)


def row(label: str, res: Dict[str, Any], key, ref=None, ref_key=None,
        ref_name: str = "") -> str:
    h = st.hits(res, key)
    per = st.per_class_hits(res, key)
    ans = st.answers(res, key)
    t = st.tie_report(res, key)
    line = "  %-40s %2d/%d   %s   answers %-11s" % (
        label, sum(h), len(h),
        " ".join("%s %2d/%-2d" % (c, per[c][0], per[c][1]) for c in CLASSES),
        "/".join("%d" % ans[c] for c in CLASSES))
    if t["ties"]:
        line += " ties %d (split %.2f)" % (t["ties"], t["split"])
    if ref is not None:
        g, l, p = st.paired(ref, res, ref_key, key)
        line += "   %+d/-%d p=%.3f%s" % (
            g, l, p, (" vs " + ref_name) if ref_name else "")
    return line


def main() -> int:
    out: List[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out.append(s)

    emit("STAGE 2 addendum -- descriptive re-reads of the stored records "
         "(not pre-registered criteria)")
    for w in ("full", "v1"):
        R = {name: load(tag, w) for name, tag in TAGS}
        for r in R.values():
            for q in r["queries"]:
                for pr in q["pairs"].values():
                    pr["neg_absence"] = -pr["edge_absence"]
                    pr["node_x_rel"] = (pr["node_coverage"]
                                        * pr["relational_multiplier"])
        emit("")
        emit("#" * 100)
        emit("# window = %s   (source digest %s)" % (
            w, R["F6+D1"]["source_digest"][:12]))
        emit("#" * 100)

        # ------------------------------------------------------------ A
        emit("")
        emit("A. each term deciding ALONE, and the score, per configuration "
             "(paired against that configuration's score)")
        for name in ("F6+D1", "D3:node", "D3", "NULL", "NULL+D3"):
            r = R[name]
            emit("  %s" % name)
            emit(row("  evidence score", r, None))
            emit(row("  node term alone", r, "node_coverage", r, None,
                     "score"))
            emit(row("  relational term alone", r, "relational_consistency",
                     r, None, "score"))
            emit(row("  absence alone (least absence wins)", r, "neg_absence",
                     r, None, "score"))
            emit(row("  node x relation (no absence penalty)", r,
                     "node_x_rel", r, None, "score"))

        # ------------------------------------------------------------ B
        emit("")
        emit("B. per-class movement, paired (queries gained / lost per true "
             "class)")
        for a, b in (("F6+D1", "D3:node"), ("F6+D1", "D3:node+abs"),
                     ("F6+D1", "D3"), ("F6+D1", "NULL"), ("D3", "NULL+D3")):
            ha, hb = st.hits(R[a]), st.hits(R[b])
            cells = []
            for c in CLASSES:
                g = sum(int(y and not x) for q, x, y in
                        zip(R[a]["queries"], ha, hb) if q["true"] == c)
                l = sum(int(x and not y) for q, x, y in
                        zip(R[a]["queries"], ha, hb) if q["true"] == c)
                cells.append("%s +%d/-%d" % (c, g, l))
            g, l, p = st.paired(R[a], R[b])
            emit("  %-12s -> %-12s %s   total +%d/-%d p=%.3f" % (
                a, b, "  ".join(cells), g, l, p))

        # ------------------------------------------------------------ C
        emit("")
        emit("C. sizes -- each reference against the queries of its own "
             "class (D3 record: boost-weighted node mass, edge count)")
        r = R["D3"]
        for c in CLASSES:
            qs = [q for q in r["queries"] if q["true"] == c]
            ref = r["queries"][0]["pairs"][c]
            qn = sorted(q["pairs"][c]["q_node_total"] for q in qs)
            qe = sorted(q["pairs"][c]["n_query_edges"] for q in qs)
            emit("  %-6s reference: node mass %7.1f  edges %5d   |  its %2d "
                 "queries: node mass median %7.1f (min %6.1f, max %6.1f)  "
                 "edges median %5d (min %4d, max %5d)   reference/median "
                 "edges %.2f" % (
                     c, ref["c_node_total"], ref["n_core_edges"], len(qs),
                     qn[len(qn) // 2], qn[0], qn[-1], qe[len(qe) // 2],
                     qe[0], qe[-1], ref["n_core_edges"] / qe[len(qe) // 2]))
        allq = sorted(q["pairs"]["apple"]["n_query_edges"]
                      for q in r["queries"])
        emit("  all 75 queries: edges median %d (min %d, max %d)" % (
            allq[len(allq) // 2], allq[0], allq[-1]))
        emit("  absence, means per reference (own-class queries only / all "
             "queries):")
        for name in ("F6+D1", "D3"):
            rr = R[name]
            emit("    %-8s %s" % (name, "   ".join(
                "%s %.3f / %.3f" % (
                    c,
                    st.mean([q["pairs"][c]["edge_absence"]
                             for q in rr["queries"] if q["true"] == c]),
                    st.mean([q["pairs"][c]["edge_absence"]
                             for q in rr["queries"]]))
                for c in CLASSES)))
        emit("  relational term, means per reference (own-class queries only "
             "/ all queries):")
        for name in ("F6+D1", "D3"):
            rr = R[name]
            emit("    %-8s %s" % (name, "   ".join(
                "%s %.3f / %.3f" % (
                    c,
                    st.mean([q["pairs"][c]["relational_consistency"]
                             for q in rr["queries"] if q["true"] == c]),
                    st.mean([q["pairs"][c]["relational_consistency"]
                             for q in rr["queries"]]))
                for c in CLASSES)))

        # ------------------------------------------------------------ D
        emit("")
        emit("D. what the correspondence leaves unpaired -- one-sided binary "
             "coverage (matched core mass / core mass) against the count a")
        emit("   maximum pairing inside each lens type reaches "
             "(COUNT-ONLY formula), F6+D1 record")
        r = R["F6+D1"]
        b = [pr["node_coverage_binary"] for q in r["queries"]
             for pr in q["pairs"].values()]
        k = [pr["count_only"] for q in r["queries"]
             for pr in q["pairs"].values()]
        gap = sorted(x - y for x, y in zip(k, b))
        emit("  mean binary coverage %.4f   mean count formula %.4f   "
             "shortfall: mean %.4f  median %.4f  max %.4f  min %.4f   pairs "
             "with a shortfall > 0.01: %d of %d" % (
                 st.mean(b), st.mean(k), st.mean(gap), gap[len(gap) // 2],
                 gap[-1], gap[0], sum(int(g > 0.01) for g in gap), len(gap)))

        # ------------------------------------------------------------ E
        emit("")
        emit("E. queries the calibrated score cannot decide (four equal "
             "scores)")
        for name in ("NULL", "NULL+D3"):
            r = R[name]
            und = [q for q in r["queries"]
                   if len({q["pairs"][c]["score"] for c in CLASSES}) == 1]
            emit("  %-8s %d %s" % (name, len(und), "; ".join(
                "%s (true %s; node term %s; N*R %s; 0.15*A %s)" % (
                    q["name"], q["true"],
                    "/".join("%.3f" % q["pairs"][c]["node_coverage"]
                             for c in CLASSES),
                    "/".join("%.3f" % (q["pairs"][c]["node_coverage"]
                                       * q["pairs"][c]["relational_multiplier"])
                             for c in CLASSES),
                    "/".join("%.3f" % q["pairs"][c]["total_absence"]
                             for c in CLASSES))
                for q in und)))
        # ------------------------------------------------------------ F
        emit("")
        emit("F. is the chance level class-neutral?  N0 is the coverage of a "
             "random pairing on the two graphs' own similarity matrix")
        for name, bal in (("NULL", False), ("NULL+D3", True)):
            r = R[name]
            for q in r["queries"]:
                for pr in q["pairs"].values():
                    pr["N"] = pr["node_coverage_raw"]
                    pr["N0"] = pr["node_coverage_chance"]
                    if bal:
                        kb = ((pr["q_matched_nm_binary"]
                               + pr["c_matched_nm_binary"])
                              / (pr["q_node_total"] + pr["c_node_total"]))
                        kc = pr["count_only_balanced"]
                    else:
                        kb = pr["node_coverage_binary"]
                        kc = pr["count_only"]
                    pr["kb"], pr["kc"] = kb, kc
                    # mean signature similarity of the pairs the matcher
                    # kept, and of a random pairing (zeros included)
                    pr["s_matched"] = pr["N"] / kb if kb > 0 else 0.0
                    pr["s_chance"] = pr["N0"] / kc if kc > 0 else 0.0
                    pr["s_gain"] = pr["s_matched"] - pr["s_chance"]
            emit("  %s" % name)
            emit(row("  N0 alone (the chance level decides)", r, "N0"))
            emit(row("  N alone", r, "N", r, "N0", "N0 alone"))
            emit(row("  calibrated node term alone", r, "node_coverage", r,
                     "N0", "N0 alone"))
            emit("    r(N, N0): pooled %.3f   between references of one "
                 "query %.3f" % (st.pooled_r(r, "N", "N0"),
                                 st.within_query_r(r, "N", "N0")))
            for key, lab in (
                    ("s_matched", "mean similarity of the pairs kept"),
                    ("s_chance", "mean similarity of a random pairing"),
                    ("s_gain", "selection gain (kept - random)"),
                    ("kb", "share of the mass paired (binary)"),
                    ("kc", "share a maximum pairing covers")):
                v = st.variance_shares(r, key)
                emit("    %-38s own-class %.4f   wrong-class %.4f   "
                     "difference %+.4f   variance: which reference %5.1f%%, "
                     "own class %5.1f%%" % (
                         lab, v["mean_own"], v["mean_wrong"],
                         v["mean_own"] - v["mean_wrong"], 100 * v["ref"],
                         100 * v["own"]))
            for key, lab in (("s_matched", "similarity of the pairs kept"),
                             ("s_chance", "similarity of a random pairing"),
                             ("s_gain", "selection gain")):
                emit(row("  decided by " + lab, r, key))

    path = os.path.join(RESULTS_DIR, "phase2_addendum.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("written: %s" % path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
