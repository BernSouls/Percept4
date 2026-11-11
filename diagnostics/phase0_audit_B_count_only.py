"""PHASE 0 / AUDIT B (read-only) -- is node credit a per-lens-type COUNT?

Handoff findings under test:
  F2  node credit is binary: ``node_coverage`` is the per-lens-type node-count
      overlap of the two graphs;
  F3  populations cannot move the score: they decide who pairs with whom, not
      how much credit a pair earns.

The audit compares, on EVAL-75, four ways of deciding each query:

  evidence      argmax ``evidence_score``                      (the benchmark)
  node term     argmax of the matcher's own ``node_coverage``
  COUNT-ONLY    argmax of  sum_lens boost*min(n_q, n_c) / sum_lens boost*n_c
                -- computed from the two graphs' per-lens-type NODE COUNTS
                and nothing else (no signature, no position, no edge)
  graded node   argmax of the node term re-read with each credited pair
                counted by its signature similarity (OBSERVABLE ONLY: this is
                a measurement of what binary credit discards, not a change)

and then removes every population vector from both graphs (arm ``nopop``).

B5 re-decides every query from ONE term of the score at a time, using only
the per-pair trace values the baseline run stored (no matcher call):

  node term        node_coverage
  edge term        relational_consistency = matched_rel_sim_sum / n_verifiable
  absence term     n_verifiable / n_core_edges
  edges as COUNT   the score with every matched edge counted 1.0 instead of
                   its graded similarity  (F1: is edge geometry read?)

and prints the size of each reference next to how often each rule answers
it (F4: the score has no query-side term, so a small reference is easy to
cover).

B6 (OBSERVABLE ONLY, not a fix) asks how much class signal each term holds
once the reference's own offset is taken out: every quantity is re-read as
"this query's value for reference R, minus the mean value R receives from the
OTHER 74 queries" (leave-one-out), and the query is re-decided.  This uses
the query set's own statistics and EVAL-75 is in-sample, so it measures where
the signal is, not what a deployable score would reach.

    python diagnostics/phase0_audit_B_count_only.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import math
import os
from typing import Any, Dict, List

from _bootstrap import RESULTS_DIR


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def main() -> int:
    import _stats as st
    import eth80_benchmark as bm
    from matching.structural_similarity import GESTALT_LENS_BOOST

    out: List[str] = []
    P = out.append
    P("PHASE 0 / AUDIT B -- count-only classifier and population control"
      " (EVAL-75, in-sample)")
    P("")
    for window in bm.WINDOWS:
        base = bm.run_cached("eval", window, "baseline", breakdown=True)
        nopop = bm.run_cached("eval", window, "nopop", breakdown=True)
        s_ev = bm.summarize(base)
        s_nc = bm.summarize(base, "node_coverage")
        s_co = bm.summarize(base, "count_only")
        s_gr = bm.summarize(base, "node_sim_coverage")
        s_np = bm.summarize(nopop)

        P("=" * 78)
        P("window = %s" % window)
        P("=" * 78)
        P("B1. deciding each query four ways")
        P("  " + bm.fmt_row("evidence_score (baseline)", s_ev))
        P("  " + bm.fmt_row("node term (node_coverage)", s_nc))
        P("  " + bm.fmt_row("COUNT-ONLY (lens node counts)", s_co))
        P("  " + bm.fmt_row("graded node term (observable)", s_gr))
        n = s_ev["n"]

        def agree(a, b):
            return sum(int(x == y) for x, y in zip(a["preds"], b["preds"]))

        P("  same prediction on N of %d queries:" % n)
        P("     count-only vs node term      %d" % agree(s_co, s_nc))
        P("     count-only vs evidence_score %d" % agree(s_co, s_ev))
        P("     node term  vs evidence_score %d" % agree(s_nc, s_ev))
        P("  paired against evidence_score (queries gained / lost, exact"
          " McNemar):")
        for label, key in (("node term", "node_coverage"),
                           ("COUNT-ONLY", "count_only"),
                           ("graded node term", "node_sim_coverage")):
            g, l, pv = st.paired(base, base, None, key)
            tr = st.tie_report(base, key)
            P("     %-18s +%d/-%d  p=%.3f%s" % (
                label, g, l, pv,
                "" if not tr["ties"] else
                "   (%d ties; counted wrong %d, split %.2f)" % (
                    tr["ties"], tr["strict"], tr["split"])))
        P("  confusion, COUNT-ONLY:")
        P(bm.fmt_confusion(s_co))
        P("  confusion, evidence_score:")
        P(bm.fmt_confusion(s_ev))
        P("")

        # ---- B2: node_coverage vs the count formula, pair by pair -------
        nc, co, gr = [], [], []
        for q in base["queries"]:
            for rc in bm.CLASSES:
                pr = q["pairs"][rc]
                nc.append(pr["node_coverage"])
                co.append(pr["count_only"])
                gr.append(pr["node_sim_coverage"])
        diffs = [c - a for a, c in zip(nc, co)]
        P("B2. the matcher's node_coverage against the count formula,"
          " %d (query, reference) pairs" % len(nc))
        P("  exact (|diff| < 1e-9): %d/%d;  count-only >= node_coverage in"
          " %d/%d" % (sum(int(abs(d) < 1e-9) for d in diffs), len(diffs),
                      sum(int(d >= -1e-9) for d in diffs), len(diffs)))
        P("  count-only - node_coverage: mean %.4f  median %.4f  max %.4f"
          % (sum(diffs) / len(diffs), sorted(diffs)[len(diffs) // 2],
             max(diffs)))
        P("  Pearson r(node_coverage, count-only) = %.3f;"
          "  r(node_coverage, graded node term) = %.3f"
          % (_pearson(nc, co), _pearson(nc, gr)))
        P("  node_coverage  mean %.4f  (min %.4f, max %.4f);"
          "  count-only mean %.4f;  graded mean %.4f"
          % (sum(nc) / len(nc), min(nc), max(nc), sum(co) / len(co),
             sum(gr) / len(gr)))
        P("")

        # ---- B3: what binary credit discards ----------------------------
        P("B3. node credit by lens type (own-class reference pairs)")
        P(bm.node_breakdown_table(base, "true"))
        a_true = base["agg"]["node_true"]
        a_all = base["agg"]["node_all"]
        P("  mean signature similarity of CREDITED pairs, own-class reference"
          " vs the three wrong-class references:")
        P("  %-18s %10s %10s %9s %12s %12s" % (
            "lens", "own-class", "wrong", "delta", "cred/min own",
            "cred/min wrong"))
        tot = {"ts": 0.0, "tc": 0.0, "ws": 0.0, "wc": 0.0}
        for lt in sorted(a_all, key=lambda k: -a_all[k]["core"]):
            t = a_true.get(lt, {})
            tc, ts = t.get("credited", 0), t.get("sim_sum", 0.0)
            wc = a_all[lt]["credited"] - tc
            ws = a_all[lt]["sim_sum"] - ts
            tm = t.get("min_qc", 0)
            wm = a_all[lt]["min_qc"] - tm
            if not (tc or wc):
                continue
            b = GESTALT_LENS_BOOST.get(lt, 1.0)
            tot["ts"] += b * ts; tot["tc"] += b * tc
            tot["ws"] += b * ws; tot["wc"] += b * wc
            m_t = ts / tc if tc else float("nan")
            m_w = ws / wc if wc else float("nan")
            P("  %-18s %10.3f %10.3f %+9.3f %11.1f%% %11.1f%%" % (
                lt, m_t, m_w, m_t - m_w,
                100.0 * tc / tm if tm else 0.0,
                100.0 * wc / wm if wm else 0.0))
        P("  %-18s %10.3f %10.3f %+9.3f   (boost-weighted)" % (
            "ALL", tot["ts"] / tot["tc"], tot["ws"] / tot["wc"],
            tot["ts"] / tot["tc"] - tot["ws"] / tot["wc"]))
        P("")

        # ---- B4: populations removed ------------------------------------
        d_score, d_nc, changed = [], [], 0
        for qa, qb in zip(base["queries"], nopop["queries"]):
            assert qa["name"] == qb["name"]
            changed += int(qa["pred"] != qb["pred"])
            for rc in bm.CLASSES:
                d_score.append(qb["pairs"][rc]["score"]
                               - qa["pairs"][rc]["score"])
                d_nc.append(qb["pairs"][rc]["node_coverage"]
                            - qa["pairs"][rc]["node_coverage"])
        P("B4. every population vector removed from both graphs (arm nopop)")
        P("  " + bm.fmt_row("baseline", s_ev))
        P("  " + bm.fmt_row("nopop", s_np) + "   "
          + st.fmt_paired(*st.paired(base, nopop)))
        P("  predictions changed: %d of %d" % (changed, n))
        P("  score change per pair:         mean %+.5f  mean|.| %.5f"
          "  max|.| %.5f" % (sum(d_score) / len(d_score),
                             sum(abs(d) for d in d_score) / len(d_score),
                             max(abs(d) for d in d_score)))
        P("  node_coverage change per pair: mean %+.5f  mean|.| %.5f"
          "  max|.| %.5f" % (sum(d_nc) / len(d_nc),
                             sum(abs(d) for d in d_nc) / len(d_nc),
                             max(abs(d) for d in d_nc)))
        for label, res in (("baseline", base), ("nopop", nopop)):
            a = res["agg"]["node_all"]
            c = sum(r["credited"] for r in a.values())
            s = sum(r["sim_sum"] for r in a.values())
            e = res["agg"]["edge_all"]
            P("  %-8s credited node pairs %d (mean sig sim %.3f);"
              " verifiable edges %d, matched %d, sim_sum %.1f" % (
                  label, c, s / c if c else 0.0,
                  sum(r["verifiable"] for r in e.values()),
                  sum(r["matched"] for r in e.values()),
                  sum(r["sim_sum"] for r in e.values())))
        P("")

        # ---- B5: one term at a time (stored trace values only) -----------
        lam = 0.15                       # memory/cores.py, lambda_missing
        worst = 0.0
        for q in base["queries"]:
            for rc in bm.CLASSES:
                pr = q["pairs"][rc]
                nv = float(pr["verifiable_core_edges_count"])
                ne = float(pr["n_core_edges"])
                nm = float(pr["n_matched_edges"])
                pr["_verif_frac"] = (nv / ne) if ne > 0 else 0.0
                rc_count = (nm / nv) if nv > 0 else 0.0
                pr["_edges_as_count"] = max(0.0, min(1.0, (
                    pr["node_coverage"] * rc_count
                    - lam * pr["edge_absence"])))
                rebuilt = max(0.0, min(1.0, (
                    pr["node_coverage"] * pr["relational_consistency"]
                    - lam * pr["edge_absence"])))
                worst = max(worst, abs(rebuilt - pr["score"]))
        rules = [
            ("evidence_score (baseline)", None),
            ("node term alone", "node_coverage"),
            ("edge term alone (rel. consist.)", "relational_consistency"),
            ("absence term alone (verif. frac)", "_verif_frac"),
            ("score, matched edges as COUNT", "_edges_as_count"),
        ]
        P("B5. one term of the score at a time (stored trace values;"
          " score rebuilt from its terms to within %.1e)" % worst)
        for label, key in rules:
            sm = bm.summarize(base, key)
            pred_n = {c: sum(sm["confusion"][t].get(c, 0)
                             for t in bm.CLASSES) for c in bm.CLASSES}
            P("  " + bm.fmt_row(label, sm))
            P("  %-34s answers: %s%s" % (
                "", "  ".join("%s %d" % (c, pred_n[c]) for c in bm.CLASSES),
                "" if key is None else
                "   | same as evidence_score on %d of %d; +%d/-%d p=%.3f" % (
                    (agree(sm, s_ev), n) + st.paired(base, base, None, key))))
        P("  true class of the %d queries:      %s" % (
            n, "  ".join("%s %d" % (c, s_ev["per_class"][c][1])
                         for c in bm.CLASSES)))
        P("  mean of each term per REFERENCE, over all %d queries whatever"
          " their class:" % n)
        for label, key in (("score", "score"),
                           ("node_coverage", "node_coverage"),
                           ("relational_consistency",
                            "relational_consistency"),
                           ("edge_absence (penalised)", "edge_absence")):
            P("     %-26s %s" % (label, "  ".join(
                "%s %.4f" % (c, st.mean([q["pairs"][c][key]
                                         for q in base["queries"]]))
                for c in bm.CLASSES)))
        zero = sum(int(q["pairs"][c]["score"] == 0.0)
                   for q in base["queries"] for c in bm.CLASSES)
        gaps = sorted(
            sorted(q["pairs"][c]["score"] for c in bm.CLASSES)[-1]
            - sorted(q["pairs"][c]["score"] for c in bm.CLASSES)[-2]
            for q in base["queries"])
        P("  scores clamped to exactly 0.0: %d of %d pairs;  winner minus"
          " runner-up: median %.4f, below 0.005 on %d of %d queries" % (
              zero, 4 * n, gaps[len(gaps) // 2],
              sum(int(x < 0.005) for x in gaps), n))
        import eth80_frontend as fe
        from eth80_split import load_split

        refs = load_split()["references"]
        P("  reference sizes in this window (the score is normalised by the"
          " REFERENCE only):")
        for c in bm.CLASSES:
            sg, sigs = bm.make_window(fe.graph_for(c, refs[c]), window)
            P("     %-6s nodes %4d   edges %4d" % (
                c, len(sg.nodes), len(sg.edges)))
        qn = {c: [] for c in bm.CLASSES}
        for c in bm.CLASSES:
            for pth in load_split()["eval"][c]:
                sg, _sg = bm.make_window(fe.graph_for(c, pth), window)
                qn[c].append((len(sg.nodes), len(sg.edges)))
        P("  query sizes, mean nodes / mean edges by true class: " + "  ".join(
            "%s %.0f/%.0f" % (c, sum(a for a, _b in qn[c]) / len(qn[c]),
                              sum(b_ for _a, b_ in qn[c]) / len(qn[c]))
            for c in bm.CLASSES))
        P("")

        # ---- B6: the reference's own offset removed (observable) ---------
        P("B6. OBSERVABLE: each term minus the mean its reference receives"
          " from the other %d queries (leave-one-out)" % (n - 1))
        own = [q["pairs"][q["true"]]["score"] for q in base["queries"]]
        wrong = [q["pairs"][c]["score"] for q in base["queries"]
                 for c in bm.CLASSES if c != q["true"]]
        all_s = own + wrong
        m_all = st.mean(all_s)
        sst = sum((x - m_all) ** 2 for x in all_s)
        ss_ref = sum(
            n * (st.mean([q["pairs"][c]["score"] for q in base["queries"]])
                 - m_all) ** 2 for c in bm.CLASSES)
        ss_own = (len(own) * (st.mean(own) - m_all) ** 2
                  + len(wrong) * (st.mean(wrong) - m_all) ** 2)
        P("  mean score: own-class reference %.4f, wrong-class reference %.4f"
          % (st.mean(own), st.mean(wrong)))
        P("  share of the score's variance over the %d pairs explained by"
          " WHICH REFERENCE it is: %.1f%%;  by own-class vs not: %.1f%%" % (
              len(all_s), 100.0 * ss_ref / sst, 100.0 * ss_own / sst))
        for label, key in (("evidence_score", "score"),
                           ("node term", "node_coverage"),
                           ("edge term (rel. consist.)",
                            "relational_consistency"),
                           ("COUNT-ONLY", "count_only")):
            tot = {c: sum(q["pairs"][c][key] for q in base["queries"])
                   for c in bm.CLASSES}
            ckey = "_centred_" + key
            for q in base["queries"]:
                for c in bm.CLASSES:
                    v = q["pairs"][c][key]
                    q["pairs"][c][ckey] = v - (tot[c] - v) / (n - 1)
            raw = bm.summarize(base, None if key == "score" else key)
            cen = bm.summarize(base, ckey)
            pred_n = {c: sum(cen["confusion"][t].get(c, 0)
                             for t in bm.CLASSES) for c in bm.CLASSES}
            g, l, pv = st.paired(base, base,
                                 None if key == "score" else key, ckey)
            P("  %-26s as is %2d/%d  ->  offset removed %2d/%d  (%s)"
              "  answers %s   offset removed vs as is: +%d/-%d p=%.4f" % (
                  label, raw["hits"], n, cen["hits"], n,
                  " ".join("%s %d/%d" % (c, cen["per_class"][c][0],
                                         cen["per_class"][c][1])
                           for c in bm.CLASSES),
                  " ".join("%s %d" % (c, pred_n[c]) for c in bm.CLASSES),
                  g, l, pv))
        P("")

    text = "\n".join(out)
    print(text)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase0_audit_B.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
