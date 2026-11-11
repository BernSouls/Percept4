"""PHASE 1 (F6) and D1 -- before/after on EVAL-75, one tree, four settings.

Two library changes, each behind one general parameter (default = new):

  F6   parameters.EDGE_UNPARAMETERISED_STRUCTURAL_MATCH
       polymorphic_edge_similarity: two same-predicate edges that both carry
       no EdgeParams read 1.0 (was 0.0 on both branches)
  D1   parameters.EVIDENCE_GRADED_NODE_CREDIT
       evidence_score: a credited node pair earns
       boost * node_weight * signature_similarity (was boost * node_weight)

Measured settings:  OFF (both False)   F6   D1   F6+D1 (the new default)

What this prints
  0  REPRODUCTION  OFF must give back the frozen Phase 0 baseline, exactly,
     on every per-pair quantity (this is also the proof that the rebuilt
     front-end graphs are the Phase 0 graphs)
  1  accuracy, per class, answers, paired against OFF -- every setting
  2  F6: which edges it touches and what moves
  3  D1: the credited mass, the offline prediction, and the SUCCESS SIGNAL
     the handoff set (the count-only stand-in must fall far behind)
  4  why D1 moves what it moves: how much similarity varies between references

Every figure is in-sample (EVAL-75, references fixed).  n = 75: a difference
of a few queries is noise; comparisons are paired on the same 75 queries.

    python diagnostics/phase1_f6_d1.py [--window full|v1|both]
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import RESULTS_DIR
from _stats import CLASSES, hits, mcnemar_exact, paired, tie_report, winners

F6 = "EDGE_UNPARAMETERISED_STRUCTURAL_MATCH"
D1 = "EVIDENCE_GRADED_NODE_CREDIT"
CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    ("OFF", {F6: False, D1: False}),
    ("F6", {F6: True, D1: False}),
    ("D1", {F6: False, D1: True}),
    ("F6+D1", {F6: True, D1: True}),
]
PHASE0 = os.path.join(RESULTS_DIR, "phase0")


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def within_query_r(res: Dict[str, Any], ka: str, kb: str) -> float:
    """Pearson r after removing each query's own mean from both quantities:
    only the differences BETWEEN the four references of one query -- the only
    thing an argmax over references can read."""
    xs: List[float] = []
    ys: List[float] = []
    for q in res["queries"]:
        a = [q["pairs"][c][ka] for c in CLASSES]
        b = [q["pairs"][c][kb] for c in CLASSES]
        ma, mb = sum(a) / 4, sum(b) / 4
        xs += [v - ma for v in a]
        ys += [v - mb for v in b]
    return pearson(xs, ys)


def pooled_r(res: Dict[str, Any], ka: str, kb: str) -> float:
    xs = [q["pairs"][c][ka] for q in res["queries"] for c in CLASSES]
    ys = [q["pairs"][c][kb] for q in res["queries"] for c in CLASSES]
    return pearson(xs, ys)


def acc_row(label: str, res: Dict[str, Any], key: Optional[str] = None,
            ref: Optional[Dict[str, Any]] = None,
            ref_key: Optional[str] = None) -> str:
    h = hits(res, key)
    per = {c: [0, 0] for c in CLASSES}
    ans: Counter = Counter()
    for q, hit in zip(res["queries"], h):
        per[q["true"]][0] += hit
        per[q["true"]][1] += 1
        ans[winners(q, key)[0][0]] += 1
    t = tie_report(res, key)
    line = "  %-30s %2d/%d = %5.1f%%   %s   answers %s   ties %d" % (
        label, sum(h), len(h), 100.0 * sum(h) / len(h),
        "  ".join("%s %2d/%-2d" % (c, per[c][0], per[c][1]) for c in CLASSES),
        "/".join("%d" % ans.get(c, 0) for c in CLASSES), int(t["ties"]))
    if ref is not None:
        g, l, p = paired(ref, res, ref_key, key)
        line += "   +%d/-%d p=%.3f" % (g, l, p)
    return line


def confusion(res: Dict[str, Any], key: Optional[str] = None) -> str:
    conf = {c: Counter() for c in CLASSES}
    for q in res["queries"]:
        conf[q["true"]][winners(q, key)[0][0]] += 1
    out = ["      true\\answer " + "".join("%7s" % c for c in CLASSES)]
    for c in CLASSES:
        out.append("      %-10s   " % c + "".join(
            "%7d" % conf[c].get(p, 0) for p in CLASSES))
    return "\n".join(out)


def add_derived(res: Dict[str, Any]) -> None:
    """Per pair: `mean_sim` = graded / binary node coverage of the same
    correspondence = the boost-weighted mean signature similarity of the
    credited pairs (0 when nothing was credited)."""
    for q in res["queries"]:
        for c in CLASSES:
            pr = q["pairs"][c]
            b = pr.get("node_coverage_binary", pr["node_coverage"])
            g = pr.get("node_sim_coverage", pr["node_coverage"])
            pr["cov_binary"] = b
            pr["cov_graded"] = g
            pr["mean_sim"] = (g / b) if b > 0 else 0.0


def changed_answers(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    return sum(int(winners(qa)[0][0] != winners(qb)[0][0])
               for qa, qb in zip(a["queries"], b["queries"]))


def edge_rows(res: Dict[str, Any], which: str, types: List[str]) -> List[str]:
    agg = res["agg"]["edge_" + which]
    out = []
    for et in types:
        r = agg.get(et)
        if r is None:
            out.append("      %-12s (no core edge of this type)" % et)
            continue
        out.append("      %-12s core %6d  verifiable %6d  no query edge %6d  "
                   "matched %5d  sim_sum %8.1f  no-param core edges %6d" % (
                       et, r["n"], r["verifiable"], r["no_query_edge"],
                       r["matched"], r["sim_sum"], r["c_params_none"]))
    tot = {k: sum(r[k] for r in agg.values())
           for k in ("n", "verifiable", "matched", "sim_sum")}
    out.append("      %-12s core %6d  verifiable %6d  %21s matched %5d  "
               "sim_sum %8.1f" % ("ALL TYPES", tot["n"], tot["verifiable"],
                                  "", tot["matched"], tot["sim_sum"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", default="both", choices=("full", "v1", "both"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    windows = ("full", "v1") if args.window == "both" else (args.window,)

    import eth80_benchmark as bm
    import parameters as P

    out: List[str] = []

    def emit(s: str = "") -> None:
        print(s)
        sys.stdout.flush()
        out.append(s)

    emit("PHASE 1 (F6) + D1 -- before / after, EVAL-75 (in-sample)")
    emit("source digest %s   harness digest %s" % (bm.source_digest()[:12],
                                                  bm.harness_digest()[:12]))
    emit("defaults in this tree: %s=%r  %s=%r" % (F6, getattr(P, F6),
                                                  D1, getattr(P, D1)))
    emit("settings: " + "   ".join(
        "%s(%s)" % (n, ", ".join("%s=%s" % (k.split("_")[0], v)
                                 for k, v in o.items())) for n, o in CONFIGS))
    emit("paired figures: +gained/-lost queries against the setting named, "
         "exact two-sided McNemar p")
    emit("")

    R: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for w in windows:
        for name, ov in CONFIGS:
            tag = "p1_" + name.replace("+", "_")
            bm.run_config(ov, window=w, breakdown=True,
                          workers=args.workers, tag=tag)
            # always read the STORED record (10 significant digits), so a
            # fresh run and a re-used one print the same tables and compare
            # like with like against the frozen Phase 0 files
            with open(os.path.join(RESULTS_DIR, "%s_%s.json" % (tag, w)),
                      "r", encoding="utf-8") as fh:
                R[name, w] = json.load(fh)
            add_derived(R[name, w])

    # ------------------------------------------------------------------ 0
    emit("=" * 100)
    emit("0. REPRODUCTION -- the OFF setting against the frozen Phase 0 "
         "baseline (pre-fix source)")
    emit("=" * 100)
    ok_all = True
    for w in windows:
        path = os.path.join(PHASE0, "eval_baseline_%s.json" % w)
        if not os.path.isfile(path):
            emit("  [%s] frozen file missing: %s" % (w, path))
            ok_all = False
            continue
        with open(path, "r", encoding="utf-8") as fh:
            old = json.load(fh)
        new = R["OFF", w]
        n_cmp = n_diff = 0
        worst = 0.0
        worst_key = ""
        names_ok = [a["name"] for a in old["queries"]] == \
                   [b["name"] for b in new["queries"]]
        for qa, qb in zip(old["queries"], new["queries"]):
            for c in CLASSES:
                pa, pb = qa["pairs"][c], qb["pairs"][c]
                for k, va in pa.items():
                    if k not in pb or not isinstance(va, (int, float)):
                        continue
                    n_cmp += 1
                    d = abs(float(va) - float(pb[k]))
                    if d != 0.0:
                        n_diff += 1
                        if d > worst:
                            worst, worst_key = d, k
        same_pred = sum(int(a["pred"] == b["pred"])
                        for a, b in zip(old["queries"], new["queries"]))
        agg_same = (old.get("agg") == new.get("agg"))
        ok = names_ok and n_diff == 0 and same_pred == len(new["queries"])
        ok_all = ok_all and ok
        emit("  [%-4s] Phase 0 source %s / front end %s   ->   now %s / %s"
             % (w, old["source_digest"][:12], old["frontend_digest"][:12],
                new["source_digest"][:12], new["frontend_digest"][:12]))
        emit("         per-pair values compared %d (stored to 10 significant "
             "digits)   differing %d   largest difference %.3g%s" % (
                 n_cmp, n_diff, worst,
                 (" (%s)" % worst_key) if worst_key else ""))
        emit("         same answer %d/%d   per-type edge and node tables "
             "identical: %s   =>  %s" % (
                 same_pred, len(new["queries"]), agg_same,
                 "REPRODUCED EXACTLY" if ok and agg_same else
                 ("scores reproduced" if ok else "NOT REPRODUCED")))
    emit("")
    if not ok_all:
        emit("  !! OFF does not reproduce Phase 0: nothing below may be read "
             "as a before/after.")
        emit("")

    # ------------------------------------------------------------------ 1
    emit("=" * 100)
    emit("1. ACCURACY PER SETTING  (evidence score; paired against OFF)")
    emit("=" * 100)
    for w in windows:
        emit("  window = %s" % w)
        for name, _ov in CONFIGS:
            emit(acc_row(name, R[name, w],
                         ref=None if name == "OFF" else R["OFF", w]))
        for name in ("OFF", "F6+D1"):
            emit("    confusion, %s" % name)
            emit(confusion(R[name, w]))
        emit("")

    # ------------------------------------------------------------------ 2
    emit("=" * 100)
    emit("2. F6 -- unparameterised same-predicate edges (OFF -> F6; node "
         "credit binary in both)")
    emit("=" * 100)
    for w in windows:
        a, b = R["OFF", w], R["F6", w]
        types = sorted(et for et, r in a["agg"]["edge_all"].items()
                       if r.get("c_params_none", 0) > 0)
        emit("  window = %s   core-edge types with at least one param-less "
             "core edge: %s" % (w, ", ".join(types) or "(none)"))
        for which, lab in (("true", "own-class pairs (75)"),
                           ("all", "every pair (300)")):
            emit("    %s -- OFF" % lab)
            for line in edge_rows(a, which, types):
                emit(line)
            emit("    %s -- F6" % lab)
            for line in edge_rows(b, which, types):
                emit(line)
        deltas = [b["queries"][i]["pairs"][c]["score"]
                  - a["queries"][i]["pairs"][c]["score"]
                  for i in range(len(a["queries"])) for c in CLASSES]
        moved = [d for d in deltas if d != 0.0]
        emit("    pairs whose score moved: %d of %d   largest |change| %.6f   "
             "mean change over moved pairs %+.6f" % (
                 len(moved), len(deltas),
                 max((abs(d) for d in moved), default=0.0),
                 (sum(moved) / len(moved)) if moved else 0.0))
        emit("    answers that changed: %d of %d" % (
            changed_answers(a, b), len(a["queries"])))
        emit(acc_row("OFF", a))
        emit(acc_row("F6", b, ref=a))
        emit("")

    # ------------------------------------------------------------------ 3
    emit("=" * 100)
    emit("3. D1 -- similarity-weighted node credit (OFF -> D1; F6 off in "
         "both)")
    emit("=" * 100)
    for w in windows:
        a, b = R["OFF", w], R["D1", w]
        emit("  window = %s" % w)
        # 3a credited mass
        own_b = [q["pairs"][q["true"]]["cov_binary"] for q in b["queries"]]
        own_g = [q["pairs"][q["true"]]["node_coverage"] for q in b["queries"]]
        all_b = [q["pairs"][c]["cov_binary"] for q in b["queries"]
                 for c in CLASSES]
        all_g = [q["pairs"][c]["node_coverage"] for q in b["queries"]
                 for c in CLASSES]
        emit("    3a node term (mean over pairs)       binary credit   graded "
             "credit   graded/binary")
        emit("       own-class pairs (75)                 %8.4f      %8.4f   "
             "     %6.4f" % (sum(own_b) / 75, sum(own_g) / 75,
                             sum(own_g) / sum(own_b)))
        emit("       every pair (300)                     %8.4f      %8.4f   "
             "     %6.4f" % (sum(all_b) / 300, sum(all_g) / 300,
                             sum(all_g) / sum(all_b)))
        # the correspondence is untouched: binary coverage under D1 == OFF's
        same_corr = max(abs(qb["pairs"][c]["cov_binary"]
                            - qa["pairs"][c]["node_coverage"])
                        for qa, qb in zip(a["queries"], b["queries"])
                        for c in CLASSES)
        same_edges = max(abs(qb["pairs"][c]["relational_consistency"]
                             - qa["pairs"][c]["relational_consistency"])
                         for qa, qb in zip(a["queries"], b["queries"])
                         for c in CLASSES)
        emit("       unchanged by D1: binary coverage of the correspondence "
             "(max |diff| %.3g), edge term (max |diff| %.3g)"
             % (same_corr, same_edges))
        # 3b offline prediction from the Phase 0 record
        worst = 0.0
        for qa, qb in zip(a["queries"], b["queries"]):
            for c in CLASSES:
                pa = qa["pairs"][c]
                pred = min(1.0, max(0.0, pa["node_sim_coverage"]
                                    * pa["relational_consistency"]
                                    - 0.15 * pa["edge_absence"]))
                worst = max(worst, abs(pred - qb["pairs"][c]["score"]))
        emit("    3b offline prediction (Phase 0 record: graded node "
             "observable x edge term - 0.15 x absence)")
        emit("       against the implemented D1 score, 300 pairs: largest "
             "|difference| %.3g" % worst)
        # 3c success signal
        emit("    3c SUCCESS SIGNAL -- \"the count-only classifier must fall "
             "far below baseline after D1\"")
        emit("       COUNT-ONLY reads the two per-lens node censuses and "
             "nothing else; it never calls the")
        emit("       matcher, so its own accuracy cannot move. The signal is "
             "the GAP to it.")
        emit(acc_row("evidence score, OFF", a))
        emit(acc_row("evidence score, D1", b, ref=a))
        emit(acc_row("COUNT-ONLY (no matcher)", a, "count_only"))
        emit(acc_row("node term alone, OFF", a, "node_coverage"))
        emit(acc_row("node term alone, D1", b, "node_coverage",
                     ref=a, ref_key="node_coverage"))
        emit(acc_row("mean similarity alone (*)", b, "mean_sim",
                     ref=a, ref_key=None))
        emit("       (*) OBSERVABLE, not a proposal: argmax of the "
             "boost-weighted mean signature similarity of the")
        emit("           credited pairs (graded / binary coverage). It "
             "ignores every unmatched node. Paired against")
        emit("           the OFF evidence score.")
        emit("       gap to COUNT-ONLY, paired on the same 75 queries "
             "(+ = queries the first gets right and COUNT-ONLY wrong):")
        for lab, res, key in (("evidence OFF", a, None),
                              ("evidence D1 ", b, None),
                              ("node term OFF", a, "node_coverage"),
                              ("node term D1 ", b, "node_coverage")):
            l, g, p = paired(res, a, key, "count_only")   # g: count-only wins
            n_same = sum(int(winners(q, key)[0][0]
                             == winners(q2, "count_only")[0][0])
                         for q, q2 in zip(res["queries"], a["queries"]))
            emit("         %-14s %+3d queries (+%d/-%d, p=%.3f)   same "
                 "answer as COUNT-ONLY on %d/75" % (
                     lab, sum(hits(res, key)) - sum(hits(a, "count_only")),
                     g, l, p, n_same))
        emit("       r(node term, COUNT-ONLY formula)      pooled 300 pairs"
             "   between references of one query")
        emit("         node term OFF (binary)                  %6.3f"
             "                 %6.3f" % (
                 pooled_r(a, "node_coverage", "count_only"),
                 within_query_r(a, "node_coverage", "count_only")))
        emit("         node term D1 (graded)                   %6.3f"
             "                 %6.3f" % (
                 pooled_r(b, "node_coverage", "count_only"),
                 within_query_r(b, "node_coverage", "count_only")))
        emit("         r(node term OFF, node term D1)          %6.3f"
             "                 %6.3f" % (
                 pearson([q["pairs"][c]["node_coverage"]
                          for q in a["queries"] for c in CLASSES],
                         [q["pairs"][c]["node_coverage"]
                          for q in b["queries"] for c in CLASSES]),
                 within_query_r(b, "cov_binary", "node_coverage")))
        n_co = sum(hits(a, "count_only"))
        g1, l1, p1 = paired(a, b, "count_only", None)       # with D1
        g0, l0, p0 = paired(a, a, "count_only", None)       # control: D1 off
        gap1, gap0 = sum(hits(b)) - n_co, sum(hits(a)) - n_co
        crit1 = gap1 > 0 and p1 < 0.05
        crit0 = gap0 > 0 and p0 < 0.05
        n_chg = changed_answers(a, b)
        emit("       READING")
        emit("         criterion written into this script BEFORE the run: "
             "evidence(D1) ahead of COUNT-ONLY, paired p < 0.05")
        emit("           with D1            %+3d queries, p=%.3f   -> %s" % (
            gap1, p1, "holds" if crit1 else "does not hold"))
        emit("           control, D1 OFF    %+3d queries, p=%.3f   -> %s" % (
            gap0, p0, "holds" if crit0 else "does not hold"))
        emit("         gap OPENED by D1     %+3d queries   (answers that "
             "changed OFF -> D1: %d of 75)" % (gap1 - gap0, n_chg))
        if crit1 and not crit0:
            verdict = "SIGNAL MET"
        elif crit1 and crit0:
            verdict = ("SIGNAL NOT MET -- the gap is there with D1 off as "
                       "well (it is the edge term's, not D1's). The")
        else:
            verdict = "SIGNAL NOT MET"
        emit("         => %s" % verdict)
        if crit1 and crit0:
            emit("            criterion as first written had no D1-off "
                 "control and would have read \"met\"; the control was")
            emit("            added AFTER seeing this result, which is said "
                 "here rather than hidden.")
        emit("")

    # ------------------------------------------------------------------ 4
    emit("=" * 100)
    emit("4. WHY -- how much the two factors of the graded node term vary "
         "BETWEEN the four references")
    emit("   graded node term = (binary coverage) x (mean similarity of the "
         "credited pairs)")
    emit("=" * 100)
    for w in windows:
        b = R["D1", w]
        emit("  window = %s" % w)
        emit("    per reference (mean over 75 queries)   binary coverage   "
             "mean similarity   graded coverage")
        for c in CLASSES:
            emit("      %-10s                            %8.4f        %8.4f "
                 "        %8.4f" % (
                     c,
                     sum(q["pairs"][c]["cov_binary"] for q in b["queries"]) / 75,
                     sum(q["pairs"][c]["mean_sim"] for q in b["queries"]) / 75,
                     sum(q["pairs"][c]["node_coverage"]
                         for q in b["queries"]) / 75))
        # within-query spread, in log units (the two factors add in logs)
        v_bin = v_sim = cov = 0.0
        rng_bin: List[float] = []
        rng_sim: List[float] = []
        n_used = 0
        for q in b["queries"]:
            lb = [q["pairs"][c]["cov_binary"] for c in CLASSES]
            ls = [q["pairs"][c]["mean_sim"] for c in CLASSES]
            if min(lb) <= 0 or min(ls) <= 0:
                continue
            n_used += 1
            rng_bin.append(max(lb) / min(lb))
            rng_sim.append(max(ls) / min(ls))
            lb = [math.log(x) for x in lb]
            ls = [math.log(x) for x in ls]
            mb, ms = sum(lb) / 4, sum(ls) / 4
            v_bin += sum((x - mb) ** 2 for x in lb)
            v_sim += sum((x - ms) ** 2 for x in ls)
            cov += sum((x - mb) * (y - ms) for x, y in zip(lb, ls))
        tot = v_bin + v_sim + 2 * cov
        emit("    between-reference variance of log(graded node term), "
             "%d queries with all four references credited:" % n_used)
        emit("      from binary coverage %5.1f%%   from mean similarity "
             "%5.1f%%   from their covariance %+5.1f%%" % (
                 100 * v_bin / tot, 100 * v_sim / tot, 200 * cov / tot))
        rng_bin.sort()
        rng_sim.sort()
        emit("      best/worst reference ratio, median over queries: binary "
             "coverage x%.3f   mean similarity x%.3f" % (
                 rng_bin[len(rng_bin) // 2], rng_sim[len(rng_sim) // 2]))
        own = [q["pairs"][q["true"]]["mean_sim"] for q in b["queries"]]
        oth = [q["pairs"][c]["mean_sim"] for q in b["queries"]
               for c in CLASSES if c != q["true"]]
        emit("      mean similarity of credited pairs: own-class reference "
             "%.4f   wrong-class references %.4f" % (
                 sum(own) / len(own), sum(oth) / len(oth)))
        emit("")

    path = os.path.join(RESULTS_DIR, "phase1_f6_d1.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("written: %s" % path)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
