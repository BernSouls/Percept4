"""STAGE 2 -- D3 (query-side terms), Phase 4 (unmeasured edge pairs) and the
null calibration of node coverage, on EVAL-75 (in-sample), both windows.

Every configuration is the tree's matcher under explicit parameter values, so
the tables do not depend on what the defaults are when the script is run.
All figures are read back from the STORED records (10 significant digits).

  0  continuity      F6+D1 with every new switch off reproduces stage 1
  1  Phase 4         EDGE_UNMEASURED_IS_NOT_EVIDENCE off / on
  2  D3              EVIDENCE_QUERY_SIDE_TERMS: (), node, node+absence, all
  3  null            EVIDENCE_NODE_NULL_CALIBRATED off / on, x D3 off / all

    python diagnostics/phase2_d3_p4_null.py [--window full|v1|both]

Paired figures are +gained/-lost queries against the configuration named,
with the exact two-sided McNemar p.  Ties are broken toward the first class
(apple); the split figure is the expected hits if ties were broken at random.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import _stats as st
from _bootstrap import RESULTS_DIR
from _stats import CLASSES

F6 = "EDGE_UNPARAMETERISED_STRUCTURAL_MATCH"
D1 = "EVIDENCE_GRADED_NODE_CREDIT"
QS = "EVIDENCE_QUERY_SIDE_TERMS"
P4 = "EDGE_UNMEASURED_IS_NOT_EVIDENCE"
NC = "EVIDENCE_NODE_NULL_CALIBRATED"
MX = "EDGE_REVERSE_FLIPS_MIXTURE"
MA = "MATCH_ASSIGNMENT_EXACT"
BL = "MATCH_SIGNATURE_MATRIX_BLOCKS"
CA = "EVIDENCE_RELATION_COUNT_APART"
LK = "EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS"
BR = "MATCH_ALIGN_BUDGET_REFINEMENT_ONLY"
AP = "MATCH_GLOBAL_ANCHOR_PRIOR"
SC = "EDGE_SPECIFICITY_CAP_SAME_SCHEMA"
ALL = ("node", "relation", "absence")

#: the stage-4 switches at the values every earlier table is measured under:
#: the block matrix on (the same floats as the per-pair loop -- that is its
#: criterion, checked by phase3_blocks_p4b.py), P4b off, the dark-path
#: parameters at their declared values (the dark path is never entered).
STAGE4_NEUTRAL: Dict[str, Any] = {BL: True, CA: False, LK: False, BR: True,
                                  AP: False, SC: False}


def cfg(qs=(), p4=False, nc=False, d1=True) -> Dict[str, Any]:
    # MX (mixture reversal) and MA (exact assignment) came after this stage
    # was measured: they are held at their historical values here, so these
    # tables are the ones registered and measured.  phase2_mixture_reversal.py
    # and phase2_exact_assignment.py measure them on their own.
    out = {F6: True, D1: d1, QS: tuple(qs), P4: p4, NC: nc, MX: False,
           MA: False}
    out.update(STAGE4_NEUTRAL)
    return out


#: name -> (file tag, overrides, one-line description)
CONFIGS: List[Tuple[str, str, Dict[str, Any], str]] = [
    ("F6+D1", "p2_R0", cfg(),
     "stage-1 state: every stage-2 switch off"),
    ("P4", "p2_P4", cfg(p4=True),
     "Phase 4 on"),
    ("D3:node", "p2_D3n", cfg(qs=("node",)),
     "decomposition arm: node term two-sided"),
    ("D3:node+abs", "p2_D3na", cfg(qs=("node", "absence")),
     "decomposition arm: node and absence two-sided"),
    ("D3", "p2_D3", cfg(qs=ALL),
     "PRIMARY: all three terms two-sided"),
    ("D3+P4", "p2_D3P4", cfg(qs=ALL, p4=True),
     "D3 primary with Phase 4 on"),
    ("D3,D1off", "p2_D3b", cfg(qs=ALL, d1=False),
     "control: D3 primary with binary node credit"),
    ("NULL", "p2_NULL", cfg(p4=True, nc=True),
     "null calibration, one-sided score (Phase 4 on)"),
    ("NULL+D3", "p2_NULLD3", cfg(qs=ALL, p4=True, nc=True),
     "null calibration with D3 primary (Phase 4 on)"),
    ("NULL,D1off", "p2_NULLb", cfg(p4=True, nc=True, d1=False),
     "mechanism check: null calibration of BINARY credit"),
]


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def add_derived(res: Dict[str, Any], balanced: bool) -> None:
    """Per pair, offline quantities used by the tables.

    count_formula   the count-only stand-in of the node formula in use
    N, N0           uncalibrated node term and its chance level (0 if none)
    excess          N - N0
    score_excess    the score with (N - N0) as the node term
    score_recomp    the score recomputed from the stored terms (self-check)
    """
    for q in res["queries"]:
        for c in CLASSES:
            pr = q["pairs"][c]
            pr["count_formula"] = (pr["count_only_balanced"] if balanced
                                   else pr["count_only"])
            n = pr.get("node_coverage_raw", pr["node_coverage"])
            n0 = pr.get("node_coverage_chance", 0.0)
            pr["N"], pr["N0"] = n, n0
            pr["excess"] = n - n0
            mult = pr["relational_multiplier"]
            pen = pr["total_absence"]
            pr["score_excess"] = clamp01((n - n0) * mult - pen)
            pr["score_recomp"] = clamp01(pr["node_coverage"] * mult - pen)


def acc_row(label: str, res: Dict[str, Any], key: Optional[str] = None,
            ref: Optional[Dict[str, Any]] = None,
            ref_key: Optional[str] = None, ref_name: str = "") -> str:
    h = st.hits(res, key)
    per = st.per_class_hits(res, key)
    ans = st.answers(res, key)
    t = st.tie_report(res, key)
    line = "  %-34s %2d/%d = %5.1f%%   %s   answers %-11s ties %2d" % (
        label, sum(h), len(h), 100.0 * sum(h) / len(h),
        " ".join("%s %2d/%-2d" % (c, per[c][0], per[c][1]) for c in CLASSES),
        "/".join("%d" % ans[c] for c in CLASSES), int(t["ties"]))
    if t["ties"]:
        line += " (split %.2f)" % t["split"]
    if ref is not None:
        g, l, p = st.paired(ref, res, ref_key, key)
        line += "   %+d/-%d p=%.3f%s" % (g, l, p,
                                         (" vs " + ref_name) if ref_name else "")
    return line


def shares_row(label: str, res: Dict[str, Any], key: Optional[str]) -> str:
    v = st.variance_shares(res, key)
    return ("  %-34s which reference %5.1f%%   own class %5.1f%%   "
            "mean own %.4f  wrong %.4f   per reference %s" % (
                label, 100.0 * v["ref"], 100.0 * v["own"], v["mean_own"],
                v["mean_wrong"],
                " ".join("%s %.4f" % (c, v["ref_means"][c]) for c in CLASSES)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", default="both", choices=("full", "v1", "both"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="phase2_d3_p4_null.txt")
    args = ap.parse_args()
    windows = ("full", "v1") if args.window == "both" else (args.window,)

    import eth80_benchmark as bm
    import eth80_frontend as fe
    import parameters as P

    out: List[str] = []

    def emit(s: str = "") -> None:
        print(s)
        sys.stdout.flush()
        out.append(s)

    emit("STAGE 2 -- D3, Phase 4, null calibration of node coverage -- "
         "EVAL-75 (in-sample)")
    emit("source digest %s   harness digest %s   front-end digest %s" % (
        bm.source_digest()[:12], bm.harness_digest()[:12],
        fe.frontend_source_digest()[:12]))
    emit("defaults in this tree: " + "  ".join(
        "%s=%r" % (k, getattr(P, k)) for k in (F6, D1, QS, P4, NC)))
    emit("configurations (explicit values; independent of the defaults):")
    for name, _tag, ov, desc in CONFIGS:
        emit("  %-11s D1=%-5s query-side=%-28s P4=%-5s null=%-5s  %s" % (
            name, ov[D1], ",".join(ov[QS]) or "()", ov[P4], ov[NC], desc))
    emit("")

    R: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for w in windows:
        for name, tag, ov, _desc in CONFIGS:
            bm.run_config(ov, window=w, breakdown=True,
                          workers=args.workers, tag=tag)
            with open(os.path.join(RESULTS_DIR, "%s_%s.json" % (tag, w)),
                      "r", encoding="utf-8") as fh:
                R[name, w] = json.load(fh)
            add_derived(R[name, w], balanced="node" in ov[QS])
            print("  [%s] %-11s %5.0fs" % (w, name, R[name, w]["seconds"]),
                  file=sys.stderr)

    verdict: Dict[str, Dict[str, Any]] = {w: {} for w in windows}

    for w in windows:
        base = R["F6+D1", w]
        emit("#" * 110)
        emit("# window = %s" % w)
        emit("#" * 110)

        # ---------------------------------------------------------------- 0
        emit("")
        emit("0. CONTINUITY -- F6+D1 with every stage-2 switch off, against "
             "the stored stage-1 record (p1_F6_D1_%s.json)" % w)
        p1_path = os.path.join(RESULTS_DIR, "p1_F6_D1_%s.json" % w)
        if os.path.isfile(p1_path):
            with open(p1_path, "r", encoding="utf-8") as fh:
                p1 = json.load(fh)
            cmp_ = st.compare_stored(p1, base)
            same_ans = len(base["queries"]) - st.changed_answers(p1, base)
            emit("  %d stored per-pair values compared (every key the two "
                 "records share): %d differ, largest difference %.3g%s; "
                 "same answer on %d/%d queries" % (
                     cmp_["n"], cmp_["differ"], cmp_["worst"],
                     (" on " + cmp_["worst_key"]) if cmp_["differ"] else "",
                     same_ans, len(base["queries"])))
            emit("  stage-1 source digest %s -> this source digest %s" % (
                p1["source_digest"][:12], base["source_digest"][:12]))
            verdict[w]["continuity"] = (cmp_["differ"] == 0)
        else:
            emit("  stage-1 record missing: %s" % p1_path)
            verdict[w]["continuity"] = None
        chk = max(abs(pr["score_recomp"] - pr["score"])
                  for name, _t, _o, _d in CONFIGS
                  for q in R[name, w]["queries"]
                  for pr in q["pairs"].values())
        emit("  self-check: score recomputed from its stored terms, every "
             "configuration, largest deviation %.2g" % chk)

        # ---------------------------------------------------------------- 1
        emit("")
        emit("=" * 110)
        emit("1. PHASE 4 -- an unmeasured edge pair is not evidence  "
             "(switch off -> on)")
        emit("=" * 110)
        for a_name, b_name, lab in (("F6+D1", "P4", "query-side terms off"),
                                    ("D3", "D3+P4", "query-side terms all on")):
            a, b = R[a_name, w], R[b_name, w]
            cmp_ = st.compare_stored(a, b)
            g, l, p = st.paired(a, b)
            emit("  %-24s %d stored per-pair values compared: %d differ "
                 "(largest %.3g%s); pairs whose SCORE differs %d of %d; "
                 "answers changed %d; accuracy %d -> %d (+%d/-%d p=%.3f)" % (
                     lab + ":", cmp_["n"], cmp_["differ"], cmp_["worst"],
                     (" on " + cmp_["worst_key"]) if cmp_["differ"] else "",
                     cmp_["score_differ"], 4 * len(a["queries"]),
                     st.changed_answers(a, b), sum(st.hits(a)),
                     sum(st.hits(b)), g, l, p))
            for which in ("true", "all"):
                ea, eb = a["agg"]["edge_" + which], b["agg"]["edge_" + which]
                moved = [(et, ea[et]["sim_sum"], eb[et]["sim_sum"],
                          ea[et]["matched"], eb[et]["matched"])
                         for et in sorted(ea)
                         if abs(ea[et]["sim_sum"] - eb[et]["sim_sum"]) > 1e-9
                         or ea[et]["matched"] != eb[et]["matched"]]
                emit("      core-side similarity sum per relation, %s pairs: "
                     "%s" % ("own-class" if which == "true" else "all",
                             "no relation moved" if not moved else ""))
                for et, sa, sb, ma, mb in moved:
                    emit("        %-26s sim_sum %.3f -> %.3f   matched "
                         "%d -> %d" % (et, sa, sb, ma, mb))
            if b_name == "D3+P4":
                qa = sum(pr.get("q_matched_rel_sim_sum", 0.0)
                         for q in a["queries"] for pr in q["pairs"].values())
                qb = sum(pr.get("q_matched_rel_sim_sum", 0.0)
                         for q in b["queries"] for pr in q["pairs"].values())
                emit("      query-side similarity sum, all pairs: "
                     "%.3f -> %.3f" % (qa, qb))
        a, b = R["F6+D1", w], R["P4", w]
        g, l, p = st.paired(a, b)
        a2, b2 = R["D3", w], R["D3+P4", w]
        g2, l2, p2 = st.paired(a2, b2)
        verdict[w]["p4_no_fall"] = not ((l > g and p < 0.05)
                                        or (l2 > g2 and p2 < 0.05))
        verdict[w]["p4_scores_moved"] = (
            st.compare_stored(a, b)["score_differ"],
            st.compare_stored(a2, b2)["score_differ"])

        # ---------------------------------------------------------------- 2
        emit("")
        emit("=" * 110)
        emit("2. D3 -- query-side terms  (Phase 4 off, null off; against "
             "F6+D1)")
        emit("=" * 110)
        d3_names = ["F6+D1", "D3:node", "D3:node+abs", "D3"]
        emit("2.1 accuracy, per class, answers apple/car/cup/horse")
        for nm in d3_names:
            emit(acc_row(nm + ("  [PRIMARY]" if nm == "D3" else ""),
                         R[nm, w], None,
                         base if nm != "F6+D1" else None, None, "F6+D1"))
        emit(acc_row("D3,D1off  [control]", R["D3,D1off", w], None, base,
                     None, "F6+D1"))
        emit("")
        emit("2.2 PRE-REGISTERED MEASURE 1 -- share of the variance over the "
             "300 pairs explained by which reference / by own class")
        emit("  evidence score")
        for nm in d3_names:
            emit(shares_row("  " + nm, R[nm, w], None))
        emit("  node term")
        for nm in d3_names:
            emit(shares_row("  " + nm, R[nm, w], "node_coverage"))
        emit("  relational term")
        for nm in ("F6+D1", "D3"):
            emit(shares_row("  " + nm, R[nm, w], "relational_consistency"))
        emit("  absence term")
        for nm in ("F6+D1", "D3:node+abs"):
            emit(shares_row("  " + nm, R[nm, w], "edge_absence"))
        sh0 = st.variance_shares(base)["ref"]
        sh1 = st.variance_shares(R["D3", w])["ref"]
        verdict[w]["d3_ref_share"] = (sh0, sh1)
        emit("")
        emit("2.3 OBSERVABLE (Audit B6, transductive): each score minus the "
             "mean its reference receives from the other 74 queries")
        for nm in d3_names:
            ck = st.add_centred(R[nm, w])
            emit(acc_row("  " + nm + ", offset removed", R[nm, w], ck,
                         R[nm, w], None, "as is"))
        emit("")
        emit("2.4 the D1 signal again -- each rule against the count-only "
             "stand-in of the SAME node formula")
        emit("    (one-sided: sum b*min(nq,nc) / sum b*nc;   two-sided: "
             "2*sum b*min(nq,nc) / (sum b*nq + sum b*nc))")
        for nm in ("F6+D1", "D3"):
            r = R[nm, w]
            emit("  %s" % nm)
            emit(acc_row("    COUNT-ONLY, same formula", r, "count_formula"))
            emit(acc_row("    node term alone", r, "node_coverage", r,
                         "count_formula", "COUNT-ONLY"))
            emit(acc_row("    evidence score", r, None, r, "count_formula",
                         "COUNT-ONLY"))
            emit("    r(node term, count formula): pooled %.3f   between "
                 "references of one query %.3f" % (
                     st.pooled_r(r, "node_coverage", "count_formula"),
                     st.within_query_r(r, "node_coverage", "count_formula")))
        r = R["D3,D1off", w]
        emit("  D3 with binary credit (control: the gap that is NOT D1's)")
        emit(acc_row("    node term alone", r, "node_coverage", r,
                     "count_formula", "COUNT-ONLY"))
        emit(acc_row("    evidence score", r, None, r, "count_formula",
                     "COUNT-ONLY"))
        g, l, p = st.paired(R["D3,D1off", w], R["D3", w])
        emit("    D1 under D3 (binary -> graded credit), evidence score: "
             "%d -> %d  +%d/-%d p=%.3f" % (
                 sum(st.hits(R["D3,D1off", w])), sum(st.hits(R["D3", w])),
                 g, l, p))
        emit("")
        emit("2.5 the three terms, means over own-class / wrong-class pairs")
        for nm in d3_names:
            r = R[nm, w]
            cells = []
            for key, lab in (("node_coverage", "node"),
                             ("relational_consistency", "relation"),
                             ("edge_absence", "absence"),
                             ("score", "score")):
                v = st.variance_shares(r, key)
                cells.append("%s %.4f / %.4f" % (lab, v["mean_own"],
                                                 v["mean_wrong"]))
            emit("  %-14s %s" % (nm, "   ".join(cells)))
        emit("")
        emit("2.6 absolute scale (the thresholds were set on the one-sided "
             "score: EVIDENCE_MIN_BALANCED 0.15, THETA_EVIDENCE_POS 0.35)")
        for nm in d3_names:
            r = R[nm, w]
            own = sorted(q["pairs"][q["true"]]["score"] for q in r["queries"])
            alls = [pr["score"] for q in r["queries"]
                    for pr in q["pairs"].values()]
            z = st.zero_report(r)
            emit("  %-14s own-class score: median %.4f  max %.4f   pairs "
                 ">= 0.15: %d of %d   >= 0.35: %d   exactly 0: %d   "
                 "undecidable queries: %d" % (
                     nm, own[len(own) // 2], own[-1],
                     sum(int(x >= 0.15) for x in alls), len(alls),
                     sum(int(x >= 0.35) for x in alls), z["pairs_zero"],
                     z["all_equal"]))
        g, l, p = st.paired(base, R["D3", w])
        verdict[w]["d3_acc"] = (sum(st.hits(base)), sum(st.hits(R["D3", w])),
                                g, l, p)

        # ---------------------------------------------------------------- 3
        emit("")
        emit("=" * 110)
        emit("3. NULL CALIBRATION of node coverage  (Phase 4 on; against the "
             "same configuration without calibration)")
        emit("=" * 110)
        pairs_nc = (("P4", "NULL", "one-sided score"),
                    ("D3+P4", "NULL+D3", "D3 primary"))
        emit("3.1 PRE-REGISTERED MEASURE 1 -- the chance level N0 next to the "
             "uncalibrated node term N (means)")
        for _off, on, lab in pairs_nc:
            r = R[on, w]
            vn, v0 = st.variance_shares(r, "N"), st.variance_shares(r, "N0")
            vc = st.variance_shares(r, "node_coverage")
            emit("  %-16s own-class: N %.4f  N0 %.4f (%.0f%% of N)  "
                 "calibrated %.4f   |   wrong-class: N %.4f  N0 %.4f "
                 "(%.0f%% of N)  calibrated %.4f" % (
                     lab, vn["mean_own"], v0["mean_own"],
                     100.0 * v0["mean_own"] / vn["mean_own"], vc["mean_own"],
                     vn["mean_wrong"], v0["mean_wrong"],
                     100.0 * v0["mean_wrong"] / vn["mean_wrong"],
                     vc["mean_wrong"]))
            emit("  %-16s per reference: N %s   N0 %s   calibrated %s" % (
                "", " ".join("%.3f" % vn["ref_means"][c] for c in CLASSES),
                " ".join("%.3f" % v0["ref_means"][c] for c in CLASSES),
                " ".join("%.3f" % vc["ref_means"][c] for c in CLASSES)))
        emit("")
        emit("3.2 PRE-REGISTERED MEASURE 2 -- variance explained by which "
             "reference / by own class")
        for off, on, lab in pairs_nc:
            emit("  %s" % lab)
            emit(shares_row("    score, uncalibrated", R[off, w], None))
            emit(shares_row("    score, calibrated", R[on, w], None))
            emit(shares_row("    node term, uncalibrated", R[off, w],
                            "node_coverage"))
            emit(shares_row("    node term, calibrated", R[on, w],
                            "node_coverage"))
            emit(shares_row("    chance level N0", R[on, w], "N0"))
        emit("")
        emit("3.3 PRE-REGISTERED MEASURE 3 -- r(node term, count formula)")
        for off, on, lab in pairs_nc:
            emit("  %-16s pooled %.3f -> %.3f   between references of one "
                 "query %.3f -> %.3f" % (
                     lab,
                     st.pooled_r(R[off, w], "node_coverage", "count_formula"),
                     st.pooled_r(R[on, w], "node_coverage", "count_formula"),
                     st.within_query_r(R[off, w], "node_coverage",
                                       "count_formula"),
                     st.within_query_r(R[on, w], "node_coverage",
                                       "count_formula")))
        emit("")
        emit("3.4 PRE-REGISTERED MEASURE 4 -- accuracy, ties split, and where "
             "the score stops deciding")
        for off, on, lab in pairs_nc:
            emit(acc_row(lab + ", uncalibrated", R[off, w]))
            emit(acc_row(lab + ", calibrated", R[on, w], None, R[off, w],
                         None, "uncalibrated"))
            z0, z1 = st.zero_report(R[off, w]), st.zero_report(R[on, w])
            emit("      pairs scoring exactly 0: %d -> %d of %d   own-class "
                 "pairs at 0: %d -> %d of %d   queries with four equal "
                 "scores: %d -> %d" % (
                     z0["pairs_zero"], z1["pairs_zero"],
                     4 * len(R[on, w]["queries"]), z0["own_zero"],
                     z1["own_zero"], len(R[on, w]["queries"]),
                     z0["all_equal"], z1["all_equal"]))
            ck = st.add_centred(R[on, w])
            emit(acc_row("      calibrated, offset removed (B6 observable)",
                         R[on, w], ck, R[on, w], None, "as is"))
        emit("")
        emit("3.5 PRE-REGISTERED MEASURE 5 -- the node term alone against "
             "COUNT-ONLY (uncalibrated count formula)")
        for off, on, lab in pairs_nc:
            emit("  %s" % lab)
            emit(acc_row("    COUNT-ONLY", R[on, w], "count_formula"))
            emit(acc_row("    node term, uncalibrated", R[off, w],
                         "node_coverage", R[off, w], "count_formula",
                         "COUNT-ONLY"))
            emit(acc_row("    node term, calibrated", R[on, w],
                         "node_coverage", R[on, w], "count_formula",
                         "COUNT-ONLY"))
            g, l, p = st.paired(R[off, w], R[on, w], "node_coverage",
                                "node_coverage")
            emit("      node term, calibrated vs uncalibrated: +%d/-%d "
                 "p=%.3f" % (g, l, p))
        emit("")
        emit("3.6 PRE-REGISTERED MEASURE 6 -- interaction with D3, and the "
             "two halves of the calibration (offline, not selected between)")
        h = {nm: sum(st.hits(R[nm, w]))
             for nm in ("P4", "D3+P4", "NULL", "NULL+D3")}
        emit("  evidence score, hits of 75:")
        emit("                         null off   null on")
        emit("    query-side off        %5d      %5d" % (h["P4"], h["NULL"]))
        emit("    query-side all        %5d      %5d" % (h["D3+P4"],
                                                         h["NULL+D3"]))
        emit("    interaction (both - D3 only - null only + neither): %+d "
             "queries" % (h["NULL+D3"] - h["D3+P4"] - h["NULL"] + h["P4"]))
        rs = {nm: st.variance_shares(R[nm, w])["ref"]
              for nm in ("P4", "D3+P4", "NULL", "NULL+D3")}
        emit("  reference share of the score's variance:")
        emit("    query-side off        %5.1f%%     %5.1f%%" % (
            100 * rs["P4"], 100 * rs["NULL"]))
        emit("    query-side all        %5.1f%%     %5.1f%%" % (
            100 * rs["D3+P4"], 100 * rs["NULL+D3"]))
        for off, on, lab in pairs_nc:
            emit("  %s -- node term alone, then the score built on it" % lab)
            emit(acc_row("    N (uncalibrated)", R[on, w], "N"))
            emit(acc_row("    N - N0", R[on, w], "excess", R[on, w], "N",
                         "N"))
            emit(acc_row("    (N - N0) / (1 - N0)", R[on, w],
                         "node_coverage", R[on, w], "N", "N"))
            emit(acc_row("    score with N", R[off, w]))
            emit(acc_row("    score with N - N0", R[on, w], "score_excess",
                         R[off, w], None, "N"))
            emit(acc_row("    score with (N - N0)/(1 - N0)", R[on, w], None,
                         R[off, w], None, "N"))
        emit("")
        emit("3.7 mechanism check -- BINARY credit, calibrated (a count is "
             "what a random pairing earns)")
        r = R["NULL,D1off", w]
        vn, v0 = st.variance_shares(r, "N"), st.variance_shares(r, "N0")
        vc = st.variance_shares(r, "node_coverage")
        z = st.zero_report(r)
        emit("  mean N %.4f / %.4f (own / wrong)   N0 %.4f / %.4f   "
             "calibrated node term %.4f / %.4f" % (
                 vn["mean_own"], vn["mean_wrong"], v0["mean_own"],
                 v0["mean_wrong"], vc["mean_own"], vc["mean_wrong"]))
        emit(acc_row("  evidence score", r))
        emit("      pairs scoring exactly 0: %d of %d   queries with four "
             "equal scores: %d" % (z["pairs_zero"], 4 * len(r["queries"]),
                                   z["all_equal"]))
        for off, on, key in pairs_nc:
            g, l, p = st.paired(R[off, w], R[on, w])
            z1 = st.zero_report(R[on, w])
            verdict[w]["null_" + on] = {
                "ref_share": (st.variance_shares(R[off, w])["ref"],
                              st.variance_shares(R[on, w])["ref"]),
                "acc": (sum(st.hits(R[off, w])), sum(st.hits(R[on, w])),
                        g, l, p),
                "all_equal": z1["all_equal"]}
        emit("")

    # ------------------------------------------------------------ verdicts
    emit("#" * 110)
    emit("# PRE-REGISTERED RULES, read off the tables above (pins are "
         "checked separately)")
    emit("#" * 110)
    for w in windows:
        v = verdict[w]
        emit("window = %s" % w)
        emit("  continuity with stage 1: %s" % (
            "exact" if v["continuity"] else
            ("NOT EXACT" if v["continuity"] is False else "not checked")))
        emit("  Phase 4: pairs whose score moved %d (query-side off), %d "
             "(query-side on); accuracy falls with p<0.05: %s" % (
                 v["p4_scores_moved"][0], v["p4_scores_moved"][1],
                 "no" if v["p4_no_fall"] else "YES"))
        a0, a1, g, l, p = v["d3_acc"]
        emit("  D3 primary: reference share of the score's variance "
             "%.1f%% -> %.1f%% (%s);  accuracy %d -> %d, +%d/-%d p=%.3f "
             "(falls with p<0.05: %s)" % (
                 100 * v["d3_ref_share"][0], 100 * v["d3_ref_share"][1],
                 "falls" if v["d3_ref_share"][1] < v["d3_ref_share"][0]
                 else "DOES NOT FALL", a0, a1, g, l, p,
                 "YES" if (l > g and p < 0.05) else "no"))
        for on, lab in (("NULL", "null, one-sided"),
                        ("NULL+D3", "null, with D3")):
            n = v["null_" + on]
            a0, a1, g, l, p = n["acc"]
            emit("  %-16s reference share %.1f%% -> %.1f%% (%s);  accuracy "
                 "%d -> %d, +%d/-%d p=%.3f (falls with p<0.05: %s);  "
                 "queries with four equal scores: %d" % (
                     lab + ":", 100 * n["ref_share"][0],
                     100 * n["ref_share"][1],
                     "falls" if n["ref_share"][1] < n["ref_share"][0]
                     else "DOES NOT FALL", a0, a1, g, l, p,
                     "YES" if (l > g and p < 0.05) else "no",
                     n["all_equal"]))
    path = os.path.join(RESULTS_DIR, args.out)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("written: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
