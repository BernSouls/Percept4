"""EXACT ASSIGNMENT -- MATCH_ASSIGNMENT_EXACT against its pre-registration.

The node correspondence was an approximate maximum-weight matching (per-row
top 8, greedy, 2-opt).  REPORT.md registers what is measured when it is
solved exactly; this script reports measures 4-6 from STORED records.
Measures 1-2 (stranding, total weight) are in
``phase2_correspondence_census.py`` and measure 3 (symmetry) in
``phase2_symmetry_probe.py``, both of which now run under either value.

Configurations (every switch explicit; EVAL-75, in-sample, both windows):

  BASE           p2m_BASE   F6, D1, Phase 4, mixture reversal on;
                            D3 and calibration off; APPROXIMATE assignment
  DEFAULTS       p2x_DEF    the same with the EXACT assignment
                            (what the tree declares)
  D3             p2x_D3h    BASE with all three terms two-sided
  D3 exact       p2x_D3     the same with the exact assignment

    python diagnostics/phase2_exact_assignment.py [--window full|v1|both]

Paired figures are +gained/-lost queries, exact two-sided McNemar p.  Ties
are broken toward the first class; "split" is the expected number of hits if
they were broken at random.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import os
import sys
from typing import Any, Dict, List

import _stats as st
from _bootstrap import RESULTS_DIR
from _stats import CLASSES
from phase2_d3_p4_null import (ALL, D1, F6, MA, MX, NC, P4, QS,
                               STAGE4_NEUTRAL)


def base(qs=(), exact=False, d1=True) -> Dict[str, Any]:
    out = {F6: True, D1: d1, QS: tuple(qs), P4: True, NC: False, MX: True,
           MA: exact}
    out.update(STAGE4_NEUTRAL)
    return out


CONFIGS = [
    ("BASE", "p2m_BASE", base()),
    ("DEFAULTS", "p2x_DEF", base(exact=True)),
    ("D3", "p2x_D3h", base(qs=ALL)),
    ("D3 exact", "p2x_D3", base(qs=ALL, exact=True)),
]

#: CONTROL for the "D1 signal" line, added after the first full run showed
#: the node term ahead of the count formula on one window under D3 with the
#: exact assignment: the same two configurations under BINARY credit.  If
#: the lead is there with binary credit too, it is not D1's.
CONTROLS = [
    ("DEFAULTS", "p2x_DEFbin", base(exact=True, d1=False)),
    ("D3 exact", "p2x_D3bin", base(qs=ALL, exact=True, d1=False)),
]


def own_wrong(res: Dict[str, Any], fn) -> str:
    own = [fn(q["pairs"][q["true"]]) for q in res["queries"]]
    wrong = [fn(q["pairs"][c]) for q in res["queries"] for c in CLASSES
             if c != q["true"]]
    return "%.4f / %.4f (%+.4f)" % (st.mean(own), st.mean(wrong),
                                    st.mean(own) - st.mean(wrong))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="both")
    args = ap.parse_args()
    windows = ["full", "v1"] if args.window == "both" else [args.window]

    import eth80_benchmark as bm
    import eth80_frontend as fe
    import parameters as P

    out: List[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        out.append(s)

    emit("EXACT ASSIGNMENT -- MATCH_ASSIGNMENT_EXACT, EVAL-75 (in-sample)")
    emit("source %s  harness %s  front end %s   environment %s" % (
        bm.source_digest()[:12], bm.harness_digest()[:12],
        fe.frontend_source_digest()[:12], bm.environment()))
    declared = {k: getattr(P, k) for k in (F6, D1, QS, P4, NC, MX, MA)}
    emit("declared defaults: %s" % "; ".join(
        "%s=%r" % kv for kv in sorted(declared.items())))
    emit("DEFAULTS is what the tree declares: %s" % (
        CONFIGS[1][2] == declared))

    fell = False
    for w in windows:
        R = {name: bm.run_config(ov, window=w, breakdown=True, tag=tag)
             for name, tag, ov in CONFIGS}
        emit("")
        emit("=" * 104)
        emit("window = %s" % w)
        emit("=" * 104)
        emit("  %-10s hits (split)  per class a/c/c/h  answers a/c/c/h   "
             "ties  pairs at 0   variance of the score: which reference / "
             "own class" % "")
        for name, _tag, _ov in CONFIGS:
            r = R[name]
            tr = st.tie_report(r)
            pc = st.per_class_hits(r)
            an = st.answers(r)
            vs = st.variance_shares(r)
            zr = st.zero_report(r)
            emit("  %-10s %2d (%5.2f)    %s        %s      %2d    %3d        "
                 "%.1f%% / %.1f%%" % (
                     name, tr["first"], tr["split"],
                     "/".join("%2d" % pc[c][0] for c in CLASSES),
                     "/".join("%2d" % an[c] for c in CLASSES),
                     tr["ties"], zr["pairs_zero"], 100 * vs["ref"],
                     100 * vs["own"]))
        for a, b in (("BASE", "DEFAULTS"), ("D3", "D3 exact")):
            ra, rb = R[a], R[b]
            g, l, p = st.paired(ra, rb)
            moved = sum(int(qa["pairs"][c]["score"] != qb["pairs"][c]["score"])
                        for qa, qb in zip(ra["queries"], rb["queries"])
                        for c in CLASSES)
            dmax = max(abs(qa["pairs"][c]["score"] - qb["pairs"][c]["score"])
                       for qa, qb in zip(ra["queries"], rb["queries"])
                       for c in CLASSES)
            emit("")
            emit("  %s against %s: paired %s; scores moved %d of 300 "
                 "(largest %.4f); answers moved %d" % (
                     b, a, st.fmt_paired(g, l, p), moved, dmax,
                     st.changed_answers(ra, rb)))
            if p < 0.05 and l > g:
                emit("    ACCURACY FELL with paired p < 0.05")
                fell = True
            emit("    means, own-class pairs / wrong-class pairs "
                 "(difference)          %-34s %s" % (a, b))
            rows = [
                ("node pairs kept", lambda pr: pr["n_corr"]),
                ("binary coverage (twin of the node term)",
                 lambda pr: pr["node_coverage_binary"]),
                ("count formula of the same form",
                 lambda pr: pr["count_only_balanced"] if b.startswith("D3")
                 else pr["count_only"]),
                ("node term (graded, uncalibrated)",
                 lambda pr: pr["node_coverage_raw"]),
                ("mean similarity of the pairs kept",
                 lambda pr: (pr["node_coverage_raw"]
                             / pr["node_coverage_binary"])
                 if pr["node_coverage_binary"] > 0 else 0.0),
                ("verifiable core edges",
                 lambda pr: pr["verifiable_core_edges_count"]),
                ("relational consistency",
                 lambda pr: pr["relational_consistency"]),
                ("edge absence", lambda pr: pr["edge_absence"]),
                ("score", lambda pr: pr["score"]),
            ]
            for label, fn in rows:
                emit("      %-42s %-34s %s" % (label, own_wrong(ra, fn),
                                               own_wrong(rb, fn)))
            emit("    each term deciding alone, hits: %s node %d, relation "
                 "%d   |   %s node %d, relation %d" % (
                     a, sum(st.hits(ra, "node_coverage")),
                     sum(st.hits(ra, "relational_consistency")),
                     b, sum(st.hits(rb, "node_coverage")),
                     sum(st.hits(rb, "relational_consistency"))))
            # the D1 signal, again: the node term against the count formula
            ck = "count_only_balanced" if b.startswith("D3") else "count_only"
            for name, r in ((a, ra), (b, rb)):
                g2, l2, p2 = st.paired(r, r, ck, "node_coverage")
                emit("    D1 signal, %-9s node term alone %d against the "
                     "count formula alone %d: +%d/-%d, p=%.3f; r(node term, "
                     "count formula) between references of one query %.3f"
                     % (name, sum(st.hits(r, "node_coverage")),
                        sum(st.hits(r, ck)), g2, l2, p2,
                        st.within_query_r(r, "node_coverage", ck)))
            for cname, ctag, cov in CONTROLS:
                if cname != b:
                    continue
                rc = bm.run_config(cov, window=w, breakdown=True, tag=ctag)
                g2, l2, p2 = st.paired(rc, rc, ck, "node_coverage")
                emit("    control, %s under BINARY credit: node term alone "
                     "%d against the count formula alone %d: +%d/-%d, "
                     "p=%.3f; r %.3f; hits %d" % (
                         cname, sum(st.hits(rc, "node_coverage")),
                         sum(st.hits(rc, ck)), g2, l2, p2,
                         st.within_query_r(rc, "node_coverage", ck),
                         sum(st.hits(rc))))
    emit("")
    emit("=" * 104)
    emit("DEFAULT RULE (pre-registered): pins green; stranding 0 and exact "
         ">= approximate on every pair (correspondence census); no paired "
         "fall with p < 0.05")
    emit("  paired fall with p < 0.05 on some window or form: %s" % (
        "YES" if fell else "no"))
    with open(os.path.join(RESULTS_DIR, "phase2_exact_assignment.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
