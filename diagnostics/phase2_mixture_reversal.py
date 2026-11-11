"""MIXTURE REVERSAL -- EDGE_REVERSE_FLIPS_MIXTURE against its pre-registration.

The bug hunt (``phase2_bug_hunt.py``, sections C and D) found that the
matcher's edge loops read an ``EdgeParamsMixture`` backwards without
reversing it, and recomputed the scores with the mixtures flipped -- offline,
through the same similarity functions and the same correspondence.  REPORT.md
records those figures as the PREDICTION for the switch; this runs the switch
and sets the two side by side.  A disagreement means the census or the
library change is wrong.

Configurations (every switch explicit; EVAL-75, in-sample, both windows):

  F6+D1          the stage-2 record p2_R0            mixtures as they were
  F6+D1 +flip    p2m_R0                              mixtures flipped
  D3             the stage-2 record p2_D3 (all three terms two-sided)
  D3 +flip       p2m_D3
  BASE           p2m_BASE: F6, D1, Phase 4 and the mixture reversal on,
                 D3 and the calibration off -- the base the exact
                 assignment is measured against (phase2_exact_assignment.py)

The assignment is the historical approximation in every configuration here.

    python diagnostics/phase2_mixture_reversal.py [--window full|v1|both]

Paired figures are +gained/-lost queries, exact two-sided McNemar p.
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
                               STAGE4_NEUTRAL, cfg)

#: the census figures recorded in REPORT.md before the switch existed:
#: form -> window -> (scores moved, largest |d score|, answers moved, hits)
PREDICTED = {
    "one-sided": {"full": (19, 0.0011, 0, 38), "v1": (18, 0.0020, 0, 34)},
    "balanced": {"full": (41, 0.0005, 1, 41), "v1": (34, 0.0011, 0, 39)},
}
#: pairs whose similarity sum moves: (core side, query side)
PREDICTED_SUMS = {"full": (21, 28), "v1": (21, 27)}


def flip(base: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out[MX] = True
    return out


CONFIGS = [
    ("F6+D1", "p2_R0", cfg()),
    ("F6+D1 +flip", "p2m_R0", flip(cfg())),
    ("D3", "p2_D3", cfg(qs=ALL)),
    ("D3 +flip", "p2m_D3", flip(cfg(qs=ALL))),
    ("BASE", "p2m_BASE", dict({F6: True, D1: True, QS: (), P4: True,
                               NC: False, MX: True, MA: False},
                              **STAGE4_NEUTRAL)),
]


def moved(a: Dict[str, Any], b: Dict[str, Any], key: str):
    """(pairs whose `key` differs, largest absolute difference)."""
    n, worst = 0, 0.0
    for qa, qb in zip(a["queries"], b["queries"]):
        for c in CLASSES:
            if key not in qa["pairs"][c]:
                continue
            d = abs(qa["pairs"][c][key] - qb["pairs"][c][key])
            if d != 0.0:
                n += 1
                worst = max(worst, d)
    return n, worst


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

    emit("MIXTURE REVERSAL -- EDGE_REVERSE_FLIPS_MIXTURE, EVAL-75 "
         "(in-sample)")
    emit("source %s  harness %s  front end %s   environment %s" % (
        bm.source_digest()[:12], bm.harness_digest()[:12],
        fe.frontend_source_digest()[:12], bm.environment()))
    declared = {k: getattr(P, k) for k in (F6, D1, QS, P4, NC, MX, MA)}
    emit("declared defaults: %s" % "; ".join(
        "%s=%r" % kv for kv in sorted(declared.items())))

    ok = True
    for w in windows:
        R = {name: bm.run_config(ov, window=w, breakdown=True, tag=tag)
             for name, tag, ov in CONFIGS}
        emit("")
        emit("=" * 100)
        emit("window = %s" % w)
        emit("=" * 100)
        emit("  %-13s hits  (ties split)  per class a/c/c/h    answers "
             "a/c/c/h" % "")
        for name, _tag, _ov in CONFIGS:
            r = R[name]
            h = st.hits(r)
            tr = st.tie_report(r)
            pc = st.per_class_hits(r)
            an = st.answers(r)
            emit("  %-13s %2d    (%.2f)       %s      %s" % (
                name, sum(h), tr["split"],
                "/".join("%2d" % pc[c][0] for c in CLASSES),
                "/".join("%2d" % an[c] for c in CLASSES)))
        for form, a, b in (("one-sided", "F6+D1", "F6+D1 +flip"),
                           ("balanced", "D3", "D3 +flip")):
            ra, rb = R[a], R[b]
            n_s, worst = moved(ra, rb, "score")
            n_core, w_core = moved(ra, rb, "matched_rel_sim_sum")
            n_query, w_query = moved(ra, rb, "q_matched_rel_sim_sum")
            ans = st.changed_answers(ra, rb)
            g, l, p = st.paired(ra, rb)
            hits_b = sum(st.hits(rb))
            pred = PREDICTED[form][w]
            emit("")
            emit("  %s score: %s against %s" % (form, b, a))
            emit("    scores moved %d of 300 (largest %.4f); answers moved "
                 "%d; hits %d -> %d, paired %s" % (
                     n_s, worst, ans, sum(st.hits(ra)), hits_b,
                     st.fmt_paired(g, l, p)))
            emit("    PREDICTED by the census: scores moved %d (largest "
                 "%.4f); answers moved %d; hits %d" % pred)
            match = (n_s == pred[0] and abs(worst - pred[1]) < 5e-5
                     and ans == pred[2] and hits_b == pred[3])
            emit("    -> %s" % ("reproduced" if match else "NOT REPRODUCED"))
            ok = ok and match
            emit("    pairs whose core-side similarity sum moved %d "
                 "(largest %.4f)%s" % (
                     n_core, w_core,
                     "; query-side %d (largest %.4f)" % (n_query, w_query)
                     if form == "balanced" else ""))
            if form == "one-sided":
                m = n_core == PREDICTED_SUMS[w][0]
            else:
                m = (n_core, n_query) == PREDICTED_SUMS[w]
            emit("    PREDICTED: core side %d%s -> %s" % (
                PREDICTED_SUMS[w][0],
                ", query side %d" % PREDICTED_SUMS[w][1]
                if form == "balanced" else "",
                "reproduced" if m else "NOT REPRODUCED"))
            ok = ok and m
            # nothing but the edge similarities may move
            still = [k for k in ("n_corr", "q_matched_nm", "c_matched_nm",
                                 "node_coverage_raw", "node_coverage_binary",
                                 "verifiable_core_edges_count",
                                 "n_core_edges", "edge_absence")
                     if moved(ra, rb, k)[0]]
            emit("    node masses, counts and the absence term moved: %s" % (
                ", ".join(still) if still else "none"))
            ok = ok and not still
            if p < 0.05 and l > g:
                emit("    ACCURACY FELL with paired p < 0.05")
                ok = False
        # the base of the next change against the stage-1 state
        g, l, p = st.paired(R["F6+D1"], R["BASE"])
        n_s, worst = moved(R["F6+D1"], R["BASE"], "score")
        emit("")
        emit("  BASE against F6+D1 (adds Phase 4 and the mixture reversal): "
             "scores moved %d (largest %.4f); paired %s" % (
                 n_s, worst, st.fmt_paired(g, l, p)))
        n_s2, _w2 = moved(R["F6+D1 +flip"], R["BASE"], "score")
        emit("  BASE against F6+D1 +flip (adds Phase 4 only): scores moved "
             "%d" % n_s2)
    emit("")
    emit("=" * 100)
    emit("DEFAULT RULE (pre-registered): pins green, census figures "
         "reproduced, no paired fall with p < 0.05")
    emit("  census figures reproduced and no fall: %s   (the pins are run "
         "separately: python -m pytest tests_matcher)" % (
             "YES" if ok else "NO"))
    with open(os.path.join(RESULTS_DIR, "phase2_mixture_reversal.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
