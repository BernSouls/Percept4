"""STAGE 4 -- D2 prerequisites, mechanism check of the DARK global alignment.

Does the relational refinement of ``matching/alignment.py`` run at all on
graphs of this size, and what does a pair cost?  Under the two accountings
of ``MATCH_ALIGN_MAX_PAIR_EVALS``:

  historical        the seed's similarity matrix is charged to the budget
                    (P.MATCH_ALIGN_BUDGET_REFINEMENT_ONLY = False)
  refinement only   the seed is not charged; a candidate of similarity 0 is
                    neither evaluated nor charged (True, the declared value)

Sample: the first two EVAL queries of each class against the four
references, 32 pairs per window.  Reported per accounting: candidates,
positive candidates, refinement sweeps run, budget state, seconds per pair.
With one round and the anchor prior the dark path should return the live
correspondence; the number of pairs where it does is reported, and where it
does not, why: a pair across lens types that share a signature class (the
dark path pairs across lens types by design), or another optimum of the
same assignment (no cross-lens pair, equal total weight).

NO SCORE AND NO ACCURACY IS COMPUTED HERE.  That would be D2, which is
gated.  ``MATCH_GLOBAL_CONSISTENCY_ENABLED`` stays False in the tree.

    python diagnostics/phase3_d2_mechanism.py [--window full|v1|both]
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List

import _stats as st
from _bootstrap import RESULTS_DIR
from _stats import CLASSES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", default="both")
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()
    windows = ["full", "v1"] if args.window == "both" else [args.window]

    import eth80_benchmark as bm
    import eth80_frontend as fe
    import matching.alignment as AL
    import parameters as P
    from eth80_split import load_split
    from memory import cores
    from phase2_d3_p4_null import STAGE4_NEUTRAL

    out: List[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        out.append(s)

    emit("STAGE 4 -- D2 prerequisites: does the dark alignment refine, and "
         "at what cost (no score, no accuracy)")
    emit("source %s  front end %s" % (bm.source_digest()[:12],
                                      fe.frontend_source_digest()[:12]))
    emit("MATCH_ALIGN_ROUNDS=%r  MATCH_ALIGN_MAX_PAIR_EVALS=%r  "
         "MATCH_SYSTEMATICITY_WEIGHT=%r  MATCH_UNMATCHED_COST=%r  "
         "MATCH_GLOBAL_CONSISTENCY_ENABLED=%r (the tree)" % (
             P.MATCH_ALIGN_ROUNDS, P.MATCH_ALIGN_MAX_PAIR_EVALS,
             P.MATCH_SYSTEMATICITY_WEIGHT, P.MATCH_UNMATCHED_COST,
             P.MATCH_GLOBAL_CONSISTENCY_ENABLED))

    seen: List[Any] = []
    real = AL.global_consistency_alignment

    def recording(*a, **k):
        al = real(*a, **k)
        seen.append(al)
        return al

    keep = {k: getattr(P, k) for k in list(STAGE4_NEUTRAL) + [
        "MATCH_ALIGN_ROUNDS", "MATCH_GLOBAL_CONSISTENCY_ENABLED",
        "MATCH_ASSIGNMENT_EXACT"]}
    AL.global_consistency_alignment = recording
    split = load_split()
    try:
        for k, v in STAGE4_NEUTRAL.items():
            setattr(P, k, v)
        P.MATCH_ASSIGNMENT_EXACT = True
        P.MATCH_GLOBAL_CONSISTENCY_ENABLED = False
        for w in windows:
            refs = {c: bm.make_window(fe.graph_for(c, split["references"][c]),
                                      w) for c in CLASSES}
            queries = [(c, os.path.basename(p),
                        bm.make_window(fe.graph_for(c, p), w))
                       for c in CLASSES
                       for p in split["eval"][c][:args.per_class]]
            emit("")
            emit("window = %s   %d pairs" % (w, 4 * len(queries)))
            rows: Dict[str, List[Dict[str, float]]] = {
                "historical": [], "refinement only": []}
            live_same = live_n = cross_only = tie_only = other = 0
            tie_nodes: List[int] = []
            for _cls, _name, (Q, q_sigs) in queries:
                for rc in CLASSES:
                    C, c_sigs = refs[rc]
                    nq = sum(1 for n in Q.nodes if n in q_sigs)
                    nc = sum(1 for n in C.nodes if n in c_sigs)
                    for label, flag in (("historical", False),
                                        ("refinement only", True)):
                        P.MATCH_ALIGN_BUDGET_REFINEMENT_ONLY = flag
                        P.MATCH_GLOBAL_ANCHOR_PRIOR = False
                        seen.clear()
                        t0 = time.perf_counter()
                        m = cores._global_correspondence(Q, C, q_sigs, c_sigs)
                        dt = time.perf_counter() - t0
                        al = seen[-1]
                        rows[label].append({
                            "cand": nq * nc, "rounds": al.rounds_used,
                            "exhausted": float(al.budget_exhausted),
                            "seconds": dt, "pairs": len(m)})
                    # one round with the anchor prior against the live path
                    P.MATCH_ALIGN_BUDGET_REFINEMENT_ONLY = True
                    P.MATCH_GLOBAL_ANCHOR_PRIOR = True
                    rounds = P.MATCH_ALIGN_ROUNDS
                    P.MATCH_ALIGN_ROUNDS = 1
                    try:
                        dark = cores._global_correspondence(Q, C, q_sigs,
                                                            c_sigs)
                    finally:
                        P.MATCH_ALIGN_ROUNDS = rounds
                        P.MATCH_GLOBAL_ANCHOR_PRIOR = False
                    live = cores._mcs_accumulate(Q, C, q_sigs, c_sigs)[6]
                    live_n += 1
                    if dark == live:
                        live_same += 1
                    else:
                        diff = {u for u in set(dark) | set(live)
                                if dark.get(u) != live.get(u)}
                        crossed = {u for u, v in dark.items()
                                   if q_sigs[u][0] != c_sigs[v][0]}
                        if crossed:
                            # a cross-lens pair, and the nodes it displaced
                            # inside their own lens types
                            cross_only += 1
                        else:
                            # no cross-lens pair: are the two correspondences
                            # two optima of the SAME assignment?  Total live
                            # weight (similarity x anchor factor) of each.
                            qn = cores._normalised_anchors(Q)
                            cn = cores._normalised_anchors(C)

                            def weight(u, v):
                                sim = cores._sig_similarity(q_sigs[u][1],
                                                            c_sigs[v][1])
                                a = (qn or {}).get(u)
                                b = (cn or {}).get(v)
                                if a is None or b is None:
                                    return sim
                                return sim * math.exp(-2.5 * math.hypot(
                                    a[0] - b[0], a[1] - b[1]))

                            wd = math.fsum(weight(u, v)
                                           for u, v in dark.items())
                            wl = math.fsum(weight(u, v)
                                           for u, v in live.items())
                            if abs(wd - wl) <= 1e-9 * max(1.0, wl):
                                tie_only += 1
                                tie_nodes.append(len(diff))
                            else:
                                other += 1
            for label in ("historical", "refinement only"):
                r = rows[label]
                emit("  %-16s candidates per pair %.0f (min %.0f, max %.0f); "
                     "refinement sweeps run: mean %.2f (pairs with none: %d "
                     "of %d); budget exhausted on %d; seconds per pair mean "
                     "%.2f max %.2f" % (
                         label, st.mean([x["cand"] for x in r]),
                         min(x["cand"] for x in r), max(x["cand"] for x in r),
                         st.mean([x["rounds"] for x in r]),
                         sum(1 for x in r if x["rounds"] == 0), len(r),
                         int(sum(x["exhausted"] for x in r)),
                         st.mean([x["seconds"] for x in r]),
                         max(x["seconds"] for x in r)))
            emit("  one round + anchor prior against the live "
                 "correspondence: identical on %d of %d pairs; of the rest, "
                 "%d contain a pair across lens types that share a signature "
                 "class; %d are another optimum of the same assignment "
                 "(equal total weight, %s nodes paired differently); %d "
                 "unexplained" % (
                     live_same, live_n, cross_only, tie_only,
                     ("%d to %d" % (min(tie_nodes), max(tie_nodes)))
                     if tie_nodes else "0", other))
    finally:
        AL.global_consistency_alignment = real
        for k, v in keep.items():
            setattr(P, k, v)
    if len(windows) == 2:
        with open(os.path.join(RESULTS_DIR, "phase3_d2_mechanism.txt"),
                  "w") as fh:
            fh.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
