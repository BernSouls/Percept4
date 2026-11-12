"""[STAGE 6 / TRACK M, 7.7 step 3] T1: the OLD matcher on the stage-6 graphs.

REPORT ONLY.  Nothing here changes a default; the walker is not run.

The old matcher (`memory.cores.evidence_score`, the switch-off path) is scored
on the ETH-80 EVAL-75 split twice, through its own harness
(`diagnostics/eth80_benchmark.py`, argmax over the four references):

  stage-5  the graphs as the repository builds them with every stage-6 switch
           OFF (the stage-5 configuration: the paired baseline)
  stage-6  the graphs with GRAPH_STAGE6_RELATIONS ON (stage 6's structure)

both windows ("full" and "v1"), paired per query.  The question is the one the
pre-registration fixed: what does the stage-6 graph change for the matcher the
repository actually ships, on its own in-sample split?

    python diagnostics/phase6_t1_old_matcher.py [--workers 2]
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import json
import os
import sys
from collections import Counter

from _bootstrap import RESULTS_DIR

STAGE6_FLAGS = ("FIELD_BORDER_DISCARD_PER_CHANNEL", "BINDER_COMPLETE_POSITION_MAP",
                "GRAPH_STAGE6_RELATIONS", "MATCH_RELATION_WALKER")


def run(workers: int, tag: str, switches_on: bool) -> dict:
    import eth80_benchmark as bench
    over = {f: bool(switches_on) for f in STAGE6_FLAGS}
    out = {}
    for w in bench.WINDOWS:
        out[w] = bench.run_config(over, split_name="eval", window=w,
                                  arm="baseline", breakdown=False,
                                  tag="%s_%s" % (tag, w), workers=workers,
                                  verbose=False)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="phase6_t1_old_matcher")
    a = ap.parse_args(argv)

    off = run(a.workers, "phase6_t1_stage5", False)
    on = run(a.workers, "phase6_t1_stage6", True)
    import eth80_benchmark as bench

    L = ["[STAGE 6 / TRACK M] T1: the OLD matcher (evidence_score) on the stage-6 graphs, "
         "EVAL-75, both windows, paired against the stage-5 graphs",
         "   REPORT ONLY: no default is changed by this run; the walker is not used.", ""]
    rows = {}
    for w in bench.WINDOWS:
        s5, s6 = bench.summarize(off[w]), bench.summarize(on[w])
        p5 = {q: r for q, r in [(x["name"], x) for x in off[w]["queries"]]}
        p6 = {q: r for q, r in [(x["name"], x) for x in on[w]["queries"]]}
        acc5, acc6 = s5.get("acc"), s6.get("acc")
        n = len(p5)
        gained = sorted(q for q in p5 if p6[q]["pred"] == p6[q]["true"] and p5[q]["pred"] != p5[q]["true"])
        lost = sorted(q for q in p5 if p5[q]["pred"] == p5[q]["true"] and p6[q]["pred"] != p6[q]["true"])
        moved = [q for q in p5 if p5[q]["pred"] != p6[q]["pred"]]
        rows[w] = {"n": n, "stage5": acc5, "stage6": acc6, "gained": gained, "lost": lost,
                   "pred_moved": moved,
                   "stage5_correct": sum(1 for q in p5 if p5[q]["pred"] == p5[q]["true"]),
                   "stage6_correct": sum(1 for q in p6 if p6[q]["pred"] == p6[q]["true"])}
        L += ["  window %-4s   n=%d   stage-5 %d/%d = %.3f   stage-6 %d/%d = %.3f   "
              "gained %d, lost %d, predictions moved %d" % (
                  w, n, rows[w]["stage5_correct"], n, acc5, rows[w]["stage6_correct"], n, acc6,
                  len(gained), len(lost), len(moved))]
    L += ["", "   per-query changes (a query whose argmax moved between the two graphs):"]
    any_change = False
    for w in bench.WINDOWS:
        for q in rows[w]["pred_moved"]:
            any_change = True
            L.append("     [%s] %-22s stage-5 -> %s   stage-6 -> %s   true %s" % (
                w, q,
                next(x for x in off[w]["queries"] if x["name"] == q)["pred"],
                next(x for x in on[w]["queries"] if x["name"] == q)["pred"],
                next(x for x in on[w]["queries"] if x["name"] == q)["true"]))
    if not any_change:
        L.append("     none: the stage-6 graph changes no argmax for this matcher on EVAL-75")
    L += ["", "   (EVAL-75, in-sample for every decision of this work; reported, never"
          " selected on.  The split and harness are the repository's own:"
          " diagnostics/eth80_benchmark.py.)"]
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
