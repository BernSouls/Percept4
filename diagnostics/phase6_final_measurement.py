"""[STAGE 6] THE FINAL MEASUREMENT (pre-registration, section 9).

Runs once per split, and only when R6 has passed (EVAL and HELDOUT) and every
M3 pin is green.  Two configurations on the same tree and the same graphs:

  A  the matcher switch OFF: `memory.cores.evidence_score`, the benchmark's
     own harness (diagnostics/eth80_benchmark.py, window "full"), argmax
  B  the matcher switch ON: the relation walker (matching/relation_walker.py)

Counting for B (section 6): an UNSUPPORTED reference cannot win; a query whose
four references are all unsupported ABSTAINS and counts as an error; a tie
between the best references is an error.

Reported: accuracy overall and per class, the paired table (both right / only
A / only B / both wrong) with the exact two-sided binomial p on the discordant
pairs, the unsupported share, the mapping sizes, the pairs examined against
the dense matrix.

    PERCEPT_SET=GRAPH_STAGE6_RELATIONS=1 python diagnostics/phase6_final_measurement.py --split eval
    python diagnostics/phase6_final_measurement.py --split eval --only-a --out NAME   # old graphs, old matcher
    ... --dry-run     # the four references as queries: exercises the code, scores nothing new
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

from _stats import CLASSES, mcnemar_exact  # noqa: E402

_STATE = {}


def _walk_query(i):
    from matching import relation_walker as W
    cls, name, (Q, q_sigs) = _STATE["queries"][i]
    rows = {}
    for rc in CLASSES:
        C, c_sigs = _STATE["refs"][rc]
        t0 = time.time()
        res = W.walk(Q, C, q_sigs, c_sigs)
        rows[rc] = {"score": res.score, "supported": bool(res.supported),
                    "mapped": len(res.mapping), "seeds": len(res.seeds),
                    "frame_q": res.frame_q, "frame_r": res.frame_r,
                    "preserved": res.preserved, "relations_q": res.relations_q,
                    "relations_r": res.relations_r, "attributes_shared": res.attributes_shared,
                    "examined": res.examined, "budget": res.budget,
                    "budget_exhausted": bool(res.budget_exhausted), "dense": res.dense,
                    "edge_checks": res.edge_checks, "order_rejections": res.order_rejections,
                    "seconds": time.time() - t0}
    return {"name": name, "true": cls, "pairs": rows}


def decide_b(q):
    """(prediction or None, outcome) under the registered counting."""
    sup = {c: q["pairs"][c]["score"] for c in CLASSES if q["pairs"][c]["supported"]}
    if not sup:
        return None, "abstain"
    best = max(sup.values())
    winners = [c for c in CLASSES if c in sup and sup[c] == best]
    if len(winners) > 1:
        return None, "tie"
    return winners[0], "decided"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=("eval", "heldout"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only-a", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import parameters as P
    import eth80_benchmark as bench
    import eth80_frontend as fe
    from eth80_split import load_split

    out = a.out or ("phase6_final_%s" % a.split)
    L = ["[STAGE 6] the final measurement, %s-75; PERCEPT_SET: %s; front-end digest %s" % (
        a.split.upper(), _bootstrap.switch_overrides_key() or "none (every stage-6 switch off)",
        fe.frontend_source_digest()[:12])]
    split = load_split()
    entries = [(c, p) for c in CLASSES for p in split[a.split][c]]
    if a.dry_run:
        entries = [(c, split["references"][c]) for c in CLASSES]
        L.append("DRY RUN: the four references as queries (nothing is scored on the split)")

    # ---- A: the old matcher, through the benchmark's own harness -----------
    assert not bool(getattr(P, "MATCH_RELATION_WALKER", False))
    if a.dry_run:
        res_a = None
    else:
        res_a = bench.run(a.split, window="full", workers=a.workers, verbose=False)
        sa = bench.summarize(res_a)
        L += ["", "A  the old matcher (evidence_score, window full)",
              "   " + bench.fmt_row("A", sa)]
    if a.only_a:
        text = "\n".join(L) + "\n"
        print(text)
        with open(os.path.join(RESULTS_DIR, out + ".txt"), "w") as fh:
            fh.write(text)
        with open(os.path.join(RESULTS_DIR, out + ".json"), "w") as fh:
            json.dump({"A": {"hits": sa["hits"], "n": sa["n"], "per_class": sa["per_class"],
                             "ties": sa["ties"], "preds": sa["preds"],
                             "names": [q["name"] for q in res_a["queries"]]}}, fh, sort_keys=True)
        return 0

    # ---- B: the relation walker ----------------------------------------------
    if not bool(getattr(P, "GRAPH_STAGE6_RELATIONS", False)):
        raise SystemExit("B needs the stage-6 relations in the graphs (PERCEPT_SET=GRAPH_STAGE6_RELATIONS=1)")
    refs = {c: bench.make_window(fe.graph_for(c, split["references"][c]), "full") for c in CLASSES}
    queries = [(c, os.path.basename(p), bench.make_window(fe.graph_for(c, p), "full"))
               for c, p in entries]
    _STATE.update(refs=refs, queries=queries)
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(_walk_query, range(len(queries)), chunksize=2)
    else:
        rows = [_walk_query(i) for i in range(len(queries))]
    seconds = time.time() - t0
    dec = [decide_b(q) for q in rows]
    hit_b = [int(pred == q["true"]) for (pred, _o), q in zip(dec, rows)]
    n = len(rows)
    per_b = {c: [sum(h for h, q in zip(hit_b, rows) if q["true"] == c),
                 sum(1 for q in rows if q["true"] == c)] for c in CLASSES}
    outcomes = Counter(o for _p, o in dec)
    decided = [i for i, (_p, o) in enumerate(dec) if o == "decided"]
    L += ["", "B  the relation walker (matcher switch ON)",
          "   B   %2d/%-2d = %5.1f%%   %s   abstained %d   tied %d" % (
              sum(hit_b), n, 100.0 * sum(hit_b) / n,
              "  ".join("%s %d/%d" % (c, per_b[c][0], per_b[c][1]) for c in CLASSES),
              outcomes["abstain"], outcomes["tie"]),
          "   over the %d queries with a decision: %d right (%.1f%%)" % (
              len(decided), sum(hit_b[i] for i in decided),
              100.0 * sum(hit_b[i] for i in decided) / max(len(decided), 1))]
    pairs = [(q, c) for q in rows for c in CLASSES]
    unsup = sum(1 for q, c in pairs if not q["pairs"][c]["supported"])
    true_pairs = [q["pairs"][q["true"]] for q in rows]
    other_pairs = [q["pairs"][c] for q in rows for c in CLASSES if c != q["true"]]
    L += ["   unsupported (query, reference) pairs: %d of %d (%.1f%%); of the TRUE-class pairs: %d of %d" % (
        unsup, len(pairs), 100.0 * unsup / len(pairs),
        sum(1 for p in true_pairs if not p["supported"]), len(true_pairs)),
          "   frame pairs mapped: true-class pairs median %d (p10 %d, p90 %d); other-class pairs median %d (p10 %d, p90 %d)" % (
              np.median([p["mapped"] for p in true_pairs]), np.percentile([p["mapped"] for p in true_pairs], 10),
              np.percentile([p["mapped"] for p in true_pairs], 90),
              np.median([p["mapped"] for p in other_pairs]), np.percentile([p["mapped"] for p in other_pairs], 10),
              np.percentile([p["mapped"] for p in other_pairs], 90)),
          "   mapped / smaller FRAME: true-class median %.2f; other-class median %.2f" % (
              np.median([p["mapped"] / max(min(p["frame_q"], p["frame_r"]), 1) for p in true_pairs]),
              np.median([p["mapped"] / max(min(p["frame_q"], p["frame_r"]), 1) for p in other_pairs])),
          "   score (supported pairs): true-class median %s; other-class median %s" % (
              ("%.3f" % np.median([p["score"] for p in true_pairs if p["supported"]]))
              if any(p["supported"] for p in true_pairs) else "-",
              ("%.3f" % np.median([p["score"] for p in other_pairs if p["supported"]]))
              if any(p["supported"] for p in other_pairs) else "-"),
          "   cost per pair: signature similarities computed median %d (budget median %d; exhausted in %d of %d pairs) "
          "against a dense matrix of median %d (%.2f%%); relation pairs compared median %d; %.2f s per pair" % (
              np.median([q["pairs"][c]["examined"] for q, c in pairs]),
              np.median([q["pairs"][c]["budget"] for q, c in pairs]),
              sum(1 for q, c in pairs if q["pairs"][c]["budget_exhausted"]), len(pairs),
              np.median([q["pairs"][c]["dense"] for q, c in pairs]),
              100.0 * np.median([q["pairs"][c]["examined"] / max(q["pairs"][c]["dense"], 1) for q, c in pairs]),
              np.median([q["pairs"][c]["edge_checks"] for q, c in pairs]),
              seconds / max(len(pairs), 1) * max(a.workers, 1))]
    result = {"B": {"hits": sum(hit_b), "n": n, "per_class": per_b, "outcomes": dict(outcomes),
                    "queries": rows, "decisions": [list(d) for d in dec]}}
    if res_a is not None:
        hit_a = [int(p == q["true"]) for p, q in zip(sa["preds"], res_a["queries"])]
        for qa, qb in zip(res_a["queries"], rows):
            assert qa["name"] == qb["name"]
        both = sum(1 for x, y in zip(hit_a, hit_b) if x and y)
        only_a = sum(1 for x, y in zip(hit_a, hit_b) if x and not y)
        only_b = sum(1 for x, y in zip(hit_a, hit_b) if y and not x)
        neither = n - both - only_a - only_b
        p = mcnemar_exact(only_b, only_a)
        L += ["", "PAIRED, the same %d queries:  both right %d | only A %d | only B %d | both wrong %d;  "
              "exact two-sided binomial p on the %d discordant pairs = %.3f" % (
                  n, both, only_a, only_b, neither, only_a + only_b, p),
              "per class (A / B): " + "  ".join(
                  "%s %d / %d of %d" % (c, sa["per_class"][c][0], per_b[c][0], per_b[c][1]) for c in CLASSES),
              "AHEAD (B has more correct answers than A): %s  (A %d, B %d)" % (
                  "YES" if sum(hit_b) > sa["hits"] else "NO", sa["hits"], sum(hit_b))]
        result["A"] = {"hits": sa["hits"], "n": sa["n"], "per_class": sa["per_class"],
                       "ties": sa["ties"], "preds": sa["preds"]}
        result["paired"] = {"both": both, "only_a": only_a, "only_b": only_b, "neither": neither, "p": p}
    text = "\n".join(L) + "\n"
    print(text)
    if not a.dry_run:
        with open(os.path.join(RESULTS_DIR, out + ".txt"), "w") as fh:
            fh.write(text)
        with open(os.path.join(RESULTS_DIR, out + ".json"), "w") as fh:
            json.dump(result, fh, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
