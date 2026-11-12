"""[STAGE 7 / PHASE 2] THE SEEDS, measured against renderer truth.

Both seed rules the walker has, on all nine registered scene pairs, with
everything else at its pre-registered value:

  TOPOLOGICAL  D1 (`SEED_TOPOLOGICAL = True`): token rarity + shared tokens +
               node similarity
  MEASURED     the rule D1 replaced (`SEED_TOPOLOGICAL = False`): the number of
               relation classes in which some relation agrees within one bin

Per pair and per rule:

  seeds           how many were chosen
  right / wrong   their two ends are the SAME PHYSICAL ROLE (the M3-6 criterion,
                  so the two phases' numbers are comparable) / another role
  no role         one end sees no object, so truth cannot judge the pair
  PRECISION       right / (right + wrong)  -- chance agreement removed
  RECALL          right / the number of true correspondent pairs in the smaller
                  FRAME
  then the mapping that grows from those seeds: mapped, pointwise P and R,
  role F1, and the seeds' own lens-type mix.

The default rule was fixed before the run (pre-registration, 7.31): MEASURED is
default iff its seed precision is higher on at least 6 of the 9 pairs.

    python diagnostics/phase7_p2_seeds.py [--out NAME] [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import phase6_walker_synthetic as H  # noqa: E402

PAIRS = H.PAIRS
RULES = (("topological", True), ("measured", False))


def build_pair(an, bn):
    def one(name):
        import parameters as P
        import eth80_frontend as fe
        sc = H.SCENES[name]
        img = np.clip(np.asarray(sc.rgb, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        P.GRAPH_STAGE6_RELATIONS = True
        tr = {}
        g = fe.build_graph(img, name=name, trace=tr)
        P.GRAPH_STAGE6_RELATIONS = False
        return (g.subgraph, g.sigs, H.roles_of(g, tr, sc))
    return one(an), one(bn)


def score_row(pa, pb, rule_flag):
    """Everything for one pair under one seed rule."""
    import matching.relation_walker as W

    (sa, ga, ra), (sb, gb, rb) = pa, pb
    old = W.SEED_TOPOLOGICAL
    W.SEED_TOPOLOGICAL = rule_flag
    try:
        res = W.walk(sa, sb, ga, gb)
    finally:
        W.SEED_TOPOLOGICAL = old

    def role_ok(u, v):
        if u not in ra or v not in rb:
            return None
        return bool(ra[u] == rb[v])

    right = wrong = norole = 0
    for u, v in res.seeds:
        ok = role_ok(u, v)
        if ok is None:
            norole += 1
        elif ok:
            right += 1
        else:
            wrong += 1
    # how many true correspondent pairs exist in the smaller frame (the ceiling)
    by_role_a, by_role_b = Counter(ra.values()), Counter(rb.values())
    ceiling = sum(min(by_role_a[r], by_role_b[r]) for r in by_role_a)
    pr = H.precision_recall(res.mapping, ra, rb)
    lens = Counter(ga[u][0] for u, _v in res.seeds)
    return {"seeds": len(res.seeds), "right": right, "wrong": wrong, "no_role": norole,
            "precision": (right / (right + wrong)) if (right + wrong) else 0.0,
            "recall": (right / ceiling) if ceiling else 0.0,
            "ceiling": ceiling,
            "mapped": len(res.mapping), "p": pr["p"], "r": pr["r"], "f1": pr["f1"],
            "seed_lens": dict(lens), "examined": res.examined, "budget": res.budget}


def main(argv=None) -> int:
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase7_p2_seeds")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    H.SCENES.update(H.scenes())
    pairs = list(PAIRS)
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            built = pool.starmap(build_pair, pairs, chunksize=1)
    else:
        built = [build_pair(*p) for p in pairs]

    rows = []
    for (an, bn), bp in zip(pairs, built):
        row = {"pair": "%s / %s" % (an, bn)}
        for label, flag in RULES:
            row[label] = score_row(bp[0], bp[1], flag)
        rows.append(row)

    L = ["[STAGE 7 / PHASE 2] THE SEEDS, nine registered pairs, against renderer truth",
         "   'right' = the seed's two ends have the SAME PHYSICAL ROLE (the M3-6",
         "   criterion); RECALL is against the true correspondent pairs in the smaller",
         "   frame, so a rule cannot win by seeding fewer.  Default rule fixed in 7.31.",
         "",
         "   %-30s %-13s %6s %6s %6s %7s %8s %9s %6s %6s %6s" % (
             "pair", "rule", "seeds", "right", "wrong", "P(seed)", "R(seed)", "ceiling",
             "mapped", "P", "R")]
    for r in rows:
        for label, _f in RULES:
            v = r[label]
            L.append("   %-30s %-13s %6d %6d %6d %7.3f %8.3f %9d %6d %6.3f %6.3f" % (
                r["pair"] if label == "topological" else "", label, v["seeds"],
                v["right"], v["wrong"], v["precision"], v["recall"], v["ceiling"],
                v["mapped"], v["p"], v["r"]))
        L.append("")

    def mean(k, rule):
        return float(np.mean([r[rule][k] for r in rows]))

    L += ["   MEANS OVER THE NINE PAIRS",
          "   %-13s %6s %6s %6s %7s %8s %7s %7s %7s" % (
              "rule", "seeds", "right", "wrong", "P(seed)", "R(seed)", "mapped", "P", "R"),
          ]
    for label, _f in RULES:
        L.append("   %-13s %6.1f %6.1f %6.1f %7.3f %8.3f %7.1f %7.3f %7.3f" % (
            label, mean("seeds", label), mean("right", label), mean("wrong", label),
            mean("precision", label), mean("recall", label), mean("mapped", label),
            mean("p", label), mean("r", label)))
    L.append("")
    higher = [r for r in rows if r["measured"]["precision"] > r["topological"]["precision"]]
    lower = [r for r in rows if r["measured"]["precision"] < r["topological"]["precision"]]
    equal = [r for r in rows if r["measured"]["precision"] == r["topological"]["precision"]]
    L += ["   THE DEFAULT RULE (7.31): MEASURED is default iff its seed precision is",
          "   higher on at least 6 of the 9 pairs.",
          "   MEASURED higher on %d, lower on %d, equal on %d" % (
              len(higher), len(lower), len(equal)),
          "   VERDICT: %s" % ("MEASURED becomes the default" if len(higher) >= 6
                             else "TOPOLOGICAL stays the default")]
    if len(higher) >= 6:
        L += ["   WHY: " + _why(rows)]
    # the registered next arm, reported only if the measured rule is under 50 % precise
    if mean("precision", "measured") < 0.5:
        L += ["", "   The measured rule is under 50 %% precise (%.3f), so the next seed arm",
              "   is the DISTINCTIVENESS RANK fixed in 7.31: rank the agreeing pairs by the",
              "   rarity of the agreeing bin pattern (its carrier count in each graph,",
              "   ascending) and take seeds in that order.  It is registered, NOT built or",
              "   run here." % mean("precision", "measured")]
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"rows": rows,
                   "verdict": ("measured" if len(higher) >= 6 else "topological"),
                   "higher": len(higher), "lower": len(lower), "equal": len(equal),
                   "means": {lab: {k: mean(k, lab) for k in
                                   ("seeds", "right", "wrong", "precision", "recall",
                                    "mapped", "p", "r", "f1")} for lab, _ in RULES}},
                  fh, indent=1)
    return 0


def _why(rows):
    """One sentence on what the topological rule loses."""
    t = float(np.mean([r["topological"]["precision"] for r in rows]))
    m = float(np.mean([r["measured"]["precision"] for r in rows]))
    tw = float(np.mean([r["topological"]["wrong"] for r in rows]))
    mw = float(np.mean([r["measured"]["wrong"] for r in rows]))
    return ("the topological rule's tokens (the far end's degree and closedness are "
            "part of the token) pick seeds that are wrong far more often: mean role "
            "precision %.3f against %.3f, %.1f wrong seeds per pair against %.1f"
            % (t, m, tw, mw))


if __name__ == "__main__":
    raise SystemExit(main())
