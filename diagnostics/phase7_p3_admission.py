"""[STAGE 7 / PHASE 3] ADMISSION BY STRENGTH: the three settings, measured.

Nine registered pairs, renderer truth, everything else pre-registered.  The
three settings are:

  shipped      no admission arm (what the walker does today)
  arm A        relative only: within ADMIT_MARGIN (1.0) of the best rival
               agreement on the same node
  arm B        arm A AND agreement >= ADMIT_FLOOR (0.5)

Per setting and pair: FAN-OUT (candidates popped per admitted true pair, i.e.
popped / role-correct mapped pairs), mapped, pointwise P and R, role F1, and the
budget spent against the pre-registered budget.

THE MECHANISM TO CONFIRM (7.32): fan-out falls AND recall rises, together.
THE FALSIFIER (7.32): fan-out falls and recall does NOT rise -> growth by
relation is the wrong mechanism for this representation; stop and say so.

    python diagnostics/phase7_p3_admission.py [--out NAME] [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import phase6_walker_synthetic as H  # noqa: E402

PAIRS = H.PAIRS
ARMS = (("shipped", False, False), ("arm A (relative)", True, False),
        ("arm B (relative + floor)", True, True))


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


def run_arm(pa, pb, rel, floor):
    import matching.relation_walker as W

    (sa, ga, ra), (sb, gb, rb) = pa, pb
    keep = (W.ADMIT_RELATIVE, W.ADMIT_FLOOR_ON)
    W.ADMIT_RELATIVE, W.ADMIT_FLOOR_ON = rel, floor
    try:
        res = W.walk(sa, sb, ga, gb)
    finally:
        W.ADMIT_RELATIVE, W.ADMIT_FLOOR_ON = keep
    pr = H.precision_recall(res.mapping, ra, rb)
    # role-correct mapped pairs: the denominator of the fan-out
    correct = sum(1 for u, v in res.mapping.items()
                  if u in ra and v in rb and ra[u] == rb[v])
    return {"mapped": len(res.mapping), "correct": correct,
            "p": pr["p"], "r": pr["r"], "f1": pr["f1"], "pairs_with_role": pr["pairs"],
            "pops": res.pops, "admission_rejections": res.admission_rejections,
            "fan_out": (res.pops / correct) if correct else float("inf"),
            "examined": res.examined, "budget": res.budget,
            "budget_spent": (res.examined / res.budget) if res.budget else 0.0,
            "seeds": len(res.seeds)}


def main(argv=None) -> int:
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase7_p3_admission")
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
        for label, rel, floor in ARMS:
            row[label] = run_arm(bp[0], bp[1], rel, floor)
        rows.append(row)

    L = ["[STAGE 7 / PHASE 3] ADMISSION BY STRENGTH, nine registered pairs",
         "   fan-out = candidates POPPED per role-correct mapped pair; the budget is",
         "   the walker's own (0.05 of the frame product, floor 8 per node).  Arms and",
         "   constants pre-registered in 7.32 (ADMIT_MARGIN 1.0, ADMIT_FLOOR 0.5).", "",
         "   %-30s %-25s %6s %6s %8s %6s %6s %6s %7s %7s" % (
             "pair", "arm", "mapped", "corr.", "fan-out", "P", "R", "F1", "spent", "budget")]
    for r in rows:
        for label, _r, _f in ARMS:
            v = r[label]
            L.append("   %-30s %-25s %6d %6d %8s %6.3f %6.3f %6.3f %7.2f %7d" % (
                r["pair"] if label == "shipped" else "", label, v["mapped"], v["correct"],
                ("%.1f" % v["fan_out"]) if np.isfinite(v["fan_out"]) else "inf",
                v["p"], v["r"], v["f1"], v["budget_spent"], v["budget"]))
        L.append("")

    def m(k, lab):
        vals = [r[lab][k] for r in rows]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    L += ["   MEANS",
          "   %-25s %7s %7s %8s %6s %6s %6s %7s" % (
              "arm", "mapped", "corr.", "fan-out", "P", "R", "F1", "spent")]
    for label, _r, _f in ARMS:
        L.append("   %-25s %7.1f %7.1f %8.1f %6.3f %6.3f %6.3f %7.2f" % (
            label, m("mapped", label), m("correct", label), m("fan_out", label),
            m("p", label), m("r", label), m("f1", label), m("budget_spent", label)))
    L.append("")

    base = "shipped"
    for label, _r, _f in ARMS[1:]:
        dfo = [r[base]["fan_out"] - r[label]["fan_out"] for r in rows]
        dr = [r[label]["r"] - r[base]["r"] for r in rows]
        dp = [r[label]["p"] - r[base]["p"] for r in rows]
        df1 = [r[label]["f1"] - r[base]["f1"] for r in rows]
        fell = sum(1 for x in dfo if x > 0)
        rose = sum(1 for x in dr if x > 0)
        L += ["   %s against the shipped setting" % label,
              "     fan-out falls on %d of 9 pairs; recall RISES on %d, falls on %d" % (
                  fell, rose, sum(1 for x in dr if x < 0)),
              "     mean fan-out %+.1f, mean recall %+.3f, mean precision %+.3f, mean F1 %+.3f" % (
                  float(np.mean(dfo)), float(np.mean(dr)), float(np.mean(dp)),
                  float(np.mean(df1)))]
        if rose >= 6 and sum(1 for x in dp if x < 0) <= 2:
            L.append("     DEFAULT RULE (7.32): QUALIFIES (recall up on >= 6, precision down on <= 2)")
        else:
            L.append("     DEFAULT RULE (7.32): does not qualify (needs recall up on >= 6 "
                     "and precision down on <= 2)")
        L.append("")

    armA = "arm A (relative)"
    fo_fell = sum(1 for r in rows if r[armA]["fan_out"] < r[base]["fan_out"])
    r_rose = sum(1 for r in rows if r[armA]["r"] > r[base]["r"])
    if fo_fell >= 5 and r_rose < 5:
        L += ["   THE FALSIFIER FIRED: fan-out fell on %d pairs and recall rose on only %d." % (
            fo_fell, r_rose),
            "   Growth by relation is the wrong mechanism for this representation; the",
            "   margin is not tuned and no arm is defaulted."]
    else:
        L += ["   THE FALSIFIER: fan-out fell on %d pairs, recall rose on %d -- the"
              " pre-registered mechanism (both, together) %s." % (
                  fo_fell, r_rose,
                  "holds" if (r_rose >= 5) else "does NOT hold")]

    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "means": {lab: {k: m(k, lab) for k in
                                                 ("mapped", "correct", "fan_out", "p", "r",
                                                  "f1", "budget_spent")}
                                           for lab, _x, _y in ARMS},
                   "fan_out_fell": fo_fell, "recall_rose": r_rose}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
