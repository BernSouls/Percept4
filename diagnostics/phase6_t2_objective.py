"""[STAGE 6 / TRACK M, Amendment 7.7 step 1] T2: IS THE OBJECTIVE RIGHT?

Before any of D0-D5 is built, the Amendment-7 objective (7.0) is scored on
THREE mappings of all nine registered synthetic pairs:

  (a) the walker's own mapping (the walker as it stands: the RED run's)
  (b) RENDERER TRUTH restricted to the FRAME: the pairs whose two nodes see
      the same labels within one scale and sit within 0.10 of the box on each
      axis, taken one-to-one by maximum cardinality, then minimum total
      position distance
  (c) the old matcher's node correspondence (memory.cores.evidence_score),
      restricted to FRAME pairs

THE TEST (fixed in 7.7): if (b) does not beat (a), the objective is still
wrong; stop and say so.  Nothing here gates, so the objective sees every
similarity it asks for (no budget).

    python diagnostics/phase6_t2_objective.py [--workers 2] [--out NAME]
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
TOL = 0.10


def scene_truth(name):
    """{node: (lens, labels, u, v)} for the positioned FRAME nodes of a scene."""
    import parameters as P
    import eth80_frontend as fe
    from matching.relation_frame import is_frame
    sc = H.SCENES[name]
    img = np.clip(np.asarray(sc.rgb, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    P.GRAPH_STAGE6_RELATIONS = True
    tr = {}
    g = fe.build_graph(img, name=name, trace=tr)
    names = sorted(sc.masks)
    h, w = sc.shape
    lab = np.full((h, w), -1, dtype=np.int32)
    depth = np.full((h, w), 10 ** 6, dtype=np.int32)
    box = {}
    for i, n in enumerate(names):
        m = np.asarray(sc.masks[n], dtype=bool)
        d = int(sc.depth.get(n, 99))
        lab[m & (d < depth)] = i
        depth[m & (d < depth)] = d
        ys, xs = np.nonzero(m)
        box[i] = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
    pos = dict(tr["anchor_positions"])
    pos.update(tr["step_positions"])
    scale = float(tr["extraction_scale"])
    yy, xx = np.mgrid[0:h, 0:w]
    where = {}
    for n, (lens, _sig) in g.sigs.items():
        if not is_frame(lens):
            continue
        p = pos.get(g.node_anchor.get(n))
        if p is None or len(p) < 2:
            continue
        x, y = float(p[0]), float(p[1])
        seen = set(int(v) for v in np.unique(lab[(xx - x) ** 2 + (yy - y) ** 2 <= scale * scale]))
        objs = sorted(v for v in seen if v >= 0)
        if not objs:
            continue
        near = min(objs, key=lambda v: (int(sc.depth.get(names[v], 99)), v))
        x0, x1, y0, y1 = box[near]
        where[n] = (lens, tuple(sorted(seen)),
                    (x - x0) / max(x1 - x0, 1.0), (y - y0) / max(y1 - y0, 1.0))
    return g, where


def truth_mapping(wa, wb):
    """Maximum-cardinality one-to-one truth mapping, minimum total position."""
    from scipy.optimize import linear_sum_assignment
    from matching import relation_walker as W
    by = {}
    for n, (lens, labels, u, v) in wb.items():
        by.setdefault((lens, labels), []).append((n, u, v))
    ua = sorted((n for n in wa), key=lambda n: W._node_key(wa[n][0], None) if False else n)
    pairs, cost = [], []
    for i, n in enumerate(ua):
        lens, labels, u1, v1 = wa[n]
        row = []
        for m, u2, v2 in by.get((lens, labels), ()):
            d = abs(u1 - u2) + abs(v1 - v2)
            if max(abs(u1 - u2), abs(v1 - v2)) <= TOL:
                row.append((d, m))
        pairs.append(row)
    BIG = 1e6
    n = len(ua)
    cand = sorted({m for row in pairs for _d, m in row})
    if not cand or not n:
        return {}
    C = np.full((n, n), BIG)
    for i, row in enumerate(pairs):
        for d, m in row:
            if m in cand:
                C[i, cand.index(m)] = d
    ri, ci = linear_sum_assignment(C)
    return {ua[i]: cand[j] for i, j in zip(ri, ci) if C[i, j] < BIG}


def one(entry):
    an, bn = entry
    from matching import relation_walker as W
    from memory.cores import evidence_score
    from matching.structural_similarity import _sig_sim
    from matching.relation_frame import is_frame
    ga, wa = scene_truth(an)
    gb, wb = scene_truth(bn)
    ia, ib = W._Index(ga.subgraph, ga.sigs), W._Index(gb.subgraph, gb.sigs)
    sim_cache: dict = {}

    def sim(u, v):
        k = (u, v)
        if k not in sim_cache:
            sim_cache[k] = float(_sig_sim(ia.sig[u], ib.sig[v])) if ia.lens[u] == ib.lens[v] else 0.0
        return sim_cache[k]

    res = W.walk(ga.subgraph, gb.subgraph, ga.sigs, gb.sigs)
    walk_map = dict(res.mapping)
    truth_map = truth_mapping(wa, wb)
    _s, corr = evidence_score(ga.subgraph, gb.subgraph, ga.sigs, gb.sigs)
    old_map = {u: v for u, v in corr.items()
               if is_frame(ia.lens.get(u, "")) and is_frame(ib.lens.get(v, ""))}
    out = {"pair": [an, bn], "frame": [len(ia.frame), len(ib.frame)],
           "walk": W.objective(walk_map, ia, ib, sim),
           "truth": W.objective(truth_map, ia, ib, sim),
           "old": W.objective(old_map, ia, ib, sim),
           "walker_score": res.score, "walker_supported": bool(res.supported),
           "walker_examined": res.examined, "walker_budget": res.budget,
           "truth_placement": sum(1 for n in wa if n in truth_map)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="phase6_t2_objective")
    a = ap.parse_args(argv)
    H.SCENES.update(H.scenes())
    pairs = [p for p in PAIRS if p[0] in H.SCENES and p[1] in H.SCENES]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(one, pairs, chunksize=1)
    else:
        rows = [one(p) for p in pairs]
    L = ["[STAGE 6 / TRACK M] T2: the Amendment-7 objective (7.0) on three mappings of the %d registered pairs" % len(rows),
         "   Q = (S + P + A) / (|M| + R); the walker as it stands (a), renderer truth restricted to the FRAME (b),",
         "   the old matcher's correspondence (c).  Nothing gates: the objective sees every similarity.",
         "",
         "   pair                                          |  WALKER  mapped   S     P     A     R   |  TRUTH   mapped   S     P     A     R   |  OLD     mapped   S     P     A     R"]
    for r in rows:
        def cell(k):
            d = r[k]
            return "%6.3f  %5d %5.1f %5.1f %5.1f %5d" % (d["q"], d["mapped"], d["S"], d["P"], d["A"], d["R"])
        L.append("   %-45s | %s | %s | %s" % ("%s / %s" % tuple(r["pair"]), cell("walk"), cell("truth"), cell("old")))
    wins = sum(1 for r in rows if r["truth"]["q"] > r["walk"]["q"])
    L += ["", "   truth Q > walker Q on %d of %d pairs; walker Q > truth Q on %d" % (
        wins, len(rows), sum(1 for r in rows if r["walk"]["q"] > r["truth"]["q"])),
        "   median Q: walker %.3f, truth %.3f, old matcher %.3f" % (
            np.median([r["walk"]["q"] for r in rows]),
            np.median([r["truth"]["q"] for r in rows]),
            np.median([r["old"]["q"] for r in rows])),
        "   median mapped: walker %.0f, truth %.0f, old %.0f (of a smaller frame of median %.0f)" % (
            np.median([r["walk"]["mapped"] for r in rows]),
            np.median([r["truth"]["mapped"] for r in rows]),
            np.median([r["old"]["mapped"] for r in rows]),
            np.median([min(r["frame"]) for r in rows])),
        "", "   THE TEST (7.7): (b) must beat (a).  VERDICT: %s" % (
            "the objective ranks truth above the walker on at least half the pairs" if wins * 2 >= len(rows)
            else "THE OBJECTIVE IS WRONG: truth does not beat the walker; STOP")]
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w") as fh:
        json.dump(rows, fh, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
