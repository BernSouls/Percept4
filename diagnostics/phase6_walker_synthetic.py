"""[STAGE 6 / TRACK M, M1 and M3-6] The relation walker on the synthetic scenes,
through the real front end, against renderer truth.

 1. M1: every scene's graph (GRAPH_STAGE6_RELATIONS on) matched against
    itself under shuffled node ids: is the mapping the identity, the score 1?
 2. M3-6 (pre-registration, section 7): on the nine registered scene pairs,
    the precision and recall of "same physical role" of the node mapping, for
      new   the walker on the graphs with the stage-6 relations
      old   `evidence_score`'s own node assignment on the graphs with every
            stage-6 switch off (the baseline configuration)
      old*  the old matcher on the NEW graphs (reported, not part of the pin)
    ROLE of a positioned FRAME node = (the set of labels within one scale of
    its position; the quadrant of its position around the centroid of the
    nearest-in-depth object of that set).  A node that only sees background
    has no role.  PRECISION = mapped pairs with the same role / mapped pairs
    with a role on both sides; RECALL = mapped pairs with the same role /
    sum over roles of min(count in A, count in B).
    THE PIN: over the pairs where old and new differ in (precision, recall),
    mean F1(new) - F1(old) > 0 and new is higher on at least as many pairs as
    it is lower.

    python diagnostics/phase6_walker_synthetic.py [--workers 2] [--out NAME]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

PAIRS = (("box_over_box", "box_over_box_test"), ("box_over_box", "box_over_box_haze"),
         ("cylinder_over_box", "cylinder_over_box_shift"), ("f_box_a", "f_box_b"),
         ("f_haze_a", "f_haze_b"), ("f_disk_a", "f_disk_b"), ("f_cyl_a", "f_cyl_b"),
         ("single_disk", "f_ctrl_disk"), ("single_box", "f_ctrl_box"))


def scenes():
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tests_figground"))
    from tests_figground.fresh_scenes import build_fresh_all
    from tests_figground.scenes import build_all
    out = dict(build_all())
    out.update(build_fresh_all())
    return out


def roles_of(g, tr, scene):
    """node id -> role, for the positioned FRAME nodes that see an object."""
    from matching.relation_frame import is_frame
    names = sorted(scene.masks)
    h, w = scene.shape
    lab = np.full((h, w), -1, dtype=np.int32)
    depth = np.full((h, w), 10 ** 6, dtype=np.int32)
    cent = {}
    for i, n in enumerate(names):
        m = np.asarray(scene.masks[n], dtype=bool)
        d = int(scene.depth.get(n, 99))
        take = m & (d < depth)
        lab[take] = i
        depth[take] = d
        ys, xs = np.nonzero(m)
        cent[i] = (float(xs.mean()), float(ys.mean())) if len(xs) else None
    pos = dict(tr["anchor_positions"])
    pos.update(tr["step_positions"])
    scale = float(tr["extraction_scale"])
    yy, xx = np.mgrid[0:h, 0:w]
    out = {}
    for n, (lens, _sig) in g.sigs.items():
        if not is_frame(lens):
            continue
        p = pos.get(g.node_anchor.get(n))
        if p is None or len(p) < 2:
            continue
        x, y = float(p[0]), float(p[1])
        disc = (xx - x) ** 2 + (yy - y) ** 2 <= scale * scale
        seen = set(int(v) for v in np.unique(lab[disc]))
        objs = sorted(v for v in seen if v >= 0)
        if not objs:
            continue
        near = min(objs, key=lambda v: (int(scene.depth.get(names[v], 99)), v))
        c = cent[near]
        ang = math.degrees(math.atan2(y - c[1], x - c[0])) % 360.0
        quadrant = int(((ang + 45.0) % 360.0) // 90.0)
        out[n] = (tuple(sorted(names[v] if v >= 0 else "background" for v in seen)), quadrant)
    return out


def precision_recall(mapping, roles_a, roles_b):
    both = [(u, v) for u, v in mapping.items() if u in roles_a and v in roles_b]
    same = sum(1 for u, v in both if roles_a[u] == roles_b[v])
    ca, cb = Counter(roles_a.values()), Counter(roles_b.values())
    possible = sum(min(ca[r], cb[r]) for r in ca if r in cb)
    p = same / len(both) if both else 0.0
    r = same / possible if possible else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"pairs": len(both), "same": same, "possible": possible, "p": p, "r": r, "f1": f1}


def relabel(g, perm):
    """The same graph under other node ids (undirected edges re-canonicalised)."""
    from core.graph_types import Subgraph
    from core.schema import EdgeParamsMixture, reverse_edge_params
    from core.types import is_undirected_edge
    sg = g.subgraph
    edges, params = set(), {}
    for (s, d, et) in sg.edges:
        ep = sg.edge_params.get((s, d, et))
        a, b = perm[s], perm[d]
        if is_undirected_edge(et) and a > b:
            a, b = b, a
            if ep is not None:
                ep = ep.flipped_copy() if isinstance(ep, EdgeParamsMixture) else reverse_edge_params(ep)
        edges.add((a, b, et))
        if ep is not None:
            params[(a, b, et)] = ep
    return (Subgraph(nodes=frozenset(perm[n] for n in sg.nodes), edges=frozenset(edges),
                     edge_params=params),
            {perm[n]: v for n, v in g.sigs.items()})


def build_one(name):
    """Both graphs of one scene (relations on / every stage-6 switch off)."""
    import parameters as P
    import eth80_frontend as fe
    sc = SCENES[name]
    img = np.clip(np.asarray(sc.rgb, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out = {}
    for flag in (True, False):
        P.GRAPH_STAGE6_RELATIONS = flag
        tr = {}
        g = fe.build_graph(img, name=name, trace=tr)
        out["new" if flag else "old"] = (g.subgraph, g.sigs, g.node_anchor, roles_of(g, tr, sc))
    P.GRAPH_STAGE6_RELATIONS = False
    # M1: the new graph against itself under shuffled ids
    from matching import relation_walker as W
    sg, sigs, _anchor, _roles = out["new"]
    ids = sorted(sigs)
    shuffled = list(ids)
    random.Random(len(ids)).shuffle(shuffled)
    perm = dict(zip(ids, (10 ** 6 + x for x in shuffled)))

    from types import SimpleNamespace
    sg2, sigs2 = relabel(SimpleNamespace(subgraph=sg, sigs=sigs), perm)
    res = W.walk(sg, sg2, sigs, sigs2)
    out["self"] = {"frame": res.frame_q, "mapped": len(res.mapping),
                   "identity": sum(1 for u, v in res.mapping.items() if perm[u] == v),
                   "score": res.score, "supported": res.supported, "seeds": len(res.seeds),
                   "examined": res.examined, "budget": res.budget, "dense": res.dense,
                   "order_rejections": res.order_rejections}
    return name, out


SCENES = {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="phase6_walker_synthetic")
    a = ap.parse_args(argv)
    SCENES.update(scenes())
    names = sorted(SCENES)
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            built = dict(pool.map(build_one, names, chunksize=1))
    else:
        built = dict(build_one(n) for n in names)

    from matching import relation_walker as W
    from memory.cores import evidence_score
    from matching.relation_frame import is_frame

    L = ["[STAGE 6 / TRACK M] the relation walker on the synthetic scenes (real front end, %d scenes)" % len(names), "",
         "1. M1: each scene's graph (stage-6 relations on) against itself under shuffled node ids",
         "   scene                       FRAME  mapped  identity  score  supported  seeds  examined / budget   dense matrix"]
    all_identity = True
    for n in names:
        s = built[n]["self"]
        all_identity &= (s["identity"] == s["mapped"] and (s["score"] is None or abs(s["score"] - 1.0) < 1e-9))
        L.append("   %-26s %6d  %6d  %8d  %5s  %-9s  %5d  %8d / %-6d   %8d" % (
            n, s["frame"], s["mapped"], s["identity"],
            "-" if s["score"] is None else "%.3f" % s["score"], s["supported"], s["seeds"],
            s["examined"], s["budget"], s["dense"]))
    mapped_share = [built[n]["self"]["mapped"] / max(built[n]["self"]["frame"], 1) for n in names]
    L.append("   every mapping is the identity with score 1 (where supported): %s; mapped / FRAME median %.2f (min %.2f); "
             "unsupported self-matches: %d" % (
                 all_identity, float(np.median(mapped_share)), min(mapped_share),
                 sum(1 for n in names if not built[n]["self"]["supported"])))

    L += ["", "2. M3-6: precision / recall of 'same physical role' on the registered pairs",
          "   pair                                          | new (walker, new graph)                         | old (evidence_score, old graph)        | old* (old matcher, new graph)",
          "                                                 | mapped  with role  P      R      F1    score      | with role  P      R      F1           | with role  P      R      F1"]
    rows = []
    for an, bn in PAIRS:
        A, B = built[an], built[bn]
        res = W.walk(A["new"][0], B["new"][0], A["new"][1], B["new"][1])
        new = precision_recall(res.mapping, A["new"][3], B["new"][3])
        ra, rb = A["new"][3], B["new"][3]
        seed_roles = [None if (u not in ra or v not in rb) else bool(ra[u] == rb[v])
                      for u, v in res.seeds]
        lens_a = {n: lt for n, (lt, _s) in A["new"][1].items()}
        seed_lens = Counter(lens_a[u] for u, _v in res.seeds)

        def old_on(key):
            _s, corr = evidence_score(A[key][0], B[key][0], A[key][1], B[key][1])
            frame_corr = {u: v for u, v in corr.items()
                          if is_frame(A[key][1][u][0]) and is_frame(B[key][1][v][0])}
            return precision_recall(frame_corr, A[key][3], B[key][3]), len(frame_corr)

        old, n_old = old_on("old")
        old_new, n_old_new = old_on("new")
        rows.append({"pair": [an, bn], "new": new, "old": old, "old_on_new_graph": old_new,
                     "mapped_new": len(res.mapping), "mapped_old": n_old, "mapped_old_new": n_old_new,
                     "score": res.score, "supported": res.supported, "seeds": len(res.seeds),
                     "examined": res.examined, "budget": res.budget, "dense": res.dense,
                     "order_rejections": res.order_rejections,
                     "budget_exhausted": bool(res.budget_exhausted),
                     "seeds_right": sum(1 for x in seed_roles if x is True),
                     "seeds_wrong": sum(1 for x in seed_roles if x is False),
                     "seeds_no_role": sum(1 for x in seed_roles if x is None),
                     "seed_lens": dict(seed_lens),
                     "frame": [res.frame_q, res.frame_r]})
        L.append("   %-45s | %6d  %9d  %.3f  %.3f  %.3f  %-9s | %9d  %.3f  %.3f  %.3f        | %9d  %.3f  %.3f  %.3f" % (
            "%s / %s" % (an, bn), len(res.mapping), new["pairs"], new["p"], new["r"], new["f1"],
            ("%.3f" % res.score) if res.score is not None else "UNSUPP.",
            old["pairs"], old["p"], old["r"], old["f1"],
            old_new["pairs"], old_new["p"], old_new["r"], old_new["f1"]))
    L += ["", "   the walker's seeds, by renderer truth (same role / another role / no role), and what it mapped",
          "   pair                                          seeds  right  wrong  no role   mapped / smaller FRAME   order rejections   budget spent"]
    for r in rows:
        L.append("   %-45s %5d  %5d  %5d  %7d   %6d / %-6d          %6d             %s" % (
            "%s / %s" % tuple(r["pair"]), r["seeds"], r["seeds_right"], r["seeds_wrong"],
            r["seeds_no_role"], r["mapped_new"], min(r["frame"]), r["order_rejections"],
            "yes" if r["budget_exhausted"] else "no"))
    differ = [r for r in rows if (round(r["new"]["p"], 9), round(r["new"]["r"], 9))
              != (round(r["old"]["p"], 9), round(r["old"]["r"], 9))]
    d = [r["new"]["f1"] - r["old"]["f1"] for r in differ]
    higher, lower = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
    mean = float(np.mean(d)) if d else 0.0
    ok = bool(d) and mean > 0 and higher >= lower
    L += ["", "   pairs where old and new differ in (precision, recall): %d of %d" % (len(differ), len(rows)),
          "   F1(new) - F1(old) on them: mean %+.3f; new higher on %d, lower on %d" % (mean, higher, lower),
          "   precision, mean over the %d pairs: new %.3f, old %.3f; recall: new %.3f, old %.3f" % (
              len(rows), np.mean([r["new"]["p"] for r in rows]), np.mean([r["old"]["p"] for r in rows]),
              np.mean([r["new"]["r"] for r in rows]), np.mean([r["old"]["r"] for r in rows])),
          "   precision alone: new higher on %d pairs, lower on %d; recall alone: new higher on %d, lower on %d" % (
              sum(1 for r in rows if r["new"]["p"] > r["old"]["p"]),
              sum(1 for r in rows if r["new"]["p"] < r["old"]["p"]),
              sum(1 for r in rows if r["new"]["r"] > r["old"]["r"]),
              sum(1 for r in rows if r["new"]["r"] < r["old"]["r"])),
          "   THE PIN (mean > 0 and higher >= lower): %s" % ("GREEN" if ok else "RED"),
          "", "   cost per pair (walker): signature similarities computed, median %d (budget median %d) against a dense "
              "matrix of median %d; frame pairs mapped, median %d of a smaller frame of median %d" % (
                  np.median([r["examined"] for r in rows]), np.median([r["budget"] for r in rows]),
                  np.median([r["dense"] for r in rows]), np.median([r["mapped_new"] for r in rows]),
                  np.median([min(r["frame"]) for r in rows]))]
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w") as fh:
        json.dump({"self": {n: built[n]["self"] for n in names}, "pairs": rows, "pin": ok}, fh, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
