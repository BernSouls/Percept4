"""[STAGE 6 / TRACK M, after the verdict] DO THE RELATIONS REPEAT?

M3-6 is red: the walker maps fewer nodes than the old matcher and, across
geometry instances, maps them worse.  R6 counted roads inside ONE graph; a
correspondence needs the SAME road in two graphs.  This measures that, per
relation class, on the synthetic scene pairs (renderer truth), and decides
nothing.

TRUE CORRESPONDENTS (finer than the role of M3-6; descriptive, defined after
the verdict): two FRAME nodes, one per scene, of the same lens type, seeing
the same set of labels within one scale, whose positions in the bounding box
of their nearest-in-depth object differ by at most 0.10 of the box on each
axis.  (Box-normalised, so it holds across the fresh scenes' new geometry.)

Per relation class (edge type, direction, lens type of the far node), over
the relations u -> u' of scene A whose two ends both have a correspondent in B:

  REPEATS   some correspondent v of u and v' of u' are joined in B by a
            relation of the same class that AGREES (every axis measured on
            both within one bin, at least one such axis)
  PRESENT   ... are joined by a relation of the same class at all
  CHANCE    for a node w of B of u's lens type that is NOT a correspondent of
            u: the share of such w that carry SOME relation of the class
            agreeing with u -> u'.  This is how easily a wrong pair is
            supported.

A relation is a road for a walker when REPEATS is high and CHANCE is low.

    python diagnostics/phase6_relation_repeatability.py [--out NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import phase6_walker_synthetic as H  # noqa: E402

SAME_GEOMETRY = (("box_over_box", "box_over_box_haze"),)
OTHER_GEOMETRY = (("box_over_box", "box_over_box_test"),
                  ("cylinder_over_box", "cylinder_over_box_shift"),
                  ("f_box_a", "f_box_b"), ("f_haze_a", "f_haze_b"),
                  ("f_disk_a", "f_disk_b"), ("f_cyl_a", "f_cyl_b"),
                  ("single_disk", "f_ctrl_disk"), ("single_box", "f_ctrl_box"))
TOL = 0.10


def build(name):
    """(walker index, {node: (lens, labels, u, v)}) of one scene."""
    import parameters as P
    import eth80_frontend as fe
    from matching import relation_walker as W
    from matching.relation_frame import is_frame
    sc = H.SCENES[name]
    img = np.clip(np.asarray(sc.rgb, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    P.GRAPH_STAGE6_RELATIONS = True
    tr = {}
    g = fe.build_graph(img, name=name, trace=tr)
    P.GRAPH_STAGE6_RELATIONS = False
    names = sorted(sc.masks)
    h, w = sc.shape
    lab = np.full((h, w), -1, dtype=np.int32)
    depth = np.full((h, w), 10 ** 6, dtype=np.int32)
    box = {}
    for i, n in enumerate(names):
        m = np.asarray(sc.masks[n], dtype=bool)
        d = int(sc.depth.get(n, 99))
        take = m & (d < depth)
        lab[take] = i
        depth[take] = d
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
        where[n] = (lens, tuple(sorted(seen)), (x - x0) / max(x1 - x0, 1.0), (y - y0) / max(y1 - y0, 1.0))
    idx = W._Index(g.subgraph, g.sigs)
    chains = [len(f) for hst, f in idx.hosts.items() if idx.lens[hst] == "PEAK_CHAIN"]
    return name, (idx, where, chains)


def correspondents(wa, wb):
    by = defaultdict(list)
    for n, (lens, labels, u, v) in wb.items():
        by[(lens, labels)].append((n, u, v))
    out = {}
    for n, (lens, labels, u, v) in wa.items():
        out[n] = [m for m, u2, v2 in by.get((lens, labels), ()) if abs(u - u2) <= TOL and abs(v - v2) <= TOL]
    return out


def measure(A, B):
    from matching import relation_walker as W
    ia, wa, _ca = A
    ib, wb, _cb = B
    corr = correspondents(wa, wb)
    lens_b = defaultdict(list)
    for n in ib.frame:
        lens_b[ib.lens[n]].append(n)
    rows = defaultdict(Counter)
    for u in ia.frame:
        cu = corr.get(u)
        if not cu:
            continue
        cu_set = set(cu)
        for cls, halves in ia.half[u].items():
            for hu in halves:
                cf = corr.get(hu.far)
                if not cf:
                    continue
                cf_set = set(cf)
                r = rows[cls]
                r["n"] += 1
                present = agree = False
                for v in cu:
                    for hv in ib.half[v].get(cls, ()):
                        if hv.far in cf_set:
                            present = True
                            if W._slack(hu, hv) is not None:
                                agree = True
                r["present"] += int(present)
                r["repeats"] += int(agree)
                wrong = [w for w in lens_b[ia.lens[u]] if w not in cu_set]
                if wrong:
                    hit = sum(1 for w in wrong
                              if any(W._slack(hu, hw) is not None for hw in ib.half[w].get(cls, ())))
                    r["chance_num"] += hit
                    r["chance_den"] += len(wrong)
    cov = {"frame": len(ia.frame), "placed": len(wa),
           "with_correspondent": sum(1 for n in wa if corr.get(n)),
           "correspondents_median": float(np.median([len(corr[n]) for n in wa if corr.get(n)] or [0]))}
    return rows, cov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase6_relation_repeatability")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    H.SCENES.update(H.scenes())
    names = sorted({n for p in SAME_GEOMETRY + OTHER_GEOMETRY for n in p})
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            built = dict(pool.map(build, names, chunksize=1))
    else:
        built = dict(build(n) for n in names)
    L = ["[STAGE 6 / TRACK M, after the verdict] do the relations repeat across two images?  (synthetic scenes, renderer truth)",
         "true correspondents: same lens type, same labels within one scale, box-normalised position within %.2f" % TOL, ""]
    out = {}
    for label, pairs in (("SAME GEOMETRY (the same layout, hazed and blurred)", SAME_GEOMETRY),
                         ("OTHER GEOMETRY (the same family, other sizes and places)", OTHER_GEOMETRY)):
        tot = defaultdict(Counter)
        covs = []
        for an, bn in pairs:
            for x, y in ((an, bn), (bn, an)):
                rows, cov = measure(built[x], built[y])
                covs.append(cov)
                for cls, r in rows.items():
                    tot[cls].update(r)
        L += [label + ": %d pairs, both directions" % len(pairs),
              "   FRAME nodes placed on an object: %d of %d; with at least one true correspondent in the other scene: %d (%.2f); "
              "correspondents per such node, median %.0f" % (
                  sum(c["placed"] for c in covs), sum(c["frame"] for c in covs),
                  sum(c["with_correspondent"] for c in covs),
                  sum(c["with_correspondent"] for c in covs) / max(sum(c["placed"] for c in covs), 1),
                  float(np.median([c["correspondents_median"] for c in covs]))),
              "   relation class (type, direction, far lens)                      relations   PRESENT   REPEATS   CHANCE   repeats / chance"]
        table = {}
        for cls in sorted(tot, key=lambda c: -tot[c]["n"]):
            r = tot[cls]
            if r["n"] < 20:
                continue
            rep = r["repeats"] / r["n"]
            ch = r["chance_num"] / max(r["chance_den"], 1)
            table["%s|%s|%s" % cls] = {"n": r["n"], "present": r["present"] / r["n"], "repeats": rep, "chance": ch}
            L.append("   %-28s %s %-18s %9d    %6.2f    %6.2f   %6.3f   %s" % (
                cls[0], cls[1], cls[2], r["n"], r["present"] / r["n"], rep, ch,
                ("%6.1f" % (rep / ch)) if ch > 0 else "   inf"))
        n_all = sum(r["n"] for r in tot.values())
        L.append("   ALL CLASSES                                                %9d    %6.2f    %6.2f   %6.3f" % (
            n_all, sum(r["present"] for r in tot.values()) / max(n_all, 1),
            sum(r["repeats"] for r in tot.values()) / max(n_all, 1),
            sum(r["chance_num"] for r in tot.values()) / max(sum(r["chance_den"] for r in tot.values()), 1)))
        L.append("")
        out[label] = table
    L.append("PEAK_CHAIN hosts per scene (features attached with an arc position): " + "; ".join(
        "%s %d chains, median %d features (largest %d)" % (
            n, len(built[n][2]), np.median(built[n][2]) if built[n][2] else 0, max(built[n][2] or [0]))
        for n in names))
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w") as fh:
        json.dump(out, fh, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
