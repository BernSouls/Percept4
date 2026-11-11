"""STAGE 2 -- what the node correspondence leaves unpaired (read-only census).

The null calibration's mechanism check showed that under binary credit the
matcher covers LESS of the core than a random maximum pairing would
(`results/phase2_d3_p4_null.txt`, 3.7).  This census says where that
shortfall comes from, for every (query, reference) pair of EVAL-75.

Per lens type the live matcher pairs query nodes with core nodes on
``signature similarity x anchor similarity`` through
``_max_weight_bipartite_matching`` (per-row top-K candidates, greedy, 2-opt).
The anchor factor is ``exp(-2.5 * distance)``: it is never 0, so a pair has
positive weight exactly when its signature similarity is positive.

For each pair and lens type:

  pairable     min(|Q_l|, |C_l|): what any maximum pairing covers
  paired       pairs the matcher returned
  stranded     pairs that could still be ADDED to the returned
               correspondence: a maximum matching between the query nodes
               and the core nodes it left unpaired, over the pairs with
               positive signature similarity.  A greedy assignment over ALL
               positive pairs can never leave one; only candidate pruning can.
  no partner   the rest of the shortfall: unpaired nodes with no positive
               partner left

Every pair is matched under BOTH values of ``MATCH_ASSIGNMENT_EXACT`` (the
approximation above, and the exact solver), and the total weight each reaches
is set against the optimum of the same weight matrix, rebuilt here.

Nothing is scored.  Diagnostics only.

    python diagnostics/phase2_correspondence_census.py [--window full|v1|both]
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from _bootstrap import RESULTS_DIR

_STATE: Dict[str, Any] = {}


MA = "MATCH_ASSIGNMENT_EXACT"
MODES = (("approx", False), ("exact", True))


def _anchor_factor(Q, C):
    """The matcher's anchor factor (``_anchor_similarity``, a closure of
    ``memory.cores._mcs_accumulate``), rebuilt.  Validated by the caller: the
    exact correspondence must be optimal for the weights built with it."""
    import math

    def bbox(amap):
        if not amap:
            return None
        xs = [float(p[0]) for p in amap.values() if p and len(p) >= 2]
        ys = [float(p[1]) for p in amap.values() if p and len(p) >= 2]
        if len(xs) < 2:
            return None
        x0, y0 = min(xs), min(ys)
        return (x0, y0, max(max(xs) - x0, 1e-9), max(max(ys) - y0, 1e-9))

    qa = getattr(Q, "anchor_of_node", None) or {}
    ca = getattr(C, "anchor_of_node", None) or {}
    qb, cb = bbox(qa), bbox(ca)

    def factor(q, c):
        if qb is None or cb is None:
            return 1.0
        pa, pb = qa.get(q), ca.get(c)
        if pa is None or pb is None or len(pa) < 2 or len(pb) < 2:
            return 1.0
        dx = ((float(pa[0]) - qb[0]) / qb[2]) - ((float(pb[0]) - cb[0]) / cb[2])
        dy = ((float(pa[1]) - qb[1]) / qb[3]) - ((float(pb[1]) - cb[1]) / cb[3])
        return math.exp(-2.5 * math.hypot(dx, dy))

    return factor


def census_pair(Q, C, q_sigs, c_sigs) -> Dict[str, Any]:
    """Both assignments on one pair: the census rows per lens type, the
    total weight each reaches and the optimum of the same weight matrix."""
    import numpy as np
    import parameters as P
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    from memory.cores import _sig_similarity, evidence_score

    q_by: Dict[str, List[int]] = defaultdict(list)
    c_by: Dict[str, List[int]] = defaultdict(list)
    for n in sorted(Q.nodes):
        if n in q_sigs:
            q_by[q_sigs[n][0]].append(n)
    for n in sorted(C.nodes):
        if n in c_sigs:
            c_by[c_sigs[n][0]].append(n)
    lenses = sorted(set(q_by) & set(c_by))
    factor = _anchor_factor(Q, C)
    S, W = {}, {}
    t_opt = 0.0
    for lt in lenses:
        qs, cs = q_by[lt], c_by[lt]
        sm = np.array([[float(_sig_similarity(q_sigs[q][1], c_sigs[c][1]))
                        for c in cs] for q in qs])
        wm = sm * np.array([[factor(q, c) for c in cs] for q in qs])
        S[lt], W[lt] = sm, wm
        r, c_ = linear_sum_assignment(wm, maximize=True)
        t_opt += float(wm[r, c_].sum())

    out: Dict[str, Any] = {"t_opt": t_opt}
    corrs = {}
    old = getattr(P, MA)
    try:
        for mode, exact in MODES:
            setattr(P, MA, exact)
            _score, corr = evidence_score(Q, C, q_sigs, c_sigs, trace={})
            corrs[mode] = corr
            used_c = set(corr.values())
            rows: Dict[str, List[float]] = {}
            total = 0.0
            for lt in lenses:
                qs, cs = q_by[lt], c_by[lt]
                qi = {q: i for i, q in enumerate(qs)}
                ci = {c: j for j, c in enumerate(cs)}
                kept = [(qi[q], ci[corr[q]]) for q in qs if q in corr]
                total += float(sum(W[lt][i, j] for i, j in kept))
                sim_sum = float(sum(S[lt][i, j] for i, j in kept))
                free_q = [qi[q] for q in qs if q not in corr]
                free_c = [ci[c] for c in cs if c not in used_c]
                stranded, best_sum, n_with = 0, 0.0, 0
                if free_q and free_c:
                    sub = S[lt][np.ix_(free_q, free_c)]
                    m = sub > 0.0
                    if m.any():
                        match = maximum_bipartite_matching(
                            csr_matrix(m), perm_type="column")
                        stranded = int((match >= 0).sum())
                        best = sub.max(axis=1)
                        best_sum = float(best[best > 0.0].sum())
                        n_with = int((best > 0.0).sum())
                rows[lt] = [len(qs), len(cs), min(len(qs), len(cs)),
                            len(kept), stranded, sim_sum, best_sum, n_with]
            out[mode] = rows
            out["t_" + mode] = total
            out["n_" + mode] = len(corr)
    finally:
        setattr(P, MA, old)
    a, e = corrs["approx"], corrs["exact"]
    out["shared"] = sum(1 for q, c in a.items() if e.get(q) == c)
    return out


def _work(i: int):
    st = _STATE
    cls, _name, (Q, q_sigs) = st["queries"][i]
    return cls, {rc: census_pair(Q, C, q_sigs, c_sigs)
                 for rc, (C, c_sigs) in st["refs"].items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", default="both", choices=("full", "v1", "both"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    windows = ("full", "v1") if args.window == "both" else (args.window,)

    import multiprocessing as mp

    import eth80_benchmark as bm
    import eth80_frontend as fe
    import parameters as P
    from eth80_split import load_split
    from matching.structural_similarity import GESTALT_LENS_BOOST

    out: List[str] = []

    def emit(s: str = "") -> None:
        print(s)
        sys.stdout.flush()
        out.append(s)

    emit("STAGE 2 -- what the node correspondence leaves unpaired "
         "(EVAL-75, read-only)")
    emit("source digest %s   THETA_SIM=%r   candidate rule of the "
         "approximation: per query node its top 8 core nodes by weight, plus "
         "every pair with weight >= THETA_SIM" % (
             bm.source_digest()[:12], getattr(P, "THETA_SIM", None)))
    emit("every pair is matched twice, under either value of %s; the other "
         "switches do not reach the correspondence" % MA)
    emit("")
    split = load_split()
    entries = [(c, p) for c in bm.CLASSES for p in split["eval"][c]]
    for w in windows:
        refs = {c: bm.make_window(fe.graph_for(c, split["references"][c]), w)
                for c in bm.CLASSES}
        queries = [(c, os.path.basename(p),
                    bm.make_window(fe.graph_for(c, p), w)) for c, p in entries]
        _STATE.update(refs=refs, queries=queries)
        with mp.get_context("fork").Pool(args.workers) as pool:
            rows = pool.map(_work, range(len(queries)), chunksize=2)
        emit("=" * 100)
        emit("window = %s   %d pairs" % (w, 4 * len(queries)))
        emit("=" * 100)
        for mode, exact in MODES:
            emit("  -- %s = %r %s" % (MA, exact, "(the historical "
                 "approximation: per-row top 8, greedy, 2-opt)" if not exact
                 else "(scipy linear_sum_assignment on the same weights)"))
            agg: Dict[str, List[float]] = defaultdict(lambda: [0.0] * 8)
            per_pair_short: List[float] = []
            per_pair_strand: List[float] = []
            for _cls, res in rows:
                for _rc, o in res.items():
                    pa = sp = st_ = 0.0
                    for lt, v in o[mode].items():
                        a = agg[lt]
                        for k in range(8):
                            a[k] += v[k]
                        b = GESTALT_LENS_BOOST.get(lt, 1.0)
                        pa += b * v[2]
                        sp += b * v[3]
                        st_ += b * v[4]
                    per_pair_short.append((pa - sp) / pa if pa else 0.0)
                    per_pair_strand.append(st_ / pa if pa else 0.0)
            emit("  %-16s %9s %9s %9s %9s %7s %9s %7s %10s %10s %12s" % (
                "lens type", "query n", "core n", "pairable", "paired",
                "share", "stranded", "share", "no partner", "mean sim",
                "stranded sim"))
            tot = [0.0] * 8
            for lt in sorted(agg, key=lambda k: -agg[k][2]):
                v = agg[lt]
                for k in range(8):
                    tot[k] += v[k]
                emit("  %-16s %9d %9d %9d %9d %6.1f%% %9d %6.1f%% %10d "
                     "%10.4f %12.4f" % (
                         lt, v[0], v[1], v[2], v[3],
                         100.0 * v[3] / v[2] if v[2] else 0.0, v[4],
                         100.0 * v[4] / v[2] if v[2] else 0.0,
                         v[2] - v[3] - v[4], v[5] / v[3] if v[3] else 0.0,
                         v[6] / v[7] if v[7] else 0.0))
            emit("  %-16s %9d %9d %9d %9d %6.1f%% %9d %6.1f%% %10d %10.4f "
                 "%12.4f" % (
                     "TOTAL", tot[0], tot[1], tot[2], tot[3],
                     100.0 * tot[3] / tot[2], tot[4],
                     100.0 * tot[4] / tot[2], tot[2] - tot[3] - tot[4],
                     tot[5] / tot[3] if tot[3] else 0.0,
                     tot[6] / tot[7] if tot[7] else 0.0))
            ps = sorted(per_pair_short)
            pt = sorted(per_pair_strand)
            emit("  per pair, boost-weighted: shortfall of pairable mass  "
                 "mean %.4f  median %.4f  max %.4f   |   of which stranded  "
                 "mean %.4f  median %.4f  max %.4f   pairs with any stranded "
                 "node: %d of %d" % (
                     sum(ps) / len(ps), ps[len(ps) // 2], ps[-1],
                     sum(pt) / len(pt), pt[len(pt) // 2], pt[-1],
                     sum(int(x > 0) for x in pt), len(pt)))
            emit("")
        emit("  columns: pairable = sum of min(|Q_l|, |C_l|); stranded = "
             "pairs that could be added to the returned correspondence "
             "(both ends unpaired, positive similarity);")
        emit("  mean sim = signature similarity of the returned pairs; "
             "stranded sim = mean, over the unpaired query nodes that have "
             "a free partner, of their BEST free partner's similarity (an "
             "upper bound per node, not the similarity of an added pairing)")
        emit("")
        emit("  -- the total weight of the assignment (signature similarity "
             "x anchor factor, summed over the pairs kept), per pair")
        pairs = [o for _cls, res in rows for o in res.values()]
        gain = sorted((o["t_exact"] - o["t_approx"]) / o["t_approx"]
                      for o in pairs if o["t_approx"] > 0)
        emit("     exact below approximate (must be 0): %d of %d pairs   "
             "largest |exact - optimum of the rebuilt weight matrix|: %.3g "
             "(must be ~0: validates the rebuilt weights)" % (
                 sum(int(o["t_exact"] < o["t_approx"] - 1e-9) for o in pairs),
                 len(pairs), max(abs(o["t_exact"] - o["t_opt"])
                                 for o in pairs)))
        emit("     approximate below the optimum: %d of %d pairs   gain of "
             "the exact total over the approximate one: mean %.2f%%  median "
             "%.2f%%  max %.2f%%" % (
                 sum(int(o["t_approx"] < o["t_opt"] - 1e-9) for o in pairs),
                 len(pairs), 100 * sum(gain) / len(gain),
                 100 * gain[len(gain) // 2], 100 * gain[-1]))
        n_a = sum(o["n_approx"] for o in pairs)
        n_e = sum(o["n_exact"] for o in pairs)
        sh = sum(o["shared"] for o in pairs)
        emit("     node pairs: approximate %d, exact %d; in both %d (%.1f%% "
             "of the approximate pairs are kept by the exact solver)" % (
                 n_a, n_e, sh, 100.0 * sh / n_a))
        emit("")
    path = os.path.join(RESULTS_DIR, "phase2_correspondence_census.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("written: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
