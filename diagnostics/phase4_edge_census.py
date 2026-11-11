"""PHASE 4, step 0 -- what each matched edge's similarity is made of.

Read-only. For every (query, reference) pair of EVAL-75 the matcher's edge
loop is re-walked for the correspondence it returned, and every VERIFIABLE
core edge (both endpoints mapped) is put in exactly one bin:

  no query edge      the query has no edge at all between the mapped endpoints
  no same-kind edge  it has edges there, none of a comparable kind
                     (similarity 0 for every candidate, none same-class)
  same kind, sim 0   a same-class candidate exists; the best similarity is 0
                     (categorical gate / sign mismatch / kernel underflow)
  PARAM-LESS         best partner and core edge both carry no EdgeParams
                     (the F6 structural match: topology, no physics)
  UNMEASURED         both carry params, but NO slot is measured on both sides
                     -- the similarity is one of the UNKNOWN floors
  CATEGORICAL ONLY   only identity (categorical) slots are measured on both
                     sides; they passed the gate; no graded axis in common
  MEASURED k         k graded axes measured on both sides (k = 1, 2, ...)

The EDGE-EVIDENCE spec says an unmeasured pair must contribute 0 and a
topological match must not pose as physical similarity. This census says how
much of `matched_rel_sim_sum` on this corpus is which.

    python diagnostics/phase4_edge_census.py [--window full|v1|both]
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from _bootstrap import RESULTS_DIR

UNKNOWN_BIN = None  # resolved at run time from core.schema


def classify_pair(Q, C, q_sigs, c_sigs) -> Tuple[Dict, Dict[str, Any]]:
    from core.schema import (UNKNOWN, edge_measured_axes,
                             polymorphic_edge_similarity)
    from memory.cores import _reverse_for_traversal
    from memory.cores import evidence_score
    from orchestration.binder import UNDIRECTED_EDGE_TYPES

    trace: Dict[str, Any] = {}
    _score, corr = evidence_score(Q, C, q_sigs, c_sigs, trace=trace)

    def _canon(e):
        if e[2] in UNDIRECTED_EDGE_TYPES:
            return (min(e[0], e[1]), max(e[0], e[1]), e[2])
        return e

    q_params = Q.edge_params or {}
    c_params = C.edge_params or {}
    q_index: Dict[Tuple[int, int], List[Tuple[Any, Any, Any]]] = defaultdict(list)
    for q_e in Q.edges:
        qs, qd, qet = q_e
        q_key = _canon(q_e)
        qp = q_params.get(q_key) or q_params.get(q_e)
        if qp is None and qet in UNDIRECTED_EDGE_TYPES:
            qp = q_params.get((q_e[1], q_e[0], q_e[2]))
        q_index[(qs, qd)].append((qet, qp, q_key))
        if qet in UNDIRECTED_EDGE_TYPES:
            # the library's own helper, so this re-walk follows
            # P.EDGE_REVERSE_FLIPS_MIXTURE as the matcher's loops do
            qp_rev = _reverse_for_traversal(qp)
            q_index[(qd, qs)].append((qet, qp_rev, q_key))

    inv = {c: q for q, c in corr.items()}
    # (edge type, bin) -> [count, sim_sum, count sim==1]
    out: Dict[Tuple[str, str], List[float]] = defaultdict(lambda: [0, 0.0, 0])
    n_ver = 0
    s_sum = 0.0
    for cs, cd, cet in C.edges:
        if cs not in inv or cd not in inv:
            continue
        n_ver += 1
        et = getattr(cet, "name", str(cet))
        c_key = _canon((cs, cd, cet))
        cp = c_params.get(c_key) or c_params.get((cs, cd, cet))
        qs, qd = inv[cs], inv[cd]
        cand = list(q_index.get((qs, qd), []))
        if cet in UNDIRECTED_EDGE_TYPES and qs != qd:
            seen = {qk for _a, _b, qk in cand}
            for qet_r, qp_r, qk_r in q_index.get((qd, qs), []):
                if qk_r not in seen:
                    cand.append((qet_r, qp_r, qk_r))
        if not cand:
            out[et, "no query edge"][0] += 1
            continue
        best, best_qp = 0.0, None
        same_kind = False
        for qet, qp, _qk in cand:
            if (qp is None and cp is None and qet == cet) or (
                    qp is not None and cp is not None
                    and getattr(qp, "class_", 0) == getattr(cp, "class_", 1)):
                same_kind = True
            sim = polymorphic_edge_similarity(
                qp, cp, use_confidence=False, edge_types=(qet, cet))
            if sim > best:
                best, best_qp = sim, qp
        s_sum += best
        if best <= 0.0:
            out[et, "same kind, sim 0" if same_kind
                else "no same-kind edge"][0] += 1
            continue
        if best_qp is None and cp is None:
            b = "PARAM-LESS"
        elif not (hasattr(best_qp, "quantized") and hasattr(cp, "quantized")):
            b = "MIXTURE (not classified)"
        elif (best_qp.class_ != cp.class_
              or len(best_qp.quantized) != len(cp.quantized)):
            b = "CROSS-SCHEMA"
        else:
            common = sum(1 for a, c in zip(best_qp.quantized, cp.quantized)
                         if a != UNKNOWN and c != UNKNOWN)
            graded = edge_measured_axes(best_qp, cp)
            if common == 0:
                b = "UNMEASURED"
            elif graded == 0:
                b = "CATEGORICAL ONLY"
            else:
                b = "MEASURED %d" % graded
        row = out[et, b]
        row[0] += 1
        row[1] += best
        row[2] += int(best == 1.0)
    chk = {"n_ver": n_ver, "s_sum": s_sum,
           "t_ver": trace.get("verifiable_core_edges_count"),
           "t_sum": trace.get("matched_rel_sim_sum")}
    return {k: list(v) for k, v in out.items()}, chk


_STATE: Dict[str, Any] = {}


def _work(i: int):
    st = _STATE
    cls, _name, (Q, q_sigs) = st["queries"][i]
    res = {}
    for rc, (C, c_sigs) in st["refs"].items():
        out, chk = classify_pair(Q, C, q_sigs, c_sigs)
        if (chk["n_ver"] != int(chk["t_ver"])
                or abs(chk["s_sum"] - chk["t_sum"]) > 1e-6):
            raise AssertionError("census drifted from the matcher: %r" % chk)
        res[rc] = out
    return cls, res


ORDER = ["no query edge", "no same-kind edge", "same kind, sim 0",
         "PARAM-LESS", "UNMEASURED", "CATEGORICAL ONLY", "CROSS-SCHEMA",
         "MIXTURE (not classified)"]


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

    out: List[str] = []

    def emit(s: str = "") -> None:
        print(s)
        sys.stdout.flush()
        out.append(s)

    emit("PHASE 4 step 0 -- what the matched-edge similarity is made of "
         "(EVAL-75, read-only)")
    emit("source digest %s   EDGE_UNPARAMETERISED_STRUCTURAL_MATCH=%r   "
         "EVIDENCE_GRADED_NODE_CREDIT=%r" % (
             bm.source_digest()[:12],
             getattr(P, "EDGE_UNPARAMETERISED_STRUCTURAL_MATCH", None),
             getattr(P, "EVIDENCE_GRADED_NODE_CREDIT", None)))
    emit("every row is a VERIFIABLE core edge (both endpoints mapped); the "
         "census totals are asserted against the matcher's own trace")
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
        for which, lab in (("own", "own-class pairs (75)"),
                           ("all", "every pair (300)")):
            agg: Dict[Tuple[str, str], List[float]] = defaultdict(
                lambda: [0, 0.0, 0])
            for cls, res in rows:
                for rc, o in res.items():
                    if which == "own" and rc != cls:
                        continue
                    for k, v in o.items():
                        a = agg[k]
                        a[0] += v[0]
                        a[1] += v[1]
                        a[2] += v[2]
            bins = sorted({b for _et, b in agg},
                          key=lambda b: (ORDER.index(b) if b in ORDER
                                         else 100 + int(b.split()[-1])))
            tot_n = sum(v[0] for v in agg.values())
            tot_s = sum(v[1] for v in agg.values()) or 1.0
            emit("=" * 100)
            emit("window = %s   %s   verifiable core edges %d   "
                 "matched_rel_sim_sum %.1f" % (w, lab, tot_n, tot_s))
            emit("=" * 100)
            emit("  %-26s %9s %8s %10s %9s %8s %9s" % (
                "bin", "edges", "% edges", "sim_sum", "% of sum",
                "sim==1", "mean sim"))
            for b in bins:
                n = sum(v[0] for (et, bb), v in agg.items() if bb == b)
                s = sum(v[1] for (et, bb), v in agg.items() if bb == b)
                e1 = sum(v[2] for (et, bb), v in agg.items() if bb == b)
                emit("  %-26s %9d %7.1f%% %10.1f %8.1f%% %8d %9.3f" % (
                    b, n, 100.0 * n / tot_n, s, 100.0 * s / tot_s, e1,
                    (s / n) if n else 0.0))
            emit("")
            emit("  per core-edge type: edges in each bin (sim_sum)")
            types = sorted({et for et, _b in agg},
                           key=lambda t: -sum(v[1] for (et, b), v
                                              in agg.items() if et == t))
            short = [b for b in bins]
            emit("  %-24s" % "edge type" + "".join(
                "%20s" % b[:19] for b in short))
            for et in types:
                cells = []
                for b in short:
                    v = agg.get((et, b))
                    cells.append("%20s" % (
                        "%d (%.0f)" % (v[0], v[1]) if v and v[0] else "."))
                emit("  %-24s" % et[:24] + "".join(cells))
            emit("")
    path = os.path.join(RESULTS_DIR, "phase4_edge_census.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("written: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
