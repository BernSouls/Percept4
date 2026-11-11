"""STAGE 4 -- what one ``evidence_score`` call costs, with the numpy kernels
and with the Python loops (``MATCH_SIGNATURE_MATRIX_BLOCKS``).

Every (query, reference) pair of EVAL-75, each window, one process, the
harness's own breakdowns excluded: the call is timed alone, once with the
switch off and once with it on, back to back (which goes first alternates
from pair to pair, after one untimed call each way), and the two results
(score, correspondence, every float of the trace) are compared with ``==``.
The declared defaults otherwise.  Run it on an idle machine: the figures
are wall time.

    python diagnostics/phase3_speed.py [--window full|v1|both] [--pairs N]

``--pairs N`` times the first N queries of each class only (N * 16 pairs).
Diagnostics only; nothing is stored but the table.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import os
import time
from typing import Any, Dict, List

from _bootstrap import RESULTS_DIR
from _stats import CLASSES


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", default="both")
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        RESULTS_DIR, "phase3_speed.txt"))
    args = ap.parse_args()
    windows = ["full", "v1"] if args.window == "both" else [args.window]

    import eth80_benchmark as bm
    import eth80_frontend as fe
    import parameters as P
    from eth80_split import load_split
    from memory.cores import evidence_score

    out: List[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        out.append(s)

    emit("STAGE 4 -- seconds per evidence_score call: Python loops against "
         "the numpy kernels (MATCH_SIGNATURE_MATRIX_BLOCKS), EVAL-75")
    emit("source %s  front end %s   environment %s" % (
        bm.source_digest()[:12], fe.frontend_source_digest()[:12],
        bm.environment()))
    split = load_split()
    keep = P.MATCH_SIGNATURE_MATRIX_BLOCKS
    differing = 0
    try:
        for w in windows:
            refs = {c: bm.make_window(
                fe.graph_for(c, split["references"][c]), w) for c in CLASSES}
            queries = []
            for c in CLASSES:
                names = split["eval"][c]
                if args.pairs:
                    names = names[:args.pairs]
                queries += [bm.make_window(fe.graph_for(c, p), w)
                            for p in names]
            pairs = [(q, refs[c]) for q in queries for c in CLASSES]
            sizes = []
            for (Q, qs), (C, cs) in pairs:
                ql = bm.lens_counts({n: qs[n] for n in Q.nodes if n in qs})
                cl = bm.lens_counts({n: cs[n] for n in C.nodes if n in cs})
                sizes.append(sum(ql[lt] * cl.get(lt, 0) for lt in ql))
            res: Dict[bool, List[Any]] = {False: [], True: []}
            secs: Dict[bool, List[float]] = {False: [], True: []}
            # one untimed call each way first (imports, lazily built tables)
            for blocks in (False, True):
                P.MATCH_SIGNATURE_MATRIX_BLOCKS = blocks
                (Q, qs), (C, cs) = pairs[0]
                evidence_score(Q, C, qs, cs, trace={})
            # each pair both ways, back to back; which goes first alternates
            for i, ((Q, qs), (C, cs)) in enumerate(pairs):
                for blocks in ((False, True) if i % 2 == 0 else (True, False)):
                    P.MATCH_SIGNATURE_MATRIX_BLOCKS = blocks
                    tr: Dict[str, Any] = {}
                    t0 = time.perf_counter()
                    score, corr = evidence_score(Q, C, qs, cs, trace=tr)
                    secs[blocks].append(time.perf_counter() - t0)
                    res[blocks].append((score, corr, tr))
            bad = sum(int(a != b) for a, b in zip(res[False], res[True]))
            differing += bad
            tl, tb = sum(secs[False]), sum(secs[True])
            ratios = [a / b for a, b in zip(secs[False], secs[True])]
            emit("")
            emit("window = %s: %d calls; node pairs compared per call "
                 "(same lens type): %d to %d, median %d" % (
                     w, len(pairs), min(sizes), max(sizes),
                     _median([float(x) for x in sizes])))
            emit("  results differing between the two (score, "
                 "correspondence, every trace value): %d%s" % (
                     bad, "   FINDING" if bad else ""))
            emit("  Python loops   total %7.1f s   per call: mean %.3f s, "
                 "median %.3f s, slowest %.3f s" % (
                     tl, tl / len(pairs), _median(secs[False]),
                     max(secs[False])))
            emit("  numpy kernels  total %7.1f s   per call: mean %.3f s, "
                 "median %.3f s, slowest %.3f s" % (
                     tb, tb / len(pairs), _median(secs[True]),
                     max(secs[True])))
            emit("  ratio of the totals x%.1f; per call: median x%.1f, "
                 "smallest x%.1f, largest x%.1f" % (
                     tl / tb, _median(ratios), min(ratios), max(ratios)))
    finally:
        P.MATCH_SIGNATURE_MATRIX_BLOCKS = keep
    with open(args.out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return 2 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main())
