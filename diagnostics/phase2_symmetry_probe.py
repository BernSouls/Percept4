"""STAGE 2 -- is the balanced (D3) score symmetric on real graphs?  (probe)

The D3 formulas are symmetric in the two graphs, and the pins show
score(Q, C) == score(C, Q) on hand-built graphs.  The node correspondence is
not symmetric by construction: the matcher keeps, per QUERY node, its top 8
core candidates (``_max_weight_bipartite_matching``), so swapping the
arguments can return a different correspondence.  This measures how much
that moves the score on a sample: every fifth EVAL query of each class (3 per
class, 12 queries) against the four references, both windows.

Every switch is set explicitly (F6, D1 and Phase 4 on, calibration off, all
three terms two-sided).  The probe runs three times: the state stage 2 was
measured in (mixtures read backwards unreversed, approximate assignment);
with EDGE_REVERSE_FLIPS_MIXTURE on -- the part of the asymmetry that was the
edge loop's; and with MATCH_ASSIGNMENT_EXACT on as well -- the part that was
the per-row pruning of the correspondence.

Read-only; nothing is stored but the text.  Not a pre-registered measure.

    python diagnostics/phase2_symmetry_probe.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import os
import sys
from typing import List

from _bootstrap import RESULTS_DIR

ALL = ("node", "relation", "absence")
MX = "EDGE_REVERSE_FLIPS_MIXTURE"
MA = "MATCH_ASSIGNMENT_EXACT"


def main() -> int:
    import eth80_benchmark as bm
    import eth80_frontend as fe
    import parameters as P
    from eth80_split import load_split
    from memory.cores import evidence_score

    out: List[str] = []

    def emit(s: str = "") -> None:
        print(s)
        sys.stdout.flush()
        out.append(s)

    split = load_split()
    emit("STAGE 2 probe -- score(Q, C) against score(C, Q) under "
         "EVIDENCE_QUERY_SIDE_TERMS = %r" % (ALL,))
    emit("source digest %s   sample: every fifth EVAL query of each class, "
         "3 per class; F6, D1, Phase 4 on, calibration off" %
         bm.source_digest()[:12])
    base = {"EDGE_UNPARAMETERISED_STRUCTURAL_MATCH": True,
            "EVIDENCE_GRADED_NODE_CREDIT": True,
            "EVIDENCE_QUERY_SIDE_TERMS": ALL,
            "EDGE_UNMEASURED_IS_NOT_EVIDENCE": True,
            "EVIDENCE_NODE_NULL_CALIBRATED": False}
    from phase2_d3_p4_null import STAGE4_NEUTRAL
    base.update(STAGE4_NEUTRAL)
    old = {k: getattr(P, k) for k in list(base) + [MX, MA]}
    for k, v in base.items():
        setattr(P, k, v)
    try:
        for flip, exact, note in (
                (False, False, "the historical edge loop and assignment "
                               "(stage 2)"),
                (True, False, "mixtures reversed; approximate assignment"),
                (True, True, "mixtures reversed; EXACT assignment")):
            setattr(P, MX, flip)
            setattr(P, MA, exact)
            emit("")
            emit("%s = %r, %s = %r   (%s)" % (MX, flip, MA, exact, note))
            for w in ("full", "v1"):
                refs = {c: bm.make_window(
                    fe.graph_for(c, split["references"][c]), w)
                    for c in bm.CLASSES}
                d: List[float] = []
                nd: List[float] = []
                rd: List[float] = []
                same_corr = 0
                flips: List[str] = []
                nq = 0
                for c in bm.CLASSES:
                    for p in split["eval"][c][::5][:3]:
                        Q, qs = bm.make_window(fe.graph_for(c, p), w)
                        nq += 1
                        fwd, bwd = {}, {}
                        for rc, (C, cs) in refs.items():
                            ta: dict = {}
                            tb: dict = {}
                            a, ca = evidence_score(Q, C, qs, cs, trace=ta)
                            b, cb = evidence_score(C, Q, cs, qs, trace=tb)
                            d.append(abs(a - b))
                            nd.append(abs(ta["node_coverage"]
                                          - tb["node_coverage"]))
                            rd.append(abs(ta["relational_consistency"]
                                          - tb["relational_consistency"]))
                            same_corr += int(
                                {(q_, c_) for q_, c_ in ca.items()}
                                == {(q_, c_) for c_, q_ in cb.items()})
                            fwd[rc], bwd[rc] = a, b
                        wa = max(bm.CLASSES, key=lambda k: fwd[k])
                        wb = max(bm.CLASSES, key=lambda k: bwd[k])
                        if wa != wb:
                            flips.append("%s (%s vs %s)" % (
                                os.path.basename(p), wa, wb))
                d.sort()
                nd.sort()
                rd.sort()
                n = len(d)
                emit("window = %-4s %d pairs: |score(Q,C) - score(C,Q)|  "
                     "mean %.4f  median %.4f  max %.4f   node term: mean "
                     "%.4f  max %.4f   relational term: mean %.4f  max %.4f"
                     "   same correspondence both ways: %d of %d pairs   "
                     "queries whose answer depends on the direction: %d of "
                     "%d%s"
                     % (w, n, sum(d) / n, d[n // 2], d[-1], sum(nd) / n,
                        nd[-1], sum(rd) / n, rd[-1], same_corr, n,
                        len(flips), nq,
                        (": " + "; ".join(flips)) if flips else ""))
    finally:
        for k, v in old.items():
            setattr(P, k, v)
    with open(os.path.join(RESULTS_DIR, "phase2_symmetry_probe.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
