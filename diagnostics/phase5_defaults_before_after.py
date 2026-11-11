"""PHASE 5 -- the declared defaults, before and after the writer fixes.

Reads four stored records (no matcher is run here):

    results/p5_before_{full,v1}.json   main's library (commit 48a4c5c)
    results/p5_after_{full,v1}.json    this tree

Both were produced by `python diagnostics/eth80_benchmark.py --tag ...` on
the same 75 EVAL queries and 4 references, in this session, on this machine.
DESCRIPTIVE: no rule was registered for it, n = 75 (one binomial standard
error is about 4.3 queries) and EVAL-75 is in-sample.

    python diagnostics/phase5_defaults_before_after.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import json
import os

from _bootstrap import RESULTS_DIR


def main() -> int:
    import _stats as st
    from eth80_benchmark import CLASSES

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s, flush=True)

    def load(tag, w):
        with open(os.path.join(RESULTS_DIR, "%s_%s.json" % (tag, w))) as fh:
            return json.load(fh)

    emit("PHASE 5 -- declared defaults before / after the writer fixes, EVAL-75 (in-sample, descriptive)")
    for w in ("full", "v1"):
        a, b = load("p5_before", w), load("p5_after", w)
        emit()
        emit("window = %s" % w)
        emit("  before: source %s  front end %s" % (a["source_digest"][:12], a["frontend_digest"][:12]))
        emit("  after:  source %s  front end %s" % (b["source_digest"][:12], b["frontend_digest"][:12]))
        ha, hb = sum(st.hits(a)), sum(st.hits(b))
        g, l, p = st.paired(a, b)
        emit("  hits %d -> %d of %d   +%d/-%d, exact McNemar p = %.3f   answers changed on %d queries"
             % (ha, hb, len(a["queries"]), g, l, p, st.changed_answers(a, b)))
        pa, pb = st.per_class_hits(a), st.per_class_hits(b)
        emit("  per class (hits): " + "   ".join(
            "%s %s -> %s" % (c, pa[c], pb[c]) for c in CLASSES))
        cmp_ = st.compare_stored(a, b)
        emit("  stored per-pair values: %d compared, %d differ; `score` differs on %d of %d pairs; "
             "largest difference %.4g on `%s`"
             % (cmp_["n"], cmp_["differ"], cmp_["score_differ"],
                len(a["queries"]) * len(CLASSES), cmp_["worst"], cmp_["worst_key"]))
    with open(os.path.join(RESULTS_DIR, "phase5_defaults_before_after.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
