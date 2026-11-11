"""PHASE 4, step 0b -- what the edge parameters of each relation can say.

Read-only, no matching.  Over the 79 EVAL graphs (4 references + 75 queries),
for every edge type and every parameter slot:

  measured   share of edges whose slot is not UNKNOWN
  values     number of distinct measured values
  H          entropy of the measured values / log(number of bins seen), 0..1
             (0 = one value everywhere: the slot cannot tell two edges apart)

and per edge type

  P(same)    probability that two edges of this type drawn at random carry
             IDENTICAL parameter tuples -- the rate at which a matched pair
             would read exactly 1.0 with no correspondence at all.

A relation whose every measured slot has H = 0 (P(same) = 1) is measured in
the letter and a count in effect: its matched similarity is 1.0 whatever was
matched.

    python diagnostics/phase4_axis_census.py [--window full|v1|both]
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

from _bootstrap import RESULTS_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", default="both", choices=("full", "v1", "both"))
    args = ap.parse_args()
    windows = ("full", "v1") if args.window == "both" else (args.window,)

    import eth80_benchmark as bm
    import eth80_frontend as fe
    from core.schema import EDGE_SCHEMAS, UNKNOWN
    from eth80_split import load_split

    out: List[str] = []

    def emit(s: str = "") -> None:
        print(s)
        sys.stdout.flush()
        out.append(s)

    split = load_split()
    entries = list(split["references"].items()) + [
        (c, p) for c in bm.CLASSES for p in split["eval"][c]]
    emit("PHASE 4 step 0b -- parameter values per relation, 79 EVAL graphs "
         "(read-only, no matching)")
    emit("front-end digest %s" % fe.frontend_source_digest()[:12])
    emit("")
    for w in windows:
        n_edges: Counter = Counter()
        no_params: Counter = Counter()
        classes: Dict[str, Counter] = defaultdict(Counter)
        tuples: Dict[str, Counter] = defaultdict(Counter)
        slots: Dict[str, Dict[int, Counter]] = defaultdict(
            lambda: defaultdict(Counter))
        for cls, rel in entries:
            sg, _sigs = bm.make_window(fe.graph_for(cls, rel), w)
            params = sg.edge_params or {}
            for e in sg.edges:
                et = e[2].name
                n_edges[et] += 1
                p = params.get(e)
                if p is None:
                    p = params.get((e[1], e[0], e[2]))
                if p is None:
                    no_params[et] += 1
                    continue
                protos = getattr(p, "protos", None) or [p]
                for pr in protos:
                    classes[et][pr.class_] += 1
                    tuples[et][(pr.class_, tuple(pr.quantized),
                                getattr(pr, "sign", 1))] += 1
                    for i, v in enumerate(pr.quantized):
                        slots[et][i][v] += 1
        emit("=" * 104)
        emit("window = %s   %d edges in %d graphs" % (
            w, sum(n_edges.values()), len(entries)))
        emit("=" * 104)
        emit("  %-26s %8s %8s %8s   %s" % (
            "edge type", "edges", "params", "P(same)",
            "slots: tag  measured%  values  H"))
        for et, n in n_edges.most_common():
            tot = sum(tuples[et].values())
            p_same = (sum(c * c for c in tuples[et].values()) / float(tot * tot)
                      if tot else float("nan"))
            cls_name = (classes[et].most_common(1)[0][0]
                        if classes[et] else None)
            schema = EDGE_SCHEMAS.get(cls_name, ()) if cls_name else ()
            cells = []
            for i in sorted(slots[et]):
                cnt = slots[et][i]
                meas = {v: c for v, c in cnt.items() if v != UNKNOWN}
                m = sum(meas.values())
                tag = (schema[i].tag if i < len(schema) else "slot%d" % i)
                if m == 0:
                    cells.append("%s 0%% - -" % tag)
                    continue
                h = -sum((c / m) * math.log(c / m) for c in meas.values())
                hn = h / math.log(len(meas)) if len(meas) > 1 else 0.0
                cells.append("%s %.0f%% %d %.2f" % (
                    tag, 100.0 * m / sum(cnt.values()), len(meas), hn))
            emit("  %-26s %8d %7.0f%% %8s   %s" % (
                et[:26], n, 100.0 * (n - no_params[et]) / n,
                ("%.3f" % p_same) if tot else "-",
                " | ".join(cells) if cells else "(no parameters)"))
        emit("")
    path = os.path.join(RESULTS_DIR, "phase4_axis_census.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("written: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
