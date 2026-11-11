"""PHASE 5 / A -- which slots hold a measurement that varies?  (read-only)

For every edge class and slot, over the cached graphs: how many edges carry
the slot, the measured share (not UNKNOWN), the number of distinct bins, the
share of the measured values that sit in the MODAL bin and in the TOP bin of
the slot's unit table.  The same for node-signature slots.

A slot is listed as
  ABSENT     never measured (0 % measured): honest, carries nothing
  FROZEN     measured, one distinct bin
  SATURATED  >= 90 % of the measured values in the top bin of the table
  NARROW     >= 90 % of the measured values in one bin (not the top)
Index slots (lookup keys, identity slots) are marked `index`: one value there
is a class, not a frozen measurement.

    python diagnostics/phase5_slot_truth.py [--split eval] [--out NAME]
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import os
from collections import Counter, defaultdict

from _bootstrap import RESULTS_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", default="phase5_slot_truth")
    args = ap.parse_args()

    import core.schema as S
    import eth80_frontend as fe
    import eth80_split as sp
    from core.registries import POPULATION_SLOTS
    from phase5_graph_census import entries_of

    split = sp.load_split()
    entries = entries_of(split, args.split)
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s, flush=True)

    edge = defaultdict(Counter)       # (class, slot index) -> Counter(bin)
    node = defaultdict(Counter)       # (lens, sig class, slot index) -> Counter
    for cls, rel in entries:
        g = fe.graph_for(cls, rel)
        sg = g.subgraph
        for e in sg.edges:
            ep = sg.edge_params.get(e)
            if ep is None:
                continue
            p = ep.dominant if isinstance(ep, S.EdgeParamsMixture) else ep
            for i, b in enumerate(p.quantized):
                edge[(p.class_, i)][b] += 1
        for _n, (lt, s) in g.sigs.items():
            for i, b in enumerate(getattr(s, "quantized", ()) or ()):
                node[(lt, getattr(s, "class_", "?"), i)][b] += 1

    emit("PHASE 5 / A -- slot truth, %d %s graphs, front-end digest %s"
         % (len(entries), args.split.upper(), fe.frontend_source_digest()[:12]))
    emit()
    emit("EDGE SLOTS  (n = edges carrying the slot; measured = not UNKNOWN)")
    emit("  %-27s %-15s %7s %6s %5s %6s %6s  %s" % (
        "class", "slot", "n", "meas%", "bins", "modal%", "top%", "state"))
    tally = Counter()
    for (cls, i), c in sorted(edge.items()):
        specs = S.EDGE_UNIT_TABLE.get(cls, ())
        spec = specs[i] if i < len(specs) else None
        name = spec.name if spec is not None else "s%d" % i
        ids = set(S.EDGE_IDENTITY_SLOTS.get(cls, ()))
        index = (i in ids) or (spec is not None and spec.quantiser == "lookup_key")
        n = sum(c.values())
        meas = {k: v for k, v in c.items() if k != S.UNKNOWN}
        m = sum(meas.values())
        if m == 0:
            state, modal, top = "ABSENT", 0.0, 0.0
        else:
            modal_bin, modal_n = max(meas.items(), key=lambda kv: (kv[1], -kv[0]))
            modal = 100.0 * modal_n / m
            top_bin = (spec.n_bins - 1) if spec is not None else max(meas)
            top = 100.0 * meas.get(float(top_bin), 0) / m
            if index:
                state = "index"
            elif len(meas) == 1:
                state = "FROZEN"
            elif top >= 90.0:
                state = "SATURATED"
            elif modal >= 90.0:
                state = "NARROW"
            else:
                state = "varies"
        tally[state] += 1
        emit("  %-27s %-15s %7d %6.1f %5d %6.1f %6.1f  %s" % (
            cls[5:], name, n, 100.0 * m / max(1, n), len(meas), modal, top, state))
    emit()
    emit("edge slots by state: %s" % dict(sorted(tally.items())))
    emit()
    emit("NODE SIGNATURE SLOTS that take ONE value over every node of the class")
    for (lt, sc, i), c in sorted(node.items()):
        if len(c) != 1:
            continue
        specs = POPULATION_SLOTS.get(sc, ())
        name = getattr(specs[i], "name", None) if i < len(specs) else None
        emit("  %-16s %-18s %-16s = %g  (n = %d)" % (
            lt, sc, name or "s%d" % i, next(iter(c)), sum(c.values())))
    with open(os.path.join(RESULTS_DIR, args.out + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
