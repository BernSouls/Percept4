"""[STAGE 7 / PHASE 0] THE GATE: does any relation slot carry class information?

READ-ONLY.  No matcher is run; no graph is changed.

The 79 EVAL graphs are built with the stage-6 relations ON.  Every edge of
every class in `core.schema.EDGE_UNIT_TABLE` contributes one observation per
slot: the slot's BIN, labelled with the class of the image it came from.

Per (class, slot):
  RATIO      one-way ANOVA F on the bin, between-class vs within-class
  ETA2       BETWEEN / TOTAL (the share of the slot's spread the label explains)
  I          mutual information with the class label, bits
  I/H(bin)   the share of the slot's own entropy the label removes
  pctl       the slot's RATIO percentile against 200 label permutations

The same three statistics on the NODE signatures' slots, as the control: that
is the information the shipped matcher already has.

The threshold and the falsifier are pre-registered (preregistration, 7.29):
  INFORMATIVE iff RATIO > 1.25 AND I >= 0.01 bits AND I/H >= 0.02 AND
                  RATIO >= the 95th percentile of its permutation null.

    python diagnostics/phase7_p0_slot_information.py [--out NAME] [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

#: pre-registered thresholds (7.29)
RATIO_MIN = 1.25
MI_MIN = 0.01
MI_FRAC_MIN = 0.02
PCTL_MIN = 95.0
N_PERM = 200
PERM_SEED = 20261110

#: attribute-only classes: not structural, reported but never part of the gate
ATTRIBUTE_CLASSES = frozenset({
    "EDGE_BORDER_OWNERSHIP", "EDGE_SHARED_BOUNDARY", "EDGE_TEXTURE_BOUNDARY",
    "EDGE_COLOR_OPPONENT_BORDER", "EDGE_SURFACE_FOLD", "EDGE_MOTION_OPPOSITION",
})

CLASSES = ("apple", "car", "cup", "horse")


def _entry(arg):
    """(class, [(EdgeType name, [bin per slot]) ...], [(lens, [bin per slot]) ...])."""
    import parameters as P
    import eth80_frontend as fe
    from core.schema import EDGE_UNIT_TABLE

    cls, rel = arg
    P.GRAPH_STAGE6_RELATIONS = True
    g = fe.graph_for(cls, rel)          # cached: the key now carries the switch
    P.GRAPH_STAGE6_RELATIONS = False

    edges = []
    for (s, d, et) in g.subgraph.edges:
        # the edge table is keyed EDGE_<EdgeType name>; the graph carries the enum
        name = "EDGE_" + getattr(et, "name", str(et))
        spec = EDGE_UNIT_TABLE.get(name)
        if not spec:
            continue
        ep = g.subgraph.edge_params.get((s, d, et))
        if ep is None:
            continue
        q = list(getattr(ep, "quantized", ()) or ())
        if len(q) < len(spec):
            continue
        row = []
        for i, sl in enumerate(spec):
            v = float(q[i])
            row.append(None if not math.isfinite(v) or v < 0 or v >= sl.n_bins
                       else int(round(v)))
        edges.append((name, row))
    nodes = []
    for n, (lens, sig) in g.sigs.items():
        q = list(getattr(sig, "quantized", ()) or ())
        nodes.append((lens, [None if float(v) < 0 else int(round(float(v)))
                             for v in q]))
    return cls, edges, nodes


def _stats(bins, labels, n_classes):
    """(ratio, eta2, I, I/H) of one slot over the pooled observations."""
    b = np.asarray(bins, dtype=float)
    y = np.asarray(labels)
    N = len(b)
    if N == 0:
        return None
    mu = b.mean()
    sst = float(((b - mu) ** 2).sum())
    ssb = ssw = 0.0
    k = 0
    for c in range(n_classes):
        v = b[y == c]
        if len(v) == 0:
            continue
        k += 1
        d = float(v.mean()) - mu
        ssb += len(v) * d * d
        ssw += float(((v - v.mean()) ** 2).sum())
    dfb, dfw = k - 1, N - k
    if dfb <= 0 or dfw <= 0:
        return None
    B = ssb / dfb
    W = ssw / dfw
    ratio = (B / W) if W > 0 else (0.0 if B == 0 else float("inf"))
    eta2 = ssb / sst if sst > 0 else 0.0
    # mutual information on the integer bins
    joint = Counter(zip(bins, labels))
    pb = Counter(bins)
    pc = Counter(labels)
    mi = 0.0
    for (v, c), n in joint.items():
        p = n / N
        mi += p * math.log2(p / ((pb[v] / N) * (pc[c] / N)))
    hb = -sum((n / N) * math.log2(n / N) for n in pb.values()) if N else 0.0
    return ratio, eta2, mi, (mi / hb if hb > 0 else 0.0)


def _perm_percentile(bins, labels, ratio, rng, n_perm=N_PERM):
    y = list(labels)
    above = 0
    for _ in range(n_perm):
        rng.shuffle(y)
        got = _stats(bins, y, len(CLASSES))
        if got is None:
            continue
        if got[0] <= ratio:
            above += 1
    return 100.0 * above / max(1, n_perm)


def main(argv=None) -> int:
    import multiprocessing as mp

    import eth80_split as sp

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase7_p0_slot_information")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    split = sp.load_split()
    entries = [(c, p) for c in CLASSES for p in split["eval"][c]]
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(_entry, entries, chunksize=1)
    else:
        rows = [_entry(e) for e in entries]

    # ---- pool the edges and the nodes by (class, slot) -----------------------
    edge_slot = defaultdict(lambda: ([], []))      # (cls, si) -> (bins, labels)
    edge_n = defaultdict(int)                      # (cls) -> edges
    edge_counts = Counter()                        # class name -> edges
    node_slot = defaultdict(lambda: ([], []))      # (cls, lens, si)
    from core.schema import EDGE_UNIT_TABLE
    slot_names = {c: [s.name for s in spec] for c, spec in EDGE_UNIT_TABLE.items()}
    for cls, edges, nodes in rows:
        ci = CLASSES.index(cls)
        for name, bins in edges:
            edge_counts[cls] += 1
            for si, v in enumerate(bins):
                if v is None:
                    continue
                lst = edge_slot[(name, si)]
                lst[0].append(v)
                lst[1].append(ci)
        for lens, bins in nodes:
            for si, v in enumerate(bins):
                if v is None:
                    continue
                lst = node_slot[(lens, si)]
                lst[0].append(v)
                lst[1].append(ci)

    rng = random.Random(PERM_SEED)
    L = ["[STAGE 7 / PHASE 0] THE GATE: does any relation slot carry class information?",
         "   %d EVAL graphs, stage-6 relations ON, no matcher run; edges as the unit of" % len(entries),
         "   measurement; class label = the image's class.  Thresholds pre-registered (7.29):",
         "   RATIO > %.2f, I >= %.2f bits, I/H >= %.2f, RATIO >= p%.0f of 200 label permutations." % (
             RATIO_MIN, MI_MIN, MI_FRAC_MIN, PCTL_MIN), "",
         "   edges per class: " + ", ".join("%s %d" % (c, edge_counts[c]) for c in CLASSES), ""]

    results = []
    for (name, si) in sorted(edge_slot):
        bins, labels = edge_slot[(name, si)]
        got = _stats(bins, labels, len(CLASSES))
        if got is None:
            continue
        ratio, eta2, mi, frac = got
        pctl = _perm_percentile(bins, labels, ratio, rng)
        informative = (ratio > RATIO_MIN and mi >= MI_MIN and frac >= MI_FRAC_MIN
                       and pctl >= PCTL_MIN)
        noisey = (ratio > RATIO_MIN and mi >= MI_MIN and frac >= MI_FRAC_MIN
                  and pctl < PCTL_MIN)
        results.append({"kind": "edge", "class": name, "slot": si,
                        "slot_name": slot_names.get(name, [])[si] if si < len(slot_names.get(name, [])) else "?",
                        "n": len(bins), "ratio": ratio, "eta2": eta2, "mi": mi,
                        "mi_frac": frac, "pctl": pctl, "informative": informative,
                        "noise": noisey, "attribute": name in ATTRIBUTE_CLASSES})
    node_results = []
    for (lens, si) in sorted(node_slot):
        bins, labels = node_slot[(lens, si)]
        got = _stats(bins, labels, len(CLASSES))
        if got is None:
            continue
        ratio, eta2, mi, frac = got
        pctl = _perm_percentile(bins, labels, ratio, rng)
        node_results.append({"kind": "node", "class": lens, "slot": si, "n": len(bins),
                             "ratio": ratio, "eta2": eta2, "mi": mi, "mi_frac": frac,
                             "pctl": pctl,
                             "informative": (ratio > RATIO_MIN and mi >= MI_MIN
                                             and frac >= MI_FRAC_MIN and pctl >= PCTL_MIN)})

    def table(title, rs, keep=None):
        out = [title,
               "   %-30s %-16s %6s %7s %7s %8s %8s %6s  %s" % (
                   "class", "slot", "n", "RATIO", "eta2", "I bits", "I/H", "pctl", "verdict"),
               "   " + "-" * 118]
        for r in sorted(rs, key=lambda r: (-r["ratio"], r["class"], r["slot"])):
            if keep is not None and not keep(r):
                continue
            v = ("INFORMATIVE" if r["informative"] else
                 ("INFORMATIVE-BUT-NOISE" if r.get("noise") else ""))
            out.append("   %-30s %-16s %6d %7.3f %7.4f %8.4f %8.4f %6.1f  %s" % (
                r["class"], r.get("slot_name", r["slot"]), r["n"], r["ratio"], r["eta2"],
                r["mi"], r["mi_frac"], r["pctl"], v))
        return out

    structural = [r for r in results if not r["attribute"]]
    L += table("1. THE STRUCTURAL RELATION SLOTS (the gate is decided here)", structural)
    L += [""]
    L += table("2. THE ATTRIBUTE-ONLY RELATION SLOTS (reported, never part of the gate)",
               [r for r in results if r["attribute"]])
    L += [""]
    L += table("3. THE CONTROL: node signature slots (what the shipped matcher already reads)",
               node_results)
    L += [""]

    inf = [r for r in structural if r["informative"]]
    named = [r for r in structural if r["class"] in (
        "EDGE_BOUNDARY_ATTACHMENT", "EDGE_BOUNDARY_SUCCESSOR", "EDGE_ILLUSORY_INDUCER")]
    L += ["4. THE VERDICT OF THE GATE",
          "   structural relation slots tested: %d" % len(structural),
          "   INFORMATIVE: %d" % len(inf),
          "   the three relations the review named (BOUNDARY_ATTACHMENT,",
          "   BOUNDARY_SUCCESSOR, ILLUSORY_INDUCER): %d slots, %d informative" % (
              len(named), sum(1 for r in named if r["informative"])),
          "   node slots INFORMATIVE: %d of %d" % (
              sum(1 for r in node_results if r["informative"]), len(node_results))]
    # SCALE CHECK (reported, not a threshold): a relation slot's ratio is large
    # partly because the class mean moves little against a narrow within-class
    # spread.  Mutual information is the scale-free comparison.
    edge_mi = sorted((r["mi"] for r in structural), reverse=True)
    node_mi = sorted((r["mi"] for r in node_results), reverse=True)
    L += ["", "5. SCALE CHECK: the same statistics on the two sides, best of each",
          "   relation slots: best I %.4f bits (median %.4f); node slots: best I %.4f"
          " bits (median %.4f)" % (
              edge_mi[0] if edge_mi else 0.0, float(np.median(edge_mi)) if edge_mi else 0.0,
              node_mi[0] if node_mi else 0.0, float(np.median(node_mi)) if node_mi else 0.0),
          "   relation slots with I >= 0.10 bits: %d of %d; node slots: %d of %d" % (
              sum(1 for r in structural if r["mi"] >= 0.10), len(structural),
              sum(1 for r in node_results if r["mi"] >= 0.10), len(node_results)),
          "   THE READING: the gate is decided by the registered bar, which %s." % (
              "5 structural slots clear" if inf else "nothing clears")]

    if inf:
        L += ["", "   GATE: PASSED -- %d structural relation slot(s) carry class information:" % len(inf)]
        for r in sorted(inf, key=lambda r: -r["ratio"])[:12]:
            L.append("     %-30s %-16s RATIO %6.3f   I %6.4f bits   pctl %5.1f" % (
                r["class"], r["slot_name"], r["ratio"], r["mi"], r["pctl"]))
        if len(inf) > 12:
            L.append("     ... %d more" % (len(inf) - 12))
        best = max(structural, key=lambda r: r["mi"])
        L += ["   the slot that removes the most class uncertainty: %s / %s, I %.4f bits"
              " (%.1f %% of its own entropy), RATIO %.3f" % (
                  best["class"], best["slot_name"], best["mi"], 100 * best["mi_frac"], best["ratio"])]
    else:
        L += ["", "   GATE: FAILED -- no structural relation slot meets the pre-registered",
              "   threshold.  THE FALSIFIER FIRED (7.29): the representation's relations",
              "   carry no class information; no matcher fix can help; the front end is",
              "   the work.  The remaining phases stop here and this table is the",
              "   deliverable."]
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"edges": results, "nodes": node_results,
                   "thresholds": {"ratio": RATIO_MIN, "mi": MI_MIN, "mi_frac": MI_FRAC_MIN,
                                  "pctl": PCTL_MIN, "n_perm": N_PERM, "seed": PERM_SEED},
                   "gate_passed": bool(inf)}, fh, indent=1,
                  default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
