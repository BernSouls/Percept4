"""[STAGE 6 / R4] THE RULERS: are the distance slots spanned to what is measured?

For every edge IN the graphs, the RAW (continuous) values its writer handed
to the edge constructor (matched to the constructor call by object identity;
BOUNDARY_ATTACHMENT, whose writer copies the object, by class and bins).

 A. The distance slots (unit `wavelengths` or `median_spacings`): measured
    values, the share in the last bin or outside the span, and, where that
    share exceeds 25 %, the new span by the registered rule (upper end = the
    99th percentile rounded up to the next 0.5), the edges that change bin
    and the entropy of the bin histogram before and after.
 B. The frozen slots, from their raw values: constant by construction, or
    varying and flattened by the quantiser?

    PERCEPT_SET=GRAPH_STAGE6_RELATIONS=1 python diagnostics/phase6_rulers.py [--workers 2] [--out NAME]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import eth80_frontend as fe  # noqa: E402
import eth80_split as sp  # noqa: E402

DISTANCE_UNITS = ("wavelengths", "median_spacings")
FROZEN = (("EDGE_CURVATURE_SIGN_CONTINUITY", "curv_diff"), ("EDGE_CONTACT", "depth_step"),
          ("EDGE_CONTOUR_CONTINUITY", "curv_agree"))


def one(entry):
    import core.schema as S
    import orchestration.binder as B
    cls, rel = entry
    rec = []
    orig = B._edge_params

    def wrap(class_name, values, *a, **k):
        ep = orig(class_name, values, *a, **k)
        rec.append((class_name, dict(values), ep))
        return ep

    B._edge_params = wrap
    try:
        g = fe.build_graph(fe._load_rgb(fe.image_path(cls, rel)), name=os.path.basename(rel),
                           record_co_located=False)
    finally:
        B._edge_params = orig
    by_id = {id(ep): vals for _c, vals, ep in rec}
    by_bins = defaultdict(list)
    for c, vals, ep in rec:
        by_bins[(c, tuple(ep.quantized))].append(vals)
    out = defaultdict(lambda: defaultdict(list))
    found = Counter()
    candidates = Counter(c for c, _v, _e in rec)
    for _e, ep in g.subgraph.edge_params.items():
        protos = ep.protos if isinstance(ep, S.EdgeParamsMixture) else [ep]
        for p in protos:
            vals = by_id.get(id(p))
            how = "identity"
            if vals is None:
                pool = by_bins.get((p.class_, tuple(p.quantized)))
                vals, how = (pool[0], "bins") if pool else (None, "none")
            found[(p.class_, how)] += 1
            if vals is None:
                continue
            for name, v in vals.items():
                out[p.class_][name].append(None if v is None else float(v))
    return {"raw": {c: dict(v) for c, v in out.items()},
            "found": {"%s|%s" % k: n for k, n in found.items()},
            "candidates": dict(candidates)}


def _bins(values, lo, hi, n):
    v = np.asarray(values, dtype=float)
    return np.clip(np.floor((v - lo) / (hi - lo) * n), 0, n - 1).astype(int)


def _entropy(b, n):
    h = np.bincount(b, minlength=n).astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def main(argv=None) -> int:
    import core.schema as S
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="phase6_rulers")
    ap.add_argument("--split", default="eval")
    a = ap.parse_args(argv)
    split = sp.load_split()
    entries = list(split["references"].items()) + [
        (c, p) for c, ps in split[a.split].items() for p in ps]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(one, entries, chunksize=1)
    else:
        rows = [one(e) for e in entries]
    raw = defaultdict(lambda: defaultdict(list))
    found, cand = Counter(), Counter()
    for r in rows:
        for c, slots in r["raw"].items():
            for name, vals in slots.items():
                raw[c][name] += vals
        found.update(r["found"])
        cand.update(r["candidates"])
    L = ["[STAGE 6 / R4] the rulers, %d graphs (4 references + %d %s); PERCEPT_SET: %s" % (
        len(rows), len(rows) - 4, a.split.upper(), _bootstrap.switch_overrides_key() or "none"),
         "edges matched to their constructor call: by identity %d, by class and bins %d, not matched %d" % (
             sum(n for k, n in found.items() if k.endswith("|identity")),
             sum(n for k, n in found.items() if k.endswith("|bins")),
             sum(n for k, n in found.items() if k.endswith("|none"))), "",
         "A. DISTANCE SLOTS (the rule: re-ruled when more than 25 % of the measured values are in the last bin or",
         "   outside the span; new upper end = 99th percentile rounded up to the next 0.5 of the unit)",
         "   class.slot                                  unit             span        n     median    p90    p99    max   last bin+   first bin   verdict"]
    ruled = {}
    for c in sorted(S.EDGE_UNIT_TABLE):
        for spec in S.EDGE_UNIT_TABLE[c]:
            if spec.unit not in DISTANCE_UNITS:
                continue
            vals = [v for v in raw.get(c, {}).get(spec.name, []) if v is not None]
            label = "%s.%s" % (c.replace("EDGE_", ""), spec.name)
            if not vals:
                L.append("   %-43s %-15s [%g, %g]   %5d   no measured value in these graphs" % (
                    label, spec.unit, spec.lo, spec.hi, 0))
                continue
            v = np.abs(np.asarray(vals, dtype=float)) if spec.lo < 0 else np.asarray(vals, dtype=float)
            lo = 0.0 if spec.lo < 0 else float(spec.lo)
            hi, n = float(spec.hi), int(spec.n_bins)
            n_eff = n // 2 if spec.lo < 0 else n          # a signed slot has n/2 bins per side
            width = (hi - lo) / n_eff
            last = float(np.mean((v >= hi - width) | (v < lo)))
            first = float(np.mean(v < lo + width))
            verdict = "kept"
            if last > 0.25:
                new_hi = math.ceil(float(np.percentile(v, 99)) / 0.5) * 0.5
                verdict = "RE-RULED -> [%g, %g]" % (spec.lo if spec.lo >= 0 else -new_hi, new_hi)
                b0, b1 = _bins(v, lo, hi, n_eff), _bins(v, lo, new_hi, n_eff)
                ruled[label] = {"class": c, "slot": spec.name, "old": [spec.lo, spec.hi],
                                "new_hi": new_hi, "n": int(len(v)),
                                "changed": int(np.sum(b0 != b1)),
                                "entropy_before": _entropy(b0, n_eff),
                                "entropy_after": _entropy(b1, n_eff),
                                "p99": float(np.percentile(v, 99)), "max": float(v.max()),
                                "hist_before": np.bincount(b0, minlength=n_eff).tolist(),
                                "hist_after": np.bincount(b1, minlength=n_eff).tolist()}
            elif first > 0.90:
                verdict = "kept (NARROW: %.0f %% in the first bin; reported, not changed)" % (100 * first)
            L.append("   %-43s %-15s [%g, %g]  %6d   %6.2f %6.2f %6.2f %6.2f     %5.2f       %5.2f     %s" % (
                label, spec.unit, spec.lo, spec.hi, len(v), np.median(v), np.percentile(v, 90),
                np.percentile(v, 99), v.max(), last, first, verdict))
    L += ["", "   the re-ruled slots: edges that change bin, entropy of the bin histogram (bits; 3 = uniform over 8 bins)",
          "   class.slot                                  old span    new span     n     change bin     entropy before -> after    histogram before -> after"]
    for label in sorted(ruled):
        r = ruled[label]
        L.append("   %-43s [%g, %g]   [%g, %g]  %6d   %6d (%.2f)      %.2f -> %.2f              %s -> %s" % (
            label, r["old"][0], r["old"][1], 0.0 if r["old"][0] >= 0 else -r["new_hi"], r["new_hi"],
            r["n"], r["changed"], r["changed"] / r["n"], r["entropy_before"], r["entropy_after"],
            r["hist_before"], r["hist_after"]))
    L += ["", "B. THE FROZEN SLOTS, from the raw values of the edges in the graphs"]
    frozen = {}
    for c, name in FROZEN:
        vals = raw.get(c, {}).get(name, [])
        meas = [v for v in vals if v is not None]
        spec = next((sp_ for sp_ in S.EDGE_UNIT_TABLE.get(c, ()) if sp_.name == name), None)
        if spec is None:
            L.append("   %s.%s: no such slot" % (c, name))
            continue
        distinct = sorted(set(round(v, 9) for v in meas))
        width = (spec.hi - spec.lo) / spec.n_bins
        varies = len(distinct) >= 3 and (max(meas) - min(meas)) > width if meas else False
        verdict = ("VARIES AND FLATTENED" if varies else
                   ("CONSTANT (one value)" if len(distinct) == 1 else
                    ("never measured" if not meas else "%d distinct values inside one bin width" % len(distinct))))
        frozen["%s.%s" % (c, name)] = {"n": len(vals), "measured": len(meas), "distinct": distinct[:12],
                                       "verdict": verdict}
        L.append("   %-44s unit %-22s span [%g, %g] x %d bins: edges %d, measured %d (None %d); distinct raw values %d%s" % (
            "%s.%s" % (c.replace("EDGE_", ""), name), spec.unit, spec.lo, spec.hi, spec.n_bins,
            len(vals), len(meas), len(vals) - len(meas), len(distinct),
            "" if not meas else "; min %.4g, median %.4g, max %.4g; first values %s" % (
                min(meas), float(np.median(meas)), max(meas), [round(x, 4) for x in distinct[:8]])))
        L.append("      -> %s" % verdict)
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w") as fh:
        json.dump({"ruled": ruled, "frozen": frozen,
                   "quantiles": {c: {n: (None if not [v for v in vs if v is not None] else
                                         [float(np.percentile([v for v in vs if v is not None], q))
                                          for q in (1, 10, 25, 50, 75, 90, 99, 100)])
                                     for n, vs in slots.items()} for c, slots in raw.items()}},
                  fh, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
