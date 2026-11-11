"""PHASE 0 / AUDIT A (read-only) -- what do CO_LOCATED edge parameters carry?

Question from the handoff: "CO_LOCATED edge params (scale_ratio,
salience_ratio) may be constant by construction."

The audit reads the parameters at three places, on real images (the four
references + the 75 EVAL queries = 79 images), and changes nothing:

  A1  the WRITER'S INPUTS: the two Detection fields
      ``Binder._run_co_located`` reads -- ``Detection.scale`` and
      ``Detection.z`` -- for the V1 detections handed to the binder;
  A2  the WRITER'S OUTPUT: every (scale_ratio, salience_ratio) pair the writer
      handed to the edge-params constructor, captured at the call (continuous
      value and bin), with the lens pair of each candidate;
  A3  the GRAPH: the parameters on the CO_LOCATED edges that reach the
      matcher (after the channel threshold and the per-edge mixture);
  A4  the MATCHER: what the baseline run credited to CO_LOCATED edges
      (verifiable, matched, similarity sum, exact-1.0 count, measured axes);
  A5  the REMOVAL ARM: every CO_LOCATED edge deleted from query and
      reference (the README's ``noedge_col``), paired against the baseline.

    python diagnostics/phase0_audit_A_colocated_hist.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from _bootstrap import RESULTS_DIR

UNKNOWN = -1.0


def _quantiles(xs: List[float], qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)):
    xs = sorted(xs)
    if not xs:
        return []
    return [xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] for q in qs]


def _hist_line(counter: Counter, keys) -> str:
    tot = sum(counter.values()) or 1
    return "  ".join("%s:%d (%.1f%%)" % (k, counter.get(k, 0),
                                         100.0 * counter.get(k, 0) / tot)
                     for k in keys)


def _entropy_bits(counter: Counter) -> float:
    tot = sum(counter.values())
    if not tot:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counter.values() if c)


def main() -> int:
    import eth80_benchmark as bm
    import eth80_frontend as fe
    from core.schema import edge_quantise
    from core.types import EdgeType
    from eth80_split import load_split
    from orchestration.binder import INFORMATIVE_LENS_TYPES

    split = load_split()
    entries = list(split["references"].items()) + [
        (c, p) for c in bm.CLASSES for p in split["eval"][c]]
    graphs = [(c, fe.graph_for(c, p)) for c, p in entries]
    out: List[str] = []
    P = out.append

    P("PHASE 0 / AUDIT A -- CO_LOCATED edge parameters on real images")
    P("images: %d (4 references + 75 EVAL queries); front end %s"
      % (len(graphs), fe.frontend_source_digest()[:12]))
    P("writer: orchestration/binder.py Binder._run_co_located")
    P("  scale_ratio    = None if either Detection.scale <= 0 else |log2(sa/sb)|")
    P("  salience_ratio = min(z1, z2) / max(z1, z2)")
    P("  slots: scale_ratio 8 bins over [0, 4] abs-log2; salience_ratio 8 bins"
      " over [0, 1]; bin -1 = UNKNOWN (no measurement)")
    P("")

    # ---------------- A1: writer inputs ----------------------------------
    scale_by_lens: Dict[str, Counter] = defaultdict(Counter)
    z_by_lens: Dict[str, List[float]] = defaultdict(list)
    for _c, g in graphs:
        for k, _alpha, z, scale in g.v1_dets:
            if k in INFORMATIVE_LENS_TYPES:
                scale_by_lens[k][scale] += 1
                z_by_lens[k].append(z)
    P("A1. WRITER INPUTS -- V1 detections of the informative lenses handed to"
      " the binder")
    P("  %-14s %8s  %-22s  %s" % ("lens", "dets", "Detection.scale values",
                                  "z: min / p25 / median / p75 / max"
                                  "   (distinct z)"))
    for k in sorted(z_by_lens):
        zs = z_by_lens[k]
        q = _quantiles(zs, (0.0, 0.25, 0.5, 0.75, 1.0))
        P("  %-14s %8d  %-22s  %.3f / %.3f / %.3f / %.3f / %.3f   (%d)" % (
            k, len(zs),
            ", ".join("%r x%d" % kv for kv in sorted(scale_by_lens[k].items())),
            q[0], q[1], q[2], q[3], q[4], len(set(round(z, 9) for z in zs))))
    n_scale_pos = sum(c for k in scale_by_lens for s, c in scale_by_lens[k].items()
                      if isinstance(s, (int, float)) and s > 0)
    P("  detections with Detection.scale > 0: %d of %d"
      % (n_scale_pos, sum(len(v) for v in z_by_lens.values())))
    P("")

    # ---------------- A2: writer output ----------------------------------
    # Re-derive each candidate from the detections with the writer's own two
    # formulas, to attach the LENS PAIR; then check the re-derivation against
    # what the writer actually handed to the constructor (exact multiset).
    raw_scale = Counter()
    raw_sal_bins = Counter()
    raw_sal_vals: List[float] = []
    joint = Counter()
    per_image_distinct: List[int] = []
    per_image_modal: List[float] = []
    by_pair: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    by_pair_vals: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    rederive_ok = 0
    for _c, g in graphs:
        img_joint = Counter()
        for row in g.co_located_raw:
            sb, lb = row["bins"]
            raw_scale["None" if row["scale_ratio"] is None else "measured"] += 1
            raw_sal_bins[int(lb)] += 1
            raw_sal_vals.append(float(row["salience_ratio"]))
            joint[(sb, lb)] += 1
            img_joint[(sb, lb)] += 1
        per_image_distinct.append(len(img_joint))
        per_image_modal.append(max(img_joint.values()) / sum(img_joint.values())
                               if img_joint else 0.0)
        by_anchor: Dict[Any, List[Tuple[str, float, Any]]] = defaultdict(list)
        for k, alpha, z, scale in g.v1_dets:
            if k in INFORMATIVE_LENS_TYPES:
                by_anchor[alpha].append((k, z, scale))
        rederived: List[float] = []
        for alpha, group in by_anchor.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    (k1, z1, _s1), (k2, z2, _s2) = group[i], group[j]
                    if k1 == k2:
                        continue
                    if math.sqrt(max(z1, 0.0) * max(z2, 0.0)) <= 0.0:
                        continue
                    sal = min(z1, z2) / max(max(z1, z2), 1e-6)
                    rederived.append(sal)
                    pair = tuple(sorted((k1, k2)))
                    b = edge_quantise("EDGE_CO_LOCATED",
                                      {"scale_ratio": None,
                                       "salience_ratio": sal})[1]
                    by_pair[pair][int(b)] += 1
                    by_pair_vals[pair].append(sal)
        rederive_ok += int(
            sorted(round(x, 12) for x in rederived)
            == sorted(round(float(r["salience_ratio"]), 12)
                      for r in g.co_located_raw))

    n_raw = sum(raw_scale.values())
    P("A2. WRITER OUTPUT -- every candidate the writer built (%d over %d images)"
      % (n_raw, len(graphs)))
    P("  scale_ratio   : " + _hist_line(raw_scale, ("None", "measured")))
    P("  salience_ratio bin histogram (bin k covers [k/8, (k+1)/8)):")
    P("     " + _hist_line(raw_sal_bins, range(8)))
    P("     entropy %.2f bits of %.2f possible; modal bin holds %.1f%%"
      % (_entropy_bits(raw_sal_bins), math.log2(8),
         100.0 * max(raw_sal_bins.values()) / n_raw))
    P("  salience_ratio continuous: min %.3f  p05 %.3f  p25 %.3f  median %.3f"
      "  p75 %.3f  p95 %.3f  max %.3f" % tuple(_quantiles(raw_sal_vals)))
    P("  distinct (scale_bin, salience_bin) tuples overall: %d -> %s"
      % (len(joint), ", ".join("%s x%d" % (k, v)
                               for k, v in sorted(joint.items()))))
    P("  per image: distinct tuples min %d / median %d / max %d;"
      " modal-tuple share min %.2f / median %.2f / max %.2f"
      % (min(per_image_distinct),
         sorted(per_image_distinct)[len(per_image_distinct) // 2],
         max(per_image_distinct), min(per_image_modal),
         sorted(per_image_modal)[len(per_image_modal) // 2],
         max(per_image_modal)))
    P("  re-derivation check (writer formula applied to the detections"
      " reproduces the captured values exactly): %d/%d images"
      % (rederive_ok, len(graphs)))
    P("")
    P("  salience bin by LENS PAIR (candidates):")
    P("  %-26s %7s  %s   %s" % ("lens pair", "n", " ".join(
        "b%d".rjust(5) % b for b in range(8)), "median"))
    for pair in sorted(by_pair, key=lambda p: -sum(by_pair[p].values())):
        h = by_pair[pair]
        n = sum(h.values())
        P("  %-26s %7d  %s   %.3f" % (
            " x ".join(pair), n,
            " ".join("%5d" % h.get(b, 0) for b in range(8)),
            _quantiles(by_pair_vals[pair], (0.5,))[0]))
    P("")

    # ---------------- A3: graph ------------------------------------------
    g_types = Counter()
    g_bins = Counter()
    g_protos = Counter()
    g_pairs: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    n_edges_all = 0
    for _c, g in graphs:
        sg = g.subgraph
        n_edges_all += len(sg.edges)
        for e in sg.edges:
            if e[2] != EdgeType.CO_LOCATED:
                continue
            ep = sg.edge_params.get(e)
            g_types[type(ep).__name__] += 1
            if ep is None:
                continue
            protos = getattr(ep, "protos", None) or [ep]
            g_protos[len(protos)] += 1
            g_bins[tuple(ep.quantized)] += 1
            pair = tuple(sorted((g.sigs[e[0]][0], g.sigs[e[1]][0])))
            g_pairs[pair][int(ep.quantized[1])] += 1
    n_co = sum(g_types.values())
    P("A3. GRAPH -- CO_LOCATED edges that reach the matcher: %d of %d edges"
      " (%.1f%%), %.0f per image" % (n_co, n_edges_all,
                                     100.0 * n_co / max(1, n_edges_all),
                                     n_co / len(graphs)))
    P("  parameter container: " + ", ".join("%s x%d" % kv
                                            for kv in sorted(g_types.items())))
    P("  mixture size (prototypes per edge): " + ", ".join(
        "%d: %d" % kv for kv in sorted(g_protos.items())))
    P("  (scale_bin, salience_bin) of the edge's founder prototype:")
    for k, v in sorted(g_bins.items()):
        P("     %-14s %6d  (%.1f%%)" % (k, v, 100.0 * v / n_co))
    n_scale_unknown = sum(v for k, v in g_bins.items() if k[0] == UNKNOWN)
    P("  scale_ratio slot UNKNOWN on %d/%d CO_LOCATED edges (%.1f%%)"
      % (n_scale_unknown, n_co, 100.0 * n_scale_unknown / max(1, n_co)))
    P("")

    # ---------------- A4: matcher ----------------------------------------
    P("A4. MATCHER -- what the baseline credited to CO_LOCATED core edges")
    for window in bm.WINDOWS:
        res = bm.run_cached("eval", window, "baseline", breakdown=True)
        for which in ("true", "all"):
            agg = res["agg"]["edge_" + which]
            co = agg.get("CO_LOCATED")
            if not co:
                continue
            tot_sim = sum(r["sim_sum"] for r in agg.values())
            tot_ver = sum(r["verifiable"] for r in agg.values())
            P("  [%s window, %s] core %d  verifiable %d  matched %d  sim_sum %.1f"
              "  exact-1.0 %d (%.1f%% of matched)  measured axes/matched %.2f"
              % (window, "own-class refs" if which == "true" else "all refs",
                 co["n"], co["verifiable"], co["matched"], co["sim_sum"],
                 co["sim_eq_1"],
                 100.0 * co["sim_eq_1"] / max(1, co["matched"]),
                 co["axes_sum"] / max(1, co["matched"])))
            P("        share of all matched edge similarity: %.1f%%;"
              " share of verifiable edges: %.1f%%"
              % (100.0 * co["sim_sum"] / max(1e-9, tot_sim),
                 100.0 * co["verifiable"] / max(1, tot_ver)))
    P("")

    # ---------------- A5: removal arm -------------------------------------
    import _stats as st

    P("A5. REMOVAL ARM -- CO_LOCATED edges deleted from query and reference"
      " (EVAL-75, in-sample)")
    for window in bm.WINDOWS:
        base = bm.run_cached("eval", window, "baseline", breakdown=True)
        rem = bm.run_cached("eval", window, "noedge_col", breakdown=True)
        g, l, pv = st.paired(base, rem)
        P("  [%s] " % window + bm.fmt_row("baseline", bm.summarize(base)))
        P("  [%s] " % window + bm.fmt_row("noedge_col", bm.summarize(rem))
          + "   " + st.fmt_paired(g, l, pv))
        changed = sum(int(qa["pred"] != qb["pred"]) for qa, qb in
                      zip(base["queries"], rem["queries"]))
        P("        predictions changed: %d of %d" % (changed,
                                                      len(base["queries"])))
    P("")

    text = "\n".join(out)
    print(text)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase0_audit_A.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
