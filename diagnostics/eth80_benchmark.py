"""ETH-80 4-class benchmark on the repository's own matcher.

One reference image per class; each query is scored against the four
references with ``memory.cores.evidence_score`` and takes the argmax.

    python diagnostics/eth80_benchmark.py                       # baseline, EVAL-75, both windows
    python diagnostics/eth80_benchmark.py --arm noedge_col      # an ablation arm
    python diagnostics/eth80_benchmark.py --set MATCH_GLOBAL_CONSISTENCY_ENABLED=True --tag d2

THIS IS A RECONSTRUCTION.  The handoff's harness (``_u32_eth80_full_benchmark``)
and its split were not in the export.  Numbers produced here are comparable
with each other and NOT with the handoff's 51/75.  See diagnostics/README.md.

Windows (what is handed to the matcher):
  full  the pipeline's situation window for the image, exactly as
        ``StatePatternIndex.record`` builds it (every lens tier).  PRIMARY:
        nothing is selected by hand.
  v1    the same window restricted to the repository's own V1 tier
        (``memory.cores.lens_tier(lens) == 0``) and the edges among those
        nodes.  This is the closest available analogue of the lens census the
        handoff reports for its headline harness (Texture / Boundary /
        Position / Color / Onset), kept as a second view.

Arms are QUERY/REFERENCE GRAPH TRANSFORMS applied outside the source tree --
they are instruments, they change no behaviour of the system.

Every split in this benchmark is IN-SAMPLE for anything tuned on it: EVAL-75
is the set every keep/revert decision is taken on.  HELDOUT-75 exists for the
final report only (``--split heldout``).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import ast
import dataclasses
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from _bootstrap import RESULTS_DIR, shims_in_use

CLASSES: Tuple[str, ...] = ("apple", "car", "cup", "horse")
WINDOWS: Tuple[str, ...] = ("full", "v1")

Graph = Tuple[Any, Dict[int, Tuple[str, Any]]]     # (Subgraph, sigs)


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------

def _restrict(sg, sigs, keep) -> Graph:
    from core.graph_types import Subgraph

    keep = frozenset(keep)
    edges = frozenset(e for e in sg.edges if e[0] in keep and e[1] in keep)
    return (
        Subgraph(
            nodes=keep,
            edges=edges,
            edge_params={k: v for k, v in sg.edge_params.items()
                         if k[0] in keep and k[1] in keep},
            weights={k: v for k, v in sg.weights.items()
                     if k[0] in keep and k[1] in keep},
            node_weights={n: w for n, w in sg.node_weights.items()
                          if n in keep},
            anchor_of_node={n: p for n, p in sg.anchor_of_node.items()
                            if n in keep},
        ),
        {n: s for n, s in sigs.items() if n in keep},
    )


def make_window(g, kind: str) -> Graph:
    """(Subgraph, sigs) for one ImageGraph under the named window."""
    if kind == "full":
        return g.subgraph, dict(g.sigs)
    if kind == "v1":
        from memory.cores import lens_tier

        keep = {n for n, (lt, _s) in g.sigs.items() if lens_tier(lt) == 0}
        return _restrict(g.subgraph, g.sigs, keep)
    raise ValueError("unknown window %r" % (kind,))


# --------------------------------------------------------------------------
# arms: graph transforms (instruments)
# --------------------------------------------------------------------------

def _replace(sg, **kw):
    from core.graph_types import Subgraph

    base = dict(nodes=sg.nodes, edges=sg.edges,
                edge_params=dict(sg.edge_params), weights=dict(sg.weights),
                node_weights=dict(sg.node_weights),
                anchor_of_node=dict(sg.anchor_of_node))
    base.update(kw)
    return Subgraph(**base)


def drop_edge_types(graph: Graph, names: Sequence[str]) -> Graph:
    """Remove every edge of the named EdgeTypes (the handoff's `noedge_*`)."""
    sg, sigs = graph
    names = set(names)

    def _keep(e):
        return getattr(e[2], "name", str(e[2])) not in names

    edges = frozenset(e for e in sg.edges if _keep(e))
    return (_replace(
        sg, edges=edges,
        edge_params={k: v for k, v in sg.edge_params.items() if _keep(k)},
        weights={k: v for k, v in sg.weights.items() if _keep(k)}), sigs)


def drop_anchors(graph: Graph) -> Graph:
    """Remove the occupancy map: the matcher's position term becomes 1.0."""
    sg, sigs = graph
    return _replace(sg, anchor_of_node={}), sigs


def shuffle_anchors(graph: Graph, seed: int, level: str) -> Graph:
    """Permute the positions in ``anchor_of_node``.  Nodes, signatures, edges
    and edge parameters are untouched: only WHERE each node is said to be.

    level='anchor'  one permutation over the DISTINCT positions; every node
                    that sat at position p moves to pi(p).  Nodes that shared
                    a patch still share a patch.
    level='node'    the positions are permuted over the nodes individually;
                    nodes that shared a patch are scattered.
    """
    sg, sigs = graph
    rng = random.Random(seed)
    amap = sg.anchor_of_node
    nodes = sorted(amap)
    if level == "anchor":
        distinct = sorted(set(amap.values()))
        perm = list(distinct)
        rng.shuffle(perm)
        mapping = dict(zip(distinct, perm))
        new = {n: mapping[amap[n]] for n in nodes}
    elif level == "node":
        vals = [amap[n] for n in nodes]
        rng.shuffle(vals)
        new = dict(zip(nodes, vals))
    else:
        raise ValueError(level)
    return _replace(sg, anchor_of_node=new), sigs


def strip_populations(graph: Graph) -> Graph:
    """Remove every graded population vector (nodes and edges); bins stay."""
    sg, sigs = graph

    def _strip(obj):
        if obj is None:
            return None
        protos = getattr(obj, "protos", None)
        if protos is not None:                       # EdgeParamsMixture
            return type(obj)([_strip(p) for p in protos], list(obj.mass),
                             obj.t_last)
        if getattr(obj, "population", None):
            return dataclasses.replace(obj, population=())
        return obj

    return (_replace(
        sg, edge_params={k: _strip(v) for k, v in sg.edge_params.items()}),
        {n: (lt, _strip(s)) for n, (lt, s) in sigs.items()})


def _arm_table() -> Dict[str, Tuple[Callable[[Graph], Graph],
                                    Callable[[Graph], Graph], str]]:
    """name -> (query transform, reference transform, description)."""
    ident = lambda g: g  # noqa: E731
    arms = {
        "baseline": (ident, ident, "no transform"),
        "noedge_col": (
            lambda g: drop_edge_types(g, ["CO_LOCATED"]),
            lambda g: drop_edge_types(g, ["CO_LOCATED"]),
            "CO_LOCATED edges removed from query and reference"),
        "no_anchor": (drop_anchors, ident,
                      "query occupancy map removed (position term = 1.0)"),
        "no_anchor_both": (drop_anchors, drop_anchors,
                           "occupancy map removed on both sides"),
        "nopop": (strip_populations, strip_populations,
                  "population vectors removed from every node and edge"),
    }
    return arms


def resolve_arm(name: str):
    arms = _arm_table()
    if name in arms:
        return arms[name]
    # shuffle_<level>_<seed>
    if name.startswith("shuffle_"):
        _s, level, seed = name.split("_")
        return (lambda g: shuffle_anchors(g, int(seed), level),
                lambda g: g,
                "query anchor positions permuted (%s level, seed %s)"
                % (level, seed))
    raise ValueError("unknown arm %r (known: %s, shuffle_<anchor|node>_<seed>)"
                     % (name, ", ".join(sorted(arms))))


# --------------------------------------------------------------------------
# count-only statistics (audit B)
# --------------------------------------------------------------------------

def lens_counts(sigs: Dict[int, Tuple[str, Any]]) -> Dict[str, int]:
    return dict(Counter(lt for lt, _s in sigs.values()))


def count_only_coverage(q_counts: Dict[str, int],
                        c_counts: Dict[str, int]) -> float:
    """sum_lens boost * min(n_q, n_c) / sum_lens boost * n_c.

    Uses NOTHING but the per-lens-type node counts of the two graphs.  It is
    what ``node_coverage`` evaluates to if every pair the per-lens assignment
    keeps is credited in full.
    """
    from matching.structural_similarity import GESTALT_LENS_BOOST

    num = den = 0.0
    for lt, n_c in c_counts.items():
        b = GESTALT_LENS_BOOST.get(lt, 1.0)
        den += b * n_c
        num += b * min(n_c, q_counts.get(lt, 0))
    return num / den if den > 0 else 0.0


def count_only_balanced(q_counts: Dict[str, int],
                        c_counts: Dict[str, int]) -> float:
    """2 * sum_lens boost * min(n_q, n_c) / (sum boost * n_q + sum boost * n_c).

    The two-sided companion of ``count_only_coverage``: what the BALANCED
    node term ([D3] "node") evaluates to if every pair the per-lens
    assignment keeps is credited in full.  Counts only.
    """
    from matching.structural_similarity import GESTALT_LENS_BOOST

    num = den = 0.0
    for lt in set(q_counts) | set(c_counts):
        b = GESTALT_LENS_BOOST.get(lt, 1.0)
        n_q, n_c = q_counts.get(lt, 0), c_counts.get(lt, 0)
        den += b * (n_q + n_c)
        num += 2.0 * b * min(n_q, n_c)
    return num / den if den > 0 else 0.0


# --------------------------------------------------------------------------
# per-edge-type evidence breakdown (instrument, validated against the trace)
# --------------------------------------------------------------------------

def edge_evidence_breakdown(Q, C, corr: Dict[int, int]) -> Dict[str, Dict[str, float]]:
    """Per core-edge type: n, verifiable, sum of best_sim, matched count,
    exact-1.0 count, and measured-axes sum.

    Re-walks the edge loop of ``memory.cores._mcs_accumulate`` for the
    correspondence it returned, calling the same similarity functions.  The
    caller asserts the totals against the matcher's own trace
    (``verifiable_core_edges_count``, ``matched_rel_sim_sum``); a mismatch
    means this instrument has drifted from the source and must not be read.
    """
    from core.schema import edge_measured_axes, polymorphic_edge_similarity
    from memory.cores import _reverse_for_traversal
    from orchestration.binder import UNDIRECTED_EDGE_TYPES

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
    out: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"n": 0, "verifiable": 0, "sim_sum": 0.0, "matched": 0,
                 "sim_eq_1": 0, "axes_sum": 0, "no_query_edge": 0,
                 "c_params_none": 0})
    for cs, cd, cet in C.edges:
        row = out[getattr(cet, "name", str(cet))]
        row["n"] += 1
        c_key = _canon((cs, cd, cet))
        cp = c_params.get(c_key) or c_params.get((cs, cd, cet))
        if cp is None:
            row["c_params_none"] += 1
        if cs not in inv or cd not in inv:
            continue
        row["verifiable"] += 1
        qs, qd = inv[cs], inv[cd]
        cand = list(q_index.get((qs, qd), []))
        if cet in UNDIRECTED_EDGE_TYPES and qs != qd:
            seen = {qk for _a, _b, qk in cand}
            for qet_r, qp_r, qk_r in q_index.get((qd, qs), []):
                if qk_r not in seen:
                    cand.append((qet_r, qp_r, qk_r))
        if not cand:
            row["no_query_edge"] += 1
        best, best_qp = 0.0, None
        for qet, qp, _qk in cand:
            sim = polymorphic_edge_similarity(
                qp, cp, use_confidence=False, edge_types=(qet, cet))
            if sim > best:
                best, best_qp = sim, qp
        row["sim_sum"] += best
        if best > 0.0:
            row["matched"] += 1
            row["axes_sum"] += edge_measured_axes(best_qp, cp)
            if best == 1.0:
                row["sim_eq_1"] += 1
    return {k: dict(v) for k, v in out.items()}


def node_credit_breakdown(q_sigs, c_sigs, corr: Dict[int, int]) -> Dict[str, Dict[str, float]]:
    """Per lens type: core / query node counts, credited pairs, and the
    SIGNATURE similarity of the credited pairs (sum, and how many fall under
    0.5 / equal 1.0).  ``memory.cores._sig_similarity`` is the function the
    assignment itself weighs pairs with.
    """
    from memory.cores import _sig_similarity

    out: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"core": 0, "query": 0, "min_qc": 0, "credited": 0,
                 "sim_sum": 0.0, "sim_lt_05": 0, "sim_eq_1": 0})
    for lt, _s in c_sigs.values():
        out[lt]["core"] += 1
    for lt, _s in q_sigs.values():
        out[lt]["query"] += 1
    for row in out.values():
        row["min_qc"] = min(row["core"], row["query"])
    for q_id, c_id in corr.items():
        lt = c_sigs[c_id][0]
        sim = float(_sig_similarity(q_sigs[q_id][1], c_sigs[c_id][1]))
        row = out[lt]
        row["credited"] += 1
        row["sim_sum"] += sim
        row["sim_lt_05"] += int(sim < 0.5)
        row["sim_eq_1"] += int(sim == 1.0)
    return {k: dict(v) for k, v in out.items()}


_SOURCE_DIGEST: Optional[str] = None


def source_digest() -> str:
    """sha256 over every .py of the source tree (diagnostics/, tests and data
    excluded).  A saved result is only ever re-used under the same digest."""
    global _SOURCE_DIGEST
    if _SOURCE_DIGEST is None:
        import hashlib

        from _bootstrap import ROOT

        h = hashlib.sha256()
        paths = []
        for base, dirs, files in os.walk(ROOT):
            dirs[:] = sorted(d for d in dirs if d not in (
                "diagnostics", "tests_figground", "tests_matcher", "data",
                ".git", "__pycache__", ".pytest_cache"))
            paths.extend(os.path.join(base, f) for f in files
                         if f.endswith(".py"))
        for path in sorted(paths):
            h.update(os.path.relpath(path, ROOT).replace(os.sep, "/").encode())
            with open(path, "rb") as fh:
                h.update(hashlib.sha256(fh.read()).digest())
        _SOURCE_DIGEST = h.hexdigest()
    return _SOURCE_DIGEST


def environment() -> Dict[str, str]:
    """Interpreter and library versions, and the hash seed, of this run.

    The repository pins none of them and the sandbox is rebuilt between
    sessions, so every stored result says what it was computed under
    (``phase2_continuity.py`` re-runs a stored configuration and compares).
    """
    out = {"python": sys.version.split()[0],
           "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "(unset)")}
    for mod in ("numpy", "scipy", "PIL"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                               # pragma: no cover
            out[mod] = "(not importable)"
    return out


def harness_digest() -> str:
    """sha256 of this file and the front-end driver: a saved result is not
    re-used after the instrument that produced it has been edited."""
    import hashlib

    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("eth80_benchmark.py", "eth80_frontend.py", "eth80_split.json"):
        with open(os.path.join(here, name), "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

_TRACE_KEYS = ("node_coverage", "relational_consistency",
               "relational_multiplier", "edge_absence", "total_absence",
               "core_coverage", "n_core_edges", "verifiable_core_edges_count",
               "matched_rel_sim_sum", "c_node_total", "c_matched_nm",
               "q_node_total", "q_matched_nm", "q_edge_total", "c_edge_total",
               "q_matched_em", "c_matched_em",
               # written by the matcher from D1 on (absent before it)
               "node_coverage_binary", "q_matched_nm_binary",
               "c_matched_nm_binary",
               # written by the matcher from D3 on; the three edge keys only
               # when a query-side edge term is switched on
               "core_node_coverage", "query_node_coverage",
               "n_query_edges", "q_verifiable_edges_count",
               "q_matched_rel_sim_sum",
               # written by the matcher from the null calibration on; the
               # chance keys only when it is switched on
               "node_coverage_raw", "node_coverage_chance",
               "q_chance_nm", "c_chance_nm",
               # written by the matcher from P4b on: the two factors of the
               # relational term and the counts behind them (the q_ keys
               # only when a query-side edge term is switched on)
               "relation_presence", "relation_physics",
               "rel_present", "rel_physical", "rel_physical_sim_sum",
               "q_rel_present", "q_rel_physical", "q_rel_physical_sim_sum")


def score_pair(Q, q_sigs, C, c_sigs, breakdown: bool = False) -> Dict[str, Any]:
    from memory.cores import evidence_score

    trace: Dict[str, Any] = {}
    score, corr = evidence_score(Q, C, q_sigs, c_sigs, trace=trace)
    row: Dict[str, Any] = {"score": float(score), "n_corr": len(corr)}
    for k in _TRACE_KEYS:
        if k in trace:
            row[k] = float(trace[k])
    axes = trace.get("coverage_axes") or []
    row["n_matched_edges"] = len(axes)
    row["coverage_axes_mean"] = (sum(axes) / len(axes)) if axes else 0.0
    row["count_only"] = count_only_coverage(lens_counts(q_sigs),
                                            lens_counts(c_sigs))
    row["count_only_balanced"] = count_only_balanced(lens_counts(q_sigs),
                                                     lens_counts(c_sigs))
    if breakdown:
        bd = edge_evidence_breakdown(Q, C, corr)
        n_ver = sum(v["verifiable"] for v in bd.values())
        s_sum = math.fsum(v["sim_sum"] for v in bd.values())
        if (n_ver != int(row.get("verifiable_core_edges_count", -1))
                or abs(s_sum - row.get("matched_rel_sim_sum", -1.0)) > 1e-6):
            raise AssertionError(
                "edge_evidence_breakdown disagrees with the matcher trace: "
                "verifiable %d vs %s, sim_sum %.9f vs %s" % (
                    n_ver, row.get("verifiable_core_edges_count"), s_sum,
                    row.get("matched_rel_sim_sum")))
        row["edge_breakdown"] = bd
        nb = node_credit_breakdown(q_sigs, c_sigs, corr)
        row["node_breakdown"] = nb
        # OBSERVABLE ONLY: what the node term would read if each credited
        # pair counted by its signature similarity instead of 1.0.
        from matching.structural_similarity import GESTALT_LENS_BOOST
        den = sum(GESTALT_LENS_BOOST.get(lt, 1.0) * r["core"]
                  for lt, r in nb.items())
        row["node_sim_coverage"] = (sum(
            GESTALT_LENS_BOOST.get(lt, 1.0) * r["sim_sum"]
            for lt, r in nb.items()) / den) if den > 0 else 0.0
    return row


_STATE: Dict[str, Any] = {}


def _score_query(i: int) -> Dict[str, Any]:
    st = _STATE
    cls, name, (Q, q_sigs) = st["queries"][i]
    rows = {}
    for rc in CLASSES:
        C, c_sigs = st["refs"][rc]
        rows[rc] = score_pair(Q, q_sigs, C, c_sigs, breakdown=st["breakdown"])
    scores = {rc: rows[rc]["score"] for rc in CLASSES}
    return {"name": name, "true": cls, "pairs": rows,
            "pred": argmax(scores)[0], "tie": argmax(scores)[1]}


def argmax(scores: Dict[str, float]) -> Tuple[str, bool]:
    """Winner (first in CLASSES order on an exact tie) and whether it tied."""
    best = max(scores[c] for c in CLASSES)
    winners = [c for c in CLASSES if scores[c] == best]
    return winners[0], len(winners) > 1


def apply_param_overrides(pairs: Sequence[str]) -> Dict[str, Any]:
    """``NAME=VALUE`` -> setattr(parameters, NAME, literal(VALUE))."""
    import parameters as P

    applied: Dict[str, Any] = {}
    for item in pairs or ():
        name, _eq, raw = item.partition("=")
        if not _eq:
            raise SystemExit("--set expects NAME=VALUE, got %r" % item)
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            val = raw
        if not hasattr(P, name):
            print("note: parameters.%s did not exist before this override"
                  % name, file=sys.stderr)
        setattr(P, name, val)
        applied[name] = val
    return applied


def run(split_name: str = "eval", window: str = "full", arm: str = "baseline",
        workers: int = 2, breakdown: bool = False,
        verbose: bool = True) -> Dict[str, Any]:
    """Score every query of the split against the four references."""
    import multiprocessing as mp

    import eth80_frontend as fe
    from eth80_split import load_split

    split = load_split()
    q_tf, c_tf, arm_desc = resolve_arm(arm)
    entries = [(c, p) for c in CLASSES for p in split[split_name][c]]
    fe.warm_cache(list(split["references"].items()) + entries,
                  workers=workers, verbose=verbose)
    refs = {c: c_tf(make_window(fe.graph_for(c, split["references"][c]),
                                window))
            for c in CLASSES}
    queries = [(c, os.path.basename(p),
                q_tf(make_window(fe.graph_for(c, p), window)))
               for c, p in entries]
    _STATE.update(refs=refs, queries=queries, breakdown=breakdown)
    t0 = time.time()
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            rows = pool.map(_score_query, range(len(queries)), chunksize=2)
    else:
        rows = [_score_query(i) for i in range(len(queries))]
    res = {
        "split": split_name, "window": window, "arm": arm,
        "arm_description": arm_desc,
        "frontend_digest": fe.frontend_source_digest(),
        "source_digest": source_digest(),
        "harness_digest": harness_digest(),
        "export_shims_in_use": shims_in_use(),
        "environment": environment(),
        "seconds": time.time() - t0,
        "queries": rows,
    }
    if breakdown:
        res["agg"] = {
            "%s_%s" % (kind, which): aggregate_breakdown(
                res, kind + "_breakdown", which)
            for kind in ("edge", "node") for which in ("true", "all")}
        res["agg_pairs"] = {"true": len(rows), "all": len(rows) * len(CLASSES)}
        for q in rows:                      # per-pair tables are not kept
            for rc in CLASSES:
                q["pairs"][rc].pop("edge_breakdown", None)
                q["pairs"][rc].pop("node_breakdown", None)
    return res


def aggregate_breakdown(res: Dict[str, Any], key: str,
                        which: str) -> Dict[str, Dict[str, float]]:
    """Sum a per-pair breakdown over (query, reference) pairs.

    which='true'  only the pair with the query's own-class reference
    which='all'   all four references
    """
    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for q in res["queries"]:
        for rc in CLASSES:
            if which == "true" and rc != q["true"]:
                continue
            for name, row in q["pairs"][rc][key].items():
                for k, v in row.items():
                    agg[name][k] += v
    return {k: dict(v) for k, v in agg.items()}


def run_cached(split_name: str = "eval", window: str = "full",
               arm: str = "baseline", workers: int = 2,
               breakdown: bool = False, tag: Optional[str] = None,
               overrides: Optional[Dict[str, Any]] = None,
               verbose: bool = True) -> Dict[str, Any]:
    """``run`` with re-use of a saved result made under the SAME source
    digest, front-end digest and parameter overrides."""
    import eth80_frontend as fe

    overrides = overrides or {}
    stem = tag or result_stem(split_name, arm, overrides)
    path = os.path.join(RESULTS_DIR, "%s_%s.json" % (stem, window))
    want = {k: repr(v) for k, v in overrides.items()}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            old = json.load(fh)
        if (old.get("source_digest") == source_digest()
                and old.get("harness_digest") == harness_digest()
                and old.get("frontend_digest") == fe.frontend_source_digest()
                and old.get("param_overrides", {}) == want
                and old.get("arm") == arm and old.get("split") == split_name
                and (not breakdown or "agg" in old)):
            return old
    res = run(split_name, window, arm, workers=workers, breakdown=breakdown,
              verbose=verbose)
    res["param_overrides"] = want
    s = summarize(res)
    res["summary"] = {k: v for k, v in s.items() if k != "preds"}
    saved = save(res, "%s_%s" % (stem, window))
    # ALWAYS hand back the STORED form.  A fresh result holds full-precision
    # floats and a cached one holds the 10 significant digits of the file;
    # returning `res` here made an exact comparison between two
    # configurations depend on which of them happened to be cached (it
    # reported "247 of 300 scores moved" for a change that moves 19).
    with open(saved, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_config(overrides: Optional[Dict[str, Any]] = None,
               split_name: str = "eval", window: str = "full",
               arm: str = "baseline", breakdown: bool = True,
               tag: Optional[str] = None, workers: int = 2,
               verbose: bool = False) -> Dict[str, Any]:
    """``run_cached`` under temporary parameter overrides.

    Sets ``parameters.NAME = value`` for every override, runs (the worker
    pool is forked inside ``run``, so the children inherit the values) and
    restores the previous values, so several configurations can be measured
    in one process.  A name that does not exist yet is an error: a typo must
    not silently measure the default.
    """
    import parameters as P

    overrides = dict(overrides or {})
    missing = [k for k in overrides if not hasattr(P, k)]
    if missing:
        raise KeyError("unknown parameters: %s" % ", ".join(sorted(missing)))
    old = {k: getattr(P, k) for k in overrides}
    for k, v in overrides.items():
        setattr(P, k, v)
    try:
        return run_cached(split_name, window, arm, workers=workers,
                          breakdown=breakdown, tag=tag, overrides=overrides,
                          verbose=verbose)
    finally:
        for k, v in old.items():
            setattr(P, k, v)


def result_stem(split_name: str, arm: str, overrides: Dict[str, Any]) -> str:
    return "_".join([split_name, arm]
                    + ["%s-%s" % kv for kv in sorted(overrides.items())])


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def summarize(res: Dict[str, Any], score_key: Optional[str] = None) -> Dict[str, Any]:
    """Accuracy, per-class hits, confusion, ties, mean margin.

    ``score_key`` re-decides every query by another per-pair quantity
    (e.g. 'node_coverage', 'count_only') instead of the evidence score.
    """
    per = {c: [0, 0] for c in CLASSES}
    conf = {c: Counter() for c in CLASSES}
    ties = 0
    margins: List[float] = []
    preds: List[str] = []
    for q in res["queries"]:
        key = score_key or "score"
        scores = {c: q["pairs"][c][key] for c in CLASSES}
        pred, tie = argmax(scores)
        preds.append(pred)
        ties += int(tie)
        per[q["true"]][1] += 1
        per[q["true"]][0] += int(pred == q["true"])
        conf[q["true"]][pred] += 1
        margins.append(scores[q["true"]]
                       - max(scores[c] for c in CLASSES if c != q["true"]))
    n = sum(v[1] for v in per.values())
    hits = sum(v[0] for v in per.values())
    return {"hits": hits, "n": n, "acc": hits / n if n else 0.0,
            "per_class": {c: tuple(per[c]) for c in CLASSES},
            "confusion": {c: dict(conf[c]) for c in CLASSES},
            "ties": ties,
            "mean_margin": sum(margins) / len(margins) if margins else 0.0,
            "preds": preds}


def fmt_row(label: str, s: Dict[str, Any]) -> str:
    pc = "  ".join("%s %2d/%-2d" % (c, s["per_class"][c][0], s["per_class"][c][1])
                   for c in CLASSES)
    return "%-34s %2d/%-2d = %5.1f%%   %s   ties %d   margin %+.4f" % (
        label, s["hits"], s["n"], 100.0 * s["acc"], pc, s["ties"],
        s["mean_margin"])


def fmt_confusion(s: Dict[str, Any]) -> str:
    lines = ["      true\\pred   " + "".join("%7s" % c for c in CLASSES)]
    for c in CLASSES:
        lines.append("      %-10s   " % c + "".join(
            "%7d" % s["confusion"][c].get(p, 0) for p in CLASSES))
    return "\n".join(lines)


def edge_breakdown_table(res: Dict[str, Any], which: str = "true") -> str:
    """Per core-edge type evidence, summed over (query, reference) pairs.

    core      core edges of the type
    verif     both endpoints mapped by the node correspondence
    no-q-edge verifiable, but the query has NO edge between the two mapped
              nodes (so nothing can be compared)
    sim=0     verifiable, a query edge exists there, similarity is 0
    matched   best similarity > 0
    sim_sum   sum of best similarity (the numerator of relational_consistency)
    sim==1    matched with similarity exactly 1.0
    axes/m    mean number of axes measured on both sides, per matched edge
    noparam   core edges that carry no EdgeParams at all
    """
    agg = (res.get("agg") or {}).get("edge_" + which)
    if agg is None:
        return "(run with --breakdown)"
    tot_sim = sum(r["sim_sum"] for r in agg.values()) or 1.0
    lines = ["  per core-edge type, summed over %d (query, %s reference) pairs"
             % (res["agg_pairs"][which],
                "own-class" if which == "true" else "every"),
             "  %-26s %7s %7s %9s %6s %7s %8s %6s %6s %6s %7s" % (
                 "edge type", "core", "verif", "no-q-edge", "sim=0",
                 "matched", "sim_sum", "share", "sim==1", "axes/m",
                 "noparam")]
    keys = ("n", "verifiable", "no_query_edge", "matched", "sim_sum",
            "sim_eq_1", "c_params_none")
    for et, r in sorted(agg.items(), key=lambda kv: (-kv[1]["sim_sum"],
                                                     -kv[1]["n"], kv[0])):
        m = r["matched"]
        lines.append(
            "  %-26s %7d %7d %9d %6d %7d %8.1f %5.1f%% %6d %6.2f %7d" % (
                et, r["n"], r["verifiable"], r["no_query_edge"],
                r["verifiable"] - r["no_query_edge"] - m, m, r["sim_sum"],
                100.0 * r["sim_sum"] / tot_sim, r["sim_eq_1"],
                (r["axes_sum"] / m) if m else 0.0, r["c_params_none"]))
    t = {k: sum(r[k] for r in agg.values()) for k in keys}
    lines.append(
        "  %-26s %7d %7d %9d %6d %7d %8.1f %5.1f%% %6d %6s %7d" % (
            "TOTAL", t["n"], t["verifiable"], t["no_query_edge"],
            t["verifiable"] - t["no_query_edge"] - t["matched"], t["matched"],
            t["sim_sum"], 100.0, t["sim_eq_1"], "", t["c_params_none"]))
    return "\n".join(lines)


def node_breakdown_table(res: Dict[str, Any], which: str = "true") -> str:
    """Per lens type node credit, summed over (query, reference) pairs.

    min(q,c)  sum over pairs of min(#query nodes, #core nodes) of the lens:
              the most pairs the per-lens assignment can form
    credited  pairs the assignment kept (each is credited boost x 1.0)
    mean sim  mean SIGNATURE similarity of the credited pairs
    """
    from matching.structural_similarity import GESTALT_LENS_BOOST

    agg = (res.get("agg") or {}).get("node_" + which)
    if agg is None:
        return "(run with --breakdown)"
    lines = ["  per lens type, summed over %d (query, %s reference) pairs"
             % (res["agg_pairs"][which],
                "own-class" if which == "true" else "every"),
             "  %-18s %5s %7s %7s %8s %8s %9s %8s %7s %7s" % (
                 "lens", "boost", "core", "query", "min(q,c)", "credited",
                 "cred/min", "mean sim", "sim<0.5", "sim==1")]
    tot = defaultdict(float)
    keys = ("core", "query", "min_qc", "credited", "sim_sum", "sim_lt_05",
            "sim_eq_1")
    for lt, r in sorted(agg.items(), key=lambda kv: (-kv[1]["core"], kv[0])):
        c = r["credited"]
        lines.append("  %-18s %5.1f %7d %7d %8d %8d %8.1f%% %8.3f %7d %7d" % (
            lt, GESTALT_LENS_BOOST.get(lt, 1.0), r["core"], r["query"],
            r["min_qc"], c, 100.0 * c / r["min_qc"] if r["min_qc"] else 0.0,
            (r["sim_sum"] / c) if c else 0.0, r["sim_lt_05"], r["sim_eq_1"]))
        for k in keys:
            tot[k] += r[k]
    c = tot["credited"]
    lines.append("  %-18s %5s %7d %7d %8d %8d %8.1f%% %8.3f %7d %7d" % (
        "TOTAL", "", tot["core"], tot["query"], tot["min_qc"], c,
        100.0 * c / tot["min_qc"] if tot["min_qc"] else 0.0,
        (tot["sim_sum"] / c) if c else 0.0, tot["sim_lt_05"],
        tot["sim_eq_1"]))
    return "\n".join(lines)


#: per-pair fields kept for instrument arms (the baseline and every
#: configuration run keep the full record)
_SLIM_KEYS = ("score", "node_coverage", "relational_consistency",
              "edge_absence", "count_only", "node_sim_coverage",
              "node_coverage_binary", "count_only_balanced",
              "core_node_coverage", "query_node_coverage",
              "node_coverage_raw", "node_coverage_chance")


def save(res: Dict[str, Any], tag: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, tag + ".json")
    if res.get("arm") != "baseline":
        res = dict(res)
        res["queries"] = [
            dict(q, pairs={rc: {k: v for k, v in pr.items()
                                if k in _SLIM_KEYS}
                           for rc, pr in q["pairs"].items()})
            for q in res["queries"]]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_round_floats(res), fh, sort_keys=True,
                  separators=(",", ":"))
        fh.write("\n")
    return path


def _round_floats(obj):
    """Floats to 10 significant digits: enough to reproduce every table and
    every argmax, and it keeps the committed result files small."""
    if isinstance(obj, float):
        return float("%.10g" % obj)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="eval", choices=("eval", "heldout"))
    ap.add_argument("--window", default="both",
                    choices=WINDOWS + ("both",))
    ap.add_argument("--arm", default="baseline")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a `parameters` attribute for this run")
    ap.add_argument("--tag", default=None,
                    help="result file stem (default: derived from the options)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--breakdown", action="store_true",
                    help="add the per-edge-type evidence breakdown")
    ap.add_argument("--confusion", action="store_true")
    a = ap.parse_args(argv)

    overrides = apply_param_overrides(a.set)
    windows = WINDOWS if a.window == "both" else (a.window,)
    stem = a.tag or result_stem(a.split, a.arm, overrides)
    print("ETH-80 reconstructed benchmark | split=%s arm=%s overrides=%s"
          % (a.split, a.arm, overrides or "{}"))
    print("source %s  front end %s" % (source_digest()[:12],
                                        __import__("eth80_frontend")
                                        .frontend_source_digest()[:12]))
    if a.split == "eval":
        print("IN-SAMPLE: every decision of this work is taken on EVAL-75.")
    for w in windows:
        res = run_cached(a.split, w, a.arm, workers=a.workers,
                         breakdown=a.breakdown, tag=stem, overrides=overrides)
        s_ = summarize(res)
        print(fmt_row("%s [%s]" % (a.arm, w), s_))
        if a.confusion:
            print(fmt_confusion(s_))
        if a.breakdown:
            print(edge_breakdown_table(res, "true"))
            print(node_breakdown_table(res, "true"))
        print("   -> %s  (%.0fs)" % (os.path.relpath(
            os.path.join(RESULTS_DIR, "%s_%s.json" % (stem, w))),
            res["seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
