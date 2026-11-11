"""BUG HUNT -- self-checks of the stage-1/2 matcher changes, measurement first.

Every check below is an identity or an invariant that must hold if the code
does what its comments, its pins and REPORT.md say.  A check that fails is a
finding; a check that passes is a bound on where a bug can still be.  Nothing
is tuned and nothing here selects between configurations: the checks read the
STORED records (sections A, B), the 79 EVAL graphs (C, F) and the matcher on
all 300 (query, reference) pairs of each window (D, E, F).  EVAL-75 only.

  A  LEDGER            every stored pair of every stored configuration is
                       recomputed from its own stored components: score,
                       node term (form, calibration, chance), relational
                       term, absence, the binary twin, the answers
  B  SWITCH INDEPENDENCE  what a score-level switch must not move (the
                       correspondence and everything accumulated from it) is
                       compared across the stored configurations
  C  GRAPH STORAGE     how undirected edges and their parameters are stored
                       (orientation, key, mixtures): what the edge loops
                       assume about them
  D  WALK              on real pairs: the mirror loop in the core direction
                       against the matcher's own trace; each matched edge
                       pair read from both ends; mixtures read in reverse;
                       the correspondence; traced against untraced scores;
                       the chance level against sampled random pairings
  E  ORDER DEPENDENCE  edge-candidate ties whose resolution depends on the
                       iteration order of an edge set (the hash seed)
  F  BLOCKS            the numpy kernels against the Python loops, == on
                       the floats: every entry of the node-similarity matrix
                       of every lens type of every pair, every anchor
                       weight, the anchor spacing and the edge-weight table
                       of every graph

    python diagnostics/phase2_bug_hunt.py                  # everything
    python diagnostics/phase2_bug_hunt.py --sections A,B   # stored only
    python diagnostics/phase2_bug_hunt.py --window full

Diagnostics only: reads the library, changes nothing.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import ast
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import _stats as st
from _bootstrap import RESULTS_DIR
from _stats import CLASSES

ALL = ("node", "relation", "absence")
TAGS = ("p2_R0", "p2_P4", "p2_D3n", "p2_D3na", "p2_D3", "p2_D3P4", "p2_D3b",
        "p2_NULL", "p2_NULLD3", "p2_NULLb")
#: the records of the two correctness fixes measured after stage 2 (mixture
#: reversal, exact assignment); read by the ledger, not by section B
TAGS_LATER = ("p2m_R0", "p2m_D3", "p2m_BASE", "p2x_DEF", "p2x_D3h", "p2x_D3",
              "p2x_DEFbin", "p2x_D3bin",
              # stage 4: the per-pair loop, and P4b on the two forms
              "p3_DEF_pairs", "p3_CA", "p3_D3_CA", "p3_CAP", "p3_D3_CAP")
#: stored values carry 10 significant digits; an identity recomputed from
#: them is exact to a few units of the 10th digit of its largest operand
TOL = 5e-9

_LINES: List[str] = []


def emit(s: str = "") -> None:
    _LINES.append(s)
    print(s, flush=True)


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def load(tag: str, window: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(RESULTS_DIR, "%s_%s.json" % (tag, window))
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        res = json.load(fh)
    res["_ov"] = {k: ast.literal_eval(v)
                  for k, v in res.get("param_overrides", {}).items()}
    return res


# ===========================================================================
# A. LEDGER
# ===========================================================================

def _form(balanced: bool, qv: float, cv: float, qt: float, ct: float,
          eps: float) -> float:
    if balanced:
        return ((qv + cv) / (qt + ct + eps)) if (qt + ct) > 0 else 0.0
    return (cv / (ct + eps)) if ct > 0 else 0.0


def ledger_pair(pr: Dict[str, float], ov: Dict[str, Any], eps: float,
                alpha: float, lam: float) -> Dict[str, float]:
    """identity name -> |stored - recomputed| for one stored pair."""
    from matching.calibration import calibrate_fraction

    terms = tuple(ov.get("EVIDENCE_QUERY_SIDE_TERMS", ()))
    cal = bool(ov.get("EVIDENCE_NODE_NULL_CALIBRATED", False))
    graded = bool(ov.get("EVIDENCE_GRADED_NODE_CREDIT", True))
    bal = "node" in terms
    d: Dict[str, float] = {}

    qt, ct = pr["q_node_total"], pr["c_node_total"]
    raw = _form(bal, pr["q_matched_nm"], pr["c_matched_nm"], qt, ct, eps)
    d["node_coverage_raw = form(matched masses)"] = abs(
        pr["node_coverage_raw"] - raw)
    if cal:
        n0 = _form(bal, pr["q_chance_nm"], pr["c_chance_nm"], qt, ct, eps)
        d["node_coverage_chance = form(chance masses)"] = abs(
            pr["node_coverage_chance"] - n0)
        # the calibration map amplifies the rounding of its inputs by
        # 1 / (1 - N0); the tolerance below is scaled accordingly
        want = calibrate_fraction(pr["node_coverage_raw"],
                                  pr["node_coverage_chance"])
        scale = 1.0 / max(1e-12, 1.0 - pr["node_coverage_chance"])
        d["node_coverage = calibrate(raw, chance)"] = abs(
            pr["node_coverage"] - want) / max(1.0, scale)
    else:
        d["node_coverage = node_coverage_raw (calibration off)"] = abs(
            pr["node_coverage"] - pr["node_coverage_raw"])
    d["core_node_coverage = c_matched / c_total"] = abs(
        pr["core_node_coverage"]
        - _form(False, 0.0, pr["c_matched_nm"], qt, ct, eps))
    d["query_node_coverage = q_matched / q_total"] = abs(
        pr["query_node_coverage"]
        - ((pr["q_matched_nm"] / (qt + eps)) if qt > 0 else 0.0))
    twin = _form(bal, pr["q_matched_nm_binary"], pr["c_matched_nm_binary"],
                 qt, ct, eps)
    d["node_coverage_binary = binary twin of the form in use"] = abs(
        pr["node_coverage_binary"] - twin)
    if not graded:
        d["binary credit: matched mass = binary mass"] = max(
            abs(pr["q_matched_nm"] - pr["q_matched_nm_binary"]),
            abs(pr["c_matched_nm"] - pr["c_matched_nm_binary"]))

    n_c, v_c = pr["n_core_edges"], pr["verifiable_core_edges_count"]
    m_c = pr["matched_rel_sim_sum"]
    edge_terms = bool(set(terms) & {"relation", "absence"}) and n_c > 0
    n_q = pr.get("n_query_edges", 0.0)
    v_q = pr.get("q_verifiable_edges_count", 0.0)
    m_q = pr.get("q_matched_rel_sim_sum", 0.0)
    d["query-side edge keys present iff an edge term is two-sided"] = float(
        ("q_verifiable_edges_count" in pr) != edge_terms)
    if n_c == 0:
        rc = 1.0
    elif "relation" in terms:
        rc = clamp01((m_c + m_q) / (v_c + v_q)) if (v_c + v_q) > 0 else 0.0
    elif v_c == 0:
        rc = 0.0
    else:
        rc = clamp01(m_c / v_c)
    apart = (bool(ov.get("EVIDENCE_RELATION_COUNT_APART", False))
             and "relation_presence" in pr)
    if "relation_presence" in pr:
        # [P4b] the two factors, from the counts behind them
        two = "relation" in terms and n_c > 0
        v = v_c + (v_q if two else 0.0)
        present = pr["rel_present"] + (pr.get("q_rel_present", 0.0)
                                       if two else 0.0)
        phys_n = pr["rel_physical"] + (pr.get("q_rel_physical", 0.0)
                                       if two else 0.0)
        phys_s = pr["rel_physical_sim_sum"] + (
            pr.get("q_rel_physical_sim_sum", 0.0) if two else 0.0)
        d["relation_presence = present / verifiable"] = abs(
            pr["relation_presence"] - ((present / v) if v > 0 else 0.0))
        d["relation_physics = physical sim sum / physical pairs"] = abs(
            pr["relation_physics"]
            - ((phys_s / phys_n) if phys_n > 0 else 1.0))
    if apart and n_c > 0:
        v = v_c + (v_q if "relation" in terms else 0.0)
        rc2 = (clamp01(pr["relation_presence"] * pr["relation_physics"])
               if v > 0 else 0.0)
        d["relational_consistency = presence x physics (P4b on)"] = abs(
            pr["relational_consistency"] - rc2)
    else:
        d["relational_consistency = sim sum / verifiable"] = abs(
            pr["relational_consistency"] - rc)
    d["relational_multiplier = alpha + (1 - alpha) x consistency"] = abs(
        pr["relational_multiplier"]
        - (alpha + (1.0 - alpha) * pr["relational_consistency"]))
    if n_c > 0 and "absence" in terms:
        ab = ((n_c - v_c) + (n_q - v_q)) / (n_c + n_q)
    elif n_c > 0:
        ab = (n_c - v_c) / n_c
    else:
        ab = 0.0
    d["edge_absence = uncovered edges / edges"] = abs(pr["edge_absence"] - ab)
    d["total_absence = lambda x edge_absence"] = abs(
        pr["total_absence"] - lam * pr["edge_absence"])
    d["core_coverage = node term x multiplier"] = abs(
        pr["core_coverage"] - pr["node_coverage"] * pr["relational_multiplier"])
    d["score = clamp(node term x multiplier - absence)"] = abs(
        pr["score"] - clamp01(pr["node_coverage"] * pr["relational_multiplier"]
                              - pr["total_absence"]))
    return d


#: prefix of a ledger line that is counted and listed but is not a finding
FACT = "[fact] "


def _mean_term(pr: Dict[str, float], terms) -> float:
    """The historical relational mean of a stored pair (edgeless core
    aside): similarity sum / verifiable edges, over the side(s) in use."""
    v = pr["verifiable_core_edges_count"]
    m = pr["matched_rel_sim_sum"]
    if "relation" in terms and pr["n_core_edges"] > 0:
        v += pr.get("q_verifiable_edges_count", 0.0)
        m += pr.get("q_matched_rel_sim_sum", 0.0)
    return (m / v) if v > 0 else 0.0


def invariants_pair(pr: Dict[str, float], ov: Dict[str, Any], eps: float
                    ) -> Dict[str, bool]:
    terms = tuple(ov.get("EVIDENCE_QUERY_SIDE_TERMS", ()))
    bal = "node" in terms
    twin = _form(bal, pr["q_matched_nm_binary"], pr["c_matched_nm_binary"],
                 pr["q_node_total"], pr["c_node_total"], eps)
    t = 1e-9
    out = {
        "node_coverage <= node_coverage_raw": (
            pr["node_coverage"] <= pr["node_coverage_raw"] + t),
        "node_coverage_raw <= binary twin (recomputed) <= 1": (
            pr["node_coverage_raw"] <= twin + t and twin <= 1.0 + t),
        "node_coverage_raw <= node_coverage_binary (as stored)": (
            pr["node_coverage_raw"] <= pr["node_coverage_binary"] + t),
        "graded mass <= binary mass, each side": (
            pr["q_matched_nm"] <= pr["q_matched_nm_binary"] + t
            and pr["c_matched_nm"] <= pr["c_matched_nm_binary"] + t),
        "binary mass <= total, each side": (
            pr["q_matched_nm_binary"] <= pr["q_node_total"] + t
            and pr["c_matched_nm_binary"] <= pr["c_node_total"] + t),
        "verifiable <= edges; sim sum <= verifiable (core side)": (
            pr["verifiable_core_edges_count"] <= pr["n_core_edges"]
            and pr["matched_rel_sim_sum"]
            <= pr["verifiable_core_edges_count"] + t),
        "0 <= score, node term, relation, absence <= 1": all(
            -t <= pr[k] <= 1.0 + t for k in (
                "score", "node_coverage", "relational_consistency",
                "edge_absence")),
    }
    if "relation_presence" in pr:
        # [P4b] physical <= present <= verifiable; the physical sum is part
        # of the whole sum, and a count-only pair adds at most 1.0 to it
        cnt = pr["rel_present"] - pr["rel_physical"]
        out["physical <= present <= verifiable (core side)"] = (
            0 <= pr["rel_physical"] <= pr["rel_present"]
            <= pr["verifiable_core_edges_count"])
        # The two sums are STORED at 10 significant digits -- an error of
        # up to 5e-10 of each -- and are of the order of 100, so comparing
        # them carries up to 1e-9 of the larger one.  (With the absolute
        # 1e-9 this line reported 623 pairs, every one by less than 6e-8:
        # the storage, not the matcher.)
        ts = t + 1e-9 * max(1.0, pr["matched_rel_sim_sum"])
        out["physical sum <= sim sum <= physical sum + count-only pairs"] = (
            pr["rel_physical_sim_sum"] <= pr["matched_rel_sim_sum"] + ts
            and pr["matched_rel_sim_sum"]
            <= pr["rel_physical_sim_sum"] + cnt + ts)
        # "The product never exceeds the mean" presupposes that a count-only
        # pair reads exactly 1.0.  That holds with the Phase-4 floors off
        # and the specificity cap off.  Under the cap a membership flag
        # reads its class's specificity (0.576), and with the floors
        # restored an all-UNKNOWN pair reads 0.2; either can sit below the
        # physics mean, and then the product is ABOVE the mean.  There the
        # line is a fact about two switches in combination, not an
        # invariant, and is counted apart.
        reads_one = (bool(ov.get("EDGE_UNMEASURED_IS_NOT_EVIDENCE", True))
                     and not bool(ov.get("EDGE_SPECIFICITY_CAP_SAME_SCHEMA",
                                         False)))
        never = (pr["relation_presence"] * pr["relation_physics"]
                 <= (_mean_term(pr, terms) + 1e-8))
        if reads_one:
            out["presence x physics <= sim sum / verifiable (never rises)"] = (
                never)
        else:
            out[FACT + "presence x physics > the mean (a count-only pair "
                "reads < 1)"] = never
    if "q_verifiable_edges_count" in pr:
        out["verifiable <= edges; sim sum <= verifiable (query side)"] = (
            pr["q_verifiable_edges_count"] <= pr["n_query_edges"]
            and pr["q_matched_rel_sim_sum"]
            <= pr["q_verifiable_edges_count"] + t)
    if "node_coverage_chance" in pr:
        n0b = _form(bal, pr["q_chance_nm"], pr["c_chance_nm"],
                    pr["q_node_total"], pr["c_node_total"], eps)
        out["0 <= chance level <= 1"] = -t <= n0b <= 1.0 + t
    return out


def section_ledger(windows: List[str]) -> int:
    import parameters as P

    eps = float(P.EPSILON)
    alpha = (float(getattr(P, "EVIDENCE_CONJUNCTION_FLOOR", 0.0))
             if getattr(P, "MATCH_NULL_CALIBRATED", False)
             else float(getattr(P, "EVIDENCE_ALPHA_CONJUNCTION", 0.85)))
    lam = 0.15
    emit("")
    emit("=" * 104)
    emit("A. LEDGER -- every stored pair recomputed from its own stored "
         "components")
    emit("=" * 104)
    emit("   constants read from the tree: EPSILON %g, conjunction floor "
         "alpha %g, absence weight %g; tolerance %g" % (eps, alpha, lam, TOL))
    findings = 0
    fails: Dict[str, Counter] = defaultdict(Counter)
    worst: Dict[str, float] = defaultdict(float)
    inv_fail: Dict[str, Counter] = defaultdict(Counter)
    inv_seen = set()
    n_pairs = Counter()
    pred_bad = Counter()
    digests = set()
    for window in windows:
        for tag in TAGS + TAGS_LATER:
            res = load(tag, window)
            if res is None:
                emit("   (missing: %s_%s)" % (tag, window))
                continue
            digests.add((res.get("source_digest", "")[:12],
                         res.get("harness_digest", "")[:12],
                         res.get("frontend_digest", "")[:12]))
            cfg = "%s %s" % (tag, window)
            for q in res["queries"]:
                scores = {c: q["pairs"][c]["score"] for c in CLASSES}
                best = max(scores.values())
                win = [c for c in CLASSES if scores[c] == best]
                if q["pred"] != win[0] or bool(q["tie"]) != (len(win) > 1):
                    pred_bad[cfg] += 1
                for c in CLASSES:
                    pr = q["pairs"][c]
                    n_pairs[cfg] += 1
                    for name, dv in ledger_pair(pr, res["_ov"], eps, alpha,
                                                lam).items():
                        worst[name] = max(worst[name], dv)
                        if dv > TOL:
                            fails[name][cfg] += 1
                    for name, ok in invariants_pair(pr, res["_ov"],
                                                    eps).items():
                        inv_seen.add(name)
                        if not ok:
                            inv_fail[name][cfg] += 1
            hits = sum(int(q["pred"] == q["true"]) for q in res["queries"])
            if "summary" in res and res["summary"].get("hits", hits) != hits:
                pred_bad[cfg + " (summary hits)"] += 1
    emit("   stored configurations read: %d, pairs %d; digests (source, "
         "harness, front end): %s" % (
             len(n_pairs), sum(n_pairs.values()),
             "; ".join("/".join(d) for d in sorted(digests))))
    emit("")
    emit("   identity                                                    "
         "      pairs failing   largest difference")
    names = sorted(set(worst) | set(fails))
    for name in names:
        n_bad = sum(fails[name].values())
        emit("   %-64s %6d   %.3g%s" % (
            name, n_bad, worst[name],
            "" if not n_bad else "   <-- in: " + ", ".join(
                "%s x%d" % kv for kv in sorted(fails[name].items()))))
        findings += int(n_bad > 0)
    emit("")
    emit("   invariant                                                   "
         "      pairs failing")
    all_inv = sorted(n for n in inv_seen if not n.startswith(FACT))
    for name in all_inv:
        n_bad = sum(inv_fail[name].values())
        emit("   %-64s %6d%s" % (
            name, n_bad, "" if not n_bad else "   <-- in: " + ", ".join(
                "%s x%d" % kv for kv in sorted(inv_fail[name].items()))))
        findings += int(n_bad > 0)
    facts = sorted(n for n in inv_seen if n.startswith(FACT))
    if facts:
        emit("")
        emit("   counted, not a finding (the Phase-4 floors restored, or the "
             "specificity cap on: a count-only pair reads below 1.0 there, "
             "so 'never rises' is not an invariant)")
        for name in facts:
            n_hit = sum(inv_fail[name].values())
            emit("   %-64s %6d%s" % (
                name, n_hit, "" if not n_hit else "   in: " + ", ".join(
                    "%s x%d" % kv for kv in sorted(inv_fail[name].items()))))
    emit("")
    emit("   answers: stored `pred`/`tie` re-derived from the stored scores "
         "(first class on an exact tie): %d queries differ" % sum(
             pred_bad.values()))
    findings += int(sum(pred_bad.values()) > 0)
    return findings


# ===========================================================================
# B. SWITCH INDEPENDENCE
# ===========================================================================

def section_independence(windows: List[str]) -> int:
    emit("")
    emit("=" * 104)
    emit("B. SWITCH INDEPENDENCE -- what a score-level switch must not move, "
         "across the stored configurations")
    emit("=" * 104)
    findings = 0
    groups = [
        ("the correspondence and the binary accumulation (all 10 "
         "configurations)", TAGS,
         ("n_corr", "q_node_total", "c_node_total", "q_matched_nm_binary",
          "c_matched_nm_binary", "n_core_edges",
          "verifiable_core_edges_count", "q_edge_total", "c_edge_total",
          "count_only", "count_only_balanced")),
        ("the graded node masses (the 8 graded-credit configurations)",
         tuple(t for t in TAGS if t not in ("p2_D3b", "p2_NULLb")),
         ("q_matched_nm", "c_matched_nm", "core_node_coverage",
          "query_node_coverage")),
        ("the core-side edge evidence, Phase 4 OFF (R0, D3n, D3na, D3, D3b)",
         ("p2_R0", "p2_D3n", "p2_D3na", "p2_D3", "p2_D3b"),
         ("matched_rel_sim_sum", "n_matched_edges", "coverage_axes_mean",
          "q_matched_em", "c_matched_em")),
        ("the core-side edge evidence, Phase 4 ON (P4, D3P4, NULL, NULLD3, "
         "NULLb)", ("p2_P4", "p2_D3P4", "p2_NULL", "p2_NULLD3", "p2_NULLb"),
         ("matched_rel_sim_sum", "n_matched_edges", "coverage_axes_mean",
          "q_matched_em", "c_matched_em")),
        ("the core-side edge evidence across Phase 4 (R0 against P4): what "
         "Phase 4 moved", ("p2_R0", "p2_P4"),
         ("matched_rel_sim_sum", "n_matched_edges", "coverage_axes_mean",
          "q_matched_em", "c_matched_em")),
        ("the query-side edge evidence, Phase 4 OFF (D3na, D3, D3b)",
         ("p2_D3na", "p2_D3", "p2_D3b"),
         ("n_query_edges", "q_verifiable_edges_count",
          "q_matched_rel_sim_sum")),
        ("the query-side edge evidence across Phase 4 (D3 against D3P4, "
         "NULLD3)", ("p2_D3", "p2_D3P4", "p2_NULLD3"),
         ("n_query_edges", "q_verifiable_edges_count",
          "q_matched_rel_sim_sum")),
        ("the chance masses (NULL against NULLD3)", ("p2_NULL", "p2_NULLD3"),
         ("q_chance_nm", "c_chance_nm")),
    ]
    for window in windows:
        emit("   window = %s" % window)
        for title, tags, keys in groups:
            recs = [(t, load(t, window)) for t in tags]
            recs = [(t, r) for t, r in recs if r is not None]
            if len(recs) < 2:
                continue
            base_tag, base = recs[0]
            n = diff = 0
            worst, where = 0.0, ""
            for t, r in recs[1:]:
                for qa, qb in zip(base["queries"], r["queries"]):
                    for c in CLASSES:
                        for k in keys:
                            if k not in qa["pairs"][c] or k not in qb["pairs"][c]:
                                continue
                            n += 1
                            dv = abs(qa["pairs"][c][k] - qb["pairs"][c][k])
                            if dv != 0.0:
                                diff += 1
                                if dv > worst:
                                    worst, where = dv, "%s vs %s, %s" % (
                                        base_tag, t, k)
            emit("     %-88s values %6d  differing %5d%s" % (
                title, n, diff,
                "" if not diff else "  largest %.3g (%s)" % (worst, where)))
            findings += int(diff > 0)
        # binary credit: graded == binary masses is in the ledger
    return findings


# ===========================================================================
# C. GRAPH STORAGE
# ===========================================================================

def section_storage(windows: List[str]) -> int:
    import eth80_benchmark as bm
    import eth80_frontend as fe
    from core.schema import (EdgeParams, EdgeParamsMixture,
                             reverse_edge_params)
    from eth80_split import load_split
    from orchestration.binder import UNDIRECTED_EDGE_TYPES

    emit("")
    emit("=" * 104)
    emit("C. GRAPH STORAGE -- what the edge loops assume about stored edges "
         "(79 EVAL graphs)")
    emit("=" * 104)
    split = load_split()
    entries = [(c, split["references"][c]) for c in CLASSES] + [
        (c, p) for c in CLASSES for p in split["eval"][c]]
    findings = 0
    for window in windows:
        n = Counter()
        mix_sens = Counter()
        for c, p in entries:
            sg, _sigs = bm.make_window(fe.graph_for(c, p), window)
            params = sg.edge_params or {}
            und_keys = set()
            for e in sg.edges:
                s, d, et = e
                n["edges"] += 1
                und = et in UNDIRECTED_EDGE_TYPES
                n["undirected" if und else "directed"] += 1
                if s == d:
                    n["self loops"] += 1
                if und:
                    if s > d:
                        n["undirected stored with src > dst"] += 1
                    k = (min(s, d), max(s, d), et)
                    if k in und_keys:
                        n["undirected stored in both orientations"] += 1
                    und_keys.add(k)
                canon = (min(s, d), max(s, d), et) if und else e
                ep = params.get(canon) or params.get(e)
                if ep is not None and und and s > d and e not in params:
                    n["undirected src > dst, parameters only under the "
                      "canonical key"] += 1
                if ep is None and und:
                    ep2 = params.get((d, s, et))
                    if ep2 is not None:
                        n["parameters found only under the REVERSED key"] += 1
                    ep = ep2
                if ep is None:
                    n["edges without parameters"] += 1
                elif isinstance(ep, EdgeParamsMixture):
                    n["mixtures"] += 1
                    n["mixture prototypes"] += len(ep.protos)
                    if und and any(reverse_edge_params(pr).quantized
                                   != pr.quantized for pr in ep.protos):
                        n["undirected mixtures a reversal would change"] += 1
                        mix_sens[et.name] += 1
                    if reverse_edge_params(ep) is ep:
                        n["mixtures reverse_edge_params returns unchanged"] += 1
                elif isinstance(ep, EdgeParams):
                    if und and reverse_edge_params(ep).quantized != ep.quantized:
                        n["undirected plain parameters a reversal changes"] += 1
            for k in params:
                if k not in sg.edges and (k[1], k[0], k[2]) not in sg.edges:
                    n["parameter entries with no edge"] += 1
        emit("   window = %s" % window)
        for k in ("edges", "directed", "undirected", "self loops",
                  "undirected stored with src > dst",
                  "undirected stored in both orientations",
                  "undirected src > dst, parameters only under the "
                  "canonical key",
                  "parameters found only under the REVERSED key",
                  "parameter entries with no edge",
                  "edges without parameters", "mixtures",
                  "mixture prototypes",
                  "mixtures reverse_edge_params returns unchanged",
                  "undirected plain parameters a reversal changes",
                  "undirected mixtures a reversal would change"):
            emit("     %-66s %7d" % (k, n[k]))
        if mix_sens:
            emit("       of those, by relation: %s" % ", ".join(
                "%s %d" % kv for kv in mix_sens.most_common()))
        # `reverse_edge_params` itself still returns a mixture unchanged;
        # that is a finding only while the edge loops rely on it.  Under
        # P.EDGE_REVERSE_FLIPS_MIXTURE they reverse a mixture through
        # `_reverse_for_traversal` (section D counts what is still read two
        # ways), and no other caller passes it a mixture.
        import parameters as _P
        relied_on = not bool(getattr(_P, "EDGE_REVERSE_FLIPS_MIXTURE", True))
        emit("       the edge loops reverse a mixture themselves "
             "(EDGE_REVERSE_FLIPS_MIXTURE=%r): %s" % (
                 not relied_on,
                 "FINDING -- these are read unreversed" if relied_on
                 else "not a finding; see section D"))
        findings += int(relied_on and n[
            "undirected mixtures a reversal would change"] > 0)
        findings += int(n["parameters found only under the REVERSED key"] > 0)
        findings += int(n["undirected src > dst, parameters only under the "
                          "canonical key"] > 0)
    return findings


# ===========================================================================
# D. WALK
# ===========================================================================

_STATE: Dict[str, Any] = {}


def _walk(A, B, a_to_b, flip_mixtures: Optional[bool] = None, record=None,
          a_is_core: bool = True):
    """The edge loop of ``memory.cores._edge_side_evidence`` (edges of A,
    searched in B under ``a_to_b``), re-walked so that it can also

      * record every candidate evaluation, keyed (query edge, core edge);
      * read a mixture in reverse through ``flipped_copy`` instead of
        ``reverse_edge_params`` (which returns a mixture unchanged).

    Returns (n_edges, n_verifiable, sim_sum, stats).  With
    ``flip_mixtures=None`` (follow P.EDGE_REVERSE_FLIPS_MIXTURE) the first
    three must equal the library function; the caller asserts it.
    """
    import parameters as P
    from core.schema import (EdgeParamsMixture, polymorphic_edge_similarity,
                             reverse_edge_params)
    from orchestration.binder import UNDIRECTED_EDGE_TYPES

    if flip_mixtures is None:
        flip_mixtures = bool(getattr(P, "EDGE_REVERSE_FLIPS_MIXTURE", False))

    def _canon(e):
        if e[2] in UNDIRECTED_EDGE_TYPES:
            return (min(e[0], e[1]), max(e[0], e[1]), e[2])
        return e

    def _rev(bp):
        if bp is None:
            return None
        if flip_mixtures and isinstance(bp, EdgeParamsMixture):
            return bp.flipped_copy()
        return reverse_edge_params(bp)

    def _sensitive(bp) -> bool:
        return isinstance(bp, EdgeParamsMixture) and any(
            reverse_edge_params(pr).quantized != pr.quantized
            for pr in bp.protos)

    a_params = getattr(A, "edge_params", None) or {}
    b_params = getattr(B, "edge_params", None) or {}
    b_index = defaultdict(list)
    for b_e in B.edges:
        bs, bd, bet = b_e
        b_key = _canon(b_e)
        bp = b_params.get(b_key) or b_params.get(b_e)
        if bp is None and bet in UNDIRECTED_EDGE_TYPES:
            bp = b_params.get((b_e[1], b_e[0], b_e[2]))
        b_index[(bs, bd)].append((bet, bp, b_key, False, False))
        if bet in UNDIRECTED_EDGE_TYPES:
            b_index[(bd, bs)].append(
                (bet, _rev(bp), b_key, True, _sensitive(bp)))

    stats = Counter()
    n_ver = 0
    sim_sum = 0.0
    for a_s, a_d, aet in A.edges:
        if a_s not in a_to_b or a_d not in a_to_b:
            continue
        n_ver += 1
        bs, bd = a_to_b[a_s], a_to_b[a_d]
        a_key = _canon((a_s, a_d, aet))
        ap = a_params.get(a_key) or a_params.get((a_s, a_d, aet))
        cand = list(b_index.get((bs, bd), []))
        if aet in UNDIRECTED_EDGE_TYPES and bs != bd:
            seen = {c[2] for c in cand}
            for c in b_index.get((bd, bs), []):
                if c[2] not in seen:
                    cand.append(c + ("second",))
        best = 0.0
        best_exposed = False
        for c in cand:
            bet, bp, b_key, is_rev, sens = c[0], c[1], c[2], c[3], c[4]
            sim = polymorphic_edge_similarity(
                bp, ap, use_confidence=False, edge_types=(bet, aet))
            stats["candidate evaluations"] += 1
            if is_rev and sens:
                stats["evaluations of a reversal-sensitive mixture read in "
                      "reverse"] += 1
            if record is not None:
                key = (b_key, a_key) if a_is_core else (a_key, b_key)
                kind = ("reversed entry" if is_rev else "stored entry") + (
                    ", second lookup" if len(c) > 5 else "")
                record.setdefault(key, []).append((sim, kind))
            if sim > best:
                best = sim
                best_exposed = bool(is_rev and sens)
        sim_sum += best
        if best_exposed:
            stats["best match is a reversal-sensitive mixture read in "
                  "reverse"] += 1
    return len(A.edges), n_ver, sim_sum, stats


def _score_from(n: float, m: float, v: float, mq: float, vq: float,
                n_c: float, n_q: float, terms) -> float:
    """clamp(N x R - 0.15 x A) from the counts, alpha = 0 (asserted in A)."""
    if n_c == 0:
        rc, ab = 1.0, 0.0
    else:
        if "relation" in terms:
            rc = clamp01((m + mq) / (v + vq)) if (v + vq) > 0 else 0.0
        else:
            rc = clamp01(m / v) if v > 0 else 0.0
        if "absence" in terms:
            ab = ((n_c - v) + (n_q - vq)) / (n_c + n_q)
        else:
            ab = (n_c - v) / n_c
    return clamp01(n * rc - 0.15 * ab)


def _walk_query(i: int) -> Dict[str, Any]:
    import numpy as np
    import parameters as P
    from matching.structural_similarity import GESTALT_LENS_BOOST
    from memory.cores import (_edge_side_evidence, _sig_similarity,
                              evidence_score)

    S = _STATE
    cls, name, (Q, q_sigs) = S["queries"][i]
    out: Dict[str, Any] = {"name": name, "true": cls, "n": Counter(),
                           "worst": defaultdict(float), "pairs": {},
                           "z": [], "kinds": Counter()}
    n = out["n"]
    for rc in CLASSES:
        C, c_sigs = S["refs"][rc]
        tr: Dict[str, Any] = {}
        score, corr = evidence_score(Q, C, q_sigs, c_sigs, trace=tr)
        inv = {c: q for q, c in corr.items()}
        n["pairs"] += 1

        # -- D5 correspondence
        if len(inv) != len(corr):
            n["correspondence not one-to-one"] += 1
        if any(q_sigs[q][0] != c_sigs[c][0] for q, c in corr.items()):
            n["a pair joins two lens types"] += 1
        if any(q not in Q.nodes or c not in C.nodes for q, c in corr.items()):
            n["a paired node is not in its graph"] += 1
        sims = [float(_sig_similarity(q_sigs[q][1], c_sigs[c][1]))
                for q, c in corr.items()]
        n["kept pairs"] += len(sims)
        n["kept pairs with signature similarity <= 0"] += sum(
            int(s <= 0.0) for s in sims)
        if tr.get("correspondence") != corr:
            n["trace correspondence differs from the returned one"] += 1

        # -- D1 mirror loop, core direction, against the trace
        ne, nv, ss = _edge_side_evidence(C, Q, inv)
        if (ne != tr["n_core_edges"]
                or nv != tr["verifiable_core_edges_count"]):
            n["mirror (core direction): counts differ from the trace"] += 1
        if ss != tr["matched_rel_sim_sum"]:
            n["mirror (core direction): sim sum not bit-identical"] += 1
            out["worst"]["mirror sim sum"] = max(
                out["worst"]["mirror sim sum"],
                abs(ss - tr["matched_rel_sim_sum"]))
        # -- D2 the instrument against the library, both directions
        rec_c: Dict[Any, list] = {}
        rec_q: Dict[Any, list] = {}
        wc = _walk(C, Q, inv, record=rec_c, a_is_core=True)
        wq = _walk(Q, C, corr, record=rec_q, a_is_core=False)
        if (wc[0], wc[1], wc[2]) != (ne, nv, ss):
            n["instrument differs from the library (core direction)"] += 1
        if (wq[0], wq[1]) != (tr["n_query_edges"],
                              tr["q_verifiable_edges_count"]) \
                or wq[2] != tr["q_matched_rel_sim_sum"]:
            n["instrument differs from the library (query direction)"] += 1
        q_ver = sum(int(s in corr and d in corr) for s, d, _t in Q.edges)
        if q_ver != tr["q_verifiable_edges_count"]:
            n["query-side verifiable count differs from its definition"] += 1
        n["verifiable core edges"] += nv
        n["verifiable query edges"] += wq[1]
        for k, v in wc[3].items():
            n["core direction: " + k] += v
        for k, v in wq[3].items():
            n["query direction: " + k] += v

        # -- D3 every edge pair read from both ends
        for key in set(rec_c) | set(rec_q):
            a, b = rec_c.get(key), rec_q.get(key)
            if a is None or b is None:
                side = "core" if b is None else "query"
                vals = a if b is None else b
                n["edge pairs evaluated from the %s side only" % side] += 1
                if max(v for v, _k in vals) > 0.0:
                    n["edge pairs evaluated from the %s side only, "
                      "similarity > 0" % side] += 1
                continue
            if len(a) > 1 or len(b) > 1:
                n["edge pairs evaluated more than once from one side"] += 1
            va, vb = max(v for v, _k in a), max(v for v, _k in b)
            n["edge pairs evaluated from both sides"] += 1
            dv = abs(va - vb)
            kind = "core: %s / query: %s" % (a[0][1], b[0][1])
            out["kinds"][kind] += 1
            if dv > 1e-12:
                n["edge pairs read differently from the two sides"] += 1
                out["kinds"][kind + " -- DIFFERENT"] += 1
                out["worst"]["edge pair, two readings"] = max(
                    out["worst"]["edge pair, two readings"], dv)

        # -- D4 mixtures read in reverse: what the OTHER value of
        #    EDGE_REVERSE_FLIPS_MIXTURE would change
        other = not bool(getattr(P, "EDGE_REVERSE_FLIPS_MIXTURE", False))
        rec_cf: Dict[Any, list] = {}
        rec_qf: Dict[Any, list] = {}
        fc = _walk(C, Q, inv, flip_mixtures=other, record=rec_cf,
                   a_is_core=True)
        fq = _walk(Q, C, corr, flip_mixtures=other, record=rec_qf,
                   a_is_core=False)
        # attribution: with every mixture FLIPPED, do the two readings of
        # each edge pair agree?
        rf_c, rf_q = (rec_cf, rec_qf) if other else (rec_c, rec_q)
        for key in set(rf_c) & set(rf_q):
            if abs(max(v for v, _k in rf_c[key])
                   - max(v for v, _k in rf_q[key])) > 1e-12:
                n["edge pairs read differently from the two sides, "
                  "mixtures flipped"] += 1
        if fc[2] != ss:
            n["pairs whose core-side sim sum a flipped mixture changes"] += 1
        if fq[2] != wq[2]:
            n["pairs whose query-side sim sum a flipped mixture changes"] += 1
        out["worst"]["core-side sim sum, flipped mixtures"] = max(
            out["worst"]["core-side sim sum, flipped mixtures"],
            abs(fc[2] - ss))
        n_core = tr["core_node_coverage"]
        n_bal = tr["node_coverage_raw"]
        args = (tr["n_core_edges"], tr["n_query_edges"])
        out["pairs"][rc] = {
            "one": _score_from(n_core, ss, nv, 0.0, 0.0, args[0], args[1],
                               ()),
            "one_flip": _score_from(n_core, fc[2], nv, 0.0, 0.0, args[0],
                                    args[1], ()),
            "bal": _score_from(n_bal, ss, nv, wq[2], wq[1], args[0], args[1],
                               ALL),
            "bal_flip": _score_from(n_bal, fc[2], nv, fq[2], wq[1], args[0],
                                    args[1], ALL),
            "d_core": fc[2] - ss, "d_query": fq[2] - wq[2],
        }

        # -- D6 traced against untraced (every 5th pair)
        if (i * len(CLASSES) + CLASSES.index(rc)) % 5 == 0:
            n["traced/untraced pairs compared"] += 1
            if evidence_score(Q, C, q_sigs, c_sigs)[0] != score:
                n["untraced score differs from the traced one"] += 1

        # -- D7 the chance level against sampled random pairings
        if name in S["mc_names"]:
            rng = np.random.default_rng(S["mc_seed"] + i * 17
                                        + CLASSES.index(rc))
            K = S["mc_k"]
            q_by, c_by = defaultdict(list), defaultdict(list)
            for nid in sorted(Q.nodes):
                if nid in q_sigs:
                    q_by[q_sigs[nid][0]].append(nid)
            for nid in sorted(C.nodes):
                if nid in c_sigs:
                    c_by[c_sigs[nid][0]].append(nid)
            qw = getattr(Q, "node_weights", None) or {}
            cw = getattr(C, "node_weights", None) or {}
            tot_q = np.zeros(K)
            tot_c = np.zeros(K)
            for lt in sorted(set(q_by) & set(c_by)):
                qi, ci = q_by[lt], c_by[lt]
                M = np.array([[min(1.0, max(0.0, float(_sig_similarity(
                    q_sigs[a][1], c_sigs[b][1])))) for b in ci] for a in qi])
                boost = GESTALT_LENS_BOOST.get(lt, 1.0)
                mq = np.array([boost * (float(qw.get(a, 0.0)) if qw else 1.0)
                               for a in qi])
                mc = np.array([boost * (float(cw.get(b, 0.0)) if cw else 1.0)
                               for b in ci])
                nq_, nc_ = len(qi), len(ci)
                small = min(nq_, nc_)
                for k in range(K):
                    if nq_ <= nc_:
                        cols = rng.permutation(nc_)[:small]
                        rows = np.arange(small)
                    else:
                        rows = rng.permutation(nq_)[:small]
                        cols = np.arange(small)
                    g = M[rows, cols]
                    tot_q[k] += float((mq[rows] * g).sum())
                    tot_c[k] += float((mc[cols] * g).sum())
            # on graphs without node weights the two sides hold the SAME
            # mass in every pairing: one independent comparison, not two
            same = bool(np.allclose(tot_q, tot_c, rtol=0.0, atol=1e-9))
            for side, tot, key in (("q", tot_q, "q_chance_nm"),
                                   ("c", tot_c, "c_chance_nm")):
                if side == "q" and same:
                    if abs(float(tr["q_chance_nm"])
                           - float(tr["c_chance_nm"])) > 1e-9:
                        n["chance masses of the two sides differ where "
                          "every pairing gives them equal"] += 1
                    continue
                se = float(tot.std(ddof=1)) / math.sqrt(K)
                closed = float(tr[key])
                out["z"].append({
                    "pair": "%s/%s" % (name, rc), "side": side,
                    "closed": closed, "mc": float(tot.mean()), "se": se,
                    "z": ((closed - float(tot.mean())) / se) if se > 0
                    else 0.0})
    return out


def section_walk(windows: List[str], workers: int) -> int:
    import multiprocessing as mp

    import eth80_benchmark as bm
    import eth80_frontend as fe
    import parameters as P
    from eth80_split import load_split

    emit("")
    emit("=" * 104)
    emit("D. WALK -- the matcher on all 300 (query, reference) pairs of each "
         "window")
    emit("=" * 104)
    settings = {"EDGE_UNPARAMETERISED_STRUCTURAL_MATCH": True,
                "EVIDENCE_GRADED_NODE_CREDIT": True,
                "EVIDENCE_QUERY_SIDE_TERMS": ALL,
                "EDGE_UNMEASURED_IS_NOT_EVIDENCE": True,
                "EVIDENCE_NODE_NULL_CALIBRATED": True,
                "EDGE_REVERSE_FLIPS_MIXTURE": bool(_STATE.get("flip", True)),
                "MATCH_ASSIGNMENT_EXACT": bool(_STATE.get("exact", True))}
    from phase2_d3_p4_null import STAGE4_NEUTRAL
    settings.update(STAGE4_NEUTRAL)
    flip_on = settings["EDGE_REVERSE_FLIPS_MIXTURE"]
    emit("   settings (every switch on, so every loop runs): %s" % "; ".join(
        "%s=%r" % kv for kv in sorted(settings.items())))
    prev = {k: getattr(P, k) for k in settings}
    for k, v in settings.items():
        setattr(P, k, v)
    findings = 0
    split = load_split()
    try:
        for window in windows:
            entries = [(c, p) for c in CLASSES for p in split["eval"][c]]
            refs = {c: bm.make_window(
                fe.graph_for(c, split["references"][c]), window)
                for c in CLASSES}
            queries = [(c, os.path.basename(p),
                        bm.make_window(fe.graph_for(c, p), window))
                       for c, p in entries]
            mc_names = {os.path.basename(split["eval"][c][0])
                        for c in CLASSES}
            _STATE.update(refs=refs, queries=queries, mc_names=mc_names,
                          mc_seed=20261110, mc_k=2000)
            t0 = time.time()
            if workers > 1:
                with mp.get_context("fork").Pool(workers) as pool:
                    rows = pool.map(_walk_query, range(len(queries)),
                                    chunksize=2)
            else:
                rows = [_walk_query(i) for i in range(len(queries))]
            n = Counter()
            worst: Dict[str, float] = defaultdict(float)
            kinds = Counter()
            zs = []
            for r in rows:
                n.update(r["n"])
                kinds.update(r["kinds"])
                zs.extend(r["z"])
                for k, v in r["worst"].items():
                    worst[k] = max(worst[k], v)
            emit("")
            emit("   window = %s   (%d pairs, %.0f s)" % (
                window, n["pairs"], time.time() - t0))

            def line(label, key, bad=True):
                v = n[key]
                emit("     %-78s %9d%s" % (label, v, (
                    "   <-- FINDING" if (bad and v) else "")))
                return int(bad and v > 0)

            emit("   D1 the mirror loop, called in the core direction, is "
                 "the matcher's own loop")
            findings += line("pairs whose counts differ from the trace",
                             "mirror (core direction): counts differ from "
                             "the trace")
            findings += line("pairs whose similarity sum is not "
                             "bit-identical to the trace",
                             "mirror (core direction): sim sum not "
                             "bit-identical")
            emit("   D2 this instrument against the library function")
            findings += line("pairs differing, core direction",
                             "instrument differs from the library (core "
                             "direction)")
            findings += line("pairs differing, query direction",
                             "instrument differs from the library (query "
                             "direction)")
            findings += line("pairs whose query-side verifiable count "
                             "differs from its definition",
                             "query-side verifiable count differs from its "
                             "definition")
            emit("   D3 every edge pair the two loops evaluate, read from "
                 "both ends")
            line("verifiable core edges", "verifiable core edges", False)
            line("verifiable query edges", "verifiable query edges", False)
            line("candidate evaluations, core direction",
                 "core direction: candidate evaluations", False)
            line("candidate evaluations, query direction",
                 "query direction: candidate evaluations", False)
            line("edge pairs evaluated from both sides",
                 "edge pairs evaluated from both sides", False)
            findings += line("  of which the two readings differ (> 1e-12)",
                             "edge pairs read differently from the two "
                             "sides")
            emit("       largest difference between the two readings: %.3g"
                 % worst["edge pair, two readings"])
            line("  ... and with every mixture read through flipped_copy "
                 "(attribution)",
                 "edge pairs read differently from the two sides, mixtures "
                 "flipped", True)
            for k, v in sorted(kinds.items()):
                emit("       %-76s %9d" % (k, v))
            findings += line("edge pairs evaluated from the core side only",
                             "edge pairs evaluated from the core side only")
            line("  of which with similarity > 0",
                 "edge pairs evaluated from the core side only, similarity "
                 "> 0", False)
            findings += line("edge pairs evaluated from the query side only",
                             "edge pairs evaluated from the query side only")
            line("  of which with similarity > 0",
                 "edge pairs evaluated from the query side only, similarity "
                 "> 0", False)
            findings += line("edge pairs evaluated more than once from one "
                             "side",
                             "edge pairs evaluated more than once from one "
                             "side")
            emit("   D4 mixtures read in reverse (reverse_edge_params "
                 "returns a mixture unchanged; flipped_copy reverses it). "
                 "The loops %s them here; the lines below say what the "
                 "other value of the switch would change"
                 % ("FLIP" if flip_on else "DO NOT FLIP"))
            for side in ("core", "query"):
                findings += line(
                    "%s direction: evaluations of a reversal-sensitive "
                    "mixture read in reverse" % side,
                    "%s direction: evaluations of a reversal-sensitive "
                    "mixture read in reverse" % side, bad=not flip_on)
                line("%s direction: of which it is the best match of its "
                     "edge" % side,
                     "%s direction: best match is a reversal-sensitive "
                     "mixture read in reverse" % side, False)
            line("pairs whose core-side similarity sum flipping would change",
                 "pairs whose core-side sim sum a flipped mixture changes",
                 False)
            line("pairs whose query-side similarity sum flipping would "
                 "change",
                 "pairs whose query-side sim sum a flipped mixture changes",
                 False)
            d_core = [r["pairs"][c]["d_core"] for r in rows for c in CLASSES]
            d_query = [r["pairs"][c]["d_query"] for r in rows for c in CLASSES]
            emit("       change of the similarity sum per pair, other value "
                 "- as is: core side mean %+.4f min %+.4f max %+.4f; query "
                 "side mean %+.4f min %+.4f max %+.4f" % (
                     st.mean(d_core), min(d_core), max(d_core),
                     st.mean(d_query), min(d_query), max(d_query)))
            for form, a, b in (("one-sided score (F6+D1 form)", "one",
                                "one_flip"),
                               ("balanced score (D3, all three terms)",
                                "bal", "bal_flip")):
                def answers(key):
                    res = []
                    for r in rows:
                        sc = {c: r["pairs"][c][key] for c in CLASSES}
                        best = max(sc.values())
                        res.append([c for c in CLASSES if sc[c] == best][0])
                    return res
                pa, pb = answers(a), answers(b)
                true = [r["true"] for r in rows]
                ds = [abs(r["pairs"][c][a] - r["pairs"][c][b])
                      for r in rows for c in CLASSES]
                emit("       %-38s scores changed %3d of %d (largest %.4f); "
                     "answers changed %d; hits %d -> %d" % (
                         form, sum(int(x > 0) for x in ds), len(ds), max(ds),
                         sum(int(x != y) for x, y in zip(pa, pb)),
                         sum(int(x == t) for x, t in zip(pa, true)),
                         sum(int(y == t) for y, t in zip(pb, true))))
            emit("   D5 the correspondence")
            line("kept node pairs", "kept pairs", False)
            for k in ("correspondence not one-to-one",
                      "a pair joins two lens types",
                      "a paired node is not in its graph",
                      "kept pairs with signature similarity <= 0",
                      "trace correspondence differs from the returned one"):
                findings += line(k, k)
            emit("   D6 a traced call against an untraced one")
            line("pairs compared", "traced/untraced pairs compared", False)
            findings += line("untraced score differs",
                             "untraced score differs from the traced one")
            emit("   D7 the chance masses against %d sampled random maximum "
                 "pairings (first query of each class x 4 references)"
                 % _STATE["mc_k"])
            if zs:
                z = [r["z"] for r in zs]
                rel = [abs(r["closed"] - r["mc"]) / r["closed"]
                       for r in zs if r["closed"] > 0]
                n3 = sum(int(abs(x) > 3.0) for x in z)
                sd = math.sqrt(sum((x - st.mean(z)) ** 2 for x in z)
                               / (len(z) - 1))
                emit("     independent masses compared %d   z = (closed form "
                     "- sample mean) / standard error: mean %+.2f  sd %.2f  "
                     "largest |z| %.2f  |z| > 3: %d" % (
                         len(z), st.mean(z), sd, max(abs(x) for x in z), n3))
                emit("     (unbiased closed form: mean z within +-%.2f, sd "
                     "within 1 +- %.2f at two standard errors)" % (
                         2.0 / math.sqrt(len(z)),
                         2.0 / math.sqrt(2.0 * (len(z) - 1))))
                emit("     relative difference closed form against sample "
                     "mean: mean %.4f%%  largest %.4f%%" % (
                         100 * st.mean(rel), 100 * max(rel)))
                findings += line(
                    "chance masses of the two sides differ where every "
                    "pairing gives them equal",
                    "chance masses of the two sides differ where every "
                    "pairing gives them equal")
                findings += int(n3 > 1
                                or abs(st.mean(z)) > 3.0 / math.sqrt(len(z)))
    finally:
        for k, v in prev.items():
            setattr(P, k, v)
    return findings


# ===========================================================================
# E. ORDER DEPENDENCE
# ===========================================================================

def _ties_query(i: int) -> Dict[str, Any]:
    """Edge-candidate ties of one query against the four references.

    The matcher's edge loop keeps the FIRST candidate that reaches the best
    similarity (`if sim > best_sim`), and the candidates are listed in the
    iteration order of ``Q.edges`` -- a frozenset whose order depends on the
    process hash seed (the tuples embed EdgeType members).  The similarity
    SUM does not care which of two tied candidates is kept; the edge MASS
    does when they differ in weight or in measured axes (``mcs_score`` /
    ``mcs_similarity`` read it).
    """
    from core.schema import (edge_measured_axes, polymorphic_edge_similarity)
    from memory.cores import (_effective_edge_weights, _reverse_for_traversal,
                              evidence_score)
    from orchestration.binder import UNDIRECTED_EDGE_TYPES

    def _canon(e):
        if e[2] in UNDIRECTED_EDGE_TYPES:
            return (min(e[0], e[1]), max(e[0], e[1]), e[2])
        return e

    S = _STATE
    _cls, _name, (Q, q_sigs) = S["queries"][i]
    n = Counter()
    worst = 0.0
    q_w = _effective_edge_weights(Q)
    q_params = Q.edge_params or {}
    q_index = defaultdict(list)
    for q_e in Q.edges:
        qs, qd, qet = q_e
        q_key = _canon(q_e)
        qp = q_params.get(q_key) or q_params.get(q_e)
        if qp is None and qet in UNDIRECTED_EDGE_TYPES:
            qp = q_params.get((q_e[1], q_e[0], q_e[2]))
        q_index[(qs, qd)].append((qet, qp, q_key))
        if qet in UNDIRECTED_EDGE_TYPES:
            q_index[(qd, qs)].append((qet, _reverse_for_traversal(qp), q_key))
    for rc in CLASSES:
        C, c_sigs = S["refs"][rc]
        _score, corr = evidence_score(Q, C, q_sigs, c_sigs)
        inv = {c: q for q, c in corr.items()}
        c_params = C.edge_params or {}
        n["pairs"] += 1
        pair_hit = False
        for cs, cd, cet in C.edges:
            if cs not in inv or cd not in inv:
                continue
            qs, qd = inv[cs], inv[cd]
            c_key = _canon((cs, cd, cet))
            cp = c_params.get(c_key) or c_params.get((cs, cd, cet))
            cand = list(q_index.get((qs, qd), []))
            if cet in UNDIRECTED_EDGE_TYPES and qs != qd:
                seen = {c[2] for c in cand}
                cand += [c for c in q_index.get((qd, qs), [])
                         if c[2] not in seen]
            n["verifiable core edges"] += 1
            if len(cand) > 1:
                n["... with more than one candidate query edge"] += 1
            sims = [(polymorphic_edge_similarity(
                qp, cp, use_confidence=False, edge_types=(qet, cet)), qp, qk)
                for qet, qp, qk in cand]
            best = max([s for s, _p, _k in sims] or [0.0])
            if best <= 0.0:
                continue
            tied = [(qp, qk) for s, qp, qk in sims if s == best]
            if len(tied) < 2:
                continue
            n["... whose best similarity is reached by several candidates"] += 1
            masses = {(float(q_w.get(qk, q_w.get(_canon(qk), 0.0))),
                       edge_measured_axes(qp, cp)) for qp, qk in tied}
            if len(masses) > 1:
                n["... and the tied candidates carry different mass"] += 1
                pair_hit = True
                vals = [w * best * (1.0 + math.log2(a + 1.0))
                        for w, a in masses]
                worst = max(worst, max(vals) - min(vals))
        n["pairs whose edge mass depends on the candidate order"] += int(
            pair_hit)
    return {"n": n, "worst": worst}


def section_order(windows: List[str], workers: int) -> int:
    import multiprocessing as mp

    import eth80_benchmark as bm
    import eth80_frontend as fe
    from eth80_split import load_split

    emit("")
    emit("=" * 104)
    emit("E. ORDER DEPENDENCE -- does a result depend on the iteration order "
         "of an edge set (the process hash seed)?")
    emit("=" * 104)
    emit("   the edge loop keeps the first candidate that reaches the best "
         "similarity; candidates are listed in frozenset order")
    import parameters as P

    P.EDGE_REVERSE_FLIPS_MIXTURE = bool(_STATE.get("flip", True))
    P.MATCH_ASSIGNMENT_EXACT = bool(_STATE.get("exact", True))
    from phase2_d3_p4_null import STAGE4_NEUTRAL
    for _k, _v in STAGE4_NEUTRAL.items():
        setattr(P, _k, _v)
    emit("   EDGE_REVERSE_FLIPS_MIXTURE=%r  MATCH_ASSIGNMENT_EXACT=%r" % (
        P.EDGE_REVERSE_FLIPS_MIXTURE, P.MATCH_ASSIGNMENT_EXACT))
    findings = 0
    split = load_split()
    for window in windows:
        entries = [(c, p) for c in CLASSES for p in split["eval"][c]]
        refs = {c: bm.make_window(fe.graph_for(c, split["references"][c]),
                                  window) for c in CLASSES}
        queries = [(c, os.path.basename(p),
                    bm.make_window(fe.graph_for(c, p), window))
                   for c, p in entries]
        _STATE.update(refs=refs, queries=queries)
        if workers > 1:
            with mp.get_context("fork").Pool(workers) as pool:
                rows = pool.map(_ties_query, range(len(queries)), chunksize=2)
        else:
            rows = [_ties_query(i) for i in range(len(queries))]
        n = Counter()
        for r in rows:
            n.update(r["n"])
        worst = max(r["worst"] for r in rows)
        emit("   window = %s" % window)
        for k in ("pairs", "verifiable core edges",
                  "... with more than one candidate query edge",
                  "... whose best similarity is reached by several "
                  "candidates",
                  "... and the tied candidates carry different mass",
                  "pairs whose edge mass depends on the candidate order"):
            emit("     %-72s %9d" % (k, n[k]))
        emit("     largest difference in one edge's mass between tied "
             "candidates: %.4f" % worst)
        findings += int(n["... and the tied candidates carry different "
                          "mass"] > 0)
    emit("   (the similarity SUM, and so the evidence score, does not depend "
         "on which tied candidate is kept; its float")
    emit("    summation order does: phase2_continuity.py under another "
         "PYTHONHASHSEED measures both)")
    return findings


# ===========================================================================
# F. the signature matrix in blocks against the per-pair loop
# ===========================================================================

def _blocks_query(i: int) -> Dict[str, Any]:
    """One query against the four references: every lens-type block of the
    correspondence, computed per pair and in blocks, compared with ``==``."""
    from matching.signature_matrix import signature_similarity_matrix
    from memory.cores import (_anchor_weights, _normalised_anchors,
                              _sig_similarity)

    st_ = _STATE
    _cls, _name, (Q, q_sigs) = st_["queries"][i]
    n = Counter()
    t_pair = t_block = 0.0
    by_lens: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0, 0.0])

    def same_float(x, y):
        return (type(y) is float and x == y
                and math.copysign(1.0, x) == math.copysign(1.0, y))

    def bbox(m):
        if not m:
            return None
        xs = [float(p[0]) for p in m.values() if p and len(p) >= 2]
        ys = [float(p[1]) for p in m.values() if p and len(p) >= 2]
        if len(xs) < 2:
            return None
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        return (x0, y0, max(x1 - x0, 1e-9), max(y1 - y0, 1e-9))

    for rc in CLASSES:
        C, c_sigs = st_["refs"][rc]
        q_by: Dict[str, List[int]] = defaultdict(list)
        c_by: Dict[str, List[int]] = defaultdict(list)
        for nid in sorted(Q.nodes):
            if nid in q_sigs:
                q_by[q_sigs[nid][0]].append(nid)
        for nid in sorted(C.nodes):
            if nid in c_sigs:
                c_by[c_sigs[nid][0]].append(nid)
        qa = getattr(Q, "anchor_of_node", None) or {}
        ca = getattr(C, "anchor_of_node", None) or {}
        qb, cb = bbox(qa), bbox(ca)

        def anchor(q, c):
            # the per-pair expression of `_mcs_accumulate`, written out
            if qb is None or cb is None:
                return 1.0
            pa, pb = qa.get(q), ca.get(c)
            if pa is None or pb is None or len(pa) < 2 or len(pb) < 2:
                return 1.0
            qx = (float(pa[0]) - qb[0]) / qb[2]
            qy = (float(pa[1]) - qb[1]) / qb[3]
            cx = (float(pb[0]) - cb[0]) / cb[2]
            cy = (float(pb[1]) - cb[1]) / cb[3]
            return math.exp(-2.5 * math.hypot(qx - cx, qy - cy))

        qn, cn = _normalised_anchors(Q), _normalised_anchors(C)
        for lt in sorted(set(q_by) & set(c_by)):
            q_ids, c_ids = q_by[lt], c_by[lt]
            A = [q_sigs[q][1] for q in q_ids]
            B = [c_sigs[c][1] for c in c_ids]
            t0 = time.perf_counter()
            ref = [[_sig_similarity(a, b) for b in B] for a in A]
            t1 = time.perf_counter()
            got = signature_similarity_matrix(A, B, _sig_similarity)
            t2 = time.perf_counter()
            t_pair += t1 - t0
            t_block += t2 - t1
            row = by_lens[lt]
            row[0] += len(A) * len(B)
            row[1] += t1 - t0
            row[2] += t2 - t1
            w_ref = [[ref[a][b] * anchor(q, c) for b, c in enumerate(c_ids)]
                     for a, q in enumerate(q_ids)]
            w_got = _anchor_weights(got, q_ids, c_ids, qn, cn)
            n["blocks"] += 1
            for ra, rb, wa, wb in zip(ref, got, w_ref, w_got):
                n["entries"] += len(ra)
                for x, y, u, v in zip(ra, rb, wa, wb):
                    if not same_float(x, y):
                        n["similarities differing"] += 1
                    if not same_float(u, v):
                        n["weights differing"] += 1
                    if x != 0.0 and not (0.0 < x < 1.0) and x != 1.0:
                        n["similarities outside [0, 1]"] += 1
    return {"n": n, "t_pair": t_pair, "t_block": t_block,
            "by_lens": {k: list(v) for k, v in by_lens.items()}}


def _reference_spacing(points) -> float:
    """``memory.cores._median_positive_nn`` as it stood before the numpy
    kernel: the double loop, written out."""
    pts = sorted({
        (round(float(p[0]), 6), round(float(p[1]), 6))
        for p in points
        if p is not None and len(p) >= 2
        and math.isfinite(float(p[0])) and math.isfinite(float(p[1]))
    })
    if len(pts) < 2:
        return 0.0
    nn = []
    for i, (x, y) in enumerate(pts):
        best = math.inf
        for j, (u, v) in enumerate(pts):
            if i == j:
                continue
            d = math.hypot(x - u, y - v)
            if 1e-12 < d < best:
                best = d
        if math.isfinite(best):
            nn.append(best)
    if not nn:
        return 0.0
    nn.sort()
    mid = len(nn) // 2
    return nn[mid] if len(nn) % 2 else 0.5 * (nn[mid - 1] + nn[mid])


def _spacing_census(graphs) -> Dict[str, Any]:
    """The anchor spacing and the redundancy-aware edge weights of every
    graph: the numpy kernel against the double loop."""
    import parameters as P
    from memory.cores import _effective_edge_weights, _median_positive_nn

    n = Counter()
    t_loop = t_kernel = 0.0
    keep = getattr(P, "MATCH_SIGNATURE_MATRIX_BLOCKS", True)
    try:
        for sg, _sigs in graphs:
            anchors = getattr(sg, "anchor_of_node", None) or {}
            pts = [p for p in anchors.values()]
            n["graphs"] += 1
            n["anchor points"] += len(pts)
            t0 = time.perf_counter()
            want = _reference_spacing(pts)
            t1 = time.perf_counter()
            P.MATCH_SIGNATURE_MATRIX_BLOCKS = True
            got = _median_positive_nn(pts)
            t2 = time.perf_counter()
            w_on = _effective_edge_weights(sg)
            P.MATCH_SIGNATURE_MATRIX_BLOCKS = False
            off = _median_positive_nn(pts)
            w_off = _effective_edge_weights(sg)
            t_loop += t1 - t0
            t_kernel += t2 - t1
            if not (type(got) is float and got == want and off == want):
                n["spacings differing"] += 1
            if not (w_on == w_off
                    and list(w_on.items()) == list(w_off.items())):
                n["edge-weight tables differing"] += 1
            n["edge weights"] += len(w_on)
    finally:
        P.MATCH_SIGNATURE_MATRIX_BLOCKS = keep
    return {"n": n, "t_loop": t_loop, "t_kernel": t_kernel}


def section_blocks(windows: List[str], workers: int) -> int:
    import multiprocessing as mp

    import eth80_benchmark as bm
    import eth80_frontend as fe
    from eth80_split import load_split

    emit("")
    emit("=" * 104)
    emit("F. SIGNATURE MATRIX IN BLOCKS -- every entry of every lens-type "
         "block against the per-pair loop, and the anchor spacing of every "
         "graph against the double loop (== on the floats)")
    emit("=" * 104)
    findings = 0
    split = load_split()
    for window in windows:
        entries = [(c, p) for c in CLASSES for p in split["eval"][c]]
        refs = {c: bm.make_window(fe.graph_for(c, split["references"][c]),
                                  window) for c in CLASSES}
        queries = [(c, os.path.basename(p),
                    bm.make_window(fe.graph_for(c, p), window))
                   for c, p in entries]
        _STATE.update(refs=refs, queries=queries)
        if workers > 1:
            with mp.get_context("fork").Pool(workers) as pool:
                rows = pool.map(_blocks_query, range(len(queries)),
                                chunksize=2)
        else:
            rows = [_blocks_query(i) for i in range(len(queries))]
        n = Counter()
        for r in rows:
            n.update(r["n"])
        tp = sum(r["t_pair"] for r in rows)
        tb = sum(r["t_block"] for r in rows)
        emit("   window = %s: %d pairs, %d blocks, %d entries" % (
            window, 4 * len(queries), n["blocks"], n["entries"]))
        for k in ("similarities differing", "weights differing",
                  "similarities outside [0, 1]"):
            emit("     %-40s %d%s" % (k, n[k], "   FINDING" if n[k] else ""))
            findings += int(n[k] > 0)
        emit("     the matrices alone: per pair %.1f s, in blocks %.1f s "
             "(x%.1f), summed over the workers" % (tp, tb, tp / max(tb, 1e-9)))
        by_lens: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0, 0.0])
        for r in rows:
            for lt, (e, a, b) in r["by_lens"].items():
                by_lens[lt][0] += e
                by_lens[lt][1] += a
                by_lens[lt][2] += b
        emit("     %-18s %12s %10s %10s %7s" % (
            "lens type", "entries", "per pair", "blocks", "ratio"))
        for lt in sorted(by_lens, key=lambda k: -by_lens[k][1]):
            e, a, b = by_lens[lt]
            emit("     %-18s %12d %9.2fs %9.2fs %6.1fx" % (
                lt, e, a, b, a / max(b, 1e-9)))
        sp = _spacing_census([refs[c] for c in CLASSES]
                             + [g for _c, _n, g in queries])
        m = sp["n"]
        emit("   anchor spacing, window = %s: %d graphs, %d anchor points, "
             "%d edge weights" % (window, m["graphs"], m["anchor points"],
                                  m["edge weights"]))
        for k in ("spacings differing", "edge-weight tables differing"):
            emit("     %-40s %d%s" % (k, m[k], "   FINDING" if m[k] else ""))
            findings += int(m[k] > 0)
        emit("     the spacing alone: double loop %.2f s, kernel %.2f s "
             "(x%.1f)" % (sp["t_loop"], sp["t_kernel"],
                          sp["t_loop"] / max(sp["t_kernel"], 1e-9)))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="A,B,C,D,E,F")
    ap.add_argument("--window", default="both")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--flip", default="1", choices=("0", "1"),
                    help="EDGE_REVERSE_FLIPS_MIXTURE for sections D, E")
    ap.add_argument("--exact", default="1", choices=("0", "1"),
                    help="MATCH_ASSIGNMENT_EXACT for sections D, E")
    ap.add_argument("--out", default=os.path.join(
        RESULTS_DIR, "phase2_bug_hunt.txt"))
    args = ap.parse_args()
    _STATE["flip"] = args.flip == "1"
    _STATE["exact"] = args.exact == "1"
    windows = ["full", "v1"] if args.window == "both" else [args.window]
    sections = [s.strip().upper() for s in args.sections.split(",")]

    import eth80_benchmark as bm
    import eth80_frontend as fe

    emit("BUG HUNT -- identities and invariants of the matcher changes, "
         "EVAL-75 (in-sample)")
    emit("tree now: source %s  harness %s  front end %s" % (
        bm.source_digest()[:12], bm.harness_digest()[:12],
        fe.frontend_source_digest()[:12]))
    total = 0
    per = {}
    if "A" in sections:
        per["A"] = section_ledger(windows)
    if "B" in sections:
        per["B"] = section_independence(windows)
    if "C" in sections:
        per["C"] = section_storage(windows)
    if "D" in sections:
        per["D"] = section_walk(windows, args.workers)
    if "E" in sections:
        per["E"] = section_order(windows, args.workers)
    if "F" in sections:
        per["F"] = section_blocks(windows, args.workers)
    total = sum(per.values())
    emit("")
    emit("=" * 104)
    emit("checks with a finding, per section: %s   total %d" % (
        "  ".join("%s=%d" % kv for kv in sorted(per.items())), total))
    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(_LINES) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
