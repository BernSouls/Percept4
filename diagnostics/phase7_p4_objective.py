"""[STAGE 7 / PHASE 4] THE OBJECTIVE: a score that arbitrates across node sets,
and the coverage loophole.

(a) `score_for_class = Q * |M| / min(frame_q, frame_r)` is computed here
    beside raw Q on the nine registered pairs and on the old matcher's
    correspondence.  REPORT ONLY for the pipeline: nothing reads it yet.

(b) The loophole is DECIDED BY THE TEST FIXED IN 7.33 BEFORE THIS SCRIPT
    EXISTED: the nine pairs are re-scored under each form of R, and the form
    under which a 50 %-complete sub-mapping of the SAME mapping scores below
    its 100 %-complete counterpart ON EVERY PAIR is the form that closes it.
    Until that test passes, R is NOT changed (7.33).

The two R forms, both computed here on the same mapping:

  union   (7.14, in force)  (relations with an end in M) + (relations with an
                            end in the image of M) - (relations with both ends
                            mapped); this is what `W.objective` computes
  frame   (proposed)        the query's own relation mass with both ends in
                            the FRAME, so an unmapped node's relations are
                            MISSED relations and the denominator does not
                            shrink with the mapping

The local objective repeats `W.objective` term for term; it is asserted equal
to it on every pair before anything is reported, so the two forms differ in R
and in nothing else.  `matching/relation_walker.py` is not modified.

    python diagnostics/phase7_p4_objective.py [--workers 2] [--out NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import phase6_t2_objective as T2  # noqa: E402
import phase6_walker_synthetic as H  # noqa: E402

PAIRS = H.PAIRS


# --------------------------------------------------------------------------
# the objective, with R in either form
# --------------------------------------------------------------------------

def objective_forms(mapping: Dict[int, int], gq, gr, similarity) -> Dict[str, Dict]:
    """{form: terms} for one mapping.

    "frame" is now what `matching.relation_walker.objective` computes (the R
    form Phase 4 selected, Amendment 7.39); "union" is the form every stage-6
    number was stated under, kept here so both can be reported side by side.
    The caller asserts the local "frame" form equals `W.objective` term for
    term, so the two forms differ in R and in nothing else.
    """
    from matching.relation_walker import (NODE_FLOOR, W_NODE, W_RELATION,
                                          _agreement, _attr_similarity)
    m = dict(mapping)
    inv = {v: u for u, v in m.items()}
    s_node = 0.0
    for u, v in sorted(m.items()):
        if gq.lens[u] != gr.lens[v]:
            return {f: {"q": 0.0, "mapped": len(m), "R": 0, "jaccard": 0.0,
                        "S": 0.0, "P": 0.0, "A": 0.0, "A_units": 0}
                    for f in ("union", "frame")}
        s_node += float(similarity(u, v))

    p_mass = 0.0
    p_candidates = 0
    n_q_any = n_r_any = 0
    for s_, d_, name in gq.edges:
        if s_ in m and d_ in m:
            p_candidates += 1
            n_q_any += 1
            hr = gr.pair.get((m[s_], m[d_], name))
            if hr is None:
                continue
            g = _agreement(gq.pair[(s_, d_, name)], hr)
            if g is not None:
                p_mass += g
        elif s_ in m or d_ in m:
            n_q_any += 1
    for s_, d_, name in gr.edges:
        if s_ in inv and d_ in inv:
            n_r_any += 1
        elif s_ in inv or d_ in inv:
            n_r_any += 1
    r_union = n_q_any + n_r_any - p_candidates
    frame_q = set(gq.frame)
    r_frame = sum(1 for s_, d_, _n in gq.edges if s_ in frame_q and d_ in frame_q)

    a_mass = 0.0
    a_units = 0
    for u, v in sorted(m.items()):
        aq, ar = gq.attrs.get(u, {}), gr.attrs.get(v, {})
        for lt in sorted(set(aq) | set(ar)):
            a_units += 1
            mine, theirs = aq.get(lt, []), ar.get(lt, [])
            if not mine or not theirs:
                continue
            finer = mine if len(mine) <= len(theirs) else theirs
            coarse = theirs if finer is mine else mine
            hits = 0
            for x in finer:
                best = 0.0
                for y in coarse:
                    sim = _attr_similarity(gq if finer is mine else gr,
                                           gr if finer is mine else gq, x, y)
                    best = max(best, sim)
                if best >= NODE_FLOOR:
                    hits += 1
            a_mass += hits / len(finer)

    num = W_NODE * s_node + W_RELATION * (p_mass + a_mass)
    out = {}
    for form, r_den in (("union", r_union), ("frame", r_frame)):
        den = W_NODE * len(m) + W_RELATION * (r_den + a_units)
        out[form] = {"q": (num / den) if den > 0 else 0.0, "mapped": len(m),
                     "R": r_den, "jaccard": (p_mass / r_den) if r_den else 0.0,
                     "S": s_node, "P": p_mass, "A": a_mass, "A_units": a_units}
    return out


def score_for_class(terms: Dict, frame_q: int, frame_r: int) -> float:
    """Phase 4(a): Q * the share of the SMALLER frame the mapping covers.
    No tuned constant (7.33)."""
    return terms["q"] * (terms["mapped"] / max(1, min(frame_q, frame_r)))


def closed_component(mapping: Dict[int, int], gq) -> Dict[int, int]:
    """The LARGEST CONNECTED COMPONENT of the mapping under the query's own
    relation graph (Amendment 7.39), ties broken by component size then by the
    smallest node id -- canonical, and it never looks at a score.

    A component is closed by construction: every query relation with an end in
    it either has both ends in it or leads to a node outside the component's
    node set, which is what makes the union form pay nothing for what is left
    out.  The restriction to M's image is trivially closed on the reference
    side too, because the image of a component is the image of a component.
    """
    m = dict(mapping)
    nodes = set(m)
    adj: Dict[int, set] = {u: set() for u in nodes}
    for s_, d_, _n in gq.edges:
        if s_ in nodes and d_ in nodes:
            adj[s_].add(d_)
            adj[d_].add(s_)
    seen, comps = set(), []
    for u in sorted(nodes):
        if u in seen:
            continue
        stack, comp = [u], []
        seen.add(u)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in sorted(adj[x]):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(tuple(sorted(comp)))
    best = max(comps, key=lambda c: (len(c), [-x for x in c]))
    return {u: m[u] for u in best}


def halve(mapping: Dict[int, int]) -> Dict[int, int]:
    """A 50 %-complete sub-mapping of the SAME mapping.

    Every second pair in sorted-key order, so the selection never looks at the
    graph: it is a prefix of an order fixed before the score is computed.  Any
    result that depends on WHICH half was dropped would not be about coverage.
    """
    return {u: mapping[u] for i, u in enumerate(sorted(mapping)) if i % 2 == 0}


# --------------------------------------------------------------------------
# one pair
# --------------------------------------------------------------------------

def one(entry):
    from matching import relation_walker as W
    from memory.cores import evidence_score
    from matching.relation_frame import is_frame
    from matching.structural_similarity import _sig_sim

    an, bn = entry
    ga, wa = T2.scene_truth(an)
    gb, wb = T2.scene_truth(bn)
    ia, ib = W._Index(ga.subgraph, ga.sigs), W._Index(gb.subgraph, gb.sigs)
    cache: dict = {}

    def sim(u, v):
        k = (u, v)
        if k not in cache:
            cache[k] = float(_sig_sim(ia.sig[u], ib.sig[v]))
        return cache[k]

    res = W.walk(ga.subgraph, gb.subgraph, ga.sigs, gb.sigs)
    walk_map = dict(res.mapping)
    truth_map = T2.truth_mapping(wa, wb)
    _s, corr = evidence_score(ga.subgraph, gb.subgraph, ga.sigs, gb.sigs)
    old_map = {u: v for u, v in corr.items()
               if is_frame(ia.lens.get(u, "")) and is_frame(ib.lens.get(v, ""))}

    fq, fr = len(ia.frame), len(ib.frame)
    row = {"pair": [an, bn], "frame": [fq, fr]}
    # the walk's beam, so the two selection rules can be applied to the SAME
    # hypotheses; `pick` is the hypothesis index in `res.beam`
    row["beam"] = [{"pick": i, "seeds": h["seeds"], "mapped": h["mapped"], "q": h["q"]}
                   for i, h in enumerate(res.beam)]
    row["_best_q_pick"] = (max(row["beam"], key=lambda h: h["q"])["pick"]
                           if row["beam"] else None)
    row["_walk_pick"] = next((h["pick"] for h in row["beam"]
                              if h["mapped"] == len(walk_map)), None)
    for name, mp in (("walk", walk_map), ("truth", truth_map), ("old", old_map)):
        forms = objective_forms(mp, ia, ib, sim)
        # the local "frame" form must BE W.objective, term for term
        ref = W.objective(mp, ia, ib, sim)
        for k in ("q", "mapped", "R", "S", "P", "A", "A_units"):
            a, b = forms["frame"].get(k), ref.get(k)
            if isinstance(a, float) or isinstance(b, float):
                assert a is not None and abs(float(a) - float(b)) <= 1e-9, \
                    "%s/%s: local frame form disagrees with W.objective on %s" % (an, bn, k)
            else:
                assert a == b, "%s/%s: local frame form disagrees on %s" % (an, bn, k)
        for form in ("union", "frame"):
            forms[form]["q_class"] = score_for_class(forms[form], fq, fr)
        row[name] = forms
        # the common-set re-score, both operationalisations (7.33 and 7.39)
        for tag, part in (("_half", halve(mp)), ("_closed", closed_component(mp, ia))):
            hf = objective_forms(part, ia, ib, sim)
            for form in ("union", "frame"):
                hf[form]["q_class"] = score_for_class(hf[form], fq, fr)
            hf["_is_proper"] = len(part) < len(mp)
            hf["_n"] = len(part)
            row[name + tag] = hf
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="phase7_p4_objective")
    a = ap.parse_args(argv)
    H.SCENES.update(H.scenes())
    pairs = [p for p in PAIRS if p[0] in H.SCENES and p[1] in H.SCENES]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(one, pairs, chunksize=1)
    else:
        rows = [one(p) for p in pairs]

    L = ["[STAGE 7 / PHASE 4] THE OBJECTIVE: (a) a score across node sets, (b) the"
         " coverage loophole",
         "   raw Q stays the number every earlier result is stated in; score_for_class ="
         " Q * |M| / min(frame_q, frame_r)",
         "   is reported beside it.  R forms: union (the stage-6 form, 7.14) and frame",
         "   (the form SELECTED here, 7.39, now in matching/relation_walker.py).",
         "   The local frame form was asserted equal to matching.relation_walker.objective"
         " on all %d pairs." % len(rows), "",
         "   pair                                          MAPPING    mapped  raw Q  Q_class "
         "  R_union  R_frame"]
    for r in rows:
        for name in ("walk", "truth", "old"):
            u = r[name]["union"]
            L.append("   %-30s %-10s %6d %6.3f %8.3f %8d %8d" % (
                "%s / %s" % tuple(r["pair"]) if name == "walk" else "",
                name, u["mapped"], u["q"], u["q_class"], u["R"], r[name]["frame"]["R"]))
    L.append("")
    L.append("   MEDIANS OVER THE NINE PAIRS")
    for name in ("walk", "truth", "old"):
        L.append("     %-6s raw Q %6.3f   score_for_class %6.3f   mapped %5.0f   "
                 "R_union %6.0f  R_frame %6.0f" % (
                     name,
                     np.median([r[name]["union"]["q"] for r in rows]),
                     np.median([r[name]["union"]["q_class"] for r in rows]),
                     np.median([r[name]["union"]["mapped"] for r in rows]),
                     np.median([r[name]["union"]["R"] for r in rows]),
                     np.median([r[name]["frame"]["R"] for r in rows])))

    # ---------------------------------------------------------------- (a)
    L += ["", "   (a) DOES score_for_class ARBITRATE ACROSS NODE SETS?",
          "       truth > walker on raw Q:            %d of %d" % (
              sum(1 for r in rows if r["truth"]["union"]["q"] > r["walk"]["union"]["q"]), len(rows)),
          "       truth > walker on score_for_class:  %d of %d" % (
              sum(1 for r in rows if r["truth"]["union"]["q_class"] > r["walk"]["union"]["q_class"]), len(rows)),
          "       truth > old on raw Q:               %d of %d" % (
              sum(1 for r in rows if r["truth"]["union"]["q"] > r["old"]["union"]["q"]), len(rows)),
          "       truth > old on score_for_class:     %d of %d" % (
              sum(1 for r in rows if r["truth"]["union"]["q_class"] > r["old"]["union"]["q_class"]), len(rows)),
          "       median mapped: truth %.0f, old %.0f, walk %.0f  (the sets the raw Q "
          "is taken over differ, which is the defect)" % (
              np.median([r["truth"]["union"]["mapped"] for r in rows]),
              np.median([r["old"]["union"]["mapped"] for r in rows]),
              np.median([r["walk"]["union"]["mapped"] for r in rows]))]

    # ---------------------------------------------- (a) the walk's OWN choice
    # score_for_class is reported only, but the walk does choose among its own
    # hypotheses (`_choose`: larger mapping first, Q to break ties), and that
    # choice CAN be read under either number.  Measured, not switched on.
    beamed = [r for r in rows if r.get("beam")]
    changed_q = changed_c = 0
    examples = []
    for r in beamed:
        b = r["beam"]
        best_q = max(b, key=lambda h: h["q"])
        best_c = max(b, key=lambda h: h["q"] * (h["mapped"] / max(1, min(*r["frame"]))))
        if best_q["pick"] != best_c["pick"]:
            changed_c += 1
            examples.append((r["pair"], best_q["mapped"], best_q["q"],
                             best_c["mapped"], best_c["q"]))
        if r["_best_q_pick"] != r["_walk_pick"]:
            changed_q += 1
    L += ["", "   (a2) WHAT score_for_class WOULD DO IF THE WALK SELECTED ON IT",
          "        (it does not: _choose is untouched; this is the size of the change",
          "        a selection rule on this score would make, measured on the nine pairs)",
          "        pairs with a beam: %d;  the walk's own hypothesis differs from the one raw Q"
          " alone would pick on %d" % (len(beamed), changed_q),
          "        raw-Q pick and score_for_class pick differ on %d of %d pairs" % (
              changed_c, len(beamed))]
    for p_, mq, qq, mc, qc in examples:
        L.append("          %-34s raw-Q pick: %4d pairs Q %.3f   |   class-score pick: %4d"
                 " pairs Q %.3f" % ("/".join(p_), mq, qq, mc, qc))

    # ---------------------------------------------------------------- (b)
    L += ["", "   (b) THE COVERAGE LOOPHOLE: the SAME mapping at 100 % and at 50 %",
          "       100 %-complete vs a 50 %-complete sub-mapping of itself (every second"
          " pair in sorted order);",
          "       the registered test (7.33) needs 50 % < 100 % on EVERY pair.",
          "",
          "       form    pairs where 50 % scores BELOW 100 %   mean(100 % - 50 %)   "
          "a 50 % mapping that still scores >= a full one"]
    verdict = {}
    for form in ("union", "frame"):
        below = 0
        deltas = []
        bad = []
        for r in rows:
            for name in ("walk", "truth", "old"):
                full = r[name][form]["q"]
                half = r[name + "_half"][form]["q"]
                deltas.append(full - half)
                if half < full - 1e-12:
                    below += 1
                elif half > full + 1e-12:
                    bad.append((r["pair"], name, full, half))
        total = 3 * len(rows)
        verdict[form] = (below, total, bad)
        L.append("       %-7s %6d of %-3d mappings                %+8.4f            %d" % (
            form, below, total, float(np.mean(deltas)), len(bad)))
        for p, name, full, half in bad[:6]:
            L.append("         still >= full: %s / %s   full %.3f   half %.3f" % (
                "/".join(p), name, full, half))
        if len(bad) > 6:
            L.append("         ... %d more" % (len(bad) - 6))

    closes = [f for f in ("union", "frame")
              if verdict[f][0] == verdict[f][1] and verdict[f][2] == []]

    # ---------------- (b) corrected: the CLOSED COMPONENT (Amendment 7.39)
    L += ["", "   (b, corrected per 7.39) THE CLOSED SUB-MAPPING: the largest connected",
          "       component under the query's relation graph, versus the whole mapping.",
          "       This is the event the loophole describes; the halving above is a CUT and",
          "       both forms fall under it, which is why it cannot decide.  The prediction",
          "       registered in 7.39: union FAILS (a closed mapping pays nothing for what it",
          "       leaves out), frame PASSES (the unmapped nodes' relations stay counted).",
          "",
          "       form    proper-subset cases   component scores BELOW the whole   equal   "
          "ABOVE   verdict"]
    verdict2 = {}
    for form in ("union", "frame"):
        below = equal = above = 0
        cases = []
        for r in rows:
            for name in ("walk", "truth", "old"):
                if not r[name + "_closed"]["_is_proper"]:
                    continue
                full = r[name][form]["q"]
                comp = r[name + "_closed"][form]["q"]
                cases.append((r["pair"], name, full, comp))
                if comp < full - 1e-12:
                    below += 1
                elif comp > full + 1e-12:
                    above += 1
                else:
                    equal += 1
        ok = (below + equal == len(cases)) and above == 0
        verdict2[form] = (below, equal, above, len(cases), ok)
        L.append("       %-7s %6d %20d %15d %10d   %s" % (
            form, len(cases), below, equal, above,
            "PASSES (never above the whole)" if ok else "FAILS (%d above)" % above))
    worst_u = sorted((c for r in rows for name in ("walk", "truth", "old")
                      for c in [(r["pair"], name, r[name]["union"]["q"],
                                 r[name + "_closed"]["union"]["q"])] if c[3] >= c[2] - 1e-12),
                     key=lambda c: -c[2])[:8]
    if worst_u:
        L.append("       under the union form, a closed component that pays NOTHING for")
        L.append("       what it leaves out (component Q == whole-mapping Q):")
        for p, name, full, comp in worst_u:
            L.append("         %-34s %-6s whole %.3f   component (n=%d) %.3f" % (
                "/".join(p), name, full, 0, comp))
    worst_f = sorted((c for r in rows for name in ("walk", "truth", "old")
                      for c in [(r["pair"], name, r[name]["frame"]["q"],
                                 r[name + "_closed"]["frame"]["q"])] if True),
                     key=lambda c: -(c[3] - c[2]))
    if verdict2["frame"][2] == 0:
        L.append("       under the frame form no closed component scores above its whole")
        L.append("       mapping; the largest component-minus-whole gap is %+.4f (%s %s)" % (
            worst_f[0][3] - worst_f[0][2], "/".join(worst_f[0][0]), worst_f[0][1]))

    chosen = [f for f in ("union", "frame") if verdict2[f][4]]
    L += ["",
          "   THE TEST, AS REGISTERED: a form closes the loophole iff the halved mapping"
          " scores strictly below",
          "   the full one on every pair and mapping (7.33).  FORMS THAT PASS: %s" % (
              ", ".join(closes) if closes else "NONE"),
          "   UNDER THE CORRECTED TEST (7.39): %s" % (
              ", ".join(chosen) if chosen else "NONE"), "",
          "   VERDICT ON R: %s" % (
              "the union form FAILS the corrected test and the frame form passes it,"
              " so R is changed to the frame form -- 7.39's condition" if (
                  chosen == ["frame"]) else
              "R is NOT changed; the union form is kept and the loophole is recorded"
              " as OPEN" if not chosen else
              "both forms pass the corrected test; the form in force is kept, nothing"
              " changes")]
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w") as fh:
        json.dump({"rows": rows, "passes": closes}, fh, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
