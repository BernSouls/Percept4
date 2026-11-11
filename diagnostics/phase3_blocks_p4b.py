"""STAGE 4 -- the signature matrix in blocks, and P4b, against their
pre-registrations (REPORT.md).

PART 1, blocks (``MATCH_SIGNATURE_MATRIX_BLOCKS``).  The declared-default
configuration is measured with the per-pair loop (p3_DEF_pairs) and set
against the same configuration measured in blocks (p2x_DEF): stored against
stored, every key.  The criterion is 0 differing values.  (Every block of
every pair is compared entry by entry in ``phase2_bug_hunt.py`` section F;
every earlier record is re-measured in blocks and compared by
``phase2_continuity.py``.)

PART 2, P4b step 0 (read-only).  What the relational term is made of, from
the records with the switch OFF: verifiable edges, edges with a counterpart,
count-only pairs (no graded axis measured on both sides), physical pairs.

PART 3, P4b (``EVIDENCE_RELATION_COUNT_APART``): the relational term as
presence x physics, on the declared defaults and with D3's three terms on
top; the registered measures and the default rule.

    python diagnostics/phase3_blocks_p4b.py [--window full|v1|both]

EVAL-75, in-sample.  Paired figures are +gained/-lost queries, exact
two-sided McNemar p.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import os
import sys
from typing import Any, Callable, Dict, List

import _stats as st
from _bootstrap import RESULTS_DIR
from _stats import CLASSES
from phase2_d3_p4_null import ALL, BL, CA, SC
from phase2_exact_assignment import base


def with_(ov: Dict[str, Any], **kw) -> Dict[str, Any]:
    out = dict(ov)
    out.update(kw)
    return out


DEF = base(exact=True)
D3 = base(qs=ALL, exact=True)
CONFIGS = [
    ("DEFAULTS", "p2x_DEF", DEF),
    ("DEFAULTS, per-pair loop", "p3_DEF_pairs", with_(DEF, **{BL: False})),
    ("DEFAULTS +P4b", "p3_CA", with_(DEF, **{CA: True})),
    ("D3", "p2x_D3", D3),
    ("D3 +P4b", "p3_D3_CA", with_(D3, **{CA: True})),
    # descriptive only: the documented specificity cap, wired (dark switch)
    ("DEFAULTS +cap", "p3_CAP", with_(DEF, **{SC: True})),
    ("D3 +cap", "p3_D3_CAP", with_(D3, **{SC: True})),
]


def pairs(res: Dict[str, Any]):
    for q in res["queries"]:
        for c in CLASSES:
            yield q, c, q["pairs"][c]


def own_wrong(res: Dict[str, Any], fn: Callable[[Dict[str, float]], float]):
    own = [fn(pr) for q, c, pr in pairs(res) if c == q["true"]]
    wrong = [fn(pr) for q, c, pr in pairs(res) if c != q["true"]]
    return st.mean(own), st.mean(wrong)


def discrimination(res: Dict[str, Any],
                   fn: Callable[[Dict[str, float]], float]) -> float:
    """Share of the (own-class, wrong-class) comparisons inside a query that
    the own-class reference wins; a tie counts half.  0.5 = no information."""
    won = n = 0.0
    for q in res["queries"]:
        own = fn(q["pairs"][q["true"]])
        for c in CLASSES:
            if c == q["true"]:
                continue
            other = fn(q["pairs"][c])
            won += 1.0 if own > other else (0.5 if own == other else 0.0)
            n += 1.0
    return won / n


def two_sided(pr: Dict[str, float]) -> bool:
    return "q_rel_present" in pr


def counts(pr: Dict[str, float]):
    """(verifiable, present, physical, sim sum, physical sim sum) over the
    side(s) the relational term reads in this record."""
    v, p = pr["verifiable_core_edges_count"], pr["rel_present"]
    f, s, fs = (pr["rel_physical"], pr["matched_rel_sim_sum"],
                pr["rel_physical_sim_sum"])
    if two_sided(pr):
        v += pr["q_verifiable_edges_count"]
        p += pr["q_rel_present"]
        f += pr["q_rel_physical"]
        s += pr["q_matched_rel_sim_sum"]
        fs += pr["q_rel_physical_sim_sum"]
    return v, p, f, s, fs


def mean_term(pr: Dict[str, float]) -> float:
    if pr["n_core_edges"] == 0:
        return 1.0
    v, _p, _f, s, _fs = counts(pr)
    return min(1.0, max(0.0, s / v)) if v > 0 else 0.0


def apart_term(pr: Dict[str, float]) -> float:
    if pr["n_core_edges"] == 0:
        return 1.0
    v, _p, _f, _s, _fs = counts(pr)
    if v <= 0:
        return 0.0
    return min(1.0, max(0.0, pr["relation_presence"] * pr["relation_physics"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", default="both")
    args = ap.parse_args()
    windows = ["full", "v1"] if args.window == "both" else [args.window]

    import eth80_benchmark as bm
    import eth80_frontend as fe
    import parameters as P

    out: List[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        out.append(s)

    emit("STAGE 4 -- signature matrix in blocks, and P4b (the count kept "
         "apart from the physics), EVAL-75 (in-sample)")
    emit("source %s  harness %s  front end %s   environment %s" % (
        bm.source_digest()[:12], bm.harness_digest()[:12],
        fe.frontend_source_digest()[:12], bm.environment()))
    emit("the tree declares: %s=%r  %s=%r" % (BL, getattr(P, BL),
                                              CA, getattr(P, CA)))

    blocks_ok = True
    rule = {"rises": 0, "disc_falls": [], "acc_falls": []}
    for w in windows:
        R = {name: bm.run_config(ov, window=w, breakdown=True, tag=tag)
             for name, tag, ov in CONFIGS}
        emit("")
        emit("=" * 108)
        emit("window = %s" % w)
        emit("=" * 108)

        # ------------------------------------------------------------ part 1
        a, b = R["DEFAULTS"], R["DEFAULTS, per-pair loop"]
        n = bad = 0
        worst = 0.0
        for (qa, c, pa), (_qb, _c, pb) in zip(pairs(a), pairs(b)):
            for k in sorted(set(pa) & set(pb)):
                if isinstance(pa[k], dict):
                    n += 1
                    bad += int(pa[k] != pb[k])
                    continue
                n += 1
                if pa[k] != pb[k]:
                    bad += 1
                    worst = max(worst, abs(pa[k] - pb[k]))
        same_ans = sum(int(qa["pred"] == qb["pred"])
                       for qa, qb in zip(a["queries"], b["queries"]))
        emit("PART 1  blocks against the per-pair loop, declared defaults: "
             "%d stored per-pair values compared (every key, the edge and "
             "node breakdowns included): %d differ (largest %g); same "
             "answer on %d/75 queries" % (n, bad, worst, same_ans))
        emit("        wall time of the 300 pairs on 2 workers, front end "
             "excluded: per pair %.0f s, in blocks %.0f s (x%.1f)" % (
                 b["seconds"], a["seconds"],
                 b["seconds"] / max(a["seconds"], 1e-9)))
        blocks_ok = blocks_ok and bad == 0 and same_ans == 75

        # ------------------------------------------------------------ part 2
        emit("")
        emit("PART 2  what the relational term is made of (switch off)")
        emit("  %-9s %-30s %12s %12s" % ("", "per pair, mean", "own class",
                                         "wrong class"))
        for name in ("DEFAULTS", "D3"):
            r = R[name]
            rows = [
                ("verifiable edges", lambda pr: counts(pr)[0]),
                ("  with a counterpart (present)", lambda pr: counts(pr)[1]),
                ("    count-only pairs",
                 lambda pr: counts(pr)[1] - counts(pr)[2]),
                ("    physical pairs", lambda pr: counts(pr)[2]),
                ("presence", lambda pr: pr["relation_presence"]),
                ("physics", lambda pr: pr["relation_physics"]),
                ("mean similarity of ALL present pairs",
                 lambda pr: (counts(pr)[3] / counts(pr)[1])
                 if counts(pr)[1] > 0 else 0.0),
                ("count-only share of the sim sum",
                 lambda pr: ((counts(pr)[3] - counts(pr)[4]) / counts(pr)[3])
                 if counts(pr)[3] > 0 else 0.0),
                ("term as it is (the mean)", mean_term),
                ("term kept apart (presence x physics)", apart_term),
            ]
            for label, fn in rows:
                o, x = own_wrong(r, fn)
                emit("  %-9s %-38s %12.4f %12.4f" % (name, label, o, x))
            few = sum(1 for _q, _c, pr in pairs(r) if 0 < counts(pr)[2] < 5)
            none = sum(1 for _q, _c, pr in pairs(r)
                       if counts(pr)[1] > 0 and counts(pr)[2] == 0)
            emit("  %-9s pairs whose physics rests on fewer than 5 physical "
                 "pairs: %d of 300; present but nothing physical: %d" % (
                     name, few, none))

        # ------------------------------------------------------------ part 3
        emit("")
        emit("PART 3  P4b on (EVIDENCE_RELATION_COUNT_APART)")
        emit("  %-15s hits (split)  per class a/c/c/h  answers a/c/c/h  "
             "pairs at 0   variance of the score: which reference / own "
             "class" % "")
        for name in ("DEFAULTS", "DEFAULTS +P4b", "D3", "D3 +P4b"):
            r = R[name]
            tr = st.tie_report(r)
            pc = st.per_class_hits(r)
            an = st.answers(r)
            vs = st.variance_shares(r)
            zr = st.zero_report(r)
            emit("  %-15s %2d (%5.2f)    %s        %s     %3d          "
                 "%.1f%% / %.1f%%" % (
                     name, tr["first"], tr["split"],
                     "/".join("%2d" % pc[c][0] for c in CLASSES),
                     "/".join("%2d" % an[c] for c in CLASSES),
                     zr["pairs_zero"], 100 * vs["ref"], 100 * vs["own"]))
        for a_name, b_name in (("DEFAULTS", "DEFAULTS +P4b"),
                               ("D3", "D3 +P4b")):
            ra, rb = R[a_name], R[b_name]
            emit("")
            emit("  %s against %s" % (b_name, a_name))
            rises = falls = same = 0
            dmax = 0.0
            for (_qa, _c, pa), (_qb, _c2, pb) in zip(pairs(ra), pairs(rb)):
                d = pb["relational_consistency"] - pa["relational_consistency"]
                rises += int(d > 1e-9)
                falls += int(d < -1e-9)
                same += int(abs(d) <= 1e-9)
                dmax = max(dmax, -d)
            rule["rises"] += rises
            emit("    1. the term: rises on %d pairs (must be 0), falls on "
                 "%d (largest fall %.4f), unchanged on %d" % (
                     rises, falls, dmax, same))
            # the factors the score used are in the record with the switch on
            for label, fn in (
                    ("presence", lambda pr: pr["relation_presence"]),
                    ("physics", lambda pr: pr["relation_physics"])):
                o, x = own_wrong(rb, fn)
                emit("    2. %-9s own %.4f  wrong %.4f  (%+.4f)" % (
                    label, o, x, o - x))
            o1, x1 = own_wrong(ra, lambda pr: pr["relational_consistency"])
            o2, x2 = own_wrong(rb, lambda pr: pr["relational_consistency"])
            emit("    2. the term  own %.4f  wrong %.4f  (%+.4f)   ->   own "
                 "%.4f  wrong %.4f  (%+.4f)" % (o1, x1, o1 - x1,
                                                o2, x2, o2 - x2))
            d_before = discrimination(
                ra, lambda pr: pr["relational_consistency"])
            d_after = discrimination(
                rb, lambda pr: pr["relational_consistency"])
            emit("    3. discrimination of the relational term (own-class "
                 "reference wins, of 225, ties half): %.4f -> %.4f   "
                 "[presence alone %.4f, physics alone %.4f]" % (
                     d_before, d_after,
                     discrimination(rb, lambda pr: pr["relation_presence"]),
                     discrimination(rb, lambda pr: pr["relation_physics"])))
            if a_name == "DEFAULTS" and d_after < d_before:
                rule["disc_falls"].append(w)
            emit("       the relational term deciding alone, hits: %d -> %d"
                 % (sum(st.hits(ra, "relational_consistency")),
                    sum(st.hits(rb, "relational_consistency"))))
            vsa, vsb = st.variance_shares(ra), st.variance_shares(rb)
            emit("    4. variance of the score, which reference / own "
                 "class: %.1f%% / %.1f%%  ->  %.1f%% / %.1f%%" % (
                     100 * vsa["ref"], 100 * vsa["own"],
                     100 * vsb["ref"], 100 * vsb["own"]))
            g, l, p = st.paired(ra, rb)
            moved = sum(int(pa["score"] != pb["score"])
                        for (_q, _c, pa), (_q2, _c2, pb)
                        in zip(pairs(ra), pairs(rb)))
            up = sum(int(pb["score"] > pa["score"] + 1e-9)
                     for (_q, _c, pa), (_q2, _c2, pb)
                     in zip(pairs(ra), pairs(rb)))
            emit("    5. accuracy %d -> %d, paired %s; scores moved %d of "
                 "300 (rose on %d); answers moved %d" % (
                     sum(st.hits(ra)), sum(st.hits(rb)),
                     st.fmt_paired(g, l, p), moved, up,
                     st.changed_answers(ra, rb)))
            if p < 0.05 and l > g:
                rule["acc_falls"].append("%s/%s" % (w, a_name))

        # ------------------------------------------------------------ part 4
        emit("")
        emit("PART 4  the specificity cap on same-schema pairs "
             "(EDGE_SPECIFICITY_CAP_SAME_SCHEMA) -- DESCRIPTIVE, no default "
             "rule, the switch stays off")
        for a_name, b_name in (("DEFAULTS", "DEFAULTS +cap"),
                               ("D3", "D3 +cap")):
            ra, rb = R[a_name], R[b_name]
            tr = st.tie_report(rb)
            pc = st.per_class_hits(rb)
            g, l, p = st.paired(ra, rb)
            vsa, vsb = st.variance_shares(ra), st.variance_shares(rb)
            o1, x1 = own_wrong(ra, lambda pr: pr["relational_consistency"])
            o2, x2 = own_wrong(rb, lambda pr: pr["relational_consistency"])
            s1, _ = own_wrong(ra, lambda pr: pr["matched_rel_sim_sum"])
            s2, _ = own_wrong(rb, lambda pr: pr["matched_rel_sim_sum"])
            emit("  %-13s accuracy %d -> %d (%s), paired %s; similarity sum "
                 "per own-class pair %.1f -> %.1f; the term own/wrong "
                 "%.4f/%.4f -> %.4f/%.4f; its discrimination %.4f -> %.4f; "
                 "deciding alone %d -> %d; variance which reference / own "
                 "class %.1f%%/%.1f%% -> %.1f%%/%.1f%%; pairs at 0: %d -> %d"
                 % (b_name, sum(st.hits(ra)), sum(st.hits(rb)),
                    "/".join("%d" % pc[c][0] for c in CLASSES),
                    st.fmt_paired(g, l, p), s1, s2, o1, x1, o2, x2,
                    discrimination(
                        ra, lambda pr: pr["relational_consistency"]),
                    discrimination(
                        rb, lambda pr: pr["relational_consistency"]),
                    sum(st.hits(ra, "relational_consistency")),
                    sum(st.hits(rb, "relational_consistency")),
                    100 * vsa["ref"], 100 * vsa["own"],
                    100 * vsb["ref"], 100 * vsb["own"],
                    st.zero_report(ra)["pairs_zero"],
                    st.zero_report(rb)["pairs_zero"]))

    emit("")
    emit("=" * 108)
    emit("BLOCKS, criterion 3 at record level (declared defaults; 0 "
         "differing values on every window measured): %s" % (
             "MET" if blocks_ok else "NOT MET"))
    emit("P4b DEFAULT RULE (pre-registered): on only if, on both windows on "
         "the declared defaults, the pins are green, the term rises on no "
         "pair, the discrimination of the relational term does not fall, and "
         "accuracy does not fall with paired p < 0.05 under either form")
    emit("  term rises on a pair: %d   discrimination falls on: %s   "
         "accuracy falls (p < 0.05) on: %s" % (
             rule["rises"], ", ".join(rule["disc_falls"]) or "no window",
             ", ".join(rule["acc_falls"]) or "no window"))
    on = (rule["rises"] == 0 and not rule["disc_falls"]
          and not rule["acc_falls"] and len(windows) == 2)
    emit("  -> the rule says: %s   (the pins are run separately: python -m "
         "pytest tests_matcher)" % (
             "ON" if on else ("OFF" if len(windows) == 2
                              else "undecided (one window measured)")))
    path = os.path.join(RESULTS_DIR, "phase3_blocks_p4b.txt")
    if len(windows) == 2:
        with open(path, "w") as fh:
            fh.write("\n".join(out) + "\n")
    return 0 if blocks_ok else 2


if __name__ == "__main__":
    sys.exit(main())
