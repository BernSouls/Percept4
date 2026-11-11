"""Small paired statistics for the audit reports.

Kept OUT of ``eth80_benchmark.py`` on purpose: that file is part of the
harness digest, and a change to it invalidates every cached result.  Nothing
here changes a score or a decision; it only re-reads saved results.

With 75 queries a difference of a few queries is not evidence of anything.
Every comparison in the audits is therefore reported PAIRED (the same 75
queries under two rules): how many queries were gained, how many were lost,
and the exact two-sided McNemar p-value on those discordant queries.
"""
from __future__ import annotations

from math import comb, sqrt
from typing import Any, Dict, List, Optional, Sequence, Tuple

CLASSES: Tuple[str, ...] = ("apple", "car", "cup", "horse")


def mcnemar_exact(gained: int, lost: int) -> float:
    """Exact two-sided McNemar (binomial, p = 1/2) on the discordant pairs."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def wilson(hits: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def winners(q: Dict[str, Any], key: Optional[str] = None) -> Tuple[List[str], float]:
    """Classes holding the maximum of ``key`` for one query, and that maximum."""
    k = key or "score"
    sc = {c: q["pairs"][c][k] for c in CLASSES}
    best = max(sc.values())
    return [c for c in CLASSES if sc[c] == best], best


def hits(res: Dict[str, Any], key: Optional[str] = None) -> List[int]:
    """Per-query hit (0/1) under the benchmark's rule: an exact tie goes to the
    first class in the order apple, car, cup, horse."""
    return [int(winners(q, key)[0][0] == q["true"]) for q in res["queries"]]


def tie_report(res: Dict[str, Any], key: Optional[str] = None) -> Dict[str, float]:
    """How much of an accuracy is the tie-break.

    first      hits under the benchmark's rule (tie -> first class in order)
    strict     hits counting every tied query as WRONG
    split      expected hits if a tie is broken uniformly at random
    ties       tied queries
    ties_zero  tied queries whose best score is exactly 0.0 (the score clamped
               to zero for every tied reference)
    """
    first = strict = ties = ties_zero = 0
    split = 0.0
    for q in res["queries"]:
        w, best = winners(q, key)
        first += int(w[0] == q["true"])
        if len(w) > 1:
            ties += 1
            ties_zero += int(best == 0.0)
        else:
            strict += int(w[0] == q["true"])
        split += (1.0 / len(w)) if q["true"] in w else 0.0
    return {"first": first, "strict": strict, "split": split, "ties": ties,
            "ties_zero": ties_zero, "n": len(res["queries"])}


def paired(res_a: Dict[str, Any], res_b: Dict[str, Any],
           key_a: Optional[str] = None,
           key_b: Optional[str] = None) -> Tuple[int, int, float]:
    """(gained, lost, p): queries B gets right that A gets wrong, the reverse,
    and the exact McNemar p.  A and B must list the same queries in order."""
    for qa, qb in zip(res_a["queries"], res_b["queries"]):
        assert qa["name"] == qb["name"], (qa["name"], qb["name"])
    ha, hb = hits(res_a, key_a), hits(res_b, key_b)
    gained = sum(int(b and not a) for a, b in zip(ha, hb))
    lost = sum(int(a and not b) for a, b in zip(ha, hb))
    return gained, lost, mcnemar_exact(gained, lost)


def fmt_paired(g: int, l: int, p: float) -> str:
    return "+%d/-%d vs baseline, McNemar p=%.3f" % (g, l, p)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------
# helpers shared by the stage scripts from stage 2 on (read-only)
# --------------------------------------------------------------------------

def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n == 0:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def pooled_r(res: Dict[str, Any], ka: str, kb: str) -> float:
    """Pearson r over all (query, reference) pairs."""
    xs = [q["pairs"][c][ka] for q in res["queries"] for c in CLASSES]
    ys = [q["pairs"][c][kb] for q in res["queries"] for c in CLASSES]
    return pearson(xs, ys)


def within_query_r(res: Dict[str, Any], ka: str, kb: str) -> float:
    """Pearson r after removing each query's own mean from both quantities:
    only the differences BETWEEN the four references of one query -- the only
    thing an argmax over references can read."""
    xs: List[float] = []
    ys: List[float] = []
    for q in res["queries"]:
        a = [q["pairs"][c][ka] for c in CLASSES]
        b = [q["pairs"][c][kb] for c in CLASSES]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        xs += [v - ma for v in a]
        ys += [v - mb for v in b]
    return pearson(xs, ys)


def variance_shares(res: Dict[str, Any],
                    key: Optional[str] = None) -> Dict[str, float]:
    """The Audit B6 measure, for any per-pair quantity.

    Over the 4 x n (query, reference) values:
      ref    share of the variance explained by WHICH REFERENCE the value
             belongs to (between-reference sum of squares / total)
      own    share explained by own-class reference vs not
    and the two means (own-class pairs, wrong-class pairs).
    """
    k = key or "score"
    qs = res["queries"]
    n = len(qs)
    own = [q["pairs"][q["true"]][k] for q in qs]
    wrong = [q["pairs"][c][k] for q in qs for c in CLASSES if c != q["true"]]
    all_s = own + wrong
    m_all = mean(all_s)
    sst = sum((x - m_all) ** 2 for x in all_s)
    ss_ref = sum(n * (mean([q["pairs"][c][k] for q in qs]) - m_all) ** 2
                 for c in CLASSES)
    ss_own = (len(own) * (mean(own) - m_all) ** 2
              + len(wrong) * (mean(wrong) - m_all) ** 2)
    return {"ref": (ss_ref / sst) if sst > 0 else float("nan"),
            "own": (ss_own / sst) if sst > 0 else float("nan"),
            "mean_own": mean(own), "mean_wrong": mean(wrong),
            "ref_means": {c: mean([q["pairs"][c][k] for q in qs])
                          for c in CLASSES}}


def add_centred(res: Dict[str, Any], key: Optional[str] = None) -> str:
    """OBSERVABLE ONLY (Audit B6): write, per pair, the value minus the mean
    its reference receives from the OTHER queries (leave-one-out). Uses the
    query set's own statistics; it says where the signal is, not what a
    deployable score reaches. Returns the name of the new per-pair key."""
    k = key or "score"
    ck = "_centred_" + k
    qs = res["queries"]
    n = len(qs)
    tot = {c: sum(q["pairs"][c][k] for q in qs) for c in CLASSES}
    for q in qs:
        for c in CLASSES:
            v = q["pairs"][c][k]
            q["pairs"][c][ck] = v - (tot[c] - v) / (n - 1)
    return ck


def answers(res: Dict[str, Any], key: Optional[str] = None) -> Dict[str, int]:
    """How often each class is answered (benchmark tie rule)."""
    out = {c: 0 for c in CLASSES}
    for q in res["queries"]:
        out[winners(q, key)[0][0]] += 1
    return out


def per_class_hits(res: Dict[str, Any],
                   key: Optional[str] = None) -> Dict[str, Tuple[int, int]]:
    per = {c: [0, 0] for c in CLASSES}
    for q, h in zip(res["queries"], hits(res, key)):
        per[q["true"]][0] += h
        per[q["true"]][1] += 1
    return {c: (v[0], v[1]) for c, v in per.items()}


def zero_report(res: Dict[str, Any],
                key: Optional[str] = None) -> Dict[str, int]:
    """Where a score stops being a score.

    pairs_zero    (query, reference) pairs whose value is exactly 0.0
    own_zero      own-class pairs whose value is exactly 0.0
    all_equal     queries whose four values are all equal (undecidable)
    all_zero      queries whose four values are all exactly 0.0
    """
    k = key or "score"
    pz = oz = ae = az = 0
    for q in res["queries"]:
        vals = [q["pairs"][c][k] for c in CLASSES]
        pz += sum(int(v == 0.0) for v in vals)
        oz += int(q["pairs"][q["true"]][k] == 0.0)
        ae += int(len(set(vals)) == 1)
        az += int(all(v == 0.0 for v in vals))
    return {"pairs_zero": pz, "own_zero": oz, "all_equal": ae, "all_zero": az}


def changed_answers(a: Dict[str, Any], b: Dict[str, Any],
                    key_a: Optional[str] = None,
                    key_b: Optional[str] = None) -> int:
    return sum(int(winners(qa, key_a)[0][0] != winners(qb, key_b)[0][0])
               for qa, qb in zip(a["queries"], b["queries"]))


def compare_stored(a: Dict[str, Any], b: Dict[str, Any],
                   keys: Optional[Sequence[str]] = None
                   ) -> Dict[str, Any]:
    """Per-pair comparison of two STORED results (same queries, same order).

    Compares every per-pair key the two records share (or `keys`).
    Returns n compared, n differing, the largest absolute difference and the
    key it is on, and the number of pairs whose `score` differs.
    """
    n_cmp = n_diff = n_score = 0
    worst, worst_key = 0.0, ""
    for qa, qb in zip(a["queries"], b["queries"]):
        assert qa["name"] == qb["name"], (qa["name"], qb["name"])
        for c in CLASSES:
            pa, pb = qa["pairs"][c], qb["pairs"][c]
            ks = keys if keys is not None else sorted(
                k for k in pa if k in pb and not k.startswith("_")
                and isinstance(pa[k], (int, float)))
            for k in ks:
                n_cmp += 1
                d = abs(float(pa[k]) - float(pb[k]))
                if d != 0.0:
                    n_diff += 1
                    if k == "score":
                        n_score += 1
                    if d > worst:
                        worst, worst_key = d, k
    return {"n": n_cmp, "differ": n_diff, "score_differ": n_score,
            "worst": worst, "worst_key": worst_key}
