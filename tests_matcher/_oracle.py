"""The node term of ``evidence_score``, written from its DEFINITION.

Not a copy of the matcher: the masses, the credit, the two forms and the
calibration are written out from the text of the switches' comments, and the
chance level is the definition itself -- the mean, over EVERY maximum pairing
inside each lens type, of the mass that pairing would earn -- by exhaustive
enumeration, not by the closed form the matcher uses.  The pins compare the
matcher's trace with this under every combination of the switches.

Definitions
    mass of node n on side X    boost(lens of n) x w_X(n);  w_X(n) is
                                X.node_weights.get(n, 0.0) when the side
                                carries a weights dict, else 1.0
    credit of a kept pair       graded: signature similarity clipped to [0, 1]
                                binary: 1.0
    matched mass of a side      sum over kept pairs of (mass of the side's
                                node) x credit
    one-sided form              core matched / core total
    balanced form (D3 "node")   (query matched + core matched)
                                / (query total + core total)
    chance mass of a side       expectation of its matched mass over a
                                uniformly random MAXIMUM pairing inside each
                                lens type (credit: graded as above; binary:
                                1.0 where the similarity is positive)
    calibrated                  clamp((N - N0) / (1 - N0)), 0 when N0 >= 1
    binary twin                 the same form, every kept pair credited 1.0,
                                never calibrated

Only small graphs: the enumeration is factorial in the size of a lens type.
"""
from __future__ import annotations

import itertools
from typing import Dict, Tuple

import parameters as P
from matching.calibration import calibrate_fraction
from matching.structural_similarity import GESTALT_LENS_BOOST


def mass(G, n) -> float:
    sg, sigs = G
    lens = sigs[n][0] if n in sigs else ""
    weights = sg.node_weights
    w = float(weights.get(n, 0.0)) if weights else 1.0
    return GESTALT_LENS_BOOST.get(lens, 1.0) * w


def sim(Q, C, q, c) -> float:
    return float(Q[1][q][1].similarity(C[1][c][1]))


def credit(s: float, graded: bool) -> float:
    if graded:
        return min(1.0, max(0.0, s))
    return 1.0 if s > 0.0 else 0.0


def chance_masses(Q, C, graded: bool) -> Tuple[float, float]:
    """(query, core) expected matched mass over ALL maximum pairings."""
    q_ch = c_ch = 0.0
    lenses = sorted({lt for lt, _s in Q[1].values()}
                    & {lt for lt, _s in C[1].values()})
    for lens in lenses:
        qs = sorted(n for n, (lt, _s) in Q[1].items() if lt == lens)
        cs = sorted(n for n, (lt, _s) in C[1].items() if lt == lens)
        if len(qs) <= len(cs):
            small, large, small_is_query = qs, cs, True
        else:
            small, large, small_is_query = cs, qs, False
        tot_q = tot_c = 0.0
        n_pairings = 0
        for image in itertools.permutations(large, len(small)):
            n_pairings += 1
            for a, b in zip(small, image):
                q, c = (a, b) if small_is_query else (b, a)
                g = credit(sim(Q, C, q, c), graded)
                tot_q += mass(Q, q) * g
                tot_c += mass(C, c) * g
        q_ch += tot_q / n_pairings
        c_ch += tot_c / n_pairings
    return q_ch, c_ch


def node_term(Q, C, corr: Dict[int, int], graded: bool, balanced: bool,
              calibrated: bool) -> Dict[str, float]:
    """Every node-term trace key, from the definitions above."""
    eps = P.EPSILON
    q_tot = sum(mass(Q, n) for n in Q[0].nodes)
    c_tot = sum(mass(C, n) for n in C[0].nodes)

    def kept(side_mass, binary=False):
        return sum(side_mass(q, c)
                   * (1.0 if binary else credit(sim(Q, C, q, c), graded))
                   for q, c in corr.items())

    q_nm = kept(lambda q, c: mass(Q, q))
    c_nm = kept(lambda q, c: mass(C, c))
    q_bin = kept(lambda q, c: mass(Q, q), binary=True)
    c_bin = kept(lambda q, c: mass(C, c), binary=True)

    def form(qv, cv):
        if balanced:
            return ((qv + cv) / (q_tot + c_tot + eps)
                    if (q_tot + c_tot) > 0 else 0.0)
        return cv / (c_tot + eps) if c_tot > 0 else 0.0

    raw = form(q_nm, c_nm)
    out = {
        "node_coverage_raw": raw,
        "node_coverage_binary": form(q_bin, c_bin),
        "core_node_coverage": c_nm / (c_tot + eps) if c_tot > 0 else 0.0,
        "query_node_coverage": q_nm / (q_tot + eps) if q_tot > 0 else 0.0,
        "node_coverage": raw,
    }
    if calibrated:
        q_ch, c_ch = chance_masses(Q, C, graded)
        n0 = form(q_ch, c_ch)
        out.update(q_chance_nm=q_ch, c_chance_nm=c_ch,
                   node_coverage_chance=n0,
                   node_coverage=calibrate_fraction(raw, n0))
    return out
