"""U11.2 gate report: direction correctness of every order cue, per scene.

Writes diagnostics/_u11_gate_after.json and prints the table.  Run with the
baseline tree for the "before" numbers (diagnostics/_u11_census_BASELINE.json
holds the pre-repair census; the gate table for the baseline is recorded in
diagnostics/_u11_handoff.md).
"""
from __future__ import annotations

import json
import os
import sys

import scenes as S
import pipeline as P
import truth as T
from scenes import CAL_SCENES, TEST_SCENES, CONTROL_SCENES
from perception import surface_fragments as SF


def row(scene, run):
    out = {}
    for cue, edges in (("occ", run.order["occlusion"]),
                       ("tjo", run.order["t_junction_order"]),
                       ("arms", run.order["t_junction_arms_diag"].get(
                           "resolved", []))):
        c = T.classify_edges(scene, run.fragments, edges)
        out[cue] = dict(emitted=len(edges), correct=len(c["correct"]),
                        wrong=len(c["wrong"]), undecided=len(c["undecided"]),
                        wrong_pairs=[[int(a), int(b), oa, ob]
                                     for a, b, oa, ob in c["wrong"]])
    sites, diag = SF.t_junction_sites(run.records)
    out["tjo_diag"] = diag
    out["totals"] = dict(peaks=len(run.peaks), chains=len(run.chains),
                         rings=sum(1 for c in run.chains if c[1]),
                         records=len(run.records),
                         fragments=len(run.fragments),
                         undecided_fragments=sum(
                             1 for f in run.fragments if not f.owner_is_figure()),
                         layers=len(run.layers))
    lay = T.layer_correctness(scene, run)
    out["layers"] = {k: v for k, v in lay.items()
                     if not isinstance(v, (list, tuple, dict))}
    return out


def main():
    all_names = tuple(CAL_SCENES) + tuple(TEST_SCENES) + tuple(CONTROL_SCENES)
    out = {}
    tot = dict(correct=0, wrong=0, wrong_test=0, undecided=0)
    for nm in all_names:
        scene = S.build(nm)
        run = P.run_pipeline(scene)
        r = row(scene, run)
        split = "TEST" if nm in TEST_SCENES else (
            "CONTROL" if nm in CONTROL_SCENES else "CAL")
        r["split"] = split
        out[nm] = r
        for cue in ("occ", "tjo", "arms"):
            tot["correct"] += r[cue]["correct"]
            tot["wrong"] += r[cue]["wrong"]
            tot["undecided"] += r[cue]["undecided"]
            if split == "TEST":
                tot["wrong_test"] += r[cue]["wrong"]
        print("%-7s %-22s frags=%3d/%3d undecided=%2d | occ %d/%d/%d | "
              "tjo %d/%d/%d | arms %d/%d/%d | layers=%d %s"
              % (split, nm, r["totals"]["fragments"], r["totals"]["records"],
                 r["totals"]["undecided_fragments"],
                 r["occ"]["correct"], r["occ"]["wrong"], r["occ"]["undecided"],
                 r["tjo"]["correct"], r["tjo"]["wrong"], r["tjo"]["undecided"],
                 r["arms"]["correct"], r["arms"]["wrong"], r["arms"]["undecided"],
                 r["totals"]["layers"], r["totals"]), flush=True)
        print("        tjo_diag=%s" % (r["tjo_diag"],))
    out["_TOTAL"] = tot
    print("\nTOTAL correct/wrong/undecided = %d/%d/%d   WRONG ON TEST = %d"
          % (tot["correct"], tot["wrong"], tot["undecided"], tot["wrong_test"]))
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "diagnostics", "_u11_gate_after.json")
    dest = os.path.normpath(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
