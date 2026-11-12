"""[STAGE 6 / Amendment 2.7] THE MAP SWITCH ON THE MULTI-STEP ROUTE.

Completing the binder's position map wakes existing edge channels, and on one
presentation they emit 76-95 % of their candidates against a target rate of
0.2.  Each channel has a homeostatic threshold (theta) that the binder moves
toward the target after every presentation.  Before the switch can default
on: where does theta go?

Protocol (pre-registered): 16 images (the 4 references and the first 3 EVAL
images of each class), 8 presentations of each, switch off and on.  Per
affected channel and per presentation: theta, the emission rate of that
presentation, candidates, emitted (median over images).  THE RULE: the switch
cannot default on while any affected channel's theta is still moving by more
than 0.01 per step at step 8, or has reached its ceiling.

    python diagnostics/phase6_map_multistep.py --map 0      # writes the off half
    python diagnostics/phase6_map_multistep.py --map 1      # writes the on half
    python diagnostics/phase6_map_multistep.py --report     # the table
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_ap = argparse.ArgumentParser()
_ap.add_argument("--map", type=int, default=None)
_ap.add_argument("--report", action="store_true")
_ap.add_argument("--steps", type=int, default=8)
_ap.add_argument("--workers", type=int, default=2)
_ap.add_argument("--out", default="phase6_map_multistep")
ARGS = _ap.parse_args()
if ARGS.map is not None:
    os.environ["PERCEPT_SET"] = "BINDER_COMPLETE_POSITION_MAP=%d" % int(bool(ARGS.map))

import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import numpy as np  # noqa: E402

AFFECTED = ("BORDER_OWNERSHIP_EDGE", "TEXTURE_BOUNDARY_EDGE", "LINE_TERMINATION",
            "PART_CONTAINMENT", "SHARED_BOUNDARY")


def one(entry):
    import eth80_frontend as fe
    cls, rel = entry
    tr = {}
    fe.build_graph(fe._load_rgb(fe.image_path(cls, rel)), name=os.path.basename(rel),
                   n_steps=ARGS.steps, trace=tr, record_co_located=False)
    steps = []
    prev = {}
    for snap in tr["channels_by_step"]:
        row = {}
        for name, st in snap.items():
            c0, e0 = prev.get(name, (0, 0))
            row[name] = {"theta": st["theta"], "r_bar": st["r_bar"],
                         "candidates": st["candidates"] - c0, "emitted": st["emitted"] - e0}
            prev[name] = (st["candidates"], st["emitted"])
        steps.append(row)
    return {"class": cls, "image": os.path.basename(rel), "steps": steps}


def _path(flag):
    return os.path.join(RESULTS_DIR, "%s_%s.json" % (ARGS.out, "on" if flag else "off"))


def run(flag):
    import parameters as P
    import eth80_split as sp
    assert bool(P.BINDER_COMPLETE_POSITION_MAP) == bool(flag)
    split = sp.load_split()
    entries = list(split["references"].items()) + [
        (c, p) for c, ps in sorted(split["eval"].items()) for p in ps[:3]]
    if ARGS.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(ARGS.workers) as pool:
            rows = pool.map(one, entries, chunksize=1)
    else:
        rows = [one(e) for e in entries]
    with open(_path(flag), "w") as fh:
        json.dump({"steps": ARGS.steps, "theta_min": float(P.THETA_CHANNEL_MIN),
                   "theta_max": float(P.THETA_CHANNEL_MAX),
                   "target": float(P.RHO_CHANNEL_TARGET), "rows": rows}, fh, sort_keys=True)
    print("wrote", _path(flag), len(rows), "images")


def report():
    data = {}
    for flag in (0, 1):
        with open(_path(flag)) as fh:
            data[flag] = json.load(fh)
    n_steps = data[1]["steps"]
    tmax, tmin, target = data[1]["theta_max"], data[1]["theta_min"], data[1]["target"]
    L = ["[STAGE 6 / Amendment 2.7] the map switch on the multi-step route: %d images, %d presentations" % (
        len(data[1]["rows"]), n_steps),
         "channel threshold theta in [%.2f, %.2f], target emission rate %.2f; medians over images" % (
             tmin, tmax, target), ""]
    names = sorted({n for r in data[1]["rows"] for s in r["steps"] for n in s})
    verdicts = []
    changed = []
    for name in names:
        def series(flag, key):
            out = []
            for t in range(n_steps):
                v = [r["steps"][t][name][key] for r in data[flag]["rows"]
                     if t < len(r["steps"]) and name in r["steps"][t]]
                out.append(float(np.median(v)) if v else float("nan"))
            return out
        c_on, c_off = series(1, "candidates"), series(0, "candidates")
        if name not in AFFECTED and all(
                (a == b) or (np.isnan(a) and np.isnan(b)) for a, b in zip(c_on, c_off)):
            continue
        changed.append(name)
        L.append("%s%s" % (name, "" if name in AFFECTED else "   (not in the list of woken channels; its candidates differ)"))
        for flag, label in ((0, "off"), (1, "on ")):
            th, cand, emit = series(flag, "theta"), series(flag, "candidates"), series(flag, "emitted")
            rate = [e / c if c else float("nan") for e, c in zip(emit, cand)]
            L.append("   %s theta      " % label + " ".join("%6.3f" % v for v in th))
            L.append("   %s candidates " % label + " ".join("%6.0f" % v for v in cand))
            L.append("   %s emitted    " % label + " ".join("%6.0f" % v for v in emit))
            L.append("   %s rate       " % label + " ".join("%6.2f" % v for v in rate))
        # the rule, per image then summarised: theta still moving at the last step, or at the ceiling
        moving = ceiling = n_img = 0
        last_move = []
        for r in data[1]["rows"]:
            ths = [s[name]["theta"] for s in r["steps"] if name in s]
            if len(ths) < 2:
                continue
            n_img += 1
            d = abs(ths[-1] - ths[-2])
            last_move.append(d)
            moving += int(d > 0.01)
            ceiling += int(ths[-1] >= tmax - 1e-9)
        med_th = series(1, "theta")
        med_move = abs(med_th[-1] - med_th[-2]) if len(med_th) > 1 else float("nan")
        bad = (med_move > 0.01) or (med_th[-1] >= tmax - 1e-9)
        verdicts.append((name, bad))
        L.append("   switch on, last step: median theta moved %.4f; images still moving by more than 0.01: %d of %d; "
                 "at the ceiling: %d of %d  ->  %s" % (
                     med_move, moving, n_img, ceiling, n_img,
                     "NOT SETTLED" if bad else "settled"))
        L.append("")
    blocked = [n for n, bad in verdicts if n in AFFECTED and bad]
    L.append("THE RULE (pre-registered): the switch cannot default on while any affected channel's theta still moves by "
             "more than 0.01 per step at step %d, or has reached its ceiling." % n_steps)
    L.append("VERDICT: %s" % (
        "the switch CANNOT default on (%s)" % ", ".join(blocked) if blocked
        else "no affected channel blocks the default by this rule"))
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, ARGS.out + ".txt"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    if ARGS.report:
        report()
    else:
        if ARGS.map is None:
            raise SystemExit("--map 0|1 or --report")
        run(ARGS.map)
