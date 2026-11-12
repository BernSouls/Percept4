"""[STAGE 7 / PHASE 1] THE FRAGMENT TANGENT: measured before and after the fix.

The claim (psychophysics appendix): the stream-2 tracer's `tangent_rad` is the
BOUNDARY TANGENT.  The code computes it as `carrier_idx * pi / nt + pi / 2`
("the Gabor carrier is normal"), and the measured error of everything
downstream is a median 90.0 deg -- the error of the NORMAL.  Phase 1 measures
which reading the bank actually obeys and reports the consequence for the
attachments.

Three measurements, all against renderer truth, on the 18 synthetic scenes:

  1. THE CARRIER.  The angle of the dominant bank channel at every pixel near a
     true boundary, against the true TANGENT, and with the +90 deg the code
     applies.  This decides the sign of the fix without any matcher.
  2. THE FRAGMENT.  `ContourFragment`'s own tangent (what the attachment reads)
     against the true boundary tangent, on smooth boundary: the same test that
     put ContourFragment at 0.00 in phase6_tangent_truth.
  3. THE AMBIGUITY.  `perception.attachment.attach` over every detection of the
     scene: the share of features that end AMBIGUOUS (two near-equal hosts,
     nothing written) and the share decided by tangent agreement.

The pre-registered expectations and the falsifier are in the pre-registration,
7.30.  Run it twice: on the tree before the fix and after it.

    python diagnostics/phase7_p1_tangent.py --out phase7_p1_tangent_before
    python diagnostics/phase7_p1_tangent.py --out phase7_p1_tangent
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
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import phase6_tangent_truth as TT  # noqa: E402
from perception.attachment import AMBIGUOUS, BY_TANGENT  # noqa: E402
import phase6_walker_synthetic as H  # noqa: E402


def axial(a: float, b: float) -> float:
    d = abs((a % math.pi) - (b % math.pi))
    return min(d, math.pi - d)


def carrier_rows(scene, obs, truth, scale, nt):
    """(angle of the dominant channel vs true tangent, +90 vs true tangent)."""
    Eg = getattr(obs, "_gabor_complex", None)
    if Eg is None:
        return [], []
    e = Eg.max(axis=1)
    st = e.max(axis=0)
    ci = np.argmax(e, axis=0)
    h, w = st.shape
    raw, turned = [], []
    for y in range(4, h - 4, 3):
        for x in range(4, w - 4, 3):
            t = truth.at((x, y), scale)
            # registered (7.30): the measurement is on SMOOTH boundary, the same
            # restriction phase6_tangent_truth applied; a corner pixel's normal
            # is not defined and would only add 0-90 deg noise to every arm.
            if t is None or not t[1]:
                continue
            a = float(ci[y, x]) * (math.pi / float(nt))
            raw.append(math.degrees(axial(a, t[0])))
            turned.append(math.degrees(axial(a + math.pi / 2.0, t[0])))
    return raw, turned


def stream2_row(scene, obs, truth, scale):
    """The association-field polylines' own local direction vs the true tangent."""
    from perception.contour_structure import _extract_association_field_contours

    Eg = getattr(obs, "_gabor_complex", None)
    if Eg is None:
        return []
    th, sf = name = getattr(obs, "_gabor_thetas"), getattr(obs, "_gabor_sfs")
    nt, _nsf, h, w = Eg.shape
    cs = _extract_association_field_contours(Eg, np.asarray(th), np.asarray(sf), h, w)
    err = []
    for c in cs:
        pl = c["polyline"]
        for i in range(3, len(pl) - 3, 3):
            a, b = pl[i - 3], pl[i + 3]
            if a == b:
                continue
            t = truth.at(pl[i], scale)
            if t is None:
                continue
            err.append(math.degrees(axial(math.atan2(b[1] - a[1], b[0] - a[0]), t[0])))
    return err


def scene_row(name):
    import parameters as P
    import eth80_frontend as fe
    from perception import attachment as A

    sc = H.SCENES[name]
    img = np.clip(np.asarray(sc.rgb, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    tr = {}
    fe.build_graph(img, name=name, trace=tr)
    obs = tr.get("obs")
    truth = TT.Truth(sc)
    scale = float(tr["extraction_scale"])
    nt = int(P.ORIENTATION_N_THETA)

    positions = dict(tr["anchor_positions"])
    positions.update(tr["step_positions"])
    from phase6_attachment_census import detections_with_nodes
    dn = detections_with_nodes(tr)
    hosts, feats = A.hosts_and_features(dn, positions, nt)
    atts, stats = A.attach(feats, hosts, scale)

    raw, turned = carrier_rows(sc, obs, truth, scale, nt)
    s2 = stream2_row(sc, obs, truth, scale)

    # THE FRAGMENT, measured on the DETECTION (no admission filter: the kind is
    # ABSENT by rule, and the point of the phase is what it would carry).
    # `frag_raw`   what `A.raw_orientation` returns for the detection
    # `frag_chord` the fragment's own mid, straight from the polar fragments
    frag_raw, frag_chord = [], []
    from perception.polar import polar_boundary_fragments
    for d, _n in dn:
        if d.k != "ContourFragment":
            continue
        own = A.raw_orientation(d, nt)
        mid = getattr(d, "_frag_mid", None)
        if mid is None or own is None:
            continue
        t = truth.at(mid, scale)
        if t is None or not t[1]:
            continue
        frag_raw.append(math.degrees(axial(own, t[0])))
    for f in polar_boundary_fragments(obs.get_contours())[0]:
        t = truth.at(f["mid"], scale)
        if t is None or not t[1] or f.get("tangent") is None:
            continue
        frag_chord.append(math.degrees(axial(f["tangent"], t[0])))

    # the census' own statistics, keyed "feature kind|host kind" -> outcome counts
    return {"scene": name, "carrier": raw, "carrier_turned": turned,
            "stream2": s2, "frag_raw": frag_raw, "frag_chord": frag_chord,
            "stats": {"%s|%s" % k: dict(v) for k, v in stats.items()}}


def _stat(v):
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=float)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "under15": float(np.mean(a < 15.0)), "under30": float(np.mean(a < 30.0))}


def main(argv=None) -> int:
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase7_p1_tangent")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    H.SCENES.update(H.scenes())
    names = sorted(H.SCENES)
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(scene_row, names, chunksize=1)
    else:
        rows = [scene_row(n) for n in names]

    def pool(key):
        out = []
        for r in rows:
            out += r[key]
        return out

    L = ["[STAGE 7 / PHASE 1] the fragment tangent, %d synthetic scenes, against renderer truth" % len(rows),
         "   pre-registered (7.30): after the fix, fragment median <= 15 deg and the share",
         "   within 15 deg >= 0.75; the ambiguity share must fall.", ""]

    c, ct = _stat(pool("carrier")), _stat(pool("carrier_turned"))
    per_scene = [(r["scene"], float(np.median(r["carrier"])) if r["carrier"] else float("nan"))
                 for r in rows]
    L += ["1. THE CARRIER -- the angle of the dominant Gabor channel at pixels near a boundary",
          "   per scene, median error of the raw index: " + ", ".join(
              "%s %.0f" % (n, v) for n, v in per_scene),
          "   as the index stands     : n %d  median %6.1f deg  p90 %6.1f  under 20 deg %.2f" % (
              c["n"], c["median"], c["p90"], c.get("under15", 0.0)) if c["n"] else "   (none)",
          "   with the +90 the code adds: n %d  median %6.1f deg  p90 %6.1f  under 20 deg %.2f" % (
              ct["n"], ct["median"], ct["p90"], ct.get("under15", 0.0)) if ct["n"] else "   (none)",
          "   VERDICT: the bank's index is the %s; the tracer's `+pi/2` is the defect." % (
              "TANGENT" if c["n"] and c["median"] < ct["median"] else "NORMAL"),
          ""]

    s2 = _stat(pool("stream2"))
    L += ["2. THE TRACER'S OWN POLYLINES -- local direction of each stream-2 polyline",
          "   n %d  median %.1f deg  p90 %.1f  under 15 deg %.2f  under 30 deg %.2f" % (
              s2["n"], s2["median"], s2["p90"], s2.get("under15", 0.0), s2.get("under30", 0.0))
          if s2["n"] else "   (none)", ""]

    for label, key in (("3. THE FRAGMENT TANGENT the detection carries (`raw_orientation`)", "frag_raw"),
                       ("4. THE FRAGMENT CHORD (`polar_boundary_fragments` -> `_frag_tangent`)", "frag_chord")):
        st = _stat(pool(key))
        L += [label,
              ("   n %d  median %.1f deg  p90 %.1f  under 15 deg %.2f  under 30 deg %.2f" % (
                  st["n"], st["median"], st["p90"], st.get("under15", 0.0), st.get("under30", 0.0)))
              if st["n"] else "   (none)", ""]

    tot = Counter()
    per = {}
    for r in rows:
        d = defaultdict(int)
        for k, v in r["stats"].items():
            if isinstance(v, dict):
                for o, n in v.items():
                    d[o] += int(n)
        per[r["scene"]] = dict(d)
        tot.update(d)
    keys = sorted({k for d in per.values() for k in d})
    L += ["5. THE ATTACHMENTS (`perception.attachment.attach`, the rule in force)", ""]
    head = "   %-34s" % "outcome" + "".join("%10s" % s[:9] for s in names)
    L += [head, "   " + "-" * (34 + 10 * len(names))]
    for k in keys:
        L.append("   %-34s" % k + "".join("%10d" % per[s].get(k, 0) for s in names))
    n_all = sum(sum(d.values()) for d in per.values())
    n_amb = sum(v for d in per.values() for k, v in d.items() if k.split("|")[0] == AMBIGUOUS)
    n_tan = sum(v for d in per.values() for k, v in d.items() if k.split("|")[0] == BY_TANGENT)
    L += ["", "   all outcomes %d; AMBIGUOUS %d = %.3f; decided by TANGENT %d = %.3f" % (
        n_all, n_amb, (n_amb / n_all if n_all else 0.0), n_tan, (n_tan / n_all if n_all else 0.0)), ""]

    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"carrier": c, "carrier_turned": ct, "stream2": s2,
                   "frag_raw": _stat(pool("frag_raw")), "frag_chord": _stat(pool("frag_chord")),
                   "outcomes": per, "totals": dict(tot),
                   "ambiguous_share": (n_amb / n_all if n_all else 0.0),
                   "tangent_share": (n_tan / n_all if n_all else 0.0)}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
