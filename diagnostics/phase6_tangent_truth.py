"""[STAGE 6 / Amendment 2.2 and 2.6] Tangent CORRECTNESS, against renderer truth.

 1. THE BAR PROBE (the registered test of the EndStopped convention): isolated
    bars at 24 known orientations; for every EndStopped detection within one
    scale of a bar END, the axial difference between the slot's theta and the
    bar's orientation.  TANGENT iff the median is below 22.5 degrees, CARRIER
    iff above 67.5, otherwise undecided.
 2. THE SYNTHETIC SCENES: the renderer knows every boundary.  Truth = the
    visible-label map; a boundary pixel differs from a 4-neighbour; its true
    tangent is perpendicular to the gradient of the blurred (sigma 2 px)
    indicator of its own label; it is SMOOTH when every boundary pixel within
    4 px has a true tangent within 20 degrees of its own.
      a. per feature kind: error of the feature's own orientation;
      b. per host kind: error of the host's tangent at its vertices;
      c. per attachment: error of `tangent_angle` against the angle between
         the two true tangents (at the feature and at its foot).
 3. THE TESTS OF AMENDMENT 3.2:
      a. Corner against the TRUE BISECTOR (a bisector is not a tangent);
      b. EndStopped: is the winning channel the locally dominant channel?
      c. the stream-2 polylines against the true tangent, as traced and with
         the tracer's 90-degree rotation removed.

    python diagnostics/phase6_tangent_truth.py [--out NAME]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import parameters as P  # noqa: E402
import eth80_frontend as fe  # noqa: E402
import orchestration.pipeline as pl  # noqa: E402
from perception import attachment as A  # noqa: E402
from perception.observation_2d import Observation2D  # noqa: E402
from phase6_attachment_census import detections_with_nodes  # noqa: E402

N_THETA = int(P.ORIENTATION_N_THETA)


def axial(a, b):
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


# --------------------------------------------------------------------------
# 1. the bar probe
# --------------------------------------------------------------------------

def bar_image(phi, size=256, length=70.0, width=3.0, fg=215, bg=60):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    cx = cy = (size - 1) / 2.0
    u = (xx - cx) * math.cos(phi) + (yy - cy) * math.sin(phi)
    v = -(xx - cx) * math.sin(phi) + (yy - cy) * math.cos(phi)
    inside = (np.clip(length / 2 - np.abs(u) + 0.5, 0, 1)
              * np.clip(width / 2 - np.abs(v) + 0.5, 0, 1))
    img = (bg + (fg - bg) * inside).astype(np.uint8)
    ends = [(cx + s * (length / 2) * math.cos(phi), cy + s * (length / 2) * math.sin(phi))
            for s in (-1, 1)]
    return np.dstack([img, img, img]), ends


def bar_probe(width=3.0):
    near, every = [], []
    for k in range(24):
        phi = k * math.pi / 24.0
        img, ends = bar_image(phi, width=width)
        obs = Observation2D.from_image(img, rich=False)
        pos = {}
        es = pl.Pipeline._derive_extraction_scale(None, obs)
        dets = pl.EndStoppedLens().step(0, None, None, pos, None, {}, es, observation=obs)
        for d in dets:
            p = pos[d.alpha]
            dist = min(math.hypot(p[0] - e[0], p[1] - e[1]) for e in ends) / es
            diff = math.degrees(axial(float(d.sigma.quantized[0]) * math.pi / N_THETA, phi))
            every.append((diff, dist))
            if dist <= 1.0:
                near.append(diff)
    return near, every


# --------------------------------------------------------------------------
# 2. renderer truth on the synthetic scenes
# --------------------------------------------------------------------------

def label_map(scene):
    names = sorted(scene.masks)
    h, w = scene.shape
    lab = np.full((h, w), -1, dtype=np.int32)
    depth = np.full((h, w), 10 ** 6, dtype=np.int32)
    for i, n in enumerate(names):
        m = np.asarray(scene.masks[n], dtype=bool)
        d = int(scene.depth.get(n, 99))
        take = m & (d < depth)
        lab[take] = i
        depth[take] = d
    return lab


class Truth:
    """Boundary pixels of the label map with their true tangent."""

    def __init__(self, scene):
        lab = label_map(scene)
        diff = np.zeros(lab.shape, dtype=bool)
        diff[:-1, :] |= lab[:-1, :] != lab[1:, :]
        diff[1:, :] |= lab[:-1, :] != lab[1:, :]
        diff[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        diff[:, 1:] |= lab[:, :-1] != lab[:, 1:]
        ys, xs = np.nonzero(diff)
        tang = np.zeros(len(ys))
        for v in np.unique(lab[ys, xs]):
            ind = ndimage.gaussian_filter((lab == v).astype(np.float64), sigma=2.0)
            gy, gx = np.gradient(ind)
            sel = lab[ys, xs] == v
            tang[sel] = (np.arctan2(gy[ys[sel], xs[sel]], gx[ys[sel], xs[sel]])
                         + math.pi / 2.0) % math.pi
        self.xy = np.column_stack([xs, ys]).astype(np.float64)
        self.tang = tang
        self.tree = cKDTree(self.xy) if len(ys) else None
        smooth = np.ones(len(ys), dtype=bool)
        if self.tree is not None:
            for i, nb in enumerate(self.tree.query_ball_point(self.xy, r=4.0)):
                d = np.abs(tang[nb] - tang[i]) % math.pi
                d = np.minimum(d, math.pi - d)
                smooth[i] = bool(d.max() <= math.radians(20.0))
        self.smooth = smooth

    def at(self, pos, radius):
        """(true tangent, smooth) at the boundary pixel nearest `pos`, or None."""
        if self.tree is None:
            return None
        d, i = self.tree.query([float(pos[0]), float(pos[1])])
        if d > radius:
            return None
        return float(self.tang[i]), bool(self.smooth[i])


def true_bisector(truth, pos, scale):
    """(true bisector, axial) at the true corner nearest `pos`, or None.

    Registered in Amendment 3.2: the corner is the centroid of the non-smooth
    boundary pixels within 4 px of the non-smooth pixel nearest the detection
    (within one scale); the bisector is the direction of the mean unit vector
    from it to the boundary pixels 0.75-1.5 scales away."""
    if truth.tree is None:
        return None
    near = [i for i in truth.tree.query_ball_point([float(pos[0]), float(pos[1])], r=scale)
            if not truth.smooth[i]]
    if not near:
        return None
    d = np.hypot(truth.xy[near, 0] - pos[0], truth.xy[near, 1] - pos[1])
    i0 = near[int(np.argmin(d))]
    cl = [i for i in truth.tree.query_ball_point(truth.xy[i0], r=4.0) if not truth.smooth[i]]
    c = truth.xy[cl].mean(axis=0)
    ring = truth.tree.query_ball_point(c, r=1.5 * scale)
    v = truth.xy[ring] - c
    r = np.hypot(v[:, 0], v[:, 1])
    keep = r >= 0.75 * scale
    if int(keep.sum()) < 6:
        return None
    u = (v[keep] / r[keep, None]).mean(axis=0)
    if float(np.hypot(u[0], u[1])) < 0.2:
        return None
    return math.atan2(float(u[1]), float(u[0])) % math.pi


def stream2_rows(obs, truth, scale):
    """Angle between the local direction of every stream-2 polyline and the
    true tangent: as traced, and with the tracer's rotation removed (the
    energy rolled by half the channels: the tracer then follows the channel
    index as it is)."""
    from perception.contour_structure import _extract_association_field_contours
    Eg = getattr(obs, "_gabor_complex", None)
    th = getattr(obs, "_gabor_thetas", None)
    sf = getattr(obs, "_gabor_sfs", None)
    if Eg is None or th is None or sf is None:
        return None
    nt, _nsf, h, w = Eg.shape
    out = {}
    for label, energy in (("as traced", Eg), ("rotation removed", np.roll(Eg, nt // 2, axis=0))):
        cs = _extract_association_field_contours(energy, np.asarray(th), np.asarray(sf), h, w)
        err, lens, closed = [], [], 0
        for c in cs:
            pl = c["polyline"]
            lens.append(float(c.get("length", 0.0)))
            closed += int(bool(c.get("is_closed")))
            for i in range(3, len(pl) - 3, 3):
                a, b = pl[i - 3], pl[i + 3]
                if a == b:
                    continue
                t = truth.at(pl[i], scale)
                if t is None:
                    continue
                err.append(math.degrees(axial(math.atan2(b[1] - a[1], b[0] - a[0]), t[0])))
        out[label] = {"n": len(cs), "len": lens, "closed": closed, "err": err}
    same = len(out["as traced"]["len"]) == len(obs.get_contours())
    out["reproduces_the_observation"] = bool(same)
    return out


def foot_point(h, pos):
    poly = A._Poly(h.pts, h.closed)
    _d, s, i = poly.project(pos)
    t = (s - poly.cum[i]) / poly.len[i] if poly.len[i] > 0.0 else 0.0
    return (float(poly.a[i, 0] + t * poly.seg[i, 0]), float(poly.a[i, 1] + t * poly.seg[i, 1]))


def scene_rows(name, scene):
    img = np.clip(np.asarray(scene.rgb, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    tr = {}
    fe.build_graph(img, name=name, trace=tr)
    positions = dict(tr["anchor_positions"])
    positions.update(tr["step_positions"])
    dn = detections_with_nodes(tr)
    hosts, feats = A.hosts_and_features(dn, positions, N_THETA)
    scale = float(tr["extraction_scale"])
    truth = Truth(scene)
    out = {"feature": [], "feature_alt": [], "host": [], "attach": [],
           "corner": [], "endstop": [], "counts": defaultdict(int)}
    # what each detection ITSELF says (every kind, admitted or not)
    raw, cls_of = {}, {}
    for d, _n in dn:
        if d.k in A.FEATURE_KINDS:
            raw[(d.k, A.feature_key(d))] = A.raw_orientation(d, N_THETA)
            if d.k == "Corner":
                cls_of[A.feature_key(d)] = str(getattr(d.sigma, "class_", "?"))
    obs = tr.get("obs")
    Eg = getattr(obs, "_gabor_complex", None)
    for d, _n in dn:
        if d.k != "EndStopped" or Eg is None:
            continue
        p = positions.get(d.alpha)
        if p is None:
            continue
        nt = Eg.shape[0]
        x = int(min(Eg.shape[3] - 1, max(0, round(p[0]))))
        y = int(min(Eg.shape[2] - 1, max(0, round(p[1]))))
        e = Eg[:, :, y, x].max(axis=1)
        it = int(round(float(d.sigma.quantized[0]) * nt / N_THETA)) % nt
        dom = int(np.argmax(e))
        step = min((it - dom) % nt, (dom - it) % nt)
        t = truth.at(p, scale)
        out["endstop"].append((step * 180.0 / nt, float(e[it] / max(e[dom], 1e-12)),
                               -1 if t is None else int(t[1])))
    for f in feats:
        out["counts"]["features|" + f.kind] += 1
        own = raw.get((f.kind, f.key))
        if f.kind == "Corner" and own is not None:
            tb = true_bisector(truth, f.pos, scale)
            out["counts"]["corner_with_true_bisector"] += int(tb is not None)
            if tb is not None:
                out["corner"].append((math.degrees(axial(own, tb)), cls_of.get(f.key, "?")))
        t = truth.at(f.pos, scale)
        if t is None:
            continue
        out["counts"]["near_boundary|" + f.kind] += 1
        if own is None:
            continue
        out["feature"].append((f.kind, math.degrees(axial(own, t[0])), int(t[1])))
        if f.kind == "EndStopped":      # the reading with a 90-degree turn, for the record
            out["feature_alt"].append((math.degrees(axial(own + math.pi / 2, t[0])), int(t[1])))
    out["stream2"] = stream2_rows(obs, truth, scale)
    for h in hosts:
        poly = A._Poly(h.pts, h.closed)
        for vi, p in enumerate(h.pts):
            t = truth.at(p, scale)
            if t is None:
                continue
            seg = min(vi, len(poly.len) - 1)
            ang = h.tangent if h.tangent is not None else poly.segment_angle(seg)
            if ang is None:
                continue
            out["host"].append((h.kind, math.degrees(axial(ang, t[0])), int(t[1])))
    atts, _stats = A.attach(feats, hosts, scale)
    for a in atts:
        if a.tangent_angle is None:
            continue
        tf = truth.at(a.feature.pos, scale)
        th = truth.at(foot_point(a.host, a.feature.pos), scale)
        if tf is None or th is None:
            continue
        true_angle = axial(tf[0], th[0])
        how = "known" if a.known and not a.by_tangent else ("by_tangent" if a.by_tangent else "unambiguous")
        out["attach"].append((a.feature.kind, a.host.kind, how,
                              math.degrees(a.tangent_angle), math.degrees(true_angle),
                              int(tf[1] and th[1])))
    out["counts"] = dict(out["counts"])
    return out


def _stat(v):
    if not len(v):
        return "n     0"
    a = np.asarray(v, dtype=float)
    return "n %5d  median %5.1f  p90 %5.1f  under 15: %.2f  under 30: %.2f" % (
        len(a), np.median(a), np.percentile(a, 90), np.mean(a < 15.0), np.mean(a < 30.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase6_tangent_truth")
    a = ap.parse_args(argv)
    L = ["[STAGE 6 / Amendment 2] tangent correctness against renderer truth", ""]

    L.append("1. THE BAR PROBE: is the ENDSTOPPED theta slot the bar's orientation?")
    for width in (3.0, 8.0):
        near, every = bar_probe(width)
        n_ = np.asarray(near)
        e_ = np.asarray([d for d, _r in every])
        L.append("   bar width %.0f px: within ONE scale of a bar end (the registered test): n %d, median %.1f deg, "
                 "under 22.5: %.2f, over 67.5: %.2f" % (
                     width, len(n_), np.median(n_) if len(n_) else float("nan"),
                     np.mean(n_ < 22.5) if len(n_) else float("nan"),
                     np.mean(n_ > 67.5) if len(n_) else float("nan")))
        L.append("      every detection of the probe: n %d, median %.1f deg, p90 %.1f, under 22.5: %.2f, over 67.5: %.2f; "
                 "distance to the nearest bar end: median %.2f scales" % (
                     len(e_), np.median(e_), np.percentile(e_, 90), np.mean(e_ < 22.5),
                     np.mean(e_ > 67.5), np.median([r for _d, r in every])))
    near, _every = bar_probe(3.0)
    med = float(np.median(near)) if near else float("nan")
    verdict = "TANGENT" if med < 22.5 else ("CARRIER" if med > 67.5 else "UNDECIDED")
    L.append("   VERDICT (median %.1f deg): %s.  The slot is the terminating bar's orientation; "
             "no 90-degree turn applies." % (med, verdict) if verdict == "TANGENT" else
             "   VERDICT (median %.1f deg): %s" % (med, verdict))

    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tests_figground"))
    from tests_figground.fresh_scenes import build_fresh_all
    from tests_figground.scenes import build_all
    scenes = dict(build_all())
    scenes.update(build_fresh_all())
    feat, feat_alt, host, att, counts = [], [], [], [], defaultdict(int)
    corner, endstop = [], []
    s2 = {"as traced": {"n": 0, "len": [], "closed": 0, "err": []},
          "rotation removed": {"n": 0, "len": [], "closed": 0, "err": []}}
    s2_same = 0
    for name, sc in scenes.items():
        r = scene_rows(name, sc)
        feat += r["feature"]
        feat_alt += r["feature_alt"]
        host += r["host"]
        att += r["attach"]
        corner += r["corner"]
        endstop += r["endstop"]
        if r["stream2"] is not None:
            s2_same += int(r["stream2"]["reproduces_the_observation"])
            for k in s2:
                s2[k]["n"] += r["stream2"][k]["n"]
                s2[k]["closed"] += r["stream2"][k]["closed"]
                s2[k]["len"] += r["stream2"][k]["len"]
                s2[k]["err"] += r["stream2"][k]["err"]
        for k, v in r["counts"].items():
            counts[k] += v
    L += ["", "2. THE SYNTHETIC SCENES (%d scenes; error in degrees against the true boundary tangent" % len(scenes),
          "   at the nearest boundary pixel, for things within one scale of a true boundary)", "",
          "   a. the feature's own orientation",
          "      kind              features  near a boundary   on SMOOTH boundary                                         on any boundary"]
    kinds = sorted({k for k, _e, _s in feat} | {k.split("|")[1] for k in counts if "|" in k})
    for k in kinds:
        sm = [e for kk, e, s in feat if kk == k and s]
        al = [e for kk, e, _s in feat if kk == k]
        L.append("      %-16s %8d  %8d          %s   | %s" % (
            k, counts.get("features|" + k, 0), counts.get("near_boundary|" + k, 0),
            _stat(sm), _stat(al)))
    L.append("      EndStopped + 90   (the rejected reading)        %s   | %s" % (
        _stat([e for e, s in feat_alt if s]), _stat([e for e, _s in feat_alt])))
    L += ["", "   b. the host's tangent at its vertices",
          "      host                                          on SMOOTH boundary                                         on any boundary"]
    for k in sorted({k for k, _e, _s in host}):
        L.append("      %-16s (%s)   %s   | %s" % (
            k, "segment at the vertex" if k == "PEAK_CHAIN" else "its own mean orientation",
            _stat([e for kk, e, s in host if kk == k and s]),
            _stat([e for kk, e, _s in host if kk == k])))
    L += ["", "   c. attachments: |tangent_angle - true angle| where both the feature and its foot are near a boundary",
          "      feature          host         how          on SMOOTH boundary                                         on any boundary"
          "                                        (measured angle, median)"]
    for key in sorted({(t[0], t[1], t[2]) for t in att}):
        g = [t for t in att if (t[0], t[1], t[2]) == key]
        L.append("      %-16s %-11s %-11s  %s   | %s   | %5.1f" % (
            key[0], key[1], key[2],
            _stat([abs(t[3] - t[4]) for t in g if t[5]]),
            _stat([abs(t[3] - t[4]) for t in g]),
            float(np.median([t[3] for t in g]))))
    allsm = [abs(t[3] - t[4]) for t in att if t[5]]
    L.append("      ALL                                        %s   | %s" % (
        _stat(allsm), _stat([abs(t[3] - t[4]) for t in att])))
    L += ["", "3. THE TESTS OF AMENDMENT 3.2", ""]
    L.append("   THE RULE: a kind's orientation is used iff at least half of its detections on smooth boundary lie within 15 degrees of truth.")
    for k in kinds:
        sm = [e for kk, e, s_ in feat if kk == k and s_]
        if not sm:
            continue
        share = float(np.mean(np.asarray(sm) < 15.0))
        L.append("      %-16s %.2f within 15 degrees of the true TANGENT (n %d) -> %s%s" % (
            k, share, len(sm), "USED" if share >= 0.5 else "ABSENT",
            "  (a bisector is not a tangent: judged in 3a)" if k == "Corner" else ""))
    c_ = np.asarray([e for e, _c in corner], dtype=float)
    L += ["", "   a. Corner against the TRUE BISECTOR: %d Corner detections, %d with a true bisector" % (
        counts.get("features|Corner", 0), len(c_))]
    if len(c_):
        share = float(np.mean(c_ < 15.0))
        L.append("      THE REGISTERED TEST (the internal bisector: the mean direction of the two arms): median %.1f deg, "
                 "p90 %.1f, within 15: %.2f, within 30: %.2f -> %s" % (
                     np.median(c_), np.percentile(c_, 90), share, np.mean(c_ < 30.0),
                     "USED" if share >= 0.5 else "ABSENT"))
        L.append("      beside it, the perpendicular reading (the external-angle bisector, which is what the CORNER slot is "
                 "documented to hold; NOT the registered truth): within 15: %.2f, within 30: %.2f" % (
                     np.mean(c_ > 75.0), np.mean(c_ > 60.0)))
        for cls in sorted({c for _e, c in corner}):
            g = np.asarray([e for e, c in corner if c == cls], dtype=float)
            L.append("         class %-9s n %3d: internal within 15: %.2f; external within 15: %.2f; median error to the internal %.1f" % (
                cls, len(g), np.mean(g < 15.0), np.mean(g > 75.0), np.median(g)))
    L += ["", "   b. EndStopped: the winning channel against the locally dominant channel (max energy over frequencies at the detection)"]
    for label, sel in (("on smooth boundary", 1), ("at corners / junctions", 0), ("away from any boundary", -1)):
        g = [(st_, ratio) for st_, ratio, where in endstop if where == sel]
        if not g:
            continue
        st_ = np.asarray([x for x, _r in g])
        ra = np.asarray([r_ for _x, r_ in g])
        L.append("      %-24s n %4d: winning = dominant %.2f; within one channel (15 deg) %.2f; median offset %.0f deg; "
                 "energy of the winning channel / dominant: median %.2f" % (
                     label, len(g), np.mean(st_ == 0), np.mean(st_ <= 15.0 + 1e-9),
                     np.median(st_), np.median(ra)))
    L += ["", "   c. stream-2 polylines (the traces behind the contour regions and the ContourFragments) against the true tangent",
          "      (the as-traced call reproduces the observation's own contour count in %d of %d scenes)" % (
              s2_same, len(scenes))]
    for k in ("as traced", "rotation removed"):
        e = np.asarray(s2[k]["err"], dtype=float)
        ln = np.asarray(s2[k]["len"], dtype=float)
        L.append("      %-17s polylines %4d, closed %3d, length median %5.1f px (p90 %5.1f, total %7.0f); "
                 "local direction vs true tangent: median %.1f deg, p10 %.1f, p90 %.1f, under 22.5: %.2f, over 67.5: %.2f" % (
                     k, s2[k]["n"], s2[k]["closed"], np.median(ln) if len(ln) else float("nan"),
                     np.percentile(ln, 90) if len(ln) else float("nan"), float(ln.sum()),
                     np.median(e), np.percentile(e, 10), np.percentile(e, 90),
                     np.mean(e < 22.5), np.mean(e > 67.5)))
    ea = np.asarray(s2["as traced"]["err"], dtype=float)
    er = np.asarray(s2["rotation removed"]["err"], dtype=float)
    across = bool(len(ea) and len(er) and np.median(ea) > 67.5 and np.median(er) < 22.5)
    L.append("      VERDICT (registered: median as traced above 67.5 AND with the rotation removed below 22.5): %s" % (
        "the stream-2 tracer runs ACROSS edges; the rotation is applied to an index that already is the edge's orientation"
        if across else "NOT established"))
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w") as fh:
        json.dump({"feature": feat, "feature_alt": feat_alt, "host": host, "attach": att,
                   "corner": corner, "endstop": endstop,
                   "stream2": {k: {"n": v["n"], "closed": v["closed"], "len": v["len"],
                                   "err": v["err"]} for k, v in s2.items()},
                   "counts": dict(counts)}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
