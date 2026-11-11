"""[STAGE 6 / R1] The attachment census: how many local features have ONE
unambiguous place on an extended boundary?

Runs `perception.attachment.attach` (the function the binder writer calls) on
the detections of each image, taken from a trace of the real front end, with
the position map completed from the lens tier.  No graph is changed; nothing
is scored.  Two rules side by side: distance alone (the first rule), and
distance with tangent agreement (Amendment 2.3).

    python diagnostics/phase6_attachment_census.py [--workers 2] [--out NAME]
                                                   [--split eval|heldout]
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
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import eth80_frontend as fe  # noqa: E402
import eth80_split as sp  # noqa: E402

OUTCOMES = ("known", "unambiguous", "by_tangent", "ambiguous", "no_host")


def detections_with_nodes(tr):
    """[(detection, node id)] for EVERY detection of the step, canonical order.

    The binder registers one detection per pattern node; the node of a pooled
    detection is found through its pattern (lens kind, signature hash)."""
    from core.hashing import _detection_sort_key, _node_signature_hash
    pat = {}
    for d, nid in list(tr["v1_node_ids"].items()) + list(tr["higher_node_ids"].items()):
        pat.setdefault((d.k, _node_signature_hash(d.sigma)), nid)
    dets = list(tr["D_v1"]) + [d for v in tr["tier_dets"].values() for d in v]
    dets = sorted(set(dets), key=_detection_sort_key)
    return [(d, pat.get((d.k, _node_signature_hash(d.sigma)))) for d in dets]


def one(entry):
    import parameters as P
    from perception import attachment as A
    cls, rel = entry
    tr = {}
    fe.build_graph(fe._load_rgb(fe.image_path(cls, rel)), name=os.path.basename(rel), trace=tr)
    positions = dict(tr["anchor_positions"])
    positions.update(tr["step_positions"])
    dn = detections_with_nodes(tr)
    hosts, feats = A.hosts_and_features(dn, positions, int(P.ORIENTATION_N_THETA))
    scale = float(tr["extraction_scale"])
    _old, stats_dist = A.attach(feats, hosts, scale, use_tangent=False)
    atts, stats = A.attach(feats, hosts, scale)
    frag = [d for d, _n in dn if d.k == "ContourFragment"]
    row = {"class": cls, "image": os.path.basename(rel), "scale": scale,
           "hosts": dict(Counter(h.kind for h in hosts)),
           "host_closed": dict(Counter(h.kind for h in hosts if h.closed)),
           "features": dict(Counter(f.kind for f in feats)),
           "features_with_tangent": dict(Counter(f.kind for f in feats if f.tangent is not None)),
           "fragments": len(frag),
           "fragments_carrying_place": sum(1 for d in frag if getattr(d, "_frag_mid", None) is not None),
           "stats": {"%s|%s" % k: v for k, v in stats.items()},
           "stats_distance_only": {"%s|%s" % k: v for k, v in stats_dist.items()},
           "atts": [(a.feature.kind, a.host.kind,
                     "known" if a.known and not a.by_tangent else
                     ("by_tangent" if a.by_tangent else "unambiguous"),
                     round(a.arc_pos, 4),
                     None if a.arc_end_dist is None else round(a.arc_end_dist, 4),
                     round(a.perp_dist, 4),
                     None if a.tangent_angle is None else round(a.tangent_angle, 4),
                     a.feature.node, a.host.node) for a in atts]}
    any_att = {(a.feature.kind, a.feature.key) for a in atts}
    row["attached_any"] = dict(Counter(k for k, _key in any_att))
    return row


def _pct(v, q):
    return float(np.percentile(np.asarray(v, dtype=float), q)) if len(v) else float("nan")


def _bin8(x, lo, hi):
    return min(7, max(0, int((x - lo) / (hi - lo) * 8)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="phase6_attachment")
    ap.add_argument("--split", default="eval")
    a = ap.parse_args(argv)
    split = sp.load_split()
    entries = list(split["references"].items()) + [
        (c, p) for c, ps in split[a.split].items() for p in ps]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(one, entries, chunksize=1)
    else:
        rows = [one(e) for e in entries]

    from perception import attachment as A
    L = ["[STAGE 6 / R1] attachment census, %d graphs (4 references + %d %s)" % (
        len(rows), len(rows) - 4, a.split.upper()),
         "admission radius %.2f scales, ambiguity margin %.2f scales; tangent agreement: parallel "
         "within %.0f degrees, the next contender at least %.0f degrees less so" % (
             A.D_MAX, A.AMBIGUITY_MARGIN, math.degrees(A.PARALLEL_MAX),
             math.degrees(A.PARALLEL_MARGIN)), ""]
    hosts, closed = Counter(), Counter()
    for r in rows:
        hosts.update(r["hosts"])
        closed.update(r["host_closed"])
    L.append("hosts: " + ", ".join("%s %d (%d closed)" % (k, v, closed.get(k, 0))
                                   for k, v in sorted(hosts.items())))
    nfrag = sum(r["fragments"] for r in rows)
    L.append("ContourFragment detections carrying their own midpoint and chord direction: %d of %d" % (
        sum(r["fragments_carrying_place"] for r in rows), nfrag))
    ft, fwt = Counter(), Counter()
    for r in rows:
        ft.update(r["features"])
        fwt.update(r["features_with_tangent"])
    L.append("features with an orientation of their own: " + ", ".join(
        "%s %d/%d" % (k, fwt.get(k, 0), v) for k, v in sorted(ft.items())))

    tot, tot0 = defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        for k, v in r["stats"].items():
            tot[k].update(v)
        for k, v in r["stats_distance_only"].items():
            tot0[k].update(v)
    L += ["", "A. outcome per feature and host kind (detections; one attempt per feature per host kind)",
          "   feature          host         features   known  unambig.  by tangent  ambiguous  no host   attached"]
    S = Counter()
    for key in sorted(tot):
        fk, hk = key.split("|")
        c = tot[key]
        n = sum(c.values())
        S.update(c)
        att = c["known"] + c["unambiguous"] + c["by_tangent"]
        L.append("   %-16s %-11s %8d  %6d  %8d  %10d  %9d  %7d   %6.3f" % (
            fk, hk, n, c["known"], c["unambiguous"], c["by_tangent"], c["ambiguous"],
            c["no_host"], att / max(n, 1)))
    n = sum(S.values())
    L.append("   ALL                          %8d  %6d  %8d  %10d  %9d  %7d   %6.3f" % (
        n, S["known"], S["unambiguous"], S["by_tangent"], S["ambiguous"], S["no_host"],
        (S["known"] + S["unambiguous"] + S["by_tangent"]) / max(n, 1)))

    L += ["", "A2. what tangent agreement recovers (ambiguous under distance alone -> by tangent)",
          "   feature          host         ambiguous before   by tangent   still ambiguous   recovered"]
    R0 = R1 = 0
    for key in sorted(tot):
        fk, hk = key.split("|")
        before = tot0[key]["ambiguous"]
        got = tot[key]["by_tangent"]
        R0 += before
        R1 += got
        L.append("   %-16s %-11s %10d       %8d      %10d        %.3f" % (
            fk, hk, before, got, tot[key]["ambiguous"], got / max(before, 1)))
    L.append("   ALL                          %10d       %8d      %10d        %.3f" % (
        R0, R1, R0 - R1, R1 / max(R0, 1)))
    S0 = Counter()
    for c in tot0.values():
        S0.update(c)
    geo0 = S0["unambiguous"] + S0["ambiguous"]
    geo1 = S["unambiguous"] + S["by_tangent"] + S["ambiguous"]
    L.append("   geometric candidates ambiguous: distance alone %d of %d (%.3f); with tangent agreement %d of %d (%.3f)" % (
        S0["ambiguous"], geo0, S0["ambiguous"] / max(geo0, 1), S["ambiguous"], geo1,
        S["ambiguous"] / max(geo1, 1)))

    feats, any_att = Counter(), Counter()
    for r in rows:
        feats.update(r["features"])
        any_att.update(r["attached_any"])
    L += ["", "B. features attached to at least one host of either kind",
          "   feature          features   attached   share"]
    for k in sorted(feats):
        L.append("   %-16s %8d  %8d    %.3f" % (k, feats[k], any_att.get(k, 0),
                                                 any_att.get(k, 0) / max(feats[k], 1)))
    L.append("   all              %8d  %8d    %.3f" % (
        sum(feats.values()), sum(any_att.values()),
        sum(any_att.values()) / max(sum(feats.values()), 1)))

    atts = [t for r in rows for t in r["atts"]]
    L += ["", "C. the measured axes (attached pairs; median / p10 / p90 and the share ABSENT)",
          "   feature          host         how          n       arc_end_dist            perp_dist               tangent_angle (deg)"]
    groups = defaultdict(list)
    for t in atts:
        groups[(t[0], t[1], t[2])].append(t)
    for key in sorted(groups):
        g = groups[key]

        def col(idx, mult=1.0):
            v = [t[idx] * mult for t in g if t[idx] is not None]
            if not v:
                return "ABSENT (all)          "
            return "%5.2f %5.2f %5.2f (%2.0f%%)" % (_pct(v, 50), _pct(v, 10), _pct(v, 90),
                                                 100.0 * (1 - len(v) / len(g)))
        L.append("   %-16s %-11s %-11s %6d   %s   %s   %s" % (
            key[0], key[1], key[2], len(g), col(4), col(5), col(6, 180.0 / math.pi)))

    SPEC = {"arc_end_dist": (4, 0.0, 0.5), "perp_dist": (5, 0.0, A.D_MAX),
            "tangent_angle": (6, 0.0, math.pi / 2), "ordinal": (3, 0.0, 1.0)}
    pairs = defaultdict(list)
    for ri, r in enumerate(rows):
        for t in r["atts"]:
            if t[7] is not None and t[8] is not None and t[7] != t[8]:
                pairs[(ri, t[0], t[1], t[8], t[7])].append(t)
    E = defaultdict(Counter)
    for (ri, fk, hk, _hn, _fn), g in pairs.items():
        e = E[(fk, hk)]
        e["edges"] += 1
        e["pooled"] += int(len(g) > 1)
        for name, (idx, lo, hi) in SPEC.items():
            vals = [t[idx] for t in g if t[idx] is not None]
            if not vals:
                e[name + "_never"] += 1
                continue
            b = [_bin8(v, lo, hi) for v in vals]
            e[name + ("_lost" if max(b) - min(b) > 1 else "_measured")] += 1
    L += ["", "D. edges (one per host node and feature node), the pooling rule applied",
          "   feature          host          edges  pooled | tangent: measured / never / lost | arc_end_dist: measured / never / lost | perp lost | ordinal lost"]
    T = Counter()
    for key in sorted(E):
        e = E[key]
        T.update(e)
        L.append("   %-16s %-11s %6d %6d | %6d (%.3f) / %5d / %4d      | %6d (%.3f) / %4d / %4d          | %5d     | %5d" % (
            key[0], key[1], e["edges"], e["pooled"],
            e["tangent_angle_measured"], e["tangent_angle_measured"] / e["edges"],
            e["tangent_angle_never"], e["tangent_angle_lost"],
            e["arc_end_dist_measured"], e["arc_end_dist_measured"] / e["edges"],
            e["arc_end_dist_never"], e["arc_end_dist_lost"], e["perp_dist_lost"], e["ordinal_lost"]))
    L.append("   ALL                          %6d %6d | %6d (%.3f) / %5d / %4d      | %6d (%.3f) / %4d / %4d          | %5d     | %5d" % (
        T["edges"], T["pooled"], T["tangent_angle_measured"],
        T["tangent_angle_measured"] / max(T["edges"], 1), T["tangent_angle_never"],
        T["tangent_angle_lost"], T["arc_end_dist_measured"],
        T["arc_end_dist_measured"] / max(T["edges"], 1), T["arc_end_dist_never"],
        T["arc_end_dist_lost"], T["perp_dist_lost"], T["ordinal_lost"]))

    L += ["", "E. the ordinal (arc position), attachments: share per decile 0-0.1 ... 0.9-1.0; exactly 0; exactly 1"]
    H = defaultdict(lambda: [0] * 10)
    Z = defaultdict(Counter)
    for t in atts:
        k = (t[1], "known" if t[2] == "known" else "by geometry")
        H[k][min(9, int(t[3] * 10))] += 1
        Z[k]["n"] += 1
        Z[k]["zero"] += int(t[3] == 0.0)
        Z[k]["one"] += int(t[3] == 1.0)
    for k in sorted(H):
        nn = Z[k]["n"]
        L.append("   %-11s %-12s n %6d | %s | =0 %.3f | =1 %.3f" % (
            k[0], k[1], nn, " ".join("%.2f" % (x / nn) for x in H[k]),
            Z[k]["zero"] / nn, Z[k]["one"] / nn))
    text = "\n".join(L) + "\n"
    print(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS_DIR, a.out + ".json"), "w") as fh:
        json.dump(rows, fh, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
