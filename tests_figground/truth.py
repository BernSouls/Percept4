"""Truth harness: is an emitted (nearer, farther) order edge CORRECT?

The session's ship gate is "zero wrong-direction edges on TEST", so every test
needs a truth oracle for an edge.  An edge is a pair of FRAGMENT INDICES; each
fragment carries an alpha (a CONTOUR_PEAK anchor) and therefore a position, and
the scene carries a per-pixel object id + ordinal depth.  So:

    object(fragment) = the NEAREST object whose mask contains a point sampled
                       a few pixels around the fragment's position, preferring
                       the nearer object when several overlap (the fragment
                       sits ON a boundary, so both its flanks are candidates
                       and the tie must be broken toward the front object --
                       that is what "the fragment belongs to the occluder's
                       contour" means).

    edge (a, b) is CORRECT   iff depth[obj(a)] <  depth[obj(b)]
         (a is nearer)      or  obj(a) == obj(b) is treated as UNDECIDED
                              (an intra-object edge makes no order claim)

The oracle is deliberately conservative: an edge whose endpoints cannot be
attributed, or whose endpoints belong to the same object, is UNDECIDED, not
wrong.  Only a strict contradiction of the renderer's depth order counts as a
wrong-direction edge.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def object_at(scene, x: float, y: float, radius: float = 3.0,
              prefer: str = "near") -> Optional[str]:
    """Nearest object id at (x, y), searched in a small disk.

    ``prefer='near'`` breaks an overlap toward the smaller depth ordinal (the
    occluder), ``prefer='far'`` toward the larger.  Returns None when the disk
    holds only background.
    """
    h, w = scene.raw_luminance.shape
    best: Optional[Tuple[int, str]] = None
    r = int(math.ceil(radius))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            xi, yi = int(round(x)) + dx, int(round(y)) + dy
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            for name, mask in scene.masks.items():
                if mask[yi, xi]:
                    d = int(scene.depth.get(name, 99))
                    key = (d, name) if prefer == "near" else (-d, name)
                    if best is None or key < best[0]:
                        best = (key, name)
    return None if best is None else best[1]


def fragment_object(scene, frag, radius: float = 3.0,
                    inset: float = 4.0) -> Optional[str]:
    """The object whose CONTOUR this fragment lies on.

    A fragment sits ON a boundary, so a disk around its position contains both
    objects and the attribution would be a coin flip.  When the fragment HAS an
    ownership verdict it carries the answer: ``owner_normal`` points to the owner
    side, so the object is sampled `inset` pixels along it (the owner's
    interior).  Otherwise the oracle falls back to the position disk, whose
    documented tie-break (prefer the nearer object at a boundary pixel) is just
    the renderer's own painter's-order semantics.

    [F9] The owner-normal inset is a verdict only when the fragment has one.
    ``owner_side == 0.0`` means the ownership mechanism CANCELLED -- both sides
    equally supported, which is exactly what happens at an occlusion junction,
    where one contour point belongs to two objects at once.  Following an
    undecided normal 4 px is arbitrary, and measured it is arbitrary in the
    worst possible direction: at 3 of the 5 box-over-box junctions in the corpus
    the CONTACT fragment has ``owner_side == 0.0`` and a normal pointing AWAY
    from the occluder, so the inset samples the FAR object and the oracle reports
    bar and terminator as the same object::

        box_over_box       J_left  contact frag#98  side=-1.000 n=(+1,0)   -> near_box
        box_over_box_test  J_left  contact frag#105 side=+0.000 n=(-1,-.08)-> far_box
        box_over_box_test  J_right contact frag#151 side=+0.000 n=(+1,-.09)-> far_box
        box_over_box_haze  J_left  contact frag#70  side=-1.000 n=(+1,0)   -> near_box
        box_over_box_haze  J_right contact frag#136 side=+0.000 n=(+1,-.08)-> far_box

    (``_frag_attribution.py``; the near box's left edge is at x=76 and its right
    edge at x=120, so a 4 px inset along those normals lands outside it.)  The
    cue had found the junction and emitted the right direction; the instrument
    could not score it.  Requiring a verdict before trusting the normal fixes
    the instrument without touching the cue -- it cannot make a wrong edge look
    right, because the fallback prefers the nearer object for BOTH endpoints
    symmetrically: where both fragments see the same object set they still
    collapse to same-object/undecided.
    """
    on = getattr(frag, "owner_normal", None)
    if on is not None and float(getattr(frag, "owner_side", 0.0) or 0.0) != 0.0:
        ox, oy = float(on[0]), float(on[1])
        m = math.hypot(ox, oy)
        if m > 1e-9:
            ox, oy = ox / m, oy / m
            got = object_at(scene, frag.position[0] + ox * inset,
                            frag.position[1] + oy * inset, 1.5)
            if got is not None:
                return got
    return object_at(scene, frag.position[0], frag.position[1], radius)


def classify_edges(scene, frags: Sequence[Any],
                   edges: Sequence[Tuple[int, int]],
                   radius: float = 3.0) -> Dict[str, Any]:
    """Split emitted (nearer, farther) edges into correct / wrong / undecided."""
    by_index = {int(f.index): f for f in frags}
    correct: List[Tuple[int, int, str, str]] = []
    wrong: List[Tuple[int, int, str, str]] = []
    undecided: List[Tuple[int, int, Any, Any]] = []
    for a, b in edges:
        fa, fb = by_index.get(int(a)), by_index.get(int(b))
        if fa is None or fb is None:
            undecided.append((a, b, None, None))
            continue
        oa = fragment_object(scene, fa, radius)
        ob = fragment_object(scene, fb, radius)
        if oa is None or ob is None:
            undecided.append((a, b, oa, ob))
            continue
        da, db = int(scene.depth[oa]), int(scene.depth[ob])
        if oa == ob or da == db:
            undecided.append((a, b, oa, ob))
        elif da < db:
            correct.append((a, b, oa, ob))
        else:
            wrong.append((a, b, oa, ob))
    n = len(correct) + len(wrong)
    return dict(correct=correct, wrong=wrong, undecided=undecided,
                n_decided=n,
                precision=(len(correct) / n) if n else None,
                emitted=len(edges))


def classify_alpha_edges(scene, run, alpha_edges: Sequence[Tuple[Any, Any]],
                         radius: float = 3.0) -> Dict[str, Any]:
    """Same classification for ALPHA edges (t_junction_arms' output)."""
    by_alpha = {f.alpha: f for f in run.fragments}
    pairs = []
    for na, fa in alpha_edges:
        a, b = by_alpha.get(na), by_alpha.get(fa)
        if a is None or b is None:
            continue
        pairs.append((a.index, b.index))
    out = classify_edges(scene, run.fragments, pairs, radius)
    out["unrepresentable"] = len(alpha_edges) - len(pairs)
    return out


def layer_correctness(scene, run, radius: float = 3.0) -> Dict[str, Any]:
    """For every weakly-layered component: is the layer order consistent with
    the renderer's depth order?  Reports per-component verdicts."""
    frags = [f for f in run.fragments if run.layers.get(f.index) is not None]
    comps: Dict[int, List[Any]] = {}
    # rebuild weak components from the emitted edges
    edges = list(run.order.get("occlusion", [])) + list(
        run.order.get("extra_edges", []))
    parent: Dict[int, int] = {}

    def find(i):
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    by_index = {int(f.index): f for f in frags}
    for f in frags:
        comps.setdefault(find(int(f.index)), []).append(f)
    rows = []
    for root, members in sorted(comps.items()):
        if len(members) < 2:
            continue
        objs = [(f, fragment_object(scene, f, radius)) for f in members]
        objs = [(f, o) for f, o in objs if o is not None]
        ok = bad = und = 0
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                (fa, oa), (fb, ob) = objs[i], objs[j]
                la, lb = run.layers[int(fa.index)], run.layers[int(fb.index)]
                da, db = scene.depth[oa], scene.depth[ob]
                if oa == ob or da == db or la == lb:
                    und += 1
                    continue
                # a LARGER layer must mean FARTHER
                if (la < lb) == (da < db):
                    ok += 1
                else:
                    bad += 1
        rows.append(dict(component=root, n=len(members),
                         objects=sorted({o for _f, o in objs}),
                         consistent=ok, inconsistent=bad, undecided=und,
                         verdict=("correct" if bad == 0 and ok > 0
                                  else ("wrong" if bad else "undecided"))))
    return dict(components=rows,
                n_components=len(rows),
                n_correct=sum(1 for r in rows if r["verdict"] == "correct"),
                n_wrong=sum(1 for r in rows if r["verdict"] == "wrong"),
                abstained=len([f for f in run.fragments
                               if run.layers.get(f.index) is None]))
