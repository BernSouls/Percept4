"""U13.0 -- THE SEALED SET.

Why this module exists
----------------------
Session U13 promotes BOTH previous corpora to development sets: the old 8 scenes
were inspected while R9/R10 were derived, and the U12 fresh set (`cb11bea2...`)
was read once in U12.4 and then used again to kill H2, so its junction
statistics are no longer unread.  A final verdict needs a set that has never
been scored, so this module renders one and seals it.

Contract
--------
* Same four families and the same parameter RANGES as the fresh set, drawn from
  NEW SEEDS (`SEEDS` below, 2301..2310, disjoint from 1201..1210).
* The haze/blur renderings are inside the fresh set's range but equal to none of
  its values, so no sealed scene is a re-render of a development scene.
* `freeze_record()` writes renderer INPUTS and renderer TRUTH ONLY -- no peaks,
  chains, sites, edges or any other pipeline output -- and its sha256 is copied
  into `test_sealed_scenes.py::SEALED_SHA256`.
* The renderer self-consistency gate (`test_renderer_truth.py`, the B37 lesson)
  runs over this corpus from the moment it is sealed: a stimulus that does not
  draw what its truth says is invalid, and finding that out AFTER the final
  measurement would void the verdict.
* NO CUE STATISTIC ON THESE SCENES IS READ UNTIL THE FINAL MEASUREMENT.  A
  diagnostic that walks the corpus must skip it; `_u13_*.py` scripts take their
  scene lists from `scenes`/`fresh_scenes` only.

Truth is analytic, from the geometry the renderer drew with (see `fresh_scenes`
for the derivation): a box crossing a straight top edge at `y = y0` gives
contacts at `x = nx0`, `nx1`; a disk gives `cx -/+ sqrt(r^2 - (y0-cy)^2)`; a
cylinder's limb gives `cx -/+ rx`.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Dict, Tuple

import numpy as np

from scenes import SIZE
from fresh_scenes import (fresh_box_over_box, fresh_cylinder_over_box,
                          fresh_disk_on_box, fresh_single_box,
                          fresh_single_disk)

#: the seeds every sealed geometry is drawn from.  Disjoint from the fresh set's
#: 1201..1210 by construction; recorded in the freeze file so a rebuild is
#: bit-identical.  Changing one of these invalidates the seal.
SEEDS = dict(
    s_box_a=2301, s_box_b=2302,
    s_haze_a=2303, s_haze_b=2304,
    s_disk_a=2305, s_disk_b=2306,
    s_cyl_a=2307, s_cyl_b=2308,
    s_ctrl_disk=2309, s_ctrl_box=2310,
)

#: the rendering variants, chosen BEFORE any measurement and inside (but not
#: equal to) the fresh set's ranges: fresh used haze 0.42/blur 1.1, haze
#: 0.55/blur 1.6 and cylinder haze 0.35.
HAZE_A = dict(haze_far=0.30, blur_far=0.9)
HAZE_B = dict(haze_far=0.62, blur_far=2.0)
CYL_B_HAZE = 0.45


def _seal(sc, name: str):
    """Re-label a scene built by a fresh_scenes builder as SEALED.

    The builders write "FRESH held-out" into `note`, which is the U12 set's
    label; the seal is a different contract (never scored until the final
    measurement), so the note says which one it is.  Geometry, masks, truth and
    params are untouched.
    """
    sc.name = name
    sc.note = "SEALED (U13.0) " + str(sc.note).replace("FRESH held-out ", "")
    return sc


BUILDERS: Dict[str, Callable[[], Any]] = {
    "s_box_a":   lambda: _seal(fresh_box_over_box(
        SEEDS["s_box_a"], "s_box_a"), "s_box_a"),
    "s_box_b":   lambda: _seal(fresh_box_over_box(
        SEEDS["s_box_b"], "s_box_b"), "s_box_b"),
    "s_haze_a":  lambda: _seal(fresh_box_over_box(
        SEEDS["s_haze_a"], "s_haze_a", **HAZE_A), "s_haze_a"),
    "s_haze_b":  lambda: _seal(fresh_box_over_box(
        SEEDS["s_haze_b"], "s_haze_b", **HAZE_B), "s_haze_b"),
    "s_disk_a":  lambda: _seal(fresh_disk_on_box(
        SEEDS["s_disk_a"], "s_disk_a"), "s_disk_a"),
    "s_disk_b":  lambda: _seal(fresh_disk_on_box(
        SEEDS["s_disk_b"], "s_disk_b"), "s_disk_b"),
    "s_cyl_a":   lambda: _seal(fresh_cylinder_over_box(
        SEEDS["s_cyl_a"], "s_cyl_a"), "s_cyl_a"),
    "s_cyl_b":   lambda: _seal(fresh_cylinder_over_box(
        SEEDS["s_cyl_b"], "s_cyl_b", haze_far=CYL_B_HAZE), "s_cyl_b"),
    "s_ctrl_disk": lambda: _seal(fresh_single_disk(
        SEEDS["s_ctrl_disk"], "s_ctrl_disk"), "s_ctrl_disk"),
    "s_ctrl_box": lambda: _seal(fresh_single_box(
        SEEDS["s_ctrl_box"], "s_ctrl_box"), "s_ctrl_box"),
}

SEALED_SCENES: Tuple[str, ...] = tuple(BUILDERS)
SEALED_OCCLUSION: Tuple[str, ...] = tuple(
    n for n in SEALED_SCENES if not n.startswith("s_ctrl"))
SEALED_CONTROL: Tuple[str, ...] = tuple(
    n for n in SEALED_SCENES if n.startswith("s_ctrl"))


def build_sealed(name: str, size: int = SIZE):
    return BUILDERS[name]()


def build_sealed_all(size: int = SIZE) -> Dict[str, Any]:
    return {n: BUILDERS[n]() for n in SEALED_SCENES}


def freeze_record(size: int = SIZE) -> Dict[str, Any]:
    """The sealed record: geometry, rendering and TRUTH ONLY."""
    out: Dict[str, Any] = dict(
        purpose=("U13.0 sealed set: the final verdict is measured here ONCE, at "
                 "the end of the session.  No cue statistic on these scenes was "
                 "read before that measurement."),
        renderer=dict(cylinder_silhouette=(
            "B37 fixed (U12): the silhouette mask is |x - cx| <= rx over the "
            "row band |y - cy| <= ry.  test_renderer_truth.py gates this "
            "corpus too, from the moment of sealing.")),
        seal=dict(
            seeds_disjoint_from_fresh="1201..1210 vs 2301..2310",
            renderings=dict(haze_a=HAZE_A, haze_b=HAZE_B,
                            cyl_b_haze=CYL_B_HAZE),
            dev_corpora=dict(
                old=("cylinder_over_box, box_over_box, box_over_box_haze, "
                     "disk_on_box, box_over_box_test, cylinder_over_box_shift, "
                     "single_disk, single_box"),
                fresh_sha256=("cb11bea2c7be88dc83fdae86d8f6b423844b55a0db45b"
                              "ccf5d474a7f9df56504")),
        ),
        size=size, seeds=dict(SEEDS), scenes={},
    )
    total = 0
    for n in SEALED_SCENES:
        sc = BUILDERS[n]()
        prm = {k: (list(v) if isinstance(v, tuple) else v)
               for k, v in getattr(sc, "params", {}).items()}
        lum = np.asarray(sc.raw_luminance, dtype=np.float64)
        out["scenes"][n] = dict(
            family=prm.get("family"), seed=prm.get("seed"), params=prm,
            n_true_junctions=len(sc.junctions),
            junctions=[dict(id=j["id"], x=round(float(j["x"]), 4),
                            y=round(float(j["y"]), 4), near=j["near"],
                            far=j["far"]) for j in sc.junctions],
            objects=sorted(sc.masks), depth=dict(sc.depth),
            img_min=round(float(lum.min()), 6),
            img_max=round(float(lum.max()), 6),
            img_mean=round(float(lum.mean()), 6),
            note=sc.note,
        )
        total += len(sc.junctions)
    out["n_scenes"] = len(SEALED_SCENES)
    out["n_occlusion_scenes"] = len(SEALED_OCCLUSION)
    out["n_control_scenes"] = len(SEALED_CONTROL)
    out["n_true_junctions"] = total
    body = json.dumps(out, sort_keys=True, default=str)
    out["_sha256_of_record"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return out


if __name__ == "__main__":
    import os
    import sys
    rec = freeze_record()
    dest = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "diagnostics",
        "_u13_sealed_scenes.json"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True, default=str)
    print("SEALED %d scenes, %d true junctions -> %s"
          % (rec["n_scenes"], rec["n_true_junctions"], dest))
    print("sha256(record) = %s" % rec["_sha256_of_record"])
    for n in SEALED_SCENES:
        s = rec["scenes"][n]
        print("  %-12s %-18s seed=%-5d junctions=%d  %s"
              % (n, s["family"], s["seed"], s["n_true_junctions"],
                 [(j["id"], j["x"], j["y"]) for j in s["junctions"]]))
    sys.exit(0)
