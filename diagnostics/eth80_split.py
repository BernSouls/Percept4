"""ETH-80 split definition for the reconstructed 4-class benchmark.

WHY THIS FILE EXISTS
--------------------
The original 75-query protocol ran on ``data/ETH-80-extended`` with a split
definition that was NOT part of the export (README: "75 queries; refs:
apple1-000-000, car1-000-000, cup_01, horse_03").  Neither the images nor the
split file reached this checkout, so the split below is a RECONSTRUCTION that
mirrors the protocol's *shape*, not its exact images:

  * 4 classes: apple, car, cup, horse;
  * ONE reference image per class -- the alphabetical-first image of the
    class (``<class>1-000-000``), the README's "alphabetical-first reference";
  * 75 queries with the README's per-class counts (apple 16, car 24, cup 18,
    horse 17).

It was written BEFORE any measurement (pre-registered) and is purely
mechanical: no image was looked at, no score was consulted.

  EVAL-75     drawn from object slots 1-5 of each class.  Every phase of the
              gap-closure work is measured on it, and every keep/revert
              decision is taken on it, so every number on this split is an
              IN-SAMPLE ("in-stent") operating point, never a held-out
              estimate.
  HELDOUT-75  same counts, drawn from object slots 6-10 (objects that never
              appear in EVAL-75 or among the references).  Not read until the
              final report, and then scored exactly once per configuration.

Selection is hash-ordered (sha256 of a fixed tag + relative path), so it does
not depend on a PRNG implementation and is identical on every machine.

Dataset source: github.com/chenchkx/ETH-80 @ 9243466b (the standard ETH-80:
8 classes x 10 objects x 41 views; Leibe & Schiele, CVPR 2003).  Folder
indices there: 1 apple, 2 car, 3 cow, 4 cup, 5 dog, 6 horse, 7 pear, 8 tomato.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List

DATASET_REPO = "chenchkx/ETH-80"
DATASET_COMMIT = "9243466b9ca5f15765c88fdaf17c4f10c15a8574"
SPLIT_TAG = "ETH80-split-v1"

#: class name -> dataset folder index
CLASS_DIR: Dict[str, int] = {"apple": 1, "car": 2, "cup": 4, "horse": 6}

#: dataset folder slot (1..10) -> object id used in the file names.  Cars are
#: the only class whose ids are not 1..10 (verified against the repo tree).
OBJECT_IDS: Dict[str, List[int]] = {
    "apple": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "car": [1, 2, 3, 11, 5, 6, 7, 12, 9, 14],
    "cup": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "horse": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

#: the 41 ETH-80 views, "<zenith>-<azimuth>" (000 = from above, 090 = equator)
VIEWS: List[str] = (
    ["000-000"]
    + ["022-%03d" % a for a in (0, 90, 180, 270)]
    + ["035-%03d" % a for a in (45, 135, 225, 315)]
    + ["045-%03d" % a for a in (0, 90, 180, 270)]
    + ["066-%03d" % a for a in (27, 63, 117, 153, 207, 243, 297, 333)]
    + ["068-%03d" % a for a in (0, 90, 180, 270)]
    + ["090-%03d" % a for a in (0, 22, 45, 68, 90, 112, 135, 158, 180, 202,
                                225, 248, 270, 292, 315, 338)]
)
assert len(VIEWS) == 41

#: per-class query counts of the original 75-query protocol (README)
QUERY_COUNTS: Dict[str, int] = {"apple": 16, "car": 24, "cup": 18, "horse": 17}
assert sum(QUERY_COUNTS.values()) == 75

EVAL_SLOTS = (1, 2, 3, 4, 5)
HELDOUT_SLOTS = (6, 7, 8, 9, 10)


def rel_path(cls: str, slot: int, view: str) -> str:
    """Path of one image inside the dataset repo."""
    oid = OBJECT_IDS[cls][slot - 1]
    return "%d/%d/%s%d-%s.png" % (CLASS_DIR[cls], slot, cls, oid, view)


def _order_key(path: str) -> str:
    return hashlib.sha256((SPLIT_TAG + "|" + path).encode("utf-8")).hexdigest()


def build_split() -> Dict[str, object]:
    refs = {cls: rel_path(cls, 1, "000-000") for cls in CLASS_DIR}
    out: Dict[str, object] = {
        "tag": SPLIT_TAG,
        "dataset_repo": DATASET_REPO,
        "dataset_commit": DATASET_COMMIT,
        "classes": list(CLASS_DIR),
        "references": refs,
        "query_counts": dict(QUERY_COUNTS),
    }
    for name, slots in (("eval", EVAL_SLOTS), ("heldout", HELDOUT_SLOTS)):
        split: Dict[str, List[str]] = {}
        for cls in CLASS_DIR:
            pool = [rel_path(cls, s, v) for s in slots for v in VIEWS]
            pool = [p for p in pool if p not in refs.values()]
            pool.sort(key=_order_key)
            split[cls] = sorted(pool[:QUERY_COUNTS[cls]])
        out[name] = split
    return out


def load_split(path: str = None) -> Dict[str, object]:
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "eth80_split.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    split = build_split()
    with open(os.path.join(here, "eth80_split.json"), "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=1, sort_keys=True)
        fh.write("\n")
    for name in ("eval", "heldout"):
        n = sum(len(v) for v in split[name].values())
        print(name, n, {c: len(v) for c, v in split[name].items()})
    print("references", split["references"])
