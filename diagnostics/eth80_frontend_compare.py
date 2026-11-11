"""Do two front-end source states build the SAME graphs?

A library edit under ``parameters/`` or ``core/`` changes the front-end
source digest, so every graph is rebuilt even when the edit cannot reach the
front end (a new matcher switch, a comment).  Results measured on the old
graphs can be set next to results on the new ones only if the graphs are the
same.  This compares the canonical content digest (``eth80_determinism``:
node ids, lens types, full signatures, edges, edge parameters, anchors) of
every graph of the split, cached under the OLD digest, with the graph the
current tree builds.

    python diagnostics/eth80_frontend_compare.py --old <old front-end digest>

``--old`` defaults to the digest recorded in a stored result
(``--from-result p2_R0_full``).  A prefix is resolved from the stored
records.  The old cache entries must still be on disk; when one is not, the
exit code is 2 and the output says that nothing was compared.
Diagnostics only.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import hashlib
import json
import os
import pickle
import sys

from _bootstrap import RESULTS_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="")
    ap.add_argument("--from-result", default="p2_R0_full")
    ap.add_argument("--which", default="eval", choices=("eval", "heldout"))
    args = ap.parse_args()

    import eth80_frontend as fe
    from eth80_determinism import graph_digest
    from eth80_split import load_split
    from _stats import CLASSES

    old = args.old
    if not old:
        with open(os.path.join(RESULTS_DIR, args.from_result + ".json")) as fh:
            old = json.load(fh)["frontend_digest"]
    if len(old) != 64:
        # The cache is keyed by the FULL digest.  A prefix (the 12 or 16
        # characters the tables print) used to find no entry and report
        # "old cache entry missing for 79" -- which reads like a cleared
        # cache.  Resolve it from the stored records, or stop.
        full = _stored_digests(old)
        if len(full) != 1:
            print("--old %r is not a full 64-character digest and matches "
                  "%d of the digests in the stored records (%s); pass the "
                  "full digest (the `frontend_digest` of a stored result)"
                  % (old, len(full),
                     ", ".join(sorted(d[:16] for d in full)) or "none"))
            return 2
        old = full.pop()
    new = fe.frontend_source_digest()
    print("old front-end digest %s" % old[:16])
    print("new front-end digest %s" % new[:16])
    if old == new:
        print("the digests are equal: nothing to compare")
        return 0
    split = load_split()
    entries = [(c, split["references"][c]) for c in CLASSES] + [
        (c, p) for c in CLASSES for p in split[args.which][c]]
    same = missing = 0
    differing = []
    for cls, rel in entries:
        path = fe.image_path(cls, rel)
        with open(path, "rb") as fh:
            img = hashlib.sha256(fh.read()).hexdigest()
        key = hashlib.sha256((old + "|" + img).encode()).hexdigest()
        old_file = os.path.join(fe.CACHE_DIR, key[:2], key + ".pkl")
        if not os.path.isfile(old_file):
            missing += 1
            continue
        with open(old_file, "rb") as fh:
            g_old = pickle.load(fh)
        g_new = fe.graph_for(cls, rel)
        if graph_digest(g_old) == graph_digest(g_new):
            same += 1
        else:
            differing.append(rel)
    print("graphs compared %d of %d (old cache entry missing for %d): "
          "identical content %d, differing %d" % (
              same + len(differing), len(entries), missing, same,
              len(differing)))
    for rel in differing[:10]:
        print("  differs: %s" % rel)
    if missing:
        print("NOT A COMPARISON for the %d missing: no cache file under the "
              "old digest (a wrong digest, or the cache was cleared). "
              "Nothing above says those graphs differ, or that they are the "
              "same." % missing)
    return 0 if not differing and not missing else 2


def _stored_digests(prefix: str):
    """Full front-end digests in the stored records that start with
    `prefix`."""
    found = set()
    for base, _dirs, files in os.walk(RESULTS_DIR):
        for f in files:
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(base, f)) as fh:
                    rec = json.load(fh)
            except (OSError, ValueError):
                continue
            d = rec.get("frontend_digest") if isinstance(rec, dict) else None
            if isinstance(d, str) and len(d) == 64 and d.startswith(prefix):
                found.add(d)
    return found


if __name__ == "__main__":
    sys.exit(main())
