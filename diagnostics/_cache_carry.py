"""Carry the cached graphs of a committed revision over to the working tree.

The front-end cache is keyed by a digest of every front-end source file, so
ANY edit empties it (8 minutes per 79 single-step graphs, more for the
multi-step route).  When an edit only adds code behind a switch that is OFF,
the graphs are the same graphs.  This tool PROVES that on a sample (fresh
build against the cached pickle: nodes, edges, edge parameters, signatures,
anchors, detection counts, binder diagnostics) and only then hard-links the
old keys to the new ones.  It refuses on the first difference.

    python diagnostics/_cache_carry.py [--rev HEAD] [--verify 4] [--routes 1,2,3]
                                       [--splits eval] [--verify-multi 1]

Run it under the same PERCEPT_SET as the cached graphs: the override is part
of the key on both sides, and the sample is built fresh under it.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402,F401
from _bootstrap import CACHE_DIR, ROOT  # noqa: E402

import eth80_frontend as fe  # noqa: E402
import eth80_split as sp  # noqa: E402


def digest_of_revision(rev: str) -> str:
    """The front-end digest the tree had at `rev` (same file list as today)."""
    h = hashlib.sha256()
    listed = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev], capture_output=True,
                            cwd=ROOT, check=True).stdout.decode().splitlines()
    dirs = tuple(d.rstrip("/") + "/" for d in fe._FRONTEND_SOURCE_DIRS)
    paths = [p for p in listed if p.endswith(".py") and p.startswith(dirs)]
    paths += [f for f in fe._FRONTEND_SOURCE_FILES if f in listed]
    for rel in sorted(set(paths)):
        h.update(rel.encode())
        data = subprocess.run(["git", "show", "%s:%s" % (rev, rel)], capture_output=True,
                              cwd=ROOT, check=True).stdout
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


def _key(digest: str, img_digest: str, n_steps: int) -> str:
    route = "" if n_steps == 1 else "|steps=%d" % n_steps
    ov = _bootstrap.switch_overrides_key()
    if ov:
        route += "|set=" + ov
    k = hashlib.sha256((digest + "|" + img_digest + route).encode()).hexdigest()
    return os.path.join(CACHE_DIR, k[:2], k + ".pkl")


def _same(old, new) -> bool:
    return (old.subgraph.nodes == new.subgraph.nodes
            and old.subgraph.edges == new.subgraph.edges
            and {k: repr(v) for k, v in old.subgraph.edge_params.items()}
            == {k: repr(v) for k, v in new.subgraph.edge_params.items()}
            and {k: (v[0], v[1].serialize()) for k, v in old.sigs.items()}
            == {k: (v[0], v[1].serialize()) for k, v in new.sigs.items()}
            and old.node_anchor == new.node_anchor
            and old.det_counts == new.det_counts
            and old.binder_diag == new.binder_diag)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--verify", type=int, default=4)
    ap.add_argument("--verify-multi", type=int, default=1)
    ap.add_argument("--routes", default="1")
    ap.add_argument("--splits", default="eval")
    a = ap.parse_args(argv)
    print("PERCEPT_SET: %s" % (_bootstrap.switch_overrides_key() or "none"))
    old, new = digest_of_revision(a.rev), fe.frontend_source_digest()
    print("digest at %s: %s   working tree: %s" % (a.rev, old[:12], new[:12]))
    if old == new:
        print("same digest: nothing to carry")
        return 0
    split = sp.load_split()
    entries = list(split["references"].items())
    for name in a.splits.split(","):
        entries += [(c, p) for c, ps in split[name].items() for p in ps]
    routes = [int(r) for r in a.routes.split(",")]

    def img_digest(cls, rel):
        with open(fe.image_path(cls, rel), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    refs = list(split["references"].items())
    ev = [(c, ps[0]) for c, ps in sorted(split["eval"].items())]
    sample = (refs[:2] + ev[2:] + refs[2:] + ev[:2])[: max(1, a.verify)]
    checks = [(c, r, 1) for c, r in sample]
    checks += [(c, r, n) for n in routes if n > 1 for c, r in sample[: a.verify_multi]]
    for cls, rel, n in checks:
        f = _key(old, img_digest(cls, rel), n)
        if not os.path.isfile(f):
            raise SystemExit("no cached graph of %s for %s (steps %d)" % (a.rev, rel, n))
        with open(f, "rb") as fh:
            cached = pickle.load(fh)
        fresh = fe.build_graph(fe._load_rgb(fe.image_path(cls, rel)),
                               name=os.path.basename(rel), n_steps=n)
        ok = _same(cached, fresh)
        print("  verify %-22s steps %d: %s" % (os.path.basename(rel), n,
                                                "identical" if ok else "DIFFERENT"))
        if not ok:
            raise SystemExit("the working tree does not build the cached graph: not carried")
    linked = missing = 0
    for cls, rel in entries:
        d = img_digest(cls, rel)
        for n in routes:
            src, dst = _key(old, d, n), _key(new, d, n)
            if not os.path.isfile(src):
                missing += 1
                continue
            if not os.path.isfile(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.link(src, dst)
                linked += 1
    print("carried %d graphs (%d not cached at %s)" % (linked, missing, a.rev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
