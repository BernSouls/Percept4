"""Determinism check for the benchmark front end.

Rebuilds a few graphs from pixels in FRESH interpreters under different
``PYTHONHASHSEED`` values and compares a canonical digest (node ids, lens
types, full signatures, edges, edge parameters, anchor positions) with the
cached graph.  A benchmark whose graphs depend on the hash seed cannot be
re-run, so this is checked before any number is reported.

    python diagnostics/eth80_determinism.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import hashlib
import os
import subprocess
import sys


def graph_digest(g) -> str:
    sg = g.subgraph
    h = hashlib.sha256()
    for nid in sorted(sg.nodes):
        lt, sig = g.sigs.get(nid, ("", None))
        h.update(("N|%d|%s|%r\n" % (nid, lt, sig)).encode())
    for e in sorted(sg.edges, key=lambda e: (e[0], e[1], e[2].value)):
        h.update(("E|%d|%d|%s|%r\n" % (
            e[0], e[1], e[2].name, sg.edge_params.get(e))).encode())
    for nid in sorted(sg.anchor_of_node):
        h.update(("A|%d|%r\n" % (nid, sg.anchor_of_node[nid])).encode())
    return h.hexdigest()


def _child(cls: str, rel: str) -> None:
    import eth80_frontend as fe

    g = fe.build_graph(fe._load_rgb(fe.image_path(cls, rel)), name=rel)
    print("DIGEST", graph_digest(g))


def main() -> int:
    import eth80_frontend as fe
    from eth80_split import load_split

    split = load_split()
    picks = [(c, split["references"][c]) for c in ("apple", "horse")]
    picks += [(c, split["eval"][c][0]) for c in ("car", "cup")]
    ok = True
    for cls, rel in picks:
        cached = graph_digest(fe.graph_for(cls, rel))
        digests = []
        for seed in ("1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            out = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--child", cls, rel],
                env=env, capture_output=True, text=True, check=True).stdout
            digests.append([l.split()[1] for l in out.splitlines()
                            if l.startswith("DIGEST")][0])
        same = all(d == cached for d in digests)
        ok &= same
        print("%-28s %s  cached=%s  rebuilt=%s" % (
            os.path.basename(rel), "IDENTICAL" if same else "DIFFERS",
            cached[:12], ",".join(d[:12] for d in digests)))
    print("front end deterministic across hash seeds: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        _child(sys.argv[2], sys.argv[3])
        raise SystemExit(0)
    raise SystemExit(main())
