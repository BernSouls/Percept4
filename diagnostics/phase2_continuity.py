"""CONTINUITY -- re-run a stored configuration and compare with the stored record.

The sandbox (interpreter, numpy, scipy, the fetched images, the graph cache)
is rebuilt between sessions and the library versions are not pinned by the
repository.  Before a new number is set next to an old one, the old one has
to be reproducible in the new environment.  This re-scores a STORED
configuration from the stored parameter overrides, rounds the fresh result the
way ``save`` does, and compares it with the stored record value by value
(stored against stored).  Nothing under ``results/`` is rewritten.

    python diagnostics/phase2_continuity.py --tags p2_R0
    python diagnostics/phase2_continuity.py --tags p2_R0,p2_D3 --windows full
    PYTHONHASHSEED=1 python diagnostics/phase2_continuity.py --tags p2_D3

The last form is the HASH-SEED check of the matcher: the front end is checked
across seeds by ``eth80_determinism.py``, the matcher is checked here (edge
sets iterate in a seed-dependent order, and ``matched_rel_sim_sum`` is a plain
running sum over one).

``--keys`` restricts the comparison (default: every numeric per-pair key the
two records share); ``--skip-keys`` leaves keys out, for a comparison across a
change that is known to move a trace key and nothing else.

Diagnostics only: reads the library, changes nothing.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import argparse
import ast
import json
import os
import sys

import _stats as st
from _bootstrap import RESULTS_DIR
from _stats import CLASSES


def versions() -> str:
    import numpy
    import scipy
    try:
        import PIL
        pil = PIL.__version__
    except Exception:                                   # pragma: no cover
        pil = "?"
    return "python %s  numpy %s  scipy %s  pillow %s  PYTHONHASHSEED=%s" % (
        sys.version.split()[0], numpy.__version__, scipy.__version__, pil,
        os.environ.get("PYTHONHASHSEED", "(unset)"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="p2_R0")
    ap.add_argument("--windows", default="full,v1")
    ap.add_argument("--keys", default="")
    ap.add_argument("--skip-keys", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--stored-dir", default="",
                    help="compare the records under results/ with the "
                         "records of the same name in this directory "
                         "instead of re-running anything")
    args = ap.parse_args()

    import eth80_benchmark as bm
    import eth80_frontend as fe
    import parameters as P

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s, flush=True)

    emit("CONTINUITY -- fresh run against the stored record, value by value "
         "(both rounded to 10 significant digits)")
    emit("environment now: %s" % versions())
    emit("digests now:     source %s  harness %s  front end %s" % (
        bm.source_digest()[:12], bm.harness_digest()[:12],
        fe.frontend_source_digest()[:12]))
    keys = [k for k in args.keys.split(",") if k] or None
    skip = set(k for k in args.skip_keys.split(",") if k)
    worst_all = 0
    for tag in args.tags.split(","):
        for window in args.windows.split(","):
            path = os.path.join(RESULTS_DIR, "%s_%s.json" % (tag, window))
            with open(path, "r", encoding="utf-8") as fh:
                old = json.load(fh)
            if args.stored_dir:
                # `old` = the earlier record, `new` = the one under results/
                new = old
                with open(os.path.join(args.stored_dir, "%s_%s.json" % (
                        tag, window)), "r", encoding="utf-8") as fh:
                    old = json.load(fh)
            else:
                overrides = {
                    k: ast.literal_eval(v)
                    for k, v in old.get("param_overrides", {}).items()}
                # a record made before a later switch existed is re-run with
                # that switch at its neutral value, whatever the tree now
                # declares for it
                from phase2_d3_p4_null import STAGE4_NEUTRAL
                for k, v in STAGE4_NEUTRAL.items():
                    overrides.setdefault(k, v)
                prev = {k: getattr(P, k) for k in overrides}
                for k, v in overrides.items():
                    setattr(P, k, v)
                try:
                    new = bm.run(old["split"], window, old["arm"], workers=2,
                                 breakdown="agg" in old, verbose=False)
                finally:
                    for k, v in prev.items():
                        setattr(P, k, v)
                new = json.loads(json.dumps(bm._round_floats(new)))
            use = keys
            if use is None and skip:
                pa = old["queries"][0]["pairs"][CLASSES[0]]
                pb = new["queries"][0]["pairs"][CLASSES[0]]
                use = sorted(k for k in pa if k in pb and k not in skip
                             and isinstance(pa[k], (int, float)))
            cmp_ = st.compare_stored(old, new, use)
            ans = sum(int(qa["pred"] != qb["pred"])
                      for qa, qb in zip(old["queries"], new["queries"]))
            only_old = sorted(set(old["queries"][0]["pairs"][CLASSES[0]])
                              - set(new["queries"][0]["pairs"][CLASSES[0]]))
            only_new = sorted(set(new["queries"][0]["pairs"][CLASSES[0]])
                              - set(old["queries"][0]["pairs"][CLASSES[0]]))
            emit("")
            emit("%s %s   stored under source %s  harness %s  front end %s" % (
                tag, window, str(old.get("source_digest"))[:12],
                str(old.get("harness_digest"))[:12],
                str(old.get("frontend_digest"))[:12]))
            if args.stored_dir:
                emit("  against the record now under results/: source %s  "
                     "harness %s  front end %s" % (
                         str(new.get("source_digest"))[:12],
                         str(new.get("harness_digest"))[:12],
                         str(new.get("frontend_digest"))[:12]))
            emit("  overrides: %s" % "; ".join(
                "%s=%s" % kv for kv in sorted(
                    old.get("param_overrides", {}).items())))
            emit("  values compared %d   differing %d   scores differing %d   "
                 "answers differing %d of %d   largest difference %.3g%s" % (
                     cmp_["n"], cmp_["differ"], cmp_["score_differ"], ans,
                     len(old["queries"]), cmp_["worst"],
                     (" (on %s)" % cmp_["worst_key"]) if cmp_["differ"] else ""))
            if skip:
                emit("  keys left out: %s" % ", ".join(sorted(skip)))
            if only_old or only_new:
                emit("  per-pair keys only in the stored record: %s; only in "
                     "the fresh run: %s" % (only_old or "-", only_new or "-"))
            if "agg" in old and "agg" in new:
                emit("  aggregated breakdown tables identical: %s" % (
                    old["agg"] == new["agg"]))
            worst_all += cmp_["differ"] + ans
    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return 0 if worst_all == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
