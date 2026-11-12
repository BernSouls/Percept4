"""Path set-up shared by every diagnostics script.

    import _bootstrap  # noqa: F401   (first import of every script here)

* puts the repository root (the original ``src/``) at the FRONT of sys.path;
* appends ``_export_shims/`` at the END, so the shims are used only where the
  export left a module out (see ``_export_shims/README.md``).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHIMS = os.path.join(HERE, "_export_shims")
DATA_DIR = os.path.join(ROOT, "data", "ETH-80")
CACHE_DIR = os.path.join(ROOT, "data", "_cache")
RESULTS_DIR = os.path.join(HERE, "results")

for _p in (HERE, ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
if SHIMS not in sys.path:
    sys.path.append(SHIMS)


def _parse_switch(text: str):
    t = text.strip()
    low = t.lower()
    if low in ("1", "true", "on", "yes"):
        return True
    if low in ("0", "false", "off", "no"):
        return False
    if low == "()":
        return ()
    if low in ("none", ""):
        return None
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def switch_overrides():
    """[STAGE 6] Switch overrides for this run, as a canonical tuple.

    ``PERCEPT_SET="NAME=value,OTHER=value"`` in the environment sets
    ``parameters.NAME`` before any library module is imported (this module is
    the first import of every diagnostics script, and worker processes inherit
    the environment).  The front end folds the canonical string into its cache
    key, so a graph built under one setting is never served under another.
    Empty = the tree's declared defaults.
    """
    raw = os.environ.get("PERCEPT_SET", "")
    out = []
    for item in raw.split(","):
        if not item.strip():
            continue
        name, _eq, value = item.partition("=")
        out.append((name.strip(), value.strip()))
    return tuple(sorted(out))


def switch_overrides_key() -> str:
    return ",".join("%s=%s" % kv for kv in switch_overrides())


def _apply_switch_overrides() -> None:
    ov = switch_overrides()
    if not ov:
        return
    import parameters as _P
    for name, value in ov:
        if not hasattr(_P, name):
            raise SystemExit("PERCEPT_SET names an unknown parameter: %s" % name)
        setattr(_P, name, _parse_switch(value))


_apply_switch_overrides()


def shims_in_use():
    """Names of the export shims that were actually imported (for the log)."""
    out = []
    for name in ("sensory_floor", "derived_lenses", "action"):
        mod = sys.modules.get(name)
        f = getattr(mod, "__file__", "") or ""
        if mod is not None and os.path.abspath(f).startswith(SHIMS):
            out.append(name)
    return out
