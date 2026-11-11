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


def shims_in_use():
    """Names of the export shims that were actually imported (for the log)."""
    out = []
    for name in ("sensory_floor", "derived_lenses", "action"):
        mod = sys.modules.get(name)
        f = getattr(mod, "__file__", "") or ""
        if mod is not None and os.path.abspath(f).startswith(SHIMS):
            out.append(name)
    return out
