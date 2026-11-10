"""Path set-up for the matcher pins.

Run from the tree root:

    python -m pytest tests_matcher -q

The repository root (the original ``src/``) goes to the front of ``sys.path``.
In the EXPORT checkout three top-level modules are absent (see
``diagnostics/_export_shims/README.md``); the shim directory is appended to
the END of the path, so a full working tree never sees it.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHIMS = os.path.join(ROOT, "diagnostics", "_export_shims")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.isdir(SHIMS) and SHIMS not in sys.path:
    sys.path.append(SHIMS)


@pytest.fixture
def params(monkeypatch):
    """Set ``parameters.NAME`` for one test; restored afterwards."""
    import parameters as P

    def _set(**kw):
        for k, v in kw.items():
            assert hasattr(P, k), "unknown parameter %r" % k
            monkeypatch.setattr(P, k, v)
        return P

    return _set
