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


#: The switch values every pin in this directory is written against.  A pin
#: that tests another value sets it through ``params``.  Stating them here
#: keeps the pins independent of the tree's defaults: flipping a default
#: reddens exactly one test (``test_declared_defaults.py``), not every pin
#: that happened to rely on it.
PIN_BASELINE = {
    "EDGE_UNPARAMETERISED_STRUCTURAL_MATCH": True,     # F6
    "EVIDENCE_GRADED_NODE_CREDIT": True,               # D1
    "EVIDENCE_QUERY_SIDE_TERMS": (),                   # D3
    "EDGE_UNMEASURED_IS_NOT_EVIDENCE": True,           # Phase 4
    "EVIDENCE_NODE_NULL_CALIBRATED": False,            # null calibration
}

#: the tree's own values, read once before any pin touches them
_DECLARED = {}


@pytest.fixture(autouse=True)
def _pin_baseline(monkeypatch):
    """Put every gap-closure switch at the value the pins are written for.

    TRIAGE: with ``PERCEPT_PIN_BASELINE=0`` in the environment the baseline
    is NOT applied, so the pins run under whatever the tree (or
    ``diagnostics/_param_overrides_plugin.py``) has set.  A pin that fails
    then is one whose expectation depends on a switch it does not set itself
    -- the instrument used to classify the pins in REPORT.md
    (``diagnostics/phase2_pin_triage.py``).

    WITH the baseline applied, a ``PERCEPT_PARAMS`` override does not reach
    these pins (the baseline is set after it), so "the matcher pins are green
    under override X" says nothing.  What a pin reads under the OTHER values
    is pinned explicitly in ``test_switch_interactions.py``, which sets every
    switch itself.
    """
    import parameters as P

    apply = os.environ.get("PERCEPT_PIN_BASELINE", "1") != "0"
    for k, v in PIN_BASELINE.items():
        assert hasattr(P, k), "unknown parameter %r" % k
        _DECLARED.setdefault(k, getattr(P, k))
        if apply:
            monkeypatch.setattr(P, k, v)


@pytest.fixture
def declared_defaults(_pin_baseline):
    """The switch values the tree declares (before the pin baseline)."""
    return dict(_DECLARED)


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
