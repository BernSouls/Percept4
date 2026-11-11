"""EXPORT SHIM -- the legacy ``derived_lenses`` import surface.

Not source code: a re-export of ``lenses.derived`` (whose own docstring
describes it as the aggregator behind this legacy name).  See
``diagnostics/_export_shims/README.md``.  Only reachable when the real
top-level ``derived_lenses.py`` is absent.
"""
import lenses.derived as _derived
from lenses.derived import *  # noqa: F401,F403

# Module-level state (e.g. ``_SIM_MEMO``) must be the SAME objects, not copies:
# ``orchestration/pipeline.py`` reads it through this module.
globals().update(
    {k: v for k, v in vars(_derived).items() if not k.startswith("__")})
