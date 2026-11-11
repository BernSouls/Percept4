"""EXPORT SHIM -- the legacy ``sensory_floor`` import surface.

Not source code: a re-export of names that already live in the tree.  See
``diagnostics/_export_shims/README.md``.  Only reachable when the real
top-level ``sensory_floor.py`` is absent (this directory is appended to the
END of ``sys.path``).
"""
from core.types import *  # noqa: F401,F403
from core.types import (  # noqa: F401
    Detection,
    TypedSignature,
    serialize_detection,
)
from perception.observation import *  # noqa: F401,F403
from perception.observation import (  # noqa: F401
    ACTION_LENS_NAME,
    ACTION_ROLE_NAME,
    action_signature,
    resolve_anchor_position,
)
from perception.polar import polar_boundary_fragments  # noqa: F401
