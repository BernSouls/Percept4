"""EXPORT SHIM -- import stub.  ``orchestration/pipeline.py`` imports this
name at module top level; the diagnostics never call it."""


def decode_action_sequence(*_args, **_kwargs):
    raise NotImplementedError(
        "action.decode is not part of this export (import stub only)")
