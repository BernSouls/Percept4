"""pytest plugin: how often does a bar call the stage-5 writers?

A bar that never calls the changed code says nothing about the change (the
same idea as PERCEPT_REACH in ``_param_overrides_plugin.py``, for the
functions stage 5 edited or read).  The wrappers call straight through.

    PYTHONPATH=.:diagnostics:diagnostics/_export_shims python -m pytest \\
        tests_figground --ignore=tests_figground/test_renderer_truth.py \\
        -q -p no:cacheprovider -p _stage5_reach_plugin

Diagnostics only -- not library, not ported.
"""
import functools

_CALLS = {}
_RET = {}
_TARGETS = (
    ("orchestration.binder", "Binder", "_run_curve_containment", False),
    ("orchestration.binder", "Binder", "_containment_values", True),
    ("orchestration.binder", "Binder", "_containment_geometry", True),
    ("orchestration.binder", "Binder", "_footprint", True),
    ("orchestration.binder", "Binder", "_size_ratio", True),
    ("orchestration.binder", "Binder", "_run_co_located", False),
    ("orchestration.binder", "Binder", "_run_radiality", False),
    ("orchestration.binder", "Binder", "_run_spirality", False),
    ("orchestration.binder", "Binder", "_run_curvature", False),
    ("orchestration.binder", "Binder", "step", False),
    ("lenses.chain_lens", "ChainLens", "step", False),
    ("lenses.v4", "CurveComplexLens", "step", False),
)


def pytest_collection_finish(session):
    import importlib
    for mod_name, owner, attr, static in _TARGETS:
        mod = importlib.import_module(mod_name)
        holder = getattr(mod, owner)
        fn = getattr(holder, attr)
        label = "%s.%s.%s" % (mod_name, owner, attr)
        _CALLS.setdefault(label, 0)

        def make(fn=fn, label=label):
            @functools.wraps(fn)
            def counted(*a, **kw):
                _CALLS[label] += 1
                out = fn(*a, **kw)
                if label.endswith("_containment_values") and isinstance(out, dict):
                    key = "measured" if any(v is not None for v in out.values()) else "all absent"
                    _RET[key] = _RET.get(key, 0) + 1
                return out
            return counted
        w = make()
        setattr(holder, attr, staticmethod(w) if static else w)


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line("")
    terminalreporter.write_line("REACH -- calls into the stage-5 writers, whole session:")
    for label in sorted(_CALLS):
        terminalreporter.write_line("  %-58s %d" % (label, _CALLS[label]))
    terminalreporter.write_line("  _containment_values results: %s" % (_RET or "none"))
