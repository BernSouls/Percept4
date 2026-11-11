"""pytest plugin: run a test session under explicit parameter values.

The frozen bars (figground pins, matcher pins) have to be green not only
under the tree's defaults but under any value a switch might be defaulted
to.  This applies ``NAME=VALUE`` pairs to ``parameters`` before collection,
and says so in the summary, so the log shows what the bar ran under.

    PERCEPT_PARAMS='EVIDENCE_QUERY_SIDE_TERMS=("node","relation","absence")' \\
    PYTHONPATH=.:diagnostics python -m pytest tests_figground \\
        -p _param_overrides_plugin -q

Pairs are separated by ``;``.  An unknown name is an error.  Diagnostics
only -- not library, not ported.

REACH.  A bar that never calls the changed code says nothing about the
change.  With ``PERCEPT_REACH=1`` the plugin counts, over the whole session,
the calls to the functions stage 1 and stage 2 edited and prints the counts
in the summary.  The wrappers call straight through; nothing is altered.
"""
from __future__ import annotations

import ast
import os

_APPLIED = {}


def _apply() -> None:
    raw = os.environ.get("PERCEPT_PARAMS", "").strip()
    if not raw or _APPLIED:
        return
    import parameters as P

    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        name, eq, val = item.partition("=")
        name = name.strip()
        if not eq or not hasattr(P, name):
            raise SystemExit("PERCEPT_PARAMS: bad or unknown entry %r" % item)
        value = ast.literal_eval(val.strip())
        setattr(P, name, value)
        _APPLIED[name] = value


def pytest_configure(config):
    _apply()


def pytest_report_header(config):
    if not _APPLIED:
        return "parameter overrides: none (tree defaults)"
    return "parameter overrides: " + "; ".join(
        "%s=%r" % kv for kv in sorted(_APPLIED.items()))


# --------------------------------------------------------------- reach count

_CALLS = {}
#: (module, owner or None, attribute) -- the functions the fixes edited
_REACH = (
    ("memory.cores", None, "evidence_score"),
    ("memory.cores", None, "_mcs_accumulate"),
    ("memory.cores", None, "mcs_score"),
    ("memory.cores", None, "mcs_similarity"),
    ("core.schema", None, "polymorphic_edge_similarity"),
    ("core.schema", "EdgeParams", "similarity"),
)


def _wrap(fn, label):
    import functools

    @functools.wraps(fn)
    def counted(*a, **kw):
        _CALLS[label] = _CALLS.get(label, 0) + 1
        return fn(*a, **kw)

    return counted


_LOADED_BY_SESSION = {}


def _install_reach() -> None:
    import importlib
    import sys as _sys

    # what the collected tests themselves imported, before this plugin
    # imports anything
    for mod_name in ("memory.cores", "core.schema", "consolidator",
                     "memory.mapper", "matching.calibration"):
        _LOADED_BY_SESSION[mod_name] = mod_name in _sys.modules

    for mod_name, owner, attr in _REACH:
        mod = importlib.import_module(mod_name)
        holder = getattr(mod, owner) if owner else mod
        label = "%s.%s%s" % (mod_name, (owner + ".") if owner else "", attr)
        _CALLS.setdefault(label, 0)
        setattr(holder, attr, _wrap(getattr(holder, attr), label))
    # names imported `from memory.cores import ...` elsewhere keep the
    # original object; re-point the ones the library itself uses
    import sys

    cores = importlib.import_module("memory.cores")
    schema = importlib.import_module("core.schema")
    for m in list(sys.modules.values()):
        if m is None or m is cores or m is schema:
            continue
        for src, name in ((cores, "evidence_score"), (cores, "mcs_score"),
                          (cores, "mcs_similarity"),
                          (schema, "polymorphic_edge_similarity")):
            try:
                cur = m.__dict__.get(name)
            except Exception:
                continue
            if cur is not None and getattr(cur, "__wrapped__", None) is None \
                    and getattr(cur, "__module__", "") == src.__name__:
                m.__dict__[name] = getattr(src, name)


def pytest_collection_finish(session):
    # after collection every test module (and what it imports) is loaded
    if os.environ.get("PERCEPT_REACH", "") == "1":
        _install_reach()


def pytest_terminal_summary(terminalreporter):
    # always: `-q` suppresses the session header, and a log has to say what
    # it ran under
    terminalreporter.write_line("")
    terminalreporter.write_line(
        "PARAMETER OVERRIDES for this session: " + (
            "; ".join("%s=%r" % kv for kv in sorted(_APPLIED.items()))
            if _APPLIED else "none (tree defaults)"))
    if os.environ.get("PERCEPT_REACH", "") != "1":
        return
    terminalreporter.write_line("")
    terminalreporter.write_line(
        "REACH -- calls into the functions the fixes edited, whole session:")
    for label in sorted(_CALLS):
        terminalreporter.write_line("  %-48s %d" % (label, _CALLS[label]))
    terminalreporter.write_line(
        "  imported by the collected tests themselves: " + ", ".join(
            "%s %s" % (m, "yes" if v else "no")
            for m, v in sorted(_LOADED_BY_SESSION.items())))
