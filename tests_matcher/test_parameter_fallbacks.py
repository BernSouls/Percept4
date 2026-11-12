"""[STAGE 7 / PHASE 7] Every `getattr(parameters.X, "...", default)` fallback
agrees with the declaration.

A fallback is a second, silent declaration.  When it disagrees with the real
one the tree has two values for one switch and nothing says which is in force:
before this pin, `MATCHER_USE_POPULATION_DISTANCE` fell back to `False` while
declared `True`, `LENS_POOL_NORMALIZE` to `False` while declared `True`,
`COCIRCULAR_MAX_PATH_ANGLE` to pi/3 while declared pi/6, `CONTOUR_CONTENT_MODE`
to "full" while declared "summary", `BO_GRAVITY_SCALE` to 4.0 while declared
10.0 -- 23 of them in all.  Every one was aligned to the declaration (the
fallback now says what the tree declares, so no behaviour changed) and this pin
keeps them aligned.

The audit is static (ast), so it sees the whole tree the same way, including
code no test reaches.
"""
from __future__ import annotations

import ast
import glob
import math
import os

import pytest

import parameters as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = ("/tests", "/diagnostics/", "/.git/", "__pycache__", "/data/", "/scripts/")


def _calls():
    """(file, line, name, default-ast, node) for every getattr(parameters, "...")."""
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)):
        rel = f.replace(ROOT, "").replace(os.sep, "/")
        if any(s in f for s in SKIP):
            continue
        if rel.endswith("test_parameter_fallbacks.py"):
            continue
        try:
            tree = ast.parse(open(f, "r", encoding="utf-8", errors="ignore").read())
        except SyntaxError:                       # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Name) and fn.id == "getattr"):
                continue
            if len(node.args) < 3:
                continue
            a0, a1, a2 = node.args[:3]
            if not (isinstance(a0, ast.Name) and a0.id == "P"):
                continue
            if not (isinstance(a1, ast.Constant) and isinstance(a1.value, str)):
                continue
            name = a1.value
            if not name.isidentifier() or name.startswith("__"):
                continue
            out.append((rel, node.lineno, name, a2))
    return out


def _value(node):
    """The fallback as a value, or raise _CallerValue when it is not one."""
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    try:
        return eval(compile(ast.Expression(node), "<fallback>", "eval"),
                    {"__builtins__": {}, "math": math})
    except Exception:
        raise _CallerValue()


class _CallerValue(Exception):
    """`getattr(P, "X", some_caller_argument)` -- not a declaration conflict."""


def test_every_fallback_exists_in_the_tree():
    """PIN: a name read with a fallback must be DECLARED.  Four were not
    (`V2_MAX_CANDIDATES_PER_CATEGORY`, `BO_CALIBRATION_SLOPE`,
    `BO_CALIBRATION_INTERCEPT`, `TEXTURE_CALIBRATION`, `JUNCTION_ARM_CLUSTER_TOL`)
    -- the knob existed only inside the read, and nothing outside the module
    could set it.  They are declared now."""
    undeclared = [(f, ln, n) for f, ln, n, _d in _calls() if not hasattr(P, n)]
    assert not undeclared, "read with a fallback but never declared: %r" % undeclared


def test_every_fallback_agrees_with_the_declaration():
    """PIN: the fallback says what the tree declares.  When the two disagree,
    the value in force depends on whether the module was imported before or
    after the declaration was set -- a difference no result could be trusted
    through."""
    bad = []
    for f, ln, name, node in _calls():
        if not hasattr(P, name):
            continue
        try:
            default = _value(node)
        except _CallerValue:
            continue
        declared = getattr(P, name)
        if isinstance(declared, bool) or isinstance(default, bool):
            same = bool(declared) == bool(default)
        else:
            same = declared == default
        if not same:
            bad.append((f, ln, name, repr(declared)[:40], repr(default)[:40]))
    assert not bad, "fallbacks that disagree with the declaration: %r" % bad[:10]
