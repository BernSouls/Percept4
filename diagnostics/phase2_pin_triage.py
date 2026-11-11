"""PIN TRIAGE -- every failing ASSERTION of the matcher pins, per switch value.

Why this exists.  A pytest run stops a test at its first failing assertion,
so "7 pins fail under D3" says which tests are affected but not what is wrong
in each: the assertions after the first one are never evaluated.  Classifying
a failure as "the pin states the old form" or "the implementation is wrong"
needs every assertion, not the first.

What it does.  For each switch setting it
  1. writes SOFT copies of ``tests_matcher/test_*.py`` to a temporary
     directory: every ``assert`` is wrapped so that a failure is recorded and
     the test goes on (the original file is not touched; line numbers and
     source text reported are those of the original);
  2. runs them with the pin baseline disabled (``PERCEPT_PIN_BASELINE=0``) and
     the setting applied by ``_param_overrides_plugin`` before collection, so a
     pin sees the setting unless it sets the switch itself;
  3. prints one line per failing assertion -- how often it failed, the first
     obtained/expected pair -- with its CLASS from ``CLASSIFICATION`` below.

Classes (the hand derivations are in REPORT.md, "Pin triage"; each S verdict
names the interaction pin that asserts the derived value):
  S  old semantics -- the assertion states the value of the form the setting
     replaces; under the setting the matcher returns the hand-derived value
     of the new form
  G  guard -- a non-degeneracy check on the pin's own scenario, which the
     setting makes degenerate; no claim about the matcher is contradicted
  B  implementation bug -- the matcher's value differs from the hand-derived
     one
  X  by design (``test_declared_defaults`` reddens on any default flip)
An assertion that fails and is not listed prints as ``?? UNCLASSIFIED`` and
the script exits 1.

    python diagnostics/phase2_pin_triage.py                 # all settings
    python diagnostics/phase2_pin_triage.py --only d3n,null
    python diagnostics/phase2_pin_triage.py --out diagnostics/results/x.txt

Diagnostics only: reads the library, changes nothing, measures no image.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PINS = os.path.join(ROOT, "tests_matcher")

ALL3 = '("node","relation","absence")'
SETTINGS = OrderedDict([
    ("defaults", ""),
    ("d3n", 'EVIDENCE_QUERY_SIDE_TERMS=("node",)'),
    ("d3", "EVIDENCE_QUERY_SIDE_TERMS=" + ALL3),
    ("null", "EVIDENCE_NODE_NULL_CALIBRATED=True"),
    ("d3n_null", 'EVIDENCE_QUERY_SIDE_TERMS=("node",);'
                 "EVIDENCE_NODE_NULL_CALIBRATED=True"),
    ("d3_null", "EVIDENCE_QUERY_SIDE_TERMS=" + ALL3 +
                ";EVIDENCE_NODE_NULL_CALIBRATED=True"),
    ("d1off", "EVIDENCE_GRADED_NODE_CREDIT=False"),
    ("f6off", "EDGE_UNPARAMETERISED_STRUCTURAL_MATCH=False"),
    ("p4off", "EDGE_UNMEASURED_IS_NOT_EVIDENCE=False"),
    ("mxoff", "EDGE_REVERSE_FLIPS_MIXTURE=False"),
    ("maoff", "MATCH_ASSIGNMENT_EXACT=False"),
    # stage 4
    ("bloff", "MATCH_SIGNATURE_MATRIX_BLOCKS=False"),
    ("caon", "EVIDENCE_RELATION_COUNT_APART=True"),
    ("lkon", "EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS=True"),
    ("broff", "MATCH_ALIGN_BUDGET_REFINEMENT_ONLY=False"),
    ("apon", "MATCH_GLOBAL_ANCHOR_PRIOR=True"),
    ("capon", "EDGE_SPECIFICITY_CAP_SAME_SCHEMA=True"),
])

_SOFT_PY = '''
import os
RECORDS = []


def record(lineno, src, exc):
    lines = [l.strip() for l in str(exc).split("\\n") if l.strip()]
    keep = [l for l in lines
            if l.startswith(("assert ", "Obtained", "Expected"))][:3]
    RECORDS.append({
        "test": os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0],
        "line": lineno, "src": src, "hard": False,
        "msg": " | ".join(keep or lines[:1])[:260]})
'''

_CONFTEST_TAIL = '''

# ---- appended by diagnostics/phase2_pin_triage.py (soft copy only) ----
import json as _json
import _soft


def pytest_runtest_makereport(item, call):
    if call.when == "call" and call.excinfo is not None:
        tb = call.excinfo.traceback[-1]
        _soft.RECORDS.append({
            "test": item.nodeid, "line": int(tb.lineno) + 1,
            "src": "<raised, soft copy line>", "hard": True,
            "msg": repr(call.excinfo.value)[:260]})


def pytest_sessionfinish(session):
    with open(os.environ["PIN_TRIAGE_OUT"], "w") as fh:
        _json.dump({"records": _soft.RECORDS,
                    "collected": session.testscollected}, fh)
'''


class _Soften(ast.NodeTransformer):
    def __init__(self, source: str):
        self.source = source
        self.n = 0

    def visit_Assert(self, node: ast.Assert):
        self.n += 1
        seg = ast.get_source_segment(self.source, node) or "assert ?"
        seg = " ".join(seg.split())
        handler = ast.ExceptHandler(
            type=ast.Name(id="AssertionError", ctx=ast.Load()),
            name="_soft_e",
            body=[ast.Expr(ast.Call(
                func=ast.Attribute(value=ast.Name(id="_soft", ctx=ast.Load()),
                                   attr="record", ctx=ast.Load()),
                args=[ast.Constant(node.lineno), ast.Constant(seg),
                      ast.Name(id="_soft_e", ctx=ast.Load())],
                keywords=[]))])
        return ast.Try(body=[node], handlers=[handler], orelse=[],
                       finalbody=[])


def write_soft_copies(dst: str) -> int:
    """Soft copies of the pins in `dst`; returns the number of assertions."""
    n_asserts = 0
    for name in sorted(os.listdir(PINS)):
        src_path = os.path.join(PINS, name)
        if not name.endswith(".py"):
            continue
        with open(src_path) as fh:
            source = fh.read()
        if name == "conftest.py":
            out = source + _CONFTEST_TAIL
        elif name.startswith("test_"):
            tree = ast.parse(source)
            tr = _Soften(source)
            tree = tr.visit(tree)
            n_asserts += tr.n
            # after the docstring and the __future__ import
            at = 0
            for i, stmt in enumerate(tree.body):
                if (isinstance(stmt, ast.Expr)
                        and isinstance(getattr(stmt, "value", None),
                                       ast.Constant)) or (
                        isinstance(stmt, ast.ImportFrom)
                        and stmt.module == "__future__"):
                    at = i + 1
            tree.body.insert(at, ast.Import(names=[ast.alias(name="_soft")]))
            ast.fix_missing_locations(tree)
            out = ast.unparse(tree) + "\n"
        else:
            out = source
        with open(os.path.join(dst, name), "w") as fh:
            fh.write(out)
    with open(os.path.join(dst, "_soft.py"), "w") as fh:
        fh.write(_SOFT_PY)
    return n_asserts


def run_setting(dst: str, overrides: str):
    out_json = os.path.join(dst, "_records.json")
    if os.path.exists(out_json):
        os.remove(out_json)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [ROOT, HERE, os.path.join(HERE, "_export_shims")])
    env["PYTHONHASHSEED"] = "0"
    env["PERCEPT_PIN_BASELINE"] = "0"
    env["PERCEPT_PARAMS"] = overrides
    env["PIN_TRIAGE_OUT"] = out_json
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", dst, "-q", "-p", "no:cacheprovider",
         "-p", "_param_overrides_plugin", "--rootdir", dst],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if not os.path.exists(out_json):
        raise SystemExit("pin run produced no record file:\n%s\n%s"
                         % (proc.stdout[-2000:], proc.stderr[-2000:]))
    with open(out_json) as fh:
        data = json.load(fh)
    tail = [l for l in proc.stdout.strip().split("\n") if l.strip()]
    return data, (tail[-1] if tail else "")


# ---------------------------------------------------------------------------
# CLASSIFICATION: phase2_pin_triage_classes.json, one entry per group of
# assertions:
#     {"test": name, "lines": [...], "settings": [...] or null (= any),
#      "class": "S" | "G" | "B" | "X", "why": "...", "pinned": "..."}
# An entry applies to a failing assertion when test, line and setting match;
# the first match wins.  Filled from the derivations in REPORT.md.  An
# unlisted failing assertion is an error; the summary also lists the entries
# that matched nothing (an assertion that no longer fails).
# ---------------------------------------------------------------------------
CLASSIFICATION: list = []
_USED = set()


def _load_classification():
    path = os.path.join(HERE, "phase2_pin_triage_classes.json")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        raw = json.load(fh)
    for i, entry in enumerate(raw["entries"]):
        CLASSIFICATION.append((i, entry))


def classify(test: str, line: int, setting: str):
    for i, entry in CLASSIFICATION:
        if entry["test"] != test or int(line) not in entry["lines"]:
            continue
        if entry.get("settings") and setting not in entry["settings"]:
            continue
        _USED.add(i)
        why = entry["why"]
        if entry.get("pinned"):
            why += "  [pinned: %s]" % entry["pinned"]
        return entry["class"], why
    return "??", "UNCLASSIFIED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="",
                    help="also dump every failing assertion as JSON")
    args = ap.parse_args()
    dump = []
    _load_classification()
    names = [n for n in SETTINGS
             if not args.only or n in args.only.split(",")]

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s, flush=True)

    tmp = tempfile.mkdtemp(prefix="pin_triage_")
    unclassified = 0
    try:
        n_asserts = write_soft_copies(tmp)
        emit("PIN TRIAGE -- every failing assertion of tests_matcher, pin "
             "baseline disabled")
        emit("soft copies: %d assertions wrapped; original line numbers "
             "reported" % n_asserts)
        summary = []
        for name in names:
            data, tail = run_setting(tmp, SETTINGS[name])
            recs = data["records"]
            by = OrderedDict()
            for r in recs:
                fn = r["test"].split("::")[-1]
                fn = fn.split("[")[0]
                key = (r["test"].split("::")[0].split("/")[-1], fn,
                       r["line"], r["hard"])
                by.setdefault(key, []).append(r)
            tests = OrderedDict()
            for (f, fn, line, hard), rs in by.items():
                tests.setdefault((f, fn), []).append((line, hard, rs))
            emit("")
            emit("=" * 100)
            emit("%s   %s" % (name, SETTINGS[name] or "(tree defaults)"))
            emit("=" * 100)
            emit("  %d tests collected; %d with a failing assertion; %d "
                 "failing assertions (%d distinct lines)   [pytest: %s]" % (
                     data["collected"], len(tests), len(recs), len(by), tail))
            counts = defaultdict(int)
            for (f, fn), items in tests.items():
                emit("  %s :: %s" % (f, fn))
                for line, hard, rs in items:
                    if hard:
                        cls, why = "!!", "raised after a soft failure"
                    else:
                        cls, why = classify(fn, line, name)
                    if cls == "??":
                        unclassified += 1
                    counts[cls] += 1
                    dump.append({"setting": name, "file": f, "test": fn,
                                 "line": line, "hard": hard, "n": len(rs),
                                 "class": cls, "src": rs[0]["src"],
                                 "msg": rs[0]["msg"]})
                    emit("    [%s] line %-4d x%-3d %s" % (
                        cls, line, len(rs), rs[0]["src"][:150]))
                    emit("         %s" % rs[0]["msg"])
                    if why:
                        emit("         -> %s" % why)
            summary.append((name, len(tests), len(by), dict(counts)))
        emit("")
        emit("=" * 100)
        emit("SUMMARY   setting: tests with a failure / distinct failing "
             "assertions / by class")
        for name, n_t, n_a, counts in summary:
            emit("  %-10s %2d / %2d   %s" % (
                name, n_t, n_a, "  ".join(
                    "%s=%d" % kv for kv in sorted(counts.items()))))
        emit("  unclassified failing assertions: %d" % unclassified)
        emit("  classes: S old semantics (the pin states the form its "
             "baseline setting gives; the other value is pinned where "
             "named)")
        emit("           G guard on the pin's own scenario   B implementation "
             "bug   X by design   !! raised after a soft failure")
        if not args.only:
            stale = [e for i, e in CLASSIFICATION if i not in _USED]
            emit("  classification entries that matched no failing "
                 "assertion: %d" % len(stale))
            for e in stale:
                emit("    [%s] %s lines %s (%s): %s" % (
                    e["class"], e["test"], e["lines"],
                    ",".join(e.get("settings") or ["any"]), e["why"][:110]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dump, fh, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return 1 if unclassified else 0


if __name__ == "__main__":
    sys.exit(main())
