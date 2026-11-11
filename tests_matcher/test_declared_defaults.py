"""PIN -- the defaults the tree declares for the gap-closure switches.

Every other pin in this directory states the switch values it is written
against (``conftest.PIN_BASELINE``), so it does not move when a default
does.  THIS is the one test that reddens when a default is flipped: change
the expected value here in the same commit, on purpose.

Why each default is what it is: ``diagnostics/REPORT.md``.
"""
from __future__ import annotations


def test_declared_defaults(declared_defaults):
    assert declared_defaults == {
        # F6 -- landed on; a parameter-less pair matches as structure
        "EDGE_UNPARAMETERISED_STRUCTURAL_MATCH": True,
        # D1 -- landed on; its success signal was NOT met
        "EVIDENCE_GRADED_NODE_CREDIT": True,
        # D3 -- in the tree, OFF: its pre-registered rule failed
        "EVIDENCE_QUERY_SIDE_TERMS": (),
        # Phase 4 -- on by the spec; never exercised on ETH-80
        "EDGE_UNMEASURED_IS_NOT_EVIDENCE": True,
        # null calibration -- in the tree, OFF: its rule failed
        "EVIDENCE_NODE_NULL_CALIBRATED": False,
        # mixture reversal -- on: the library's own contract (flipped_copy)
        "EDGE_REVERSE_FLIPS_MIXTURE": True,
        # exact assignment -- on: the contract of the function's name
        "MATCH_ASSIGNMENT_EXACT": True,
        # signature matrix in blocks -- on: the same floats, faster
        "MATCH_SIGNATURE_MATRIX_BLOCKS": True,
        # P4b, presence x physics -- in the tree; default by its rule
        "EVIDENCE_RELATION_COUNT_APART": False,
        # P4b, the flag in f(n) -- OFF: nothing here can measure it
        "EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS": False,
    }
