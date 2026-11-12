"""[STAGE 7 / PHASE 6] PINS THAT REACH THE REAL FRONT END.

Every pin in `test_stage6_relations.py` before this file took the relations
apart with hand-built hosts, features and detections, or called
`Binder._stage6_*` directly.  NONE of them built a graph the way the repository
builds one, so a change that stopped a relation from ever being WRITTEN in the
real pipeline -- a lens that stops emitting, a writer that is never called, a
switch that is read at a dead site -- would have left the whole matcher suite
green.  `tests_figground` reaches the binder but not the matcher, and no
diagnostic is a pin.

These pins run `diagnostics/eth80_frontend.build_graph` on a synthetic image:
the real perception stack, the real lens tier, the real binder, the real
writers.  They assert what the graph CONTAINS, not how a writer behaves when
called by hand:

  1. all three stage-6 relations are WRITTEN on a real graph, and the switch
     off writes none of them (the end-to-end form of the switch pin);
  2. their slots take more than one value -- a relation written with one
     constant parameter is a count, not a measurement (7.35);
  3. the FRAME/ATTRIBUTE partition holds on the nodes the front end emits;
  4. an ambiguous attachment writes ABSENT and is counted, never guessed;
  5. the graph cache key moves when a graph-side switch moves (7.28's pin,
     restated here as the end-to-end form the review asked for).

Build cost is ~11 s per graph on this machine and the pins share one module
level cache, so the file is one build per switch setting, not one per pin.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pytest

import core.schema as S
from core.types import EdgeType
from matching.relation_frame import (ATTRIBUTE_LENS_TYPES, FRAME_LENS_TYPES,
                                     is_frame)

NEW = (EdgeType.BOUNDARY_ATTACHMENT, EdgeType.BOUNDARY_SUCCESSOR,
       EdgeType.ILLUSORY_INDUCER)
NAMES = {getattr(t, "name", str(t)): t for t in NEW}


def _diag_on_path():
    here = os.path.dirname(os.path.abspath(__file__))
    diag = os.path.join(os.path.dirname(here), "diagnostics")
    if diag not in sys.path:
        sys.path.insert(0, diag)


def _disk_image(size=160, r=0.22):
    """A disk on a flat ground: closed boundary, one object, several views of
    it inside one image (the lens tier finds its own peaks, chains, corners)."""
    yy, xx = np.mgrid[0:size, 0:size]
    disk = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) <= (size * r) ** 2
    a = np.where(disk, 215, 60).astype(np.uint8)
    return np.dstack([a, a, a])


_CACHE = {}


def graph(switches_on: bool):
    """The real front end's graph of the scene, built once per switch setting."""
    key = bool(switches_on)
    if key in _CACHE:
        return _CACHE[key]
    _diag_on_path()
    import eth80_frontend as fe
    import parameters as P
    old = {k: getattr(P, k) for k in ("GRAPH_STAGE6_RELATIONS",
                                      "BINDER_COMPLETE_POSITION_MAP")}
    try:
        P.GRAPH_STAGE6_RELATIONS = bool(switches_on)
        P.BINDER_COMPLETE_POSITION_MAP = bool(switches_on)
        g = fe.build_graph(_disk_image(), name="phase6_pin_disk",
                          record_co_located=False)
    finally:
        for k, v in old.items():
            setattr(P, k, v)
    _CACHE[key] = g
    return g


@pytest.fixture(scope="module")
def on():
    return graph(True)


@pytest.fixture(scope="module")
def off():
    return graph(False)


def _edges(g, et):
    return [e for e in g.subgraph.edges if e[2] == et]


def _params(g, et):
    return [g.subgraph.edge_params.get(tuple(e)) for e in _edges(g, et)]


# --------------------------------------------------------------------------
# 1. THE RELATIONS EXIST, END TO END
# --------------------------------------------------------------------------

def test_the_real_front_end_writes_all_three_stage6_relations(on):
    """PIN (Phase 6, DONE condition).  On a graph built by the real perception
    stack, all three relations are present.  This is the pin that fails if any
    of them stops being written -- not if a writer misbehaves when called by
    hand, which the hand-built pins already cover."""
    assert len(on.subgraph.nodes) > 50, "the scene must actually be perceived"
    got = {t.name: len(_edges(on, t)) for t in NEW}
    for name in ("BOUNDARY_ATTACHMENT", "BOUNDARY_SUCCESSOR", "ILLUSORY_INDUCER"):
        assert got[name] > 0, "%s is not written by the real front end: %r" % (name, got)
    # and the binder reports what it did, so a silent writer is visible
    assert "stage6_attachment" in on.binder_diag


def test_the_switch_off_writes_none_of_them(off):
    """PIN: the switch is the only way in.  A relation that appears with the
    switch off is being written through a second, unswitched path."""
    for t in NEW:
        assert _edges(off, t) == [], "%s is written with the switch OFF" % t.name


# --------------------------------------------------------------------------
# 2. THE SLOTS VARY
# --------------------------------------------------------------------------

def test_the_slots_of_the_written_relations_take_more_than_one_value(on):
    """RULE (7.35): a relation written with ONE parameter value everywhere is a
    count with a constant attached, not a measurement, and nothing downstream
    can read a difference out of it.  Each relation's own slot is therefore
    required to vary on a real graph."""
    seen = {}
    for et in NEW:
        vals = set()
        for ep in _params(on, et):
            if ep is None:
                continue
            q = getattr(ep, "quantized", None)
            if q:
                vals.add(tuple(round(float(x), 6) for x in q))
        seen[et.name] = len(vals)
    for name, n in seen.items():
        assert n > 1, "%s carries %d distinct parameter values on a real graph" % (name, n)


# --------------------------------------------------------------------------
# 3. THE FRAME / ATTRIBUTE PARTITION, ON REAL NODES
# --------------------------------------------------------------------------

def test_every_node_the_front_end_emits_is_on_exactly_one_side(on):
    """RULE: the partition decides which nodes may ever carry a correspondence.
    It is pinned as a LIST in test_relation_frame_partition.py; here it is
    checked against the lens types the real front end actually emits, so a new
    lens type cannot slip in unpinned."""
    seen = Counter(lt for lt, _s in on.sigs.values())
    unknown = set(seen) - (FRAME_LENS_TYPES | ATTRIBUTE_LENS_TYPES)
    assert not unknown, "lens types emitted but in neither side: %r" % sorted(unknown)
    assert FRAME_LENS_TYPES.isdisjoint(ATTRIBUTE_LENS_TYPES)
    # every emitted type is on the side is_frame() says it is
    for lt in seen:
        assert is_frame(lt) == (lt in FRAME_LENS_TYPES)
    # the graph really contains both sides, or the check above is vacuous
    assert any(is_frame(lt) for lt in seen) and any(not is_frame(lt) for lt in seen)


# --------------------------------------------------------------------------
# 4. AMBIGUITY IS ABSENT, NOT A GUESS
# --------------------------------------------------------------------------

def test_an_ambiguous_attachment_is_written_absent_and_counted(on):
    """RULE (Amendment 2.4 / 7.35).  When the evidence does not pick a host the
    attachment is ABSENT and the binder COUNTS the refusal; it never guesses.

    The pin is written on the two things that are actually invariant, because
    the obvious third one is NOT: a feature may be attached to several hosts
    of one kind when the data NAMES it as a member of each -- that is a fact,
    not a rival guess (Amendment 3.1), and on this scene it is exactly what the
    CONTOUR_PEAK / PEAK_CHAIN memberships do.

    (i)  ambiguity happens on this scene and every refusal is counted, not
         silently dropped;
    (ii) every attachment that IS written carries a measured slot -- an
         unmeasured attachment would enter the walker's relation set and be
         compared as if it had been measured, which is the one way an
         ambiguous attachment could still reach the matcher.
    """
    from matching.relation_frame import has_measured_axis

    diag = on.binder_diag["stage6_attachment"]
    att = _edges(on, EdgeType.BOUNDARY_ATTACHMENT)
    assert att, "the scene must write attachments"

    outcomes = defaultdict(int)
    for k, v in diag.items():
        if "|" in k:
            outcomes[k.split("|")[-1]] += v
    assert outcomes.get("ambiguous", 0) > 0, (
        "the binder reports no ambiguity on a scene where ties must exist: %r" % diag)
    # every outcome the rule can reach is a COUNTED outcome, never a written
    # attachment: the ambiguous count is separate from the written edges
    assert outcomes.get("unambiguous", 0) > 0
    # (ii) nothing unmeasured is written
    for e in att:
        ep = on.subgraph.edge_params.get(tuple(e))
        assert ep is not None and has_measured_axis(ep), (
            "an attachment with no measured slot was written: %r" % (e,))


# --------------------------------------------------------------------------
# 5. THE CACHE KEY MOVES WITH A GRAPH-SIDE SWITCH
# --------------------------------------------------------------------------

def test_the_graph_cache_key_moves_when_a_graph_switch_moves(params):
    """PIN (7.28, in its end-to-end form): two graphs that differ must never
    share a cache key.  The key is read from the same function `graph_for`
    uses, and the two graphs are the two this module built."""
    _diag_on_path()
    import eth80_frontend as fe

    params(GRAPH_STAGE6_RELATIONS=False, BINDER_COMPLETE_POSITION_MAP=False)
    k_off = fe.frontend_source_digest()
    params(GRAPH_STAGE6_RELATIONS=True, BINDER_COMPLETE_POSITION_MAP=True)
    k_on = fe.frontend_source_digest()
    assert k_off != k_on
    # and the graphs the two keys name really are different
    assert len(_edges(off := graph(False), EdgeType.BOUNDARY_ATTACHMENT)) != \
        len(_edges(graph(True), EdgeType.BOUNDARY_ATTACHMENT))
    del off
