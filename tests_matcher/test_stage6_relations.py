"""[STAGE 6 / TRACK R] The measured relations: what each one holds, and when it
holds nothing.

Each pin names the rule, then shows it on the real code with hand-built
geometry whose answer is known in closed form.

  ATTACHMENT (perception/attachment.py)
    a feature's place on a boundary is arc position, distance and angle;
    the distance from the nearer end does not depend on the walking direction;
    two near-equal hosts are not a measurement;
    a host the data names needs no test, and two named hosts are not one fact;
    a ring has no end; a feature without an orientation gets no angle;
    a chain's tangent is its segment, a Contour's is its own mean orientation.
  POOLING (Binder._stage6_pool)
    one node at two places of one boundary has no one place on it.
  THE WRITERS (Binder._run_boundary_attachment / _successor / _illusory_inducer)
  THE ORDINAL is carried, never compared.
  THE MAP MERGE: the binder's own entries win.
  R5 (perception/complex_links.py): each contour end is linked at most once.
  THE BORDER ZONES (perception/contour_field.py).
  DETERMINISM across PYTHONHASHSEED.
  THE SWITCH: off, the graph is the old graph.
"""
from __future__ import annotations

import dataclasses
import math
import os
import random
import subprocess
import sys
from collections import defaultdict

import pytest

import core.schema as S
import parameters as P
from core.types import Detection, EdgeType, TypedSignature
from orchestration.binder import Binder
from perception import attachment as A
from perception import complex_links as CL

R = 10.0                      # one extraction scale, in position units
N_THETA = int(P.ORIENTATION_N_THETA)
ATT = "EDGE_BOUNDARY_ATTACHMENT"


# --------------------------------------------------------------------------
# hand-built hosts, features and detections
# --------------------------------------------------------------------------

def _host(key, pts, kind="PEAK_CHAIN", node=None, closed=False, tangent=None):
    return A.Host(kind=kind, key=key, node=node, pts=tuple(pts), closed=closed,
                  tangent=tangent)


def _feat(key, pos, tangent=None, kind="EndStopped", node=None, known=()):
    return A.Feature(kind=kind, key=key, node=node, pos=pos, tangent=tangent,
                     known=tuple(known))


def _sig(*q):
    return TypedSignature(kind="CANONICAL", class_="TEST", relation="T",
                          quantized=tuple(float(v) for v in q), population=())


def _det(k, alpha, q, rho=()):
    return Detection(k=k, source_layer=1, scale=0, sigma=_sig(*q), alpha=alpha,
                     z=1.0, t=0, rho=tuple(rho))


def _chain(alpha, members, closed=False):
    return _det("PEAK_CHAIN", alpha, (0, 0, 0, 1 if closed else 0, 0, 0),
                (("entity", alpha, 0, "+"),)
                + tuple(("member", a, i, "+") for i, a in enumerate(members)))


def _peak(alpha, theta_idx=0):
    return _det("CONTOUR_PEAK", alpha, (theta_idx, 0, 0, 0, 0, 0, 0, 0))


def _endstop(alpha, theta_bin=0):
    return _det("EndStopped", alpha, (theta_bin, 0, 0, 0))


def _contour(alpha, members, mean_theta_bin=0, closed=False):
    return _det("Contour", alpha,
                (len(members), mean_theta_bin, 0, 1 if closed else 0),
                (("entity", alpha, 0, "+"),)
                + tuple(("member", a, i, "+") for i, a in enumerate(members)))


class _Capture(Binder):
    """A Binder whose channels record their candidates instead of writing."""

    def __init__(self, higher=(), positions=None, tier=None, v1=()):
        self.seen = {}
        self._step_extraction_scale = R
        self.diagnostics = {"stage6_attachment": defaultdict(int)}
        self._v1_dets_this_step = tuple(d for d, _n in v1)
        self._v1_node_ids = {d: n for d, n in v1}
        self._stage6_higher_all = list(higher)
        self._step_anchor_positions = positions
        self._tier_anchor_positions = tier

    def _process_channel(self, edge_type, candidates):
        self.seen[edge_type] = list(candidates)


LINE = [(0.0, 0.0), (100.0, 0.0)]


# --------------------------------------------------------------------------
# the attachment measurement
# --------------------------------------------------------------------------

def test_a_features_place_on_a_boundary_is_arc_distance_and_angle():
    """RULE: the foot is the closest point of the host; the arc position is
    the foot's arc length over the host's length; the distance is in scales;
    the angle is between the feature's tangent and the host's, in [0, pi/2]."""
    f1 = _feat("f1", (25.0, 5.0), tangent=0.0)
    f2 = _feat("f2", (80.0, -3.0), tangent=math.pi / 2)
    atts, stats = A.attach([f1, f2], [_host("h", LINE)], R)
    a1, a2 = atts
    assert (a1.arc_pos, a1.arc_end_dist, a1.perp_dist, a1.tangent_angle) == \
        pytest.approx((0.25, 0.25, 0.5, 0.0))
    assert (a2.arc_pos, a2.arc_end_dist, a2.perp_dist, a2.tangent_angle) == \
        pytest.approx((0.80, 0.20, 0.3, math.pi / 2))
    assert stats[("EndStopped", "PEAK_CHAIN")][A.UNAMBIGUOUS] == 2
    # a feature beyond the end projects on the end vertex
    (a3,), _ = A.attach([_feat("f3", (104.0, 3.0))], [_host("h", LINE)], R)
    assert a3.arc_pos == 1.0 and a3.perp_dist == pytest.approx(0.5)


def test_the_distance_from_the_nearer_end_ignores_the_walking_direction():
    """RULE: a host's member order is the chain former's choice (a gauge).
    Walking it the other way turns arc_pos into 1 - arc_pos and changes
    nothing else: arc_end_dist, the distance and the angle are measurements."""
    f = _feat("f", (25.0, 5.0), tangent=0.3)
    (a,), _ = A.attach([f], [_host("h", LINE)], R)
    (b,), _ = A.attach([f], [_host("h", LINE[::-1])], R)
    assert b.arc_pos == pytest.approx(1.0 - a.arc_pos)
    assert (b.arc_end_dist, b.perp_dist, b.tangent_angle) == pytest.approx(
        (a.arc_end_dist, a.perp_dist, a.tangent_angle))


def test_two_near_equal_hosts_are_not_a_measurement():
    """RULE (the ambiguity rule): attached iff d1 <= D_MAX and d2 - d1 >=
    AMBIGUITY_MARGIN.  Otherwise NOTHING is written."""
    f = _feat("f", (50.0, 3.0))
    near = _host("near", LINE)                                  # d = 0.3
    rival = _host("rival", [(0.0, 7.0), (100.0, 7.0)])          # d = 0.4
    far = _host("far", [(0.0, 20.0), (100.0, 20.0)])            # d = 1.7
    atts, stats = A.attach([f], [near, rival], R)
    assert atts == [] and stats[("EndStopped", "PEAK_CHAIN")] == {
        A.KNOWN: 0, A.KNOWN_SEVERAL: 0, A.UNAMBIGUOUS: 0, A.BY_TANGENT: 0,
        A.AMBIGUOUS: 1, A.NO_HOST: 0}
    (a,), stats = A.attach([f], [near, far], R)
    assert a.host.key == "near" and stats[("EndStopped", "PEAK_CHAIN")][A.UNAMBIGUOUS] == 1
    # the margin is exactly AMBIGUITY_MARGIN: 0.3 against 0.8 passes
    edge = _host("edge", [(0.0, 11.0), (100.0, 11.0)])
    assert len(A.attach([f], [near, edge], R)[0]) == 1
    # beyond the admission radius: no host, whatever the margin
    lone = _feat("lone", (50.0, 10.0 * A.D_MAX + 0.5))
    atts, stats = A.attach([lone], [near], R)
    assert atts == [] and stats[("EndStopped", "PEAK_CHAIN")][A.NO_HOST] == 1


def test_ambiguity_is_judged_inside_one_host_kind():
    """RULE: a chain and a Contour are two different relations.  A Contour as
    near as the chain does not make the chain ambiguous."""
    f = _feat("f", (50.0, 3.0))
    chain = _host("c", LINE, kind="PEAK_CHAIN")
    contour = _host("k", [(0.0, 5.0), (100.0, 5.0)], kind="Contour")
    atts, _ = A.attach([f], [chain, contour], R)
    assert sorted(a.host.kind for a in atts) == ["Contour", "PEAK_CHAIN"]


def test_a_host_the_data_names_needs_no_test():
    """RULE: membership IDENTIFIES the host; the attachment MEASURES the place.
    A named host is used even when a rival is as near.  Several named hosts
    of one kind: see `test_several_named_hosts_are_several_facts`."""
    near = _host("near", LINE)
    rival = _host("rival", [(0.0, 6.0), (100.0, 6.0)])
    f = _feat("f", (50.0, 3.0), known=(("PEAK_CHAIN", "rival"),))
    (a,), stats = A.attach([f], [near, rival], R)
    assert a.host.key == "rival" and a.known and a.perp_dist == pytest.approx(0.3)
    assert stats[("EndStopped", "PEAK_CHAIN")][A.KNOWN] == 1
    # a named host is not bound by the admission radius: it is a fact
    g = _feat("g", (50.0, 60.0), known=(("PEAK_CHAIN", "near"),))
    (b,), _ = A.attach([g], [near], R)
    assert b.known and b.perp_dist == pytest.approx(6.0)
    # a named host that is not among the hosts (it has no polyline) is not a
    # fact about any host: the feature is then placed by geometry, like any other
    gone = _feat("h", (50.0, 3.0), known=(("PEAK_CHAIN", "missing"),))
    (c,), stats = A.attach([gone], [near], R)
    assert c.host.key == "near" and not c.known


def test_a_ring_has_no_end():
    """RULE: on a closed host the arc position is still an ordinal, but the
    distance from the nearer END is ABSENT: there is no end."""
    square = [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)]
    f = _feat("f", (20.0, -2.0))
    (ring,), _ = A.attach([f], [_host("r", square, closed=True)], R)
    (arc,), _ = A.attach([f], [_host("r", square, closed=False)], R)
    assert ring.arc_end_dist is None and ring.arc_pos == pytest.approx(20.0 / 160.0)
    assert arc.arc_end_dist == pytest.approx(20.0 / 120.0)
    # the closing side exists on the ring only
    g = _feat("g", (-2.0, 20.0))
    (on_ring,), _ = A.attach([g], [_host("r", square, closed=True)], R)
    assert on_ring.perp_dist == pytest.approx(0.2)


def test_a_feature_without_an_orientation_gets_no_angle():
    """RULE: ABSENT is not a number.  No tangent on the feature: no angle."""
    (a,), _ = A.attach([_feat("f", (50.0, 3.0), tangent=None)], [_host("h", LINE)], R)
    assert a.tangent_angle is None and a.perp_dist == pytest.approx(0.3)


def test_a_chains_tangent_is_its_segment_a_contours_is_its_own_orientation():
    """RULE (the host-tangent source).  A PEAK_CHAIN's vertices are peaks: the
    segment at the foot IS its tangent.  A Contour's vertices are grid cells,
    whose segments only take the grid's directions: its tangent is the
    Contour's own measured `mean_theta` slot."""
    pos = {"a": (0.0, 0.0), "b": (50.0, 0.0), "c": (50.0, 50.0)}
    chain = A.host_of(_chain("C", ["a", "b", "c"]), pos, 1, N_THETA)
    assert chain.tangent is None and not chain.closed
    f_low = _feat("f1", (25.0, 2.0), tangent=0.0)        # on the horizontal segment
    f_up = _feat("f2", (52.0, 25.0), tangent=0.0)        # on the vertical segment
    a_low, a_up = A.attach([f_low, f_up], [chain], R)[0]
    assert a_low.tangent_angle == pytest.approx(0.0)
    assert a_up.tangent_angle == pytest.approx(math.pi / 2)
    # the same polyline as a Contour whose mean orientation is bin 3
    contour = A.host_of(_contour("K", ["a", "b", "c"], mean_theta_bin=3), pos, 2, N_THETA)
    own = (3 + 0.5) * math.pi / N_THETA
    assert contour.tangent == pytest.approx(own)
    k_low, k_up = A.attach([f_low, f_up], [contour], R)[0]
    assert k_low.tangent_angle == pytest.approx(own)
    assert k_up.tangent_angle == pytest.approx(own)      # NOT the grid segment
    # one positioned vertex is not a boundary
    assert A.host_of(_chain("C1", ["a"]), pos, 3, N_THETA) is None


def test_attach_does_not_depend_on_the_order_it_is_handed_things():
    """RULE: determinism.  Shuffling hosts and features changes nothing."""
    rng = random.Random(7)
    hosts = [_host("h%d" % i, [(0.0, 30.0 * i), (100.0, 30.0 * i + rng.uniform(-5, 5))])
             for i in range(5)]
    feats = [_feat("f%d" % i, (rng.uniform(0, 100), rng.uniform(-10, 130)),
                   tangent=rng.uniform(0, math.pi)) for i in range(40)]

    def run(hs, fs):
        atts, stats = A.attach(fs, hs, R)
        return (sorted((a.feature.key, a.host.key, a.arc_pos, a.perp_dist,
                        a.tangent_angle) for a in atts), stats)

    base = run(hosts, feats)
    for seed in range(5):
        hs, fs = hosts[:], feats[:]
        random.Random(seed).shuffle(hs)
        random.Random(seed + 100).shuffle(fs)
        assert run(hs, fs) == base


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------

def test_one_node_at_two_places_has_no_one_place():
    """RULE (the pooling rule).  Several detections of one node on one host:
    an axis is written when their values share a bin or sit in ADJACENT bins
    (the mean), and is ABSENT otherwise."""
    pool = Binder._stage6_pool
    width = 0.5 / 8                                   # one arc_end_dist bin
    assert pool(ATT, "arc_end_dist", [0.01, 0.05]) == pytest.approx(0.03)        # same bin
    assert pool(ATT, "arc_end_dist", [0.01, width + 0.01]) == pytest.approx(
        0.01 + width / 2)                                                        # adjacent
    assert pool(ATT, "arc_end_dist", [0.01, 2 * width + 0.01]) is None           # two apart
    assert pool(ATT, "arc_end_dist", [0.01, 0.02, 0.45]) is None
    assert pool(ATT, "perp_dist", [0.30]) == 0.30                                # one value
    assert pool(ATT, "perp_dist", [None, 0.30, None]) == 0.30                    # None is not a value
    assert pool(ATT, "tangent_angle", [None, None]) is None
    assert pool(ATT, "tangent_angle", []) is None


# --------------------------------------------------------------------------
# the writers
# --------------------------------------------------------------------------

def _straight_chain():
    pos = {"p0": (0.0, 0.0), "p1": (50.0, 0.0), "p2": (100.0, 0.0), "e": (100.0, 4.0)}
    chain = _chain("C", ["p0", "p1", "p2"])
    higher = [(chain, 100), (_peak("p0"), 1), (_peak("p1"), 2), (_peak("p2"), 3),
              (_endstop("e"), 7)]
    return pos, higher


def test_the_attachment_writer_lays_one_edge_per_host_and_feature_node():
    """RULE: host -> feature, one edge per node pair, the slots quantised from
    the measurement and the ordinal carried beside them."""
    pos, higher = _straight_chain()
    b = _Capture(higher=higher, positions={}, tier=pos)
    b._run_boundary_attachment()
    cands = b.seen[EdgeType.BOUNDARY_ATTACHMENT]
    assert [(c[0], c[1]) for c in cands] == [(100, 1), (100, 2), (100, 3), (100, 7)]
    assert all(c[2] == 1.0 for c in cands)            # a fact, not an activation
    by = {c[1]: c[4] for c in cands}
    # member peaks: on the chain (distance 0), tangent along it, arc 0 / 0.5 / 1
    for node, arc in ((1, 0.0), (2, 0.5), (3, 1.0)):
        assert by[node].quantized == S.edge_quantise(ATT, {
            "arc_end_dist": min(arc, 1 - arc), "tangent_angle": 0.0, "perp_dist": 0.0})
        assert by[node].ordinal == pytest.approx(arc)
    # the terminator: beyond the end, 0.4 scales off.  Its angle is ABSENT:
    # truth does not support an EndStopped orientation (Amendment 3.2)
    assert A.feature_tangent(_endstop("e"), N_THETA) is None
    assert by[7].quantized == S.edge_quantise(ATT, {
        "arc_end_dist": 0.0, "tangent_angle": None, "perp_dist": 0.4})
    assert by[7].quantized != S.edge_quantise(ATT, {
        "arc_end_dist": 0.0, "tangent_angle": 0.0, "perp_dist": 0.4})
    assert by[7].ordinal == pytest.approx(1.0)
    d = b.diagnostics["stage6_attachment"]
    assert d["CONTOUR_PEAK|PEAK_CHAIN|known"] == 3
    assert d["EndStopped|PEAK_CHAIN|unambiguous"] == 1


def test_a_pooled_node_keeps_the_axes_that_agree_and_loses_the_ones_that_do_not():
    """RULE: the two ends of a chain share one node (same signature).  Its
    distance from the nearer end is 0 at both places: written.  Its ordinal
    is 0 at one and 1 at the other: ABSENT."""
    pos = {"p0": (0.0, 0.0), "p1": (50.0, 0.0), "p2": (100.0, 0.0)}
    higher = [(_chain("C", ["p0", "p1", "p2"]), 100),
              (_peak("p0"), 5), (_peak("p1"), 2), (_peak("p2"), 5)]
    b = _Capture(higher=higher, positions={}, tier=pos)
    b._run_boundary_attachment()
    by = {c[1]: c[4] for c in b.seen[EdgeType.BOUNDARY_ATTACHMENT]}
    assert sorted(by) == [2, 5]
    assert by[5].quantized == S.edge_quantise(ATT, {
        "arc_end_dist": 0.0, "tangent_angle": 0.0, "perp_dist": 0.0})
    assert by[5].ordinal is None and by[2].ordinal == pytest.approx(0.5)
    d = b.diagnostics["stage6_attachment"]
    assert d["pooled|node_pairs"] == 1 and d["pooled|ordinal_absent"] == 1
    assert d["pooled|arc_end_dist_absent"] == 0


def test_the_ordinal_is_carried_never_compared():
    """RULE: the arc position is a gauge quantity.  It rides on the edge and
    enters no equality, no hash, no repr and no similarity."""
    ep = S.edge_params(ATT, {"arc_end_dist": 0.2, "tangent_angle": 0.1, "perp_dist": 0.3})
    other = dataclasses.replace(ep, ordinal=0.8)
    assert ep.ordinal is None and other.ordinal == 0.8
    assert other == ep and hash(other) == hash(ep) and repr(other) == repr(ep)
    assert other.similarity(ep) == ep.similarity(ep) == 1.0
    assert "ordinal" not in repr(other)
    # and it is not a slot of the class
    assert [s.name for s in S.EDGE_UNIT_TABLE[ATT]] == [
        "arc_end_dist", "tangent_angle", "perp_dist"]


def test_the_successor_joins_consecutive_member_peaks():
    """RULE: next along the chain: the gap in scales and the UNSIGNED turn
    between the two tangents.  Undirected.  A node is not its own successor."""
    pos = {"p0": (0.0, 0.0), "p1": (6.0, 0.0), "p2": (6.0, 8.0), "q0": (50.0, 0.0),
           "q1": (56.0, 0.0)}
    higher = [(_chain("C", ["p0", "p1", "p2"]), 100),
              (_peak("p0", 0), 3), (_peak("p1", 0), 2), (_peak("p2", 6), 1),
              (_chain("D", ["q0", "q1"]), 101), (_peak("q0", 0), 9), (_peak("q1", 0), 9)]
    b = _Capture(higher=higher, positions={}, tier=pos)
    b._run_boundary_successor()
    cands = b.seen[EdgeType.BOUNDARY_SUCCESSOR]
    cls = "EDGE_BOUNDARY_SUCCESSOR"
    assert [(c[0], c[1]) for c in cands] == [(1, 2), (2, 3)]      # (min, max) keys
    by = {(c[0], c[1]): c[4] for c in cands}
    assert by[(2, 3)].quantized == S.edge_quantise(cls, {"arc_gap": 0.6, "turn": 0.0})
    assert by[(1, 2)].quantized == S.edge_quantise(cls, {"arc_gap": 0.8, "turn": math.pi / 2})
    assert b.diagnostics["stage6_attachment"]["successor|same_node"] == 1
    assert EdgeType.BOUNDARY_SUCCESSOR in __import__("core.types").types.UNDIRECTED_EDGE_TYPES


def test_a_closed_chain_has_a_closing_successor():
    """RULE: a ring's last member is followed by its first."""
    pos = {"a": (0.0, 0.0), "b": (8.0, 0.0), "c": (8.0, 8.0)}
    members = ["a", "b", "c"]
    peaks = [(_peak("a"), 1), (_peak("b"), 2), (_peak("c"), 3)]
    open_ = _Capture(higher=[(_chain("C", members), 100)] + peaks, positions={}, tier=pos)
    open_._run_boundary_successor()
    ring = _Capture(higher=[(_chain("C", members, closed=True), 100)] + peaks,
                    positions={}, tier=pos)
    ring._run_boundary_successor()
    assert [(c[0], c[1]) for c in open_.seen[EdgeType.BOUNDARY_SUCCESSOR]] == [(1, 2), (2, 3)]
    assert [(c[0], c[1]) for c in ring.seen[EdgeType.BOUNDARY_SUCCESSOR]] == [(1, 2), (1, 3), (2, 3)]


def test_an_illusory_contour_is_tied_to_each_of_its_inducers():
    """RULE: the inducers are named in the completion's own `rho`: nothing to
    guess.  `gap` is their separation in scales; `alignment` is the angle the
    LENS tested when it paired them: the inducer's slot (bin centre) turned by
    90 degrees, against the joining line.  The slot is the terminating bar's
    orientation, so the completion runs ACROSS the bars; `alignment` is how
    far from square that is."""
    pos = {"e1": (0.0, 0.0), "e2": (30.0, 0.0)}
    e1, e2 = _endstop("e1", 12), _endstop("e2", 13)
    ic = _det("IllusoryContour", "ic", (0, 0, 0),
              (("entity", "ic", 0, "+"), ("inducer_1", "e1", 0, "+"),
               ("inducer_2", "e2", 0, "+")))
    b = _Capture(higher=[(ic, 50), (e1, 7), (e2, 8)], positions={}, tier=pos)
    b._run_illusory_inducer()
    cands = b.seen[EdgeType.ILLUSORY_INDUCER]
    cls = "EDGE_ILLUSORY_INDUCER"
    assert [(c[0], c[1]) for c in cands] == [(50, 7), (50, 8)]
    half = 0.5 * math.pi / N_THETA
    # bar at bin 12 (90 degrees): square to the horizontal line but for the
    # lens' half-bin; bar at bin 13: one bin further
    for c, align in zip(cands, (half, math.pi / N_THETA + half)):
        assert c[4].quantized == S.edge_quantise(cls, {"gap": 3.0, "alignment": align})


# --------------------------------------------------------------------------
# the map merge
# --------------------------------------------------------------------------

def test_the_binders_own_positions_win_the_merge(params):
    """RULE (direction 1).  The lens tier registers positions after the
    binder's map was copied.  Completing the map ADDS the tier's anchors; an
    anchor the binder already had keeps the binder's value.  With the switch
    off the step's map is untouched, and only the stage-6 writers read the
    completed view."""
    tier = {"a": (9.0, 9.0), "p": (2.0, 2.0)}
    params(BINDER_COMPLETE_POSITION_MAP=False)
    b = _Capture(positions={"a": (1.0, 1.0)})
    b.note_tier_positions(tier)
    assert b._step_anchor_positions == {"a": (1.0, 1.0)}
    assert b._stage6_positions() == {"a": (1.0, 1.0), "p": (2.0, 2.0)}
    params(BINDER_COMPLETE_POSITION_MAP=True)
    b = _Capture(positions={"a": (1.0, 1.0)})
    b.note_tier_positions(tier)
    assert b._step_anchor_positions == {"a": (1.0, 1.0), "p": (2.0, 2.0)}
    assert tier == {"a": (9.0, 9.0), "p": (2.0, 2.0)}      # the caller's map is not edited
    # a binder that was handed no map at all
    b = _Capture(positions=None)
    b.note_tier_positions(tier)
    assert b._step_anchor_positions == tier and b._step_anchor_positions is not tier


# --------------------------------------------------------------------------
# R5: the CurveComplex link rule
# --------------------------------------------------------------------------

def _seg(key, x0, y0, x1, y1):
    return CL.contour_ends(key, [(x0, y0), (x1, y1)], None)


def test_gap_and_bend_bound_a_link():
    """RULE: a link needs gap <= GAP_MAX scales AND bend <= BEND_MAX."""
    a = _seg(0, 0, 0, 20, 0)

    def linked(b):
        comps, _ = CL.group([a, b], [], R)
        return len(comps) == 1

    assert linked(_seg(1, 35, 0, 55, 0))                       # gap 1.5, straight on
    assert linked(_seg(1, 20 + 10 * CL.GAP_MAX, 0, 60, 0))     # gap exactly the bound
    assert not linked(_seg(1, 20 + 10 * CL.GAP_MAX + 1, 0, 60, 0))
    up = math.radians(44.0)
    assert linked(_seg(1, 20, 0, 20 + 20 * math.cos(up), 20 * math.sin(up)))
    up = math.radians(46.0)
    assert not linked(_seg(1, 20, 0, 20 + 20 * math.cos(up), 20 * math.sin(up)))
    assert not linked(_seg(1, 20, 0, 20, 20))                  # a right angle
    # a parallel neighbour is near but not a continuation
    assert not linked(_seg(1, 0, 8, 20, 8))
    gap, bend = CL.link_evidence(a, 1, _seg(1, 35, 0, 55, 0), 0, R)
    assert (gap, bend) == pytest.approx((1.5, 0.0))


def test_each_contour_end_is_linked_at_most_once():
    """RULE: degree <= 2.  Three contours leaving one end: the best one links,
    the others do not, and the complex is a path, not a star."""
    a = _seg(0, 0, 0, 20, 0)
    straight = _seg(1, 26, 0, 46, 0)
    up = _seg(2, 26, 3, 46, 9)
    down = _seg(3, 26, -3, 46, -9)
    comps, stats = CL.group([a, straight, up, down], [], R)
    assert stats["candidates"] == 3 and stats["links"] == 1
    assert stats["max_contour_degree"] == 1
    assert comps == [([0, 1], []), ([2], []), ([3], [])]


def test_complexes_are_paths_or_cycles_on_random_layouts():
    """RULE: whatever the layout, no Contour takes more than two links and no
    complex has more links than members (a path has n - 1, a cycle n)."""
    for seed in range(40):
        rng = random.Random(seed)
        cs = []
        for k in range(14):
            x, y, th = rng.uniform(0, 120), rng.uniform(0, 120), rng.uniform(0, 2 * math.pi)
            L = rng.uniform(8, 30)
            cs.append(_seg(k, x, y, x + L * math.cos(th), y + L * math.sin(th)))
        comps, stats = CL.group(cs, [], R)
        assert stats["max_contour_degree"] <= 2
        assert sorted(k for c, _j in comps for k in c) == list(range(14))
        assert stats["links"] <= sum(len(c) for c, _j in comps if len(c) > 1)


def test_a_junction_takes_the_ends_near_it():
    """RULE: a junction within GAP_MAX of an end holds that end, and may hold
    any number.  An end a junction holds is not linked to another contour.
    Two junctions about equally near: neither takes it."""
    stem = _seg(0, 50, 0, 50, 40)
    left = _seg(1, 0, 50, 40, 50)
    right = _seg(2, 60, 50, 100, 50)
    comps, stats = CL.group([stem, left, right], [("J", (50.0, 50.0))], R)
    assert comps == [([0, 1, 2], ["J"])]
    assert stats["ends_on_junctions"] == 3 and stats["links"] == 0
    # without the junction: left and right continue each other (gap 2.0 > 1.8: no)
    comps, stats = CL.group([left, _seg(2, 55, 50, 100, 50)], [], R)
    assert comps == [([1, 2], [])] and stats["links"] == 1
    # the same pair with a junction between them: the junction holds both ends
    comps, stats = CL.group([left, _seg(2, 55, 50, 100, 50)], [("J", (47.0, 50.0))], R)
    assert comps == [([1, 2], ["J"])] and stats["links"] == 0
    # two junctions equally near one end: ambiguous, the end stays free
    comps, stats = CL.group([left], [("J1", (45.0, 55.0)), ("J2", (45.0, 45.0))], R)
    assert comps == [([1], [])] and stats["ends_junction_ambiguous"] == 1
    # a junction that holds nothing is in no complex
    comps, _ = CL.group([left], [("far", (500.0, 500.0))], R)
    assert comps == [([1], [])]


def test_the_grouping_reads_no_weight_and_no_order():
    """RULE: one step of geometry.  The grouping takes positions only (there
    is no graph to read a weight from), and does not depend on input order."""
    rng = random.Random(3)
    cs = []
    for k in range(12):
        x, y = rng.uniform(0, 100), rng.uniform(0, 100)
        th = rng.uniform(0, 2 * math.pi)
        cs.append(_seg(k, x, y, x + 15 * math.cos(th), y + 15 * math.sin(th)))
    js = [("j%d" % i, (rng.uniform(0, 100), rng.uniform(0, 100))) for i in range(3)]
    base = CL.group(cs, js, R)
    for seed in range(5):
        c2, j2 = cs[:], js[:]
        random.Random(seed).shuffle(c2)
        random.Random(seed).shuffle(j2)
        assert CL.group(c2, j2, R) == base
    import inspect
    assert list(inspect.signature(CL.group).parameters)[:3] == ["contours", "junctions", "scale"]


def test_a_one_vertex_contour_heads_both_ways_along_its_orientation():
    """RULE: a single cell has no segment; it continues along its own mean
    orientation, either way, and is one place (never linked twice to the same
    partner)."""
    dot = CL.contour_ends("dot", [(50.0, 0.0)], 0.0)            # horizontal
    assert dot.ends[0] == dot.ends[1] and dot.out == ((-1.0, -0.0), (1.0, 0.0))
    left = _seg("l", 0, 0, 35, 0)
    right = _seg("r", 65, 0, 100, 0)
    comps, stats = CL.group([left, dot, right], [], R)
    assert comps == [(["dot", "l", "r"], [])] and stats["links"] == 2
    assert CL.contour_ends("x", [(1.0, 1.0)], None) is None     # nothing to go on
    assert CL.contour_ends("x", [], 0.0) is None


def test_the_lens_groups_by_geometry_under_the_switch():
    """RULE: the CurveComplex lens' components come from positions, one
    Contour per node, keyed by node id."""
    from lenses.v4 import CurveComplexLens
    pos = {"a0": (0.0, 0.0), "a1": (20.0, 0.0), "b0": (35.0, 0.0), "b1": (55.0, 0.0),
           "c0": (0.0, 60.0), "c1": (20.0, 60.0)}
    dets = {11: _contour("a0", ["a0", "a1"]), 12: _contour("b0", ["b0", "b1"]),
            13: _contour("c0", ["c0", "c1"])}
    kinds = {11: "Contour", 12: "Contour", 13: "Contour"}
    comps = CurveComplexLens._geometric_components(dets, kinds, pos, R)
    assert comps == [{11, 12}, {13}]
    # a contour with no positioned member is its own component
    dets[14] = _contour("zz", ["zz"])
    kinds[14] = "Contour"
    comps = CurveComplexLens._geometric_components(dets, kinds, pos, R)
    assert {14} in comps and {11, 12} in comps


# --------------------------------------------------------------------------
# the border zones
# --------------------------------------------------------------------------

def test_the_frame_is_cut_into_nested_zones_of_border_distance():
    """RULE (per-channel border discard): a channel is used where its OWN
    kernel fits.  The innermost zone has every channel; each step outwards
    drops the channels that no longer fit."""
    import numpy as np
    import perception.contour_field as ctf
    zones = ctf._channel_zones((56, 37, 24, 16, 16))
    assert [(lo, chans) for lo, _hi, chans in zones] == [
        (56, [0, 1, 2, 3, 4]), (37, [1, 2, 3, 4]), (24, [2, 3, 4]), (16, [3, 4])]
    assert [hi for _lo, hi, _c in zones][1:] == [56, 37, 24]
    assert ctf._channel_zones((42, 42, 42)) == [(42, 1 << 30, [0, 1, 2])]
    # a channel is removed on the WHOLE frame, so no band edge is left to read
    a = np.ones((2, 3, 4, 4), dtype=np.float32)
    only = ctf._only_channels(a, [1])
    assert only[:, 1].min() == 1.0 and only[:, 0].max() == 0.0 and only[:, 2].max() == 0.0
    assert a.min() == 1.0                                   # the input is not edited
    assert ctf._only_channels(a, [0, 1, 2]) is a


def test_each_channel_has_its_own_kernel_support():
    """RULE: the bank pads every kernel to the largest one; a channel's own
    support is the extent of its non-zero block (56 / 37 / 24 / 15 / 10 px at
    256 px), and its margin is that, never below the extraction floor."""
    import perception.contour_field as ctf
    bank = ctf._get_bank(ctf.front_end_scale(256, 256))
    assert ctf._channel_half_supports(bank) == [56, 37, 24, 15, 10]
    assert ctf._channel_margins(bank, 256, 256) == [56, 37, 24, 16, 16]
    assert ctf._border_margin_px(bank) == 56


def test_the_border_switch_is_part_of_the_field_fingerprint(params):
    """RULE: no cache hands one setting the other's field."""
    import perception.contour_field as ctf
    params(FIELD_BORDER_DISCARD_PER_CHANNEL=False)
    off = ctf.field_fingerprint()
    params(FIELD_BORDER_DISCARD_PER_CHANNEL=True)
    assert ctf.field_fingerprint() != off


# --------------------------------------------------------------------------
# determinism across PYTHONHASHSEED
# --------------------------------------------------------------------------

def _scenario_digest() -> str:
    """Candidates of the three writers and the R5 grouping on a scene whose
    anchors are STRINGS (so their hash, and any set order, moves with
    PYTHONHASHSEED).  Printed by the subprocess of the pin below."""
    rng = random.Random(11)
    pos, higher, node = {}, [], 0
    chains = []
    for c in range(6):
        x0, y0 = rng.uniform(0, 200), rng.uniform(0, 200)
        th = rng.uniform(0, math.pi)
        members = []
        for i in range(rng.randint(2, 6)):
            a = "c%d_p%d" % (c, i)
            pos[a] = (x0 + 7.0 * i * math.cos(th), y0 + 7.0 * i * math.sin(th))
            members.append(a)
            node += 1
            higher.append((_peak(a, rng.randint(0, 11)), 1 + rng.randint(0, 30)))
        chains.append(members)
        higher.append((_chain("chain%d" % c, members), 100 + c))
    for i in range(25):
        a = "e%d" % i
        pos[a] = (rng.uniform(0, 200), rng.uniform(0, 200))
        higher.append((_endstop(a, rng.randint(0, N_THETA - 1)), 200 + rng.randint(0, 12)))
    ic = _det("IllusoryContour", "ic", (0, 0, 0),
              (("entity", "ic", 0, "+"), ("inducer_1", "e0", 0, "+"),
               ("inducer_2", "e1", 0, "+")))
    higher.append((ic, 300))
    # hand the writer its detections through a SET, as the pipeline does
    higher = list({(d, n) for d, n in higher})
    b = _Capture(higher=higher, positions={}, tier=pos)
    b._run_boundary_attachment()
    b._run_boundary_successor()
    b._run_illusory_inducer()
    out = []
    for et in (EdgeType.BOUNDARY_ATTACHMENT, EdgeType.BOUNDARY_SUCCESSOR,
               EdgeType.ILLUSORY_INDUCER):
        for c in b.seen[et]:
            out.append((et.name, c[0], c[1], c[2], tuple(c[4].quantized), c[4].ordinal))
    cs = [CL.contour_ends("k%d" % i, [pos[m] for m in ms], None)
          for i, ms in enumerate(chains)]
    comps, stats = CL.group(list({c for c in cs}), [("j", (100.0, 100.0))], R)
    return repr((out, sorted(b.diagnostics["stage6_attachment"].items()), comps,
                 sorted(stats.items())))


def test_the_same_answer_under_every_hash_seed():
    """RULE: determinism.  Anchors are strings and the writer is handed a
    set; the candidates, their order, the counters and the R5 complexes are
    byte for byte the same under three hash seeds."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); "
            "sys.path.append(%r); import test_stage6_relations as T; "
            "print(T._scenario_digest())" % (
                root, here, os.path.join(root, "diagnostics", "_export_shims")))
    outs = []
    for seed in ("0", "1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, cwd=root)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(r.stdout)
    assert len(outs[0]) > 500
    assert outs[0] == outs[1] == outs[2]
    assert outs[0].strip() == _scenario_digest()


# --------------------------------------------------------------------------
# the switch
# --------------------------------------------------------------------------

def _disk_image(size=160):
    import numpy as np
    yy, xx = np.mgrid[0:size, 0:size]
    disk = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) <= (size * 0.22) ** 2
    img = np.where(disk, 215, 60).astype(np.uint8)
    return np.dstack([img, img, img])


def test_switch_off_is_the_old_graph_and_switch_on_adds_only_the_relations(params):
    """RULE: the relations are one grouped switch.  Off: none of the three
    edge types exists.  On: they appear; every node that is not a CurveComplex
    is the same node, and every other edge among those nodes is the same edge
    (the CurveComplex lens regroups under the same switch, R5)."""
    here = os.path.dirname(os.path.abspath(__file__))
    diag = os.path.join(os.path.dirname(here), "diagnostics")
    if diag not in sys.path:
        sys.path.insert(0, diag)
    import eth80_frontend as fe

    new = {EdgeType.BOUNDARY_ATTACHMENT, EdgeType.BOUNDARY_SUCCESSOR,
           EdgeType.ILLUSORY_INDUCER}
    img = _disk_image()
    params(GRAPH_STAGE6_RELATIONS=False, BINDER_COMPLETE_POSITION_MAP=False)
    off = fe.build_graph(img, name="disk", record_co_located=False)
    params(GRAPH_STAGE6_RELATIONS=True)
    on = fe.build_graph(img, name="disk", record_co_located=False)

    assert not [e for e in off.subgraph.edges if e[2] in new]
    assert "stage6_attachment" not in off.binder_diag
    got = {t.name: sum(1 for e in on.subgraph.edges if e[2] == t) for t in new}
    assert got["BOUNDARY_ATTACHMENT"] > 0 and got["BOUNDARY_SUCCESSOR"] > 0

    def plain(g):
        cc = {n for n, (lt, _s) in g.sigs.items() if lt == "CurveComplex"}
        nodes = {n: (lt, s.serialize()) for n, (lt, s) in g.sigs.items() if n not in cc}
        edges = {(e[0], e[1], e[2]): repr(g.subgraph.edge_params.get((e[0], e[1], e[2])))
                 for e in g.subgraph.edges
                 if e[0] not in cc and e[1] not in cc and e[2] not in new}
        return nodes, edges

    assert plain(on) == plain(off)
    # an attachment edge runs host -> feature and carries its ordinal
    lens = {n: lt for n, (lt, _s) in on.sigs.items()}
    hosts = {lens[e[0]] for e in on.subgraph.edges if e[2] == EdgeType.BOUNDARY_ATTACHMENT}
    assert hosts <= {"PEAK_CHAIN", "Contour"}
    eps = [on.subgraph.edge_params[(e[0], e[1], e[2])] for e in on.subgraph.edges
           if e[2] == EdgeType.BOUNDARY_ATTACHMENT]
    assert any(getattr(ep, "ordinal", None) is not None for ep in eps)


# --------------------------------------------------------------------------
# Amendment 2: the feature's own orientation, tangent agreement, terminators
# --------------------------------------------------------------------------

def test_the_endstopped_slot_is_the_bars_orientation_not_a_carrier():
    """RULE: the ENDSTOPPED `theta` slot is the bank channel whose response
    ends.  On isolated bars that is the bar (a median 7.5 degrees from it,
    none beyond 67.5), so what the detection says is the slot AS IT IS: a
    second 90-degree turn is a defect."""
    for b in (0, 5, 12, 23):
        assert A.raw_orientation(_endstop("e", b), N_THETA) == pytest.approx(
            b * math.pi / N_THETA)


def test_an_orientation_truth_does_not_support_is_absent_and_the_place_stays():
    """RULE (Amendment 3.2): a kind's own orientation is used only if renderer
    truth supports it.  EndStopped (a leak channel on a boundary: 9 % within
    15 degrees), ContourFragment (its chord is the boundary normal by
    construction) and Corner (24 % within 15 degrees of the true bisector)
    say something, and it is not used: the angle is ABSENT.  Where the feature
    sits on its host is still measured.  A fabricated angle is worse than
    none."""
    assert A.TANGENT_KINDS == ("CONTOUR_PEAK", "Orientation")
    frag = _det("ContourFragment", "region", (4, 16, 7, 2),
                (("entity", "region", 0, "+"), ("fragment", 0, 0, "+")))
    object.__setattr__(frag, "_frag_mid", (25.0, 4.0))
    object.__setattr__(frag, "_frag_tangent", 0.25)
    corner = Detection(
        k="Corner", source_layer=1, scale=0, alpha="c", z=1.0, t=0, rho=(),
        sigma=TypedSignature(kind="CANONICAL", class_="CORNER", relation="T",
                             quantized=(4.0, 3.0, 2.0, 1.0), population=()))
    for det in (_endstop("e", 5), frag, corner):
        assert A.raw_orientation(det, N_THETA) is not None      # it does say something
        assert A.feature_tangent(det, N_THETA) is None          # and it is not used
    for det in (_peak("p", 3), _det("Orientation", "o", (3, 0, 0))):
        assert A.feature_tangent(det, N_THETA) == A.raw_orientation(det, N_THETA)
    hosts, feats = A.hosts_and_features(
        [(_chain("C", ["a", "b"]), 1), (_endstop("e", 5), 2), (frag, 3)],
        {"a": (0.0, 0.0), "b": (100.0, 0.0), "e": (50.0, 3.0), "region": (500.0, 500.0)},
        N_THETA)
    atts, _ = A.attach(feats, hosts, R)
    got = {a.feature.kind: a for a in atts}
    assert got["EndStopped"].tangent_angle is None
    assert (got["EndStopped"].arc_pos, got["EndStopped"].perp_dist) == pytest.approx((0.5, 0.3))
    assert got["ContourFragment"].tangent_angle is None
    assert (got["ContourFragment"].arc_pos, got["ContourFragment"].perp_dist) == pytest.approx(
        (0.25, 0.4))
    # ... and a kind with no usable orientation cannot be told apart by it
    flat = _host("flat", [(0.0, 0.0), (100.0, 0.0)])
    steep = _host("steep", [(50.0, -40.0), (50.0, 40.0)])
    hosts2, feats2 = A.hosts_and_features(
        [(_endstop("e", 0), 2)], {"e": (53.0, 3.0)}, N_THETA)
    atts2, stats2 = A.attach(feats2, [flat, steep], R)
    assert atts2 == [] and stats2[("EndStopped", "PEAK_CHAIN")][A.AMBIGUOUS] == 1


def test_a_corners_orientation_is_its_bisector():
    """RULE: two arms are not a reason for silence.  What a Corner says its
    direction is: its BISECTOR, the CORNER class' own `orient` slot (bin
    centre of 12 over [0, pi)).  A junction-class Corner with two arms:
    ref_theta + gap0 / 2; with three arms there is no one direction.
    (Whether that is USED is truth's call: see the pin on TANGENT_KINDS.)"""
    def corner(cls, q):
        sig = TypedSignature(kind="CANONICAL", class_=cls, relation="T",
                             quantized=tuple(float(v) for v in q), population=())
        return Detection(k="Corner", source_layer=1, scale=0, sigma=sig,
                         alpha="c", z=1.0, t=0, rho=())

    assert A.raw_orientation(corner("CORNER", (4, 3, 2, 1)), N_THETA) == pytest.approx(
        3.5 * math.pi / 12.0)
    ref, gap = 2.5 * 2 * math.pi / 24, 1.5 * 2 * math.pi / 8
    assert A.raw_orientation(corner("JUNCTION", (2, 2, 0, 0, 1)), N_THETA) == pytest.approx(
        (ref + gap / 2) % math.pi)
    assert A.raw_orientation(corner("JUNCTION", (3, 2, 0, 0, 1)), N_THETA) is None
    assert A.raw_orientation(corner("TEST", (4, 3, 2, 1)), N_THETA) is None
    # on its own chain the bisector-to-arm angle is (pi - turn) / 2
    square = [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0)]
    from lenses.corner_lens import _turn_and_bisector
    turn, bisector = _turn_and_bisector(*square)
    f = _feat("c", (40.0, 0.0), tangent=bisector, kind="Corner",
              known=(("PEAK_CHAIN", "h"),))
    (a,), _ = A.attach([f], [_host("h", square)], R)
    assert a.tangent_angle == pytest.approx((math.pi - abs(turn)) / 2.0)


def test_a_fragment_carries_its_own_place_and_chord_direction():
    """RULE: the ContourFragment signature keeps the tangent only as an
    UNSIGNED difference from the radial angle, so theta and tvr leave two
    candidates; and the fragment's anchor is its parent contour's region.  The
    lens measures both the chord direction and the midpoint: they ride on the
    detection, outside its identity."""
    from perception.polar import polar_boundary_fragments

    def line(x0, y0, x1, y1, n=40):
        return {"polyline": [(x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1))
                             for i in range(n)]}

    # a horizontal contour above the object's centre (the centre is the
    # centroid of both contours, at (50, 100))
    frs, _meta = polar_boundary_fragments([line(0, 0, 100, 0), line(0, 200, 100, 200)])
    mine = [f for f in frs if f["contour"] == 0]
    assert len(mine) >= 2
    signs = set()
    for f in mine:
        assert A.axial_difference(f["tangent"], 0.0) == pytest.approx(0.0, abs=1e-9)
        plus = A.axial_difference(f["theta"] + f["tvr_c"], 0.0)
        minus = A.axial_difference(f["theta"] - f["tvr_c"], 0.0)
        assert min(plus, minus) == pytest.approx(0.0, abs=1e-9)
        if abs(plus - minus) > 1e-6:
            signs.add(+1 if plus < minus else -1)
    assert signs == {+1, -1}          # theta + tvr is right for some, wrong for others

    frag = _det("ContourFragment", "region", (4, 16, 7, 2),
                (("entity", "region", 0, "+"), ("fragment", 0, 0, "+")))
    twin = _det("ContourFragment", "region", (4, 16, 7, 2),
                (("entity", "region", 0, "+"), ("fragment", 0, 0, "+")))
    pos = {"region": (500.0, 500.0)}
    assert A.raw_orientation(frag, N_THETA) is None                    # nothing carried
    assert A.feature_position(frag, pos) == (500.0, 500.0)
    object.__setattr__(frag, "_frag_mid", (25.0, 4.0))
    object.__setattr__(frag, "_frag_tangent", 0.25)
    assert A.raw_orientation(frag, N_THETA) == 0.25
    assert A.feature_position(frag, pos) == (25.0, 4.0)                # not the anchor
    assert frag == twin and hash(frag) == hash(twin)                   # the same detection
    # two fragments of one contour share the anchor: the fragment index is
    # part of the feature's identity
    other = _det("ContourFragment", "region", (4, 17, 7, 2),
                 (("entity", "region", 0, "+"), ("fragment", 1, 0, "+")))
    assert A.feature_key(frag) == ("region", 0) != A.feature_key(other)
    hosts, feats = A.hosts_and_features(
        [(_chain("C", ["a", "b"]), 1), (frag, 2)],
        dict(pos, a=(0.0, 0.0), b=(100.0, 0.0)), N_THETA)
    (a,), _ = A.attach(feats, hosts, R)
    assert (a.arc_pos, a.perp_dist) == pytest.approx((0.25, 0.4))


def test_a_feature_attaches_to_the_near_equidistant_host_it_parallels():
    """RULE (tangent agreement).  Where distance cannot decide, a feature that
    has a tangent attaches to the contender it parallels (within PARALLEL_MAX)
    when no other contender is nearly as parallel (PARALLEL_MARGIN).  Otherwise
    nothing is written."""
    flat = _host("flat", [(0.0, 0.0), (100.0, 0.0)])
    steep = _host("steep", [(50.0, -40.0), (50.0, 40.0)])        # crosses it
    key = ("Orientation", "PEAK_CHAIN")
    p = (53.0, 3.0)                                              # 0.3 from each
    (a,), stats = A.attach([_feat("f", p, tangent=0.05, kind="Orientation")], [flat, steep], R)
    assert a.host.key == "flat" and a.by_tangent and not a.known
    assert a.tangent_angle == pytest.approx(0.05) and stats[key][A.BY_TANGENT] == 1
    (b,), _ = A.attach([_feat("f", p, tangent=math.pi / 2 - 0.05, kind="Orientation")], [flat, steep], R)
    assert b.host.key == "steep" and b.by_tangent
    # parallel to neither: 45 degrees from both
    atts, stats = A.attach([_feat("f", p, tangent=math.pi / 4, kind="Orientation")], [flat, steep], R)
    assert atts == [] and stats[key][A.AMBIGUOUS] == 1
    # no tangent: nothing to agree with
    assert A.attach([_feat("f", p, tangent=None, kind="Orientation")], [flat, steep], R)[0] == []
    # the distance-only rule, for the before/after report
    assert A.attach([_feat("f", p, tangent=0.05, kind="Orientation")], [flat, steep], R, use_tangent=False)[0] == []
    # two PARALLEL near-equidistant hosts: the tangent cannot tell them apart
    twin = _host("twin", [(0.0, 6.0), (100.0, 6.0)])
    atts, stats = A.attach([_feat("f", (50.0, 3.0), tangent=0.0, kind="Orientation")], [flat, twin], R)
    assert atts == [] and stats[key][A.AMBIGUOUS] == 1
    # the rival must be a full PARALLEL_MARGIN less parallel
    def crossing(alpha):
        return _host("x", [(53.0 - 40 * math.cos(alpha), -40 * math.sin(alpha)),
                           (53.0 + 40 * math.cos(alpha), 40 * math.sin(alpha))])

    f = _feat("f", (53.0, 2.0), tangent=0.0, kind="Orientation")
    assert A.attach([f], [flat, crossing(0.2)], R)[0] == []          # 11.5 degrees apart
    (c,), _ = A.attach([f], [flat, crossing(0.4)], R)                # 22.9 degrees apart
    assert c.host.key == "flat" and c.by_tangent
    # a contender whose own tangent cannot be read cannot be ruled out
    dot = A.Host(kind="PEAK_CHAIN", key="dot", node=None,
                 pts=((53.0, 5.0), (53.0, 5.0), (60.0, 30.0)), closed=False)
    atts, stats = A.attach([_feat("f", (53.0, 3.0), tangent=0.0, kind="Orientation")], [flat, dot], R)
    assert atts == [] and stats[key][A.AMBIGUOUS] == 1


def test_several_named_hosts_are_several_facts():
    """RULE (Amendment 3.1): a cell where the cell graph branches is listed by
    every Contour that ends on it.  Recorded memberships are facts, not rival
    guesses: the feature is attached to EACH host that names it and measured
    on each.  (The ambiguity rule is about a guess.)"""
    flat = _host("flat", [(50.0, 0.0), (100.0, 0.0)], kind="Contour", tangent=0.0)
    steep = _host("steep", [(50.0, 0.0), (50.0, 40.0)], kind="Contour",
                  tangent=math.pi / 2)
    third = _host("third", [(0.0, 0.0), (50.0, 0.0)], kind="Contour", tangent=0.0)
    named = (("Contour", "flat"), ("Contour", "steep"), ("Contour", "third"))
    f = _feat("f", (50.0, 0.0), tangent=0.1, kind="Orientation", known=named)
    atts, stats = A.attach([f], [flat, steep, third], R)
    assert [a.host.key for a in atts] == ["flat", "steep", "third"]
    assert all(a.known and not a.by_tangent and a.perp_dist == 0.0 for a in atts)
    assert [a.arc_end_dist for a in atts] == [0.0, 0.0, 0.0]        # the end of each
    assert [a.tangent_angle for a in atts] == pytest.approx([0.1, math.pi / 2 - 0.1, 0.1])
    assert stats[("Orientation", "Contour")] == {
        A.KNOWN: 0, A.KNOWN_SEVERAL: 1, A.UNAMBIGUOUS: 0, A.BY_TANGENT: 0,
        A.AMBIGUOUS: 0, A.NO_HOST: 0}
    # a host named twice is one host
    twice = _feat("g", (50.0, 0.0), tangent=0.1, kind="Orientation",
                  known=(("Contour", "flat"), ("Contour", "flat")))
    (only,), stats = A.attach([twice], [flat, steep], R)
    assert only.host.key == "flat" and stats[("Orientation", "Contour")][A.KNOWN] == 1
    # the Amendment-2 treatment, kept for the before/after report: the one it
    # parallels, or nothing
    (a,), stats = A.attach([f], [flat, steep], R, every_named=False)
    assert a.host.key == "flat" and a.known and a.by_tangent
    assert stats[("Orientation", "Contour")][A.BY_TANGENT] == 1
    assert A.attach([f], [flat, steep, third], R, every_named=False)[0] == []


def test_the_writer_attaches_a_branch_cell_to_every_contour_that_names_it(params):
    """RULE: the binder writer follows `ATTACHMENT_EVERY_NAMED_HOST`."""
    o = _det("Orientation", "o", (0, 0, 0))
    k1 = _contour("k1", ["o", "x"], mean_theta_bin=0)
    k2 = _contour("k2", ["o", "y"], mean_theta_bin=6)
    pos = {"o": (0.0, 0.0), "x": (40.0, 0.0), "y": (0.0, 40.0)}

    def edges(flag):
        params(ATTACHMENT_EVERY_NAMED_HOST=flag)
        b = _Capture(higher=[(k1, 10), (k2, 11)], positions=dict(pos), v1=[(o, 1)])
        b._run_boundary_attachment()
        return sorted((c[0], c[1]) for c in b.seen[EdgeType.BOUNDARY_ATTACHMENT])

    assert edges(True) == [(10, 1), (11, 1)]
    assert edges(False) == [(10, 1)]         # Amendment 2: only the one it parallels


def test_a_terminator_with_two_near_equal_parents_writes_nothing(params):
    """RULE (Amendment 2.4).  With the position map completed, EndStopped
    terminators enter LINE_TERMINATION; "the nearest parent" of two
    near-equidistant ones is a guess, so nothing is written.  Switch off: the
    old writer, which takes the nearest."""
    term = _det("EndStopped", "t", (0, 0, 0, 7))
    near = _contour("k1", ["k1"], mean_theta_bin=0)
    rival = _contour("k2", ["k2"], mean_theta_bin=6)
    far = _contour("k3", ["k3"], mean_theta_bin=6)
    nodes = {term: 1, near: 2, rival: 3, far: 4}

    def run(pos, dets):
        b = _Capture()
        b._run_line_termination(tuple(dets), nodes, pos)
        return [(c[0], c[1]) for c in b.seen[EdgeType.LINE_TERMINATION]], b

    close = {"t": (0.0, 0.0), "k1": (10.0, 0.0), "k2": (0.0, 12.0)}      # 1.0 and 1.2 scales
    apart = {"t": (0.0, 0.0), "k1": (10.0, 0.0), "k3": (0.0, 20.0)}      # 1.0 and 2.0 scales
    params(BINDER_COMPLETE_POSITION_MAP=False)
    assert (1, 2) in run(close, [term, near, rival])[0]                  # the old writer guesses
    params(BINDER_COMPLETE_POSITION_MAP=True)
    edges, b = run(close, [term, near, rival])
    assert edges == []
    assert b.diagnostics["stage6_attachment"]["line_termination|ambiguous_parent"] == 1
    assert (1, 2) in run(apart, [term, near, far])[0]                    # a clear nearest is kept
