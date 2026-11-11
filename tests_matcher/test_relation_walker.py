"""[STAGE 6 / TRACK M] The relation walker: the contract (M2) and the pins (M3).

Hand-built graphs, no image and no front end: a boundary (a PEAK_CHAIN host)
with features attached at known arc positions, successor links between
consecutive features, attributes hung on features through CO_LOCATED.  Every
pin states the rule it holds.  The renderer-truth pin (M3-6) is measured in
diagnostics/phase6_walker_synthetic.py on the synthetic scenes through the
real front end.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys

import pytest

import core.schema as S
from core.graph_types import Subgraph
from core.types import EdgeType, TypedSignature, is_undirected_edge
from matching import relation_walker as W

from _graphs import tex

ATT, SUC = "EDGE_BOUNDARY_ATTACHMENT", "EDGE_BOUNDARY_SUCCESSOR"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def chain_sig(n=3, closed=False):
    return TypedSignature(kind="CANONICAL", class_="PEAK_CHAIN",
                          quantized=(float(n), 2.0, 1.0, 1.0 if closed else 0.0, 2.0, 1.0))


def att(arc, tangent=0.0, perp=0.0, closed=False):
    ep = S.edge_params(ATT, {"arc_end_dist": None if closed else min(arc, 1.0 - arc),
                             "tangent_angle": tangent, "perp_dist": perp})
    return dataclasses.replace(ep, ordinal=float(arc))


def suc(gap=0.5, turn=0.0):
    return S.edge_params(SUC, {"arc_gap": gap, "turn": turn})


def params_of(cls, **values):
    """Edge parameters of a class; every slot not named is 'no measurement'."""
    full = {spec.name: None for spec in S.EDGE_UNIT_TABLE[cls]}
    full.update(values)
    return S.edge_params(cls, full)


class Build:
    """A graph under construction."""

    def __init__(self):
        self.nodes, self.edges, self.params = {}, set(), {}

    def node(self, nid, lens, sig):
        assert nid not in self.nodes
        self.nodes[nid] = (lens, sig)
        return nid

    def edge(self, s, d, et, ep=None):
        if is_undirected_edge(et) and s > d:
            s, d = d, s
            ep = S.reverse_edge_params(ep) if ep is not None else None
        self.edges.add((s, d, et))
        if ep is not None:
            self.params[(s, d, et)] = ep

    def boundary(self, host, sig, feats, closed=False, successors=True):
        """`feats` = [(node id, signature, arc position[, tangent[, perp]])]."""
        self.node(host, "PEAK_CHAIN", sig)
        prev = None
        for f in sorted(feats, key=lambda f: f[2]):
            fid, fsig, arc = f[0], f[1], f[2]
            if fid not in self.nodes:
                self.node(fid, "CONTOUR_PEAK", fsig)
            self.edge(host, fid, EdgeType.BOUNDARY_ATTACHMENT,
                      att(arc, *(f[3:5]), closed=closed))
            if successors and prev is not None:
                self.edge(prev[0], fid, EdgeType.BOUNDARY_SUCCESSOR,
                          suc(gap=min(1.0, 4.0 * (arc - prev[2]))))
            prev = f
        return host

    def attribute(self, frame, nid, lens, sig):
        self.node(nid, lens, sig)
        self.edge(frame, nid, EdgeType.CO_LOCATED,
                  params_of("EDGE_CO_LOCATED", salience_ratio=1.0, scale_ratio=1.0))

    def done(self):
        sg = Subgraph(nodes=frozenset(self.nodes), edges=frozenset(self.edges),
                      edge_params=dict(self.params))
        return sg, dict(self.nodes)


def relabel(sg, sigs, perm):
    """The same graph under other node ids (undirected edges re-canonicalised)."""
    b = Build()
    for n, (lens, sig) in sigs.items():
        b.node(perm[n], lens, sig)
    for (s, d, et) in sg.edges:
        b.edge(perm[s], perm[d], et, sg.edge_params.get((s, d, et)))
    return b.done()


def two_boundaries():
    """Two chains, eleven features, three attributes."""
    b = Build()
    b.boundary(100, chain_sig(6), [
        (1, tex(1, 1), 0.05), (2, tex(2, 1), 0.20), (3, tex(3, 1), 0.35),
        (4, tex(1, 2), 0.50), (5, tex(2, 2), 0.70), (6, tex(3, 2), 0.90)])
    b.boundary(200, chain_sig(5), [
        (11, tex(5, 5), 0.10, 0.6), (12, tex(6, 5), 0.30, 0.6), (13, tex(7, 5), 0.45, 0.6),
        (14, tex(5, 6), 0.65, 0.6), (15, tex(6, 6), 0.95, 0.6)])
    b.attribute(1, 301, "Color", tex(4, 4))
    b.attribute(4, 302, "Color", tex(9, 1))
    b.attribute(12, 303, "Texture", tex(2, 7))
    return b.done()


# --------------------------------------------------------------------------
# M3-1  self-match under shuffled ids
# --------------------------------------------------------------------------

def test_a_graph_matched_against_itself_under_other_ids_is_the_identity():
    """RULE (M3-1): the same graph under shuffled node ids maps every frame
    node to itself with score 1.  Nothing in the walk reads a node id."""
    sg, sigs = two_boundaries()
    frame = sorted(n for n, (lens, _s) in sigs.items() if W.is_frame(lens))
    for seed in range(5):
        ids = sorted(sigs)
        shuffled = list(ids)
        random.Random(seed).shuffle(shuffled)
        perm = dict(zip(ids, (1000 + x for x in shuffled)))
        sg2, sigs2 = relabel(sg, sigs, perm)
        res = W.walk(sg, sg2, sigs, sigs2)
        assert res.mapping == {n: perm[n] for n in frame}
        assert res.supported and res.score == pytest.approx(1.0)
        assert res.attributes_shared == 3
        back = W.walk(sg2, sg, sigs2, sigs)
        assert back.mapping == {perm[n]: n for n in frame} and back.score == pytest.approx(1.0)


def test_only_the_frame_is_mapped_and_attributes_are_attached_afterwards():
    """RULE (M2): the frame is grown; attributes never enter the mapping.
    They are read after growth, through CO_LOCATED on a mapped frame pair,
    and a shared attribute is a preserved relation of the score."""
    sg, sigs = two_boundaries()
    res = W.walk(sg, sg, sigs, sigs)
    assert not ({301, 302, 303} & (set(res.mapping) | set(res.mapping.values())))
    # the same graph with one attribute changed beyond the gate: one relation
    # is no longer preserved, the mapping is the same mapping
    b_sg, b_sigs = two_boundaries()
    b_sigs = dict(b_sigs)
    b_sigs[302] = ("Color", tex(1, 9))
    other = W.walk(sg, b_sg, sigs, b_sigs)
    assert other.mapping == res.mapping
    assert other.attributes_shared == res.attributes_shared - 1
    assert other.preserved == res.preserved - 1 and other.score < 1.0


# --------------------------------------------------------------------------
# M3-2  relation-less distractors
# --------------------------------------------------------------------------

def test_a_node_with_no_measured_relation_is_never_matched():
    """RULE (M3-2): identical signatures are not a reason to pair two nodes.
    A frame node with no relation (or only CO_LOCATED / proximity edges, or an
    edge with nothing measured) is left out of the mapping."""
    def build():
        b = Build()
        sg, sigs = two_boundaries()
        b.nodes, b.edges, b.params = dict(sigs), set(sg.edges), dict(sg.edge_params)
        b.node(401, "Corner", tex(3, 3))                    # no edge at all
        b.node(402, "Corner", tex(4, 3))
        b.node(403, "Corner", tex(5, 3))
        b.edge(402, 403, EdgeType.CONTACT,                  # proximity, measured
               params_of("EDGE_CONTACT", dist=0.3))
        b.attribute(402, 404, "Color", tex(1, 1))           # same-place attribute
        b.node(405, "EndStopped", tex(6, 3))
        b.edge(100, 405, EdgeType.BOUNDARY_ATTACHMENT,      # a relation, nothing measured
               S.edge_params(ATT, {"arc_end_dist": None, "tangent_angle": None,
                                   "perp_dist": None}))
        return b.done()

    sg, sigs = build()
    res = W.walk(sg, sg, sigs, sigs)
    assert not ({401, 402, 403, 404, 405} & set(res.mapping))
    assert len(res.mapping) == 13 and res.score == pytest.approx(1.0)
    assert not W.is_growth_edge_type(EdgeType.CONTACT)
    assert not W.is_growth_edge_type(EdgeType.CO_LOCATED)
    assert W.is_growth_edge_type(EdgeType.BOUNDARY_ATTACHMENT)


# --------------------------------------------------------------------------
# M3-3  same signatures, different arrangement
# --------------------------------------------------------------------------

def _arranged(swap):
    """One chain, eight features.  `swap` exchanges the PLACES of the two
    features whose signatures are tex(1,1) and tex(2,1)."""
    b = Build()
    first, second = (0.45, 0.06) if swap else (0.06, 0.45)
    b.boundary(100, chain_sig(8), [
        (1, tex(1, 1), first), (2, tex(2, 1), second),
        (3, tex(8, 8), 0.15), (4, tex(8, 1), 0.25), (5, tex(1, 8), 0.35),
        (6, tex(5, 5), 0.60), (7, tex(5, 9), 0.75), (8, tex(9, 5), 0.90)],
        successors=False)
    return b.done()


def test_the_mapping_follows_the_relations_not_the_signatures():
    """RULE (M3-3): two graphs with the same signatures, arranged differently.
    A pair is admitted BECAUSE a relation around it agrees: the feature at
    the start of the boundary goes to the feature at the start of the other
    boundary, although the node with its own signature exists elsewhere on
    it.  (Both candidates pass the signature gate; the gate does not rank.)"""
    a_sg, a_sigs = _arranged(False)
    b_sg, b_sigs = _arranged(True)
    res = W.walk(a_sg, b_sg, a_sigs, b_sigs)
    assert res.mapping[1] == 2 and res.mapping[2] == 1        # by place
    assert all(res.mapping[n] == n for n in (100, 3, 4, 5, 6, 7, 8))
    # the old matcher pairs by signature
    from memory.cores import evidence_score
    _score, corr = evidence_score(a_sg, b_sg, a_sigs, b_sigs)
    pairs = set(corr.items()) | {(v, k) for k, v in corr.items()}
    assert (1, 1) in pairs and (2, 2) in pairs


def test_identical_signatures_with_disagreeing_relations_are_not_paired():
    """RULE (M2): signature similarity is only a gate.  A feature whose
    signature is identical in both graphs but which sits at another place on
    the boundary (more than one bin away on the measured axis) is not
    admitted on the strength of its signature."""
    def build(arc):
        b = Build()
        b.boundary(100, chain_sig(8), [
            (1, tex(1, 1), arc), (3, tex(8, 8), 0.15), (4, tex(8, 1), 0.25),
            (5, tex(1, 8), 0.35), (6, tex(5, 5), 0.60), (7, tex(5, 9), 0.75),
            (8, tex(9, 5), 0.90), (9, tex(9, 9), 0.50)], successors=False)
        return b.done()

    a_sg, a_sigs = build(0.03)
    b_sg, b_sigs = build(0.44)
    res = W.walk(a_sg, b_sg, a_sigs, b_sigs)
    assert 1 not in res.mapping and len(res.mapping) == 8


def test_a_pair_that_fails_the_gate_is_not_admitted_whatever_its_relations():
    """RULE (M2): the gate is a gate.  Same place, same relations, a
    signature too far (similarity under SIGNATURE_GATE): not admitted."""
    a_sg, a_sigs = _arranged(False)
    b_sg, b_sigs = _arranged(False)
    b_sigs = dict(b_sigs)
    b_sigs[6] = ("CONTOUR_PEAK", tex(0, 0))                 # was tex(5, 5)
    res = W.walk(a_sg, b_sg, a_sigs, b_sigs)
    assert 6 not in res.mapping and res.gate_rejections >= 1
    assert W.SIGNATURE_GATE == 0.5


def test_an_axis_measured_on_one_side_only_weighs_nothing():
    """RULE (M2): unmeasured axes weigh nothing either way.  Two relations
    agree when every axis measured on BOTH is within one bin; an axis one
    side lacks is ignored; with no axis measured on both they do not agree."""
    def half(bins):
        return W._Half(cls=("X", "o", "L"), far=0, bins=tuple(bins), cycles=(),
                       ordinal=None, etype=None)

    assert W._slack(half([(0, 3.0), (1, 5.0)]), half([(0, 3.0), (1, 5.0)])) == 0
    assert W._slack(half([(0, 3.0), (1, 5.0)]), half([(0, 4.0), (1, 5.0)])) == 1
    assert W._slack(half([(0, 3.0), (1, 5.0)]), half([(0, 5.0), (1, 5.0)])) is None
    assert W._slack(half([(0, 3.0), (1, 5.0)]), half([(0, 3.0)])) == 0       # axis 1 ignored
    assert W._slack(half([(0, 3.0)]), half([(0, 3.0), (1, 5.0)])) == 0       # ... either way
    assert W._slack(half([(0, 3.0)]), half([(1, 3.0)])) is None              # nothing shared
    wrap = W._Half(cls=("X", "o", "L"), far=0, bins=((0, 0.0),), cycles=((0, 8),),
                   ordinal=None, etype=None)
    assert W._slack(wrap, half([(0, 7.0)])) == 1                             # circular axis


def _candidates(r1, r2):
    """A boundary whose feature `u` (tex(5,5), bin 3 of the distance from the
    end) has two candidates on the other boundary, both through the gate."""
    def build(feats):
        b = Build()
        b.boundary(100, chain_sig(5), feats + [
            (7, tex(0, 0), 0.03), (8, tex(9, 9), 0.47), (9, tex(0, 9), 0.40)],
            successors=False)
        return b.done()

    return build([(1, tex(5, 5), 0.22)]), build([(1,) + r1, (2,) + r2])


def test_among_equally_supported_candidates_the_better_agreement_is_served_first():
    """RULE (Amendment 4.1): proposals are served by (supporting pairs; total
    slack; signature similarity; canonical key).  Exact agreement on the
    measured axes is better evidence than agreement within a bin, and the
    canonical key order is arbitrary with respect to the measurement, so it
    comes last."""
    # r1 has the smaller key but sits one bin off; r2 is in the same bin
    (a_sg, a_sigs), (b_sg, b_sigs) = _candidates((tex(4, 5), 0.27), (tex(6, 5), 0.20))
    assert W.walk(a_sg, b_sg, a_sigs, b_sigs).mapping[1] == 2
    # the same two the other way round: the key does not decide
    (a_sg, a_sigs), (b_sg, b_sigs) = _candidates((tex(4, 5), 0.20), (tex(6, 5), 0.27))
    assert W.walk(a_sg, b_sg, a_sigs, b_sigs).mapping[1] == 1
    # equal agreement (one bin): the more similar signature, not the smaller key
    (a_sg, a_sigs), (b_sg, b_sigs) = _candidates((tex(3, 5), 0.21), (tex(6, 5), 0.23))
    assert W.walk(a_sg, b_sg, a_sigs, b_sigs).mapping[1] == 2


# --------------------------------------------------------------------------
# M3-4  the order constraint
# --------------------------------------------------------------------------

def _ordered(arcs):
    b = Build()
    sigs = [tex(1, 1), tex(5, 1), tex(9, 1), tex(1, 9), tex(9, 9)]
    b.boundary(100, chain_sig(5), [(i + 1, sigs[i], a) for i, a in enumerate(arcs)],
               successors=False)
    return b.done()


def test_a_mapping_that_reverses_part_of_a_boundary_is_rejected():
    """RULE (M3-4): features on an admitted pair of boundaries keep their arc
    order.  Two features exchange places symmetrically about the middle, so
    every measured axis still agrees (the distance from the nearer end is the
    same); only the ORDER tells.  One of the two is refused."""
    a_sg, a_sigs = _ordered([0.10, 0.20, 0.40, 0.60, 0.80])
    b_sg, b_sigs = _ordered([0.10, 0.20, 0.60, 0.40, 0.80])     # 3 and 4 exchanged
    res = W.walk(a_sg, b_sg, a_sigs, b_sigs)
    assert res.order_rejections >= 1
    assert {1, 2, 5, 100} <= set(res.mapping)
    assert len({3, 4} & set(res.mapping)) == 1                  # not both
    same = W.walk(a_sg, a_sg, a_sigs, a_sigs)
    assert same.order_rejections == 0 and len(same.mapping) == 6


def test_a_boundary_walked_the_other_way_is_the_same_boundary():
    """RULE: the direction a chain is walked in is the chain former's choice
    (a gauge).  The whole boundary mirrored keeps every relation and one
    order: it is accepted in full.  What is rejected is a mapping that is not
    monotone."""
    arcs = [0.10, 0.20, 0.40, 0.60, 0.80]
    a_sg, a_sigs = _ordered(arcs)
    b_sg, b_sigs = _ordered([1.0 - a for a in arcs])
    res = W.walk(a_sg, b_sg, a_sigs, b_sigs)
    assert res.order_rejections == 0
    assert res.mapping == {100: 100, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def test_the_order_test_itself():
    """RULE: open boundary, one sign over all pairs; closed boundary, a
    rotation of a rising or of a falling sequence; equal positions constrain
    nothing; fewer than three features cannot disagree."""
    ok = W._order_consistent
    assert ok([(0.1, 0.1), (0.2, 0.3), (0.3, 0.9)], False)
    assert ok([(0.1, 0.9), (0.2, 0.3), (0.3, 0.1)], False)              # walked backwards
    assert not ok([(0.1, 0.1), (0.2, 0.9), (0.3, 0.5)], False)
    assert ok([(0.1, 0.9), (0.2, 0.1)], False)
    assert ok([(0.1, 0.1), (0.2, 0.5), (0.2, 0.9), (0.3, 0.95)], False)
    assert not ok([(0.1, 0.5), (0.2, 0.3), (0.2, 0.6)], False)          # one pair each way
    assert ok([(0.1, 0.5), (0.1, 0.2), (0.3, 0.9)], False)              # a tie constrains nothing
    ring = [(0.1, 0.6), (0.3, 0.8), (0.5, 0.0), (0.7, 0.2)]
    assert not ok(ring, False) and ok(ring, True)                       # a rotation
    assert ok([(a, 1.0 - b) for a, b in ring], True)                    # and its mirror
    assert not ok([(0.1, 0.1), (0.3, 0.6), (0.5, 0.3), (0.7, 0.8), (0.9, 0.5)], True)


def test_on_a_closed_boundary_the_origin_is_free():
    """RULE: a ring has no origin.  The same ring cut at another place has
    every arc position shifted; the order is the same cyclic order."""
    sigs = [tex(1, 1), tex(5, 1), tex(9, 1), tex(1, 9), tex(9, 9), tex(5, 5)]

    def ring(shift, arcs=(0.05, 0.2, 0.4, 0.55, 0.7, 0.9)):
        b = Build()
        b.boundary(100, chain_sig(6, closed=True),
                   [(i + 1, sigs[i], (a + shift) % 1.0) for i, a in enumerate(arcs)],
                   closed=True, successors=False)
        b.boundary(200, chain_sig(4), [(i + 1, sigs[i], 0.1 + 0.15 * i) for i in range(6)],
                   successors=False)
        return b.done()

    a_sg, a_sigs = ring(0.0)
    b_sg, b_sigs = ring(0.5)
    res = W.walk(a_sg, b_sg, a_sigs, b_sigs)
    assert res.order_rejections == 0 and len(res.mapping) == 8
    assert all(u == v for u, v in res.mapping.items())


# --------------------------------------------------------------------------
# M3-5  determinism and the budget
# --------------------------------------------------------------------------

_SEED_SCRIPT = r"""
import hashlib, json, sys
sys.path.insert(0, %(root)r); sys.path.insert(0, %(here)r)
sys.path.append(%(shims)r)
import test_relation_walker as T
from matching import relation_walker as W
out = []
for a, b in ((T.two_boundaries(), T.two_boundaries()),
             (T._arranged(False), T._arranged(True)),
             (T._ordered([0.10, 0.20, 0.40, 0.60, 0.80]),
              T._ordered([0.10, 0.20, 0.60, 0.40, 0.80]))):
    r = W.walk(a[0], b[0], a[1], b[1])
    out.append([sorted(r.mapping.items()), r.seeds, r.score, r.examined, r.edge_checks,
                r.order_rejections, r.preserved, r.relations_q, r.relations_r])
print(hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest())
"""


def test_the_same_walk_under_every_hash_seed():
    """RULE (M3-5): determinism across PYTHONHASHSEED: the mapping, the
    seeds, the score and the cost are the same numbers."""
    script = _SEED_SCRIPT % {"root": ROOT, "here": HERE,
                             "shims": os.path.join(ROOT, "diagnostics", "_export_shims")}
    seen = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env, cwd=ROOT)
        assert out.returncode == 0, out.stderr[-2000:]
        seen.add(out.stdout.strip())
    assert len(seen) == 1 and len(next(iter(seen))) == 64


def test_the_walk_stops_at_its_budget(monkeypatch):
    """RULE (M3-5): pairs examined (signature similarities computed) never
    exceed BUDGET_FRACTION of |FRAME_q| x |FRAME_r|; a walk that runs out
    says so and keeps what it has."""
    sg, sigs = two_boundaries()
    full = W.walk(sg, sg, sigs, sigs)
    assert (W.BUDGET_FRACTION, W.BUDGET_FLOOR_PER_NODE) == (0.05, 8)
    # 5 % of the product on big frames, 8 looks per node on small ones, never
    # more than the product
    assert W.budget(300, 280) == 4200 and W.budget(13, 13) == 104 and W.budget(4, 5) == 20
    assert full.budget == 104 and full.examined <= full.budget and not full.budget_exhausted
    monkeypatch.setattr(W, "budget", lambda a, b: 4)
    cut = W.walk(sg, sg, sigs, sigs)
    assert cut.budget == 4 and cut.examined == 4 and cut.budget_exhausted
    assert len(cut.mapping) < len(full.mapping)
    assert all(u == v for u, v in cut.mapping.items())


def test_seeds_are_few_rare_and_one_to_one():
    """RULE (M2): at most N_SEEDS seeds; a seed shares at least two relation
    tokens that at most two nodes carry in each graph; one-to-one."""
    sg, sigs = two_boundaries()
    res = W.walk(sg, sg, sigs, sigs)
    assert 1 <= len(res.seeds) <= W.N_SEEDS == 8
    assert len({u for u, _v in res.seeds}) == len(res.seeds) == len({v for _u, v in res.seeds})
    assert all(u == v for u, v in res.seeds)
    # a boundary whose every feature sits in ONE bin has one token: no seed
    b = Build()
    b.boundary(100, chain_sig(4), [(i + 1, tex(i, i), 0.26 + 0.004 * i) for i in range(4)],
               successors=False)
    lone_sg, lone_sigs = b.done()
    lone = W.walk(lone_sg, lone_sg, lone_sigs, lone_sigs)
    assert lone.seeds == [] and lone.mapping == {} and not lone.supported


# --------------------------------------------------------------------------
# the score, the support, the switch
# --------------------------------------------------------------------------

def test_below_the_minimum_support_there_is_no_score():
    """RULE (M2): a mapping is SUPPORTED iff it has at least MIN_SUPPORT_PAIRS
    frame pairs AND at least MIN_SUPPORT_FRACTION of the smaller frame.  Below
    it the pair of graphs is UNSUPPORTED: the score is None, not a number."""
    b = Build()
    b.boundary(100, chain_sig(4), [(i + 1, tex(2 * i, 1), 0.1 + 0.2 * i) for i in range(4)])
    sg, sigs = b.done()
    res = W.walk(sg, sg, sigs, sigs)
    assert len(res.mapping) == 5 < W.MIN_SUPPORT_PAIRS == 8
    assert not res.supported and res.score is None
    assert (W.MIN_SUPPORT_PAIRS, W.MIN_SUPPORT_FRACTION) == (8, 0.10)
    big_sg, big_sigs = two_boundaries()
    assert W.walk(big_sg, big_sg, big_sigs, big_sigs).supported


def test_the_score_is_a_jaccard_on_preserved_relations_over_the_mapped_part():
    """RULE (M2): preserved / (query relations among mapped nodes + reference
    relations among their images - preserved).  A relation the other side
    lacks counts against; nodes outside the mapping count for nothing."""
    a_sg, a_sigs = two_boundaries()
    res = W.walk(a_sg, a_sg, a_sigs, a_sigs)
    # 11 attachments + 9 successors + 3 attributes, all preserved
    assert (res.preserved, res.relations_q, res.relations_r) == (23, 23, 23)
    # drop two successor links from the reference: 21 preserved of 23 + 21 - 21
    b = Build()
    b.nodes, b.edges, b.params = dict(a_sigs), set(a_sg.edges), dict(a_sg.edge_params)
    for key in ((2, 3, EdgeType.BOUNDARY_SUCCESSOR), (12, 13, EdgeType.BOUNDARY_SUCCESSOR)):
        b.edges.discard(key)
        b.params.pop(key)
    b_sg, b_sigs = b.done()
    res2 = W.walk(a_sg, b_sg, a_sigs, b_sigs)
    assert res2.mapping == res.mapping
    assert (res2.preserved, res2.relations_q, res2.relations_r) == (21, 23, 21)
    assert res2.score == pytest.approx(21.0 / 23.0)
    # an edge with NOTHING measured is not a relation: one more of them on one
    # side changes no count
    c = Build()
    c.nodes, c.edges, c.params = dict(a_sigs), set(a_sg.edges), dict(a_sg.edge_params)
    c.edge(1, 3, EdgeType.BOUNDARY_SUCCESSOR, params_of(SUC))
    c_sg, c_sigs = c.done()
    res3 = W.walk(c_sg, a_sg, c_sigs, a_sigs)
    assert (res3.preserved, res3.relations_q, res3.relations_r) == (23, 23, 23)
    assert res3.score == pytest.approx(1.0)


def test_the_switch_off_is_the_old_matcher(params, monkeypatch):
    """RULE: ONE switch, read at ONE place.  Off (default): `match_evidence`
    IS `evidence_score` (same arguments, its own return value) and the walker
    is not run.  On: the walker; an unsupported pair has no score."""
    from memory.cores import evidence_score
    a_sg, a_sigs = _arranged(False)
    b_sg, b_sigs = _arranged(True)
    P = params(MATCH_RELATION_WALKER=False)
    calls = []
    real_walk = W.walk
    monkeypatch.setattr(W, "walk", lambda *a, **k: calls.append(1) or real_walk(*a, **k))
    off = W.match_evidence(a_sg, b_sg, a_sigs, b_sigs)
    assert off == evidence_score(a_sg, b_sg, a_sigs, b_sigs) and calls == []
    params(MATCH_RELATION_WALKER=True)
    trace = {}
    score, mapping = W.match_evidence(a_sg, b_sg, a_sigs, b_sigs, trace=trace)
    assert calls == [1] and mapping == trace["relation_walk"].mapping
    assert score == trace["relation_walk"].score and mapping[1] == 2
    assert bool(P.MATCH_RELATION_WALKER) is True
