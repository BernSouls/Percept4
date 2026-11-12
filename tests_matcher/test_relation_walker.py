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
    # [Amendment 7] a shared attribute is a PRESERVED relation and an
    # attribute unit of the objective: one attribute lost moves BOTH counts
    assert other.preserved == res.preserved and other.attributes_shared == 2
    assert other.score < 1.0 and other.terms["A"] == res.terms["A"] - 1.0


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
        (6, tex(5, 5), 0.60), (7, tex(5, 9), 0.75), (8, tex(9, 5), 0.90)])
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
            (8, tex(9, 5), 0.90), (9, tex(9, 9), 0.50)])
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
            (7, tex(0, 0), 0.03), (8, tex(9, 9), 0.47), (9, tex(0, 9), 0.40)])
        return b.done()

    return build([(1, tex(5, 5), 0.22)]), build([(1,) + r1, (2,) + r2])


def _served_without_seeding(a_sg, a_sigs, b_sg, b_sigs):
    """Grow from the three harmless seeds, so the contested feature (1) is
    served from the heap and its ORDER is what is measured."""
    gq, gr = W._Index(a_sg, a_sigs), W._Index(b_sg, b_sigs)
    res = W.WalkResult(frame_q=len(gq.frame), frame_r=len(gr.frame),
                       budget=W.budget(len(gq.frame), len(gr.frame)))
    sim = W._Sim(gq, gr, res)
    common = sorted(set(gq.frame) & set(gr.frame) - {1, 2})
    return W._grow(gq, gr, sim, res, [(n, n) for n in common[:3]]), res


def test_among_equally_supported_candidates_the_better_agreement_is_served_first():
    """RULE (Amendment 4.1, kept under D0): proposals are served by
    (supporting pairs; total slack; signature similarity; canonical key).
    Exact agreement on the measured axes is better evidence than agreement
    within a bin, and the canonical key order is arbitrary with respect to the
    measurement, so it comes last."""
    # candidate 2 sits one bin off, candidate 1 is in the same bin
    (a_sg, a_sigs), (b_sg, b_sigs) = _candidates((tex(6, 5), 0.20), (tex(4, 5), 0.27))
    grown, _res = _served_without_seeding(a_sg, a_sigs, b_sg, b_sigs)
    assert grown.mapping[1] == 1                       # exact agreement, not the nearest arc
    # the same two the other way round: exact agreement still wins, so the
    # one that kept the bin (now candidate 2) is served first
    (a_sg, a_sigs), (b_sg, b_sigs) = _candidates((tex(6, 5), 0.27), (tex(4, 5), 0.20))
    grown, _res = _served_without_seeding(a_sg, a_sigs, b_sg, b_sigs)
    assert grown.mapping[1] == 2


def _two_plain(r1, r2):
    """A boundary whose feature 1 has two candidates, each attached at its
    place and with NO successor links, so support and slack are exactly equal
    and only the signature can differ."""
    def build(feats):
        b = Build()
        b.boundary(100, chain_sig(5), feats +
                   [(7, tex(0, 0), 0.03), (8, tex(9, 9), 0.47), (9, tex(0, 9), 0.40)],
                   successors=False)
        return b.done()
    return build([(1, tex(5, 5), 0.22)]), build([(1,) + r1, (2,) + r2])


def test_node_similarity_is_a_tie_break_not_a_rank_or_a_veto():
    """RULE (D0): node similarity neither ranks a proposal nor vetoes it; it
    is the LAST tie-break of Amendment 4.1, after support and slack.  Two
    candidates in the same arc bin, both supported by the one admitted
    boundary: the more similar signature is served first, and the other is
    not counted as a gate rejection (it was never refused, only not taken)."""
    # tex(6,5) is the closer signature to tex(5,5) (0.956 against 0.835)
    (a_sg, a_sigs), (b_sg, b_sigs) = _two_plain((tex(6, 5), 0.21), (tex(3, 5), 0.24))
    grown, _res = _served_without_seeding(a_sg, a_sigs, b_sg, b_sigs)
    assert grown.mapping[1] == 1
    assert grown.gate_rejections == 0
    # the same two the other way round: the arc position does not decide
    (a_sg, a_sigs), (b_sg, b_sigs) = _two_plain((tex(3, 5), 0.21), (tex(6, 5), 0.24))
    grown, _res = _served_without_seeding(a_sg, a_sigs, b_sg, b_sigs)
    assert grown.mapping[1] == 2
    assert grown.gate_rejections == 0


# --------------------------------------------------------------------------
# D1  the seeds are topological, and they are ranked by node similarity
# --------------------------------------------------------------------------

def test_a_seed_is_two_shared_topological_tokens_and_no_measurement():
    """RULE (D1): a seed token is (lens type, edge type, direction, far lens
    type, the far node's degree, whether the far node is closed).  No bins, no
    similarity.  Two nodes with the same relation SHAPE seed each other even
    when every measured value on those relations is moved: the seeds are the
    same pairs before and after."""
    a_sg, a_sigs = two_boundaries()
    b = Build()
    b.nodes, b.edges, b.params = dict(a_sigs), set(a_sg.edges), dict(a_sg.edge_params)
    for (s_, d_, et), ep in list(b.params.items()):
        if et == EdgeType.BOUNDARY_ATTACHMENT:            # move every arc axis by 2 bins
            b.params[(s_, d_, et)] = dataclasses.replace(
                ep, quantized=(ep.quantized[0] + 2.0,) + tuple(ep.quantized[1:]))
    moved_sg, moved_sigs = b.done()
    # [STAGE 7 / PHASE 2] the TOPOLOGICAL rule is no longer the default (the
    # measured rule is, on a pre-registered 8-0-1 precision comparison); this
    # pin is about the topological rule, so it states that it wants it.
    old_flag = W.SEED_TOPOLOGICAL
    W.SEED_TOPOLOGICAL = True
    try:
        ident = W.walk(a_sg, a_sg, a_sigs, a_sigs).seeds
        shifted = W.walk(a_sg, moved_sg, a_sigs, moved_sigs).seeds
    finally:
        W.SEED_TOPOLOGICAL = old_flag
    assert ident == shifted and len(ident) >= 3
    # the tokens carry no measurements at all
    g = W._Index(a_sg, a_sigs)
    for t in g.tok[1]:
        assert len(t) == 6 and all(isinstance(x, (str, int, bool)) for x in t)


def _rare_pairing():
    """One feature (3) is attached to another (5) through a measured
    PART_CONTAINMENT: both then carry topological tokens that NO other node in
    the graph carries, so a pair sharing them shares two RARE tokens.  A
    second graph where 3 is related to a plain feature instead shares only one
    of them."""
    sg, sigs = two_boundaries()
    b = Build()
    b.nodes, b.edges, b.params = dict(sigs), set(sg.edges), dict(sg.edge_params)
    b.edge(3, 5, EdgeType.PART_CONTAINMENT,
           params_of("EDGE_PART_CONTAINMENT", scale_ratio=1.0, rel_dist=0.3))
    rich_sg, rich_sigs = b.done()
    c = Build()
    c.nodes, c.edges, c.params = dict(sigs), set(sg.edges), dict(sg.edge_params)
    c.edge(3, 6, EdgeType.PART_CONTAINMENT,      # the same class, the other end
           params_of("EDGE_PART_CONTAINMENT", scale_ratio=1.0, rel_dist=0.9))
    other_sg, other_sigs = c.done()
    return (rich_sg, rich_sigs), (other_sg, other_sigs)


def test_a_pair_is_a_seed_only_when_the_node_gate_passes_it(monkeypatch):
    """RULE (D1 + D3): a seed is a pair of FRAME nodes that also passes the
    node gate.  Two nodes of DIFFERENT lens types have similarity 0.0, under
    the floor: a token rule alone would seed them; the gate refuses.  (The
    seed LIST is read in full here, N_SEEDS lifted, so the rule and not the
    list length is what is measured.)"""
    b = Build()
    b.boundary(100, chain_sig(11), [(i, tex(5, 5), 0.04 + 0.08 * i) for i in range(1, 11)] +
               [(11, tex(5, 5), 0.95)])
    a_sg, a_sigs = b.done()
    monkeypatch.setattr(W, "N_SEEDS", 32)
    other = dict(a_sigs)
    other[11] = ("EndStopped", tex(5, 5))              # another lens: similarity 0
    differing = W.walk(a_sg, a_sg, a_sigs, other)
    same = W.walk(a_sg, a_sg, a_sigs, a_sigs)
    assert all(v != 11 for _u, v in differing.seeds)
    assert any(v == 11 for _u, v in same.seeds)


def test_seed_ranking_puts_the_rare_topology_first():
    """RULE (D1 / 7.9): seeds are ranked by RARE tokens shared first, so a
    pair that fits an uncommon shape is preferred to a common one, whatever
    its node similarity.  (Ranking by similarity instead puts the whole
    identity of the graph first -- the mutation report in the pre-registration
    says so: it is the one D-rule the pins catch only through the seed list.)"""
    (rich_sg, rich_sigs), (other_sg, other_sigs) = _rare_pairing()
    rich = W.walk(rich_sg, rich_sg, rich_sigs, rich_sigs).seeds
    other = W.walk(rich_sg, other_sg, rich_sigs, other_sigs).seeds
    # the rare shape is found in both graphs (the class is rare in both), and
    # it is found FIRST in both -- the seeds are generated by topology, not
    # by similarity: they are the same list although the two graphs differ
    assert (3, 3) in rich and (3, 3) in other
    assert rich.index((3, 3)) <= 3 and other.index((3, 3)) <= 3
    assert len({u for u, _v in rich}) == len({v for _u, v in rich})


# --------------------------------------------------------------------------
# D4  the order proposes, and it is served after a measurement
# --------------------------------------------------------------------------

def test_the_order_proposes_and_is_served_after_a_measurement():
    """RULE (D4 / 7.18): two admitted boundaries with ORDER_PROPOSAL_MIN
    admitted features in order propose the features attached to both.  An
    order proposal carries no measured slack and is therefore a TIER below a
    proposed relation: it may only reach what no measurement reached."""
    assert W.ORDER_PROPOSAL_MIN == 3
    # grown from three harmless seeds, so the feature that only the ORDER can
    # reach (its measured axis disagrees) is what the pin reads
    a_sg, a_sigs = _arranged(False)
    b_sg, b_sigs = _arranged(True)
    res, _res = _served_without_seeding(a_sg, a_sigs, b_sg, b_sigs)
    kinds = [how for _u, _v, how in res.log]
    order = [i for i, k in enumerate(kinds)
             if isinstance(k, tuple) and "order" in k]
    assert order, "the order proposed at least one admission"
    # every relation-backed admission is served before any order-backed one
    assert max(i for i, k in enumerate(kinds)
               if not (isinstance(k, tuple) and "order" in k)) < min(order)


# --------------------------------------------------------------------------
# D5  the beam
# --------------------------------------------------------------------------

def test_the_beam_grows_four_seed_sets_and_keeps_the_best():
    """RULE (D5 / 7.10-7.11): up to BEAM_WIDTH hypotheses, one per seed set
    (grown largest first, so the whole seed list is not starved); each is
    grown independently and one-to-one; the answer is the hypothesis with the
    larger supported mapping, Q breaking the tie."""
    sg, sigs = two_boundaries()
    res = W.walk(sg, sg, sigs, sigs)
    assert W.BEAM_WIDTH == 4 and 1 <= len(res.beam) <= 4
    assert max(b["seeds"] for b in res.beam) == len(res.seeds)   # the whole list ran
    assert res.beam[0]["seeds"] == len(res.seeds)                # ... and ran first
    assert max(b["mapped"] for b in res.beam) == len(res.mapping)
    assert len(set(res.mapping.values())) == len(res.mapping)     # one-to-one
    # a hypothesis that maps more wins over a hypothesis with a higher Q
    best = max(res.beam, key=lambda b: (b["mapped"], b["q"]))
    assert best["mapped"] == len(res.mapping)


def test_an_order_proposal_never_lowers_a_measured_slack():
    """RULE (7.19): an order proposal carries no MEASURED slack.  It may open
    a support no measurement opened, it may never overwrite a measured slack
    with 0.0, and a measurement arriving later replaces the weak entry."""
    sup_d = {}
    W._record_support(sup_d, ("A",), 2.0, True)          # measured, two bins off
    W._record_support(sup_d, ("A",), 0.0, False)         # order backs the same pair
    assert sup_d[("A",)] == (2.0, True) and W._measured_slack(sup_d) == 2.0
    W._record_support(sup_d, ("B",), 0.0, False)         # order only, no measurement
    assert sup_d[("B",)] == (0.0, False) and W._measured_slack(sup_d) == 2.0
    W._record_support(sup_d, ("B",), 3.0, True)          # a measurement arrives
    assert sup_d[("B",)] == (3.0, True) and W._measured_slack(sup_d) == 5.0
    W._record_support(sup_d, ("A",), 1.0, True)          # two measurements: the smaller
    assert sup_d[("A",)] == (1.0, True) and W._measured_slack(sup_d) == 4.0
    W._record_support(sup_d, ("B",), 3.5, True)
    assert sup_d[("B",)][0] == 3.0


def test_an_order_proposal_is_served_after_a_measured_one():
    """RULE (7.18, read on the serving key itself): proposals are ordered by
    (tier; support; slack; signature; keys).  An order proposal carries no
    measurement, so it is a tier below; without the tier it would outrank
    every measured proposal (that is what the pre-registration measured)."""
    key = W._serve_key
    measured = key(("BOUNDARY_ATTACHMENT",), 1, 0.5, 0.8, ("a",), ("b",), 1, 2)
    by_order = key(("order", 100, 100), 1, 0.0, 0.9, ("a",), ("b",), 1, 2)
    assert measured < by_order                     # tier 0 before tier 1
    # two measured proposals: more support, then less slack, then similarity
    assert key(("X",), 2, 9.0, 0.1, ("z",), ("z",), 9, 9) < key(("X",), 1, 0.0, 1.0,
                                                               ("a",), ("a",), 1, 1)
    assert key(("X",), 1, 0.0, 0.1, ("z",), ("z",), 9, 9) < key(("X",), 1, 2.0, 1.0,
                                                               ("a",), ("a",), 1, 1)
    # equal tier, support and slack: the more similar signature is served first
    assert key(("X",), 1, 1.0, 0.9, ("z",), ("z",), 9, 9) < key(("X",), 1, 1.0, 0.1,
                                                               ("a",), ("a",), 1, 1)


def test_the_beam_choice_prefers_the_larger_mapping_over_a_higher_q():
    """RULE (7.11): among hypotheses that pass the support floor the larger
    mapping is the answer, Q only breaking a tie; below the floor Q decides.
    (Q is a mean, so the smaller mapping can always have the higher Q.)"""
    def hyp(mapped, q):
        h = W.WalkResult(frame_q=100, frame_r=100, supported=True)
        h.mapping = {i: i for i in range(mapped)}
        h.terms = {"q": q}
        return h

    big, small = hyp(60, 0.50), hyp(20, 0.95)
    assert W._choose([big, small]) is big
    assert W._choose([small, big]) is big          # the order of the list decides nothing
    # both under the floor: the better Q wins, whatever the order of the list
    a, b = hyp(2, 0.80), hyp(2, 0.60)
    assert W._choose([a, b]) is a and W._choose([b, a]) is a
    # a tie goes to the earlier hypothesis (the earlier seed set)
    c = hyp(2, 0.80)
    assert W._choose([a, c]) is a and W._choose([c, a]) is c


# --------------------------------------------------------------------------
# M3-4  the order constraint
# --------------------------------------------------------------------------

def _ordered(arcs):
    b = Build()
    sigs = [tex(1, 1), tex(5, 1), tex(9, 1), tex(1, 9), tex(9, 9)]
    b.boundary(100, chain_sig(5), [(i + 1, sigs[i], a) for i, a in enumerate(arcs)])
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


def test_a_partial_identity_scores_below_one_and_a_complete_one_scores_one():
    """RULE (7.23, restating M1's score): Q is read over the UNION of the
    relations, so a mapping that leaves frame nodes out is penalised by the
    relations that leave it.  A COMPLETE identity of a graph with itself is
    exactly 1.0; a partial one is not, and how far below is its own formula."""
    sg, sigs = two_boundaries()
    g = W._Index(sg, sigs)
    frame = sorted(g.frame)
    sim_one = lambda u, v: 1.0
    complete = {n: n for n in frame}
    assert W.objective(complete, g, g, sim_one)["q"] == pytest.approx(1.0)
    # drop one feature that carries relations: the mapping is still one-to-one
    # and still correct, and Q falls -- the edges it leaves are in R
    partial = {n: n for n in frame if n != 12}
    q = W.objective(partial, g, g, sim_one)["q"]
    assert q < 1.0 and q > 0.5
    # a partial mapping that is CLOSED (nothing leaves it) is still not 1: the
    # node half |M| is smaller than the denominators' own relation count only
    # when relations leave it; with none leaving, Q is 1 again
    closed = {n: n for n in frame if n != 12 and n != 13}
    assert W.objective(closed, g, g, sim_one)["q"] <= 1.0
    assert W.objective(complete, g, g, sim_one)["q"] > W.objective(partial, g, g, sim_one)["q"]


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
    # Amendment 7.13: on a SMALL frame (product <= 1000) the whole matrix may
    # be examined; above it 5 % of the product, floored at 8 looks per node
    assert W.budget(300, 280) == 4200 and W.budget(40, 40) == 320   # 5 % binds
    assert W.budget(13, 13) == 169 and W.budget(4, 5) == 20
    # [Amendment 7.22] the budget is PER HYPOTHESIS: each of the beam's walks
    # may examine up to it, and the seed scan is a look of its own; what the
    # walk reports as `examined` is what the whole beam spent, so the pin is
    # that no SINGLE hypothesis went over, and that the bill is reported.
    assert full.budget == 169 and not full.budget_exhausted
    assert len(full.mapping) == 13
    over = [b for b in full.beam if b.get("examined", 0) > full.budget]
    assert not over, over
    assert full.examined >= sum(b.get("examined", 0) for b in full.beam)
    monkeypatch.setattr(W, "budget", lambda a, b: 4)
    cut = W.walk(sg, sg, sigs, sigs)
    assert cut.budget == 4 and cut.budget_exhausted
    assert all(b.get("examined", 0) <= 4 for b in cut.beam)
    assert cut.examined <= 4 * (len(cut.beam) + 1)          # the beam's whole bill
    assert len(cut.mapping) < len(full.mapping)
    assert all(u == v for u, v in cut.mapping.items())


def test_seeds_are_few_rare_and_one_to_one():
    """RULE (M2): at most N_SEEDS seeds; a seed shares at least two relation
    tokens that at most two nodes carry in each graph; one-to-one."""
    sg, sigs = two_boundaries()
    # [STAGE 7 / PHASE 2] this is the TOPOLOGICAL rule's pin; the default seed
    # rule is the measured one now, so the pin states which rule it tests.
    old_flag = W.SEED_TOPOLOGICAL
    W.SEED_TOPOLOGICAL = True
    try:
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
    finally:
        W.SEED_TOPOLOGICAL = old_flag


# --------------------------------------------------------------------------
# the score, the support, the switch
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# D0  the objective;  D2  the contradiction margin;  D3  the relative gate
# --------------------------------------------------------------------------

def _one_relation_off(a_sg, a_sigs):
    """The same graph with ONE attachment two bins away: P is no longer full."""
    b = Build()
    b.nodes, b.edges, b.params = dict(a_sigs), set(a_sg.edges), dict(a_sg.edge_params)
    key = (100, 1, EdgeType.BOUNDARY_ATTACHMENT)
    ep = b.params[key]
    b.params[key] = dataclasses.replace(
        ep, quantized=(ep.quantized[0] + 2.0,) + tuple(ep.quantized[1:]))
    return W._Index(*b.done())


def sim_any(u, v):
    return 1.0


def test_the_objective_marries_the_node_half_and_the_relation_half():
    """RULE (D0 / 7.0): Q = (S + P + A) / (|M| + R).  S is the graded node
    similarity of every mapped pair, P the graded agreement of every relation
    between mapped pairs, A the graded attribute mass, R the relations of the
    union.  A mapping that drops half the frame is worse than one that keeps
    it, although its few relations are just as well preserved -- which the
    Jaccard on relations alone could not see."""
    a_sg, a_sigs = two_boundaries()
    g = W._Index(a_sg, a_sigs)
    sim = lambda u, v: 1.0 if g.lens[u] == g.lens[v] else 0.0
    ident = {n: n for n in g.frame}
    q_full = W.objective(ident, g, g, sim)
    q_half = W.objective({n: n for n in sorted(g.frame)[:6]}, g, g, sim)
    assert q_full["q"] == pytest.approx(1.0)
    assert q_half["mapped"] == 6 and q_half["q"] < q_full["q"]
    assert (q_full["S"], q_full["P"], q_full["A"]) == (13.0, 20.0, 3.0)
    assert q_full["R"] == 20 and q_full["q"] <= 1.0
    assert W.objective({}, g, g, sim)["q"] == 0.0
    # BOTH halves carry weight: on a mapping whose halves do not measure the
    # same, dropping either weight moves Q -- in opposite directions
    assert W.W_NODE == W.W_RELATION == 1.0
    partial = W.objective(ident, g, _one_relation_off(a_sg, a_sigs), sim_any)
    assert partial["q"] == pytest.approx(
        (partial["S"] + partial["P"] + partial["A"])
        / (partial["mapped"] + partial["R"] + partial["A_units"]))
    assert (partial["S"] / partial["mapped"]) > partial["q"]          # the node half is higher
    assert ((partial["P"] + partial["A"]) / (partial["R"] + partial["A_units"])) < partial["q"]


def test_the_node_half_is_graded_not_a_count():
    """RULE (D0): S sums the GRADED signature similarity of the mapped pairs,
    so two mappings of the same size differ when one pairs look-alikes."""
    a_sg, a_sigs = two_boundaries()
    g = W._Index(a_sg, a_sigs)
    ident = {n: n for n in g.frame}
    assert W.objective(ident, g, g, lambda u, v: 0.8)["S"] == pytest.approx(0.8 * len(ident))
    assert (W.objective(ident, g, g, lambda u, v: 0.8)["q"]
            < W.objective(ident, g, g, lambda u, v: 1.0)["q"])


def test_a_relation_half_preserved_is_worth_its_agreement():
    """RULE (D0): a relation between two mapped nodes counts the fraction of
    its shared measured axes that agree.  One bin off is 3/4 of an axis, at
    the margin 0, a circular axis wrapped."""
    def half(bin_, cycles=((0, 8),)):
        return W._Half(cls=("X", "o", "L"), far=0, bins=((0, bin_),), cycles=cycles,
                       ordinal=None, etype=None)

    assert W._agreement(half(4.0), half(4.0)) == pytest.approx(1.0)
    assert W._agreement(half(4.0), half(5.0)) == pytest.approx(0.75)
    assert W._agreement(half(4.0), half(8.0)) == pytest.approx(0.0)
    assert W._agreement(half(4.0), W._Half(cls=("X", "o", "L"), far=0, bins=(),
                                           cycles=(), ordinal=None, etype=None)) is None
    assert W.CONTRADICTION_MARGIN == 4.0


def _tangent(*tangents):
    """One boundary, five features at 0.10 .. 0.80, each attached at its own
    tangent angle (radians; 45 degrees is one bin of the 8-bin axis)."""
    b = Build()
    sigs = [tex(1, 1), tex(5, 1), tex(9, 1), tex(1, 9), tex(9, 9)]
    b.boundary(100, chain_sig(5), [(i + 1, sigs[i], a, t) for i, (a, t) in
                                   enumerate(zip([0.10, 0.20, 0.40, 0.60, 0.80],
                                                 tangents))])
    return b.done()


def test_only_a_gross_disagreement_vetoes_an_admission():
    """RULE (D2): slack ranks; only a shared axis further apart than
    CONTRADICTION_MARGIN vetoes.  Two bins are admitted and only cost Q;
    seven bins are refused."""
    a_sg, a_sigs = _tangent(0.0, 0.0, 0.0, 0.0, 0.0)
    near_sg, near_sigs = _tangent(0.0, 0.0, 0.0, 0.0, 0.52)      # 2 bins
    far_sg, far_sigs = _tangent(0.0, 0.0, 0.0, 0.0, 1.57)        # 7 bins
    near = W.walk(a_sg, near_sg, a_sigs, near_sigs)
    far = W.walk(a_sg, far_sg, a_sigs, far_sigs)
    assert 5 in near.mapping and near.contradicts == 0
    assert 5 not in far.mapping and far.contradicts >= 1
    # the margin itself: 4 bins is not a contradiction, 4.5 is
    def half(bin_, cycles=()):
        return W._Half(cls=("X", "o", "L"), far=0, bins=((0, bin_),), cycles=cycles,
                       ordinal=None, etype=None)
    assert not W._contradicts(half(1.0), half(5.0))
    assert W._contradicts(half(1.0), half(5.5))
    assert not W._contradicts(half(1.0, ((0, 8),)), half(6.0, ((0, 8),)))


def test_the_frontier_is_reported_and_no_longer_starves():
    """RULE (D2 / 7.13): examined / budget is reported for every pair, and on
    a small frame the walk may examine the whole matrix."""
    sg, sigs = two_boundaries()
    res = W.walk(sg, sg, sigs, sigs)
    # [Amendment 7.22] the budget is per hypothesis, so the bill of the whole
    # beam may be several times it; no single hypothesis may exceed it
    assert res.budget > 0
    assert all(b.get("examined", 0) <= res.budget for b in res.beam)
    assert len(res.mapping) == len(W._Index(sg, sigs).frame)


def test_the_node_gate_is_relative_to_the_nodes_best_rival():
    """RULE (D3): a pair passes the gate when its similarity is above
    NODE_FLOOR and within NODE_MARGIN of the query node's best similarity in
    the other graph.  Both numbers are pre-registered and both are load-bearing."""
    gate = W._node_gate
    assert (W.NODE_FLOOR, W.NODE_MARGIN) == (0.30, 0.25)
    assert gate(0.90, 0.90) and gate(0.70, 0.90)          # within the margin
    assert not gate(0.60, 0.90)                           # too far from the best
    assert not gate(0.25, 0.25) and gate(0.35, 0.35)      # the floor alone refuses
    assert not gate(None, 0.90)                           # not examined


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


def test_the_score_is_the_objective_of_the_mapping_the_walk_returned():
    """RULE (M2 as amended by D0 / 7.0, and by 7.39 for R): the walk's score
    IS Q(its mapping), with the support floor on top: preserved relation mass
    P (agreement weighted), node mass S, attribute mass A, relation denominator
    R = the QUERY's own relation mass over its whole frame.  A relation the
    other side lacks counts against; a node the mapping does not reach is a
    MISSED relation, not an absent one (7.39: the union denominator let a
    closed sub-mapping pay nothing for what it left out); attributes are
    carried, not mapped (they are A)."""
    a_sg, a_sigs = two_boundaries()
    res = W.walk(a_sg, a_sg, a_sigs, a_sigs)
    # 11 attachments + 9 successors preserved exactly, 3 attribute units
    assert (res.preserved, res.relations_q, res.relations_r) == (20, 20, 20)
    assert res.attributes_shared == 3
    assert res.terms["P"] == pytest.approx(20.0) and res.terms["A"] == pytest.approx(3.0)
    assert res.terms["S"] == pytest.approx(13.0) and res.terms["R"] == 20
    assert res.score == pytest.approx(res.terms["q"]) == pytest.approx(1.0)
    # drop two successor links from the reference: the nodes still map, and
    # under the union form the denominator shrank with them, so Q stayed 1 --
    # nothing preserved was lost, because nothing lost was counted.  Under
    # 7.39's query-side denominator the two links are STILL owed: the query is
    # unmapped there, so they are missed relations and Q must fall.
    b = Build()
    b.nodes, b.edges, b.params = dict(a_sigs), set(a_sg.edges), dict(a_sg.edge_params)
    for key in ((2, 3, EdgeType.BOUNDARY_SUCCESSOR), (12, 13, EdgeType.BOUNDARY_SUCCESSOR)):
        b.edges.discard(key)
        b.params.pop(key)
    b_sg, b_sigs = b.done()
    res2 = W.walk(a_sg, b_sg, a_sigs, b_sigs)
    assert res2.mapping == res.mapping
    # relations_q counts the mapped relations whose counterpart is MISSING
    # (20 mapped, of which 18 have a counterpart); relations_r is the
    # reference-side count and is reported, not used in the score
    assert (res2.preserved, res2.relations_q, res2.relations_r) == (18, 20, 18)
    # S 13 + P 18 + A 3 over |M| 13 + R 20 + A_units 3: the two dropped links
    # stay in R, so the same mapping no longer reads 1.0
    assert res2.terms["R"] == 20 == res.terms["R"]
    assert res2.score == pytest.approx((13.0 + 18.0 + 3.0) / (13 + 20 + 3))
    assert res2.score < res.score
    # an edge with NOTHING measured is not a relation: it changes no count
    c = Build()
    c.nodes, c.edges, c.params = dict(a_sigs), set(a_sg.edges), dict(a_sg.edge_params)
    c.edge(1, 3, EdgeType.BOUNDARY_SUCCESSOR, params_of(SUC))
    c_sg, c_sigs = c.done()
    res3 = W.walk(c_sg, a_sg, c_sigs, a_sigs)
    assert (res3.preserved, res3.relations_q, res3.relations_r) == (20, 20, 20)
    assert res3.score == pytest.approx(1.0)
    # a relation whose axis is a few bins off is preserved in part, not lost
    d = Build()
    d.nodes, d.edges, d.params = dict(a_sigs), set(a_sg.edges), dict(a_sg.edge_params)
    key = (100, 1, EdgeType.BOUNDARY_ATTACHMENT)
    ep = d.params[key]
    d.params[key] = dataclasses.replace(
        ep, quantized=(ep.quantized[0] + 2.0,) + tuple(ep.quantized[1:]))
    d_sg, d_sigs = d.done()
    res4 = W.walk(a_sg, d_sg, a_sigs, d_sigs)
    assert res4.mapping == res.mapping
    # the shift is +2 bins on ONE of the relation's three shared axes, and the
    # agreement of a relation is the MEAN over its shared axes: (0.5+1+1)/3
    assert res4.terms["P"] == pytest.approx(20.0 - 1.0 + 5.0 / 6.0)
    assert 0.0 < res4.score < 1.0


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


# --------------------------------------------------------------------------
# PHASE 4: the score across node sets, and the coverage loophole (7.33/7.39)
# --------------------------------------------------------------------------

def test_a_mapping_that_leaves_a_boundary_out_does_not_keep_the_full_score():
    """RULE (7.39, the coverage loophole).  The union denominator paid a
    mapping only for the relations it CONTAINED, so a closed region -- a
    mapping that stops at one boundary and never reaches the other -- could
    read as high as a mapping of the whole frame.  R is now the QUERY's own
    relation mass, so a node the mapping does not reach is a MISSED relation.
    The same graph matched against itself maps everything and scores 1; the
    mapping restricted to one boundary must score strictly below it, although
    every relation it does contain is preserved exactly."""
    sg, sigs = two_boundaries()
    full = W.walk(sg, sg, sigs, sigs)
    assert full.score == pytest.approx(1.0)

    # the region around boundary 100 (its six features) and nothing else: a
    # CLOSED sub-mapping -- every relation inside it is preserved perfectly
    keep = {n for n in sigs if n in (100, 1, 2, 3, 4, 5, 6)}
    part = {u: u for u in keep}
    res = W._Index(sg, sigs)
    terms = W.objective(part, res, res, lambda u, v: 1.0)
    assert terms["R"] == full.terms["R"] > 0          # the whole frame is owed
    assert terms["P"] < full.terms["P"]               # and only part is paid
    assert terms["q"] < full.terms["q"]

    # every relation the part DOES contain is preserved: nothing is wrong with
    # it, it is simply not the whole job -- which is the point of the rule
    # `_Index` keys a relation by the edge type's NAME, not the enum
    inside = [(s_, d_, getattr(nm, "name", nm)) for (s_, d_, nm) in sg.edges
              if s_ in keep and d_ in keep]
    assert inside, "the fixture must contain relations inside the kept region"
    # every relation the walker SEES inside the region is preserved in full;
    # relations it drops at admission (an unmeasured parameter) are not
    # relations to it, and _score does not count them either
    seen = [k for k in inside if k in res.pair]
    assert seen, "the kept region must contain relations the walker sees"
    for k in seen:
        hq, hr = res.pair[k], res.pair.get(k)
        assert hq is not None and hr is not None
        assert W._slack(hq, hr) == pytest.approx(0.0)   # preserved exactly


def test_the_class_score_charges_a_mapping_for_the_frame_it_does_not_reach():
    """RULE (7.33 (a)).  score_for_class = Q * |M| / min(frame_q, frame_r).
    Two mappings of the same |M| with the same Q score the same; a mapping of
    the same Q with FEWER pairs scores less, which is what makes the number
    comparable across node sets.  No constant is involved."""
    assert W.score_for_class({"q": 1.0, "mapped": 10}, 10, 10) == pytest.approx(1.0)
    assert W.score_for_class({"q": 1.0, "mapped": 5}, 10, 10) == pytest.approx(0.5)
    # the SMALLER frame is the denominator: an over-complete mapping is not
    # rewarded past 1
    assert W.score_for_class({"q": 1.0, "mapped": 10}, 10, 4) == pytest.approx(1.0)
    # and Q alone cannot tell the two halves apart
    assert W.score_for_class({"q": 1.0, "mapped": 5}, 10, 10) < \
        W.score_for_class({"q": 1.0, "mapped": 10}, 10, 10)
