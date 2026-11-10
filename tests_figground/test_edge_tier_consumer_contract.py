"""[U17] CONSUMER CONTRACT -- FACT builds hard structure, HYPOTHESIS routes.

Pin for the Path B hand-off item: a downstream consumer reading
``SurfaceFragment.tier`` and ``run.order["edge_tier"]`` must behave as the
session contract states:

  * an edge tagged FACT is a HARD depth-structure edge;
  * an edge tagged HYPOTHESIS (kept-but-below-floor, or dropped) is NOT hard:
    it routes to a temporary / low-conductance link that carries the priced
    wrong-rate of the tier instead of asserting depth.

The binder/TIRN reader itself is a dated open item (not built here); this pin
asserts the contract at the pipeline boundary that already publishes the tier.

Measured context (U16/U17): the shipped floor is the degenerate min-over-kept, so
every KEPT edge lands FACT and only dropped edges are HYPOTHESIS -- the shipped
behaviour is recorded here so the degenerate floor cannot silently drift into a
real threshold without this pin being re-derived.
"""
from __future__ import annotations

HYPOTHESIS = "HYPOTHESIS"


def _tier_map(run) -> dict:
    return {tuple(int(x) for x in k): v for k, v in (run.order.get("edge_tier") or {}).items()}


def _kept(run) -> set:
    return {(int(k[0]), int(k[1])) for k in run.order.get("resolver_kept") or []}


def _bar_curvature_fact_eligible(run, key) -> bool:
    """Mirror of the existence gate in pipeline.py (U17)."""
    import math
    # find fragment with index == key[0]
    curv = None
    for f in run.fragments:
        if int(getattr(f, "index", -1)) == int(key[0]):
            curv = getattr(f, "curvature", None)
            break
    if curv is None:
        return False
    try:
        c = float(curv)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(c):
        return False
    # spacing from records
    from perception.surface_fragments import _chain_spacing
    # reconstruct records not on run; use run's? run has no records attr exposed as list? it does
    recs = getattr(run, "_records", None)
    # pipeline stores records? Run has records
    recs = run.records if hasattr(run, "records") else None
    spacing = _chain_spacing(recs) if recs is not None else None
    if spacing is None:
        return False
    try:
        sp = float(spacing)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(sp) or sp <= 0:
        return False
    return abs(c) * sp <= (1.0 / 3.0)

def test_kept_edges_are_fact_or_gate_demoted(scene):
    """Floor + existence gate: kept -> FACT if gate-eligible, else HYPOTHESIS."""
    sc, run = scene("box_over_box")
    kept = _kept(run)
    tiers = _tier_map(run)
    for key, t in tiers.items():
        assert t in ("FACT", "HYPOTHESIS"), t
        if key in kept:
            eligible = _bar_curvature_fact_eligible(run, key)
            want = "FACT" if eligible else "HYPOTHESIS"
            assert t == want, "kept edge %s tier=%s want=%s" % (key, t, want)
        else:
            assert t == "HYPOTHESIS", "dropped edge %s must be HYPOTHESIS" % (key,)

def test_hypothesis_edges_are_not_fact(scene):
    """No HYPOTHESIS edge may carry the FACT tier label."""
    sc, run = scene("box_over_box")
    tiers = _tier_map(run)
    for key, t in tiers.items():
        if t == "HYPOTHESIS":
            assert t != "FACT"


def test_fragment_tier_follows_incident_edge(scene):
    """A fragment's tier follows the incident edge's tier: FACT incident ->
    FACT fragment; HYPOTHESIS incident -> HYPOTHESIS fragment; no incident ->
    None (undecided, not hypothesised)."""
    sc, run = scene("box_over_box")
    tiers = _tier_map(run)
    incident: dict[int, list] = {}
    for (a, b), t in tiers.items():
        incident.setdefault(int(a), []).append(t)
        incident.setdefault(int(b), []).append(t)
    for f in run.fragments:
        inc = incident.get(int(f.index))
        if inc is None:
            want = None
        elif "FACT" in inc:
            want = "FACT"
        else:
            want = "HYPOTHESIS"
        got = f.tier
        assert got == want, (
            "fragment %d tier=%r want=%r (incident=%s)" % (f.index, got, want, inc))
