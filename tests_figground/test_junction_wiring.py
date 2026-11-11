"""Finding F4 -- the live T-junction detector exists and is never called, so
ownership's T-junction cue is a permanent non-vote AND the consensus has no
junction mask.

Grep over the supplied tree:

    perception/contour_structure.py:167  def _detect_t_junctions(...)   # defined
    (no other hit)                                                     # never called

    perception/fragment_ownership.py:311 getattr(observation, "get_t_junctions", None)
    perception/fragment_ownership.py:520 t_junction_positions=None     # never passed

`lenses/v2.py::BorderOwnershipLens.step` calls

    fragment_ownership_records(chain_dets, anchor_positions,
                               observation=observation,
                               extraction_scale=extraction_scale)

with no `t_junction_positions`, and the observation type exposes no
`get_t_junctions`.  So `_t_junctions` returns [] and:

  1. `_t_junction_cue` returns None for every member -- a cue with a fitted
     weight (`DEFAULT_WEIGHTS["t_junction"] = 0.8`) that never votes, so the
     log-odds are convexity + enclosure only;
  2. `_consensus`'s `junction` mask is all-False, so the ONLY places ownership
     may keep its local value are corners (|turn| >= BO_CORNER_FLIP_MIN_TURN).
     At a junction the local value is smoothed away by the moving average --
     the consensus is ERASING junction switches, not protecting them.

These tests pin both halves, plus the record-shape fragility of the explicit
path (`rec[2] if (not isinstance(rec, (tuple, list)) or len(rec) == 0) else rec`
-- a condition that reads as if it were written for a different record shape).
"""
from __future__ import annotations

import math

import pytest

from perception import fragment_ownership as FO
from perception import contour_structure as CS


def test_ownership_reads_a_getter_the_observation_does_not_have():
    """The shipped wiring: `getattr(observation, "get_t_junctions", None)`.
    An observation without it yields no sites, so the cue cannot vote."""
    class Shipped:
        def get_luminance(self):
            return None

    assert FO._t_junctions(Shipped(), None) == []
    assert FO._t_junctions(None, None) == []


def test_t_junction_cue_is_a_non_vote_without_sites():
    assert FO._t_junction_cue((10.0, 10.0), (0.0, -1.0), [], 3.0) is None


# --------------------------------------------------------------------------
# consequence 1: the cue's weight is dead mass in the log-odds
# --------------------------------------------------------------------------

def test_weights_include_a_cue_that_never_votes():
    w = FO._weights()
    assert "t_junction" in w and w["t_junction"] != 0.0
    cues = {"convexity": 0.3, "enclosure": 0.6, "t_junction": None}
    from perception.ownership import ownership_logodds
    with_cue = ownership_logodds(cues, w)
    without = ownership_logodds({"convexity": 0.3, "enclosure": 0.6}, w)
    assert with_cue == pytest.approx(without), (
        "a None cue must be a non-vote (it is), so the fitted t_junction "
        "weight contributes nothing on the shipped wiring")


# --------------------------------------------------------------------------
# consequence 2: the consensus has no junction mask
# --------------------------------------------------------------------------

def _ring(n=40, r=40.0, cx=96.0, cy=96.0):
    pts = []
    for i in range(n):
        ang = 2.0 * math.pi * i / n
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def test_consensus_smothers_a_junction_switch_when_no_sites_are_supplied():
    """A chain whose local side vote flips at ONE member -- the signature of a
    junction, where ownership legitimately changes side.  With an empty `tjs`
    and no corner there, the moving average erases the flip.  Supplying the site
    must preserve it: that is the entire purpose of the junction mask."""
    import parameters as P
    pts = _ring(n=40)
    n = len(pts)
    frames = FO._frame(pts, closed=True)
    s = [0.9] * n
    flip = 5
    s[flip] = -0.9                              # the junction switch
    frames[flip] = (frames[flip][0], frames[flip][1], 0.0)   # not a corner
    radius = float(P.BO_PARTNER_RADIUS_SCALE) * 2.5

    out_no_sites = FO._consensus(list(s), frames, True, pts, [], radius)
    site = [pts[flip]]
    out_with_sites = FO._consensus(list(s), frames, True, pts, site, radius)

    assert out_with_sites[flip] < 0.0, (
        "with the junction site supplied the local switch must survive")
    assert out_no_sites[flip] > 0.0, (
        "PRECONDITION of the finding: with no sites the switch is smoothed "
        "away (got %.3f).  If this stops holding, the consensus changed and "
        "the wiring finding must be re-measured." % out_no_sites[flip])


# --------------------------------------------------------------------------
# the explicit path: record shape
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rec,expected", [
    ((0, 1, (12.0, 34.0), 0.9), (12.0, 34.0)),      # _detect_t_junctions shape
    ((12.0, 34.0), (12.0, 34.0)),                   # a bare point
    ([5.0, 6.0], (5.0, 6.0)),                       # a bare point, list
])
def test_explicit_junction_records_are_read_as_contact_points(rec, expected):
    got = FO._t_junctions(None, [rec])
    assert got == [expected], (
        "explicit junction record %r was read as %r, expected the contact "
        "point %r (F4c)" % (rec, got, expected))


def test_explicit_and_observation_paths_agree():
    """`_detect_t_junctions` returns (stem, top, contact, confidence) and the
    observation path reads `rec[2]`.  The explicit path must read the same
    field, or the two wirings disagree about where the junction is."""
    rec = (3, 7, (20.0, 30.0), 0.75)

    class Wired:
        def get_t_junctions(self):
            return [rec]

    from_obs = FO._t_junctions(Wired(), None)
    from_explicit = FO._t_junctions(None, [rec])
    assert from_obs == from_explicit == [(20.0, 30.0)]


def test_malformed_records_are_skipped_not_raised():
    recs = [None, (), (1, 2), (1, 2, None, 0.5), "junk", (1, 2, (3.0,), 0.5)]
    got = FO._t_junctions(None, recs)          # must not raise
    assert all(isinstance(p, tuple) and len(p) == 2 for p in got)
