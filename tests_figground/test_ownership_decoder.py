"""Finding F1 -- `perception/ownership.py` advertises two functions it does not
define, and the signed decode must round-trip with the producer's quantiser.

`__all__` lists:
    BO_BINS, ownership_signed, ownership_strength, figure_ground_pair,
    combine_ownership_cues, combine_ownership_logodds, ownership_logodds,
    ownership_probability

The module body defines only:
    ownership_signed, ownership_logodds, ownership_probability,
    combine_ownership_logodds

so `ownership_strength`, `figure_ground_pair` and `combine_ownership_cues` are
missing.  The module docstring says `figure_ground_pair` exists precisely so
"the binder can emit the relation as a SIGNED edge (positive coupling binds the
border to the figure it owns, negative coupling (OCCLUSION, sign=-1) marks the
ground / T-occlusion side)" -- i.e. the signed figure-ground EDGE that the whole
U11 thread is about is advertised, documented and absent.  Any
`from perception.ownership import *` raises, and any binder call to
`figure_ground_pair(det)` raises AttributeError.

A private helper `_neighbor(det)` -- the reader that `figure_ground_pair` needs
to name the occluded side -- IS defined and is dead code, which is the
fingerprint of a lost definition rather than a deliberate omission.
"""
from __future__ import annotations

import math

import pytest

from perception import ownership as OWN
from core.quantize import _q_signed
from core.types import Detection, TypedSignature


def _bo_det(signed: float, neighbor=None, partner=None):
    """A BorderOwnership detection exactly as lenses/v2.py emits one."""
    slots = (0.0, float(_q_signed(signed, max_bins=8)), 0.0, 0.0)
    rho = [("entity", 7, 0, "+"), ("vertex", 0, 0, "+")]
    if neighbor is not None:
        rho.append(("neighbor", neighbor, 0, "+"))
    if partner is not None:
        rho.append(("partner", partner, 0, "+"))
    return Detection(k="BorderOwnership", source_layer=1, scale=0,
                     sigma=TypedSignature(
        kind="CANONICAL", class_="BORDER_OWNERSHIP",
        relation="FIGURE_SIDE_FIELD", quantized=tuple(slots)),
        alpha=7, z=abs(signed), t=0, rho=tuple(rho))


# --------------------------------------------------------------------------
# the advertised surface must exist
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(OWN.__all__))
def test_all_exports_are_defined(name):
    assert hasattr(OWN, name), (
        "perception/ownership.py advertises %r in __all__ but does not define "
        "it (F1)" % name)


def test_star_import_does_not_raise():
    ns: dict = {}
    exec("from perception.ownership import *", ns)   # AttributeError today
    assert "ownership_signed" in ns


def test_combine_ownership_cues_is_the_equal_authority_mean():
    """The docstring of `combine_ownership_logodds` names
    `combine_ownership_cues` as "the equal-authority mean used elsewhere"."""
    cues = {"convexity": 0.5, "enclosure": -0.25, "t_junction": None}
    got = OWN.combine_ownership_cues(cues, {"convexity": 3.0,
                                            "enclosure": 1.0,
                                            "t_junction": 9.0})
    # equal authority: weights ignored, None omitted, mean of the votes
    assert got == pytest.approx((0.5 + -0.25) / 2.0, abs=1e-12)


def test_figure_ground_pair_signs_the_edge():
    """+1 binds the border to its own figure; -1 marks the occluded side and
    names the neighbour (the module docstring's own contract)."""
    det = _bo_det(+0.9, neighbor=42)
    got = OWN.figure_ground_pair(det)
    assert got is not None
    other, sign = got
    assert other == 42
    assert sign == -1                      # a named neighbour is OCCLUSION

    det2 = _bo_det(-0.9, neighbor=42)      # the neighbour is the figure
    other2, sign2 = OWN.figure_ground_pair(det2)
    assert (other2, sign2) == (7, +1)      # this detection's own entity

    assert OWN.figure_ground_pair(_bo_det(0.0)) is None   # ambiguous: abstain


# --------------------------------------------------------------------------
# the decode itself (this part is correct and must stay correct)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("signed", [-1.0, -0.75, -0.2, 0.0, 0.2, 0.75, 1.0])
def test_signed_decode_round_trips_the_producer(signed):
    """v2 writes `_q_signed(signed, max_bins=8)`; ownership reads the bin
    CENTRE `(b + 0.5)/8*2 - 1`.  The round trip must land inside the bin."""
    det = _bo_det(signed)
    got = OWN.ownership_signed(det)
    width = 2.0 / OWN.BO_BINS
    assert abs(got - signed) <= width / 2.0 + 1e-12
    assert -1.0 <= got <= 1.0


def test_decode_matches_binder_bo_depth_constraints_reading():
    """ownership.py's stated purpose: one decode so the three readers cannot
    drift.  The only correct reading is `(bin + 0.5)/8*2 - 1`."""
    for b in range(8):
        det = Detection(k="BorderOwnership", source_layer=1, scale=0,
                        sigma=TypedSignature(kind="CANONICAL",
                                             class_="BORDER_OWNERSHIP",
                                             relation="FIGURE_SIDE_FIELD",
                                             quantized=(0.0, float(b), 0.0, 0.0)),
                        alpha=7, z=0.0, t=0, rho=(("entity", 7, 0, "+"),))
        assert OWN.ownership_signed(det) == pytest.approx((b + 0.5) / 8 * 2 - 1)


def test_short_signature_abstains():
    _base = dict(k="BorderOwnership", source_layer=1, scale=0,
                sigma=None, alpha=7, z=0.0, t=0, rho=())
    assert OWN.ownership_signed(Detection(
        k="BorderOwnership", source_layer=1, scale=0,
        sigma=TypedSignature(kind="CANONICAL", quantized=(1.0,)), alpha=7,
        z=0.0, t=0, rho=())) == 0.0
    assert OWN.ownership_signed(Detection(
        k="BorderOwnership", source_layer=1, scale=0, sigma=None, alpha=7,
        z=0.0, t=0, rho=())) == 0.0


# --------------------------------------------------------------------------
# the two combiners
# --------------------------------------------------------------------------

def test_logodds_and_tanh_agree_in_sign_and_none_is_a_non_vote():
    cues = {"convexity": 0.6, "enclosure": 0.4, "t_junction": None}
    w = {"convexity": 1.0, "enclosure": 1.2, "t_junction": 0.8}
    raw = OWN.ownership_logodds(cues, w)
    squashed = OWN.combine_ownership_logodds(cues, w)
    assert raw > 0.0
    assert squashed > 0.0
    assert squashed == pytest.approx(math.tanh(raw), abs=1e-12)
    # a None cue must not dilute: dropping its weight changes nothing
    w2 = {"convexity": 1.0, "enclosure": 1.2}
    assert OWN.ownership_logodds(cues, w) == pytest.approx(
        OWN.ownership_logodds(cues, w2))


def test_logodds_is_unbounded_but_tanh_is_clamped():
    """The module documents WHY the raw sum is kept: the tanh compresses a
    single strong vote into the same band as three weak ones."""
    strong = OWN.ownership_logodds({"enclosure": 0.999}, {"enclosure": 1.2})
    weak = OWN.ownership_logodds({"a": 0.2, "b": 0.2, "c": 0.2},
                                 {"a": 0.4, "b": 0.4, "c": 0.4})
    assert strong > 5.0                     # raw is unbounded
    assert OWN.combine_ownership_logodds({"enclosure": 0.999},
                                         {"enclosure": 1.2}) == pytest.approx(1.0, abs=1e-6)
    assert weak < strong


def test_probability_is_monotone_in_magnitude_and_saturated():
    assert OWN.ownership_probability(0.0) == pytest.approx(0.5)
    assert OWN.ownership_probability(-3.0) == OWN.ownership_probability(3.0)
    assert OWN.ownership_probability(3.0) > OWN.ownership_probability(1.0)
    assert OWN.ownership_probability(1e9) == 1.0
    # the SIGN is the claim, so a negative slope is not a mirror: confidence
    # can only depend on |evidence|, and a negative slope means "less sure".
    assert OWN.ownership_probability(-1e9, slope=-1.0) == 0.0
