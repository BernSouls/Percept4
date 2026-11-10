"""matching/population_distance.py -- graded population distance over edge axes.

The metric half of the U31 matcher upgrade: **discrete topology + graded
populations**.  This module is deliberately UNWIRED: nothing imports it yet, so
landing it cannot change a single match score.  See diagnostics/_matcher_audit.md
§7 for the derivation and §10 for the staged roll-out.

Design (stated so it can be audited):

* **One distance per axis, in the axis's own unit.**  The activity vectors are
  distributions over bin centres; the distance between them is the 1-D
  Wasserstein-1 (earth mover's) distance, which transports mass instead of
  charging a full bin for a bump that moved half a bin.  It is therefore the
  continuous reading of a quantity whose bins are only a lookup index -- the
  codebase's own representation thesis (core/population.py).

* **Topology comes from the axis declaration.**  A circular axis (axial bearing
  over [0, pi), polar angle over [0, 2*pi), a wrapped turn angle) uses the
  classic circular EMD: the optimal flow adds one constant circulation, and the
  cost is minimised at the MEDIAN of the CDF difference.  A linear axis cuts at
  the ends.  No doubling trick, no special-casing per category.

* **ABSENT is not a value.**  A slot the writer could not measure carries a
  uniform vector.  A uniform vector is skipped, never scored: absence lowers
  the reported COVERAGE of the comparison and contributes no evidence either
  way.  (The existing `population_similarity` divides by every axis weight,
  including absent ones, so absence currently dilutes the score -- audit M5.)

* **Lookup-key slots are topology, not distance.**  A categorical slot
  (`topo_class`, `source_cue`, `relation`, `side`) is gated by
  `identity_mismatch`, never by the mover's distance.

* **Deterministic.**  Sorted/fixed iteration, `math.fsum` for the sums, no
  randomness, no wall clock.  Same inputs -> bit-identical outputs.

Pure: imports only `core.schema` (the U30 unit table) and stdlib.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from core.schema import U30_EDGE_UNIT_TABLE, UNKNOWN

#: Gaussian width on the NORMALISED distance, matching the codebase's
#: `range / 3` convention for graded metrics.  At d = 1/3 the similarity is
#: 0.61; at d = 1 it is 0.011.
DEFAULT_SIGMA = 1.0 / 3.0

#: Absence is NEUTRAL (AGIFINAL): an unmeasured axis contributes no evidence
#: and must never dilute or penalise the score.  With
#: `conf = floor + (1 - floor) * coverage`, neutrality is floor = 1.0 (floor
#: 0.0 is the MAXIMUM discount).  `coverage` is still returned as the size of
#: the evidence base; callers that want to weight it can, but the similarity
#: itself is not charged for what is missing.
DEFAULT_COVERAGE_FLOOR = 1.0

#: Per-axis veto: an axis at the largest distance its own declaration allows
#: is MAXIMALLY wrong.  One such axis zeroes the comparison, so four good axes
#: can never outvote a decisive mismatch.  Measured in normalised units, so
#: the threshold is the same for every axis.
DEFAULT_VETO_DISTANCE = 1.0

#: Two vectors within this of each other are the same constant -> uniform.
_UNIFORM_TOL = 1e-9


@dataclass(frozen=True)
class AxisUnit:
    """One graded axis plus the PHYSICAL declaration its distance needs.

    `core.population.PopAxis` carries the bin count and circularity but not the
    span, because it was built for the overlap metric (span-free).  A distance
    in the axis's own unit needs the span, so this carries it alongside.
    """
    name: str
    n_bins: int
    lo: float = 0.0
    hi: float = 1.0
    circular: bool = False
    weight: float = 1.0
    quantiser: str = "floor"

    @property
    def span(self) -> float:
        return float(self.hi) - float(self.lo)

    @property
    def is_key(self) -> bool:
        """A categorical lookup key: topology, not a graded quantity."""
        return self.quantiser == "lookup_key"

    @property
    def default_sigma(self) -> float:
        """ONE BIN of slack, on the normalised distance scale.

        Derived from the axis's own resolution, not a global constant: a
        12-bin circular axis normalises by span/2, so one bin is 2/n; a
        linear axis normalises by span*(n-1)/n, so one bin is 1/(n-1).
        """
        if self.circular:
            return 2.0 / float(max(self.n_bins, 2))
        return 1.0 / float(max(self.n_bins - 1, 1))

    @property
    def max_w1(self) -> float:
        """The largest mover's distance the axis can express (normaliser).

        Circular: antipodal point masses travel span/2 (half the
        circumference).  Linear: point masses at the two extreme BIN CENTRES
        travel (n-1)/n of the span, not the full span (the span includes the
        half-bin margins outside the outermost centres)."""
        if self.circular:
            return self.span / 2.0
        n = max(self.n_bins, 2)
        return self.span * (n - 1) / float(n)


def u30_axis_units(class_name: str) -> Tuple[AxisUnit, ...]:
    """The graded axis declarations of one U30 edge class ("EDGE_PROXIMITY")."""
    specs = U30_EDGE_UNIT_TABLE.get(class_name, ())
    return tuple(
        AxisUnit(
            name=s.name,
            n_bins=int(s.n_bins),
            lo=float(s.lo),
            hi=float(s.hi),
            circular=bool(s.circular),
            quantiser=str(getattr(s, "quantiser", "floor")),
        )
        for s in specs
    )


def category_axis_units(category: Any) -> Tuple[AxisUnit, ...]:
    """`u30_axis_units` for an `EdgeCategory` member or a bare name."""
    name = getattr(category, "name", None)
    if name is None:
        name = str(category)
    if not name.startswith("EDGE_"):
        name = "EDGE_" + name
    return u30_axis_units(name)


def _as_floats(vec: Sequence[float]) -> list:
    return [float(v) for v in vec]


def _is_uniform(vec: Sequence[float]) -> bool:
    """True when the vector carries no usable measurement.

    Informativeness is judged at the axis's own resolution: a population is
    uninformative when it displaces less than ONE BIN's share of mass from
    flat (total-variation distance from uniform < 1/n).  The earlier absolute
    tolerance (max - min <= 1e-9) accepted a vector with 0.1% noise as
    informative, so its "mode" was noise and the opposed-axes veto fired on
    pairs with NO evidence (3.6-7.8% of noise-only axis pairs; a noise-only
    CONTINUITY edge was forced to similarity 0.0 in 17.6% of pairs).
    """
    if not vec:
        return True
    vals = [float(v) for v in vec]
    n = len(vals)
    lo = min(vals)
    hi = max(vals)
    if (hi - lo) <= _UNIFORM_TOL:
        return True
    total = math.fsum(vals)
    if total <= 0.0:
        return True
    flat = 1.0 / n
    tv = 0.5 * math.fsum(abs(v / total - flat) for v in vals)
    return tv < flat


def _cdf(vec: Sequence[float]) -> list:
    """Cumulative distribution, normalised so the last entry is 1."""
    total = math.fsum(float(v) for v in vec)
    if total <= 0.0:
        return [0.0] * len(vec)
    acc = 0.0
    out = []
    for v in vec:
        acc += float(v)
        out.append(acc / total)
    return out


def population_distance(
    vec_a: Optional[Sequence[float]],
    vec_b: Optional[Sequence[float]],
    unit: AxisUnit,
) -> Optional[float]:
    """Normalised mover's distance on one axis, in [0, 1].

    Returns None when the axis carries NO comparable measurement: either side
    missing, arity mismatch, a lookup-key axis (topology, not distance), or a
    uniform (uninformative) vector on either side.  None is "no evidence", and
    must never be read as 0.0 (identical) or 1.0 (maximally different).
    """
    if vec_a is None or vec_b is None:
        return None
    if unit.is_key or unit.n_bins < 2 or unit.max_w1 <= 0.0:
        return None
    a = _as_floats(vec_a)
    b = _as_floats(vec_b)
    if len(a) != unit.n_bins or len(b) != unit.n_bins:
        return None
    if _is_uniform(a) or _is_uniform(b):
        return None

    n = unit.n_bins
    delta = unit.span / float(n)
    ca, cb = _cdf(a), _cdf(b)
    if unit.circular:
        # Optimal circulation: subtract the median of the CDF difference.
        diff = [ca[k] - cb[k] for k in range(n)]
        med = sorted(diff)[n // 2]
        w1 = delta * math.fsum(abs(d - med) for d in diff)
    else:
        # The last CDF difference is identically 0 (both distributions sum to 1).
        w1 = delta * math.fsum(abs(ca[k] - cb[k]) for k in range(n - 1))
    return min(1.0, max(0.0, w1 / unit.max_w1))


def axis_similarity_from_distance(distance: float, sigma: float = DEFAULT_SIGMA) -> float:
    """Gaussian reading of a normalised axis distance, in (0, 1]."""
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    return float(math.exp(-0.5 * (float(distance) / sigma) ** 2))


def _mode_bin(vec: Sequence[float]) -> int:
    """Deterministic mode of a population vector (smallest index on ties)."""
    best_i, best_v = 0, float(vec[0])
    for i in range(1, len(vec)):
        v = float(vec[i])
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def axes_opposed(vec_a: Sequence[float], vec_b: Sequence[float],
                 unit: AxisUnit) -> bool:
    """True when the two supports sit at the axis's MAXIMUM separation.

    `population_distance` reaches 1.0 only for point masses; a soft bump
    shifted antipodally tops out near 0.88 (the kernel's width costs a little
    transport), so a literal `distance >= 1.0` veto would never fire on real
    emitted populations.  The veto therefore reads the axis's own topology:
    the two ends of a linear axis, or antipodal modes on a circular one.
    Either is "maximally wrong" in the units the axis declares.
    """
    a = _as_floats(vec_a)
    b = _as_floats(vec_b)
    if len(a) != unit.n_bins or len(b) != unit.n_bins:
        return False
    if _is_uniform(a) or _is_uniform(b):
        return False
    i, j = _mode_bin(a), _mode_bin(b)
    if unit.circular:
        sep = abs(i - j)
        sep = min(sep, unit.n_bins - sep)
        return 2 * sep >= unit.n_bins
    return abs(i - j) >= unit.n_bins - 1


def population_similarity_masked(
    pop_a: Optional[Sequence[Sequence[float]]],
    pop_b: Optional[Sequence[Sequence[float]]],
    units: Sequence[AxisUnit],
    sigma: Optional[float] = None,
    agg: str = "mean",
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
    veto: bool = True,
) -> Tuple[Optional[float], float]:
    """Axis similarity over the axes BOTH sides measured, plus its coverage.

    Returns ``(similarity, coverage)``:
      * similarity: `raw * (floor + (1 - floor) * coverage)`, where `raw` is the
        weighted mean (or the MINIMUM, agg="min") over the measured axes only.
        ABSENT axes cannot dilute `raw`; the floor makes a thin evidence base
        pay, so one measured axis cannot read the same as four.
      * coverage: measured weight / total non-key weight, in [0, 1].

    `sigma=None` uses each axis's own one-bin width (`AxisUnit.default_sigma`).
    Vectors are normalised to distributions before the distance is taken, so
    raw mass cannot change the result.
    """
    if not units:
        return None, 0.0
    total_w = 0.0
    measured: list = []
    vetoed = False
    for i, unit in enumerate(units):
        if unit.is_key:
            continue
        total_w += unit.weight
        if pop_a is None or pop_b is None:
            continue
        if i >= len(pop_a) or i >= len(pop_b):
            continue
        d = population_distance(pop_a[i], pop_b[i], unit)
        if d is None:
            continue
        if veto and (d >= DEFAULT_VETO_DISTANCE - 1e-9
                     or axes_opposed(pop_a[i], pop_b[i], unit)):
            vetoed = True
        s = axis_similarity_from_distance(
            d, unit.default_sigma if sigma is None else sigma)
        measured.append((unit.weight, s))
    if not measured:
        return None, 0.0
    den = math.fsum(w for w, _ in measured)
    if den <= 0.0:
        return None, 0.0
    if agg == "min":
        raw = min(s for _, s in measured)
    elif agg == "mean":
        raw = math.fsum(w * s for w, s in measured) / den
    else:
        raise ValueError(f"unknown aggregation {agg!r}")
    coverage = (den / total_w) if total_w > 0.0 else 0.0
    if vetoed:
        # A decisive mismatch: no amount of good evidence outvotes it.
        return 0.0, min(1.0, max(0.0, coverage))
    floor = min(1.0, max(0.0, float(coverage_floor)))
    conf = floor + (1.0 - floor) * min(1.0, coverage)
    return min(1.0, max(0.0, raw * conf)), min(1.0, max(0.0, coverage))


def per_axis_distances(
    pop_a: Optional[Sequence[Sequence[float]]],
    pop_b: Optional[Sequence[Sequence[float]]],
    units: Sequence[AxisUnit],
) -> Tuple[Dict[str, Optional[float]], float]:
    """Per-axis normalised distances (for diagnostics / a pre-filter), plus
    the measured weight fraction.  `None` marks an axis with no evidence."""
    out: Dict[str, Optional[float]] = {}
    measured_w = 0.0
    total_w = 0.0
    for i, unit in enumerate(units):
        if unit.is_key:
            continue
        total_w += unit.weight
        if pop_a is None or pop_b is None or i >= len(pop_a) or i >= len(pop_b):
            out[unit.name] = None
            continue
        d = population_distance(pop_a[i], pop_b[i], unit)
        out[unit.name] = d
        if d is not None:
            measured_w += unit.weight
    return out, (measured_w / total_w if total_w > 0.0 else 0.0)


def _is_measured(value: float) -> bool:
    if value == UNKNOWN:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def identity_mismatch(
    quantized_a: Sequence[float],
    quantized_b: Sequence[float],
    units: Sequence[AxisUnit],
) -> bool:
    """True when the DISCRETE topology contradicts: a lookup-key slot differs.

    Follows the codebase's NAP convention: UNKNOWN on either side is a wildcard
    and passes; a measured difference of at least half a bin is a contradiction.
    """
    for i, unit in enumerate(units):
        if not unit.is_key:
            continue
        if i >= len(quantized_a) or i >= len(quantized_b):
            continue
        a = quantized_a[i]
        b = quantized_b[i]
        if not _is_measured(a) or not _is_measured(b):
            continue
        if abs(float(a) - float(b)) >= 0.5:
            return True
    return False


def edge_population_distance(
    ep_a: Any,
    ep_b: Any,
    class_name: Optional[str] = None,
    units: Optional[Sequence[AxisUnit]] = None,
) -> Tuple[Optional[float], float]:
    """Convenience for two `EdgeParams`: (similarity, coverage).

    Returns (None, 0.0) when either side carries no stored population, which is
    the signal for the caller to keep the existing bins path untouched.
    """
    if ep_a is None or ep_b is None:
        return None, 0.0
    cls = class_name or getattr(ep_a, "class_", None) or getattr(ep_b, "class_", None)
    if not cls:
        return None, 0.0
    if getattr(ep_a, "class_", None) != getattr(ep_b, "class_", None):
        return None, 0.0
    if units is None:
        units = u30_axis_units(str(cls))
    if not units:
        return None, 0.0
    pop_a = getattr(ep_a, "population", ()) or ()
    pop_b = getattr(ep_b, "population", ()) or ()
    if not pop_a or not pop_b:
        return None, 0.0
    return population_similarity_masked(pop_a, pop_b, units)


# ---------------------------------------------------------------------------
# Self-test: the invariants the live wiring will rely on.
# ---------------------------------------------------------------------------

def _selftest() -> None:
    from core.population import soft_bump_bins

    # Linear axis, 8 bins over [0, 8) (spacings).
    lin = AxisUnit("distance", 8, 0.0, 8.0, circular=False)
    a = soft_bump_bins(2.0, _axis(lin))
    b = soft_bump_bins(2.0, _axis(lin))
    c = soft_bump_bins(6.0, _axis(lin))
    assert population_distance(a, b, lin) == 0.0, "identical must be 0"
    d = population_distance(a, c, lin)
    assert d is not None and 0.4 < d <= 1.0, d
    assert d == population_distance(c, a, lin), "symmetry"
    assert 0.0 <= d <= 1.0

    # Uniform => ABSENT (no evidence), never a penalty.
    uniform = tuple([1.0 / 8] * 8)
    assert population_distance(uniform, a, lin) is None
    sim, cov = population_similarity_masked(
        (uniform, a), (uniform, b), (lin, lin), coverage_floor=0.0)
    # One of two axes is ABSENT: raw is 1.0, coverage 0.5, floor 0.0 -> 0.5.
    assert sim is not None and abs(sim - 0.5) < 1e-12, (sim, cov)
    assert abs(cov - 0.5) < 1e-12, cov
    # Absence is neutral: one identical measured axis reads 1.0, not 0.75.
    sim_floor, _ = population_similarity_masked((uniform, a), (uniform, b),
                                                (lin, lin))
    assert sim_floor is not None and abs(sim_floor - 1.0) < 1e-12, sim_floor
    # Both axes measured identically: coverage 1.0 -> similarity 1.0.
    sim_full, cov_full = population_similarity_masked(
        (a, a), (b, b), (lin, lin))
    assert abs(sim_full - 1.0) < 1e-12 and abs(cov_full - 1.0) < 1e-12

    # No comparable axis at all => None, not 0.0.
    sim_none, cov_none = population_similarity_masked((uniform,), (uniform,), (lin,))
    assert sim_none is None and cov_none == 0.0

    # Per-axis veto: opposite extremes on ONE axis zero the comparison even
    # though the other measured axis is identical.
    far_a = (1.0,) + (0.0,) * 7
    far_b = (0.0,) * 7 + (1.0,)
    sim_veto, cov_veto = population_similarity_masked(
        (far_a, a), (far_b, b), (lin, lin), coverage_floor=0.0)
    assert sim_veto == 0.0 and abs(cov_veto - 1.0) < 1e-12, (sim_veto, cov_veto)

    # Circular axis over [0, pi) for 12 bins: 0.05pi and 0.95pi are adjacent.
    circ = AxisUnit("bearing", 12, 0.0, math.pi, circular=True)
    edge_lo = soft_bump_bins(0.05 * 12, _axis(circ, 12))
    edge_hi = soft_bump_bins(0.95 * 12, _axis(circ, 12))
    mid = soft_bump_bins(6.0, _axis(circ, 12))
    assert population_distance(edge_lo, edge_hi, circ) < 0.3, "wrap must be cheap"
    assert population_distance(edge_lo, mid, circ) > 0.5, "opposite must be far"

    # Axial fold over [-pi/2, pi/2]: +1.4 and -1.4 are neighbours.
    fold = AxisUnit("turn_angle", 12, -0.5 * math.pi, 0.5 * math.pi, circular=True)
    plus = soft_bump_bins((1.4 - fold.lo) / fold.span * 12, _axis(fold, 12))
    minus = soft_bump_bins((-1.4 - fold.lo) / fold.span * 12, _axis(fold, 12))
    assert population_distance(plus, minus, fold) < 0.3, "fold must be cheap"

    # Identity gate: lookup-key slots, with UNKNOWN as wildcard.
    # EDGE_JUNCTION = (angle_delta, topo_class[key], arm_count).
    units = u30_axis_units("EDGE_JUNCTION")
    assert units and units[1].is_key
    assert identity_mismatch((0.0, 0.0, 3.0), (0.0, 1.0, 3.0), units) is True
    assert identity_mismatch((0.0, 2.0, 3.0), (0.0, 2.0, 5.0), units) is False
    assert identity_mismatch((0.0, UNKNOWN, 3.0), (0.0, 4.0, 3.0), units) is False

    # Lookup-key axes never enter the graded mean: EDGE_OCCLUSION has two keys
    # and two graded axes; with everything uniform there is no evidence at all.
    occ = u30_axis_units("EDGE_OCCLUSION")
    occ_pop = tuple(tuple([1.0 / u.n_bins] * u.n_bins) for u in occ)
    sim_key, cov_key = population_similarity_masked(occ_pop, occ_pop, occ)
    assert sim_key is None and cov_key == 0.0, (sim_key, cov_key)
    print("population_distance self-test: OK")


def _axis(unit: AxisUnit, n_bins: Optional[int] = None):
    """A PopAxis view of an AxisUnit, for soft_bump_bins in the self-test."""
    from core.population import PopAxis
    return PopAxis(unit.name, n_bins or unit.n_bins, circular=unit.circular)


if __name__ == "__main__":
    _selftest()
