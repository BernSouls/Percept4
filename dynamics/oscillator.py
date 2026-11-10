"""dynamics/oscillator.py -- Stage 6 coupled-phase oscillator (pure math).

No extraction, no perception: this module only manipulates phase vectors and a
coupling matrix, so it is testable without any image.

Model
-----
Kuramoto-style coupled phases,

    theta_i' = omega_i + K * sum_j w_ij * sin(theta_j - theta_i)

with three properties the caller depends on:

* **Gauge-pinned.** Only phase DIFFERENCES carry information, so every step the
  global mean phase is subtracted. Without pinning the whole configuration
  drifts and two runs of the same input are not comparable.
* **Co-rotating by default.** `omega` is zero unless the caller sets it, i.e.
  the phases are integrated in the frame that rotates with the mean field. A
  node with a nonzero omega does not lock unless the coupling beats its
  frequency offset, which is a *model choice*, so it must be explicit.
* **Tolerance equality.** Partitions are compared with a tolerance, never with
  `==`, because fixed-point iteration only converges up to float error.

Anti-phase coupling is expressed by negative edge weights: a negative w_ij makes
sin(theta_j - theta_i) push theta_j - theta_i toward pi rather than 0, which is
what splits two cliques into anti-phase clusters.

Stability of the explicit Euler update
--------------------------------------
This is forward Euler, so the textbook condition applies and there is no new
theorem here. Linearising one step around a configuration gives

    delta' = (I + dt*K*M)*delta,

where M is the coupling Laplacian built from the effective edge weights
w_ij*cos(theta_j - theta_i). A mode with eigenvalue mu therefore carries the
multiplier 1 - dt*K*mu, and the map is stable only while dt*K*mu < 2. At
mu = 2/dt*K the multiplier is exactly -1, which is the period-2 boundary: past
it the map overshoots, the sine nonlinearity bounds the amplitude, and the
intended partition is lost (measured in
work/diagnostics/_exp_oscillator_instability.py).

mu grows with coupling degree, so the SAME coupling that is stable for a small
clique is unstable for a large one. That is what the `normalize` modes fix:

* ``"row"`` (default) divides each row by sum_j|w_ij|, so every node's total
  coupling weight is exactly 1. M is then the random-walk normalised
  Laplacian G*(I - D^-1*A), a diagonal contractive scaling of a matrix similar
  to a symmetric PSD matrix, so mu <= 2 for EVERY graph regardless of size or
  hub structure, and the criterion collapses to dt*K < 1. Convergence speed
  also stops depending on graph size, because no node's drive is diluted by a
  large neighbourhood. The cost is that total evidence is discarded: a node
  with many weak edges gets the same drive as one with a single strong edge.
* ``"row_sat"`` re-weights each normalised row by the saturating gain
  g(d) = d/(d + c) of its original weight sum d, restoring evidence-dependent
  drive while every row stays a contraction, so mu <= 2 still holds. `c` is
  UNMEASURED and must be swept, not guessed: log it with the fingerprint.
* ``"spectral"`` divides the whole matrix by its spectral radius. This also
  bounds mu, but it couples every node to the global radius, so a small clique
  embedded in a large graph is dragged down and may not converge within a
  fixed settle budget. Kept for comparison, not as the default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

#: Normalisation modes accepted by `PhaseOscillator(normalize=...)`.
NORMALIZE_MODES = ("row", "row_sat", "spectral", "none")


def _check_mode(mode: Optional[str]) -> str:
    """Validate and canonicalise a normalisation mode."""
    if mode is None or mode == "none":
        return "none"
    if mode not in NORMALIZE_MODES:
        raise ValueError(
            "normalize must be one of %s or None, got %r"
            % (list(NORMALIZE_MODES), mode))
    return mode


def row_normalize(w: np.ndarray) -> np.ndarray:
    """Divide each row by its sum of absolute weights.

    Rows with no coupling keep zero weight (a division by zero must not
    produce NaN, because a NaN silently turns the whole node's drive into NaN).
    """
    d = np.abs(w).sum(axis=1, keepdims=True)
    out = np.zeros_like(w, dtype=np.float64)
    return np.divide(w, d, out=out, where=d > 0)


def spectral_normalize(w: np.ndarray) -> np.ndarray:
    """Scale a coupling matrix so its spectral radius is at most 1.

    Only the magnitude is rescaled: the sign structure (which encodes in-phase
    vs anti-phase coupling) and the relative weights are kept. An all-zero
    coupling has radius 0 and is returned unchanged.
    """
    ws = 0.5 * (w + w.T)
    if ws.size == 0:
        return w
    radius = float(np.max(np.abs(np.linalg.eigvalsh(ws))))
    if radius <= 0.0 or not np.isfinite(radius):
        return w
    return w / radius if radius > 1.0 else w


def saturating_row_normalize(w: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Row-normalise, then scale each row by the saturating gain g(d).

    d_i = sum_j|w_ij| is the node's total coupling evidence and
    g(d) = d/(d + c) is in (0, 1) for d > 0, so every row stays a contraction
    and the eigenvalue bound survives. Nodes with a lot of evidence keep nearly
    their full normalised drive; weakly-evidenced nodes keep less.
    """
    rn = row_normalize(w)
    d = np.abs(w).sum(axis=1)
    if c is None or float(c) <= 0.0:
        return rn
    gain = d / (d + float(c))
    return rn * gain[:, None]


def _normalize_matrix(w: np.ndarray, mode: str,
                      gain_c: Optional[float] = None) -> np.ndarray:
    if mode == "row":
        return row_normalize(w)
    if mode == "row_sat":
        return saturating_row_normalize(w, c=1.0 if gain_c is None else gain_c)
    if mode == "spectral":
        return spectral_normalize(w)
    return w


@dataclass
class SettleResult:
    """Outcome of a convergence-aware settle.

    `converged` is the caller's gate: a non-converged graph is UNDECIDED, not a
    clustering. Never read the partition of a result with
    `converged is False` as if it were stable.
    """

    converged: bool
    steps: int
    final_jump: float
    peak_jump: float
    locked_fraction: float
    theta: np.ndarray


class PhaseOscillator:
    """Coupled phase network over n nodes with a fixed coupling matrix.

    STATE (ROUND 11 / O4): DARK — no live caller anywhere in ``src/``; the
    class is exercised only by ``tests/dynamics/test_oscillator.py`` in
    isolation. It is the anti-phase/resonance mechanism parked behind
    ``ANTIPHASE_ENABLED=False``; see
    ``diagnostics/_O4_DEAD_MECHANISM_LEDGER.md``.
    See the module docstring for the stability argument; the short version is
    that the explicit Euler map is stable while dt*K*mu < 2 and the default
    `"row"` normalisation makes mu <= 2 independent of graph size.
    """

    def __init__(
        self,
        n: int,
        coupling: np.ndarray,
        omega: Optional[Sequence[float]] = None,
        k: float = 1.0,
        dt: float = 0.1,
        seed: Optional[int] = None,
        normalize: Optional[str] = "row",
        gain_c: Optional[float] = None,
    ) -> None:
        if coupling.shape != (n, n):
            raise ValueError(f"coupling must be ({n}, {n}), got {coupling.shape}")
        self.n = int(n)
        self.w = np.asarray(coupling, dtype=np.float64)
        self.normalize = _check_mode(normalize)
        self.gain_c = gain_c
        if self.normalize != "none":
            self.w = _normalize_matrix(self.w, self.normalize, self.gain_c)
        self.omega = (np.zeros(self.n) if omega is None
                      else np.asarray(omega, dtype=np.float64))
        if self.omega.shape != (self.n,):
            raise ValueError("omega must have length n")
        self.k = float(k)
        self.dt = float(dt)
        self._rng = np.random.default_rng(seed)
        self.theta = self._rng.uniform(-math.pi, math.pi, size=self.n)

    # -- core update ------------------------------------------------------
    def step(self) -> np.ndarray:
        """Advance one Euler step and re-pin the global phase gauge."""
        d = self.theta[None, :] - self.theta[:, None]          # d[j, i] = theta_i - theta_j
        coupling = self.w * np.sin(d)                          # negative w => anti-phase
        drive = np.sum(coupling, axis=1) * self.k + self.omega
        self.theta = self.theta + self.dt * drive
        # gauge pin: remove the global mean phase
        self.theta = self.theta - float(np.mean(self.theta))
        return self.theta

    def run(self, steps: int) -> np.ndarray:
        """Run a FIXED number of steps and return the phases.

        A fixed step count is not a convergence criterion: use `settle()` when
        the answer must be trusted, and check `SettleResult.converged`.
        """
        for _ in range(int(steps)):
            self.step()
        return self.theta

    def settle(
        self,
        steps: int = 2000,
        eps: float = 1e-3,
        patience: int = 3,
    ) -> SettleResult:
        """Iterate until the configuration stops moving, and say so.

        Stops as soon as every node's per-step phase change stays at or below
        `eps` for `patience` consecutive steps. Runs at most `steps` steps; a
        graph that never settles comes back with `converged=False`, which the
        caller must treat as undecided rather than as a clustering.

        `eps` is deliberately far below the 64-bin readout quantum
        (2*pi/64 = 0.0982 rad) so that a settled state is genuinely static at
        readout resolution, not merely sub-bin.
        """
        max_steps = max(1, int(steps))
        quiet = 0
        final_jump = float("inf")
        peak_jump = 0.0
        quiet_flags = np.zeros(self.n, dtype=bool)
        used = 0
        prev = self.theta
        for used in range(1, max_steps + 1):
            cur = self.step()
            jump = np.abs(cur - prev)
            final_jump = float(np.max(jump))
            peak_jump = max(peak_jump, final_jump)
            quiet_flags = jump <= eps
            prev = cur
            if bool(np.all(quiet_flags)):
                quiet += 1
                if quiet >= max(1, int(patience)):
                    return SettleResult(
                        converged=True, steps=used, final_jump=final_jump,
                        peak_jump=peak_jump,
                        locked_fraction=float(np.mean(quiet_flags)),
                        theta=cur)
            else:
                quiet = 0
        return SettleResult(
            converged=False, steps=max_steps, final_jump=final_jump,
            peak_jump=peak_jump,
            locked_fraction=float(np.mean(quiet_flags)) if self.n else 0.0,
            theta=self.theta)

    # -- readouts ---------------------------------------------------------
    def order_parameter(self) -> Tuple[float, float]:
        """Magnitude and angle of the Kuramoto order parameter."""
        z = complex(0.0, 0.0)
        for t in self.theta:
            z += complex(math.cos(t), math.sin(t))
        z /= max(self.n, 1)
        return abs(z), math.atan2(z.imag, z.real)

    def partition(self, tol: float = 0.25) -> FrozenSet[FrozenSet[int]]:
        """Group nodes whose phases agree within `tol` radians.

        Greedy single-linkage over the circular distance, so the result is a set
        of clusters and never depends on node ordering.
        """
        return self._partition(self.theta, tol)

    def partition_quantized(self, bins: int = 64,
                            tol: float = 0.25) -> FrozenSet[FrozenSet[int]]:
        """Partition after rounding each phase to the nearest of `bins` bins.

        This is what a fixed-width integer readout sees. The internal state
        always stays float: quantising only at readout time is what keeps weak
        coupling from rounding to a standstill, because the motion accumulates
        in the float state before it is ever rounded.

        The bin width (2*pi/bins) must stay below `tol`, otherwise two nodes
        that agree in float can round to distant bins and split a cluster.
        """
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}")
        if (2.0 * math.pi) / bins > tol:
            raise ValueError(
                "bin width %.4f exceeds tol %.4f: nodes that agree in float "
                "could round apart, so the readout is not comparable"
                % (2.0 * math.pi / bins, tol))
        if self.n == 0:
            return frozenset()
        q = np.round(self.theta * (bins / (2.0 * math.pi)))
        q = np.mod(q + 0.5 * bins, bins) - 0.5 * bins      # wrap into [-bins/2, bins/2)
        phases = q * (2.0 * math.pi / bins)
        return self._partition(phases, tol)

    def _partition(self, phases: np.ndarray,
                   tol: float) -> FrozenSet[FrozenSet[int]]:
        """Single-linkage clustering of `phases` by circular distance."""
        parent = list(range(self.n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = abs(math.atan2(math.sin(phases[i] - phases[j]),
                                   math.cos(phases[i] - phases[j])))
                if d <= tol:
                    parent[find(i)] = parent[find(j)]

        groups: Dict[int, List[int]] = {}
        for i in range(self.n):
            groups.setdefault(find(i), []).append(i)
        return frozenset(frozenset(g) for g in groups.values())

    # -- construction helpers ---------------------------------------------
    @classmethod
    def two_cliques(cls, m: int, k: float = 2.0, anti: float = -1.0,
                    within: float = 1.0, seed: Optional[int] = None,
                    normalize: Optional[str] = "row") -> "PhaseOscillator":
        """Two disjoint cliques of size m coupled anti-phase to each other.

        Intra-clique coupling is positive (`within`); inter-clique coupling is
        negative (`anti`), which is the anti-phase condition the Stage 6 tests
        assert on. Pass `normalize=None` for the raw coupling, which is what
        shows the Euler instability once dt*k*m >= 1; the default `"row"`
        normalisation keeps the anti-phase solution stable for any m.
        """
        n = 2 * m
        w = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                same = (i < m) == (j < m)
                w[i, j] = within if same else anti
        return cls(n, w, k=k, seed=seed, normalize=normalize)



def adjusted_rand_index(a: Sequence[int], b: Sequence[int]) -> float:
    """Adjusted Rand index between two labellings, from the contingency table.

    Returns 1.0 for identical partitions (including two all-singleton
    partitions of the same length), 0.0 when agreement is no better than
    chance, and negative when it is worse than chance.
    """
    a = list(a)
    b = list(b)
    if len(a) != len(b):
        raise ValueError("labellings must have the same length")
    n = len(a)
    if n == 0:
        return 1.0
    pairs: Dict[Tuple[int, int], int] = {}
    for x, y in zip(a, b):
        pairs[(x, y)] = pairs.get((x, y), 0) + 1
    sum_ij = sum(v * (v - 1) / 2.0 for v in pairs.values())
    ci: Dict[int, int] = {}
    cj: Dict[int, int] = {}
    for x in a:
        ci[x] = ci.get(x, 0) + 1
    for y in b:
        cj[y] = cj.get(y, 0) + 1
    sum_i = sum(v * (v - 1) / 2.0 for v in ci.values())
    sum_j = sum(v * (v - 1) / 2.0 for v in cj.values())
    total = n * (n - 1) / 2.0
    expected = sum_i * sum_j / total if total else 0.0
    max_index = (sum_i + sum_j) / 2.0
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 1.0 if abs(sum_ij - expected) < 1e-12 else 0.0
    return (sum_ij - expected) / denom


def connected_components(n: int, edges: Sequence[Tuple[int, int, float, float]]
                         ) -> List[List[int]]:
    """Connected components of the POSITIVE subgraph of a signed edge list.

    This is the baseline a signed oscillator has to beat: it is what you get
    from coupling alone, ignoring every occlusion signal.
    """
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, sign, _w in edges:
        if sign > 0:
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if g]


def _median_of(values: Sequence[float]) -> float:
    """Median of the finite values; 0.0 when none. Deterministic."""
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _spatial_neighbourhood(
    positions: Any, n: int, spacing: Optional[float],
) -> Tuple[List[Optional[Tuple[float, float]]], float, int]:
    """(xy, radius, K) derived from the graph's OWN node spacing.

    radius is the median nearest-neighbour distance (or the caller-supplied
    `spacing`); K is the median number of neighbours within that radius,
    rounded to the nearest integer. Neither is a hand-set constant: both are
    statistics of the positions the graph itself carries. Only measured
    positions take part; unmeasured nodes are not bounded.

    `positions` may be a sequence indexed by node id or a mapping id -> (x, y).
    """
    xy: List[Optional[Tuple[float, float]]] = [None] * n
    for i in range(n):
        if isinstance(positions, dict):
            p = positions.get(i)
        else:
            p = positions[i] if 0 <= i < len(positions) else None
        if p is None or len(p) < 2:
            continue
        x, y = float(p[0]), float(p[1])
        if math.isfinite(x) and math.isfinite(y):
            xy[i] = (x, y)
    if spacing is None or float(spacing) <= 0.0:
        # Co-located pattern nodes (several lens types minted at the same
        # anchor) are legitimate zero-distance couplings but they are NOT the
        # spacing unit: using them makes the median nearest-neighbour distance
        # 0 and disables the bound. Derive the unit from the median POSITIVE
        # nearest-neighbour distance instead; zero-distance pairs still count
        # as neighbours once the radius is known.
        nn: List[float] = []
        for i in range(n):
            if xy[i] is None:
                continue
            best = math.inf
            for j in range(n):
                if i == j or xy[j] is None:
                    continue
                d = math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1])
                if 1e-12 < d < best:
                    best = d
            if math.isfinite(best):
                nn.append(best)
        spacing = _median_of(nn)
    radius = float(spacing)
    if not (radius > 0.0):
        return xy, 0.0, 0
    counts: List[int] = []
    for i in range(n):
        if xy[i] is None:
            continue
        c = 0
        for j in range(n):
            if i == j or xy[j] is None:
                continue
            if math.hypot(xy[i][0] - xy[j][0],
                          xy[i][1] - xy[j][1]) <= radius + 1e-12:
                c += 1
        counts.append(c)
    k = int(math.floor(_median_of(counts) + 0.5)) if counts else 0
    return xy, radius, max(0, k)


def signed_graph_from_edges(
    n: int,
    edges: Sequence[Tuple[int, int, float, float]],
    *,
    positions: Any = None,
    spacing: Optional[float] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Dense symmetric signed coupling matrix from an (i, j, sign, weight) list.

    Opposite signs on the same pair partially cancel, which is the point: a pair
    with both joint evidence and occlusion evidence must not be asserted into
    either cluster.

    [U32] The K-nearest bound belongs HERE, not at graph emission. The ConceptGraph is
    the physical ground truth and keeps every edge; this constructor is where
    the O(N^2) coupling cost is paid. With `positions`, a pair must lie within
    the graph's median nearest-neighbour spacing to be a coupling candidate,
    and each node accepts at most K, where K is the median number of neighbours
    inside that radius (both derived from the positions, no magic constant).
    Ties break on (-|weight|, repr(i), repr(j)): strongest first, then the
    canonical endpoint order, so the matrix is deterministic across runs.

    Without `positions` the behaviour is byte-identical to before (all edges;
    the existing callers and tests rely on that).

    `diagnostics`, when supplied, receives the per-scene truncation record:
    coupling_edges_in / coupling_edges_kept / couplings_dropped,
    coupling_pairs_in / coupling_pairs_kept / coupling_pairs_dropped,
    coupling_spacing, coupling_k, coupling_radius, coupling_positions_missing.
    """
    kept: Sequence[Tuple[int, int, float, float]] = edges
    if positions is not None and n > 0 and len(edges) > 0:
        xy, radius, k = _spatial_neighbourhood(positions, n, spacing)
        pair_w: Dict[Tuple[int, int], float] = {}
        for i, j, _sign, weight in edges:
            if not (0 <= i < n and 0 <= j < n) or i == j:
                continue
            key = (i, j) if i < j else (j, i)
            pair_w[key] = max(pair_w.get(key, 0.0), abs(float(weight)))
        if radius > 0.0 and k > 0:
            # Candidates: measured pairs inside the radius, plus any pair with
            # an unmeasured endpoint (the bound cannot be evaluated there).
            within: List[Tuple[float, int, int]] = []
            for i, j in sorted(pair_w):
                if xy[i] is not None and xy[j] is not None:
                    d = math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1])
                    if d > radius + 1e-12:
                        continue
                within.append((pair_w[(i, j)], i, j))
            within.sort(key=lambda t: (-t[0], repr(t[1]), repr(t[2])))
            degree: Dict[int, int] = {}
            keep_pairs: set = set()
            for _w, i, j in within:
                if degree.get(i, 0) >= k or degree.get(j, 0) >= k:
                    continue
                keep_pairs.add((i, j))
                degree[i] = degree.get(i, 0) + 1
                degree[j] = degree.get(j, 0) + 1
            kept = [e for e in edges
                    if (min(e[0], e[1]), max(e[0], e[1])) in keep_pairs]
            if diagnostics is not None:
                diagnostics.update({
                    "coupling_nodes": int(n),
                    "coupling_positions_missing": int(
                        sum(1 for v in xy if v is None)),
                    "coupling_spacing": float(radius),
                    "coupling_k": int(k),
                    "coupling_radius": float(radius),
                    "coupling_pairs_in": int(len(pair_w)),
                    "coupling_pairs_kept": int(len(keep_pairs)),
                    "coupling_pairs_dropped": int(
                        len(pair_w) - len(keep_pairs)),
                    "coupling_edges_in": int(len(edges)),
                    "coupling_edges_kept": int(len(kept)),
                    "couplings_dropped": int(len(edges) - len(kept)),
                })
        elif diagnostics is not None:
            diagnostics.update({
                "coupling_nodes": int(n),
                "coupling_positions_missing": int(
                    sum(1 for v in xy if v is None)),
                "coupling_spacing": float(radius),
                "coupling_k": int(k),
                "coupling_radius": float(radius),
                "coupling_pairs_in": int(len(pair_w)),
                "coupling_pairs_kept": int(len(pair_w)),
                "coupling_pairs_dropped": 0,
                "coupling_edges_in": int(len(edges)),
                "coupling_edges_kept": int(len(edges)),
                "couplings_dropped": 0,
            })
    elif diagnostics is not None:
        diagnostics.update({
            "coupling_nodes": int(n),
            "coupling_edges_in": int(len(edges)),
            "coupling_edges_kept": int(len(edges)),
            "couplings_dropped": 0,
            "coupling_pairs_in": int(len(edges)),
            "coupling_pairs_kept": int(len(edges)),
            "coupling_pairs_dropped": 0,
        })

    w = np.zeros((n, n))
    for i, j, sign, weight in kept:
        w[i, j] += sign * weight
        w[j, i] += sign * weight
    np.fill_diagonal(w, 0.0)
    return w
