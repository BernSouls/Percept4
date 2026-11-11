"""matching/signature_matrix.py -- the node-similarity matrix of a lens type,
computed in blocks.

``signature_similarity_matrix(sigs_a, sigs_b)`` returns

    [[_sig_similarity(a, b) for b in sigs_b] for a in sigs_a]

BIT FOR BIT, and several times faster.  It is a performance change only: no similarity, no weight, no
correspondence and no score may differ from the per-pair loop by a single
bit, and ``tests_matcher/test_signature_matrix.py`` pins exactly that.

How.  ``TypedSignature.similarity`` of two CANONICAL signatures of the same
class and relation that both carry a population of the declared layout is
``core.population.population_similarity`` (the first branch of both
similarity implementations the tree ships).  That function is a short chain
of IEEE-754 double operations; here the SAME operations are applied in the
SAME order to whole arrays:

    mean            the builtin ``sum`` over the stored tuple, divided by n
                    (called per vector, so the block follows the scalar on
                    any interpreter, whatever its ``sum`` does)
    centred values  x - mean, elementwise
    num, va, vb     accumulated bin by bin, starting from 0.0
    cosine          num / sqrt(va * vb), clamped to [-1, 1]
    axis overlap    the positive part; 0.0 when either variance <= 1e-18
    result          sum over the axes of weight * overlap, divided by the
                    sum of the weights

No BLAS product, no ``numpy.sum``, no fused operation: elementwise numpy
arithmetic on float64 is the same IEEE operation as Python's.

The second block is the BINS path of the Gaussian kernel
(``core.types._similarity_gaussian`` below its population branches): two
CANONICAL signatures of the same class, relation and arity, neither of which
can reach a population branch, no slot ABSENT:

    d               abs(a - b); min(d, n - d) on a circular slot
    term            (w / W_sum) * (d / sigma) ** 2, added slot by slot
    result          exp(-total); exactly 1.0 when the two tuples are equal

Subtraction, absolute value, minimum, division, multiplication and addition
are elementwise numpy.  ``** 2`` and ``exp`` are NOT: numpy's square and
exponential are not guaranteed to round as the C library's ``pow`` and
``exp`` that Python calls, so both are applied element by element with the
interpreter's own ``pow`` and ``math.exp``.  The per-slot constants (sigma,
w / W_sum, the cycle length) are computed once by the scalar's expressions.

Every pair that is on neither path -- another signature kind, a missing or
mis-shaped population, an ABSENT slot, a class with its own slot rule
(``core.slots``), a class without ranges, an unknown similarity
implementation -- is evaluated by the per-pair function itself.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: no intermediate of the kernel can overflow below this magnitude
_MAX_ABS = 1e50

#: Below this many pairs a block costs MORE than the per-pair loop (the fixed
#: cost of the numpy calls): a small block is left to the per-pair function.
#: Both paths return the same floats, so these two numbers are a matter of
#: speed only; the measured break-even is about 36 pairs for the population
#: block and about 12 for the Gaussian one.
_MIN_PAIRS_POPULATION = 48
_MIN_PAIRS_GAUSSIAN = 16


def population_similarity_block(pops_a: Sequence[Any], pops_b: Sequence[Any],
                                axes: Sequence[Any]):
    """``population_similarity(pops_a[i], pops_b[j], axes)`` for every (i, j),
    as a float64 array, bit for bit.

    The caller guarantees what ``population_similarity`` would otherwise
    check: one vector per axis on every population, ``axis.n_bins`` finite
    values in each.
    """
    import numpy as np

    na, nb = len(pops_a), len(pops_b)
    total = np.zeros((na, nb), dtype=np.float64)
    wsum = 0.0
    tmp = np.empty((na, nb), dtype=np.float64)
    num = np.empty((na, nb), dtype=np.float64)
    cos = np.empty((na, nb), dtype=np.float64)
    for k, axis in enumerate(axes):
        n = axis.n_bins
        da = np.array([p[k] for p in pops_a], dtype=np.float64).reshape(na, n)
        db = np.array([p[k] for p in pops_b], dtype=np.float64).reshape(nb, n)
        # ma = sum(a) / n -- the builtin on the stored tuple, as the scalar
        da -= np.array([sum(p[k]) / n for p in pops_a],
                       dtype=np.float64).reshape(na, 1)
        db -= np.array([sum(p[k]) / n for p in pops_b],
                       dtype=np.float64).reshape(nb, 1)
        va = np.zeros(na, dtype=np.float64)
        vb = np.zeros(nb, dtype=np.float64)
        num.fill(0.0)
        for t in range(n):
            ca = da[:, t]
            cb = db[:, t]
            va += ca * ca                      # va += dx * dx
            vb += cb * cb                      # vb += dy * dy
            np.multiply(ca[:, None], cb[None, :], out=tmp)
            np.add(num, tmp, out=num)          # num += dx * dy
        ok = (va > 1e-18)[:, None] & (vb > 1e-18)[None, :]
        np.multiply(va[:, None], vb[None, :], out=tmp)
        np.sqrt(tmp, out=tmp)                  # math.sqrt(va * vb)
        cos.fill(0.0)
        np.divide(num, tmp, out=cos, where=ok)
        np.minimum(cos, 1.0, out=cos)          # max(-1.0, min(1.0, .))
        np.maximum(cos, -1.0, out=cos)
        cos[~(cos > 0.0)] = 0.0                # c if c > 0.0 else 0.0
        np.multiply(cos, axis.weight, out=cos)
        np.add(total, cos, out=total)          # total += weight * overlap
        wsum += axis.weight
    if wsum <= 0.0:
        total.fill(0.0)
        return total
    np.divide(total, wsum, out=total)
    return total


def gaussian_bins_block(qa: Sequence[Any], qb: Sequence[Any],
                        plan: Sequence[Tuple[float, float, Optional[float]]]):
    """The bins path of ``_similarity_gaussian`` for every (i, j) of two lists
    of quantized tuples of one class, relation and arity, as a float64
    array, bit for bit.

    `plan` is ``_gaussian_plan`` of the class: one (sigma, w / W_sum, cycle
    length or None) per slot.  The caller guarantees what the scalar would
    otherwise check: plain finite floats, no ABSENT slot, the declared
    arity, no population branch reachable.
    """
    import math
    from itertools import repeat

    import numpy as np

    na, nb, k = len(qa), len(qb), len(plan)
    A = np.array(qa, dtype=np.float64).reshape(na, k)
    B = np.array(qb, dtype=np.float64).reshape(nb, k)
    total = np.zeros((na, nb), dtype=np.float64)
    same = np.ones((na, nb), dtype=bool)
    d = np.empty((na, nb), dtype=np.float64)
    for i, (sigma, coef, cycle) in enumerate(plan):
        a = A[:, i].reshape(na, 1)
        b = B[:, i].reshape(1, nb)
        same &= (a == b)                       # self.quantized == other.quantized
        np.subtract(a, b, out=d)
        np.abs(d, out=d)                       # d = abs(a - b)
        if cycle is not None:
            np.minimum(d, cycle - d, out=d)    # d = min(d, n - d)
        np.divide(d, sigma, out=d)             # d / sigma
        # (d / sigma) ** 2 -- the interpreter's own pow, element by element
        sq = np.array(list(map(pow, d.ravel().tolist(), repeat(2))),
                      dtype=np.float64).reshape(na, nb)
        np.multiply(sq, coef, out=sq)          # (w / W_sum) * (...)
        np.add(total, sq, out=total)           # total_sq += ...
    np.negative(total, out=total)
    # math.exp(-total_sq) -- the interpreter's own exp, element by element
    out = np.array(list(map(math.exp, total.ravel().tolist())),
                   dtype=np.float64).reshape(na, nb)
    out[same] = 1.0                            # equal tuples return 1.0 first
    return out


def _gaussian_plan(class_name: str, arity: int):
    """The per-slot constants of the bins path of ``_similarity_gaussian``
    for a class -- [(sigma, w / W_sum, cycle length or None), ...] -- each
    computed by the scalar's own expression; or None when the class is left
    to the per-pair function: no ranges or another arity, a class-specific
    slot rule, a weight lookup that raises, a constant outside the range in
    which no intermediate can overflow."""
    import core.types as T
    from core.registries import (CIRCULAR_CYCLE_LENGTHS,
                                 get_similarity_ranges, get_slot_weights)
    from core.slots import _CLASS_SLOT_RULES

    if class_name in _CLASS_SLOT_RULES:
        return None
    try:
        ranges = get_similarity_ranges(class_name)
        if ranges is None or len(ranges) != arity:
            return None
        cycle_lengths = CIRCULAR_CYCLE_LENGTHS.get(class_name)
        weights = get_slot_weights(class_name, len(ranges), context=None)
        W_sum = float(sum(weights)) if weights else float(arity)
        if W_sum <= 0.0:
            W_sum = float(arity) or 1.0
        plan = []
        for i in range(arity):
            if ranges[i] > 0.0:
                sigma = ranges[i] / 3.0
            else:
                sigma = T._SIGMA_FALLBACK_RANGE0
            w = weights[i] if (weights and i < len(weights)) else 1.0
            sigma = float(sigma)
            coef = float(w / W_sum)
            cycle = None
            if cycle_lengths and i in cycle_lengths:
                cycle = float(cycle_lengths[i])
                if not (-_MAX_ABS <= cycle <= _MAX_ABS):
                    return None
            # NaN fails every comparison below
            if not (1.0 / _MAX_ABS <= sigma <= _MAX_ABS):
                return None
            if not (0.0 <= coef <= _MAX_ABS):
                return None
            plan.append((sigma, coef, cycle))
        return plan
    except Exception:
        return None


def _block_key(sig: Any, axes_of: Callable[[str], Any],
               populations: bool = True):
    """The block `sig` belongs to, or None when it is left to the per-pair
    function.

    ("P", class, relation): on the strict population branch for every
    partner with the same key.
    ("G", class, relation, arity): on the bins path for every partner --
    it carries no population, or its class declares no layout, so neither
    population branch of the similarity can be taken whatever the partner
    carries -- with plain finite floats and no ABSENT slot.

    `populations=False`: the matrix is too small for a population block to
    pay, so a signature that carries one is not examined at all.
    """
    import core.types as T

    if type(sig) is not T.TypedSignature or sig.kind != "CANONICAL":
        return None
    axes = axes_of(sig.class_)
    pop = sig.population
    if pop and axes is not None:
        if not populations or len(pop) != len(axes):
            return None
        for v, ax in zip(pop, axes):
            if len(v) != ax.n_bins:
                return None
            for x in v:
                # finite and far from overflow; NaN fails both comparisons
                if not (-_MAX_ABS <= x <= _MAX_ABS):
                    return None
        return ("P", sig.class_, sig.relation)
    q = sig.quantized
    if not q:
        return None
    absent = T.ABSENT
    for x in q:
        if (type(x) is not float or x == absent
                or not (-_MAX_ABS <= x <= _MAX_ABS)):
            return None
    return ("G", sig.class_, sig.relation, len(q))


def signature_similarity_matrix(
    sigs_a: Sequence[Any],
    sigs_b: Sequence[Any],
    scalar: Callable[[Any, Any], float],
) -> List[List[float]]:
    """``[[scalar(a, b) for b in sigs_b] for a in sigs_a]`` with the
    population branch and the Gaussian bins path evaluated in blocks.
    `scalar` is the per-pair function (``memory.cores._sig_similarity``); it
    decides every pair the blocks do not."""
    na, nb = len(sigs_a), len(sigs_b)
    if na == 0 or nb == 0:
        return [[] for _ in range(na)]
    if na * nb < min(_MIN_PAIRS_POPULATION, _MIN_PAIRS_GAUSSIAN):
        # too small for any block to pay for itself
        return [[scalar(a, b) for b in sigs_b] for a in sigs_a]

    import parameters as P
    import core.types as T
    from core.registries import get_population_axes

    blocks_ok = (
        bool(getattr(P, "POPULATION_CODING_ENABLED", False))
        and T._SIMILARITY_IMPL in (T._similarity_gaussian,
                                   T._similarity_linear)
    )
    if not blocks_ok:
        return [[scalar(a, b) for b in sigs_b] for a in sigs_a]
    gaussian = T._SIMILARITY_IMPL is T._similarity_gaussian
    populations = na * nb >= _MIN_PAIRS_POPULATION

    plans: Dict[Tuple[str, int], Any] = {}

    def _key(sig):
        k = _block_key(sig, get_population_axes, populations)
        if k is not None and k[0] == "G":
            if not gaussian:
                return None
            pk = (k[1], k[3])
            if pk not in plans:
                plans[pk] = _gaussian_plan(k[1], k[3])
            if plans[pk] is None:
                return None
        return k

    keys_a = [_key(s) for s in sigs_a]
    keys_b = [_key(s) for s in sigs_b]
    groups_a: Dict[Any, List[int]] = {}
    groups_b: Dict[Any, List[int]] = {}
    for i, k in enumerate(keys_a):
        if k is not None:
            groups_a.setdefault(k, []).append(i)
    for j, k in enumerate(keys_b):
        if k is not None:
            groups_b.setdefault(k, []).append(j)

    def _pays(key, n_pairs):
        return n_pairs >= (_MIN_PAIRS_POPULATION if key[0] == "P"
                           else _MIN_PAIRS_GAUSSIAN)

    def _block(key, ia, jb):
        if key[0] == "P":
            return population_similarity_block(
                [sigs_a[i].population for i in ia],
                [sigs_b[j].population for j in jb],
                get_population_axes(key[1])).tolist()
        return gaussian_bins_block(
            [sigs_a[i].quantized for i in ia],
            [sigs_b[j].quantized for j in jb],
            plans[(key[1], key[3])]).tolist()

    # the usual case: one block on both sides, every node in it -- the
    # block IS the matrix
    if (len(groups_a) == 1 and len(groups_b) == 1
            and next(iter(groups_a)) == next(iter(groups_b))
            and len(next(iter(groups_a.values()))) == na
            and len(next(iter(groups_b.values()))) == nb
            and _pays(next(iter(groups_a)), na * nb)):
        return _block(next(iter(groups_a)), range(na), range(nb))

    out: List[List[Optional[float]]] = [[None] * nb for _ in range(na)]
    for key, ia in groups_a.items():
        jb = groups_b.get(key)
        if not jb or not _pays(key, len(ia) * len(jb)):
            continue                      # left to the per-pair function
        block = _block(key, ia, jb)
        for r, i in enumerate(ia):
            row = out[i]
            brow = block[r]
            for c, j in enumerate(jb):
                row[j] = brow[c]
    for i in range(na):
        row = out[i]
        a = sigs_a[i]
        ka = keys_a[i]
        for j in range(nb):
            if row[j] is None:
                kb = keys_b[j]
                if ka is not None and kb is not None and ka != kb and (
                        ka[1] != kb[1] or ka[2] != kb[2]
                        or (ka[0] == "G" and kb[0] == "G")):
                    # two CANONICAL signatures whose class or relation
                    # differs: 0.0 at the per-pair function's first test.
                    # Two bins-path signatures of one class and relation
                    # and different arity: 0.0 at its length test.
                    # (Equal keys reach this loop only when their block
                    # was too small to pay: the per-pair function below.)
                    row[j] = 0.0
                else:
                    row[j] = scalar(a, sigs_b[j])
    return out  # type: ignore[return-value]
