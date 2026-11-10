"""[Pass 13 / B23] Semantic slot weighting (context-conditional similarity).

Calculates per-context slot weights from between-concept vs within-concept spread
using a Fisher discriminant ratio: weight ~ between / within.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from core.registries import (  # noqa: F401
    CIRCULAR_CYCLE_LENGTHS,
    CONSTITUTIONAL_RANGES,
    SLOT_WEIGHTS_BY_CONTEXT,
    SLOT_WEIGHT_PRIORS,
)


class SlotWeightLearner:
    """Per-context slot weights from between-concept vs within-concept spread.

    A slot is worth a lot exactly when concepts of DIFFERENT identity differ on
    it while members of the SAME concept agree on it. That ratio is a Fisher
    discriminant, so it is used directly: weight ~ between / within.
    """

    def __init__(
        self,
        class_name: str,
        beta: float = 0.5,
        floor: float = 0.25,
        cap: float = 4.0,
        min_members: int = 3,
        min_concepts: int = 2,
    ):
        self.cls = class_name
        self.ranges = CONSTITUTIONAL_RANGES[class_name]
        self.cycles = CIRCULAR_CYCLE_LENGTHS.get(class_name, {})
        self.n = len(self.ranges)
        self.beta, self.floor, self.cap = beta, floor, cap
        self.min_members, self.min_concepts = min_members, min_concepts
        self._members: Dict[Tuple[Any, Any], List[Tuple[float, ...]]] = defaultdict(list)

    def _sigma(self, j: int) -> float:
        """Per-slot spread scale. Same convention as similarity's sigma."""
        r = self.ranges[j]
        return (r / 3.0) if r > 0 else 1.0

    def _diff(self, a: float, b: float, j: int) -> float:
        """Normalised difference, circular where the slot is circular."""
        d = a - b
        if j in self.cycles:
            n = self.cycles[j]
            d = (d + n / 2.0) % n - n / 2.0
        return d / self._sigma(j)

    def observe(
        self,
        context: Any,
        concept_id: Any,
        quantized: Tuple[float, ...] | List[float],
    ) -> None:
        self._members[(context, concept_id)].append(tuple(float(x) for x in quantized))

    def _mean(self, vecs: List[Tuple[float, ...] | List[float]], j: int) -> float:
        """Circular mean for circular slots, else the arithmetic mean."""
        if j in self.cycles:
            n = self.cycles[j]
            s = sum(math.sin(2.0 * math.pi * v[j] / n) for v in vecs)
            c = sum(math.cos(2.0 * math.pi * v[j] / n) for v in vecs)
            return (math.atan2(s, c) % (2.0 * math.pi)) * n / (2.0 * math.pi)
        return sum(v[j] for v in vecs) / len(vecs)

    def fit(self, context: Any) -> Optional[Tuple[float, ...]]:
        groups = {
            cid: v
            for (ctx, cid), v in sorted(
                self._members.items(),
                key=lambda kv: str(kv[0]),
            )
            if ctx == context and len(v) >= self.min_members
        }
        if len(groups) < self.min_concepts:
            return None
        prior = SLOT_WEIGHT_PRIORS.get(self.cls, (1.0,) * self.n)
        means = {
            cid: [self._mean(v, j) for j in range(self.n)]
            for cid, v in groups.items()
        }
        fisher = []
        for j in range(self.n):
            within = 0.0
            cnt = 0
            for cid, vecs in groups.items():
                for v in vecs:
                    within += self._diff(v[j], means[cid][j], j) ** 2
                    cnt += 1
            within /= max(cnt, 1)
            grand = self._mean(list(means.values()), j)
            between = sum(
                self._diff(m[j], grand, j) ** 2
                for m in means.values()
            ) / len(means)
            fisher.append(between / (within + 1e-6))

        # Normalise to mean 1, then blend with the prior: beta=1 is fully
        # learned, beta=0 is fully prior. floor/cap bound the damage a single
        # noisy slot can do.
        mf = sum(fisher) / len(fisher) + 1e-9
        s = [min(self.cap, max(self.floor, f / mf)) for f in fisher]
        raw = [
            p * ((1.0 - self.beta) + self.beta * sj)
            for p, sj in zip(prior, s)
        ]
        # Rescale so total weight matches the prior's, keeping similarity
        # calibrated against the class-level baseline.
        scale = sum(prior) / (sum(raw) + 1e-12)
        w = tuple(r * scale for r in raw)
        SLOT_WEIGHTS_BY_CONTEXT[(self.cls, context)] = w
        return w


# ---------------------------------------------------------------------------
# [O7] THE WIRING.
#
# The learner was never instantiated in the production tree, so
# SLOT_WEIGHTS_BY_CONTEXT was always EMPTY and the similarity kernel's
# `context` parameter was consulted by nobody: the learned-metric path
# existed and was unreachable.
#
# This trainer connects it.  The groups are the CONCEPTS: the positives
# are one concept, and each negative subgraph is a group of its own.  A
# slot is worth a lot exactly when concepts of DIFFERENT identity differ
# on it while the members of the SAME concept agree on it -- which is
# the Fisher ratio the learner computes, and exactly the question the
# discriminative core was built to answer.
#
# The CONTEXT is the class itself, so the kernel looks the weights up
# with the key it uses everywhere else: (class, class).
#
# Nothing changes for a class that has no learned weights: the kernel
# falls back to the class prior, and then to the uniform default.
# ---------------------------------------------------------------------------


def train_slot_weights_from_pn(
    positives: List[Any],
    negatives: List[Any],
    sig_lookup: Any,
) -> Dict[str, Tuple[float, ...]]:
    """Learn the per-class slot weights from a discriminative extraction.

    ``positives`` and ``negatives`` are subgraphs; ``sig_lookup`` maps a
    node id to (lens_type, TypedSignature).  Returns the weights that
    were learned, by class.  Deterministic: the same positives and
    negatives in the same order give the same weights.
    """
    if not positives:
        return {}
    # Collect, per class, the member vectors of every group.
    per_class: Dict[str, Dict[Any, List[Tuple[float, ...]]]] = defaultdict(
        lambda: defaultdict(list))
    groups: List[Tuple[Any, Any]] = []
    for i, sg in enumerate(positives):
        groups.append(("pos", i, sg))
    for j, sg in enumerate(negatives):
        groups.append(("neg", j, sg))
    for group_id, idx, sg in groups:
        for nid in getattr(sg, "nodes", ()):  # a Subgraph's node set
            try:
                info = sig_lookup(nid)
            except Exception:
                info = None
            if info is None:
                continue
            lt, sig = info
            cls = getattr(sig, "class_", None)
            if not cls:
                continue
            q = tuple(float(x) for x in getattr(sig, "quantized", ()))
            per_class[cls][group_id].append(q)
    learned: Dict[str, Tuple[float, ...]] = {}
    for cls, groups_by_id in sorted(per_class.items()):
        if cls not in CONSTITUTIONAL_RANGES:
            continue
        try:
            learner = SlotWeightLearner(cls)
        except KeyError:
            continue
        for group_id, vectors in sorted(groups_by_id.items(),
                                         key=lambda kv: str(kv[0])):
            if not vectors:
                continue
            for q in vectors:
                learner.observe(cls, group_id, q)
        try:
            w = learner.fit(cls)
        except Exception:
            w = None
        if w is not None:
            learned[cls] = w
    return learned
