# STAGE 7 — DELIVERABLE

One table per phase: what was measured, before and after, the mechanism the
change was meant to move, and whether the falsifier fired.  Every falsifier and
every constant was written into `phase6_preregistration.md` (7.29–7.43) before
the run that produced the number beside it.

Nothing in this stage flips a pipeline default except in two cases where a rule
fixed in advance selected the new value, and both are named: the seed rule
(7.31, Phase 2) and the relation denominator (7.39, Phase 4b).  The walker stays
behind `MATCH_RELATION_WALKER`, still OFF.

---

## PHASE 0 — THE GATE: does any relation carry class information?

Read-only.  79 EVAL graphs, 91,283 edges, class-6 relations ON.  Bar registered
in 7.29: RATIO > 1.25, I ≥ 0.01 bits, I/H ≥ 0.02, RATIO ≥ p95 of 200 label
permutations.

| | slots tested | passed the bar | best I | median I |
|---|---|---|---|---|
| structural relation slots | 63 | **5** | 0.1274 bits (LINE_TERMINATION/curvature); best overall 0.2689 (CONTACT/normal_angle) | 0.0200 |
| **BOUNDARY_ATTACHMENT** (3 slots) | 3 | **0** | 0.0207 | 0.0062 |
| **BOUNDARY_SUCCESSOR** (2 slots) | 2 | **0** | 0.0191 | 0.0111 |
| **ILLUSORY_INDUCER** (2 slots) | 2 | **0** | — | — |
| node signatures (control) | 210 | **30** | 0.5218 | 0.0120 |

The five that pass: JUNCTION_CONTOUR/topology_class, PARALLEL/perp_dist,
RADIALITY/hub_bearing, CONCENTRICITY/ecc_difference, LINE_TERMINATION/curvature.

**FALSIFIER: DID NOT FIRE** — the gate passes.  But it passes on five slots
that are none of the three the review named, and the effect sizes are small:
eta² beside every large RATIO is 0.002–0.043, and by mutual information the
relations are an order of magnitude poorer than the node signatures they would
have to improve on (6 of 63 slots ≥ 0.10 bits against 22 of 210).

**Mechanism it was meant to move:** whether matcher work is even possible.
**Verdict: possible, narrowly, and not through the relations the walker grows on.**

---

## PHASE 1 — THE FRONT-END TANGENT

| | before | after | target |
|---|---|---|---|
| tracer's polyline direction vs truth | 90.0° median | **0.0° median** (p90 18.1°, 87 % within 15°) | 15° |
| fragment tangent (`raw_orientation`) | — | **2.0° median**, 91 % within 15° | ≤ 15° |
| fragment chord | — | **2.0° median**, 90 % within 15° | ≤ 15° |
| AMBIGUOUS attachments | 514 / 6301 = **0.082** | 569 / 6739 = **0.084** | must fall |
| decided by TANGENT | 105 = 0.017 | 105 = 0.016 | must rise |
| ContourFragment's tangent used in a tie-break | 0 | **0** | must fire |

**FALSIFIER: FIRED.**  The tangent is now correct at the source and the
ambiguity rate did not move (0.082 → 0.084, and the tangent-decided share went
down, not up).  By 7.30 that says the ambiguity is NOT direction-limited, and
ContourFragment stays **ABSENT** — its tangent is not fabricated into a
decision.

**Mechanism it was meant to move:** the ~30 % of attachments that were
ambiguous.  **It did not move them: the fix is in the geometry and the
ambiguity is somewhere else (host choice, not feature direction).**

---

## PHASE 2 — SEEDS

Both rules, all nine registered pairs, against renderer truth.  Recall is
against the true correspondent pairs in the smaller frame, so a rule cannot win
by seeding fewer.

| rule | seeds | right | wrong | **P(seed)** | R(seed) | mapped | P | R | **role F1** |
|---|---|---|---|---|---|---|---|---|---|
| topological (D1, the old default) | 7.7 | 3.0 | 4.6 | **0.421** | 0.024 | 62.3 | 0.578 | 0.303 | **0.383** |
| measured (the rule D1 replaced) | 5.7 | 2.8 | 2.8 | **0.639** | 0.024 | 68.4 | 0.625 | 0.353 | **0.444** |

Measured is more precise on 8 of 9 pairs, less on 0, equal on 1.

**DEFAULT APPLIED: `SEED_TOPOLOGICAL = False`** by the rule registered in 7.31
before the run.  The topological rule stays behind the flag, unchanged.

**FALSIFIER: none registered for this phase** (7.31 registered a default rule
instead, and it fired).  Both rules are far below the 50 % precision the review
made the condition for a third arm: measured is 0.639, so the distinctiveness
rank registered in 7.31 as "the next arm" is **not** built.

**Mechanism it was meant to move:** seed precision (diagnosis: 15 % right).
**From 0.421 to 0.639, on 5.7 seeds.  Still 2.8 wrong seeds per pair against a
median 124 true correspondent pairs — the recall problem is untouched.**

---

## PHASE 3 — ADMISSION BY STRENGTH

Arm A: admit only within `ADMIT_MARGIN` = 1.0 bin of the node's best rival.
Arm B: Arm A plus `ADMIT_FLOOR` = 0.5.  Both registered in 7.32.

| arm | mapped | correct | **fan-out** | P | R | role F1 | budget spent |
|---|---|---|---|---|---|---|---|
| shipped | 68.4 | 46.7 | **89.0** | 0.625 | 0.353 | **0.444** | 2.39 |
| A (relative) | 64.1 | 43.4 | **106.4** | 0.588 | 0.327 | **0.411** | 2.34 |
| B (relative + floor) | 64.1 | 43.4 | **106.2** | 0.588 | 0.327 | **0.411** | 2.33 |

**FALSIFIER: FIRED.**  Fan-out fell on 3 of 9 pairs and rose on the mean
(89.0 → 106.4); recall rose on 1 pair and fell on 3.  The registered mechanism
— fan-out falls AND recall rises together — does not hold.  Neither arm is
defaulted; both stay OFF.

A detail the review's diagnosis did not have: fan-out and recall are
**inversely** related across pairs (best recall 0.750 at 58.6 pops; worst 0.051
at 249.8).  The recall is not being eaten by rivals; it is thin because the
seeds are thin.

**Mechanism it was meant to move:** recall (0.11 in the diagnosis).
**0.353 → 0.353 → 0.327: it moved nothing.**  Growth by relation is not the
recall lever for this representation, and no margin was tuned to make it one.

---

## PHASE 4 — THE OBJECTIVE

(a) `score_for_class = Q · |M| / min(frame_q, frame_r)`, reported, not selected on.

| mapping | mapped (median) | raw Q (median) | score_for_class (median) |
|---|---|---|---|
| renderer truth | 75 | 0.226 | **0.105** |
| old matcher | 169 | 0.417 | **0.403** |
| walker | 58 | 0.248 | **0.085** |

truth > walker: **raw Q 5 of 9 → score_for_class 9 of 9**.  truth > old matcher:
**0 of 9 under both.**  A clamp bug found by the new pin (a mapping could read
2.5 on a 4-node frame) was fixed in the function, not in the pin.

(b) The coverage loophole.  7.33's first test (halve the mapping) could not
decide — a halved mapping CUTS relations, so both forms fell on 27 of 27.  The
corrected test was registered in 7.39 before its run: compare a **closed**
region (largest connected component) with the whole mapping.

| form | proper-subset cases | region BELOW whole | equal | ABOVE | verdict |
|---|---|---|---|---|---|
| union (7.14, in force) | 22 | 14 | 0 | **8** | **FAILS** |
| frame (query-side mass) | 22 | **22** | 0 | 0 | **PASSES** |

**DEFAULT APPLIED, by the rule registered in 7.39: R is the query's own relation
mass over its whole frame.**  `R_union` is still computed for every earlier
result's baseline.  Pin verified red on the union form.

**FALSIFIER: none registered** (7.33/7.39 registered a selection rule instead,
and it selected exactly one form).

**Mechanism it was meant to move:** whether Q can arbitrate across node sets.
**It could not (5 of 9, a coin toss); paid for coverage it can (9 of 9).**  It
does not rescue the walker against the old matcher, and it moves nothing on
M3-6 — R decides how a mapping is SCORED, and M3-6 scores by role.

---

## PHASE 5 — THE SHIPPED MATCHER'S ABSENCE BIAS

Criterion fixed in 7.34 before the run: `B = max_r |p_r − e_r| < 0.10` **and**
rank(p) = rank(e).

| window | form | acc | B | Spearman ρ | criterion |
|---|---|---|---|---|---|
| full | core (default) | 0.453 | 0.376 | **−1.00** | fail |
| full | symmetric (union) | 0.453 | 0.283 | −0.20 | fail |
| v1 | core (default) | 0.480 | 0.385 | **−0.80** | fail |
| v1 | symmetric (union) | 0.520 | 0.264 | −0.40 | fail |

Per-reference predictions (full window, of 75): apple — 16 of 75 queries, a
third of the edges, predicted **0** times; car 18; cup 27; horse 30.

**The bias is exactly a size effect:** the prediction distribution is the
INVERSE of reference size on both windows.  The symmetric form halves it and
does not cross the bar.

**DEFAULT: NOT CHANGED.**  7.34 required the criterion to pass; it does not.
`EVIDENCE_QUERY_SIDE_TERMS` stays `()`.  A real, direction-correct improvement
that misses a bar fixed beforehand does not become a default.

**Mechanism it was meant to move:** whether a reference's size decides how often
it wins.  **It does (ρ = −1).  The symmetric form is half a fix; recorded OPEN.**

---

## PHASE 6 — PINS THAT REACH THE REAL FRONT END

`tests_matcher/test_frontend_relations.py`, six pins, each building a graph
through `diagnostics/eth80_frontend.build_graph` — real perception stack, real
lens tier, real binder, real writers.

| pin | asserts | verified |
|---|---|---|
| all three relations written | BOUNDARY_ATTACHMENT 190, BOUNDARY_SUCCESSOR 26, ILLUSORY_INDUCER 4 | **RED** when the `_run_boundary_successor()` call site is deleted |
| switch off writes none | 0 of each | — |
| slots vary | > 1 distinct parameter value per relation | — |
| FRAME partition | every emitted lens type on exactly one side, both sides non-empty | — |
| ambiguity | refusals counted (7 here); every written attachment measured | — |
| cache key | moves with a graph-side switch, and the graphs differ | — |

One pin had to be corrected: it asserted "one host per feature kind", which the
real scene refutes correctly — 6 of 128 features attach to several NAMED hosts
(Amendment 3.1).  The pin now fixes the two invariants that exist.

**FALSIFIER: none (a construction phase).**  **DONE as specified: a pin that
fails if any of the three relations stops being written.**

---

## PHASE 7 — CLEANUP

| item | before | after |
|---|---|---|
| switches read nowhere | 4 | 4, marked DEAD in place (kept: export checkout) |
| getattr fallbacks disagreeing with their declaration | **23** | **0** |
| names read with a fallback but declared nowhere | **5** | **0** |
| stale "budget shared by the beam" docstring | present | corrected |
| dead code from the ablations | none found | — |
| pins for the above | 0 | 2 |

None of this changed behaviour: every fallback was aligned to the DECLARATION,
which was always the value in force.

**It did change a number, and here it is.**  M3-6 on the cleaned tree reads
mean F1(new) − F1(old) = **−0.219**, against the −0.274 of record.  Measured
three ways: topological seeds −0.281, measured seeds −0.219, measured seeds
with R back to the union form −0.219.  So the 0.062 is Phase 2's seed default,
and the Phase-4 R change moves nothing here.  The pin stays RED.

---

## THE VERDICT

**M3-6, on the walker as it now stands: RED.**  Mean F1 difference −0.219, new
better on **0 of 9**; precision 0.625 vs 0.641, recall 0.353 vs 0.690.  Every
stage-6 default stays OFF, `MATCH_RELATION_WALKER` included.  The final
measurement does not run.

### (a) Does the representation carry class information?

**Barely, and not through the relations the matcher grew on.**

- The gate passes on 5 of 63 structural slots.  **None** of the seven slots of
  the three relations the walker grows on (BOUNDARY_ATTACHMENT,
  BOUNDARY_SUCCESSOR, ILLUSORY_INDUCER) reaches the bar.
- The node signatures carry an order of magnitude more: 30 of 210 slots pass,
  best 0.52 bits against 0.27 for the best relation slot.
- Phase 1 confirms the geometry is not the obstacle: the tangent is now correct
  at the source (0.0° median) and the ambiguity rate did not move.

So the answer to (a) is **no, not enough to carry a matcher**: what the
relations encode is mostly geometry-invariant detail (rapidly varying
parameters with narrow within-class spread — eta² 0.002–0.043 wherever the
ratio is large), and their class information per bit is a tenth of the node
signatures'.

### (b) Do seeds + admission strength close the gap to the old matcher's pointwise P 0.64 / R 0.63?

**No.**

| | P | R |
|---|---|---|
| old matcher (the bar) | 0.641 | 0.690 |
| walker, before Phase 2/3 | 0.578 | 0.303 |
| walker, after Phase 2 (seeds) | 0.625 | 0.353 |
| walker, after Phase 3 arm A | 0.588 | 0.327 |

Precision came to within 0.016 of the bar.  **Recall stayed at half of it** and
admission strength made it worse, not better.  The gap that remains is not a
tuning gap: the walker maps 68 nodes where the old matcher maps 169, and its
relations do not distinguish the classes well enough to grow further.

### What this means for the work

The front end is the work.  Phases 0 and 1 say it in two numbers: the relations
that could carry a correspondence carry a tenth of the class information the
node signatures carry, and the one geometric defect that looked like it was
blocking 30 % of attachments was real, is fixed, and changed the ambiguity rate
by +0.002 (the wrong way).  No matcher change was retained that a pre-registered
rule did not select, and the four defects that were fixed (the tangent, the
seed rule, the objective's denominator, the cache key) each moved the number
their rule predicted and no others.
