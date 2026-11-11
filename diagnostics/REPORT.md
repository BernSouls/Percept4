# Gap-closure report — matcher edge evidence on ETH-80

Running report. Newest stage first; the pre-registrations follow the stages,
then the port manifest (cumulative), the pin triage and, last, the bug hunt
and the design changes in one list. Every figure is **EVAL-75, in-sample**, on the
reconstructed settle-route harness (see `README.md` in this directory for
what that harness is and is not). n = 75: one binomial standard error is
about 4.3 queries, so a difference of a few queries is noise. Comparisons are
paired on the same 75 queries: `+gained/-lost`, exact two-sided McNemar p.
Pairs of numbers are `full / v1` window.

## Where things stand

| item | state | default in the tree |
|---|---|---|
| Phase 0 — baseline, audits A, B, C | done, frozen in `results/phase0/` | – |
| Phase 1 — F6, parameter-less edges match as structure | landed, measured | on |
| D1 — similarity-weighted node credit | landed, measured; success signal **not met** (stage 1 §4; control in stage 4 §4) | on |
| D3 — query-side terms | measured; **fails its rule on full** | off |
| Phase 4 — an unmeasured edge pair is not evidence | landed by the spec; **unexercised** on this corpus | on |
| null calibration of node coverage | measured; negative | off |
| mixture reversal (correctness) | landed; the prediction held to the last figure | on |
| exact assignment (correctness) | landed; every prediction held | on |
| numpy kernels (performance) | landed; no value moves; a call is × 8.2 \| × 4.0 faster | on |
| P4b — the count kept apart from the physics | measured; **fails its rule on full**, by one comparison of 225 | off |
| P4b — a lookup key is not a measured axis | pinned; cannot be measured in this harness | off; your decision |
| D2 prerequisites — the dark alignment refines | landed on the dark path; mechanism checked, no score | budget fix on, anchor prior off |
| the documented specificity cap, wired | dark; measured descriptively | off; your decision |
| D2 — edge-aware correspondence | **gated** on the graded-admission experiment; not started | – |
| Phase 3 — population wiring | gate open; **not started** | – |
| HELDOUT-75 | **untouched** | – |

Hits on the declared defaults: **37 | 39** of 75 (Phase 0 baseline 36 | 34;
chance 18.75; one binomial standard error 4.3). Across four stages no
change moved accuracy by an amount distinguishable from chance, and the
three changes whose rule asked for a measurable improvement in what the
score reads (D3, the null calibration, P4b) are off. What is on by default
is a correctness fix, the spec's wording, or a performance change that
moves no value.

Bars on the final tree (`results/logs/`): matcher pins **390 passed**; pin
triage **0 unclassified** over 17 settings; bug hunt sections A–F **0
findings**; figground **99 passed, 4 xfailed** (the same four as before any
change); `_u27_selftest.py` all pass (run on this branch; `main` deleted
the script after the branch point, so it is no longer a bar there, and the
stored records' source digest will not match a tree without it — see
`README.md`, "The stored records and `main`"). Not checkable here: the 329-test
figground bar (103 tests exist in this checkout), U20 WRONG=0, `_u24_*`,
the per-class L0/L1 thresholds. The figground bar calls the matcher 0
times, so it says nothing about a matcher change (stage 2 §4).

---

## Assessment — what I think the code base needs (opinion; each point names the measurement it rests on)

1. **The instruments belong in the repository.** The export has none of
   the `_u32_*` harnesses the README's figures come from, 103 of 329
   figground tests, three top-level modules missing, no dependency pins,
   and — before this work — no test that calls the matcher. Every figure
   here is therefore from a reconstruction with one reference view per
   class; the README's 68 % cannot be re-run. Nothing below can be settled
   until the real harness, the matcher pins and a requirements file are
   checked in.
2. **The correspondence is the problem, not the score formula.** It is
   chosen from node similarity and position alone; the relations are
   checked afterwards on whatever came out. The result: four relations in
   five have no counterpart, own class or not (presence 0.197 against
   0.189 on full, stage 4 §3); the node term under an exact solver is the
   node count (r = 1.000 under binary credit, stage 4 §4); and where the
   optimum is not unique the choice is an accident of node order (7 of 32
   full-window pairs, stage 4 §5). Four score-side changes — D1, D3, the
   null calibration, P4b — were each measured and none moved accuracy
   beyond chance. What is left to try is what the gate already names:
   admit pairs at a price, and let the relations vote on who is paired
   (D2). Its prerequisites are now in place.
3. **n = 75 cannot decide anything.** One standard error is 4.3 queries
   and the largest change seen in four stages is 5. The kernels make a
   larger protocol affordable (0.074 s per full-window call): all views of
   the held-in objects, several reference views per class, the rule fixed
   beforehand. Decide on the terms' discrimination, not on hits.
4. **Find out what the higher-tier lenses contribute.** The full window
   is not ahead of v1 (37 against 39 hits) at twice the cost per call, and
   adding its nodes takes the relational term's discrimination from 0.72 to
   0.54 (stage 4 §3). A per-lens-type ablation of the two terms would say
   which lens types carry class information and which only add counts.
5. **Fix the writers, not the reader.** A membership flag written as a
   literal 1.0 into a graded slot (16,528 edges), `CURVE_CONTAINMENT` with
   two measured slots that never vary, `PEAK_CHAIN` writing three axes
   against a four-axis layout, two slot tables that disagree on four
   slots. P4b and the lookup-key switch are reader-side workarounds for the
   first and the last.
6. **Make comments fail when the code disagrees.** Five times a comment
   or a parameter description stated a behaviour the code did not have (F6
   existed as a comment; all-UNKNOWN "is 0" returned 0.2; the specificity
   cap is unreachable; `MATCH_NULL_CALIBRATED` did not calibrate; the
   alignment budget "for the refinement" was spent on the seed). Each is
   now a pin. The same treatment for the rest of `parameters/_text.py` —
   one pin per switch, both values — would retire the dead ones.
7. **Next performance step: prepare a graph once.** What is left in a
   call is per-call bookkeeping on data that does not change between calls
   (canonical edge keys, the edge-weight table, normalised anchors, block
   keys). A reference is matched against every query; preparing it once is
   the next factor of two or three and needs no arithmetic to change.

What I would not do: flip D3, the null calibration, P4b or the cap on the
strength of ±3 queries; add a term to the score; or tune a weight on
EVAL-75.

---

## Stage 4 — performance, P4b, the dark alignment's prerequisites, the specificity cap: results against the pre-registered rules

EVAL-75, in-sample; "a | b" is full | v1. Tables:
`results/phase3_blocks_p4b.txt`, `results/phase3_speed.txt`,
`results/phase3_d2_mechanism.txt`, `results/phase2_bug_hunt.txt` (section F),
`results/logs/stage4_vs_before_blocks.txt`. The whole measurement, in order:
`results/logs/measurement_chain_stage4.sh` (and `…_rerun.sh`, see §7). The
scripts are named `phase3_*` for the third round of measurement; they are
not the spec's Phase 3 (population wiring), which is not started.

### 0. Verdicts

| change | rule (fixed beforehand) | full | v1 | default |
|---|---|---|---|---|
| numpy kernels, `MATCH_SIGNATURE_MATRIX_BLOCKS` | no value differs from the Python loops: every matrix entry on EVAL, the pins, every stored record | 0 of 14,989,430 entries; 0 of 10,500 record values | 0 of 5,854,028; 0 of 10,500 | stays **on** |
| P4b, `EVIDENCE_RELATION_COUNT_APART` | on **both** windows: pins green; the term rises on no pair; the discrimination of the relational term does not fall; no paired fall with p < 0.05 | rises on 0; discrimination 0.5378 → 0.5333, **falls** (one comparison of 225); 37 → 37 | rises on 0; 0.7200 → 0.7244; 39 → 38, +0/−1, p = 1.00 | stays **`False`** — rule fails on full |
| P4b, `EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS` | none: it moves `mcs_score` / `mcs_similarity` only, which this harness does not read | – | – | stays `False`; your decision |
| D2 prerequisites, `MATCH_ALIGN_BUDGET_REFINEMENT_ONLY`, `MATCH_GLOBAL_ANCHOR_PRIOR` | pins green; no stored value moves; mechanism only, no score | as built: 0 refinement sweeps on 32 of 32 pairs; now 3.0 sweeps, budget never exhausted | as built: 2.0 sweeps, exhausted on 27 of 32; now 3.0, never | `True` / `False`, on the dark path; D2 stays gated |
| specificity cap, `EDGE_SPECIFICITY_CAP_SAME_SCHEMA` | none: descriptive, the default was fixed at `False` beforehand | 37 → 36; under D3 37 → 34 | 39 → 36; under D3 40 → 43 | stays `False`; your decision |

No accuracy figure in this stage is distinguishable from chance, and none
was predicted to be. Continuity: every stage-2 and stage-3 configuration
re-measured with the kernels reproduces the record stored before them — 32
records, 307,800 stored values, 0 differ. Under `PYTHONHASHSEED=1` five
configurations (the two P4b ones included) reproduce 112,200 of 112,200
values. The front end builds the same 79 graphs as before the stage (content
digests identical, deterministic across hash seeds). Bars: matcher pins
**390 passed**; figground **99 passed, 4 xfailed**; `_u27_selftest.py` all
pass.

### 1. The kernels return the loops' floats

Three kernels sit behind the one switch: the population branch of
`TypedSignature.similarity`, the bins path of its Gaussian kernel, and the
nearest-neighbour search of the anchor spacing. The contract is equality on
the floats, not closeness, and it was checked at the three registered
levels.

| level | what was compared | differing |
|---|---|---|
| every entry, EVAL | the node-similarity matrix of every lens type of every pair, and the anchor-weighted matrix: 5,672 \| 1,800 matrices, 14,989,430 \| 5,854,028 entries | 0 \| 0 |
| every graph, EVAL | the anchor spacing and the redundancy-aware edge-weight table of the 79 graphs (66,951 \| 36,319 weights) | 0 \| 0 |
| the call | `evidence_score` on all 300 pairs both ways: score, correspondence, every trace value | 0 \| 0 |
| stored records | the declared defaults measured with the loops and with the kernels, every stored key | 0 of 10,500 \| 0 of 10,500 |
| stored records | the 16 stage-2 and stage-3 configurations against the records stored before the kernels existed | 0 of 307,800 |
| pins | 139 for the matrices (the Gaussian block on 66 of the 67 classes that register ranges; `REGION_SHAPE` has a slot rule of its own and stays per pair), 69 for the spacing (against the function as it stood, copied into the pin) | green |

The pins were themselves tested by breaking the kernels on purpose. Of
nine mutations, seven reddened the pins at once: the circular minimum
dropped, the slot weights ignored, an ABSENT slot let through, a class's
own slot rule ignored, a too-small block answered with 0 instead of the
loop, `sqrt` of the squared distance in place of `math.hypot`, the 10⁻¹²
floor moved. Two did not: a population of the wrong shape sent to the bins
block, and the strict 10⁻¹² guard dropped. Both holes were closed with a
pin that fails under the mutation. Widening the candidate margin of the
spacing kernel from 10⁻⁹ to 0.5 changes nothing, as it must: the margin
decides how many candidates `math.hypot` is asked about, not the answer.

### 2. What it buys

| | full | v1 |
|---|---|---|
| one `evidence_score` call, Python loops: mean (median) | 0.610 s (0.572 s) | 0.136 s (0.134 s) |
| the same call, numpy kernels | 0.074 s (0.071 s) | 0.034 s (0.032 s) |
| ratio over all 300 calls | **× 8.2** | **× 4.0** |
| ratio per call: median, smallest, largest | × 7.8, × 2.2, × 15.3 | × 4.1, × 1.1, × 5.7 |
| the node-similarity matrices alone | 161.0 s → 11.1 s, × 14.5 | 29.4 s → 3.2 s, × 9.3 |
| the anchor spacing alone, 79 graphs | 0.45 s → 0.10 s | 0.44 s → 0.09 s |
| one 300-pair configuration in the harness (2 workers, breakdowns included) | 99 s → 17 s | 23 s → 7 s |

One process, each pair timed both ways back to back, the order alternating
(`phase3_speed.py`). An earlier figure of × 8.7 for the full window came
from a 16-call smoke run; the complete run is × 8.2.

- **Where the time was.** `CONTOUR_PEAK` and `TextureBoundary` were 61 % of
  the matrix time on full (54 s and 45 s of 161 s; × 36 and × 31 now). The
  bins-only lens types — every node of v1 — were not touched by the first
  form of the change; the Gaussian block takes `Position` from 13.8 s to
  1.4 s (× 10) and the other four by × 7 to × 9.
- **What is not faster.** `PEAK_CHAIN` (three axes written against a
  four-axis layout: the aligned comparison, per pair, × 1.0); the anchor
  factor (`math.exp` and `math.hypot` have no bit-identical array form); any
  block under 48 pairs (population), 16 pairs (Gaussian) or 24 points
  (spacing), where numpy's fixed cost exceeds the loop. Those thresholds
  exist because the first form was about 8 times *slower* than the loop on
  a 2 × 2 block, and more on a single pair.
- **What is left in a call** (profiler, full window, indicative): about a
  third the matrices (half of that the kernels themselves, a quarter
  deciding which block each signature belongs to); a tenth the anchor
  factor; a sixth the edge-weight table; the remaining 40 % the edge loops
  — dictionary look-ups, canonical keys, enum hashing — with no single
  item above 10 %. The next factor of two is there, and it needs the edge
  data laid out once per graph instead of once per call. That is a
  structural change, not a kernel, and is not attempted here.

### 3. P4b — fails its rule on full by one comparison of 225; the two factors are worth reading

What the relational term is made of, declared defaults, own-class pair /
wrong-class pair (means):

| | full | v1 |
|---|---|---|
| verifiable edges | 666 / 597 | 390 / 346 |
| … with a counterpart | 137 / 117 | 47 / 37 |
| **presence** (the per-pair share, mean) | 0.197 / 0.189 | 0.119 / 0.106 |
| … of which count-only (flag, no graded axis in common) | 24.8 / 18.3 | 7.5 / 5.3 |
| share of the similarity sum that is count-only | 25.7 % / 23.3 % | 20.6 % / 20.1 % |
| mean similarity of the physical pairs (**physics**) | 0.658 / 0.637 | 0.677 / 0.621 |
| the term as it is (one mean) | 0.142 / 0.131 | 0.087 / 0.071 |
| the term kept apart (presence × physics) | 0.130 / 0.120 | 0.081 / 0.066 |

Against the registered measures, on the declared defaults (under D3's three
terms in brackets):

| measure | full | v1 |
|---|---|---|
| pairs on which the term rises / falls | 0 / 300 [0 / 300] | 0 / 292 [0 / 292] |
| largest fall | 0.022 | 0.020 |
| discrimination of the term inside a query | 0.5378 → 0.5333 [0.6044 → 0.5867] | 0.7200 → 0.7244 [0.7200 → 0.7511] |
| … of presence alone, of physics alone | 0.542, 0.618 | 0.689, 0.671 |
| the term deciding alone, hits | 23 → 23 [27 → 25] | 41 → 41 [43 → 47] |
| variance of the score: which reference / own class | 12.7 / 7.8 → 13.0 / 7.5 % | 30.7 / 11.2 → 32.8 / 10.9 % |
| pairs scoring exactly 0 | 30 → 44 | 48 → 62 |
| hits, paired | 37 → 37, +0/−0 [37 → 38, +2/−1] | 39 → 38, +0/−1 [40 → 41, +1/−0] |

- **The rule.** The term rises on no pair and no accuracy falls, but the
  discrimination falls on full — 121 comparisons won of 225 become 120 —
  so the switch stays off. On v1 it rises by the same one comparison. The
  honest reading is that keeping the count apart changes nothing that can
  be measured here: the constant it removes is nearly class-blind (25.7 %
  against 23.3 % of the similarity sum), as the pre-registration expected.
  Both predictions held (the term falls on every pair with a flag and a
  physical match; more pairs clamp to 0).
- **What the two factors show.** On full four relations in five have no
  counterpart at all, on v1 seven in eight (presence 0.20 | 0.12). On full,
  whether a relation has one says almost nothing about the class (presence
  alone 0.542; 0.5 is no information); the agreement of the matched
  relations says more (physics alone 0.618). So on the full window the
  relational term — 23 hits when it decides alone, chance 18.75 — is
  carried by a count that does not discriminate. On v1 both factors do, and
  the term alone (41) is ahead of the whole score (39). What the full
  window adds to v1 — the higher-tier nodes and their relations — lowers
  the term's discrimination from 0.72 to 0.54.
- **A second exception to "never rises".** The stored-record ledger found
  5 pairs measured under the specificity cap where presence × physics is
  above the mean. Under the cap a flag reads its class's specificity
  (0.576) in the mean and is a full count once the count is kept apart. It
  is the Phase-4-floors exception again, by another route: the identity
  needs every count-only pair to read 1.0. Pinned
  (`test_specificity_cap.py`); the ledger no longer counts it as a
  violation where its premise does not hold. Both switches are off, and
  the comment in `evidence_score` names only the first exception — it
  gets the clause with the next library edit (changing it now would
  re-open every digest for a comment).
- **`EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS`** is pinned on synthetic
  graphs and unmeasured: nothing in this harness reads the edge mass it
  changes.

### 4. D1's signal — the binary-credit control

Stage 3 left one line open: under D3 with the exact assignment the graded
node term alone beat the count formula alone on full (49 against 41,
p = 0.021). The control asks whether that lead is the grading's.

| node term alone against the count formula alone | full | v1 |
|---|---|---|
| declared defaults, graded credit | 39 against 34, +7/−2, p = 0.18 | 18 against 19, +0/−1 |
| the same under binary credit | 34 against 34, +0/−0; r = 1.000 | 19 against 19, +0/−0; r = 0.998 |
| D3, graded credit | 49 against 41, +9/−1, p = 0.021 | 45 against 48, +1/−4, p = 0.38 |
| the same under binary credit | 40 against 41, +0/−1; r = 0.999 | 48 against 48, +0/−0; r = 0.995 |

Under binary credit and the exact assignment the node term **is** the
count formula: the same answers, a correlation of 1.000 between the
references of a query. So whatever the graded term has beyond the count is
the grading, and on full it is worth 5 to 8 queries when the node term
decides alone; on v1, where no node carries a population, it is worth
nothing or less. This is not the registered signal. That one is the gap
the grading opens between the *whole score* and the count-only classifier,
and the control measures it directly: graded against binary, 37 against 35
on full and 39 against 40 on v1 (under D3: 37 against 36, 40 against 41).
+2 | −1, as in stage 1. **The D1 signal remains not met.** What the
control adds is narrower: the grade carries class information where the
nodes carry populations, and the score as built does not pass it on.

### 5. The dark alignment — it could not refine; now it does; what it costs

`phase3_d2_mechanism.py`, 32 pairs per window, no score computed.

| | full | v1 |
|---|---|---|
| candidate pairs per graph pair | 577,000 (355,000 – 790,000) | 85,000 (58,000 – 111,000) |
| as built: refinement sweeps run; budget exhausted | **0.00**; on 32 of 32 | 2.03; on 27 of 32 |
| as built: seconds per pair | 0.20 | 0.61 |
| budget counts the refinement only: sweeps; exhausted | 3.00; on 0 | 3.00; on 0 |
| seconds per pair | 0.67 (max 0.92) | 0.23 (max 0.35) |
| one round + anchor prior returns the live correspondence | 7 of 32 | 32 of 32 |

- **As built, `MATCH_GLOBAL_CONSISTENCY_ENABLED` would have done nothing on
  the full window**: the seed's similarity matrix used up the budget on
  every pair, and the function returned the unrefined seed with
  `budget_exhausted = True`. What stage 1 read from the code is now
  measured.
- **The 25 full-window pairs that differ from the live correspondence** are
  accounted for: 18 contain a pair across lens types that share a
  signature class (`Corner`, `Junction`, `T` and `Y` all write `JUNCTION`;
  the dark path pairs across them by design), and 7 are **another optimum
  of the same assignment** — equal total weight, 1 to 8 nodes paired
  differently (`EndStopped`, `BorderOwnership`). The node assignment has
  ties on real graphs, and which optimum is returned is an accident of the
  matrix layout. The live path is deterministic (same node order, same
  answer, also across hash seeds), but its choice among equal optima
  carries no meaning, and the edges it then verifies depend on it.
- **Cost.** With the refinement actually running, a full-window pair costs
  0.67 s — nine times the whole live call (0.074 s). The refinement is a
  Python loop over the positive candidates; D2 at this cost is a
  300-pair run of several minutes per configuration, affordable for an
  experiment and not for a default.

### 6. The specificity cap — descriptive

| | full | v1 |
|---|---|---|
| similarity sum per own-class pair | 98.5 → 79.9 | 34.6 → 23.0 |
| the term, own / wrong | 0.142 / 0.131 → 0.114 / 0.108 | 0.087 / 0.071 → 0.057 / 0.047 |
| its discrimination | 0.5378 → 0.5378 | 0.7200 → 0.6756 |
| … under D3 | 0.6044 → 0.5422 | 0.7200 → 0.6667 |
| the term deciding alone, hits | 23 → 22 | 41 → 39 |
| pairs scoring exactly 0 | 30 → 59 | 48 → 89 |
| hits, paired | 37 → 36, +0/−1 | 39 → 36, +0/−3, p = 0.25 |
| under D3: hits, paired | 37 → 34, +1/−4, p = 0.38 | 40 → 43, +3/−0, p = 0.25 |

The prediction held where it was firm (the term falls on every pair with a
matched generic edge) and is undecided where it was conditional (accuracy:
three falls and one rise, none distinguishable from chance). What the cap
does to the one term it acts on is clearer: the generic edges'
full-strength matches carry class information, and capping them lowers the
term's discrimination on v1 (0.720 → 0.676) and, under D3, on full
(0.604 → 0.542); it raises it nowhere. It also breaks the property that a
graph scores 1.0 against itself (pinned). Of the library's two texts, this
harness sides with the later one ("[U32] effective weights are near-flat
by necessity"). The switch stays off and the cap stays dead code until you
decide which text stands.

### 7. Instruments corrected during the stage

Three, none of them in the library, each a false reading that had to be
explained before the figures above could be written.

- **"0 of 79 graphs compared."** After the rebuild the graph comparison
  reported that no old cache entry existed. The entries were there: I had
  passed the 16-character prefix of the old digest and the tool looks
  entries up by the full 64. With the full digest: 79 of 79 identical. The
  tool now resolves a prefix from the stored records or stops with an
  error.
- **"A = 2" in the bug hunt.** Two new ledger lines failed on the stored
  records: 623 pairs on "sim sum ≤ physical sum + count-only pairs", every
  one by less than 6 × 10⁻⁸ — the two sums are stored at 10 significant
  digits and are of the order of 100, and the line used an absolute 10⁻⁹;
  and 5 pairs on "never rises", which is the cap × P4b exception of §3.
  The tolerance is now scaled to the stored precision and the second line
  is applied only where its premise holds. Re-run: 0 findings in A–F.
- **The mechanism check could not say why 7 correspondences differed.** It
  now compares total weights and reports the ties of §5.

The first bug-hunt output is kept
(`results/logs/bug_hunt_stage4_first_run_A2.log`). The re-runs of the three
corrected instruments, and of the pins and the triage after one pin was
added, are `results/logs/measurement_chain_stage4_rerun.sh`; no library or
harness file changed between the chain and the re-run, so the digests of
every stored record are the tree's.

---

## Stage 3 — two correctness fixes of the correspondence and the edge loop: results against the pre-registered rules

EVAL-75, in-sample; "a | b" is full | v1. Tables:
`results/phase2_mixture_reversal.txt`, `results/phase2_exact_assignment.txt`,
`results/phase2_correspondence_census.txt`,
`results/phase2_symmetry_probe.txt`. Both changes restore a contract the
library already states; neither was predicted or required to gain accuracy.

### 0. Verdicts

| change | rule (fixed beforehand) | full | v1 | default |
|---|---|---|---|---|
| mixture reversal, `EDGE_REVERSE_FLIPS_MIXTURE` | pins green; the run reproduces the census figures; no paired fall with p < 0.05 | reproduced exactly; 38 → 38, 41 → 41 | reproduced exactly; 34 → 34, 39 → 39 | stays **on** |
| exact assignment, `MATCH_ASSIGNMENT_EXACT` | pins green; stranding 0 and exact ≥ approximate on every pair; no paired fall with p < 0.05 | 0 stranded, 0 pairs below; 38 → 37, +1/−2, p = 1.00; under D3 41 → 37, +2/−6, p = 0.29 | 0 stranded, 0 pairs below; 34 → 39, +6/−1, p = 0.125; under D3 39 → 40, +4/−3 | stays **on** |

Order dependence: the three reference configurations re-measured under
`PYTHONHASHSEED=1` reproduce the stored records with 0 of 55,800 values
differing (`results/logs/hashseed1_vs_seed0.txt`).

### 1. Mixture reversal — the prediction held to the last figure

| | scores that move | largest | answers | hits |
|---|---|---|---|---|
| one-sided score, predicted from the census | 19 \| 18 | 0.0011 \| 0.0020 | 0 \| 0 | 38 \| 34 |
| one-sided score, measured under the switch | 19 \| 18 | 0.0011 \| 0.0020 | 0 \| 0 | 38 \| 34 |
| balanced score, predicted | 41 \| 34 | 0.0005 \| 0.0011 | 1 \| 0 | 41 \| 39 |
| balanced score, measured | 41 \| 34 | 0.0005 \| 0.0011 | 1 \| 0 | 41 \| 39 |

The similarity sum moves on 21 pairs per window on the core side and 28 | 27
on the query side, as predicted; no node mass, no count and no absence value
moves. After it, no edge pair is read two ways from its two ends (bug hunt,
section D: 45 | 44 before, 0 after), and one query whose answer depended on
the direction of the comparison on full no longer does.

*A false alarm, and the harness defect behind it.* The measurement chain
printed "NOT REPRODUCED — 247 of 300 scores moved" for the full window. The
library was right and the instrument was wrong: `run_cached` handed back
full-precision floats for a record it had just computed and the stored 10
significant digits for a record it found on disk, and the script compared
one of each. Stored against stored, the figures are the ones above. The
harness now always returns the stored form; the false output is kept as
`results/logs/mixture_reversal_fresh_vs_cached_artefact.txt`. No earlier
table was affected (the stage-2 script compares stored records).

### 2. Exact assignment — every prediction held; what it shows is the objective

| measure | approximation | exact |
|---|---|---|
| pairable node slots left stranded | 13,175 (7.3 %) \| 4,161 (5.4 %) | **0 \| 0** |
| pairs whose total weight is below the optimum | 300 of 300 \| 300 of 300 | 0 \| 0 (exact below approximate: 0 \| 0) |
| gain in total weight | – | mean 4.5 % \| 2.1 %, max 12.1 % \| 4.4 % |
| node pairs; share of the old pairs kept | 166,283 \| 72,683 | 179,474 \| 76,844; 78.5 % \| 86.0 % |
| same correspondence for (Q, C) and (C, Q), 48 probed pairs | 0 \| 0 | **48 \| 48** |
| \|score(Q, C) − score(C, Q)\|, balanced score, mean | 0.0040 \| 0.0031 | **0.0000 \| 0.0000** (max 0) |

The symmetry prediction was "most pairs, not all": it is all 48 on both
windows, so the balanced score is now symmetric on real graphs and not only
as a formula.

Own-class / wrong-class means on the declared defaults:

| | full, approximate | full, exact | v1, approximate | v1, exact |
|---|---|---|---|---|
| binary coverage | 0.799 / 0.741 | 0.862 / 0.798 | 0.896 / 0.862 | 0.948 / 0.908 |
| the count formula (no matching at all) | 0.865 / 0.801 | 0.865 / 0.801 | 0.950 / 0.912 | 0.950 / 0.912 |
| mean similarity of the pairs kept | 0.891 / 0.879 | 0.872 / 0.859 | 0.926 / 0.918 | 0.910 / 0.900 |
| verifiable core edges | 578 / 520 | 666 / 597 | 338 / 302 | 390 / 346 |
| relational consistency | 0.152 / 0.137 | 0.142 / 0.131 | 0.094 / 0.075 | 0.087 / 0.071 |
| edge absence | 0.318 / 0.379 | 0.224 / 0.294 | 0.235 / 0.289 | 0.127 / 0.194 |
| pairs scoring exactly 0 | 53 | 30 | 88 | 48 |
| variance of the score: which reference / own class | 10.1 % / 10.1 % | 12.7 % / 7.8 % | 29.4 % / 10.4 % | 30.7 % / 11.2 % |
| hits (apple / car / cup / horse) | 38 (2 / 6 / 16 / 14) | 37 (1 / 7 / 16 / 13) | 34 (0 / 7 / 14 / 13) | 39 (2 / 8 / 15 / 14) |

Every direction was predicted: coverage rises to the count formula, the
similarity of the kept pairs falls, more edges become verifiable, absence
falls, relational consistency falls.

- **What it shows.** With every pairable node paired, binary coverage *is*
  the count formula (0.862 against 0.865), and the graded node term follows
  it: its correlation with the count formula between the references of one
  query goes from 0.946 to 0.983 | 0.960 to 0.977. The approximation's
  stranded nodes were noise; under the correct optimum of the stated
  objective, the node term says how many nodes of each lens type the two
  graphs have, and little else. The share of the score's variance that is
  *which reference* rises on full (10.1 → 12.7) and the own-class share
  falls (10.1 → 7.8).
- **D3 under the exact assignment** (reported; D3's rule was judged in
  stage 2 and is not re-judged): 41 → 37 on full, 39 → 40 on v1; the
  reference share goes 10.5 → 18.5 | 18.3 → 31.2. The three queries D3
  seemed to gain on full were inside the approximation's noise.
- **D1's signal.** Under D3 with the exact assignment the node term alone
  beats the count formula alone on full (49 against 41, +9/−1, p = 0.021)
  and not on v1 (45 against 48, +1/−4). One window of two, among many
  comparisons; the binary-credit control is in stage 4 §4. It is not a met
  signal.
- **What follows for the gated items.** An exact matcher of this objective
  is a counter. Whatever is to separate classes has to come from *which*
  pairs are admitted (a priced unmatched decision — the graded-admission
  experiment) or from the relations (D2), not from pairing more.

---

## Stage 2 — D3, Phase 4, null calibration: results against the pre-registered rules

EVAL-75, in-sample; "a | b" is full | v1; paired figures are +gained/−lost
queries with the exact McNemar p. Tables: `results/phase2_d3_p4_null.txt`
(the registered measures), `results/phase2_addendum.txt` (descriptive, A–F),
`results/phase2_correspondence_census.txt`, `results/phase2_symmetry_probe.txt`.
Every configuration sets every switch explicitly. The pre-registrations are
further down, unchanged.

### 0. Verdicts

| change | rule (fixed beforehand) | full | v1 | default |
|---|---|---|---|---|
| Phase 4, `EDGE_UNMEASURED_IS_NOT_EVIDENCE` | pins green; no paired fall with p < 0.05 | 0 scores move | 0 scores move | stays **on** — and is **unexercised** here |
| D3, `EVIDENCE_QUERY_SIDE_TERMS` (all three terms) | pins green; reference share of the score's variance falls on **both** windows; no paired fall with p < 0.05 | 10.1 % → 10.5 %, **does not fall**; 38 → 41, +15/−12, p = 0.70 | 29.4 % → 18.3 %, falls; 34 → 39, +17/−12, p = 0.46 | stays **`()`** — rule fails on full |
| null calibration, `EVIDENCE_NODE_NULL_CALIBRATED`, in the configuration it would default in (one-sided) | pins green; reference share falls on both windows; no paired fall; no query with four equal scores | 10.1 % → 12.9 %, **rises**; 38 → 39 | 29.4 % → 35.8 %, **rises**; 34 → 36 | stays **`False`** |
| the same on top of D3 (reported, not the deciding configuration) | | 10.5 % → 5.5 %; 41 → 41; **2** undecidable queries | 18.3 % → 14.1 %; 39 → 40 | — |

Continuity: F6+D1 with every stage-2 switch off reproduces the stage-1 record
exactly (0 of 7,800 values differ). All figures are in-sample, there are many
comparisons, and none of the accuracy differences is distinguishable from
chance.

### 1. D3 — fails its rule on full; what it does is real and one-sided

Hits, and per class (apple / car / cup / horse, of 16 / 24 / 18 / 17):

| | full | v1 |
|---|---|---|
| F6+D1 | 38 (2 / 6 / 16 / 14) | 34 (0 / 7 / 14 / 13) |
| node term two-sided only | 39 (3 / 6 / 15 / 15), +2/−1 | 38, +5/−1, p = 0.22 |
| all three terms (primary) | 41 (10 / 12 / 14 / 5), +15/−12, p = 0.70 | 39 (6 / 16 / 16 / 1), +17/−12, p = 0.46 |

- **Measure 1** (variance of the score explained by *which reference* / by
  *own class*): F6+D1 10.1 / 10.0 | 29.4 / 10.3; node only 4.4 / 11.6 |
  21.1 / 13.7; primary 10.5 / 12.6 | 18.3 / 19.5. The rule was coded as "the
  reference share falls"; on full it moves from 10.1 to 10.5. Read as a
  ratio to the own-class share it would pass (1.01 → 0.83). The reading
  coded beforehand applies. Both are stated so the choice is visible.
- **The node-only arm meets every criterion** (4.4 % | 21.1 %, no fall). It
  was registered as "reported alongside and not selected between", so it is
  not made a default here; it is a candidate for HELDOUT-75, named as such.
- **Mechanism.** The gain and the loss are the same thing. One-sided, the
  absence term is 56.6 % a property of the reference: mean absence per
  reference 0.54 / 0.38 / 0.36 / **0.18** (apple / car / cup / horse). The
  horse reference is sparse (462 edges against a median 761 in its queries;
  225 against 412 on v1) and is the top-down view, so almost any query
  covers it: "least absence" alone answers horse 57 | 71 times of 75.
  Two-sided, the absence means are 0.42 / 0.36 / 0.37 / 0.44; horse loses
  the free answers (14 → 5 | 13 → 1) and apple and car gain them
  (2 → 10, 6 → 12 | 0 → 6, 7 → 16). Accuracy barely moves because one
  reference-size effect has been exchanged for its removal, not for
  recognition.
- **D1's signal under D3: not met.** Node term alone against COUNT-ONLY of
  the same formula: 48 vs 41, +11/−4, p = 0.118 | 38 vs 48, +3/−13,
  p = 0.021 (worse). With binary credit the same gap is there on full
  (47 vs 41), so it is not D1's.
- **Scale.** No pair reaches 0.15 under any configuration (largest own-class
  score 0.1425 one-sided, 0.1332 under D3); `EVIDENCE_MIN_BALANCED` (0.15)
  and `THETA_EVIDENCE_POS` (0.35) cannot be met on this corpus by either
  form.
- **Symmetry.** The balanced score is symmetric as a *formula*. On real
  graphs score(Q, C) ≠ score(C, Q) in every probed pair, because the
  correspondence was not the same both ways — see Stage 3, where that is
  traced to the assignment and fixed. One exception is by design: with
  exactly one edgeless side the two directions read 1.0 and 0.0 (the
  registered "edgeless core is vacuous"; pinned).

### 2. Phase 4 — on by the spec, never exercised

No value, score or answer moves in either window, with the query-side terms
off or on. The censuses said so beforehand: no matched edge pair has zero
commonly measured slots. Its support is the spec and the synthetic pins;
`False` restores the floors. **What the censuses do show** is the spec's
*second* line, which Phase 4 did not implement: the binder writes a literal
`containment = 1.0` on `CONTOUR_ORIENTATION` (10,303 edges),
`REGION_CONTENT` (5,489) and `JUNCTION_ORIENTATION` (736), the schema scores
it as a measured graded slot, and every such pair matches at exactly 1.000:
1,758 own-class matches, **25.6 %** of the own-class similarity sum on full
(`REGION_CONTENT` alone is 21.5 % on v1). A further 421 `CURVE_CONTAINMENT`
matches (6.1 %) read 1.000 because both of their measured slots are constant
over the corpus; with them it is 2,179 matches and 31.7 %. That is a count
read as physics. It is the next change (P4b, pre-registered below and
reported in stage 4), not part of this one.

### 3. Null calibration — negative, and the reason is structural

- One-sided: reference share 10.1 → 12.9 | 29.4 → 35.8; hits 38 → 39 |
  34 → 36; pairs scoring exactly 0: 53 → 128 | 88 → 118.
- With D3: 10.5 → 5.5 | 18.3 → 14.1; hits 41 → 41 | 39 → 40; pairs at 0:
  47 → 140 | 102 → 150; 2 | 0 queries left with four equal scores.
- Binary-credit check, as predicted: 300 | 297 of 300 pairs score 0.
- **Why it cannot work in this form.** N0 carries the same coverage factor
  as N, so the reference-size offset survives the subtraction. The random
  pairing is not class-neutral (mean similarity 0.513 own-class against
  0.503 wrong-class | 0.459 / 0.453), and what the matcher adds over it is
  class-blind (selection gain 0.3773 / 0.3758 | 0.4664 / 0.4648). The fixed
  0.15 absence weight, set on the uncalibrated scale, then zeroes 118–150 of
  300 scores.
- It implements `matching/calibration.py`'s `NullModel.from_matrix` and
  nothing else: `background_weights` and a uniform null are different forms,
  not registered, not tried.
- The closed form of the chance level is right: it equals exhaustive
  enumeration of every maximum pairing on 77 pinned scenarios (boosts,
  weights, partial weights) and agrees with 2,000 sampled random pairings on
  real pairs (see the bug hunt). The negative result is the form's, not an
  implementation error.

### 4. What the frozen bars can see — a correction to stage 1

The figground pins in this checkout call the edited matcher functions
**0 times** (`PERCEPT_REACH=1`, `results/logs/bars*_figground_*.txt`), and
`_u27_selftest.py` is perception only. "Figground green" is therefore true
and says nothing about any matcher switch — including the "figground green
after F6+D1" reported in stage 1, which I withdraw as evidence. Only
`tests_matcher/` exercises these changes.

---

## Stage 1 — Phase 1 (F6) and D1

Full tables: `results/phase1_f6_d1.txt`. Script: `phase1_f6_d1.py`.

### 0. The before is the Phase 0 baseline, exactly

Both changes sit behind a parameter. With both set to `False`, the edited
tree gives back the frozen Phase 0 baseline on all 6,900 stored per-pair
values per window (24 quantities x 300 pairs), on the per-type edge and node
tables, and on 75/75 answers. Largest difference: 0. This is also the proof
that the rebuilt front-end graphs are the Phase 0 graphs (the sandbox lost
its cache; the digest changed because `core/` and `parameters/` did).

### 1. Per class, every setting

`OFF` = both switches off. `F6+D1` = the tree's new default.

| window | setting | acc | apple | car | cup | horse | answers a/c/c/h | vs OFF |
|---|---|---|---|---|---|---|---|---|
| full | OFF | 36/75 = 48.0 % | 2/16 | 6/24 | 16/18 | 12/17 | 2/12/30/31 | |
| full | F6 | 36/75 = 48.0 % | 2/16 | 6/24 | 16/18 | 12/17 | 2/12/30/31 | +0/-0 |
| full | D1 | 38/75 = 50.7 % | 2/16 | 6/24 | 16/18 | 14/17 | 2/11/29/33 | +2/-0, p = 0.50 |
| full | F6+D1 | 38/75 = 50.7 % | 2/16 | 6/24 | 16/18 | 14/17 | 2/11/29/33 | +2/-0, p = 0.50 |
| v1 | OFF | 34/75 = 45.3 % | 0/16 | 7/24 | 14/18 | 13/17 | 0/12/20/43 | |
| v1 | F6 | 34/75 = 45.3 % | 0/16 | 7/24 | 14/18 | 13/17 | 0/12/21/42 | +0/-0 |
| v1 | D1 | 34/75 = 45.3 % | 0/16 | 7/24 | 14/18 | 13/17 | 0/12/20/43 | +0/-0 |
| v1 | F6+D1 | 34/75 = 45.3 % | 0/16 | 7/24 | 14/18 | 13/17 | 0/12/20/43 | +0/-0 |

Nothing in this table is distinguishable from no change. The expected D1
accuracy drop did not happen either.

### 2. F6 — what it does

`polymorphic_edge_similarity` returned 0.0 on **both** branches of its
BLACK-HOLE block, so the fix its own comment describes never ran. It now
returns 1.0 when both sides carry no `EdgeParams` and the predicate pair the
caller passes is equal.

- Only `TOPOLOGY` has param-less core edges on this corpus.
- Own-class pairs: of 850 / 818 verifiable `TOPOLOGY` core edges, **14**
  become matched (0 before). The other 836 / 804 have no query edge between
  the mapped endpoints at all — that is F1 (the correspondence), not F6.
- 13 of 300 pair scores move, by at most 0.0049 / 0.0129. Answers that
  change: 0 / 1. Accuracy and every per-class figure: unchanged.

Decisions I took, so you can overrule them:

- **Params on one side only stay 0.0** under either setting. A missing
  measurement should not read as a perfect one, and it keeps
  `EdgeParams.similarity(None) == 0.0` true from both directions.
- **Callers that pass no `edge_types` are unaffected** (the two calls in
  `matching/structural_similarity.py`).
- The match reads **1.0 in `matched_rel_sim_sum`**, as the comment asks. That
  is a topological fact entering the sum as if it were physical similarity —
  exactly what the EDGE-EVIDENCE spec says must not happen. It measures zero
  axes (`edge_measured_axes(None, None) == 0`), so Phase 4 can separate it.
  On this corpus it is 14 edges.

### 3. D1 — what it does

`evidence_score` now credits a kept node pair
`GESTALT_LENS_BOOST x node_weight x TypedSignature.similarity` instead of the
full `boost x weight`. The correspondence and the whole edge accumulation are
untouched (checked: binary coverage and edge term identical on 300/300 pairs).

- Own-class node term: 0.799 -> 0.712 (x0.890) / 0.896 -> 0.830 (x0.926).
- An offline prediction from the Phase 0 record (graded node observable x
  edge term - 0.15 x absence) matches the implemented score on 300/300 pairs
  to 5e-13. The implementation computes what Audit B said it would.

Scope decision: **`evidence_score` only.** `mcs_score` and `mcs_similarity`
keep binary mass, because they feed containment and fusion gates
(`THETA_FUSE`, `mcs_contained`, `_classify`) whose thresholds are calibrated
on it and which nothing in this checkout can test. One keyword argument on
`_mcs_accumulate` extends it if you want it there too.

### 4. D1 success signal — NOT MET

> the count-only classifier must fall FAR below baseline after D1

COUNT-ONLY reads the two per-lens node censuses and never calls the matcher,
so its own accuracy cannot move (34 / 19). The signal has to be read as the
gap to it, and as how far the node term has stopped tracking it.

| | full | v1 |
|---|---|---|
| evidence, OFF vs COUNT-ONLY | +2 queries (p = 0.84) | +15 (p = 0.004) |
| evidence, D1 vs COUNT-ONLY | +4 queries (p = 0.54) | +15 (p = 0.004) |
| **gap opened by D1** | **+2** | **0** |
| answers D1 changed | 2 of 75 | 0 of 75 |
| node term alone, OFF -> D1 | 31 -> 37 (+6/-0, p = 0.031) | 17 -> 17 |
| r(node term, count formula), OFF -> D1 | 0.978 -> 0.974 | 0.957 -> 0.942 |
| same, between the 4 references of one query | 0.954 -> 0.946 | 0.963 -> 0.960 |

The v1 gap of +15 is there with D1 off; it is the edge term's. The criterion
I coded before the run (D1 evidence ahead of COUNT-ONLY, paired p < 0.05)
had no D1-off control and would have read "met" on v1. I added the control
after seeing the result; the output says so.

One correction to the stage-check wording: Audit B did not show that the
*signal* is binary — it showed the *code* credits in full, and that the
graded quantity is collinear with the binary one (r = 0.997). D1 confirms
the second half: making the code graded changes almost nothing.

### 5. Why — the grade is real but it is multiplied by a count

graded node term = (binary coverage) x (mean similarity of credited pairs)

| between the four references of one query | full | v1 |
|---|---|---|
| variance of log node term from binary coverage | 96.0 % | 83.2 % |
| ... from mean similarity | 1.1 % | 1.7 % |
| best/worst reference, median: coverage | x1.193 | x1.233 |
| best/worst reference, median: similarity | x1.024 | x1.028 |
| mean similarity, own-class vs wrong-class reference | 0.8905 vs 0.8790 | 0.9257 vs 0.9176 |

Similarity is about 0.88-0.93 against *every* reference: with several hundred
candidates per lens, the assignment finds a near-twin for almost any node.
Coverage — a count, and the part that carries the reference-size offset of
F4 — varies eight times more.

The similarity factor is not empty, though. Taken alone (argmax of the mean
similarity of the credited pairs, ignoring every unmatched node) it scores
**47/75 = 62.7 %** on the full window, with all four classes alive (apple
11/16, car 13/24, cup 13/18, horse 10/17; +19/-8 against the OFF evidence
score, p = 0.052), and 27/75 on v1. This is a post-hoc in-sample observable,
not a proposal and not a result to quote. What it supports: the information
D1 was meant to let through exists in the full window, and the product with
coverage buries it. On v1, where no node carries a population, there is much
less of it.

### 6. What this means for the gated items

- **D3 (query-side term)** is now the item with the largest measured
  leverage: the factor that dominates the node term is the one that carries
  the reference-size offset.
- **Phase 3 (populations)**: the similarity signal is weak exactly where
  populations are absent (v1). Wiring them is the direct test of whether the
  grade can be widened. It can only be read through the similarity factor,
  not through accuracy, until coverage stops dominating.
- **The D1 signal should be re-read after D3**, against the matching
  count-only stand-in (the same formula on the censuses).

### 7. Open question 1 — how much is already half-built

More than the README suggests. `matching/` holds a complete "Phase 5" graded
matcher that the live path does not use:

| piece | where | state |
|---|---|---|
| similarity-weighted **symmetric** node coverage, `2 * sum(quality) / (\|A\|+\|B\|)` — D1 and D3 together | `matching/fuzzy.py:162`, `matching/alignment.py` | built, not called by `evidence_score` |
| **edge-aware correspondence** with a priced unmatched decision — D2 | `matching/alignment.py:244`, `memory/cores.py:_global_correspondence` | built, dark behind `MATCH_GLOBAL_CONSISTENCY_ENABLED = False` |
| **null calibration** of coverage, `(observed - chance) / (1 - chance)` | `matching/calibration.py` | built; `MATCH_NULL_CALIBRATED`'s comment says `evidence_score` calibrates node coverage against the null — **it does not**; only the 0.85 floor was removed |
| **topological count separate from physical similarity**, `edge_coverage x edge_param_mean` — Phase 4's separation | `matching/fuzzy.py:168` | built, not called by `evidence_score` |
| the F6 fix | `core/schema.py` | was written as a comment only; now live |
| the `[U32]` "all-UNKNOWN is not evidence" rule | `core/schema.py`, `EdgeParams.similarity` | comment says 0; code returns **0.2**; a pair with no commonly measured axis returns **1.0** |

Two cautions about the dark D2 path as it stands. It builds a dense
`|Q| x |C|` similarity matrix across all lens types in pure Python, and it
charges those evaluations to `MATCH_ALIGN_MAX_PAIR_EVALS = 200000` before
the first refinement round. An ETH-80 pair is about 860 x 750 = 645,000, so
on graphs of this size the budget is spent before any relational refinement
runs: the flag would return the unrefined seed. It also drops the anchor
factor. Both are read from the code, not yet measured.

Null calibration is the code base's own answer to section 5 (a similarity of
0.89 that every reference reaches is not evidence). It is not on the
authorized list, so I have not touched it. It is the first thing I would ask
you about.

---

## Pre-registration — D3 (written before any D3 number was computed)

**Form.** Taken from the code base's own symmetric designs (`mcs_similarity`:
`(matched_q + matched_c) / (q_mass + c_mass)`; `matching/fuzzy.py`: weighted
Dice), not invented here. Every ratio in `evidence_score` reads both sides:

    node term   N = (q_matched_nm + c_matched_nm) / (q_node_total + c_node_total)
    edge term   R = (core-side sim sum + query-side sim sum)
                    / (verifiable core edges + verifiable query edges)
    absence     A = (unverifiable core edges + unverifiable query edges)
                    / (core edges + query edges)
    score         = clamp(N x R - 0.15 x A)

The query-side edge quantities come from the mirror of the existing core-side
loop: a query edge is verifiable when both endpoints are mapped, and its
similarity is the best over the core edges between the mapped endpoints. The
mirror is self-checking: called in the core direction it must return the
matcher's own `verifiable_core_edges_count` and `matched_rel_sim_sum`.

One general parameter selects which terms read both sides; the empty setting
is today's score. Primary arm: all three. Decomposition arms, reported
alongside and not selected between: node only; node + absence.

**What is measured.**

1. Share of score variance explained by *which reference* against *own
   class* (Audit B6: 9.3 % vs 9.0 % full, 28.6 % vs 10.6 % v1). It should fall.
2. The answer distribution. The apple reference wins 2 and 0 of 75 answers
   with 16 apple queries; horse takes 33 and 43.
3. Accuracy per class, paired against F6+D1. Reported whatever it is. B6's
   transductive centring gained 4 / 12 queries, which sets a direction, not a
   size: D3 is a different operation.
4. The D1 signal again, against the count-only stand-in of the *same*
   formula.

**Default rule, fixed now.** The switch defaults to the balanced form only if
the pins stay green, (1) holds on both windows, and accuracy does not fall
with paired p < 0.05 on either window. Otherwise it stays in the tree, off,
and is reported as a negative.

**Known risk I cannot test here.** A symmetric score changes absolute values
wherever the query is much larger than the core (a part matched inside a
scene): `THETA_EVIDENCE_POS`, the per-class thresholds and U20 are calibrated
on the one-sided score. ETH-80 single-object images cannot show that.

**Addendum, recorded before any D3 number was computed.** Two details were
fixed while writing the code and are pinned in
`tests_matcher/test_d3_query_side_terms.py`; neither was in the text above.
(a) An edgeless core keeps its vacuous relational reading (R = 1, A = 0)
under every setting: a core that asserts no relation cannot be contradicted
by the query's relations. (b) Under `EVIDENCE_ONLY_EDGES` only the node term
reads both sides; that branch keeps its own relation rule.

---

## Pre-registration — Phase 4, edge honesty (written after the step-0 censuses and the code; before any score under the switch was computed)

**Step 0, already measured (read-only, no switch involved).**
`results/phase4_edge_census.txt` puts every verifiable core edge of every
(query, reference) pair in one bin, and asserts its totals against the
matcher's own trace. `results/phase4_axis_census.txt` lists, per relation,
what its parameter slots can say over the 79 EVAL graphs.

- **No matched edge pair has zero commonly measured slots**, on either
  window, in 300 pairs: the bins UNMEASURED and CATEGORICAL ONLY are empty.
  The only matches without a measurement are F6's parameter-less TOPOLOGY
  pairs: 14 own-class, 30 over all pairs. So the 0.2 and 1.0 UNKNOWN floors
  the spec names are **not exercised on this corpus**.
- Cross-schema matches exist: 506 own-class edges (full), mean similarity
  0.31, 2.3 % of the similarity sum.
- The front end does not depend on the change: 79 of 79 graphs have the same
  content digest before and after it.

**What is implemented** (`core/schema.py`, switch
`EDGE_UNMEASURED_IS_NOT_EVIDENCE`, default `True`): the spec's first and
third lines. A pair of edge parameters with no slot measured on both sides
reads 0.0 instead of 0.2 (both all-UNKNOWN) or 1.0 (tuples differ, nothing
in common). In the cross-schema path the mean runs over the shared axes
measured on both sides instead of over all shared axes, and reads 0.0 when
there is none. Any pair with a slot measured on both sides is untouched.

**Prediction, from the census.** No same-schema similarity changes. A
cross-schema similarity can only stay or fall. No answer changes.

**What is measured.** Pairs whose score differs between the two settings and
the largest difference; the similarity sum per relation, before and after;
answers changed; accuracy, paired. Once with the query-side terms off and
once with all three on (the query-side loop compares edges the census did
not look at).

**Default rule, fixed now.** This is the spec's rule, not an accuracy fix:
no gain is predicted or required. It stays on if the pins stay green and
accuracy does not fall with paired p < 0.05 on either window.

**Not in this step** — the spec's second line (the count kept apart from the
physics). See the Phase 4 results section for what the censuses say about
where a count poses as physics on this corpus; it is not where the floors
are.

---

## Pre-registration — null calibration of node coverage (written before any null-calibration number was computed, and before any D3 or Phase 4 number)

**Form.** Taken from `matching/calibration.py` (built, dark), not invented
here. A match fraction is evidence only by what it exceeds under an
uninformative correspondence:

    calibrate_fraction(observed, chance) = clamp((observed - chance) / (1 - chance), 0, 1)
                                           and 0 when chance >= 1

The null is the module's `NullModel.from_matrix`: a random pairing, whose
expected credit per pair is the mean of the candidate matrix. It is applied
where the matcher forms pairs — inside each lens type — and composed with
the same masses as the observed term:

    chance credit of core node c, lens l   = sum over q in Q_l of g(q, c) / max(|Q_l|, |C_l|)
    chance credit of query node q, lens l  = sum over c in C_l of g(q, c) / max(|Q_l|, |C_l|)
    g(q, c)      = the credit the live rule gives the pair if it is paired:
                   the signature similarity (D1), or 1 when it is positive (binary credit)
    chance mass  = sum of boost x node weight x chance credit
    N0           = chance mass / total mass, over the side or sides the node term reads
                   (the core; both under D3 "node")
    node term    = calibrate_fraction(N, N0)

`1 / max(|Q_l|, |C_l|)` is the probability that a given pair belongs to a
uniformly random maximum pairing of the two sets. The signature similarity
is the credited quantity; the anchor factor steers who pairs with whom and
is not credit, so it is not in the null either. Nothing else changes: the
relational term, the absence term and every constant stay as they are.
One switch, `EVIDENCE_NODE_NULL_CALIBRATED`, default `False`. The existing
`MATCH_NULL_CALIBRATED` (which only removes the 0.85 floor, whatever its
comment says) is left alone.

**What the form can and cannot do — stated before measuring.**

- It removes the credit any pairing would earn. Under binary credit nearly
  all of the node term is such credit, so the calibrated term must be about
  0 there. Pinned on synthetic graphs.
- It does not remove the gain of a *best* pairing over a random one, and
  that gain grows with the number of candidates. Removing it needs the
  matcher's own score on unrelated graphs, a corpus statistic the module
  rules out ("no corpus state").
- A graph whose nodes are interchangeable inside every lens type scores 0
  against itself: chance is 1. That is the module's rule; it is pinned and
  it is a behaviour change.

**What is measured.**

1. N0 itself: its size next to N, on own-class and on wrong-class pairs.
2. Share of the variance of the score, and of the node term, explained by
   *which reference* against *own class* (the Audit B6 measure). The
   reference share should fall.
3. r(node term, count formula), 0.974 / 0.942 after D1. It should fall.
4. Accuracy per class, ties split, paired against the same configuration
   without calibration. The number of pairs whose score is exactly 0, and
   the number of queries left with four equal scores: the absence penalty
   (0.15) was set on the uncalibrated scale.
5. The node term alone against COUNT-ONLY (the D1 signal). The calibrated
   term has no count-only stand-in — it is 0 under binary credit — so the
   comparison is with the uncalibrated count formulas.
6. Interaction with D3: (query-side terms off / all three) x (calibration
   off / on), per window, on accuracy and on measure 2. From the traced N
   and N0, offline, reported and not selected between: N - N0 alone and
   (N - N0) / (1 - N0).

**Default rule, fixed now.** Applied to the configuration it would be the
default in (with D3 as its own rule leaves it). The switch defaults to on
only if the pins stay green with it on, the reference share of the score's
variance falls on both windows, accuracy does not fall with paired p < 0.05
on either window, and no query is left with four equal scores. Otherwise it
stays in the tree, off, reported as a negative.

**Known risks I cannot test here.** Absolute values change everywhere, so
`THETA_EVIDENCE_POS`, the per-class thresholds and U20 are affected as under
D3. The self-match of a homogeneous graph going to 0 can matter to any
caller that re-recognises a stored core from its own instance.

---

## Pre-registration — mixtures read against their stored direction (written after the step-0 census of the bug hunt and before the switch existed or was run)

**What the census found** (`diagnostics/phase2_bug_hunt.py`, sections C and D;
the pre-fix output is kept in `results/logs/bug_hunt_before_fixes_*.txt`).
The matcher's edge loop reads an undirected query edge from either end and
reverses its parameters when it reads it against its stored direction
(`reverse_edge_params`). That function returns anything that is not an
`EdgeParams` unchanged, so an `EdgeParamsMixture` is read backwards with its
prototypes **not** reversed. The library has the right operation
(`EdgeParamsMixture.flipped_copy`, "for inverted endpoint traversals") and
its other matcher uses it (`matching/structural_similarity.py:233, 251`);
`memory/cores.py` does not. The mirror loop D3 added copies the historical
loop call for call, so it has the same defect.

- 3,525 of 66,951 edges in the 79 EVAL graphs carry a mixture (3,121 with one
  prototype). 198 of them, all `ORIENTATION_CONTINUITY`, have a prototype
  that a reversal changes.
- Per window, 86 candidate evaluations in the core direction and 67 | 65 in
  the query direction read such a mixture backwards; it is the best match of
  its edge in 17 and 25 | 24 cases.
- Read from its two ends, the same edge pair gets two different similarities
  in 45 | 44 of 81,842 | 14,876 pairs (largest difference 0.966). With every
  mixture read through `flipped_copy` that number is 0: the unreversed
  mixtures are the whole of the disagreement, and `polymorphic_edge_similarity`
  itself is symmetric on every pair measured.
- The census also recomputed the scores with the mixtures flipped, offline,
  through the same similarity functions and the same correspondence. One-sided
  score: 19 | 18 of 300 scores move (largest 0.0011 | 0.0020), no answer, hits
  38 | 34 unchanged. Balanced score: 41 | 34 move (largest 0.0005 | 0.0011),
  1 | 0 answers, hits 41 | 39 unchanged.

So the switch run is a **confirmation**, not a discovery: I have seen its
result. It is recorded here as a prediction all the same, because it can
fail — if the library change and the census disagree, one of them is wrong.

**Form.** One helper in `memory/cores.py`, used by both edge loops where the
reversed index entry is built: a mixture is reversed with `flipped_copy`,
anything else with `reverse_edge_params` as before. One switch,
`EDGE_REVERSE_FLIPS_MIXTURE`, default `True`; `False` is the historical
reading. Nothing else changes: the correspondence does not read edge
parameters, and a graph without mixtures, or whose mixtures are symmetric
under reversal, is untouched.

**Prediction.** Under the switch, with every other switch as in F6+D1 and as
in D3 (all three terms): exactly the census figures above — the scores that
move, the largest difference, the answers, the hits. `matched_rel_sim_sum`
moves on 21 pairs per window and the query-side sum on 28 | 27. Every
single-prototype mixture then reads exactly as its plain prototype would,
from either end (pinned).

**Default rule, fixed now** (the Phase 4 rule: this is the library's own
contract, not an accuracy fix, and no gain is predicted or required). The
switch stays on if the pins stay green, the run reproduces the census
figures, and accuracy does not fall with paired p < 0.05 on either window
under either form. Otherwise it goes off and is reported as a negative.

**Known risk I cannot test here.** The loop is shared: `mcs_score` and
`mcs_similarity` read the same edge similarities, and they feed the fusion
and containment gates (`THETA_FUSE`, `mcs_contained`, `_classify`), whose
thresholds were set on the historical values. Single-view graphs carry few
multi-prototype mixtures (404 of 3,525 here); consolidated cores carry more,
so the change can be larger there than anything ETH-80 shows. The frozen
bars in this checkout do not call the matcher at all.

---

## Pre-registration — exact assignment (written before any score, accuracy or coverage under the switch was computed; one count was seen by accident, see below)

**The defect, already measured** (`results/phase2_correspondence_census.txt`,
`results/phase2_symmetry_probe.txt`). The node correspondence of all three
matcher scores comes from `_max_weight_bipartite_matching`
(`matching/structural_similarity.py`), which is not a maximum-weight
matching but an approximation of one: per-row top-8 pruning, a greedy pass,
three 2-opt passes. Its docstring says why: it replaced a pure-Python
Hungarian that was the dominant cost.

- It leaves a node unpaired while a partner with a positive weight is still
  free: 7.3 % | 5.4 % of the pairable node slots, in all 300 pairs of either
  window; worst TextureBoundary 22.2 %, Orientation 11.5 %.
- The pruning is per row of the matrix, so the correspondence of (Q, C) is
  not that of (C, Q): the same correspondence both ways in 0 of 48 probed
  pairs. This is why the balanced score is symmetric as a formula only.
- The total weight is below the maximum (on 300 random 1–6 × 1–6 matrices
  the approximation misses the optimum in 48; the exact solver equals brute
  force in all 300).

**Form.** `scipy.optimize.linear_sum_assignment` on the same weight matrix
(signature similarity × anchor factor; nothing about the weights changes),
solved in one orientation whatever the argument order, pairs of weight 0
dropped. scipy is already a dependency of the tree (`core/spatial.py`). The
assignment was never the cost: on a 100 × 120 matrix the exact solver takes
0.9 ms against 27 ms for the approximation, and 90 % of an `evidence_score`
call is the dense signature matrix (profiled). One switch,
`MATCH_ASSIGNMENT_EXACT`, default `True`; `False` is the approximation,
untouched. The switch sits in the function, so every caller follows it: the
three matcher scores, the consolidator, `matching/structural_similarity.py`
and the seed of the global alignment.

**Seen before this text was written — disclosed.** While the module was
being edited, a census process that was already running re-imported it for
its second window and so ran under the exact solver. One figure came out of
that: the number of verifiable core edges on v1 over the 300 pairs, 107,190
against 93,377 with the approximation. No score, no coverage value and no
answer was computed or looked at. The predictions below were not adjusted to
it, except that I can no longer claim not to know the sign of the change in
verifiable edges.

**What is measured**, on EVAL-75, both windows, every switch explicit, with
the declared defaults as the base (F6, D1, Phase 4 and the mixture reversal
on) and with D3's three terms two-sided on top:

1. Stranding, by the census's own definition. It must be 0.
2. The total weight of the assignment, exact against approximate, per pair.
   Exact must be ≥ approximate on every pair.
3. The symmetry probe: pairs with the same correspondence both ways (0 of 48
   now) and |score(Q, C) − score(C, Q)| under the balanced score (mean
   0.0040 | 0.0031 now).
4. What the newly paired nodes are worth: the similarity of the pairs kept,
   binary coverage against the count formula, verifiable edges, relational
   consistency — own class against wrong class.
5. Share of score variance explained by *which reference* against *own
   class* (the Audit B6 measure).
6. Accuracy per class, ties split, paired against the same configuration
   with the approximation.

**Predictions.**

- (1) 0 and (2) no pair below, by construction; a failure is a bug.
- (3) The correspondence becomes the same both ways in most pairs but not
  all (the weight matrix of (C, Q) is the transpose of that of (Q, C) only
  if the signature similarity is symmetric to the last bit, and ties are
  the solver's); |Δscore| falls by an order of magnitude.
- (4) Binary coverage rises to the count formula wherever every candidate
  has a positive weight, so the node term under binary credit becomes *more*
  of a count, not less. The newly paired nodes are the ones the
  approximation could not place: their best free partner had similarity
  about 0.50 against 0.88 for the pairs kept, so graded coverage rises by
  roughly half the stranded share and the mean similarity of kept pairs
  falls. More nodes are mapped, so more core edges are verifiable and the
  absence term falls; the relations between weakly matched nodes are
  mostly absent in the query, so relational consistency falls.
- (5), (6) No prediction of direction. The added credit is large for a
  small core against a large query and small the other way round, which is
  the reference-size effect D3 was aimed at; it can go either way.

**Default rule, fixed now.** This restores the contract of the function's
name; no gain is predicted or required. The switch stays on if the pins stay
green, (1) and (2) hold on every pair, and accuracy does not fall with
paired p < 0.05 on either window under either form. Otherwise it goes off
and is reported as a negative — and the stranding stays a known defect.

**Known risks I cannot test here.** Every score of the approximate
correspondence moves, for every caller: the fusion and containment gates,
`THETA_EVIDENCE_POS`, the per-class thresholds and U20 were set on the
approximation. Between equally good matchings the solver chooses; two
interchangeable nodes can swap between scipy versions (the version is now
recorded in every result).

---

## Pre-registration — the signature matrix computed in blocks (performance; written before the kernel was run on any real pair)

**Measured already.** One `evidence_score` call on a 701 × 753-node pair
takes 0.58 s. About 90 % of it is the dense node-similarity matrix of the
correspondence, built pair by pair in pure Python (`_sig_similarity` →
`TypedSignature.similarity` → `population_similarity` → `centered_cosine`);
the assignment is under 4 %. A window of 300 pairs takes about 110 s on the
two cores here, a stage about 40 minutes.

**Form.** One function, used by `_mcs_accumulate` where the matrix of a lens
type is built. For signatures on the strict population branch (both
CANONICAL, same class and relation, a population of the declared layout on
both sides) it evaluates `population_similarity` for the whole block with
numpy, doing **the same floating-point operations in the same order** as the
scalar code: the mean with the builtin `sum`, the centred products
accumulated bin by bin, one square root, one division, the clamp, the
weighted sum over the axes. No BLAS product and no `numpy.sum` — nothing
whose rounding could differ. Every other pair goes through `_sig_similarity`
unchanged. The anchor factor stays per pair (`math.hypot` and `math.exp`
have no bit-identical numpy form); each node's normalised position is
computed once instead of once per pair. Switch
`MATCH_SIGNATURE_MATRIX_BLOCKS`, default `True`; `False` is the per-pair
loop, untouched.

*Amended before the implementation was run:* the first draft of this
paragraph skipped the anchor factor where the similarity is exactly 0. It
is not skipped: with a non-finite anchor the per-pair product is 0 × NaN =
NaN, and skipping would have returned 0.

**Criterion, fixed now — equality, not closeness.**

1. On every lens-type block of every (query, reference) pair of EVAL-75, in
   both windows, every similarity and every weight equals the per-pair value
   bit for bit (`==` on the floats; bug hunt, section F).
2. Pins: the same equality on random signatures, the degenerate ones
   included (a population with no variance, a missing population, a wrong
   layout, mixed classes and relations, OPAQUE signatures, `None`).
3. Every stored record of stages 2 and 3 is reproduced with 0 differing
   values.

If a single value differs anywhere the switch goes off and the difference is
reported; no tolerance is granted afterwards. The speed is reported; beyond
"faster" it is not a criterion.

**Limits.** Equality is to the arithmetic of the interpreter in use: the
builtin `sum` is called for the means precisely so that the block follows
the scalar on any Python version. A signature that is not a plain
`TypedSignature`, or a similarity implementation other than the two the tree
ships, takes the per-pair path.

*Amended a second time, after the first implementation was profiled and
before any stage-4 record was measured. The criterion is unchanged; the
form grew by three things, all under the same switch.*

- **The bins path of the Gaussian kernel.** With the population block in
  place, 63 % of a call on the full window was `_similarity_gaussian` on
  signatures that carry no population (Position, Boundary, Orientation,
  Color, Texture, Onset: every node of the v1 window, which the first form
  did not speed up at all). It is now a second block. Subtraction, absolute
  value, the circular minimum, division, multiplication and addition are
  elementwise numpy; `** 2` and `exp` are **not** (numpy's square and
  exponential are not guaranteed to round as the C library's `pow` and
  `exp`), so both are applied element by element by the interpreter's own
  `pow` and `math.exp`. A signature with an ABSENT slot or a slot that is
  not a float, a class with its own slot rule (`REGION_SHAPE`) and a class
  without ranges stay per pair; so does `PEAK_CHAIN`, whose three-axis
  population against a four-axis layout takes the aligned comparison.
- **The anchor spacing** of the redundancy-aware edge mass
  (`_median_positive_nn`): an O(n²) Python loop over the anchor points, run
  for both graphs on every call — 16 % of what was left on the full window
  once the Gaussian block was in, by the profiler (which overstates a
  Python loop; by the clock it is 5.7 ms per graph against 1.3 ms with the
  kernel, stage 4 §2). numpy now
  finds, for each point, the candidates for its nearest neighbour (every
  point whose squared distance is within a relative 10⁻⁹ of the smallest);
  the distance of those candidates is computed by `math.hypot` on the same
  two differences the loop forms, and the minimum of those is taken. A
  candidate at or below the loop's 10⁻¹² guard sends that point back to the
  loop.
- **Size thresholds.** A numpy call has a fixed cost: on a 2 × 2 block of
  a population class the first implementation was 7 to 8 times *slower*
  than the loop it replaced, and 3 to 4 times on a 3 × 3 block; it broke
  even near 36 pairs. A block below 48 pairs (population), 16 pairs
  (Gaussian) or 24 points (spacing) is now left to the loop. Both paths
  return the same floats, so the thresholds are a matter of speed only.

Criterion 1 now also covers the spacing and the edge-weight table of every
EVAL graph (bug hunt, section F); criterion 2 the new kernels (the Gaussian
block on all 66 classes that register ranges; the spacing against the
function as it stood, copied into the pin); criterion 3 as written.

---

## Pre-registration — P4b, the count kept apart from the physics (the spec's second line; written after the stage-2 censuses and the code, before either switch existed)

**What is already known** (read-only: `results/phase4_edge_census.txt`,
`results/phase4_axis_census.txt`, the two slot tables in `core/schema.py`).

- The library holds two tables about edge slots and they disagree on four
  slots. `EDGE_UNIT_TABLE` declares 12 slots as categorical lookup keys. For
  8 of them `EDGE_SCHEMAS` agrees: they are identity slots, gated and never
  graded. The other four — `containment` on `CONTOUR_ORIENTATION`,
  `JUNCTION_ORIENTATION` and `REGION_CONTENT`, and `source` on `OCCLUSION` —
  are `linear` or ungated in `EDGE_SCHEMAS`. The three containment classes
  are also listed in `EDGE_INDEX_ONLY`: "membership is a 0/1 flag … there
  is NO graded quantity to populate".
- So in the matcher a flag pair reads 1.000 *as a similarity*, and
  `edge_measured_axes` counts the flag as a measured graded axis — against
  its own docstring ("categorical gates are not graded axes") — so the pair
  earns f(1) = 2 instead of f(0) = 1 in the edge mass of `mcs_score` and
  `mcs_similarity`.
- On EVAL (stage-2 census: approximate assignment, full window, own-class
  pairs) 1,758 of the 9,527 matched core edges are flag pairs: 18.5 % of the
  matches and **25.6 %** of the similarity sum (21.5 % on v1, all
  `REGION_CONTENT`). Stage 2 §2 quoted 2,179 and 31.7 %; that figure also
  held 421 `CURVE_CONTAINMENT` matches, which read 1.000 for a different
  reason (their two measured slots are constant) and are not flags. Both are
  now stated separately there.

**Form — two switches, because one of them cannot be measured here.**

- `core/schema.py::edge_graded_axes(ep1, ep2)`: the number of GRADED axes
  measured on both sides. Same schema: slots that are not UNKNOWN on either
  side, not identity slots and not lookup keys in the unit table.
  Cross-schema: the shared axes under the same conditions. A mixture: the
  best over its prototypes. No parameters on either side: 0.
- **`EVIDENCE_RELATION_COUNT_APART`** (`evidence_score`; default by the rule
  below). The relational term becomes *presence × physics*.
  - presence = verifiable edges that have a counterpart (best similarity
    > 0) / verifiable edges. A count; every relation is in it, flags and
    parameter-less relations included.
  - physics = mean similarity over the present pairs that have a graded axis
    in common; 1.0 when none has — nothing was measured, so nothing
    contradicts (`matching/fuzzy.py`'s convention for its
    `edge_coverage × edge_param_mean`).
  - Under D3's two-sided relation term both factors are summed over the two
    sides before dividing, as the term is now.
  - The `EVIDENCE_ONLY_EDGES` branch keeps RULE B16's reading untouched.
  - Identity: with `t` count-only pairs and `p` physical pairs of mean
    similarity `m` among `v` verifiable edges, the term goes from
    `(t + p·m) / v` to `(t + p)·m / v`. It can only stay or fall; it stays
    when there is no count-only pair, when `m` = 1, or when nothing physical
    was matched.
  - *Amended after the pins were written and triaged, before the switch was
    run on any EVAL pair:* the identity assumes every count-only pair reads
    exactly 1.0. That holds for a flag, a parameter-less pair and a
    categorical agreement — everything the censuses found on this corpus —
    but not in general. With the Phase-4 floors restored
    (`EDGE_UNMEASURED_IS_NOT_EVIDENCE = False`) an all-UNKNOWN pair reads
    0.2, is a counterpart with nothing measured, and becomes one full count:
    the term rises from 0.2 to 1.0 for a lone such pair (pinned). The pin
    triage found it: a Phase-4 pin failed under the new switch. "Present"
    is a best similarity above 0 — exactly the pairs the mean already
    credits; nothing is newly credited under the declared defaults. The
    registered check stands as written: on EVAL the term must rise on no
    pair.
  - *Added after the measurement (stage 4 §3):* there is a second setting
    in which a count-only pair does not read 1.0 — the specificity cap,
    under which a flag reads its class's specificity. The stored-record
    ledger found it on 5 pairs measured under the cap. Pinned. It does not
    touch the registered check, which is on the declared defaults, where
    the term rose on no pair.
- **`EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS`** (`edge_measured_axes`;
  default **`False`**). A lookup-key slot stops counting as a measured axis
  in f(n). It moves `mcs_score` and `mcs_similarity` only; `evidence_score`
  does not read that mass, so nothing in this harness can measure it. It is
  pinned on synthetic graphs, left off, and is your decision.

**What is measured for `EVIDENCE_RELATION_COUNT_APART`** — EVAL-75, both
windows, every switch explicit, on the declared defaults and with D3's three
terms on top; a step-0 census on the final tree first (read-only).

1. Per pair: verifiable, present, count-only and physical edges, the two
   factors, the term before and after. The term must rise on no pair.
2. Own-class against wrong-class means of presence, physics and the term.
3. Discrimination of the relational term inside a query: the share of the
   225 (own-class, wrong-class) comparisons per window that the own-class
   reference wins, ties counted half — before and after.
4. Variance of the score explained by *which reference* / by *own class*.
5. Accuracy per class, ties split, paired; pairs scoring exactly 0.

**Predictions.** The term falls on every pair that has both a flag match and
a physical match with similarity below 1 — most pairs on full; on v1
`REGION_CONTENT` is the only flag class. Scores fall with it, so more pairs
clamp to 0. No prediction for discrimination or accuracy: the pairs reading
1.000 are 31.7 % of the similarity sum in own-class pairs against 29.7 % in
wrong-class pairs on full, so the constant is nearly class-blind, and the
physics of matched pairs is only slightly better for the own class (0.68
against 0.67 with two measured axes).

**Default rule, fixed now.** The spec fixes the principle, not the formula,
and the formula is mine; so the rule is of D3's kind, not Phase 4's. The
switch goes on only if, on **both** windows on the declared defaults: the
pins are green, the term rises on no pair, the discrimination of the
relational term (measure 3) does not fall, and accuracy does not fall with
paired p < 0.05 under either form. Otherwise it stays off and is reported as
measured.

**What it cannot fix.** A relation whose *measured* slots are constant is
physics by declaration. `CURVE_CONTAINMENT` is the case on this corpus. That
is a writer matter (last section), not this switch's.

---

## Pre-registration — D2 prerequisites: the dark alignment does what its parameters say (written from the code; nothing run)

**Defects, by the path's own documentation.**

1. `MATCH_ALIGN_MAX_PAIR_EVALS` is documented as the "pair-evaluation budget
   for the refinement". `global_consistency_alignment` charges the seed's
   similarity matrix to it when it builds the matrix itself, and not when
   the matrix is supplied. The same inputs therefore give two answers by
   entry point; and on any pair with more than 200,000 candidate pairs —
   every ETH-80 pair has 528,000 to 645,000 — the route the matcher uses
   (`_global_correspondence`) returns the unrefined seed with
   `budget_exhausted = True`.
2. Each refinement round evaluates the relational agreement of all
   |A| × |B| pairs, including those of similarity 0, whose weight is 0
   whatever the agreement is.
3. `_global_correspondence` builds the dense matrix pair by pair in Python.
4. It weighs candidates by signature similarity alone; the live
   correspondence also uses the anchor-position factor. Turning the flag on
   therefore changes two things at once.

**Form.** (1) and (2): a keyword of `global_consistency_alignment`, fed from
a new parameter `MATCH_ALIGN_BUDGET_REFINEMENT_ONLY` (default `True`): the
seed is never charged; a pair of similarity 0 is skipped and not charged.
`False` is the historical accounting. (3): the block matrix of the section
above, passed as `sim=` — the same values. (4): parameter
`MATCH_GLOBAL_ANCHOR_PRIOR` (default `False`, as it is today): the seed and
the refinement weigh candidates by similarity × anchor factor, the live
path's weight. All of it is on the dark path:
`MATCH_GLOBAL_CONSISTENCY_ENABLED` stays `False` and D2 stays gated.

**What is checked.** Pins: the two entry points return the same alignment; a
pair of similarity 0 is never evaluated and never matched; the historical
accounting reproduces the old behaviour, budget exhaustion included; with
one round and the anchor prior the dark path returns the live
correspondence. A mechanism check on 8 EVAL pairs: rounds used, budget
state and seconds per pair under both accountings — **no score and no
accuracy is computed**; that would be D2 itself. Continuity: no stored
value moves. *(As run: 32 pairs per window — the first two EVAL queries of
each class against the four references — instead of 8; nothing else
changed.)*

**What D2 has to register when its gate opens** (not decided here): the
seed (with or without the anchor prior), the systematicity weight, the
unmatched cost (the graded-admission experiment), and whether the cost per
pair measured above is affordable.

---

## Pre-registration — the specificity cap on same-schema pairs (a documented mechanism that is dead code; written from the code and the stage-2 census, before the switch existed)

**What the code says and does.** `core/schema.py` documents "[P2] EDGE
SPECIFICITY — the MATCHING-TIME cap on the generic edges": the similarity of
two generic proximity edges (`CONTACT`, `SURFACE_GROUPING`, `CO_LOCATED`,
`TEXTURE_BOUNDARY`, `CONTOUR_ORIENTATION`) is to be multiplied by the
class's measured specificity (0.282 for `CO_LOCATED`, 0.576 for
`CONTOUR_ORIENTATION`), and `EdgeParams.similarity` says "the cap lives at
the comparison site (`polymorphic_edge_similarity`), not here". In
`polymorphic_edge_similarity` the cap is applied in the cross-schema branch
only, after the same-schema branch has returned. No generic class has a
cross-schema partner (`EDGE_FAMILY_COMPATIBILITY` lists seven classes, none
of them generic), so the factor is 1.0 wherever it is reached: **the cap
never acts in the matcher.** Its only live use is the mass of a relational
memory entry (`memory/relational.py`).

**Why I do not call it a bug to be fixed by default.** A later note in the
same library (`memory/cores.py`, "[U32] effective weights are near-flat by
necessity") records that deflating the low-information edge mass was tried
and inverted the winner on the author's own benchmark. The two texts cannot
both describe the intended matcher, the specificity table was measured by a
script that is not in this checkout, and `CO_LOCATED` is 26 % of the edges
here. Which one stands is your decision.

**Form.** `EDGE_SPECIFICITY_CAP_SAME_SCHEMA`, default **`False`** (today's
behaviour) and not subject to any default rule: under it the same-schema
branch multiplies by `edge_match_weight` of the class, as the cross-schema
branch does.

**What is measured — descriptive only.** EVAL-75, both windows, on the
declared defaults and with D3's terms on top: the similarity sum per
relation, the relational term own-class against wrong-class and its
discrimination, the variance shares, accuracy paired. It answers README
question 4 for this harness: what the generic edges' full-strength matches
are worth to recognition. Whatever comes out, the default stays `False`
here.

**Prediction.** The relational term falls on every pair with a matched
generic edge; `CO_LOCATED` and `CONTOUR_ORIENTATION` matches were 34 % of
the own-class similarity sum on full in the stage-2 census. If README
retraction 1 ("CO_LOCATED removal costs 8.0 pts") carries over to this
harness, accuracy falls.

---

## Phase 0 — summary (frozen; full tables in `results/phase0/`)

Baseline: 36/75 = 48.0 % / 34/75 = 45.3 % (apple 2 and 0 of 16). No ties;
scores <= 0.16; median winner-minus-runner-up 0.030.

| finding | reading |
|---|---|
| F1 edges act as a count | 70.5 % of verifiable own-class core edges find **no query edge** between the mapped endpoints. Where an edge is matched it *is* graded here (61 % below 1.0) — unlike the atom-bin harness. |
| F2 node credit is binary | True by code. The graded quantity is collinear with it (r = 0.997). D1 confirms. |
| F3 populations cannot move the score | True for V1-tier nodes (they carry none). Edge populations do move it. `nopop` arm 34 / 32, not significant. |
| F4 no query-side term | The largest effect measured. Removing each reference's own offset (transductive, observable only) recovers 4 / 12 queries on the evidence score and 15 / 31 on the node term. |
| F5 position is a hidden prior | Removing the anchor factor costs nothing (36 / 39). Scrambling it costs 2.5-3 queries (full, n.s.) and 5-6 (v1), ties split. |
| F6 `polymorphic_edge_similarity` | 14 own-class edges. |
| Audit A — CO_LOCATED | `scale_ratio` is UNKNOWN on all 17,516 edges (`Detection.scale == 0`); `salience_ratio` restates node salience (re-derived exactly on 79/79 images). Removal arm 34 / 34: the README's -8 pts is not seen on this harness. |
| Audit C — anchor shuffle | accuracy 33.0-33.6 / 27.9-29.3 (ties split, 5 seeds, two shuffles); own-class mean score falls by 70-90 %. Never quote a shuffle accuracy without the tie-split figure: the score clamps to 0.0 for all references on many queries. |

---

## Port manifest (cumulative) — on hold until you say

### (b) Library source — port these

Seven files, one of them new: +1,530 / −34 lines against the upload
(`bd27128`). Every behaviour change sits behind one general parameter; the
two changes without a switch move no score (trace and validation only).
Line counts are per change; `patch` is the file in `diagnostics/patches/`
that carries the change together with its switch and its pin.

| file | change | +/− | switch (default) | patch |
|---|---|---|---|---|
| `parameters/_text.py` | a header and 13 switches after `MATCH_SEARCH_MAX_OPS` (CRLF kept) | +153 | — | each patch its own lines |
| `core/schema.py` | F6: `polymorphic_edge_similarity`, the parameter-less branch | +23 / −2 | `EDGE_UNPARAMETERISED_STRUCTURAL_MATCH` (`True`) | 0001 |
| `core/schema.py` | Phase 4: `EdgeParams.similarity` and the cross-schema mean | +32 / −1 | `EDGE_UNMEASURED_IS_NOT_EVIDENCE` (`True`) | 0004 |
| `memory/cores.py` | D1: `_mcs_accumulate(graded_node_credit=…)`, matched-mass loops, trace | +64 / −8 | `EVIDENCE_GRADED_NODE_CREDIT` (`True`) | 0002 |
| `memory/cores.py` | D3: `_edge_side_evidence`, `_query_side_terms`, balanced terms in `evidence_score` | +155 / −2 | `EVIDENCE_QUERY_SIDE_TERMS` (`()`) | 0003 |
| `memory/cores.py` | null calibration: `_chance_credit`, chance masses, calibrated node term | +141 / −5 | `EVIDENCE_NODE_NULL_CALIBRATED` (`False`) | 0005 |
| `memory/cores.py` | trace: `node_coverage_binary` is the binary twin; docstring of the trace keys | +27 / −6 | none — trace only | 0007 |
| `memory/cores.py` | D3 terms read and validated first; dropped under `EVIDENCE_ONLY_EDGES` | +21 / −8 | none — no score changes | 0008 |
| `memory/cores.py` | mixture reversal: `_reverse_for_traversal`, used by both edge loops | +30 / −2 | `EDGE_REVERSE_FLIPS_MIXTURE` (`True`) | 0010 |
| `matching/structural_similarity.py` | exact assignment: `_exact_max_weight_matching` and the branch in `_max_weight_bipartite_matching` (CRLF kept) | +60 | `MATCH_ASSIGNMENT_EXACT` (`True`) | 0011 |
| `matching/signature_matrix.py` (**new**) | kernels: the population block, the Gaussian bins block, the size thresholds | +367 | `MATCH_SIGNATURE_MATRIX_BLOCKS` (`True`) | 0013 |
| `memory/cores.py` | kernels: the block path of `_mcs_accumulate`, `_normalised_anchors`, `_anchor_weights`; the nearest-neighbour kernel of `_median_positive_nn` | +202 / −15 | the same switch | 0013 |
| `core/schema.py` | P4b: `edge_graded_axes` and the lookup-key slots; `edge_measured_axes` under its switch | +102 | `EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS` (`False`) | 0014 |
| `memory/cores.py` | P4b: presence and physics traced by both edge loops; the term as their product in `evidence_score` | +100 / −5 | `EVIDENCE_RELATION_COUNT_APART` (`False`) | 0014 |
| `matching/alignment.py`, `matching/fuzzy.py` | D2 prerequisites: the budget counts the refinement only; zero-similarity pairs skipped; seed evaluations passed in | +37 / −2 | `MATCH_ALIGN_BUDGET_REFINEMENT_ONLY` (`True`) | 0015 |
| `memory/cores.py` | D2 prerequisites: `_global_correspondence` hands over the block matrix; optional anchor prior | +25 | `MATCH_GLOBAL_ANCHOR_PRIOR` (`False`) | 0015 |
| `core/schema.py` | the specificity cap in the same-schema branch of `polymorphic_edge_similarity` | +14 / −1 | `EDGE_SPECIFICITY_CAP_SAME_SCHEMA` (`False`) | 0016 |

Pins that travel with (b): `tests_matcher/`, 20 files, 390 tests — patches
0001–0016 (0006, 0009 and 0012 are tests only). Dependencies: numpy was
already imported by `matching/structural_similarity.py`; scipy
(`scipy.optimize.linear_sum_assignment`, the exact assignment) is new to
the matcher and was already used by the front end. The repository pins
neither; every stored result records the versions it was computed under.

**Read before porting.**

- *On by default and able to move a score outside `evidence_score`.* F6,
  the mixture reversal and the exact assignment act in the edge loop or
  the correspondence that `mcs_score` and `mcs_similarity` share with
  `evidence_score`, so they change scores that feed gates I cannot run
  here. Phase 4 sits in the same shared code; it moved no value on this
  corpus and can on one that has all-UNKNOWN edge pairs. Each is one
  parameter away from the historical behaviour.
- *On by default, moves no value.* The kernels (bit for bit, on every
  entry point) and the alignment's budget accounting (dark path only:
  nothing calls it while `MATCH_GLOBAL_CONSISTENCY_ENABLED` is `False` —
  but whoever turns that flag on now gets an alignment that refines, which
  the historical accounting never did on graphs of this size).
- *On by default, `evidence_score` only.* D1.
- *Off; nothing changes unless switched on.* D3, the null calibration,
  P4b (both switches), the anchor prior, the specificity cap.

**Decisions that are yours.** `EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS`
(moves `mcs_score` only; unmeasurable here); `EDGE_SPECIFICITY_CAP_SAME_SCHEMA`
(which of the library's two texts stands; stage 4 §6); the D2 gate; the
`CURVE_CONTAINMENT` writer (two measured slots that never vary); which
configurations to score on HELDOUT-75.

**Commits.** The work is on the branch `arena/8ef05428-percept4`. Two
set-up commits come first: `.gitignore`, and the import shims of
`diagnostics/_export_shims/`, without which `matching/` and `memory/`
cannot be imported in this export checkout (`sensory_floor` is not in the
export). The next 16 commits are the library changes, one per change, each
with its switch and its pins; the files in `diagnostics/patches/0001…0016`
are those commits exported (`git format-patch -k --zero-commit`). Applied to a fresh
clone of `bd27128` with `git am -k --keep-cr` they reproduce the seven
files and the pins byte for byte, and the pins are green at each of the 16
commits (11, 20, 32, 41, 56, 57, 80, 85, 105, 120, 129, 135, 343, 367, 385,
390 tests, each in a fresh checkout). The commits after them add the seven
uploaded figground files, the harness, the stored results and this report,
and touch no library file. (Earlier in the session local commits were lost
when the sandbox was rebuilt; the hashes quoted in earlier reports name
nothing any more.)

### (a) Diagnostics / harness — not library

`.gitignore`; everything under `diagnostics/` (see `diagnostics/README.md`
for the map): the harness (`_bootstrap.py`, `_stats.py`, `_export_shims/`,
`eth80_*.py`, `eth80_split.json`), the phase scripts (`phase0_audit_*.py`,
`phase1_f6_d1.py`, `phase4_*_census.py`, `phase2_*.py`, `phase3_*.py`),
`_param_overrides_plugin.py`, `phase2_pin_triage_classes.json`, `patches/`,
`results/`.

Copied unmodified from your uploads into `tests_figground/`: `truth.py`,
`test_junction_wiring.py`, `test_occlusion_order_edges.py`,
`test_ownership_decoder.py`, `test_renderer_truth.py`,
`test_ring_closure.py`, `test_t_junction_interrupted.py`.

---

## Pin triage — every failing assertion classified, bugs fixed first

**Why the earlier answer was not enough.** After stage 2 the matcher pins
failed when run under another switch value (7 under D3, 12 under the null
calibration, their union under both), and I answered by giving every pin a
stated baseline (`conftest.PIN_BASELINE`). That made them green without
saying, pin by pin, whether the failure was the pin or the code. It also
made "the matcher pins are green under override X" as empty as the figground
bar: with the baseline applied, an override never reaches them.

**Instrument** (`diagnostics/phase2_pin_triage.py`). pytest stops a test at
its first failing assertion, so "7 pins fail" hides what follows it. The tool
writes soft copies of the pins (every `assert` records and continues), runs
them with the baseline disabled under one changed switch, and lists every
failing assertion with its class from
`diagnostics/phase2_pin_triage_classes.json`. An unclassified assertion is an
error. For each class S entry the value the matcher returns under the other
setting was derived by hand and is now asserted by a pin that sets every
switch itself (`tests_matcher/test_switch_interactions.py`); for the node
term the derivation is also mechanical — `tests_matcher/_oracle.py` writes
the term from its definition, with the chance level by exhaustive
enumeration of every maximum pairing, and the matcher equals it under all 12
combinations of graded/binary × one-sided/balanced × calibrated or not on 77
scenarios.

**Before the fixes** (the 57 original pins, pre-fix library;
`results/phase2_pin_triage_before.txt`). Tests failing / distinct failing
assertions, by class:

| setting | tests | assertions | B bug | S old semantics | G guard | X by design |
|---|---|---|---|---|---|---|
| D3, node term only | 7 + 1 | 13 | **2** | 10 | – | 1 |
| D3, all three terms | 7 + 1 | 13 | **2** | 10 | – | 1 |
| null calibration | 13 + 1 | 26 | – | 25 | – | 1 |
| D3 + null calibration | 16 + 1 | 32 | **1** | 30 | – | 1 |
| D1 off | 16 + 1 | 35 | – | 33 | 1 | 1 |
| F6 off | 8 + 1 | 17 | – | 13 | 2 | 1 (+1 raised after a soft failure) |
| Phase 4 off | 6 + 1 | 12 | – | 11 | – | 1 |

("+ 1" is `test_declared_defaults`, which reddens on any default flip by
design. "12 under the null calibration" was counted before I added one more
null pin; with it, 13.)

**The 7 pins under D3.**

| pin | assertion | class | what the matcher returns under D3, derived |
|---|---|---|---|
| D1 `binary_credit_tied_them…` | `node_coverage == node_coverage_binary` | **B** | 6/(6+ε) against 3/(3+ε): the binary key kept the one-sided form while the term was balanced |
| D1 `graded_coverage_never_exceeds_binary…` | `node_coverage <= node_coverage_binary` | **B** | false in 10 of 40 random pairs, for the same reason |
| the same pin | value of the one-sided fraction | S | 2 Σ sims / (n_q + n_c) on every pair |
| D1 `node_weight_and_lens_boost…` | one-sided weighted mean | S | (1.25 s₁ + 3b s₂) / (3.25 + 3b) = 0.92488, weights on the core only |
| D1 `an_unmatched_core_node…` | 1/2 | S | (1 + 1)/(1 + 2) = 2/3; the one-sided `core_node_coverage` is still 1/2 |
| null `chance_is_the_mean…` | one-sided N0, N, calibrated | S | N0 = 2·total/15 = 0.28575, N = 0.73114, calibrated 0.62358 |
| null `chance_carries_the_lens_boost` | one-sided N0 | S | 2 (tex + b·con) / (q total + c total) = 0.43778 |
| null `chance_carries_the_node_weights` | one-sided N0 | S | (q chance + c chance) / (4.0 + 2.5) = 0.55261 |

**The 13 pins under the null calibration** — all S.

- Five D1 pins and three D3 pins state the uncalibrated value. Calibrated,
  the matcher returns the derived one: e.g. 0.91644 = (N − N0)/(1 − N0) with
  N0 the mean of the 3 × 3 candidate matrix; 0.38883 for D3's fragment
  (N = 6/9, N0 = ΣS/27); and **0** wherever a lens type has a single
  candidate on each side, because the only pairing is the random pairing.
- Five of the null pins' own "off" legs did not state the switch, so they
  ran calibrated.

**D1 off, F6 off, Phase 4 off.** The same two kinds: a pin of the switch's
own subject that did not state it, or a scenario built on another change
(D3's relations are parameter-less edges, which only F6 matches). Three
assertions are guards on the scenario (a non-degeneracy check such as
`0 < score`), where the claim under test still holds.

**One implementation bug, found twice, fixed first.**
`trace["node_coverage_binary"]` — D1's "binary coverage of the same
correspondence (== node_coverage when graded credit is off)" — was left as
the one-sided core fraction when D3 made the term balanced and the null
calibration made it calibrated. The comment was false and D1's invariant
"graded never exceeds binary" with it. No score reads the key, and the
harness had worked round it, so no reported number changes; but on the
stored EVAL records 3,576 of 3,600 balanced pairs carried a twin of the
wrong form and in 727 the term exceeded it. Fix: the key is the binary twin
of the term in use, never calibrated, so in every setting
`node_coverage ≤ node_coverage_raw ≤ node_coverage_binary ≤ 1` and under
binary credit `node_coverage_raw` is the twin bit for bit
(`tests_matcher/test_trace_binary_twin.py`; trace only, no switch).

**Decisions on the semantic mismatches.**

| mismatch | decision |
|---|---|
| the binary twin's form and calibration | **adjust the implementation** (above) |
| a pin states the one-sided / uncalibrated / graded / F6 value and the other setting gives another | **update the pins**: the assertion is right for its setting and stays; the setting is stated (baseline); the value under the other setting is pinned in `test_switch_interactions.py` |
| a pin of a switch's own subject does not state that switch | **update the pins**: stated by the baseline |
| guards | nothing to decide; listed |
| D3 under `EVIDENCE_ONLY_EDGES` — the pre-registration addendum said it was pinned and it was not | **adjust the implementation and add the pin**: the score was as stated, the trace was not (the mirror loop ran for nothing and wrote a balanced relation the branch never reads); the names that cannot act are now dropped before any work, and a mistyped name is an error on the early exits too (`tests_matcher/test_d3_only_edges_branch.py`) |
| binary credit × calibration is 0 for every pair | **keep**, it is the registered mechanism check; now said in the docstring and pinned as "the calibration presupposes graded credit" |
| one edgeless side breaks the balanced score's symmetry (1.0 against 0.0) | **keep**, it is the registered "edgeless core is vacuous"; the symmetry claim is now stated with its exception and pinned |

**After the fixes** (stage 3, 129 pins): the class B assertions no longer
fail under any setting; every remaining failing assertion is S, G or X;
none is unclassified. The pins are run three ways and all three have to
hold: with the baseline, the triage (0 unclassified), and the interaction
pins, which set every switch themselves.

**After stage 4** (`results/phase2_pin_triage.txt`, all **390** pins, 17
settings — the seven above, the two correctness switches, and the six
stage-4 switches each at its other value). Tests with a failing assertion /
distinct failing assertions, by class:

| setting | tests | assertions | S | G | X | other |
|---|---|---|---|---|---|---|
| declared defaults | 0 | 0 | – | – | – | – |
| D3, node term only | 6 + 1 | 11 | 10 | – | 1 | – |
| D3, all three terms | 6 + 1 | 11 | 10 | – | 1 | – |
| null calibration | 13 + 1 | 26 | 25 | – | 1 | – |
| D3 (either form) + null calibration | 16 + 1 | 31 | 30 | – | 1 | – |
| D1 off | 16 + 1 | 35 | 33 | 1 | 1 | – |
| F6 off | 8 + 1 | 17 | 13 | 2 | 1 | 1 raised after a soft failure |
| Phase 4 off | 6 + 1 | 12 | 11 | – | 1 | – |
| mixture reversal off | 0 + 1 | 1 | – | – | 1 | – |
| exact assignment off | 0 + 1 | 1 | – | – | 1 | – |
| kernels off | 0 + 1 | 1 | – | – | 1 | – |
| P4b on | 1 + 1 | 2 | 1 | – | 1 | – |
| lookup key on | 0 + 1 | 1 | – | – | 1 | – |
| budget accounting historical | 0 + 1 | 1 | – | – | 1 | – |
| anchor prior on | 0 + 1 | 1 | – | – | 1 | – |
| specificity cap on | 12 + 1 | 17 | 13 | 3 | 1 | – |

No class B assertion, none unclassified. Three things the triage found in
stage 4, each before the switch was run on EVAL:

- **P4b's "the term can only stay or fall" was false as written.** A
  Phase-4 pin failed under the switch: with the floors restored an
  all-UNKNOWN pair reads 0.2 in the mean and is a full count once the count
  is kept apart. The pre-registration, the code comment and the parameter
  comment were amended, and the case is pinned.
- **The new pins leaned on the baseline.** Twenty assertions of the
  stage-4 pins were unclassified at first because they relied on D3's
  terms, F6 or the cap being at their baseline value without saying so.
  Each now states the switches its numbers depend on.
- **Under the cap a graph no longer scores 1.0 against itself.** Thirteen
  assertions that expect 1.0 (or a value built from it) for an identical
  generic edge read the class's specificity instead. They state the
  uncapped matcher and stay; the capped values are pinned in
  `test_specificity_cap.py`. It is the cost of the cap in one sentence,
  and part of why it is not a default.

---

## Bug hunt and design changes — one list, all stages

### The hunt on the final tree: 0 findings

`diagnostics/phase2_bug_hunt.py` checks identities that must hold if the
code does what its comments, its pins and this report say. A failing check
is a finding; a passing one bounds where a bug can still be
(`results/phase2_bug_hunt.txt`).

| section | what is checked | on the final tree |
|---|---|---|
| A ledger | every stored pair of 46 stored configurations (13,800 pairs) recomputed from its own stored components: 18 identities (the node term's form, calibration and chance, the relational term in both forms, absence, the score), 12 invariants, the answers | 0 fail |
| B switch independence | what a score-level switch must not move, across the stored configurations: 111,600 values | 0 differ |
| C graph storage | how the 79 graphs store undirected edges and their parameters: orientation, key, mixtures | as the loops assume; 198 undirected mixtures per window that a reversal changes — handled in the loops since stage 3 |
| D walk | the matcher on all 300 pairs per window: the mirror loop against the matcher's own; 90,562 \| 16,429 edge pairs read from both ends; the correspondence (179,474 \| 76,844 kept pairs: one-to-one, none across lens types, none of similarity 0); traced against untraced; the chance masses against 2,000 sampled pairings | 0 differ; no \|z\| above 3 |
| E order dependence | edge-candidate ties whose resolution would depend on set order | 0 ties in 184,307 \| 107,190 verifiable edges; `PYTHONHASHSEED=1` reproduces 112,200 of 112,200 values |
| F kernels | every matrix entry, anchor weight, spacing and edge-weight table against the Python loops | 0 of 14,989,430 \| 5,854,028 |

What the hunt does not cover: `mcs_score` and `mcs_similarity` as called by
the rest of the library (they share the loop and the correspondence; their
callers and gates are not in this harness), the front end, and the dark
alignment beyond its pins and the mechanism check.

### Defects found, by where they were

**In the library as uploaded** — defects, and two gaps the spec names (2,
10); each behind a switch:

| # | defect | how it was found | what was done | stage |
|---|---|---|---|---|
| 1 | two parameter-less edges of one predicate scored 0.0; the fix existed as a comment | Phase 0 census (14 own-class edges) | F6, on | 1 |
| 2 | node credit is all-or-nothing: any kept pair earns its full mass (the spec's D1) | Audit B measured it | D1, graded credit, on | 1 |
| 3 | an all-UNKNOWN edge pair read 0.2 and a pair with no commonly measured axis 1.0, against "[U32] all-UNKNOWN is not evidence" | reading the code against its comment; censuses | Phase 4, on; unexercised on this corpus | 2 |
| 4 | `MATCH_NULL_CALIBRATED`'s description says `evidence_score` calibrates node coverage; it did not | reading the code | null calibration implemented, off (its rule failed) | 2 |
| 5 | a mixture read against its stored direction was not reversed: the same edge pair read two ways from its two ends on 45 \| 44 pairs | bug hunt, section D | mixture reversal, on | 3 |
| 6 | the assignment was an approximation: 7.3 % \| 5.4 % of pairable node slots stranded, below the optimum on 300 of 300 pairs, a different correspondence for (Q, C) and (C, Q) | correspondence census, symmetry probe | exact assignment, on | 3 |
| 7 | the dark alignment charged its seed to the refinement's budget: on a full-window pair it never refined; it also evaluated pairs of similarity 0 and built its matrix pair by pair | reading the code (stage 1), measured in stage 4 | budget accounting, on (dark path) | 4 |
| 8 | the dark alignment weighs candidates without the anchor factor the live path uses | reading the code | anchor prior, off (as built) | 4 |
| 9 | the documented specificity cap is unreachable: applied only in a branch no generic edge can enter | reading the code against its comment | wired behind a switch, off; your decision | 4 |
| 10 | two slot tables disagree on four slots; a membership flag is scored as a measured graded axis and reads 1.000 as a similarity — a quarter of the own-class similarity sum on full (the spec's "count may not masquerade as physics") | step-0 censuses | P4b, off (its rule failed by one comparison); the lookup-key switch, off, unmeasurable here | 4 |
| 11 | the node-similarity matrix and the anchor spacing were Python loops: 0.61 s per full-window call | profiling | numpy kernels, on; bit for bit | 4 |

Not fixed, reported: `CURVE_CONTAINMENT` writes two measured slots that are
constant over the corpus, so every such pair reads 1.000 as physics (421
own-class matches, 6.1 % of the similarity sum) — a writer matter;
`PEAK_CHAIN` writes three population axes against a four-axis layout; the
node assignment has several optima of equal weight on 7 of 32 sampled
full-window pairs, and which one is returned is an accident of node order.

**In my own changes** (found by the pin triage or the hunt, fixed before the
figures were written):

| defect | found by | fix |
|---|---|---|
| `trace["node_coverage_binary"]` kept the one-sided form when D3 made the term balanced: 3,576 of 3,600 balanced stored pairs carried a twin of the wrong form | pin triage (class B, twice) | the key is the binary twin of the term in use; trace only, no score read it |
| D3 under `EVIDENCE_ONLY_EDGES`: the score was as stated, the trace was not, and a mistyped term name was not an error on the early exits | writing the pin the pre-registration said existed | terms validated first; the names that cannot act are dropped before any work |
| P4b's "can only stay or fall" does not hold where a count-only pair reads below 1.0 (the Phase-4 floors restored; the specificity cap) | pin triage; the stored-record ledger | statement amended; both cases pinned |
| the first form of the kernels was slower than the loop on small blocks (8 times on a 2 × 2 block), and left every bins-only lens type — all of v1 — per pair | timing before measuring | size thresholds; the Gaussian block |
| two holes in the kernels' pins | mutation of the kernels | two pins added |

**In the instruments** (each produced a false reading; none changed a
reported figure, because each was explained before the figure was written):

| false reading | cause | fix |
|---|---|---|
| "NOT REPRODUCED — 247 of 300 scores moved" (mixture reversal, full) | `run_cached` returned full precision for a fresh record and 10 significant digits for a stored one; the script compared one of each | the harness always returns the stored form |
| chain stopped on a breakdown assertion | the edge breakdown did not follow the matcher's loop after the reversal fix | breakdown and census follow the loop |
| bug hunt "C = 2" | a storage fact (198 mixtures) counted as a violation after the loops stopped relying on it | counted only when the switch is off |
| "0 of 79 graphs compared" | a digest prefix passed where the cache is keyed by the full digest | prefixes are resolved or refused; a missing entry is reported as "not a comparison" |
| bug hunt "A = 2" | an absolute tolerance of 10⁻⁹ on sums of the order of 100 stored at 10 digits; an invariant applied where its premise does not hold | tolerance scaled to the stored precision; the premise is checked |
| the pins were green under every override | the baseline that made them independent of the defaults also made an override unable to reach them | the triage runs them with the baseline disabled |

### Design changes — what behaves differently by default

| change | what a caller sees | reach |
|---|---|---|
| F6 | an edge with no parameters on either side matches as structure (1.0) instead of 0.0 | every entry point |
| D1 | a matched node earns its similarity instead of its full mass | `evidence_score` only |
| Phase 4 | an edge pair with nothing measured on both sides is 0, not 0.2 / 1.0 | every entry point; no value moved on this corpus |
| mixture reversal | a mixture traversed backward is reversed like a plain tuple | every entry point |
| exact assignment | the correspondence is the maximum-weight one; `score(Q, C)` and `score(C, Q)` use the same pairs | every entry point, and the other callers of `_max_weight_bipartite_matching` (the consolidator, the structural similarity, the seed of the global alignment) |
| numpy kernels | nothing; a call is 4 to 8 times faster | every entry point |
| alignment budget | nothing while `MATCH_GLOBAL_CONSISTENCY_ENABLED` is off; with it on, the alignment refines | dark path |

Each is one parameter from the historical behaviour, and
`tests_matcher/test_declared_defaults.py` reddens when one is flipped. The
first, the third, the fourth and the fifth move scores that feed gates
outside this harness; that is the part of the port I cannot check for you.

### Decisions I took that you may want to reverse

- **Two rules were applied as written although the measurement was
  noise-level either way.** D3 is off because the reference share of the
  variance did not fall on full (10.1 → 10.5 %); P4b is off because the
  term's discrimination fell by one comparison of 225. A looser rule would
  have turned both on, and I do not think the data support that either.
- **Two correctness fixes are on although neither was required to gain
  accuracy** (mixture reversal, exact assignment). Their rule was "no
  significant loss", and the exact assignment's cost under D3 on full
  (41 → 37, +2/−6, p = 0.29) is inside it.
- **The kernels are on** on the strength of equality alone. If a platform's
  C library rounds `pow` or `exp` differently from the one here, the block
  still equals the loop *on that platform* (both call the same functions);
  what it will not equal is a record computed elsewhere.
- **HELDOUT-75 has not been opened.** Which configurations to score on it,
  once, is yours to say.
