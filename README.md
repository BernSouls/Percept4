# Take25/work — external-analysis export

This folder is a **self-contained copy** for an outside code reviewer.
It was assembled on 2026-11-10 from `E:\Take25\work`. Nothing here was modified —
all files are verbatim copies (SHA256-verified where noted) of the working tree.

## Layout

```
src_flat/
  <every folder of Take25/work/src>   93 .py files, verbatim (no .pyc, no __pycache__)
  _u27_selftest.py (left in place; top-level src files were NOT copied — see README note)
  export/
    README.md                         this file
    diagnostics/                      benchmark + audit harnesses and their measured outputs
    tests_figground/                  renderer-truth infrastructure + the junction-pin suites
    images/<class>/                   15 curated ETH-80 images (references + failure cases)
```

`src_flat` root contains the \*.py source tree copied 1:1 from `Take25/work/src`
(11 folders). Only the top-level `src\*.py` files were deliberately **not** copied
(the reviewer wanted only the folders + package files; 20 top-level .py files were
removed by the user after copying). Everything in `export/` is an addition.

## What you can do with this set

- Run the ETH-80 benchmark harness:
  `cd export/diagnostics && PYTHONPATH=../../src python _u32_eth80_full_benchmark.py --config baseline`
  (uses cached subgraphs if present; rebuilds otherwise).
- Run figground pin suites:
  `pytest` over `export/tests_figground/` (run from the work root with the same
  PYTHONPATH; scenes.py builds scenes in memory — no image files needed).
- Reproduce the occlusion-gate recall/precision:
  `python _u32_occlusion_recall_measure.py`

## Measured results (read this before re-running anything)

All numbers are measured on this codebase, in this tree, with the exact harnesses
in `export/diagnostics/`. **Do not re-derive claims that are already refuted.**

### ETH-80 full-split (75 queries; refs: apple1-000-000, car1-000-000, cup_01, horse_03)
| config | acc | per-class | margin |
|---|---|---|---|
| baseline | **51/75 = 68.0%** | apple 6/16 · car 18/24 · cup 17/18 · horse 10/17 | +0.0101 |
| noeff_color (color boost = 0) | 49/75 = 65.3% | 6/16 · 18/24 · 17/18 · 8/17 | +0.0105 |
| noedge_col (CO_LOCATED edges removed) | 45/75 = 60.0% | 10/16 · 10/24 · 18/18 · 7/17 | +0.0135 |

Determinism verified (full-split margins rerun and reproduce exactly).
The 75-query protocol vs alphabetical-first reference is an **in-stent** benchmark:
the 16-image tuning subsample overlaps 11/15 images with the eval pool — nothing in
this document is a held-out estimate. Full-corpus LOO 1-NN was never run (it timed
out on this machine); the subsample's LOO is 15/15 but the subsample saturates, so
it is uninformative.

### Edge-evidence decomposition (bug hunt arms A/B)
- Every matched edge in all 300 (query, ref) pairs has `best_sim == 1.0` exactly
  (`matched_rel_sim_sum == n_matched_edges`); `mean_coverage_axes_matched = 2.0` for
  every row. The edge layer contributes a **count** (co-assignment fraction), no
  graded geometry.
- CO_LOCATED removal costs 8.0 pts; color-boost removal costs 2.7 pts.
- Remove-arm = argmax over node_coverage, reconstructs exactly the measured 45/75
  (verified against the measured file).
- Zero-arm (CO_LOCATED similarity := 0, not the remove-arm) is degenerate
  (score≡0 → argmax ties → 16/75, all-apple). Multiplier is multiplicative
  (alpha = 0, `MATCH_NULL_CALIBRATED=True`).
- Partial correlations: `r(n_matched, score | node_coverage) = 0.85`,
  `r(rc, score | nc) = 0.78`; `r(n_verifiable, nc) = 0.96`, `r(n_matched, nc) = 0.105`.
- Correct refs beat wrong refs on n_matched for every class (car 78.1 vs 63.1;
  apple 89.6 vs 73.9; cup 93.7 vs 58.2; horse 66.1 vs 58.9).

### Population coding (arm C)
- In atom-bin wiring (the 68% headline): populations absent by construction
  (TypedSignature got `population=()`); only `COLOR_CONTENT` is even registered.
- In the structural (settle-loop) path: 256/467 matched pairs (54.8%) carry
  populations on both sides; mean full_sim 0.796 vs bins-only 0.750; ablations
  (`POPULATION_CODING_ENABLED=False`) score **48.0% and 46.7% vs 45.3%** baseline →
  population contributes ~0 net discriminative power on this corpus.
- The 6 high-mass lens classes that actually fire on ETH-80 (Texture 61, Boundary 45,
  Position 38, Color 37, Onset 12, Action 12 nodes/image) carry NO populations:
  their writers still emit `TEXTURE`, `BOUNDARY_ANCHORED`, `POSITION`,
  `COLOR_LEGACY`, `ONSET` class names — none of which have registered population
  axes. Other lens classes are 100% populated already (PEAK_CHAIN, CONTOUR,
  CURVE_COMPLEX(44 axes), SURFACE_PATCH, ...). Wiring the 4 codable classes is
  ~30 lines (unit tables + 2 lines each) but is unimplemented and unproven.

### Occlusion / interruption-gate track (arm E)
- figground corpus (6 scenes with renderer-true junctions):
  **recall 7/12, precision 7/7**. Controls (single_box, single_disk) emit 0 sites.
- The 5 misses, by named sub-condition:
  cylinder_over_box J_right → **turn gate** (candidate pairs with near-parallel
  vertical contour; the true horizontal bar is not an interior candidate);
  box_over_box_haze J_left, J_right → **interruption/continuation gate**, lateral
  band test (blur-tilted stem ray overshoots the reappearing far contour;
  `continuation=None`);
  disk_on_box J_left, J_right → **stem-on-owner-side gate** (outward·owner_normal
  = −0.997 vs the CLOSED-ring occluder limb; disk-occluder is a ring).
- Relaxation matrix over all 8 scenes: **no** single or pair relaxation gains true
  junctions without false ones (no_interrupted: +1 true, +1 false;
  no_owner+no_interrupted: +3 true, +4 false; tol 2.5/3.0: +3 false;
  turn_0.35: −1 true). => **The interruption gate admits no free recall. The root
  cause is upstream of the gate: fragment records near those junctions exist only
  as 2-sample ghosts; the reappearing collinear contour is not chained/owned.**
- ETH-80 is not rendered with occlusions; frontend emits 0 OCCLUSION-typed edges.
  The 1 apple1 site and 3 apple10 sites in the table-crossing band are gate
  hypotheses, not verified junctions (no ground truth exists).

### Frozen bars / constraints (always intact)
- figground: 329 passed, 4 xfailed (pins stay; any reddening reverts the change).
- `_u24_score.py` / `_u24_decide.py` untouched; U20 WRONG=0 intact.
- per-class thresholds preserved at L0/L1.
- Everything reproducible: every ablation ran with full per-class tables;
  no number in this document is unlanded.

## Claims RETRACTED this session (do not use)
1. "edges inert" / "matcher ignores relational signal" — falsified by the full
   split (CO_LOCATED removal costs 8.0 pts).
2. "corpus out of scope" — withdrawn (the 68% is real; track record: Part-1
   enrichment, population coding measured, all neutral).
3. "junction scale dependence" — artifact of summing different images; per-image
   junctions are 2–4 at every canon (64/96/128/192/256).
4. Ownership decisive — all (alpha,neighbor) buckets are single votes; the
   binder requires n≥2; abstains everywhere on this corpus (0–6 votes/image).
5. "junction structural value → fix with bar-side global relaxation" — measured:
   +1 false junction, zero recall gain, broke 3 pins → CLOSED-FOR-GOOD, reverted.
6. 6–8 month structural-value estimate — withdrawn.
7. noedge_col == baseline on the subsample carried no meaning (subsample saturates:
   ~10 configs 11/11 == 100%).
8. Any 79-image number is **in-stent** (tuning⊂eval), not held out.

## Open questions the reviewer should focus on
1. The EDGE-EVIDENCE model, as spec'd (below), implemented behind pins —
   is the redesign the right move and is any part already half-built?
2. The interruption-gate failure modes above — is my attribution correct and is
   the fix upstream (fragment chaining) rather than in the gate?
3. Registration-gap list in the arm-C table — do these unit tables exist and is my
   "wire ≈ 30 lines" estimate correct?
4. Why do atom-bin `CO_LOCATED` matched pairs hit sim=1.0 with *independent
   channel indices* as parameters — is that an acceptable wiring of the harness,
   or a benchmark artifact that flattens edge evidence into a count by construction?

### EDGE-EVIDENCE model (spec only; UNIMPLEMENTED)
- Edge similarity depends on measured axes only; an all-UNKNOWN pair contributes 0.
- Topological count separated from physical relation; the count may not masquerade
  as physics in the score; coverage mass `f(n)=1+log2(n+1)` stays range-guarded.
- Unmeasured axis contributes no mass (no UNKNOWN floor 1.0/0.2 acting as evidence).

## Tuning history (context)
Full-split numbers transfer from the subsample only after this session's correction
(CO_LOCATED −8 pts, color −2.7 pts). Subsample saturates: baseline and noedge_col
identical 11/11; noeff_color wins by +0.0018 (rounding noise). step_005 collapses to
27.3%; `tabs_025/040/060` monotone degradation; `cboost*` and `colboost15` all
11/11 within rounding noise of baseline. **No tuned operating point is honest.**

## Files to read first (in order)
1. `export/README.md` (this file)
2. `diagnostics/_u32_full_benchmark_baseline.txt` (headline result + per-class)
3. `diagnostics/_u32_eth80_bug_hunt_report.txt` (A/B decomposition)
4. `diagnostics/_u32_occlusion_recall_measure.txt` + `_u32_occlusion_named_audit.txt`
   (arm E)
5. `src/memory/cores.py` (`_mcs_accumulate`, `evidence_score`),
   `src/core/schema.py` (EdgeParams, similarity paths),
   `src/perception/surface_fragments.py` (`t_junction_sites`, `_continuation`),
   `src/orchestration/binder.py` (CO_LOCATED emission, `_bo_depth_constraints`)
6. `tests_figground/test_t_junction_interrupted.py` (the pin gate),
   `tests_figground/scenes.py` + `pipeline.py` (renderer truth + probe run)
7. `diagnostics/_u32_eth80_structural_benchmark_v3.txt` (structural path + the
   population ablation in `_u32_eth80_structural_ablated{,_v2}.txt`)
