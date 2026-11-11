# diagnostics/ — ETH-80 gap-closure harness

Everything in this directory is measurement. Nothing here is imported by the
library. The findings are in [`REPORT.md`](REPORT.md); this file is the map
and the commands.

## What this harness is, and is not

It is a **reconstruction**. The export does not contain the `_u32_*` scripts
or the instruments named in the top-level README, so the 68.0 % headline
harness ("atom-bin wiring") cannot be re-run and was not imitated. This
harness drives the real writers instead:

    Observation2D.from_image(rich=False) -> SettleLoop.step -> surround_suppress
    -> Binder.step -> Pipeline._run_higher_lens -> StatePatternIndex.record
    -> memory.cores.evidence_score(query, reference)

That is the README's *structural settle-loop path* (45.3 % there). Landing in
the same band is not a reproduction of it: the references differ (the README's
`cup_01` / `horse_03` do not name an ETH-80 view), and there is no top-down
feedback, no recognition lens and no action stream here.

- **Split** `ETH80-split-v1` (`eth80_split.json`): references
  `<class>1-000-000`; EVAL-75 from object slots 1-5; HELDOUT-75 from slots
  6-10. HELDOUT-75 is scored once per configuration, at the end.
- **Windows**: `full` (every node the front end wrote) and `v1` (V1-tier lens
  nodes only).
- **n = 75.** One binomial standard error is about 4.3 queries. Every
  comparison is paired on the same 75 queries (gained / lost, exact McNemar).
- Three top-level modules are absent from the export; `_export_shims/` holds
  re-export stubs so the tree imports (see its README). They sit last on
  `sys.path` and are inert in a full working tree.

## Set-up

    python3 -m venv ~/.venv && . ~/.venv/bin/activate
    pip install numpy==2.4.6 scipy==1.17.1 pytest==9.2.0 pillow==13.0.0   # Python 3.11.2
    export PYTHONHASHSEED=0
    python diagnostics/eth80_fetch.py                       # 154 PNGs -> data/ETH-80 (git-ignored)
    python diagnostics/eth80_frontend.py --which eval --workers 2   # graph cache, ~9 min

The versions above are the ones every stored result was computed under; the
repository pins none, so each result record now carries its own
(`environment`). After rebuilding the environment, check before measuring:
`python diagnostics/phase2_continuity.py --tags p2_R0` re-runs a stored
configuration and compares it value by value (it must report 0 differing).
`scipy` matters twice: the front end uses `scipy.ndimage`, and the exact
assignment is `scipy.optimize.linear_sum_assignment`.

The graph cache is keyed by a digest of the front-end source (`perception/`,
`lenses/`, `parameters/`, `core/`, `dynamics/`, binder, pipeline,
`concept_graph.py`). Editing any of those rebuilds it. Saved results are keyed
by the whole source tree plus the harness, and are never re-used across a
change.

**The stored records and `main`.** The records carry the digests of the
branch they were measured on (source `f66ea9ac0eea`). The source digest
covers every `.py` outside `diagnostics/`, the tests and `data/`, root-level
scripts included. `main` deleted `_u27_selftest.py` after the branch point
(`5be15a1`, its only commit since), so on a tree without that file the
source digest is `ec8a7ebce62d`, and a script that re-uses records measures
again and rewrites the record under the new digest (17 s on full and 7 s on
v1 per configuration; the front-end digest is unchanged, so the graph cache
is re-used). Nothing imports that script, so no measured value depends on
it. Checked on a scratch merge of `main` into the branch: the merge is
clean, the pins and figground pass as on the branch, and five stored
configurations re-run on both windows give 0 of 112,200 values differing —
`results/logs/merge_with_main_5be15a1.txt`.

## Commands

| what | command |
|---|---|
| one configuration | `python diagnostics/eth80_benchmark.py --window both --breakdown --confusion [--set NAME=VALUE ...]` |
| Phase 0 audits | `python diagnostics/phase0_audit_A_colocated_hist.py` · `..._B_count_only.py` · `..._C_anchor_shuffle.py --seeds 5` |
| Phase 1 (F6) + D1 | `python diagnostics/phase1_f6_d1.py` |
| front-end determinism | `python diagnostics/eth80_determinism.py` |
| figground pins | `PYTHONPATH=.:diagnostics/_export_shims python -m pytest tests_figground --ignore=tests_figground/test_renderer_truth.py -q -p no:cacheprovider` |
| matcher pins | `python -m pytest tests_matcher -q -p no:cacheprovider` |
| stage 2: D3, Phase 4, null calibration | `python diagnostics/phase2_d3_p4_null.py` then `phase2_addendum.py` (descriptive A–F) |
| mixture reversal against its pre-registration | `python diagnostics/phase2_mixture_reversal.py` |
| exact assignment against its pre-registration | `python diagnostics/phase2_exact_assignment.py`, `phase2_correspondence_census.py`, `phase2_symmetry_probe.py` |
| stage 4: blocks against the loops (stored records), P4b against its rule, the specificity cap (descriptive) | `python diagnostics/phase3_blocks_p4b.py` |
| stage 4: the dark global alignment — does it refine, what does a pair cost (no score) | `python diagnostics/phase3_d2_mechanism.py` |
| stage 4: seconds per `evidence_score` call, numpy kernels against the Python loops | `python diagnostics/phase3_speed.py` (idle machine) |
| bug hunt (identities on stored records and real graphs) | `python diagnostics/phase2_bug_hunt.py [--sections A,B,C,D,E,F]` |
| pin triage (every failing assertion, baseline disabled) | `python diagnostics/phase2_pin_triage.py` |
| a stored configuration re-run and compared | `python diagnostics/phase2_continuity.py --tags p2_R0` · under another seed: `PYTHONHASHSEED=1 python diagnostics/phase2_continuity.py --tags p2x_DEF` |
| a bar under explicit switch values, with reach | `PERCEPT_PARAMS='NAME=VALUE;…' PERCEPT_REACH=1 PYTHONPATH=.:diagnostics:diagnostics/_export_shims python -m pytest <dir> -p _param_overrides_plugin -q` |
| do two front-end states build the same graphs | `python diagnostics/eth80_frontend_compare.py --old <digest>` (a prefix is resolved from the stored records) |
| the whole measurement of the last session, in order | `results/logs/measurement_chain_stage4.sh` (stage 3: `measurement_chain_final.sh`) |
| selftest | `PYTHONPATH=.:diagnostics/_export_shims python _u27_selftest.py` — the script exists on the branch this work was done on; `main` deleted it afterwards (`5be15a1`) |

`--set` writes `parameters.NAME` before the run. Every library change made in
this work sits behind one such general parameter, so "before" and "after" are
measured in one tree.

## Files

| file | role |
|---|---|
| `_bootstrap.py` | paths (`ROOT`, `DATA_DIR`, cache, results), shim detection |
| `_stats.py` | McNemar exact, Wilson, tie-split accuracy, paired helpers |
| `eth80_split.py`, `eth80_split.json`, `eth80_fetch.py` | the split and the downloader |
| `eth80_frontend.py` | image -> graph (settle route), disk cache, digest |
| `eth80_determinism.py` | rebuilds four graphs under other hash seeds |
| `eth80_benchmark.py` | windows, arms, scoring, per-type breakdowns, `run_config` |
| `phase0_audit_*.py` | Phase 0 audits A, B, C |
| `phase1_f6_d1.py` | OFF / F6 / D1 / F6+D1 before-after |
| `phase4_edge_census.py`, `phase4_axis_census.py` | Phase 4 step 0: where matched edge pairs get their similarity; what each parameter slot can say |
| `phase2_d3_p4_null.py`, `phase2_addendum.py` | stage 2 against its pre-registered measures; descriptive tables |
| `phase2_correspondence_census.py` | what the node correspondence leaves unpaired, under the approximate and the exact assignment; total weight against the optimum |
| `phase2_symmetry_probe.py` | score(Q, C) against score(C, Q) under three states of the loop and the assignment |
| `phase2_mixture_reversal.py`, `phase2_exact_assignment.py` | the two correctness fixes against their pre-registrations |
| `phase3_blocks_p4b.py` | stage 4: the declared defaults measured with the Python loops and with the numpy kernels, stored against stored; P4b's registered measures and default rule; the specificity cap, descriptive |
| `phase3_d2_mechanism.py` | stage 4: the dark alignment under both budget accountings — sweeps run, budget state, seconds; no score |
| `phase3_speed.py` | stage 4: one `evidence_score` call timed both ways on every EVAL pair, results compared with `==` |
| `phase2_bug_hunt.py` | A ledger of every stored pair, B switch independence, C graph storage, D walk on real pairs, E order dependence, F the numpy kernels against the Python loops on every pair and graph |
| `phase2_pin_triage.py`, `phase2_pin_triage_classes.json` | soft-assert triage of the matcher pins and its classification |
| `phase2_continuity.py` | re-run a stored configuration (new environment, other hash seed) or compare two stored sets |
| `eth80_frontend_compare.py` | graph content under two front-end digests |
| `_param_overrides_plugin.py` | pytest plugin: switch values before collection, call counts of the edited functions |
| `patches/` | the library changes as an ordered series of 16, one change per patch — see `patches/README.md` |
| `results/phase0/` | Phase 0 as measured on the pre-fix source — **frozen** |
| `results/*.txt` | the tables each script prints |
| `results/*.json` | per-pair records (10 significant digits) |
| `results/logs/` | pin runs, front-end and determinism logs |

`tests_figground/test_renderer_truth.py` cannot be collected in this checkout
(it imports `u14_sealed_scenes`, which is not in the export), hence the
`--ignore`. Only 103 of the 329 figground tests exist here.

The `phase3_*.py` scripts are the third round of measurement (stage 4 of
the report). They are not the spec's Phase 3, population wiring, which is
not started.

## The three ways the matcher pins are run

1. `python -m pytest tests_matcher` — every pin under the values it states
   (`conftest.PIN_BASELINE`). Green whatever the tree's defaults are, except
   `test_declared_defaults.py`, which reddens on a default flip on purpose.
2. `python diagnostics/phase2_pin_triage.py` — the baseline disabled, one
   switch changed, every failing assertion listed and classified. It must
   report 0 unclassified.
3. `tests_matcher/test_switch_interactions.py` (part of 1) — sets every
   switch itself and asserts the value under each combination.

With the baseline applied, `PERCEPT_PARAMS` does not reach the pins: "green
under override X" from run 1 says nothing. The figground pins call the
matcher 0 times (`PERCEPT_REACH=1`), so they say nothing about a matcher
switch either.
