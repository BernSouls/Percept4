#!/bin/bash
# The single measurement chain of stage 4: all code is landed, measure once.
cd /home/user/Percept4 && . /home/user/.venv/bin/activate && export PYTHONHASHSEED=0
L=diagnostics/results/logs; R=diagnostics/results
TAGS=p2_R0,p2_P4,p2_D3n,p2_D3na,p2_D3,p2_D3P4,p2_D3b,p2_NULL,p2_NULLD3,p2_NULLb,p2m_BASE,p2m_D3,p2m_R0,p2x_DEF,p2x_D3h,p2x_D3
step() { echo "=== $(date -u +%H:%M:%S) $1"; }
step "pins"
python -m pytest tests_matcher -q -p no:cacheprovider > $L/pins_stage4.txt 2>&1; echo "pins rc=$?"; tail -1 $L/pins_stage4.txt
step "stage 2 re-measure (in blocks)"
python diagnostics/phase2_d3_p4_null.py > $L/phase2_measure_4.log 2> $L/phase2_measure_4.err; echo "stage2 rc=$?"
step "addendum"
python diagnostics/phase2_addendum.py > $L/phase2_addendum_4.log 2>&1; echo "addendum rc=$?"
step "mixture reversal"
python diagnostics/phase2_mixture_reversal.py > $L/phase2_mixture_4.log 2>&1; echo "mixture rc=$?"; tail -3 $L/phase2_mixture_4.log
step "exact assignment (+ binary-credit controls)"
python diagnostics/phase2_exact_assignment.py > $L/phase2_exact_4.log 2>&1; echo "exact rc=$?"; tail -2 $L/phase2_exact_4.log
step "correspondence census"
python diagnostics/phase2_correspondence_census.py > $L/phase2_census_4.log 2>&1; echo "census rc=$?"
step "symmetry probe"
python diagnostics/phase2_symmetry_probe.py > $L/phase2_probe_4.log 2>&1; echo "probe rc=$?"
step "blocks criterion 3: every stage-2/3 record against the records stored before the blocks"
python diagnostics/phase2_continuity.py --tags $TAGS --stored-dir $R/stage3_before_blocks --out $L/stage4_vs_before_blocks.txt > /dev/null 2>&1; echo "continuity rc=$?"
grep -c "differing 0 " $L/stage4_vs_before_blocks.txt; tail -2 $L/stage4_vs_before_blocks.txt
step "blocks part 1, P4b, specificity cap"
python diagnostics/phase3_blocks_p4b.py > $L/phase3_blocks_p4b.log 2>&1; echo "p4b rc=$?"; tail -4 $L/phase3_blocks_p4b.log
step "D2 prerequisites: mechanism"
python diagnostics/phase3_d2_mechanism.py > $L/phase3_d2_mechanism.log 2>&1; echo "d2 rc=$?"; tail -2 $L/phase3_d2_mechanism.log
step "bug hunt, final tree (A-F)"
python diagnostics/phase2_bug_hunt.py > $L/bug_hunt_stage4.log 2>&1; echo "hunt rc=$?"; tail -1 $L/bug_hunt_stage4.log
step "hash seed 1"
PYTHONHASHSEED=1 python diagnostics/phase2_continuity.py --tags p2_R0,p2x_DEF,p2x_D3,p3_CA,p3_D3_CA --out $L/hashseed1_vs_seed0_stage4.txt > /dev/null 2>&1; echo "seed rc=$?"; tail -2 $L/hashseed1_vs_seed0_stage4.txt
step "pin triage, final tree"
python diagnostics/phase2_pin_triage.py --out $R/phase2_pin_triage.txt > $L/pin_triage_stage4.log 2>&1; echo "triage rc=$?"; grep "unclassified" $R/phase2_pin_triage.txt
step "frozen bars: figground under the declared defaults, with reach"
PERCEPT_REACH=1 PYTHONPATH=.:diagnostics:diagnostics/_export_shims python -m pytest tests_figground --ignore=tests_figground/test_renderer_truth.py -q -p no:cacheprovider -p _param_overrides_plugin > $L/bars4_figground_defaults.txt 2>&1; echo "figground rc=$?"; grep -E "passed|failed" $L/bars4_figground_defaults.txt | tail -1
PYTHONPATH=.:diagnostics/_export_shims python _u27_selftest.py > $L/bars4_u27.txt 2>&1; echo "u27 rc=$?"; tail -1 $L/bars4_u27.txt
step "speed (idle machine, one process)"
python diagnostics/phase3_speed.py > $L/phase3_speed.log 2>&1; echo "speed rc=$?"; tail -12 $L/phase3_speed.log
step "done"
