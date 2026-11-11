#!/bin/bash
# The single measurement chain of this turn: all code is landed, measure once.
cd /home/user/Percept4 && . /home/user/.venv/bin/activate && export PYTHONHASHSEED=0
L=diagnostics/results/logs; R=diagnostics/results
step() { echo "=== $(date +%H:%M:%S) $1"; }
step "pins"
python -m pytest tests_matcher -q -p no:cacheprovider > $L/pins_final.txt 2>&1; echo "pins rc=$?"; tail -1 $L/pins_final.txt
step "stage 2 re-measure"
python diagnostics/phase2_d3_p4_null.py > $L/phase2_measure_2.log 2> $L/phase2_measure_2.err; echo "stage2 rc=$?"
step "stage 2 against the records stored before the fixes"
python diagnostics/phase2_continuity.py --tags p2_R0,p2_P4,p2_D3n,p2_D3na,p2_D3,p2_D3P4,p2_D3b,p2_NULL,p2_NULLD3,p2_NULLb --stored-dir $R/stage2_before_fixes --out $L/stage2_vs_before_fixes_all_keys.txt > /dev/null 2>&1; echo "cmp(all keys) rc=$?"
python diagnostics/phase2_continuity.py --tags p2_R0,p2_P4,p2_D3n,p2_D3na,p2_D3,p2_D3P4,p2_D3b,p2_NULL,p2_NULLD3,p2_NULLb --stored-dir $R/stage2_before_fixes --skip-keys node_coverage_binary --out $L/stage2_vs_before_fixes.txt > /dev/null 2>&1; echo "cmp(skip twin) rc=$?"
grep -c "differing 0 " $L/stage2_vs_before_fixes.txt
step "addendum"
python diagnostics/phase2_addendum.py > $L/phase2_addendum_2.log 2>&1; echo "addendum rc=$?"
step "mixture reversal"
python diagnostics/phase2_mixture_reversal.py > $L/phase2_mixture.log 2>&1; echo "mixture rc=$?"; tail -3 $L/phase2_mixture.log
step "exact assignment"
python diagnostics/phase2_exact_assignment.py > $L/phase2_exact.log 2>&1; echo "exact rc=$?"; tail -2 $L/phase2_exact.log
step "correspondence census (both assignments)"
python diagnostics/phase2_correspondence_census.py > $L/phase2_census_2.log 2>&1; echo "census rc=$?"
step "symmetry probe"
python diagnostics/phase2_symmetry_probe.py > $L/phase2_probe_2.log 2>&1; echo "probe rc=$?"
step "bug hunt, final tree"
python diagnostics/phase2_bug_hunt.py > $L/bug_hunt_final.log 2>&1; echo "hunt rc=$?"; tail -1 $L/bug_hunt_final.log
step "hash seed 1"
PYTHONHASHSEED=1 python diagnostics/phase2_continuity.py --tags p2_R0,p2x_DEF,p2x_D3 --out $L/hashseed1_vs_seed0.txt > /dev/null 2>&1; echo "seed rc=$?"
step "pin triage, final tree"
python diagnostics/phase2_pin_triage.py --out $R/phase2_pin_triage.txt > $L/pin_triage_final.log 2>&1; echo "triage rc=$?"; tail -4 $R/phase2_pin_triage.txt | head -2
step "frozen bars: figground under the declared defaults, with reach"
PERCEPT_REACH=1 PYTHONPATH=.:diagnostics:diagnostics/_export_shims python -m pytest tests_figground --ignore=tests_figground/test_renderer_truth.py -q -p no:cacheprovider -p _param_overrides_plugin > $L/bars3_figground_defaults.txt 2>&1; echo "figground rc=$?"; grep -E "passed|failed" $L/bars3_figground_defaults.txt | tail -1
PYTHONPATH=.:diagnostics/_export_shims python _u27_selftest.py > $L/bars3_u27.txt 2>&1; echo "u27 rc=$?"; tail -1 $L/bars3_u27.txt
step "done"
