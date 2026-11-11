#!/bin/bash
# After the stage-4 chain: three instruments were corrected (no library or
# harness file changed, the digests are the chain's) and are run again.
#   - phase3_d2_mechanism.py: a third category for a differing correspondence
#     (another optimum of the same assignment)
#   - phase2_bug_hunt.py section A: tolerance of one invariant scaled to the
#     stored precision; "never rises" not applied where a count-only pair
#     reads below 1.0 (the cap on, or the Phase-4 floors restored)
#   - one pin added (the cap in combination with P4b): pins and triage again
cd /home/user/Percept4 && . /home/user/.venv/bin/activate && export PYTHONHASHSEED=0
L=diagnostics/results/logs; R=diagnostics/results
step() { echo "=== $(date -u +%H:%M:%S) $1"; }
step "D2 prerequisites: mechanism (idle machine)"
python diagnostics/phase3_d2_mechanism.py > $L/phase3_d2_mechanism_2.log 2>&1; echo "d2 rc=$?"; tail -6 $L/phase3_d2_mechanism_2.log
step "bug hunt, final tree (A-F)"
python diagnostics/phase2_bug_hunt.py > $L/bug_hunt_stage4_2.log 2>&1; echo "hunt rc=$?"; tail -1 $L/bug_hunt_stage4_2.log
step "pins"
python -m pytest tests_matcher -q -p no:cacheprovider > $L/pins_stage4.txt 2>&1; echo "pins rc=$?"; tail -1 $L/pins_stage4.txt
step "pin triage, final tree"
python diagnostics/phase2_pin_triage.py --out $R/phase2_pin_triage.txt > $L/pin_triage_stage4.log 2>&1; echo "triage rc=$?"; grep "unclassified" $R/phase2_pin_triage.txt
step "done"
