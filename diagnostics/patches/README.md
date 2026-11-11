# The library changes, one per patch

These are 16 consecutive commits of the branch `arena/8ef05428-percept4`
— the third to the eighteenth; the two before them add `.gitignore` and the
import shims — exported with `git format-patch -k --zero-commit`: one
library change per patch. (`--zero-commit` leaves the commit hash out of
each file's first line, so the files stay the same if the branch is rebased
or squashed.) They are kept as files as well because the port into another
tree is done by hand, change by change. They apply in order to the upload
(`bd27128`) and reproduce the tree's seven library files —
`parameters/_text.py`, `core/schema.py`, `memory/cores.py`,
`matching/structural_similarity.py`, `matching/signature_matrix.py` (new),
`matching/alignment.py`, `matching/fuzzy.py` — and `tests_matcher/` byte
for byte (checked: `git am` on a fresh clone of the upload, then `cmp`).

    git am -k --keep-cr diagnostics/patches/00*.patch

`--keep-cr` is required: `parameters/_text.py` and
`matching/structural_similarity.py` have CRLF line endings and the patches
carry them. `-k` keeps the subjects as written (`[F6] …`, `[P4b] …`);
without it the patches apply the same and `git am` drops the bracketed tag.

| patch | change | switch (default) | library files |
|---|---|---|---|
| 0001 | F6: two parameter-less edges of one predicate are a structural match | `EDGE_UNPARAMETERISED_STRUCTURAL_MATCH` (True) | `core/schema.py` |
| 0002 | D1: node credit graded by signature similarity | `EVIDENCE_GRADED_NODE_CREDIT` (True) | `memory/cores.py` |
| 0003 | D3: terms that read the query side | `EVIDENCE_QUERY_SIDE_TERMS` (`()`, off: rule failed) | `memory/cores.py` |
| 0004 | Phase 4: an unmeasured edge pair is not evidence | `EDGE_UNMEASURED_IS_NOT_EVIDENCE` (True) | `core/schema.py` |
| 0005 | null calibration of node coverage | `EVIDENCE_NODE_NULL_CALIBRATED` (False, off: rule failed) | `memory/cores.py` |
| 0006 | pins state their switch values | – | tests only |
| 0007 | trace: `node_coverage_binary` is the binary twin of the term in use | none (trace only) | `memory/cores.py` |
| 0008 | D3 under `EVIDENCE_ONLY_EDGES`; term names validated first | none (no score changes) | `memory/cores.py` |
| 0009 | interaction pins and the node term's definition | – | tests only |
| 0010 | mixture reversal on backward traversal | `EDGE_REVERSE_FLIPS_MIXTURE` (True) | `memory/cores.py` |
| 0011 | exact assignment | `MATCH_ASSIGNMENT_EXACT` (True) | `matching/structural_similarity.py` |
| 0012 | Phase 4 pin: categorical agreement alone | – | tests only |
| 0013 | numpy kernels: the node-similarity matrix (population block, Gaussian bins block) and the anchor spacing, bit for bit | `MATCH_SIGNATURE_MATRIX_BLOCKS` (True; performance only) | `matching/signature_matrix.py` (new), `memory/cores.py` |
| 0014 | P4b: the relational term as presence × physics; a lookup key is not a measured axis | `EVIDENCE_RELATION_COUNT_APART` (False, off: rule failed), `EDGE_LOOKUP_KEY_IS_NOT_A_MEASURED_AXIS` (False, unmeasured) | `core/schema.py`, `memory/cores.py` |
| 0015 | D2 prerequisites: the alignment's budget counts the refinement only; block matrix; optional anchor prior | `MATCH_ALIGN_BUDGET_REFINEMENT_ONLY` (True), `MATCH_GLOBAL_ANCHOR_PRIOR` (False); dark path | `matching/alignment.py`, `matching/fuzzy.py`, `memory/cores.py` |
| 0016 | the documented specificity cap, wired for same-schema pairs | `EDGE_SPECIFICITY_CAP_SAME_SCHEMA` (False; a decision for the owner) | `core/schema.py` |

Each patch carries its switch in `parameters/_text.py` and its pin file; the
pins are green after every patch (11, 20, 32, 41, 56, 57, 80, 85, 105, 120,
129, 135, 343, 367, 385, 390 tests; checked at each of the 16 commits in a
fresh checkout).

**Running the pins needs `sensory_floor` to be importable.** In the export
checkout that module is absent, `matching/` and `memory/` cannot be
imported without it, and every pin that imports them fails with
`ModuleNotFoundError`. `diagnostics/_export_shims/` supplies it there;
`tests_matcher/conftest.py` appends that directory to the end of the path
when it exists. The shims are not in the series and are not ported: a full
working tree has the real module and never sees them.

To drop a change, leave its patch out
and the later ones that touch the same function will need a hand merge; to
switch it off, set its parameter. 0015 needs 0013 (it hands the block
matrix to the alignment); 0014 and 0016 are independent of the other
stage-4 patches except for their lines in the shared files.

**How the series was made.** The first five changes were originally separate
commits that were lost with the sandbox; the files survived. The states were
rebuilt by removing each change's blocks from the surviving files. The line
counts of the rebuilt steps equal the ones recorded before the loss for F6
(+23/−2), D1 (+64/−8), Phase 4 (+32/−1) and the null calibration (+141/−5);
D3 is +155/−2 against a recorded +156/−3 — one edited line is attributed to
a neighbouring step. The end state is exact.

The four stage-4 patches were carved out of the final tree the same way:
the difference to patch 0012's state was split into hunks, each hunk
assigned to one change (27 in `memory/cores.py`, 4 in `core/schema.py`,
the per-switch blocks of `parameters/_text.py` and of the two pin files
that list the switches), and the changes committed in order with the pins
run after each.

Diagnostics (`diagnostics/`, `.gitignore`) are not in the series: they are
not library and are not ported.
