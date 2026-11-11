# `_export_shims/` -- import shims for the EXPORT checkout only

The export this repository was created from copied the source *folders* and
deliberately left out the 20 top-level `src/*.py` files and the `action/`
package (see the top-level `README.md`, "Layout").  Three of those names are
imported by modules that the diagnostics and the `tests_figground` pins need:

| missing module | imported by | what the shim does |
|---|---|---|
| `sensory_floor` | 12 modules, at import time (e.g. `orchestration/binder.py`, `lenses/base.py`) | re-exports the names from where they now live (`core.types`, `perception.observation`, `perception.polar`) |
| `derived_lenses` | `orchestration/pipeline.py`, `lenses/derived_common.py` users | re-exports `lenses.derived` (the module that file's docstring calls "the legacy `derived_lenses` surface") |
| `action.decode` | `orchestration/pipeline.py` (top level) | import stub; `decode_action_sequence` raises if it is ever called |

Nothing here defines behaviour: the first two are pure re-exports of code that
is already in the tree, the third only lets an `import` statement succeed.

**They are never preferred over real modules.**  `diagnostics/_bootstrap.py`
appends this directory to the *end* of `sys.path`, so on a full working tree
(where `sensory_floor.py`, `derived_lenses.py` and `action/` exist) the real
modules are found first and these files are dead.

To run the pins in this checkout:

    PYTHONPATH=.:diagnostics/_export_shims python -m pytest tests_figground \
        --ignore=tests_figground/test_renderer_truth.py -q
