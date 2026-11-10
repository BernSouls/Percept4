"""Shared fixtures: the corpus, and ONE cached pipeline run at a time.

A full field+chain+ownership run is ~10 s per scene, so runs are cached -- but
the box has ~1 GB of RAM and a `Run` retains the interior field and the fragment
/ record lists, so caching all eight scenes at once OOMs the interpreter before
the corpus finishes.  The cache therefore holds ONE run: asking for a different
scene frees the previous one.  `run_pipeline` is deterministic (the tree's own
contract), so rebuilding cannot mask a nondeterminism bug -- those get their own
test that runs the field twice.

`scene(name)` returns `(Scene, Run)`.  A test that loops over several scenes
holds only the current one; a test that needs two runs at once must say so by
calling `run_pipeline` itself.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scenes import build_all                     # noqa: E402
from scenes import (CAL_SCENES, TEST_SCENES,      # noqa: E402,F401
                    OCCLUSION_SCENES, CONTROL_SCENES)
from pipeline import run_pipeline                 # noqa: E402

#: the scenes that carry renderer-true junctions
OCCLUSION_SCENES = OCCLUSION_SCENES
#: scenes that must produce no order claim at all
CONTROL_SCENES = CONTROL_SCENES
#: CAL / TEST split (session rule 4: derive on CAL, verify on TEST)
CAL = CAL_SCENES
TEST = TEST_SCENES


class _OneRunCache:
    """Cache of exactly one (scene name -> Run) pair."""

    def __init__(self, corpus):
        self.corpus = corpus
        self._name = None
        self._run = None
        self.builds = 0

    def get(self, name, **kwargs):
        if self._name != name or kwargs:
            self._run = None
            self._name = name
            self._run = run_pipeline(self.corpus[name], **kwargs)
            self.builds += 1
        return self.corpus[name], self._run


@pytest.fixture(scope="session")
def corpus():
    return build_all()


@pytest.fixture(scope="session")
def cache(corpus):
    return _OneRunCache(corpus)


@pytest.fixture(scope="session")
def scene(cache):
    """`scene(name) -> (Scene, Run)`, holding at most one run at a time."""
    return cache.get


@pytest.fixture(scope="session")
def wired_scene(cache):
    """`wired_scene(name) -> (Scene, Run)` with the live junction sites
    injected (H5).  Only scenes with true junctions are wired."""
    def _get(name):
        sc = cache.corpus[name]
        return cache.get(name, with_live_junctions=bool(sc.junctions))
    return _get


class _LazyRuns:
    """Mapping-like view kept for diagnostics that iterate the corpus.

    Indexing rebuilds (and frees the previous run), so `runs[a]` and `runs[b]`
    must not be held at the same time.
    """

    def __init__(self, cache):
        self._cache = cache

    def __getitem__(self, name):
        return self._cache.get(name)[1]

    def __contains__(self, name):
        return name in self._cache.corpus

    def __iter__(self):
        return iter(self._cache.corpus)

    def keys(self):
        return self._cache.corpus.keys()


@pytest.fixture(scope="session")
def runs(cache):
    return _LazyRuns(cache)


@pytest.fixture(scope="session")
def wired_runs(cache):
    out = {}
    for name, sc in cache.corpus.items():
        if sc.junctions:
            out[name] = cache.get(name, with_live_junctions=True)[1]
    return out
