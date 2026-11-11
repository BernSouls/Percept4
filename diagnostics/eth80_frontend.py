"""Image -> match graph, by the repository's OWN front end.

This module contains NO perception or binding logic of its own.  For one
image it performs the perception + binding half of one ``Pipeline.step`` on a
fresh ``ConceptGraph``, calling the same objects the pipeline calls, in the
same order:

  1. ``Observation2D.from_image(img, rich=False)``
  2. ``dynamics.settle.SettleLoop.step``         (the V1 lens tier, through
     the raw-image settle route exactly as ``Pipeline._settle_loop_step``
     calls it; it owns the single ``SensoryFloor.step``)
  3. ``orchestration.pipeline.surround_suppress`` (as ``Pipeline.step`` does
     when ``P.HIGHER_LENS_ENABLED``)
  4. ``Binder.step``                             (every V1 edge channel,
     including ``_run_co_located``)
  5. the higher lens tier, in ``declared_lens_order()``, each lens invoked
     through ``Pipeline._run_higher_lens`` itself, followed by the
     ``bind_intra_step`` / ``emit_pending_edges`` choreography transcribed
     from ``Pipeline.step`` (orchestration/pipeline.py, "[P0-1] THE TIER IS
     DRIVEN BY THE DECLARED INPUTS")
  6. ``StatePatternIndex.record``                (the pipeline's own
     situation-window ``Subgraph`` builder: induced edges, ``edge_params_for``,
     ``anchor_positions_for``)

WHY NOT ``Pipeline`` ITSELF.  ``Pipeline.__init__`` needs an environment, the
action repertoire, the prediction module, the utility engine and the
consolidator; none of them is in the export.  Steps 1-6 need none of them.

WHAT IS *NOT* REPRODUCED (stated, not hidden):
  * top-down feedback.  The settle loop runs, but on a fresh graph there are
    no concepts, hence no hypotheses and no attentional gain: its later
    iterations only re-run the V1 lenses against their own hysteresis state.
  * the recognition / operator / chunk lenses (``derived_lenses.step_13..16``):
    they emit from crystallised concepts, and a fresh graph has none.
  * multi-step episodes and actions.  One step per image: there are no Action
    nodes and Onset collapses to a single node.

``rich=False`` is the contour-only observation mode the repository's own
gate/cache paths use; under it the Color lens takes the legacy RGB path
(class ``COLOR_LEGACY``), which is the writer set the handoff describes
(``TEXTURE``, ``BOUNDARY_ANCHORED``, ``POSITION``, ``COLOR_LEGACY``, ``ONSET``).

The result is cached on disk, keyed by the image bytes and by the bytes of
every perception / binding source file, so a matcher-only change re-uses the
cache and a front-end change invalidates it automatically.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)

import hashlib
import io
import logging
import os
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from _bootstrap import CACHE_DIR, DATA_DIR, ROOT

#: Source files whose bytes decide what the front end emits.  A change to any
#: of them invalidates the graph cache.  The matcher (memory/, matching/) is
#: deliberately absent.
_FRONTEND_SOURCE_DIRS = ("perception", "lenses", "parameters", "core",
                         "dynamics")
_FRONTEND_SOURCE_FILES = ("orchestration/binder.py",
                          "orchestration/pipeline.py",
                          "concept_graph.py",
                          "diagnostics/eth80_frontend.py")

_log = logging.getLogger("eth80_frontend")


@dataclass
class ImageGraph:
    """One image's graph, in the exact types the matcher consumes."""

    name: str
    #: the pipeline's situation window (core.graph_types.Subgraph)
    subgraph: Any
    #: node id -> (lens_type, TypedSignature)   (the SignatureProvider view)
    sigs: Dict[int, Tuple[str, Any]]
    #: node id -> observation anchor it was first bound from
    node_anchor: Dict[int, Any]
    #: per-lens detection counts: raw V1, after surround suppression, higher
    det_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    #: binder channel diagnostics (candidates / emitted / dropped per type)
    binder_diag: Dict[str, Dict[str, int]] = field(default_factory=dict)
    #: the lenses of the higher tier that were skipped (empty declared input)
    skipped_lenses: List[str] = field(default_factory=list)
    #: number of T-junction sites the observation handed to the binder
    n_t_junctions: int = 0
    #: raw CO_LOCATED writer measurements (audit A); see ``build_graph``
    co_located_raw: List[Dict[str, Any]] = field(default_factory=list)
    #: (lens, anchor, z, Detection.scale) of every V1 detection handed to the
    #: binder -- the two Detection fields the CO_LOCATED writer reads
    v1_dets: List[Tuple[str, Any, float, Any]] = field(default_factory=list)
    seconds: float = 0.0


# --------------------------------------------------------------------------
# cache key
# --------------------------------------------------------------------------

_SOURCE_DIGEST: Optional[str] = None


def frontend_source_digest() -> str:
    """sha256 over every front-end source file (path + bytes)."""
    global _SOURCE_DIGEST
    if _SOURCE_DIGEST is not None:
        return _SOURCE_DIGEST
    h = hashlib.sha256()
    paths: List[str] = []
    for d in _FRONTEND_SOURCE_DIRS:
        for base, _dirs, files in os.walk(os.path.join(ROOT, d)):
            for f in files:
                if f.endswith(".py"):
                    paths.append(os.path.join(base, f))
    paths.extend(os.path.join(ROOT, f) for f in _FRONTEND_SOURCE_FILES)
    for p in sorted(paths):
        h.update(os.path.relpath(p, ROOT).replace(os.sep, "/").encode())
        with open(p, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    _SOURCE_DIGEST = h.hexdigest()
    return _SOURCE_DIGEST


# --------------------------------------------------------------------------
# the front end
# --------------------------------------------------------------------------

class _PipelineHost:
    """The attributes ``Pipeline._run_higher_lens`` reads from ``self``.

    Built exactly as ``Pipeline.__init__`` builds them (same classes, no
    arguments).  Nothing else of the pipeline is constructed.
    """

    def __init__(self, cg):
        import orchestration.pipeline as pl
        from lenses.base import PolarityLens
        from lenses.chain_lens import ChainLens
        from lenses.corner_lens import CornerLens
        from lenses.peak_lens import PeakLens

        self.concept_graph = cg
        self.providers = pl.PipelineProviders(cg)
        self.border_ownership_lens = pl.BorderOwnershipLens()
        self.surface_lens = pl.SurfaceLens()
        self.contour_lens = pl.ContourLens()
        self.junction_lens = pl.JunctionLens()
        self.peak_lens = PeakLens()
        self.chain_lens = ChainLens()
        self.corner_lens = CornerLens()
        self.illusory_contour_lens = pl.IllusoryContourLens()
        self.curve_complex_lens = pl.CurveComplexLens()
        self.region_shape_lens = pl.RegionShapeLens()
        self.polarity_lens = PolarityLens()
        self.endstopped_lens = pl.EndStoppedLens()
        self.texture_boundary_lens = pl.TextureBoundaryLens()


def _load_rgb(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def build_graph(image: np.ndarray, name: str = "",
                record_co_located: bool = True) -> ImageGraph:
    """Run steps 1-6 of the module docstring on one uint8 RGB image.

    ``record_co_located`` wraps (does not replace) the binder's edge-params
    constructor for the duration of the call, to capture the CONTINUOUS values
    the CO_LOCATED writer hands it -- the instrument audit A reads.  The
    wrapper forwards its arguments untouched and returns the original result.
    """
    import parameters as P
    import orchestration.binder as binder_mod
    import orchestration.pipeline as pl
    from concept_graph import ConceptGraph
    from core.hashing import _detection_sort_key
    from dynamics.settle import SettleLoop
    from lenses.derived_common import (LENS_DECLARATION_BY_NAME,
                                       declared_lens_order, junction_alias)
    from orchestration.binder import Binder
    from perception.floor import SensoryFloor
    from perception.observation_2d import Observation2D

    t_start = time.time()
    t = 0
    cg = ConceptGraph()
    binder = Binder(cg)
    floor = SensoryFloor()
    host = _PipelineHost(cg)

    co_raw: List[Dict[str, Any]] = []
    orig_edge_params = binder_mod._edge_params

    def _recording_edge_params(class_name, values, *a, **kw):
        ep = orig_edge_params(class_name, values, *a, **kw)
        if class_name == "EDGE_CO_LOCATED":
            co_raw.append({
                "scale_ratio": values.get("scale_ratio"),
                "salience_ratio": values.get("salience_ratio"),
                "bins": tuple(ep.quantized),
            })
        return ep

    if record_co_located:
        binder_mod._edge_params = _recording_edge_params
    try:
        # 1. observation
        obs = Observation2D.from_image(image, rich=False)

        binder.note_step_begin(t)
        # 2. V1 lens tier through the settle route (Pipeline.step, "Step 2",
        #    raw-image branch; Pipeline._settle_loop_step).  `_ap` is the
        #    caller-owned map the GABOR runner writes its tuple anchors into.
        loop = SettleLoop(sensory_floor=floor, binder=binder,
                          self_binders=None, derived=None, cg=cg)
        _ap = pl.Pipeline._derive_anchor_positions(None, obs)
        _es = pl.Pipeline._derive_extraction_scale(None, obs)
        D_raw, _derived, _S_t, s_t, _T_prev = loop.step(
            t, obs, None, None, set(),
            anchor_positions=_ap, extraction_scale=_es)
        _lens_positions = _ap

        anchor_positions = pl.Pipeline._derive_anchor_positions(None, obs)
        binder_positions = dict(anchor_positions)
        if _lens_positions:
            binder_positions.update(_lens_positions)
        t_junctions = obs.get_t_junctions()
        extraction_scale = pl.Pipeline._derive_extraction_scale(None, obs)
        modality = pl.observation_modality(obs)

        # 3. surround suppression
        D_v1: FrozenSet = frozenset(D_raw)
        if P.HIGHER_LENS_ENABLED:
            D_v1 = pl.surround_suppress(
                frozenset(D_raw), anchor_positions, modality,
                extraction_scale)
        D_bind = set(D_v1)

        # 4. binder
        binder.step(
            t, frozenset(D_bind),
            anchor_positions=(binder_positions if binder_positions else None),
            anchor_velocities=None,
            t_junctions=t_junctions,
            extraction_scale=extraction_scale,
            structural_anchors=obs.get_structural_anchors(),
            surface_anchors=obs.get_surface_anchors(),
            contour_surface_contacts=obs.get_contour_surface_contacts(),
        )

        # 5. higher lens tier (transcription of Pipeline.step's tier loop)
        tier_dets: Dict[str, List[Any]] = {}
        skipped: List[str] = []
        if P.HIGHER_LENS_ENABLED:
            by_k: Dict[str, List[Any]] = defaultdict(list)
            for d in sorted(D_bind, key=_detection_sort_key):
                by_k[d.k].append(d)
            sig_provider = getattr(floor, "signature_provider", None)
            lens_dets: List[Any] = []
            for lens_name in declared_lens_order():
                decl = LENS_DECLARATION_BY_NAME[lens_name]
                if decl.missing_inputs(by_k):
                    skipped.append(lens_name)
                    tier_dets[lens_name] = []
                    continue
                out = pl.Pipeline._run_higher_lens(
                    host, lens_name, t, cg, sig_provider,
                    (binder_positions
                     if (lens_name == "Junction"
                         or (lens_name == "Contour" and P.CONTOUR_SPINE_MAP))
                     else anchor_positions),
                    by_k, extraction_scale, obs)
                out = (out if not pl._HIGHER_LENS_OUTPUT_SORTED[lens_name]
                       else sorted(out, key=_detection_sort_key))
                tier_dets[lens_name] = out
                if lens_name != "CurveComplex":
                    if lens_name == "Junction":
                        for d in out:
                            if d.k != "Junction":
                                by_k[d.k].append(d)
                        by_k["Junction"].extend(junction_alias(out))
                    else:
                        for d in out:
                            by_k[d.k].append(d)
                if lens_name in ("EndStopped", "RegionShape", "Polarity",
                                 "TextureBoundary"):
                    lens_dets.extend(out)
                # --- the bind choreography, per lens ---
                if lens_name in ("BorderOwnership", "Surface"):
                    binder.bind_intra_step(t, frozenset(out))
                elif lens_name == "Corner":
                    binder.bind_intra_step(t, frozenset(
                        list(tier_dets.get("Contour", []))
                        + list(tier_dets.get("Junction", []))
                        + list(tier_dets.get("Peak", []))
                        + list(tier_dets.get("Chain", []))
                        + list(out)))
                elif lens_name == "TextureBoundary":
                    binder.bind_intra_step(t, frozenset(lens_dets))
                elif lens_name == "IllusoryContour":
                    for d in out:
                        inds = [r for r in d.rho
                                if r[0] in ("inducer_1", "inducer_2")]
                        pts = [anchor_positions.get(r[1]) for r in inds]
                        pts = [p for p in pts if p]
                        if len(pts) == 2:
                            anchor_positions[d.alpha] = (
                                (pts[0][0] + pts[1][0]) / 2.0,
                                (pts[0][1] + pts[1][1]) / 2.0,
                                0.0)
                    binder.bind_intra_step(t, frozenset(out))
                    binder.bind_intra_step(t, frozenset(
                        list(tier_dets.get("Contour", []))
                        + list(tier_dets.get("Junction", []))
                        + list(tier_dets.get("Peak", []))
                        + list(tier_dets.get("Chain", []))
                        + list(tier_dets.get("Corner", []))
                        + list(out)))
                    binder.emit_pending_edges("v2")
                elif lens_name == "CurveComplex":
                    binder.bind_intra_step(t, frozenset(out))
                    binder.emit_pending_edges("v4")

        # D_bind_all, in the pipeline's fixed union order
        D_v2 = (list(tier_dets.get("Contour", []))
                + list(tier_dets.get("Junction", []))
                + list(tier_dets.get("Peak", []))
                + list(tier_dets.get("Chain", []))
                + list(tier_dets.get("Corner", []))
                + list(tier_dets.get("IllusoryContour", [])))
        D_v4 = (list(tier_dets.get("CurveComplex", []))
                + list(tier_dets.get("EndStopped", []))
                + list(tier_dets.get("RegionShape", []))
                + list(tier_dets.get("Polarity", []))
                + list(tier_dets.get("TextureBoundary", [])))
        D_bind_all = (set(D_bind)
                      .union(tier_dets.get("BorderOwnership", []))
                      .union(tier_dets.get("Surface", []))
                      .union(D_v2).union(D_v4))

        # 6. the pipeline's situation window
        index = pl.StatePatternIndex(cg)
        index.record(s_t, {d for d in D_bind_all
                           if d.k != pl.ACTION_LENS_NAME})
        subgraph = index(s_t)
        diag = binder.get_diagnostics()
        binder.finalize_step()
    finally:
        binder_mod._edge_params = orig_edge_params

    provider = pl.make_signature_provider(cg)
    sigs: Dict[int, Tuple[str, Any]] = {}
    for nid in sorted(subgraph.nodes):
        info = provider(nid)
        if info is not None:
            sigs[nid] = info
    anchor_map = cg.pattern_node_anchor_map()

    counts: Dict[str, Dict[str, int]] = {
        "v1_raw": dict(Counter(d.k for d in D_raw)),
        "v1_kept": dict(Counter(d.k for d in D_v1)),
        "higher": {k: len(v) for k, v in tier_dets.items()},
    }
    binder_diag = {
        key: {str(k): int(v) for k, v in diag.get(key, {}).items() if v}
        for key in ("candidates_by_type", "emitted_by_type",
                    "dropped_below_threshold", "candidate_cap_hits")
    }
    return ImageGraph(
        name=name,
        subgraph=subgraph,
        sigs=sigs,
        node_anchor={n: anchor_map.get(n) for n in subgraph.nodes
                     if n in anchor_map},
        det_counts=counts,
        binder_diag=binder_diag,
        skipped_lenses=skipped,
        n_t_junctions=len(t_junctions or ()),
        co_located_raw=co_raw,
        v1_dets=sorted(((d.k, d.alpha, float(d.z), d.scale) for d in D_v1),
                       key=repr),
        seconds=time.time() - t_start,
    )


# --------------------------------------------------------------------------
# cached access
# --------------------------------------------------------------------------

def image_path(cls: str, rel: str) -> str:
    return os.path.join(DATA_DIR, cls, os.path.basename(rel))


def graph_for(cls: str, rel: str, *, use_cache: bool = True) -> ImageGraph:
    """Cached ``build_graph`` for one split entry (class, repo-relative path)."""
    path = image_path(cls, rel)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "%s is missing -- run `python diagnostics/eth80_fetch.py`" % path)
    with open(path, "rb") as fh:
        img_digest = hashlib.sha256(fh.read()).hexdigest()
    key = hashlib.sha256(
        (frontend_source_digest() + "|" + img_digest).encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, key[:2], key + ".pkl")
    if use_cache and os.path.isfile(cache_file):
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)
    g = build_graph(_load_rgb(path), name=os.path.basename(rel))
    if use_cache:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        tmp = cache_file + ".tmp%d" % os.getpid()
        with open(tmp, "wb") as fh:
            pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_file)
    return g


def _warm(args) -> Tuple[str, float]:
    cls, rel = args
    g = graph_for(cls, rel)
    return g.name, g.seconds


def warm_cache(entries, workers: int = 2, verbose: bool = True) -> None:
    """Build every missing graph (process pool; the front end is CPU bound)."""
    import multiprocessing as mp

    entries = list(entries)
    t0 = time.time()
    if workers <= 1:
        for i, e in enumerate(entries):
            _warm(e)
            if verbose and (i + 1) % 10 == 0:
                print("  front end %d/%d  (%.0fs)" % (
                    i + 1, len(entries), time.time() - t0), flush=True)
        return
    with mp.get_context("fork").Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_warm, entries, chunksize=1)):
            if verbose and (i + 1) % 10 == 0:
                print("  front end %d/%d  (%.0fs)" % (
                    i + 1, len(entries), time.time() - t0), flush=True)


def main(argv=None) -> int:
    import argparse

    from eth80_split import load_split

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--which", default="eval",
                    choices=("refs", "eval", "heldout", "all"))
    a = ap.parse_args(argv)
    split = load_split()
    entries = list(split["references"].items())
    if a.which in ("eval", "all"):
        entries += [(c, p) for c, ps in split["eval"].items() for p in ps]
    if a.which in ("heldout", "all"):
        entries += [(c, p) for c, ps in split["heldout"].items() for p in ps]
    print("front-end source digest", frontend_source_digest()[:16])
    warm_cache(entries, workers=a.workers)
    print("done: %d graphs" % len(entries))
    return 0


if __name__ == "__main__":
    # Re-enter through the importable module so ImageGraph pickles as
    # `eth80_frontend.ImageGraph`, not `__main__.ImageGraph`.
    import eth80_frontend as _self

    raise SystemExit(_self.main())
