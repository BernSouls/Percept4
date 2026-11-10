"""Observation-level memoisation of the SurfaceFragment field.

The U30 ontology ruling makes SurfaceFragment an INTERMEDIATE computation,
not a graph node.  Emitters and lenses must therefore share one measured
fragment field per observation, not re-derive it per caller.

The cache is content-hash keyed, exactly like ``peak_lens._contour_field_cache``:
a fragment field built from a particular luminance under a particular builder
configuration is stored under ``(shape, hash(luminance), fingerprint)``, not
under object identity.  Two observations carrying the same image therefore
share their fragments; two observations with different images never do.

The cache is intentionally module-local and lives at module scope; the
individual ``observation`` object carries an attribute describing which key
(if any) it currently holds.  Reading paths:

- ``observation.surface_fragments``            — attribute accessor (list)
- ``observation.get_surface_fragments()``     — callable accessor (list)
- ``surface_cache.get_surface_fragments(obs)`` — content-hash memoised access
- ``surface_cache.install_surface_fragments_on_observation(obs, fragments)``
  — called once a fragment field has been derived; future reads are free.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# The fingerprint version tracks the builder's PARAMETERS, not the
# observation's runtime content.  A behavioural change in the builder MUST
# bump this constant or two parameter states will alias under one key.
#
# [U30] v1 — first registration; covers the POLYLINE_SWEEP and list ordering
# in ``build_surface_fragments``; future bumps bump the leading byte(s).
SURFACE_FRAGMENT_FINGERPRINT_VERSION = 1

_SURFACE_FRAGMENT_CACHE: Dict[Tuple[tuple, bytes, bytes], List[Any]] = {}


def surface_fingerprint() -> bytes:
    """Cheap content digest of the builder's behavioural parameters.

    Derived from the names and values of every ``SURFACE_*`` constant on the
    parameter module; any behavioural change to the surface suite invalidates
    cached fragments.  Cheap because it is already materialised for _text.py
    startup at import.
    """
    import parameters as P
    pieces = []
    for name in sorted(dir(P)):
        if "SURFACE_" in name.upper():
            pieces.append("%s=%s" % (name, getattr(P, name, "")))
    text = "|".join(pieces)
    return hashlib.blake2b(
        ("SURFACE_FRAGMENT|%d|" % SURFACE_FRAGMENT_FINGERPRINT_VERSION
         + text).encode("utf-8"),
        digest_size=16).digest()


def _luminance_digest(luminance: Any) -> Optional[Tuple[tuple, bytes]]:
    if luminance is None:
        return None
    arr = np.asarray(luminance)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    return (arr.shape, hashlib.blake2b(
        np.ascontiguousarray(arr).tobytes(),
        digest_size=16).digest())


def _key(luminance: Any, rgb: Any) -> Optional[Tuple[tuple, bytes, bytes]]:
    parts = _luminance_digest(luminance)
    if parts is None:
        return None
    shape, h1 = parts
    h2 = hashlib.blake2b(
        np.ascontiguousarray(np.asarray(rgb)).tobytes(),
        digest_size=16).digest() if rgb is not None else b"\x00" * 16
    return (shape, h1, h2 + surface_fingerprint())


def install_surface_fragments_on_observation(
    observation: Any,
    fragments: List[Any],
) -> None:
    """Record a fragment field on the observation object.

    Mirrors the ``_contour_field_cache`` pattern: the fragments ride on the
    observation so every lens/emitter on that observation reads the same
    field.  Calling this twice for the same observation with the same image
    is a no-op for the second caller.
    """
    if observation is None:
        return
    try:
        observation._surface_fragment_cache = list(fragments)
    except Exception:
        pass


def get_surface_fragments(
    observation: Any,
    *,
    builder: Optional[Any] = None,
) -> Optional[List[Any]]:
    """Memoised surface fragments for this observation.

    Returns the cached list when present, otherwise invokes ``builder`` (or
    bails out as ``None``), otherwise builds and caches the field.  Pure;
    no detections are emitted here.
    """
    if observation is None:
        return None

    cached = getattr(observation, "_surface_fragment_cache", None)
    if cached is not None:
        return list(cached)

    luminance = None
    if hasattr(observation, "get_luminance"):
        try:
            luminance = observation.get_luminance()
        except Exception:
            luminance = None
    if luminance is None:
        luminance = getattr(observation, "luminance", None)
    rgb = getattr(observation, "rgb", None)

    key = _key(luminance, rgb)
    if key is not None and key in _SURFACE_FRAGMENT_CACHE:
        cached = _SURFACE_FRAGMENT_CACHE[key]
        install_surface_fragments_on_observation(observation, cached)
        return list(cached)

    if builder is not None:
        fragments = list(builder())
        install_surface_fragments_on_observation(observation, fragments)
        if key is not None:
            _SURFACE_FRAGMENT_CACHE[key] = list(fragments)
        return list(fragments)

    return None


def reset_surface_fragment_cache() -> None:
    """Clear the global cache (tests only)."""
    _SURFACE_FRAGMENT_CACHE.clear()
