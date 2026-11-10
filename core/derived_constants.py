"""core/derived_constants.py -- shared constants and signature helpers for the
derived lenses (moved out of lenses/derived_common.py, Lead 4).

This module owns the operator/chunk thresholds and the signature-compatibility
helpers so that `memory` no longer needs to import from `perception`.  The
definitions here are the single source of truth; `lenses.derived_common`
re-exports them for backwards compatibility.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Sequence

import parameters as P
from core.types import TypedSignature


# --- Operator emission (§2.28) ---
THETA_PROMO_OP = 0.5

RHO_PROMO_RF = 0.1

SIGMA_MAX_OP = 10

TAU_PROMO_MIN = 200

TAU_OP_COOLDOWN = 200

K_OP = 10

Y_FLOOR_OP = 0.01

THETA_FLOOR_OP = 0.05

ETA_CTX_OP = 0.05

ETA_FAT_OP = 0.05

ETA_FAT_DECAY_OP = 0.95

DELTA_OP = 0.02

THETA_HYST_OP = 0.5

KAPPA_OP = 0.05

RHO_OP_TARGET = 0.2

KAPPA_BIRTH = 0.05

RHO_BIRTH = 0.1

THETA_SIM = 0.7

# --- Chunk emission (§5.10) ---
W_CHUNK = 20

W_PLAN_CHUNK = 100

K_CHUNK_CAND = 20

K_CHUNK_PLAN = math.ceil(K_CHUNK_CAND / 3)

K_CHUNK_STAR = math.ceil(K_CHUNK_CAND / 3)

THETA_CHUNK_EMIT = 0.5

RHO_CHUNK_TARGET = 0.1

KAPPA_CHUNK_HOMEO = 0.05

THETA_CHUNK_EMIT_MIN = 0.1

THETA_CHUNK_EMIT_MAX = 0.9

D_CHUNK_MAX = 3

DELTA_CHUNK = 0.02

B_MAX = 3

THETA_CATEGORIZE = P.THETA_CATEGORIZE

K_CATEGORIZE = P.K_CATEGORIZE

KAPPA_CAT = P.KAPPA_CAT

RHO_CAT_TARGET = P.RHO_CAT_TARGET

THETA_CATEGORIZE_MIN = P.THETA_CATEGORIZE_MIN

THETA_CATEGORIZE_MAX = P.THETA_CATEGORIZE_MAX

DELTA_THETA_CAT_MAX = P.DELTA_THETA_CAT_MAX

TAU_PROGRESS = P.TAU_PROGRESS

RHO_PROGRESS_MIN = P.RHO_PROGRESS_MIN

TAU_REVERT = P.TAU_REVERT

DELTA_CAT = P.DELTA_CAT

KAPPA_EMA = 0.05

# --- Active-edge predicate (§1.12) ---
THETA_EDGE_ACTIVE = 0.1

TAU_EDGE_INACTIVE = 500

# --- Promotion breadth/fertility ---
ARITY_MAX = 3

RF_MAX = 10 ** 9

# --- signature helpers ---
_SIM_MEMO: Dict[Tuple[bytes, bytes], float] = {}
_SIM_MEMO_CAP = 200_000


def _sig_compatible(sig1: TypedSignature, sig2: TypedSignature) -> bool:
    """Structural compatibility (§A.4)."""
    if sig1.kind == "CANONICAL" and sig2.kind == "CANONICAL":
        return (
            sig1.class_ == sig2.class_
            and sig1.relation == sig2.relation
            and len(sig1.quantized) == len(sig2.quantized)
        )
    return False


def _compat_key(sig: TypedSignature) -> Optional[Tuple]:
    """Compatibility group key (§A.4 _sig_compatible, hashed form)."""
    if sig.kind != "CANONICAL":
        return None
    return ("CANONICAL", sig.class_, sig.relation, len(sig.quantized))


def _sim_memo(a: TypedSignature, b: TypedSignature) -> float:
    key = (a.serialize(), b.serialize())
    val = _SIM_MEMO.get(key)
    if val is not None:
        return val
    val = a.similarity(b)
    if len(_SIM_MEMO) >= _SIM_MEMO_CAP:
        _SIM_MEMO.clear()
    _SIM_MEMO[key] = val
    return val
