"""memory/cores.py -- core library, MCS matching and evidence scoring
(Move 5c).

MOVE ONLY from consolidator.py. Whole top-level statements, verbatim.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, FrozenSet
import parameters as P
from matching.structural_similarity import (  # noqa: F401
    _max_weight_bipartite_matching as _max_weight_match,
    GESTALT_LENS_BOOST,
)
from core.graph_types import (  # noqa: F401
    Subgraph,
)
from core.hashing import (  # noqa: F401
    _node_signature_hash,
    clamp,
    hash_128,
    serialize_bytes,
    serialize_int,
    serialize_tuple,
    serialize_value,
)
from core.schema import (  # noqa: F401
    coarsen_edge_params_for_wl,
    polymorphic_edge_similarity,
    reverse_edge_params,
)
from core.types import (  # noqa: F401
    EdgeType,
    TypedSignature,
    UNDIRECTED_EDGE_TYPES,
)
from memory.content import (  # noqa: F401
    SignatureProvider,
)
from memory.resonance import (  # noqa: F401
    CoreKind,
    CorePayload_Action,
    CorePayload_Category,
    CoreStatus,
)




THETA_PAIR_TRIALS: int = P.THETA_PAIR_TRIALS

THETA_CATEGORY_INSTANCES: int = P.THETA_CATEGORY_INSTANCES

THETA_CATEGORY_DIVERSITY: float = P.THETA_CATEGORY_DIVERSITY

THETA_CONTAINMENT: float = P.THETA_CONTAINMENT

K_CONSOLIDATE: int = P.K_CONSOLIDATE

THETA_GROWTH_PLATEAU: float = P.THETA_GROWTH_PLATEAU

M_PLATEAU: int = P.M_PLATEAU

CORE_SIZE_CAP_NODES: int = P.CORE_SIZE_CAP_NODES

CORE_SIZE_CAP_EDGES: int = P.CORE_SIZE_CAP_EDGES

KAPPA_RELIABILITY: float = P.KAPPA_RELIABILITY

# Numeric floor.
_EPS: float = 1e-12

# ---------------------------------------------------------------------------
# [K.1] Tier classes for attrition accounting.
#
# Tier numbers follow the binder's own level partition — see the EdgeType
# comment in binder.py ("partitioned by level: v2 = 21/22, v4 = 23,
# lo_pit = 24/25, ait = 26") — and the derived-lens headers in
# derived_lenses.py (B.5/B.6/B.7 = V2, B.8 = V4, B.9 = LO/pIT, B.10 = aIT):
#
#   0 = V1 : the eight production V1 lenses (lenses.py) and the order-1
#            edge binders run by Binder.step
#   1 = V2 : BorderOwnership / Contour / Junction, their lateral links, and
#            their links back to the Orientation segments they summarize
#   2 = V4 : CurveComplex and its containment edges
#   3 = IT : Part / ObjectShape and their containment edges
#
# -1 = not classified. Callers bucket -1 as "*_unknown" so an unclassified
#      type is visible rather than silently folded into a tier.
# ---------------------------------------------------------------------------
_V1_LENSES = frozenset({
    "Boundary", "Color", "Motion", "Onset", "Orientation", "Position",
    "Size", "Texture",
})

# [U20 LEAD 3] The junction FAMILIES are V2 detections: the family split
# renames the node kind, it does not move the tier.  "Corner" is listed here
# for BOTH producers (JunctionLens' corner family and CornerLens' turn
# extrema, which are V2 detections too); leaving it out would bucket a V2
# node as unclassified and make the tier report lie.
_V2_LENSES = frozenset({
    "BorderOwnership", "Contour", "Junction", "Corner", "T", "Y",
})

_V4_LENSES = frozenset({"CurveComplex"})

_IT_LENSES = frozenset()

_V1_EDGE_TYPES = frozenset({
    EdgeType.CO_ACTIVATION, EdgeType.TRANSITION, EdgeType.CO_MOTION,
    EdgeType.CONTRAST, EdgeType.PERIODICITY, EdgeType.PERSISTENCE,
    EdgeType.TOPOLOGY, EdgeType.DEGREE, EdgeType.CONTACT, EdgeType.SEQUENCE,
    EdgeType.SYNCHRONY,
    EdgeType.ORIENTATION_CONTINUITY, EdgeType.SURFACE_GROUPING,
    EdgeType.CURVATURE, EdgeType.OCCLUSION, EdgeType.CO_LOCATED,
    EdgeType.ORIENTATION_BEND, EdgeType.ORIENTATION_JUNCTION,
})

_V2_EDGE_TYPES = frozenset({
    EdgeType.CONTOUR_CONTINUITY, EdgeType.JUNCTION_CONTOUR,
    EdgeType.CONTOUR_ORIENTATION, EdgeType.JUNCTION_ORIENTATION,
    EdgeType.AMODAL_LINK, EdgeType.COCIRCULAR, EdgeType.PARALLEL,
    EdgeType.BORDER_OWNERSHIP_EDGE,
    EdgeType.CURVATURE_SIGN_CONTINUITY, EdgeType.SHARED_BOUNDARY,
})

_V4_EDGE_TYPES = frozenset({EdgeType.CURVE_CONTAINMENT, EdgeType.SYMMETRY})

_IT_EDGE_TYPES = frozenset({
    EdgeType.PART_CONTAINMENT, EdgeType.REGION_CONTENT,
})

def lens_tier(lens_type: str) -> int:
    """[K.1] Tier of a lens type (a detection's `k`); -1 == unclassified."""
    if lens_type in _V1_LENSES:
        return 0
    if lens_type in _V2_LENSES:
        return 1
    if lens_type in _V4_LENSES:
        return 2
    if lens_type in _IT_LENSES:
        return 3
    return -1

def edge_tier(edge_type: EdgeType) -> int:
    """[K.1] Tier of an edge type; -1 == unclassified.

    [U31 / M3] An EdgeCategory carries its tier in EDGE_CATEGORY_TIER, so a v2
    edge is no longer silently unclassified (-1) in the mass traces.  Every
    legacy EdgeType takes exactly the path it took before.
    """
    from core.types import EDGE_CATEGORY_TIER, EdgeCategory
    if isinstance(edge_type, EdgeCategory):
        return int(EDGE_CATEGORY_TIER.get(edge_type, -1))
    if edge_type in _V1_EDGE_TYPES:
        return 0
    if edge_type in _V2_EDGE_TYPES:
        return 1
    if edge_type in _V4_EDGE_TYPES:
        return 2
    if edge_type in _IT_EDGE_TYPES:
        return 3
    return -1

# [K.7] Containment-edge taxonomy for _minimize_core's subtree phase.
#
# These two names are specified by the K.7 brief but did not exist in this
# tree (PF-3); they are defined here from the emitters in binder.py, whose
# docstrings carry the orientation convention. Three higher-tier relations
# are directed container -> contained, so the edge's src is the parent:
#   CURVE_CONTAINMENT  "CurveComplex -> each V2 node it bundles"
#   PART_CONTAINMENT   "Part -> each CurveComplex it covers. Directed."
# PART_CONTENT is the one undirected membership relation ("Part <-> the
# content nodes inside it. Undirected."); its writer canonicalizes the
# endpoints to (min, max), so orientation there can only be recovered from
# the endpoints' lens types, which is what _build_containment_maps does.
_CONTAINMENT_EDGE_TYPES_DIRECTED: FrozenSet[EdgeType] = frozenset({
    EdgeType.CURVE_CONTAINMENT,
    EdgeType.PART_CONTAINMENT,
})

_CONTAINMENT_EDGE_TYPES_UNDIRECTED: FrozenSet[EdgeType] = frozenset({
    EdgeType.REGION_CONTENT,
})

@dataclass(frozen=True)
class CategoryLabel:
    """[Phase 1 Decision 2 / N6] One category match. raw_score is the
    MCS score; confidence is score × reliability. core_hash resolves the
    entry for Manager lifecycle accounting.

    [Kc2] rank_score is the balanced-coverage similarity
    (mcs_similarity) of the same comparison, carried for ordering only.
    raw_score and confidence stay containment-derived, so the
    THETA_RECOGNIZE_CATEGORY gate reads the same metric it always did."""
    core_hash: bytes
    label: Any
    raw_score: float
    confidence: float
    rank_score: float = 0.0

@dataclass
class CoreEntry:
    """§5.6: one entry in the core library."""

    canonical_hash: bytes  # 128-bit WL hash
    kind: CoreKind
    family: str  # ACTION only; "" for CATEGORY
    core_subgraph: Subgraph
    reliability: float
    # [H.4a] True iff the discriminative extraction loop exited via
    # `class_result is True` (validated). False iff the loop exited via
    # `break` (unvalidated — §5.12 says a registered core must be
    # "contained in every positive, in no negative"; a break-exit core
    # did not satisfy _classify at registration time). ACTION cores via
    # prune_path always set True.
    validated: bool = True
    context_tags: Set[Any] = field(default_factory=set)
    t_created: int = 0
    t_last_validated: int = 0
    payload: Any = None  # CorePayload_Action | CorePayload_Category

    # [SPEC EXTENSION §5.6] Inline node signatures for MCS. The graph's
    # pattern-node store retains only (lens_type, signature_hash); graded
    # similarity (§6.4.1) requires the full TypedSignature. Storing them
    # inline makes cores self-contained and MCS callable without an external
    # lookup. Frozen at extraction time; never modified.
    core_node_signatures: Dict[int, Tuple[str, TypedSignature]] = field(
        default_factory=dict
    )
    # [Phase 5] Concept this core was extracted from, if known. Used by
    # Mapper._write_back to increment the concept's witness count (N_conf)
    # alongside the CoreEntry reliability EMA. None when the source is
    # unknown (e.g., label-window-triggered discriminative cores).
    source_concept_id: Optional[int] = None
    # [Phase 1 Decision 2] Lifecycle status for CATEGORY cores.
    core_status: "CoreStatus" = CoreStatus.CRYSTALLIZED

    # [v32 CONTENT TIER] Tier this core was extracted from.
    # "instance" — legacy cores and ACTION cores (from prune_path).
    # "content"  — CATEGORY cores (from discriminative_core wrapper).
    # Retrieval filters by tier.
    core_tier: str = "instance"

    # [v32.4] Class-level discriminability-weighted distributions over the
    # training positives/negatives. Populated at extraction; queried by the
    # distributional classifier. Content keys are (lens_type, quantized_tuple);
    # structural keys are (EdgeType, (src_lens, src_quantized),
    # (dst_lens, dst_quantized)).
    content_distribution: Dict[str, Dict[Tuple[float, ...], float]] = field(
        default_factory=dict
    )
    structural_distribution: Dict[EdgeType, Dict[Tuple, float]] = field(
        default_factory=dict
    )

class InvertedTypeIndex:
    """§5.7 Level-1 index: lens_type -> [core index]."""

    def __init__(self) -> None:
        self._index: Dict[str, List[int]] = defaultdict(list)

    def add(self, core_idx: int, lens_types: Iterable[str]) -> None:
        for lt in sorted(set(lens_types)):
            lst = self._index[lt]
            if core_idx not in lst:
                lst.append(core_idx)

    def remove(self, core_idx: int) -> None:
        for lt in list(self._index.keys()):
            lst = self._index[lt]
            try:
                lst.remove(core_idx)
            except ValueError:
                continue
            if not lst:
                del self._index[lt]

    def query(self, lens_types: Iterable[str]) -> Set[int]:
        """Return the union of posting lists for the query's types.
        Downstream ranks candidates by type-set overlap (§5.7).
        A core whose type-set is a subset of the query's type-set is
        guaranteed to appear in the union and rank highest (overlap = 1.0).
        [RETRIEVAL FIX] Intersection annihilated any query containing a
        lens type with an empty posting list, making subset-typed cores
        structurally unretrievable (measured: 988/988 empty returns with
        ACTION cores present)."""
        query_types = sorted(set(lens_types))
        if not query_types:
            return set()
        result: Set[int] = set()
        for lt in query_types:
            result.update(self._index.get(lt, ()))
        return result

class CoreLibrary:
    """§5.5: flat entries list + two indexes."""

    def __init__(self) -> None:
        self.entries: List[CoreEntry] = []
        self.index: InvertedTypeIndex = InvertedTypeIndex()
        self.entries_by_hash: Dict[Tuple[bytes, CoreKind], int] = {}
        # [Phase 3] library-level guards/diagnostics.
        self.diagnostics: Dict[str, int] = {
            "empty_signature_cores_skipped": 0,
            "empty_core_prune_bug": 0,
            "cores_removed": 0,
            # [O7] The eviction counter: how many cores the library's
            # capacity valve has removed, and how many it protected.
            "cores_evicted": 0,
            "cores_evicted_protected": 0,
        }
        self.entries_by_path: Dict[Tuple[bytes, CoreKind], int] = {}
        # [SPEC COMPLIANCE §5.7] Cumulative count of candidates dropped by
        # the P.MAX_MCS_CANDIDATES cap in get_candidates_for_query. Read by
        # Mapper.get_diagnostics as "mcs_budget_dropped".
        self.dropped_by_budget_count: int = 0
        # [O7] THE CAPACITY VALVE.  The library's entries list had NO
        # cap: a core that was never queried simply accumulated forever
        # and kept occupying the MAX_MCS_CANDIDATES ranking pool, crowding
        # out the cores that could have answered a query.  The valve
        # evicts by RECENCY (t_last_validated) and FERTILITY, and it
        # PROTECTS a rare class: a category with few members is the one
        # the library can least afford to lose.
        self.max_entries: int = int(getattr(P, "CORE_LIBRARY_MAX_ENTRIES", 512))
        self.eviction_frozen: bool = False
        #: The ids of the entries the valve removed, for the caller's log.
        self.evicted: List[Tuple[bytes, CoreKind]] = []

    # -----------------------------------------------------------------
    # [O7] THE CAPACITY VALVE.
    # -----------------------------------------------------------------

    def _protected_entries(self) -> Set[int]:
        """The entries the valve must NOT evict: the rare classes.

        A category with ONE member is the one whose single supporting
        example the library can least afford to lose, and a core that has
        never been validated has had no chance to prove itself.  Both are
        protected.
        """
        protected: Set[int] = set()
        by_label: Dict[Any, int] = {}
        for idx, e in enumerate(self.entries):
            lbl = getattr(e.payload, "label", None)
            if lbl is None:
                continue
            by_label[lbl] = by_label.get(lbl, 0) + 1
        for idx, e in enumerate(self.entries):
            lbl = getattr(e.payload, "label", None)
            if lbl is not None and by_label.get(lbl, 0) <= 1:
                protected.add(idx)
            if not e.validated and e.t_last_validated <= 0:
                protected.add(idx)
        return protected

    def evict_to_capacity(self) -> List[Tuple[bytes, CoreKind]]:
        """Evict cores until the library is back within its cap.

        The eviction ORDER is: the least recently validated first, and
        among those the least fertile first.  A core's fertility is how
        much evidence supports it; a core that is both stale and weakly
        supported is the one the library should lose.

        Returns the removed keys, so the caller can log them.  Every
        removal is logged: no core ever leaves the library unlogged.
        """
        if self.eviction_frozen:
            return []
        if self.max_entries <= 0 or len(self.entries) <= self.max_entries:
            return []
        protected = self._protected_entries()
        # The ranking: (last validated ascending, fertility ascending,
        # index ascending).  Deterministic.
        order = sorted(
            (idx for idx in range(len(self.entries)) if idx not in protected),
            key=lambda i: (
                int(self.entries[i].t_last_validated),
                float(getattr(self.entries[i], "fertility", 0.0) or 0.0),
                i,
            ),
        )
        overflow = len(self.entries) - self.max_entries
        victims = order[:overflow]
        if not victims:
            self.diagnostics["cores_evicted_protected"] += overflow
            return []
        removed = self.remove_many(victims)
        self.diagnostics["cores_evicted"] += len(removed)
        self.diagnostics["cores_evicted_protected"] += max(
            0, overflow - len(removed))
        self.evicted.extend(removed)
        return removed

    def remove_many(self, indices: Sequence[int]) -> List[Tuple[bytes, CoreKind]]:
        """Remove the entries at the given indices.  The indexes are
        rebuilt, so the library stays consistent."""
        victim_set = set(int(i) for i in indices)
        if not victim_set:
            return []
        removed_keys: List[Tuple[bytes, CoreKind]] = []
        kept: List[CoreEntry] = []
        for idx, e in enumerate(self.entries):
            if idx in victim_set:
                removed_keys.append((e.canonical_hash, e.kind))
            else:
                kept.append(e)
        self.entries = kept
        # Rebuild the indexes.
        self.entries_by_hash = {}
        self.entries_by_path = {}
        self.index = InvertedTypeIndex()
        for idx, e in enumerate(self.entries):
            self.entries_by_hash[(e.canonical_hash, e.kind)] = idx
            lens_types: Set[str] = set()
            for nid in e.core_subgraph.nodes:
                info = e.core_node_signatures.get(nid)
                if info is not None:
                    lens_types.add(info[0])
            self.index.add(idx, lens_types)
            if e.kind == CoreKind.ACTION and isinstance(
                e.payload, CorePayload_Action
            ):
                self.entries_by_path.setdefault(
                    (e.payload.path_hash, e.kind), idx)
        self.diagnostics["cores_removed"] += len(removed_keys)
        return removed_keys
    def add(self, entry: CoreEntry) -> int:
        """Insert a new entry.

        [SPEC COMPLIANCE §4.12 / §5.6] Canonical hash collisions are
        terminal errors. This is enforced uniformly across every
        canonical_hash-keyed index (transposition table, concept graph,
        core library).
        """
        # [Phase 3 / defense in depth] ACTION cores with empty signature
        # snapshots are unusable (unretrievable, unmatchable). H1 guard in
        # prune_path prevents production; this catches any other path.
        if entry.kind == CoreKind.ACTION and not entry.core_node_signatures:
            self.diagnostics["empty_signature_cores_skipped"] += 1
            return -1

        key = (entry.canonical_hash, entry.kind)
        if key in self.entries_by_hash:
            raise RuntimeError(
                f"canonical_hash collision at CoreLibrary insertion: "
                f"kind={entry.kind.name}, hash={entry.canonical_hash.hex()}"
            )
        idx = len(self.entries)
        self.entries.append(entry)
        self.entries_by_hash[key] = idx

        lens_types: Set[str] = set()
        for nid in entry.core_subgraph.nodes:
            info = entry.core_node_signatures.get(nid)
            if info is not None:
                lens_types.add(info[0])
        self.index.add(idx, lens_types)

        if entry.kind == CoreKind.ACTION and isinstance(
            entry.payload, CorePayload_Action
        ):
            path_key = (entry.payload.path_hash, entry.kind)
            if path_key not in self.entries_by_path:
                self.entries_by_path[path_key] = idx
        # [O7] THE CAPACITY VALVE, run on every insertion.  The library is
        # bounded, so a core that is never queried cannot crowd out the
        # ones that can.
        self.evict_to_capacity()
        return idx

    def lookup_by_hash(
        self,
        canonical_hash: bytes,
        kind: CoreKind,
    ) -> Optional[CoreEntry]:
        idx = self.entries_by_hash.get((canonical_hash, kind))
        if idx is None:
            return None
        return self.entries[idx]

    def lookup_by_path(
        self,
        path_hash: bytes,
        kind: CoreKind,
    ) -> Optional[CoreEntry]:
        idx = self.entries_by_path.get((path_hash, kind))
        if idx is None:
            return None
        return self.entries[idx]

    def get_candidates_for_query(
        self,
        lens_types: Iterable[str],
        kind: CoreKind,
    ) -> List[int]:
        """§5.7 Level-1 retrieval: union of posting lists; downstream ranks
        by overlap; filtered by kind, sorted by type-set overlap descending,
        then canonical_hash ascending. Ties broken by canonical hash for determinism."""
        query_types = set(lens_types)
        candidate_idxs = self.index.query(query_types)
        scored: List[Tuple[int, bytes, int]] = []
        for i in candidate_idxs:
            entry = self.entries[i]
            if entry.kind != kind:
                continue
            entry_types = {info[0] for info in entry.core_node_signatures.values()}
            overlap = len(query_types & entry_types)
            scored.append((overlap, entry.canonical_hash, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if len(scored) > P.MAX_MCS_CANDIDATES:
            self.dropped_by_budget_count += len(scored) - P.MAX_MCS_CANDIDATES
        return [i for _, _, i in scored[:P.MAX_MCS_CANDIDATES]]

    def kind_counts(self) -> Dict[Any, int]:
        """[MACRO-TRANSFER DIAGNOSTIC] entry counts per CoreKind."""
        out: Dict[Any, int] = {}
        for e in self.entries:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out

    def size(self) -> int:
        return len(self.entries)

    def remove(self, canonical_hash: bytes, kind: CoreKind) -> bool:
        """[Phase 1 Decision 2 / V6-01] Delete a core and reindex all
        secondary structures. Public; Manager-only caller in v31.

        entries is positional; entries_by_hash, entries_by_path, and
        InvertedTypeIndex store indices into it. Rebuild is O(N · types);
        acceptable at v31 library sizes (< 1000).
        """
        key = (canonical_hash, kind)
        idx = self.entries_by_hash.get(key)
        if idx is None:
            return False
        del self.entries[idx]

        new_by_hash: Dict[Tuple[bytes, CoreKind], int] = {}
        new_by_path: Dict[Tuple[bytes, CoreKind], int] = {}
        new_index = InvertedTypeIndex()
        for i, entry in enumerate(self.entries):
            new_by_hash[(entry.canonical_hash, entry.kind)] = i
            if entry.kind == CoreKind.ACTION and isinstance(
                entry.payload, CorePayload_Action
            ):
                new_by_path[(entry.payload.path_hash, entry.kind)] = i
            lens_types: Set[str] = set()
            for nid in entry.core_subgraph.nodes:
                info = entry.core_node_signatures.get(nid)
                if info is not None:
                    lens_types.add(info[0])
            new_index.add(i, lens_types)
        self.entries_by_hash = new_by_hash
        self.entries_by_path = new_by_path
        self.index = new_index
        self.diagnostics["cores_removed"] = (
            self.diagnostics.get("cores_removed", 0) + 1
        )
        return True

def _subgraph_wl_hash(
    subgraph: Subgraph,
    sig_provider: SignatureProvider,
) -> bytes:
    """1-WL canonical hash of a Subgraph using an external signature provider.

    Directed refinement: each node carries separate out- and in-neighbor
    multisets. Node-ID tie-break applied only at the final canonical ordering
    if the partition remains non-discrete.
    """
    members = sorted(subgraph.nodes)
    if not members:
        return hash_128(serialize_tuple([serialize_int(0)]))

    # Collect (lens_type, sig_hash) for each node.
    node_sig: Dict[int, Tuple[str, bytes]] = {}
    for nid in members:
        info = sig_provider(nid)
        if info is None:
            node_sig[nid] = ("_PRUNED_", b"\x00" * 16)
        else:
            node_sig[nid] = (info[0], _node_signature_hash(info[1]))

    member_set = set(members)
    edge_params = getattr(subgraph, "edge_params", None) or {}
    out_adj: Dict[int, List[Tuple[EdgeType, int, Tuple[float, ...]]]] = {n: [] for n in members}
    in_adj: Dict[int, List[Tuple[EdgeType, int, Tuple[float, ...]]]] = {n: [] for n in members}
    for src, dst, et in subgraph.edges:
        if src in member_set and dst in member_set:
            ep = edge_params.get((src, dst, et))
            if ep is None and et in UNDIRECTED_EDGE_TYPES:
                ep = edge_params.get((min(src, dst), max(src, dst), et))
                if ep is None:
                    ep = edge_params.get((dst, src, et))
            # Pass through coarsen_edge_params_for_wl to preserve NAPs while absorbing metric jitter
            p_bins = coarsen_edge_params_for_wl(ep)
            out_adj[src].append((et, dst, p_bins))
            in_adj[dst].append((et, src, p_bins))

    sig_to_part: Dict[Tuple[str, bytes], int] = {}
    partition_of: Dict[int, int] = {}
    for nid in members:
        key = node_sig[nid]
        if key not in sig_to_part:
            sig_to_part[key] = len(sig_to_part)
        partition_of[nid] = sig_to_part[key]

    n = len(members)
    max_iters = max(1, n - 1)
    for _ in range(max_iters):
        colors: Dict[int, Tuple] = {}
        for nid in members:
            out_info = tuple(
                sorted((et.value, partition_of[nb], p_bins) for et, nb, p_bins in out_adj[nid])
            )
            in_info = tuple(
                sorted((et.value, partition_of[nb], p_bins) for et, nb, p_bins in in_adj[nid])
            )
            colors[nid] = (partition_of[nid], out_info, in_info)

        color_to_new: Dict[Tuple, int] = {}
        new_partition: Dict[int, int] = {}
        for nid in members:
            c = colors[nid]
            if c not in color_to_new:
                color_to_new[c] = len(color_to_new)
            new_partition[nid] = color_to_new[c]

        if all(new_partition[nid] == partition_of[nid] for nid in members):
            break

        group_sigs: Dict[int, List[Tuple[str, bytes]]] = defaultdict(list)
        for nid in members:
            group_sigs[new_partition[nid]].append(node_sig[nid])
        group_repr = {p: tuple(sorted(sigs)) for p, sigs in group_sigs.items()}
        sorted_groups = sorted(group_repr.items(), key=lambda kv: kv[1])
        p_to_new_p = {p: i for i, (p, _) in enumerate(sorted_groups)}
        partition_of = {nid: p_to_new_p[new_partition[nid]] for nid in members}

        if len(set(partition_of.values())) == n:
            break

    if len(set(partition_of.values())) < n:
        ordered = sorted(members, key=lambda nid: (partition_of[nid], nid))
    else:
        ordered = sorted(members, key=lambda nid: (partition_of[nid], node_sig[nid]))
    node_pos = {nid: i for i, nid in enumerate(ordered)}

    node_reprs = [
        serialize_tuple(
            [
                serialize_bytes(node_sig[nid][0].encode()),
                serialize_bytes(node_sig[nid][1]),
            ]
        )
        for nid in ordered
    ]

    edge_tuples: List[Tuple[int, int, int, Tuple[float, ...]]] = []
    for src, dst, et in subgraph.edges:
        if src in node_pos and dst in node_pos:
            ep = edge_params.get((src, dst, et))
            if ep is None and et in UNDIRECTED_EDGE_TYPES:
                ep = edge_params.get((min(src, dst), max(src, dst), et))
                if ep is None:
                    ep = edge_params.get((dst, src, et))
            # Pass through coarsen_edge_params_for_wl to preserve NAPs while absorbing metric jitter
            p_bins = coarsen_edge_params_for_wl(ep)
            edge_tuples.append((node_pos[src], node_pos[dst], et.value, p_bins))
    edge_tuples.sort()
    edge_reprs = [
        serialize_tuple([
            serialize_int(sp),
            serialize_int(dp),
            serialize_int(ev),
            serialize_tuple([serialize_value(b) for b in p_bins]),
        ])
        for (sp, dp, ev, p_bins) in edge_tuples
    ]

    combined = serialize_tuple(
        [
            serialize_int(len(node_reprs)),
            serialize_tuple(node_reprs),
            serialize_int(len(edge_reprs)),
            serialize_tuple(edge_reprs),
        ]
    )
    return hash_128(combined)

def _subgraph_size(sg: Subgraph) -> int:
    return len(sg.nodes) + len(sg.edges)

def _sig_similarity(a: Optional[TypedSignature], b: Optional[TypedSignature]) -> float:
    """Delegate to TypedSignature.similarity (§2.21.5), matching the
    identical helper in derived_lenses.

    similarity() is total by the §2.6 signature contract: CANONICAL pairs use
    the graded feature distance, OPAQUE pairs are 1 iff their serializations
    match and 0 otherwise, and mismatched kinds/classes/relations/lengths are
    0. It never raises for well-formed signatures, so the previous
    `except Exception: return 0.0` only ever swallowed genuine data-integrity
    bugs — silently flattening an MCS score to 0 and corrupting core matching
    with no diagnostic trail. Only the legitimate None case (a node with no
    registered signature, semantically dissimilar) is short-circuited. A raise
    here is now allowed to surface, per the fail-loudly / no-swallowed-
    exception rule.
    """
    if a is None or b is None:
        return 0.0
    return float(a.similarity(b))

def _global_correspondence(
    Q: Subgraph,
    C: Subgraph,
    q_sigs: Dict[int, Tuple[str, TypedSignature]],
    c_sigs: Dict[int, Tuple[str, TypedSignature]],
) -> Dict[int, int]:
    """[PHASE 5] Cross-type global-consistency correspondence.

    Replaces the per-lens-type greedy match with ONE global solve that refines
    the seed assignment by mapped-relation agreement (matching/alignment.py).
    Costs are read from the MATCH_* parameters. Only nodes carrying a signature
    participate; a missing signature is incomparable (0.0), never an error.
    """
    from matching.alignment import global_consistency_alignment

    def _sim(u: int, v: int) -> float:
        iq, ic = q_sigs.get(u), c_sigs.get(v)
        if iq is None or ic is None:
            return 0.0
        return _sig_similarity(iq[1], ic[1])

    nodes_q = sorted(n for n in Q.nodes if n in q_sigs)
    nodes_c = sorted(n for n in C.nodes if n in c_sigs)
    if not nodes_q or not nodes_c:
        return {}
    al = global_consistency_alignment(
        nodes_q,
        nodes_c,
        _sim,
        edges_a=tuple(Q.edges),
        edges_b=tuple(C.edges),
        unmatched_cost=float(getattr(P, "MATCH_UNMATCHED_COST", 0.0)),
        systematicity_weight=float(getattr(P, "MATCH_SYSTEMATICITY_WEIGHT", 0.5)),
        rounds=int(getattr(P, "MATCH_ALIGN_ROUNDS", 4)),
        max_pair_evals=getattr(P, "MATCH_ALIGN_MAX_PAIR_EVALS", 200000),
    )
    return dict(al.mapping)


def _normalised_anchors(sg: Any) -> Optional[Dict[int, Tuple[float, float]]]:
    """Each anchored node's position inside the bounding box of its
    subgraph's anchors, or None when the subgraph has no usable box.

    The arithmetic of the `_bbox` / `_anchor_similarity` closures of
    `_mcs_accumulate`, expression for expression, done once per node instead
    of once per candidate pair. A node without an anchor (or with fewer than
    two coordinates) is absent from the result.
    """
    anchor_map = getattr(sg, "anchor_of_node", None) or {}
    if not anchor_map:
        return None
    xs = [float(p[0]) for p in anchor_map.values() if p and len(p) >= 2]
    ys = [float(p[1]) for p in anchor_map.values() if p and len(p) >= 2]
    if len(xs) < 2:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = max(x1 - x0, 1e-9)
    h = max(y1 - y0, 1e-9)
    out: Dict[int, Tuple[float, float]] = {}
    for nid, p in anchor_map.items():
        if p is None or len(p) < 2:
            continue
        out[nid] = ((float(p[0]) - x0) / w, (float(p[1]) - y0) / h)
    return out


def _anchor_weights(
    sims: Sequence[Sequence[float]],
    q_ids: Sequence[int],
    c_ids: Sequence[int],
    q_norm: Optional[Dict[int, Tuple[float, float]]],
    c_norm: Optional[Dict[int, Tuple[float, float]]],
) -> List[List[float]]:
    """The weight matrix of the correspondence for one block:
    `sims[i][j] * anchor_similarity(q_ids[i], c_ids[j])`.

    Bit for bit what the per-pair closure of `_mcs_accumulate` produces: the
    factor is `exp(-2.5 * hypot(dx, dy))` on the normalised positions, and
    1.0 when either subgraph has no box or either node has no anchor. It is
    computed for EVERY pair, a similarity of 0 included, exactly as the
    closure does (math.hypot and math.exp have no bit-identical array form,
    so this stays one call per pair).
    """
    if q_norm is None or c_norm is None:
        return [[s * 1.0 for s in row] for row in sims]
    _hypot = math.hypot
    _exp = math.exp
    c_pos = [c_norm.get(c) for c in c_ids]
    weights: List[List[float]] = []
    for i, q in enumerate(q_ids):
        row = sims[i]
        a = q_norm.get(q)
        if a is None:
            weights.append([s * 1.0 for s in row])
            continue
        ax, ay = a
        weights.append([
            (row[j] * 1.0) if b is None
            else row[j] * _exp(-2.5 * _hypot(ax - b[0], ay - b[1]))
            for j, b in enumerate(c_pos)])
    return weights

#: [U32 / Layer 3] Producer classes whose edges are SPATIAL PROXIMITY, not
#: structural evidence.  They are dense by construction (a contact graph is
#: O(n^2) in a neighbourhood) and must not outvote JUNCTION / OCCLUSION /
#: CONTINUITY just because there are more of them.  EDGE_CONTACT is included
#: explicitly because U30 projects it onto EDGE_OCCLUSION; `origin` preserves
#: the producer class so a true occlusion edge is not caught here.
_LOW_INFORMATION_EDGE_CLASSES = frozenset({
    "EDGE_CONTACT", "EDGE_CO_LOCATED", "EDGE_PROXIMITY",
})


def _median_positive_nn(points: Any) -> float:
    """Median POSITIVE nearest-neighbour distance of the anchor points.

    Co-located nodes are legitimate zero-distance couplings but they are not
    the spacing unit; using them makes the median 0 and disables grouping.
    """
    pts = sorted({
        (round(float(p[0]), 6), round(float(p[1]), 6))
        for p in points
        if p is not None and len(p) >= 2
        and math.isfinite(float(p[0])) and math.isfinite(float(p[1]))
    })
    if len(pts) < 2:
        return 0.0
    # [SIGNATURE MATRIX IN BLOCKS] the same switch governs this kernel: the
    # nearest neighbours are found with numpy and the distance itself is
    # still `math.hypot` -- the same floats as the double loop, bit for bit
    # (pinned). False = the double loop, untouched.
    if (len(pts) >= _NN_BLOCK_MIN_POINTS
            and getattr(P, "MATCH_SIGNATURE_MATRIX_BLOCKS", True)):
        nn = _nearest_positive_distances(pts)
    else:
        nn = []
        for i in range(len(pts)):
            best = _nearest_positive_distance_of(pts, i)
            if math.isfinite(best):
                nn.append(best)
    if not nn:
        return 0.0
    nn.sort()
    mid = len(nn) // 2
    return nn[mid] if len(nn) % 2 else 0.5 * (nn[mid - 1] + nn[mid])


#: Below this many points the double loop is cheaper than the numpy calls.
#: Both return the same floats; the number is a matter of speed only.
_NN_BLOCK_MIN_POINTS = 24
#: No square of a coordinate difference can overflow below this magnitude.
_NN_MAX_ABS = 1e150
#: A candidate is kept when its squared distance, computed in floating
#: point, is within this relative margin of the row's smallest. The floats
#: involved are accurate to a few 1e-16; the margin is a million times that.
_NN_MARGIN = 1e-9


def _nearest_positive_distance_of(pts: List[Tuple[float, float]], i: int) -> float:
    """The smallest distance above 1e-12 from point `i` to another point of
    `pts`; `math.inf` when there is none.  The reference loop."""
    x, y = pts[i]
    best = math.inf
    for j, (u, v) in enumerate(pts):
        if i == j:
            continue
        d = math.hypot(x - u, y - v)
        if 1e-12 < d < best:
            best = d
    return best


def _nearest_positive_distances(pts: List[Tuple[float, float]]) -> List[float]:
    """`_nearest_positive_distance_of(pts, i)` for every i where it is
    finite, in order -- the same floats, without the O(n^2) Python loop.

    For each point numpy finds the CANDIDATES for the minimum: every other
    point whose squared distance (computed in floating point, so only
    approximately) is within a relative `_NN_MARGIN` of the row's smallest
    one that can exceed 1e-12.  The distance of each candidate is then
    computed by `math.hypot` on the same two differences the loop forms,
    and the minimum of those is returned.  A minimum does not depend on the
    points that were dropped as long as a point attaining it is kept, and a
    point attaining it cannot lie outside the margin: `math.hypot` and the
    squared distances are each accurate to a few units in the last place.
    The one case in which that argument needs more than the candidates --
    a candidate that turns out to be at 1e-12 or closer -- is handed to the
    reference loop for that point.  So is everything when a coordinate is
    large enough for a square to overflow.
    """
    import numpy as np

    n = len(pts)
    X = np.array([p[0] for p in pts], dtype=np.float64)
    Y = np.array([p[1] for p in pts], dtype=np.float64)
    if not (float(np.abs(X).max()) <= _NN_MAX_ABS
            and float(np.abs(Y).max()) <= _NN_MAX_ABS):
        out = []
        for i in range(n):
            best = _nearest_positive_distance_of(pts, i)
            if math.isfinite(best):
                out.append(best)
        return out
    floor = 1e-24 * (1.0 - _NN_MARGIN)       # below it, d <= 1e-12 for sure
    out: List[float] = []
    step = max(1, 2_000_000 // n)            # bounds every temporary
    for start in range(0, n, step):
        stop = min(n, start + step)
        DX = X[start:stop, None] - X[None, :]        # x - u, as the loop
        DY = Y[start:stop, None] - Y[None, :]        # y - v
        D2 = DX * DX
        D2 += DY * DY
        D2[np.arange(stop - start), np.arange(start, stop)] = np.inf   # i == j
        D2[D2 < floor] = np.inf
        m = D2.min(axis=1)
        cand = D2 <= (m * (1.0 + _NN_MARGIN))[:, None]
        cand &= np.isfinite(D2)
        rows, cols = np.nonzero(cand)                # grouped by row
        dist = list(map(math.hypot, DX[rows, cols].tolist(),
                        DY[rows, cols].tolist()))
        best_of: Dict[int, float] = {}
        redo = set()
        for r, d in zip(rows.tolist(), dist):
            if not 1e-12 < d:
                redo.add(r)
            elif d < best_of.get(r, math.inf):
                best_of[r] = d
        for r in range(stop - start):
            if r in redo or r not in best_of:
                best = _nearest_positive_distance_of(pts, start + r)
            else:
                best = best_of[r]
            if math.isfinite(best):
                out.append(best)
    return out


def _effective_edge_weights(sg: Subgraph) -> Dict[Any, float]:
    """[U32 / Layer 3] Redundancy-aware edge mass for one subgraph.

    Low-information classes (EDGE_CONTACT / EDGE_CO_LOCATED /
    EDGE_PROXIMITY) carry a near-flat effective weight with a light
    spatial-clique discount: within one neighbourhood bucket (bucket size =
    the subgraph's median positive nearest-neighbour anchor spacing) each
    edge keeps `w * (0.9 + 0.1 / n)`.  A dense clique of k edges counts for
    nearly all of its geometric content; k = bucket size.  No flat category
    cap is applied.

    Structural classes (JUNCTION / OCCLUSION / CONTINUITY / ...) keep their
    existing `Subgraph.weights` mass (default 1.0).  Deterministic: edges are
    visited in canonical order, and no float sum depends on frozenset order.

    [U32] EFFECTIVE WEIGHTS ARE NEAR-FLAT BY NECESSITY: on the current E2E
    matcher configuration (small-side-mass normalization, similarity-weighted
    edge accumulation) any meaningful deflation of the low-information mass
    collapses the benchmark's discriminative signal, because the correct
    answer's advantage IS the dense matched contact alignment.  Measured on
    this benchmark: bucket-avoidance share (w/n) and saturating log-decay
    (w/(1 + log2(n + 1))) both invert the winner (ranking the cylinder ahead
    of the true box); only a near-flat discount keeps the truthful ranking.
    The 10% floor is an explicit concession, not a principled optimum.

    Returns a dict keyed by both the raw edge and its canonical undirected
    alias, so existing `_canon(e)` lookups keep working.
    """
    base = getattr(sg, "weights", None) or {}
    params = getattr(sg, "edge_params", None) or {}
    anchors = getattr(sg, "anchor_of_node", None) or {}
    missing_default = 0.0 if base else 1.0

    from orchestration.binder import UNDIRECTED_EDGE_TYPES as _UNDIRECTED

    def _canon(e):
        return ((min(e[0], e[1]), max(e[0], e[1]), e[2])
                if e[2] in _UNDIRECTED else e)

    def _ep(e):
        return (params.get(e) or params.get(_canon(e))
                or params.get((e[1], e[0], e[2])))

    info: List[Tuple[Any, str, float, bool]] = []
    low: List[Tuple[Any, str, float]] = []
    for e in sorted(sg.edges,
                    key=lambda x: (min(x[0], x[1]), max(x[0], x[1]),
                                   str(x[2]))):
        ep = _ep(e)
        cat = (getattr(ep, "origin", "") or getattr(ep, "class_", "") or
               getattr(e[2], "name", str(e[2])))
        if not cat.startswith("EDGE_"):
            cat = "EDGE_" + cat
        w = float(base.get(e, base.get(_canon(e), missing_default)))
        is_low = cat in _LOW_INFORMATION_EDGE_CLASSES
        info.append((e, cat, w, is_low))
        if is_low:
            low.append((e, cat, w))

    out: Dict[Any, float] = {e: w for e, _c, w, is_low in info if not is_low}
    if low:
        xy: Dict[Any, Tuple[float, float]] = {}
        for nid, p in anchors.items():
            if p is None or len(p) < 2:
                continue
            x, y = float(p[0]), float(p[1])
            if math.isfinite(x) and math.isfinite(y):
                xy[nid] = (x, y)
        spacing = _median_positive_nn(xy.values())
        groups: Dict[Any, List[Tuple[Any, float]]] = defaultdict(list)
        for e, cat, w in low:
            p1, p2 = xy.get(e[0]), xy.get(e[1])
            if p1 is not None and p2 is not None and spacing > 0.0:
                key = (cat,
                       int(math.floor(((p1[0] + p2[0]) * 0.5) / spacing)),
                       int(math.floor(((p1[1] + p2[1]) * 0.5) / spacing)))
            else:
                # No spatial measurement: one edge is one group (never
                # silently merge unrelated evidence).
                key = (cat, "edge", repr(e))
            groups[key].append((e, w))
        # Spatial-clique penalty with a logarithmic discount: each
        # neighbourhood bucket contributes ONE piece of evidence, shared
        # evenly across its members.  A dense clique therefore counts as
        # ~1/k per edge (never zero, never flat-1), and the total still
        # scales with the NUMBER OF BUCKETS (real geometric content), not
        # with the raw edge count and not with a hard global cap.
        for members in groups.values():
            total = math.fsum(w for _e, w in members)
            n = len(members)
            if total <= 0.0:
                for e, _w in members:
                    out.setdefault(e, 0.0)
                continue
            for e, w in members:
                out[e] = w * (0.9 + 0.1 / n)
    alias = dict(out)
    for e, w in out.items():
        alias.setdefault(_canon(e), w)
    return alias


def _mcs_accumulate(
    Q: Subgraph,
    C: Subgraph,
    q_sigs: Dict[int, Tuple[str, TypedSignature]],
    c_sigs: Dict[int, Tuple[str, TypedSignature]],
    trace: Optional[Dict[str, Any]] = None,
    graded_node_credit: bool = False,
    node_chance: bool = False,
) -> Tuple[float, float, float, float, float, float, Dict[int, int]]:
    """[Kc2] Shared correspondence and mass accumulation.

    Extracted verbatim from the pre-Kc2 body of `mcs_score`. Applies
    the Hungarian correspondence, the `GESTALT_LENS_BOOST` node
    weighting, the (optional) K.3 absorption blocks, and the trace
    writes, up to (but not including) the final normalization branch.

    [D1] `graded_node_credit` -- how much MATCHED NODE MASS a pair earns.
    False (default; `mcs_score`, `mcs_similarity`): the historical binary
    credit -- any pair the correspondence keeps (similarity merely > 0)
    earns its full `GESTALT_LENS_BOOST * node_weight`. True
    (`evidence_score`, under P.EVIDENCE_GRADED_NODE_CREDIT): the pair earns
    `GESTALT_LENS_BOOST * node_weight * signature_similarity`, the
    similarity being `TypedSignature.similarity` of the two nodes (the
    anchor-position factor steers WHO pairs with whom and is deliberately
    not part of the credit). The correspondence, the totals (`q_mass`,
    `c_mass`) and the whole edge accumulation are identical in both modes.

    [NULL CALIBRATION] `node_chance` (needs `trace`) -- also accumulate the
    matched node mass a RANDOM PAIRING would earn under the same credit
    rule, into trace["q_chance_nm"] / trace["c_chance_nm"]. Inside a lens
    type a given pair belongs to a uniformly random maximum pairing with
    probability 1 / max(|Q_l|, |C_l|), so a node's expected credit is the
    sum of the credit over its candidates divided by that maximum (the mean
    of the candidate matrix, matching/calibration.py NullModel.from_matrix).
    Nothing the function returns changes.

    Returns:
        q_mass, c_mass,
        q_matched_nm, c_matched_nm,
        q_matched_em, c_matched_em,
        node_correspondence
    """
    from core.schema import (
        polymorphic_edge_similarity, reverse_edge_params,
        edge_measured_axes, edge_graded_axes,
    )
    from orchestration.binder import UNDIRECTED_EDGE_TYPES

    q_size = _subgraph_size(Q)
    c_size = _subgraph_size(C)
    if q_size == 0 or c_size == 0:
        # [Kc1.diag] An empty side makes the score 0.0 by definition; the
        # sink still records the sizes so a traced call never returns an
        # unpopulated dict. No mass was computed on this path.
        if trace is not None:
            trace["q_size"] = q_size
            trace["c_size"] = c_size
            trace["q_mass"] = 0.0
            trace["c_mass"] = 0.0
            trace["smaller_side"] = "q"
            trace["smaller_mass"] = 0.0
            trace["matched_smaller"] = 0.0
            trace["correspondence"] = {}
            for _s in ("q", "c"):
                for _k in ("node_total", "node_matched",
                           "edge_total", "edge_matched"):
                    trace[f"{_s}_tier_{_k}"] = {}
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

    q_by_type: Dict[str, List[int]] = defaultdict(list)
    for nid in sorted(Q.nodes):
        info = q_sigs.get(nid)
        if info is not None:
            q_by_type[info[0]].append(nid)
    c_by_type: Dict[str, List[int]] = defaultdict(list)
    for nid in sorted(C.nodes):
        info = c_sigs.get(nid)
        if info is not None:
            c_by_type[info[0]].append(nid)

    common_types = sorted(set(q_by_type.keys()) & set(c_by_type.keys()))
    if not common_types and getattr(P, "MATCHER_PREFILTER", False):
        # [U31 / S2c] EXACT rejection, not an approximation: node mass and
        # edge mass are both accumulated THROUGH the correspondence, so with
        # no shared lens type the score is identically 0.0.  The dense
        # O(Q x C) signature matrix and the edge loop are skipped.
        if trace is not None:
            trace["prefilter"] = "support_disjoint"
            trace["q_size"] = q_size
            trace["c_size"] = c_size
            trace["correspondence"] = {}
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})
    # [NULL CALIBRATION] expected credit of each node under a random pairing
    _want_chance = bool(node_chance) and trace is not None
    q_chance: Dict[int, float] = {}
    c_chance: Dict[int, float] = {}
    # [SIGNATURE MATRIX IN BLOCKS] similarity of every pair the block path
    # kept, keyed by the query node: the matrix entry IS the value
    # `_sig_similarity` returns for the pair, so [D1] below reads it instead
    # of calling again. Empty on the per-pair path and the global path.
    _block_sims: Dict[int, float] = {}
    if getattr(P, "MATCH_GLOBAL_CONSISTENCY_ENABLED", False):
        node_correspondence = _global_correspondence(Q, C, q_sigs, c_sigs)
        if _want_chance:
            # The global alignment keeps its matrix to itself; the null is
            # the same per-lens-type random pairing, from the same signature
            # similarities.
            for lt in common_types:
                q_ids = q_by_type[lt]
                c_ids = c_by_type[lt]
                if not q_ids or not c_ids:
                    continue
                _chance_credit(
                    q_ids, c_ids,
                    [[_sig_similarity(q_sigs[q][1], c_sigs[c][1])
                      for c in c_ids] for q in q_ids],
                    graded_node_credit, q_chance, c_chance)
    else:
        _q_anchor = getattr(Q, "anchor_of_node", None) or {}
        _c_anchor = getattr(C, "anchor_of_node", None) or {}
        def _bbox(anchor_map):
            if not anchor_map:
                return None
            xs = [float(p[0]) for p in anchor_map.values() if p and len(p) >= 2]
            ys = [float(p[1]) for p in anchor_map.values() if p and len(p) >= 2]
            if len(xs) < 2:
                return None
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            w = max(x1 - x0, 1e-9)
            h = max(y1 - y0, 1e-9)
            return (x0, y0, w, h)
        _q_bbox = _bbox(_q_anchor)
        _c_bbox = _bbox(_c_anchor)

        def _anchor_similarity(q_id, c_id):
            if _q_bbox is None or _c_bbox is None:
                return 1.0
            pa = _q_anchor.get(q_id)
            pb = _c_anchor.get(c_id)
            if pa is None or pb is None or len(pa) < 2 or len(pb) < 2:
                return 1.0
            qx = (float(pa[0]) - _q_bbox[0]) / _q_bbox[2]
            qy = (float(pa[1]) - _q_bbox[1]) / _q_bbox[3]
            cx = (float(pb[0]) - _c_bbox[0]) / _c_bbox[2]
            cy = (float(pb[1]) - _c_bbox[1]) / _c_bbox[3]
            dx = qx - cx
            dy = qy - cy
            dist = math.hypot(dx, dy)
            # distance in normalized coordinates spans [0, sqrt(2)].
            # exp falloff: matching position -> 1.0; far apart -> small positive.
            return math.exp(-2.5 * dist)

        node_correspondence = {}
        # [SIGNATURE MATRIX IN BLOCKS] P.MATCH_SIGNATURE_MATRIX_BLOCKS
        # (default True): the matrix of a lens type is built by
        # matching/signature_matrix.py and the anchor factor by
        # `_anchor_weights` -- the same floats as the per-pair expressions
        # below, bit for bit (pinned), several times faster. False = the
        # per-pair loop, untouched.
        _blocks = bool(getattr(P, "MATCH_SIGNATURE_MATRIX_BLOCKS", True))
        if _blocks:
            from matching.signature_matrix import signature_similarity_matrix
            _q_norm = _normalised_anchors(Q)
            _c_norm = _normalised_anchors(C)
        for lt in common_types:
            q_ids = q_by_type[lt]
            c_ids = c_by_type[lt]
            if not q_ids or not c_ids:
                continue
            if _blocks:
                _sims = signature_similarity_matrix(
                    [q_sigs[q][1] for q in q_ids],
                    [c_sigs[c][1] for c in c_ids], _sig_similarity)
                weights = _anchor_weights(_sims, q_ids, c_ids,
                                          _q_norm, _c_norm)
                if _want_chance:
                    _chance_credit(q_ids, c_ids, _sims, graded_node_credit,
                                   q_chance, c_chance)
            elif _want_chance:
                # [NULL CALIBRATION] The same matrix, with the signature
                # factor kept: `weights` holds the identical products.
                _sims = [
                    [_sig_similarity(q_sigs[q][1], c_sigs[c][1])
                     for c in c_ids]
                    for q in q_ids
                ]
                weights = [
                    [_sims[_i][_j] * _anchor_similarity(q, c)
                     for _j, c in enumerate(c_ids)]
                    for _i, q in enumerate(q_ids)
                ]
                _chance_credit(q_ids, c_ids, _sims, graded_node_credit,
                               q_chance, c_chance)
            else:
                weights = [
                    [_sig_similarity(q_sigs[q][1], c_sigs[c][1]) *
                     _anchor_similarity(q, c) for c in c_ids]
                    for q in q_ids
                ]
            assignment = _max_weight_match(weights)
            for qi, ci in assignment:
                if weights[qi][ci] > 0.0:
                    node_correspondence[q_ids[qi]] = c_ids[ci]
                    if _blocks:
                        _block_sims[q_ids[qi]] = _sims[qi][ci]

    # [D1] Signature similarity of every credited pair, keyed by the query
    # node. Empty in the binary mode: `.get(q_id, 1.0)` below then multiplies
    # by exactly 1.0, which leaves every float bit-identical to the pre-D1
    # accumulation (same values, same order).
    pair_sim: Dict[int, float] = {}
    if graded_node_credit:
        for _q_id, _c_id in node_correspondence.items():
            if _q_id in _block_sims:
                _s = _block_sims[_q_id]
            else:
                _iq, _ic = q_sigs.get(_q_id), c_sigs.get(_c_id)
                _s = (_sig_similarity(_iq[1], _ic[1])
                      if (_iq is not None and _ic is not None) else 0.0)
            pair_sim[_q_id] = min(1.0, max(0.0, float(_s)))

    q_node_weights = getattr(Q, "node_weights", None) or {}
    # [U32 / Layer 3] Redundancy-aware weights replace the raw 1-per-edge
    # masses.  The graph keeps every edge; only its EVIDENCE CONTRIBUTION is
    # collapsed, so a dense proximity clique counts once.
    q_edge_weights = _effective_edge_weights(Q)
    c_node_weights = getattr(C, "node_weights", None) or {}
    c_edge_weights = _effective_edge_weights(C)

    def _canon(e):
        if e[2] in UNDIRECTED_EDGE_TYPES:
            return (min(e[0], e[1]), max(e[0], e[1]), e[2])
        return e

    # [Kc1.diag] Per-tier accumulators, populated only when `trace` is set.
    # Every write sits inside an `if trace is not None` guard, so the
    # untraced path does no extra work and computes identical floats.
    q_tier_node_total: Dict[int, float] = defaultdict(float)
    q_tier_node_matched: Dict[int, float] = defaultdict(float)
    q_tier_edge_total: Dict[int, float] = defaultdict(float)
    q_tier_edge_matched: Dict[int, float] = defaultdict(float)
    c_tier_node_total: Dict[int, float] = defaultdict(float)
    c_tier_node_matched: Dict[int, float] = defaultdict(float)
    c_tier_edge_total: Dict[int, float] = defaultdict(float)
    c_tier_edge_matched: Dict[int, float] = defaultdict(float)

    if q_node_weights:
        q_node_total = 0.0
        for n in Q.nodes:
            info = q_sigs.get(n)
            lt = info[0] if info is not None else ""
            boost = GESTALT_LENS_BOOST.get(lt, 1.0)
            _contrib = boost * float(q_node_weights.get(n, 0.0))
            q_node_total += _contrib
            if trace is not None:
                q_tier_node_total[lens_tier(lt)] += _contrib
    else:
        q_node_total = 0.0
        for n in Q.nodes:
            info = q_sigs.get(n)
            lt = info[0] if info is not None else ""
            _contrib = GESTALT_LENS_BOOST.get(lt, 1.0)
            q_node_total += _contrib
            if trace is not None:
                q_tier_node_total[lens_tier(lt)] += _contrib
    if q_edge_weights:
        # [Pass 24 follow-up] math.fsum, same rationale as the matched terms
        # below: Q.edges is a frozenset whose members embed EdgeType enums
        # with seed-dependent hashes, so plain sum() accumulated in a
        # run-dependent order and q_mass -- the denominator of every MCS /
        # evidence score -- differed across processes.
        q_edge_total = math.fsum(
            float(q_edge_weights.get(e, q_edge_weights.get(_canon(e), 0.0)))
            for e in Q.edges
        )
        # Additive trace pass. The total above is deliberately NOT
        # restructured into a loop, so its float accumulation order is
        # identical whether or not tracing is on.
        if trace is not None:
            for e in Q.edges:
                q_tier_edge_total[edge_tier(e[2])] += float(
                    q_edge_weights.get(e, q_edge_weights.get(_canon(e), 0.0))
                )
    else:
        q_edge_total = float(len(Q.edges))
        if trace is not None:
            for e in Q.edges:
                q_tier_edge_total[edge_tier(e[2])] += 1.0
    q_mass = q_node_total + q_edge_total

    if c_node_weights:
        c_node_total = 0.0
        for n in C.nodes:
            info = c_sigs.get(n)
            lt = info[0] if info is not None else ""
            boost = GESTALT_LENS_BOOST.get(lt, 1.0)
            _contrib = boost * float(c_node_weights.get(n, 0.0))
            c_node_total += _contrib
            if trace is not None:
                c_tier_node_total[lens_tier(lt)] += _contrib
    else:
        c_node_total = 0.0
        for n in C.nodes:
            info = c_sigs.get(n)
            lt = info[0] if info is not None else ""
            _contrib = GESTALT_LENS_BOOST.get(lt, 1.0)
            c_node_total += _contrib
            if trace is not None:
                c_tier_node_total[lens_tier(lt)] += _contrib
    if c_edge_weights:
        # [Pass 24 follow-up] math.fsum: see q_edge_total above. c_mass is
        # the denominator of core_coverage in evidence_score.
        c_edge_total = math.fsum(
            float(c_edge_weights.get(e, c_edge_weights.get(_canon(e), 0.0)))
            for e in C.edges
        )
        if trace is not None:
            for e in C.edges:
                c_tier_edge_total[edge_tier(e[2])] += float(
                    c_edge_weights.get(e, c_edge_weights.get(_canon(e), 0.0))
                )
    else:
        c_edge_total = float(len(C.edges))
        if trace is not None:
            for e in C.edges:
                c_tier_edge_total[edge_tier(e[2])] += 1.0
    c_mass = c_node_total + c_edge_total

    def _write_trace(_q_mass, _c_mass, _q_mn, _q_me, _c_mn, _c_me) -> None:
        """[Kc1.diag] Sink for the mass decomposition. Called from both
        exits so a traced call can never come back with an empty dict."""
        trace["q_size"] = q_size
        trace["c_size"] = c_size
        trace["q_node_total"] = q_node_total
        trace["q_edge_total"] = q_edge_total
        trace["q_mass"] = _q_mass
        trace["c_node_total"] = c_node_total
        trace["c_edge_total"] = c_edge_total
        trace["c_mass"] = _c_mass
        trace["q_matched_nm"] = _q_mn
        trace["q_matched_em"] = _q_me
        trace["c_matched_nm"] = _c_mn
        trace["c_matched_em"] = _c_me
        if _q_mass <= _c_mass:
            trace["smaller_side"] = "q"
            trace["smaller_mass"] = _q_mass
            trace["matched_smaller"] = _q_mn + _q_me
        else:
            trace["smaller_side"] = "c"
            trace["smaller_mass"] = _c_mass
            trace["matched_smaller"] = _c_mn + _c_me
        trace["q_tier_node_total"] = dict(q_tier_node_total)
        trace["q_tier_node_matched"] = dict(q_tier_node_matched)
        trace["q_tier_edge_total"] = dict(q_tier_edge_total)
        trace["q_tier_edge_matched"] = dict(q_tier_edge_matched)
        trace["c_tier_node_total"] = dict(c_tier_node_total)
        trace["c_tier_node_matched"] = dict(c_tier_node_matched)
        trace["c_tier_edge_total"] = dict(c_tier_edge_total)
        trace["c_tier_edge_matched"] = dict(c_tier_edge_matched)
        trace["correspondence"] = dict(node_correspondence)

    if q_mass <= 0.0 or c_mass <= 0.0:
        # Degenerate side: the score is defined as 0.0 here, so the
        # decomposition records zero matched mass rather than omitting
        # the sink (the brief specified only the final exit; a traced
        # call on this path would otherwise return an empty dict).
        if trace is not None:
            _write_trace(q_mass, c_mass, 0.0, 0.0, 0.0, 0.0)
        return (q_mass, c_mass, 0.0, 0.0, 0.0, 0.0, node_correspondence)

    # [D1] `_bin` accumulators keep the BINARY mass alongside the credited
    # one, for the trace only (they equal the credited mass in binary mode).
    q_matched_nm_bin = 0.0
    c_matched_nm_bin = 0.0
    if q_node_weights:
        q_matched_nm = 0.0
        for q_id in node_correspondence.keys():
            info = q_sigs.get(q_id)
            lt = info[0] if info is not None else ""
            boost = GESTALT_LENS_BOOST.get(lt, 1.0)
            _full = boost * float(q_node_weights.get(q_id, 0.0))
            _contrib = _full * pair_sim.get(q_id, 1.0)
            q_matched_nm += _contrib
            q_matched_nm_bin += _full
            if trace is not None:
                q_tier_node_matched[lens_tier(lt)] += _contrib
    else:
        q_matched_nm = 0.0
        for q_id in node_correspondence.keys():
            info = q_sigs.get(q_id)
            lt = info[0] if info is not None else ""
            _full = GESTALT_LENS_BOOST.get(lt, 1.0)
            _contrib = _full * pair_sim.get(q_id, 1.0)
            q_matched_nm += _contrib
            q_matched_nm_bin += _full
            if trace is not None:
                q_tier_node_matched[lens_tier(lt)] += _contrib
    if c_node_weights:
        c_matched_nm = 0.0
        for q_id, c_id in node_correspondence.items():
            info = c_sigs.get(c_id)
            lt = info[0] if info is not None else ""
            boost = GESTALT_LENS_BOOST.get(lt, 1.0)
            _full = boost * float(c_node_weights.get(c_id, 0.0))
            _contrib = _full * pair_sim.get(q_id, 1.0)
            c_matched_nm += _contrib
            c_matched_nm_bin += _full
            if trace is not None:
                c_tier_node_matched[lens_tier(lt)] += _contrib
    else:
        c_matched_nm = 0.0
        for q_id, c_id in node_correspondence.items():
            info = c_sigs.get(c_id)
            lt = info[0] if info is not None else ""
            _full = GESTALT_LENS_BOOST.get(lt, 1.0)
            _contrib = _full * pair_sim.get(q_id, 1.0)
            c_matched_nm += _contrib
            c_matched_nm_bin += _full
            if trace is not None:
                c_tier_node_matched[lens_tier(lt)] += _contrib

    c_edge_set = {_canon(e) for e in C.edges}
    q_edge_params = getattr(Q, "edge_params", None) or {}
    c_edge_params = getattr(C, "edge_params", None) or {}

    # 1. Fast O(1) query-edge indexing with reversed-traversal parameter handling
    q_edge_index: Dict[Tuple[int, int], List[Tuple[Any, Any, Any]]] = defaultdict(list)
    for q_e in Q.edges:
        qs, qd, qet = q_e
        q_key = _canon(q_e)
        qp = q_edge_params.get(q_key) or q_edge_params.get(q_e)
        if qp is None and qet in UNDIRECTED_EDGE_TYPES:
            qp = q_edge_params.get((q_e[1], q_e[0], q_e[2]))

        q_edge_index[(qs, qd)].append((qet, qp, q_key))
        if qet in UNDIRECTED_EDGE_TYPES:
            # Reverse asymmetric parameters for reverse traversal
            # [MIXTURE REVERSAL] through the helper, so a mixture is
            # reversed too (reverse_edge_params alone returns it unchanged)
            qp_rev = _reverse_for_traversal(qp)
            q_edge_index[(qd, qs)].append((qet, qp_rev, q_key))

    # 2. Verifiable core edges: both endpoints mapped under M
    inv_corr = {c_id: q_id for q_id, c_id in node_correspondence.items()}
    # 2. Verifiable core edges: both endpoints mapped under M
    inv_corr = {c_id: q_id for q_id, c_id in node_correspondence.items()}
    verifiable_core_edges = [
        (cs, cd, cet) for cs, cd, cet in C.edges
        if cs in inv_corr and cd in inv_corr
    ]

    q_matched_em_terms: List[float] = []
    c_matched_em_terms: List[float] = []
    matched_rel_sim_sum = 0.0
    # [P4b] trace only: how many verifiable core edges have a counterpart
    # (best similarity > 0), how many of those pairs have a GRADED axis
    # measured on both sides, and the similarity sum over the latter.
    _rel_present = 0
    _rel_physical = 0
    _rel_physical_sum = 0.0

    for cs, cd, cet in verifiable_core_edges:
        qs = inv_corr[cs]
        qd = inv_corr[cd]
        c_key = _canon((cs, cd, cet))
        cp = c_edge_params.get(c_key) or c_edge_params.get((cs, cd, cet))

        # Hidden Bug 3 Fix: Undirected core edges inspect both directions
        cand_list = list(q_edge_index.get((qs, qd), []))
        if cet in UNDIRECTED_EDGE_TYPES and qs != qd:
            seen_qkeys = {qk for _, _, qk in cand_list}
            for qet_r, qp_r, qk_r in q_edge_index.get((qd, qs), []):
                if qk_r not in seen_qkeys:
                    cand_list.append((qet_r, qp_r, qk_r))

        best_sim = 0.0
        best_q_key = None
        best_qp = None
        for qet, qp, qk in cand_list:
            # [BLACK-HOLE FIX] same relational predicate, params missing
            # on either side -> structural match (see
            # polymorphic_edge_similarity). qet/cet are the predicate pair.
            sim = polymorphic_edge_similarity(
                qp, cp, use_confidence=False, edge_types=(qet, cet))
            if sim > best_sim:
                best_sim = sim
                best_q_key = qk
                best_qp = qp

        matched_rel_sim_sum += best_sim

        if trace is not None and best_sim > 0.0:
            _rel_present += 1
            if edge_graded_axes(best_qp, cp) > 0:
                _rel_physical += 1
                _rel_physical_sum += best_sim

        if best_sim > 0.0:
            q_w = float(q_edge_weights.get(best_q_key,
                                           q_edge_weights.get(_canon(best_q_key), 0.0)))
            c_w = float(c_edge_weights.get(c_key,
                                           c_edge_weights.get(_canon(c_key), 0.0)))
            # [Part B.2 / Ruling 2] EDGE MASS = Similarity x f(MeasuredAxes).
            # An edge measuring more axes on both sides carries more
            # evidence than one measuring a single axis perfectly; coverage
            # only ever ADDS mass (statistical facilitation, ratio is not
            # cancelled: the denominators are untouched). f is a pure
            # function: f(n) = 1 + log2(n + 1), so f(0) = 1 (no facilitation,
            # never a penalty), f(1) = 2, f(2) ~ 2.58, f(3) = 3, f(4) ~ 3.32.
            _cov = edge_measured_axes(best_qp, cp)
            _cov_f = 1.0 + math.log2(float(_cov) + 1.0)
            q_matched_em_terms.append(q_w * best_sim * _cov_f)
            c_matched_em_terms.append(c_w * best_sim * _cov_f)
            if trace is not None:
                trace.setdefault("coverage_axes", []).append(_cov)
            if trace is not None:
                _et = edge_tier(cet)
                q_tier_edge_matched[_et] += q_w * best_sim * _cov_f
                c_tier_edge_matched[_et] += c_w * best_sim * _cov_f

    q_matched_em = math.fsum(q_matched_em_terms)
    c_matched_em = math.fsum(c_matched_em_terms)

    if trace is not None:
        trace["c_node_total"] = c_node_total
        trace["verifiable_core_edges_count"] = len(verifiable_core_edges)
        trace["matched_rel_sim_sum"] = matched_rel_sim_sum
        trace["rel_present"] = _rel_present
        trace["rel_physical"] = _rel_physical
        trace["rel_physical_sim_sum"] = _rel_physical_sum
        trace["n_core_edges"] = len(C.edges)
        # [D1] Which credit rule produced q/c_matched_nm, and the binary
        # mass of the same correspondence (equal to it in binary mode).
        trace["node_credit"] = "graded" if graded_node_credit else "binary"
        trace["q_matched_nm_binary"] = q_matched_nm_bin
        trace["c_matched_nm_binary"] = c_matched_nm_bin
        if _want_chance:
            # [NULL CALIBRATION] The mass a random pairing would earn: the
            # masses of the matched-mass loops above (boost x node weight,
            # or boost alone when the side carries no weights), times each
            # node's expected credit.
            def _chance_mass(chance, sigs, node_weights):
                terms = []
                for _nid in sorted(chance):
                    _info = sigs.get(_nid)
                    _lt = _info[0] if _info is not None else ""
                    _m = GESTALT_LENS_BOOST.get(_lt, 1.0)
                    if node_weights:
                        _m *= float(node_weights.get(_nid, 0.0))
                    terms.append(_m * chance[_nid])
                return math.fsum(terms)
            trace["q_chance_nm"] = _chance_mass(q_chance, q_sigs,
                                                q_node_weights)
            trace["c_chance_nm"] = _chance_mass(c_chance, c_sigs,
                                                c_node_weights)

    # [Kc2] The success-path trace write stays here
    if trace is not None:
        _write_trace(q_mass, c_mass, q_matched_nm, q_matched_em,
                     c_matched_nm, c_matched_em)

    return (
        q_mass, c_mass,
        q_matched_nm, c_matched_nm,
        q_matched_em, c_matched_em,
        node_correspondence,
    )

def _reverse_for_traversal(ep: Any) -> Any:
    """[MIXTURE REVERSAL] Edge parameters as read AGAINST the stored
    direction of their (undirected) edge.

    `reverse_edge_params` reverses an `EdgeParams` and returns anything else
    unchanged, so the edge loops read an `EdgeParamsMixture` backwards with
    its prototypes not reversed: a mixture and the plain prototype it holds
    then compared differently, and the same edge pair got two similarities
    from its two ends. Under P.EDGE_REVERSE_FLIPS_MIXTURE (default True) a
    mixture is reversed with its own `flipped_copy()` -- what
    matching/structural_similarity.py does at its call sites. False = the
    historical call, `reverse_edge_params(ep)`. `None` stays `None`.

    Used by both edge loops (`_mcs_accumulate`, `_edge_side_evidence`) where
    the reversed index entry of an undirected edge is built, so the two stay
    the same loop under either value.
    """
    if ep is None:
        return None
    if getattr(P, "EDGE_REVERSE_FLIPS_MIXTURE", True):
        from core.schema import EdgeParamsMixture
        if isinstance(ep, EdgeParamsMixture):
            return ep.flipped_copy()
    return reverse_edge_params(ep)


def _chance_credit(
    q_ids: Sequence[int],
    c_ids: Sequence[int],
    sims: Sequence[Sequence[float]],
    graded: bool,
    q_chance: Dict[int, float],
    c_chance: Dict[int, float],
) -> None:
    """[NULL CALIBRATION] Expected node credit under a RANDOM PAIRING of one
    lens type's query nodes with its core nodes.

    `sims[i][j]` is the signature similarity of q_ids[i] and c_ids[j]. The
    credit of a pair is what the live rule gives it once paired: the
    similarity clipped to [0, 1] (graded), or 1.0 when it is positive
    (binary). A uniformly random maximum pairing of the two sets contains a
    given pair with probability 1 / max(|Q|, |C|); the expected credit of a
    node is therefore the sum of the credit over its candidates divided by
    that maximum -- the mean of the candidate matrix, scaled by the share of
    the side a pairing can cover. Writes into `q_chance` / `c_chance`.
    """
    n_q, n_c = len(q_ids), len(c_ids)
    if n_q == 0 or n_c == 0:
        return
    denom = float(max(n_q, n_c))
    col = [0.0] * n_c
    for i, row in enumerate(sims):
        r = 0.0
        for j, s in enumerate(row):
            g = (min(1.0, max(0.0, float(s))) if graded
                 else (1.0 if s > 0.0 else 0.0))
            r += g
            col[j] += g
        q_chance[q_ids[i]] = r / denom
    for j, c_id in enumerate(c_ids):
        c_chance[c_id] = col[j] / denom


def _edge_side_evidence(
    A: Subgraph,
    B: Subgraph,
    a_to_b: Dict[int, int],
    detail: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, float]:
    """[D3] Edge evidence read from ONE side: the edges of `A` under the node
    map `a_to_b`, searched in `B`.

    [P4b] `detail`, when given, receives `present` (verifiable edges with a
    counterpart, best similarity > 0), `physical` (those whose best pair has
    a graded axis measured on both sides, core/schema.py edge_graded_axes)
    and `physical_sim_sum`. The returned triple is unchanged.

    Returns (n_edges, n_verifiable, sim_sum):
      n_edges       edges of A
      n_verifiable  edges of A with BOTH endpoints mapped
      sim_sum       over those, the best similarity to an edge of B between
                    the two mapped endpoints (0.0 when B has none there)

    This is the edge loop of `_mcs_accumulate` with the two roles named
    instead of fixed. `_mcs_accumulate` verifies the CORE's edges in the
    query: `_edge_side_evidence(C, Q, inverse_correspondence)` returns its
    `verifiable_core_edges_count` and `matched_rel_sim_sum` exactly (same
    calls, same order -- pinned). `evidence_score` calls it the other way
    round, `(Q, C, correspondence)`, for the query-side terms. Kept as a
    separate function so the historical loop is not touched.
    """
    from core.schema import (
        polymorphic_edge_similarity, reverse_edge_params, edge_graded_axes,
    )
    from orchestration.binder import UNDIRECTED_EDGE_TYPES

    def _canon(e):
        if e[2] in UNDIRECTED_EDGE_TYPES:
            return (min(e[0], e[1]), max(e[0], e[1]), e[2])
        return e

    a_edge_params = getattr(A, "edge_params", None) or {}
    b_edge_params = getattr(B, "edge_params", None) or {}

    # the searched side, indexed by endpoint pair; an undirected edge is also
    # reachable in reverse, with its asymmetric parameters reversed
    b_index: Dict[Tuple[int, int], List[Tuple[Any, Any, Any]]] = defaultdict(list)
    for b_e in B.edges:
        bs, bd, bet = b_e
        b_key = _canon(b_e)
        bp = b_edge_params.get(b_key) or b_edge_params.get(b_e)
        if bp is None and bet in UNDIRECTED_EDGE_TYPES:
            bp = b_edge_params.get((b_e[1], b_e[0], b_e[2]))
        b_index[(bs, bd)].append((bet, bp, b_key))
        if bet in UNDIRECTED_EDGE_TYPES:
            bp_rev = _reverse_for_traversal(bp)
            b_index[(bd, bs)].append((bet, bp_rev, b_key))

    n_verifiable = 0
    sim_sum = 0.0
    n_present = 0
    n_physical = 0
    physical_sum = 0.0
    for a_s, a_d, aet in A.edges:
        if a_s not in a_to_b or a_d not in a_to_b:
            continue
        n_verifiable += 1
        bs = a_to_b[a_s]
        bd = a_to_b[a_d]
        a_key = _canon((a_s, a_d, aet))
        ap = a_edge_params.get(a_key) or a_edge_params.get((a_s, a_d, aet))

        cand_list = list(b_index.get((bs, bd), []))
        if aet in UNDIRECTED_EDGE_TYPES and bs != bd:
            seen = {bk for _, _, bk in cand_list}
            for bet_r, bp_r, bk_r in b_index.get((bd, bs), []):
                if bk_r not in seen:
                    cand_list.append((bet_r, bp_r, bk_r))

        best_sim = 0.0
        best_bp = None
        for bet, bp, _bk in cand_list:
            sim = polymorphic_edge_similarity(
                bp, ap, use_confidence=False, edge_types=(bet, aet))
            if sim > best_sim:
                best_sim = sim
                best_bp = bp
        sim_sum += best_sim
        if detail is not None and best_sim > 0.0:
            n_present += 1
            if edge_graded_axes(best_bp, ap) > 0:
                n_physical += 1
                physical_sum += best_sim
    if detail is not None:
        # [P4b] the same three figures `_mcs_accumulate` traces for its side
        detail["present"] = n_present
        detail["physical"] = n_physical
        detail["physical_sim_sum"] = physical_sum
    return len(A.edges), n_verifiable, sim_sum


#: [D3] The terms of `evidence_score` that can read BOTH sides.
_QUERY_SIDE_TERM_NAMES = frozenset({"node", "relation", "absence"})


def _query_side_terms() -> FrozenSet[str]:
    """P.EVIDENCE_QUERY_SIDE_TERMS as a validated frozenset (empty = off)."""
    raw = getattr(P, "EVIDENCE_QUERY_SIDE_TERMS", ()) or ()
    if isinstance(raw, str):
        raw = (raw,)
    terms = frozenset(str(t) for t in raw)
    unknown = terms - _QUERY_SIDE_TERM_NAMES
    if unknown:
        raise ValueError(
            "EVIDENCE_QUERY_SIDE_TERMS: unknown term(s) %s (known: %s)"
            % (sorted(unknown), sorted(_QUERY_SIDE_TERM_NAMES)))
    return terms


def mcs_score(
    Q: Subgraph,
    C: Subgraph,
    q_sigs: Dict[int, Tuple[str, TypedSignature]],
    c_sigs: Dict[int, Tuple[str, TypedSignature]],
    trace: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[int, int]]:
    """[Kc2 refactor] Containment score. Behavior byte-identical to the
    pre-Kc2 `mcs_score`. Uses the min(q_mass, c_mass) normalization.

    Semantics: "what fraction of the smaller side is present in the
    larger side?" Correct for ACTION containment gates and for the
    `_classify` loop; the small-side normalization is what those
    callers rely on.
    """
    q_mass, c_mass, q_nm, c_nm, q_em, c_em, corr = _mcs_accumulate(
        Q, C, q_sigs, c_sigs, trace=trace
    )
    if q_mass <= 0.0 or c_mass <= 0.0:
        return 0.0, corr
    if q_mass <= c_mass:
        score = (q_nm + q_em) / q_mass
    else:
        score = (c_nm + c_em) / c_mass
    return min(1.0, float(score)), corr

def mcs_similarity(
    Q: Subgraph,
    C: Subgraph,
    q_sigs: Dict[int, Tuple[str, TypedSignature]],
    c_sigs: Dict[int, Tuple[str, TypedSignature]],
    trace: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[int, int]]:
    """[DEPRECATED in Step 8: superseded by evidence_score]
    [Kc2] Balanced-coverage similarity score, in [0, 1].

        score = (matched_q + matched_c) / (q_mass + c_mass)

    where matched_q is query-side matched mass, matched_c is core-side
    matched mass, and q_mass / c_mass are the total masses of each side.

    This is the balanced (Dice-equivalent) intersection fraction — the
    graph-theoretic analog of the symmetric normalization used by
    divisive normalization in visual cortex (Carandini & Heeger 2012):
    a response is normalized by the summed magnitudes on both sides,
    not by the smaller of the two.

    Neural recognition is similarity-tuned (Kayaert, Biederman & Vogels
    2005; Yamins et al. 2014): the response of a shape-selective cell
    scales with overlap between input and preferred stimulus. Sparse
    distributed memory (Kanerva 1988) uses population overlap as
    similarity. Both are symmetric.

    [Failure mode fixed] `mcs_score` normalizes by min(q_mass, c_mass).
    When the query is a whole-image content subgraph (mass ~8,000) and
    the core is a filtered template (mass ~30), min is always the core,
    and the score degenerates to "fraction of the core present in the
    query" — which rewards small generic cores. `mcs_similarity` weighs
    both sides, so a small core must still explain a meaningful fraction
    of the query.

    [Compatibility] Reduces to the same value as `mcs_score` when
    `q_mass == c_mass`. Bounded in [0, 1]. Returns the same
    (score, correspondence) tuple shape.

    Does NOT replace `mcs_score` for containment gates. `mcs_contained`,
    `_classify`, `_mcs_margin`, and `_replay` continue to use
    `mcs_score`.
    """
    q_mass, c_mass, q_nm, c_nm, q_em, c_em, corr = _mcs_accumulate(
        Q, C, q_sigs, c_sigs, trace=trace
    )
    denom = q_mass + c_mass
    if denom <= 0.0:
        return 0.0, corr
    score = (q_nm + q_em + c_nm + c_em) / denom
    return min(1.0, float(score)), corr

def evidence_score(
    Q: Subgraph,
    C: Subgraph,
    q_sigs: Dict[int, Tuple[str, TypedSignature]],
    c_sigs: Dict[int, Tuple[str, TypedSignature]],
    lambda_absence: Optional[float] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[int, int]]:
    """Conjunctive evidence score with unconfounded two-component absence.

    [D1] Node coverage reads GRADED node credit under
    P.EVIDENCE_GRADED_NODE_CREDIT (default True): a credited pair earns
    boost * weight * signature_similarity, so two cores that both find a
    partner for every node no longer tie on the node term -- the one whose
    partners are more alike scores higher. See `_mcs_accumulate`.

    [D3] P.EVIDENCE_QUERY_SIDE_TERMS names the terms that read both sides
    (any of "node", "relation", "absence"; empty = the historical one-sided
    score):

        node      (q_matched_nm + c_matched_nm) / (q_node_total + c_node_total)
        relation  (core-side sim sum + query-side sim sum)
                  / (verifiable core edges + verifiable query edges)
        absence   (uncovered core edges + uncovered query edges)
                  / (core edges + query edges)

    An edgeless core keeps the vacuous relational reading (multiplier 1.0, no
    absence) in every setting; so with all three terms the score is symmetric
    in its two arguments when both sides have edges or neither has, and not
    when exactly one side is edgeless. Under P.EVIDENCE_ONLY_EDGES only "node"
    acts: that branch keeps its own relational rule, the other names are
    accepted and dropped, and the trace's `query_side_terms` lists the terms
    the score used. An unknown name is an error before any matching is done.

    [NULL CALIBRATION] Under P.EVIDENCE_NODE_NULL_CALIBRATED the node term is
    read as evidence above chance, (N - N0) / (1 - N0) clamped to [0, 1]
    (matching/calibration.py), N0 being the coverage a random pairing inside
    each lens type would earn under the same credit rule over the same
    side(s). A coverage any pairing would reach is then worth 0. The
    calibration presupposes graded credit: under binary credit a count is
    exactly what a random pairing earns, so the term is 0 for every pair.

    Trace: `node_coverage` is the node term the score used;
    `node_coverage_raw` the same term before calibration;
    `node_coverage_binary` its binary twin (same form, every kept pair
    credited in full, never calibrated); `core_node_coverage` and
    `query_node_coverage` the two one-sided graded fractions;
    `relation_presence` and `relation_physics` the two factors of the
    relational term kept apart ([P4b]; the term is their product only under
    P.EVIDENCE_RELATION_COUNT_APART, see `relation_count_apart`). On the
    early exits (an empty side, no lens type in common) none of them is
    written.
    """
    internal_trace = trace if trace is not None else {}
    null_calibrated = bool(getattr(P, "EVIDENCE_NODE_NULL_CALIBRATED", False))
    # [D3] Which terms read BOTH sides. Empty (the historical score): every
    # term is a fraction of the CORE, so nothing asks how much of the QUERY
    # the core explains and a small core is covered by almost anything.
    # Balanced form, the one `mcs_similarity` and matching/fuzzy.py already
    # use: numerator and denominator are each summed over the two sides.
    # Read (and validated) before any work, so a mistyped name is an error on
    # every path, the early exits included.
    qs_terms = _query_side_terms()
    if getattr(P, "EVIDENCE_ONLY_EDGES", False):
        # RULE B16 fixes the relational reading of that branch (similarity
        # over ALL core edges, no absence term): only the node term can read
        # both sides there. The names that cannot act are dropped HERE, so
        # the mirror loop is not run for them and the trace reports the terms
        # the score used.
        qs_terms = qs_terms & frozenset({"node"})
    q_mass, c_mass, q_nm, c_nm, q_em, c_em, corr = _mcs_accumulate(
        Q, C, q_sigs, c_sigs, trace=internal_trace,
        graded_node_credit=bool(
            getattr(P, "EVIDENCE_GRADED_NODE_CREDIT", True)),
        node_chance=null_calibrated,
    )
    if c_mass <= 0.0:
        return 0.0, corr

    # 1. Node Coverage (Part Matching)
    c_node_total = internal_trace.get("c_node_total", c_nm)
    q_node_total = internal_trace.get("q_node_total", q_nm)
    core_node_coverage = (
        (c_nm / (c_node_total + P.EPSILON)) if c_node_total > 0 else 0.0)
    query_node_coverage = (
        (q_nm / (q_node_total + P.EPSILON)) if q_node_total > 0 else 0.0)
    if "node" in qs_terms:
        _nt = q_node_total + c_node_total
        node_coverage = ((q_nm + c_nm) / (_nt + P.EPSILON)) if _nt > 0 else 0.0
    else:
        node_coverage = core_node_coverage

    # [NULL CALIBRATION] Node coverage as evidence ABOVE CHANCE. N0 is the
    # coverage a random pairing inside each lens type would earn under the
    # same credit rule, over the same side(s) and with the same masses as
    # the coverage itself; the map is matching/calibration.py's
    # (observed - chance) / (1 - chance), clamped, 0 when chance >= 1.
    node_coverage_raw = node_coverage
    node_coverage_chance = 0.0
    if null_calibrated:
        from matching.calibration import calibrate_fraction
        _q_ch = float(internal_trace.get("q_chance_nm", 0.0))
        _c_ch = float(internal_trace.get("c_chance_nm", 0.0))
        if "node" in qs_terms:
            _nt = q_node_total + c_node_total
            node_coverage_chance = (
                ((_q_ch + _c_ch) / (_nt + P.EPSILON)) if _nt > 0 else 0.0)
        else:
            node_coverage_chance = (
                (_c_ch / (c_node_total + P.EPSILON))
                if c_node_total > 0 else 0.0)
        node_coverage = calibrate_fraction(node_coverage_raw,
                                           node_coverage_chance)

    # 2. Relational Consistency
    n_core_edges = internal_trace.get("n_core_edges", len(C.edges))
    n_verifiable = internal_trace.get("verifiable_core_edges_count", 0)
    matched_rel_sim_sum = internal_trace.get("matched_rel_sim_sum", c_em)

    # [D3] Query-side edge evidence: the mirror of the core-side loop. Only
    # read when a term asks for it AND the core has edges -- an edgeless core
    # asserts no relation and keeps its vacuous reading in every setting.
    n_query_edges = len(Q.edges)
    q_verifiable = 0
    q_rel_sim_sum = 0.0
    _q_detail: Dict[str, Any] = {}
    _edge_terms = bool(qs_terms & {"relation", "absence"}) and n_core_edges > 0
    if _edge_terms:
        n_query_edges, q_verifiable, q_rel_sim_sum = _edge_side_evidence(
            Q, C, corr, detail=_q_detail)

    if n_core_edges == 0:
        relational_consistency = 1.0  # Edgeless core: relations vacuously satisfied
    elif "relation" in qs_terms:
        _nv = n_verifiable + q_verifiable
        relational_consistency = (
            clamp((matched_rel_sim_sum + q_rel_sim_sum) / float(_nv), 0.0, 1.0)
            if _nv > 0 else 0.0)
    elif n_verifiable == 0:
        relational_consistency = 0.0  # Relations exist in C, but none were verifiable under M
    else:
        relational_consistency = clamp(matched_rel_sim_sum / float(n_verifiable), 0.0, 1.0)

    # [P4b] THE COUNT KEPT APART FROM THE PHYSICS. The mean above credits a
    # matched pair with its similarity whatever that similarity is made of.
    # A pair with no graded axis measured on both sides (a membership flag,
    # a parameter-less predicate, a categorical agreement) reads exactly 1.0
    # there: the fact that the relation EXISTS on both sides, entered as a
    # perfect MEASUREMENT. Kept apart:
    #     presence = verifiable edges with a counterpart / verifiable edges
    #     physics  = mean similarity of the pairs that measured something
    #                (1.0 when none did: nothing measured, nothing
    #                contradicted -- matching/fuzzy.py's convention)
    #     term     = presence * physics
    # With t count-only pairs READING 1.0 and p physical pairs of mean
    # similarity m among v verifiable edges the term goes from (t + p*m)/v
    # to (t + p)*m/v: it can only stay or fall. (A count-only pair reads 1.0
    # when it is a flag, a parameter-less pair or a categorical agreement.
    # With the Phase-4 floors restored an all-UNKNOWN pair reads 0.2 and
    # becomes a full count here, so the term can rise in that setting.)
    # "Present" is a best similarity > 0, exactly the pairs the mean
    # credits: nothing is newly credited. The two factors read the side(s)
    # the term reads
    # (both under D3 "relation"). Always traced; they replace the mean only
    # under P.EVIDENCE_RELATION_COUNT_APART (default False), and never in the
    # EVIDENCE_ONLY_EDGES branch, whose reading RULE B16 fixes.
    _rel_v = n_verifiable
    _rel_present = float(internal_trace.get("rel_present", 0))
    _rel_phys_n = float(internal_trace.get("rel_physical", 0))
    _rel_phys_sum = float(internal_trace.get("rel_physical_sim_sum", 0.0))
    if "relation" in qs_terms and n_core_edges > 0:
        _rel_v = n_verifiable + q_verifiable
        _rel_present += float(_q_detail.get("present", 0))
        _rel_phys_n += float(_q_detail.get("physical", 0))
        _rel_phys_sum += float(_q_detail.get("physical_sim_sum", 0.0))
    relation_presence = (_rel_present / float(_rel_v)) if _rel_v > 0 else 0.0
    relation_physics = (
        (_rel_phys_sum / _rel_phys_n) if _rel_phys_n > 0 else 1.0)
    count_apart = (
        bool(getattr(P, "EVIDENCE_RELATION_COUNT_APART", False))
        and not getattr(P, "EVIDENCE_ONLY_EDGES", False))
    if count_apart and n_core_edges > 0:
        relational_consistency = (
            clamp(relation_presence * relation_physics, 0.0, 1.0)
            if _rel_v > 0 else 0.0)

    # 3. Conjunctive Relational Multiplier
    if getattr(P, "EVIDENCE_ONLY_EDGES", False):
        # [ITEM 3 / BUG-16 -- RULE B16, pre-registered 2026-10-16 before any
        # result]: an edge counts only by its OWN evidence. Unmatched
        # (endpoints unmapped), unverifiable, and ungraded (sim = 0) edges
        # contribute 0, and the denominator is n_core_edges -- so an
        # uninformative edge can no longer award the 0.85 conjunction floor
        # just because both endpoints mapped. Edgeless cores stay vacuous
        # 1.0. The separate absence term is dropped in this branch: with
        # n_core_edges as the denominator it would double-count the same
        # missing edges. Dark flag, UNCONFIRMED -- one object per class,
        # views only. Category scores of record come from
        # AspectGraph.category_scores (memory/aspects.py); this path is the
        # mapper's core-memory scoring only.
        if n_core_edges == 0:
            relational_multiplier = 1.0
        else:
            relational_multiplier = clamp(
                matched_rel_sim_sum / float(n_core_edges), 0.0, 1.0)
        core_coverage = node_coverage * relational_multiplier
        # Structural absence remains a TRACE value here (edges whose
        # endpoints never mapped); it does not enter the score -- with
        # n_core_edges as the denominator the uninformative edges are
        # already counted at 0.
        if n_core_edges > 0:
            edge_absence = (float(n_core_edges) - float(n_verifiable)) / float(n_core_edges)
        else:
            edge_absence = 0.0
        absence = 0.0
        score = clamp(core_coverage - absence, 0.0, 1.0)
    else:
        # [PHASE 5 -- null-calibrated, floor-free conjunction] The relational
        # multiplier's null is 0: an uninformative (verifiable but ungraded)
        # edge set yields 0 graded similarity, so conjugating with floor 0 is
        # the null-calibrated choice. This RETIRES the historical 0.85 floor
        # (EVIDENCE_ALPHA_CONJUNCTION) on the default path; that constant is
        # retained only for the legacy A/B path (MATCH_NULL_CALIBRATED=False).
        if getattr(P, "MATCH_NULL_CALIBRATED", False):
            alpha = float(getattr(P, "EVIDENCE_CONJUNCTION_FLOOR", 0.0))
        else:
            alpha = getattr(P, "EVIDENCE_ALPHA_CONJUNCTION", 0.85)
        relational_multiplier = alpha + (1.0 - alpha) * relational_consistency
        core_coverage = node_coverage * relational_multiplier

        # 4. Structural absence: core edges the correspondence failed to cover.
        #    Unit-correct by construction -- both terms are EDGE COUNTS:
        #      core_edges       = number of edges in the core subgraph
        #      verifiable_edges = core edges with BOTH endpoints inside the
        #                         correspondence, no similarity threshold
        #    (verifiable_core_edges is defined at _mcs_accumulate's step 2,
        #     memory/cores.py:865-868, and is exposed as n_verifiable.)
        #
        #    The graded mismatch term is intentionally absent
        #    (lambda_mismatch = 0): similarity magnitudes are already read by
        #    relational_multiplier above, so re-using them here would double
        #    count the same signal in one score.
        lambda_missing = 0.15
        if lambda_absence is not None and lambda_absence > 0.0:
            lambda_missing = float(lambda_absence)

        if n_core_edges > 0 and "absence" in qs_terms:
            # [D3] edges of EITHER side the correspondence failed to cover
            edge_absence = (
                (float(n_core_edges) - float(n_verifiable))
                + (float(n_query_edges) - float(q_verifiable))
            ) / float(n_core_edges + n_query_edges)
        elif n_core_edges > 0:
            edge_absence = (float(n_core_edges) - float(n_verifiable)) / float(n_core_edges)
        else:
            edge_absence = 0.0

        absence = lambda_missing * edge_absence

        score = clamp(core_coverage - absence, 0.0, 1.0)

    if trace is not None:
        trace["node_coverage"] = node_coverage
        # [D1] The BINARY TWIN of `node_coverage_raw`: the same form (the
        # one-sided core fraction, or the balanced one under D3 "node") over
        # the same correspondence, every kept pair credited in full. Never
        # calibrated. In every setting
        #     node_coverage <= node_coverage_raw <= node_coverage_binary
        # and under binary credit node_coverage_raw IS this value (the same
        # floats in the same expression). Trace only: no score reads it.
        _c_nm_bin = internal_trace.get("c_matched_nm_binary", c_nm)
        _q_nm_bin = internal_trace.get("q_matched_nm_binary", q_nm)
        if "node" in qs_terms:
            _nt = q_node_total + c_node_total
            trace["node_coverage_binary"] = (
                ((_q_nm_bin + _c_nm_bin) / (_nt + P.EPSILON))
                if _nt > 0 else 0.0)
        else:
            trace["node_coverage_binary"] = (
                (_c_nm_bin / (c_node_total + P.EPSILON))
                if c_node_total > 0 else 0.0)
        trace["relational_consistency"] = relational_consistency
        # [P4b] the two factors (always), and whether the score used them
        trace["relation_presence"] = relation_presence
        trace["relation_physics"] = relation_physics
        trace["relation_count_apart"] = count_apart
        trace["relational_multiplier"] = relational_multiplier
        trace["edge_absence"] = edge_absence
        trace["total_absence"] = absence
        trace["core_coverage"] = core_coverage
        # [D3] `node_coverage` above is the node term the score USED. The two
        # one-sided coverages are always traced, whichever was used.
        trace["query_side_terms"] = tuple(sorted(qs_terms))
        trace["core_node_coverage"] = core_node_coverage
        trace["query_node_coverage"] = query_node_coverage
        # [NULL CALIBRATION] `node_coverage` is what the score used; the
        # uncalibrated value and the chance level are traced next to it.
        trace["node_null_calibrated"] = null_calibrated
        trace["node_coverage_raw"] = node_coverage_raw
        if null_calibrated:
            trace["node_coverage_chance"] = node_coverage_chance
        if _edge_terms:
            trace["n_query_edges"] = n_query_edges
            trace["q_verifiable_edges_count"] = q_verifiable
            trace["q_matched_rel_sim_sum"] = q_rel_sim_sum
            # [P4b] the query side's present / physical pairs
            trace["q_rel_present"] = _q_detail.get("present", 0)
            trace["q_rel_physical"] = _q_detail.get("physical", 0)
            trace["q_rel_physical_sim_sum"] = _q_detail.get(
                "physical_sim_sum", 0.0)

    return float(score), corr

def decide_with_gap(
    scores: Dict[str, float],
    theta_min: Optional[float] = None,
    theta_gap: Optional[float] = None,
) -> Tuple[bool, str, float]:
    """Winner-take-all with lateral-inhibition gap.

    accepted iff best >= theta_min AND (best − runner_up) >= theta_gap,
    or only one candidate.

    Returns (accepted, predicted_label, gap). Gap is always finite;
    single-candidate case returns 1.0.
    """
    if theta_min is None:
        theta_min = float(P.THETA_EVIDENCE_POS)
    if theta_gap is None:
        theta_gap = float(P.EVIDENCE_GAP_THRESHOLD)

    if not scores:
        return False, "", 0.0

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_label, best_score = ordered[0]
    if best_score < theta_min:
        return False, best_label, 0.0
    if len(ordered) == 1:
        return True, best_label, 1.0
    runner_up = ordered[1][1]
    gap = best_score - runner_up
    return gap >= theta_gap, best_label, float(gap)

def mcs_contained(
    Q: Subgraph,
    C: Subgraph,
    q_sigs: Dict[int, Tuple[str, TypedSignature]],
    c_sigs: Dict[int, Tuple[str, TypedSignature]],
    theta: float = THETA_CONTAINMENT,
) -> bool:
    """§6.4.1 / §5.3.2: Q ⊆ C iff MCS_score(Q, C) >= theta.

    [DIRECTIONAL FIX] mcs_score normalizes by min(|Q|, |C|). When
    |Q| > |C| and C is a structural subset of Q, min picks |C|, the
    score evaluates to |C|/|C| = 1.0, and containment falsely reports
    True. Containment is directional: Q cannot fit inside a smaller C.
    Enforce |Q| <= |C| before delegating to mcs_score. For |Q| <= |C|,
    min(|Q|, |C|) = |Q| and the score is already the containment ratio.
    """
    q_size = _subgraph_size(Q)
    c_size = _subgraph_size(C)
    if q_size == 0 or q_size > c_size:
        return False
    score, _ = mcs_score(Q, C, q_sigs, c_sigs)
    return score >= theta

def _extract_actions_from_subgraph(
    G: Subgraph,
    sig_provider: SignatureProvider,
) -> List[Any]:
    """[SPEC GAP §5.3.4] Extract the ordered action sequence implied by G.

    §5.3.4 states "actions(G) = the ordered list of action descriptors in
    G's subgraph". Under v31's representational foundation (§1), pattern
    nodes do NOT carry action descriptors — only TypedSignatures. Actions
    are recovered here as the topological order of SEQUENCE edges (node IDs).

    Callers whose action_edge_map.predict expects real action strings must
    supply a downstream translation layer. The Builder's synthetic SEQUENCE
    detections carry (antecedent, consequent) role descriptors that a future
    mapper can translate.

    Returns [] if G has no SEQUENCE edges; replay of an empty sequence is a
    well-defined no-op (checks only the current state's outcome).
    """
    seq_edges = [(src, dst) for (src, dst, et) in G.edges if et == EdgeType.SEQUENCE]
    if not seq_edges:
        return []

    out_adj: Dict[int, List[int]] = defaultdict(list)
    in_degree: Dict[int, int] = {nid: 0 for nid in G.nodes}
    for src, dst in seq_edges:
        out_adj[src].append(dst)
        in_degree[dst] = in_degree.get(dst, 0) + 1

    queue: deque = deque(sorted(n for n in G.nodes if in_degree.get(n, 0) == 0))
    order: List[int] = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for nb in sorted(out_adj.get(n, ())):
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)

    remaining = [n for n in sorted(G.nodes) if n not in set(order)]
    order.extend(remaining)
    return order

def _replay(
    G: Subgraph,
    O: Subgraph,
    current_state: int,
    action_edge_map: Any,
    state_pattern_provider: Callable[[int], Optional[Subgraph]],
    sig_provider: SignatureProvider,
    theta_containment: float = THETA_CONTAINMENT,
    action_extractor: Optional[Callable[[Subgraph, int], Optional[List[Any]]]] = None,
    diagnostics: Optional[Dict[str, int]] = None,
    schema_confirmer: Optional[Callable[[Any], bool]] = None,
    concept_graph: Optional[Any] = None,
) -> bool:
    """§5.3.4: graph-simulation replay.

    Action resolution (B2 order):
      1. injected `action_extractor` (the pipeline wires
         pipeline_support.make_action_extractor — Action-lens registry
         decode first, AEM walk fallback for legacy concepts);
      2. without an extractor, the Action-lens registry decode via
         `concept_graph` (exact action objects — the §5.3.4 vocabulary
         recovery the spec requires);
      3. the historical node-order fallback: since pattern nodes are not
         actions (§1), predict() then returns None and replay fails
         honestly — counted, so the miswiring is observable instead of
         silent.

    ARCHITECT RULING (Item 1) — confirmed-transition gate: cores are
    PERMANENT macro-edges, so every replayed transition must have reached
    mass ≥ N_edge (5) with dominance ≥ θ_dominance (0.8). Environments
    keep n_edge = 1 for bootstrap navigation (tentative traversal), but
    unconfirmed noise must never be encoded into the CoreLibrary.
    """
    if action_extractor is not None:
        actions = action_extractor(G, current_state)
        if actions is None:
            if diagnostics is not None:
                diagnostics["replay_action_extraction_failed"] += 1
            return False
    else:
        decoded: Optional[List[Any]] = None
        if concept_graph is not None:
            from action.decode import decode_action_sequence

            decoded = decode_action_sequence(G, concept_graph)
        if decoded is not None:
            actions = decoded
        else:
            if diagnostics is not None:
                diagnostics["replay_no_extractor"] += 1
            actions = _extract_actions_from_subgraph(G, sig_provider)
    s_sim = current_state
    for a in actions:
        predict_fn = getattr(action_edge_map, "predict", None)
        if predict_fn is None:
            # Caller passed an action_edge_map without predict; fail loudly.
            raise AttributeError(
                "_replay requires action_edge_map.predict(state, action) "
                "-> Optional[int] (§5.3.4)"
            )
        s_pred = predict_fn(s_sim, a)
        if s_pred is None:
            return False
        # Ruling Item 1: permanence gate on confirmed transitions.
        # Confirmation is satisfied by EITHER:
        #   (a) per-state edge mass ≥ N_edge (the literal AEM count), OR
        #   (b) DESIGN DECISION D4 (flagged for architect ratification):
        #       the action's causal schema link is counterfactually
        #       VALIDATED — the injected schema_confirmer answers from
        #       H_nec, which itself requires N_cf ≥ N_cf_min ≥ 5 resolved
        #       counterfactual opportunities AND ΔP̄ ≥ θ_enter: a strictly
        #       stronger noise filter than a raw per-state count.
        # Rationale: fine-grained state hashes make per-state repetition
        # rare for one-shot causal effects (a door is opened once, then
        # the goal-satisfied skip correctly stops re-toggling), so gate
        # (a) alone starves Trigger 1A permanently. Dominance ≥
        # θ_dominance remains mandatory under BOTH routes.
        outcomes_fn = getattr(action_edge_map, "get_outcomes", None)
        if callable(outcomes_fn):
            outcomes = outcomes_fn(s_sim, a)
            mass = float(sum(outcomes.values())) if outcomes else 0.0
            schema_ok = bool(schema_confirmer(a)) if schema_confirmer is not None else False
            if mass < P.N_EDGE and not schema_ok:
                if diagnostics is not None:
                    diagnostics["replay_unconfirmed_transition"] += 1
                return False
            if mass > 0:
                dominant = max(outcomes.values())
                if dominant / (mass + 1e-9) < P.THETA_DOMINANCE:
                    if diagnostics is not None:
                        diagnostics["replay_nondominant_transition"] += 1
                    return False
            elif not schema_ok:
                return False
        s_sim = s_pred

    observed = state_pattern_provider(s_sim)
    if observed is None:
        return False

    # Signature maps for the MCS call.
    o_sigs: Dict[int, Tuple[str, TypedSignature]] = {}
    for nid in O.nodes:
        info = sig_provider(nid)
        if info is not None:
            o_sigs[nid] = info
    obs_sigs: Dict[int, Tuple[str, TypedSignature]] = {}
    for nid in observed.nodes:
        info = sig_provider(nid)
        if info is not None:
            obs_sigs[nid] = info

    score, _ = mcs_score(O, observed, o_sigs, obs_sigs)
    return score >= theta_containment

THETA_UPDATE = 0.85

THETA_CREATE = 0.60

THETA_FUSE = 0.90

THETA_RECOGNIZE_CATEGORY = 0.70

THETA_RECOGNIZE_MARGIN = 0.15

N_DEMOTE = 3

N_QUERY_FLOOR_FOR_DEMOTION = 5

# [Phase 1.6 GATE 2] Minimum number of steps a category must REMAIN in
# STABLE before a demotion is honoured. Without it, `observe()` calls
# `_evaluate_q5` and then `_evaluate_degradation` in the SAME tick
# (this module:285-286) while demotion preserves the graduation evidence
# counters (`positive_recognition_count`, `confirmed_rejection_count`,
# `_t_first_pos`), so a category that graduates on tick T can be demoted on
# tick T+2 and re-graduated on tick T+3 — indefinitely, and with no
# evidence change in between. The cooldown bounds that churn without
# disabling degradation: once a category has held STABLE for T_STABLE_MIN
# steps the demotion path behaves exactly as before. Inert on any run where
# no category graduates (all three pinned envs today), so it cannot move the
# 425/98/597 pins.
T_STABLE_MIN = 100

N_FUSE_QUERY_MIN = 3

N_FUSE_FALSE_POS_MAX = 2

Q5_POS_MIN = 5

Q5_NEG_MIN = 3

Q5_AGE_MIN = 200

Q3_SPLIT_MIN_HISTORY = 6

Q3_SPLIT_MIN_MODE = 3

Q3_SEPARATION_THRESHOLD = 1.0

HISTORY_MAXLEN = 32

_LABEL_TO_CIDS_SOFT_CAP = 500

@dataclass(frozen=True)
class RecognizeResult:
    assigned_hash: Optional[bytes]
    top_candidate_hash: Optional[bytes]
    label: Any
    top_candidate_score: float

class CategoryManager:
    def __init__(
        self,
        core_library: Any,
        mapper: Any,
        window_buffer: Any,
        concept_graph: Any,
        consolidation_diagnostics: Dict[str, Any],
        extract_fn: Any,
        diversity_fn: Any,
    ) -> None:
        self.cl = core_library
        self.mapper = mapper
        self.buffer = window_buffer
        self.cg = concept_graph
        self.diag = consolidation_diagnostics
        self._extract_fn = extract_fn
        self._diversity_fn = diversity_fn

        self._next_cat_id: int = 0
        self._core_of_cat: Dict[int, bytes] = {}
        self._cat_of_core: Dict[bytes, int] = {}
        self._counters: Dict[int, Dict[str, int]] = {}
        self._history: Dict[int, List[Tuple[Any, float]]] = {}
        self._t_first_pos: Dict[int, int] = {}
        self._t_stable_entered: Dict[int, int] = {}
        self._label_to_cids: Dict[Any, Set[int]] = defaultdict(set)

    def start_session(self) -> None:
        self._next_cat_id = 0
        self._core_of_cat = {}
        self._cat_of_core = {}
        self._counters = {}
        self._history = {}
        self._t_first_pos = {}
        self._t_stable_entered = {}
        self._label_to_cids = defaultdict(set)
        for e in sorted(self.cl.entries, key=lambda x: x.canonical_hash):
            if e.kind != CoreKind.CATEGORY:
                continue
            payload = e.payload
            if not isinstance(payload, CorePayload_Category):
                continue
            if not _is_effect_atom_label(payload.label):
                continue
            self._register_new_category(e.canonical_hash, payload.label)

    def _register_new_category(self, core_hash: bytes, label: Any) -> int:
        cid = self._next_cat_id
        self._next_cat_id += 1
        self._core_of_cat[cid] = core_hash
        self._cat_of_core[core_hash] = cid
        self._counters[cid] = {
            "query_count": 0,
            "false_positive_count": 0,
            "confirmed_rejection_count": 0,
            "positive_recognition_count": 0,
            "declined_below_theta": 0,
            "declined_below_margin": 0,
            "consecutive_positive_failures": 0,
            "cross_label_assignments": 0,
        }
        self._history[cid] = []
        if label is not None:
            self._label_to_cids[label].add(cid)
        return cid

    def _unregister_category(self, cid: int) -> None:
        ch = self._core_of_cat.pop(cid, None)
        if ch is not None:
            self._cat_of_core.pop(ch, None)
        self._counters.pop(cid, None)
        self._history.pop(cid, None)
        self._t_first_pos.pop(cid, None)
        self._t_stable_entered.pop(cid, None)
        for label, cids in list(self._label_to_cids.items()):
            if cid in cids:
                cids.discard(cid)
                if not cids:
                    del self._label_to_cids[label]

    def purge_stale(self, label: Any) -> None:
        cids = list(self._label_to_cids.get(label, ()))
        dead: List[int] = []
        for cid in cids:
            ch = self._core_of_cat.get(cid)
            if ch is None:
                dead.append(cid)
                continue
            if self.cl.lookup_by_hash(ch, CoreKind.CATEGORY) is None:
                dead.append(cid)
        for cid in dead:
            self._unregister_category(cid)
            self.diag["staleness_purges"] = self.diag.get("staleness_purges", 0) + 1

    def _sweep_orphan_labels(self) -> None:
        for label in list(self._label_to_cids.keys()):
            cids = self._label_to_cids.get(label)
            if not cids:
                del self._label_to_cids[label]
                continue
            alive = False
            for cid in cids:
                ch = self._core_of_cat.get(cid)
                if ch is None:
                    continue
                if self.cl.lookup_by_hash(ch, CoreKind.CATEGORY) is not None:
                    alive = True
                    break
            if not alive:
                del self._label_to_cids[label]

    def step(self, t: int, label_signals: Any) -> None:
        labels = sorted(label_signals, key=lambda ls: serialize_value(ls.label))
        for ls in labels:
            if self.buffer.ready(ls.label):
                self.on_label_ready(ls.label, t)
        if len(self._label_to_cids) > _LABEL_TO_CIDS_SOFT_CAP:
            self._sweep_orphan_labels()

    def recognize(
        self, label: Any, polarity: int, window: Any, t: int,
    ) -> RecognizeResult:
        self.purge_stale(label)
        match = self.mapper.query_category(t, label, window)
        if not match.labels:
            return RecognizeResult(None, None, label, 0.0)
        same_label = [cl for cl in match.labels if cl.label == label]
        if not same_label:
            return RecognizeResult(None, None, label, 0.0)
        # [Kc2] Ordered by balanced-coverage similarity (rank_score), not by
        # containment. The gate below still reads top.raw_score (MCS
        # containment) against THETA_RECOGNIZE_CATEGORY, unchanged.
        same_label.sort(key=lambda cl: (-cl.rank_score, cl.core_hash))
        top = same_label[0]

        assigned_hash: Optional[bytes] = None
        if top.raw_score >= THETA_RECOGNIZE_CATEGORY:
            if len(same_label) < 2 or (
                top.raw_score - same_label[1].raw_score
            ) >= THETA_RECOGNIZE_MARGIN:
                assigned_hash = top.core_hash

        for cl in same_label:
            cid = self._cat_of_core.get(cl.core_hash)
            if cid is not None:
                self._counters[cid]["query_count"] += 1

        return RecognizeResult(
            assigned_hash=assigned_hash,
            top_candidate_hash=top.core_hash,
            label=label,
            top_candidate_score=float(top.raw_score),
        )

    def observe(
        self,
        label: Any,
        polarity: int,
        window: Any,
        assigned_hash: Optional[bytes],
        top_candidate_hash: Optional[bytes],
        top_candidate_score: float,
        t: int,
    ) -> None:
        self.purge_stale(label)

        matched_same_label = False
        if assigned_hash is not None:
            cid = self._cat_of_core.get(assigned_hash)
            if cid is not None:
                entry = self.cl.lookup_by_hash(assigned_hash, CoreKind.CATEGORY)
                if entry is not None:
                    payload = getattr(entry, "payload", None)
                    if payload is not None and getattr(payload, "label", None) == label:
                        matched_same_label = True
                    else:
                        # Unreachable under current call graph; safety net.
                        self._counters[cid]["cross_label_assignments"] += 1

        if polarity == 1:
            if matched_same_label:
                cid = self._cat_of_core[assigned_hash]
                self._counters[cid]["positive_recognition_count"] += 1
                self._counters[cid]["consecutive_positive_failures"] = 0
                if cid not in self._t_first_pos:
                    self._t_first_pos[cid] = t
                self._history[cid].append((window, float(top_candidate_score)))
                if len(self._history[cid]) > HISTORY_MAXLEN:
                    self._history[cid].pop(0)
            elif top_candidate_score < THETA_RECOGNIZE_CATEGORY:
                cid = (
                    self._cat_of_core.get(top_candidate_hash)
                    if top_candidate_hash is not None else None
                )
                if cid is not None:
                    self._counters[cid]["consecutive_positive_failures"] += 1
                    self._counters[cid]["declined_below_theta"] += 1
            else:
                cid = (
                    self._cat_of_core.get(top_candidate_hash)
                    if top_candidate_hash is not None else None
                )
                if cid is not None:
                    self._counters[cid]["declined_below_margin"] += 1
        elif polarity == -1:
            if matched_same_label:
                cid = self._cat_of_core[assigned_hash]
                self._counters[cid]["false_positive_count"] += 1
            else:
                cid = (
                    self._cat_of_core.get(top_candidate_hash)
                    if top_candidate_hash is not None else None
                )
                if cid is not None:
                    self._counters[cid]["confirmed_rejection_count"] += 1

        self._evaluate_q5(label, t)
        self._evaluate_degradation(label, t)

    def on_label_ready(self, label: Any, t: int) -> None:
        self.purge_stale(label)
        pair = self.buffer.pop_windows(label)
        if pair is None:
            return
        positives, negatives = pair
        if not positives:
            return

        new_core = self._extract_fn(label, positives, negatives, t)
        if new_core is None:
            return
        new_hash = new_core.canonical_hash

        q1_action, q1_cid = self._evaluate_q1(
            label, new_hash, positives, negatives, t
        )
        if q1_action == "create":
            # Cross-label hash collision guard: if _extract_fn returned a
            # hash already tracked under a different label, do NOT create
            # a duplicate mapping. The core is legitimately tracked
            # elsewhere; skip and log.
            if new_hash in self._cat_of_core:
                self.diag["cross_label_hash_collision"] = (
                    self.diag.get("cross_label_hash_collision", 0) + 1
                )
                return
            self._register_new_category(new_hash, label)
        elif q1_action == "update":
            self._apply_update(
                q1_cid, new_hash, positives, negatives, t, label
            )
        elif q1_action == "no_op":
            # [CRITICAL] _extract_fn unconditionally inserted the
            # extracted core into the library BEFORE Q1 ran. On no_op
            # the manager refuses to track it; without removal, the
            # orphan would match queries but never graduate/demote/
            # fuse/split — a semantic leak, not just memory bloat.
            # Only remove if untracked elsewhere (cross-label case).
            if new_hash not in self._cat_of_core:
                self.cl.remove(new_hash, CoreKind.CATEGORY)
            return

        self._evaluate_q3(label, t, positives, negatives)
        self._evaluate_q2(label, t, positives, negatives)

    def _evaluate_q1(self, label, new_hash, positives, negatives, t):
        same_label_cids = self._live_same_label_cids(label)
        if not same_label_cids:
            return ("create", None)
        best_sim = 0.0
        best_cid: Optional[int] = None
        for cid in same_label_cids:
            ch = self._core_of_cat.get(cid)
            if ch is None:
                continue
            s = self._mcs_score(new_hash, ch)
            if s > best_sim or (
                s == best_sim
                and best_cid is not None
                and ch < self._core_of_cat[best_cid]
            ):
                best_sim = s
                best_cid = cid
        if best_sim >= THETA_UPDATE and best_cid is not None:
            return ("update", best_cid)
        if best_sim <= THETA_CREATE:
            return ("create", None)
        return ("no_op", None)

    def _apply_update(
        self, cid, new_hash, positives, negatives, t, label
    ) -> None:
        """On hash-changing update: reset the running failure streak.
        Cumulative counters and t_first_pos carry forward.
        Defensive: stale cid degrades to fresh registration."""
        old_hash = self._core_of_cat.get(cid)
        if old_hash is None:
            self._unregister_category(cid)
            if new_hash not in self._cat_of_core:
                self._register_new_category(new_hash, label)
            return
        if old_hash == new_hash:
            entry = self.cl.lookup_by_hash(new_hash, CoreKind.CATEGORY)
            if entry is not None:
                entry.t_last_validated = t
            return
        if new_hash in self._cat_of_core and self._cat_of_core[new_hash] != cid:
            self.cl.remove(old_hash, CoreKind.CATEGORY)
            self._unregister_category(cid)
            return
        self.cl.remove(old_hash, CoreKind.CATEGORY)
        self._cat_of_core.pop(old_hash, None)
        self._core_of_cat[cid] = new_hash
        self._cat_of_core[new_hash] = cid
        self._counters[cid]["consecutive_positive_failures"] = 0

    def _evaluate_q2(self, label, t, positives, negatives) -> None:
        same_label_cids = sorted(self._live_same_label_cids(label))
        if len(same_label_cids) < 2:
            return
        for i in range(len(same_label_cids)):
            a_cid = same_label_cids[i]
            a_ch = self._core_of_cat.get(a_cid)
            if a_ch is None:
                continue
            ac = self._counters[a_cid]
            if ac["query_count"] < N_FUSE_QUERY_MIN:
                continue
            if ac["false_positive_count"] >= N_FUSE_FALSE_POS_MAX:
                continue
            for j in range(i + 1, len(same_label_cids)):
                b_cid = same_label_cids[j]
                b_ch = self._core_of_cat.get(b_cid)
                if b_ch is None:
                    continue
                bc = self._counters[b_cid]
                if bc["query_count"] < N_FUSE_QUERY_MIN:
                    continue
                if bc["false_positive_count"] >= N_FUSE_FALSE_POS_MAX:
                    continue
                if self._mcs_score(a_ch, b_ch) < THETA_FUSE:
                    continue
                pos_union = (
                    [w for (w, _s) in self._history[a_cid]]
                    + [w for (w, _s) in self._history[b_cid]]
                )
                fused = self._extract_fn(label, pos_union, list(negatives), t)
                if fused is None:
                    continue
                self.cl.remove(a_ch, CoreKind.CATEGORY)
                self.cl.remove(b_ch, CoreKind.CATEGORY)
                self._unregister_category(a_cid)
                self._unregister_category(b_cid)
                if fused.canonical_hash in self._cat_of_core:
                    self.diag["fusion_child_collapsed"] = (
                        self.diag.get("fusion_child_collapsed", 0) + 1
                    )
                    self.diag["fusion_count"] = self.diag.get("fusion_count", 0) + 1
                    return
                self._register_new_category(fused.canonical_hash, label)
                self.diag["fusion_count"] = self.diag.get("fusion_count", 0) + 1
                return

    def _evaluate_q3(self, label, t, positives, negatives) -> None:
        for cid in sorted(self._live_same_label_cids(label)):
            history = self._history.get(cid, [])
            if len(history) < Q3_SPLIT_MIN_HISTORY:
                continue
            scores = [s for (_w, s) in history]
            split_result = self._bimodal_split(scores)
            if split_result is None:
                continue
            low_idx, high_idx = split_result
            if len(low_idx) < Q3_SPLIT_MIN_MODE or len(high_idx) < Q3_SPLIT_MIN_MODE:
                continue
            windows = [w for (w, _s) in history]
            low_pos = [windows[i] for i in low_idx]
            high_pos = [windows[i] for i in high_idx]
            old_hash = self._core_of_cat.get(cid)
            if old_hash is None:
                continue

            self.cl.remove(old_hash, CoreKind.CATEGORY)
            self._unregister_category(cid)

            created = 0
            for group in (low_pos, high_pos):
                child = self._extract_fn(label, group, list(negatives), t)
                if child is None:
                    continue
                if child.canonical_hash in self._cat_of_core:
                    self.diag["split_child_collapsed"] = (
                        self.diag.get("split_child_collapsed", 0) + 1
                    )
                    continue
                self._register_new_category(child.canonical_hash, label)
                created += 1
            if created >= 1:
                self.diag["split_count"] = self.diag.get("split_count", 0) + 1
                return

    def _evaluate_q5(self, label: Any, t: int) -> None:
        for cid in self._live_same_label_cids(label):
            counters = self._counters.get(cid)
            if counters is None:
                continue
            if counters["positive_recognition_count"] < Q5_POS_MIN:
                continue
            if counters["confirmed_rejection_count"] < Q5_NEG_MIN:
                continue
            t_first = self._t_first_pos.get(cid)
            if t_first is None or (t - t_first) < Q5_AGE_MIN:
                continue
            ch = self._core_of_cat.get(cid)
            if ch is None:
                continue
            entry = self.cl.lookup_by_hash(ch, CoreKind.CATEGORY)
            if entry is None:
                continue
            if getattr(entry, "core_status", CoreStatus.CRYSTALLIZED) == CoreStatus.STABLE:
                continue
            entry.core_status = CoreStatus.STABLE
            self._t_stable_entered[cid] = t
            self.diag["graduation_count"] = self.diag.get("graduation_count", 0) + 1

    def _evaluate_degradation(self, label: Any, t: int) -> None:
        for cid in self._live_same_label_cids(label):
            counters = self._counters.get(cid)
            if counters is None:
                continue
            if counters["consecutive_positive_failures"] < N_DEMOTE:
                continue
            if counters["query_count"] < N_QUERY_FLOOR_FOR_DEMOTION:
                continue
            ch = self._core_of_cat.get(cid)
            if ch is None:
                continue
            entry = self.cl.lookup_by_hash(ch, CoreKind.CATEGORY)
            if entry is None:
                continue
            if getattr(entry, "core_status", CoreStatus.CRYSTALLIZED) != CoreStatus.STABLE:
                continue
            # [Phase 1.6 GATE 2] Demotion cooldown: a fresh STABLE category
            # gets T_STABLE_MIN steps of grace. Without this guard, the
            # single-tick graduate/demote pair re-graduates immediately on
            # the surviving evidence counters (see T_STABLE_MIN above).
            t_entered = self._t_stable_entered.get(cid)
            if t_entered is not None and (t - t_entered) < T_STABLE_MIN:
                continue
            entry.core_status = CoreStatus.CRYSTALLIZED
            counters["consecutive_positive_failures"] = 0
            self.diag["demotion_count"] = self.diag.get("demotion_count", 0) + 1

    def _live_same_label_cids(self, label: Any) -> List[int]:
        out: List[int] = []
        for cid in list(self._label_to_cids.get(label, ())):
            ch = self._core_of_cat.get(cid)
            if ch is None:
                self._label_to_cids[label].discard(cid)
                continue
            entry = self.cl.lookup_by_hash(ch, CoreKind.CATEGORY)
            if entry is None:
                continue
            if getattr(entry, "core_status", CoreStatus.CRYSTALLIZED) == CoreStatus.REJECTED:
                continue
            payload = getattr(entry, "payload", None)
            if not isinstance(payload, CorePayload_Category):
                continue
            if payload.label != label:
                continue
            out.append(cid)
        return out

    def _mcs_score(self, hash_a: bytes, hash_b: bytes) -> float:
        ea = self.cl.lookup_by_hash(hash_a, CoreKind.CATEGORY)
        eb = self.cl.lookup_by_hash(hash_b, CoreKind.CATEGORY)
        if ea is None or eb is None:
            return 0.0
        score, _ = mcs_score(
            ea.core_subgraph, eb.core_subgraph,
            ea.core_node_signatures, eb.core_node_signatures,
        )
        return float(score)

    def _bimodal_split(self, scores):
        if len(scores) < Q3_SPLIT_MIN_HISTORY:
            return None
        ordered = sorted(range(len(scores)), key=lambda i: scores[i])
        mid = len(ordered) // 2
        low, high = ordered[:mid], ordered[mid:]
        for _ in range(2):
            ml = sum(scores[i] for i in low) / max(1, len(low))
            mh = sum(scores[i] for i in high) / max(1, len(high))
            nl, nh = [], []
            for i in range(len(scores)):
                if abs(scores[i] - ml) <= abs(scores[i] - mh):
                    nl.append(i)
                else:
                    nh.append(i)
            if nl == low and nh == high:
                break
            low, high = nl, nh
        if not low or not high:
            return None
        ml = sum(scores[i] for i in low) / len(low)
        mh = sum(scores[i] for i in high) / len(high)
        sl = math.sqrt(sum((scores[i] - ml) ** 2 for i in low) / len(low))
        sh = math.sqrt(sum((scores[i] - mh) ** 2 for i in high) / len(high))
        if (mh - ml) / (sl + sh + 1e-9) < Q3_SEPARATION_THRESHOLD:
            return None
        return (low, high)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "category_count": len(self._core_of_cat),
            "fusion_count": self.diag.get("fusion_count", 0),
            "split_count": self.diag.get("split_count", 0),
            "split_child_collapsed": self.diag.get("split_child_collapsed", 0),
            "fusion_child_collapsed": self.diag.get("fusion_child_collapsed", 0),
            "cross_label_hash_collision": self.diag.get("cross_label_hash_collision", 0),
            "demotion_count": self.diag.get("demotion_count", 0),
            "graduation_count": self.diag.get("graduation_count", 0),
            "staleness_purges": self.diag.get("staleness_purges", 0),
            "cross_label_assignments": sum(
                c["cross_label_assignments"] for c in self._counters.values()
            ),
            "label_to_cids_size": len(self._label_to_cids),
        }

def _is_effect_atom_label(label: Any) -> bool:
    return (
        isinstance(label, tuple)
        and len(label) == 2
        and label[0] == "effect_atom"
    )

