"""memory/assembly.py -- the ASSEMBLY CONTRACT.

DESIGN AND DATA STRUCTURES ONLY. Formation and readout *algorithms* are
SPECIFIED here (in the docstrings and the constants below) and are BUILT in the
consolidation session. This module defines the stores, the records, the
pre-registered description-length coding scheme, the reader handles, the
retention rule, the legacy list and the operational statistics -- everything a
later session needs to implement formation and readout without re-deciding a
contract.

PRINCIPLES (binding)
  1. A concept is a CLUSTER of nodes and edges, never a node. A reader unit is
     a function of the cluster (Buzsaki), not the concept. A label (name) is an
     ordinary member that co-occurs by repetition; there is no supervised side
     channel.
  2. Parts and objects are not types; they are statistics computed on demand.
  3. One cluster mechanism at all levels. The existing cores and the ART-style
     view memory are early forms of it; migrate toward it with no parallel
     scorers.
  4. Expectations modulate, never create: zero bottom-up evidence stays zero.
  5. No fabrication: nothing enters a cluster that was not observed.

STORES
  1. Memory nodes: persistent, position-invariant, population-coded (existing
     core nodes; referenced here only by id).
  2. Episode store (fast, append-only): one `EpisodeRecord` per experience --
     the memory nodes active together, their relation summaries (arrangement),
     and any label symbols.
  3. Assembly store (slow): `AssemblyRecord`s with graded members, internal
     relations, support, parent assemblies, overlap links and provenance.

RELATION TO THE EXISTING MECHANISM
  The cluster mechanism already exists: `concept_graph.py` (`ConceptGraph`,
  `Concept`, `Candidate`, the crystallization predicate) plus
  `memory/builder.py` (spine-guided commitment search over `CandidateSubgraph`
  fronts). Those are the EARLY FORM this contract migrates toward; the shapes
  here are the migration TARGET and must NOT grow into a second, parallel
  scorer. Mapping: `AssemblyRecord` <- `Concept`; candidate assembly <-
  `Candidate`/`CandidateSubgraph`; members <- `Concept.members` (today an
  unweighted frozenset); internal relations <- `role_descriptors` + induced
  `EdgeParams`; support <- `N_conf`/`N_opp`; episode store <- the
  `CandidateSubgraph.contexts` set; formation <- the Builder; admission <-
  `_crystallization_predicate` (thresholds); decay/pruning <- the dormancy /
  capacity / prune steps. What this contract ADDS (to be closed by migration,
  not a rewrite): graded member weight+tolerance, explicit relation prototypes,
  description-length admission, a real episode store with retention, maximal-
  clique growth, overlap links, bounded reader recursion, the one-shot scorer
  switch, label members.

READOUT CONTRACT (specified; dynamics owned by the resonance session)
  Graded activation = the overlap of the active graph with the record: the
  population overlap of active member nodes plus the consistency of the
  recorded internal relations. The same function yields an expected-activation
  vector for NON-active members by pattern completion (a member's expected
  activation is the record's activation restricted through the relation it
  participates in). That vector is handed to the resonance layer as a BOUNDED
  GAIN applied only where bottom-up evidence exists -- never a source of
  activation. Reader activations may themselves be members of higher
  assemblies`over whose readout is capped by `ASSEMBLY_READER_MAX_DEPTH` and
  `ASSEMBLY_READER_MAX_READERS` (specified, not run here).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.hashing import serialize_value
import parameters as P
from parameters import EPSILON


# ---------------------------------------------------------------------------
# PRE-REGISTERED CODING SCHEME (one scheme; chosen now, not after seeing data)
# ---------------------------------------------------------------------------
# The mission allows exactly one coding scheme for the length argument. We
# register DESCRIPTION LENGTH (a minimum-description-length criterion) rather
# than a Bayesian evidence with non-informative priors, because every term of
# the length has a direct, representation-derived interpretation (a node must
# be named, an edge presence must be stated, a graded parameter must be given
# to a precision set by its own tuning width), so the admission test cannot be
# gamed by a hidden prior scale. Discrimination comes from the LABEL as an
# ordinary member, from competition between overlapping assemblies, and from
# the null -- compression alone finds redundancy, not discrimination.
CODING_SCHEME = "description_length"

#: Cost, in bits, of an assembly header (a record's identity, its size, and the
#: checksum that lets a reader reject a malformed record). Stated once here; a
#: matched member reuses this header and is not charged a second time.
ASSEMBLY_HEADER_BITS = 16.0

#: Extra header bits per member/relation slot beyond the index cost, covering
#: the boolean "is this a sub-assembly reference" and the weight/tolerance
#: presence flags.
ASSEMBLY_SLOT_FLAG_BITS = 2.0

#: Default graded-parameter width (isotropic fallback) when a parameter's own
#: population width is unavailable. Expressed as a fraction of the signed range.
DEFAULT_PARAMETER_WIDTH_FRAC = 1.0 / 3.0


def node_index_cost_bits(n_memory_nodes: int) -> float:
    """Bits to name one memory node: ``log2(number of memory nodes)``.

    A cluster member is an INDEX into the persistent node store, so naming it
    costs exactly the address length of that store. At least 0 bits when the
    store is empty or singular (nothing to distinguish).
    """
    n = int(n_memory_nodes)
    if n <= 1:
        return 0.0
    return math.log2(float(n))


def edge_presence_cost_bits(n_candidate_pairs: int) -> float:
    """Bits to state that ONE relation is present: ``log2(candidate pairs)``.

    An internal relation is a selected pair among the pairs the members could
    form, so stating its presence costs the log-count of the candidate pair
    set. At least 0 bits when there is no choice to make.
    """
    n = int(n_candidate_pairs)
    if n <= 1:
        return 0.0
    return math.log2(float(n))


def parameter_precision_cost_bits(
    value_range: float, population_width: float
) -> float:
    """Bits to state a graded parameter to its own resolution.

    ``log2(range / population width)`` -- the number of distinguishable levels
    the parameter can take, set by the *representation's own* tuning width, not
    a hand-chosen precision. Falls back to ``DEFAULT_PARAMETER_WIDTH_FRAC`` of
    the range when the width is missing, and to 0 bits for a degenerate range.
    """
    rng = float(value_range)
    if rng <= 0.0:
        return 0.0
    w = float(population_width)
    if w <= 0.0:
        w = DEFAULT_PARAMETER_WIDTH_FRAC * rng
    ratio = rng / max(w, EPSILON)
    if ratio <= 1.0:
        return 0.0
    return math.log2(ratio)


@dataclass(frozen=True)
class CodingCost:
    """The part-wise description length of a candidate assembly."""
    header_bits: float
    member_bits: float
    edge_bits: float
    parameter_bits: float
    error_bits: float = 0.0

    @property
    def total_bits(self) -> float:
        return (self.header_bits + self.member_bits + self.edge_bits
                + self.parameter_bits + self.error_bits)


def description_length_bits(
    n_members: int,
    member_ref_ids: Sequence[int],
    n_relations: int,
    n_candidate_pairs: int,
    n_memory_nodes: int,
    parameter_widths: Sequence[Tuple[float, float]] = (),
    error_bits: float = 0.0,
) -> CodingCost:
    """Total description length of a candidate cluster under ``CODING_SCHEME``.

    ``member_ref_ids`` are the ids of the member references (memory nodes or
    sub-assemblies); duplicate sub-assembly references cost the same index
    twice. ``parameter_widths`` is one ``(range, population_width)`` pair per
    graded internal relation. ``error_bits`` is the residual of the cluster
    after the members and relations are stated (zero for a lossless candidate).
    """
    header = float(ASSEMBLY_HEADER_BITS)
    n_nodes = int(n_memory_nodes)
    index_bits = node_index_cost_bits(n_nodes)
    member = sum(
        index_bits + ASSEMBLY_SLOT_FLAG_BITS for _ in range(int(n_members))
    )
    edge = (edge_presence_cost_bits(n_candidate_pairs) + ASSEMBLY_SLOT_FLAG_BITS) * float(
        max(0, int(n_relations))
    )
    parameter = 0.0
    rng_hint = 0.0
    for (rng, width) in parameter_widths:
        rng_hint = max(rng_hint, float(rng))
        parameter += parameter_precision_cost_bits(rng, width)
    if parameter == 0.0 and int(n_relations) > 0:
        parameter = parameter_precision_cost_bits(
            rng_hint if rng_hint > 0.0 else 1.0, 0.0
        ) * float(int(n_relations))
    return CodingCost(
        header_bits=header,
        member_bits=member,
        edge_bits=edge,
        parameter_bits=parameter,
        error_bits=float(error_bits),
    )


def unclustered_baseline_bits(
    n_members: int,
    n_relations: int,
    n_candidate_pairs: int,
    n_memory_nodes: int,
    parameter_widths: Sequence[Tuple[float, float]] = (),
    error_bits: float = 0.0,
) -> CodingCost:
    """Description length of the SAME observations stored UNCLUSTERED.

    The baseline pays a full header PER member (no shared cluster header) and
    states every relation without the cluster's shared context, so a candidate
    must beat it to be worth forming. This is the comparison the shadow-MDL
    instrumentation logs beside the threshold predicate.
    """
    index_bits = node_index_cost_bits(n_memory_nodes)
    member = float(max(0, int(n_members))) * (
        float(ASSEMBLY_HEADER_BITS) + index_bits + ASSEMBLY_SLOT_FLAG_BITS
    )
    edge = (edge_presence_cost_bits(n_candidate_pairs) + ASSEMBLY_SLOT_FLAG_BITS) * float(
        max(0, int(n_relations))
    )
    parameter = 0.0
    for (rng, width) in parameter_widths:
        parameter += parameter_precision_cost_bits(rng, width)
    if parameter == 0.0 and int(n_relations) > 0:
        parameter = parameter_precision_cost_bits(1.0, 0.0) * float(int(n_relations))
    return CodingCost(
        header_bits=float(ASSEMBLY_HEADER_BITS),
        member_bits=member,
        edge_bits=edge,
        parameter_bits=parameter,
        error_bits=float(error_bits),
    )


def admits_by_length(candidate: CodingCost, baseline: CodingCost) -> bool:
    """Admission test (b): does the cluster shorten the description?

    A candidate is admitted only when its total length is strictly SHORTER
    than the baseline length of storing the same observations unclustered.
    Equality is not admission. (Test (a), the sample-size null, is the existing
    ``matching/calibration.NullModel`` rule and is applied by the caller.)
    """
    return candidate.total_bits < baseline.total_bits


#: Formation budget: smallest indivisible cluster is a pair; growth targets
#: MAXIMAL cliques of size 3 and larger (addendum 4), so a cluster is not
#: pairwise-only.
ASSEMBLY_MIN_CLUSTER_SIZE = 2


# ---------------------------------------------------------------------------
# RECORDS (data structures; population-coded, position-invariant, never hashed
# into a node's bytes -- a cluster is a separate store entry, not a node)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MemberRef:
    """One graded member of an assembly.

    ``ref_id`` names a persistent memory node (``is_assembly=False``) or
    another assembly (``is_assembly=True``; a reader handle, so recursion is
    possible). ``weight`` is the graded support for the member; ``tolerance``
    is the width within which the member's population may vary and still count
    as the same member. ``is_label`` marks an ordinary name member that
    co-occurs by repetition -- it is NOT a supervised channel (see the label
    addendum): it enters activation and completion like any member.
    """
    ref_id: int
    weight: float
    tolerance: float
    is_assembly: bool = False
    is_label: bool = False


@dataclass(frozen=True)
class InternalRelation:
    """A relation observed INSIDE the cluster.

    ``class_`` is the registered edge type; ``params`` is the graded
    ``EdgeParams`` prototype (a population-coded arrangement, stored as data);
    ``weight`` is its graded support. Relations exist only where observed.

    [O6] ``lag`` is the relation's ORDER: how many steps separated the
    two endpoints when the relation was observed.  It is zero for the
    relations that carry no order (every relation except SEQUENCE), and
    it is what lets a stored episode say "A then B" and not only "A and
    B".
    """
    class_: str
    params: object
    weight: float
    lag: int = 0


@dataclass(frozen=True)
class OverlapLink:
    """A graded overlap to a COMPETING assembly.

    Stored as data only; the resonance session owns any inhibition dynamics
    that read it.
    """
    other_id: int
    overlap: float


@dataclass(frozen=True)
class Provenance:
    """Where an assembly came from. ``episode_ids`` is bounded by retention."""
    first_experience: int
    last_experience: int
    episode_ids: Tuple[int, ...] = ()
    parent_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class AssemblyRecord:
    """A slow-store cluster: the unit that carries a concept.

    ``members`` may reference sub-assemblies (recursion). ``relations`` are the
    internal-relation prototypes. ``support`` is the experience count; the
    parent assemblies are recorded both on the record and in provenance so
    ``reuse`` is a pure query. ``overlap_links`` point at competitors.
    """
    assembly_id: int
    members: Tuple[MemberRef, ...] = ()
    relations: Tuple[InternalRelation, ...] = ()
    support: int = 0
    parent_ids: Tuple[int, ...] = ()
    overlap_links: Tuple[OverlapLink, ...] = ()
    provenance: Optional[Provenance] = None

    def label_members(self) -> Tuple[MemberRef, ...]:
        return tuple(m for m in self.members if m.is_label)

    def is_label_assisted(self) -> bool:
        """True when this assembly currently holds a name member.

        Every number produced with such a member present is tagged
        "label-assisted" (the declared label cheat)."""
        return any(m.is_label for m in self.members)

    def member_ref_ids(self) -> Tuple[int, ...]:
        return tuple(m.ref_id for m in self.members)

    def fingerprint(self) -> bytes:
        """Deterministic content fingerprint (stable member/relation order)."""
        return serialize_value((
            int(self.assembly_id),
            tuple((int(m.ref_id), float(m.weight), float(m.tolerance),
                   bool(m.is_assembly), bool(m.is_label)) for m in self.members),
            tuple((str(r.class_), float(r.weight)) for r in self.relations),
            int(self.support),
            tuple(int(p) for p in self.parent_ids),
            tuple((int(o.other_id), float(o.overlap)) for o in self.overlap_links),
        ))


@dataclass(frozen=True)
class EpisodeRecord:
    """One fast-store experience.

    ``node_ids`` are the memory nodes active together; ``relation_summaries``
    are the arrangement seen among them; ``label_symbols`` are the name members
    observed (the declared cheat). Append-only; compressed into assemblies and
    then dropped once support suffices.
    """
    episode_id: int
    node_ids: Tuple[int, ...] = ()
    relation_summaries: Tuple[InternalRelation, ...] = ()
    label_symbols: Tuple[int, ...] = ()
    # [O6] THE EPISODE'S ORDER.  ``node_ids`` is kept in FIRST-ACTIVATION
    # ORDER (it used to be sorted, which made every episode an unordered
    # set), and ``sequence`` carries the relations in the order they were
    # observed, with their lags.  A stored episode can therefore say
    # "A then B" and not only "A and B".
    sequence: Tuple[Tuple[int, int, str, float, int], ...] = ()

    def fingerprint(self) -> bytes:
        return serialize_value((
            int(self.episode_id),
            tuple(int(n) for n in self.node_ids),
            tuple((str(r.class_), float(r.weight)) for r in self.relation_summaries),
            tuple(int(s) for s in self.label_symbols),
            tuple((int(a), int(b), str(c), float(w), int(l))
                  for (a, b, c, w, l) in self.sequence),
        ))


# ---------------------------------------------------------------------------
# STORES
# ---------------------------------------------------------------------------
class EpisodeStore:
    """Fast, append-only store; NEVER unbounded.

    Retention rule (specified; the pruning step is run by the consolidation
    session and is logged): an episode is COMPRESSED once every one of its
    active nodes appears in some admitted assembly whose support has reached
    ``min_support``; from that point it is dropped. ``compressible`` reports
    which episode ids satisfy the rule; ``drop_compressed`` removes them and
    returns the dropped ids so the caller can log the pruning.

    The coverage rule alone does NOT bound the store: an episode whose nodes
    are never covered stays forever, so a run with no admitted assembly would
    grow the store by one record per experience. A CAPACITY VALVE therefore
    backs the retention rule: when the store holds more than ``max_records``
    episodes, the OLDEST are dropped (they have had the most chances to be
    covered) and their ids are recorded in ``capacity_drops`` so the caller
    logs the drop -- no episode ever leaves the store unlogged.
    """
    def __init__(self, min_support: int = 2,
                 max_records: Optional[int] = None) -> None:
        self._records: Dict[int, EpisodeRecord] = {}
        self._order: list = []
        self._min_support = int(min_support)
        self._max_records = (None if max_records is None
                             else int(max_records))
        #: Episode ids dropped by the capacity valve, oldest first, awaiting
        #: the caller's log line.
        self.capacity_drops: List[int] = []

    def append(self, record: EpisodeRecord) -> None:
        if record.episode_id not in self._records:
            self._order.append(record.episode_id)
        self._records[record.episode_id] = record
        self._apply_capacity()

    def _apply_capacity(self) -> None:
        if self._max_records is None:
            return
        overflow = len(self._records) - self._max_records
        if overflow <= 0:
            return
        for eid in list(self._order)[:overflow]:
            self._records.pop(eid, None)
            self._order.remove(eid)
            self.capacity_drops.append(eid)

    def take_capacity_drops(self) -> Tuple[int, ...]:
        """The episode ids the capacity valve dropped since the last call."""
        out = tuple(self.capacity_drops)
        self.capacity_drops = []
        return out

    def get(self, episode_id: int) -> Optional[EpisodeRecord]:
        return self._records.get(int(episode_id))

    def records(self) -> Tuple[EpisodeRecord, ...]:
        return tuple(self._records[eid] for eid in self._order)

    # -----------------------------------------------------------------
    # [O6] THE READ API.
    #
    # The store was WRITE-ONLY: the pipeline appended an episode per
    # step, bounded it, logged its drops -- and nothing ever asked it a
    # question.  A memory that cannot be recalled is a log.
    #
    # The read is CUE-DRIVEN and PURE: a cue is a set of memory nodes,
    # and the store answers with the episodes that contain the cue, in a
    # deterministic order, together with the EXPECTED ACTIVATION of the
    # episode's other members under that cue.
    #
    # There is NO completion dynamics here: the expected activation is a
    # function of the cue and the STORED members, nothing else.  The
    # resonance layer owns the dynamics; this layer only reports what
    # was stored.
    # -----------------------------------------------------------------

    def recall(self, cue: Sequence[int],
               limit: Optional[int] = None) -> Tuple[Tuple[EpisodeRecord, float], ...]:
        """The episodes that contain the cue, best first.

        The score is the FRACTION of the episode's members the cue
        covers, tie-broken by the episode's id (oldest first): a cue that
        covers more of an episode is a better match, and among equals the
        older episode is the one that has had more chances to be covered.
        """
        cue_set = {int(n) for n in cue}
        if not cue_set:
            return ()
        scored = []
        for eid in self._order:
            rec = self._records.get(eid)
            if rec is None or not rec.node_ids:
                continue
            members = set(rec.node_ids)
            hit = len(cue_set & members)
            if hit == 0:
                continue
            scored.append((hit / float(len(members)), int(eid), rec))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out = [(rec, score) for score, _eid, rec in scored]
        if limit is not None:
            out = out[:int(limit)]
        return tuple(out)

    @staticmethod
    def expected_activation(cue: Sequence[int],
                             record: EpisodeRecord) -> Dict[int, float]:
        """The expected activation of an episode's members under a cue.

        A PURE function of the cue and the STORED members and relations:
        a member the cue covers is expected at 1.0; a member that was
        OBSERVED in a relation with a covered member is expected at that
        relation's own weight; a member with neither is expected at 0.0.
        Nothing is added that was not observed -- this is a report, not
        a completion.
        """
        cue_set = {int(n) for n in cue}
        members = list(record.node_ids)
        if not members:
            return {}
        out: Dict[int, float] = {n: (1.0 if n in cue_set else 0.0)
                                  for n in members}
        covered = {n for n in members if n in cue_set}
        if not covered:
            return out
        for (parent, child, _cls, w, _lag) in record.sequence:
            if parent in covered and child not in covered:
                out[child] = max(out.get(child, 0.0), float(w))
            elif child in covered and parent not in covered:
                out[parent] = max(out.get(parent, 0.0), float(w))
        return out

    def __len__(self) -> int:
        return len(self._records)

    def compressible(self, assembly_store: "AssemblyStore") -> Tuple[int, ...]:
        supported: set = set()
        for record in assembly_store.records():
            if record.support >= self._min_support:
                supported.update(record.member_ref_ids())
        out = []
        for eid in self._order:
            rec = self._records[eid]
            if rec.node_ids and all(n in supported for n in rec.node_ids):
                out.append(eid)
        return tuple(out)

    def drop_compressed(self, assembly_store: "AssemblyStore") -> Tuple[int, ...]:
        dropped = self.compressible(assembly_store)
        for eid in dropped:
            self._records.pop(eid, None)
        if dropped:
            self._order = [e for e in self._order if e in self._records]
        return dropped

    def drop_covered(self, covered_node_ids) -> Tuple[int, ...]:
        """Retention for a runtime caller that knows its own coverage.

        Drops every episode whose active nodes are all in ``covered_node_ids``
        (the set of nodes held by assemblies whose support suffices). This is
        the SAME retention rule as ``compressible``/``drop_compressed``,
        expressed without a fixture ``AssemblyStore`` so the pipeline can apply
        it against the live cluster mechanism. Returns the dropped ids for a
        single logged pruning line.
        """
        covered = set(int(n) for n in covered_node_ids)
        dropped = []
        for eid in self._order:
            rec = self._records[eid]
            if rec.node_ids and all(n in covered for n in rec.node_ids):
                dropped.append(eid)
        for eid in dropped:
            self._records.pop(eid, None)
        if dropped:
            self._order = [e for e in self._order if e in self._records]
        return tuple(dropped)


class AssemblyStore:
    """Slow store of `AssemblyRecord`s, addressed by assembly id."""
    def __init__(self) -> None:
        self._records: Dict[int, AssemblyRecord] = {}

    def add(self, record: AssemblyRecord) -> None:
        self._records[int(record.assembly_id)] = record

    def get(self, assembly_id: int) -> Optional[AssemblyRecord]:
        return self._records.get(int(assembly_id))

    def records(self) -> Tuple[AssemblyRecord, ...]:
        return tuple(self._records[k] for k in sorted(self._records))

    def __len__(self) -> int:
        return len(self._records)

    def containing(self, ref_id: int) -> Tuple[int, ...]:
        """Assembly ids whose members include ``ref_id`` (parents of a node)."""
        rid = int(ref_id)
        return tuple(sorted(
            rec.assembly_id for rec in self._records.values()
            if rid in rec.member_ref_ids()
        ))


# ---------------------------------------------------------------------------
# READER HANDLES (stable; recursion bounded)
# ---------------------------------------------------------------------------
#: Depth cap and fan-out budget for reading readers-of-readers. Stated here so
#: the readout is bounded, never an open recursion. The depth cap is
#: RECONCILED with the graph's own nesting limit `P.D_MAX`: one notion of depth
#: across the cluster mechanism, not two.
ASSEMBLY_READER_MAX_DEPTH = int(P.D_MAX)
ASSEMBLY_READER_MAX_READERS = 64


@dataclass(frozen=True)
class AssemblyReader:
    """A stable handle: an assembly id plus on-demand activation.

    The activation itself is computed by the consolidation session; the handle
    only carries the identity and the recursion position so the bound is a
    property of the handle, not of a caller's bookkeeping.
    """
    assembly_id: int
    depth: int = 0

    def can_recurse(self) -> bool:
        return int(self.depth) < ASSEMBLY_READER_MAX_DEPTH


# ---------------------------------------------------------------------------
# LABELS -- DECLARED CHEAT (current build/test phase only)
# ---------------------------------------------------------------------------
#: Labels currently enter as ordinary co-occurring members and MAY assist
#: activation/completion during build/test. This is a declared cheat: it is
#: logged in the session notebook and every number produced while a label
#: member is present is tagged "label-assisted". The eventual form is labels as
#: physically recognized visual chunks (word/symbol shapes through the visual
#: front end); an invisible abstract id tag is never allowed.
LABEL_ASSIST_ENABLED = True


# ---------------------------------------------------------------------------
# SCORER-OF-RECORD (switches ONCE, after information parity, logged)
# ---------------------------------------------------------------------------
SCORER_VIEW_MEMORY = "view_memory"
SCORER_ASSEMBLIES = "assemblies"


@dataclass
class ScorerOfRecord:
    """The single category scorer. View memory is the scorer of record until
    assemblies show information parity under the shuffled-label control on the
    same protocol; then ONE logged switch. No mid-flight swaps."""
    current: str = SCORER_VIEW_MEMORY
    switched: bool = False

    def switch_once(self, parity_ok: bool) -> str:
        if parity_ok and not self.switched and self.current != SCORER_ASSEMBLIES:
            self.current = SCORER_ASSEMBLIES
            self.switched = True
        return self.current


# ---------------------------------------------------------------------------
# LEGACY LIST (recorded; deletion only by a logged pruning step, never here)
# ---------------------------------------------------------------------------
#: Legacy node classes: superseded by the single cluster mechanism. Retained in
#: the registry (no deletions) but not built upon.
LEGACY_NODE_CLASSES: Tuple[str, ...] = ()

#: RegionShape is labelled legacy but STAYS ACTIVE until assemblies show
#: information parity under the shuffled-label control (see ScorerOfRecord).
LEGACY_ACTIVE_UNTIL_PARITY: Tuple[str, ...] = ("REGION_SHAPE",)

