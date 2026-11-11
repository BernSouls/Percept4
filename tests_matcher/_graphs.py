"""Tiny hand-built graphs for the matcher pins (no image, no front end)."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from core.graph_types import Subgraph
from core.types import EdgeType, TypedSignature

Sigs = Dict[int, Tuple[str, TypedSignature]]


def tex(a: float, b: float) -> TypedSignature:
    """A bins-only TEXTURE signature (two graded slots, range 10 each)."""
    return TypedSignature(kind="CANONICAL", class_="TEXTURE",
                          quantized=(float(a), float(b)))


def graph(nodes: Dict[int, Tuple[str, TypedSignature]],
          edges: Iterable[Tuple[int, int, EdgeType]] = (),
          edge_params: Optional[dict] = None,
          node_weights: Optional[Dict[int, float]] = None,
          ) -> Tuple[Subgraph, Sigs]:
    """(Subgraph, sigs) from ``{node_id: (lens_type, signature)}``."""
    sg = Subgraph(nodes=frozenset(nodes), edges=frozenset(edges),
                  edge_params=dict(edge_params or {}),
                  node_weights=dict(node_weights or {}))
    return sg, dict(nodes)
