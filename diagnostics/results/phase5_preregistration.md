# Stage 5 pre-registration (written before any library edit of this stage)

Tree: `main` 9315d24, front-end digest c1ea81ae1b6c, 79 graphs (4 references + 75 EVAL).
Baseline bars on this checkout, before any edit: matcher pins 390 passed, figground
99 passed / 4 xfailed. `_u27_selftest.py` is not in this tree (deleted on `main`), so that
bar cannot be run as written; see REPORT.md.

## Phase A: what each fix is expected to change in the rebuilt graphs

| fix | rule | expected in the census (`phase5_graph_census_after.txt`) | would count as failure |
|---|---|---|---|
| A2 | A lens that runs inside step t reads the association strength valid during step t (`edge_association_strength`), not the EMA weight that is written after the tier. | CurveComplex components with more than one member appear; CURVE_CONTAINMENT same-anchor share falls below 100 %; `rel_dist` and `polar_angle` take more than one value; fewer CurveComplex detections per image (components merge); SYMMETRY / CONCENTRICITY / PART_CONTAINMENT counts move. | The median graph has one complex holding more than half of its Contour nodes (the image-spanning weld the code warns about). Then the link rule is wrong, not just its reader, and the fix is not kept as is. |
| A2b | The parent of a containment relation is located at the centroid of its footprint, the same centre `_member_geometry` and PART_CONTAINMENT use. A bearing of a zero separation is undefined and is ABSENT. | `polar_angle` is unmeasured exactly on the edges whose child sits at the parent centre (single-member complexes); measured elsewhere. | `polar_angle` measured with one value. |
| A1 | A size ratio is written only when the two ends have independently measured sizes. For a whole and its member that is the footprint extent of each (covering radius of the positioned member anchors). | CURVE_CONTAINMENT `scale_ratio` measured on edges whose two ends each have at least two positioned members, values <= 0 (a member is not larger than its whole), more than one value. CO_LOCATED `scale_ratio` stays 0 % measured: both ends are pooled over the same anchor, so the ratio is a tautology (pinned), not a measurement. `Detection.scale` is not touched. | CURVE_CONTAINMENT `scale_ratio` positive on any edge, or frozen. |
| A3 | Absence is never a neutral number. | No graph changes (the neutral triple never reached an edge). The bins helper returns UNKNOWN. | Any edge changes. |
| A4 | A lookup key is an index: categorical in both tables, uniform (no-information) activity in the population. | No bin changes. Edge populations of the flag classes unchanged (already uniform). | Any bin changes. |
| A5 | A population layout axis the writer can measure is written. Closure is a two-level topological fact, so its activity is one-hot. | PEAK_CHAIN populations carry 4 axes; node identity unchanged (the population is not in the signature bytes). | PEAK_CHAIN node count per graph changes. |

Known before the rebuild: the PEAK_CHAIN closure bin is 0 on all 1,442 chain nodes of the
baseline graphs, so the filled axis will be a true constant on this data.

## Phase B: definitions and stop rule (fixed before the census is computed)

FRAME lens types (pinned list, C1): Contour, ContourFragment, Corner, Junction, T, Y,
CurveComplex, Orientation, CONTOUR_PEAK, PEAK_CHAIN, EndStopped, IllusoryContour,
RegionShape. ATTRIBUTE lens types: Color, Texture, Position, Boundary, Polarity,
TextureBoundary, BorderOwnership, Surface, Onset. A lens type in neither list is reported
and counted as ATTRIBUTE (it can never drive growth). A STRUCTURAL edge is an edge whose
two endpoints are FRAME nodes.

Per graph: (1) FRAME node count; (2) FRAME nodes with >= 1 and >= 2 structural edges;
(3) largest connected component over structural edges; (4) share of structural edges with
at least one measured graded axis (a slot that is neither UNKNOWN, nor an identity slot,
nor a lookup key); (5) the reachable set: the largest connected component over structural
edges that have at least one measured graded axis. (5) is the upper bound of what the
growth rule can ever admit from its best seed, because admission requires a structural
relation that agrees on a measured axis.

STOP RULE. Stop and do not build the matcher if, over the 79 graphs, the median of (3)
is below 8 nodes, or the median of (5) divided by (1) is below 0.25. The 10th percentile
of both is reported next to the median. If the rule does not fire, Phase C is built.
