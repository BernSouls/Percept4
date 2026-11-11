# Stage 6 pre-registration: "build the roads, then build the walker"

Written BEFORE any stage-6 census number, any library edit and any score.
`phase5_preregistration.md` is not edited.  Amendments to this file are
appended at the bottom, dated, with the reason; nothing above an amendment is
rewritten.

> RESTORATION NOTE.  This file was first committed locally as `174bbed` on the
> branch `arena/dea2ca5a-percept4`, before any stage-6 number.  That commit was
> never pushed, and the workspace was reset when the branch was merged.  The
> text of sections 0-9 and of Amendment 1 below is restored word for word from
> the session transcript.  Everything from Amendment 2 on is new and is dated.

## 0. What had been seen when this was written

Honest list, so the reader can judge what could have shaped a rule:

* everything in the stage-5 report (single-step census, slot truth, the
  post-verdict what-ifs 0.287 / 0.302 / 0.48; the what-ifs shape no rule here);
* a smoke test of the multi-step route on ONE image (`apple1-000-000`) at
  1, 2, 3 and 5 steps: node and edge counts per type only (Contours 33 / 30 /
  19, Orientation nodes 73 / 69 / 47, CurveComplex 24 / 3 / 10; V1 Orientation
  detections 114 / 106 / 76 and 43 at 5 steps).  No census, no reachability;
* the 0b closure census (`phase6_closure.txt`, a Phase-0 diagnostic with no
  criterion attached): extraction margin 56 px, share of the silhouette inside
  the peak interior, cycles in the chain adjacency, 0 rings of 1,569 chains;
* a position trace of the same one image: which detection kinds have a position
  in the binder's map and which only in the lens tier's map;
* code reading of the lenses, the binder writers and the chain former.

Not seen: any attachment statistic, any distribution of a new slot, any complex
grouping under a new rule, any matcher output, any number from HELDOUT.

Units.  "Scale" below is the step's extraction scale (12.8 px on a 256 px
image); it is the unit the binder already calls a wavelength
(`dist / extraction_scale`).  The region grid cell is 1.25 scales; half a cell
diagonal is 0.88 scales.  Angles are radians.

## 1. Phase 0a: the route

* The multi-step route is the same image presented N times to ONE concept
  graph, binder, sensory floor and settle loop, with
  `ConceptGraph.step_28_update_edges` and `Binder.finalize_step` between
  presentations (the order of `Pipeline.step`).  The graph read out is the last
  presentation's window, exactly as the single-step route reads its one step.
* N = 3 is the registered multi-step route (the smallest N at which a
  one-step-delayed reader of `Edge.weight` has run twice on a weight the EMA
  has updated).  N = 2 is reported beside it.  The repository declares no N for
  a still image; this is my choice and is labelled as such.
* The census is the stage-5 Phase B census, unchanged (`phase5_feasibility.py`
  with a `--steps` option), over the same 79 graphs (4 references + 75 EVAL).
* "A stage-5 finding survives" means, on the N = 3 route:
  * F1 (the verdict): median reachable / FRAME < 0.25;
  * F2 (the ceiling): median largest structural component / FRAME < 0.5;
  * F3 (islands): each of CONTOUR_PEAK, EndStopped, PEAK_CHAIN,
    ContourFragment, RegionShape, IllusoryContour still has 0 measured
    structural edges (judged per type);
  * F4: EndStopped and RegionShape still have no edge of any kind;
  * F5 (A2): median (largest complex / Contours) per graph is reported; the
    single-step finding "every complex is one Contour" survives if the median
    number of member Contours per complex is 1; the stage-5 trial finding
    "everything welds" is reproduced if the median largest-complex share
    exceeds 0.5;
  * F6: no chain is closed.
* Decision rule.  Stage 6 builds and measures on the single-step route (the
  route of every earlier stage and of the benchmark) unless 0a shows F1 does
  not survive; in that case I stop and report before building.  Whatever 0a
  shows, both columns are reported.

## 2. R1 / R2 / R3: attachment

One writer, one relation: a local feature's place on an extended boundary.

* HOST: a detection that carries an ordered polyline of at least two
  positioned vertices: a PEAK_CHAIN (its member peaks in chain order) or a
  Contour (its member anchors in member order).  Closed when the detection
  says so.
* FEATURE: a detection of kind Orientation, CONTOUR_PEAK, EndStopped, Corner or
  ContourFragment that has a position.  (R3 adds RegionShape and
  IllusoryContour through relations that are KNOWN, see below.)
* MEASUREMENTS, per (feature, host):
  * foot = the closest point of the host polyline to the feature's position;
  * `arc_pos` = arc length from the host's first vertex to the foot / total
    length, in [0, 1], in the host's own member order;
  * `perp_dist` = |feature - foot| / scale;
  * `tangent_angle` = angle between the feature's own orientation and the host
    segment at the foot, folded to [0, pi/2]; ABSENT when the feature kind
    carries no orientation or the host segment has zero length.
* ADMISSION RADIUS D_max = 1.0 scale.  Reason (geometry, no data): a
  cell-anchored feature can sit up to 0.88 scales from a boundary crossing its
  own cell, and the next cell centre is 1.25 scales away.
* KNOWN HOST.  When the data structure itself names the host (a peak is a
  listed member of its chain; a Corner's `rho` names its chain and vertex; an
  Orientation anchor is a listed member of a Contour), the host is that one and
  no test is needed: it is a recorded fact, not a guess.
* THE AMBIGUITY TEST (for every other feature / host-kind pair).  Let d1 <= d2
  be the feature's two smallest distances to DISTINCT hosts of the same kind
  (d2 over all such hosts, with no radius cut-off; d2 = infinity if there is
  one host).  The attachment is written iff d1 <= D_max AND d2 - d1 >= 0.5
  scales.  Otherwise nothing is written (ABSENT) and the feature is counted as
  "ambiguous" (d1 <= D_max, margin failed) or "no host" (d1 > D_max).  The 0.5
  margin is half a grid cell less a pixel-rounding allowance: the position
  uncertainty of a cell-anchored feature.  It is one constant for every kind.
* POOLING RULE.  A graph node pools detections.  If several detections of ONE
  feature node attach to ONE host node, an axis is written only when their
  values fall in the same bin of that axis or in adjacent bins (then the mean
  is written); otherwise that axis is ABSENT on that edge.  Reported.
* GAUGE.  A chain's member order is chosen by the chain former in the image
  frame (`_diameter_path`); it has no physical meaning.  So `arc_pos` is an
  ORDINAL coordinate (stored, read only by the order constraint, never compared
  across graphs as a value and not counted as a measured axis), and the
  comparable measured axis is `arc_end_dist` = min(arc_pos, 1 - arc_pos) in
  [0, 0.5], the distance from the nearer end.  On a CLOSED host there is no end
  and no origin: `arc_end_dist` is ABSENT there.  This corrects the brief's
  "arc-length position 0..1" as a comparable slot; the reason is the gauge.
* R2.  PEAK_CHAIN -> CONTOUR_PEAK is the known-host case of the same writer
  (`perp_dist` is then 0 by construction: it is written as measured, and its
  constancy is reported, not hidden).  The successor relation joins
  consecutive member peaks of a chain: `arc_gap` (distance between them /
  scale) and `turn` (unsigned angle between their tangents, [0, pi/2]).
  Unsigned because the sign flips with the gauge.
* R3.  EndStopped is a FEATURE of the writer above ("what it terminates" is
  the host it attaches to, with `arc_end_dist` saying how near the end it
  sits).  RegionShape: diagnosed first; if (as the code suggests) it is the
  shape of ONE chain, the relation is RegionShape -> that chain (known host)
  with the geometry of the shape's centroid measured against the chain
  polyline; anything that needs an enclosed region is ABSENT on an open chain.
  IllusoryContour -> each of its two inducers (named in its `rho`): `gap`
  (inducer separation / scale) and `alignment` (angle between the inducer's
  tangent and the completion line).
* REPORTED per feature kind and host kind: features, with a host within D_max,
  unambiguous, ambiguous, no host; axes ABSENT by the pooling rule.

## 3. R4: rulers

* Scope: every edge slot whose unit is a distance in scales.  A slot is
  RE-RULED when, over the 79 graphs, more than 25 % of its measured values fall
  outside its span or in its last bin.
* The rule: lower end 0; upper end = the 99th percentile of the measured values
  over the 79 graphs, rounded UP to the next 0.5 scale.  If the writer has a
  hard admission radius below that, the upper end is the admission radius.
  Number of levels unchanged.  Bins stay linear.
* Reported per slot: old span, new span, edges that change bin, entropy of the
  bin histogram (bits) before and after.
* `CURVATURE_SIGN_CONTINUITY.curv_diff` and `CONTACT.depth_step` are diagnosed
  first from the writer's RAW values: if the raw value varies and the quantiser
  flattens it, it is mis-scaled and the same percentile rule applies on its own
  unit; if the raw value is constant by construction, the slot is unmeasured
  and is written ABSENT.
* Own patch, own switch, own bar run.

## 4. R5: the CurveComplex link rule

* Evidence is one step of geometry between Contour ENDS.  For an end e of
  contour a and an end f of contour b: `gap` = distance between the two end
  vertices / scale; the end tangent points outward along the last segment (for
  a one-vertex contour: the contour's mean orientation, axial); `bend` = the
  larger of (angle between a's outward tangent and the connecting vector) and
  (angle between the connecting vector and b's inward direction); with zero
  gap, the angle between a's outward tangent and b's inward direction.
* A link is possible iff gap <= 1.8 scales AND bend <= pi/4.  Reasons
  (geometry, no data): contour vertices sit on the region grid, so consecutive
  pieces of one boundary end in adjacent cells, at most one cell diagonal
  (1.77 scales) apart; and one step of direction on that grid is pi/4.
* Each contour END takes at most one link: candidates in ascending
  (gap / 1.8 + bend / (pi/4), canonical key) order, accepted when both ends are
  free.  So a contour has degree <= 2 and a complex is a simple path or a
  cycle.
* A Junction within 1.8 scales of a contour end takes that end (the nearest
  junction; two junctions within 0.5 scales of each other in distance: none).
  An end taken by a junction is not available for a contour link.  A junction
  may hold any number of ends.
* The temporal EMA (`Edge.weight`) gates nothing and is not read.
* FAILURE CRITERIA (the brief's), counted over the 79 graphs.  The rule FAILS
  if in 20 or more graphs (more than 25 %) any of these holds: (a) one complex
  holds more than half of the graph's Contours; (b) every Contour is in one
  complex (graphs with at least 4 Contours); (c) every complex has one member.
  The distribution of complex sizes is reported whatever the verdict.
* The rule is MEASURED OFFLINE on the 79 graphs before it is built into the
  lens.  If it fails, the measurement and a proposed alternative are reported
  and nothing is built.

## 5. R6: the gate

Unchanged from `phase5_preregistration.md`: the same FRAME list, the same
definitions of structural edge and measured axis, over the 79 graphs.  PASS iff
median largest structural component >= 8 nodes AND median (largest measured
component / FRAME) >= 0.25.  Then the same census on HELDOUT (4 references +
75 HELDOUT images): the gate holds only if HELDOUT passes the same two
thresholds.  HELDOUT graphs are built for this census only; nothing is scored.

`arc_pos` (ordinal, gauge-dependent) does not count as a measured axis.  A
relation whose every measured axis is constant by construction over all its
edges is reported as such next to the gate figure.

## 6. Track M: the matcher

Contract as in the brief.  The constants, fixed here:

* SIGNATURE GATE: same lens type, and the existing graded signature similarity
  of the two nodes >= 0.5.  A gate only: it never adds to a score.
* RELATION CLASS: (edge type, direction as seen from the admitted node, lens
  type of the far node).  Growth edges are structural edges only (both ends
  FRAME).  CO_LOCATED and every proximity-only type are never growth edges.
* AGREEMENT of two edges of one class: on every axis measured on BOTH, the
  quantised values differ by at most one bin (circular axes wrapped), and at
  least one such axis exists.  An axis measured on one side only, or on
  neither, weighs nothing either way.
* SEEDS: 8.  A seed is a pair passing the signature gate that shares at least
  two relation tokens (class + exact bins of its measured axes) each of which
  is carried by at most 2 nodes in its own graph on each side.  Ranked by
  (shared rare tokens, descending; signature similarity, descending; canonical
  node keys), one-to-one.
* GROWTH: (u', v') is proposed only through an admitted (u, v) with u-u' and
  v-v' of one class; admitted iff the gate passes, the two edges agree and the
  order constraint holds.  Proposals are served best-first by (number of
  admitted pairs supporting it with an agreeing relation, descending; canonical
  keys).  One-to-one; the first admitted wins; no backtracking.
* ORDER CONSTRAINT.  For an admitted host pair, the features admitted on it
  must be in the same arc order on both hosts up to the gauge: the first two
  admitted features with distinct arc positions fix the direction (same or
  reversed); every later feature must fall on the same side of every admitted
  one in both hosts under that direction.  A proposal that breaks this is
  rejected.  On closed hosts: the cyclic order under one of the two
  directions.  A whole host traversed the other way is the same boundary (the
  gauge), so it is accepted with the direction bit recorded; what is rejected
  is a mapping that is not monotone.  (This is how I read pin 4; if the brief
  means that a full mirror must be rejected, the chain direction would first
  have to be a measured quantity, which it is not today.)
* BUDGET: proposals examined <= 5 % of |FRAME_q| x |FRAME_r| per pair of
  graphs.  Reported next to the dense count.
* ATTRIBUTES are attached after growth: for a mapped frame pair, an attribute
  type is "shared" when both nodes have a same-anchor CO_LOCATED attribute of
  that type and the two pass the signature gate.
* SCORE: Jaccard over the mapped part: preserved / (relations of the query
  among mapped nodes + relations of the reference among their images -
  preserved), where a relation is a structural edge between mapped frame nodes
  (preserved when its image exists in the same class and agrees) or an
  attribute attachment of a mapped frame node (preserved when shared).
* MINIMUM SUPPORT: a mapping is SUPPORTED iff it has at least 8 frame pairs AND
  at least 10 % of the smaller graph's FRAME nodes.  Below that the pair of
  graphs is UNSUPPORTED: no score.
* BENCHMARK COUNTING: an unsupported reference cannot win.  A query whose four
  references are all unsupported ABSTAINS; an abstention counts as an error in
  the accuracy over all 75 queries, and the unsupported share and the accuracy
  over the supported queries are reported beside it.  A tie between
  references is an error, as in the existing benchmark.

## 7. M3 pin 6: what "same physical role" means

* Scenes: the synthetic figground scenes, built through the real front end
  (`eth80_frontend.build_graph`, single step).  Pairs, fixed here:
  (box_over_box, box_over_box_test), (box_over_box, box_over_box_haze),
  (cylinder_over_box, cylinder_over_box_shift), (f_box_a, f_box_b),
  (f_haze_a, f_haze_b), (f_disk_a, f_disk_b), (f_cyl_a, f_cyl_b),
  (single_disk, f_ctrl_disk), (single_box, f_ctrl_box).
* The renderer's label map: the object mask a pixel belongs to (nearest in
  depth), else "background".
* ROLE of a positioned FRAME node: (the set of labels within one scale of its
  position; the quadrant, in 90-degree sectors centred on the image axes, of
  its position around the centroid of the nearest-in-depth object of that
  set).  A node whose set is {"background"} alone has no role and is left out.
* Two nodes, one per scene, have the SAME ROLE iff their roles are equal
  (objects correspond by mask name; all pairs are upright).
* PRECISION of a mapping = mapped pairs with the same role / mapped pairs with
  a role on both sides.  RECALL = mapped pairs with the same role / the sum
  over roles of min(count in A, count in B).
* THE PIN: over the scene pairs where the old and the new matcher differ in
  (precision, recall), the mean of F1(new) - F1(old) is > 0 and the new matcher
  is higher on at least as many pairs as it is lower.  The number of differing
  pairs, and each pair's two figures, are reported.  The old matcher's mapping
  is the node assignment its own alignment returns.

## 8. Switches and defaults

* Stage-5 writer-truth set: one grouped rollback switch, default ON.
* Stage-6 relations (R1, R2, R3, R5): one grouped switch.  Stage-6 rulers
  (R4): its own switch.  Both default ON iff R6 passes (EVAL and HELDOUT);
  otherwise default OFF, kept and labelled.
* The new matcher: ONE switch, default OFF.  With it OFF the old path is byte
  for byte the old path.
* No level count changes.

## 9. The final measurement

Runs once, and only if R6 passes and every M3 pin is green.

* EVAL-75, two configurations on the same tree and the same graphs: (A) old
  defaults = the matcher switch OFF; (B) the matcher switch ON.  Per class and
  overall accuracy, the paired table (both right / only A / only B / both
  wrong) with the exact two-sided binomial p on the discordant pairs, the
  unsupported share, the mapping-size distribution, candidates examined.
* Then HELDOUT-75, once, the same two configurations, last.
* "Ahead" = B has more correct answers than A on EVAL-75.  HELDOUT is the
  check, not a second chance: whatever it shows is reported.
* Nothing is re-run with a changed constant.  If B is not ahead, the report
  says what the census and the pins showed and where the representation falls
  short.

## Amendment 1 (same day, before any census number): three directions from the user

The user read the Phase-0 closure census and gave three directions.  They widen
the scope; they are recorded here before the work is done.

1. THE POSITION MAP.  Hypothesis to verify, not assume: the binder's position
   map is copied before the higher lens tier runs, so the anchors that tier
   registers (EndStopped, and RegionShape's chain anchor) never reach the
   binder, and `_run_line_termination` / `_run_part_containment` skip them for
   lack of a position.  If confirmed it is plumbing: fix the map, not the lens.
   CRITERION for "confirmed", fixed now: over the 79 graphs, (i) the share of
   EndStopped detections whose anchor is in the binder's map is 0 and the
   share in the lens tier's map is 1; (ii) with the map completed and nothing
   else changed, `_run_line_termination` receives EndStopped terminators and
   emits edges from them.  For RegionShape the three gates the user names are
   counted separately (own anchor position; `len(q) >= 5`; CurveComplex member
   positions), plus the one I add: whether the level the writer lives in is
   ever emitted.
2. THE BORDER DISCARD.  Before any change: the clearance each spatial-frequency
   channel needs, MEASURED as the distance from the border beyond which the
   channel's response on a window agrees with the response computed with the
   true surround (relative error below 5 % of the channel's maximum), beside
   the half-support the kernel's own envelope implies; and the share of the
   silhouette perimeter inside the interior under a per-channel rule.  Then the
   per-channel discard is built behind a switch and the closure census is
   re-run.  THE TEST, fixed now: "closure was broken by the margin" iff rings
   appear on ETH-80 under the per-channel rule; if none appears, the margin
   was not the reason and the per-cycle census (3) names the next cause.
   The switch stays ON only if the matcher pins and figground stay green.
3. THE PER-CYCLE VERDICT CENSUS: the rejection histogram of every fundamental
   cycle the ring test can consider (`phase6_closure.py`, section B2).

Phase 0 ends when these three are reported.  R1 is next, and its ambiguity
rate is the first number.
