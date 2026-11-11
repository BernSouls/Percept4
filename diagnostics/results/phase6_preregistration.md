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

## Amendment 2 (new branch, after the user's review of R1-R3 and R5)

WHAT HAD BEEN SEEN when this amendment was written: the first attachment
census (79 graphs: 39.2 % of geometric candidates ambiguous; Orientation ->
Contour 18 unambiguous against 789 ambiguous; tangent_angle present on 86.2 %
of edges, absent on every Corner and ContourFragment edge; EndStopped against
chains at a 56-degree median); the R5 offline verdict (does not fail: 6 / 0 /
0 graphs against a bound of 20); the map switch on one ETH image and on 18
synthetic scenes; the closure census under the per-channel discard.  NOT seen:
any reachability figure of a stage-6 graph, any tangent error against truth,
any multi-step trajectory of a woken channel, any matcher output, HELDOUT.

### 2.1 Three deviations already made, recorded

a. THE MAP FIX HAS ITS OWN SWITCH (`BINDER_COMPLETE_POSITION_MAP`), apart from
   the relations switch (`GRAPH_STAGE6_RELATIONS`).  Section 8 said "one
   grouped switch".  Reason: completing the map wakes five EXISTING channels
   (+1,195 edges on one image); that must be measurable alone.
b. A CONTOUR HOST'S TANGENT IS ITS OWN `mean_theta` SLOT, not "the host segment
   at the foot" (section 2).  Reason, measured: a Contour's vertices are
   region-grid cells, so its segments only take the grid's directions and
   every feature kind read a 45-degree median against them.  A PEAK_CHAIN host
   keeps the segment at the foot (its vertices are pixel positions).
c. R5 WAS WIRED INTO THE LENS BEFORE ITS OFFLINE VERDICT CAME IN (behind the
   switch, off).  Section 4 said measure first, then build.  The verdict
   passed; the rule is run on the ETH graphs through the real lens before any
   default (2.7).

### 2.2 The feature's own tangent, kind by kind

* ContourFragment.  The signature's `theta` and `tangent_vs_radial` do NOT
  determine the tangent: `tangent_vs_radial` is stored as an absolute value
  (`abs(_axial_wrap_sf(chord - theta))`, perception/polar.py), so the tangent
  is theta + tvr OR theta - tvr.  Two answers are not a measurement.  The lens
  computes the chord direction itself and throws its sign away; the fix is to
  carry the chord direction (axial) on the fragment and on its detection.
  Also found: every fragment's anchor is its parent contour's REGION, so all
  fragments of a contour share one position; the fragment's own midpoint is
  carried the same way and is the position the attachment uses.
* Corner.  The orientation is the corner's BISECTOR (the Corner signature's
  own `orient` slot: the external-angle bisector folded to [0, pi), 12 bins,
  bin centre).  Chosen over the turn angle because the bisector is a direction
  in the image, comparable to a host tangent, while the turn is a scalar the
  signature already holds.  On its own chain the angle between the bisector
  and an arm is (pi - turn) / 2, so the slot is informative there too.  A
  Corner of the JUNCTION class (emitted by the junction lens) with two arms:
  bisector = ref_theta + gap0 / 2; with any other arm count: ABSENT.
* EndStopped.  Hypothesis (the user's): the ENDSTOPPED `theta` slot is already
  the terminating bar's orientation, and `carrier_to_tangent` adds a second
  90 degrees.  THE TEST, fixed now: a rendered probe of isolated bars at known
  orientations (every 15 degrees, plus the half-steps); for every EndStopped
  detection within one scale of a bar END, the axial difference between the
  slot's theta and the bar's orientation.  TANGENT iff the median is below
  22.5 degrees; CARRIER iff above 67.5; otherwise undecided and the EndStopped
  tangent is ABSENT.  The ILLUSORY_INDUCER `alignment` keeps the lens' own
  quantity (the angle the lens tested), whatever the outcome.
* Orientation and CONTOUR_PEAK: unchanged (slot 0, tangent), now CHECKED
  against truth (2.6) instead of against a host.

### 2.3 Tangent agreement as a disambiguator (the design gap)

The ambiguity test of section 2 uses distance alone.  It leaves 39 % of the
geometric candidates unattached and nearly all non-member Orientations.

* MEMBERSHIP IDENTIFIES, ATTACHMENT MEASURES.  A membership flag (a peak
  listed in a chain, an Orientation anchor listed in a Contour) says WHICH
  host; it carries no geometry.  The attachment supplies where on the host,
  how far and at what angle.  Reading the flag to choose the host and then
  measuring against that host is not circular.
* THE RULE.  It applies only where the distance test says "ambiguous" (or
  where the data names several hosts of one kind), and only to a feature that
  has a tangent.  CONTENDERS = the hosts of the kind with d <= D_max and
  d - d1 < 0.5 scales (or the named hosts).  For each contender, a = the angle
  between the feature's tangent and that host's tangent at the feature's foot
  on it.  With a1 <= a2 the two smallest: the feature attaches to the
  contender of a1 iff a1 <= pi/12 AND a2 - a1 >= pi/12.  Otherwise it stays
  ambiguous and nothing is written.
* The constants: pi/12 (15 degrees) is one step of the peak's orientation
  channels and two Orientation bins: "parallel" means within one step, and
  the rival must be at least one step further.  Not tuned on any count.
* REPORTED: the outcome table with a fifth column "by tangent"; the recovery
  among the 789 Orientation -> Contour cases and among all 7,597; the
  `tangent_angle` of tangent-resolved edges separately (it is <= 15 degrees
  by selection, which is a selection effect, not a finding).

### 2.4 LINE_TERMINATION gets an ambiguity test

`_run_line_termination` binds a terminator to its NEAREST parent within 2.5
scales with no test.  With the map completed, EndStopped terminators enter
it (3,399 candidates on the 79 graphs).  Under `BINDER_COMPLETE_POSITION_MAP`
the parent choice follows the same rule as everywhere else: if a second parent
is within 0.5 scales of the nearest in distance, the terminator writes
nothing (counted).  Off, the writer is the old writer, byte for byte.

### 2.5 R6: which graph the gate is computed on

The gate (section 5, unchanged thresholds) is computed with
`GRAPH_STAGE6_RELATIONS` ALONE.  The map switch is a separate row, reported,
never part of the verdict.  The per-channel border switch is off.

### 2.6 Tangent correctness (not presence)

* TRUTH, on the synthetic figground scenes: the visible-label map (nearest
  object in depth, else background); a boundary pixel is one whose label
  differs from a 4-neighbour; the true tangent there is perpendicular to the
  gradient of the Gaussian-blurred (sigma 2 px) indicator of the pixel's own
  label.  A boundary pixel is SMOOTH when the true tangents of the boundary
  pixels within 4 px of it all lie within 20 degrees of its own; the others
  (corners, junctions) are reported apart.
* REPORTED per feature kind: for features within one scale of a true
  boundary, the axial error between the feature's tangent and the true
  tangent at the nearest boundary pixel (median, p90, share under 15 and
  under 30 degrees), on smooth pixels and on all.  Per host kind: the same
  for the host's tangent at the foot.  For attachments: the error of
  `tangent_angle` against the angle between the two true tangents.
* On ETH-80 (no truth): the distribution of `tangent_angle` after the fixes,
  per feature and host kind, with the known-host rows as the reference.

### 2.7 Before any default

* R5 through the real lens on the 79 ETH graphs: the three failure criteria of
  section 4 recomputed on the built graphs (complex membership read from the
  CurveComplex detections), beside the offline figures.
* THE MAP SWITCH ON THE MULTI-STEP ROUTE.  The woken channels emit 76-95 % of
  their candidates against a target rate of 0.2.  Protocol: 16 images (the 4
  references and the first 3 EVAL images of each class), 8 presentations,
  switch off and on; per affected channel and per step: theta, the emission
  rate, candidates, emitted (median over images).  The switch cannot default
  on while any affected channel's theta is still moving by more than 0.01 per
  step at step 8, or has reached its ceiling.

## Amendment 3 (after the first truth table and the census under Amendment 2)

WHAT HAD BEEN SEEN when this amendment was written: the attachment census under
Amendment 2 (`phase6_attachment.txt` as committed in 22aaea1: 7,571 ambiguous
under distance alone, 1,662 recovered by tangent agreement; Orientation ->
Contour 789 -> 28 recovered); the first truth table (bar probe: 20 detections
at bar ends, median 7.5 degrees; synthetic scenes, share of a kind's
orientation within 15 degrees of the true tangent on smooth boundary:
CONTOUR_PEAK 0.97, Orientation 0.99, EndStopped 0.09 (0.41 after a 90-degree
turn), ContourFragment 0.00 (median error 90.0), Corner 0.37 on 19; hosts:
Contour 0.97, PEAK_CHAIN 0.98); a 12-graph look at the Orientation features
several Contours name.  NOT seen: any reachability figure of a stage-6 graph,
the Corner bisector against a true bisector, the stream-2 polylines traced
without the rotation, any multi-step trajectory, any matcher output, HELDOUT.

### 3.1 Several named hosts are several facts (correction to 2.3)

Amendment 2.3 let tangent agreement decide "where the data names several hosts
of one kind", and counted the rest as ambiguous.  That clause is WITHDRAWN.

* What the 789 are (12 graphs, 119 such Orientation features): the Contour
  lens lists the feature as a member of 3 Contours (89), 4 (17), 2 (12) or 5
  (1), and in 101 of the 119 the feature is the END vertex of every one of
  them.  They are branch cells of the cell-adjacency graph: the lens ends each
  branch on the shared cell.  The Contours' mean orientations are within one
  bin of the feature's own (the gap between the two best is under 15 degrees in
  97 %), so no tangent rule can pick one, and none should: the feature is on
  all of them.
* The ambiguity rule is about a GUESS between candidates.  A recorded
  membership is a fact; three recorded memberships are three facts.  By the
  principle 2.3 itself states (membership identifies, attachment measures),
  the feature is attached to EACH host that names it, and each attachment is
  measured on its own host (where on it, how far, at what angle).
* Counted apart ("known, several") in every table.  Tangent agreement keeps
  its registered role for geometric contenders only.
* BECAUSE THIS RULE WAS WRITTEN AFTER SEEING PRESENCE COUNTS, its contribution
  to the gate is isolated: a parameter `ATTACHMENT_EVERY_NAMED_HOST` (True;
  False restores the Amendment-2 behaviour) and R6 is reported both ways.  The
  verdict row is the amended rule (True); the False row is printed beside it.

### 3.2 An orientation must pass truth before it is used (addition to 2.6)

2.6 fixed what is reported and no consequence.  The consequence, one rule for
every feature kind:

* A kind's own orientation enters `tangent_angle` and tangent agreement only
  if, on the synthetic scenes, AT LEAST HALF of that kind's detections within
  one scale of a smooth true boundary pixel lie within 15 degrees of the true
  tangent there.  Otherwise the kind's orientation is ABSENT (its place, i.e.
  `arc_end_dist`, `perp_dist` and the ordinal, is unaffected).
* 15 degrees is the constant of 2.3 (one channel step); "half" because a
  median beyond one step means the typical value is in another bin.
* This rule is written after the first truth table, so it is not blind for
  the four kinds already seen; their figures are far from the line on either
  side (0.97, 0.99 against 0.09, 0.00).  It IS blind for Corner (below).
* EndStopped.  The registered bar test (2.2) returns TANGENT and stands as a
  statement about bars: on an isolated bar the slot is the bar's orientation,
  so no reader should turn it by 90 degrees.  On the boundaries of filled
  shapes the same slot is within 15 degrees of the tangent in 9 % of the
  detections: by the rule above the orientation is ABSENT.  Hypothesis for the
  cause, to be measured in the same diagnostic: the end-stop score is a
  RELATIVE suppression, so on a smooth edge an off-axis channel that passes
  the energy gate scores full suppression along its own axis and wins; the
  test is the share of EndStopped detections whose winning channel is the
  locally dominant channel.
* ContourFragment.  The chord carried under 2.2 is, against truth, the
  boundary NORMAL (median error 90.0 degrees, none within 30).  Hypothesis for
  the cause: the stream-2 tracer (`_extract_association_field_contours`)
  follows `carrier + pi/2`, while the bank's channel index already is the
  orientation of the bar it answers (the kernel's docstring; Orientation and
  CONTOUR_PEAK, which read the index as it is, are within 4 degrees of
  truth), so the traces run ACROSS edges.  THE TEST, fixed now: on the
  synthetic scenes, the angle between the local direction of every stream-2
  polyline and the true tangent, as traced and with the rotation removed (the
  energy rolled by half the channels, which is the same tracer following the
  index as it is); "across" iff the median as traced is above 67.5 degrees and
  the median with the rotation removed is below 22.5.  Whatever the outcome
  the fragment's orientation is ABSENT under the rule above (an angle that is
  90 degrees by construction is not a measurement); the fragment's own
  midpoint stays.  The tracer is NOT changed in this stage: it feeds the
  contour regions, the fragments and the shape lenses, so changing it changes
  the benchmark's nodes.  Reported with the evidence, and asked.
* Corner.  A bisector is not a tangent, so its truth is the TRUE BISECTOR:
  the true corner is the centroid of the non-smooth boundary pixels within 4
  px of the non-smooth pixel nearest the detection (within one scale); the
  true bisector is the direction of the mean of the unit vectors from it to
  the boundary pixels between 0.75 and 1.5 scales away (undefined when that
  mean is shorter than 0.2 or rests on fewer than 6 pixels).  The Corner
  orientation is used iff at least half of the Corner detections that have a
  true bisector lie within 15 degrees of it.  Not yet computed.

### 3.3 The convention sites, listed (none changed in this stage)

`core.quantize.carrier_to_tangent` documents "bank theta is the carrier
(normal) axis".  The sites that apply it or its equivalent, with what each
would mean if the index is the bar's own orientation: the stream-2 tracer
(tested, 3.2); `_oriented_tangents` (the curvature sign of a Contour: a lead
for the frozen `curv_agree` / `curv_diff` slots of R4, to be diagnosed
there); `_opens_toward` (tests the open side against the axis turned by 90
degrees, while `compute_endstopped_map` measures it along the axis);
`_arm_tangent_stats` (a doubled-angle concentration: a global turn changes
nothing); `IllusoryContourLens` (joins terminators across their bars, which
is what a line-end contour is: correct if the slot is the bar).

## Amendment 4 (Track M, written with the walker's first draft, before it had run on anything)

WHAT HAD BEEN SEEN: no output of the walker on any graph, real or synthetic;
no pin of M3 had been run.  The census figures listed under Amendment 3.

Section 6 stands except where a point below says otherwise.

* 4.1 THE PROPOSAL ORDER gains two keys.  Section 6: "(number of admitted
  pairs supporting it, descending; canonical keys)".  Now: (supporting pairs,
  descending; TOTAL SLACK, ascending; signature similarity, descending;
  canonical keys).  Slack of one supporting pair = the number of shared
  measured axes on which its best agreeing relation differs by one bin; total
  slack = the sum over the supporting pairs.  REASON: with canonical keys
  alone a graph matched against itself need not come out as the identity (a
  feature can be proposed toward a neighbour with a smaller key before its
  own image is proposed); with slack and similarity ahead of the keys the
  identity proposal is never behind a competitor (its support is at least as
  large, its slack is 0, its similarity is 1).  The similarity therefore has
  to be known when a proposal is made, so it is computed then; the BUDGET
  counts every distinct pair whose similarity is computed (5 % of
  |FRAME_q| x |FRAME_r|, unchanged), seeds included.  Similarity still adds
  to no score: it gates, and it orders ties.
* 4.2 GROWTH EDGES, stated exactly: an edge between two FRAME nodes whose
  type is declared in `core.types.LEGACY_TO_CATEGORY` with a category other
  than PROXIMITY (so CO_LOCATED, CONTACT, TOPOLOGY, DEGREE and
  CROSS_SCALE_IDENTITY never are), and which carries at least one measured
  axis.  An edge with no measured axis cannot agree with anything: it is
  neither a growth edge nor a relation of the score.
* 4.3 THE SCORE, one clarification: a pair of corresponding edges with no
  axis measured on BOTH weighs nothing (counted on neither side).  Section 6
  left this case open.
* 4.4 SEEDS pass the order constraint like every other pair.
* 4.5 THE CANONICAL NODE KEY is (lens type, signature class, relation,
  quantised slots).  A graph's nodes are pattern nodes, so the key is unique
  inside a graph; node ids break a tie only between nodes with one key.
* 4.6 THE SWITCH `MATCH_RELATION_WALKER` is read at one place,
  `matching.relation_walker.match_evidence`, which is `evidence_score` when
  the switch is off.  The memory system's own callers of `evidence_score`
  (mapper, consolidator) are not rewired in this stage: the walker is
  measured on the benchmark first.
* 4.7 ORDER CONSTRAINT, the test made exact.  Open boundaries: over every two
  admitted features with distinct arc positions on both boundaries, the
  product of the two position differences has one sign.  Closed boundaries:
  the reference positions, read in query order, are a rotation of a rising or
  of a falling sequence.  Features at one position on either side constrain
  nothing.
* 4.8 THE BUDGET ON A SMALL FRAME (added when the first pins ran, on
  hand-built graphs of 6 to 13 frame nodes; no real-graph result motivated
  it).  5 % of |FRAME_q| x |FRAME_r| is less than one look per node when the
  frames are small (13 x 13: 8 pairs), so no walk can even confirm a graph
  against itself.  Budget = max(5 % of the product, 8 looks per node of the
  smaller frame), never more than the product.  On the ETH graphs (frames of
  158 to 402 nodes) the 5 % term is the larger one from 160 nodes up, so the
  registered budget is what binds there.
* 4.1, CORRECTION OF ITS REASON (written when the first mutation run showed
  that no pin held the rule).  The reason given above is wrong: with the
  canonical keys alone a graph matched against itself DOES come out as the
  identity (whenever (u, v) is proposed, (v, v) and (u, u) are proposed with
  at least the same support and one of them has the smaller key, so an
  identity pair is always served first).  The two keys are kept for another
  reason, stated now: among equally supported proposals the registered order
  was the canonical key order, which is arbitrary with respect to the
  measurement, while the walk is meant to be best-first; exact agreement on
  the measured axes is better evidence than agreement within a bin, and of
  two candidates that agree equally the more similar one is the safer.  No
  result motivated it (the walker had run on the four reference graphs only,
  with this order, and nothing was compared).  Pinned.
