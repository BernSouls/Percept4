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

## Amendment 5 (R4, written before any raw slot value had been collected)

WHAT HAD BEEN SEEN: the stage-5 bin census (which slots are frozen or
saturated, as bins); the list of slots and units in `EDGE_UNIT_TABLE`.  NOT
seen: any raw (continuous) value of any slot.

Section 3 stands.  Made exact:

* SCOPE.  The slots whose unit is `wavelengths` (the extraction scale: 18
  slots) and `median_spacings` (3 slots: the same kind of quantity, a distance
  in the writer's own length unit).  For the second group "0.5 scale" reads
  "0.5 of its unit".  The signed slot `PARALLEL.stagger_dist` is judged and
  ruled on |value|, symmetrically.
* THE VALUES are the raw values the writer handed to the edge constructor for
  the edges that are IN the 79 graphs (emitted edges, matched to their
  constructor call by object identity; BOUNDARY_ATTACHMENT, whose writer
  copies the object, by class and bins).  Graphs built with
  GRAPH_STAGE6_RELATIONS on (the configuration that becomes the default if R6
  holds on HELDOUT), so the new slots are judged with the old ones.
* "IN ITS LAST BIN OR OUTSIDE ITS SPAN": value >= hi - (hi - lo) / n_bins, or
  value < lo.
* ENTROPY: Shannon entropy in bits of the histogram of the slot's bins over
  its measured values.  "Edges that change bin": measured values whose bin
  under the new span differs from their bin under the old span.
* ADMISSION RADIUS: read from the writer's code for each slot that is
  re-ruled, and stated next to the new span.
* THE TWO FROZEN SLOTS (`CURVATURE_SIGN_CONTINUITY.curv_diff`,
  `CONTACT.depth_step`) and, because Amendment 3.3 names the same suspect
  (`_oriented_tangents`), `CONTOUR_CONTINUITY.curv_agree`: the raw values are
  listed (count, distinct values, range).  VARIES AND FLATTENED iff the raw
  values take at least 3 distinct values spanning more than one bin width of
  a sensible span; CONSTANT BY CONSTRUCTION iff one value (or the writer's
  code shows the argument is a literal).  The first is re-ruled by the
  percentile rule on its own unit, the second is written ABSENT.
* A slot that is NARROW (most values in its first bins because the span is
  too wide) is reported and NOT changed: the registered rule is about
  saturation.
* THE SWITCH: `EDGE_RULERS_FROM_MEASUREMENT`, default OFF until its own bar
  run (pins, figground) is green and R6 is recomputed with it.

## Amendment 6 (after M3-6 came in RED)

WHAT HAD BEEN SEEN: R6 on EVAL (0.869, largest component 248; the two
separate rows 0.869 and 0.895); the walker's pins (18 green, 9 mutations each
caught); the synthetic harness (M1: identity on 18 of 18; M3-6: F1 lower than
the old matcher on 9 of 9 pairs, precision higher on 4, recall lower on 9);
one scratch trace of three pairs (seeds against truth).  NOT seen: any score
of either matcher on EVAL-75 or HELDOUT-75 under the stage-6 graphs.

* 6.1 THE FINAL MEASUREMENT DOES NOT RUN.  Section 9 and the brief: only if
  R6 passes and every M3 pin is green.  M3-6 is red.  The script
  (`phase6_final_measurement.py`) exists and has only been dry-run on the four
  references.  Nothing in the walker is changed to turn the pin: its constants
  are the registered ones and stay.
* 6.2 THE DEFAULTS STAY OFF; section 8's rule is NOT applied.  Section 8 said
  the relations switch and the rulers switch default ON iff R6 passes.  That
  rule assumed the relations would have a reader.  They have none that is
  adopted: the walker is not ahead, and the old matcher would receive some
  31,000 new edges on the 79 graphs with no measurement of what they do to it
  (that measurement was configuration A of the final measurement, which does
  not run).  A default that changes the working matcher's input unmeasured is
  not taken.  `GRAPH_STAGE6_RELATIONS`, `BINDER_COMPLETE_POSITION_MAP`,
  `FIELD_BORDER_DISCARD_PER_CHANNEL`, `EDGE_RULERS_FROM_MEASUREMENT` and
  `MATCH_RELATION_WALKER` are all OFF; with them off the graphs and the
  matcher are the stage-5 ones.  Whether to measure the old matcher on the new
  graphs is the user's call.
* 6.3 WHAT IS REPORTED INSTEAD: the walker's seeds against renderer truth per
  pair, its precision and recall apart, and the mechanism (which relation
  carries a right pair and which a wrong one; how far the two images cut
  their boundaries alike).  No constant is moved on the strength of it.

## Amendment 7 (the user's correction: forget Jaccard alone, marry the two halves; D0-D5)

The user read the RED M3-6 and corrected the contract.  Recorded verbatim in
substance, and MADE EXACT here, before any of it is measured.

WHAT HAD BEEN SEEN when this was written: M3-6 RED (F1(new) - F1(old) mean
-0.306, 0 of 9 pairs better); M1 green on 18 of 18; the seed table of the nine
pairs; one scratch trace of three pairs (which relation class carried a right
pair, which a wrong one, how far two images cut their boundaries alike).  NOT
seen: any T2 number (no objective was ever scored against truth), any number
of D0-D5, any score of any matcher on EVAL-75 or HELDOUT-75 under the stage-6
graphs, and the relation-repeatability table (its script was written; its run
was overtaken by this correction and is not reported).

### 7.0 THE OBJECTIVE (D0), the exact formula

One mapping M (one-to-one, FRAME -> FRAME) is scored by the MCS form over the
two halves it marries:

    Q(M) = [ W_N * S(M) + W_R * (P(M) + A(M)) ] / [ W_N * |M| + W_R * R(M) ]

  S(M)  NODE half, graded: the sum over mapped pairs of the graded signature
        similarity s(u, M(u)) in [0, 1] (matching.structural_similarity._sig_sim:
        the same measure the old matcher weights node credit with).
  P(M)  RELATION half, preserved: a measured growth edge (both ends FRAME,
        declared, not PROXIMITY, at least one measured axis) between two mapped
        nodes u, v counts 1 when the image edge exists in the same class and
        no shared axis contradicts it, and it counts the FRACTION of its shared
        measured axes that agree, agreement of one axis being
        max(0, 1 - |difference in bins| / CONTRADICTION_MARGIN) for a circular
        axis and the same on |difference| otherwise.  So exact agreement is 1,
        a look-alike one bin off is 2/3, and at the margin it is 0.
  A(M)  ATTRIBUTE half, graded: for every mapped pair whose frame nodes both
        carry CO_LOCATED attributes, for every attribute lens type either side
        has, the share of the finer side's attributes that pass the node gate
        against the other side, summed over (pair, type).  Carried by W_R.
  R(M)  the union's denominator, relations only: the number of measured growth
        edges with at least one end in M (query side) plus the number with at
        least one end in the reference image of M, minus the number with both
        ends mapped (those are the P candidates, counted on both sides before
        subtraction).
  |M|   the number of mapped pairs.

THE WEIGHT, stated now and never tuned: W_N = W_R = 1.  The two halves are
made commensurable in their own units before they are added (a node pair
contributes its similarity in [0, 1]; a relation contributes its agreement in
[0, 1]) and neither is worth more than the other.  Q(M) is in [0, 1] when
nothing is measured on one side (then that side's numerator and denominator
terms are both 0).

WHY THIS FIXES WHAT WAS WRONG: the old score (Jaccard on relations alone)
threw away every node's own evidence; the old matcher's S is a coverage count
over an assignment grown from node similarity, and its relation term is read
after the fact.  Here admission is still the walk (relations), but what a
mapping is WORTH is read on both halves, so a walk that drops most of the
frame (recall 0.287) is now visibly worse than one that maps it.

### 7.1 SEEDS (D1)

A SEED TOKEN is topological: (lens type of the node, edge type, direction
seen from the node, lens type of the far node, the far node's in-degree in
the growth graph, whether the far node is a closed host).  No bins, no
similarity.  A token is RARE when at most RARE_TOKEN_MAX_NODES (2) nodes carry
it in each graph; a candidate pair shares at least SEED_MIN_SHARED_TOKENS (2)
of them; the N_SEEDS (8) one-to-one pairs with the most shared tokens are
taken, ranked by (shared tokens, descending; node similarity, descending;
canonical keys).  The Amendment-4 tokens (class + exact bins) are still
computed, as the SCORE's relation identity, and the seed precision of BOTH
rules is reported (Amendment-7's own "before and after").

### 7.2 ADMISSION (D2)

A proposal reaching an unassigned node is admitted when the node gate (7.3)
passes and no shared axis CONTRADICTS:

    CONTRADICTION_MARGIN = 4.0 bins: a shared axis whose bin difference
    exceeds 4 bins (a quarter of a turn on an 8-bin angular axis, half of a
    distance span) contradicts; anything less only lowers the agreement
    fraction of 7.0 and is ranked by it.

So slack ranks and only a gross disagreement vetoes.  The FRONTIER is the
number of pairs examined against the budget, reported per pair; it must no
longer starve (it stood at 8 % of the dense matrix on the synthetic pairs).

### 7.3 THE NODE GATE (D3), relative

For a query node u and a candidate v:

    s(u, v) > NODE_FLOOR (0.30)   AND
    s(u, v) >= s(u, v*) - NODE_MARGIN (0.25), where v* is the most similar
        FRAME node of the reference image to u, over the whole frame.

Both numbers fixed now.  The old gate (absolute 0.5, and a similarity that
never entered the score) is replaced; the relation that proposed the pair is
what makes it an admission candidate at all.

### 7.4 ORDER PROPOSES (D4)

On two admitted boundary pairs (hosts h <-> h'), a feature f attached to h
(arc p) and a feature f' attached to h' (arc p') with s(f, f') passing the node
gate are PROPOSED as a pair when at least 3 features already admitted on that
pair of hosts keep their order with them (the order test of 4.7, on the
extended set).  Such proposals enter the frontier with the same rank as a
relation proposal; the order constraint then applies to them as to everything
else.

### 7.5 THE BEAM (D5)

The growth is a beam search of width BEAM_WIDTH = 4: from the seed set (a
prefix of the seed list: the top 1, then the top 2, ... up to 8 seeds), up to
four hypotheses are grown independently to exhaustion, each one-to-one inside
itself; the hypothesis with the best Q(M) (7.0) is the walk's answer; ties go
to the earlier seed set.  Budget and frontier are shared across the beam and
reported.

### 7.6 UNCHANGED

Determinism (canonical keys; one answer under any hash seed), the budget
(0.05 of the product, floor 8 looks per node, shared by the beam), the support
floor (8 pairs and 10 % of the smaller frame; below it the pair is
UNSUPPORTED and its score is None, which abstains), and the pins M1, the
distractors, same signatures with another arrangement, the order reversal, the
hash seeds.  M3-6 stays the verdict.  Accuracy is reported, never selected on.

### 7.7 THE ORDER OF WORK, fixed

1. T2: with the 7.0 objective, score (a) the walker's mapping as it is today
   (the RED run's walker), (b) renderer truth restricted to FRAME, (c) the old
   matcher's mapping, on all nine synthetic pairs.  If (b) does not beat (a),
   the objective is wrong; STOP and say so (nothing of D0-D5 is built).
2. Build D0-D5 behind MATCH_RELATION_WALKER (off by default); re-run M1 and
   M3-6; the pins are extended for each of D0-D5.
3. T1: the OLD matcher on the stage-6 graphs, EVAL-75, both windows, paired
   against the stage-5 graphs.  Reported, no default changed.
4. Only then, if M3-6 is green, the single final measurement, once.
5. If M3-6 stays red after D0-D5: report which defect's fix moved which
   number and stop.  No constant is tuned to turn the pin.

### 7.8 A CORRECTION TO 7.0, made the same day after the first T2 run

7.0 wrote the objective's denominator as `|M|` plus relations only, and said
the result lies in [0, 1].  It does not: the attribute mass `A` is in the
numerator and nothing in the denominator bounds it, so a mapping whose pairs
carry many shared attributes can score above 1 (a full identity on the
hand-built pin graph gives 36/33).  THE DENOMINATOR NOW INCLUDES THE
ATTRIBUTE UNITS: `W_R * (R + G)`, where `G` is the number of (mapped pair,
attribute lens type) units `A` is summed over.  Then `A <= G`, `P <= R` and
`S <= |M|`, and Q is in [0, 1].  T2 is re-run with this denominator; both
runs are reported, the first one labelled as written under 7.0.

### 7.9 SEED RANKING, made exact while building (before any synthetic run of D1)

7.1 said "the 8 one-to-one pairs with the most shared tokens, ranked by (shared
tokens, descending; node similarity, descending; canonical keys)".  A token
count that includes COMMON tokens ranks a shallow node (one relation, carried
by many nodes) above a deep one, which is the opposite of the intent.  The
rank is therefore (RARE tokens shared, descending; tokens shared, descending;
node similarity, descending; canonical keys); the RARE-threshold rule of 7.1
is unchanged.  7.1's own sentence ("the seed precision of both rules is
reported") is kept: the topological rule and the Amendment-4 rule are both
measured against truth in the M3-6 run.

### 7.10 THE BEAM'S SEED SETS, made exact while building

7.5 said "from the seed set (a prefix of the seed list: the top 1, then the
top 2, ... up to 8 seeds), up to four hypotheses".  Read literally that is one
hypothesis per prefix length, i.e. up to eight, not four.  The four are: the
first seed, the prefix at one third, the prefix at two thirds, and the whole
seed list (fewer when there are fewer seeds).  The whole seed list is always
among them, so a bigger seed set is never thrown away unexamined.

### 7.11 HOW THE BEAM CHOOSES, made exact while building

7.5 said "the hypothesis with the best Q(M) is the walk's answer".  Q is a
ratio (a mean per unit), so it prefers a SMALL mapping that is very similar to
a LARGE one that is slightly less similar (measured on the first hand-built
graph while building: a 10-pair hypothesis at Q 0.412 beat a 6-pair one at
Q 0.433, so Q alone would have picked the 6).  The choice is therefore: among
the hypotheses that pass the support floor, the larger mapping; Q breaks a tie
in size, and decides alone when no hypothesis passes the floor.  Ties go to
the earlier seed set.

### 7.12 THE BEAM'S ORDER, a second fix made while building (7.10 corrected)

The budget (7.6, unchanged) is SHARED by the beam.  Grown smallest seed set
first, the alternatives spent it before the hypothesis the seed list was built
for ran: measured on the first hand-built graph, the 13-frame graph matched
against itself under shuffled ids mapped 6 nodes in the first three hypotheses
and could not finish the fourth (104 looks of budget for 13 x 13).  The seed
sets are therefore grown LARGEST FIRST, and the whole seed list is always the
first hypothesis.  The alternatives are what the remainder of the budget can
still afford; their cost is reported.

### 7.13 THE BUDGET ON A SMALL FRAME (D2: the frontier must not starve)

7.6 keeps the budget unchanged: 5 % of |FRAME_q| x |FRAME_r|, floored at 8
looks per node of the smaller frame.  On the hand-built pins (13 x 13) that is
104 looks, and with D0-D5's growth the whole seed list needs 123: the walk
stopped at 9 of 13 nodes, i.e. the frontier starved exactly as the user warned
it would.  A SMALL frame therefore (product <= 1000 pairs, fewer than ~32 nodes
a side) may examine the whole matrix: the dense matrix there costs less than
the bookkeeping and the 5 % rule was written for graphs of hundreds of nodes,
where it still stands unchanged (frames of 111 to 402 nodes: 5 % binds and is
used at 0.1-8 % of the matrix).  Report the frontier (examined / budget) per
pair either way.

### 7.14 R(M) COUNTED, not a set union of keys (bug found while building)

7.0 writes R as a union of relation sets.  Keys of relation sets are (min id,
max id, class), and the query graph and the reference graph number their nodes
in DIFFERENT spaces (in the pins, the reference is the same graph relabelled
1000+).  Unioning the two key sets therefore counts every relation twice as
soon as the numbering differs -- measured: a graph matched against itself
under shuffled ids scored Q 0.643 instead of 1.0, with R = 40 where 7.0's
formula gives 20.  R is computed with the counts 7.0 describes: edges with at
least one end in M, plus edges with at least one end in the image of M, minus
the edges with both ends mapped.  No constant changes; the identity pin is
what found it.

### 7.15 THE VETO APPLIES TO SEEDS TOO (D2 read literally)

7.2's admission rule is written for "a proposal reaching an unassigned node";
a seed is such a proposal (its own class of it, 7.1).  7.1 did not exempt
seeds from the contradiction veto and 7.2 says "only a pre-registered
contradiction margin vetoes", so the veto is applied to the seed list as well:
a seed pair that shares a measured relation whose axis differs by more than
CONTRADICTION_MARGIN is skipped and the next ranked seed is taken.  Without
this, D1's topological tokens (which carry no bins) can seed two nodes that
the measurement itself says are far apart -- measured on the M3-3 pin: all
eight seeds were the identity (arc 0.06 against arc 0.45, 7 bins), and the
place-swapped feature was never found.

### 7.16 THE VETO IS ON THE PAIR, not on the proposing relation (7.15 completed)

7.15 applied the contradiction veto to seeds; applying it to seeds only left
ordinary admissions vetoed per PROPOSING relation, so a node whose attachment
axis is seven bins away (a contradiction) was admitted because the successor
gap that proposed it agreed.  Measured on the pin graph built for 7.2: the
feature at 1.57 rad against 0.0 rad was admitted through its successor edge
and the contradiction counted but did not refuse.  The veto is therefore one
rule for the pair: before admission, every relation class the two nodes share
is compared, and ONE shared axis beyond CONTRADICTION_MARGIN refuses the
pair.  Slack still ranks; the veto only vetoes.

### 7.17 WHICH RELATIONS THE VETO MAY COMPARE (7.16 corrected)

7.16 compared every relation of one class across the two nodes, which is too
wide: on the identity pin the two boundary hosts share six attachments at
arc bins 0, 7, 2, 6, 4 and 4, so "the host against itself" was refused because
one of its own features sits seven bins from another (measured: the identity
pin mapped 11 of 13 nodes, 100 and 200 refused).  A relation may only be
compared with a relation whose far end is its image, or -- when the far ends
are unknown, as at seed time -- the class must have NO agreeing reading at
all: the veto holds when every pair of relations in a shared class
contradicts, and also when a pair whose far ends ARE mapped to each other
contradicts.  A pair that agrees anywhere in a class is not vetoed by the
measurement; it is ranked by slack, as 7.2 says.  Both the M3-3 seed pin
(arc 0.06 against 0.45: refused) and the identity pin (accepted in full) are
what this rule is measured on.

### 7.18 ORDER PROPOSES AFTER MEASUREMENT, not ahead of it (D4, made exact)

7.4 gives the order a proposing role; the serving rank (Amendment 4.1) is
(supporting pairs; total slack; signature similarity; canonical key).  An
order proposal has no slack of its own, so recorded as slack 0 it outranked
every measured proposal: measured on the two-candidate pin, the feature whose
arc position was one bin off on the measured axis was admitted ahead of the
candidate in the same bin, because the order proposed it first.  Order is
weaker evidence than a measurement, so it is a TIER: proposals with a
measured relation behind them are served first, and an order proposal is
served when no measured proposal is pending.  Within a tier the 4.1 rank is
unchanged.  (D4's purpose -- reaching what no measured relation reaches -- is
what a tier keeps; what it loses is the power to overrule a measurement.)

### 7.19 AN ORDER PROPOSAL MAY NOT WRITE INTO A MEASURED SLACK (the 7.18 defect)

7.18 made order proposals a tier below measurements, and that hid a bug in the
bookkeeping rather than fixing it: proposals were stored per pair per support
as ONE slack, and `min(existing, incoming)` let an order proposal -- which has
no measured slack, so it carried 0.0 -- overwrite a measured slack of 2 bins
with 0.  The pair then looked perfectly supported.  Fixed: the support map
stores (slack, measured) and an order proposal may only OPEN a support that no
measurement opened; it never lowers a measured slack, and a measurement that
arrives after it REPLACES it.  The serving rank reads the sum of MEASURED
slacks only.

THE TIER (7.18) IS NOW BEHAVIOURALLY REDUNDANT on the hand-built pins: with
7.19 fixed, removing the tier changes no mapping in the pin set (measured:
32 of 33 pins green with the tier removed, the one red being the pin that
tests the key itself).  It is KEPT, on the record, because an order-only
support still sums to 0.0 measured slack -- indistinguishable in the rank
from an exact measurement -- and because tuning it out would be a change made
after seeing a number.  If a later measurement shows the tier costs
correspondence, it comes out by its own amendment.

### 7.20 THE MUTATION REPORT, written down (not just run)

diagnostics/results/phase6_walker_mutations.md holds the table: 18 single-rule
breaks, 16 red with the pin named, 2 survivors.  The survivors are named and
explained there, one behaviourally equivalent by construction (`_measured_slack`
counting a 0.0 order support as 0.0), one equivalent only in the graphs the
pins use (a rarity rule that changes ranking alone, and whose ranking never
separates a candidate on those graphs).  No mutation was swapped for a
catchable one; a mutation that did not catch anything was replaced by a pin on
the rule it actually tests (the node gate on a seed), and that is recorded.

### 7.22 THE BUDGET IS PER HYPOTHESIS (D5's design flaw, found by M1)

7.6 kept the budget and D5 gave the walk a beam; 7.10 made the beam share that
one budget.  Four hypotheses at ~1500 looks each against ~1470 in total starve
all four.  Measured on the M1 run: `examined == budget` on 16 of 18
self-matches, mapped/FRAME median 0.29 where the pre-D0 walker managed 0.83,
and on M3-6 the median pair spent 1427 of 1470 where before D0 it spent 300-700
of 900-2000 and never bound.  This is a flaw in the design, not a tuning miss:
each hypothesis IS a walk, and the pre-registered number is the budget of a
walk.  Each hypothesis now gets the whole budget; the beam's bill is their sum
and is reported (`WalkResult.examined` is the sum, each beam entry carries its
own).  Nothing else about the budget changes: 0.05 of the product, floored at
8 looks per node, the whole matrix on a small frame (7.13).  The seed scan is
charged to the walk (it is one look per pair it needs) and is reported apart.

### 7.23 WHAT A PARTIAL IDENTITY SCORES, stated (a reading of 7.0, not a change)

A graph matched against ITSELF, mapping 50 of 174 frame nodes, all 50 correctly,
scores 0.535 -- not 1.0.  This is deliberate and it is 7.0 read literally: R(M)
is "the number of measured edges with at least one end in M, plus the number
with at least one end in the image of M, minus the edges with both ends
mapped".  An edge from a mapped node to an UNMAPPED one is in R and cannot be
in P, so every node left out of the correspondence costs the objective.  7.0
states the intent in as many words ("a walk that drops most of the frame is now
visibly worse than one that maps it"); the MCS form is over the union, not over
the mapped part alone.  Consequences, written down before the next measurement:
  * M1's pin is restated.  The invariant is "the mapping is the identity AND
    every mapped pair is the same node AND Q = 1 WHEN THE MAPPING IS COMPLETE".
    A partial identity must show Q < 1, and its Q is its own formula's number:
    checked directly, the full identity of the same graph scores exactly 1.0
    (P = R = 453, S = |M| = 174, A = A_units).  An M1 row that maps less than
    the whole frame and still reads 1.000 is saying that the mapping is CLOSED
    (no measured edge leaves it), which on a 126-of-174 mapping of one graph is
    possible and is reported, not assumed.
  * The walk's own `score` is therefore a number that can be compared across
    pairs of different sizes, and 1.0 is achievable only by a complete,
    closed mapping.  Any claim that a partial mapping "scores 1" is withdrawn.

### 7.25 T2, M1 AND T1, each re-run ONCE under the current walker and objective

Order as the review set it: T2 (the gate), M1, M3-6, then T1.  Nothing was
selected on: no constant moved between these runs, and MATCH_RELATION_WALKER
stays OFF.

**T2 (results/phase6_t2_objective_d05.txt).**  Under 7.0/7.8/7.14 with the
walker as it now stands, on the nine registered pairs: truth Q > walker Q on
6 of 9 (walker > truth on 3); median Q walker 0.252, truth 0.239, old matcher
0.421; median mapped walker 43, truth 71, old 158.  The registered test is
"truth must beat the walker on at least half the pairs" and it passes (6 of 9),
but the MEDIAN says the walker's own mapping now scores higher than renderer
truth, because the walker maps far fewer nodes and Q is a mean over the union:
a small closed mapping can beat a large honest one.  Both readings are
reported; the test as pre-registered is met, the median is not flattering.

**M1 (results/phase6_walker_synthetic_d05b.txt).**  After 7.22 (budget per
hypothesis): identity on 18 of 18 self-matches, score 1.000 on 16 of 18,
mapped/FRAME median 0.82 (min 0.13); the two below 1 are partial identities
(7.23).  Before 7.22 the same run mapped 0.29 of FRAME and read 0.436-1.000.

**M3-6: RED.**  F1(new) - F1(old) mean -0.274, better on 0 of 9; precision
mean 0.556 against 0.632, recall 0.301 against 0.678.  This is worse than the
pre-D0 run's -0.306 only in the sense that recall rose (0.166 -> 0.301) while
the old matcher's own numbers did not move.  THE VERDICT DOES NOT CHANGE: the
walker loses to the matcher the repository ships, on the metric the review
fixed, on every pair.  The final measurement therefore does not run.

**T1 (results/phase6_t1_old_matcher.txt): flat.**  The old matcher on EVAL-75,
both windows, stage-5 graphs against stage-6 graphs: full 38/75 both, v1
39/75 both, no argmax moved by a single query.  The stage-6 structure -- the
relations R6 counted 848 of, the 248-node component, the measured share 0.83 --
changes NOTHING for the matcher the repository actually uses.  Read with R6,
this is the sharpest negative result of the stage: a graph change that
increases structure by an order of magnitude leaves the matcher's accuracy
untouched to the query, which says the matcher is not reading the structure
it is given.

### 7.26 WHAT MOVED WHICH NUMBER, listed once

Each fix, and the number it moved, all measured (not estimated):
  * 7.22 budget per hypothesis: M1 mapped/FRAME 0.29 -> 0.82, M1 score
    0.436-1.000 -> 1.000 on 16 of 18, M3-6 recall 0.166 -> 0.301.
  * 7.19 order slack bug: no mapping move on any pin (measured); it removed a
    latent mis-ranking and is pinned.
  * 7.14 R counted instead of set-unioned: identity Q 0.643 -> 1.000 on the
    pin graph.
  * 7.13 small-frame budget: 13-node pin growth 9 of 13 -> 13 of 13.
  * D1 topological seeds: F1 0.451 (the rule it replaced) -> 0.378.  WORSE.
  * D2 contradiction margin: F1 0.420 (no veto) -> 0.378.  WORSE; too tight.
  * D3 relative gate: F1 0.390 (floor only) -> 0.378.  WORSE by a little.
  * D4 order proposing: no change to three decimals.  DEAD on this workload.
  * D5 beam: F1 0.372 (one hypothesis) -> 0.378.  The only D that helps.

### 7.27 DO THE RELATIONS REPEAT ACROSS TWO IMAGES? (descriptive, after the verdict)

M3-6 is red, and the question underneath it is not "are there relations in a
graph" (R6 counted those inside one graph) but "is the SAME relation there in
the other graph".  `diagnostics/phase6_relation_repeatability.py` measures it
against renderer truth and decides nothing; the numbers are in
results/phase6_relation_repeatability.txt.  True correspondents: same lens
type, same labels within one scale, box-normalised position within 0.10 of the
box (finer than M3-6's role, and defined after the verdict saw the numbers --
recorded as such).

  other geometry, 8 pairs both directions, ALL CLASSES: 4,688 relations,
  present 0.85, REPEATS 0.73, chance 0.462.  Same geometry (1 pair): 0.90 of
  0.92 against chance 0.594.
  Best above chance: AMODAL_LINK 7.7x, RADIALITY 6.3x, CURVE_CONTAINMENT 9.6x
  and 4.9x (but present only 0.67/0.71), COCIRCULAR 3.4x,
  BOUNDARY_ATTACHMENT i PEAK_CHAIN 3.8x / 2.5x, JUNCTION_CONTOUR i Corner 2.3x
  with repeats 1.00.
  Below chance: BOUNDARY_ATTACHMENT o EndStopped (0.15 repeats against 0.141
  chance, present 0.44) -- the EndStopped class that the tangent work already
  found ABSENT as a measurement (truth v2, 0.09 used).

Read with T1: the relations DO repeat across images (0.73 against 0.462
chance), and the old matcher's accuracy does not move at all when they are
added (T1: 0 of 75 queries changed).  So the stage-6 relations are real and
repeated, and the shipped matcher does not read them.  That is the honest
summary of the stage: the roads were built and they are there; the walker that
would use them loses to the matcher on the metric that was fixed in advance.

### 7.28 THE T1 RESULT IN 7.25 WAS NOT A MEASUREMENT.  THE CACHE KEY DID NOT CARRY THE SWITCHES, AND T1 IS RE-RUN

Found by checking the review's own question -- "the front-end digest is in the
cache key, so why would the two arms be the same?" -- against the stored files
instead of against the intent.

**WHAT WAS WRONG.**  `eth80_frontend.graph_for` keys the graph cache on
`frontend_source_digest()`, and that was sha256 over the front-end SOURCE BYTES
only.  `eth80_benchmark.run_config` sets the front-end switches on
`parameters` INSIDE the process and forks the scoring pool -- the source tree
never changes.  So both arms of T1 keyed to the same cache entry: whatever the
first arm built, the second arm was served.  The stored files say it plainly:
`phase6_t1_stage5_full_full.json` and `phase6_t1_stage6_full_full.json` carry
the SAME `frontend_digest` (5b32547b) and the SAME first-pair numbers (208
matched edges, score 0.079634), although one arm claims all four stage-6
switches ON and the other all four OFF.  7.25's T1 row -- "38/75 both, 0 of 75
argmaxes moved" -- was one graph measured twice.  Recorded here rather than
deleted: it is the second time in this stage that a flat reading was accepted
without first asking whether the two arms could differ.

**THE FIX (in the code, not in a test).**

  1. `frontend_source_digest()` now folds in `switch_suffix()`: the value of
     every parameter in `_FRONTEND_SWITCHES`
     (FIELD_BORDER_DISCARD_PER_CHANNEL, BINDER_COMPLETE_POSITION_MAP,
     GRAPH_STAGE6_RELATIONS, ATTACHMENT_EVERY_NAMED_HOST) and the run's
     `PERCEPT_SET` override.  The memo is keyed by that suffix, not by one
     value: `run_config` changes the switches between two runs in ONE process,
     and a one-slot memo would have handed the second arm the first arm's key.
  2. `phase6_t1_old_matcher.py` now REFUSES TO REPORT when the two arms share a
     front-end digest, prints both digests and the first pair's matched-edge
     count, and states whether the v1 window is literally the same graph in the
     two arms.
  3. `MATCH_RELATION_WALKER` was in T1's `STAGE6_FLAGS`.  The script calls
     `evidence_score` directly (`score_pair`), so the flag changed no score --
     but it is the one flag an arm must not touch and the arm's own header read
     "the old matcher" while the switch said the walker was on.  Removed from
     the tuple; the arms now differ in the three front-end switches only.
  4. `diagnostics/_cache_carry.py` looked the old entries up under the
     bytes-only key, which after the fix carries nothing.  It now tries both
     key shapes and uses the one that exists; it refuses on the first sample
     that differs, as before.
  5. Pin: `test_the_graph_cache_key_carries_the_switch_values`
     (tests_matcher/test_stage6_relations.py) -- one switch, two values, one
     process, two keys.  Verified RED against a reverted digest (the pin fails
     with the switch suffix removed from the hash) and GREEN with it.

**THE RE-RUN (results/phase6_t1_old_matcher.txt, same script, same split,
nothing else changed).**  The arms are now different graphs: digests
b4a1bcc0e9e4c083 (stage-5) and 71a7be72c8d740ed (stage-6); the first query
against its own class has 208 matched edges in the stage-5 arm and 362 in the
stage-6 arm.

  window full:  stage-5 38/75 = 0.507, stage-6 40/75 = 0.533,
                gained 17, lost 15, predictions moved 34 of 75
  window v1:    stage-5 39/75 = 0.520, stage-6 39/75 = 0.520,
                gained 0, lost 0, moved 0 -- and here the zero is REAL: the
                two arms' v1 windows are identical graph for graph, because
                the stage-6 relations are written to V1 nodes and v1 keeps
                only V1 nodes.  The stage-6 structure shows in the full
                window, where it moves 34 of 75 answers and nets +2.

**WHAT THIS CHANGES IN THE STAGE'S SUMMARY.**  Not the verdict -- M3-6 stays
RED, no default changes, and the walker still loses to the shipped matcher --
but the sentence in 7.27 "the shipped matcher does not read them (T1: 0 of 75
queries changed)" is WITHDRAWN: it reads them enough to move 34 answers and
change net +2 correct.  The relations are real, they repeat (7.27), and
adding them to the shipped matcher's graph is a wash on accuracy with a large
reshuffle underneath.  That is the honest summary, and it is a weaker claim
than the one 7.25 made.

No constant, threshold or default was touched to produce any of the numbers
above; the only code changes are the cache key and the T1 script's own guards.

# STAGE 7 — "FIX THE REST".  PRE-REGISTRATION, WRITTEN BEFORE ANY OF IT RAN

The review fixed the diagnosis and the order.  This section fixes the
thresholds, the constants and the default rules BEFORE the numbers exist, so
that each phase's falsifier is a prediction and not a rationalisation.  Every
number below is reported whatever it says; a negative result is a deliverable.

## 7.29 PHASE 0 — THE GATE: is any relation slot class-informative?

**Population.** The 79 EVAL graphs as the front end builds them with
`GRAPH_STAGE6_RELATIONS` ON (the stage-6 structure under test).  No matcher is
run.

**Unit of measurement: the edge.**  For every edge class in
`core.schema.EDGE_UNIT_TABLE` and every one of its slots, the observation is the
slot's bin on ONE edge.  Each edge is labelled with the class of the image it
came from (`apple`, `car`, `cup`, `horse`).

**Statistics, per slot.**  Let the bins be integers `b`, `c` the class label:

  * BETWEEN/WITHIN RATIO.  `mu_c` = the mean bin over the edges of class `c`,
    `mu` = the mean over all edges, `n_c` the edge count of class `c`,
    `N = sum_c n_c`, `K` the number of classes with `n_c > 0`.
      BETWEEN = (1 / (K - 1)) * sum_c n_c * (mu_c - mu)^2      (eta-squared
                numerator: variance of the class means, weighted)
      WITHIN  = (1 / (N - K)) * sum_c sum_{e in c} (b_e - mu_c)^2
      RATIO   = BETWEEN / WITHIN, and 0 when WITHIN == 0 and BETWEEN == 0.
    This is the F ratio of a one-way ANOVA on the bin, with NO model of
    within-class structure: it asks how much of a slot's spread is explained by
    the class label.  It is reported with eta-squared (BETWEEN / TOTAL) beside
    it, because the ratio alone rewards a slot that barely varies at all.
  * MUTUAL INFORMATION.  `I(b; c)` in bits, with the empirical joint
    distribution over the same edges, and the NORMALISED form
    `I / H(b)` (the share of the slot's own entropy that the class removes).
  * A PERMUTATION NULL.  The class labels are permuted 200 times
    (`random.Random(20261110)`), the statistic recomputed, and the slot's
    PERCENTILE against that null is reported.  Shape statistics are
    scale-dominated and a large-N slot wins on N alone; the permutation says
    how much of the ratio is N.

**Controls.**  The same three statistics on the node signatures' own slots
(every lens type in `EDGE_UNIT_TABLE`'s sibling node tables, read through
`TypedSignature.quantized`), so a relation slot can be compared with the thing
the shipped matcher already uses.  The node control is the matcher's own
information, not a baseline to beat by tuning.

**THE THRESHOLD, FIXED NOW.**  A slot is INFORMATIVE iff ALL THREE hold:

  1. RATIO > 1.25.  The margin over 1.0 is 25 %: an F ratio of 1.25 on ~600
     edges per class is a real separation, and a slot that cannot clear a
     quarter of the within-class variance is not a lever.  (Stated as a
     margin, per the brief: "exceeds 1 by a margin you state".)
  2. I(b; c) >= 0.01 bits AND I / H(b) >= 0.02.  0.01 bits is 1 % of a
     two-outcome bit -- a small, checkable amount of class information; a slot
     at 0 has none however large its ratio.
  3. The ratio is at or above the 95th percentile of its own permutation null.

   A slot that meets 1 and 2 but not 3 is reported as INFORMATIVE-BUT-NOISE:
   the number is above threshold and indistinguishable from a relabelling.

**THE FALSIFIER, FIXED NOW.**  If NO slot of BOUNDARY_ATTACHMENT,
BOUNDARY_SUCCESSOR, ILLUSORY_INDUCER or any other structural relation class is
INFORMATIVE, then STOP: the representation carries no class information in its
relations, no matcher fix can help, and the report says the front end is the
work.  "Structural relation class" means every class in `EDGE_UNIT_TABLE`
except the attribute-only ones (BORDER_OWNERSHIP, SHARED_BOUNDARY,
TEXTURE_BOUNDARY, COLOR_OPPONENT_BORDER, SURFACE_FOLD, MOTION_OPPOSITION), and
the gate is decided on all of them, not only the three named.

## 7.30 PHASE 1 — THE FRAGMENT TANGENT

**The claim under test.**  `polar_boundary_fragments` takes the fragment's
tangent as the chord `atan2(p1 - p0)` of its pixel polyline.  The polyline is
traced along the association field's ridge, i.e. along the boundary, so its
chord is the boundary's TANGENT -- but the measured error is a median 90.0 deg,
which is the error of the NORMAL.  Something in the chain turns the tangent by
90 deg.  Phase 1 finds it and fixes it at the source; it does not add a
constant 90 deg at a consumer.  (If the turn is a `dx, dy` swap in one call,
the fix is that swap; if it is a normal being stored where a tangent belongs,
the fix is the stored quantity.  The commit says which.)

**Pre-registered expectations, before the fix is written.**

  * TANGENT ERROR against renderer truth for ContourFragment on smooth
    boundary (the phase6_tangent_truth measurement, same code path): the
    as-traced median is 90.0 deg.  Expected AFTER: median <= 15 deg, and the
    share within 15 deg >= 0.75.  If the corrected tangent lands at 15-30 deg
    the fix is directionally right and still not a measurement.
  * AMBIGUOUS ATTACHMENTS, share of all features, on the same 18 scenes and
    through the same census: expected to DROP.  The diagnosis says ~30 % of
    attachments are ambiguous; a tangent can only decide between near-equal
    hosts, so the expected drop is whatever share of the ambiguous sets is
    decided by tangent agreement.  No number is fixed for the drop -- the
    direction is.

**THE FALSIFIER, FIXED NOW.**  If the corrected fragment tangent does NOT
reduce the ambiguous share AND does not raise the tangent-decided share, the
ambiguity is not direction-limited, Phase 1 was the wrong lever, and no
consumer change is made.  EndStopped and Corner stay ABSENT: no direction is
fabricated for them, whatever this phase finds.

## 7.31 PHASE 2 — SEEDS

**Bookkeeping correction, stated before the run.**  The diagnosis says "D1's
topological tokens lost to the rule they replaced" with "F1 0.378 vs 0.451".
Those two numbers are not two seed rules: 0.378 is the all-on row and 0.451 is
the row with `SEED_TOPOLOGICAL` OFF (diagnostics/results/
phase6_ablation_d0_d5.txt), so the 0.451 row carries D1 off TOGETHER WITH
whatever else the flag reaches.  This phase measures the two seed rules
against renderer truth directly, on all nine pairs, and the ablation row is
only a pointer to it.

**Both rules, as they stand in the code** (`matching/relation_walker.py`):
  * TOPOLOGICAL (`SEED_TOPOLOGICAL = True`, D1): token rarity + shared tokens
    + node similarity.
  * MEASURED (`SEED_TOPOLOGICAL = False`): the number of relation classes in
    which some relation agrees within one bin, then similarity.

**Metrics, against renderer truth.**  Per pair and per rule: seeds chosen,
seeds whose two ends are TRUE CORRESPONDENTS (the M3-6 "same physical role"
criterion, so the two phases' numbers are comparable), SEED PRECISION =
right / chosen, SEED RECALL = right / the number of true correspondent pairs
in the smaller FRAME (a count, not a rate, so a rule cannot win by seeding
less), and then the mapping that grows from those seeds: mapped, pointwise
precision and recall, role F1.

**THE DEFAULT RULE, FIXED NOW.**  MEASURED is the default iff its seed
precision is higher than TOPOLOGICAL's on at least 6 of the 9 pairs (a strict
majority, paired); otherwise TOPOLOGICAL stays the default.  The flag that is
not defaulted stays behind its switch, and the reason is written with the
table.  I note here, before the numbers, that the ablation's F1 0.451 belongs
to the MEASURED rule, so the prior is that MEASURED wins; that prior does not
change the rule.

**THE NEXT ARM, registered in advance but NOT run.**  If the measured-
agreement rule is still under 50 % precise, the next seed arm is a
DISTINCTIVENESS RANK: rank the agreeing pairs by the rarity of the agreeing
bin pattern (the carrier count of the pattern in each graph, ascending), and
take seeds in that order.  No third rule is written before the second is
measured; this paragraph only fixes what the third would be.

## 7.32 PHASE 3 — ADMISSION BY STRENGTH

**What is already there.**  Agreement is the mean over the shared axes of
`max(0, 1 - d / MARGIN)`, `MARGIN = 4.0` bins, computed by `_agreement`.  The
admission code is `agree = _agreement(hu, hv) or 0.0`: a proposal always has an
agreement value, and an UNMEASURED relation (`None`) becomes 0.0 instead of
rejecting.  Requiring a measurement (`is not None`) rejects 0 pairs in the
current data -- measured, so it is not the lever.

**Arm A -- RELATIVE ONLY.**  At admission, compare the proposal's agreement `a`
with the best agreement `a*` offered by ANY rival on the same node: admit iff
`a >= a* - ADMIT_MARGIN`.

    ADMIT_MARGIN = 1.0   (one bin, of four)

  WHY 1.0: `MARGIN = 4.0` bins is where agreement reaches 0, so one bin is a
  quarter of the whole agreement scale -- the largest tolerance that still
  separates two proposals by more than their quantisation error (slots are
  8-12 bins wide, and `BIN_TOLERANCE = 1.0` is already the walker's declared
  "same bin" unit everywhere else).  It is the widest band that is still a
  comparison rather than a pass.

**Arm B -- FLOOR + RELATIVE.**  Arm A AND `a >= ADMIT_FLOOR`.

    ADMIT_FLOOR = 0.5   (half the agreement scale)

  WHY 0.5: a proposal must agree on average within 2 of the 4 bins on every
  shared axis to score 0.5.  `NODE_FLOOR = 0.30` already rejects signatures
  below 0.30, so 0.5 is the smallest floor that is STRICTER than the gate the
  pipeline already applies and therefore the most conservative floor that can
  be told apart from the node gate.

**How it is applied without a second mechanism.**  The proposal heap already
serves best-supported first; the arms gate the POP, not the push: a popped
candidate is admitted only if its agreement satisfies the arm.  Both arms are
run on the nine pairs with everything else at its pre-registered value.
FAN-OUT is counted as candidate POPs per admitted true pair; the budget spent
is `res.examined` against `res.budget`.

**THE MECHANISM TO CONFIRM.**  Fan-out falls AND recall rises, together.

**THE FALSIFIER, FIXED NOW.**  If fan-out falls and recall does NOT rise, then
growth-by-relation is the wrong mechanism for this representation: stop, say
so, do not tune the margin, and the walker stays off.

**DEFAULT RULE, FIXED NOW.**  An arm is defaulted only if it raises pointwise
recall on at least 6 of the 9 pairs against the shipped setting AND does not
lower pointwise precision on more than 2 of them.  If both arms qualify, the
one with the higher mean role F1 is defaulted; if neither qualifies, neither is
defaulted and the walker stays exactly as it is.

## 7.33 PHASE 4 — THE OBJECTIVE

**(a) A SCORE THAT CAN ARBITRATE ACROSS NODE SETS.**  Raw Q stays as it is (the
mean over the union; it is the number every stage-6 result is stated in).
Added, for argmax only:

    score_for_class = Q * |M| / min(frame_q, frame_r)

No tuned constant: the factor is the share of the SMALLER frame that the
mapping covers, so a closed sub-mapping is paid exactly for the frame it does
not map.  It is reported beside raw Q on the nine pairs, and the loophole's
status is decided on the measured number, not asserted.

**(b) THE COVERAGE LOOPHOLE -- which form closes it, and the experiment.**
If a closed sub-mapping of 82 % of the frame still reads 1.000 because every
relation it contains is preserved, then (a) alone moves it to 0.82 while a
mapping that maps the same nodes badly is unaffected.  The alternative is that
R must count the relations incident to the FRAME, not to M: the denominator
becomes the query's own relation mass and an unmapped node's relations are
missed relations, not absent ones.  WHICH FORM IS USED IS DECIDED BY THE
COMMON-SET RE-SCORE:

    the nine pairs re-scored with (a) and with each R form, and the form that
    makes a 50 %-complete sub-mapping of the SAME mapping score below its
    100 %-complete counterpart on every pair is the form that closes it.

The R form is NOT changed unless that test passes; the objective change (a) is
reported either way.

## 7.34 PHASE 5 — THE SHIPPED MATCHER'S ABSENCE BIAS

**The criterion, fixed BEFORE the measurement.**  Let `p_r` be the share of
EVAL-75 queries whose argmax is reference `r`, and `e_r` the share of the
four references' edges that `r` holds.  The bias statistic is
`B = max_r |p_r - e_r|`.  The criterion is a PAIR, both fixed now:

    (i)  `B` must fall below 0.10, and
    (ii) the ranking of the four references by `p_r` must agree with their
         ranking by `e_r` (Spearman rho = 1 on four points).

`B` is not a function of accuracy, so this cannot be passed by moving accuracy
up: it asks only whether a reference's size decides how often it wins.  The
two forms measured are the current `edge_absence` and the symmetric (union)
absence form.  Reported: per-reference prediction counts and per-class hits
under both.

**THE DEFAULT RULE, FIXED NOW.**  The symmetric form is defaulted iff it meets
the criterion above on BOTH windows (full, v1) AND does not lower EVAL-75
accuracy.  Accuracy alone never flips this: a noise-level accuracy change is
neither necessary nor sufficient.

## 7.35 PHASE 6 AND 7 — PINS AND CLEANUP

Phase 6's pins are stated before writing them: a pin on each of
BOUNDARY_ATTACHMENT, BOUNDARY_SUCCESSOR and ILLUSORY_INDUCER that FAILS if the
relation stops being written; a pin that their slots take more than one value
on a real scene; the FRAME partition pin (exists); an ambiguity pin (AMBIGUOUS
writes ABSENT, never a guess); and a pin that the cache key moves when a
graph-side switch moves (written in 7.28).  Phase 7 is mechanical and is done
last, after every number above exists, so no diagnostic can be re-run against
a half-cleaned tree.
