# The walker's pins against single-rule breaks (stage 6, D0-D5)

Method: each row rewrites ONE rule in matching/relation_walker.py (a constant,
a branch, a term), runs `pytest tests_matcher/test_relation_walker.py`, and
restores the file.  A row is only counted as caught when the pin that fails
NAMES the rule it holds.  Run at `4c3b74e` + Amendment 7.19.

| mutation | result | pin that catches it |
|---|---|---|
| D0 node weight 1 -> 0 | RED | `test_the_objective_marries_the_node_half_and_the_relation_half` |
| D0 relation weight 1 -> 0 | RED | `test_only_the_frame_is_mapped_and_attributes_are_attached_afterwards` |
| D0 agreement graded -> binary | RED | `test_among_equally_supported_candidates_the_better_agreement_is_served_first` |
| D2 margin 4 -> 100 (nothing vetoes) | RED | `test_the_mapping_follows_the_relations_not_the_signatures` |
| D2 margin 4 -> 0 (everything vetoes) | RED | `test_a_graph_matched_against_itself_under_other_ids_is_the_identity` |
| D2 pair veto removed at admission | RED | `test_the_mapping_follows_the_relations_not_the_signatures` |
| D2 veto removed for SEEDS | RED | `test_the_mapping_follows_the_relations_not_the_signatures` |
| D3 relative margin dropped (floor only) | RED | `test_the_node_gate_is_relative_to_the_nodes_best_rival` |
| D3 floor dropped (margin only) | RED | `test_the_node_gate_is_relative_to_the_nodes_best_rival` |
| D1 seeds blind to the node gate | RED | `test_a_pair_is_a_seed_only_when_the_node_gate_passes_it` |
| D1 one shared token is enough | RED | `test_the_order_proposes_and_is_served_after_a_measurement` |
| D4 order never proposes | RED | `test_the_order_proposes_and_is_served_after_a_measurement` |
| D4 order tier removed | RED | `test_an_order_proposal_is_served_after_a_measured_one` |
| D5 beam width 4 -> 1 | RED | `test_the_beam_grows_four_seed_sets_and_keeps_the_best` |
| D5 Q chooses instead of size | RED | `test_the_beam_choice_prefers_the_larger_mapping_over_a_higher_q` |
| R union -> one graph's count | RED | `test_the_score_is_the_objective_of_the_mapping_the_walk_returned` |
| budget small-frame rule off | RED | `test_a_graph_matched_against_itself_under_other_ids_is_the_identity` |
| 7.19 order proposal overwrites a measured slack | RED | `test_an_order_proposal_never_lowers_a_measured_slack` |

## The survivors, named

**`is_rare = True` (rarity of a seed token ignored).**  SURVIVING.  Not
behaviourally equivalent in general; equivalent IN THESE GRAPHS, and here is
the reason, measured rather than argued: with rarity on and off, the seed list
is IDENTICAL on six hand-built graphs (two_boundaries 8 seeds, _arranged 8,
_ordered 5, _candidates 4, _two_plain 0, and a 6-long-against-2-short host
graph 8).  The reason is that rarity changes only the RANKING
(`-rare, -n, -similarity, keys`), and on these graphs every pair that passes
`SEED_MIN_SHARED_TOKENS` shares the same rare count, so the rank never
separates them.  What would catch it is a graph with more than N_SEEDS = 8
qualifying candidates where a common-token pair has the better similarity --
constructing one is exactly the "more adversarial graphs" the review asked me
not to build in place of pinning, so it is reported instead.  The M3-6 run
reports the seed precision of the topological rule and of the Amendment-4
rule, which is where a rarity mistake would show as a number.

**`_measured_slack` counting order-only supports as 0 slack.**  SURVIVING.
This one IS behaviourally equivalent, by construction: an order-only support
records slack 0.0 either way, so including it adds 0.  The value that matters
(never lowering a measured slack) is pinned by
`test_an_order_proposal_never_lowers_a_measured_slack`, and the mutation that
restores the 7.19 bug is caught there.

## What was changed, and why (recorded, not swapped silently)

The first mutation set ranked seeds by similarity FIRST
(`(-s, -rare, -n, keys)`) and the pins did not catch it: on every pin graph the
whole identity of the graph is the seed list whatever the order, and the first
eight candidates are the same eight.  I did NOT replace that mutation with a
catchable one to make the table look better; it is recorded above as the
rarity survivor, and the seed rule that IS catchable (the node gate on a seed)
got its own pin.  The pure functions `_serve_key`, `_choose`, `_record_support`
and `_measured_slack` were extracted and pinned directly, as asked, instead of
building more graphs: that is what caught the D4 tier and the D5 choice.
