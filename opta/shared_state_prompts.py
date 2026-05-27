"""Prompts used by the OPTA chunk-report controller.

At each context-folding round, the controller runs two LLM phases:

Phase A: ``SHARED_STATE_BUILD_PROMPT``
    Reads every alive trajectory's latest accumulated report, then synthesises
    a single, schema-fixed *shared state* with four buckets:
        - ``supported_findings``    (findings supported by ≥1 trajectory with
                                     concrete sources, tool outputs,
                                     calculations, derivations, or other
                                     prompt-grounded reasoning)
        - ``dead_ends``             (entities/paths/hypotheses that ANY
                                     trajectory has definitively ruled out)
        - ``unresolved_conflicts``  (claims that genuinely disagree across
                                     trajectories and need verification)
        - ``progress_summary``      (one short overall snapshot + per-traj
                                     micro-summaries of what each one is doing)

Phase B: ``PER_TRAJ_INJECTION_PROMPT``
    Run ONCE per alive trajectory.  Given that trajectory's own accumulated
    report and the shared state from phase A, the LLM decides which (if any)
    items from the shared state are *useful* for THIS trajectory and emits a
    new ``<enriched_accumulated_report>`` that the agent will receive as its
    next-session opening context.  The LLM is allowed (and encouraged) to
    inject NOTHING when the shared state would not help.

Both phases are advisory only.  All trajectories keep running.
"""


SHARED_STATE_SYSTEM_PROMPT = (
    "You are a careful research aggregator. Follow the user's instructions "
    "exactly and produce the requested XML structure with no extra prose."
).strip()


SHARED_STATE_BUILD_PROMPT = """You are the cross-trajectory aggregator. {sample_count} independent trajectories have been working in parallel on the same question. Each trajectory has just produced its latest *accumulated progress report* (covering everything it has discovered up to chunk round {round_idx}).

**Original Question:**
{question}

**Your job:** Read every trajectory's accumulated report and synthesise a single SHARED STATE that captures, in a structured way:
  1. Findings or intermediate reasoning results that are **well-supported**.
  2. **Dead ends** (entities / paths / hypotheses that have been definitively ruled out).
  3. **Unresolved conflicts** (claims that genuinely disagree across trajectories and still need verification).
  4. A short **progress summary** (overall status + a 1–2 sentence note for each trajectory).

You must produce exactly the XML structure shown below. No prose outside the XML block.

**Hard rules:**
- Use ONLY information present in the provided accumulated reports. Do not invent findings, sources, URLs, tool outputs, calculations, or reasoning steps.
- An item belongs to ``supported_findings`` only if at least one trajectory provides concrete support for it. Support may be source evidence (a URL, search result, visited page content, PDF text, or tool output) OR reasoning evidence (a calculation, derivation, code/output trace, constraint check, or prompt-grounded inference). Mere unsupported assertion does NOT qualify.
- A target/path goes into ``dead_ends`` only if some trajectory has shown it definitively contradicts a hard constraint of the question, OR multiple trajectories independently established it as a wrong lead. Do NOT put a path here just because it is "weak". Cite which trajectories established it as dead.
- A disagreement goes into ``unresolved_conflicts`` only if (a) trajectories make competing claims about the same target/attribute AND (b) the conflict cannot be settled from the existing reports alone. If one side is clearly supported by strong evidence and the other is unsupported assumption, treat the unsupported side as a dead end instead.
- Each item must list the trajectory ids that contributed it (e.g. ``<source_trajectories>[1, 3]</source_trajectories>``). Use 1-indexed ids exactly as labelled in the input.
- Keep every item short and grounded. Do not summarise the whole reports — only extract reusable building blocks, including useful intermediate reasoning, that another trajectory could benefit from.
- If a bucket has no qualifying items, emit it as empty (e.g. ``<dead_ends></dead_ends>``). Do not invent items just to fill it.

Below are the {sample_count} accumulated reports:

{trajectory_reports}

Return your answer in EXACTLY this XML format (no extra commentary):
<shared_state>
  <supported_findings>
    <finding>
      <id>SF1</id>
      <statement>...</statement>
      <evidence>Source URL/tool output/calculation/derivation/constraint check supporting the statement.</evidence>
      <source_trajectories>[...]</source_trajectories>
    </finding>
    <!-- more <finding> blocks as needed; omit entirely if none qualify -->
  </supported_findings>
  <dead_ends>
    <dead_end>
      <id>D1</id>
      <entity_or_path>...</entity_or_path>
      <reason>...</reason>
      <discovered_by>[...]</discovered_by>
    </dead_end>
    <!-- more <dead_end> blocks as needed -->
  </dead_ends>
  <unresolved_conflicts>
    <conflict>
      <id>C1</id>
      <issue>...</issue>
      <competing_views>...</competing_views>
      <verification_needed>...</verification_needed>
      <involved_trajectories>[...]</involved_trajectories>
    </conflict>
    <!-- more <conflict> blocks as needed -->
  </unresolved_conflicts>
  <progress_summary>
    <overall>One concise paragraph on where the swarm collectively stands.</overall>
    <per_trajectory>
      <traj id="1">1–2 sentences on what this trajectory is currently pursuing and how far it has gotten.</traj>
      <!-- one <traj> block per alive trajectory -->
    </per_trajectory>
  </progress_summary>
</shared_state>""".strip()


PER_TRAJ_INJECTION_SYSTEM_PROMPT = (
    "You are a careful research aggregator. You decide what cross-trajectory "
    "knowledge to share with a SPECIFIC trajectory. Follow the user's "
    "instructions exactly and produce the requested XML structure with no "
    "extra prose."
).strip()


PER_TRAJ_INJECTION_PROMPT = """You are deciding which pieces of the SHARED STATE (built from all alive trajectories) should be injected into trajectory #{trajectory_id}'s next-session opening context, as additions to its own accumulated report.

**Original Question:**
{question}

**Trajectory #{trajectory_id}'s own accumulated report (this trajectory's private state):**
{own_report}

**Shared State (synthesised across all alive trajectories at chunk round {round_idx}):**
{shared_state_xml}

---

**Your task:** Choose ONLY the shared-state items that are likely USEFUL for trajectory #{trajectory_id} to know going into its next session. Then emit an *enriched accumulated report* that the trajectory will see as its new starting context.

**Selection rules:**
1. Inject a ``<finding>`` ONLY if it gives this trajectory supported information or a useful reasoning result it does not already have, or directly resolves an open question it was investigating. Do NOT inject findings that this trajectory already knows.
2. Inject a ``<dead_end>`` ONLY if this trajectory is currently pursuing (or might soon pursue) that path/entity. Use this to actively warn the trajectory away from a confirmed wrong lead. Do NOT inject dead ends about paths this trajectory has clearly abandoned or never considered.
3. Inject an ``<unresolved_conflict>`` ONLY if this trajectory is well placed to help resolve it (e.g., it is investigating one side of the conflict, or it has tools/leads relevant to verification).
4. Inject the ``<overall>`` progress summary ONLY if it materially changes this trajectory's situational awareness (e.g., it shows that the swarm has converged on a different region of the search space). Otherwise omit it. NEVER inject other trajectories' per-trajectory summaries — they are noise to this trajectory.
5. **It is perfectly valid to inject NOTHING.** If the shared state offers no actionable benefit to this trajectory, return its original accumulated report unchanged.
6. NEVER tell the trajectory which entity to commit to. The injected content is supplemental evidence or reasoning support, not orders.
7. NEVER paraphrase, weaken, or strengthen the original accumulated report. The trajectory's own findings must be preserved verbatim. You may only ADD a clearly demarcated "Cross-Trajectory Aggregator Notes" section.

**Output format:**
- First, list which shared-state ids you chose to inject (or ``[]`` if none) inside the ``<injection_decision>`` block.
- Then emit the full ``<enriched_accumulated_report>`` block that the trajectory will see. It must contain:
    a. The trajectory's own accumulated report, **unchanged**, under the heading ``## Your Own Accumulated Progress`` (verbatim copy of the input own report).
    b. A clearly separated section ``## Cross-Trajectory Aggregator Notes`` that contains ONLY the injected items (supported findings, dead-end warnings, unresolved conflicts that this trajectory could help resolve, and optionally the overall summary). If you injected nothing, emit this section with the literal text ``(No cross-trajectory notes to inject for this trajectory at this round.)``.

Return your answer in EXACTLY this XML format (no extra commentary outside it):
<injection_decision>
  <injected_findings>[...ids or empty list]</injected_findings>
  <injected_dead_ends>[...ids or empty list]</injected_dead_ends>
  <injected_conflicts>[...ids or empty list]</injected_conflicts>
  <injected_overall>true|false</injected_overall>
  <reasoning>One short paragraph: why these items (and only these) help this specific trajectory.</reasoning>
</injection_decision>
<enriched_accumulated_report>
## Your Own Accumulated Progress

<verbatim copy of the input own report>

## Cross-Trajectory Aggregator Notes

<only the injected items, neatly formatted as bullet points; or the literal "(No cross-trajectory notes to inject for this trajectory at this round.)" if none>
</enriched_accumulated_report>""".strip()


# Used for selecting the final answer after every trajectory finishes.
FINAL_INTEGRATE_PROMPT = """You are selecting the final answer from multiple problem-solving trajectories for the same question.

Each trajectory has produced (a) a final accumulated progress report, and (b) a candidate final answer.

Question:
{question}

Below are the trajectories you must consider for selecting the final answer:

{trajectory_reports}

Instructions:
1. Use only the provided accumulated reports and candidate answers.
2. Prefer the answer supported by the strongest, most coherent evidence or reasoning and the fewest unresolved contradictions.
3. Do not invent a new answer unless it is clearly implied by the provided material.
4. If at least one trajectory has a candidate answer other than [No Prediction], you MUST select exactly one of those candidate answers. In this case, [No Prediction] is forbidden.
5. Select the trajectory id whose candidate answer you choose. Use NONE and return [No Prediction] only if every trajectory's candidate answer is empty or [No Prediction].

Return:
<trajectory_id>SELECTED_TRAJECTORY_ID_OR_NONE</trajectory_id>
<answer>BEST_FINAL_ANSWER</answer>""".strip()
