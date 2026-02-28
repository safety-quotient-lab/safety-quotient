---
name: hunt
description: Find the highest-value next work — scans TODO.md, task list, git diff, stale plans, orphaned data/scripts, doc drift, and research gaps to surface actionable items ranked by value and orthogonality to in-flight work.
user-invocable: true
argument-hint: "[constraint or focus area, e.g. 'orthogonal', 'stale', 'quick wins', 'while waiting', 'data', 'extrapolate']"
allowed-tools: Read, Grep, Glob, Bash, Task, TaskList, AskUserQuestion
---

# Hunt — Systematic Work Discovery (PSQ)

Find the most valuable next work given current constraints. This aggregates all the ways you'd search for "what's next" in the PSQ research project into one structured sweep.

## Trigger Phrases

This skill matches any of these user intents:
- "what's next?" / "what else?" / "what can we do?"
- "look for orthogonal work" / "find stale work"
- "anything to do while we wait?" (e.g. while training runs)
- "what should I focus on?" / "what's highest value?"
- "find work at the edges of context"

## Arguments

Parse `$ARGUMENTS` to determine constraints:

| Argument | Constraint |
|---|---|
| *(empty)* or `all` | Full sweep — all sources, rank by value |
| `orthogonal` | Only surface work that doesn't touch files currently being modified |
| `stale` | Focus on drift and rot — stale docs, orphaned files, dead references |
| `quick` or `quick wins` | Only items that take <5 minutes (single-file fixes, doc updates) |
| `blocked` | Show what's blocked and what would unblock it |
| `while waiting` or `parallel` | Same as `orthogonal` — safe to do alongside in-flight training |
| `data` | Focus on labeling opportunities, batch scoring, DB coverage gaps |
| `extrapolate` or `deep` | Go beyond the backlog — mine for gaps between what's built and what's possible |

Multiple constraints can be combined: `orthogonal quick wins`

If `$ARGUMENTS` doesn't match any keyword above (e.g., "for improvements to hunt", "for scripts", "for DA"), treat it as a **domain/topic filter**: run the full sweep (all sources) but surface only items relevant to that topic. If truly ambiguous, default to full sweep and note the interpretation at the top of the results.

## Phase 1: Establish Context

Before hunting, understand what's currently in-flight:

1. **Read `TODO.md`** — the canonical backlog
2. **Run `TaskList`** — active task tracker (in-progress items = in-flight)
3. **Run `git status`** — what files are currently modified (avoid these for orthogonal work)
4. **Run `git log --oneline -20`** — what was recently changed; note any "pending metrics", "TODO", or "fix X" patterns (used in Source 8 — no need to re-scan there)
5. **Read `MEMORY.md`** — `~/.claude/projects/-home-kashif-projects-psychology-safety-quotient/memory/MEMORY.md` — current model version, DB state, known gotchas
6. **Glob `.claude/plans/*.md`** — existing plans (may have unfinished work)

Collect this into a mental model of: **what's active, what's done, what's blocked, what's untouched**.

## Phase 2: Scan Sources

Work through each source. For each, extract candidate work items with a rough value estimate.

### Source 1: TODO.md Backlog
- Read `TODO.md` (project root)
- Extract all unchecked `- [ ]` items
- Note dependencies (does completing item X unblock item Y?)
- Flag items whose prerequisites are now met

### Source 2: Task List
- Check for pending tasks not yet started
- Check for blocked tasks whose blockers may have been resolved
- Check for stale in-progress tasks (started but abandoned)

### Source 3: Stale Plans
- Glob `.claude/plans/*.md`
- For each plan, check if it has unfinished items
- Skip plans clearly superseded or completed (e.g., plan predates current model version)

### Source 4: Orphaned Data & Script Files
Research-specific rot patterns:

- **Scored-but-not-ingested batches:** Check `/tmp/psq_separated/` for score files that exist but were never assembled/ingested. Compare with `label_separated.py status`.
- **Stale /tmp files:** Any `/tmp/psq_*` scratch files (scored dimension JSON files, helper outputs) from past sessions that have already been ingested
- **Assembled-but-not-ingested JSONLs:** Glob `data/labeling-batch-*-labeled.jsonl` and cross-check each against DB (run a quick count query to see if those texts exist in `scores` table)
- **Orphaned scripts:** Grep `scripts/` for references to deleted files or deprecated function names (e.g., references to `map_new_datasets.py` which was deleted)
- **Old training logs or checkpoint directories:** `ls models/` — are there model directories with no corresponding EXPERIMENTS.md row?

```bash
# Useful scan commands:
ls /tmp/psq_* 2>/dev/null
ls models/
python scripts/label_separated.py status 2>/dev/null
```

### Source 5: Documentation Drift
- **MEMORY.md vs reality:** Check the current model version, DB counts, and labeling batches list against actual state (`python3 -c "import sqlite3; ..."`)
- **EXPERIMENTS.md vs models/ directory:** Does every `models/psq-v*` directory have a corresponding row?
- **DATA-PROVENANCE.md vs DB:** Do the "Total separated-llm" and "Total texts in DB" numbers match the DB?
- **lab-notebook.md open questions:** Check the Open Questions section — have any been answered since it was last updated?
- **distillation-research.md status line** (top of file): Does it reflect the current best model and next priority?
- **Cross-references:** Grep for any references to removed scripts (`map_new_datasets.py`, `batch_label_llm.js`, `relabel_separated.js`) still appearing in docs

### Source 6: Research Gaps in Scripts
Run targeted searches for common decay patterns:

```bash
# TODO/FIXME comments in scripts
grep -r "TODO\|FIXME\|HACK\|XXX" scripts/ --include="*.py"

# Likely debug prints (bare string, not formatted progress output)
grep -rn "^\s*print(['\"]" scripts/ --include="*.py" | grep -v "# " | head -20

# Hardcoded paths that may be stale
grep -r "/tmp/psq_" scripts/ --include="*.py"
```

Also check: are there `assert` statements or hardcoded record counts that need updating after new data ingestion?

### Source 7: Data Coverage Analysis
Quick DB scan for dimension-level coverage gaps:

```python
import sqlite3
db = sqlite3.connect('data/psq.db')
c = db.cursor()
# Separated-LLM coverage per dimension
# Scorer name: read from MEMORY.md ("scorer=claude-sonnet-4-6" in Labeling policy)
c.execute("""
    SELECT dimension, COUNT(*) as n, AVG(score) as mean,
           MIN(score) as lo, MAX(score) as hi,
           ROUND(100.0*SUM(CASE WHEN score=5 THEN 1 ELSE 0 END)/COUNT(*),1) as pct_5
    FROM scores WHERE scorer='claude-sonnet-4-6'
    GROUP BY dimension ORDER BY n ASC
""")
for row in c.fetchall():
    print(row)
```

Flag dimensions with:
- Fewer than 2,000 separated-LLM records (data-starved)
- >30% of scores at exact 5.0 (concentration problem)
- Range compressed (max < 8 or min > 3)

### Source 8: Git History Patterns
- Git log was already read in Phase 1 — use those results here rather than re-running
- Look for patterns of "fix X" commits suggesting recurring issues
- Check if any recent commits introduced known follow-up work (commit message says "pending metrics", "TODO", or "pending results")

## Phase 2b: Deep Extrapolation (for `extrapolate` / `deep` constraint)

When asked to "extrapolate", "find new work", or "go deeper than the backlog", go beyond TODO.md and task lists. Scan three layers:

### Layer 1: Dark Data — collected but not leveraged
What does the system produce that no analysis uses?

- **Unlabeled pool coverage:** How many texts in `data/unlabeled-pool.jsonl` have never been scored? How many are in the "informative band" (g ∈ [3,4.5)∪[5.5,7])? Which sources dominate the unused pool?
- **Proxy labels for dropped dimensions:** TE/TC/CC/AD/ED proxy rows still exist in DB. Could they be used for error analysis (identify where proxy was systematically wrong) even if not for training?
- **Confidence scores:** The model outputs confidence per dimension. Is confidence calibrated? Has anyone checked whether high-confidence errors cluster by source or dimension?
- **Labeling log timing data:** `data/labeling_log.jsonl` has per-session timing. Has anyone computed per-source or per-dimension scoring difficulty from this?
- **Split metadata:** Some texts appear in val/test but have only 1-2 dimensions scored. What are the most under-covered texts in the evaluation set?

### Layer 2: Structural Gaps — natural research extensions
What's the natural next analysis given what's built?

- **Held-out per-source breakdown:** `held-out-test.jsonl` has 5 sources (check MEMORY.md for current count). Has the current best model been evaluated broken down by source? Which sources is the model strongest/weakest on?
- **Error analysis:** For held-out texts where the model is worst (largest |predicted - actual|), what do they have in common? Source? Length? Dimension?
- **Criterion validity update:** Check MEMORY.md for the current best model. If it's newer than the last time CaSiNo/CGA-Wiki/CMV/DonD were run, re-running with the current model's predictions could strengthen publication claims.
- **Calibration check:** Is confidence calibrated for the current best model? (Does high-confidence predict lower error?) Run `scripts/evaluate_calibration.py` if it exists.
- **CO dimension deep-dive:** CO is weakest at 0.504. What's the distribution of CO scores in the held-out set? Is the model systematically biased (always predicts ~5)?
- **New validation dataset:** Are there publicly available datasets with explicit contractual/norm clarity labels that could serve as additional proxy? (CaSiNo negotiation outcomes correlate with CO — could reuse.)

### Layer 3: Methodology-Implementation Gaps
What does the research design call for that isn't implemented yet?

- **psq-definition.md vs distill.py:** Do the scoring rubrics in the definition match what the Claude prompts in `label_separated.py` actually ask? Any drift?
- **Expert validation protocol:** `expert-validation-protocol.md` exists. Are there any preparatory computational steps that could be done now (stratified sample selection, inter-rater reliability power analysis)?
- **Publication readiness:** `TODO.md` has publication phase items. What's the gap between current documentation and what a journal submission would require? Which sections of `journal.md` are incomplete?
- **ONNX inference validation:** `models/psq-student/model_quantized.onnx` is INT8. Has anyone verified that quantized predictions on the held-out set are within acceptable tolerance? (The max diff was 0.777 on random inputs — what is it on real text?)
- **student.js:** Is there a JS inference wrapper that uses the ONNX model? Does it work with the re-exported tokenizer?

### Presenting extrapolation results

Organize findings into two buckets:
1. **New TODOs** — concrete gaps that should be fixed (missing analysis, broken reference, calibration not run)
2. **New IDEAS** — enhancement opportunities (new validation dataset, new error analysis angle, new visualization)

For each finding: what it is, where in the codebase or data (file:line or DB query), severity/value, and effort estimate.

## Phase 3: Classify & Rank

For each candidate found, assign:

### Value Rating
- **HIGH**: Fixes a data quality issue, improves model reliability, enables publication, or unblocks future work
- **MED**: Improves documentation completeness, fills an analysis gap, cleans up stale state
- **LOW**: Style improvements, minor optimizations, nice-to-have analyses

### Effort Rating
- **XS**: <2 minutes (delete a file, fix a typo, update a count)
- **S**: 2-10 minutes (single-file doc update, quick DB query, ingest a batch)
- **M**: 10-30 minutes (new analysis, score a labeling batch, multi-file update)
- **L**: 30+ minutes (training run, new labeling batch from scratch, new validation study)

### Orthogonality
- **SAFE**: Doesn't touch any in-flight files or running processes (e.g., distill.py training)
- **ADJACENT**: Touches related but not identical files
- **OVERLAPPING**: Would conflict with in-flight work — defer

## Phase 4: Present Results

Format output as a ranked list, grouped by constraint match:

```
## Hunt Results

**Context:** [1-line summary of what's in-flight and what constraint was applied]
**Model:** vXX (held-out_r=X.XXX) | **DB:** XX,XXX texts / XX,XXX scores

### Top Picks (recommended next)
1. **[Subject]** — [1-line description]
   Value: HIGH | Effort: S | Where: `path/to/file` or `psq.db dimension=X`

2. **[Subject]** — [1-line description]
   Value: MED | Effort: XS | Where: `path/to/file`

### Backlog Candidates (from TODO.md)
- **[Item]** — [status/blocker note]

### Stale Items (needs attention)
- **[Item]** — [why stale, what to do]

### Data Opportunities (labeling / analysis)
- **[Dimension/batch]** — [gap description, estimated texts needed]

### Blocked (needs unblocking first)
- **[Item]** — blocked by: [what]
```

### Presentation Rules
- **Max 10 items** in Top Picks — don't overwhelm
- **Bold the subject**, keep descriptions to one line
- **Always include effort estimate** — knowing "this is 2 minutes" vs "this is 30 minutes" matters for task initiation
- **Group by theme** if multiple items relate (e.g., "3 stale doc entries" = 1 item, not 3)
- **If orthogonal/parallel constraint**: explicitly state "v23 training is running; these items don't touch distill.py or psq.db writes"
- **If data constraint**: lead with labeling opportunities, sorted by expected model impact
- **End with a recommendation**: "I'd suggest starting with #1 because [reason]"

## Phase 5: Decision Refinement with Knock-On Analysis

When a hunt surfaces items requiring a **choice between approaches** (not just "do this"), shift into decision-assist mode. This applies to:
- Backlog items with multiple implementation strategies
- Deferred items where the decision is "do it now / later / never"
- Data decisions (add more labels vs. fix proxy vs. wait for training results)
- Architectural choices (new dimension vs. absorb into existing, proxy retain vs. drop)
- Any item where the user says "what do you think?" or hasn't committed to an approach

### Step 1: Identify 2-3 distinct options

Frame the decision as concrete, mutually exclusive choices. Not vague ("maybe improve CO") — specific ("add 200 CO-targeted texts now" vs "wait for v23 results first" vs "run CO error analysis to diagnose before labeling"). Each option should be a real action the user could take right now.

### Step 2: Classify the decision domain

Identify which domain(s) apply — this determines what effect vectors to trace:

| Domain | Signal | Effect vectors |
|---|---|---|
| **Training data** | adding/removing labels or proxy rows | DB scores table → distill.py training view → model weights → held-out_r |
| **Proxy quality** | dropping/reweighting a proxy source | effective training set size → dimension coverage → held-out vs test paradox |
| **Scoring protocol** | changing rubric, scale, or prompt | label quality → halo artifact → factor structure → construct validity |
| **Model architecture** | new head, bifactor, dimension count | capacity allocation → per-dim r → publication claims |
| **Documentation** | updating or deferring doc changes | reproducibility → expert reviewer confidence → publication readiness |
| **Validation** | running or deferring criterion validity | external validity evidence → publication strength → expert recruitment timing |
| **Research scope** | adding/removing a dimension | measurement breadth vs. DA construct validity risk |

### Step 3: Ground the analysis

Before asking questions or tracing effects, answer these (read files or run DB queries if needed):
- What does the changed thing actually do?
- What depends on it? (what scripts read from it, what docs reference it, what training data views consume it)
- What would break silently vs. loudly if it changed?

This prevents speculation at low orders and makes questions more targeted.

### Step 4: Ask clarifying questions

Now that you've grounded the analysis, use `AskUserQuestion` with targeted questions that would change which option is best. Focus on:
- **Motivation**: What's driving the interest? Model quality concern, publication deadline, curiosity, or architectural preference?
- **Constraints**: What matters more — speed to next version, data efficiency, or thoroughness?
- **Context the user has that you don't**: Timeline pressure, upcoming expert review, known failure modes
- **Dealbreakers**: Any hard requirements that eliminate an option?

Keep questions concrete with option labels, not open-ended. Multiple-choice is easier to engage with than "what do you think about X?"

### Step 5: Six-order cascade — for each option

Use this exact format. One tight paragraph per order, bold descriptive label:

**Label guidance:** Labels are short noun phrases that name the *effect* at that order — not "Effect 1" or "Change 1". Good: "Training view update", "Dimension head gradient shift", "Held-out metric delta". Bad: "Order 1 effect". The label should tell the reader what happened before they read the paragraph.

---

**Knock-on analysis (6 orders) — [option label]:**

**1. [Label]** *(certain)*
The direct, immediate effect. Name specific files, DB tables, scripts, or dimensions. No vagueness.

**2. [Label]** *(certain–likely)*
What systems or data flows are activated by Order 1. What gets written, what gets invalidated, what training view changes.

**3. [Label]** *(likely)*
What consumes Order 2's outputs. Model training behavior, held-out evaluation results, documentation consistency, downstream scripts that read from changed data.

**4. [Label]** *(likely–possible)*
Aggregate or systemic effects — what emerges from accumulation. Metric skew, dimension trade-offs, silent degradation of non-targeted dimensions, documentation drift. State key assumptions.

**5. [Label]** *(possible)*
How Order 4 affects what humans observe or trust: held-out_r trajectory, expert reviewer impressions, publication reviewers' statistical scrutiny, DA construct validity risk. This is where silent data quality problems become visible — or permanently hidden.

**6. [Label]** *(speculative)*
How Orders 1–5 compound over time or constrain future work: lock-in on a labeling approach, reduced model headroom, publication framing consequences, technical debt in scoring infrastructure.

---

**Confidence discipline:**
- Orders 1–2: state as fact (direct causal effects — verify with DB queries or file reads if uncertain)
- Order 3: "likely" — based on known data flow; run query if uncertain
- Orders 4–5: "possible" — requires accumulation or compounding; state key assumptions explicitly
- Order 6: "speculative" — be honest; say "orders 5–6 are too speculative without knowing X" if true
- If a branch diverges into two significant paths at any order, note both

After the cascade for each option, list:

```
**Key mitigations for [option]:**
- [Concrete action] — addresses Order N: [specific risk]
- [Concrete action] — addresses Order N: [specific risk]
```

Prioritize non-obvious mitigations (Order 3+). Be specific: name the file, query, or script to change.

**Assumptions made:**
```
- [Assumption] — if wrong, Order N changes to [alternative]
```

### Step 6: Comparison table

End with a crisp comparison across axes that actually differentiate the options. Output this as a proper markdown table (not in a code block):

| | Model impact | Effort | Reversibility | Publication value | Risk |
|---|---|---|---|---|---|
| **Option 1** | ... | ... | ... | ... | ... |
| **Option 2** | ... | ... | ... | ... | ... |
| **Option 3** | ... | ... | ... | ... | ... |

Common useful axes for PSQ: expected Δheld-out_r, labeling cost, reversibility if wrong, strengthens which publication claim, risk of silent degradation. Pick only axes that differentiate the options — drop any column where all rows would say the same thing.

### PSQ-specific cross-domain effect patterns

Always check these when analyzing a PSQ decision:

**Training data changes (add/remove labels or proxy rows):**
- New labels added → `migrate.py --ingest` changes `best_scores` view priority → `distill.py training_data` view updates → model sees different ground truth
- Proxy rows removed → effective dataset shrinks for that dimension → model relies entirely on LLM labels → held-out improves if proxy was adversarial (TE lesson), degrades if proxy was useful (HI/RB lesson)
- Score concentration changes → `_cap_score_concentration()` applies differently → dimension head receives different gradient balance

**Scoring protocol changes (rubric, prompt, scale):**
- Prompt changes → all future labels are systematically different from past labels → mixed-signal training data → worse differentiation
- Scale changes (0-10 → 0-4 → pct) → `label_separated.py ingest` conversion handles it, but factor structure changes (pct collapses dimensions, shown by EV1 94.1% in pct vs 67.3% in integer)
- Never modify scoring prompt in production — test as controlled experiment first (scoring-research-plan.md)

**DB schema/query changes:**
- `best_scores` view priority order is `separated-llm > synthetic > joint-llm > composite-proxy` — any change here reorders which score wins for texts with multiple labels
- `training_data` view feeds directly into `distill.py` — schema changes here ripple immediately into next training run
- `splits` table is hash-based (deterministic) — adding texts doesn't re-shuffle existing splits

**Model architecture changes:**
- New output head → capacity competition in shared 384-dim projection → other dims may regress (bifactor v19b lesson: CC +0.150 but RC/AD/TE all lost)
- Changing N_DIMS breaks checkpoint compatibility — old `.pt` files can't be loaded into new architecture

**Documentation changes:**
- Stale version numbers in MEMORY.md → next session starts with wrong mental model → may re-do completed work
- distillation-research.md status line not updated → reader has wrong understanding of current best model
- DATA-PROVENANCE.md counts not updated → licensing audit finds discrepancy → credibility concern for publication

### When to skip decision refinement

- All hunt results are straightforward tasks (no choice needed) — go straight to Phase 6
- User says "just do it" or picks an item without hesitation — execute, don't deliberate
- Effort is XS/S — the cost of deciding exceeds the cost of doing. Just do it.

## Phase 6: Offer Next Steps

After presenting results (and completing Phase 5 decision refinement if needed), offer concrete next actions.

**If Phase 5 was run:** Lead with the winning option or ask the user to confirm their choice, then offer to execute it. Don't re-list items that the Phase 5 analysis already resolved.

**If Phase 5 was skipped** (items were straightforward tasks), offer:
- "Want me to tackle #1-3 (quick wins)?"
- "Want me to score [dimension] for [batch]?"
- "Want me to run the error analysis on CO?"
- "Want me to run `/cycle` to clean up after these changes?"

If no meaningful work is found:
- "Project is in good shape. Next meaningful work is [item] which requires [prerequisite — likely v23 training results]."

## Efficiency Notes

**Skip matrix — which sources to run per constraint:**

| Constraint | Sources to run | Sources to skip |
|---|---|---|
| *(empty)* / `all` | 1–8 + Phase 2b | — |
| `quick` / `quick wins` | 1, 2, 5 (spot-check only) | 3, 4, 6, 7, 8, Phase 2b |
| `stale` | 3, 4, 5, 8 | 1, 2, 6, 7, Phase 2b |
| `blocked` | 1, 2 | 3–8, Phase 2b |
| `orthogonal` / `while waiting` | 1–8 (filter by SAFE orthogonality) | Phase 2b |
| `data` | 1, 2, 4, 7, Phase 2b Layer 1 | 3, 5, 6, 8 |
| `extrapolate` / `deep` | 1, 2, Phase 2b | 3–8 (maintenance checks not needed) |

**Other notes:**
- **Use parallel Bash calls** for Sources 4, 6, 7 — independent queries, no ordering dependency
- **Source 8 is free** — git log was already read in Phase 1; just re-use the result
- **Don't re-scan** what was already found in the same session — note what's been covered and focus on what's changed
- The goal is **actionable items in <60 seconds** for quick/blocked mode, **comprehensive in <3 minutes** for full sweep
