# PSQ Project Memory

**Note:** Stable project conventions are now in `CLAUDE.md` (repo root, auto-read).
This file holds volatile state only — current model, DB counts, batch lists, in-progress work.

## Snapshots
- `memory/snapshot-20260301-1225-paradigm-shift.md` — Summary format + MY REASONING voice protocol. Git tag: `paradigm-shift-summary-format` (919f496). Full 9-order knock-on analysis of voice choice. **Restore point for format paradigm.**

## Output format (ADHD/autism accommodation)
**ASCII dashboard format for all session summaries, hunt results, and cycle reports.**
- **Verbosity: 1.618× base (golden ratio).** Every section gets a line of context, not bare facts. Full format structure always present. Not terse, not verbose — calibrated.
- **Whitespace: 1.618× base (golden ratio).** Content density stays at baseline — no extra words. Whitespace does the structural work:
  - Double blank lines between major sections
  - Sub-separator lines (`──────`) under each section heading
  - Blank line after each sub-item label before its body
  - Dense paragraphs broken into evidence / cost / conclusion chunks
  - Lists use aligned columns, not inline prose
  - Colons removed from section headers (cleaner visual hierarchy)
- Max one page/screen. Use ASCII box-drawing tables, not markdown tables.
- One line per item. Abbreviate dimension names (TE, HI, AD, etc.).
- Visual symbols: ✓ pass, ✗ fail, ★ important, ↑↓≈ for deltas, ⚑ flag, ⚠ warning.
- Structure: HEADER (model+DB) → WHAT HAPPENED (✓/✗ table with 1-line context each) → THEORY (brief) → ⚑ EPISTEMIC FLAGS → SKIPPED (with reasons) → MY REASONING → WHAT'S NEXT (tiered: immediate → tech debt → pub blockers → horizon)
- PUB SCORECARD: include when new psychometric/criterion evidence generated; omit for doc-only cycles.
- **MY REASONING section voice protocol** (calibrated via 9-order knock-on analysis):
  - **Chain visibility:** Show the chain (A→B→C therefore D), not just conclusions.
  - **Calibration:** Show 2-3 competing hypotheses ranked by plausibility, not single answers.
  - **Disagreement:** Evidence first — present the data, let the user draw conclusions.
  - **Error tracking:** Running correction log — "Last session I said X. Data shows Y. Updating."
  - **Sycophancy check:** Flag contrarian claims with ⚡ when I expect pushback. "You may disagree, but: [claim + evidence]." Makes the anti-sycophancy check visible and auditable.
  - Content types: pattern spotting, honest concerns, hypotheses, strategic advice, chain analysis.
  - Always challenge when I think direction is wrong. Never optimize for approval over truth.
- **⚑ EPISTEMIC FLAGS section is mandatory.** Surface anything that threatens:
  - Metric reproducibility, labeling protocol fidelity, calibration, construct validity,
    demographic fairness, data provenance, internal doc consistency, reviewer risk.
  - Severity: ██░░ HIGH, █░░░ MOD, ░░░░ LOW. 1-2 line explanation each.
- Generous whitespace between sections. Prose for context (1-2 sentences per item).
- SKIPPED section always present with 1-phrase reason per doc.

## Formatting policy
**APA 7th edition in 4 publication-facing documents only:**
- `journal.md`, `psychometric-evaluation.md`, `criterion-validity-summary.md`, `distillation-research.md`
- In-text citations: (Author, Year). Statistics: *r*(df) = .684, *p* < .001 (suppress leading zeros, include df, exact *p*, CIs on AUC). No asterisk significance markers.
- **Target venue: psychology journal** (Behavior Research Methods or Journal of Personality Assessment).
- Operational logs (EXPERIMENTS.md, lab-notebook.md, TODO.md, CLAUDE.md, etc.) keep current shorthand — APA formatting is incompatible with machine-parseable tables and living operational docs.
- Decided via 6-order knock-on analysis (2026-03-01): "everywhere" rejected (3 certain orders against); targeted policy adopted.

## Decision-making: knock-on auto-resolution
**TRIGGER:** When I encounter ANY decision point with 2+ options, run knock-on analysis (3-6 orders) and attempt resolution by consensus or parsimony BEFORE presenting to the user. Do not present bare forks ("your call"). Present recommendations with reasoning already attached.
- **If consensus/parsimony resolves it:** State the recommendation, show the reasoning, proceed unless user overrides.
- **If it doesn't resolve:** Terminate with full reasoning + unresolved tensions. Present the genuine ambiguity to the user with the specific questions that would resolve it (use AskUserQuestion).
- **Apply to agents:** Instruct them to resolve by consensus/parsimony or terminate with full context. Never force conclusions.
- **Effort scaling:** XS/S decisions get 3-order analysis inline. M/L decisions get 6-order, optionally via agent.

## Edit discipline
**When adding a new preference or setting, APPEND — never overwrite an adjacent existing one.** Read the surrounding context before editing to confirm what's already there. The cost of a redundant line is near-zero; the cost of silently dropping a calibrated preference is a full recovery cycle. (Learned 2026-03-01: verbosity preference was overwritten when whitespace preference was added.)

## Date/time policy
**Always run `date -Idate` before writing any date into documentation.**
Do NOT trust the `currentDate` system context or assume the date from conversation history.
The system clock is the single source of truth. Timezone: CST (UTC-6).
All script timestamps use local time (`datetime.now().astimezone()`), not UTC.

## Rubric policy
**The PSQ construct definitions (psq-definition.md) are externally authored.**
Dimension names and core definitions are stable. However, scoring rubric anchors may be experimentally modified as part of the halo mitigation research (see `scoring-research-plan.md` Avenue 2: structurally dissimilar rubrics). Any rubric changes should be tracked as experiments, not silent edits.

## AD rename: WILL NOT RENAME
`authority_dynamics` stays as-is. Rationale: fidelity with the official psychological safety taxonomy (Edmondson 1999, French & Raven 1959). The criterion validity finding (AD measures peer-context status negotiation) is an empirical refinement *within* the construct, not a reason to depart from the established nomenclature. **Do not revisit this.** What can change: the rubric anchors in psq-definition.md §9 (description update to reflect peer-context status negotiation, not formal authority).

## Labeling policy
**All LLM labeling is done by Claude Code in conversation. No API scripts.**
- No ANTHROPIC_API_KEY needed; it is intentionally empty in `.env`
- `batch_label_llm.js` was deleted — it used joint scoring (halo problem) and API calls
- `relabel_separated.js` was deleted — same reason
- Canonical labeling tool: `scripts/label_separated.py`

## Labeling workflow (separated scoring)
One dimension per Claude Code session to eliminate halo effect.

```
# 1. Extract batch files (one per dimension)
python scripts/label_separated.py extract --input <file.jsonl>
# For 0-100 percentage scale (finer granularity):
python scripts/label_separated.py extract --input <file.jsonl> --pct

# 2. In each session: I read the batch, score texts, output compact JSON
#    Format: {"dim": "te", "scores": {"0": [score, conf], "1": [score, conf], ...}}
#    With --pct: scores are 0-100, auto-converted to 0-10 on ingest

# 3. Ingest scored dimension (auto-detects --pct from session metadata)
python scripts/label_separated.py ingest --dim <dim> --scores /tmp/scored.json

# 4. Check progress
python scripts/label_separated.py status

# 5. When all 10 done, assemble
python scripts/label_separated.py assemble --input <original.jsonl> --output <out.jsonl>
```

Batch files land in `/tmp/psq_separated/`. Scores persist there across sessions.

## Scale awareness
- Held-out set (100 texts): scorable in 1-2 sessions per dimension
- train-llm.jsonl (4,199 texts): requires sub-batching across many sessions
- For training data: prefer **adding new separated-scored texts** over relabeling all 4,199
  - Unlabeled pool: `data/unlabeled-pool.jsonl` (~7K texts available)
  - Each session: extract a batch of 50-100 texts, score 1-2 dimensions, ingest

## Current model: v23 (production best, held-out_r=0.684 corrected)
- **v23**: held-out_r=**0.684** (corrected from 0.696 after max_length eval bug fix). test_r=0.387. 2026-02-28.
  - `--drop-proxy-dims` (TE/TC/CC/AD/ED). Data: +550 texts (ccda+proxy-audit+held-out-expand). 8 epochs.
  - ONNX re-exported 2026-03-01: model.onnx=254.4 MB, model_quantized.onnx=64.0 MB (INT8).
- **v27** (2026-03-01): held-out_r=**0.655** (−0.029). +368 texts (ucc/civil/extreme-adco). **Regressed — not promoted.** Possible same-session halo contamination.
- **max_length eval bug** (fixed 2026-03-01): eval_held_out.py/calibrate.py/distill.py PSQDataset all had max_length=256 instead of 128. All historical held-out_r inflated ~0.012. Fixed.
- **max_length criterion bugs** (discovered 2026-03-01, NOT YET FIXED): `criterion_validity_cmv.py` uses 512 (should be 128), `criterion_cgawiki_temporal.py` uses 256 (should be 128). DonD/CaSiNo clean. CMV AUC=0.5735 and CGA-Wiki T2 results may be inflated.
- **Historical re-eval DONE** (2026-03-01): 11 models (v14–v22c) re-evaluated at max_length=128. Correction NOT uniform (+0.033 to −0.049). **v22a=0.706 > v23=0.698** — relative ordering shifted. Production calibration re-fit saved.
- **DISCREPANCY RESOLVED**: 0.698 was my bug (wrong dim names in manual computation). v23 _avg_r=0.684 confirmed. BUT v22a _avg_r=**0.695** — v22a is actually the best model after correction. v23 is 2nd.
- **H1 halo CONFIRMED**: Rapid-batch mean |r|=0.658 vs existing sep-llm mean |r|=0.582 (Δ=+0.077). Rapid batches exceed 0.65 threshold. Same-session halo contamination is real.
- **CMV corrected** (2026-03-01): AUC=0.5549 (was 0.5735, Δ=−0.019). DA still top predictor. Core findings preserved.
- **CGA-Wiki T2 corrected** (2026-03-01): T2 still NOT SUPPORTED. Temporal trajectory unchanged. Q4 collapse preserved.
- **v22a vs v23 RESOLVED (hold v23)**: v22a _avg_r=0.695 > v23=0.684, but Δ=0.011 < SE≈0.05. Knock-on: promoting v22a would invalidate all 4 criterion studies (run with v23), require full doc update, and be superseded by v28 retrain anyway. Parsimony: hold v23, note correction, promote v28 after revert.
- **3,680 rapid-scored records REVERTED** (2026-03-01): H1 halo confirmed (mean |r|=0.658 vs 0.582). Deleted sep-llm scores for 368 texts (ucc/civil/extreme-adco). DB backup: psq.db.bak-pre-revert-20260301. Texts remain in DB for re-scoring properly (one dim per session).
- **Pending**: Re-score 368 texts properly (~7 sessions), then retrain v28.
- **Confidence calibration**: POOR — 8/10 dims inverted (higher conf → higher error). validate_confidence_calibration.py rewritten to use DB.
- **Context length sweep** (COMPLETE): 128 > 512 (0.692) > 256 (0.670). 128-token context optimal.
- Score-concentration cap: `_cap_score_concentration()` in distill.py (>30% → weight 1.5)
- distill.py: `--out DIR`, `--no-save`, `--no-cap`, `--bifactor`, `--drop-proxy-dims`
- Smoke test: `python scripts/distill.py --no-save --epochs 1`; Production: `--out models/psq-vN`

## Factor analysis (2026-02-28, v2)
- **v2**: EFA on N=1,970 separated-llm-only texts: g-factor eigenvalue **6.727 (67.3% variance)** — up from 4.844 (48.4%)
- KMO = **0.902** ("Superb") — up from 0.819
- Parallel analysis: **1 factor only** (was 2 in v1). 5-factor structure collapsed — Factor 1 absorbs 8/10 dims.
- g-factor loadings all >0.66: TC (0.930), DA (0.914), CC (0.864), RC (0.854)
- Mean inter-dim |r| = 0.632 (up from 0.417 mixed, 0.564 sep-llm v1)
- **Integer vs pct scoring**: Integer uses 11 bins, pct uses 35 unique values — but pct COLLAPSES dimensions.
- Pct within-text SD=0.448 vs int=0.717 (1.6× less differentiated). 8/10 dims <5% unique variance in pct.
- g-factor eigenvalue: INT=6.727 (67.3%), PCT=9.410 (94.1%) — g-factor is REAL, NOT inflated by integer bias.
- **Revert to integer scoring.** Pct anchoring-and-adjustment destroys dimension differentiation.
- Residual structure survives: parallel analysis retains 3 factors after removing text mean in pct data.
- v1 5-factor structure retained for reference: Hostility/Threat (HI,TE,CC), Relational Contract (CO,TC), Internal Resources (RB,RC,DA), Power Dynamics (AD), Stress/Energy (ED)
- Key docs: psychometric-evaluation.md §3c, distillation-research.md §26/§42/§43, journal.md §18/§28

## Database (psq.db)
- `data/psq.db` — SQLite (22,304 texts, 90,361 scores, 34,850 separated-llm — post-revert 2026-03-01)
- Splits: train=17,708 / val=2,160 / test=2,235 / held-out=100
- Schema: `data/schema.sql` — texts, scores, splits, labeling_sessions, models, calibrations, dataset_mappings
- Migration: `scripts/migrate.py` — bootstraps from existing JSONLs; `--ingest JSONL` for incremental ingest
  - Use `--ingest` for new assembled JSONLs (separated-llm, filtered: skips confidence<=0.15 placeholders)
- Key view: `best_scores` (priority: separated-llm > synthetic > joint-llm > composite-proxy)
- `data/dataset_mappings.json` — canonical config for all 11 source datasets (replaces map_new_datasets.py)
- `scripts/build_composite_ground_truth.py` — now config-driven; reads dataset_mappings.json
- `scripts/map_new_datasets.py` — DELETED (superseded)
- Provenance triple on all new labels: `scorer=claude-sonnet-4-6`, `provider=anthropic`, `interface=claude-code`

## Labeling batches (all scored+ingested unless noted)
- weak-dims(200), rc(150), ad(300), co(200), rb(200), cc(200), te(200), broad(300), pct-200(200), midg(250), test-clean(200), ccda(200), proxy-audit(200), held-out-expand(150) — all complete
- ucc(150), civil(100), extreme-adco(118) — **REVERTED** (H1 halo confirmed). Texts remain for re-scoring properly.
- Scoring batches of 50 texts per response (avoid 32K output token limit)
  - Partial files: `/tmp/psq_separated/{dim}_partial.json` (accumulate across 4 batches, then ingest)

## Labeling timing
- `label_separated.py ingest --started-at <ISO>` logs timing to `data/labeling_log.jsonl`
- `label_separated.py timing` shows per-dimension and aggregate stats
- Record start time (UTC) before scoring, pass to ingest after
- Full CO batch: 200 texts × 10 dims in 25.3 min = ~4,743 texts/hr average
- Careful scoring (first encounter): 3,200–5,100 texts/hr; fast (already in context): 23,000–24,400 texts/hr

## Context limit mitigation
- Large labeling sessions (>100 texts × 10 dims) can exhaust context before post-processing
- Score files in `/tmp/psq_separated/` persist across sessions — no work lost
- **Assemble after every 2-3 dims** instead of waiting for all 10
- Budget context for assemble + ingest + docs at end of session
- Use `--offset`/`--limit` to sub-batch very large files across sessions

## Working state docs
- `working-state.md` — current project state (updated each session)
- `working-state-snapshot-YYYYMMDD-HHMM.md` — point-in-time snapshots

## Expert validation protocol
- `expert-validation-protocol.md` — full study design for human expert validation
- 5 expert psychologists, 200 stratified texts, all 10 dims, 10,000 ratings
- Primary: ICC(2,1) per dimension (target ≥ 0.70)
- DA-specific decision tree: ICC<0.50 → deprecate; partial r<0.30 → retain; R²>0.80 → absorb
- Expert vs LLM convergent validity on 20 held-out overlap texts
- Expert factor structure comparison (Tucker's φ)
- Estimated: 7-9 weeks, $5,625-$15,000 for 5 raters
- Status: protocol designed, recruitment not started

## Criterion validity evidence
- **CaSiNo** (commit b460e52): 1,030 negotiation dialogues, 3 outcomes (satisfaction, likeness, points)
  - 9/10 dims predict satisfaction (r≈0.08-0.13***), 9/10 predict likeness
  - Incremental R² = +0.016 (sat), +0.023 (like) beyond sentiment + text length
  - DA is top individual predictor after controls (paradox: weakest factor loading, strongest criterion predictor)
  - Docs: distillation-research.md §30, psychometric-evaluation.md §3g, journal.md §20
- **CGA-Wiki**: 4,188 Wikipedia talk-page conversations, derailment prediction
  - AUC=0.599 (10-dim), g-PSQ near-chance (0.515) — profile shape predicts, average does not
  - AD strongest predictor (r_pb=-0.105***), replicates CaSiNo finding
  - Temporal gradient: AUC 0.519→0.570→0.599 (1st turn→early→all). PSQ measures process, not static content.
  - **T2 cross-lagged analysis in progress** (criterion_cgawiki_temporal.py, task be6a8et6r on CPU, ~53% done). Tests whether AD_t predicts HI_{t+1} more than HI_t predicts AD_{t+1} in derailing conversations.
  - Docs: distillation-research.md §31, psychometric-evaluation.md §3g, journal.md §21
- **CMV** (Change My View): 4,263 matched pairs, persuasion prediction. **v23 rerun 2026-02-28.**
  - 10-dim AUC=0.5735 (was 0.590 v16), g-PSQ=0.5227 — profile >> average (gap 0.051)
  - DA top predictor (r_pb=+0.059***), CO not significant (p=0.155), 7/10 dims significant
  - TE near-zero (v16 TE significance was adversarial proxy artifact)
  - Docs: distillation-research.md §34, journal.md §25
- **DonD** (Deal or No Deal): 12,234 negotiation dialogues, deal/no-deal outcome. **v23 rerun 2026-02-28.**
  - AUC=0.732 (+0.046 vs v18), g-PSQ=0.700 — Q4/Q1 gap 88.5%/59.7% = **28.7pp** (was 15.9pp)
  - **TE top predictor** (d=+0.801) — TE held-out 0.492→0.800 with v23. ED 7th bivariate (partial r ≈ TE after length control).
  - AD now positive (r_pb=+0.138), multivariate suppressor (coef=-0.746 — largest coefficient)
  - **T3b CONFIRMED**: AD predicts deal (+0.138***) but predicts points NEGATIVELY (-0.070***). AD=relational safety, not strategic advantage.
  - Docs: distillation-research.md §39, journal.md §27, criterion-validity-summary.md §2d

## ED construct validity (assessed 2026-02-28, updated v23)
- ED is a valid genuine singleton: lowest g-loading (R²=0.447), promax +0.77
- **DonD v23**: ED 7th bivariate (d=+0.535), but partial r=0.209 ≈ TE partial r=0.203 after length control
- Context-dependent: strong for experience (CaSiNo) and behavioral negotiation (DonD), weak for derailment (CGA-Wiki)
- Docs: distillation-research.md §37, §39

## DA construct validity concern
- DA max promax loading: 0.332 (below 0.35 threshold at 5-factor level)
- Mean r with other 9 dims: 0.480; separated-llm: DA-TC=0.825, DA-RC=0.768
- 49% of separated-llm DA scores are exact 5.0, std=1.13 (2nd lowest)
- Resolution requires human expert validation, not more LLM data

## Scoring experiments (COMPLETE — all interventions REJECTED)
- All 4 phases executed 2026-02-28. Full results in `scoring-experiments.md`.
- **Phase 0 (test-retest):** Δ_noise=0.011, 6/10 dims r≥0.80, AD unstable (r=0.156). Qualified GO.
- **Exp 1 (halo-awareness):** Initially ADOPTED, then **REVERSED** after g-factor structural analysis.
- **Exp 2 (dissimilar rubrics):** REJECTED. Construct redefinition, not halo reduction.
- **Exp 3 (scale format):** RETAINED 0-10. Scale has zero effect on halo.
- **G-factor structural analysis (§51):** g-factor is real co-variation (range/extremity effect), NOT scorer halo.
  - Extreme texts (g<3 or g>7): EV1=82.8%, uniform PC1 loadings (SD=0.023) — pure valence
  - Middle texts (g 4-6): EV1=38.7%, structured loadings (SD=0.117) — genuine differentiation
  - Halo-aware instruction's individual |Δ|=0.217 < test-retest noise floor 0.54
  - CC bias (+0.33 mean shift) and CO decoupling account for ~1/3 of headline SD improvement
- **Hierarchical model:** PSQ (g) → 2-3 clusters → 5 groups → 10 dimensions. g IS the PSQ at broadest level.
  - NOT bifactor (which makes g orthogonal to group factors, flattening the hierarchy)
  - Correct approach: middle-g text enrichment (g ∈ [3, 4.5) ∪ [5.5, 7]) for dimension-specific training signal
- Key principle: **never modify scoring prompt/rubrics in production** — test as controlled experiment first
- **No changes to scoring prompt.** Current prompt without halo-aware instruction is correct.

## Proxy data audit (2026-02-28)
- Proxy: 30,803 rows, 17.8% effective weight. 1 sep-llm row = 5.8× 1 proxy row.
- Corpus-wide proxy-LLM r: RB=0.539, RC=0.497, HI=0.488, DA=0.448 (usable); AD=0.155, CC=0.102, TC=0.071 (harmful); TE=-0.260 (actively harmful); ED=constant
- **Source-specific audit (proxy-audit batch, goemotions/ucc/casino/berkeley):** All dims show near-zero or negative r (TE=0.223, HI=-0.126, AD=-0.129, TC=-0.200, CC=-0.293, RB=-0.203, RC=0.004). The positive corpus-wide r values for "retained" dims come from OTHER sources (dreaddit/empathetic_dialogues), not from these four.
- **Action items complete:** Proxy dropped for TE/TC/CC/AD/ED. v22a confirmed best at held-out_r=0.682.
- Docs: distillation-research.md §52, §55

## Key files
- `TODO.md` — project-level task list | `EXPERIMENTS.md` — training run log
- `scripts/label_separated.py` — canonical labeling tool | `scripts/distill.py` — training
- `data/unlabeled-pool.jsonl` — 17,451 unlabeled texts | `data/held-out-test.jsonl` — 100 held-out texts
- `distillation-research.md` — running research notes | `journal.md` — research narrative
- `criterion-validity-summary.md` — cross-study criterion validity (v23 reruns 2026-02-28)
