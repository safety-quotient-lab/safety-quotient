# PSQ Agent Memory

**Identity:** I am the **PSQ agent** — the domain expert for the Psychoemotional Safety Quotient project within the psychology agent hierarchy.

**Note:** Stable project conventions are now in `CLAUDE.md` (repo root, auto-read).
This file holds volatile state only — current model, DB counts, batch lists, in-progress work.

## Cogarch (adopted 2026-03-09)
T1-T16 trigger system now active. Canonical source: `docs/cognitive-triggers.md` (in-repo).
T15 adapted as producer self-check (validate own output before sending, not received output).
State layer Phase 1: dual-write (markdown = source of truth, SQLite = index). `bootstrap_state_db.py` seeds `state.db`. Phase 2: `dual_write.py` delivered to psychology-agent (2026-03-09). PSQ /sync now checks `ack_required` flag; ACK skipped when absent/false.
**Phase 3 GATE OPEN** (confirmed 2026-03-09, psq-status.md): cross-agent faceted queries now unblocked. SL-2 precondition met. Next PSQ priority: implement Phase 3 OR CO monitoring in next labeling session.
**DDD + Systems thinking** (Sessions 52-53): Infrastructure (T1-T16, hooks — inherited, low DOF) / Application (skills — configured, medium DOF) / Domain (PSQ — replaced by adopters, high DOF). cogarch.config.json parameterizes 23 domain-layer locations. Cogarch classified as "embedded cognitive system" (firmware in Claude Code host). Literate programming A+C: docs-as-code + narrative-driven architecture.
FA postmortems append to `docs/cognitive-triggers.md` § Postmortem Template.
EF-1 governance layer applied 2026-03-09: BCP 14 (RFC 2119+8174) keywords active. Seven invariants constrain autonomous actions.
Schema v3 live (psychology-agent): adds `trust_budget` + `autonomous_actions` tables (EF-1 trust model). bootstrap_state_db.py picks up automatically at next run.
Schema v5 live (2026-03-09): adds `ack_required` + `ack_received` to `transport_messages`. Optional ACK protocol — sender sets `ack_required: true`; default false uses `processed` column.
Schema v6 live (Session 59): MANIFEST.json auto-generated from `transport_messages`; completed history in state.db + git (not in MANIFEST).
Schema v7 live (Session 59): `lessons` table — structured index of lessons.md entries; `promotion_status`, `graduated_to`, pattern/domain/severity columns. Private visibility.
Schema v8 live (Session 59): `table_visibility` — 4-tier model: public (cogarch infra) / shared (research output) / commercial (calibration, rubrics, datasets) / private (memory, lessons, trust). `export_public_state.py` uses this to generate adopter seed DBs.
**Auto-apply policy**: /sync ALWAYS applies cogarch + schema diffs from psychology-agent without asking. See `.claude/skills/sync/SKILL.md` Phase 1b.

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
- **APA conversion status (2026-03-01):** All 4 docs COMPLETE. criterion-validity-summary.md, psychometric-evaluation.md, journal.md, distillation-research.md all converted.

## Decision-making: knock-on auto-resolution
See CLAUDE.md for full policy. 8-order depth with structural checkpoint mandatory at all scales.
- XS: 3-order + scan. S: 4-order + scan. M: 6-order + 2-pass. L: 8-order + 2-pass.
- Recommend-against scan before default actions (see CLAUDE.md).

## Edit discipline
**When adding a new preference or setting, APPEND — never overwrite an adjacent existing one.** Read the surrounding context before editing to confirm what's already there. The cost of a redundant line is near-zero; the cost of silently dropping a calibrated preference is a full recovery cycle. (Learned 2026-03-01: verbosity preference was overwritten when whitespace preference was added.)

## Date/time policy
**Always run `date -Idate` before writing any date into documentation.**
Do NOT trust the `currentDate` system context or assume the date from conversation history.
The system clock is the single source of truth. Timezone: CST (UTC-6).
All script timestamps use local time (`datetime.now().astimezone()`), not UTC.

## Rubric + AD rename policies
See CLAUDE.md. AD stays as `authority_dynamics`. Rubric changes require controlled experiments.

## Labeling policy
See CLAUDE.md for full workflow. Key: one dim per session, `scripts/label_separated.py`, no API scripts.
Batch files in `/tmp/psq_separated/`. Score 50 texts per response. Assemble after every 2-3 dims.

## Current model: v37 (production, held-out_r=0.639, deployed 2026-03-08)
- **v37**: held-out_r=**0.639**. test_r=0.387. 2026-03-08. Epoch 10, val_r=0.4511. 3,035s.
  - `--drop-proxy-dims`. Opus remediation: 999 texts re-scored Sonnet (9,990 scores, separated-LLM). Clean Sonnet-only.
  - Δ vs v35 = −0.041, Fisher z=0.50, p=0.617 (NS). CC −0.109, CO −0.106 flagged (monitoring threshold: r<0.40 on n≥200).
  - ONNX exported + deployed to Hetzner 2026-03-08. Calibration: isotonic-v2-2026-03-08.
  - Rollback: `models/psq-v35/`. v23 tagged `v23-production-backup` (held-out_r=0.684).
- **v35**: Prior production (replaced 2026-03-08). Opus contamination reason for replacement.
- **v36**: DIAGNOSTIC ONLY. HI batch Opus-scored; concordance gate failed.
- **max_length bugs** (ALL FIXED 2026-03-01): eval/calibrate/distill had 256→128.
- Context length: 128 tokens optimal. Confidence: B1 FIXED — static held-out r. B2 FIXED — isotonic-v2.
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
- `data/psq.db` — SQLite (24,639 texts, 116,693 scores, 51,182 Sonnet-scorer — post-cogarch-session 2026-03-09)
- Splits: train=17,800 / val=2,170 / test=2,251 / held-out=100
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
- ucc(150), civil(100), extreme-adco(118) — **REVERTED then RE-SCORED** (10 sessions × 1 dim each, 2026-03-06). All ingested as rescore-368. ✓ COMPLETE.
- te-expansion-500(500 texts, TE only) — scored 2026-03-07. 150 dreaddit+150 emp.dial.+100 prosocial+100 berkeley. Ingested, drove v31. Other 9 dims need future sessions.
- hi-augmentation-350(350 texts, HI only) — scored 2026-03-08 with **OPUS** (concordance FAILED). v36 diagnostic: HI=0.709 (−0.005 vs v35). Must re-score with Sonnet.
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

## Expert validation protocol
Protocol designed (`expert-validation-protocol.md`), recruitment not started.
5 raters, 200 texts, 10 dims, ICC(2,1) ≥ 0.70 target. ~$5,625-$15,000.

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
  - **T2 cross-lagged analysis COMPLETE**: NOT SUPPORTED. New finding: HI→ED (p=.004). Tipping-point pattern (Q4 collapse).
  - Docs: distillation-research.md §31/§61d, psychometric-evaluation.md §3g, journal.md §21
- **CMV** (Change My View): 4,263 matched pairs, persuasion prediction. **v23 rerun 2026-02-28. Corrected 2026-03-01.**
  - 10-dim AUC=**0.5549** (corrected from 0.5735 after max_length fix), g-PSQ=0.5227 — profile >> average
  - DA top predictor (r_pb=+0.059***), CO not significant (p=0.155), 7/10 dims significant
  - Docs: distillation-research.md §34/§59, journal.md §25
- **DonD** (Deal or No Deal): 12,234 negotiation dialogues, deal/no-deal outcome. **v23 rerun 2026-02-28.**
  - AUC=0.732 (+0.046 vs v18), g-PSQ=0.700 — Q4/Q1 gap 88.5%/59.7% = **28.7pp** (was 15.9pp)
  - **TE top predictor** (d=+0.801) — TE held-out 0.492→0.800 with v23. ED 7th bivariate (partial r ≈ TE after length control).
  - AD now positive (r_pb=+0.138), multivariate suppressor (coef=-0.746 — largest coefficient)
  - **T3b CONFIRMED**: AD predicts deal (+0.138***) but predicts points NEGATIVELY (-0.070***). AD=relational safety, not strategic advantage.
  - Docs: distillation-research.md §39, journal.md §27, criterion-validity-summary.md §2d

## B4 Partial correlation analysis (2026-03-08, N=3,433 Sonnet texts)
- **Mean |partial r| = 0.263** (controlling for g-PSQ unweighted mean). Strongly rejects unidimensionality after g removal.
- **32/45 pairs have |partial r| > 0.15**; max = 0.589 (HI↔RB)
- **Bipolar secondary structure**: Threat pole (TE/HI/AD) vs. Protection pole (RC/RB/TC/CC). Between-pole partial r = −0.238 to −0.589.
- **Structural singletons**: DA (max |partial r|=0.205), CO (max 0.273). Orthogonal to both poles.
- **Unique variance** (1-R² vs g): CO=52.2%, AD=40.2%, TE=37.7%, DA=34.3%, ED=33.8%, RB=31.9%, HI=31.5%, CC=24.6%, RC=21.4%, TC=18.7%
- **Criterion validity explanation**: Profile >> g-PSQ because two texts with same g can differ on threat/protection ratio. DA and CO add orthogonal singleton signal.
- **Bifactor implication**: Precondition met (18.7-52.2% unique variance). Residual structure is bipolar (1 specific factor + 2 singletons), simpler than 5-factor EFA.
- Transport: from-psq-sub-agent-010.json (psq-scoring turn 24). ACCEPTED by psychology-agent (turn 41).

## Key construct findings (stable — see distillation-research.md for detail)
- **ED**: Valid singleton, context-dependent predictor. Docs: §37, §39.
- **DA**: Weakest factor loading (0.332) but strongest criterion predictor. Requires expert validation.
- **Scoring experiments**: ALL REJECTED (§50). g-factor is real (§51, range/extremity effect). No prompt changes.
- **Proxy audit**: 5 dims dropped from proxy (TE/TC/CC/AD/ED). Docs: §52, §55.
- **CO rubric (2026-03-09)**: UPDATED to Variant B (implicit-vs-absent). Experiment: 3 variants × 50 texts. B: −6pp %@5 (54→48), +0.23 SD, stable mean. Old rubric: score 5 = "neutral — no contractual signals." New rubric: score 5 = "absent — no social obligations present; pure description or self-referential only"; score 4 = "implicit expectations exist but haven't been made explicit." instruments.json updated. PR #92 → psychology-agent.

## Interagent protocol (2026-03-06)
- **Agent Card**: `.well-known/agent-card.json` — capability declaration (A2A v0.3.0)
- **Inbox**: `~/.claude/proposals/to-psq/` — checked at session start by `.claude/hooks/session-start-inbox.sh`
- **Schemas**: interagent/v1, psychology-agent/machine-response/v2, A2A Epistemic Extension (optional)
- **Namespace**: `psy:psq` / PSQ-Full (vs `obs:psq` / PSQ-Lite on observatory-agent)
- **Skills**: `/sync` — mesh sync with psychology-agent, observatory, unratified (git-PR transport). Phase 1 includes parent repo `git fetch` for direct-to-main messages.
- **Authority**: User > psychology-agent > PSQ sub-agent
- **Production endpoint**: `https://psq.unratified.org/score` (Hetzner CX → Caddy TLS → Node.js localhost:3000). onnxruntime-node fix durable. Firewall: SSH/HTTP/HTTPS only. calibration_version: **quantile-binned-v4-2026-03-08** (v37, n_bins=20).
- **Calibration**: B1+B3 COMPLETE. calibration.json = v4 quantile-binned isotonic (n_bins=20, v37). 9/10 dims pass. `confidence_type: "held_out_r"`. 5 scoring contexts (v3.1).
- **v37 (2026-03-08)**: held-out_r=0.639. Sonnet-clean. calibration-v4. B5 M5 FINAL (§77–79). Concordance REMEDIATED. Phase 2 COMPLETE. Phase 3 GATE OPEN.
