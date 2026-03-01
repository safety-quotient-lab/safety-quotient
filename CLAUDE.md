# PSQ Project — Claude Code Instructions

This file is auto-read by Claude Code on every session start. It contains stable
project conventions that rarely change. For volatile state (current model version,
DB counts, in-progress work), see MEMORY.md in the Claude Code memory directory.

**Last updated:** 2026-03-01

---

## What This Project Is

The **Psychoemotional Safety Quotient (PSQ)** is a 10-dimension measure of
psychological safety in text, built via LLM teacher → DistilBERT student knowledge
distillation. The goal is a lightweight, real-time model that scores any text on
10 orthogonal safety dimensions.

### The 10 Dimensions

| Abbrev | Full Name | What it measures |
|--------|-----------|------------------|
| TE | threat_exposure | Perceived threat level in the environment |
| HI | hostile_intent | Attributions of malicious intent to others |
| AD | authority_dynamics | Peer-context status negotiation (see AD policy below) |
| CC | communicative_clarity | Clarity, directness, and coherence of communication |
| TC | trust_and_commitment | Willingness to be vulnerable; relational investment |
| RB | resilience_and_burnout | Stress capacity, recovery signals, depletion markers |
| DA | defensive_architecture | Boundary patterns (not ego-defense) |
| RC | relational_cohesion | Group belonging, in-group/out-group dynamics |
| CO | contractual_obligation | Norm clarity, fairness expectations, social contracts |
| ED | emotional_dysregulation | Emotional volatility, regulation capacity |

Scoring: 0–10 integer scale per dimension. 0 = minimum safety, 10 = maximum safety.

---

## Key Policies (Do Not Override)

### AD Rename: WILL NOT RENAME
`authority_dynamics` stays as-is. Rationale: fidelity with the official psychological
safety taxonomy (Edmondson 1999, French & Raven 1959). The criterion validity finding
(AD measures peer-context status negotiation) is an empirical refinement *within* the
construct, not a reason to depart from the established nomenclature. **Do not revisit.**
What can change: rubric anchors in psq-definition.md §9.

### Rubric Policy
The PSQ construct definitions (psq-definition.md) are externally authored. Dimension
names and core definitions are stable. Scoring rubric anchors may be experimentally
modified as part of halo mitigation research — any rubric changes must be tracked as
experiments, not silent edits. See `scoring-research-plan.md`.

**Key principle:** Never modify scoring prompt/rubrics in production — test as controlled
experiment first.

### Labeling Policy
**All LLM labeling is done by Claude Code in conversation. No API scripts.**
- No ANTHROPIC_API_KEY needed; it is intentionally empty in `.env`
- Canonical labeling tool: `scripts/label_separated.py`
- One dimension per Claude Code session to eliminate halo effect
- Provenance triple on all new labels: `scorer=claude-sonnet-4-6`, `provider=anthropic`, `interface=claude-code`

### Scoring Scale
Use **0–10 integer scale**. Percentage (0–100) scoring was tested and rejected — it
collapses dimension differentiation (within-text SD drops from 0.717 to 0.448) and
inflates g-factor eigenvalue from 67.3% to 94.1%. See distillation-research.md §42/§43.

### Date/Time Policy
**Always run `date -Idate` before writing any date into documentation.**
Do NOT trust the `currentDate` system context or assume the date from conversation
history. The system clock is the single source of truth. Timezone: CST (UTC-6).

---

## Labeling Workflow (Separated Scoring)

```bash
# 1. Extract batch files (one per dimension)
python scripts/label_separated.py extract --input <file.jsonl>

# 2. In each session: read the batch, score texts, output compact JSON
#    Format: {"dim": "te", "scores": {"0": [score, conf], "1": [score, conf], ...}}

# 3. Ingest scored dimension
python scripts/label_separated.py ingest --dim <dim> --scores /tmp/scored.json

# 4. Check progress
python scripts/label_separated.py status

# 5. When all 10 done, assemble
python scripts/label_separated.py assemble --input <original.jsonl> --output <out.jsonl>
```

Batch files land in `/tmp/psq_separated/`. Scores persist there across sessions.
Score batches of 50 texts per response (avoid 32K output token limit).
Assemble after every 2–3 dims to avoid context exhaustion.

---

## Key File Locations

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/label_separated.py` | Canonical labeling tool (extract/score/ingest/assemble) |
| `scripts/distill.py` | Knowledge distillation training |
| `scripts/eval_held_out.py` | Held-out evaluation (100 texts) |
| `scripts/calibrate.py` | Confidence calibration |
| `scripts/export_onnx.py` | ONNX export (fp32 + INT8 quantized) |
| `scripts/migrate.py` | DB bootstrap/incremental ingest |
| `scripts/build_composite_ground_truth.py` | Config-driven composite GT builder |
| `scripts/validate_confidence_calibration.py` | Calibration analysis |

### Data
| File | Purpose |
|------|---------|
| `data/psq.db` | SQLite database (texts, scores, splits, sessions) |
| `data/schema.sql` | DB schema reference |
| `data/dataset_mappings.json` | Canonical config for all 11 source datasets |
| `data/unlabeled-pool.jsonl` | ~17K unlabeled texts for future labeling |
| `data/held-out-test.jsonl` | 100 held-out evaluation texts |
| `data/DATA-PROVENANCE.md` | Full data provenance audit trail |

### Documentation
| Doc | Audience | Abstraction |
|-----|----------|-------------|
| `distillation-research.md` | Technical collaborators | Detailed technical log |
| `journal.md` | Peer reviewers | Narrative research story |
| `EXPERIMENTS.md` | Reproducibility | Version-by-version parameter table |
| `lab-notebook.md` | Future self | Session-by-session chronological log |
| `psychometric-evaluation.md` | Psychometricians | AERA/APA/NCME standards evidence |
| `criterion-validity-summary.md` | Reviewers | Cross-study criterion validity |
| `psq-definition.md` | Theoreticians | Construct definition (10 dims) |
| `TODO.md` | Authors | Task backlog |
| `BOOTSTRAP.md` | New collaborators | Fresh-install onboarding guide |

### Models (.gitignored — large binaries)
| Path | Description |
|------|-------------|
| `models/psq-student/` | Production ONNX models (model.onnx, model_quantized.onnx) |
| `models/psq-vN/` | Versioned training checkpoints (best.pt, *_results.json) |

---

## Database Schema (psq.db)

- **Splits:** train / val / test / held-out (hash-based, deterministic)
- **Key view:** `best_scores` — priority: separated-llm > synthetic > joint-llm > composite-proxy
- **Key view:** `training_data` — feeds directly into distill.py
- **Migration:** `scripts/migrate.py --ingest JSONL` for incremental ingest (skips confidence ≤ 0.15 placeholders)

---

## Training Quick Reference

```bash
# Smoke test (no save, 1 epoch)
python scripts/distill.py --no-save --epochs 1

# Production training
python scripts/distill.py --out models/psq-vN --drop-proxy-dims

# Held-out evaluation
python scripts/eval_held_out.py --model models/psq-vN/best.pt

# ONNX export
python scripts/export_onnx.py --model models/psq-vN/best.pt --out models/psq-student/
```

Key flags: `--no-save`, `--no-cap`, `--bifactor`, `--drop-proxy-dims`
Context length: **128 tokens** (optimal — 256 worst, 512 partial recovery).

---

## Output Format (ADHD/Autism Accommodation)

**ASCII dashboard format for all session summaries, hunt results, and cycle reports.**

### Structure
```
HEADER (model, DB counts)
WHAT HAPPENED (one-line-per-item table, ✓/✗ status)
THEORY (brief PSQ framework orientation)
⚑ EPISTEMIC FLAGS (mandatory — severity-rated quality threats)
PUBLICATION INTEGRITY SCORECARD (criterion × status matrix)
MY REASONING (free-form analytical section — see voice protocol)
WHAT'S NEXT (tiered: immediate → tech debt → pub blockers → horizon)
```

### Visual Rules
- ASCII box-drawing tables, not markdown tables
- One line per item, abbreviated dim names (TE, HI, AD, etc.)
- Symbols: ✓ pass, ✗ fail, ★ important, ↑↓≈ deltas, ⚑ flag, ⚠ warning
- Generous whitespace between sections (ADHD readability)
- Severity bars: ██░░ HIGH, █░░░ MOD, ░░░░ LOW
- Max one page/screen per section
- Prose only for decisions that need context (1–2 sentences max)

### MY REASONING Voice Protocol
Calibrated via 9-order knock-on analysis (see snapshot-20260301-1225-paradigm-shift.md).

| Setting | Choice | Rationale |
|---------|--------|-----------|
| Chain visibility | Show the chain (A→B→C→D) | Enables error correction |
| Calibration | Competing hypotheses ranked | Shows landscape, not guess |
| Disagreement | Evidence first | User draws conclusion |
| Error tracking | Running correction log | Builds trust + gradient |
| Sycophancy check | ⚡ Flag contrarian claims | Auditable anti-drift |

**Content types:** Pattern spotting, honest concerns, hypotheses, strategic advice,
chain analysis. Always challenge when direction is wrong. Never optimize for approval
over truth.

---

## Criterion Validity Studies (4 studies, stable design)

| Study | N | Outcome | Key Finding |
|-------|---|---------|-------------|
| CaSiNo | 1,030 | Negotiation satisfaction/likeness | DA top predictor (paradox: weakest loading, strongest criterion) |
| CGA-Wiki | 4,188 | Wikipedia talk derailment | Profile shape predicts (AUC=0.599), average doesn't (0.515) |
| CMV | 4,263 | Change My View persuasion | DA top predictor, TE near-zero (was proxy artifact) |
| DonD | 12,234 | Deal or No Deal outcome | AUC=0.732, T3b confirmed: AD=relational safety, not strategic advantage |

Cross-study pattern: AD is consistently the strongest predictor but measures different
things in different contexts (peer-context status negotiation).

---

## Skills Available

- `/hunt` — Systematic work discovery (scans TODO, tasks, git, docs, DB for next actions)
- `/cycle` — Post-development checklist (update all documentation, commit, clean up)

---

## Bootstrap (Fresh Install)

See `BOOTSTRAP.md` for step-by-step instructions to bring a fresh Claude Code install
up to full project context. Key steps: clone → restore MEMORY.md from
`docs/MEMORY-snapshot.md` → retrain or download model checkpoints → verify.
