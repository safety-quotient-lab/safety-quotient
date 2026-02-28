# Research Session Transcripts

Raw conversation transcripts from Claude Code research sessions on the PSQ project. These constitute the complete audit trail of research decisions, analysis, scoring, and development work.

## Purpose

In this project, Claude Code serves as both research assistant and measurement instrument (LLM-as-judge scoring). The session transcripts document:

- **Research decisions** — why specific analyses were run, how results were interpreted
- **Scoring sessions** — the actual LLM scoring of texts across 10 dimensions (separated protocol)
- **Data pipeline operations** — batch extraction, ingestion, training runs, evaluations
- **Exploratory analysis** — factor analysis, criterion validity, error analysis
- **Design decisions** — architecture choices, hyperparameter rationale, construct discussions

## Format

Each file is a JSONL transcript from Claude Code, named `YYYYMMDD-HHMM_<session-id>.jsonl`. Each line is a JSON object with message metadata and content.

## Session Index

| File | Date | Size | Topics |
|---|---|---|---|
| `20260227-1447_d2248669.jsonl` | 2026-02-27 14:47 | 12K | Initial setup, help commands |
| `20260227-1451_baa1de15.jsonl` | 2026-02-27 14:51 | 4K | Brief session |
| `20260227-1740_4a3bffb6.jsonl` | 2026-02-27 17:40 | 8.7M | Major session: SQLite migration, dataset_mappings.json, v14 training, separated scoring infrastructure, labeling provenance, working-state.md creation |
| `20260227-1901_cab0552f.jsonl` | 2026-02-27 19:01 | 1.9M | RC batch scoring (150 texts × 10 dims), context limit encountered, batch mismatch fix |
| `20260227-1948_88f676c1.jsonl` | 2026-02-27 19:48 | 4.5M | AD batch scoring (300 texts × 10 dims), v15 training, CO regression diagnosis, score-concentration cap implementation, CO/RB/CC batch creation |
| `20260228-1105_9e5127a1.jsonl` | 2026-02-28 11:05 | 68M | Multi-day marathon: CO/RB/CC/TE batch scoring, v16-v21 training, factor analysis (EFA, promax), expert validation protocol, 4 criterion validity studies (CaSiNo, CGA-Wiki, CMV, DonD), bifactor evaluation, percentage scoring experiment, scoring experiments (halo-awareness, rubric variants), proxy data audit, v22 ablation series, middle-g batch, range-dependent g-factor discovery |

## Storage

These files are stored in git via Git LFS (or excluded from git and backed up separately) due to their size (~83MB total). The scientific findings from these sessions are documented in:

- `distillation-research.md` — technical analysis and results
- `journal.md` — narrative research story
- `EXPERIMENTS.md` — training run parameters and metrics
- `psychometric-evaluation.md` — validation evidence

The transcripts provide the complete audit trail behind those summaries.

## Provenance

All sessions used Claude Code (Anthropic) as the interface. Model versions varied across sessions (claude-sonnet-4-6 for scoring, claude-opus-4-6 for analysis and development). The scorer, provider, and interface are recorded on all ingested labels in the database.
