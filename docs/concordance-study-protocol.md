# Cross-Scorer Concordance Study: Opus vs Sonnet

**Status:** Protocol designed, execution pending
**Gate:** Psychology-agent T16 — must complete before v36 promotion or further Opus-scored batches enter training
**Date:** 2026-03-08

---

## Purpose

Measure agreement between Claude Opus (claude-opus-4-6) and Claude Sonnet (claude-sonnet-4-6) on PSQ dimension scoring. Two Opus-scored batches (1,000-text rescore + 350 HI augmentation) have entered training data. This study determines whether the two scorers produce interchangeable labels.

## Design

| Parameter | Value |
|-----------|-------|
| Sample | 50 texts, stratified by source (11 datasets) |
| Dimensions | All 10 PSQ dimensions |
| Protocol | Separated-LLM (1 dim per session, blind to other dims) |
| Sonnet scores | Already in DB (clean `claude-sonnet-4-6` provenance) |
| Opus scores | To be scored blind (no Sonnet scores visible) |
| Total comparisons | 500 (50 texts × 10 dims) |
| Sessions required | 10 (1 per dimension) |

## Sample

Source-stratified sample from 3,433 texts with verified `claude-sonnet-4-6` provenance and full 10-dimension separated-llm coverage. Random seed = 42.

| Source | N |
|--------|---|
| berkeley | 10 |
| prosocial | 10 |
| esconv | 8 |
| dreaddit | 7 |
| empathetic_dialogues | 7 |
| civil_comments | 2 |
| ucc | 2 |
| casino | 1 |
| goemotions | 1 |
| politeness_stack-exchange | 1 |
| politeness_wikipedia | 1 |

## Files

| File | Purpose |
|------|---------|
| `data/concordance-study-sample.jsonl` | Full sample with Sonnet reference scores (for analysis) |
| `data/concordance-opus-blind.jsonl` | Blind sample for Opus scoring (no Sonnet scores) |

## Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Per-dimension ICC(2,1) | >= 0.70 | Matches expert validation protocol target (Cicchetti, 1994) |
| Per-dimension Pearson r | >= 0.80 (desirable) | Strong agreement |
| Mean absolute difference | < 1.0 per dimension | Within 1 scale point |

If any dimension falls below ICC 0.70, expand to n=100 for that dimension before concluding (per psychology-agent T18 guidance).

## Execution Protocol

1. Extract concordance batch: `label_separated.py extract --input data/concordance-opus-blind.jsonl`
2. Score each dimension in a separate session (1 dim per session, standard separated-llm protocol)
3. Scorer provenance: `claude-opus-4-6` / `anthropic` / `claude-code`
4. Do NOT look at `data/concordance-study-sample.jsonl` during scoring — it contains Sonnet reference scores
5. After all 10 dims scored, run analysis script

## Analysis Plan

For each dimension:

1. **Pearson r** — linear agreement
2. **ICC(2,1)** — absolute agreement (two-way random, single measures)
3. **Mean difference** (Opus − Sonnet) — systematic bias detection
4. **Mean absolute difference** — magnitude of disagreement
5. **Bland-Altman plot** — bias × magnitude relationship

Aggregate:
- Mean ICC across dimensions
- Number of dimensions passing threshold
- Systematic bias direction (does Opus score higher or lower than Sonnet?)

## Interpretation Guide

| Outcome | Action |
|---------|--------|
| All 10 dims ICC >= 0.70 | Gate clears. Opus-scored data validated. v36 eligible for promotion. |
| 7-9 dims pass, 1-3 fail | Expand failing dims to n=100. If still fail, retrain with Sonnet-only labels for those dims. |
| < 7 dims pass | Opus scoring not interchangeable. Retrain v36 with Sonnet-only labels. Consider Opus-to-Sonnet score adjustment. |

## Epistemic Flags

- Sample size (n=50) provides moderate power. ICC confidence intervals will be wide (~±0.15). Marginal results (ICC 0.65-0.75) will be ambiguous.
- The sample is drawn from training data, not held-out. This is necessary because held-out texts have unclear provenance ("claude-code" scorer, not "claude-sonnet-4-6").
- Opus and Sonnet may differ in calibration (systematic offset) even if rank-order agreement is high. Check mean difference alongside ICC.
- The study cannot determine which scorer is "correct" — it only measures agreement. Both may be wrong in the same way.
