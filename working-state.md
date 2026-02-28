# PSQ Project — Working State

This file captures the current operational state of the PSQ Content Evaluator project.
It is updated at the end of each working session. Snapshots are saved as
`working-state-snapshot-YYYYMMDD-HHMM.md`.

---

## Current Model: psq-v21 (production)

| Metric | Value |
|---|---|
| Architecture | DistilBERT → 10-dim regression (knowledge distillation) |
| Test r (avg 10 dims) | 0.504 |
| Held-out r (avg 10 dims) | **0.630** (best ever, +0.030 vs v19) |
| Epochs | 6 (early stopped epoch 9, best epoch 6) |
| Production checkpoint | `models/psq-student/best.pt` |
| ONNX | re-exported from v21 (254 MB / 64 MB quantized) |

### Per-dimension held-out r (v21)

| Dimension | Held-out r | Δ vs v19 | Status |
|---|---|---|---|
| regulatory_capacity | 0.729 | +0.019 | **best** — ceiling dim |
| cooling_capacity | 0.687 | +0.085 | strong gain |
| authority_dynamics | 0.674 | +0.017 | improved |
| trust_conditions | 0.674 | +0.038 | improved |
| energy_dissipation | 0.636 | -0.013 | slight regression |
| hostility_index | 0.658 | +0.087 | strong gain |
| resilience_baseline | 0.600 | -0.024 | slight regression |
| defensive_architecture | 0.566 | +0.028 | improved |
| contractual_clarity | 0.555 | +0.042 | improved |
| threat_exposure | 0.492 | -0.003 | flat — still weakest |

---

## v22 Ablation Experiment (in progress)

Testing two independent interventions: (A) proxy removal for 4 dims, (B) middle-g enrichment data.

| Run | Intervention | Status | test_r | Notes |
|---|---|---|---|---|
| v22a | `--drop-proxy-dims` (removes TE,TC,CC,AD composite-proxy rows) | **Done** | 0.446 | Proxy removal alone hurt — need held-out eval |
| v22b | +midg batch only (250 texts × 10 dims, no proxy drop) | Pending | — | `--out models/psq-v22b` |
| v22c | Both (proxy drop + midg data) | Pending | — | `--out models/psq-v22c --drop-proxy-dims` |

**v22a per-dim test_r:** CC 0.721 (strong), ED 0.592, RB 0.520, HI 0.520, RC 0.491, DA 0.444, CC 0.403, AD 0.358, TC 0.285, TE 0.228

---

## Database: data/psq.db

| Table | Count |
|---|---|
| texts | 21,877 |
| scores | 82,861 |
| separated-llm | 29,271 |
| composite-proxy | 40,487 |
| joint-llm | 12,257 |
| synthetic | 846 |
| held-out | 100 texts (separate, not in training) |
| train split | 17,458 texts |
| val split | 2,122 texts |
| test split | 2,203 texts |

---

## Completed Labeling Batches

| Batch | File | Texts | Dims | Focus |
|---|---|---|---|---|
| Weak dims | `labeling-batch-weak-dims.jsonl` | 200 | all 10 | te/rc/co |
| RC focus | `labeling-batch-rc.jsonl` | 150 | all 10 | regulatory_capacity |
| AD focus | `labeling-batch-ad.jsonl` | 300 | all 10 | authority_dynamics |
| CO focus | `labeling-batch-co.jsonl` | 200 | all 10 | contractual_clarity (keyword-filtered) |
| RB focus | `labeling-batch-rb.jsonl` | 200 | all 10 | resilience_baseline |
| CC focus | `labeling-batch-cc.jsonl` | 200 | all 10 | cooling_capacity |
| TE focus | `labeling-batch-te.jsonl` | 200 | all 10 | threat_exposure (TE mean=3.17) |
| Broad spectrum | `labeling-batch-broad.jsonl` | 300 | all 10 | 150 random + 100 single-dim + 50 multi-dim |
| Pct pilot | `labeling-batch-pct.jsonl` | 200 | all 10 | percentage-scale scoring pilot |
| CO #2 | `labeling-batch-co2.jsonl` | 200 | all 10 | CO-keyword texts (CO mean=4.36) |
| Middle-g | `labeling-batch-midg.jsonl` | 250 | all 10 | g∈[3,4.5)∪[5.5,7] differentiated texts |

**Total separated-llm labels ingested:** 29,271 scores

---

## Criterion Validity Studies

| Study | Dataset | Outcome | Key Result | Status |
|---|---|---|---|---|
| CaSiNo | 1,030 negotiations | satisfaction, likeness | AD top predictor (r=0.127***) | Complete (§30) |
| CGA-Wiki | 4,188 Wikipedia talks | derailment | AD top predictor, AUC=0.599 | Complete (§31) |
| CMV | 4,263 paired replies | persuasion (delta) | DA top predictor, AUC=0.590 | Complete (§34) |
| **DonD** | **12,234 dialogues** | **deal reached** | **ED top (d=+0.614), AUC=0.686** | **Complete (§39)** |

**Key cross-study finding:** Context-dependent primacy across 4 studies. AD dominates contested-status/relational (CaSiNo, CGA-Wiki). ED dominates sustained engagement/behavioral (DonD). DA dominates fixed-status (CMV). Profile >> average in all 4 studies (g-PSQ: 0.515–0.622).

---

## Factor Structure

### v2 (separated-llm only, N=1,970)
- g-factor eigenvalue: **6.727 (67.3% variance)** — up from 4.844 (48.4%) in v1
- KMO = **0.902** ("Superb")
- Parallel analysis retains **1 factor only**
- g-factor loadings all >0.66: TC (0.930), DA (0.914), CC (0.864), RC (0.854)
- Mean inter-dim |r| = **0.632**
- **Integer-only scoring bias discovered**: LLM uses 11 bins (integers 0-10), not continuous scale
- g-factor may be partly inflated by score-5 concentration (24-61% across dims)

---

## Key Architecture Decisions

- **Separated scoring**: one LLM call per dimension per text (eliminates halo effect)
- **Score-concentration cap**: `_cap_score_concentration()` — when >30% of a dim's scores share one value, excess rows downweighted from 5.0 to 1.5
- **SQLite schema**: scores as observations, texts/dimensions as axes
- **best_scores view**: priority separated-llm > synthetic > joint-llm > composite-proxy
- **Splits**: frozen md5(text) hash assignments in `splits` table
- **`--drop-proxy-dims`**: removes composite-proxy rows for TE, TC, CC, AD (experimental)

---

## Known Issues

- **Integer-only scoring bias** (MITIGATED): 0-100 percentage scale implemented but RETRACTED — FA v3 showed dimension collapse (eigenvalue 9.41 = 94.1% shared variance). All production scoring uses integer scale.
- **CO score-5 concentration** (IMPROVED): Score-concentration cap mitigates. Middle-g batch CO still 92.8% at 5 (expected — texts not CO-relevant).
- **DA construct validity**: max promax loading 0.332 (below 0.35 threshold). 49% of scores are exact 5.0. Requires human expert validation.
- **g-factor inflation uncertainty**: g-factor eigenvalue 6.727 (67.3%) may be partly artifactual from integer-only bias. Resolution requires expert validation data.
- **Proxy data quality**: Composite-proxy data for TE, TC, CC, AD may be low-quality (v22a ablation testing removal).

---

## Key Design Constraint

**The PSQ construct definitions and scoring rubric (psq-definition.md) are externally authored and immutable.** All 10 dimension definitions, score anchors, and the overall PSQ formula are treated as fixed inputs to the measurement system. When score distributions show concentration or other statistical issues, the response is to improve data sourcing and model training — never to revise the rubric itself.

---

## What's Next

1. **Complete v22 ablation** — Run v22b (midg only) and v22c (both). Evaluate all three on held-out. Determine if proxy removal, midg enrichment, or both improve held-out_r.
2. **Held-out eval for v22a** — v22a training complete (test_r=0.446). Need held-out evaluation to see if lower test_r translates to held-out.
3. **Expert validation** — protocol designed (`expert-validation-protocol.md`), recruitment not started.

### Completed This Session

- Middle-g batch scored: 250 texts × 10 dims = 2,500 new separated-llm labels
- Midg batch ingested into psq.db (21,877 texts, 82,861 scores)
- `--drop-proxy-dims` flag added to distill.py
- v22a training completed (test_r=0.446, proxy removal only)
- §53 added to distillation-research.md documenting v22 intervention design

---

*Last updated: 2026-02-28 (v22 ablation + middle-g batch)*
