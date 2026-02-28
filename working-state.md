# PSQ Project — Working State

This file captures the current operational state of the PSQ Content Evaluator project.
It is updated at the end of each working session. Snapshots are saved as
`working-state-snapshot-YYYYMMDD-HHMM.md`.

---

## Current Model: psq-v18 (production)

| Metric | Value |
|---|---|
| Architecture | DistilBERT → 10-dim regression (knowledge distillation) |
| Test r (avg 10 dims) | 0.525 |
| Held-out r (avg 10 dims) | 0.568 (+0.007 vs v16, new best) |
| Epochs | 10 (no early stop) |
| Production checkpoint | `models/psq-student/best.pt` |
| ONNX | needs re-export from v18 |

### Per-dimension held-out r (v18)

| Dimension | Held-out r | Δ vs v16 | Status |
|---|---|---|---|
| regulatory_capacity | 0.679 | +0.116 | **best** — massive jump |
| resilience_baseline | 0.651 | +0.088 | strong gain |
| trust_conditions | 0.620 | +0.045 | improved |
| cooling_capacity | 0.618 | -0.025 | good |
| authority_dynamics | 0.599 | -0.026 | good |
| energy_dissipation | 0.562 | -0.030 | good |
| hostility_index | 0.557 | -0.047 | good |
| contractual_clarity | 0.533 | -0.001 | flat |
| defensive_architecture | 0.488 | -0.035 | moderate |
| threat_exposure | 0.370 | +0.023 | correlation artifact |

### g-Factor Prerequisite Check

r(mean_pred, mean_target) = **0.644** on held-out (n=87). Well below 0.95 — bifactor Option A is worth building.

---

## Database: data/psq.db

| Table | Count |
|---|---|
| texts | 21,127 |
| scores | 73,361 |
| separated-llm | 19,771 |
| composite-proxy | 40,487 |
| joint-llm | 12,257 |
| synthetic | 846 |
| held-out | 100 texts (separate, not in training) |

---

## Completed Labeling Batches

| Batch | File | Texts | Dims | Focus |
|---|---|---|---|---|
| Weak dims | `labeling-batch-weak-dims.jsonl` | 200 | all 10 | te/rc/co |
| RC focus | `labeling-batch-rc.jsonl` | 150 | all 10 | regulatory_capacity |
| AD focus | `labeling-batch-ad.jsonl` | 300 | all 10 | authority_dynamics |
| CO focus | `labeling-batch-co.jsonl` | 200 | all 10 | contractual_clarity (keyword-filtered, CO mean=4.36) |
| RB focus | `labeling-batch-rb.jsonl` | 200 | all 10 | resilience_baseline |
| CC focus | `labeling-batch-cc.jsonl` | 200 | all 10 | cooling_capacity |
| TE focus | `labeling-batch-te.jsonl` | 200 | all 10 | threat_exposure (TE mean=3.17) |
| **Broad spectrum** | `labeling-batch-broad.jsonl` | 300 | 0/10 | **Not yet scored.** 150 random + 100 single-dim + 50 multi-dim. |

---

## Criterion Validity Studies

| Study | Dataset | Outcome | Key Result | Status |
|---|---|---|---|---|
| CaSiNo | 1,030 negotiations | satisfaction, likeness | AD top predictor (r=0.127***) | Complete (§30) |
| CGA-Wiki | 4,188 Wikipedia talks | derailment | AD top predictor, AUC=0.599 | Complete (§31) |
| **CMV** | **4,263 paired replies** | **persuasion (delta)** | **DA top predictor, AUC=0.590** | **Complete (§34)** |
| Deal or No Deal | Lewis et al. 2017 | deal/points | — | Running in background |

**Key cross-study finding:** AD is top predictor when status is *contested* (CaSiNo, CGA-Wiki). DA is top predictor when status is *fixed* (CMV). Profile >> average in all 3 studies (g-PSQ near-chance: 0.515–0.531). See journal §24–25 for theoretical analysis.

---

## Factor Structure

- EFA on 2,359 complete texts → 5-factor BIC-optimal
- Dominant g-factor: eigenvalue 4.844 (48.4% variance)
- 5 clusters: Hostility/Threat (HI,TE,CC), Relational Contract (CO,TC), Internal Resources (RB,RC,DA), Power Dynamics (AD), Stress/Energy (ED)
- AD and ED are singleton factors — genuinely independent
- 5-factor preserves 88% of 10-dim info (avg R²=0.881)
- g-PSQ alone AUC=0.515 (near-chance) vs 10-dim AUC=0.599

---

## AD Paradox: Three Theories (journal §24)

1. **Meta-conversation** (Watzlawick 1967): AD measures command channel, not report
2. **Leading indicator**: AD deteriorates before HI/TE in derailing conversations
3. **Status negotiation** (Tajfel & Turner 1979): AD measures epistemic/moral status positioning

CMV results favor Theory 3 (context-dependent). Construct rename: authority_dynamics → power_positioning.

---

## Key Architecture Decisions

- **Separated scoring**: one LLM call per dimension per text (eliminates halo effect)
- **Score-concentration cap**: `_cap_score_concentration()` — when >30% of a dim's scores share one value, excess rows downweighted from 5.0 to 1.5
- **SQLite schema**: scores as observations, texts/dimensions as axes
- **best_scores view**: priority separated-llm > synthetic > joint-llm > composite-proxy
- **Splits**: frozen md5(text) hash assignments in `splits` table

---

## Known Issues

- **AD data provenance**: 70.4% LLM-generated effective training signal (middle of pack, not outlier). Needs expert validation to confirm.
- **TE correlation artifact**: held-out r=0.370 is misleadingly low; MAE actually improved. Caused by restricted variance in held-out TE scores.
- **DA construct validity**: max promax loading 0.332 (below 0.35 threshold). 49% of scores are exact 5.0. Requires human expert validation.

---

## What's Next

See `TODO.md` for full task list. Immediate priorities:

1. **ONNX re-export from v18** — v18 promoted, ONNX needs re-export + INT8 quantization
2. **DonD results** — when script completes, document (tests AD deal vs points prediction)
3. **Score broad-spectrum batch** — 300 texts × 10 dims in `/tmp/psq_separated/`
4. **Bifactor Option A** — g-factor prerequisite confirmed (r=0.644), implement g-head
5. **Score distribution remediation** — CO at 63.2% score-5 needs rubric revision, not just more data
6. **Scoring rubric review** — check if psq-definition.md anchors still match learned constructs

---

*Last updated: 2026-02-28*
