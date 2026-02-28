# PSQ Project — Working State

This file captures the current operational state of the PSQ Content Evaluator project.
It is updated at the end of each working session. Snapshots are saved as
`working-state-snapshot-YYYYMMDD-HHMM.md`.

---

## Current Model: psq-v21 (production) / v22a (promotion candidate)

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

## v22 Ablation Experiment (complete)

Testing two independent interventions: (A) proxy removal for 4 dims, (B) middle-g enrichment data.

| Run | Intervention | Status | test_r | held-out_r | Δ vs v21 | Notes |
|---|---|---|---|---|---|---|
| v22a | `--drop-proxy-dims` (removes TE,TC,CC,AD composite-proxy rows) | **Done** | 0.457 | **0.682** | **+0.052** | **NEW BEST.** Dominant intervention. |
| v22b | +midg batch only (250 texts × 10 dims, no proxy drop) | **Done** | — | 0.578 | **-0.052** | **WORSE than v21.** Proxy noise overwhelms midg signal. |
| v22c | Both (proxy drop + midg data) | Pending | — | — | — | `--out models/psq-v22c --drop-proxy-dims` |

**Key conclusion:** Data quality > data quantity. Proxy removal (v22a: +0.052) is the dominant intervention. Midg data without proxy cleanup (v22b: -0.052) is neutral-to-negative. The ablation is symmetric — the adversarial proxy rows actively fight the midg signal.

**v22a held-out highlights:** TE 0.492→**0.805** (+0.313, largest ever), 9/10 dims improved. Only CC regressed (-0.051). TE proxy was adversarial (r=-0.260); removal unleashed separated-LLM signal.

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

### Current (N=2,420 complete texts, separated-llm)
- Overall g-factor eigenvalue: **7.06 (70.6% variance)**
- **RANGE-DEPENDENT:** g-factor is not uniform across the safety continuum
  - Extreme texts (g<3.5 or g>6.5, N=469): EV1=**79.6%**, mean |r|=0.772 — pure valence
  - Middle-g texts (4≤g≤6, N=1,602): EV1=**39.0%**, mean |r|=0.286 — genuine differentiation
- g-factor is real co-variation, not artifact: extreme texts genuinely are uniformly extreme
- Middle-g band satisfies conventional discriminant validity (mean |r|=0.286, well below 0.50 threshold)
- **Implication:** g-PSQ is most informative at extremes; dimension profile is most informative in the middle band

### v2 (separated-llm only, N=1,970) — provenance
- g-factor eigenvalue: **6.727 (67.3% variance)** — up from 4.844 (48.4%) in v1
- KMO = **0.902** ("Superb")
- Parallel analysis retains **1 factor only**
- g-factor loadings all >0.66: TC (0.930), DA (0.914), CC (0.864), RC (0.854)
- Mean inter-dim |r| = **0.632**

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
- **g-factor inflation uncertainty** (PARTIALLY RESOLVED): g-factor is range-dependent — EV1=39.0% in middle-g texts (N=1,602), rising to 79.6% in extreme texts. The overall 70.6% figure is not artifactual; it reflects genuine co-variation in extreme texts. Discriminant validity in the middle-g band is acceptable (mean |r|=0.286). Resolution of the residual uncertainty requires expert validation data.
- **Proxy data quality** (RESOLVED for TE/TC/CC/AD): v22a ablation confirmed proxy removal improves held-out performance (+0.052 vs v21). v22b confirmed midg data alone is insufficient (-0.052). Default `--drop-proxy-dims` now includes ED (constant 5.0, r=NaN).
- **CC regression in v22a**: contractual_clarity dropped from 0.555 to 0.504 after proxy removal. CC-targeted batch (200 texts, keyword-filtered) prepared to address.

---

## Key Design Constraint

**The PSQ construct definitions and scoring rubric (psq-definition.md) are externally authored and immutable.** All 10 dimension definitions, score anchors, and the overall PSQ formula are treated as fixed inputs to the measurement system. When score distributions show concentration or other statistical issues, the response is to improve data sourcing and model training — never to revise the rubric itself.

---

## What's Next

1. **Promote v22a to production** — held-out_r=0.682 (+0.052 vs v21). v22b confirms v22a is the superior model. No need to wait for v22c.
2. **Score CC-targeted batch** — `data/labeling-batch-ccda.jsonl` (200 texts) prepared. CC is the only dim that regressed in v22a (0.555→0.504). Then train v22c (proxy removal + midg + CC-enriched).
3. **Expert validation** — protocol designed (`expert-validation-protocol.md`), recruitment not started. Priority: DA (lowest construct validity confidence) and TE (highest recent gain — expert confirmation warranted).
4. **Expand held-out set** — `data/labeling-batch-held-out-expand.jsonl` (150 texts) prepared. Expanding from 100 to 250 texts will reduce evaluation variance.

### Completed This Session

- v22b: held-out_r=0.578, WORSE than v21 by -0.052. All 10 dims regressed.
- **Ablation complete:** proxy removal (+0.052) >> midg data alone (-0.052). Data quality > data quantity confirmed.
- **Range-dependent g-factor documented:** middle-g texts (N=1,602) EV1=39.0% vs extreme texts (N=469) EV1=79.6%. g is real, range-sensitive, correctly measuring the constructs.
- Source-level profile analysis: berkeley g=3.85, dreaddit g=4.10, esconv g=4.48, empathetic_dialogues g=5.05, prosocial g=5.14. TE η²=0.627, CO η²=0.105.
- Text-length analysis: 25.1% texts exceed 128 words; r(length, g)=+0.018 (ns); long texts show lower g-variance (SD=0.84 vs 1.26).
- Infrastructure: ED added to default --drop-proxy-dims; --curriculum flag implemented; 4 new batch files prepared.
- §55 added to distillation-research.md; §33 added to journal.md; psychometric-evaluation.md §3c updated with range-dependent g-factor; EXPERIMENTS.md v22b row added; working-state.md updated.

---

*Last updated: 2026-02-28 (v22 ablation complete; range-dependent g-factor documented)*
