# PSQ Project — Working State

This file captures the current operational state of the PSQ Content Evaluator project.
It is updated at the end of each working session. Snapshots are saved as
`working-state-snapshot-YYYYMMDD-HHMM.md`.

---

## Current Model: psq-v19 (production)

| Metric | Value |
|---|---|
| Architecture | DistilBERT → 10-dim regression (knowledge distillation) |
| Test r (avg 10 dims) | 0.509 |
| Held-out r (avg 10 dims) | **0.600** (+0.032 vs v18, new best) |
| Epochs | 10 (early stopped epoch 7, best epoch 4) |
| Production checkpoint | `models/psq-student/best.pt` |
| ONNX | re-exported from v19 (254 MB / 64 MB quantized) |

### Per-dimension held-out r (v19)

| Dimension | Held-out r | Δ vs v18 | Status |
|---|---|---|---|
| regulatory_capacity | 0.710 | +0.031 | **best** — continued improvement |
| authority_dynamics | 0.657 | +0.058 | strong gain |
| energy_dissipation | 0.649 | +0.087 | strong gain |
| trust_conditions | 0.636 | +0.016 | improved |
| resilience_baseline | 0.624 | -0.027 | slight regression |
| cooling_capacity | 0.602 | -0.016 | good |
| hostility_index | 0.571 | +0.014 | good |
| defensive_architecture | 0.538 | +0.050 | improved |
| contractual_clarity | 0.513 | -0.020 | slight regression |
| threat_exposure | 0.495 | **+0.125** | **massive recovery** |

### Bifactor Status

- `--bifactor` flag implemented and smoke-tested (1 epoch, g_psq r=0.5277)
- g-factor prerequisite confirmed: r(mean_pred, mean_target) = 0.644 on held-out
- Full bifactor training deferred pending v19 cycle completion

---

## Database: data/psq.db

| Table | Count |
|---|---|
| texts | 21,427 |
| scores | 76,361 |
| separated-llm | 22,771 |
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
| Broad spectrum | `labeling-batch-broad.jsonl` | 300 | all 10 | broad-spectrum (150 random + 100 single-dim + 50 multi-dim) |

---

## Criterion Validity Studies

| Study | Dataset | Outcome | Key Result | Status |
|---|---|---|---|---|
| CaSiNo | 1,030 negotiations | satisfaction, likeness | AD top predictor (r=0.127***) | Complete (§30) |
| CGA-Wiki | 4,188 Wikipedia talks | derailment | AD top predictor, AUC=0.599 | Complete (§31) |
| CMV | 4,263 paired replies | persuasion (delta) | DA top predictor, AUC=0.590 | Complete (§34) |
| **DonD** | **12,234 dialogues** | **deal reached** | **ED top (d=+0.614), AUC=0.686** | **Complete (§39)** |

**Key cross-study finding:** Context-dependent primacy across 4 studies. AD dominates contested-status/relational (CaSiNo, CGA-Wiki). ED dominates sustained engagement/behavioral (DonD). DA dominates fixed-status (CMV). Profile >> average in all 4 studies (g-PSQ: 0.515–0.622). See journal §24–25, §27 for theoretical analysis.

---

## Factor Structure

### v2 (separated-llm only, N=1,970)
- g-factor eigenvalue: **6.727 (67.3% variance)** — up from 4.844 (48.4%) in v1
- KMO = **0.902** ("Superb") — up from 0.819
- Parallel analysis retains **1 factor only** (was 2 in v1)
- 5-factor structure **collapsed**: Factor 1 absorbs 8/10 dims; only CO, ED, AD separate weakly
- g-factor loadings all >0.66: TC (0.930), DA (0.914), CC (0.864), RC (0.854)
- Mean inter-dim |r| = **0.632** (up from 0.417 in mixed data)
- **Integer-only scoring bias discovered**: LLM uses 11 bins (integers 0-10), not continuous scale
- g-factor may be partly inflated by score-5 concentration (24-61% across dims)

### v1 (retained for comparison, N=2,359 mixed)
- 5 clusters: Hostility/Threat (HI,TE,CC), Relational Contract (CO,TC), Internal Resources (RB,RC,DA), Power Dynamics (AD), Stress/Energy (ED)
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

- **Integer-only scoring bias** (MITIGATED): 0-100 percentage scale implemented (`--pct` flag). Pilot confirms: non-integer 2.1%→77.8%, exact-5 41.3%→7.2%. All future batches should use `--pct`. Existing 22,771 separated-llm scores remain integer-only.
- **CO score-5 concentration** (IMPROVED): Pilot shows CO exact-5 drops from 60.8% to 10.0% with percentage scale. Structural mitigation: score-concentration cap + `--pct` scoring.
- **AD data provenance**: 70.4% LLM-generated effective training signal (middle of pack, not outlier). Needs expert validation to confirm.
- **DA construct validity**: max promax loading 0.332 (below 0.35 threshold). 49% of scores are exact 5.0. Requires human expert validation.
- **g-factor inflation uncertainty**: g-factor eigenvalue 6.727 (67.3%) may be partly artifactual from integer-only bias. Resolution requires pct-scored data + re-run EFA. Factor analysis v3 yielded no new info (same N=1,970, AD is bottleneck).

---

## Key Design Constraint

**The PSQ construct definitions and scoring rubric (psq-definition.md) are externally authored and immutable.** All 10 dimension definitions, score anchors, and the overall PSQ formula are treated as fixed inputs to the measurement system. When score distributions show concentration or other statistical issues, the response is to improve data sourcing and model training — never to revise the rubric itself.

---

## What's Next

See `TODO.md` for full task list. Immediate priorities:

1. **Production pct-scored batch** — Extract 200+ texts with `--pct`, score using separated protocol (1 dim per session). This is the critical path to resolving g-factor inflation question.
2. **Bifactor v19b evaluation** — Training in progress. Evaluate g-PSQ head quality on held-out.
3. **Factor analysis with pct data** — After production pct batch is scored and ingested, re-run EFA to test if g-factor eigenvalue drops.
4. **Expert validation** — protocol designed, recruitment not started

### Completed This Session

- v19 promoted to production (ONNX re-exported)
- `--pct` flag implemented in `label_separated.py`
- Percentage scoring pilot: 50 texts × 10 dims, confirms 37× improvement in non-integer scores
- Factor analysis v3: same N=1,970 (AD bottleneck), no new findings
- Bifactor v19b training launched

---

*Last updated: 2026-02-28 (pct pilot + bifactor cycle)*
