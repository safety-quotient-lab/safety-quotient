# Data Provenance Card

All datasets used in PSQ student model training, with licenses, sizes, quality assessments, and audit status.

## Source Datasets

### Tier 1: Primary Proxy Sources (in composite from v1)

| Dataset | License | Records | PSQ Dimensions | Quality | Audit Status |
|---|---|---|---|---|---|
| **Berkeley Measuring Hate Speech** | CC-BY 4.0 | 2,000 | threat_exposure, hostility_index | Good (IRT-derived scores, multi-annotator) | Clean |
| **Civil Comments** (Google/Jigsaw) | CC0 1.0 | 2,000 | hostility_index | Good (crowd-annotated, large scale) | threat_exposure REMOVED |
| **GoEmotions** (Google) | Apache 2.0 | 2,000 | regulatory_capacity, resilience_baseline, trust_conditions, cooling_capacity, hostility_index, energy_dissipation, defensive_architecture | Moderate (27 emotion labels, proxy gap to PSQ constructs) | Clean |
| **UCC** (Unhealthy Conversations) | CC-BY 4.0 | 1,949 | authority_dynamics, trust_conditions, cooling_capacity, hostility_index, defensive_architecture | Mixed | Partially removed |

**UCC audit notes:**
- `condescending → authority_dynamics`: retained but confidence halved (0.25x) — narrow construct, +2.8 bias
- `generalisation_unfair → contractual_clarity`: **REMOVED** in v3b — bias -2.32, negative r, actively harmful
- Other mappings (hostile, dismissive, sarcastic): retained with reduced confidence

**Civil Comments audit notes:**
- `threat + severe_toxicity → threat_exposure`: **REMOVED** in v10 — 95% of records (1,754/1,853) scored 9-10 ("perfectly safe") because proxy maps "not threatening from author's POV" to "safe for target". Texts describing harassment, rape, ethnic cleansing got score 10.0. Actively poisoned threat_exposure training.
- `toxicity + insult → hostility_index`: retained — works correctly (toxicity is a valid hostility proxy)

### Tier 2: Expanded Sources (added v2d/v3)

| Dataset | License | Records | PSQ Dimensions | Quality | Audit Status |
|---|---|---|---|---|---|
| **Dreaddit** (Reddit stress) | CC-BY-SA 4.0 | 2,000 | energy_dissipation | Good (binary stress, clean mapping) | Clean |
| **ESConv** (Emotional Support) | MIT | 1,300 | regulatory_capacity | Good (strategy labels) | Clean |
| **Empathetic Dialogues** (Facebook) | CC-BY 4.0 | 2,000 | resilience_baseline, regulatory_capacity | Moderate (emotion labels as proxy) | Clean |
| **CaSiNo** (negotiation) | CC-BY 4.0 | 396 | contractual_clarity | Good (negotiation strategies) | Clean |
| **Stanford Politeness** | CC-BY 4.0 | 2,000 | authority_dynamics | Moderate (politeness != power dynamics) | De-weighted |
| **ProsocialDialog** (Allen AI) | CC-BY 4.0 | 1,998 | defensive_architecture | Moderate (safety labels, not defense mechanisms) | Clean |

**Politeness audit notes:** Confidence halved (0.15-0.30) — politeness is a narrow proxy for authority dynamics. Compressed range (std=0.73 vs LLM 1.72), over-predicts +0.90-1.45.

### Removed Sources

| Dataset | License | Records | Was mapping | Reason for removal | Removed in |
|---|---|---|---|---|---|
| **Diplomacy** (FAIR) | MIT | 525 | trust_conditions | Sender intent != textual trust. Deceptive-but-believed records are unlearnable from text. MAE 2.405 (worst source). | v3b |
| **UCC** contractual proxy | CC-BY 4.0 | ~2,000 | contractual_clarity | Bias -2.32, negative r (-0.10), taught wrong signal. | v3b |

### Tier 3: LLM Gold-Standard Labels (train-llm.jsonl)

**API labels (1,353 records):** Claude LLM teacher labels from unlabeled pool. These were originally 1,376 but 23 internal duplicates removed in v9. Covers all 10 dimensions, 50-450 per dim depending on composite coverage.

**Synthetic labels (1,905 records as of 2026-02-27):** Targeted generation for signal-starved dimensions. All written by Claude, scored on single primary dimension.

| Batch | File | Count | Target dimension | Date |
|---|---|---|---|---|
| ad_1–ad_3 | psq_synthetic_ad_1..3.json | ~20 | authority_dynamics | 2026-02-27 |
| ad_4 | psq_synthetic_ad_4.json | 107 | authority_dynamics (workplace) | 2026-02-27 |
| ad_5 | psq_synthetic_ad_5.json | 87 | authority_dynamics (education/family) | 2026-02-27 |
| ad_6 | psq_synthetic_ad_6.json | 87 | authority_dynamics (institutional/government) | 2026-02-27 |
| ad_7 | psq_synthetic_ad_7.json | 100 | authority_dynamics (community/online) | 2026-02-27 |
| ad_8 | psq_synthetic_ad_8.json | 305 | authority_dynamics (all contexts, balanced) | 2026-02-27 |
| co_2 | psq_synthetic_co_2.json | 185 | contractual_clarity | 2026-02-27 |
| co_3 | psq_synthetic_co_3.json | 183 | contractual_clarity | 2026-02-27 |
| te_2 | psq_synthetic_te_2.json | 200 | threat_exposure (low-score heavy) | 2026-02-27 |
| ed_2 | psq_synthetic_ed_2.json | 150 | energy_dissipation | 2026-02-27 |
| da_2 | psq_synthetic_da_2.json | 191 | defensive_architecture | 2026-02-27 |
| tc_2 | psq_synthetic_tc_2.json | 30 | trust_conditions | 2026-02-27 |
| cc_2 | psq_synthetic_cc_2.json | 30 | cooling_capacity | 2026-02-27 |
| rb_2 | psq_synthetic_rb_2.json | 20 | resilience_baseline | 2026-02-27 |
| rc_2 | psq_synthetic_rc_2.json | 20 | regulatory_capacity | 2026-02-27 |

**Relabeled records (941 records as of 2026-02-27):** Existing composite texts relabeled by Claude on specific dimensions where proxy labels were broken.

| Batch | File | Count | Target dimension | Date |
|---|---|---|---|---|
| relabel_thre | psq_relabeled_thre.json | 250 | threat_exposure | 2026-02-27 |
| relabel_ener | psq_relabeled_ener.json | 250 | energy_dissipation | 2026-02-27 |
| relabel_regu | psq_relabeled_regu.json | 250 | regulatory_capacity | 2026-02-27 |
| relabel_defe | psq_relabeled_defe.json | 250 | defensive_architecture | 2026-02-27 |

**Design notes:**
- Auth_dynamics zeroed in composite for politeness/UCC sources (3,515 records) — too noisy for proxy labeling
- Secondary dimension scores removed from synthetic records where authority_dynamics is not primary — prevents neutral-band noise
- Dedup in distill.py: when same text in both composite and LLM, keeps LLM version (5x weight)

### Tier 4: Held-Out Real-World Test Set

| File | Records | Source | Date | Purpose |
|---|---|---|---|---|
| data/held-out-test.jsonl | 100 | unlabeled-pool (20 per source) | 2026-02-27 | Generalization test, no composite/LLM overlap |

Originally assembled from `/tmp/held_out_labeled_a.json` (50 texts) and `/tmp/held_out_labeled_b.json` (50 texts), both LLM-labeled jointly (all 10 dims per call). Abbreviation keys mapped to full dimension names by `scripts/assemble_held_out.py`.

**Re-scored with separated labeling (2026-02-27):** All 100 texts re-scored one dimension at a time (10 separate LLM calls per text) to eliminate halo effect. Original joint-scored file archived as `data/held-out-test-joint.jsonl`. Separated scoring performed via `scripts/label_separated.py` extract/ingest/assemble workflow. All 100 records now have complete 10-dimension coverage with `teacher: "separated-llm"`. Halo validation confirmed mean inter-dimension |r| dropped from 0.766 (joint) to 0.656 (separated).

### Tier 5: Separated-LLM Labeling Batches (training data)

| Batch | File | Texts | Dims | Date | Notes |
|---|---|---|---|---|---|
| Weak dims (te/rc/co) | `data/labeling-batch-weak-dims.jsonl` | 200 | 3 | 2026-02-27 | First separated batch; te/rc/co only |
| All dims expansion | `data/labeling-batch-weak-dims.jsonl` | 200 | all 10 | 2026-02-27 | Extended to all 10 dims (same 200 texts) |
| RC focus batch | `data/labeling-batch-rc.jsonl` | 150 | all 10 | 2026-02-27 | Targeting rc regression; diverse sources |
| AD focus batch | `data/labeling-batch-ad.jsonl` | 300 | all 10 | 2026-02-27 | All dims scored and ingested |
| CO focus batch | `data/labeling-batch-co.jsonl` | 200 | 10/10 | 2026-02-28 | Complete. Keyword-filtered for co-relevant content (≥2 CO keywords). Integer scale. Assembled → `labeling-batch-co-scored.jsonl`, ingested. |
| RB focus batch | `data/labeling-batch-rb.jsonl` | 200 | 10/10 | 2026-02-27 | Complete. Keyword-filtered for rb-relevant content (resilience, coping, recovery). |
| CC focus batch | `data/labeling-batch-cc.jsonl` | 200 | 10/10 | 2026-02-27 | Complete. Keyword-filtered for cc-relevant content (calm, de-escalation, anger, composure). |
| TE focus batch | `data/labeling-batch-te.jsonl` | 200 | 10/10 | 2026-02-27 | Complete. Keyword-filtered for te-relevant content (threat, violence, abuse, harm). TE score mean=3.17, good low-end variance. |
| Broad spectrum | `data/labeling-batch-broad.jsonl` | 300 | 10/10 | 2026-02-28 | Complete. 150 random + 100 single-dim keyword + 50 multi-dim keyword. Best distributions: TE 18.3%, ED 25.7% score-5. |
| Pct scale batch | `data/labeling-batch-pct-200.jsonl` | 200 | 10/10 | 2026-02-28 | Complete. 0-100 percentage scale (÷10 on ingest). 86.2% non-integer, 4.8% exact-5, 35 unique values. Sources: empathetic_dialogues 73, berkeley 56, prosocial 50, dreaddit 16, esconv 5. |
| CO #2 batch | `data/labeling-batch-co2.jsonl` | 200 | 10/10 | 2026-02-28 | Complete. CO-keyword filtered texts from unlabeled pool. CO mean=4.36. Drove v21 (+0.030 held-out). |
| Middle-g batch | `data/labeling-batch-midg.jsonl` | 250 | 10/10 | 2026-02-28 | Complete. g∈[3,4.5)∪[5.5,7] from unlabeled pool (model-scored selection). Sources: dreaddit 80, berkeley 70, prosocial 50, empathetic_dialogues 30, esconv 20. TE 22.8%, HI 34.4% score-5 (good variance). |

**Total separated-llm training labels:** 29,271 scores (across 21,877 texts in DB).

## Composite Summary (post-midg batch, 2026-02-28)

| Metric | Value |
|---|---|
| Total texts in DB | 21,877 |
| Total scores in DB | 82,861 |
| Composite proxy records | 40,487 (auth zeroed for 3,515 politeness/UCC, CC threat_exposure REMOVED) |
| Joint-LLM records | 12,257 |
| Separated-LLM records | 29,271 |
| Synthetic records | 846 |
| Held-out test records | 100 (separate, not in training) |
| Train / Val / Test | 17,458 / 2,122 / 2,203 (hash-based text split) |
| Source datasets | 11 active (2 removed, 1 partially removed) |
| Dimensions covered | 10/10 |

## Data Pipeline

```
build_composite_ground_truth.py → composite-ground-truth.jsonl (17,643)
  ↑ config-driven via data/dataset_mappings.json (11 source datasets)
label_separated.py              → separated-llm scores (per-dim, halo-free)
  extract → score in Claude Code → ingest → assemble → migrate.py --ingest
migrate.py                      → data/psq.db (canonical store)
  --ingest JSONL for incremental additions
build_unlabeled_pool.py         → unlabeled-pool.jsonl (17,451 texts)
  (raw dataset texts not in composite)
                                   ↓
distill.py reads from psq.db training_data view (auto-dedup, LLM 5x weight)
```

## Data Quality Audit (v4 composite, 2026-02-26)

### Issues Found

| Issue | Severity | Dimension | Detail |
|---|---|---|---|
| Ceiling effect | CRITICAL | threat_exposure | 34.7% of scores are exactly 10.0; 52.5% in [8-10]. Model will over-predict high safety. |
| Range compression | HIGH | energy_dissipation | Max score is 6.8 — model can never learn to predict 7-10. |
| Under-representation | HIGH | contractual_clarity | Only 796 labels (10x fewer than largest dim). |
| Duplicate texts | MEDIUM | all | 1,873 texts appear 2-4x. 247 dimension-level conflicts >2.0 pts (worst: 7.0 spread on threat_exposure between berkeley and claude_code). |
| Stale records | MEDIUM | trust_conditions | 15 diplomacy records remain despite source being "removed". |
| Narrow source diversity | LOW | per-dim | Most dims trained on 1-2 source datasets. Only goemotions covers all 10. |

### Recommendations for v5

1. **Threat exposure**: Resample to reduce 10.0 ceiling, or add more low-threat examples
2. **Energy dissipation**: Add LLM-labeled examples scoring 7-10 to extend the range
3. **Duplicates**: Deduplicate by text, keep highest-confidence label or average scores
4. **Diplomacy**: Remove remaining 15 records from composite
5. **Source diversity**: For weakest dims, source additional proxy datasets or increase LLM label budget

### Distribution Summary

| Dimension | Records | Mean Score | Score Range | Skew |
|---|---|---|---|---|
| hostility_index | 7,999 | 6.38 | [0.0, 10.0] | -0.30 |
| threat_exposure | 6,050 | 7.53 | [0.0, 10.0] | -0.48 |
| regulatory_capacity | 5,350 | 4.83 | [1.0, 8.0] | +0.19 |
| energy_dissipation | 4,050 | 4.39 | [0.0, 6.8] | -1.01 |
| resilience_baseline | 3,982 | 5.03 | [1.0, 8.5] | 0.00 |
| trust_conditions | 3,999 | 6.21 | [0.5, 10.0] | -0.15 |
| cooling_capacity | 3,999 | 5.96 | [0.5, 10.0] | -0.02 |
| defensive_architecture | 3,513 | 4.08 | [0.5, 9.0] | -0.11 |
| authority_dynamics | 4,009 | 5.52 | [0.5, 10.0] | +0.08 |
| contractual_clarity | 796 | 5.77 | [1.0, 10.0] | -0.84 |

## Licensing

All source datasets use permissive licenses (CC-BY, CC0, Apache 2.0, MIT). The PSQ student model trained on this data can be distributed under the project's dual license (research-free / commercial-paid).
