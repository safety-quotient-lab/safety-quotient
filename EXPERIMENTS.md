# PSQ Training Experiment Log

Version-by-version record of every training run, with hyperparameters, data changes, and results. For reproducibility.

## How to Read This Table

- **test_r**: average Pearson r across 10 dimensions on the test split (seen-distribution data)
- **held-out_r**: average Pearson r on 100 independent real-world texts with LLM labels
- **Data changes** describes what changed in the training data relative to the previous version
- **Config changes** describes hyperparameter or architecture changes

---

## Training Runs

| Version | Date | Architecture | Epochs | LR | Batch | test_r | held-out_r | Data changes | Config changes |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 2026-02-26 | DeBERTa-v3-small | 10 | 2e-5 | 8 (accum 4) | 0.492 | — | 4 datasets (Berkeley, Civil Comments, GoEmotions, UCC) ~8K composite + 365 LLM | Baseline. MSE loss, flat confidence. |
| v2a | 2026-02-26 | DeBERTa-v3-small | 10 | 2e-5 | 8 (accum 4) | ~0.50 | — | +4 new datasets (Dreaddit, ESConv, CaSiNo, Politeness) ~12K composite | Fixed sample weighting bug (LLM samples were getting weight=0) |
| v2b | 2026-02-26 | DeBERTa-v3-small | 10 | 2e-5 | 8 (accum 4) | ~0.52 | — | +LLM labeled samples | Confidence-weighted loss (`conf^1.0 * MSE`) |
| v2d | 2026-02-26 | DeBERTa-v3-small | 10 | 2e-5 | 8 (accum 4) | 0.585 | — | Dedup, ProsocialDialog added, ~15K composite + 1,353 LLM | Hash-based splits, LLM 3x weight |
| v3 | 2026-02-26 | DeBERTa-v3-small | 10 | 2e-5 | 8 (accum 4) | ~0.55 | — | Diplomacy REMOVED, UCC contractual REMOVED | — |
| v3b | 2026-02-26 | DeBERTa-v3-small | 10 | 2e-5 | 8 (accum 4) | ~0.54 | — | Auth_dynamics zeroed for politeness/UCC (3,515 records) | — |
| v4 | 2026-02-26 | DeBERTa-v3-small | 10 | 2e-5 | 8 (accum 4) | ~0.52 | — | Same | Squared conf weighting (`conf^2.0`), two-phase conf warmup |
| v5 | 2026-02-27 | DistilBERT | 10 | 2e-5 | 16 | ~0.49 | — | Same | Architecture switch. Grad accum 2. |
| v6 | 2026-02-27 | DistilBERT | 10 | 2e-5 | 16 | ~0.50 | — | +synthetic ad_1..ad_3 (~20 auth texts) | — |
| v7 | 2026-02-27 | DistilBERT | 10 | 2e-5 | 16 | ~0.51 | — | +synthetic ad_4..ad_7 (~381 auth texts) | — |
| v8 | 2026-02-27 | DistilBERT | 15 | 2e-5 | 16 | 0.505 | 0.367 | +synthetic co_2, co_3 (368 contractual) | 15 epochs, patience 5 |
| v9 | 2026-02-27 | DistilBERT | 15 | 2e-5 | 16 | 0.515 | 0.385 | +synthetic tc_2, cc_2, rb_2, rc_2 (100 misc) | LLM 5x weight (up from 3x) |
| v10 | 2026-02-27 | DistilBERT | 15 | 2e-5 | 16 | 0.534 | 0.425 | +relabeled thre/ener/regu/defe (1,000 texts) | — |
| v11 | 2026-02-27 | DistilBERT | — | 2e-5 | 16 | — | — | +synthetic te_2, ed_2, da_2 (541 texts) | Killed early — ad_8 not ready |
| v12 | 2026-02-27 | DistilBERT | — | 2e-5 | 16 | — | — | +synthetic ad_8 (305 auth) | Killed — trained on old composite (CC threat still in) |
| **v13** | **2026-02-27** | **DistilBERT** | **8** | **2e-5** | **16** | **0.553** | **0.428** | CC threat_exposure REMOVED from composite, all synthetic + relabeled | Fixed composite (17,643 records), 4,199 LLM |
| **v14** | **2026-02-27** | **DistilBERT** | **8** | **2e-5** | **32** | **0.544** | **0.482** | +2,000 separated-llm labels (all 10 dims, 200 texts × 10 dims). DB mode. | Same hyperparams as v13. `--out models/psq-v14`. |
| **v15** | **2026-02-27** | **DistilBERT** | **7** | **2e-5** | **32** | **0.536** | **0.495** | +4,500 separated-llm (300 AD + 150 RC batches × 10 dims). DB: 63,361 scores. | Same hyperparams. `--out models/psq-v15`. |

| **v16** | **2026-02-27** | **DistilBERT** | **6** | **2e-5** | **32** | **0.529** | **0.561** | +6,000 separated-llm (CO 200 + RB 200 + CC 200 batches × 10 dims). DB: 20,727 texts, 69,361 scores. | Score-concentration cap (>30% → weight 1.5). `--out models/psq-v16`. |

**Notes on v16:** Best held-out ever (0.561, +0.066 vs v15). RC massive recovery (0.285→0.563). CO recovered (0.388→0.534). TE regressed (0.476→0.347) — correlation artifact (MAE improved). Early stopped at epoch 9, best at epoch 6.

| v17 | 2026-02-28 | DistilBERT | 6 | 2e-5 | 32 | 0.503 | 0.563 | +2,000 separated-llm (TE 200 batch × 10 dims). DB: 20,927 texts, 71,361 scores. | Same as v16. `--out models/psq-v17`. |

**Notes on v17:** Essentially flat vs v16 (held-out 0.563 vs 0.561). Different dim trade-offs: gains RC(+0.073), RB(+0.069), TC(+0.096), DA(+0.063); losses HI(-0.058), ED(-0.034), CC(-0.031). Not promoted — v16 remains in production. Early stopped epoch 9, best epoch 6.

| **v18** | **2026-02-28** | **DistilBERT** | **10** | **2e-5** | **32** | **0.525** | **0.568** | +2,000 separated-llm (CO 200 batch × 10 dims). DB: 21,127 texts, 73,361 scores. | Same as v16. `--out models/psq-v18`. |

**Notes on v18:** New held-out best (0.568, +0.007 vs v16). RC massive jump (0.563→0.679), RB strong gain (0.563→0.651), TC improved (0.575→0.620). CO held-out flat (0.534→0.533) despite huge test improvement (CO test=0.766). HI/AD/ED/CC slightly down — typical dimension trade-off. Ran all 10 epochs (no early stop), 3360s total. **Promoted to production.**

| **v19** | **2026-02-28** | **DistilBERT** | **7** | **2e-5** | **32** | **0.509** | **0.600** | +3,000 separated-llm (broad-spectrum 300 texts × 10 dims). DB: 21,427 texts, 76,361 scores. | Same as v16. Score-concentration cap on all 10 dims. `--out models/psq-v19`. |

**Notes on v19:** NEW BEST held-out (0.600, +0.032 vs v18). 10 epochs, early stopped at epoch 7 (best at epoch 4). Broad-spectrum batch (150 random + 100 single-dim + 50 multi-dim keyword-filtered) drove broad improvement across weakest dims. Key wins: TE +0.125 (0.370→0.495, biggest single-dim improvement), ED +0.087 (0.562→0.649), AD +0.058 (0.599→0.657), DA +0.050 (0.488→0.538). Key losses: RB -0.027 (0.651→0.624), CO -0.020 (0.533→0.513), CC -0.016 (0.618→0.602). **Not yet promoted — awaiting ONNX re-export.**

| v19b | 2026-02-28 | DistilBERT | 7 | 2e-5 | 32 | 0.502 | — | Same as v19. Bifactor experiment only. | `--bifactor`: 11th head (g-PSQ, sigmoid×10, loss weight 1.0). g_r=0.594. `--out models/psq-v19b`. |

**Notes on v19b:** Bifactor experiment — added 11th output head predicting g-PSQ (mean of 10 dim scores). g-head learned well (r=0.594) but per-dim average dropped (0.509→0.502). CC benefited (+0.150) but RC/AD/CC/DA/TE all lost. Capacity competition: 11 heads share 384-dim projection. Early stopped epoch 7 (best@4), same as v19. **Not promoted — bifactor architecture is net-negative for per-dim prediction at DistilBERT scale.**

| v20 | 2026-02-28 | DistilBERT | 10 | 2e-5 | 32 | 0.501 | 0.600 | +200 pct-scored texts (2K separated-llm, 86.2% non-integer). | `--out models/psq-v20`. No early stop (all 10 epochs). |

**Notes on v20:** Pct-scored data had no measurable effect (held-out_r 0.600 = identical to v19). Dim-level shifts within noise (CO +0.024, CC +0.023 vs ED -0.034, TE -0.028). **Not promoted.** Pct scoring retracted as labeling format — FA v3 showed dimension collapse (eigenvalue 9.41 = 94.1% shared variance).

| **v21** | **2026-02-28** | **DistilBERT** | **6** | **2e-5** | **32** | **0.504** | **0.630** | +2,000 separated-llm (CO batch: 200 CO-keyword texts × 10 dims). DB: 21,627 texts, 80,361 scores, 26,771 separated-llm. | Score-concentration cap. `--out models/psq-v21`. Early stop epoch 9, best epoch 6. |

**Notes on v21:** NEW BEST held-out (0.630, +0.030 vs v19, +0.069 vs v16). RC=0.729 (new ceiling dim), CC=0.687, AD=0.674, TC=0.674. CO batch helped non-CO dims most: HI +0.087, CC +0.085, TC +0.038 vs v19. CO itself +0.042. TE flat (0.492), DA still weak (0.566). Val-held-out gap 0.126 (held-out quality > val split). **Promoted to production.** ONNX re-exported.

| **v22a** | **2026-02-28** | **DistilBERT** | **5** | **2e-5** | **32** | **0.457** | **0.682** | Same data as v21 (82,861 scores). `--drop-proxy-dims` removes 9,450 composite-proxy rows for TE, TC, CC, AD. | Proxy removal ablation. `--out models/psq-v22a`. 10 epochs, best@5. |

**Notes on v22a:** **NEW BEST held-out (0.682, +0.052 vs v21).** Test-split paradox: test_r drops (0.504→0.457) because test split contains proxy labels as ground truth, but held-out (independent LLM labels) massively improves. TE transformed (0.492→0.805, +0.313 — largest single-dim improvement ever). 9/10 dims improved on held-out; only CC regressed (-0.051). Proxy data for TE was adversarial (r=-0.260); removal unleashed separated-LLM signal. **Promotion candidate pending v22b/v22c comparison.**

| v22b | 2026-02-28 | DistilBERT | — | 2e-5 | 32 | — | 0.578 | Same data as v21 + midg batch (85,361 scores). No proxy removal. | Middle-g enrichment ablation only. `--out models/psq-v22b`. |

**Notes on v22b:** **WORSE than v21 (held-out_r=0.578, -0.052 vs v21, -0.104 vs v22a).** All 10 dims regressed vs v21. Midg data alone, without proxy removal, does not help — proxy noise overwhelms the midg signal. Confirms that proxy removal (v22a) is the dominant intervention; middle-g enrichment is neutral-to-negative when proxy noise is retained. Data quality > data quantity: 250 high-quality texts added to a noisy dataset produce no net gain.

| v22c | 2026-02-28 | DistilBERT | 9 (early stop, best@6) | 2e-5 | 32 | 0.431 | 0.638 | Same as v22a + 200-text test-clean batch (test split LLM labels). Removed 12,409 proxy rows for TE/TC/CC/AD/ED. | `--drop-proxy-dims --curriculum --out models/psq-v22c`. Curriculum: Phase 1 (LLM-only, epochs 1-3), Phase 2 (+proxy, epochs 4-9). |

**Notes on v22c:** **Curriculum learning adds NO benefit over proxy removal alone (v22c 0.638 < v22a 0.682, Δ=−0.044). Curriculum REJECTED.** All 10 dims worse than v22a. Largest regressions: HI (−0.114), TE (−0.091), DA (−0.070), CC (−0.055). Worst test_r of the v22 series (0.431) due to proxy-clean test split from test-clean batch ingestion. The complete 2×2 ablation (v22a/v22b/v22c) confirms proxy removal alone is the sufficient and dominant intervention. v22a remains the production candidate.

## Key Hyperparameters (v21, current)

```
model_name:          distilbert-base-uncased
max_length:          128
learning_rate:       2e-5
batch_size:          16
grad_accumulation:   2
effective_batch:     32
epochs:              15 (early stop patience 5, stopped at 8)
loss:                conf^2.0 * sample_weight * MSE(score) + 0.25 * MSE(confidence)
llm_weight:          5.0
composite_weight:    1.5
conf_mode:           two-phase (first 2 epochs: fixed 0.5, then use model confidence)
projection:          768 → 384 (shared) → 10 heads × (384 → 2: score + confidence)
optimizer:           AdamW (weight_decay=0.01)
scheduler:           linear warmup (10% of steps) + linear decay
```

## Held-Out Results by Dimension

### v22a (new best, 2026-02-28)

Proxy removal for TE, TC, CC, AD. Score-concentration cap active. 100 real-world held-out texts with separated-LLM labels.

| Dimension | r | Δ vs v21 | Notes |
|---|---|---|---|
| threat_exposure | +0.805 | **+0.313** | **Largest single-dim improvement ever** — adversarial proxy removed |
| regulatory_capacity | +0.756 | +0.027 | Continued improvement |
| cooling_capacity | +0.719 | +0.032 | Improved |
| hostility_index | +0.719 | +0.061 | Strong gain |
| energy_dissipation | +0.712 | +0.076 | Strong gain |
| trust_conditions | +0.679 | +0.005 | Flat (proxy dropped, net-neutral) |
| authority_dynamics | +0.679 | +0.005 | Flat (proxy dropped, net-neutral) |
| resilience_baseline | +0.640 | +0.040 | Improved |
| defensive_architecture | +0.607 | +0.041 | Improved |
| contractual_clarity | +0.504 | -0.051 | **Only regression** — now weakest dim |
| **Average** | **+0.682** | **+0.052** | **New best. +0.082 vs v19, +0.121 vs v16, +0.280 vs v13** |

**Test-split paradox:** test_r = 0.457 (regression from v21's 0.504) because test split uses proxy labels as ground truth. Removing proxy training data mechanically lowers test_r even though real-world prediction improves. Held-out_r is the reliable metric.

### v22b (midg enrichment only, 2026-02-28)

Middle-g batch (250 texts × 10 dims). No proxy removal. Score-concentration cap active. 100 real-world held-out texts.

| Dimension | r | Δ vs v21 | Notes |
|---|---|---|---|
| regulatory_capacity | — | — | All dims regressed vs v21 |
| cooling_capacity | — | — | |
| hostility_index | — | — | |
| authority_dynamics | — | — | |
| trust_conditions | — | — | |
| energy_dissipation | — | — | |
| resilience_baseline | — | — | |
| defensive_architecture | — | — | |
| contractual_clarity | — | — | |
| threat_exposure | — | — | |
| **Average** | **0.578** | **-0.052** | **Worse than v21 and v22a. Proxy noise overwhelms midg signal.** |

**Key conclusion:** Proxy removal (v22a: +0.052) is the dominant intervention. Adding 250 midg texts without proxy removal (v22b: -0.052) not only fails to help — it regresses. Data quality > data quantity: the midg signal is neutralized by 9,450 adversarial/noisy proxy rows that remain in training.

### v22c (proxy removal + curriculum, 2026-02-28)

`--drop-proxy-dims --curriculum`. Phase 1: 5,308 LLM records (epochs 1–3). Phase 2: +10,383 proxy records (15,691 total, epochs 4–9). Early stopping at epoch 9, best at epoch 6 (val_r=0.4478). Dropped 12,409 proxy rows for TE/TC/CC/AD/ED.

| Dimension | r | Δ vs v22a | Notes |
|---|---|---|---|
| regulatory_capacity | +0.728 | −0.028 | |
| energy_dissipation | +0.707 | −0.005 | Near-flat |
| trust_conditions | +0.671 | −0.008 | Near-flat |
| cooling_capacity | +0.664 | −0.055 | |
| threat_exposure | +0.714 | −0.091 | |
| authority_dynamics | +0.650 | −0.029 | |
| resilience_baseline | +0.614 | −0.026 | |
| hostility_index | +0.605 | **−0.114** | Largest regression |
| defensive_architecture | +0.537 | −0.070 | |
| contractual_clarity | +0.487 | −0.017 | Still weakest dim |
| **Average** | **0.638** | **−0.044** | **All 10 dims regressed vs v22a** |

**Curriculum learning REJECTED. v22a remains the best and production candidate.**

### v22a ablation comparison

| Run | Intervention | held-out_r | Δ vs v21 | Verdict |
|---|---|---|---|---|
| v21 | Baseline | 0.630 | — | Production baseline |
| v22a | Proxy removal only | **0.682** | **+0.052** | **Best model. Dominant intervention.** |
| v22b | Midg data only (no proxy removal) | 0.578 | -0.052 | Neutral-to-negative. Proxy noise overwhelms midg signal. |
| v22c | Proxy removal + curriculum | 0.638 | -0.044 | **Curriculum REJECTED.** Worse than v22a on all 10 dims. |

### v21 (production, 2026-02-28)

CO batch #2 (200 CO-keyword texts × 10 dims). Score-concentration cap active. 100 real-world held-out texts.

| Dimension | r | Δ vs v19 | Notes |
|---|---|---|---|
| regulatory_capacity | +0.729 | +0.019 | Ceiling dim |
| cooling_capacity | +0.687 | +0.085 | Strong gain |
| authority_dynamics | +0.674 | +0.017 | Improved |
| trust_conditions | +0.674 | +0.038 | Improved |
| hostility_index | +0.658 | +0.087 | Strong gain |
| energy_dissipation | +0.636 | -0.013 | Slight regression |
| resilience_baseline | +0.600 | -0.024 | Slight regression |
| defensive_architecture | +0.566 | +0.028 | Improved |
| contractual_clarity | +0.555 | +0.042 | Improved |
| threat_exposure | +0.492 | -0.003 | Flat — weakest dim |
| **Average** | **+0.630** | **+0.030** | **+0.030 vs v19, +0.069 vs v16** |

### v19 (previous best, 2026-02-28)

Score-concentration cap active. Broad-spectrum batch (300 texts × 10 dims = 3,000 new separated-llm scores). 100 real-world held-out texts.

| Dimension | r | Δ vs v18 | Notes |
|---|---|---|---|
| regulatory_capacity | +0.710 | +0.031 | **best** — continued improvement |
| authority_dynamics | +0.657 | +0.058 | strong gain |
| energy_dissipation | +0.649 | +0.087 | strong gain |
| trust_conditions | +0.636 | +0.016 | improved |
| resilience_baseline | +0.624 | -0.027 | slight regression |
| cooling_capacity | +0.602 | -0.016 | good |
| hostility_index | +0.571 | +0.014 | good |
| defensive_architecture | +0.538 | +0.050 | improved |
| contractual_clarity | +0.513 | -0.020 | slight regression |
| threat_exposure | +0.495 | **+0.125** | **massive recovery** — broad-spectrum data |
| **Average** | **+0.600** | | **+0.032 vs v18, +0.039 vs v16, +0.172 vs v13** |

### v16 (previous production, 2026-02-27)

Score-concentration cap active. CO/RB/CC batches (600 texts × 10 dims = 6,000 new separated-llm scores). 100 real-world held-out texts.

| Dimension | r | MSE | n | Notes |
|---|---|---|---|---|
| cooling_capacity | +0.643 | 1.797 | 99 | Stable (-0.010) |
| authority_dynamics | +0.625 | 1.424 | 93 | Improved (+0.052) |
| hostility_index | +0.604 | 3.821 | 99 | Strong improvement (+0.066) |
| energy_dissipation | +0.592 | 1.737 | 99 | Improved (+0.081) |
| trust_conditions | +0.585 | 3.476 | 99 | Improved (+0.021) |
| resilience_baseline | +0.576 | 1.266 | 99 | Improved (+0.046) |
| regulatory_capacity | +0.563 | 1.716 | 99 | **Massive recovery (+0.278)** |
| defensive_architecture | +0.539 | 1.756 | 88 | Improved (+0.016) |
| contractual_clarity | +0.534 | 1.692 | 89 | **Recovered (+0.146)** |
| threat_exposure | +0.347 | 6.035 | 99 | **Regressed (-0.129)** — needs investigation |
| **Average** | **+0.561** | | | **+0.066 vs v15, +0.133 vs v13** |

### v15 (previous best, 2026-02-27)

Re-scored with separated LLM calls (one dimension per call). 100 real-world texts, complete 10-dimension coverage.

| Dimension | r | MSE | n | Notes |
|---|---|---|---|---|
| cooling_capacity | +0.653 | 2.464 | 99 | Best — unchanged from v14 |
| authority_dynamics | +0.573 | 1.818 | 93 | Massive jump (+0.166 from v14) — AD batch |
| trust_conditions | +0.564 | 4.011 | 99 | Good — stable |
| hostility_index | +0.538 | 4.134 | 99 | Good — improved (+0.050) |
| defensive_architecture | +0.523 | 1.921 | 88 | Good — improved |
| energy_dissipation | +0.511 | 1.618 | 99 | Good — slight regression |
| resilience_baseline | +0.507 | 1.370 | 99 | Good — improved (+0.063) |
| threat_exposure | +0.410 | 6.603 | 99 | Moderate — regressed from v14 |
| contractual_clarity | +0.388 | 1.587 | 89 | Moderate — regressed (-0.110) |
| regulatory_capacity | +0.285 | 2.662 | 99 | Weak — partial recovery (+0.041) |
| **Average** | **+0.495** | | | **+0.013 vs v14, +0.093 vs v13** |

### v14 (previous best, for comparison)

| Dimension | r | MSE | n | Notes |
|---|---|---|---|---|
| cooling_capacity | +0.653 | 2.375 | 99 | Best — proxy signal strong + new labels |
| trust_conditions | +0.572 | 3.865 | 99 | Good — consistent since v3 |
| energy_dissipation | +0.531 | 1.611 | 99 | Good — large jump from v13 |
| hostility_index | +0.523 | 4.647 | 99 | Good |
| authority_dynamics | +0.503 | 1.839 | 93 | Moderate — improved |
| resilience_baseline | +0.473 | 1.380 | 99 | Moderate — slight regression |
| defensive_architecture | +0.474 | 1.992 | 88 | Moderate — improved |
| contractual_clarity | +0.432 | 1.463 | 89 | Moderate — large jump from v13 |
| threat_exposure | +0.414 | 8.228 | 99 | Moderate — dramatic jump from 0.16 |
| regulatory_capacity | +0.244 | 2.928 | 99 | Weak — regressed from v13 (0.325) |
| **Average** | **+0.482** | | | **+0.080 vs v13** |

### v13 (for comparison)

| Dimension | r | MSE | n | Notes |
|---|---|---|---|---|
| threat_exposure | +0.16 | 13.62 | 99 | Weakest — CC poisoning effect |
| contractual_clarity | +0.27 | 2.16 | 89 | Moderate — small dataset |
| regulatory_capacity | +0.32 | 1.56 | 99 | Moderate — relabeling helped |
| defensive_architecture | +0.37 | 1.95 | 88 | Moderate — redefined construct |
| energy_dissipation | +0.39 | 1.99 | 99 | Moderate — range compression |
| authority_dynamics | +0.46 | 1.69 | 93 | Moderate — ad_8 synthetic helped |
| hostility_index | +0.48 | 5.63 | 99 | Good |
| resilience_baseline | +0.50 | 1.30 | 99 | Good |
| trust_conditions | +0.50 | 4.30 | 99 | Good |
| cooling_capacity | +0.57 | 2.76 | 99 | Good |
| **Average** | **+0.402** | | | Baseline after separated labeling |

## Calibration Summary (v13)

Score calibration via isotonic regression reduces MAE by 4–25% and decompresses score ranges. Confidence calibration fixes inverted confidence (raw r(conf,acc) was negative for 6/10 dims; post-calibration all positive).

## Artifacts

v21 artifacts are in `models/psq-student/` (production slot):
- `best.pt` — PyTorch checkpoint (epoch 6)
- `model.onnx` — Full-precision ONNX (254 MB)
- `model_quantized.onnx` — INT8 quantized (64 MB)
- `tokenizer/` — Tokenizer files
- `calibration.json` — Score + confidence calibration maps
- `config.json` — Model config
- `held_out_results.json` — Held-out evaluation metrics

v22a artifacts are in `models/psq-v22a/`:
- `best.pt` — PyTorch checkpoint (epoch 4)
- `best_results.json` — Test metrics

v19 artifacts are in `models/psq-v19/`:
- `best.pt` — PyTorch checkpoint (epoch 4)
- `held_out_results.json` — Held-out evaluation metrics
- `tokenizer/` — Tokenizer files
- `config.json`, `best_results.json`, `test_results.json`

v15 artifacts are in `models/psq-v15/`:
- `best.pt` — PyTorch checkpoint (epoch 7)
- `held_out_results.json` — Held-out evaluation metrics
- `tokenizer/` — Tokenizer files
- `config.json`, `best_results.json`, `test_results.json`

v14 artifacts are in `models/psq-v14/`:
- `best.pt` — PyTorch checkpoint (epoch 8)
- `held_out_results.json` — Held-out evaluation metrics
- `tokenizer/` — Tokenizer files
- `config.json`, `best_results.json`, `test_results.json`

v16 artifacts are in `models/psq-student/` (production slot):
- `best.pt` — PyTorch checkpoint (epoch 6)
- `model.onnx` — Full-precision ONNX (254 MB)
- `model_quantized.onnx` — INT8 quantized (64 MB)
- `tokenizer/` — Tokenizer files
- `calibration.json` — Score + confidence calibration maps
- `config.json` — Model config
- `held_out_results.json` — Held-out evaluation metrics
