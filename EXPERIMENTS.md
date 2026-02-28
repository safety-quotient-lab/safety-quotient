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

## Key Hyperparameters (v15, current)

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

### v15 (current best, 2026-02-27)

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

v13 artifacts are in `models/psq-student/` (production slot):
- `best.pt` — PyTorch checkpoint
- `model.onnx` — Full-precision ONNX (254 MB)
- `model_quantized.onnx` — INT8 quantized (64 MB)
- `tokenizer/` — Tokenizer files
- `calibration.json` — Score + confidence calibration maps
- `config.json` — Model config
- `held_out_results.json` — Held-out evaluation metrics
