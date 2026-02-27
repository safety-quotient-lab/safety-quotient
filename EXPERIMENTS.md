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
| v14 | 2026-02-27 | DistilBERT | — | 2e-5 | 32 | — | — | +2,000 separated-llm labels (all 10 dims, 200 texts). DB mode. | Same hyperparams as v13. `--out models/psq-v14`. Training in progress. |

## Key Hyperparameters (v13, current)

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

## Held-Out Results by Dimension (v13, separated labels)

Re-scored with separated LLM calls (one dimension per call) to eliminate halo effect. All 100 texts now have complete 10-dimension coverage.

| Dimension | r | MSE | n | Notes |
|---|---|---|---|---|
| threat_exposure | +0.16 | 13.62 | 99 | Still weakest — CC poisoning effect persists in model |
| hostility_index | +0.48 | 5.63 | 99 | Strong — best proxy coverage |
| authority_dynamics | +0.46 | 1.69 | 93 | Improved — ad_8 synthetic helped |
| energy_dissipation | +0.39 | 1.99 | 99 | Moderate — range compression from composite |
| regulatory_capacity | +0.32 | 1.56 | 99 | Moderate — relabeling helped |
| resilience_baseline | +0.50 | 1.30 | 99 | Strong — GoEmotions + Empathetic Dialogues |
| trust_conditions | +0.50 | 4.30 | 99 | Strong — reliable since v3 |
| cooling_capacity | +0.57 | 2.76 | 99 | Strong — GoEmotions proxy works |
| defensive_architecture | +0.37 | 1.95 | 88 | Moderate — redefined construct |
| contractual_clarity | +0.27 | 2.16 | 89 | Moderate — small dataset |
| **Average** | **+0.402** | | | **Down from 0.428 (joint) — harder benchmark** |

## Calibration Summary (v13)

Score calibration via isotonic regression reduces MAE by 4–25% and decompresses score ranges. Confidence calibration fixes inverted confidence (raw r(conf,acc) was negative for 6/10 dims; post-calibration all positive).

## Artifacts

All artifacts are in `models/psq-student/`:
- `best.pt` — PyTorch checkpoint
- `model.onnx` — Full-precision ONNX (254 MB)
- `model_quantized.onnx` — INT8 quantized (64 MB)
- `tokenizer/` — Tokenizer files
- `calibration.json` — Score + confidence calibration maps
- `config.json` — Model config
- `held_out_results.json` — Held-out evaluation metrics
