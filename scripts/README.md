# PSQ Scripts

All scripts for the PSQ student model distillation pipeline.

## Data Pipeline

| Script | Description | Key Args | Outputs |
|---|---|---|---|
| `validate_proxy.py` | Validate detoxify as proxy teacher against Berkeley hate speech | — | `data/proxy-validation/*.png`, correlation table |
| `benchmark_proxies.py` | Benchmark 5 proxy models against Berkeley ground truth | — | stdout correlation table |
| `benchmark_ground_truth.py` | Benchmark ground truth datasets against proxy stack | — | `data/ground_truth_comparison.json` |
| `build_composite_ground_truth.py` | Merge all proxy datasets into composite training JSONL | — | `data/composite-ground-truth.jsonl` |
| `map_new_datasets.py` | Map CaSiNo, Politeness, ProsocialDialog to PSQ dimensions | — | `data/new-dataset-ground-truth.jsonl` |
| `label_proxy.py` | Batch-label texts with multi-signal proxy (detoxify + sentiment + emotion) | `--input`, `--output`, `--batch-size` | JSONL with proxy scores |

## Training

| Script | Description | Key Args | Outputs |
|---|---|---|---|
| `distill.py` | Train PSQ student model (transformer encoder + 10 dim heads) | `--model-name`, `--epochs`, `--batch-size`, `--grad-accum`, `--conf-mode`, `--conf-power`, `--max-length` | `models/psq-student/best.pt`, `config.json`, `test_results.json`, `tokenizer/` |
| `calibrate.py` | Fit isotonic calibration for scores + confidence on validation set | `--model-dir` | `models/psq-student/calibration.json` |
| `launch_v3b.sh` | Back up v3 results and launch v3b training | — | Training log |
| `launch_v4_deberta.sh` | Back up v3b results and launch v4 DeBERTa training | — | Training log |

## Evaluation

| Script | Description | Key Args | Outputs |
|---|---|---|---|
| `eval.py` | Per-dimension r, MSE, tier accuracy, calibration analysis | `--checkpoint`, `--test-file` | stdout table, `eval_results.json` |
| `error_analysis.py` | Error stats, top-20 high-error samples, systematic bias detection | `--split`, `--checkpoint` | `models/psq-student/error_analysis.json` |
| `compare_versions.py` | Auto-detect and compare model versions side-by-side | `[versions...]`, `--metric r/mse/n/all`, `--use-val` | stdout comparison table + bar chart |

## Validation (Psychometric)

| Script | Description | Key Args | Outputs |
|---|---|---|---|
| `test_retest_reliability.py` | Perturbation-based ICC(3,1) stability analysis (5 perturbation types) | `--model-type`, `--n-samples` | `test_retest_results.json` |
| `validate_discriminant_sentiment.py` | PSQ vs VADER sentiment — proves PSQ ≠ just sentiment | — | `discriminant_validity_results.json` |
| `validate_confidence_calibration.py` | Confidence bin analysis — checks if conf predicts accuracy | — | `confidence_calibration_results.json` |
| `validate_known_groups.py` | ANOVA across source datasets — group separation test | — | `known_groups_results.json` |
| `validate_calibrated.py` | Re-run all 3 validations above using calibrated outputs | — | `validation_calibrated_results.json` |

## Export

| Script | Description | Key Args | Outputs |
|---|---|---|---|
| `export_onnx.py` | Export to ONNX with optional INT8 quantization | `--checkpoint`, `--no-quantize` | `model.onnx`, `model_quantized.onnx`, `tokenizer/` |

## Typical Workflow

```
# 1. Build composite training data
python scripts/build_composite_ground_truth.py

# 2. Train model
python scripts/distill.py --model-name distilbert-base-uncased --epochs 10

# 3. Fit calibration
python scripts/calibrate.py

# 4. Evaluate
python scripts/eval.py
python scripts/error_analysis.py

# 5. Run psychometric validation
python scripts/test_retest_reliability.py
python scripts/validate_calibrated.py

# 6. Export to ONNX
python scripts/export_onnx.py

# 7. Compare versions
python scripts/compare_versions.py v2d v3b

# 8. (For DeBERTa) Use reduced batch with gradient accumulation
python scripts/distill.py --model-name microsoft/deberta-v3-small \
    --batch-size 8 --grad-accum 4 --max-length 256 \
    --conf-mode two-phase --conf-warmup-epochs 2 --conf-power 2.0
```
