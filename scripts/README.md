# PSQ Scripts

All scripts for the PSQ student model distillation pipeline.

## Core Scripts (active)

| Script | Description | Key Args |
|---|---|---|
| `distill.py` | Train PSQ student model (DistilBERT + 10 dim heads) | `--out DIR`, `--no-save`, `--no-cap`, `--epochs` |
| `label_separated.py` | Extract/ingest/assemble separated-scoring batches | `extract --input`, `ingest --dim`, `assemble`, `status`, `timing` |
| `migrate.py` | Bootstrap DB from JSONLs, incremental ingest | `--ingest JSONL` |
| `eval_held_out.py` | Evaluate model against 100-text held-out benchmark | `--model-dir` |
| `build_composite_ground_truth.py` | Rebuild composite training JSONL from dataset_mappings.json | — |
| `export_onnx.py` | Export to ONNX with optional INT8 quantization | `--checkpoint`, `--no-quantize` |
| `calibrate.py` | Fit isotonic calibration for scores + confidence | `--model-dir` |

## Criterion Validity Studies

| Script | Description | Dataset |
|---|---|---|
| `criterion_casino.py` | CaSiNo negotiation outcomes | Lewis et al. 2017 |
| `criterion_cgawiki.py` | CGA-Wiki derailment prediction | Zhang et al. 2018 |
| `criterion_dealornodeal.py` | Deal or No Deal negotiation | Lewis et al. 2017 |
| `criterion_validity_cmv.py` | Change My View persuasion | Tan et al. 2016 |

## Legacy/Support Scripts

| Script | Description |
|---|---|
| `validate_proxy.py` | Validate detoxify as proxy teacher |
| `benchmark_proxies.py` | Benchmark proxy models |
| `benchmark_ground_truth.py` | Benchmark ground truth datasets |
| `label_proxy.py` | Batch-label texts with proxy stack |
| `test_retest_reliability.py` | Perturbation-based ICC stability |
| `validate_discriminant_sentiment.py` | PSQ vs VADER sentiment |
| `validate_confidence_calibration.py` | Confidence calibration check |
| `validate_known_groups.py` | ANOVA across source datasets |
| `compare_versions.py` | Compare model versions side-by-side |
| `error_analysis.py` | Error stats and bias detection |
| `label_batch_helper.py` | Batch extraction helper (pool management) |

## Deleted Scripts

- `map_new_datasets.py` — superseded by config-driven `build_composite_ground_truth.py`
- `batch_label_llm.js` — used joint scoring (halo problem) and API calls
- `relabel_separated.js` — same reason

## Typical Workflow

```bash
# Smoke test training
python scripts/distill.py --no-save --epochs 1

# Full training
python scripts/distill.py --out models/psq-vN

# Evaluate on held-out
python scripts/eval_held_out.py --model-dir models/psq-vN

# Export to ONNX
python scripts/export_onnx.py

# Labeling workflow (separated scoring)
python scripts/label_separated.py extract --input data/labeling-batch-X.jsonl
# ... score in Claude Code sessions ...
python scripts/label_separated.py ingest --dim te --scores /tmp/scored.json
python scripts/label_separated.py assemble --input data/labeling-batch-X.jsonl --output data/out.jsonl
python scripts/migrate.py --ingest data/out.jsonl
```
