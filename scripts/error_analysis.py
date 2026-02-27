"""
Error analysis for the PSQ student model (v2d, DistilBERT-base-uncased).

Loads the trained model and ground truth data, runs inference, and computes
per-dimension error statistics including:
  - Per-sample absolute error
  - Top 20 highest-error samples per dimension
  - Mean error by source dataset
  - Score distribution comparisons (predicted vs actual)
  - Systematic bias detection (mean signed error)

Usage:
  python scripts/error_analysis.py
  python scripts/error_analysis.py --split test     # only test split
  python scripts/error_analysis.py --split all       # all data (default)
"""

import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent.parent

# Import model architecture constants from distill.py
DIMENSIONS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]
DIM_TO_IDX = {d: i for i, d in enumerate(DIMENSIONS)}
N_DIMS = len(DIMENSIONS)

SEED = 42


# =====================================================================
# Model (duplicated from distill.py to avoid import side-effects)
# =====================================================================
import torch.nn as nn

class PSQStudent(nn.Module):
    """DistilBERT encoder with 10 dimension heads."""

    def __init__(self, model_name, n_dims=N_DIMS):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden // 2, 2) for _ in range(n_dims)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        projected = self.proj(cls)

        scores = []
        confs = []
        for head in self.heads:
            out = head(projected)
            s = torch.sigmoid(out[:, 0]) * 10.0
            c = torch.sigmoid(out[:, 1])
            scores.append(s)
            confs.append(c)

        scores = torch.stack(scores, dim=1)
        confs = torch.stack(confs, dim=1)
        return scores, confs


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_data():
    """Load composite ground truth and LLM labels, matching distill.py logic."""
    all_records = []
    data_dir = ROOT / "data"

    for fname in ["composite-ground-truth.jsonl", "train-llm.jsonl"]:
        fpath = data_dir / fname
        if fpath.exists():
            count = 0
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_records.append(json.loads(line))
                        count += 1
            print(f"  {fname}: {count} records")
        else:
            print(f"  {fname}: not found (skipping)")

    return all_records


def split_data(all_records):
    """Replicate the exact 80/10/10 split from distill.py with seed 42."""
    np.random.seed(SEED)
    np.random.shuffle(all_records)
    n = len(all_records)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_recs = all_records[:n_train]
    val_recs = all_records[n_train:n_train + n_val]
    test_recs = all_records[n_train + n_val:]
    return train_recs, val_recs, test_recs


def extract_labels(rec):
    """Extract scores, confidences, mask, source from a record."""
    scores = np.full(N_DIMS, np.nan)
    confs = np.full(N_DIMS, np.nan)
    mask = np.zeros(N_DIMS, dtype=bool)

    dims = rec.get("dimensions", {})
    for dim_id, val in dims.items():
        if dim_id not in DIM_TO_IDX:
            continue
        idx = DIM_TO_IDX[dim_id]
        score = val.get("score")
        conf = val.get("confidence", 0.5)
        if score is not None and not (isinstance(score, float) and math.isnan(score)):
            scores[idx] = score
            confs[idx] = conf
            mask[idx] = True

    source = rec.get("source", rec.get("teacher", "unknown"))
    return scores, confs, mask, source


def run_inference(model, tokenizer, records, device, max_length=128, batch_size=32):
    """Run model inference on all records, return predictions array [N, 10]."""
    model.eval()
    all_preds = []
    all_pred_confs = []

    n = len(records)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = [rec["text"] for rec in records[start:end]]

        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            pred_scores, pred_confs = model(input_ids, attention_mask)
            all_preds.append(pred_scores.cpu().numpy())
            all_pred_confs.append(pred_confs.cpu().numpy())

        if (start // batch_size) % 50 == 0:
            print(f"    Inference: {end}/{n} records...", flush=True)

    return np.concatenate(all_preds, axis=0), np.concatenate(all_pred_confs, axis=0)


def analyze_errors(records, predictions, pred_confs, split_labels):
    """Compute comprehensive error analysis.

    Args:
        records: list of raw record dicts
        predictions: [N, 10] array of predicted scores
        pred_confs: [N, 10] array of predicted confidences
        split_labels: dict mapping record index -> 'train'/'val'/'test'

    Returns:
        dict with per-dimension error analysis
    """
    n = len(records)
    results = {}

    # Pre-extract all labels
    all_scores = np.full((n, N_DIMS), np.nan)
    all_confs = np.full((n, N_DIMS), np.nan)
    all_masks = np.zeros((n, N_DIMS), dtype=bool)
    all_sources = []

    for i, rec in enumerate(records):
        scores, confs, mask, source = extract_labels(rec)
        all_scores[i] = scores
        all_confs[i] = confs
        all_masks[i] = mask
        all_sources.append(source)

    all_sources = np.array(all_sources)

    # Overall summary
    overall = {
        "total_records": n,
        "split_counts": {
            "train": sum(1 for v in split_labels.values() if v == "train"),
            "val": sum(1 for v in split_labels.values() if v == "val"),
            "test": sum(1 for v in split_labels.values() if v == "test"),
        },
    }

    # Per-dimension analysis
    dim_results = {}
    for dim_idx, dim_name in enumerate(DIMENSIONS):
        # Valid samples for this dimension
        valid = all_masks[:, dim_idx] & ~np.isnan(all_scores[:, dim_idx])
        valid_idx = np.where(valid)[0]

        if len(valid_idx) < 2:
            dim_results[dim_name] = {"n_valid": int(len(valid_idx)), "skipped": True}
            continue

        pred_vals = predictions[valid_idx, dim_idx]
        true_vals = all_scores[valid_idx, dim_idx]
        true_confs_dim = all_confs[valid_idx, dim_idx]
        pred_confs_dim = pred_confs[valid_idx, dim_idx]

        # Absolute and signed errors
        abs_errors = np.abs(pred_vals - true_vals)
        signed_errors = pred_vals - true_vals  # positive = over-predicting

        # Basic stats
        mae = float(np.mean(abs_errors))
        mse = float(np.mean((pred_vals - true_vals) ** 2))
        rmse = float(np.sqrt(mse))
        mean_signed = float(np.mean(signed_errors))
        std_signed = float(np.std(signed_errors))

        # Pearson r
        from scipy import stats as sp_stats
        r, p_val = sp_stats.pearsonr(pred_vals, true_vals)

        # Score distributions
        pred_dist = {
            "mean": float(np.mean(pred_vals)),
            "std": float(np.std(pred_vals)),
            "min": float(np.min(pred_vals)),
            "max": float(np.max(pred_vals)),
            "median": float(np.median(pred_vals)),
            "q25": float(np.percentile(pred_vals, 25)),
            "q75": float(np.percentile(pred_vals, 75)),
        }
        true_dist = {
            "mean": float(np.mean(true_vals)),
            "std": float(np.std(true_vals)),
            "min": float(np.min(true_vals)),
            "max": float(np.max(true_vals)),
            "median": float(np.median(true_vals)),
            "q25": float(np.percentile(true_vals, 25)),
            "q75": float(np.percentile(true_vals, 75)),
        }

        # Top 20 highest-error samples
        top_error_idx = np.argsort(abs_errors)[::-1][:20]
        top_errors = []
        for rank, ei in enumerate(top_error_idx):
            global_idx = valid_idx[ei]
            rec = records[global_idx]
            text = rec["text"]
            snippet = text[:150] + "..." if len(text) > 150 else text
            top_errors.append({
                "rank": rank + 1,
                "text_snippet": snippet,
                "predicted": round(float(pred_vals[ei]), 3),
                "actual": round(float(true_vals[ei]), 3),
                "error": round(float(abs_errors[ei]), 3),
                "signed_error": round(float(signed_errors[ei]), 3),
                "true_confidence": round(float(true_confs_dim[ei]), 3),
                "pred_confidence": round(float(pred_confs_dim[ei]), 3),
                "source": all_sources[global_idx],
                "split": split_labels.get(global_idx, "unknown"),
            })

        # Mean error by source dataset
        source_errors = defaultdict(lambda: {"errors": [], "signed_errors": [], "count": 0})
        for ei in range(len(valid_idx)):
            global_idx = valid_idx[ei]
            src = all_sources[global_idx]
            source_errors[src]["errors"].append(abs_errors[ei])
            source_errors[src]["signed_errors"].append(signed_errors[ei])
            source_errors[src]["count"] += 1

        source_summary = {}
        for src, data in sorted(source_errors.items()):
            errs = np.array(data["errors"])
            serrs = np.array(data["signed_errors"])
            source_summary[src] = {
                "count": data["count"],
                "mae": round(float(np.mean(errs)), 4),
                "rmse": round(float(np.sqrt(np.mean(errs**2))), 4),
                "mean_signed_error": round(float(np.mean(serrs)), 4),
                "std_error": round(float(np.std(errs)), 4),
            }

        # Error by split
        split_errors = defaultdict(lambda: {"errors": [], "count": 0})
        for ei in range(len(valid_idx)):
            global_idx = valid_idx[ei]
            sp = split_labels.get(global_idx, "unknown")
            split_errors[sp]["errors"].append(abs_errors[ei])
            split_errors[sp]["count"] += 1

        split_summary = {}
        for sp, data in sorted(split_errors.items()):
            errs = np.array(data["errors"])
            split_summary[sp] = {
                "count": data["count"],
                "mae": round(float(np.mean(errs)), 4),
            }

        # Error buckets: how often is the model off by 0-1, 1-2, 2-3, 3+?
        bucket_edges = [0, 1, 2, 3, 5, 10]
        buckets = {}
        for i in range(len(bucket_edges) - 1):
            lo, hi = bucket_edges[i], bucket_edges[i+1]
            label = f"{lo}-{hi}"
            in_bucket = ((abs_errors >= lo) & (abs_errors < hi)).sum()
            buckets[label] = int(in_bucket)

        dim_results[dim_name] = {
            "n_valid": int(len(valid_idx)),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "pearson_r": round(float(r), 4),
            "p_value": float(p_val),
            "mean_signed_error": round(mean_signed, 4),
            "std_signed_error": round(std_signed, 4),
            "bias_direction": "over-predicts" if mean_signed > 0.1 else ("under-predicts" if mean_signed < -0.1 else "neutral"),
            "predicted_distribution": pred_dist,
            "actual_distribution": true_dist,
            "range_compression": round(pred_dist["std"] / max(true_dist["std"], 0.01), 4),
            "error_buckets": buckets,
            "by_source": source_summary,
            "by_split": split_summary,
            "top_20_errors": top_errors,
        }

    results["overall"] = overall
    results["dimensions"] = dim_results

    return results


def print_summary(results):
    """Print a human-readable summary of the error analysis."""
    print("\n" + "=" * 80)
    print("PSQ STUDENT MODEL - ERROR ANALYSIS SUMMARY")
    print("=" * 80)

    overall = results["overall"]
    print(f"\nTotal records: {overall['total_records']}")
    print(f"Splits: {overall['split_counts']}")

    # Summary table
    print(f"\n{'Dimension':<25s} {'n':>5s} {'MAE':>7s} {'RMSE':>7s} {'r':>7s} {'Bias':>8s} {'Direction':<15s} {'PredStd':>8s} {'TrueStd':>8s} {'Compress':>8s}")
    print("-" * 105)

    dims = results["dimensions"]
    for dim_name in DIMENSIONS:
        d = dims[dim_name]
        if d.get("skipped"):
            print(f"  {dim_name:<25s}  (skipped, n={d['n_valid']})")
            continue
        print(f"{dim_name:<25s} {d['n_valid']:5d} {d['mae']:7.3f} {d['rmse']:7.3f} {d['pearson_r']:+7.3f} {d['mean_signed_error']:+8.3f} {d['bias_direction']:<15s} {d['predicted_distribution']['std']:8.3f} {d['actual_distribution']['std']:8.3f} {d['range_compression']:8.3f}")

    # Error bucket summary
    print(f"\n--- Error Magnitude Distribution ---")
    print(f"{'Dimension':<25s} {'0-1':>7s} {'1-2':>7s} {'2-3':>7s} {'3-5':>7s} {'5-10':>7s}")
    print("-" * 60)
    for dim_name in DIMENSIONS:
        d = dims[dim_name]
        if d.get("skipped"):
            continue
        b = d["error_buckets"]
        n = d["n_valid"]
        print(f"{dim_name:<25s} {b.get('0-1',0)/n*100:6.1f}% {b.get('1-2',0)/n*100:6.1f}% {b.get('2-3',0)/n*100:6.1f}% {b.get('3-5',0)/n*100:6.1f}% {b.get('5-10',0)/n*100:6.1f}%")

    # Worst dimensions by MAE
    print(f"\n--- Dimensions Ranked by MAE (worst first) ---")
    sorted_dims = sorted(
        [(name, d) for name, d in dims.items() if not d.get("skipped")],
        key=lambda x: x[1]["mae"],
        reverse=True,
    )
    for rank, (name, d) in enumerate(sorted_dims, 1):
        print(f"  {rank}. {name}: MAE={d['mae']:.3f}, RMSE={d['rmse']:.3f}, bias={d['mean_signed_error']:+.3f}")

    # Source dataset analysis (aggregated across dimensions)
    print(f"\n--- Source Dataset Error Summary (all dimensions) ---")
    source_agg = defaultdict(lambda: {"total_error": 0, "count": 0, "total_signed": 0})
    for dim_name in DIMENSIONS:
        d = dims[dim_name]
        if d.get("skipped"):
            continue
        for src, stats in d["by_source"].items():
            source_agg[src]["total_error"] += stats["mae"] * stats["count"]
            source_agg[src]["total_signed"] += stats["mean_signed_error"] * stats["count"]
            source_agg[src]["count"] += stats["count"]

    print(f"{'Source':<30s} {'n':>7s} {'MAE':>8s} {'Bias':>8s}")
    print("-" * 55)
    for src, data in sorted(source_agg.items(), key=lambda x: x[1]["total_error"]/max(x[1]["count"],1), reverse=True):
        mae = data["total_error"] / max(data["count"], 1)
        bias = data["total_signed"] / max(data["count"], 1)
        print(f"{src:<30s} {data['count']:7d} {mae:8.3f} {bias:+8.3f}")

    # Top errors across all dimensions
    print(f"\n--- Top 5 Highest-Error Samples Per Dimension ---")
    for dim_name in DIMENSIONS:
        d = dims[dim_name]
        if d.get("skipped"):
            continue
        print(f"\n  [{dim_name}] (MAE={d['mae']:.3f}, bias={d['mean_signed_error']:+.3f})")
        for item in d["top_20_errors"][:5]:
            print(f"    #{item['rank']} err={item['error']:.2f} (pred={item['predicted']:.1f} actual={item['actual']:.1f}) "
                  f"src={item['source']} split={item['split']}")
            print(f"       \"{item['text_snippet'][:100]}\"")

    # Systematic bias analysis
    print(f"\n--- Systematic Bias Analysis ---")
    biased = [(name, d) for name, d in dims.items()
              if not d.get("skipped") and abs(d["mean_signed_error"]) > 0.3]
    if biased:
        for name, d in sorted(biased, key=lambda x: abs(x[1]["mean_signed_error"]), reverse=True):
            direction = "OVER" if d["mean_signed_error"] > 0 else "UNDER"
            print(f"  {name}: {direction}-predicts by {abs(d['mean_signed_error']):.3f} on average")
            print(f"    Predicted range: [{d['predicted_distribution']['min']:.1f}, {d['predicted_distribution']['max']:.1f}] "
                  f"mean={d['predicted_distribution']['mean']:.2f} std={d['predicted_distribution']['std']:.2f}")
            print(f"    Actual range:    [{d['actual_distribution']['min']:.1f}, {d['actual_distribution']['max']:.1f}] "
                  f"mean={d['actual_distribution']['mean']:.2f} std={d['actual_distribution']['std']:.2f}")
            print(f"    Range compression ratio: {d['range_compression']:.2f} (1.0=perfect, <1=compressed)")
    else:
        print("  No dimensions with systematic bias > 0.3")

    # Range compression analysis
    print(f"\n--- Range Compression Analysis ---")
    print("  (ratio < 0.8 means model predictions are more compressed than actuals)")
    compressed = [(name, d) for name, d in dims.items()
                  if not d.get("skipped") and d["range_compression"] < 0.8]
    if compressed:
        for name, d in sorted(compressed, key=lambda x: x[1]["range_compression"]):
            print(f"  {name}: compression={d['range_compression']:.2f} "
                  f"(pred_std={d['predicted_distribution']['std']:.2f}, true_std={d['actual_distribution']['std']:.2f})")
    else:
        print("  No dimensions with severe compression (<0.8)")


def main():
    parser = argparse.ArgumentParser(description="Error analysis for PSQ student model")
    parser.add_argument("--split", default="all", choices=["all", "test", "val"],
                        help="Which split to analyze (default: all)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (default: models/psq-student/best.pt)")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    all_records = load_data()
    print(f"  Total: {len(all_records)} records")

    # Split data (same seed/logic as distill.py)
    train_recs, val_recs, test_recs = split_data(all_records)
    print(f"  Split: {len(train_recs)} train / {len(val_recs)} val / {len(test_recs)} test")

    # Reload data fresh (shuffle mutated the list) to get original indices
    # We need to re-load and re-split to track indices properly
    all_records_fresh = load_data()
    np.random.seed(SEED)
    np.random.shuffle(all_records_fresh)
    n = len(all_records_fresh)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    # Build split label map
    split_labels = {}
    for i in range(n):
        if i < n_train:
            split_labels[i] = "train"
        elif i < n_train + n_val:
            split_labels[i] = "val"
        else:
            split_labels[i] = "test"

    # Select records based on --split
    if args.split == "test":
        records = all_records_fresh[n_train + n_val:]
        # Remap split_labels to new indices
        new_split = {i: "test" for i in range(len(records))}
        split_labels = new_split
        print(f"\n  Analyzing TEST split only: {len(records)} records")
    elif args.split == "val":
        records = all_records_fresh[n_train:n_train + n_val]
        new_split = {i: "val" for i in range(len(records))}
        split_labels = new_split
        print(f"\n  Analyzing VAL split only: {len(records)} records")
    else:
        records = all_records_fresh
        print(f"\n  Analyzing ALL data: {len(records)} records")

    # Load model
    ckpt_path = args.checkpoint or str(ROOT / "models" / "psq-student" / "best.pt")
    print(f"\nLoading model from {ckpt_path}...")
    model = PSQStudent(args.model_name).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("  Model loaded successfully.")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Run inference
    print(f"\nRunning inference on {len(records)} records...")
    predictions, pred_confs = run_inference(
        model, tokenizer, records, device,
        max_length=args.max_length, batch_size=args.batch_size,
    )
    print(f"  Predictions shape: {predictions.shape}")

    # Analyze errors
    print("\nAnalyzing errors...")
    results = analyze_errors(records, predictions, pred_confs, split_labels)

    # Print summary
    print_summary(results)

    # Save detailed results
    output_path = ROOT / "models" / "psq-student" / "error_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
