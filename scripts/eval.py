"""
Evaluate the trained PSQ student model.

Reports:
  - Per-dimension Pearson r, MSE, and score distributions
  - Comparison vs teacher labels
  - Per-tier accuracy breakdown
  - Score calibration analysis

Usage:
  python scripts/eval.py                               # eval best checkpoint
  python scripts/eval.py --checkpoint models/psq-student/best.pt
  python scripts/eval.py --test-file data/test-set.jsonl
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import math
import numpy as np
from pathlib import Path
from scipy import stats

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Import model and dataset classes from distill.py
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from distill import PSQStudent, PSQDataset, DIMENSIONS, DIM_TO_IDX, N_DIMS

# Tier classification for reporting
TIERS = {
    "A": ["hostility_index", "threat_exposure", "energy_dissipation"],
    "B": ["authority_dynamics", "regulatory_capacity", "trust_conditions",
           "cooling_capacity", "contractual_clarity"],
    "C": ["resilience_baseline", "defensive_architecture"],
}
DIM_TO_TIER = {}
for tier, dims in TIERS.items():
    for d in dims:
        DIM_TO_TIER[d] = tier


def load_model(checkpoint_path, config_path=None):
    """Load trained model and tokenizer."""
    save_dir = Path(checkpoint_path).parent
    if config_path is None:
        config_path = save_dir / "config.json"

    with open(config_path) as f:
        config = json.load(f)

    model_name = config["model_name"]
    max_length = config["max_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PSQStudent(model_name)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    return model, tokenizer, max_length, config


def evaluate_detailed(model, dataloader, device):
    """Run evaluation with detailed per-dimension statistics."""
    model.eval()

    all_pred_scores = []
    all_pred_confs = []
    all_true_scores = []
    all_true_confs = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pred_s, pred_c = model(input_ids, attention_mask)
            all_pred_scores.append(pred_s.cpu())
            all_pred_confs.append(pred_c.cpu())
            all_true_scores.append(batch["scores"])
            all_true_confs.append(batch["confidences"])
            all_masks.append(batch["mask"])

    pred_scores = torch.cat(all_pred_scores).numpy()
    pred_confs = torch.cat(all_pred_confs).numpy()
    true_scores = torch.cat(all_true_scores).numpy()
    true_confs = torch.cat(all_true_confs).numpy()
    masks = torch.cat(all_masks).numpy()

    results = {}
    for i, dim_id in enumerate(DIMENSIONS):
        m = masks[:, i] & ~np.isnan(true_scores[:, i])
        n = int(m.sum())

        if n < 5:
            results[dim_id] = {
                "n": n, "r": float("nan"), "mse": float("nan"),
                "tier": DIM_TO_TIER.get(dim_id, "?"),
            }
            continue

        ps = pred_scores[m, i]
        ts = true_scores[m, i]
        pc = pred_confs[m, i]
        tc = true_confs[m, i]

        r, p_val = stats.pearsonr(ps, ts)
        mse = float(np.mean((ps - ts) ** 2))
        mae = float(np.mean(np.abs(ps - ts)))

        # Score distribution stats
        pred_mean, pred_std = float(np.mean(ps)), float(np.std(ps))
        true_mean, true_std = float(np.mean(ts)), float(np.std(ts))

        # Confidence calibration: does predicted confidence correlate with actual error?
        errors = np.abs(ps - ts)
        conf_r = float("nan")
        if n > 10:
            # Higher confidence should correlate with lower error (negative r)
            conf_r, _ = stats.pearsonr(pc, errors)

        # Accuracy at thresholds
        within_1 = float(np.mean(np.abs(ps - ts) <= 1.0))
        within_2 = float(np.mean(np.abs(ps - ts) <= 2.0))

        results[dim_id] = {
            "n": n,
            "tier": DIM_TO_TIER.get(dim_id, "?"),
            "r": round(r, 4),
            "p_value": float(p_val),
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "within_1pt": round(within_1, 3),
            "within_2pt": round(within_2, 3),
            "pred_mean": round(pred_mean, 2),
            "pred_std": round(pred_std, 2),
            "true_mean": round(true_mean, 2),
            "true_std": round(true_std, 2),
            "conf_error_r": round(conf_r, 4) if not np.isnan(conf_r) else None,
        }

    # Tier averages
    for tier, dims in TIERS.items():
        tier_rs = [results[d]["r"] for d in dims if d in results and not np.isnan(results[d]["r"])]
        tier_ns = [results[d]["n"] for d in dims if d in results and not np.isnan(results[d]["r"])]
        if tier_rs:
            avg_r = sum(r * n for r, n in zip(tier_rs, tier_ns)) / sum(tier_ns)
            results[f"_tier_{tier}_avg_r"] = round(avg_r, 4)

    # Overall average
    all_rs = [v["r"] for k, v in results.items() if not k.startswith("_") and not np.isnan(v.get("r", float("nan")))]
    all_ns = [v["n"] for k, v in results.items() if not k.startswith("_") and not np.isnan(v.get("r", float("nan")))]
    results["_overall_avg_r"] = round(sum(r * n for r, n in zip(all_rs, all_ns)) / max(sum(all_ns), 1), 4)

    return results


def print_results(results):
    """Pretty-print evaluation results."""
    print(f"\n{'='*80}")
    print("PSQ STUDENT MODEL EVALUATION")
    print(f"{'='*80}")

    print(f"\n  {'Dimension':<25s} {'Tier':>4s} {'r':>8s} {'MSE':>8s} {'MAE':>8s} "
          f"{'±1pt':>6s} {'±2pt':>6s} {'n':>6s} {'pred μ':>7s} {'true μ':>7s}")
    print(f"  {'-'*90}")

    for dim_id in DIMENSIONS:
        r = results.get(dim_id, {})
        tier = r.get("tier", "?")
        pr = r.get("r", float("nan"))
        pr_str = f"{pr:+.4f}" if not np.isnan(pr) else "   N/A"
        mse = r.get("mse", float("nan"))
        mse_str = f"{mse:.4f}" if not np.isnan(mse) else "   N/A"
        mae = r.get("mae", float("nan"))
        mae_str = f"{mae:.4f}" if not np.isnan(mae) else "   N/A"
        w1 = r.get("within_1pt", float("nan"))
        w1_str = f"{w1:.1%}" if not np.isnan(w1) else " N/A"
        w2 = r.get("within_2pt", float("nan"))
        w2_str = f"{w2:.1%}" if not np.isnan(w2) else " N/A"
        n = r.get("n", 0)
        pm = r.get("pred_mean", float("nan"))
        pm_str = f"{pm:.1f}" if not np.isnan(pm) else "N/A"
        tm = r.get("true_mean", float("nan"))
        tm_str = f"{tm:.1f}" if not np.isnan(tm) else "N/A"

        print(f"  {dim_id:<25s} {tier:>4s} {pr_str:>8s} {mse_str:>8s} {mae_str:>8s} "
              f"{w1_str:>6s} {w2_str:>6s} {n:>6d} {pm_str:>7s} {tm_str:>7s}")

    print(f"\n  Tier Averages:")
    for tier in ["A", "B", "C"]:
        key = f"_tier_{tier}_avg_r"
        if key in results:
            print(f"    Tier {tier}: r={results[key]:+.4f}")

    overall = results.get("_overall_avg_r", float("nan"))
    print(f"\n  Overall: r={overall:+.4f}")

    # Confidence calibration
    print(f"\n  Confidence Calibration (r between predicted conf and |error|):")
    print(f"  (negative = well-calibrated: high conf → low error)")
    for dim_id in DIMENSIONS:
        r = results.get(dim_id, {})
        cr = r.get("conf_error_r")
        if cr is not None:
            marker = "good" if cr < -0.1 else "weak" if cr < 0 else "miscalibrated"
            print(f"    {dim_id:<25s}  r={cr:+.4f}  ({marker})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PSQ student model")
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "psq-student" / "best.pt"))
    parser.add_argument("--test-file", default=None, help="Custom test JSONL")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, tokenizer, max_length, config = load_model(args.checkpoint)
    model = model.to(device)

    # Load test data
    if args.test_file:
        test_path = Path(args.test_file)
    else:
        # Use composite ground truth with 10% test split
        test_path = ROOT / "data" / "composite-ground-truth.jsonl"

    print(f"Loading test data from {test_path}")
    records = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # If using composite, take last 10% as test
    if not args.test_file:
        np.random.seed(42)
        np.random.shuffle(records)
        n_test = max(len(records) // 10, 100)
        records = records[-n_test:]
        print(f"  Using last {len(records)} records as test set")

    ds = PSQDataset(records, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    # Evaluate
    results = evaluate_detailed(model, loader, device)
    print_results(results)

    # Save
    save_dir = Path(args.checkpoint).parent
    out_path = save_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
