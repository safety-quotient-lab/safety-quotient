"""
Confidence calibration analysis for the PSQ student model.

Tests whether the model's confidence output is informative:
- When confidence is high, is the model actually more accurate?
- Are reliability diagrams monotonically decreasing (higher conf → lower error)?

Bins predictions by confidence, computes MAE per bin, and tests calibration.
"""

import hashlib
import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

DIMENSIONS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]
DIM_TO_IDX = {d: i for i, d in enumerate(DIMENSIONS)}


def load_test_data():
    """Load test split using same hash-based split as distill.py."""
    data_dir = ROOT / "data"
    all_records = []

    for fname in ["composite-ground-truth.jsonl", "train-llm.jsonl"]:
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line))

    test_records = []
    for rec in all_records:
        h = int(hashlib.md5(rec["text"].encode()).hexdigest(), 16) % 100
        if h >= 90:
            test_records.append(rec)

    return test_records


def load_onnx_model(model_dir):
    """Load ONNX model and return a scoring function."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_path = model_dir / "model_quantized.onnx"
    if not model_path.exists():
        model_path = model_dir / "model.onnx"

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def score_batch(texts, max_length=128):
        enc = tokenizer(texts, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="np")
        results = session.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })
        return results[0], results[1]

    return score_batch


def main():
    np.random.seed(42)
    model_dir = ROOT / "models" / "psq-student"
    batch_size = 64

    # Load data
    print("Loading test data...")
    test_records = load_test_data()
    print(f"  {len(test_records)} test records")

    texts = [r["text"] for r in test_records]

    # Extract ground truth
    gt_scores = {}   # {dim: [(idx, gt_score), ...]}
    gt_confs = {}    # {dim: [(idx, gt_conf), ...]}
    for dim in DIMENSIONS:
        gt_scores[dim] = []
        gt_confs[dim] = []

    for i, rec in enumerate(test_records):
        for d, v in rec.get("dimensions", {}).items():
            if d in DIM_TO_IDX and v.get("score") is not None:
                gt_scores[d].append((i, v["score"]))
                gt_confs[d].append((i, v.get("confidence", 0.5)))

    # Score with PSQ model
    print("\nScoring with PSQ model (ONNX)...")
    score_fn = load_onnx_model(model_dir)
    t0 = time.time()
    all_scores = []
    all_confs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        s, c = score_fn(batch)
        all_scores.append(s)
        all_confs.append(c)
    psq_scores = np.vstack(all_scores)
    psq_confs = np.vstack(all_confs)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 75)
    print("CONFIDENCE CALIBRATION ANALYSIS")
    print("=" * 75)

    # --- Overall confidence distribution ---
    print("\n--- Predicted Confidence Distribution ---")
    for dim in DIMENSIONS:
        dim_idx = DIM_TO_IDX[dim]
        confs = psq_confs[:, dim_idx]
        print(f"  {dim:<25} mean={np.mean(confs):.3f}  std={np.std(confs):.3f}  "
              f"range=[{np.min(confs):.3f}, {np.max(confs):.3f}]")

    # --- Confidence vs error correlation ---
    print("\n--- Confidence-Error Correlation (higher conf → lower error?) ---")
    print(f"  {'Dimension':<25} {'r(conf,err)':>11} {'p':>10} {'n':>6}  {'Direction'}")
    print("-" * 70)

    conf_error_results = {}
    for dim in DIMENSIONS:
        dim_idx = DIM_TO_IDX[dim]
        if len(gt_scores[dim]) < 20:
            print(f"  {dim:<25}         --         -- {len(gt_scores[dim]):>6}  (too few)")
            continue

        indices = [x[0] for x in gt_scores[dim]]
        gt_vals = np.array([x[1] for x in gt_scores[dim]])
        pred_vals = psq_scores[indices, dim_idx]
        pred_confs = psq_confs[indices, dim_idx]

        errors = np.abs(pred_vals - gt_vals)
        r, p = stats.pearsonr(pred_confs, errors)

        # Negative r = correct (higher confidence → lower error)
        direction = "CORRECT" if r < -0.05 else "FLAT" if abs(r) < 0.05 else "INVERTED"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        conf_error_results[dim] = {"r": float(r), "p": float(p), "n": len(gt_vals), "direction": direction}
        print(f"  {dim:<25} {r:>11.3f} {p:>10.2e} {len(gt_vals):>6}  {direction} {sig}")

    # --- Binned reliability diagram ---
    print("\n--- Reliability Diagram (binned by predicted confidence) ---")
    print("  Bins: predicted confidence quartiles")
    print()

    N_BINS = 5
    overall_bin_data = {b: {"errors": [], "confs": []} for b in range(N_BINS)}

    for dim in DIMENSIONS:
        dim_idx = DIM_TO_IDX[dim]
        if len(gt_scores[dim]) < 50:
            continue

        indices = [x[0] for x in gt_scores[dim]]
        gt_vals = np.array([x[1] for x in gt_scores[dim]])
        pred_vals = psq_scores[indices, dim_idx]
        pred_confs = psq_confs[indices, dim_idx]
        errors = np.abs(pred_vals - gt_vals)

        # Bin by confidence using quantiles
        try:
            bin_edges = np.quantile(pred_confs, np.linspace(0, 1, N_BINS + 1))
            # Make edges unique
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 3:
                continue
        except Exception:
            continue

        bin_indices = np.digitize(pred_confs, bin_edges[1:-1])

        for b in range(len(bin_edges) - 1):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                bin_errors = errors[mask]
                bin_confs_vals = pred_confs[mask]
                if b < N_BINS:
                    overall_bin_data[b]["errors"].extend(bin_errors.tolist())
                    overall_bin_data[b]["confs"].extend(bin_confs_vals.tolist())

    # Print overall reliability diagram
    print(f"  {'Bin':<6} {'Avg Conf':>9} {'MAE':>6} {'n':>7}  {'Visual'}")
    print("-" * 55)

    prev_mae = None
    monotonic = True
    for b in range(N_BINS):
        if len(overall_bin_data[b]["errors"]) > 0:
            avg_conf = np.mean(overall_bin_data[b]["confs"])
            mae = np.mean(overall_bin_data[b]["errors"])
            n = len(overall_bin_data[b]["errors"])
            bar = "█" * int(mae * 5)  # visual bar proportional to error
            if prev_mae is not None and mae > prev_mae + 0.01:
                monotonic = False
            prev_mae = mae
            print(f"  {b + 1:<6} {avg_conf:>9.3f} {mae:>6.3f} {n:>7}  {bar}")

    # --- Per-dimension binned analysis ---
    print("\n--- Per-Dimension: MAE at Low vs High Confidence ---")
    print(f"  {'Dimension':<25} {'MAE(low)':>9} {'MAE(high)':>10} {'Δ':>7} {'n_low':>6} {'n_high':>7}  {'Useful?'}")
    print("-" * 80)

    calibration_results = {}
    for dim in DIMENSIONS:
        dim_idx = DIM_TO_IDX[dim]
        if len(gt_scores[dim]) < 50:
            print(f"  {dim:<25}        --         --      -- {len(gt_scores[dim]):>6}       --  (too few)")
            continue

        indices = [x[0] for x in gt_scores[dim]]
        gt_vals = np.array([x[1] for x in gt_scores[dim]])
        pred_vals = psq_scores[indices, dim_idx]
        pred_confs = psq_confs[indices, dim_idx]
        errors = np.abs(pred_vals - gt_vals)

        # Split at median confidence
        median_conf = np.median(pred_confs)
        low_mask = pred_confs <= median_conf
        high_mask = pred_confs > median_conf

        if np.sum(low_mask) < 10 or np.sum(high_mask) < 10:
            continue

        mae_low = np.mean(errors[low_mask])
        mae_high = np.mean(errors[high_mask])
        delta = mae_high - mae_low  # Negative = correct calibration

        useful = "YES" if delta < -0.1 else "marginal" if delta < 0 else "NO"

        calibration_results[dim] = {
            "mae_low_conf": round(float(mae_low), 3),
            "mae_high_conf": round(float(mae_high), 3),
            "delta": round(float(delta), 3),
            "useful": useful,
        }
        print(f"  {dim:<25} {mae_low:>9.3f} {mae_high:>10.3f} {delta:>+7.3f} {np.sum(low_mask):>6} {np.sum(high_mask):>7}  {useful}")

    # --- Ground truth confidence vs predicted confidence ---
    print("\n--- GT Teacher Confidence vs Predicted Confidence ---")
    print(f"  {'Dimension':<25} {'r':>6} {'n':>6}")
    print("-" * 45)

    for dim in DIMENSIONS:
        dim_idx = DIM_TO_IDX[dim]
        if len(gt_confs[dim]) < 20:
            continue

        indices = [x[0] for x in gt_confs[dim]]
        gt_conf_vals = np.array([x[1] for x in gt_confs[dim]])
        pred_conf_vals = psq_confs[indices, dim_idx]

        r, p = stats.pearsonr(pred_conf_vals, gt_conf_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {dim:<25} {r:>6.3f} {len(gt_conf_vals):>6}  {sig}")

    # --- Summary ---
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    n_correct = sum(1 for d in conf_error_results.values() if d["direction"] == "CORRECT")
    n_flat = sum(1 for d in conf_error_results.values() if d["direction"] == "FLAT")
    n_inverted = sum(1 for d in conf_error_results.values() if d["direction"] == "INVERTED")
    n_useful = sum(1 for d in calibration_results.values() if d["useful"] == "YES")

    print(f"  Confidence-error direction: {n_correct} correct, {n_flat} flat, {n_inverted} inverted")
    print(f"  Dimensions where high conf → lower MAE: {n_useful}/{len(calibration_results)}")
    print(f"  Reliability diagram monotonic: {'YES' if monotonic else 'NO'}")

    if n_correct >= 6:
        verdict = "GOOD — confidence is informative for most dimensions"
    elif n_correct >= 3:
        verdict = "MIXED — confidence is partially informative"
    else:
        verdict = "POOR — confidence provides little useful signal"
    print(f"  Verdict: {verdict}")

    print("\n  Recommendations:")
    if n_inverted > 0:
        print(f"  - {n_inverted} dimensions have INVERTED calibration (higher conf → higher error)")
        print("    Consider: isotonic regression post-hoc fix, or replace with MC dropout uncertainty")
    if not monotonic:
        print("  - Reliability diagram is non-monotonic — Platt scaling may help")
    if n_correct >= 6:
        print("  - Confidence is usable for gating decisions (e.g., skip dim if conf < 0.4)")

    # Save results
    results = {
        "n_test_records": len(test_records),
        "confidence_error_correlations": conf_error_results,
        "calibration_by_dimension": calibration_results,
        "n_correct_direction": n_correct,
        "n_flat": n_flat,
        "n_inverted": n_inverted,
        "reliability_monotonic": monotonic,
        "verdict": verdict,
    }
    out_path = model_dir / "confidence_calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
