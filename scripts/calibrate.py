"""
Post-training calibration for PSQ student model.

Fits per-dimension isotonic regression on validation set to de-compress
predicted score ranges back to the true distribution. This addresses the
range compression problem where the model hedges toward the mean.

Usage:
    python scripts/calibrate.py [--model-dir models/psq-student]

Outputs:
    models/psq-student/calibration.json — per-dimension calibration maps
    Prints before/after metrics for each dimension

The calibration maps can be applied at inference time:
    calibrated_score = calibration_map[dimension](raw_score)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import hashlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

# Try sklearn for isotonic regression
try:
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed. Using linear rescaling fallback.")
    print("  Install with: pip install scikit-learn")


DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]


def load_data(data_dir):
    """Load composite + LLM data, split into val set using hash-based split."""
    records = []
    composite_path = data_dir / "composite-ground-truth.jsonl"
    llm_path = data_dir / "train-llm.jsonl"

    for fpath in [composite_path, llm_path]:
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                records.append(json.loads(line))

    # Hash-based split — same as distill.py
    val_records = []
    for rec in records:
        h = int(hashlib.md5(rec["text"].encode()).hexdigest(), 16) % 100
        if 80 <= h < 90:  # val split
            val_records.append(rec)

    return val_records


def get_predictions(model_dir, val_records):
    """Run inference on val records to get raw predictions."""
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config to get model_name
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_name = config.get("model_name", "distilbert-base-uncased")
    else:
        model_name = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Rebuild model architecture — must match distill.py PSQStudent exactly
    class PSQStudent(nn.Module):
        def __init__(self, enc_name, n_dims=10):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(enc_name, use_safetensors=True).float()
            hidden = self.encoder.config.hidden_size
            self.proj = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.heads = nn.ModuleList([nn.Linear(hidden // 2, 2) for _ in range(n_dims)])

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0, :]
            projected = self.proj(cls)
            scores, confs = [], []
            for head in self.heads:
                out = head(projected)
                scores.append(torch.sigmoid(out[:, 0]) * 10.0)
                confs.append(torch.sigmoid(out[:, 1]))
            return torch.stack(scores, dim=1), torch.stack(confs, dim=1)

    model = PSQStudent(model_name).to(device)
    checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # Run inference in batches
    predictions = {d: [] for d in DIMS}
    actuals = {d: [] for d in DIMS}
    pred_confs_map = {d: [] for d in DIMS}
    actual_confs_map = {d: [] for d in DIMS}

    batch_size = 64
    texts = [r["text"] for r in val_records]

    print(f"  Running inference on {len(texts)} val records...")
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                          max_length=256, return_tensors="pt").to(device)
            scores, confs = model(enc["input_ids"], enc["attention_mask"])
            scores = scores.cpu().numpy()
            confs = confs.cpu().numpy()

            for j, rec in enumerate(val_records[i:i + batch_size]):
                for dim_idx, dim_name in enumerate(DIMS):
                    if dim_name in rec.get("dimensions", {}):
                        predictions[dim_name].append(float(scores[j, dim_idx]))
                        actuals[dim_name].append(rec["dimensions"][dim_name]["score"])
                        pred_confs_map[dim_name].append(float(confs[j, dim_idx]))
                        actual_confs_map[dim_name].append(
                            rec["dimensions"][dim_name].get("confidence", 0.5))

    return predictions, actuals, pred_confs_map, actual_confs_map


def fit_isotonic(predictions, actuals):
    """Fit isotonic regression per dimension."""
    calibration = {}

    for dim in DIMS:
        preds = np.array(predictions[dim])
        trues = np.array(actuals[dim])

        if len(preds) < 10:
            print(f"  {dim}: skipped (only {len(preds)} val samples)")
            calibration[dim] = {"method": "identity", "n": len(preds)}
            continue

        if HAS_SKLEARN:
            # Isotonic regression: monotonic mapping from predicted → actual
            iso = IsotonicRegression(y_min=0.0, y_max=10.0, out_of_bounds="clip")
            iso.fit(preds, trues)

            # Store as piecewise linear lookup table
            calibrated = iso.predict(preds)
            x_thresholds = iso.X_thresholds_.tolist() if hasattr(iso, 'X_thresholds_') else sorted(set(preds.tolist()))
            y_thresholds = iso.y_thresholds_.tolist() if hasattr(iso, 'y_thresholds_') else sorted(set(trues.tolist()))

            calibration[dim] = {
                "method": "isotonic",
                "n": len(preds),
                "x_thresholds": [round(x, 4) for x in x_thresholds],
                "y_thresholds": [round(y, 4) for y in y_thresholds],
            }
        else:
            # Linear rescaling fallback: shift and scale to match true distribution
            pred_mean, pred_std = np.mean(preds), np.std(preds)
            true_mean, true_std = np.mean(trues), np.std(trues)

            if pred_std > 0.01:
                scale = true_std / pred_std
                shift = true_mean - pred_mean * scale
            else:
                scale = 1.0
                shift = true_mean - pred_mean

            calibrated = np.clip(preds * scale + shift, 0, 10)

            calibration[dim] = {
                "method": "linear",
                "n": len(preds),
                "scale": round(float(scale), 4),
                "shift": round(float(shift), 4),
            }

        # Report improvement
        pre_mae = np.mean(np.abs(preds - trues))
        post_mae = np.mean(np.abs(calibrated - trues))
        pre_std = np.std(preds)
        post_std = np.std(calibrated)
        true_std = np.std(trues)

        compression_before = pre_std / true_std if true_std > 0 else 1.0
        compression_after = post_std / true_std if true_std > 0 else 1.0

        improvement = (pre_mae - post_mae) / pre_mae * 100 if pre_mae > 0 else 0

        print(f"  {dim:28s}  n={len(preds):4d}  "
              f"MAE {pre_mae:.3f}→{post_mae:.3f} ({improvement:+.1f}%)  "
              f"compress {compression_before:.2f}→{compression_after:.2f}  "
              f"range [{min(calibrated):.1f},{max(calibrated):.1f}]")

    return calibration


def fit_confidence_calibration(pred_confs, predictions, actuals):
    """Fit isotonic regression to map predicted confidence → actual accuracy.

    Maps: predicted_conf → 1 - |error|/5 (actual accuracy on 0-1 scale).
    """
    conf_calibration = {}

    for dim in DIMS:
        preds = np.array(predictions[dim])
        trues = np.array(actuals[dim])
        confs = np.array(pred_confs[dim])

        if len(preds) < 20:
            print(f"  {dim}: skipped (only {len(preds)} val samples)")
            conf_calibration[dim] = {"method": "identity", "n": len(preds)}
            continue

        # Actual accuracy: 1 - |error|/5, clamped to [0, 1]
        errors = np.abs(preds - trues)
        actual_accuracy = np.clip(1.0 - errors / 5.0, 0.0, 1.0)

        # Check pre-calibration: correlation between predicted conf and actual accuracy
        from scipy import stats as sp_stats
        pre_r, pre_p = sp_stats.pearsonr(confs, actual_accuracy)

        if HAS_SKLEARN:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(confs, actual_accuracy)
            calibrated = iso.predict(confs)

            x_thresholds = iso.X_thresholds_.tolist() if hasattr(iso, 'X_thresholds_') else sorted(set(confs.tolist()))
            y_thresholds = iso.y_thresholds_.tolist() if hasattr(iso, 'y_thresholds_') else sorted(set(actual_accuracy.tolist()))

            conf_calibration[dim] = {
                "method": "isotonic",
                "n": len(confs),
                "x_thresholds": [round(x, 4) for x in x_thresholds],
                "y_thresholds": [round(y, 4) for y in y_thresholds],
            }
        else:
            # Linear fallback
            pred_mean, pred_std = np.mean(confs), np.std(confs)
            true_mean, true_std = np.mean(actual_accuracy), np.std(actual_accuracy)
            scale = true_std / pred_std if pred_std > 0.01 else 1.0
            shift = true_mean - pred_mean * scale
            calibrated = np.clip(confs * scale + shift, 0, 1)

            conf_calibration[dim] = {
                "method": "linear",
                "n": len(confs),
                "scale": round(float(scale), 4),
                "shift": round(float(shift), 4),
            }

        post_r, _ = sp_stats.pearsonr(calibrated, actual_accuracy)

        print(f"  {dim:28s}  n={len(confs):4d}  "
              f"r(conf,acc) {pre_r:+.3f}→{post_r:+.3f}  "
              f"conf range [{min(calibrated):.2f},{max(calibrated):.2f}]")

    return conf_calibration


def main():
    parser = argparse.ArgumentParser(description="Calibrate PSQ student model predictions")
    parser.add_argument("--model-dir", default="models/psq-student",
                       help="Directory containing best.pt")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path("data")

    print("Loading validation data...")
    val_records = load_data(data_dir)
    print(f"  {len(val_records)} val records")

    print("\nRunning inference...")
    predictions, actuals, pred_confs, actual_confs = get_predictions(model_dir, val_records)

    print("\nFitting score calibration...")
    calibration = fit_isotonic(predictions, actuals)

    print("\nFitting confidence calibration...")
    conf_calibration = fit_confidence_calibration(pred_confs, predictions, actuals)

    # Merge into single calibration file
    for dim in DIMS:
        if dim in calibration and dim in conf_calibration:
            calibration[dim]["confidence_calibration"] = conf_calibration[dim]

    # Save calibration map
    out_path = model_dir / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Calibration includes both score and confidence maps.")
    print("Score calibration: isotonic regression on raw scores → de-compressed scores")
    print("Confidence calibration: isotonic regression on raw conf → actual accuracy")


if __name__ == "__main__":
    main()
