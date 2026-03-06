"""
B2 fix: Re-calibrate hostility_index using quantile-binned isotonic regression.

The standard isotonic regression for HI produced a dead zone (raw 5.854–7.650 →
calibrated 6.69, constant). Diagnostic showed 271/882 val points (30.7%) in this
range, with a mild non-monotonicity: true mean rises 6.03→6.78 across raw 5.9–7.0,
then dips to 6.67 at 7.0–7.7. PAVA pools this pair into a constant step.

Fix: pre-bin predictions into quantile buckets before fitting isotonic regression.
Bin means smooth out the local inversion, letting isotonic see a gradual slope.

Usage:
    python scripts/recalibrate_hi_b2.py [--bins N] [--model-dir PATH] [--dry-run]

Outputs:
    models/psq-student/calibration.json — HI entry updated in-place
    Before/after MAE, dead zone differentiation report
"""

import argparse
import json
import hashlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression


DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]
HI_IDX = DIMS.index("hostility_index")

# Dead zone boundary from current calibration.json (the flat step range)
DEAD_ZONE_RAW_LOW = 5.854
DEAD_ZONE_RAW_HIGH = 7.650


def load_val_data(data_dir):
    """Load val records with HI labels (hash-based split, 80–90 bucket)."""
    records = []
    for filename in ["composite-ground-truth.jsonl", "train-llm.jsonl"]:
        fpath = data_dir / filename
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                records.append(json.loads(line))

    val_records = [
        record for record in records
        if 80 <= int(hashlib.md5(record["text"].encode()).hexdigest(), 16) % 100 < 90
        and "hostility_index" in record.get("dimensions", {})
    ]
    return val_records


def run_hi_inference(model_dir, val_records):
    """Run inference on val records, return (raw_hi_preds, hi_true_scores)."""
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel

    with open(model_dir / "config.json") as f:
        config = json.load(f)
    model_name = config.get("model_name", "distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class PSQStudent(nn.Module):
        def __init__(self, encoder_name, n_dims=10):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                encoder_name, use_safetensors=True
            ).float()
            hidden = self.encoder.config.hidden_size
            self.proj = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.heads = nn.ModuleList(
                [nn.Linear(hidden // 2, 2) for _ in range(n_dims)]
            )

        def forward(self, input_ids, attention_mask):
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_token = output.last_hidden_state[:, 0, :]
            projected = self.proj(cls_token)
            scores, confs = [], []
            for head in self.heads:
                logits = head(projected)
                scores.append(torch.sigmoid(logits[:, 0]) * 10.0)
                confs.append(torch.sigmoid(logits[:, 1]))
            return torch.stack(scores, dim=1), torch.stack(confs, dim=1)

    device = torch.device("cpu")
    model = PSQStudent(model_name).to(device)
    checkpoint = torch.load(
        model_dir / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint)
    model.eval()

    raw_hi_predictions = []
    hi_true_scores = []
    texts = [record["text"] for record in val_records]
    true_scores = [
        record["dimensions"]["hostility_index"]["score"] for record in val_records
    ]

    print(f"  Running inference on {len(texts)} HI val records...")
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch_texts = texts[i : i + 64]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            all_scores, _ = model(encoded["input_ids"], encoded["attention_mask"])
            raw_hi_predictions.extend(
                all_scores[:, HI_IDX].cpu().numpy().tolist()
            )
            hi_true_scores.extend(true_scores[i : i + 64])

    return np.array(raw_hi_predictions), np.array(hi_true_scores)


def fit_isotonic_quantile_binned(raw_predictions, true_scores, n_bins):
    """
    Fit isotonic regression on quantile-binned (bin_center, bin_mean_true) pairs.

    Standard isotonic regression on individual (raw, true) pairs allows local
    non-monotonicities from noise to trigger PAVA pooling, creating flat steps.
    Pre-binning into quantile buckets smooths noise before fitting — PAVA then
    sees a cleaner trend and produces a gradual slope instead of a flat step.
    """
    quantile_boundaries = np.percentile(
        raw_predictions, np.linspace(0, 100, n_bins + 1)
    )

    bin_centers = []
    bin_mean_trues = []
    bin_sample_counts = []

    for bin_index in range(n_bins):
        bin_low = quantile_boundaries[bin_index]
        bin_high = quantile_boundaries[bin_index + 1]
        if bin_index == n_bins - 1:
            in_bin = (raw_predictions >= bin_low) & (raw_predictions <= bin_high)
        else:
            in_bin = (raw_predictions >= bin_low) & (raw_predictions < bin_high)

        if in_bin.sum() < 3:
            continue

        bin_centers.append(float((bin_low + bin_high) / 2))
        bin_mean_trues.append(float(true_scores[in_bin].mean()))
        bin_sample_counts.append(int(in_bin.sum()))

    bin_centers = np.array(bin_centers)
    bin_mean_trues = np.array(bin_mean_trues)
    bin_sample_counts = np.array(bin_sample_counts)

    print(f"\n  Quantile bins ({n_bins} bins, {len(bin_centers)} non-empty):")
    for i, (center, mean_true, count) in enumerate(
        zip(bin_centers, bin_mean_trues, bin_sample_counts)
    ):
        print(f"    bin {i+1:2d}: raw_center={center:.3f}  true_mean={mean_true:.3f}  n={count}")

    iso = IsotonicRegression(y_min=0.0, y_max=10.0, out_of_bounds="clip")
    iso.fit(bin_centers, bin_mean_trues, sample_weight=bin_sample_counts)

    return iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()


def apply_calibration_piecewise(raw_value, x_thresholds, y_thresholds):
    """Piecewise linear interpolation (matches student.js interpolate())."""
    if raw_value <= x_thresholds[0]:
        return y_thresholds[0]
    if raw_value >= x_thresholds[-1]:
        return y_thresholds[-1]
    for i in range(1, len(x_thresholds)):
        if raw_value <= x_thresholds[i]:
            t = (raw_value - x_thresholds[i - 1]) / (
                x_thresholds[i] - x_thresholds[i - 1]
            )
            return y_thresholds[i - 1] + t * (y_thresholds[i] - y_thresholds[i - 1])
    return y_thresholds[-1]


def compute_mae_with_calibration(raw_predictions, true_scores, x_thresholds, y_thresholds):
    """Compute MAE after applying piecewise linear calibration."""
    calibrated = np.array([
        apply_calibration_piecewise(raw, x_thresholds, y_thresholds)
        for raw in raw_predictions
    ])
    return float(np.mean(np.abs(calibrated - true_scores))), calibrated


def main():
    parser = argparse.ArgumentParser(
        description="B2 fix: re-calibrate hostility_index with quantile binning"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of quantile bins for pre-binning (default: 20)",
    )
    parser.add_argument(
        "--model-dir",
        default="models/psq-student",
        help="Directory containing best.pt and calibration.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show results without writing to calibration.json",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path("data")

    # Load current calibration
    calibration_path = model_dir / "calibration.json"
    with open(calibration_path) as f:
        calibration = json.load(f)

    current_hi_cal = calibration["hostility_index"]
    print(
        f"Current HI calibration: method={current_hi_cal['method']}, "
        f"n={current_hi_cal['n']}, "
        f"knots={len(current_hi_cal.get('x_thresholds', []))}"
    )
    print(f"Dead zone: raw [{DEAD_ZONE_RAW_LOW}, {DEAD_ZONE_RAW_HIGH}] → "
          f"calibrated constant {current_hi_cal['y_thresholds'][current_hi_cal['x_thresholds'].index(DEAD_ZONE_RAW_HIGH) if DEAD_ZONE_RAW_HIGH in current_hi_cal['x_thresholds'] else -1]:.4f}")

    print("\nLoading val data...")
    val_records = load_val_data(data_dir)
    print(f"  {len(val_records)} HI-labeled val records")

    print("\nRunning inference...")
    raw_hi_preds, hi_true_scores = run_hi_inference(model_dir, val_records)

    # Before: MAE with current calibration
    before_mae, before_calibrated = compute_mae_with_calibration(
        raw_hi_preds,
        hi_true_scores,
        current_hi_cal["x_thresholds"],
        current_hi_cal["y_thresholds"],
    )

    dead_zone_mask = (raw_hi_preds >= DEAD_ZONE_RAW_LOW) & (
        raw_hi_preds <= DEAD_ZONE_RAW_HIGH
    )
    print(f"\nDead zone coverage: {dead_zone_mask.sum()} points ({100*dead_zone_mask.mean():.1f}%)")
    print(f"Before calibration — MAE: {before_mae:.4f}")
    print(
        f"Before dead zone range: calibrated [{before_calibrated[dead_zone_mask].min():.4f}, "
        f"{before_calibrated[dead_zone_mask].max():.4f}]  "
        f"(flat: {before_calibrated[dead_zone_mask].std():.6f} std)"
    )

    # Fit new calibration with quantile binning
    print(f"\nFitting quantile-binned isotonic regression (n_bins={args.bins})...")
    new_x_thresholds, new_y_thresholds = fit_isotonic_quantile_binned(
        raw_hi_preds, hi_true_scores, args.bins
    )

    # After: MAE with new calibration
    after_mae, after_calibrated = compute_mae_with_calibration(
        raw_hi_preds, hi_true_scores, new_x_thresholds, new_y_thresholds
    )

    print(f"\nAfter calibration — MAE: {after_mae:.4f}  (delta: {after_mae - before_mae:+.4f})")
    print(
        f"After dead zone range: calibrated [{after_calibrated[dead_zone_mask].min():.4f}, "
        f"{after_calibrated[dead_zone_mask].max():.4f}]  "
        f"(std: {after_calibrated[dead_zone_mask].std():.4f})"
    )

    # Show sub-range differentiation within dead zone
    print("\nDead zone differentiation (binned):")
    for raw_low, raw_high in [
        (5.854, 6.2), (6.2, 6.6), (6.6, 7.0), (7.0, 7.35), (7.35, 7.650)
    ]:
        sub_mask = (raw_hi_preds >= raw_low) & (raw_hi_preds < raw_high)
        if sub_mask.sum() == 0:
            continue
        before_sub = before_calibrated[sub_mask].mean()
        after_sub = after_calibrated[sub_mask].mean()
        print(
            f"  raw [{raw_low:.3f},{raw_high:.3f}): n={sub_mask.sum():3d}  "
            f"before={before_sub:.3f}  after={after_sub:.3f}  "
            f"true_mean={hi_true_scores[sub_mask].mean():.3f}"
        )

    if args.dry_run:
        print("\n[DRY RUN] Not writing calibration.json")
        return

    # Validate: MAE must not regress by more than 5% relative
    relative_mae_change = (after_mae - before_mae) / before_mae
    if relative_mae_change > 0.05:
        print(
            f"\nABORTED: MAE regression {relative_mae_change:+.1%} exceeds 5% threshold. "
            f"Calibration not written."
        )
        return

    # Write back: update HI entry only, preserve other dimensions
    calibration["hostility_index"]["x_thresholds"] = [round(x, 4) for x in new_x_thresholds]
    calibration["hostility_index"]["y_thresholds"] = [round(y, 4) for y in new_y_thresholds]
    calibration["hostility_index"]["n"] = len(raw_hi_preds)
    calibration["hostility_index"]["b2_fix"] = (
        f"quantile-binned isotonic, n_bins={args.bins}, "
        f"MAE {before_mae:.4f}→{after_mae:.4f} ({relative_mae_change:+.1%})"
    )

    with open(calibration_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nWritten: {calibration_path}")
    print(f"  HI knots: {len(current_hi_cal['x_thresholds'])} → {len(new_x_thresholds)}")
    print(f"  MAE: {before_mae:.4f} → {after_mae:.4f} ({relative_mae_change:+.1%})")
    print("  b2_fix annotation added to hostility_index entry")


if __name__ == "__main__":
    main()
