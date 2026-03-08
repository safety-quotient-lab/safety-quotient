"""
Recalibrate historical PSQ scores from one calibration version to another.

Maps scores through: old_calibrated → raw (inverse of old model) → new_calibrated.
Primary input is raw_score (available in v3+ API responses). Falls back to inverting
old calibration when raw_score is unavailable.

Usage:
    # Single score
    python scripts/recalibrate.py --dim threat_exposure --score 6.5 --from v2 --to v3

    # Batch (JSON on stdin)
    echo '[{"dimension": "threat_exposure", "calibrated_score": 6.5}]' | \
        python scripts/recalibrate.py --from v2 --to v3 --batch

    # Using raw score directly (preferred — no inversion needed)
    python scripts/recalibrate.py --dim threat_exposure --raw-score 4.2 --to v3

See distillation-research.md §72 (B3 recalibration) for context.
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path


def load_calibration(version, model_dir="models/psq-student"):
    """Load a calibration file by version label."""
    path = Path(model_dir) / f"calibration-{version}.json"
    if not path.exists():
        # Try the default (unlabeled) calibration.json
        if version == "current":
            path = Path(model_dir) / "calibration.json"
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
    with open(path) as f:
        return json.load(f)


def calibration_forward(raw_score, thresholds):
    """Apply calibration: raw_score → calibrated_score via piecewise linear interpolation."""
    x = np.array(thresholds["x_thresholds"])
    y = np.array(thresholds["y_thresholds"])
    return float(np.clip(np.interp(raw_score, x, y), 0, 10))


def calibration_inverse(calibrated_score, thresholds):
    """Invert calibration: calibrated_score → raw_score.

    Since isotonic regression is piecewise constant (not strictly monotonic),
    the inverse is set-valued. We return the midpoint of the preimage interval.
    """
    x = np.array(thresholds["x_thresholds"])
    y = np.array(thresholds["y_thresholds"])

    # Find all x values that map to this y (or bracket it)
    if calibrated_score <= y[0]:
        return float(x[0])
    if calibrated_score >= y[-1]:
        return float(x[-1])

    # Find the segment where y crosses calibrated_score
    for i in range(len(y) - 1):
        if y[i] <= calibrated_score <= y[i + 1]:
            if abs(y[i + 1] - y[i]) < 1e-6:
                # Flat segment — return midpoint of x range
                return float((x[i] + x[i + 1]) / 2)
            # Linear interpolation within segment
            fraction = (calibrated_score - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + fraction * (x[i + 1] - x[i]))

    return float(x[-1])


def recalibrate(dimension, calibrated_score=None, raw_score=None,
                old_cal=None, new_cal=None):
    """Recalibrate a single score from old calibration to new.

    Args:
        dimension: PSQ dimension name
        calibrated_score: score under old calibration (inverted to get raw)
        raw_score: raw model output (preferred — no inversion needed)
        old_cal: old calibration dict (needed if using calibrated_score)
        new_cal: new calibration dict

    Returns:
        dict with raw_score, old_calibrated, new_calibrated
    """
    if raw_score is not None:
        raw = raw_score
    elif calibrated_score is not None and old_cal is not None:
        if dimension not in old_cal:
            raise ValueError(f"Dimension {dimension} not in old calibration")
        raw = calibration_inverse(calibrated_score, old_cal[dimension])
    else:
        raise ValueError("Must provide either raw_score or (calibrated_score + old_cal)")

    if dimension not in new_cal:
        raise ValueError(f"Dimension {dimension} not in new calibration")

    new_calibrated = calibration_forward(raw, new_cal[dimension])

    result = {
        "dimension": dimension,
        "raw_score": round(raw, 4),
        "new_calibrated": round(new_calibrated, 4),
    }

    if calibrated_score is not None:
        result["old_calibrated"] = round(calibrated_score, 4)

    return result


def main():
    parser = argparse.ArgumentParser(description="Recalibrate historical PSQ scores")
    parser.add_argument("--dim", help="Dimension name")
    parser.add_argument("--score", type=float, help="Old calibrated score to recalibrate")
    parser.add_argument("--raw-score", type=float, help="Raw model score (preferred)")
    parser.add_argument("--from", dest="from_version", default="v2",
                        help="Old calibration version (default: v2)")
    parser.add_argument("--to", dest="to_version", default="v3",
                        help="New calibration version (default: v3)")
    parser.add_argument("--model-dir", default="models/psq-student",
                        help="Model directory containing calibration files")
    parser.add_argument("--batch", action="store_true",
                        help="Read JSON array from stdin")
    args = parser.parse_args()

    new_cal = load_calibration(args.to_version, args.model_dir)
    old_cal = None
    if args.from_version:
        old_cal = load_calibration(args.from_version, args.model_dir)

    if args.batch:
        records = json.load(sys.stdin)
        results = []
        for rec in records:
            result = recalibrate(
                dimension=rec["dimension"],
                calibrated_score=rec.get("calibrated_score"),
                raw_score=rec.get("raw_score"),
                old_cal=old_cal,
                new_cal=new_cal,
            )
            results.append(result)
        print(json.dumps(results, indent=2))
    else:
        if not args.dim:
            parser.error("--dim required for single-score mode")
        result = recalibrate(
            dimension=args.dim,
            calibrated_score=args.score,
            raw_score=args.raw_score,
            old_cal=old_cal,
            new_cal=new_cal,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
