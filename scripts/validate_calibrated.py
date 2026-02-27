"""
Re-run validation battery (discriminant, known-groups, confidence) using calibrated v3b outputs.

Applies score calibration (isotonic regression) and confidence calibration from
calibration.json to the raw ONNX model outputs, then computes the same metrics
as the original validation scripts.

Usage:
  python scripts/validate_calibrated.py
"""
import json
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]


def load_calibration(model_dir):
    """Load isotonic calibration maps for scores and confidence."""
    cal_path = model_dir / "calibration.json"
    if not cal_path.exists():
        print("WARNING: No calibration.json found. Using raw outputs.")
        return None
    with open(cal_path) as f:
        return json.load(f)


def interpolate(value, xs, ys):
    """Linear interpolation between calibration breakpoints."""
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if value <= xs[i]:
            t = (value - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def apply_calibration(raw_scores, raw_confs, calibration):
    """Apply isotonic calibration to scores and confidence."""
    cal_scores = np.copy(raw_scores)
    cal_confs = np.copy(raw_confs)

    if calibration is None:
        return cal_scores, cal_confs

    for i, dim in enumerate(DIMENSIONS):
        if dim not in calibration:
            continue
        dim_cal = calibration[dim]

        # Score calibration
        if "x_thresholds" in dim_cal and "y_thresholds" in dim_cal:
            xs = dim_cal["x_thresholds"]
            ys = dim_cal["y_thresholds"]
            for j in range(len(cal_scores)):
                cal_scores[j, i] = interpolate(raw_scores[j, i], xs, ys)

        # Confidence calibration
        conf_cal = dim_cal.get("confidence_calibration", {})
        if conf_cal.get("method") == "isotonic" and "x_thresholds" in conf_cal:
            cxs = conf_cal["x_thresholds"]
            cys = conf_cal["y_thresholds"]
            for j in range(len(cal_confs)):
                cal_confs[j, i] = interpolate(raw_confs[j, i], cxs, cys)

    return cal_scores, cal_confs


def load_test_records():
    """Load test split using deterministic hash."""
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


def load_onnx_and_score(model_dir, texts, max_length=128, batch_size=64):
    """Score texts with ONNX model, return raw scores and confs."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_name = config.get("model_name", "distilbert-base-uncased")
        max_length = config.get("max_length", max_length)
    else:
        model_name = "distilbert-base-uncased"

    model_path = model_dir / "model_quantized.onnx"
    if not model_path.exists():
        model_path = model_dir / "model.onnx"
    if not model_path.exists():
        # Fall back to PyTorch
        return score_pytorch(model_dir, model_name, texts, max_length, batch_size)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_scores = []
    all_confs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="np")
        results = session.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })
        all_scores.append(results[0])
        all_confs.append(results[1])

    return np.vstack(all_scores), np.vstack(all_confs)


def score_pytorch(model_dir, model_name, texts, max_length, batch_size):
    """Fallback: score with PyTorch model."""
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / "scripts"))
    from distill import PSQStudent, N_DIMS

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PSQStudent(model_name, N_DIMS)
    ckpt = model_dir / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    all_scores = []
    all_confs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, max_length=max_length, padding="max_length",
                            truncation=True, return_tensors="pt")
            scores, confs = model(enc["input_ids"], enc["attention_mask"])
            all_scores.append(scores.numpy())
            all_confs.append(confs.numpy())

    return np.vstack(all_scores), np.vstack(all_confs)


# ============================================================
# Validation 1: Discriminant validity vs VADER sentiment
# ============================================================

def validate_discriminant(test_records, cal_scores, dim_index):
    """Correlate calibrated PSQ scores with VADER sentiment."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from scipy.stats import pearsonr

    sia = SentimentIntensityAnalyzer()
    texts = [r["text"] for r in test_records]
    vader_scores = [sia.polarity_scores(t)["compound"] for t in texts]

    print("\n" + "=" * 60)
    print("DISCRIMINANT VALIDITY vs VADER SENTIMENT (CALIBRATED)")
    print("=" * 60)

    results = {}
    for i, dim in enumerate(DIMENSIONS):
        scores = cal_scores[:, i]
        r, p = pearsonr(scores, vader_scores)
        results[dim] = {"r": round(r, 3), "p": round(p, 4)}
        direction = "+" if r > 0 else "-"
        strength = "HIGH" if abs(r) > 0.5 else "MOD" if abs(r) > 0.3 else "LOW"
        print(f"  {dim:30s}  r={r:+.3f}  p={p:.4f}  [{strength}]")

    mean_abs_r = np.mean([abs(v["r"]) for v in results.values()])
    print(f"\n  Mean |r| = {mean_abs_r:.3f}")
    print(f"  {'STRONG' if mean_abs_r < 0.3 else 'WEAK'} discriminant validity "
          f"(lower |r| = more distinct from sentiment)")
    return results


# ============================================================
# Validation 2: Known-groups validity
# ============================================================

def validate_known_groups(test_records, cal_scores, dim_index):
    """Compare calibrated mean scores across source datasets."""
    from scipy.stats import f_oneway
    print("\n" + "=" * 60)
    print("KNOWN-GROUPS VALIDITY (CALIBRATED)")
    print("=" * 60)

    # Group by source
    sources = {}
    for idx, rec in enumerate(test_records):
        src = rec.get("source", "unknown")
        if src not in sources:
            sources[src] = []
        sources[src].append(idx)

    results = {}
    for i, dim in enumerate(DIMENSIONS):
        groups = {}
        for src, indices in sources.items():
            vals = [cal_scores[j, i] for j in indices]
            if len(vals) >= 5:
                groups[src] = vals

        if len(groups) < 2:
            continue

        # ANOVA
        group_vals = list(groups.values())
        f_stat, p_val = f_oneway(*group_vals)
        means = {src: round(np.mean(vals), 2) for src, vals in groups.items()}
        sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)

        results[dim] = {
            "f_stat": round(f_stat, 2),
            "p_value": round(p_val, 4),
            "means": means,
            "top": sorted_means[0][0],
            "bottom": sorted_means[-1][0],
        }
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {dim:30s}  F={f_stat:7.1f}  p={p_val:.4f} {sig}")
        print(f"    highest: {sorted_means[0][0]} ({sorted_means[0][1]:.2f})"
              f"  lowest: {sorted_means[-1][0]} ({sorted_means[-1][1]:.2f})")

    return results


# ============================================================
# Validation 3: Confidence calibration check
# ============================================================

def validate_confidence(test_records, cal_scores, cal_confs, dim_index):
    """Check if calibrated confidence predicts accuracy."""
    from scipy.stats import spearmanr
    print("\n" + "=" * 60)
    print("CONFIDENCE CALIBRATION CHECK (CALIBRATED)")
    print("=" * 60)

    results = {}
    for i, dim in enumerate(DIMENSIONS):
        preds = []
        actuals = []
        confs = []
        for idx, rec in enumerate(test_records):
            dims = rec.get("dimensions", {})
            if dim in dims:
                actual = dims[dim]
                if isinstance(actual, dict):
                    actual = actual.get("score", actual.get("value"))
                if actual is not None and not (isinstance(actual, float) and np.isnan(actual)):
                    preds.append(cal_scores[idx, i])
                    actuals.append(float(actual))
                    confs.append(cal_confs[idx, i])

        if len(preds) < 30:
            continue

        preds = np.array(preds)
        actuals = np.array(actuals)
        confs = np.array(confs)
        errors = np.abs(preds - actuals)

        # Confidence-error correlation (should be negative: high conf → low error)
        rho, p = spearmanr(confs, errors)
        # Confidence-accuracy correlation (should be positive)
        accuracy = 1.0 - errors / 5.0
        rho_acc, p_acc = spearmanr(confs, accuracy)

        status = "CORRECT" if rho < 0 else "INVERTED"
        results[dim] = {
            "n": len(preds),
            "rho_error": round(rho, 3),
            "rho_accuracy": round(rho_acc, 3),
            "mean_conf": round(float(np.mean(confs)), 3),
            "conf_range": [round(float(np.min(confs)), 3), round(float(np.max(confs)), 3)],
            "status": status,
        }
        print(f"  {dim:30s}  rho(conf,err)={rho:+.3f}  rho(conf,acc)={rho_acc:+.3f}"
              f"  conf=[{np.min(confs):.2f},{np.max(confs):.2f}]  [{status}]")

    correct = sum(1 for v in results.values() if v["status"] == "CORRECT")
    print(f"\n  {correct}/{len(results)} dimensions have CORRECT calibration")
    return results


# ============================================================
# Main
# ============================================================

def main():
    model_dir = ROOT / "models" / "psq-student"

    print("Loading test records...")
    test_records = load_test_records()
    print(f"  {len(test_records)} test records")

    print("\nScoring with ONNX model...")
    texts = [r["text"] for r in test_records]
    raw_scores, raw_confs = load_onnx_and_score(model_dir, texts)

    print("Applying calibration...")
    calibration = load_calibration(model_dir)
    cal_scores, cal_confs = apply_calibration(raw_scores, raw_confs, calibration)

    # Build dimension index
    dim_index = {d: i for i, d in enumerate(DIMENSIONS)}

    # Run all 3 validations
    disc_results = validate_discriminant(test_records, cal_scores, dim_index)
    groups_results = validate_known_groups(test_records, cal_scores, dim_index)
    conf_results = validate_confidence(test_records, cal_scores, cal_confs, dim_index)

    # Compare raw vs calibrated
    print("\n" + "=" * 60)
    print("RAW vs CALIBRATED COMPARISON")
    print("=" * 60)
    for i, dim in enumerate(DIMENSIONS):
        raw_mean = np.mean(raw_scores[:, i])
        cal_mean = np.mean(cal_scores[:, i])
        raw_std = np.std(raw_scores[:, i])
        cal_std = np.std(cal_scores[:, i])
        print(f"  {dim:30s}  raw={raw_mean:.2f}±{raw_std:.2f}  "
              f"cal={cal_mean:.2f}±{cal_std:.2f}  "
              f"Δmean={cal_mean - raw_mean:+.2f}  Δstd={cal_std - raw_std:+.2f}")

    # Save results
    all_results = {
        "mode": "calibrated",
        "n_test": len(test_records),
        "discriminant": disc_results,
        "known_groups": groups_results,
        "confidence": conf_results,
    }
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    out_path = model_dir / "validation_calibrated_results.json"
    with open(out_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
