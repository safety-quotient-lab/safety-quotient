"""
Discriminant validity: PSQ dimensions vs simple sentiment.

If PSQ is just measuring "positive vs negative," we'd see high correlations
with sentiment across all dimensions. If PSQ captures construct-specific
information, most dimensions should show low-to-moderate sentiment correlation.

Uses VADER (rule-based, no training) as the sentiment baseline.
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

# Protective factors score HIGH when environment is safe (positive sentiment)
# Threat factors score HIGH when environment is threatening (negative sentiment)
THREAT_DIMS = {"threat_exposure", "hostility_index", "authority_dynamics", "energy_dissipation"}
PROTECTIVE_DIMS = {"regulatory_capacity", "resilience_baseline", "trust_conditions",
                   "cooling_capacity", "defensive_architecture", "contractual_clarity"}


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


def get_vader_scores(texts):
    """Score texts with VADER sentiment analyzer."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)

    sia = SentimentIntensityAnalyzer()
    scores = []
    for text in texts:
        vs = sia.polarity_scores(text)
        scores.append({
            "compound": vs["compound"],   # -1 to +1
            "pos": vs["pos"],
            "neg": vs["neg"],
            "neu": vs["neu"],
        })
    return scores


def main():
    np.random.seed(42)
    model_dir = ROOT / "models" / "psq-student"
    batch_size = 64

    # Load data
    print("Loading test data...")
    test_records = load_test_data()
    print(f"  {len(test_records)} test records")

    # Subsample for speed
    if len(test_records) > 800:
        np.random.shuffle(test_records)
        test_records = test_records[:800]
    print(f"  Using: {len(test_records)} samples")

    texts = [r["text"] for r in test_records]

    # Get ground truth dimensions per record
    gt_dims = []
    for rec in test_records:
        dims = {}
        for d, v in rec.get("dimensions", {}).items():
            if d in DIM_TO_IDX and v.get("score") is not None:
                dims[d] = v["score"]
        gt_dims.append(dims)

    # Score with PSQ model
    print("\nScoring with PSQ model (ONNX)...")
    score_fn = load_onnx_model(model_dir)
    t0 = time.time()
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        s, c = score_fn(batch)
        all_scores.append(s)
    psq_scores = np.vstack(all_scores)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Score with VADER
    print("\nScoring with VADER sentiment...")
    t0 = time.time()
    vader_scores = get_vader_scores(texts)
    vader_compound = np.array([v["compound"] for v in vader_scores])
    vader_neg = np.array([v["neg"] for v in vader_scores])
    vader_pos = np.array([v["pos"] for v in vader_scores])
    print(f"  Done in {time.time() - t0:.1f}s")

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 75)
    print("DISCRIMINANT VALIDITY: PSQ DIMENSIONS vs SENTIMENT")
    print("=" * 75)
    print(f"Samples: {len(texts)}")
    print(f"Sentiment model: VADER (rule-based, compound score)")
    print()

    # --- PSQ predicted scores vs VADER compound ---
    print("--- PSQ Predicted Score vs VADER Compound (all samples) ---")
    print(f"  {'Dimension':<25} {'r':>6} {'p':>10}  {'Type':<12}  {'Expected r sign':<16}  {'Match'}")
    print("-" * 85)

    correlations = {}
    for dim in DIMENSIONS:
        dim_idx = DIM_TO_IDX[dim]
        psq_dim = psq_scores[:, dim_idx]

        r, p = stats.pearsonr(psq_dim, vader_compound)
        correlations[dim] = {"r": r, "p": p}

        dim_type = "Threat" if dim in THREAT_DIMS else "Protective"
        # Threat dims: high score = more threat = negative sentiment → expect r < 0
        # Protective dims: high score = more safety = positive sentiment → expect r > 0
        expected_sign = "negative" if dim in THREAT_DIMS else "positive"
        actual_sign = "negative" if r < 0 else "positive"
        match = "YES" if expected_sign == actual_sign else "NO"

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {dim:<25} {r:>6.3f} {p:>10.2e}  {dim_type:<12}  {expected_sign:<16}  {match} {sig}")

    # --- Key metric: how many dimensions are DISTINCT from sentiment? ---
    print("\n--- Discriminant Validity Summary ---")
    n_low = sum(1 for d in correlations.values() if abs(d["r"]) < 0.30)
    n_moderate = sum(1 for d in correlations.values() if 0.30 <= abs(d["r"]) < 0.60)
    n_high = sum(1 for d in correlations.values() if abs(d["r"]) >= 0.60)

    print(f"  Low correlation with sentiment (|r| < 0.30):     {n_low}/10 dimensions")
    print(f"  Moderate correlation (0.30 ≤ |r| < 0.60):        {n_moderate}/10 dimensions")
    print(f"  High correlation (|r| ≥ 0.60):                   {n_high}/10 dimensions")

    avg_abs_r = np.mean([abs(d["r"]) for d in correlations.values()])
    print(f"\n  Mean |r| with sentiment: {avg_abs_r:.3f}")

    if avg_abs_r < 0.30:
        verdict = "STRONG discriminant validity — PSQ is clearly distinct from sentiment"
    elif avg_abs_r < 0.50:
        verdict = "MODERATE discriminant validity — PSQ overlaps with sentiment but adds information"
    else:
        verdict = "WEAK discriminant validity — PSQ may be largely measuring sentiment"
    print(f"  Verdict: {verdict}")

    # --- Ground truth scores vs VADER (where available) ---
    print("\n--- Ground Truth Score vs VADER Compound (where GT available) ---")
    print(f"  {'Dimension':<25} {'r':>6} {'n':>6}  {'Note'}")
    print("-" * 60)

    for dim in DIMENSIONS:
        gt_vals = []
        vader_vals = []
        for i, rec_dims in enumerate(gt_dims):
            if dim in rec_dims:
                gt_vals.append(rec_dims[dim])
                vader_vals.append(vader_compound[i])
        if len(gt_vals) >= 20:
            r, p = stats.pearsonr(gt_vals, vader_vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {dim:<25} {r:>6.3f} {len(gt_vals):>6}  {sig}")
        else:
            print(f"  {dim:<25}    --  {len(gt_vals):>6}  (too few samples)")

    # --- Incremental validity: does PSQ predict GT beyond sentiment? ---
    print("\n--- Incremental Validity (PSQ over sentiment) ---")
    print("  For each dimension: R² of (VADER alone) vs R² of (VADER + PSQ predicted)")
    print(f"  {'Dimension':<25} {'R²_sent':>8} {'R²_sent+psq':>12} {'ΔR²':>6} {'Incremental'}")
    print("-" * 70)

    for dim in DIMENSIONS:
        gt_vals = []
        vader_vals = []
        psq_vals = []
        dim_idx = DIM_TO_IDX[dim]
        for i, rec_dims in enumerate(gt_dims):
            if dim in rec_dims:
                gt_vals.append(rec_dims[dim])
                vader_vals.append(vader_compound[i])
                psq_vals.append(psq_scores[i, dim_idx])
        if len(gt_vals) >= 30:
            gt_arr = np.array(gt_vals)
            vader_arr = np.array(vader_vals)
            psq_arr = np.array(psq_vals)

            # R² for sentiment alone
            r_sent = stats.pearsonr(vader_arr, gt_arr)[0]
            r2_sent = r_sent ** 2

            # R² for PSQ alone
            r_psq = stats.pearsonr(psq_arr, gt_arr)[0]
            r2_psq = r_psq ** 2

            # Simple incremental: R² of PSQ over sentiment
            delta_r2 = r2_psq - r2_sent
            incremental = "YES" if delta_r2 > 0.02 else "marginal" if delta_r2 > 0 else "NO"

            print(f"  {dim:<25} {r2_sent:>8.3f} {r2_psq:>12.3f} {delta_r2:>+6.3f} {incremental}")
        else:
            print(f"  {dim:<25}     --           --     --   (n={len(gt_vals)})")

    # Save results
    results = {
        "n_samples": len(texts),
        "sentiment_model": "VADER",
        "correlations": {d: {"r": round(float(v["r"]), 4), "p": float(v["p"])}
                         for d, v in correlations.items()},
        "mean_abs_r": round(float(avg_abs_r), 4),
        "n_low_correlation": n_low,
        "n_moderate_correlation": n_moderate,
        "n_high_correlation": n_high,
        "verdict": verdict,
    }
    out_path = model_dir / "discriminant_validity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
