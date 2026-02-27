"""
Test-retest reliability analysis for the PSQ student model.

For a deterministic model, exact retest is trivially r=1.0. The meaningful
analogue is perturbation stability: how consistent are scores when input text
undergoes minor, meaning-preserving transformations?

Perturbation types (5):
  1. Typo insertion — swap two adjacent characters in a random word
  2. Punctuation removal — strip all punctuation
  3. Case change — random case mutations (not just upper/lower)
  4. Word deletion — remove 1 random non-critical word
  5. Whitespace noise — add/remove extra spaces

Metrics:
  - ICC(3,1) — two-way mixed, single measures (consistency)
  - Mean absolute difference per perturbation type
  - Per-dimension stability breakdown

Usage:
  python scripts/test_retest_reliability.py
  python scripts/test_retest_reliability.py --model-type onnx    # use ONNX model
  python scripts/test_retest_reliability.py --model-type pytorch  # use best.pt
"""

import argparse
import hashlib
import json
import random
import re
import string
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIMENSIONS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]
DIM_TO_IDX = {d: i for i, d in enumerate(DIMENSIONS)}


# =====================================================================
# Perturbation functions
# =====================================================================

def perturb_typo(text, rng):
    """Swap two adjacent characters in a random word."""
    words = text.split()
    if len(words) < 2:
        return text
    # Pick a word with length >= 3
    candidates = [i for i, w in enumerate(words) if len(w) >= 3]
    if not candidates:
        return text
    idx = rng.choice(candidates)
    word = list(words[idx])
    pos = rng.randint(0, len(word) - 2)
    word[pos], word[pos + 1] = word[pos + 1], word[pos]
    words[idx] = "".join(word)
    return " ".join(words)


def perturb_punctuation(text, _rng):
    """Remove all punctuation."""
    return re.sub(r'[^\w\s]', '', text)


def perturb_case(text, rng):
    """Randomly change case of ~30% of characters."""
    result = []
    for ch in text:
        if ch.isalpha() and rng.random() < 0.3:
            result.append(ch.swapcase())
        else:
            result.append(ch)
    return "".join(result)


def perturb_word_deletion(text, rng):
    """Delete one random word (not the first or last)."""
    words = text.split()
    if len(words) <= 3:
        return text
    idx = rng.randint(1, len(words) - 2)
    return " ".join(words[:idx] + words[idx + 1:])


def perturb_whitespace(text, rng):
    """Add random extra spaces between words."""
    words = text.split()
    result = []
    for w in words:
        result.append(w)
        if rng.random() < 0.3:
            result.append("")  # extra space via join
    return " ".join(result)


PERTURBATIONS = {
    "typo": perturb_typo,
    "no_punct": perturb_punctuation,
    "case_change": perturb_case,
    "word_drop": perturb_word_deletion,
    "whitespace": perturb_whitespace,
}


# =====================================================================
# Model loading
# =====================================================================

def load_onnx_model(model_dir):
    """Load ONNX model and return a scoring function."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_path = model_dir / "model_quantized.onnx"
    if not model_path.exists():
        model_path = model_dir / "model.onnx"

    print(f"  Loading ONNX: {model_path.name}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def score_batch(texts, max_length=128):
        """Score a batch of texts, return (scores, confidences) arrays."""
        enc = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        results = session.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })
        # results[0] = scores [batch, 10], results[1] = confidences [batch, 10]
        return results[0], results[1]

    return score_batch


def load_pytorch_model(model_dir):
    """Load PyTorch checkpoint and return a scoring function."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    import torch.nn as nn

    # Inline model definition (same as distill.py)
    class PSQStudent(nn.Module):
        def __init__(self, model_name, n_dims=10):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
            hidden = self.encoder.config.hidden_size
            self.proj = nn.Sequential(
                nn.Dropout(0.1), nn.Linear(hidden, hidden // 2),
                nn.GELU(), nn.Dropout(0.1),
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

    device = "cpu"  # GPU is busy training
    print(f"  Loading PyTorch checkpoint on CPU...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Load config to get model_name
    config_path = model_dir / "v2d_config.json"
    if not config_path.exists():
        config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    model_name = config.get("model_name", "distilbert-base-uncased")

    model = PSQStudent(model_name).to(device)
    # Load v2d checkpoint if available, else best.pt
    ckpt_path = model_dir / "v2d_best.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "best.pt"
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    def score_batch(texts, max_length=128):
        enc = tokenizer(
            texts, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            s, c = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        return s.cpu().numpy(), c.cpu().numpy()

    return score_batch


# =====================================================================
# Data loading
# =====================================================================

def load_test_data():
    """Load test split using same hash-based split as distill.py."""
    data_dir = ROOT / "data"
    all_records = []

    # Load composite
    composite_path = data_dir / "composite-ground-truth.jsonl"
    if composite_path.exists():
        with open(composite_path) as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    # Load LLM labels
    llm_path = data_dir / "train-llm.jsonl"
    if llm_path.exists():
        with open(llm_path) as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    # Filter to test split (hash % 100 >= 90)
    test_records = []
    for rec in all_records:
        h = int(hashlib.md5(rec["text"].encode()).hexdigest(), 16) % 100
        if h >= 90:
            test_records.append(rec)

    return test_records


# =====================================================================
# ICC computation
# =====================================================================

def compute_icc31(ratings_matrix):
    """Compute ICC(3,1) — two-way mixed, single measures, consistency.

    ratings_matrix: (n_subjects, k_raters) — each row is one text,
    columns are [original, perturb1, perturb2, ...].

    ICC(3,1) = (MS_subjects - MS_error) / (MS_subjects + (k-1)*MS_error)

    Interpretation:
      < 0.50 = poor
      0.50–0.75 = moderate
      0.75–0.90 = good
      > 0.90 = excellent
    """
    n, k = ratings_matrix.shape
    if n < 2 or k < 2:
        return np.nan

    # Grand mean
    grand_mean = np.mean(ratings_matrix)

    # Mean squares
    row_means = np.mean(ratings_matrix, axis=1)
    col_means = np.mean(ratings_matrix, axis=0)

    SS_subjects = k * np.sum((row_means - grand_mean) ** 2)
    SS_raters = n * np.sum((col_means - grand_mean) ** 2)
    SS_total = np.sum((ratings_matrix - grand_mean) ** 2)
    SS_error = SS_total - SS_subjects - SS_raters

    df_subjects = n - 1
    df_error = (n - 1) * (k - 1)

    MS_subjects = SS_subjects / df_subjects if df_subjects > 0 else 0
    MS_error = SS_error / df_error if df_error > 0 else 0

    denom = MS_subjects + (k - 1) * MS_error
    if denom == 0:
        return np.nan

    icc = (MS_subjects - MS_error) / denom
    return icc


# =====================================================================
# Main analysis
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="PSQ test-retest reliability")
    parser.add_argument("--model-type", choices=["onnx", "pytorch"], default="onnx",
                        help="Model type to use (default: onnx)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Max number of test samples to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    model_dir = ROOT / "models" / "psq-student"

    # Load model
    print(f"\nLoading model ({args.model_type})...")
    if args.model_type == "onnx":
        score_fn = load_onnx_model(model_dir)
    else:
        score_fn = load_pytorch_model(model_dir)

    # Load test data
    print("\nLoading test data...")
    test_records = load_test_data()
    print(f"  Total test records: {len(test_records)}")

    # Subsample if needed
    if len(test_records) > args.n_samples:
        rng.shuffle(test_records)
        test_records = test_records[:args.n_samples]
    print(f"  Using: {len(test_records)} samples")

    texts = [r["text"] for r in test_records]

    # Extract ground truth dimensions per record (for filtering)
    record_dims = []
    for rec in test_records:
        dims = set()
        for d, v in rec.get("dimensions", {}).items():
            if d in DIM_TO_IDX and v.get("score") is not None:
                dims.add(d)
        record_dims.append(dims)

    # Score original texts
    print("\nScoring original texts...")
    t0 = time.time()
    all_orig_scores = []
    all_orig_confs = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        s, c = score_fn(batch)
        all_orig_scores.append(s)
        all_orig_confs.append(c)
    orig_scores = np.vstack(all_orig_scores)   # [n, 10]
    orig_confs = np.vstack(all_orig_confs)      # [n, 10]
    print(f"  Done in {time.time() - t0:.1f}s")

    # Score perturbed versions
    perturbation_scores = {}  # {perturb_name: scores_array [n, 10]}
    perturbation_confs = {}

    for pname, pfunc in PERTURBATIONS.items():
        print(f"\nScoring perturbation: {pname}...")
        perturbed_texts = [pfunc(t, rng) for t in texts]

        t0 = time.time()
        all_scores = []
        all_confs = []
        for i in range(0, len(perturbed_texts), args.batch_size):
            batch = perturbed_texts[i:i + args.batch_size]
            s, c = score_fn(batch)
            all_scores.append(s)
            all_confs.append(c)
        perturbation_scores[pname] = np.vstack(all_scores)
        perturbation_confs[pname] = np.vstack(all_confs)
        print(f"  Done in {time.time() - t0:.1f}s")

    # ===================================================================
    # Analysis
    # ===================================================================
    n_samples = len(texts)
    n_perturbations = len(PERTURBATIONS)
    perturb_names = list(PERTURBATIONS.keys())

    print("\n" + "=" * 70)
    print("TEST-RETEST RELIABILITY ANALYSIS (Perturbation Stability)")
    print("=" * 70)
    print(f"Model: {args.model_type}")
    print(f"Samples: {n_samples}")
    print(f"Perturbation types: {n_perturbations}")
    print(f"Total scorings: {n_samples * (1 + n_perturbations):,}")

    # --- Per-dimension ICC across all perturbations ---
    print("\n--- ICC(3,1) per Dimension (across all perturbation types) ---")
    print(f"{'Dimension':<25} {'ICC(3,1)':>8}  {'Interpret':<12}  {'MAD':>6}  {'Max Δ':>6}")
    print("-" * 70)

    dim_iccs = {}
    dim_mads = {}

    for dim_name in DIMENSIONS:
        dim_idx = DIM_TO_IDX[dim_name]

        # Build ratings matrix: [n_samples, 1+n_perturbations]
        # Columns: original, typo, no_punct, case_change, word_drop, whitespace
        ratings = np.zeros((n_samples, 1 + n_perturbations))
        ratings[:, 0] = orig_scores[:, dim_idx]
        for j, pname in enumerate(perturb_names):
            ratings[:, j + 1] = perturbation_scores[pname][:, dim_idx]

        icc = compute_icc31(ratings)
        dim_iccs[dim_name] = icc

        # Mean absolute difference from original
        diffs = np.abs(ratings[:, 1:] - ratings[:, 0:1])
        mad = np.mean(diffs)
        max_diff = np.max(diffs)
        dim_mads[dim_name] = mad

        if icc >= 0.90:
            interp = "Excellent"
        elif icc >= 0.75:
            interp = "Good"
        elif icc >= 0.50:
            interp = "Moderate"
        else:
            interp = "Poor"

        print(f"  {dim_name:<23} {icc:>8.3f}  {interp:<12}  {mad:>6.3f}  {max_diff:>6.2f}")

    avg_icc = np.mean(list(dim_iccs.values()))
    avg_mad = np.mean(list(dim_mads.values()))
    print(f"\n  {'AVERAGE':<23} {avg_icc:>8.3f}  {'':12}  {avg_mad:>6.3f}")

    # --- Per-perturbation breakdown ---
    print("\n--- Mean Absolute Difference by Perturbation Type ---")
    print(f"{'Perturbation':<15}", end="")
    for d in DIMENSIONS:
        print(f"  {d[:4]:>5}", end="")
    print(f"  {'AVG':>5}")
    print("-" * (15 + 6 * 11))

    for pname in perturb_names:
        print(f"  {pname:<13}", end="")
        diffs_per_dim = []
        for dim_name in DIMENSIONS:
            dim_idx = DIM_TO_IDX[dim_name]
            diff = np.mean(np.abs(perturbation_scores[pname][:, dim_idx] - orig_scores[:, dim_idx]))
            diffs_per_dim.append(diff)
            print(f"  {diff:>5.3f}", end="")
        print(f"  {np.mean(diffs_per_dim):>5.3f}")

    # --- Confidence stability ---
    print("\n--- Confidence Stability (Mean Δ confidence) ---")
    print(f"{'Perturbation':<15}", end="")
    for d in DIMENSIONS:
        print(f"  {d[:4]:>5}", end="")
    print(f"  {'AVG':>5}")
    print("-" * (15 + 6 * 11))

    for pname in perturb_names:
        print(f"  {pname:<13}", end="")
        diffs_per_dim = []
        for dim_name in DIMENSIONS:
            dim_idx = DIM_TO_IDX[dim_name]
            diff = np.mean(np.abs(perturbation_confs[pname][:, dim_idx] - orig_confs[:, dim_idx]))
            diffs_per_dim.append(diff)
            print(f"  {diff:>5.3f}", end="")
        print(f"  {np.mean(diffs_per_dim):>5.3f}")

    # --- Pairwise Pearson r (original vs each perturbation) ---
    print("\n--- Pearson r (original vs perturbed) per Dimension ---")
    print(f"{'Perturbation':<15}", end="")
    for d in DIMENSIONS:
        print(f"  {d[:4]:>5}", end="")
    print(f"  {'AVG':>5}")
    print("-" * (15 + 6 * 11))

    for pname in perturb_names:
        print(f"  {pname:<13}", end="")
        rs = []
        for dim_name in DIMENSIONS:
            dim_idx = DIM_TO_IDX[dim_name]
            from scipy.stats import pearsonr
            r, _ = pearsonr(orig_scores[:, dim_idx], perturbation_scores[pname][:, dim_idx])
            rs.append(r)
            print(f"  {r:>5.3f}", end="")
        print(f"  {np.mean(rs):>5.3f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if avg_icc >= 0.90:
        verdict = "EXCELLENT — scores are highly stable under perturbation"
    elif avg_icc >= 0.75:
        verdict = "GOOD — scores are reasonably stable, minor surface sensitivity"
    elif avg_icc >= 0.50:
        verdict = "MODERATE — some dimensions sensitive to surface-level changes"
    else:
        verdict = "POOR — model relies heavily on surface features, not construct"

    print(f"  Average ICC(3,1): {avg_icc:.3f}")
    print(f"  Average MAD:      {avg_mad:.3f} (on 0-10 scale)")
    print(f"  Verdict:          {verdict}")
    print()

    # Identify most/least stable dimensions
    sorted_dims = sorted(dim_iccs.items(), key=lambda x: x[1], reverse=True)
    print(f"  Most stable:  {sorted_dims[0][0]} (ICC={sorted_dims[0][1]:.3f})")
    print(f"  Least stable: {sorted_dims[-1][0]} (ICC={sorted_dims[-1][1]:.3f})")

    # Interpretation notes
    print("\n  Interpretation notes:")
    print("  - ICC(3,1) > 0.75 is the standard threshold for 'good' reliability")
    print("  - MAD < 0.5 on a 0-10 scale means < 5% shift from perturbations")
    print("  - Perturbation stability is the neural model analogue of test-retest")
    print("  - Low ICC may indicate reliance on surface features (punctuation,")
    print("    specific words) rather than underlying construct")

    # Save results
    results = {
        "model_type": args.model_type,
        "n_samples": n_samples,
        "n_perturbation_types": n_perturbations,
        "perturbation_types": perturb_names,
        "per_dimension": {
            dim: {
                "icc_31": round(float(dim_iccs[dim]), 4),
                "mad": round(float(dim_mads[dim]), 4),
            }
            for dim in DIMENSIONS
        },
        "average_icc": round(float(avg_icc), 4),
        "average_mad": round(float(avg_mad), 4),
        "verdict": verdict,
    }

    out_path = model_dir / "test_retest_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
