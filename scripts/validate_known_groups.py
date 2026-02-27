"""
Known-groups validity for the PSQ student model.

Tests whether the model correctly differentiates texts that should
score differently on PSQ dimensions:

Groups (by source dataset):
  - Berkeley hate speech (high threat, high hostility)
  - ProsocialDialog (high protective factors)
  - Civil Comments neutral (moderate/baseline)
  - LLM-labeled (diverse, known ground truth)

For each group, we verify:
  1. Mean scores are in the expected direction
  2. Group separation (Cohen's d) is meaningful
  3. Ranking of groups matches theory
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

THREAT_DIMS = {"threat_exposure", "hostility_index", "authority_dynamics", "energy_dissipation"}
PROTECTIVE_DIMS = {"regulatory_capacity", "resilience_baseline", "trust_conditions",
                   "cooling_capacity", "defensive_architecture", "contractual_clarity"}


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def load_records_by_source():
    """Load all records, split by source, filter to test set."""
    data_dir = ROOT / "data"
    groups = {
        "berkeley": [],
        "civil_comments": [],
        "goemotions": [],
        "prosocial": [],
        "politeness": [],
        "llm": [],
        "other": [],
    }

    # Load composite
    composite_path = data_dir / "composite-ground-truth.jsonl"
    if composite_path.exists():
        with open(composite_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                h = int(hashlib.md5(rec["text"].encode()).hexdigest(), 16) % 100
                if h < 90:
                    continue  # only test split
                source = rec.get("source", rec.get("teacher", "other"))
                if "berkeley" in source:
                    groups["berkeley"].append(rec)
                elif "civil" in source:
                    groups["civil_comments"].append(rec)
                elif "goemotion" in source:
                    groups["goemotions"].append(rec)
                elif "prosocial" in source:
                    groups["prosocial"].append(rec)
                elif "politeness" in source:
                    groups["politeness"].append(rec)
                else:
                    groups["other"].append(rec)

    # Load LLM labels
    llm_path = data_dir / "train-llm.jsonl"
    if llm_path.exists():
        with open(llm_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                h = int(hashlib.md5(rec["text"].encode()).hexdigest(), 16) % 100
                if h >= 90:
                    groups["llm"].append(rec)

    return groups


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


def score_group(texts, score_fn, batch_size=64):
    """Score a list of texts, return (scores, confs) arrays."""
    if not texts:
        return np.array([]).reshape(0, 10), np.array([]).reshape(0, 10)
    all_scores, all_confs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        s, c = score_fn(batch)
        all_scores.append(s)
        all_confs.append(c)
    return np.vstack(all_scores), np.vstack(all_confs)


def main():
    np.random.seed(42)
    model_dir = ROOT / "models" / "psq-student"

    # Load data by source
    print("Loading test data by source...")
    groups = load_records_by_source()
    for name, recs in groups.items():
        print(f"  {name}: {len(recs)} test records")

    # Load model
    print("\nLoading ONNX model...")
    score_fn = load_onnx_model(model_dir)

    # Score each group
    print("\nScoring groups...")
    group_scores = {}
    group_confs = {}
    group_texts = {}
    for name, recs in groups.items():
        if len(recs) < 10:
            print(f"  Skipping {name} (only {len(recs)} records)")
            continue
        texts = [r["text"] for r in recs]
        # Cap at 500 per group for speed
        if len(texts) > 500:
            np.random.shuffle(texts)
            texts = texts[:500]
        group_texts[name] = texts
        t0 = time.time()
        s, c = score_group(texts, score_fn)
        group_scores[name] = s
        group_confs[name] = c
        print(f"  {name}: {len(texts)} texts scored in {time.time() - t0:.1f}s")

    active_groups = list(group_scores.keys())

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 80)
    print("KNOWN-GROUPS VALIDITY ANALYSIS")
    print("=" * 80)
    print(f"Groups: {', '.join(f'{g} (n={len(group_texts[g])})' for g in active_groups)}")

    # --- Mean scores per group per dimension ---
    print("\n--- Mean Predicted Score by Group and Dimension ---")
    header = f"  {'Dimension':<25}"
    for g in active_groups:
        header += f"  {g[:8]:>8}"
    print(header)
    print("-" * (27 + 10 * len(active_groups)))

    dim_group_means = {}
    for dim in DIMENSIONS:
        dim_idx = DIM_TO_IDX[dim]
        dim_group_means[dim] = {}
        row = f"  {dim:<25}"
        for g in active_groups:
            mean = np.mean(group_scores[g][:, dim_idx])
            dim_group_means[dim][g] = float(mean)
            row += f"  {mean:>8.2f}"
        print(row)

    # --- Theoretical predictions and tests ---
    print("\n--- Theoretical Predictions ---")

    predictions = []

    # Prediction 1: Berkeley (hate speech) should have HIGH threat_exposure and hostility_index
    if "berkeley" in active_groups and "prosocial" in active_groups:
        predictions.append({
            "prediction": "Berkeley > Prosocial on threat dimensions",
            "dims": ["threat_exposure", "hostility_index"],
            "high_group": "berkeley",
            "low_group": "prosocial",
        })

    # Prediction 2: Prosocial should have HIGH protective factors
    if "prosocial" in active_groups and "berkeley" in active_groups:
        predictions.append({
            "prediction": "Prosocial > Berkeley on protective dimensions",
            "dims": ["regulatory_capacity", "resilience_baseline", "trust_conditions", "cooling_capacity"],
            "high_group": "prosocial",
            "low_group": "berkeley",
        })

    # Prediction 3: Civil Comments (neutral-ish) should be between Berkeley and Prosocial
    if "civil_comments" in active_groups and "berkeley" in active_groups:
        predictions.append({
            "prediction": "Berkeley > Civil Comments on hostility_index",
            "dims": ["hostility_index"],
            "high_group": "berkeley",
            "low_group": "civil_comments",
        })

    # Prediction 4: Politeness should score lower on authority_dynamics (polite = healthy power)
    if "politeness" in active_groups and "berkeley" in active_groups:
        predictions.append({
            "prediction": "Berkeley > Politeness on authority_dynamics",
            "dims": ["authority_dynamics"],
            "high_group": "berkeley",
            "low_group": "politeness",
        })

    print(f"\n  Testing {len(predictions)} theoretical predictions:")
    print(f"  {'Prediction':<55} {'Dim':<20} {'d':>6} {'p':>10} {'Result'}")
    print("-" * 105)

    n_confirmed = 0
    n_total = 0
    all_prediction_results = []

    for pred in predictions:
        for dim in pred["dims"]:
            dim_idx = DIM_TO_IDX[dim]
            high_scores = group_scores[pred["high_group"]][:, dim_idx]
            low_scores = group_scores[pred["low_group"]][:, dim_idx]

            d = cohens_d(high_scores, low_scores)
            t_stat, p_val = stats.ttest_ind(high_scores, low_scores)

            confirmed = d > 0.2 and p_val < 0.05
            if confirmed:
                n_confirmed += 1
            n_total += 1

            effect = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            result = f"CONFIRMED ({effect})" if confirmed else f"NOT confirmed ({effect})"

            print(f"  {pred['prediction'][:53]:<55} {dim[:18]:<20} {d:>6.2f} {p_val:>10.2e} {result} {sig}")

            all_prediction_results.append({
                "prediction": pred["prediction"],
                "dimension": dim,
                "high_group": pred["high_group"],
                "low_group": pred["low_group"],
                "cohens_d": round(float(d), 3),
                "p_value": float(p_val),
                "confirmed": bool(confirmed),
                "effect_size": effect,
            })

    # --- Group separation: one-way ANOVA per dimension ---
    print("\n--- Group Separation (One-Way ANOVA across all groups) ---")
    print(f"  {'Dimension':<25} {'F':>8} {'p':>10} {'η²':>6}  {'Separation'}")
    print("-" * 65)

    anova_results = {}
    for dim in DIMENSIONS:
        dim_idx = DIM_TO_IDX[dim]
        group_vals = [group_scores[g][:, dim_idx] for g in active_groups if len(group_scores[g]) > 0]

        if len(group_vals) < 2:
            continue

        F, p = stats.f_oneway(*group_vals)

        # Eta squared (effect size for ANOVA)
        all_vals = np.concatenate(group_vals)
        grand_mean = np.mean(all_vals)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_vals)
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        if eta_sq > 0.14:
            sep = "Large"
        elif eta_sq > 0.06:
            sep = "Medium"
        elif eta_sq > 0.01:
            sep = "Small"
        else:
            sep = "Negligible"

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        anova_results[dim] = {"F": float(F), "p": float(p), "eta_sq": round(float(eta_sq), 4), "separation": sep}
        print(f"  {dim:<25} {F:>8.1f} {p:>10.2e} {eta_sq:>6.3f}  {sep} {sig}")

    # --- Rank order test ---
    print("\n--- Group Rank Order (by mean score) ---")
    print("  Expected for threat dims: berkeley > civil > prosocial")
    print("  Expected for protective dims: prosocial > civil > berkeley")

    for dim in DIMENSIONS:
        dim_idx = DIM_TO_IDX[dim]
        group_means = [(g, np.mean(group_scores[g][:, dim_idx])) for g in active_groups]
        group_means.sort(key=lambda x: x[1], reverse=True)
        ranking = " > ".join(f"{g[0][:6]}({g[1]:.1f})" for g in group_means)
        dim_type = "THREAT" if dim in THREAT_DIMS else "PROTECT"
        print(f"  {dim:<25} [{dim_type}]  {ranking}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    pct = (n_confirmed / n_total * 100) if n_total > 0 else 0
    print(f"  Theoretical predictions confirmed: {n_confirmed}/{n_total} ({pct:.0f}%)")

    n_large_sep = sum(1 for r in anova_results.values() if r["separation"] in ("Large", "Medium"))
    print(f"  Dimensions with medium+ group separation: {n_large_sep}/{len(anova_results)}")

    if pct >= 80:
        verdict = "STRONG known-groups validity — model differentiates groups as theory predicts"
    elif pct >= 60:
        verdict = "MODERATE known-groups validity — most predictions confirmed"
    elif pct >= 40:
        verdict = "PARTIAL known-groups validity — some predictions confirmed"
    else:
        verdict = "WEAK known-groups validity — model does not differentiate groups as expected"
    print(f"  Verdict: {verdict}")

    # Save results
    results = {
        "groups": {g: len(group_texts.get(g, [])) for g in active_groups},
        "group_means": dim_group_means,
        "predictions": all_prediction_results,
        "predictions_confirmed": n_confirmed,
        "predictions_total": n_total,
        "anova_results": anova_results,
        "verdict": verdict,
    }
    out_path = model_dir / "known_groups_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
