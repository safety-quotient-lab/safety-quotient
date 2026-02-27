"""
Benchmark alternative ground truth datasets against our multi-signal proxy stack.

For each candidate dataset, compute correlations between our proxy signals
(detoxify, sentiment, emotion) and the dataset's ground truth labels.

Goal: find which dataset gives the highest correlation ceiling, indicating
it's the best ground truth for training a PSQ student model.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

N_SAMPLE = 500  # per dataset


def corr(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 10:
        return float("nan")
    return stats.pearsonr(x[mask], y[mask])[0]


def load_proxy_signals(texts):
    """Run all proxy models on texts, return dict of signal arrays."""
    from detoxify import Detoxify
    from transformers import pipeline, logging
    logging.set_verbosity_error()

    signals = {}

    # Detoxify
    print("    detoxify...", end="", flush=True)
    model = Detoxify("original")
    detox = model.predict(texts)
    for attr in ["toxicity", "severe_toxicity", "insult", "threat", "identity_attack", "obscene"]:
        signals[f"detox_{attr}"] = np.array(detox[attr])
    del model
    print(" done")

    # Sentiment
    print("    sentiment...", end="", flush=True)
    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                     device=-1, truncation=True, max_length=512, top_k=None)
    preds = pipe(texts, batch_size=32)
    signals["sent_negative"] = np.array([next((x["score"] for x in p if x["label"] == "negative"), 0) for p in preds])
    signals["sent_positive"] = np.array([next((x["score"] for x in p if x["label"] == "positive"), 0) for p in preds])
    del pipe
    print(" done")

    # Emotion
    print("    emotion...", end="", flush=True)
    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest",
                     device=-1, truncation=True, max_length=512, top_k=None)
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id or 1
    preds = pipe(texts, batch_size=16)
    for emo in ["anger", "fear", "joy", "disgust", "sadness", "surprise"]:
        signals[f"emo_{emo}"] = np.array([next((x["score"] for x in p if x["label"] == emo), 0) for p in preds])
    del pipe
    print(" done")

    return signals


# =====================================================================
# DATASET 1: Civil Comments
# =====================================================================
def bench_civil_comments(signals_func):
    print("\n" + "=" * 70)
    print("DATASET: Jigsaw Civil Comments (google/civil_comments)")
    print("  Continuous 0-1 scores, 7 attributes, ~2M comments")
    print("=" * 70)

    from datasets import load_dataset
    print("  Loading...", end="", flush=True)
    ds = load_dataset("google/civil_comments", split="train", trust_remote_code=True)
    print(f" {len(ds)} rows")

    # Stratified sample by toxicity
    df = ds.to_pandas()
    df = df[df["text"].str.len() > 20].copy()
    # Create strata
    df["stratum"] = pd.cut(df["toxicity"], bins=[0, 0.1, 0.3, 0.5, 0.8, 1.0],
                            labels=["clean", "mild", "moderate", "high", "extreme"])
    samples = []
    for s in df["stratum"].unique():
        subset = df[df["stratum"] == s]
        take = min(N_SAMPLE // 5, len(subset))
        if take > 0:
            samples.append(subset.sample(n=take, random_state=42))
    sample = pd.concat(samples).reset_index(drop=True)
    print(f"  Stratified sample: {len(sample)} texts")

    texts = sample["text"].tolist()
    print("  Running proxy signals:")
    signals = signals_func(texts)

    # Correlate all signals with all Civil Comments labels
    labels = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
    print(f"\n  {'Signal':<25s}", end="")
    for l in labels:
        print(f" {l:>12s}", end="")
    print()
    print("  " + "-" * (25 + 13 * len(labels)))

    results = {}
    best_per_label = {}
    for sig_name, sig_vals in sorted(signals.items()):
        print(f"  {sig_name:<25s}", end="")
        for label in labels:
            r = corr(sig_vals, sample[label].values)
            print(f" {r:+12.3f}", end="")
            key = f"{sig_name}_vs_{label}"
            results[key] = round(r, 4)
            if label not in best_per_label or abs(r) > abs(best_per_label[label][1]):
                best_per_label[label] = (sig_name, r)
        print()

    print(f"\n  Best proxy per label:")
    for label, (sig, r) in best_per_label.items():
        print(f"    {label:20s} <- {sig:25s} r={r:+.4f}")

    return results, sample, signals


# =====================================================================
# DATASET 2: GoEmotions
# =====================================================================
def bench_goemotions(signals_func):
    print("\n" + "=" * 70)
    print("DATASET: GoEmotions (google-research-datasets/go_emotions)")
    print("  27 emotion labels, multi-label, ~58K comments")
    print("=" * 70)

    from datasets import load_dataset
    print("  Loading...", end="", flush=True)
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train", trust_remote_code=True)
    print(f" {len(ds)} rows")

    # GoEmotions has label IDs — map them
    label_names = ds.features["labels"].feature.names
    df = ds.to_pandas()
    df = df[df["text"].str.len() > 20].copy()

    # Create binary columns for each emotion
    for i, name in enumerate(label_names):
        df[name] = df["labels"].apply(lambda x: 1 if i in x else 0)

    # Sample: ensure representation of key emotions
    # Take random sample (emotions are multi-label so stratification is complex)
    sample = df.sample(n=min(N_SAMPLE, len(df)), random_state=42).reset_index(drop=True)
    print(f"  Sample: {len(sample)} texts")

    texts = sample["text"].tolist()
    print("  Running proxy signals:")
    signals = signals_func(texts)

    # PSQ-relevant emotions grouped by dimension
    psq_emotion_map = {
        "threat_exposure": ["fear", "nervousness"],
        "hostility_index": ["anger", "annoyance", "disgust"],
        "energy_dissipation": ["sadness", "grief", "disappointment"],
        "regulatory_capacity": ["anger", "fear", "nervousness", "confusion"],
        "resilience_baseline": ["optimism", "pride", "relief"],
        "cooling_capacity": ["relief", "caring", "gratitude"],
        "trust_conditions": ["approval", "admiration", "disapproval"],
    }

    results = {}
    print(f"\n  PSQ Dimension Correlations (proxy signal vs emotion cluster):")
    print(f"  {'PSQ Dimension':<25s} {'Best Signal':<25s} {'r':>8s} {'Emotions used':>30s}")
    print("  " + "-" * 90)

    for dim, emotions in psq_emotion_map.items():
        available = [e for e in emotions if e in sample.columns]
        if not available:
            continue
        # Composite: mean of relevant emotion binary labels
        composite = sample[available].mean(axis=1).values

        best_sig = None
        best_r = 0
        for sig_name, sig_vals in signals.items():
            r = corr(sig_vals, composite)
            if abs(r) > abs(best_r):
                best_r = r
                best_sig = sig_name

        print(f"  {dim:<25s} {best_sig:<25s} {best_r:+8.4f} {','.join(available):>30s}")
        results[dim] = {"best_signal": best_sig, "r": round(best_r, 4), "emotions": available}

    return results, sample, signals


# =====================================================================
# DATASET 3: SBIC (Social Bias Inference Corpus)
# =====================================================================
def bench_sbic(signals_func):
    print("\n" + "=" * 70)
    print("DATASET: SBIC (allenai/social_bias_frames)")
    print("  Power dynamics, intentionality, implied statements")
    print("=" * 70)

    from datasets import load_dataset
    print("  Loading...", end="", flush=True)
    ds = load_dataset("allenai/social_bias_frames", split="train", trust_remote_code=True)
    print(f" {len(ds)} rows")

    df = ds.to_pandas()
    # SBIC has: post, offensiveYN, intentYN, sexYN, whoTarget, targetCategory,
    #           targetStereotype, targetMinority, speakerMinorityYN, dataSource
    df = df[df["post"].str.len() > 20].copy()

    # Deduplicate on post text and aggregate labels
    agg = df.groupby("post").agg({
        "offensiveYN": "mean",
        "intentYN": "mean",
        "sexYN": "mean",
    }).reset_index()

    # Stratified sample by offensiveness
    agg["stratum"] = pd.cut(agg["offensiveYN"], bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
                             labels=["not", "mild", "moderate", "high"])
    samples = []
    for s in agg["stratum"].dropna().unique():
        subset = agg[agg["stratum"] == s]
        take = min(N_SAMPLE // 4, len(subset))
        if take > 0:
            samples.append(subset.sample(n=take, random_state=42))
    sample = pd.concat(samples).reset_index(drop=True)
    print(f"  Stratified sample: {len(sample)} texts")

    texts = sample["post"].tolist()
    print("  Running proxy signals:")
    signals = signals_func(texts)

    # Correlate with SBIC labels
    labels = ["offensiveYN", "intentYN", "sexYN"]
    label_desc = {
        "offensiveYN": "Offensiveness (hostility_index)",
        "intentYN": "Intentionality (authority_dynamics)",
        "sexYN": "Sexual content",
    }

    results = {}
    print(f"\n  {'Signal':<25s}", end="")
    for l in labels:
        print(f" {l:>15s}", end="")
    print()
    print("  " + "-" * (25 + 16 * len(labels)))

    best_per_label = {}
    for sig_name, sig_vals in sorted(signals.items()):
        print(f"  {sig_name:<25s}", end="")
        for label in labels:
            r = corr(sig_vals, sample[label].values)
            print(f" {r:+15.3f}", end="")
            if label not in best_per_label or abs(r) > abs(best_per_label[label][1]):
                best_per_label[label] = (sig_name, r)
        print()

    print(f"\n  Best proxy per SBIC label:")
    for label, (sig, r) in best_per_label.items():
        desc = label_desc.get(label, label)
        print(f"    {desc:40s} <- {sig:25s} r={r:+.4f}")

    return results, sample, signals


# =====================================================================
# DATASET 4: Berkeley (current baseline, for comparison)
# =====================================================================
def bench_berkeley(signals_func):
    print("\n" + "=" * 70)
    print("DATASET: Berkeley Measuring Hate Speech (baseline)")
    print("=" * 70)

    df = pd.read_parquet("data/measuring-hate-speech.parquet")
    agg = df.groupby("text").agg({
        "hate_speech_score": "first",
        "hatespeech": "mean",
        "insult": "mean",
        "violence": "mean",
        "dehumanize": "mean",
    }).reset_index()

    np.random.seed(42)
    agg["quintile"] = pd.qcut(agg["hate_speech_score"], q=5, labels=False, duplicates="drop")
    samples = []
    for q in sorted(agg["quintile"].unique()):
        stratum = agg[agg["quintile"] == q]
        samples.append(stratum.sample(n=min(N_SAMPLE // 5, len(stratum)), random_state=42))
    sample = pd.concat(samples).reset_index(drop=True)
    print(f"  Stratified sample: {len(sample)} texts")

    texts = sample["text"].tolist()
    print("  Running proxy signals:")
    signals = signals_func(texts)

    labels = ["hate_speech_score", "hatespeech", "insult", "violence", "dehumanize"]
    print(f"\n  {'Signal':<25s}", end="")
    for l in labels:
        print(f" {l:>18s}", end="")
    print()
    print("  " + "-" * (25 + 19 * len(labels)))

    best_per_label = {}
    for sig_name, sig_vals in sorted(signals.items()):
        print(f"  {sig_name:<25s}", end="")
        for label in labels:
            r = corr(sig_vals, sample[label].values)
            print(f" {r:+18.3f}", end="")
            if label not in best_per_label or abs(r) > abs(best_per_label[label][1]):
                best_per_label[label] = (sig_name, r)
        print()

    print(f"\n  Best proxy per Berkeley label:")
    for label, (sig, r) in best_per_label.items():
        print(f"    {label:20s} <- {sig:25s} r={r:+.4f}")

    return {}, sample, signals


# =====================================================================
# MAIN
# =====================================================================
def main():
    all_results = {}

    # Run Berkeley first (baseline)
    r_berk, _, _ = bench_berkeley(load_proxy_signals)
    all_results["berkeley"] = r_berk

    # Civil Comments
    r_civil, _, _ = bench_civil_comments(load_proxy_signals)
    all_results["civil_comments"] = r_civil

    # GoEmotions
    r_goemo, _, _ = bench_goemotions(load_proxy_signals)
    all_results["goemotions"] = r_goemo

    # SBIC
    r_sbic, _, _ = bench_sbic(load_proxy_signals)
    all_results["sbic"] = r_sbic

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Which ground truth gives the highest proxy correlation?")
    print(f"{'='*70}")
    print("""
The question: which dataset should we use as ground truth for training
the student model? Higher correlation = the proxy signals carry more
information about that dataset's labels = better training signal.
""")

    # Save
    out_dir = Path("data/proxy-validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ground_truth_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Full results saved to {out_dir / 'ground_truth_comparison.json'}")


if __name__ == "__main__":
    main()
