"""
Benchmark proxy teacher models against Berkeley hate speech ground truth.
Runs on CPU. Suppresses progress bars.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from transformers import pipeline, logging
import json

logging.set_verbosity_error()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# Load and prepare stratified sample (500 texts)
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
    samples.append(stratum.sample(n=min(100, len(stratum)), random_state=42))
sample = pd.concat(samples).reset_index(drop=True)
texts = sample["text"].tolist()
print(f"Sample: {len(sample)} texts")
print()

results = {}

def corr(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    return stats.pearsonr(x[mask], y[mask])[0]


def bench(name, model_id, pos_labels):
    """Benchmark a binary classifier."""
    print(f"  {name}...", end="", flush=True)
    pipe = pipeline("text-classification", model=model_id, device=-1,
                     truncation=True, max_length=512)
    preds = pipe(texts, batch_size=32)

    scores = []
    for p in preds:
        if p["label"] in pos_labels:
            scores.append(p["score"])
        else:
            scores.append(1 - p["score"])
    scores = np.array(scores)

    r_hs = corr(scores, sample["hate_speech_score"].values)
    r_hate = corr(scores, sample["hatespeech"].values)
    r_ins = corr(scores, sample["insult"].values)
    r_vio = corr(scores, sample["violence"].values)
    r_deh = corr(scores, sample["dehumanize"].values)

    print(f"  hs={r_hs:+.3f}  hate={r_hate:+.3f}  ins={r_ins:+.3f}  vio={r_vio:+.3f}  deh={r_deh:+.3f}")
    results[name] = {
        "hate_speech_score": round(r_hs, 4),
        "hatespeech": round(r_hate, 4),
        "insult": round(r_ins, 4),
        "violence": round(r_vio, 4),
        "dehumanize": round(r_deh, 4),
    }
    del pipe
    return scores


# ===== HATE SPEECH MODELS =====
print("HATE SPEECH MODELS")
print("-" * 70)
fb = bench("fb_dynabench", "facebook/roberta-hate-speech-dynabench-r4-target", {"hate"})
ca = bench("cardiff_hate", "cardiffnlp/twitter-roberta-base-hate-latest", {"HATE"})
dh = bench("dehatebert", "Hate-speech-CNERG/dehatebert-mono-english", {"HATE"})
try:
    tg = bench("toxigen_hatebert", "tomh/toxigen_hatebert", {"LABEL_1"})
except Exception as e:
    print(f"  toxigen_hatebert... SKIPPED ({e})")

# ===== TOXICITY MODELS =====
print("\nTOXICITY MODELS")
print("-" * 70)
tb = bench("toxic_bert", "unitary/toxic-bert", {"toxic"})

# Detoxify (multi-head)
print("  detoxify (multi-head)...", end="", flush=True)
from detoxify import Detoxify
model = Detoxify("original")
detox = pd.DataFrame(model.predict(texts))
print()
for attr in ["toxicity", "severe_toxicity", "insult", "threat", "identity_attack"]:
    r = corr(detox[attr].values, sample["hate_speech_score"].values)
    print(f"    .{attr:20s} vs hate_speech_score: r={r:+.4f}")

comp_h = detox[["toxicity", "insult", "identity_attack"]].mean(axis=1).values
comp_t = detox[["severe_toxicity", "threat"]].mean(axis=1).values
r_ch = corr(comp_h, sample["hate_speech_score"].values)
r_ct = corr(comp_t, sample["violence"].values)
print(f"    composite_hostility  vs hate_speech_score: r={r_ch:+.4f}")
print(f"    composite_threat     vs violence:          r={r_ct:+.4f}")
results["detoxify_composite"] = {"hate_speech_score": round(r_ch, 4), "violence": round(r_ct, 4)}
results["detoxify_toxicity"] = {"hate_speech_score": round(corr(detox["toxicity"].values, sample["hate_speech_score"].values), 4)}

# ===== SENTIMENT + EMOTION =====
print("\nSENTIMENT + EMOTION MODELS")
print("-" * 70)

print("  sentiment...", end="", flush=True)
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1, truncation=True, max_length=512, top_k=None)
preds = pipe(texts, batch_size=32)
neg = np.array([next((x["score"] for x in p if x["label"] == "negative"), 0) for p in preds])
pos = np.array([next((x["score"] for x in p if x["label"] == "positive"), 0) for p in preds])
print(f"  P(neg) vs hs={corr(neg, sample['hate_speech_score'].values):+.3f}  P(pos) vs hs={corr(pos, sample['hate_speech_score'].values):+.3f}")
results["sentiment_neg"] = {"hate_speech_score": round(corr(neg, sample["hate_speech_score"].values), 4)}
results["sentiment_pos"] = {"hate_speech_score": round(corr(pos, sample["hate_speech_score"].values), 4)}
del pipe

print("  emotion...", end="", flush=True)
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest",
                device=-1, truncation=True, max_length=512, top_k=None)
pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id or 1
preds = pipe(texts, batch_size=16)
print()
for emo in ["anger", "fear", "joy", "disgust", "sadness"]:
    vals = np.array([next((x["score"] for x in p if x["label"] == emo), 0) for p in preds])
    r_hs = corr(vals, sample["hate_speech_score"].values)
    r_vio = corr(vals, sample["violence"].values)
    print(f"    P({emo:10s}) vs hs={r_hs:+.4f}  violence={r_vio:+.4f}")
    results[f"emotion_{emo}"] = {"hate_speech_score": round(r_hs, 4), "violence": round(r_vio, 4)}
del pipe

# ===== LEADERBOARD =====
print(f"\n{'='*70}")
print("LEADERBOARD: r vs Berkeley hate_speech_score (IRT)")
print(f"{'='*70}")
lb = []
for name, vals in results.items():
    if "hate_speech_score" in vals:
        lb.append((name, vals["hate_speech_score"]))
lb.sort(key=lambda x: abs(x[1]), reverse=True)
for name, r in lb:
    bar = "#" * int(abs(r) * 40)
    marker = " << PASS" if abs(r) > 0.7 else " << CLOSE" if abs(r) > 0.6 else ""
    print(f"  {name:30s}  r={r:+.4f}  {bar}{marker}")

# Save
os.makedirs("data/proxy-validation", exist_ok=True)
with open("data/proxy-validation/model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to data/proxy-validation/model_comparison.json")
