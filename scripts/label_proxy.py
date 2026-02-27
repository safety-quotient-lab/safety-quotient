"""
Multi-signal proxy labeler for PSQ distillation.

Reads JSONL with text, runs 3 proxy model families (detoxify, sentiment,
emotion = 13 signals), maps to PSQ dimension scores using validated
correlation weights, and outputs JSONL with teacher:"proxy" tag.

Confidence per dimension reflects the tier from proxy validation:
  Tier A (r>0.4): conf 0.5-0.7  — hostility, threat, energy
  Tier B (0.2<r<0.4): conf 0.2-0.4 — authority, regulatory, trust, cooling, contractual
  Tier C (r<0.2): conf 0.15-0.2  — resilience, defensive

Usage:
  python scripts/label_proxy.py --input data/unlabeled.jsonl --output data/train-proxy.jsonl
  python scripts/label_proxy.py --input data/unlabeled.jsonl --output data/train-proxy.jsonl --batch-size 64
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_models():
    """Load all proxy models. Returns a callable that takes texts and returns signals."""
    from detoxify import Detoxify
    from transformers import pipeline, logging
    logging.set_verbosity_error()

    print("Loading models...")
    detox = Detoxify("original")
    print("  detoxify loaded")

    sent_pipe = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1, truncation=True, max_length=512, top_k=None,
    )
    print("  sentiment loaded")

    emo_pipe = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest",
        device=-1, truncation=True, max_length=512, top_k=None,
    )
    emo_pipe.tokenizer.pad_token_id = emo_pipe.tokenizer.eos_token_id or 1
    print("  emotion loaded")

    def predict(texts, batch_size=32):
        """Run all proxy models, return dict of signal name -> np.array."""
        signals = {}

        # Detoxify
        d = detox.predict(texts)
        for attr in ["toxicity", "severe_toxicity", "insult", "threat", "identity_attack", "obscene"]:
            signals[f"detox_{attr}"] = np.array(d[attr])

        # Sentiment
        preds = sent_pipe(texts, batch_size=batch_size)
        signals["sent_negative"] = np.array([
            next((x["score"] for x in p if x["label"] == "negative"), 0) for p in preds
        ])
        signals["sent_positive"] = np.array([
            next((x["score"] for x in p if x["label"] == "positive"), 0) for p in preds
        ])

        # Emotion
        preds = emo_pipe(texts, batch_size=min(batch_size, 16))
        for emo in ["anger", "fear", "joy", "disgust", "sadness"]:
            signals[f"emo_{emo}"] = np.array([
                next((x["score"] for x in p if x["label"] == emo), 0) for p in preds
            ])

        return signals

    return predict


# Mapping: PSQ dimension → how to compute score from proxy signals
# Each entry: list of (signal_name, weight, invert)
# Score formula: 5 + sum(weight * (signal - 0.5) * direction * scale)
# where direction = -1 for threat dims, +1 for protective
# Confidence is preset per dimension tier.

DIMENSION_MAPPINGS = {
    # Tier A: strong proxy coverage
    "hostility_index": {
        "signals": [
            ("detox_toxicity", 0.35),
            ("emo_anger", 0.30),
            ("emo_disgust", 0.15),
            ("sent_negative", 0.20),
        ],
        "type": "threat",
        "base_conf": 0.55,
        "conf_boost": 0.15,  # added when signals are strong
    },
    "threat_exposure": {
        "signals": [
            ("detox_threat", 0.35),
            ("detox_severe_toxicity", 0.30),
            ("emo_fear", 0.20),
            ("sent_negative", 0.15),
        ],
        "type": "threat",
        "base_conf": 0.45,
        "conf_boost": 0.15,
    },
    "energy_dissipation": {
        "signals": [
            ("emo_sadness", 0.45),
            ("emo_disgust", 0.25),
            ("sent_negative", 0.20),
            ("emo_joy", -0.10),  # negative weight = inverted
        ],
        "type": "threat",
        "base_conf": 0.40,
        "conf_boost": 0.15,
    },

    # Tier B: partial proxy coverage
    "authority_dynamics": {
        "signals": [
            ("detox_toxicity", 0.35),
            ("sent_negative", 0.30),
            ("emo_anger", 0.20),
            ("detox_identity_attack", 0.15),
        ],
        "type": "threat",
        "base_conf": 0.25,
        "conf_boost": 0.10,
    },
    "regulatory_capacity": {
        "signals": [
            ("emo_fear", -0.30),  # fear signals dysregulation
            ("emo_anger", -0.30),
            ("emo_joy", 0.20),
            ("sent_positive", 0.20),
        ],
        "type": "protective",
        "base_conf": 0.25,
        "conf_boost": 0.10,
    },
    "trust_conditions": {
        "signals": [
            ("sent_positive", 0.35),
            ("detox_toxicity", -0.30),
            ("emo_joy", 0.20),
            ("emo_anger", -0.15),
        ],
        "type": "protective",
        "base_conf": 0.25,
        "conf_boost": 0.10,
    },
    "cooling_capacity": {
        "signals": [
            ("sent_positive", 0.35),
            ("emo_joy", 0.25),
            ("detox_toxicity", -0.25),
            ("emo_sadness", -0.15),
        ],
        "type": "protective",
        "base_conf": 0.25,
        "conf_boost": 0.10,
    },
    "contractual_clarity": {
        "signals": [
            ("detox_identity_attack", -0.35),
            ("detox_toxicity", -0.25),
            ("sent_negative", -0.20),
            ("emo_disgust", -0.20),
        ],
        "type": "protective",
        "base_conf": 0.25,
        "conf_boost": 0.10,
    },

    # Tier C: LLM-only (proxy gives neutral + very low confidence)
    "resilience_baseline": {
        "signals": [
            ("emo_joy", 0.40),
            ("sent_positive", 0.35),
            ("emo_sadness", -0.25),
        ],
        "type": "protective",
        "base_conf": 0.15,
        "conf_boost": 0.05,
    },
    "defensive_architecture": {
        "signals": [
            ("emo_joy", 0.30),  # weak sarcasm proxy
            ("emo_disgust", -0.30),
            ("sent_negative", -0.20),
            ("emo_fear", -0.20),
        ],
        "type": "protective",
        "base_conf": 0.15,
        "conf_boost": 0.05,
    },
}


def signals_to_psq(signals, idx):
    """Convert proxy signals at index idx to PSQ dimension scores."""
    dimensions = {}

    for dim_id, mapping in DIMENSION_MAPPINGS.items():
        # Compute weighted signal value
        weighted_sum = 0.0
        weight_total = 0.0
        signal_strength = 0.0

        for sig_name, weight in mapping["signals"]:
            if sig_name not in signals:
                continue
            val = float(signals[sig_name][idx])
            abs_weight = abs(weight)
            direction = 1.0 if weight > 0 else -1.0

            # Signal contribution: how far from neutral (0.5) in the expected direction
            deviation = (val - 0.5) * direction
            weighted_sum += deviation * abs_weight
            weight_total += abs_weight
            signal_strength += abs(val - 0.5) * abs_weight

        if weight_total == 0:
            dimensions[dim_id] = {"score": 5.0, "confidence": 0.1}
            continue

        # Normalize
        normalized = weighted_sum / weight_total  # range roughly -0.5 to +0.5

        # Map to PSQ score
        # For threat dimensions: positive normalized (signals firing) → low PSQ score
        # For protective: positive normalized → high PSQ score
        if mapping["type"] == "threat":
            score = 5.0 - (normalized * 8.0)  # range ~1-9
        else:
            score = 5.0 + (normalized * 8.0)  # range ~1-9

        score = round(float(np.clip(score, 0.5, 9.5)), 1)

        # Confidence: base + boost proportional to signal strength
        avg_strength = signal_strength / weight_total
        conf = mapping["base_conf"] + mapping["conf_boost"] * min(avg_strength * 4, 1.0)
        conf = round(float(np.clip(conf, 0.1, 0.75)), 2)

        dimensions[dim_id] = {"score": score, "confidence": conf}

    return dimensions


def main():
    parser = argparse.ArgumentParser(description="Multi-signal proxy labeler for PSQ")
    parser.add_argument("--input", required=True, help="Input JSONL file with 'text' field")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--limit", type=int, default=None, help="Max records to process")
    args = parser.parse_args()

    # Load input
    print(f"Reading {args.input}...")
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "text" not in rec:
                continue
            records.append(rec)
            if args.limit and len(records) >= args.limit:
                break
    print(f"  {len(records)} records loaded")

    if not records:
        print("No records to process")
        return

    # Load models
    predict = load_models()

    # Process in batches
    texts = [r["text"] for r in records]
    n_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    output_records = []

    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        print(f"  Batch {batch_num}/{n_batches} ({len(batch_texts)} texts)...", end="", flush=True)

        signals = predict(batch_texts, args.batch_size)

        for j in range(len(batch_texts)):
            dimensions = signals_to_psq(signals, j)
            rec = records[i + j].copy()
            rec["teacher"] = "proxy"
            rec["dimensions"] = dimensions
            # Store raw signals for potential later use
            rec["proxy_signals"] = {
                name: round(float(vals[j]), 4) for name, vals in signals.items()
            }
            output_records.append(rec)

        print(" done")

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWritten {len(output_records)} records to {out_path}")

    # Summary stats
    print(f"\nDimension score distributions:")
    for dim_id in DIMENSION_MAPPINGS:
        scores = [r["dimensions"][dim_id]["score"] for r in output_records]
        confs = [r["dimensions"][dim_id]["confidence"] for r in output_records]
        print(f"  {dim_id:25s}  score: {np.mean(scores):.1f} ± {np.std(scores):.1f}  "
              f"conf: {np.mean(confs):.2f} ± {np.std(confs):.2f}")


if __name__ == "__main__":
    main()
