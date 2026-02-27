"""
Build composite ground truth for PSQ student model training.

Loads 4 datasets (Berkeley, Civil Comments, GoEmotions, UCC), maps their
labels to PSQ dimension scores (0-10 scale), and outputs a unified JSONL
training set. Each record contains text + per-dimension score/confidence
derived from the dataset's ground truth labels.

PSQ Dimension → Dataset Mapping:
  hostility_index       ← Berkeley hate_speech_score
  threat_exposure       ← Civil Comments (threat, severe_toxicity)
  energy_dissipation    ← GoEmotions (sadness, grief, disappointment) + Dreaddit (stress)
  authority_dynamics    ← UCC (condescending)
  regulatory_capacity   ← GoEmotions (anger, fear, nervousness, confusion)
                        + ESConv (emotion intensity change, strategy labels)
                        + EmpatheticDialogues (emotion context)
  trust_conditions      ← GoEmotions (approval, disapproval) + UCC (dismissive)
  cooling_capacity      ← GoEmotions (relief, caring, gratitude) + UCC (healthy)
  contractual_clarity   ← UCC (generalisation_unfair)
  resilience_baseline   ← EmpatheticDialogues (emotion context for adversity/growth)
  defensive_architecture ← UCC (sarcastic) + LLM-labeled (Claude Code, DSQ-40/TKI rubric)

Output: data/composite-ground-truth.jsonl
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# PSQ dimension IDs (must match instruments.json)
DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]

# Dimensions that are threat factors (lower score = more threat)
THREAT_DIMS = {"threat_exposure", "hostility_index", "authority_dynamics", "energy_dissipation"}

SAMPLES_PER_DATASET = 2000


def scale_to_psq(values, source_min, source_max, invert=False):
    """Map values from source range to PSQ 0-10 scale.

    For threat dimensions: high source value (e.g. high hate) → low PSQ score (0 = max threat).
    For protective dimensions: high source value → high PSQ score.

    Args:
        invert: if True, high source = low PSQ (threat-like mapping)
    """
    values = np.clip(values, source_min, source_max)
    normalized = (values - source_min) / (source_max - source_min)  # 0-1
    if invert:
        normalized = 1 - normalized
    return np.round(normalized * 10, 1)


def make_record(text, dimensions, source):
    """Create a JSONL record."""
    return {
        "text": text,
        "source": source,
        "dimensions": dimensions,  # dict of dim_id -> {score, confidence}
    }


# =====================================================================
# BERKELEY → hostility_index
# =====================================================================
def load_berkeley(n=SAMPLES_PER_DATASET):
    print(f"\n{'='*60}")
    print("Loading Berkeley → hostility_index")
    print(f"{'='*60}")

    df = pd.read_parquet("data/measuring-hate-speech.parquet")
    agg = df.groupby("text").agg({
        "hate_speech_score": "first",
        "hatespeech": "mean",
        "insult": "mean",
        "violence": "mean",
        "dehumanize": "mean",
        "std_err": "first",
    }).reset_index()
    agg = agg[agg["text"].str.len() > 20].copy()

    # Stratified sample by hate_speech_score quintiles
    agg["quintile"] = pd.qcut(agg["hate_speech_score"], q=5, labels=False, duplicates="drop")
    samples = []
    per_q = n // 5
    for q in sorted(agg["quintile"].unique()):
        stratum = agg[agg["quintile"] == q]
        take = min(per_q, len(stratum))
        samples.append(stratum.sample(n=take, random_state=42))
    sample = pd.concat(samples).reset_index(drop=True)
    print(f"  Sampled: {len(sample)} texts")

    records = []
    for _, row in sample.iterrows():
        # hate_speech_score range: -8.3 to +6.3
        # High hate → low hostility_index PSQ score (more threat)
        hostility_score = scale_to_psq(
            row["hate_speech_score"], source_min=-8.3, source_max=6.3, invert=True
        )
        # Confidence: based on std_err (lower error = higher confidence)
        # std_err range: 0.1 to 1.9, mean 0.475
        conf = float(np.clip(1.0 - (row["std_err"] / 2.0), 0.3, 0.95))

        # Also derive threat_exposure from violence subscale
        # violence range: 0-4 ordinal (mean across annotators), higher = more violent
        threat_score = scale_to_psq(row["violence"], source_min=0, source_max=4, invert=True)
        threat_conf = 0.5  # ordinal, noisy

        dims = {
            "hostility_index": {"score": float(hostility_score), "confidence": round(conf, 2)},
            "threat_exposure": {"score": float(threat_score), "confidence": threat_conf},
        }
        records.append(make_record(row["text"], dims, "berkeley"))

    print(f"  Records: {len(records)} (hostility_index + threat_exposure)")
    return records


# =====================================================================
# CIVIL COMMENTS → threat_exposure (primary)
# =====================================================================
def load_civil_comments(n=SAMPLES_PER_DATASET):
    print(f"\n{'='*60}")
    print("Loading Civil Comments → threat_exposure")
    print(f"{'='*60}")

    from datasets import load_dataset
    ds = load_dataset("google/civil_comments", split="train")
    df = ds.to_pandas()
    df = df[df["text"].str.len() > 20].copy()
    print(f"  Total: {len(df)} texts")

    # Stratified sample by toxicity
    df["stratum"] = pd.cut(df["toxicity"], bins=[0, 0.05, 0.2, 0.5, 0.8, 1.0],
                            labels=["clean", "mild", "moderate", "high", "extreme"],
                            include_lowest=True)
    samples = []
    per_s = n // 5
    for s in ["clean", "mild", "moderate", "high", "extreme"]:
        subset = df[df["stratum"] == s]
        take = min(per_s, len(subset))
        if take > 0:
            samples.append(subset.sample(n=take, random_state=42))
    sample = pd.concat(samples).reset_index(drop=True)
    print(f"  Sampled: {len(sample)} texts")

    records = []
    for _, row in sample.iterrows():
        # threat (0-1): fraction of annotators flagging threat
        # Higher threat → lower threat_exposure PSQ score
        threat_score = scale_to_psq(row["threat"], source_min=0, source_max=1, invert=True)
        # Use number of annotators who agreed as confidence proxy
        # Civil Comments values are fractions, so higher = more agreement
        threat_conf = float(np.clip(0.4 + row["threat"] * 0.5, 0.4, 0.85))

        # severe_toxicity contributes to threat_exposure too
        sev_score = scale_to_psq(row["severe_toxicity"], source_min=0, source_max=1, invert=True)

        # Combined threat score: weighted average
        combined_threat = (float(threat_score) * 0.6 + float(sev_score) * 0.4)

        # Also get hostility from insult + toxicity
        hostility_score = scale_to_psq(
            (row["toxicity"] + row["insult"]) / 2, source_min=0, source_max=1, invert=True
        )
        hostility_conf = float(np.clip(0.4 + row["toxicity"] * 0.4, 0.4, 0.80))

        dims = {
            "threat_exposure": {"score": round(combined_threat, 1), "confidence": round(threat_conf, 2)},
            "hostility_index": {"score": float(hostility_score), "confidence": round(hostility_conf, 2)},
        }
        records.append(make_record(row["text"], dims, "civil_comments"))

    print(f"  Records: {len(records)} (threat_exposure + hostility_index)")
    return records


# =====================================================================
# GOEMOTIONS → emotional dimensions
# =====================================================================
def load_goemotions(n=SAMPLES_PER_DATASET):
    print(f"\n{'='*60}")
    print("Loading GoEmotions → emotional PSQ dimensions")
    print(f"{'='*60}")

    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    df = ds.to_pandas()
    df = df[df["text"].str.len() > 20].copy()
    print(f"  Total: {len(df)} texts")

    # Get label names
    label_names = ds.features["labels"].feature.names

    # Expand multi-label to binary columns
    for i, name in enumerate(label_names):
        df[name] = df["labels"].apply(lambda x: 1 if i in x else 0)

    # Sample
    sample = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    print(f"  Sampled: {len(sample)} texts")

    # PSQ dimension clusters
    # Each maps to a set of GoEmotions labels with polarity
    # Threat dimensions: presence of negative emotions → low score (more threat)
    # Protective dimensions: presence of positive emotions → high score
    dim_clusters = {
        "threat_exposure": {
            "negative": ["fear", "nervousness"],
            "positive": [],
            "type": "threat",
        },
        "hostility_index": {
            "negative": ["anger", "annoyance", "disgust"],
            "positive": [],
            "type": "threat",
        },
        "energy_dissipation": {
            "negative": ["sadness", "grief", "disappointment"],
            "positive": [],
            "type": "threat",
        },
        "regulatory_capacity": {
            "negative": ["anger", "fear", "nervousness", "confusion"],
            "positive": ["realization"],
            "type": "protective",
        },
        "resilience_baseline": {
            "negative": [],
            "positive": ["optimism", "pride", "relief"],
            "type": "protective",
        },
        "trust_conditions": {
            "negative": ["disapproval"],
            "positive": ["approval", "admiration"],
            "type": "protective",
        },
        "cooling_capacity": {
            "negative": [],
            "positive": ["relief", "caring", "gratitude"],
            "type": "protective",
        },
    }

    records = []
    for _, row in sample.iterrows():
        dims = {}
        for dim_id, cluster in dim_clusters.items():
            neg_labels = [l for l in cluster["negative"] if l in row.index]
            pos_labels = [l for l in cluster["positive"] if l in row.index]

            neg_sum = sum(row[l] for l in neg_labels) if neg_labels else 0
            pos_sum = sum(row[l] for l in pos_labels) if pos_labels else 0
            total_labels = len(neg_labels) + len(pos_labels)

            if total_labels == 0:
                continue

            # For threat dimensions: negative emotions push score down
            # For protective: positive emotions push score up
            if cluster["type"] == "threat":
                # More negative emotions present → lower PSQ score (more threat)
                # No negative emotions → neutral (5)
                if neg_labels:
                    neg_frac = neg_sum / len(neg_labels)  # 0 or 1 per label
                    score = 5.0 - (neg_frac * 4.0)  # range: 1-5
                else:
                    score = 5.0
            else:
                # Protective: positive emotions raise score, negative lower it
                if pos_labels and neg_labels:
                    pos_frac = pos_sum / len(pos_labels)
                    neg_frac = neg_sum / len(neg_labels)
                    score = 5.0 + (pos_frac * 3.0) - (neg_frac * 3.0)  # range: 2-8
                elif pos_labels:
                    pos_frac = pos_sum / len(pos_labels)
                    score = 5.0 + (pos_frac * 3.0)  # range: 5-8
                elif neg_labels:
                    neg_frac = neg_sum / len(neg_labels)
                    score = 5.0 - (neg_frac * 3.0)  # range: 2-5
                else:
                    score = 5.0

            score = round(np.clip(score, 0, 10), 1)

            # Confidence: based on how many labels in the cluster were active
            active = neg_sum + pos_sum
            if active == 0:
                conf = 0.25  # no signal — low confidence
            elif active == 1:
                conf = 0.45
            else:
                conf = 0.60

            dims[dim_id] = {"score": score, "confidence": conf}

        if dims:
            records.append(make_record(row["text"], dims, "goemotions"))

    print(f"  Records: {len(records)} (up to 7 emotional dimensions each)")
    return records


# =====================================================================
# UCC → subtle conversational dimensions
# =====================================================================
def load_ucc(n=SAMPLES_PER_DATASET):
    print(f"\n{'='*60}")
    print("Loading UCC → authority, trust, contractual, defensive, cooling")
    print(f"{'='*60}")

    df = pd.read_csv("data/ucc.csv")
    labels = ["hostile", "condescending", "dismissive", "antagonize",
              "sarcastic", "generalisation_unfair", "healthy"]
    agg = df.groupby("comment")[labels].mean().reset_index()
    agg = agg[agg["comment"].str.len() > 20].copy()
    print(f"  Unique comments: {len(agg)}")

    # Stratified by hostile
    agg["stratum"] = pd.cut(agg["hostile"], bins=[-0.01, 0.1, 0.3, 0.6, 1.01],
                             labels=["none", "mild", "moderate", "high"])
    samples = []
    per_s = n // 4
    for s in agg["stratum"].dropna().unique():
        subset = agg[agg["stratum"] == s]
        take = min(per_s, len(subset))
        if take > 0:
            samples.append(subset.sample(n=take, random_state=42))
    sample = pd.concat(samples).reset_index(drop=True)
    print(f"  Sampled: {len(sample)} texts")

    records = []
    for _, row in sample.iterrows():
        dims = {}

        # hostility_index ← hostile + antagonize (0-1 each)
        hostility_raw = (row["hostile"] + row["antagonize"]) / 2
        dims["hostility_index"] = {
            "score": float(scale_to_psq(hostility_raw, 0, 1, invert=True)),
            "confidence": round(0.45 + hostility_raw * 0.3, 2),
        }

        # authority_dynamics ← condescending (0-1)
        # Condescension = power assertion → threat
        # NOTE: proxy-LLM gap is +2.8 points. Halved twice (0.5 * 0.5 = 0.25x original)
        # to let 100 LLM authority_dynamics labels at 5x weight dominate.
        # Condescension is a real signal but too narrow — misses power dynamics breadth.
        dims["authority_dynamics"] = {
            "score": float(scale_to_psq(row["condescending"], 0, 1, invert=True)),
            "confidence": round((0.35 + row["condescending"] * 0.3) * 0.25, 2),
        }

        # trust_conditions ← dismissive (0-1)
        # Dismissiveness erodes trust → protective dimension goes down
        dims["trust_conditions"] = {
            "score": float(scale_to_psq(row["dismissive"], 0, 1, invert=True)),
            "confidence": round(0.35 + row["dismissive"] * 0.3, 2),
        }

        # cooling_capacity ← healthy (0-1)
        # Healthy conversation supports recovery → protective goes up
        dims["cooling_capacity"] = {
            "score": float(scale_to_psq(row["healthy"], 0, 1, invert=False)),
            "confidence": round(0.35 + abs(row["healthy"] - 0.5) * 0.5, 2),
        }

        # contractual_clarity ← generalisation_unfair (0-1)
        # REMOVED in v3b: proxy-LLM gap is +4.4 points, systematic bias of -2.32,
        # and v3 training shows negative r (-0.10 → -0.02) — this proxy actively
        # teaches the wrong signal. Contractual clarity will be learned from LLM
        # labels only (150 samples at llm_weight=5x).
        # dims["contractual_clarity"] = {
        #     "score": float(scale_to_psq(row["generalisation_unfair"], 0, 1, invert=True)),
        #     "confidence": round((0.30 + row["generalisation_unfair"] * 0.35) * 0.5, 2),
        # }

        # defensive_architecture ← sarcastic (0-1)
        # Sarcasm = defense mechanism indicator (weak signal)
        if row["sarcastic"] > 0.1:
            dims["defensive_architecture"] = {
                "score": round(5.0 - row["sarcastic"] * 2.0, 1),  # mild shift from neutral
                "confidence": 0.15,  # very low — proxy-LLM gap confirmed, weak signal
            }

        records.append(make_record(row["comment"], dims, "ucc"))

    print(f"  Records: {len(records)} (up to 6 dimensions each)")
    return records


# =====================================================================
# DREADDIT → energy_dissipation
# =====================================================================
def load_dreaddit(n=SAMPLES_PER_DATASET):
    print(f"\n{'='*60}")
    print("Loading Dreaddit → energy_dissipation")
    print(f"{'='*60}")

    df = pd.read_csv("data/dreaddit/dreaddit-train.csv")
    test = pd.read_csv("data/dreaddit/dreaddit-test.csv")
    df = pd.concat([df, test], ignore_index=True)
    df = df[df["text"].str.len() > 40].copy()
    print(f"  Total: {len(df)} posts")

    # Use both label and LIWC features for nuanced scoring
    # label=1 (stressed), label=0 (not stressed)
    # Also use lex_liwc_negemo, lex_liwc_sad, lex_liwc_Tone for gradient

    # Balanced sample
    stressed = df[df["label"] == 1].sample(n=min(n // 2, len(df[df["label"] == 1])), random_state=42)
    unstressed = df[df["label"] == 0].sample(n=min(n // 2, len(df[df["label"] == 0])), random_state=42)
    sample = pd.concat([stressed, unstressed]).reset_index(drop=True)
    print(f"  Sampled: {len(sample)} ({len(stressed)} stressed, {len(unstressed)} not)")

    # Subreddit severity mapping (from most severe to least)
    subreddit_severity = {
        "ptsd": 0.9, "domesticviolence": 0.85, "survivorsofabuse": 0.8,
        "homeless": 0.7, "almosthomeless": 0.6, "anxiety": 0.5,
        "stress": 0.4, "relationships": 0.3, "food_pantry": 0.5,
        "assistance": 0.4,
    }

    records = []
    for _, row in sample.iterrows():
        label = row["label"]
        sub_sev = subreddit_severity.get(row["subreddit"], 0.5)

        # Base score from label
        if label == 1:  # stressed → low energy_dissipation (energy trapped)
            # Subreddit severity modulates: PTSD stress = 1-2, mild stress = 3-4
            base_score = 1.0 + (1.0 - sub_sev) * 3.0  # range: 1.3 - 4.0
        else:  # not stressed → neutral to positive energy
            base_score = 5.0 + sub_sev * 1.5  # range: 5.0 - 6.4

        # LIWC refinement: use negemo and Tone if available
        if "lex_liwc_negemo" in row.index and not pd.isna(row["lex_liwc_negemo"]):
            negemo = row["lex_liwc_negemo"]
            # High negative emotion → push score down slightly
            base_score -= min(negemo / 10.0, 1.0)

        if "lex_liwc_Tone" in row.index and not pd.isna(row["lex_liwc_Tone"]):
            tone = row["lex_liwc_Tone"]
            # High tone (positive) → push score up slightly
            base_score += tone / 200.0  # small adjustment, max +0.5

        score = round(np.clip(base_score, 0, 10), 1)

        # Confidence: annotator agreement + label clarity
        conf_base = float(row.get("confidence", 0.5))
        conf = round(np.clip(0.3 + conf_base * 0.4, 0.35, 0.70), 2)

        dims = {
            "energy_dissipation": {"score": score, "confidence": conf},
        }
        records.append(make_record(row["text"], dims, "dreaddit"))

    print(f"  Records: {len(records)} (energy_dissipation)")
    return records


# =====================================================================
# ESConv → regulatory_capacity
# =====================================================================
def load_esconv(n=SAMPLES_PER_DATASET):
    print(f"\n{'='*60}")
    print("Loading ESConv → regulatory_capacity")
    print(f"{'='*60}")

    import json as json_mod

    # Load all splits
    records = []
    for split_file in ["esconv-train.jsonl"]:
        fpath = Path("data") / split_file
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                rec = json_mod.loads(line)
                conv = json_mod.loads(rec["text"])
                records.append(conv)

    # Also load validation/test if available
    from datasets import load_dataset
    try:
        ds = load_dataset("thu-coai/esconv")
        for split in ["validation", "test"]:
            if split in ds:
                for row in ds[split]:
                    conv = json_mod.loads(row["text"])
                    records.append(conv)
    except Exception:
        pass

    print(f"  Total conversations: {len(records)}")

    # Emotion type → dysregulation severity (how much regulatory strain)
    emotion_severity = {
        "anxiety": 0.7, "depression": 0.8, "sadness": 0.6, "anger": 0.75,
        "fear": 0.7, "disgust": 0.5, "shame": 0.65, "nervousness": 0.6,
        "guilt": 0.55, "jealousy": 0.5, "pain": 0.8,
    }

    # Strategy → regulation signal
    regulation_strategies = {
        "Reflection of feelings", "Restatement or Paraphrasing",
        "Providing Suggestions", "Affirmation and Reassurance",
    }

    output_records = []
    for conv in records:
        dialog = conv.get("dialog", [])
        if len(dialog) < 4:
            continue

        emotion = conv.get("emotion_type", "unknown")
        survey = conv.get("survey_score", {})

        # Extract seeker text (the person showing regulation or dysregulation)
        seeker_turns = [t["text"] for t in dialog if t.get("speaker") == "usr"]
        seeker_text = " ".join(seeker_turns)

        if len(seeker_text) < 40:
            continue

        # Score based on emotion type severity
        severity = emotion_severity.get(emotion, 0.5)

        # Emotion intensity change from survey (key signal for regulation)
        seeker_survey = survey.get("seeker", {})
        initial_intensity = int(seeker_survey.get("initial_emotion_intensity", 3))
        final_intensity = int(seeker_survey.get("final_emotion_intensity", 3))
        intensity_drop = initial_intensity - final_intensity  # positive = improvement

        # Count regulation strategies used by supporter (indicates regulation context)
        supporter_strategies = [t.get("strategy", "") for t in dialog if t.get("speaker") == "sys"]
        reg_strategy_count = sum(1 for s in supporter_strategies if s in regulation_strategies)

        # Base score: high severity emotion = low regulatory capacity shown
        base_score = 5.0 - severity * 3.5  # range: 1.5 - 3.25 for most emotions

        # Adjust for improvement: if intensity dropped, the person showed regulation
        if intensity_drop > 0:
            base_score += intensity_drop * 1.0  # +1 per intensity point drop
        elif intensity_drop < 0:
            base_score += intensity_drop * 0.5  # got worse = less regulation

        score = round(np.clip(base_score, 0, 10), 1)

        # Confidence: higher when we have survey data
        conf = 0.45
        if initial_intensity != 3 or final_intensity != 3:  # non-default values
            conf = 0.55
        if reg_strategy_count > 3:
            conf = 0.60

        dims = {
            "regulatory_capacity": {"score": score, "confidence": conf},
        }
        output_records.append(make_record(seeker_text, dims, "esconv"))

    # Limit to n
    if len(output_records) > n:
        np.random.seed(42)
        indices = np.random.choice(len(output_records), n, replace=False)
        output_records = [output_records[i] for i in indices]

    print(f"  Records: {len(output_records)} (regulatory_capacity)")
    return output_records


# =====================================================================
# EMPATHETIC DIALOGUES → resilience_baseline + regulatory_capacity
# =====================================================================
def load_empathetic_dialogues(n=SAMPLES_PER_DATASET):
    print(f"\n{'='*60}")
    print("Loading Empathetic Dialogues → resilience_baseline + regulatory_capacity")
    print(f"{'='*60}")

    df = pd.read_csv("data/empatheticdialogues/train.csv", on_bad_lines="skip")
    print(f"  Total utterances: {len(df)}")

    # Get situation descriptions (first turn of each conversation, speaker_idx=1)
    # These are the most psychologically rich — the person describing their experience
    situations = df[df["utterance_idx"] == 1][["conv_id", "context", "prompt", "utterance"]].copy()
    situations = situations.drop_duplicates("conv_id")
    print(f"  Unique conversations: {len(situations)}")

    # Emotion → resilience mapping
    # High resilience emotions: those showing recovery, growth, strength
    # Low resilience: those showing overwhelm, collapse, fragility
    resilience_map = {
        # High resilience (score 7-9)
        "proud": 8.0, "confident": 7.5, "prepared": 7.5, "hopeful": 7.0,
        "grateful": 7.5, "joyful": 7.0, "content": 7.0, "excited": 6.5,
        "caring": 7.0, "faithful": 7.0, "trusting": 7.0,
        # Moderate (5-6) — neutral or mild
        "surprised": 5.5, "anticipating": 6.0, "impressed": 6.0,
        "nostalgic": 5.5, "sentimental": 5.5,
        # Low resilience (2-4) — adversity/overwhelm
        "afraid": 3.5, "terrified": 2.0, "apprehensive": 4.0,
        "devastated": 1.5, "lonely": 3.0, "sad": 3.5,
        "ashamed": 3.0, "embarrassed": 3.5, "guilty": 3.5,
        "angry": 4.0, "furious": 3.0,
        "anxious": 3.5, "disgusted": 4.0,
        "disappointed": 3.5, "jealous": 4.0,
    }

    # Emotion → regulatory capacity mapping
    # High reg: emotions showing control, composure
    # Low reg: emotions showing dysregulation, overwhelm
    regulatory_map = {
        # High regulation (7-8)
        "content": 7.5, "grateful": 7.0, "caring": 7.0, "proud": 7.0,
        "prepared": 7.5, "confident": 7.0, "faithful": 7.0, "trusting": 6.5,
        # Moderate (5-6)
        "surprised": 5.5, "hopeful": 6.0, "anticipating": 6.0,
        "impressed": 6.0, "nostalgic": 5.5, "sentimental": 5.5,
        "joyful": 6.5, "excited": 5.5,
        # Low regulation (2-4)
        "terrified": 2.0, "furious": 2.5, "devastated": 2.0,
        "angry": 3.0, "anxious": 3.0, "afraid": 3.5,
        "disgusted": 3.5, "ashamed": 3.5, "embarrassed": 4.0,
        "lonely": 4.0, "sad": 4.0, "guilty": 4.0,
        "disappointed": 4.0, "apprehensive": 4.0,
        "annoyed": 4.0, "jealous": 4.0,
    }

    records = []
    for _, row in situations.iterrows():
        emotion = row["context"]
        # Use the situation description (prompt) as the text — it's richer
        text = str(row["prompt"]).replace("_comma_", ",").replace("_period_", ".")
        if len(text) < 30:
            continue

        dims = {}

        # Resilience
        if emotion in resilience_map:
            r_score = resilience_map[emotion]
            # Add small random noise to avoid exact duplicates per emotion
            r_score = round(np.clip(r_score + np.random.uniform(-0.5, 0.5), 0, 10), 1)
            dims["resilience_baseline"] = {
                "score": r_score,
                "confidence": 0.40,  # indirect mapping, moderate confidence
            }

        # Regulatory capacity
        if emotion in regulatory_map:
            reg_score = regulatory_map[emotion]
            reg_score = round(np.clip(reg_score + np.random.uniform(-0.5, 0.5), 0, 10), 1)
            dims["regulatory_capacity"] = {
                "score": reg_score,
                "confidence": 0.35,  # indirect mapping, lower confidence
            }

        if dims:
            records.append(make_record(text, dims, "empathetic_dialogues"))

    # Sample down if needed
    if len(records) > n:
        np.random.seed(42)
        indices = np.random.choice(len(records), n, replace=False)
        records = [records[i] for i in indices]

    print(f"  Records: {len(records)} (resilience_baseline + regulatory_capacity)")
    return records


# =====================================================================
# MAIN: Combine and output
# =====================================================================
# =====================================================================
# NEW DATASETS → trust, authority, contractual, defensive
# =====================================================================
def load_new_datasets():
    """Load pre-mapped records from Diplomacy, CaSiNo, Politeness, ProsocialDialog.

    These are generated by scripts/map_new_datasets.py and saved to
    data/new-dataset-ground-truth.jsonl.
    """
    new_path = ROOT / "data" / "new-dataset-ground-truth.jsonl"
    if not new_path.exists():
        print(f"  [new_datasets] Not found at {new_path}, skipping")
        print(f"  Run: python scripts/map_new_datasets.py")
        return []

    records = []
    with open(new_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"  [new_datasets] {len(records)} records loaded")
    source_counts = {}
    dim_counts = {}
    for r in records:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
        for d in r["dimensions"]:
            dim_counts[d] = dim_counts.get(d, 0) + 1
    for s, c in sorted(source_counts.items()):
        print(f"    {s}: {c}")
    for d, c in sorted(dim_counts.items()):
        print(f"    → {d}: {c}")

    return records


def load_llm_labels():
    """Load LLM-labeled data (Claude Code scoring sessions).

    These are gold-standard labels from manual scoring using DSQ-40 + TKI rubric.
    Currently covers defensive_architecture (50 records).
    Records are weighted 3x vs heuristic labels in the training script.
    """
    llm_path = ROOT / "data" / "train-llm.jsonl"
    if not llm_path.exists():
        print(f"  [llm_labels] No LLM labels found at {llm_path}, skipping")
        return []

    records = []
    with open(llm_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                # Normalize to composite format
                records.append({
                    "text": rec["text"],
                    "source": "llm_labeled",
                    "dimensions": rec["dimensions"],
                })

    print(f"  [llm_labels] {len(records)} LLM-labeled records loaded")
    dim_counts = {}
    for r in records:
        for d in r["dimensions"]:
            dim_counts[d] = dim_counts.get(d, 0) + 1
    for d, c in sorted(dim_counts.items()):
        print(f"    {d}: {c}")

    return records


def main():
    print("Building composite ground truth for PSQ student model")
    print(f"Target: {SAMPLES_PER_DATASET} samples per dataset\n")

    all_records = []

    # Load each dataset
    all_records.extend(load_berkeley())
    all_records.extend(load_civil_comments())
    all_records.extend(load_goemotions())
    all_records.extend(load_ucc())
    all_records.extend(load_dreaddit())
    all_records.extend(load_esconv())
    all_records.extend(load_empathetic_dialogues())
    all_records.extend(load_new_datasets())
    # Note: LLM labels are NOT included here — they are loaded separately by
    # distill.py with their own weight (llm_weight=5x). Including them here
    # would double-count them in training.

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_records)} records")
    print(f"{'='*60}")

    # Statistics
    dim_counts = {d: 0 for d in DIMS}
    source_counts = {}
    for rec in all_records:
        source_counts[rec["source"]] = source_counts.get(rec["source"], 0) + 1
        for dim_id in rec["dimensions"]:
            dim_counts[dim_id] += 1

    print(f"\n  By source:")
    for src, count in sorted(source_counts.items()):
        print(f"    {src:20s}: {count:5d} records")

    print(f"\n  By dimension (records with ground truth):")
    for dim_id in DIMS:
        count = dim_counts[dim_id]
        tier = "A" if count > 2000 else "B" if count > 500 else "C"
        print(f"    {dim_id:25s}: {count:5d}  (tier {tier})")

    # Output JSONL
    out_path = Path("data/composite-ground-truth.jsonl")
    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n  Written to {out_path} ({size_mb:.1f} MB)")
    print(f"  {len(all_records)} records total")


if __name__ == "__main__":
    main()
