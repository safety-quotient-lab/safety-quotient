"""
Map 4 new datasets to PSQ dimension ground truth records.

Datasets:
  1. Diplomacy Deception → trust_conditions (sender/receiver truthfulness)
  2. CaSiNo Negotiation → contractual_clarity (negotiation strategy + fairness)
  3. Stanford Politeness → authority_dynamics (power-politeness correlation)
  4. ProsocialDialog → defensive_architecture (safety level + RoTs)

Output: data/new-dataset-ground-truth.jsonl
(Merged into composite ground truth by build_composite_ground_truth.py)
"""
import json
import csv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def make_record(text, dimensions, source):
    return {"text": text, "source": source, "dimensions": dimensions}


# =====================================================================
# 1. DIPLOMACY → trust_conditions
# =====================================================================
def load_diplomacy(n=2000):
    print(f"\n{'='*60}")
    print("Loading Diplomacy → trust_conditions")
    print(f"{'='*60}")

    all_recs = []
    for split in ["train", "validation", "test"]:
        fpath = ROOT / "data" / "diplomacy" / f"{split}.jsonl"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                all_recs.append(json.loads(line))

    print(f"  Total conversations: {len(all_recs)}")

    # sender_labels: 1 = truthful, 0 = deceptive
    # receiver_labels: 0 = perceived lie, 1 = perceived truth, 2 = perceived truth (high confidence?)
    records = []
    for conv in all_recs:
        messages = conv["messages"]
        sender_labels = conv["sender_labels"]
        receiver_labels = conv["receiver_labels"]

        for i, msg in enumerate(messages):
            if len(msg.strip()) < 30:
                continue

            sl = sender_labels[i] if i < len(sender_labels) else None
            rl = receiver_labels[i] if i < len(receiver_labels) else None

            if sl is None:
                continue

            # Trust scoring logic:
            # Truthful sender + perceived truthful → high trust environment (7-9)
            # Truthful sender + perceived deceptive → trust breakdown despite honesty (4-5)
            # Deceptive sender + perceived truthful → trust violation in progress (1-3)
            # Deceptive sender + perceived deceptive → low trust, caught (3-4)
            if sl == 1 and rl in (1, 2):  # truthful + perceived truthful
                score = 7.5 + np.random.uniform(-0.5, 1.0)
                conf = 0.55
            elif sl == 1 and rl == 0:  # truthful but perceived as lie
                score = 4.5 + np.random.uniform(-0.5, 0.5)
                conf = 0.50
            elif sl == 0 and rl in (1, 2):  # deceptive but believed
                score = 2.0 + np.random.uniform(-0.5, 0.5)
                conf = 0.60
            elif sl == 0 and rl == 0:  # deceptive and caught
                score = 3.5 + np.random.uniform(-0.5, 0.5)
                conf = 0.55
            else:
                continue

            score = round(np.clip(score, 0, 10), 1)
            dims = {"trust_conditions": {"score": score, "confidence": conf}}
            records.append(make_record(msg, dims, "diplomacy"))

    # Stratified sample: balance deceptive vs truthful to avoid skew
    # (raw data is 95% truthful → mean 7.3, too high)
    low_trust = [r for r in records if r["dimensions"]["trust_conditions"]["score"] < 5]
    high_trust = [r for r in records if r["dimensions"]["trust_conditions"]["score"] >= 5]
    print(f"  Before balance: {len(low_trust)} low-trust, {len(high_trust)} high-trust")

    np.random.seed(42)
    half = n // 2
    if len(low_trust) < half:
        # Oversample low-trust (deceptive messages are rare)
        low_sample = low_trust * (half // max(len(low_trust), 1) + 1)
        np.random.shuffle(low_sample)
        low_sample = low_sample[:half]
    else:
        idx = np.random.choice(len(low_trust), half, replace=False)
        low_sample = [low_trust[i] for i in idx]

    high_take = min(half, len(high_trust))
    idx = np.random.choice(len(high_trust), high_take, replace=False)
    high_sample = [high_trust[i] for i in idx]

    records = low_sample + high_sample
    np.random.shuffle(records)

    print(f"  Records: {len(records)} (trust_conditions, balanced)")
    return records


# =====================================================================
# 2. CASINO → contractual_clarity
# =====================================================================
def load_casino(n=2000):
    print(f"\n{'='*60}")
    print("Loading CaSiNo → contractual_clarity")
    print(f"{'='*60}")

    fpath = ROOT / "data" / "casino" / "train.jsonl"
    if not fpath.exists():
        print("  CaSiNo data not found, skipping")
        return []

    all_recs = []
    with open(fpath) as f:
        for line in f:
            all_recs.append(json.loads(line))

    print(f"  Total dialogues: {len(all_recs)}")

    # Negotiation strategy → contractual clarity mapping
    # Strategies that improve clarity: elicit-pref, promote-coordination, self-need
    # Strategies that reduce clarity: uv-part (unknown value), no-need (deception about needs)
    clarity_positive = {"elicit-pref", "promote-coordination", "self-need", "other-need"}
    clarity_negative = {"uv-part", "no-need"}  # hiding info, false claims about needs
    clarity_neutral = {"empathy", "small-talk"}

    records = []
    for conv in all_recs:
        chat_logs = conv.get("chat_logs", [])
        annotations = conv.get("annotations", [])
        p_info = conv.get("participant_info", {})

        # Get satisfaction scores for both agents
        sats = {}
        for agent_key in ["mturk_agent_1", "mturk_agent_2"]:
            info = p_info.get(agent_key, {})
            outcomes = info.get("outcomes", {})
            sats[agent_key] = outcomes.get("satisfaction", "Unknown")

        # Build full dialogue text
        full_text = " ".join(
            entry.get("text", "") for entry in chat_logs
            if entry.get("text", "").strip()
        )

        if len(full_text) < 50:
            continue

        # Score based on strategy distribution
        pos_count = 0
        neg_count = 0
        total_ann = 0
        for ann in annotations:
            if len(ann) < 2:
                continue
            strategies = ann[1].split(",")
            for s in strategies:
                s = s.strip()
                total_ann += 1
                if s in clarity_positive:
                    pos_count += 1
                elif s in clarity_negative:
                    neg_count += 1

        if total_ann == 0:
            continue

        # Base score from strategy ratio
        pos_ratio = pos_count / total_ann
        neg_ratio = neg_count / total_ann
        base_score = 5.0 + pos_ratio * 3.0 - neg_ratio * 4.0

        # Adjust for satisfaction
        sat_bonus = 0
        for sat in sats.values():
            if "Very satisfied" in str(sat):
                sat_bonus += 0.5
            elif "Slightly satisfied" in str(sat):
                sat_bonus += 0.25
            elif "dissatisfied" in str(sat).lower():
                sat_bonus -= 0.5

        # Check if deal was reached
        last_data = ""
        for entry in reversed(chat_logs):
            d = entry.get("task_data", {}).get("data", "")
            if d:
                last_data = d
                break

        if "reject" in last_data:
            base_score -= 1.5  # failed negotiation
        elif "accept" in last_data:
            base_score += 0.5  # successful deal

        score = round(np.clip(base_score + sat_bonus, 0, 10), 1)
        conf = round(np.clip(0.40 + total_ann * 0.01, 0.40, 0.65), 2)

        dims = {"contractual_clarity": {"score": score, "confidence": conf}}
        records.append(make_record(full_text[:2000], dims, "casino"))

    if len(records) > n:
        np.random.seed(42)
        idx = np.random.choice(len(records), n, replace=False)
        records = [records[i] for i in idx]

    print(f"  Records: {len(records)} (contractual_clarity)")
    return records


# =====================================================================
# 3. STANFORD POLITENESS → authority_dynamics
# =====================================================================
def load_politeness(n=2000):
    print(f"\n{'='*60}")
    print("Loading Stanford Politeness → authority_dynamics")
    print(f"{'='*60}")

    records = []
    for src in ["wikipedia", "stack-exchange"]:
        fpath = ROOT / "data" / "politeness" / "Stanford_politeness_corpus" / f"{src}.annotated.csv"
        if not fpath.exists():
            continue

        with open(fpath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"  {src}: {len(rows)} annotated requests")

        for row in rows:
            text = row.get("Request", "").strip()
            if len(text) < 30:
                continue

            norm_score = float(row["Normalized Score"])
            # Normalized Score range: ~-3 to +3 (mean 0)
            # Higher politeness = more aware of authority dynamics, more face-saving
            # Lower politeness = more authority assertion, less consideration of power

            # Map to authority_dynamics PSQ score:
            # Very impolite (< -1.5) → low authority_dynamics (2-3, domineering)
            # Impolite (-1.5 to -0.5) → moderate-low (3-4)
            # Neutral (-0.5 to 0.5) → moderate (5-6)
            # Polite (0.5 to 1.5) → moderate-high (6-7)
            # Very polite (> 1.5) → high (7-8, healthy authority relations)

            # Linear mapping from [-3, +3] to [2, 8]
            psq_score = 5.0 + norm_score * 1.0  # range ~2-8
            psq_score = round(np.clip(psq_score, 1, 9), 1)

            # Confidence from annotator agreement
            individual_scores = []
            for i in range(1, 6):
                s = row.get(f"Score{i}")
                if s:
                    try:
                        individual_scores.append(float(s))
                    except ValueError:
                        pass

            if len(individual_scores) >= 3:
                std = np.std(individual_scores)
                # Low std = high agreement = higher confidence
                # NOTE: halved confidence (was 0.30-0.65) — politeness ≠ authority dynamics.
                # Compressed range (std=0.73 vs LLM 1.72), over-predicts +0.90-1.45.
                # At 0.15-0.30, 100 LLM labels at 5x weight will dominate.
                conf = round(np.clip(0.30 - std * 0.015, 0.15, 0.30), 2)
            else:
                conf = 0.20

            dims = {"authority_dynamics": {"score": psq_score, "confidence": conf}}
            records.append(make_record(text, dims, f"politeness_{src}"))

    if len(records) > n:
        np.random.seed(42)
        idx = np.random.choice(len(records), n, replace=False)
        records = [records[i] for i in idx]

    print(f"  Records: {len(records)} (authority_dynamics)")
    return records


# =====================================================================
# 4. PROSOCIALDIALOG → defensive_architecture
# =====================================================================
def load_prosocial(n=2000):
    print(f"\n{'='*60}")
    print("Loading ProsocialDialog → defensive_architecture")
    print(f"{'='*60}")

    fpath = ROOT / "data" / "prosocial" / "train.jsonl"
    if not fpath.exists():
        print("  ProsocialDialog data not found, skipping")
        return []

    all_recs = []
    with open(fpath) as f:
        for line in f:
            all_recs.append(json.loads(line))

    print(f"  Total records: {len(all_recs)}")

    # Safety label → defensive_architecture mapping
    # The problematic utterances (context) often exhibit defensive patterns:
    # - __needs_intervention__: severe defensive breakdown or acting out (score 1-2)
    # - __needs_caution__: immature defenses visible (score 2-4)
    # - __probably_needs_caution__: moderate defensive issues (score 3-5)
    # - __possibly_needs_caution__: mild issues (score 4-6)
    # - __casual__: normal/healthy functioning (score 5-7)

    safety_to_score = {
        "__needs_intervention__": (1.5, 0.60),
        "__needs_caution__": (3.0, 0.50),
        "__probably_needs_caution__": (4.0, 0.40),
        "__possibly_needs_caution__": (5.0, 0.35),
        "__casual__": (6.0, 0.30),
    }

    records = []
    for rec in all_recs:
        context = rec.get("context", "").strip()
        if len(context) < 30:
            continue

        label = rec.get("safety_label", "")
        if label not in safety_to_score:
            continue

        base_score, base_conf = safety_to_score[label]

        # Boost confidence when annotators agree
        anns = rec.get("safety_annotations", [])
        if len(anns) >= 3:
            agreement = len(set(anns))
            if agreement == 1:  # unanimous
                base_conf += 0.10
            elif agreement == 2:
                base_conf += 0.05

        # Add noise to avoid clustering
        score = round(np.clip(base_score + np.random.uniform(-0.8, 0.8), 0, 10), 1)
        conf = round(np.clip(base_conf, 0.25, 0.65), 2)

        dims = {"defensive_architecture": {"score": score, "confidence": conf}}
        records.append(make_record(context, dims, "prosocial"))

    # Stratified sample to balance safety levels
    if len(records) > n:
        by_label = {}
        for rec in records:
            s = rec["dimensions"]["defensive_architecture"]["score"]
            bucket = "low" if s < 3 else "mid" if s < 5 else "high"
            by_label.setdefault(bucket, []).append(rec)

        np.random.seed(42)
        per_bucket = n // 3
        sampled = []
        for bucket in ["low", "mid", "high"]:
            pool = by_label.get(bucket, [])
            take = min(per_bucket, len(pool))
            if take > 0:
                idx = np.random.choice(len(pool), take, replace=False)
                sampled.extend(pool[i] for i in idx)
        records = sampled

    print(f"  Records: {len(records)} (defensive_architecture)")
    return records


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("Mapping 4 new datasets to PSQ ground truth")

    all_records = []
    # REMOVED in v3b: Diplomacy labels measure sender intent (truthful vs deceptive),
    # not textual trust indicators. 525 deceptive-but-believed records are fundamentally
    # unlearnable — cooperative text scored 1.5-2.5 because sender was secretly lying.
    # This teaches the model that cooperative language = low trust, contradicting all
    # other sources. MAE 2.405 (worst of any source), 56.7% of trust_conditions error.
    # See distillation-research.md §8z for full audit.
    # all_records.extend(load_diplomacy())
    all_records.extend(load_casino())
    all_records.extend(load_politeness())
    all_records.extend(load_prosocial())

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_records)} new records")
    print(f"{'='*60}")

    # Stats
    dim_counts = {}
    source_counts = {}
    for rec in all_records:
        source_counts[rec["source"]] = source_counts.get(rec["source"], 0) + 1
        for dim_id in rec["dimensions"]:
            dim_counts[dim_id] = dim_counts.get(dim_id, 0) + 1

    print(f"\n  By source:")
    for src, count in sorted(source_counts.items()):
        print(f"    {src:25s}: {count:5d} records")

    print(f"\n  By dimension:")
    for dim_id, count in sorted(dim_counts.items()):
        print(f"    {dim_id:25s}: {count:5d}")

    # Output
    out_path = ROOT / "data" / "new-dataset-ground-truth.jsonl"
    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n  Written to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    np.random.seed(42)
    main()
