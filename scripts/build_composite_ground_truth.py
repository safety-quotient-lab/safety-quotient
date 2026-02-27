"""
Build composite ground truth for PSQ student model training.

All design decisions (scales, formula params, disabled dims + reasons,
emotion tables, subreddit weights, etc.) live in data/dataset_mappings.json.
Edit that file to change a mapping. Never hardcode parameters here.

Absorbs map_new_datasets.py — all 11 source datasets in one script.

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

DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]


# ============================================================
# Config loader
# ============================================================

def load_config():
    path = ROOT / "data" / "dataset_mappings.json"
    with open(path) as f:
        return json.load(f)


# ============================================================
# Scoring helpers  (read all params from cfg dicts)
# ============================================================

def scale_to_psq(value, source_min, source_max, invert=False):
    value = np.clip(value, source_min, source_max)
    normalized = (value - source_min) / (source_max - source_min)
    if invert:
        normalized = 1 - normalized
    return round(float(normalized * 10), 1)


def apply_scoring(row, scoring_cfg, dataset_cfg=None):
    """Return score (0-10) or None if record should be skipped for this dim."""
    t = scoring_cfg["type"]

    if t == "scale":
        v = row[scoring_cfg["field"]]
        return scale_to_psq(v, scoring_cfg["source_min"], scoring_cfg["source_max"],
                             scoring_cfg.get("invert", False))

    if t == "weighted_scale":
        total_w = sum(f["weight"] for f in scoring_cfg["fields"])
        combined = sum(
            f["weight"] * np.clip(row[f["field"]], f["source_min"], f["source_max"])
            / (f["source_max"] - f["source_min"])
            for f in scoring_cfg["fields"]
        ) / total_w
        if scoring_cfg["fields"][0].get("invert", True):
            combined = 1 - combined
        return round(float(combined * 10), 1)

    if t == "cluster_binary":
        neg_labels = scoring_cfg.get("neg_labels", [])
        pos_labels = scoring_cfg.get("pos_labels", [])
        dim_type   = scoring_cfg["dim_type"]
        neutral    = scoring_cfg.get("neutral_score", 5.0)
        rng        = scoring_cfg.get("range", 4.0)

        neg_sum = sum(row[l] for l in neg_labels if l in row.index)
        pos_sum = sum(row[l] for l in pos_labels if l in row.index)
        neg_n   = len([l for l in neg_labels if l in row.index])
        pos_n   = len([l for l in pos_labels if l in row.index])
        total   = neg_n + pos_n
        if total == 0:
            return None

        if dim_type == "threat":
            score = neutral - (neg_sum / neg_n * rng) if neg_n else neutral
        else:  # protective
            if pos_n and neg_n:
                score = neutral + (pos_sum / pos_n * rng) - (neg_sum / neg_n * rng)
            elif pos_n:
                score = neutral + (pos_sum / pos_n * rng)
            elif neg_n:
                score = neutral - (neg_sum / neg_n * rng)
            else:
                score = neutral

        return round(float(np.clip(score, 0, 10)), 1)

    if t == "linear":
        v = row[scoring_cfg["field"]]
        if "min_field_value" in scoring_cfg and v < scoring_cfg["min_field_value"]:
            return None  # skip record for this dim
        score = scoring_cfg["base"] + v * scoring_cfg["multiplier"]
        return round(float(np.clip(score, scoring_cfg.get("clip_min", 0),
                                        scoring_cfg.get("clip_max", 10))), 1)

    if t == "emotion_lookup":
        table_name = scoring_cfg["table"]
        table = dataset_cfg[table_name]
        emotion = row.get("emotion", row.get("context", ""))
        if emotion not in table:
            return None
        noise = dataset_cfg.get("noise_range", 0)
        base = table[emotion]
        score = base + (np.random.uniform(-noise, noise) if noise else 0)
        return round(float(np.clip(score, 0, 10)), 1)

    raise ValueError(f"Unknown scoring type: {t}")


def apply_confidence(row, conf_cfg, active_count=None, dataset_cfg=None):
    """Return confidence (0-1)."""
    t = conf_cfg["type"]

    if t == "fixed":
        return conf_cfg["value"]

    if t == "linear_clip":
        v = float(row.get(conf_cfg["field"], 0))
        c = conf_cfg["base"] + v * conf_cfg["multiplier"]
        c *= conf_cfg.get("scale", 1.0)
        return round(float(np.clip(c, conf_cfg.get("min", 0), conf_cfg.get("max", 1))), 2)

    if t == "std_err":
        v = float(row[conf_cfg["field"]])
        c = 1.0 - (v / conf_cfg["divisor"])
        return round(float(np.clip(c, conf_cfg["min"], conf_cfg["max"])), 2)

    if t == "abs_from_midpoint":
        v = float(row[conf_cfg["field"]])
        c = conf_cfg["base"] + abs(v - conf_cfg["midpoint"]) * conf_cfg["multiplier"]
        return round(float(np.clip(c, conf_cfg.get("min", 0), conf_cfg.get("max", 1))), 2)

    if t == "active_label_count":
        if active_count is None:
            return 0.5
        thresholds = conf_cfg["thresholds"]
        for max_active, val in thresholds:
            if active_count <= max_active:
                return val
        return conf_cfg["default"]

    if t == "annotator_std":
        scores = []
        for f in conf_cfg["score_fields"]:
            s = row.get(f)
            if s:
                try:
                    scores.append(float(s))
                except ValueError:
                    pass
        if len(scores) < conf_cfg.get("min_annotators", 3):
            return conf_cfg.get("min", 0.15)
        std = np.std(scores)
        c = conf_cfg["base"] + std * conf_cfg["std_multiplier"]
        c *= conf_cfg.get("scale", 1.0)
        return round(float(np.clip(c, conf_cfg.get("min", 0), conf_cfg.get("max", 1))), 2)

    raise ValueError(f"Unknown confidence type: {t}")


def active_label_count(row, scoring_cfg):
    """Count active neg+pos labels for cluster_binary confidence."""
    neg_sum = sum(row[l] for l in scoring_cfg.get("neg_labels", []) if l in row.index)
    pos_sum = sum(row[l] for l in scoring_cfg.get("pos_labels", []) if l in row.index)
    return int(neg_sum + pos_sum)


def make_record(text, dimensions, source):
    return {"text": text, "source": source, "dimensions": dimensions}


def apply_mappings(row, mappings, dataset_cfg):
    """Apply all enabled mappings to a row. Return dims dict."""
    dims = {}
    for m in mappings:
        if not m.get("enabled", True):
            continue
        dim = m["dimension"]
        score = apply_scoring(row, m["scoring"], dataset_cfg)
        if score is None:
            continue
        if m["scoring"]["type"] == "cluster_binary":
            ac = active_label_count(row, m["scoring"])
            conf = apply_confidence(row, m["confidence"], active_count=ac)
        else:
            conf = apply_confidence(row, m["confidence"], dataset_cfg=dataset_cfg)
        dims[dim] = {"score": score, "confidence": conf}
    return dims


# ============================================================
# Dataset loaders
# ============================================================

def load_berkeley(cfg, global_cfg):
    print(f"\n{'='*60}\nLoading Berkeley → hostility_index\n{'='*60}")
    df = pd.read_parquet(cfg["file"])
    agg_cols = {"hate_speech_score": "first", "hatespeech": "mean",
                "insult": "mean", "violence": "mean", "dehumanize": "mean", "std_err": "first"}
    agg = df.groupby(cfg["text_field"]).agg(agg_cols).reset_index()
    agg = agg[agg[cfg["text_field"]].str.len() > global_cfg["min_text_length"]].copy()

    sc = cfg["sampling"]
    agg["quintile"] = pd.qcut(agg[sc["field"]], q=5, labels=False, duplicates="drop")
    per_q = sc["n"] // 5
    samples = [agg[agg["quintile"] == q].sample(n=min(per_q, len(agg[agg["quintile"] == q])),
               random_state=sc["random_state"])
               for q in sorted(agg["quintile"].unique())]
    sample = pd.concat(samples).reset_index(drop=True)
    print(f"  Sampled: {len(sample)}")

    records = []
    for _, row in sample.iterrows():
        dims = apply_mappings(row, cfg["mappings"], cfg)
        if dims:
            records.append(make_record(row[cfg["text_field"]], dims, "berkeley"))
    print(f"  Records: {len(records)}")
    return records


def load_civil_comments(cfg, global_cfg):
    print(f"\n{'='*60}\nLoading Civil Comments → hostility_index\n{'='*60}")
    from datasets import load_dataset
    ds = load_dataset(cfg["source"], split="train")
    df = ds.to_pandas()
    df = df[df["text"].str.len() > global_cfg["min_text_length"]].copy()
    print(f"  Total: {len(df)}")

    sc = cfg["sampling"]
    df["stratum"] = pd.cut(df[sc["field"]], bins=sc["bins"], labels=sc["labels"],
                            include_lowest=True)
    per_s = sc["n"] // len(sc["labels"])
    samples = [df[df["stratum"] == s].sample(n=min(per_s, len(df[df["stratum"] == s])),
               random_state=sc["random_state"])
               for s in sc["labels"] if len(df[df["stratum"] == s]) > 0]
    sample = pd.concat(samples).reset_index(drop=True)
    print(f"  Sampled: {len(sample)}")

    records = []
    for _, row in sample.iterrows():
        dims = apply_mappings(row, cfg["mappings"], cfg)
        if dims:
            records.append(make_record(row["text"], dims, "civil_comments"))
    print(f"  Records: {len(records)}")
    return records


def load_goemotions(cfg, global_cfg):
    print(f"\n{'='*60}\nLoading GoEmotions → emotional dimensions\n{'='*60}")
    from datasets import load_dataset
    ds = load_dataset(cfg["source"], cfg.get("source_config", "simplified"), split="train")
    df = ds.to_pandas()
    df = df[df["text"].str.len() > global_cfg["min_text_length"]].copy()
    label_names = ds.features["labels"].feature.names
    for i, name in enumerate(label_names):
        df[name] = df["labels"].apply(lambda x: 1 if i in x else 0)

    sc = cfg["sampling"]
    sample = df.sample(n=min(sc["n"], len(df)), random_state=sc["random_state"]).reset_index(drop=True)
    print(f"  Sampled: {len(sample)}")

    records = []
    for _, row in sample.iterrows():
        dims = apply_mappings(row, cfg["mappings"], cfg)
        if dims:
            records.append(make_record(row["text"], dims, "goemotions"))
    print(f"  Records: {len(records)}")
    return records


def load_ucc(cfg, global_cfg):
    print(f"\n{'='*60}\nLoading UCC → authority, trust, contractual, defensive, cooling\n{'='*60}")
    df = pd.read_csv(cfg["file"])
    labels = cfg["aggregate_labels"]
    agg = df.groupby(cfg["text_field"])[labels].mean().reset_index()
    agg = agg[agg[cfg["text_field"]].str.len() > global_cfg["min_text_length"]].copy()
    print(f"  Unique comments: {len(agg)}")

    sc = cfg["sampling"]
    agg["stratum"] = pd.cut(agg[sc["field"]], bins=sc["bins"], labels=sc["labels"])
    per_s = sc["n"] // len(sc["labels"])
    samples = [agg[agg["stratum"] == s].sample(n=min(per_s, len(agg[agg["stratum"] == s])),
               random_state=sc["random_state"])
               for s in agg["stratum"].dropna().unique()]
    sample = pd.concat(samples).reset_index(drop=True)
    print(f"  Sampled: {len(sample)}")

    records = []
    for _, row in sample.iterrows():
        dims = apply_mappings(row, cfg["mappings"], cfg)
        if dims:
            records.append(make_record(row[cfg["text_field"]], dims, "ucc"))
    print(f"  Records: {len(records)}")
    return records


def load_dreaddit(cfg, global_cfg):
    print(f"\n{'='*60}\nLoading Dreaddit → energy_dissipation\n{'='*60}")
    dfs = [pd.read_csv(f) for f in cfg["files"] if Path(f).exists()]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["text"].str.len() > cfg.get("min_text_length", global_cfg["min_text_length"])].copy()
    print(f"  Total: {len(df)}")

    sc = cfg["sampling"]
    stressed   = df[df["label"] == 1].sample(n=min(sc["n"] // 2, len(df[df["label"] == 1])),
                                              random_state=sc["random_state"])
    unstressed = df[df["label"] == 0].sample(n=min(sc["n"] // 2, len(df[df["label"] == 0])),
                                              random_state=sc["random_state"])
    sample = pd.concat([stressed, unstressed]).reset_index(drop=True)
    print(f"  Sampled: {len(sample)} ({len(stressed)} stressed, {len(unstressed)} not)")

    sub_sev = cfg["subreddit_severity"]
    default_sev = cfg["default_subreddit_severity"]
    sf = cfg["score_formula"]

    records = []
    for _, row in sample.iterrows():
        sev = sub_sev.get(row.get("subreddit", ""), default_sev)
        if row["label"] == 1:
            base = sf["stressed"]["base"] + (1.0 - sev) * sf["stressed"]["severity_multiplier"]
        else:
            base = sf["not_stressed"]["base"] + sev * sf["not_stressed"]["severity_multiplier"]

        if sf["liwc_negemo_field"] in row.index and not pd.isna(row[sf["liwc_negemo_field"]]):
            base -= min(row[sf["liwc_negemo_field"]] / sf["liwc_negemo_divisor"],
                        sf["liwc_negemo_max_shift"])
        if sf["liwc_tone_field"] in row.index and not pd.isna(row[sf["liwc_tone_field"]]):
            base += row[sf["liwc_tone_field"]] / sf["liwc_tone_divisor"]

        score = round(float(np.clip(base, 0, 10)), 1)
        conf_cfg = cfg["mappings"][0]["confidence"]
        conf = apply_confidence(row, conf_cfg)
        dims = {"energy_dissipation": {"score": score, "confidence": conf}}
        records.append(make_record(row["text"], dims, "dreaddit"))

    print(f"  Records: {len(records)}")
    return records


def load_esconv(cfg, global_cfg):
    print(f"\n{'='*60}\nLoading ESConv → regulatory_capacity\n{'='*60}")
    convs = []
    for fpath in cfg["files"]:
        p = Path(fpath)
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                convs.append(json.loads(rec["text"]))

    from datasets import load_dataset
    try:
        ds = load_dataset(cfg["hf_source"])
        for split in cfg.get("hf_splits", []):
            if split in ds:
                for row in ds[split]:
                    convs.append(json.loads(row["text"]))
    except Exception:
        pass

    print(f"  Total conversations: {len(convs)}")
    em_sev = cfg["emotion_severity"]
    default_sev = cfg["default_emotion_severity"]
    reg_strats = set(cfg["regulation_strategies"])
    sf = cfg["score_formula"]
    min_len = cfg.get("min_seeker_text_length", 40)
    min_turns = cfg.get("min_dialog_turns", 4)

    output = []
    for conv in convs:
        dialog = conv.get("dialog", [])
        if len(dialog) < min_turns:
            continue
        emotion = conv.get("emotion_type", "unknown")
        seeker_text = " ".join(t["text"] for t in dialog if t.get("speaker") == "usr")
        if len(seeker_text) < min_len:
            continue

        sev = em_sev.get(emotion, default_sev)
        base = sf["base_from_severity"]["neutral_score"] + sev * sf["base_from_severity"]["severity_multiplier"]

        survey = conv.get("survey_score", {}).get("seeker", {})
        initial = int(survey.get("initial_emotion_intensity", 3))
        final   = int(survey.get("final_emotion_intensity", 3))
        drop = initial - final
        if drop > 0:
            base += drop * sf["intensity_drop_bonus"]
        elif drop < 0:
            base += drop * sf["intensity_increase_penalty"]

        supporter_strats = [t.get("strategy", "") for t in dialog if t.get("speaker") == "sys"]
        reg_count = sum(1 for s in supporter_strats if s in reg_strats)

        score = round(float(np.clip(base, 0, 10)), 1)

        cc = cfg["mappings"][0]["confidence"]
        conf = cc["base"]
        if initial != 3 or final != 3:
            conf = cc["survey_threshold"]
        if reg_count >= cc["strategy_min_count"]:
            conf = cc["strategy_threshold"]

        dims = {"regulatory_capacity": {"score": score, "confidence": conf}}
        output.append(make_record(seeker_text, dims, "esconv"))

    sc = cfg["sampling"]
    if len(output) > sc["n"]:
        np.random.seed(sc["random_seed"])
        idx = np.random.choice(len(output), sc["n"], replace=False)
        output = [output[i] for i in idx]

    print(f"  Records: {len(output)}")
    return output


def load_empathetic_dialogues(cfg, global_cfg):
    print(f"\n{'='*60}\nLoading Empathetic Dialogues → resilience + regulatory\n{'='*60}")
    df = pd.read_csv(cfg["file"], on_bad_lines="skip")
    situations = (df[df["utterance_idx"] == 1][["conv_id", "context", "prompt"]]
                  .drop_duplicates("conv_id"))
    print(f"  Unique conversations: {len(situations)}")

    noise = cfg.get("noise_range", 0)

    records = []
    for _, row in situations.iterrows():
        text = str(row["prompt"]).replace("_comma_", ",").replace("_period_", ".")
        if len(text) < 30:
            continue
        dims = apply_mappings(row, cfg["mappings"], cfg)
        if dims:
            records.append(make_record(text, dims, "empathetic_dialogues"))

    sc = cfg["sampling"]
    if len(records) > sc["n"]:
        np.random.seed(sc["random_seed"])
        idx = np.random.choice(len(records), sc["n"], replace=False)
        records = [records[i] for i in idx]

    print(f"  Records: {len(records)}")
    return records


def load_diplomacy(cfg, global_cfg):
    print(f"\n{'='*60}\nLoading Diplomacy → trust_conditions\n{'='*60}")
    enabled = [m for m in cfg["mappings"] if m.get("enabled", True)]
    if not enabled:
        print("  All mappings disabled — skipping")
        return []

    all_recs = []
    for fpath in cfg["files"]:
        p = Path(fpath)
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                all_recs.append(json.loads(line))

    matrix = cfg["trust_score_matrix"]
    records = []
    for conv in all_recs:
        msgs = conv["messages"]
        sl_list = conv["sender_labels"]
        rl_list = conv["receiver_labels"]
        for i, msg in enumerate(msgs):
            if len(msg.strip()) < 30:
                continue
            sl = sl_list[i] if i < len(sl_list) else None
            rl = rl_list[i] if i < len(rl_list) else None
            if sl is None:
                continue
            if sl == 1 and rl in (1, 2):
                key = "truthful_perceived_truthful"
            elif sl == 1 and rl == 0:
                key = "truthful_perceived_deceptive"
            elif sl == 0 and rl in (1, 2):
                key = "deceptive_perceived_truthful"
            elif sl == 0 and rl == 0:
                key = "deceptive_perceived_deceptive"
            else:
                continue
            mc = matrix[key]
            noise = mc["score_noise"]
            score = round(float(np.clip(mc["score_base"] + np.random.uniform(*noise), 0, 10)), 1)
            dims = {"trust_conditions": {"score": score, "confidence": mc["confidence"]}}
            records.append(make_record(msg, dims, "diplomacy"))

    sc = cfg["sampling"]
    low  = [r for r in records if r["dimensions"]["trust_conditions"]["score"] < 5]
    high = [r for r in records if r["dimensions"]["trust_conditions"]["score"] >= 5]
    half = sc["n"] // 2
    np.random.seed(sc["random_seed"])
    if len(low) < half:
        low_sample = (low * (half // max(len(low), 1) + 1))[:half]
        np.random.shuffle(low_sample)
    else:
        idx = np.random.choice(len(low), half, replace=False)
        low_sample = [low[i] for i in idx]
    high_take = min(half, len(high))
    idx = np.random.choice(len(high), high_take, replace=False)
    high_sample = [high[i] for i in idx]
    records = low_sample + high_sample
    np.random.shuffle(records)

    print(f"  Records: {len(records)}")
    return records


def load_casino(cfg, global_cfg):
    print(f"\n{'='*60}\nLoading CaSiNo → contractual_clarity\n{'='*60}")
    p = Path(cfg["file"])
    if not p.exists():
        print("  Not found — skipping")
        return []

    all_recs = []
    with open(p) as f:
        for line in f:
            all_recs.append(json.loads(line))
    print(f"  Total dialogues: {len(all_recs)}")

    pos_strats = set(cfg["strategy_clarity"]["positive"])
    neg_strats = set(cfg["strategy_clarity"]["negative"])
    sat_bonus  = cfg["satisfaction_bonus"]
    out_bonus  = cfg["outcome_bonus"]
    sf         = cfg["score_formula"]
    cf         = cfg["confidence_formula"]
    max_len    = cfg.get("max_text_length", 2000)

    records = []
    for conv in all_recs:
        chat_logs   = conv.get("chat_logs", [])
        annotations = conv.get("annotations", [])
        p_info      = conv.get("participant_info", {})

        full_text = " ".join(e.get("text", "") for e in chat_logs if e.get("text", "").strip())
        if len(full_text) < 50:
            continue

        pos_c = neg_c = total_ann = 0
        for ann in annotations:
            if len(ann) < 2:
                continue
            for s in ann[1].split(","):
                s = s.strip()
                total_ann += 1
                if s in pos_strats:
                    pos_c += 1
                elif s in neg_strats:
                    neg_c += 1

        if total_ann == 0:
            continue

        score = (sf["base"] + (pos_c / total_ann) * sf["pos_ratio_multiplier"]
                             + (neg_c / total_ann) * sf["neg_ratio_multiplier"])

        for agent_key in ["mturk_agent_1", "mturk_agent_2"]:
            sat = str(p_info.get(agent_key, {}).get("outcomes", {}).get("satisfaction", ""))
            for key, bonus in sat_bonus.items():
                if key in sat:
                    score += bonus

        last_data = next((e.get("task_data", {}).get("data", "") for e in reversed(chat_logs)
                          if e.get("task_data", {}).get("data", "")), "")
        for key, bonus in out_bonus.items():
            if key in last_data:
                score += bonus

        score = round(float(np.clip(score, 0, 10)), 1)
        conf  = round(float(np.clip(cf["base"] + total_ann * cf["per_annotation"],
                                    cf["min"], cf["max"])), 2)
        dims  = {"contractual_clarity": {"score": score, "confidence": conf}}
        records.append(make_record(full_text[:max_len], dims, "casino"))

    sc = cfg["sampling"]
    if len(records) > sc["n"]:
        np.random.seed(sc["random_seed"])
        idx = np.random.choice(len(records), sc["n"], replace=False)
        records = [records[i] for i in idx]

    print(f"  Records: {len(records)}")
    return records


def load_politeness(cfg, global_cfg):
    print(f"\n{'='*60}\nLoading Stanford Politeness → authority_dynamics\n{'='*60}")
    records = []
    for src_name, fpath in cfg["files"].items():
        p = Path(fpath)
        if not p.exists():
            print(f"  {src_name}: not found")
            continue
        import csv
        with open(p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"  {src_name}: {len(rows)} rows")

        sf = cfg["score_formula"]
        cf = cfg["confidence_formula"]

        for row in rows:
            text = row.get("Request", "").strip()
            if len(text) < 30:
                continue
            try:
                norm = float(row[sf["source_field"]])
            except (ValueError, KeyError):
                continue
            score = round(float(np.clip(
                sf["target_center"] + norm * sf["multiplier"],
                sf["target_min"], sf["target_max"]
            )), 1)

            ind_scores = []
            for f_name in cf["score_fields"]:
                s = row.get(f_name)
                if s:
                    try:
                        ind_scores.append(float(s))
                    except ValueError:
                        pass

            if len(ind_scores) >= cf.get("min_annotators", 3):
                std = np.std(ind_scores)
                c = cf["base"] + std * cf["std_multiplier"]
                c *= cf.get("scale", 1.0)
                conf = round(float(np.clip(c, cf.get("min", 0), cf.get("max", 1))), 2)
            else:
                conf = cf.get("min", 0.15)

            dims = {"authority_dynamics": {"score": score, "confidence": conf}}
            records.append(make_record(text, dims, f"politeness_{src_name}"))

    sc = cfg["sampling"]
    if len(records) > sc["n"]:
        np.random.seed(sc["random_seed"])
        idx = np.random.choice(len(records), sc["n"], replace=False)
        records = [records[i] for i in idx]

    print(f"  Records: {len(records)}")
    return records


def load_prosocial(cfg, global_cfg):
    print(f"\n{'='*60}\nLoading ProsocialDialog → defensive_architecture\n{'='*60}")
    p = Path(cfg["file"])
    if not p.exists():
        print("  Not found — skipping")
        return []

    all_recs = []
    with open(p) as f:
        for line in f:
            all_recs.append(json.loads(line))
    print(f"  Total records: {len(all_recs)}")

    safety_map   = cfg["safety_to_score"]
    agree_bonus  = cfg["annotation_agreement_bonus"]
    score_noise  = cfg.get("score_noise", 0.8)
    min_text_len = cfg.get("min_text_length", global_cfg["min_text_length"])

    records = []
    for rec in all_recs:
        ctx = rec.get("context", "").strip()
        if len(ctx) < min_text_len:
            continue
        label = rec.get("safety_label", "")
        if label not in safety_map:
            continue
        mc = safety_map[label]
        base_score = mc["base_score"]
        base_conf  = mc["base_conf"]

        anns = rec.get("safety_annotations", [])
        if len(anns) >= 3:
            if len(set(anns)) == 1:
                base_conf += agree_bonus["unanimous"]
            elif len(set(anns)) == 2:
                base_conf += agree_bonus["majority"]

        score = round(float(np.clip(base_score + np.random.uniform(-score_noise, score_noise), 0, 10)), 1)
        conf  = round(float(np.clip(base_conf, 0.25, 0.65)), 2)
        dims  = {"defensive_architecture": {"score": score, "confidence": conf}}
        records.append(make_record(ctx, dims, "prosocial"))

    sc = cfg["sampling"]
    sc_buckets = sc.get("buckets", {})
    if sc_buckets:
        by_bucket = {}
        for rec in records:
            s = rec["dimensions"]["defensive_architecture"]["score"]
            for name, (lo, hi) in sc_buckets.items():
                if lo <= s < hi:
                    by_bucket.setdefault(name, []).append(rec)
                    break
        np.random.seed(sc["random_seed"])
        per_bucket = sc["n"] // len(sc_buckets)
        sampled = []
        for bucket, pool in by_bucket.items():
            take = min(per_bucket, len(pool))
            if take > 0:
                idx = np.random.choice(len(pool), take, replace=False)
                sampled.extend(pool[i] for i in idx)
        records = sampled
    elif len(records) > sc["n"]:
        np.random.seed(sc["random_seed"])
        idx = np.random.choice(len(records), sc["n"], replace=False)
        records = [records[i] for i in idx]

    print(f"  Records: {len(records)}")
    return records


# ============================================================
# Dispatch table
# ============================================================

LOADERS = {
    "berkeley":              load_berkeley,
    "civil_comments":        load_civil_comments,
    "goemotions":            load_goemotions,
    "ucc":                   load_ucc,
    "dreaddit":              load_dreaddit,
    "esconv":                load_esconv,
    "empathetic_dialogues":  load_empathetic_dialogues,
    "diplomacy":             load_diplomacy,
    "casino":                load_casino,
    "politeness":            load_politeness,
    "prosocial":             load_prosocial,
}


# ============================================================
# Main
# ============================================================

def main():
    config     = load_config()
    global_cfg = config["global"]
    np.random.seed(global_cfg["random_seed"])

    print(f"Building composite ground truth  (config version: {config['version']})")
    print(f"Mappings config: data/dataset_mappings.json\n")

    all_records = []
    for dataset_id, dataset_cfg in config["datasets"].items():
        if dataset_id not in LOADERS:
            print(f"  [SKIP] {dataset_id}: no loader")
            continue
        records = LOADERS[dataset_id](dataset_cfg, global_cfg)
        all_records.extend(records)

    print(f"\n{'='*60}\nTOTAL: {len(all_records)} records\n{'='*60}")

    dim_counts    = {d: 0 for d in DIMS}
    source_counts = {}
    for rec in all_records:
        source_counts[rec["source"]] = source_counts.get(rec["source"], 0) + 1
        for dim_id in rec["dimensions"]:
            if dim_id in dim_counts:
                dim_counts[dim_id] += 1

    print("\n  By source:")
    for src, count in sorted(source_counts.items()):
        print(f"    {src:30s}: {count:5d}")

    print("\n  By dimension:")
    for dim_id in DIMS:
        count = dim_counts[dim_id]
        tier  = "A" if count > 2000 else "B" if count > 500 else "C"
        print(f"    {dim_id:28s}: {count:5d}  (tier {tier})")

    out_path = Path("data/composite-ground-truth.jsonl")
    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n  Written: {out_path}  ({size_mb:.1f} MB, {len(all_records)} records)")
    print(f"  Config version: {config['version']}")


if __name__ == "__main__":
    main()
