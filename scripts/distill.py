"""
Train a student model to predict PSQ dimension scores from text.

Architecture: transformer encoder → 10 dimension heads,
each outputting (score, confidence).

Supported encoders (via --model-name):
  - distilbert-base-uncased    (default, 66M params, WordPiece tokenizer)
  - microsoft/deberta-v3-small (141M params, SentencePiece tokenizer)

Training data sources (default: data/psq.db via best_scores view):
  - DB path read via --db (default: data/psq.db)
  - best_scores view picks highest-priority label per (text, dim):
      separated-llm(1) > synthetic(2) > joint-llm(3) > composite-proxy(4)
  - Split assignments persisted in splits table (80/10/10 md5 hash, frozen)
  - Per-dimension sample weights: 5.0 for LLM/synthetic, 1.5 for proxy

  --no-db falls back to reading JSONL files directly (legacy mode):
    composite-ground-truth.jsonl, train-proxy.jsonl, train-llm.jsonl

Loss: MSE(score) + 0.25 * MSE(confidence), masked where conf < threshold.
Per-dimension weighting: sample_weight * conf^2 so low-confidence proxy
data contributes proportionally less than gold-standard LLM labels.

Usage:
  python scripts/distill.py
  python scripts/distill.py --epochs 15 --lr 3e-5
  python scripts/distill.py --no-db   # legacy JSONL mode
  python scripts/distill.py --eval-only --checkpoint models/psq-student/best
"""
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force unbuffered stdout so training progress is visible in real-time
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import math
import sqlite3
import time
import numpy as np
from pathlib import Path
from scipy import stats

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

DIMENSIONS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]
DIM_TO_IDX = {d: i for i, d in enumerate(DIMENSIONS)}
N_DIMS = len(DIMENSIONS)

# Default hyperparameters
DEFAULTS = {
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 32,
    "epochs": 10,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "conf_loss_weight": 0.25,
    "conf_mask_threshold": 0.15,
    "llm_weight": 5.0,
    "composite_weight": 1.5,
    "proxy_weight": 1.0,
    "patience": 3,
    "seed": 42,
}


# =====================================================================
# Dataset
# =====================================================================
class PSQDataset(Dataset):
    """PSQ training dataset.

    Each item has:
      - text: input string
      - scores: [N_DIMS] float tensor (0-10 scale, NaN where missing)
      - confidences: [N_DIMS] float tensor (0-1, NaN where missing)
      - mask: [N_DIMS] bool tensor (True where dimension has a label)
      - weights: [N_DIMS] float tensor — per-dimension sample weight.
          DB mode: from training_data.sample_weight (5.0 LLM/synthetic, 1.5 proxy)
          JSONL mode: scalar broadcast from teacher field
    """

    def __init__(self, records, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []

        for rec in records:
            scores = np.full(N_DIMS, np.nan)
            confs = np.full(N_DIMS, np.nan)
            mask = np.zeros(N_DIMS, dtype=bool)
            # per-dim weights: default to proxy weight, overridden per dim
            dim_weights = np.full(N_DIMS, DEFAULTS["proxy_weight"])

            dims = rec.get("dimensions", {})
            # DB records carry "dim_weights" dict; JSONL records do not
            jsonl_dim_weights = rec.get("dim_weights", {})

            for dim_id, val in dims.items():
                if dim_id not in DIM_TO_IDX:
                    continue
                idx = DIM_TO_IDX[dim_id]
                score = val.get("score")
                conf = val.get("confidence", 0.5)
                if score is not None and not (isinstance(score, float) and math.isnan(score)):
                    scores[idx] = score
                    confs[idx] = conf
                    mask[idx] = True
                    if dim_id in jsonl_dim_weights:
                        dim_weights[idx] = jsonl_dim_weights[dim_id]

            # JSONL mode: derive scalar weight from teacher/source, broadcast to all dims
            if not jsonl_dim_weights:
                source = rec.get("teacher", rec.get("source", "proxy"))
                if source in ("llm", "llm_labeled", "separated-llm"):
                    scalar_w = DEFAULTS["llm_weight"]
                elif source in ("synthetic", "relabeled"):
                    scalar_w = DEFAULTS["llm_weight"]
                elif source in ("berkeley", "civil_comments", "goemotions", "ucc",
                                "dreaddit", "esconv", "empathetic_dialogues",
                                "diplomacy", "casino", "prosocial",
                                "politeness_wikipedia", "politeness_stack-exchange"):
                    scalar_w = DEFAULTS["composite_weight"]
                else:
                    scalar_w = DEFAULTS["proxy_weight"]
                # Apply scalar weight only to labeled dims
                dim_weights = np.where(mask, scalar_w, DEFAULTS["proxy_weight"])

            self.items.append({
                "text": rec["text"],
                "scores": scores,
                "confidences": confs,
                "mask": mask,
                "weights": dim_weights,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "scores": torch.tensor(item["scores"], dtype=torch.float32),
            "confidences": torch.tensor(item["confidences"], dtype=torch.float32),
            "mask": torch.tensor(item["mask"], dtype=torch.bool),
            "weights": torch.tensor(item["weights"], dtype=torch.float32),
        }


# =====================================================================
# Model
# =====================================================================
class PSQStudent(nn.Module):
    """Transformer encoder with 10 dimension heads.

    Each head outputs 2 values: predicted score (0-10) and confidence (0-1).
    Works with any HuggingFace encoder that provides last_hidden_state
    (e.g. DistilBERT, DeBERTa-v3).

    When bifactor=True, adds an 11th head predicting the general factor
    (g-PSQ = mean of 10 dimension scores). The g-head shares the same
    [CLS] projection as the dimension heads.
    """

    def __init__(self, model_name, n_dims=N_DIMS, bifactor=False):
        super().__init__()
        self.bifactor = bifactor
        # use_safetensors=True avoids torch.load CVE issue on torch <2.6;
        # .float() ensures float32 even if weights are stored as float16
        # (e.g. microsoft/deberta-v3-small ships fp16 safetensors).
        self.encoder = AutoModel.from_pretrained(
            model_name, use_safetensors=True
        ).float()
        hidden = self.encoder.config.hidden_size

        # Shared projection
        self.proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Per-dimension heads: each outputs (score, confidence)
        self.heads = nn.ModuleList([
            nn.Linear(hidden // 2, 2) for _ in range(n_dims)
        ])

        # Bifactor: general factor head (g-PSQ), predicts single score (0-10)
        if bifactor:
            self.g_head = nn.Linear(hidden // 2, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls = outputs.last_hidden_state[:, 0, :]
        projected = self.proj(cls)

        scores = []
        confs = []
        for head in self.heads:
            out = head(projected)  # [batch, 2]
            # Score: sigmoid * 10 → range 0-10
            s = torch.sigmoid(out[:, 0]) * 10.0
            # Confidence: sigmoid → range 0-1
            c = torch.sigmoid(out[:, 1])
            scores.append(s)
            confs.append(c)

        # Stack: [batch, n_dims]
        scores = torch.stack(scores, dim=1)
        confs = torch.stack(confs, dim=1)

        if self.bifactor:
            # g-PSQ: sigmoid * 10 → range 0-10
            g_score = torch.sigmoid(self.g_head(projected).squeeze(-1)) * 10.0
            return scores, confs, g_score

        return scores, confs


# =====================================================================
# Training
# =====================================================================
def compute_loss(pred_scores, pred_confs, true_scores, true_confs, mask, weights,
                 conf_weight=0.25, conf_threshold=0.15, conf_mode="teacher",
                 conf_power=2.0, pred_g=None, true_g=None, g_mask=None):
    """Masked loss: only compute where ground truth exists and confidence exceeds threshold.

    Loss is weighted by true_confs**conf_power * sample_weight so low-confidence proxy
    data contributes proportionally less than high-confidence gold-standard labels.

    weights: [batch, N_DIMS] — per-dimension sample weights from training_data.sample_weight.
      5.0 for separated-llm/synthetic/joint-llm, 1.5 for composite-proxy.

    conf_power controls how aggressively low-confidence data is down-weighted:
      1.0 — linear (original): conf=0.26 contributes 50% of conf=0.52
      2.0 — squared (default): conf=0.26 contributes 25% of conf=0.52
      3.0 — cubed: conf=0.26 contributes 12.5% of conf=0.52

    conf_mode controls what the confidence head learns:
      "teacher"  — reproduce teacher confidence (original behavior, but INVERTED for proxy data)
      "accuracy" — predict own accuracy: conf_target = 1 - |score_error| / 5
                   so a prediction off by 2.5 → conf_target = 0.5, perfect → 1.0
      "off"      — no confidence loss (conf_weight forced to 0)

    Bifactor (when pred_g, true_g, g_mask are provided):
      Adds MSE loss for g-PSQ (general factor) prediction, weighted 1.0.
      g-PSQ target is mean of available dimension scores per text.
    """
    # Additional mask: exclude very low confidence ground truth
    conf_mask = mask & (true_confs > conf_threshold)

    if conf_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Replace NaN with 0 for safe computation (masked out anyway)
    safe_true_scores = torch.where(conf_mask, true_scores, torch.zeros_like(true_scores))
    safe_true_confs = torch.where(conf_mask, true_confs, torch.zeros_like(true_confs))

    # weights is [batch, N_DIMS] (per-dim sample weights from DB or broadcast scalar)
    # conf^power * sample_weight: proxy data (conf~0.2, weight=1.5) gets ~0.06x
    # vs separated-llm (conf~0.7, weight=5.0) at ~2.45x — 40x difference
    conf_weights = (safe_true_confs ** conf_power) * conf_mask.float() * weights

    # Score loss (MSE)
    score_diff = (pred_scores - safe_true_scores) ** 2
    score_loss = (score_diff * conf_weights).sum() / conf_weights.sum()

    # Confidence loss
    if conf_mode == "off" or conf_weight == 0:
        total_loss = score_loss
    else:
        if conf_mode == "accuracy":
            # Train confidence to predict own accuracy: 1 - |error|/5 clamped to [0, 1]
            # Error of 0 → target 1.0, error of 5+ → target 0.0
            score_errors = torch.abs(pred_scores.detach() - safe_true_scores)
            conf_targets = torch.clamp(1.0 - score_errors / 5.0, 0.0, 1.0)
        else:
            # Original: reproduce teacher confidence
            conf_targets = safe_true_confs

        conf_diff = (pred_confs - conf_targets) ** 2
        conf_loss = (conf_diff * conf_weights).sum() / conf_weights.sum()
        total_loss = score_loss + conf_weight * conf_loss

    # Bifactor: g-PSQ loss (weighted 1.0, simple MSE on available texts)
    if pred_g is not None and true_g is not None and g_mask is not None:
        if g_mask.sum() > 0:
            g_loss = ((pred_g - true_g) ** 2 * g_mask.float()).sum() / g_mask.float().sum()
            total_loss = total_loss + g_loss

    return total_loss


def evaluate(model, dataloader, device, bifactor=False):
    """Evaluate model, return per-dimension Pearson r and MSE.

    When bifactor=True, also evaluates g-PSQ prediction (mean of available
    dimension scores) and reports it under the 'g_psq' key.
    """
    model.eval()
    all_pred_scores = []
    all_pred_g = []
    all_true_scores = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if bifactor:
                pred_s, pred_c, pred_g = model(input_ids, attention_mask)
                all_pred_g.append(pred_g.cpu())
            else:
                pred_s, pred_c = model(input_ids, attention_mask)

            all_pred_scores.append(pred_s.cpu())
            all_true_scores.append(batch["scores"])
            all_masks.append(batch["mask"])

    pred = torch.cat(all_pred_scores, dim=0).numpy()
    true = torch.cat(all_true_scores, dim=0).numpy()
    masks = torch.cat(all_masks, dim=0).numpy()

    results = {}
    for i, dim_id in enumerate(DIMENSIONS):
        m = masks[:, i] & ~np.isnan(true[:, i])
        if m.sum() < 5:
            results[dim_id] = {"r": float("nan"), "mse": float("nan"), "n": int(m.sum())}
            continue
        p, t = pred[m, i], true[m, i]
        r, _ = stats.pearsonr(p, t)
        mse = float(np.mean((p - t) ** 2))
        results[dim_id] = {"r": round(r, 4), "mse": round(mse, 4), "n": int(m.sum())}

    # Weighted average r (by n) — always based on 10 dims only, not g-PSQ
    rs = [v["r"] for v in results.values() if not np.isnan(v["r"])]
    ns = [v["n"] for v in results.values() if not np.isnan(v["r"])]
    avg_r = sum(r * n for r, n in zip(rs, ns)) / sum(ns) if ns else 0
    results["_avg_r"] = round(avg_r, 4)

    # Bifactor: evaluate g-PSQ prediction
    if bifactor and all_pred_g:
        pred_g = torch.cat(all_pred_g, dim=0).numpy()
        # Compute true g-PSQ = mean of available dim scores per text
        safe_true = np.where(masks & ~np.isnan(true), true, 0.0)
        dim_counts = (masks & ~np.isnan(true)).sum(axis=1)  # [N]
        # Only evaluate texts with at least 1 valid dimension
        g_valid = dim_counts >= 1
        true_g = np.where(g_valid, safe_true.sum(axis=1) / np.maximum(dim_counts, 1), np.nan)

        if g_valid.sum() >= 5:
            g_r, _ = stats.pearsonr(pred_g[g_valid], true_g[g_valid])
            g_mse = float(np.mean((pred_g[g_valid] - true_g[g_valid]) ** 2))
            results["g_psq"] = {"r": round(g_r, 4), "mse": round(g_mse, 4), "n": int(g_valid.sum())}
        else:
            results["g_psq"] = {"r": float("nan"), "mse": float("nan"), "n": int(g_valid.sum())}

    return results


def _load_splits_from_db(db_path, no_cap=False, drop_proxy_dims=None):
    """Load train/val/test records from psq.db using persisted split assignments.

    Returns (train_recs, val_recs, test_recs) where each record has:
      text, dimensions: {dim: {score, confidence}}, dim_weights: {dim: float}

    Uses best_scores view: separated-llm > synthetic > joint-llm > composite-proxy.
    Split assignments come from the splits table (persisted, not re-derived).

    drop_proxy_dims: list of dimension names to exclude proxy labels for.
        Proxy labels for these dims have poor LLM agreement and add noise.
    """
    con = sqlite3.connect(db_path)

    # training_data view is already filtered to split='train' with sample_weight
    if drop_proxy_dims:
        # Fetch with method column so we can filter proxy rows for specific dims
        all_train = con.execute(
            "SELECT text_id, text, dimension, score, confidence, sample_weight, method "
            "FROM training_data"
        ).fetchall()
        drop_set = set(drop_proxy_dims)
        train_rows = []
        dropped = 0
        for r in all_train:
            if r[2] in drop_set and r[6] == 'composite-proxy':
                dropped += 1
            else:
                train_rows.append(r[:6])  # strip method column
        if dropped:
            print(f"  Dropped {dropped} proxy rows for dims: {', '.join(sorted(drop_set))}")
    else:
        train_rows = con.execute(
            "SELECT text_id, text, dimension, score, confidence, sample_weight "
            "FROM training_data"
        ).fetchall()

    if not no_cap:
        train_rows = _cap_score_concentration(train_rows)

    # val and test: query best_scores joined to splits
    val_rows = con.execute(
        """SELECT t.id, t.text, bs.dimension, bs.score, bs.confidence,
                  CASE bs.method
                      WHEN 'separated-llm'   THEN 5.0
                      WHEN 'synthetic'       THEN 5.0
                      WHEN 'joint-llm'       THEN 5.0
                      WHEN 'composite-proxy' THEN 1.5
                  END
           FROM best_scores bs
           JOIN texts  t  ON t.id = bs.text_id
           JOIN splits sp ON sp.text_id = t.id AND sp.split = 'val'"""
    ).fetchall()

    test_rows = con.execute(
        """SELECT t.id, t.text, bs.dimension, bs.score, bs.confidence,
                  CASE bs.method
                      WHEN 'separated-llm'   THEN 5.0
                      WHEN 'synthetic'       THEN 5.0
                      WHEN 'joint-llm'       THEN 5.0
                      WHEN 'composite-proxy' THEN 1.5
                  END
           FROM best_scores bs
           JOIN texts  t  ON t.id = bs.text_id
           JOIN splits sp ON sp.text_id = t.id AND sp.split = 'test'"""
    ).fetchall()
    con.close()

    return _rows_to_records(train_rows), _rows_to_records(val_rows), _rows_to_records(test_rows)


def _rows_to_records(rows):
    """Group (text_id, text, dim, score, conf, weight) rows into per-text records."""
    texts = {}
    for text_id, text, dim, score, conf, weight in rows:
        if text_id not in texts:
            texts[text_id] = {"text": text, "dimensions": {}, "dim_weights": {}}
        texts[text_id]["dimensions"][dim] = {"score": score, "confidence": conf}
        texts[text_id]["dim_weights"][dim] = weight
    return list(texts.values())


def _cap_score_concentration(rows, cap=0.30, reduced_weight=1.5):
    """Reduce weight of over-concentrated score values per dimension.

    If >cap fraction of a dimension's rows share the same rounded score,
    randomly select excess rows and reduce their sample_weight to reduced_weight.
    """
    from collections import defaultdict
    import random as _rng

    # Group row indices by dimension
    by_dim = defaultdict(list)
    for i, (text_id, text, dim, score, conf, weight) in enumerate(rows):
        by_dim[dim].append(i)

    # Convert to mutable lists
    rows = [list(r) for r in rows]

    capped_dims = []
    for dim, indices in by_dim.items():
        total = len(indices)
        score_groups = defaultdict(list)
        for i in indices:
            rounded = round(rows[i][3])  # score is index 3
            score_groups[rounded].append(i)

        for score_val, group_indices in score_groups.items():
            frac = len(group_indices) / total
            if frac > cap:
                max_keep = int(total * cap)
                excess = len(group_indices) - max_keep
                _rng.seed(42)
                _rng.shuffle(group_indices)
                for i in group_indices[max_keep:]:
                    rows[i][5] = reduced_weight  # weight is index 5
                capped_dims.append(
                    f"{dim} score={score_val} ({len(group_indices)}/{total}"
                    f"={frac:.0%}, {excess} down-weighted)"
                )

    if capped_dims:
        print(f"  Score concentration cap ({cap:.0%}):")
        for msg in capped_dims:
            print(f"    {msg}")

    return [tuple(r) for r in rows]


def _load_splits_from_jsonl(data_dir):
    """Legacy: load from JSONL files and split by md5 hash (matching v13 behavior)."""
    import hashlib
    all_records = []

    for fname in ["composite-ground-truth.jsonl", "train-proxy.jsonl", "train-llm.jsonl"]:
        fpath = data_dir / fname
        if fpath.exists():
            count = 0
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_records.append(json.loads(line))
                        count += 1
            print(f"  {fname}: {count} records")
        else:
            print(f"  {fname}: not found (skipping)")

    # Deduplicate: keep LLM version over composite
    seen_texts = {}
    dedup_records = []
    for rec in all_records:
        text = rec["text"]
        teacher = rec.get("teacher", rec.get("source", "proxy"))
        is_llm = teacher in ("llm", "llm_labeled")
        if text in seen_texts:
            prev_idx, prev_is_llm = seen_texts[text]
            if is_llm and not prev_is_llm:
                dedup_records[prev_idx] = rec
                seen_texts[text] = (prev_idx, True)
        else:
            seen_texts[text] = (len(dedup_records), is_llm)
            dedup_records.append(rec)

    n_removed = len(all_records) - len(dedup_records)
    if n_removed > 0:
        print(f"  Deduplicated: removed {n_removed} duplicates (kept LLM over composite)")

    train_recs, val_recs, test_recs = [], [], []
    for rec in dedup_records:
        h = int(hashlib.md5(rec["text"].encode()).hexdigest(), 16) % 100
        if h < 80:
            train_recs.append(rec)
        elif h < 90:
            val_recs.append(rec)
        else:
            test_recs.append(rec)

    return train_recs, val_recs, test_recs


def train(args):
    torch.manual_seed(DEFAULTS["seed"])
    np.random.seed(DEFAULTS["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\nLoading training data...")
    data_dir = ROOT / "data"
    db_path = ROOT / args.db

    if not args.no_db and db_path.exists():
        drop_dims = args.drop_proxy_dims.split(",") if args.drop_proxy_dims else None
        train_recs, val_recs, test_recs = _load_splits_from_db(
            db_path, no_cap=args.no_cap, drop_proxy_dims=drop_dims)
        print(f"  DB: {db_path.name}")
        print(f"  Split: {len(train_recs)} train / {len(val_recs)} val / {len(test_recs)} test")
    else:
        if not args.no_db:
            print(f"  WARNING: {db_path} not found, falling back to JSONL")
        train_recs, val_recs, test_recs = _load_splits_from_jsonl(data_dir)
        print(f"  Split: {len(train_recs)} train / {len(val_recs)} val / {len(test_recs)} test")

    if not train_recs:
        print("ERROR: No training data found")
        return

    np.random.shuffle(train_recs)

    # Tokenizer + datasets
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = PSQDataset(train_recs, tokenizer, args.max_length)
    val_ds = PSQDataset(val_recs, tokenizer, args.max_length)
    test_ds = PSQDataset(test_recs, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    # Model
    print(f"\nLoading model: {args.model_name}")
    bifactor = getattr(args, "bifactor", False)
    model = PSQStudent(args.model_name, bifactor=bifactor).to(device)
    if bifactor:
        print(f"  Bifactor: g-PSQ head enabled (11th output head)")
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_val_r = -1
    patience_counter = 0
    if args.no_save:
        import tempfile
        _tmp_dir = tempfile.mkdtemp(prefix="psq-smoke-")
        save_dir = Path(_tmp_dir)
    else:
        save_dir = Path(args.out)
        if not save_dir.is_absolute():
            save_dir = ROOT / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs, {len(train_loader)} batches/epoch")
    eff_batch = args.batch_size * args.grad_accum
    print(f"  LR: {args.lr}, batch: {args.batch_size} (effective: {eff_batch}), patience: {args.patience}")
    print(f"  Confidence mode: {args.conf_mode}", end="")
    if args.conf_mode == "two-phase":
        print(f" (off for {args.conf_warmup_epochs} epochs, then accuracy)")
    else:
        print()
    if args.conf_power != 1.0:
        print(f"  Confidence power: {args.conf_power} (conf^{args.conf_power} weighting)")

    train_start = time.time()
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            true_scores = batch["scores"].to(device)
            true_confs = batch["confidences"].to(device)
            mask = batch["mask"].to(device)
            weights = batch["weights"].to(device)  # [batch, N_DIMS]

            # Replace NaN with 0 for torch
            true_scores = torch.where(torch.isnan(true_scores), torch.zeros_like(true_scores), true_scores)
            true_confs = torch.where(torch.isnan(true_confs), torch.zeros_like(true_confs), true_confs)

            if bifactor:
                pred_scores, pred_confs, pred_g = model(input_ids, attention_mask)
                # g-PSQ target = mean of available dim scores per text
                g_conf_mask = mask & (true_confs > args.conf_threshold)
                safe_scores_for_g = torch.where(g_conf_mask, true_scores, torch.zeros_like(true_scores))
                g_counts = g_conf_mask.float().sum(dim=1)  # [batch]
                g_mask_valid = g_counts >= 1  # at least 1 valid dim
                true_g = torch.where(g_mask_valid, safe_scores_for_g.sum(dim=1) / g_counts.clamp(min=1), torch.zeros_like(g_counts))
            else:
                pred_scores, pred_confs = model(input_ids, attention_mask)
                pred_g, true_g, g_mask_valid = None, None, None

            # Determine confidence mode for this epoch
            if args.conf_mode == "two-phase":
                epoch_conf_mode = "off" if epoch <= args.conf_warmup_epochs else "accuracy"
            else:
                epoch_conf_mode = args.conf_mode

            loss = compute_loss(pred_scores, pred_confs, true_scores, true_confs, mask, weights,
                                args.conf_weight, args.conf_threshold, conf_mode=epoch_conf_mode,
                                conf_power=args.conf_power,
                                pred_g=pred_g, true_g=true_g, g_mask=g_mask_valid)

            # Scale loss for gradient accumulation
            if args.grad_accum > 1:
                loss = loss / args.grad_accum

            loss.backward()

            if (n_batches + 1) % args.grad_accum == 0 or (n_batches + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * (args.grad_accum if args.grad_accum > 1 else 1)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_results = evaluate(model, val_loader, device, bifactor=bifactor)
        val_r = val_results["_avg_r"]

        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)

        print(f"  Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  val_r={val_r:.4f}  [{epoch_elapsed:.0f}s]", end="")

        # Per-dimension detail
        for dim_id in DIMENSIONS:
            r = val_results[dim_id]["r"]
            if not np.isnan(r):
                print(f"  {dim_id[:4]}={r:.2f}", end="")
        # g-PSQ if bifactor
        if bifactor and "g_psq" in val_results and not np.isnan(val_results["g_psq"]["r"]):
            print(f"  g={val_results['g_psq']['r']:.2f}", end="")
        print()

        # Early stopping
        if val_r > best_val_r:
            best_val_r = val_r
            patience_counter = 0
            # Save best
            torch.save(model.state_dict(), save_dir / "best.pt")
            with open(save_dir / "best_results.json", "w") as f:
                json.dump({"epoch": epoch, "val_r": val_r, "val_results": val_results}, f, indent=2, cls=NumpyEncoder)
            print(f"    -> New best (r={val_r:.4f}), saved to {save_dir / 'best.pt'}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    -> Early stopping (patience={args.patience})")
                break

    # Timing summary
    total_elapsed = time.time() - train_start
    avg_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    print(f"\nTiming: {total_elapsed:.0f}s total, {avg_epoch:.0f}s/epoch avg, "
          f"{len(epoch_times)} epochs, {len(train_recs)} train samples")

    # Final evaluation on test set
    print(f"\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(save_dir / "best.pt", weights_only=True))
    test_results = evaluate(model, test_loader, device, bifactor=bifactor)

    print(f"\nTest Results (best model):")
    print(f"  {'Dimension':<25s} {'r':>8s} {'MSE':>8s} {'n':>6s}")
    print(f"  {'-'*49}")
    for dim_id in DIMENSIONS:
        r = test_results[dim_id]
        print(f"  {dim_id:<25s} {r['r']:+8.4f} {r['mse']:8.4f} {r['n']:6d}")
    if bifactor and "g_psq" in test_results:
        g = test_results["g_psq"]
        if not np.isnan(g["r"]):
            print(f"  {'g_psq':<25s} {g['r']:+8.4f} {g['mse']:8.4f} {g['n']:6d}")
        else:
            print(f"  {'g_psq':<25s} {'N/A':>8s} {'N/A':>8s} {g['n']:6d}")
    print(f"  {'AVERAGE':<25s} {test_results['_avg_r']:+8.4f}")

    # Save test results
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, cls=NumpyEncoder)

    if not args.no_save:
        # Save config
        config = {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "dimensions": DIMENSIONS,
            "n_dims": N_DIMS,
            "bifactor": bifactor,
            "data_source": "db" if (not args.no_db and (ROOT / args.db).exists()) else "jsonl",
            "hyperparams": {k: getattr(args, k, v) for k, v in DEFAULTS.items()},
            "timing": {
                "total_seconds": round(total_elapsed, 1),
                "avg_epoch_seconds": round(avg_epoch, 1),
                "epoch_times": [round(t, 1) for t in epoch_times],
                "n_epochs_run": len(epoch_times),
                "n_train_samples": len(train_recs),
                },
            }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)

        # Save tokenizer for ONNX/JS inference
        tokenizer_dir = save_dir / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_dir))
        print(f"  Tokenizer saved to {tokenizer_dir}/")
        print(f"\nModel saved to {save_dir}/")
    else:
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)
        print("\n[--no-save] Checkpoint discarded.")
    return test_results


def main():
    parser = argparse.ArgumentParser(description="Train PSQ student model")
    parser.add_argument("--model-name", default=DEFAULTS["model_name"])
    parser.add_argument("--max-length", type=int, default=DEFAULTS["max_length"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--conf-weight", type=float, default=DEFAULTS["conf_loss_weight"])
    parser.add_argument("--conf-threshold", type=float, default=DEFAULTS["conf_mask_threshold"])
    parser.add_argument("--conf-mode", choices=["teacher", "accuracy", "two-phase"],
                       default="two-phase",
                       help="Confidence training mode. 'teacher': reproduce teacher conf (original). "
                            "'accuracy': predict own error. 'two-phase': no conf loss for first 2 "
                            "epochs, then switch to accuracy mode.")
    parser.add_argument("--conf-warmup-epochs", type=int, default=2,
                       help="In two-phase mode, epochs with no confidence loss before switching to accuracy.")
    parser.add_argument("--conf-power", type=float, default=2.0,
                       help="Exponent for confidence weighting. 1.0=linear (original), 2.0=squared "
                            "(default, strongly down-weights low-conf proxy data).")
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--grad-accum", type=int, default=1,
                       help="Gradient accumulation steps. Effective batch = batch-size * grad-accum.")
    parser.add_argument("--db", default="data/psq.db",
                       help="Path to psq.db (relative to project root, default: data/psq.db)")
    parser.add_argument("--no-db", action="store_true",
                       help="Force JSONL loading even if psq.db exists (legacy mode)")
    parser.add_argument("--out", default="models/psq-student",
                       help="Output directory for checkpoints and config (default: models/psq-student)")
    parser.add_argument("--no-cap", action="store_true",
                       help="Disable score-concentration cap (default: cap enabled)")
    parser.add_argument("--drop-proxy-dims", nargs="?",
                       const="threat_exposure,trust_conditions,contractual_clarity,authority_dynamics",
                       default=None,
                       help="Drop proxy labels for these dims (poor LLM agreement). "
                            "No args = default set (TE, TC, CC, AD). "
                            "Or specify comma-separated dims.")
    parser.add_argument("--no-save", action="store_true",
                       help="Discard checkpoints after training (smoke-test mode)")
    parser.add_argument("--bifactor", action="store_true",
                       help="Add g-PSQ general factor head (11th output). "
                            "g-PSQ target = mean of available dim scores. "
                            "Loss weight 1.0, same [CLS] projection as dim heads.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if args.eval_only:
        ckpt = args.checkpoint or str(ROOT / "models" / "psq-student" / "best.pt")
        print(f"Eval-only mode: use scripts/eval_held_out.py instead")
        return

    train(args)


if __name__ == "__main__":
    main()
