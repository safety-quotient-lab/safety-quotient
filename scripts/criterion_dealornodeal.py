#!/usr/bin/env python3
"""
Criterion validity study: Deal or No Deal negotiation dataset
Lewis et al. (2017) "Deal or No Deal? End-to-End Learning for Negotiation Dialogues"

Replicates methodology from CaSiNo (§30) and CGA-Wiki (§31) studies.

Outcomes:
  - Deal agreement (binary): did the negotiation reach a deal?
  - Points scored (continuous, 0-10): how many points did each participant get?
  - Joint value (continuous, 0-20): total points for both participants (Pareto efficiency)

Analysis:
  - Group comparison (deal vs. no-deal): Mann-Whitney U, Cohen's d
  - Logistic regression: AUC for deal prediction
  - Pearson/point-biserial correlations
  - Text length baseline
  - g-PSQ vs 10-dim comparison
  - Temporal analysis (early turns vs all turns)
  - Partial correlations controlling text length
"""

import sys, os, json, warnings
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Model definition ───
class PSQStudent(nn.Module):
    DIMS = ["threat_exposure","hostility_index","authority_dynamics","energy_dissipation",
            "regulatory_capacity","resilience_baseline","trust_conditions","cooling_capacity",
            "defensive_architecture","contractual_clarity"]
    DIM_SHORT = ["TE","HI","AD","ED","RC","RB","TC","CC","DA","CO"]

    def __init__(self, model_name="distilbert-base-uncased", n_dims=10):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
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

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        projected = self.proj(cls)

        scores = []
        confs = []
        for head in self.heads:
            out = head(projected)  # [batch, 2]
            s = torch.sigmoid(out[:, 0]) * 10.0
            c = torch.sigmoid(out[:, 1])
            scores.append(s)
            confs.append(c)

        return torch.stack(scores, dim=1), torch.stack(confs, dim=1)


def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tok_path = os.path.join(base, "models/psq-student/tokenizer")
    model_path = os.path.join(base, "models/psq-student/best.pt")

    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    model = PSQStudent()
    cp = torch.load(model_path, map_location="cpu", weights_only=False)
    # Checkpoint may be raw state_dict or wrapped dict
    if isinstance(cp, dict) and "model_state_dict" in cp:
        model.load_state_dict(cp["model_state_dict"])
    else:
        model.load_state_dict(cp)
    model.eval()
    return model, tokenizer


def score_texts(model, tokenizer, texts, batch_size=64, max_length=128):
    """Score a list of texts, return (n, 10) array of scores."""
    all_scores = []
    device = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        # Remove token_type_ids if present (not accepted by DistilBERT model)
        enc.pop("token_type_ids", None)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            scores, confs = model(**enc)
        all_scores.append(scores.cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  Scored {min(i+batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(all_scores)


def parse_dataset():
    """Load and parse the Deal or No Deal dataset."""
    print("Loading Deal or No Deal dataset...")
    ds = load_dataset('deal_or_no_dialog', trust_remote_code=True)

    records = []
    for split in ['train', 'test', 'validation']:
        for row in ds[split]:
            dialogue = row['dialogue']
            output = row['output']
            you_values = row['input']['value']
            you_counts = row['input']['count']
            partner_values = row['partner_input']['value']
            partner_counts = row['partner_input']['count']

            # Parse deal outcome (<disagree>, <no_agreement>, <disconnect> all mean no deal)
            is_deal = ('<disagree>' not in output
                       and '<no_agreement>' not in output
                       and '<disconnect>' not in output)

            # Parse points
            you_points = 0
            partner_points = 0
            if is_deal:
                parts = output.split()
                you_items = [int(p.split('=')[1]) for p in parts[:3]]
                partner_items = [int(p.split('=')[1]) for p in parts[3:]]
                you_points = sum(v * c for v, c in zip(you_values, you_items))
                partner_points = sum(v * c for v, c in zip(partner_values, partner_items))

            # Max possible points for each
            you_max = sum(v * c for v, c in zip(you_values, you_counts))
            partner_max = sum(v * c for v, c in zip(partner_values, partner_counts))

            # Clean dialogue text: remove <eos> and <selection> markers
            clean = dialogue.replace('<eos>', '.').replace('<selection>', '').strip()
            # Remove trailing period if exists
            clean = clean.rstrip('. ')

            # Count turns
            turns = [t.strip() for t in dialogue.split('<eos>') if t.strip() and '<selection>' not in t]
            n_turns = len(turns)

            # Separate YOU and THEM turns
            you_turns = [t.strip().replace('YOU:', '').strip() for t in turns if t.strip().startswith('YOU:')]
            them_turns = [t.strip().replace('THEM:', '').strip() for t in turns if t.strip().startswith('THEM:')]

            # Build early turns text (first half)
            half = max(1, n_turns // 2)
            early_turns = turns[:half]
            early_text = '. '.join(t.strip() for t in early_turns)

            # First turn only
            first_turn_text = turns[0].strip() if turns else clean

            records.append({
                'text': clean,
                'early_text': early_text,
                'first_turn_text': first_turn_text,
                'is_deal': is_deal,
                'you_points': you_points,
                'partner_points': partner_points,
                'joint_points': you_points + partner_points,
                'you_max': you_max,
                'partner_max': partner_max,
                'you_efficiency': you_points / you_max if you_max > 0 else 0,
                'n_turns': n_turns,
                'text_length': len(clean),
                'split': split,
                'you_text': '. '.join(you_turns),
                'them_text': '. '.join(them_turns),
            })

    return records


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def partial_corr(x, y, z):
    """Partial correlation between x and y controlling for z."""
    from scipy.stats import pearsonr
    from numpy.linalg import lstsq
    # Residualize x and y on z (with intercept)
    z = np.array(z).reshape(-1, 1)
    z_int = np.column_stack([np.ones(len(z)), z])
    bx = lstsq(z_int, x, rcond=None)[0]
    by = lstsq(z_int, y, rcond=None)[0]
    rx = x - z_int @ bx
    ry = y - z_int @ by
    return pearsonr(rx, ry)


def main():
    DIMS = PSQStudent.DIMS
    DIM_SHORT = PSQStudent.DIM_SHORT

    # ─── 1. Load data ───
    records = parse_dataset()
    print(f"\nTotal records: {len(records)}")
    n_deal = sum(1 for r in records if r['is_deal'])
    n_nodeal = sum(1 for r in records if not r['is_deal'])
    print(f"Deal: {n_deal} ({n_deal/len(records)*100:.1f}%)")
    print(f"No deal: {n_nodeal} ({n_nodeal/len(records)*100:.1f}%)")

    # ─── 2. Load model and score (with caching) ───
    cache_path = "/tmp/dond_psq_scores.npz"
    if os.path.exists(cache_path):
        print(f"\nLoading cached scores from {cache_path}")
        cached = np.load(cache_path)
        scores_all = cached['scores_all']
        scores_early = cached['scores_early']
        scores_first = cached['scores_first']
        assert len(scores_all) == len(records), f"Cache mismatch: {len(scores_all)} vs {len(records)}"
    else:
        print("\nLoading PSQ student model (psq-student/best.pt)....")
        model, tokenizer = load_model()

        texts = [r['text'] for r in records]
        print(f"Scoring {len(texts)} dialogues (all turns)...")
        scores_all = score_texts(model, tokenizer, texts)

        early_texts = [r['early_text'] for r in records]
        print(f"Scoring {len(early_texts)} dialogues (early turns)...")
        scores_early = score_texts(model, tokenizer, early_texts)

        first_texts = [r['first_turn_text'] for r in records]
        print(f"Scoring {len(first_texts)} dialogues (first turn)...")
        scores_first = score_texts(model, tokenizer, first_texts)

        np.savez(cache_path, scores_all=scores_all, scores_early=scores_early, scores_first=scores_first)
        print(f"Saved scores to {cache_path}")

    # g-PSQ = mean of all 10 dimensions
    g_psq_all = scores_all.mean(axis=1)
    g_psq_early = scores_early.mean(axis=1)
    g_psq_first = scores_first.mean(axis=1)

    # ─── 3. Dataset description ───
    print("\n" + "="*80)
    print("CRITERION VALIDITY: DEAL OR NO DEAL NEGOTIATION DATASET")
    print("Lewis et al. (2017)")
    print("="*80)

    print(f"\n### Dataset Description")
    print(f"  Total dialogues: {len(records)}")
    print(f"  Deal reached:    {n_deal} ({n_deal/len(records)*100:.1f}%)")
    print(f"  No deal:         {n_nodeal} ({n_nodeal/len(records)*100:.1f}%)")

    # Text statistics
    lengths = [r['text_length'] for r in records]
    turns = [r['n_turns'] for r in records]
    print(f"\n  Text length: mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}, "
          f"min={np.min(lengths)}, max={np.max(lengths)}")
    print(f"  Turn count:  mean={np.mean(turns):.1f}, median={np.median(turns):.0f}, "
          f"min={np.min(turns)}, max={np.max(turns)}")

    # Points statistics (deals only)
    deal_records = [r for r in records if r['is_deal']]
    you_pts = [r['you_points'] for r in deal_records]
    joint_pts = [r['joint_points'] for r in deal_records]
    efficiency = [r['you_efficiency'] for r in deal_records]
    print(f"\n  Points scored (deals only, n={len(deal_records)}):")
    print(f"    YOU points:   mean={np.mean(you_pts):.2f}, std={np.std(you_pts):.2f}")
    print(f"    Joint points: mean={np.mean(joint_pts):.2f}, std={np.std(joint_pts):.2f}")
    print(f"    Efficiency:   mean={np.mean(efficiency):.3f}, std={np.std(efficiency):.3f}")

    # Split distribution
    for split in ['train', 'test', 'validation']:
        n = sum(1 for r in records if r['split'] == split)
        print(f"  {split}: {n}")

    print(f"\n  Outcomes available:")
    print(f"    - Deal agreement (binary): {n_deal} deals, {n_nodeal} no-deal")
    print(f"    - Points scored (continuous, 0-10): for deal participants only")
    print(f"    - Joint value (continuous, 0-20): Pareto efficiency measure")
    print(f"    - No subjective ratings (satisfaction, likeness) available")

    # PSQ score summary
    print(f"\n### PSQ Score Summary (all turns)")
    print(f"  {'Dimension':<25} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for i, (dim, short) in enumerate(zip(DIMS, DIM_SHORT)):
        col = scores_all[:, i]
        print(f"  {dim:<25} {np.mean(col):6.3f} {np.std(col):6.3f} {np.min(col):6.3f} {np.max(col):6.3f}")
    print(f"  {'g-PSQ':<25} {np.mean(g_psq_all):6.3f} {np.std(g_psq_all):6.3f} "
          f"{np.min(g_psq_all):6.3f} {np.max(g_psq_all):6.3f}")

    # ─── 4. Group comparison: Deal vs No-Deal ───
    print(f"\n### Group Comparison: Deal vs No-Deal (Mann-Whitney U)")
    print(f"\n  {'Dimension':<25} {'Deal Mean':>10} {'NoDeal Mean':>12} {'Cohen d':>9} {'U-stat':>12} {'p-value':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*9} {'-'*12} {'-'*10}")

    deal_mask = np.array([r['is_deal'] for r in records])
    results_group = []

    for i, (dim, short) in enumerate(zip(DIMS, DIM_SHORT)):
        deal_scores = scores_all[deal_mask, i]
        nodeal_scores = scores_all[~deal_mask, i]
        d = cohens_d(deal_scores, nodeal_scores)
        u_stat, p_val = stats.mannwhitneyu(deal_scores, nodeal_scores, alternative='two-sided')
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        results_group.append((dim, short, np.mean(deal_scores), np.mean(nodeal_scores), d, u_stat, p_val, sig))
        print(f"  {dim:<25} {np.mean(deal_scores):10.3f} {np.mean(nodeal_scores):12.3f} "
              f"{d:+9.3f} {u_stat:12.0f} {p_val:10.4f}{sig}")

    # g-PSQ
    deal_g = g_psq_all[deal_mask]
    nodeal_g = g_psq_all[~deal_mask]
    d = cohens_d(deal_g, nodeal_g)
    u_stat, p_val = stats.mannwhitneyu(deal_g, nodeal_g, alternative='two-sided')
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f"  {'g-PSQ':<25} {np.mean(deal_g):10.3f} {np.mean(nodeal_g):12.3f} "
          f"{d:+9.3f} {u_stat:12.0f} {p_val:10.4f}{sig}")

    # Sort by absolute Cohen's d
    results_group.sort(key=lambda x: abs(x[4]), reverse=True)
    print(f"\n  Ranked by |Cohen's d|:")
    for dim, short, dm, nm, d, u, p, sig in results_group:
        print(f"    {short:>3} ({dim:<25}): d={d:+.3f}, p={p:.4f}{sig}")

    # ─── 5. Point-Biserial Correlations (deal vs no-deal) ───
    print(f"\n### Point-Biserial Correlations: PSQ scores vs Deal (1) / No-Deal (0)")
    print(f"\n  {'Dimension':<25} {'r_pb':>8} {'p-value':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*10}")

    deal_binary = deal_mask.astype(float)
    pb_results = []
    for i, (dim, short) in enumerate(zip(DIMS, DIM_SHORT)):
        r_pb, p_val = stats.pointbiserialr(deal_binary, scores_all[:, i])
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        pb_results.append((dim, short, r_pb, p_val, sig))
        print(f"  {dim:<25} {r_pb:+8.3f} {p_val:10.4f}{sig}")

    r_pb, p_val = stats.pointbiserialr(deal_binary, g_psq_all)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f"  {'g-PSQ':<25} {r_pb:+8.3f} {p_val:10.4f}{sig}")

    pb_results.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\n  Ranked by |r_pb|:")
    for dim, short, r, p, sig in pb_results:
        print(f"    {short:>3} ({dim:<25}): r_pb={r:+.3f}, p={p:.4f}{sig}")

    # ─── 6. Pearson Correlations with Points (deals only) ───
    print(f"\n### Pearson Correlations: PSQ scores vs Points Scored (deals only, n={len(deal_records)})")
    print(f"\n  {'Dimension':<25} {'r (you_pts)':>12} {'p':>8} {'r (joint)':>12} {'p':>8} {'r (effic)':>12} {'p':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*8}")

    deal_idx = np.where(deal_mask)[0]
    deal_scores = scores_all[deal_idx]
    you_pts_arr = np.array([records[i]['you_points'] for i in deal_idx])
    joint_pts_arr = np.array([records[i]['joint_points'] for i in deal_idx])
    effic_arr = np.array([records[i]['you_efficiency'] for i in deal_idx])

    for i, (dim, short) in enumerate(zip(DIMS, DIM_SHORT)):
        r1, p1 = stats.pearsonr(deal_scores[:, i], you_pts_arr)
        r2, p2 = stats.pearsonr(deal_scores[:, i], joint_pts_arr)
        r3, p3 = stats.pearsonr(deal_scores[:, i], effic_arr)
        sig1 = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else ''
        sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else ''
        sig3 = '***' if p3 < 0.001 else '**' if p3 < 0.01 else '*' if p3 < 0.05 else ''
        print(f"  {dim:<25} {r1:+12.3f}{sig1:>4} {p1:8.4f} {r2:+12.3f}{sig2:>4} {p2:8.4f} {r3:+12.3f}{sig3:>4} {p3:8.4f}")

    deal_g_pts = g_psq_all[deal_idx]
    r1, p1 = stats.pearsonr(deal_g_pts, you_pts_arr)
    r2, p2 = stats.pearsonr(deal_g_pts, joint_pts_arr)
    r3, p3 = stats.pearsonr(deal_g_pts, effic_arr)
    sig1 = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else ''
    sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else ''
    sig3 = '***' if p3 < 0.001 else '**' if p3 < 0.01 else '*' if p3 < 0.05 else ''
    print(f"  {'g-PSQ':<25} {r1:+12.3f}{sig1:>4} {p1:8.4f} {r2:+12.3f}{sig2:>4} {p2:8.4f} {r3:+12.3f}{sig3:>4} {p3:8.4f}")

    # ─── 7. Text Length Baseline ───
    print(f"\n### Text Length Analysis")
    text_lengths = np.array([r['text_length'] for r in records])

    # Text length vs deal
    r_len_deal, p_len_deal = stats.pointbiserialr(deal_binary, text_lengths)
    print(f"  Text length vs deal (r_pb): {r_len_deal:+.3f}, p={p_len_deal:.4f}")

    # Text length vs points (deals only)
    deal_lengths = text_lengths[deal_idx]
    r_len_pts, p_len_pts = stats.pearsonr(deal_lengths, you_pts_arr)
    print(f"  Text length vs YOU points (r):  {r_len_pts:+.3f}, p={p_len_pts:.4f}")
    r_len_joint, p_len_joint = stats.pearsonr(deal_lengths, joint_pts_arr)
    print(f"  Text length vs joint points (r): {r_len_joint:+.3f}, p={p_len_joint:.4f}")

    # Text length correlations with PSQ
    print(f"\n  Text length vs PSQ dimensions:")
    for i, (dim, short) in enumerate(zip(DIMS, DIM_SHORT)):
        r, p = stats.pearsonr(text_lengths, scores_all[:, i])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"    {short:>3}: r={r:+.3f}{sig}")
    r, p = stats.pearsonr(text_lengths, g_psq_all)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"    g-PSQ: r={r:+.3f}{sig}")

    # ─── 8. Partial Correlations (controlling text length) ───
    print(f"\n### Partial Correlations: PSQ vs Deal, controlling text length")
    print(f"\n  {'Dimension':<25} {'Raw r_pb':>10} {'Partial r':>10} {'p':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for i, (dim, short) in enumerate(zip(DIMS, DIM_SHORT)):
        raw_r, _ = stats.pointbiserialr(deal_binary, scores_all[:, i])
        pr, pp = partial_corr(deal_binary, scores_all[:, i], text_lengths)
        sig = '***' if pp < 0.001 else '**' if pp < 0.01 else '*' if pp < 0.05 else ''
        print(f"  {dim:<25} {raw_r:+10.3f} {pr:+10.3f} {pp:8.4f}{sig}")

    raw_r, _ = stats.pointbiserialr(deal_binary, g_psq_all)
    pr, pp = partial_corr(deal_binary, g_psq_all, text_lengths)
    sig = '***' if pp < 0.001 else '**' if pp < 0.01 else '*' if pp < 0.05 else ''
    print(f"  {'g-PSQ':<25} {raw_r:+10.3f} {pr:+10.3f} {pp:8.4f}{sig}")

    # ─── 9. Logistic Regression: AUC for Deal Prediction ───
    print(f"\n### Logistic Regression: Deal Prediction (AUC)")

    # Use train split for training, test+val for evaluation (replicating CGA-Wiki approach)
    train_mask = np.array([r['split'] == 'train' for r in records])
    test_mask = ~train_mask  # test + validation

    y_train = deal_binary[train_mask]
    y_test = deal_binary[test_mask]

    scaler = StandardScaler()

    # 10-dim PSQ
    X_train_10 = scaler.fit_transform(scores_all[train_mask])
    X_test_10 = scaler.transform(scores_all[test_mask])
    lr10 = LogisticRegression(max_iter=1000, random_state=42)
    lr10.fit(X_train_10, y_train)
    auc_10 = roc_auc_score(y_test, lr10.predict_proba(X_test_10)[:, 1])
    acc_10 = lr10.score(X_test_10, y_test)

    # g-PSQ only
    scaler_g = StandardScaler()
    X_train_g = scaler_g.fit_transform(g_psq_all[train_mask].reshape(-1, 1))
    X_test_g = scaler_g.transform(g_psq_all[test_mask].reshape(-1, 1))
    lr_g = LogisticRegression(max_iter=1000, random_state=42)
    lr_g.fit(X_train_g, y_train)
    auc_g = roc_auc_score(y_test, lr_g.predict_proba(X_test_g)[:, 1])

    # Text length only
    scaler_len = StandardScaler()
    X_train_len = scaler_len.fit_transform(text_lengths[train_mask].reshape(-1, 1))
    X_test_len = scaler_len.transform(text_lengths[test_mask].reshape(-1, 1))
    lr_len = LogisticRegression(max_iter=1000, random_state=42)
    lr_len.fit(X_train_len, y_train)
    auc_len = roc_auc_score(y_test, lr_len.predict_proba(X_test_len)[:, 1])

    # PSQ + text length
    X_train_combo = np.column_stack([X_train_10, X_train_len])
    X_test_combo = np.column_stack([X_test_10, X_test_len])
    lr_combo = LogisticRegression(max_iter=1000, random_state=42)
    lr_combo.fit(X_train_combo, y_train)
    auc_combo = roc_auc_score(y_test, lr_combo.predict_proba(X_test_combo)[:, 1])

    # n_turns only
    n_turns_arr = np.array([r['n_turns'] for r in records])
    scaler_turns = StandardScaler()
    X_train_turns = scaler_turns.fit_transform(n_turns_arr[train_mask].reshape(-1, 1))
    X_test_turns = scaler_turns.transform(n_turns_arr[test_mask].reshape(-1, 1))
    lr_turns = LogisticRegression(max_iter=1000, random_state=42)
    lr_turns.fit(X_train_turns, y_train)
    auc_turns = roc_auc_score(y_test, lr_turns.predict_proba(X_test_turns)[:, 1])

    print(f"\n  {'Model':<30} {'AUC':>6} {'Accuracy':>10}")
    print(f"  {'-'*30} {'-'*6} {'-'*10}")
    print(f"  {'10-dim PSQ':<30} {auc_10:6.3f} {acc_10*100:9.1f}%")
    print(f"  {'PSQ + text length':<30} {auc_combo:6.3f}")
    print(f"  {'Text length only':<30} {auc_len:6.3f}")
    print(f"  {'Turn count only':<30} {auc_turns:6.3f}")
    print(f"  {'g-PSQ only':<30} {auc_g:6.3f}")

    # 5-fold CV on full data
    scaler_cv = StandardScaler()
    X_all_10 = scaler_cv.fit_transform(scores_all)
    cv_scores = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                                X_all_10, deal_binary, cv=5, scoring='roc_auc')
    print(f"\n  5-fold CV (10-dim PSQ): AUC = {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Feature weights
    print(f"\n  Logistic Regression Feature Weights (10-dim model):")
    print(f"  {'Rank':>4} {'Dimension':<25} {'Coefficient':>12}")
    print(f"  {'-'*4} {'-'*25} {'-'*12}")
    coefs = lr10.coef_[0]
    ranked = sorted(zip(DIMS, DIM_SHORT, coefs), key=lambda x: abs(x[2]), reverse=True)
    for rank, (dim, short, coef) in enumerate(ranked, 1):
        print(f"  {rank:4d} {dim:<25} {coef:+12.3f}")

    # ─── 10. Temporal Analysis ───
    print(f"\n### Temporal Signal Analysis")

    # Score early turns and first turn
    # (Already scored above)

    # AUC for each temporal condition
    for label, s, g in [("All turns", scores_all, g_psq_all),
                         ("Early turns", scores_early, g_psq_early),
                         ("First turn", scores_first, g_psq_first)]:

        scaler_t = StandardScaler()
        X_train_t = scaler_t.fit_transform(s[train_mask])
        X_test_t = scaler_t.transform(s[test_mask])
        lr_t = LogisticRegression(max_iter=1000, random_state=42)
        lr_t.fit(X_train_t, y_train)
        auc_t = roc_auc_score(y_test, lr_t.predict_proba(X_test_t)[:, 1])

        d_g = cohens_d(g[deal_mask], g[~deal_mask])

        # Count significant dims
        n_sig = 0
        for i in range(10):
            _, p = stats.mannwhitneyu(s[deal_mask, i], s[~deal_mask, i], alternative='two-sided')
            if p < 0.05:
                n_sig += 1

        print(f"  {label:<15} AUC={auc_t:.3f}  d(g-PSQ)={d_g:+.3f}  sig dims={n_sig}/10")

    # Per-dimension temporal correlations
    print(f"\n  Per-dimension point-biserial r by temporal condition:")
    print(f"  {'Dimension':<25} {'All turns':>10} {'Early':>10} {'First':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for i, (dim, short) in enumerate(zip(DIMS, DIM_SHORT)):
        r_all, _ = stats.pointbiserialr(deal_binary, scores_all[:, i])
        r_early, _ = stats.pointbiserialr(deal_binary, scores_early[:, i])
        r_first, _ = stats.pointbiserialr(deal_binary, scores_first[:, i])
        print(f"  {dim:<25} {r_all:+10.3f} {r_early:+10.3f} {r_first:+10.3f}")
    r_all, _ = stats.pointbiserialr(deal_binary, g_psq_all)
    r_early, _ = stats.pointbiserialr(deal_binary, g_psq_early)
    r_first, _ = stats.pointbiserialr(deal_binary, g_psq_first)
    print(f"  {'g-PSQ':<25} {r_all:+10.3f} {r_early:+10.3f} {r_first:+10.3f}")

    # ─── 11. Extreme Group Comparison ───
    print(f"\n### Extreme Group Comparison (Q1 vs Q4 g-PSQ)")
    q1 = np.percentile(g_psq_all, 25)
    q4 = np.percentile(g_psq_all, 75)
    low_mask = g_psq_all <= q1
    high_mask = g_psq_all >= q4

    deal_rate_low = deal_binary[low_mask].mean()
    deal_rate_high = deal_binary[high_mask].mean()
    print(f"  Low PSQ (Q1, n={low_mask.sum()}):  deal rate = {deal_rate_low:.3f} ({deal_rate_low*100:.1f}%)")
    print(f"  High PSQ (Q4, n={high_mask.sum()}): deal rate = {deal_rate_high:.3f} ({deal_rate_high*100:.1f}%)")
    print(f"  Difference: {deal_rate_high - deal_rate_low:+.3f} ({(deal_rate_high-deal_rate_low)*100:+.1f}pp)")

    # Points comparison (deals only)
    low_deal_mask = low_mask & deal_mask
    high_deal_mask = high_mask & deal_mask
    if low_deal_mask.sum() > 0 and high_deal_mask.sum() > 0:
        low_pts = np.array([records[i]['you_points'] for i in np.where(low_deal_mask)[0]])
        high_pts = np.array([records[i]['you_points'] for i in np.where(high_deal_mask)[0]])
        low_joint = np.array([records[i]['joint_points'] for i in np.where(low_deal_mask)[0]])
        high_joint = np.array([records[i]['joint_points'] for i in np.where(high_deal_mask)[0]])
        print(f"\n  Points scored (deals only):")
        print(f"    Low PSQ:  YOU pts mean={np.mean(low_pts):.2f}, joint={np.mean(low_joint):.2f} (n={len(low_pts)})")
        print(f"    High PSQ: YOU pts mean={np.mean(high_pts):.2f}, joint={np.mean(high_joint):.2f} (n={len(high_pts)})")
        d_you = cohens_d(high_pts, low_pts)
        d_joint = cohens_d(high_joint, low_joint)
        print(f"    Cohen's d (YOU pts): {d_you:+.3f}")
        print(f"    Cohen's d (joint):   {d_joint:+.3f}")

    # ─── 12. Incremental Validity (PSQ beyond text length + turn count) ───
    print(f"\n### Incremental Validity: PSQ beyond Text Length + Turn Count")
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # For deal prediction (logistic)
    # Base: text length + turn count
    base_features = np.column_stack([text_lengths, n_turns_arr])
    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(base_features[train_mask])
    X_test_base = scaler_base.transform(base_features[test_mask])
    lr_base = LogisticRegression(max_iter=1000, random_state=42)
    lr_base.fit(X_train_base, y_train)
    auc_base = roc_auc_score(y_test, lr_base.predict_proba(X_test_base)[:, 1])

    # Base + PSQ
    X_train_full = np.column_stack([X_train_base, X_train_10])
    X_test_full = np.column_stack([X_test_base, X_test_10])
    lr_full = LogisticRegression(max_iter=1000, random_state=42)
    lr_full.fit(X_train_full, y_train)
    auc_full = roc_auc_score(y_test, lr_full.predict_proba(X_test_full)[:, 1])

    print(f"  Deal prediction (AUC):")
    print(f"    Text length + turns:           {auc_base:.3f}")
    print(f"    Text length + turns + PSQ-10:   {auc_full:.3f}")
    print(f"    Incremental AUC:               {auc_full - auc_base:+.3f}")

    # For points prediction (linear, deals only)
    deal_text_len = text_lengths[deal_idx]
    deal_turns = n_turns_arr[deal_idx]
    base_deal = np.column_stack([deal_text_len, deal_turns])

    from sklearn.model_selection import cross_val_score as cvs
    lr_base_pts = LinearRegression()
    r2_base = cvs(lr_base_pts, base_deal, you_pts_arr, cv=5, scoring='r2').mean()

    full_deal = np.column_stack([base_deal, deal_scores])
    lr_full_pts = LinearRegression()
    r2_full = cvs(lr_full_pts, full_deal, you_pts_arr, cv=5, scoring='r2').mean()

    print(f"\n  Points prediction (R², 5-fold CV, deals only):")
    print(f"    Text length + turns:           {r2_base:.4f}")
    print(f"    Text length + turns + PSQ-10:   {r2_full:.4f}")
    print(f"    Incremental R²:                {r2_full - r2_base:+.4f}")

    # Joint points
    r2_base_j = cvs(LinearRegression(), base_deal, joint_pts_arr, cv=5, scoring='r2').mean()
    r2_full_j = cvs(LinearRegression(), full_deal, joint_pts_arr, cv=5, scoring='r2').mean()
    print(f"\n  Joint points prediction (R², 5-fold CV, deals only):")
    print(f"    Text length + turns:           {r2_base_j:.4f}")
    print(f"    Text length + turns + PSQ-10:   {r2_full_j:.4f}")
    print(f"    Incremental R²:                {r2_full_j - r2_base_j:+.4f}")

    # ─── 13. Leave-One-Dimension-Out Analysis ───
    print(f"\n### Leave-One-Dimension-Out: AUC for Deal Prediction")
    print(f"  {'Dropped dim':<25} {'AUC (9-dim)':>12} {'Delta AUC':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*10}")
    for drop_i, (dim, short) in enumerate(zip(DIMS, DIM_SHORT)):
        keep = [j for j in range(10) if j != drop_i]
        scaler_d = StandardScaler()
        X_train_d = scaler_d.fit_transform(scores_all[train_mask][:, keep])
        X_test_d = scaler_d.transform(scores_all[test_mask][:, keep])
        lr_d = LogisticRegression(max_iter=1000, random_state=42)
        lr_d.fit(X_train_d, y_train)
        auc_d = roc_auc_score(y_test, lr_d.predict_proba(X_test_d)[:, 1])
        print(f"  {dim:<25} {auc_d:12.3f} {auc_d - auc_10:+10.3f}")

    print(f"\n  Full 10-dim AUC: {auc_10:.3f}")

    # ─── 14. Cross-study comparison table ───
    print(f"\n### Cross-Study Comparison (with CaSiNo and CGA-Wiki)")
    print(f"\n  {'Finding':<40} {'CaSiNo':>15} {'CGA-Wiki':>15} {'DealOrNoDeal':>15}")
    print(f"  {'-'*40} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'Domain':<40} {'Campsite negot.':>15} {'Wiki disputes':>15} {'Item negot.':>15}")
    print(f"  {'N':<40} {'1,030':>15} {'4,188':>15} {f'{len(records):,}':>15}")
    print(f"  {'Outcome type':<40} {'Subjective':>15} {'Behavioral':>15} {'Deal + points':>15}")
    print(f"  {'AUC (10-dim, if binary)':<40} {'—':>15} {'0.599':>15} {f'{auc_10:.3f}':>15}")
    print(f"  {'AUC (g-PSQ)':<40} {'—':>15} {'0.515':>15} {f'{auc_g:.3f}':>15}")
    print(f"  {'AUC (text length)':<40} {'—':>15} {'0.542':>15} {f'{auc_len:.3f}':>15}")

    # Find top predictor by |r_pb|
    top_pb = max(pb_results, key=lambda x: abs(x[2]))
    print(f"  {'Top individual predictor':<40} {'DA':>15} {'AD':>15} {f'{top_pb[1]}':>15}")

    n_sig_dims = sum(1 for _, _, _, p, _ in pb_results if p < 0.05)
    print(f"  {'Sig dims (p<0.05)':<40} {'9/10':>15} {'8/10':>15} {f'{n_sig_dims}/10':>15}")

    print(f"\n" + "="*80)
    print(f"Analysis complete.")
    print(f"="*80)


if __name__ == "__main__":
    main()
