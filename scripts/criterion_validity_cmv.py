#!/usr/bin/env python3
"""
Criterion validity study: PSQ student model (psq-student/best.pt) vs Change My View (CMV) dataset.

Dataset: Tan et al. (2016) "Winning Arguments" — Reddit ChangeMyView
Outcome: Delta awarded (persuasion success) vs no delta
Design: 4,263 matched pairs (same OP, one delta reply vs one non-delta reply)

Extends criterion validity battery beyond CaSiNo and CGA-Wiki.
"""

import json
import warnings
import sys
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT = Path("/home/kashif/projects/psychology/safety-quotient")
CORPUS = Path("/home/kashif/.convokit/saved-corpora/winning-args-corpus")

DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]
DIM_ABBREV = ["TE", "HI", "AD", "ED", "RC", "RB", "TC", "CC", "DA", "CO"]


# ── Model ────────────────────────────────────────────────────────────────────
class PSQStudent(nn.Module):
    """Matches the architecture in scripts/distill.py exactly."""

    def __init__(self, model_name="distilbert-base-uncased", n_dims=10):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
        hidden = self.encoder.config.hidden_size

        # Shared projection (matches distill.py)
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

        scores = torch.stack(scores, dim=1)
        confs = torch.stack(confs, dim=1)
        return scores, confs


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        str(PROJECT / "models/psq-student/tokenizer")
    )
    model = PSQStudent()
    state_dict = torch.load(
        str(PROJECT / "models/psq-student/best.pt"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer


# ── Data loading ─────────────────────────────────────────────────────────────
def load_cmv_pairs():
    """Load matched pairs of delta vs non-delta top-level replies."""
    pair_map = defaultdict(lambda: {"success": [], "failure": []})
    utterance_text = {}

    with open(CORPUS / "utterances.jsonl") as f:
        for line in f:
            u = json.loads(line)
            meta = u.get("meta", {})
            pair_ids = meta.get("pair_ids", [])
            success = meta.get("success")
            reply_to = u.get("reply-to")
            root = u.get("root")

            utterance_text[u["id"]] = u.get("text", "")

            # Top-level replies only (direct reply to OP)
            if reply_to == root and pair_ids:
                for pid in pair_ids:
                    if success == 1:
                        pair_map[pid]["success"].append(u["id"])
                    else:
                        pair_map[pid]["failure"].append(u["id"])

    # Build paired dataset: one delta text and one non-delta text per pair
    pairs = []
    for pid in sorted(pair_map.keys(), key=lambda x: int(x.split("_")[1])):
        pdata = pair_map[pid]
        if pdata["success"] and pdata["failure"]:
            # Take first of each (standard in Tan et al.)
            s_id = pdata["success"][0]
            f_id = pdata["failure"][0]
            s_text = utterance_text[s_id]
            f_text = utterance_text[f_id]
            if s_text.strip() and f_text.strip():
                pairs.append({
                    "pair_id": pid,
                    "delta_text": s_text,
                    "delta_id": s_id,
                    "no_delta_text": f_text,
                    "no_delta_id": f_id,
                })

    return pairs


# ── Scoring ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_texts(texts, model, tokenizer, batch_size=64, max_len=512):
    """Score a list of texts, return (n_texts, 10) array of dimension scores."""
    import time
    device = next(model.parameters()).device
    all_scores = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}
        scores, confs = model(**enc)
        all_scores.append(scores.cpu().numpy())
        done = min(i + batch_size, len(texts))
        if done % 320 == 0 or done == len(texts):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(texts) - done) / rate if rate > 0 else 0
            print(f"  Scored {done}/{len(texts)} ({rate:.0f} texts/s, ETA {eta:.0f}s)",
                  flush=True)
    return np.concatenate(all_scores, axis=0)


# ── Analysis ─────────────────────────────────────────────────────────────────
def cohens_d(x, y):
    """Cohen's d for independent samples."""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    return (mx - my) / pooled if pooled > 0 else 0.0


def paired_cohens_d(x, y):
    """Cohen's d for paired samples (d_z)."""
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0


def run_analysis(pairs, model, tokenizer):
    print(f"\n{'='*80}")
    print("CRITERION VALIDITY STUDY: PSQ student × Change My View (CMV)")
    print(f"{'='*80}")

    # ── Dataset description ──────────────────────────────────────────────
    print(f"\n## Dataset Description")
    print(f"  Source: Tan et al. (2016) 'Winning Arguments' via ConvoKit")
    print(f"  Subreddit: r/ChangeMyView")
    print(f"  Design: Matched pairs — same OP, one delta-awarded reply vs one non-delta")
    print(f"  N pairs: {len(pairs)}")
    print(f"  Total texts scored: {len(pairs) * 2}")

    delta_texts = [p["delta_text"] for p in pairs]
    no_delta_texts = [p["no_delta_text"] for p in pairs]

    delta_lens = np.array([len(t) for t in delta_texts])
    no_delta_lens = np.array([len(t) for t in no_delta_texts])

    print(f"\n  Text lengths (chars):")
    print(f"    Delta:    mean={np.mean(delta_lens):.0f}, median={np.median(delta_lens):.0f}, "
          f"sd={np.std(delta_lens):.0f}")
    print(f"    No-delta: mean={np.mean(no_delta_lens):.0f}, median={np.median(no_delta_lens):.0f}, "
          f"sd={np.std(no_delta_lens):.0f}")
    t_len, p_len = stats.ttest_rel(delta_lens, no_delta_lens)
    d_len = paired_cohens_d(delta_lens, no_delta_lens)
    print(f"    Paired t-test: t={t_len:.3f}, p={p_len:.2e}, d={d_len:.3f}")

    # ── Score texts ──────────────────────────────────────────────────────
    print(f"\n## Scoring with PSQ student model (psq-student/best.pt)...")
    print(f"  Scoring delta texts ({len(delta_texts)})...")
    delta_scores = score_texts(delta_texts, model, tokenizer)
    print(f"  Scoring no-delta texts ({len(no_delta_texts)})...")
    no_delta_scores = score_texts(no_delta_texts, model, tokenizer)

    # g-PSQ (mean of 10 dims)
    delta_gpsq = delta_scores.mean(axis=1)
    no_delta_gpsq = no_delta_scores.mean(axis=1)

    # ── Group comparison (paired) ────────────────────────────────────────
    print(f"\n{'='*80}")
    print("## Group Comparison: Delta-Awarded vs Non-Delta Replies (Paired)")
    print(f"{'='*80}")
    print(f"\n{'Dimension':<26} {'Delta':>8} {'No-Δ':>8} {'Diff':>8} "
          f"{'d_z':>7} {'t':>8} {'p':>12} {'Sig':>5}")
    print("-" * 95)

    sig_dims = []
    for i, (dim, abbr) in enumerate(zip(DIMS, DIM_ABBREV)):
        d_vals = delta_scores[:, i]
        nd_vals = no_delta_scores[:, i]
        t_stat, p_val = stats.ttest_rel(d_vals, nd_vals)
        d_z = paired_cohens_d(d_vals, nd_vals)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {abbr} ({dim[:20]:<20}) {np.mean(d_vals):8.3f} {np.mean(nd_vals):8.3f} "
              f"{np.mean(d_vals)-np.mean(nd_vals):+8.4f} {d_z:+7.4f} "
              f"{t_stat:8.3f} {p_val:12.2e} {sig:>5}")
        if p_val < 0.05:
            sig_dims.append(abbr)

    # g-PSQ row
    t_g, p_g = stats.ttest_rel(delta_gpsq, no_delta_gpsq)
    d_g = paired_cohens_d(delta_gpsq, no_delta_gpsq)
    sig_g = "***" if p_g < 0.001 else "**" if p_g < 0.01 else "*" if p_g < 0.05 else ""
    print("-" * 95)
    print(f"  {'g-PSQ':<28} {np.mean(delta_gpsq):8.3f} {np.mean(no_delta_gpsq):8.3f} "
          f"{np.mean(delta_gpsq)-np.mean(no_delta_gpsq):+8.4f} {d_g:+7.4f} "
          f"{t_g:8.3f} {p_g:12.2e} {sig_g:>5}")

    print(f"\n  Significant dims (p<.05): {len(sig_dims)}/10 — {', '.join(sig_dims)}")
    print(f"  Note: d_z = paired Cohen's d (within-pair effect size)")

    # ── Bonferroni correction ────────────────────────────────────────────
    print(f"\n## Bonferroni-Corrected Results (α = 0.005 for 10 tests)")
    bonf_sig = []
    for i, (dim, abbr) in enumerate(zip(DIMS, DIM_ABBREV)):
        d_vals = delta_scores[:, i]
        nd_vals = no_delta_scores[:, i]
        t_stat, p_val = stats.ttest_rel(d_vals, nd_vals)
        if p_val < 0.005:
            bonf_sig.append(abbr)
    print(f"  Survive Bonferroni: {len(bonf_sig)}/10 — {', '.join(bonf_sig) if bonf_sig else 'none'}")

    # ── Point-biserial correlations ──────────────────────────────────────
    print(f"\n{'='*80}")
    print("## Point-Biserial Correlations (delta=1 vs no-delta=0)")
    print(f"{'='*80}")

    # Unpaired view: all texts with label
    all_scores_combined = np.vstack([delta_scores, no_delta_scores])
    all_labels = np.array([1] * len(delta_scores) + [0] * len(no_delta_scores))
    all_gpsq = np.concatenate([delta_gpsq, no_delta_gpsq])
    all_lens = np.concatenate([delta_lens, no_delta_lens])

    print(f"\n{'Dimension':<26} {'r_pb':>8} {'p':>12} {'Sig':>5}")
    print("-" * 55)
    for i, (dim, abbr) in enumerate(zip(DIMS, DIM_ABBREV)):
        r, p = stats.pointbiserialr(all_labels, all_scores_combined[:, i])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {abbr} ({dim[:20]:<20}) {r:+8.4f} {p:12.2e} {sig:>5}")

    r_g, p_g = stats.pointbiserialr(all_labels, all_gpsq)
    sig_g = "***" if p_g < 0.001 else "**" if p_g < 0.01 else "*" if p_g < 0.05 else ""
    print("-" * 55)
    print(f"  {'g-PSQ':<26} {r_g:+8.4f} {p_g:12.2e} {sig_g:>5}")

    r_len, p_len = stats.pointbiserialr(all_labels, all_lens)
    sig_len = "***" if p_len < 0.001 else "**" if p_len < 0.01 else "*" if p_len < 0.05 else ""
    print(f"  {'text_length':<26} {r_len:+8.4f} {p_len:12.2e} {sig_len:>5}")

    # ── Logistic regression AUC ──────────────────────────────────────────
    print(f"\n{'='*80}")
    print("## Logistic Regression: Predicting Delta Award (AUC)")
    print(f"{'='*80}")

    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    # Use 5-fold cross-validation for all AUC estimates (avoid train=test bias)
    def cv_auc(X, y):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
        return scores.mean(), scores.std()

    # 1. Text length only (baseline)
    auc_len, auc_len_sd = cv_auc(all_lens.reshape(-1, 1), all_labels)

    # 2. g-PSQ only
    auc_gpsq, auc_gpsq_sd = cv_auc(all_gpsq.reshape(-1, 1), all_labels)

    # 3. 10-dim PSQ
    auc_10, auc_10_sd = cv_auc(all_scores_combined, all_labels)

    # 4. 10-dim + text length
    auc_10_len, auc_10_len_sd = cv_auc(
        np.hstack([all_scores_combined, all_lens.reshape(-1, 1)]), all_labels)

    # 5. Text length + g-PSQ
    auc_len_gpsq, auc_len_gpsq_sd = cv_auc(
        np.hstack([all_lens.reshape(-1, 1), all_gpsq.reshape(-1, 1)]), all_labels)

    # Fit full model for coefficients (on all data, for interpretation only)
    scaler = StandardScaler()
    X_10 = scaler.fit_transform(all_scores_combined)
    lr_10 = LogisticRegression(max_iter=1000, random_state=42)
    lr_10.fit(X_10, all_labels)

    print(f"\n  {'Model':<35} {'AUC (5-CV)':>10} {'±SD':>8} {'ΔAUCvLen':>10}")
    print("  " + "-" * 65)
    print(f"  {'Text length only (baseline)':<35} {auc_len:10.4f} {auc_len_sd:8.4f} {'—':>10}")
    print(f"  {'g-PSQ only':<35} {auc_gpsq:10.4f} {auc_gpsq_sd:8.4f} {auc_gpsq - auc_len:+10.4f}")
    print(f"  {'Text length + g-PSQ':<35} {auc_len_gpsq:10.4f} {auc_len_gpsq_sd:8.4f} {auc_len_gpsq - auc_len:+10.4f}")
    print(f"  {'10-dim PSQ':<35} {auc_10:10.4f} {auc_10_sd:8.4f} {auc_10 - auc_len:+10.4f}")
    print(f"  {'10-dim PSQ + text length':<35} {auc_10_len:10.4f} {auc_10_len_sd:8.4f} {auc_10_len - auc_len:+10.4f}")

    incr_10 = auc_10_len - auc_len
    print(f"\n  Incremental AUC (10-dim beyond length): {incr_10:+.4f}")
    print(f"  g-PSQ vs 10-dim gap: {auc_10 - auc_gpsq:+.4f}")

    # Logistic regression coefficients (10-dim model)
    print(f"\n  Logistic Regression Coefficients (10-dim model, standardized):")
    print(f"  {'Dimension':<26} {'Coef':>8} {'|Coef|':>8}")
    print("  " + "-" * 45)
    coef_order = np.argsort(-np.abs(lr_10.coef_[0]))
    for idx in coef_order:
        print(f"  {DIM_ABBREV[idx]} ({DIMS[idx][:20]:<20}) {lr_10.coef_[0][idx]:+8.4f} "
              f"{abs(lr_10.coef_[0][idx]):8.4f}")

    # ── Paired analysis (McNemar-style / paired accuracy) ────────────────
    print(f"\n{'='*80}")
    print("## Paired Analysis: Within-Pair Prediction Accuracy")
    print(f"{'='*80}")
    print("  For each pair: does the model assign a higher score to the delta reply?")

    # Per-dimension accuracy
    print(f"\n  {'Dimension':<26} {'Accuracy':>10} {'p (binom)':>12} {'Sig':>5}")
    print("  " + "-" * 55)
    for i, (dim, abbr) in enumerate(zip(DIMS, DIM_ABBREV)):
        # For each pair, check if delta reply has higher score
        correct = np.sum(delta_scores[:, i] > no_delta_scores[:, i])
        ties = np.sum(delta_scores[:, i] == no_delta_scores[:, i])
        n_eff = len(pairs) - ties
        acc = correct / len(pairs)
        # Binomial test: is accuracy significantly > 0.5?
        p_binom = stats.binomtest(correct, n=len(pairs), p=0.5, alternative="greater").pvalue
        sig = "***" if p_binom < 0.001 else "**" if p_binom < 0.01 else "*" if p_binom < 0.05 else ""
        print(f"  {abbr} ({dim[:20]:<20}) {acc:10.4f} {p_binom:12.2e} {sig:>5}")

    # g-PSQ accuracy
    correct_g = np.sum(delta_gpsq > no_delta_gpsq)
    acc_g = correct_g / len(pairs)
    p_g_binom = stats.binomtest(correct_g, n=len(pairs), p=0.5, alternative="greater").pvalue
    sig_g = "***" if p_g_binom < 0.001 else "**" if p_g_binom < 0.01 else "*" if p_g_binom < 0.05 else ""
    print("  " + "-" * 55)
    print(f"  {'g-PSQ':<26} {acc_g:10.4f} {p_g_binom:12.2e} {sig_g:>5}")

    # Text length accuracy (baseline)
    correct_len = np.sum(delta_lens > no_delta_lens)
    acc_len = correct_len / len(pairs)
    p_len_binom = stats.binomtest(correct_len, n=len(pairs), p=0.5, alternative="greater").pvalue
    sig_len = "***" if p_len_binom < 0.001 else "**" if p_len_binom < 0.01 else "*" if p_len_binom < 0.05 else ""
    print(f"  {'text_length (baseline)':<26} {acc_len:10.4f} {p_len_binom:12.2e} {sig_len:>5}")

    # ── Per-dimension individual logistic regression AUCs ────────────────
    print(f"\n{'='*80}")
    print("## Individual Dimension AUCs (single-predictor logistic regression)")
    print(f"{'='*80}")
    print(f"\n  {'Dimension':<26} {'AUC':>7} {'ΔAUCvLen':>10}")
    print("  " + "-" * 45)
    dim_aucs = []
    for i, (dim, abbr) in enumerate(zip(DIMS, DIM_ABBREV)):
        X_dim = scaler.fit_transform(all_scores_combined[:, i].reshape(-1, 1))
        lr_dim = LogisticRegression(max_iter=1000, random_state=42)
        lr_dim.fit(X_dim, all_labels)
        auc_dim = roc_auc_score(all_labels, lr_dim.predict_proba(X_dim)[:, 1])
        dim_aucs.append(auc_dim)
        print(f"  {abbr} ({dim[:20]:<20}) {auc_dim:7.4f} {auc_dim - auc_len:+10.4f}")

    best_dim_idx = np.argmax(dim_aucs)
    print(f"\n  Best single dimension: {DIM_ABBREV[best_dim_idx]} (AUC={dim_aucs[best_dim_idx]:.4f})")

    # ── Controlling for text length (partial correlations) ───────────────
    print(f"\n{'='*80}")
    print("## Partial Correlations (controlling for text length)")
    print(f"{'='*80}")
    print(f"\n  {'Dimension':<26} {'r_partial':>10} {'p':>12} {'Sig':>5}")
    print("  " + "-" * 55)

    # Partial correlation: residualize both DV and IV on text length
    from sklearn.linear_model import LinearRegression
    lr_resid = LinearRegression()

    # Residualize label on length
    lr_resid.fit(all_lens.reshape(-1, 1), all_labels)
    label_resid = all_labels - lr_resid.predict(all_lens.reshape(-1, 1))

    for i, (dim, abbr) in enumerate(zip(DIMS, DIM_ABBREV)):
        # Residualize dim score on length
        lr_resid.fit(all_lens.reshape(-1, 1), all_scores_combined[:, i])
        score_resid = all_scores_combined[:, i] - lr_resid.predict(all_lens.reshape(-1, 1))
        r, p = stats.pearsonr(label_resid, score_resid)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {abbr} ({dim[:20]:<20}) {r:+10.4f} {p:12.2e} {sig:>5}")

    # g-PSQ partial
    lr_resid.fit(all_lens.reshape(-1, 1), all_gpsq)
    gpsq_resid = all_gpsq - lr_resid.predict(all_lens.reshape(-1, 1))
    r_gp, p_gp = stats.pearsonr(label_resid, gpsq_resid)
    sig_gp = "***" if p_gp < 0.001 else "**" if p_gp < 0.01 else "*" if p_gp < 0.05 else ""
    print("  " + "-" * 55)
    print(f"  {'g-PSQ':<26} {r_gp:+10.4f} {p_gp:12.2e} {sig_gp:>5}")

    # ── Effect size comparison across validity studies ────────────────────
    print(f"\n{'='*80}")
    print("## Cross-Study Comparison (Criterion Validity Battery)")
    print(f"{'='*80}")
    print(f"\n  {'Study':<25} {'Outcome':<25} {'10-dim AUC':>10} {'g-PSQ AUC':>10} {'Gap':>8}")
    print("  " + "-" * 80)
    print(f"  {'CGA-Wiki':<25} {'Derailment':<25} {'0.599':>10} {'0.515':>10} {'0.084':>8}")
    print(f"  {'CMV (this study)':<25} {'Delta awarded':<25} {auc_10:10.4f} {auc_gpsq:10.4f} "
          f"{auc_10 - auc_gpsq:8.4f}")

    # ── Score distribution summary ───────────────────────────────────────
    print(f"\n{'='*80}")
    print("## Score Distribution Summary")
    print(f"{'='*80}")
    print(f"\n  {'Dimension':<26} {'Overall Mean':>12} {'Overall SD':>10} "
          f"{'Delta Mean':>11} {'NoΔ Mean':>10}")
    print("  " + "-" * 72)
    for i, (dim, abbr) in enumerate(zip(DIMS, DIM_ABBREV)):
        overall = all_scores_combined[:, i]
        print(f"  {abbr} ({dim[:20]:<20}) {np.mean(overall):12.3f} {np.std(overall):10.3f} "
              f"{np.mean(delta_scores[:, i]):11.3f} {np.mean(no_delta_scores[:, i]):10.3f}")
    print("  " + "-" * 72)
    print(f"  {'g-PSQ':<26} {np.mean(all_gpsq):12.3f} {np.std(all_gpsq):10.3f} "
          f"{np.mean(delta_gpsq):11.3f} {np.mean(no_delta_gpsq):10.3f}")

    # ── Direction of effects interpretation ──────────────────────────────
    print(f"\n{'='*80}")
    print("## Direction of Effects: Delta > No-Delta means...")
    print(f"{'='*80}")
    for i, (dim, abbr) in enumerate(zip(DIMS, DIM_ABBREV)):
        diff = np.mean(delta_scores[:, i]) - np.mean(no_delta_scores[:, i])
        direction = "HIGHER" if diff > 0 else "LOWER" if diff < 0 else "EQUAL"
        t_stat, p_val = stats.ttest_rel(delta_scores[:, i], no_delta_scores[:, i])
        sig = " ***" if p_val < 0.001 else " **" if p_val < 0.01 else " *" if p_val < 0.05 else ""
        print(f"  {abbr}: Delta replies score {direction} ({diff:+.4f}){sig}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    print("Loading CMV pairs...")
    pairs = load_cmv_pairs()
    print(f"  Loaded {len(pairs)} matched pairs")

    print("Loading PSQ student model (psq-student/best.pt)...")
    model, tokenizer = load_model()
    print("  Model loaded successfully")

    run_analysis(pairs, model, tokenizer)
