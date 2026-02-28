#!/usr/bin/env python3
"""Scoring experiment analysis — compare control vs treatment scoring conditions.

Usage:
  # Test-retest (Phase 0): compare retest against gold labels
  python scripts/scoring_experiment_analysis.py \
    --control data/held-out-test.jsonl --treatment /tmp/retest_scores.jsonl \
    --mode retest --n 20

  # A/B experiment (Phases 1-2): compare control vs treatment on same texts
  python scripts/scoring_experiment_analysis.py \
    --control /tmp/exp1_control.jsonl --treatment /tmp/exp1_treatment.jsonl \
    --mode ab

  # Cross-scale (Phase 3): compare two different scales (uses rank-based metrics only)
  python scripts/scoring_experiment_analysis.py \
    --control /tmp/exp3_0to10.jsonl --treatment /tmp/exp3_1to7.jsonl \
    --mode crossscale

Input format: JSONL with {"text": "...", "dimensions": {"dim_name": {"score": N, "confidence": N}}}
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]

DIM_SHORT = {
    "threat_exposure": "TE", "hostility_index": "HI",
    "authority_dynamics": "AD", "energy_dissipation": "ED",
    "regulatory_capacity": "RC", "resilience_baseline": "RB",
    "trust_conditions": "TC", "cooling_capacity": "CC",
    "defensive_architecture": "DA", "contractual_clarity": "CO",
}


def load_scores(path, n=None):
    """Load JSONL, return (texts, score_matrix) where matrix is N×10."""
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    if n:
        recs = recs[:n]

    texts = [r.get("text", "")[:80] for r in recs]
    matrix = np.full((len(recs), len(DIMS)), np.nan)
    for i, r in enumerate(recs):
        dims = r.get("dimensions", r.get("scores", {}))
        for j, d in enumerate(DIMS):
            if d in dims:
                val = dims[d]
                if isinstance(val, dict):
                    matrix[i, j] = val["score"]
                else:
                    matrix[i, j] = val
    return texts, matrix


def within_text_sd(matrix):
    """Mean SD of dimension scores per text (across 10 dims)."""
    return np.nanstd(matrix, axis=1).mean()


def inter_dim_corr(matrix):
    """Mean |r| across all 45 dimension pairs."""
    n_dims = matrix.shape[1]
    rs = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            mask = ~np.isnan(matrix[:, i]) & ~np.isnan(matrix[:, j])
            if mask.sum() >= 5:
                r, _ = stats.pearsonr(matrix[mask, i], matrix[mask, j])
                rs.append(abs(r))
    return np.mean(rs), rs


def exact_neutral_rate(matrix, neutral=5.0):
    """Fraction of scores that are exactly the neutral value, per dim."""
    rates = {}
    for j, d in enumerate(DIMS):
        col = matrix[:, j]
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            rates[d] = (col[valid] == neutral).sum() / valid.sum()
    return rates


def eigenvalue_ratio(matrix):
    """First eigenvalue / total variance on z-scored data."""
    # Z-score each column
    z = np.copy(matrix)
    for j in range(z.shape[1]):
        col = z[:, j]
        mask = ~np.isnan(col)
        if mask.sum() > 2:
            z[mask, j] = (col[mask] - np.nanmean(col)) / max(np.nanstd(col), 1e-8)
    # Drop rows with any NaN
    valid = ~np.isnan(z).any(axis=1)
    z_clean = z[valid]
    if z_clean.shape[0] < 5:
        return np.nan, []
    cov = np.cov(z_clean.T)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    total = eigvals.sum()
    return eigvals[0] / total, eigvals


def entropy_per_dim(matrix):
    """Shannon entropy per dimension (how many effective score levels used)."""
    ents = {}
    for j, d in enumerate(DIMS):
        col = matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            ents[d] = 0.0
            continue
        # Round to 1 decimal for binning
        rounded = np.round(valid, 1)
        _, counts = np.unique(rounded, return_counts=True)
        probs = counts / counts.sum()
        ents[d] = -np.sum(probs * np.log2(probs + 1e-12))
    return ents


def per_dim_correlation(matrix_a, matrix_b, method="spearman"):
    """Per-dimension correlation between two score matrices."""
    cors = {}
    for j, d in enumerate(DIMS):
        col_a = matrix_a[:, j]
        col_b = matrix_b[:, j]
        mask = ~np.isnan(col_a) & ~np.isnan(col_b)
        if mask.sum() < 5:
            cors[d] = (np.nan, np.nan)
            continue
        if method == "spearman":
            r, p = stats.spearmanr(col_a[mask], col_b[mask])
        else:
            r, p = stats.pearsonr(col_a[mask], col_b[mask])
        cors[d] = (r, p)
    return cors


def per_dim_icc(matrix_a, matrix_b):
    """ICC(3,1) — two-way mixed, single measures, consistency."""
    iccs = {}
    for j, d in enumerate(DIMS):
        col_a = matrix_a[:, j]
        col_b = matrix_b[:, j]
        mask = ~np.isnan(col_a) & ~np.isnan(col_b)
        n = mask.sum()
        if n < 5:
            iccs[d] = np.nan
            continue
        a = col_a[mask]
        b = col_b[mask]
        # Two-way ANOVA decomposition
        grand_mean = (a.mean() + b.mean()) / 2
        ss_between = 2 * (((a + b) / 2 - grand_mean) ** 2).sum()
        ss_within = (((a - (a + b) / 2) ** 2) + ((b - (a + b) / 2) ** 2)).sum()
        ms_between = ss_between / (n - 1)
        ms_within = ss_within / n
        icc = (ms_between - ms_within) / (ms_between + ms_within)
        iccs[d] = max(-1.0, min(1.0, icc))
    return iccs


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def analyze_retest(control, treatment, n):
    """Phase 0: Test-retest analysis."""
    _, mat_c = load_scores(control, n=n)
    _, mat_t = load_scores(treatment, n=n)

    print_header("PHASE 0: TEST-RETEST BASELINE")

    # Per-dim correlations
    pearson = per_dim_correlation(mat_c, mat_t, method="pearson")
    spearman = per_dim_correlation(mat_c, mat_t, method="spearman")
    iccs = per_dim_icc(mat_c, mat_t)

    print(f"\n{'Dim':4s}  {'Pearson r':>10s}  {'Spearman ρ':>10s}  {'ICC(3,1)':>10s}  {'MAD':>6s}")
    print("-" * 50)
    pearson_rs = []
    mads = []
    for j, d in enumerate(DIMS):
        pr, _ = pearson[d]
        sr, _ = spearman[d]
        icc = iccs[d]
        mask = ~np.isnan(mat_c[:, j]) & ~np.isnan(mat_t[:, j])
        mad = np.abs(mat_c[mask, j] - mat_t[mask, j]).mean() if mask.sum() > 0 else np.nan
        print(f"{DIM_SHORT[d]:4s}  {pr:10.3f}  {sr:10.3f}  {icc:10.3f}  {mad:6.2f}")
        if not np.isnan(pr):
            pearson_rs.append(pr)
        if not np.isnan(mad):
            mads.append(mad)

    print(f"\nMean Pearson r:  {np.mean(pearson_rs):.3f}")
    print(f"Mean MAD:        {np.mean(mads):.3f}")

    # Within-text SD comparison
    sd_c = within_text_sd(mat_c)
    sd_t = within_text_sd(mat_t)
    delta_noise = abs(sd_c - sd_t)
    print(f"\nWithin-text SD:  control={sd_c:.3f}, retest={sd_t:.3f}")
    print(f"Δ_noise:         {delta_noise:.3f}")
    print(f"  → Treatment effects must exceed 2×Δ_noise = {2 * delta_noise:.3f}")

    # Go/no-go
    go = sum(1 for r in pearson_rs if r >= 0.80)
    total = len(pearson_rs)
    print(f"\nGO/NO-GO: {go}/{total} dims have test-retest r ≥ 0.80")
    if go >= 7:
        print("  → GO: scoring is stable enough for experiments")
    else:
        print("  → CAUTION: scoring may be too unstable")

    return delta_noise


def analyze_ab(control, treatment):
    """Phases 1-2: A/B experiment analysis."""
    _, mat_c = load_scores(control)
    _, mat_t = load_scores(treatment)

    print_header("A/B EXPERIMENT ANALYSIS")

    # M1: Within-text SD
    sd_c = within_text_sd(mat_c)
    sd_t = within_text_sd(mat_t)
    pct_change = (sd_t - sd_c) / sd_c * 100

    print(f"\nM1 — Within-text SD:")
    print(f"  Control:    {sd_c:.3f}")
    print(f"  Treatment:  {sd_t:.3f}")
    print(f"  Change:     {pct_change:+.1f}%")

    # M2: Inter-dimension |r|
    corr_c, _ = inter_dim_corr(mat_c)
    corr_t, _ = inter_dim_corr(mat_t)
    print(f"\nM2 — Mean inter-dimension |r|:")
    print(f"  Control:    {corr_c:.3f}")
    print(f"  Treatment:  {corr_t:.3f}")
    print(f"  Change:     {corr_t - corr_c:+.3f}")

    # M3: Exact-neutral rate
    neut_c = exact_neutral_rate(mat_c)
    neut_t = exact_neutral_rate(mat_t)
    print(f"\nM3 — Exact-5 rate (per dim):")
    print(f"  {'Dim':4s}  {'Control':>8s}  {'Treatment':>10s}  {'Change':>8s}")
    print(f"  {'-'*36}")
    for d in DIMS:
        c = neut_c.get(d, 0)
        t = neut_t.get(d, 0)
        print(f"  {DIM_SHORT[d]:4s}  {c:8.0%}  {t:10.0%}  {t-c:+8.0%}")

    # M4: Control-treatment correlation (construct stability)
    cors = per_dim_correlation(mat_c, mat_t, method="spearman")
    print(f"\nM4 — Control-treatment Spearman ρ:")
    rhos = []
    for d in DIMS:
        r, p = cors[d]
        flag = " ⚠" if (not np.isnan(r) and r < 0.70) else ""
        print(f"  {DIM_SHORT[d]:4s}  {r:.3f}  (p={p:.3f}){flag}")
        if not np.isnan(r):
            rhos.append(r)
    print(f"  Mean ρ: {np.mean(rhos):.3f}")

    # M5: Eigenvalue ratio
    ev_c, _ = eigenvalue_ratio(mat_c)
    ev_t, _ = eigenvalue_ratio(mat_t)
    print(f"\nM5 — Eigenvalue ratio (EV1/total, z-scored):")
    print(f"  Control:    {ev_c:.3f} ({ev_c*100:.1f}%)")
    print(f"  Treatment:  {ev_t:.3f} ({ev_t*100:.1f}%)")

    # M6: Entropy
    ent_c = entropy_per_dim(mat_c)
    ent_t = entropy_per_dim(mat_t)
    print(f"\nM6 — Entropy per dimension:")
    print(f"  {'Dim':4s}  {'Control':>8s}  {'Treatment':>10s}  {'Change':>8s}")
    print(f"  {'-'*36}")
    for d in DIMS:
        c = ent_c.get(d, 0)
        t = ent_t.get(d, 0)
        print(f"  {DIM_SHORT[d]:4s}  {c:8.2f}  {t:10.2f}  {t-c:+8.2f}")


def analyze_crossscale(control, treatment):
    """Phase 3: Cross-scale analysis (rank-based metrics only)."""
    _, mat_c = load_scores(control)
    _, mat_t = load_scores(treatment)

    print_header("CROSS-SCALE ANALYSIS")
    print("(Only scale-invariant metrics shown)")

    # M2: Inter-dimension |r|
    corr_c, _ = inter_dim_corr(mat_c)
    corr_t, _ = inter_dim_corr(mat_t)
    print(f"\nM2 — Mean inter-dimension |r|:")
    print(f"  Scale A:  {corr_c:.3f}")
    print(f"  Scale B:  {corr_t:.3f}")

    # M5: Eigenvalue ratio (z-scored)
    ev_c, _ = eigenvalue_ratio(mat_c)
    ev_t, _ = eigenvalue_ratio(mat_t)
    print(f"\nM5 — Eigenvalue ratio (z-scored):")
    print(f"  Scale A:  {ev_c:.3f} ({ev_c*100:.1f}%)")
    print(f"  Scale B:  {ev_t:.3f} ({ev_t*100:.1f}%)")

    # M6: Entropy
    ent_c = entropy_per_dim(mat_c)
    ent_t = entropy_per_dim(mat_t)
    print(f"\nM6 — Entropy per dimension:")
    print(f"  {'Dim':4s}  {'Scale A':>8s}  {'Scale B':>10s}")
    print(f"  {'-'*24}")
    for d in DIMS:
        c = ent_c.get(d, 0)
        t = ent_t.get(d, 0)
        print(f"  {DIM_SHORT[d]:4s}  {c:8.2f}  {t:10.2f}")

    # Cross-scale rank correlation
    cors = per_dim_correlation(mat_c, mat_t, method="spearman")
    print(f"\nCross-scale Spearman ρ:")
    for d in DIMS:
        r, p = cors[d]
        print(f"  {DIM_SHORT[d]:4s}  {r:.3f}")


def main():
    parser = argparse.ArgumentParser(description="PSQ scoring experiment analysis")
    parser.add_argument("--control", required=True, help="Control condition JSONL")
    parser.add_argument("--treatment", required=True, help="Treatment condition JSONL")
    parser.add_argument("--mode", choices=["retest", "ab", "crossscale"],
                        default="ab", help="Analysis mode")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N records (for retest)")
    args = parser.parse_args()

    if args.mode == "retest":
        analyze_retest(args.control, args.treatment, args.n)
    elif args.mode == "ab":
        analyze_ab(args.control, args.treatment)
    elif args.mode == "crossscale":
        analyze_crossscale(args.control, args.treatment)


if __name__ == "__main__":
    main()
