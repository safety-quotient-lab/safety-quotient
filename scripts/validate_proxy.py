"""
Validate detoxify as proxy teacher for PSQ distillation.

Loads the Berkeley Measuring Hate Speech dataset, runs detoxify on a
stratified sample, and computes Pearson r correlations between detoxify
attributes and Berkeley ground-truth labels.

Decision gate: if r > 0.7 for hostility-related mappings, detoxify is
a viable proxy teacher. Otherwise, fall back to LLM-only labeling.

Output:
  - Correlation table (stdout)
  - Scatter plots + correlation matrix heatmap (data/proxy-validation/)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "proxy-validation"

PARQUET_PATH = DATA_DIR / "measuring-hate-speech.parquet"
SAMPLE_SIZE = 1000

# Mapping: detoxify attribute → Berkeley label
PROXY_PAIRS = [
    ("toxicity",       "hate_speech_score", "Toxicity vs Hate Speech Score"),
    ("insult",         "insult",            "Insult vs Insult"),
    ("threat",         "violence",          "Threat vs Violence"),
    ("identity_attack","dehumanize",        "Identity Attack vs Dehumanize"),
    ("severe_toxicity","hatespeech",        "Severe Toxicity vs Hatespeech"),
]

# PSQ dimension mapping (what we'll use if validation passes)
PSQ_PROXY_MAP = {
    "hostility_index":  ["toxicity", "insult", "identity_attack"],
    "threat_exposure":  ["severe_toxicity", "threat"],
}


def load_and_sample(path, n):
    """Load parquet, deduplicate on text, stratified sample by hate_speech_score."""
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"  Raw rows: {len(df)}")

    # Deduplicate — Berkeley has multiple annotator rows per comment
    df_unique = df.drop_duplicates(subset="text").copy()
    print(f"  Unique texts: {len(df_unique)}")

    # Stratified sample: split into 5 quintiles, sample equally from each
    df_unique["quintile"] = pd.qcut(
        df_unique["hate_speech_score"], q=5, labels=False, duplicates="drop"
    )
    per_q = n // 5
    samples = []
    for q in sorted(df_unique["quintile"].unique()):
        stratum = df_unique[df_unique["quintile"] == q]
        take = min(per_q, len(stratum))
        samples.append(stratum.sample(n=take, random_state=42))

    result = pd.concat(samples).reset_index(drop=True)
    print(f"  Stratified sample: {len(result)} texts")
    return result


def run_detoxify(texts):
    """Run detoxify on a list of texts, return DataFrame of scores."""
    from detoxify import Detoxify

    print(f"\nRunning detoxify on {len(texts)} texts...")
    model = Detoxify("original")
    results = model.predict(texts)
    print("  Done.")
    return pd.DataFrame(results)


def compute_correlations(berkeley_df, detox_df):
    """Compute Pearson r for each proxy pair."""
    print("\n" + "=" * 70)
    print("CORRELATION RESULTS: Detoxify vs Berkeley Ground Truth")
    print("=" * 70)

    results = []
    for detox_col, berkeley_col, label in PROXY_PAIRS:
        if detox_col not in detox_df.columns:
            print(f"  SKIP: {detox_col} not in detoxify output")
            continue
        if berkeley_col not in berkeley_df.columns:
            print(f"  SKIP: {berkeley_col} not in Berkeley data")
            continue

        x = detox_df[detox_col].values
        y = berkeley_df[berkeley_col].values

        # Drop NaNs
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        r, p = stats.pearsonr(x, y)
        results.append({
            "detoxify": detox_col,
            "berkeley": berkeley_col,
            "label": label,
            "pearson_r": r,
            "p_value": p,
            "n": len(x),
        })
        print(f"  {label:40s}  r={r:+.4f}  p={p:.2e}  n={len(x)}")

    return results


def compute_psq_proxy_correlations(berkeley_df, detox_df):
    """Compute composite proxy scores for PSQ dimensions and correlate."""
    print("\n" + "=" * 70)
    print("PSQ DIMENSION PROXY QUALITY")
    print("=" * 70)

    results = []
    for dim, detox_cols in PSQ_PROXY_MAP.items():
        available = [c for c in detox_cols if c in detox_df.columns]
        if not available:
            print(f"  {dim}: no detoxify columns available")
            continue

        # Composite: mean of available detoxify scores
        composite = detox_df[available].mean(axis=1).values

        # Best Berkeley correlate for this dimension
        if dim == "hostility_index":
            berkeley_col = "hate_speech_score"
        elif dim == "threat_exposure":
            berkeley_col = "violence"
        else:
            continue

        y = berkeley_df[berkeley_col].values
        mask = ~(np.isnan(composite) | np.isnan(y))
        x, y = composite[mask], y[mask]

        r, p = stats.pearsonr(x, y)
        results.append({
            "dimension": dim,
            "proxy_cols": available,
            "berkeley_col": berkeley_col,
            "pearson_r": r,
            "p_value": p,
        })
        gate = "PASS" if abs(r) > 0.7 else "FAIL"
        print(f"  {dim:25s}  r={r:+.4f}  [{gate}]  (proxy: {'+'.join(available)} vs {berkeley_col})")

    return results


def save_plots(berkeley_df, detox_df):
    """Save scatter plots and correlation heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Individual scatter plots
    for detox_col, berkeley_col, label in PROXY_PAIRS:
        if detox_col not in detox_df.columns or berkeley_col not in berkeley_df.columns:
            continue

        x = detox_df[detox_col].values
        y = berkeley_df[berkeley_col].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, alpha=0.3, s=10)
        r, _ = stats.pearsonr(x, y)
        ax.set_xlabel(f"Detoxify: {detox_col}")
        ax.set_ylabel(f"Berkeley: {berkeley_col}")
        ax.set_title(f"{label}\nr = {r:.4f}, n = {len(x)}")

        # Fit line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8)

        fig.tight_layout()
        fig.savefig(OUT_DIR / f"scatter_{detox_col}_vs_{berkeley_col}.png", dpi=150)
        plt.close(fig)

    # Correlation heatmap: all detoxify cols vs all Berkeley cols
    detox_cols = [p[0] for p in PROXY_PAIRS if p[0] in detox_df.columns]
    berkeley_cols = [p[1] for p in PROXY_PAIRS if p[1] in berkeley_df.columns]

    corr_matrix = np.zeros((len(detox_cols), len(berkeley_cols)))
    for i, dc in enumerate(detox_cols):
        for j, bc in enumerate(berkeley_cols):
            x = detox_df[dc].values
            y = berkeley_df[bc].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                corr_matrix[i, j], _ = stats.pearsonr(x[mask], y[mask])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(berkeley_cols)))
    ax.set_xticklabels(berkeley_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(detox_cols)))
    ax.set_yticklabels(detox_cols)
    ax.set_title("Detoxify vs Berkeley: Pearson r Correlation Matrix")

    # Annotate cells
    for i in range(len(detox_cols)):
        for j in range(len(berkeley_cols)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")

    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {OUT_DIR}/")


def main():
    if not PARQUET_PATH.exists():
        print(f"ERROR: Dataset not found at {PARQUET_PATH}")
        print("Run: curl -L 'https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/main/measuring-hate-speech.parquet' -o data/measuring-hate-speech.parquet")
        sys.exit(1)

    # Load and sample
    berkeley_df = load_and_sample(PARQUET_PATH, SAMPLE_SIZE)

    # Run detoxify
    texts = berkeley_df["text"].tolist()
    detox_df = run_detoxify(texts)

    # Correlations
    pair_results = compute_correlations(berkeley_df, detox_df)
    psq_results = compute_psq_proxy_correlations(berkeley_df, detox_df)

    # Plots
    save_plots(berkeley_df, detox_df)

    # Decision gate
    print("\n" + "=" * 70)
    print("DECISION GATE")
    print("=" * 70)

    hostility_r = None
    for res in psq_results:
        if res["dimension"] == "hostility_index":
            hostility_r = abs(res["pearson_r"])

    if hostility_r is None:
        print("  FAIL — could not compute hostility_index proxy correlation")
        print("  Recommendation: fall back to LLM-only labeling")
        sys.exit(1)

    if hostility_r > 0.7:
        print(f"  PASS — hostility_index proxy r = {hostility_r:.4f} > 0.7")
        print("  Recommendation: proceed with detoxify as proxy teacher for")
        print("  hostility_index and threat_exposure dimensions.")
        print("  Other 8 dimensions: score 5, confidence 0.2 (no coverage)")
    elif hostility_r > 0.5:
        print(f"  MARGINAL — hostility_index proxy r = {hostility_r:.4f}")
        print("  Recommendation: use detoxify with caution, increase LLM gold samples")
    else:
        print(f"  FAIL — hostility_index proxy r = {hostility_r:.4f} < 0.5")
        print("  Recommendation: fall back to LLM-only labeling")

    # Save summary
    summary = {
        "sample_size": len(berkeley_df),
        "pair_correlations": pair_results,
        "psq_proxy_correlations": [
            {k: v for k, v in r.items()} for r in psq_results
        ],
        "hostility_r": hostility_r,
        "decision": "pass" if hostility_r > 0.7 else "marginal" if hostility_r > 0.5 else "fail",
    }

    import json
    summary_path = OUT_DIR / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
