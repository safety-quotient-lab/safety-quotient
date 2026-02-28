#!/usr/bin/env python3
"""
Turn-by-turn temporal PSQ analysis — CGA-Wiki (Conversations Gone Awry).

Tests prediction T2 from journal §24:
  In conversations that eventually derail, AD scores should deteriorate
  *before* HI and TE scores — i.e., authority dynamics is a leading indicator
  of hostility/threat, not a co-occurring dimension.

Method: Cross-lagged correlation analysis on consecutive utterance pairs.
  For each consecutive pair (t, t+1) within a conversation:
    r(AD_t, HI_{t+1}) vs r(HI_t, AD_{t+1})
  T2 predicts: r(AD→HI) > r(HI→AD) in derailing conversations.

Corpus: Zhang et al. (2018) "Conversations Gone Awry"
  ~/.convokit/saved-corpora/conversations-gone-awry-corpus/
  4,188 conversations (2,094 derailing + 2,094 control), all splits used.

Output:
  - Cross-lagged correlation table (AD/HI/TE, lead vs. lag)
  - Temporal trajectory: mean PSQ by relative position x outcome
  - T2 verdict: SUPPORTED / NOT SUPPORTED / INCONCLUSIVE
"""

import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT  = Path("/home/kashif/projects/psychology/safety-quotient")
CORPUS   = Path("/home/kashif/.convokit/saved-corpora/conversations-gone-awry-corpus")
MODEL_DIR = PROJECT / "models" / "psq-v23"

DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]
DIM_ABBREV = ["TE", "HI", "AD", "ED", "RC", "RB", "TC", "CC", "DA", "CO"]
DIM_IDX = {a: i for i, a in enumerate(DIM_ABBREV)}

BATCH_SIZE  = 64
MAX_LENGTH  = 256
MIN_TEXT_LEN = 20   # skip very short utterances (wiki formatting artifacts)


# ── Model ────────────────────────────────────────────────────────────────────

class PSQStudent(nn.Module):
    """Matches the architecture in scripts/distill.py exactly."""

    def __init__(self, model_name="distilbert-base-uncased", n_dims=10):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
        hidden = self.encoder.config.hidden_size

        self.proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden // 2, 2) for _ in range(n_dims)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        projected = self.proj(cls)

        scores, confs = [], []
        for head in self.heads:
            out = head(projected)
            scores.append(torch.sigmoid(out[:, 0]) * 10.0)
            confs.append(torch.sigmoid(out[:, 1]))

        return torch.stack(scores, dim=1), torch.stack(confs, dim=1)


def load_model(device):
    print(f"Loading model from {MODEL_DIR.name}...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))
    model = PSQStudent()
    ckpt = torch.load(MODEL_DIR / "best.pt", map_location="cpu", weights_only=False)
    # Checkpoint is the state dict directly (no wrapper)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, tok


# ── Corpus loading ────────────────────────────────────────────────────────────

def load_corpus():
    """Returns (conversations dict, utterances_by_convo dict)."""
    print("Loading CGA-Wiki corpus...", flush=True)

    # Load conversation metadata
    convos = json.loads((CORPUS / "conversations.json").read_text())

    # Load utterances, group by conversation, sorted by timestamp
    utts_by_convo: dict[str, list[dict]] = defaultdict(list)
    with open(CORPUS / "utterances.jsonl") as f:
        for line in f:
            u = json.loads(line)
            if u["meta"].get("is_section_header", False):
                continue
            text = u.get("text", "").strip()
            if len(text) < MIN_TEXT_LEN:
                continue
            utts_by_convo[u["conversation_id"]].append(u)

    # Sort each conversation's utterances by timestamp (ascending)
    for cid in utts_by_convo:
        utts_by_convo[cid].sort(key=lambda u: u.get("timestamp") or 0)

    # Filter: keep only conversations with ≥ 2 utterances
    utts_by_convo = {k: v for k, v in utts_by_convo.items() if len(v) >= 2}

    n_derail = sum(1 for cid in utts_by_convo
                   if convos.get(cid, {}).get("conversation_has_personal_attack", False))
    n_control = len(utts_by_convo) - n_derail
    total_utts = sum(len(v) for v in utts_by_convo.values())
    print(f"  {len(utts_by_convo)} conversations ({n_derail} derailing / {n_control} control)")
    print(f"  {total_utts} utterances for scoring")

    return convos, utts_by_convo


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_utterances(utts_flat, model, tokenizer, device):
    """Score all utterances in flat list. Returns np.ndarray [N, 10]."""
    texts = [u["text"] for u in utts_flat]
    all_scores = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            scores, _ = model(enc["input_ids"], enc["attention_mask"])
        all_scores.append(scores.cpu().numpy())

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  Scored {min(i + BATCH_SIZE, len(texts))}/{len(texts)} utterances",
                  end="\r", flush=True)

    print()
    return np.concatenate(all_scores, axis=0)


# ── Cross-lagged analysis ─────────────────────────────────────────────────────

def fisher_z(r):
    """Fisher r-to-z transformation."""
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))


def fisher_z_test(r1, n1, r2, n2):
    """Two-tailed Fisher z-test comparing two correlations. Returns z, p."""
    z1 = fisher_z(r1)
    z2 = fisher_z(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def compute_cross_lagged(convos, utts_by_convo, scores_by_uid):
    """
    For each consecutive pair (t, t+1) within a conversation,
    record PSQ vectors. Return lagged pairs for derailing and control.

    Returns: (derail_pairs, control_pairs)
      Each is a list of (psq_t: np.ndarray[10], psq_t1: np.ndarray[10])
    """
    derail_pairs, control_pairs = [], []

    for cid, utts in utts_by_convo.items():
        is_derail = convos.get(cid, {}).get("conversation_has_personal_attack", False)
        target = derail_pairs if is_derail else control_pairs

        for i in range(len(utts) - 1):
            uid_t  = utts[i]["id"]
            uid_t1 = utts[i + 1]["id"]
            if uid_t not in scores_by_uid or uid_t1 not in scores_by_uid:
                continue
            target.append((scores_by_uid[uid_t], scores_by_uid[uid_t1]))

    return derail_pairs, control_pairs


def cross_lagged_table(pairs, label):
    """
    Compute cross-lagged correlations for all dimension pairs of interest.
    Prints a table: dim_A → dim_B (r, n, p) and dim_B → dim_A (r, n, p).
    Returns dict of results.
    """
    if not pairs:
        print(f"  No pairs for {label}")
        return {}

    psq_t  = np.array([p[0] for p in pairs])   # [N, 10]
    psq_t1 = np.array([p[1] for p in pairs])   # [N, 10]
    n = len(pairs)

    print(f"\n{'─'*72}")
    print(f"Cross-lagged correlations — {label} (N={n} consecutive pairs)")
    print(f"{'─'*72}")
    print(f"{'A → B':8} {'B → A':8}  {'Δr':6}  z-stat  p-val   Verdict")
    print(f"{'r(A_t,B_t+1)':14} {'r(B_t,A_t+1)':14}")
    print(f"{'─'*72}")

    # Focal pairs for T2: AD→HI, AD→TE, ED→HI, ED→TE
    pairs_of_interest = [
        ("AD", "HI"),
        ("AD", "TE"),
        ("ED", "HI"),
        ("ED", "TE"),
        ("HI", "TE"),  # control pair: expect symmetry
    ]

    results = {}
    for a, b in pairs_of_interest:
        ia, ib = DIM_IDX[a], DIM_IDX[b]
        r_ab = float(np.corrcoef(psq_t[:, ia], psq_t1[:, ib])[0, 1])  # A_t → B_{t+1}
        r_ba = float(np.corrcoef(psq_t[:, ib], psq_t1[:, ia])[0, 1])  # B_t → A_{t+1}
        z, p = fisher_z_test(r_ab, n, r_ba, n)
        delta = r_ab - r_ba

        if p < 0.01:
            verdict = "**" if abs(delta) > 0.02 else "~"
        elif p < 0.05:
            verdict = "*"
        else:
            verdict = "ns"

        direction = f"{a}→{b}" if r_ab > r_ba else f"{b}→{a}"
        print(f"{a}↔{b}:   r={r_ab:+.3f}   r={r_ba:+.3f}   Δ={delta:+.3f}  z={z:+5.2f}  p={p:.3f}  {verdict} ({direction} leads)")

        results[(a, b)] = {"r_ab": r_ab, "r_ba": r_ba, "delta": delta, "z": z, "p": p, "n": n}

    return results


# ── Temporal trajectory ───────────────────────────────────────────────────────

def temporal_trajectory(convos, utts_by_convo, scores_by_uid):
    """
    Bin turns by relative position (0%, 25%, 50%, 75%, 100%) and compute
    mean PSQ scores per bin × outcome. Shows whether AD deteriorates before HI/TE.
    """
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    bin_labels = ["Q1 (start)", "Q2", "Q3", "Q4 (end)"]

    # Accumulate scores per bin
    # Structure: {is_derail: {bin_idx: {dim_idx: [scores]}}}
    accum: dict = {True: {i: defaultdict(list) for i in range(4)},
                   False: {i: defaultdict(list) for i in range(4)}}

    for cid, utts in utts_by_convo.items():
        is_derail = convos.get(cid, {}).get("conversation_has_personal_attack", False)
        n = len(utts)
        for pos, u in enumerate(utts):
            if u["id"] not in scores_by_uid:
                continue
            rel_pos = pos / max(n - 1, 1)
            # Assign to bin
            bin_idx = min(int(rel_pos / 0.25), 3)
            psq = scores_by_uid[u["id"]]
            for d in range(10):
                accum[is_derail][bin_idx][d].append(psq[d])

    # Focus dimensions: AD, HI, TE, ED
    focus = ["AD", "HI", "TE", "ED"]
    focus_idx = [DIM_IDX[d] for d in focus]

    print(f"\n{'─'*72}")
    print("Temporal trajectory — mean PSQ by conversation quartile")
    print(f"{'─'*72}")

    for label, is_derail in [("DERAILING", True), ("CONTROL", False)]:
        print(f"\n{label}:")
        header = f"{'Quarter':<14}" + "".join(f"{d:>8}" for d in focus)
        print(header)
        print("─" * len(header))
        for bin_idx, bin_lbl in enumerate(bin_labels):
            row = f"{bin_lbl:<14}"
            for d_idx in focus_idx:
                vals = accum[is_derail][bin_idx][d_idx]
                m = np.mean(vals) if vals else float("nan")
                row += f"{m:8.3f}"
            print(row)

    # Difference: derailing minus control per quarter per focus dim
    print(f"\nDERAILING − CONTROL (positive = higher in derailing convos):")
    header = f"{'Quarter':<14}" + "".join(f"{d:>8}" for d in focus)
    print(header)
    print("─" * len(header))
    for bin_idx, bin_lbl in enumerate(bin_labels):
        row = f"{bin_lbl:<14}"
        for d_idx in focus_idx:
            d_vals = accum[True][bin_idx][d_idx]
            c_vals = accum[False][bin_idx][d_idx]
            diff = (np.mean(d_vals) if d_vals else float("nan")) - \
                   (np.mean(c_vals) if c_vals else float("nan"))
            row += f"{diff:+8.3f}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_model(device)
    convos, utts_by_convo = load_corpus()

    # Flatten all utterances for batch scoring
    utts_flat = []
    for utts in utts_by_convo.values():
        utts_flat.extend(utts)

    print(f"\nScoring {len(utts_flat)} utterances...", flush=True)
    scores_matrix = score_utterances(utts_flat, model, tokenizer, device)

    # Build uid → score vector lookup
    scores_by_uid = {u["id"]: scores_matrix[i] for i, u in enumerate(utts_flat)}

    # Cross-lagged analysis
    print("\nComputing cross-lagged correlations...", flush=True)
    derail_pairs, control_pairs = compute_cross_lagged(convos, utts_by_convo, scores_by_uid)
    print(f"  Derailing pairs: {len(derail_pairs)}, Control pairs: {len(control_pairs)}")

    derail_results = cross_lagged_table(derail_pairs, "DERAILING conversations")
    control_results = cross_lagged_table(control_pairs, "CONTROL conversations")

    # Temporal trajectory
    temporal_trajectory(convos, utts_by_convo, scores_by_uid)

    # T2 Verdict
    print(f"\n{'═'*72}")
    print("T2 VERDICT — AD leads HI in derailing conversations")
    print(f"{'═'*72}")

    if ("AD", "HI") in derail_results and ("AD", "HI") in control_results:
        d = derail_results[("AD", "HI")]
        c = control_results[("AD", "HI")]
        print(f"Derailing: r(AD_t, HI_{{t+1}})={d['r_ab']:+.3f}  vs  r(HI_t, AD_{{t+1}})={d['r_ba']:+.3f}  Δ={d['delta']:+.3f}  p={d['p']:.3f}")
        print(f"Control:   r(AD_t, HI_{{t+1}})={c['r_ab']:+.3f}  vs  r(HI_t, AD_{{t+1}})={c['r_ba']:+.3f}  Δ={c['delta']:+.3f}  p={c['p']:.3f}")

        ad_leads_derail = d["r_ab"] > d["r_ba"] and d["p"] < 0.05
        ad_leads_control = c["r_ab"] > c["r_ba"] and c["p"] < 0.05
        delta_difference = d["delta"] - c["delta"]

        if ad_leads_derail and not ad_leads_control:
            verdict = "SUPPORTED — AD leads HI specifically in derailing conversations (not in controls)"
        elif ad_leads_derail and ad_leads_control:
            verdict = "PARTIAL — AD leads HI in both groups; not derailment-specific"
        elif d["p"] > 0.05 and d["delta"] > 0:
            verdict = "INCONCLUSIVE — directional but non-significant (AD_t→HI stronger)"
        elif d["delta"] <= 0:
            verdict = "NOT SUPPORTED — HI leads AD (or no directional asymmetry)"
        else:
            verdict = "NOT SUPPORTED"

        print(f"\nVerdict: {verdict}")
        print(f"Δr difference (derailing minus control): {delta_difference:+.3f}")
    else:
        print("Could not compute T2 verdict (missing results)")

    print()


if __name__ == "__main__":
    main()
