"""
Evaluate trained PSQ model against held-out real-world texts.

These texts are from the unlabeled pool (no overlap with composite or LLM training data)
and are LLM-labeled independently. This provides a fair generalization test that is
immune to synthetic writing style bias.

Usage:
    python scripts/eval_held_out.py [--model models/psq-student/best.pt]
"""

import argparse
import json
import sys
from pathlib import Path
from scipy.stats import pearsonr
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
HELD_OUT = ROOT / "data" / "held-out-test.jsonl"
DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]


class PSQStudent(nn.Module):
    """Mirror of the training architecture in distill.py."""

    def __init__(self, model_name="distilbert-base-uncased", n_dims=10):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name).float()
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

        scores = []
        confs = []
        for head in self.heads:
            out = head(projected)
            s = torch.sigmoid(out[:, 0]) * 10.0
            c = torch.sigmoid(out[:, 1])
            scores.append(s)
            confs.append(c)

        return torch.stack(scores, dim=1), torch.stack(confs, dim=1)


def load_model(model_path):
    """Load trained PSQ model from checkpoint."""
    from transformers import AutoTokenizer

    # Load state dict
    state = torch.load(model_path, map_location="cpu", weights_only=False)

    # Detect model type from checkpoint keys
    has_config = isinstance(state, dict) and "config" in state
    if has_config:
        config = state["config"]
        model_name = config.get("model_name", "distilbert-base-uncased")
        encoder_sd = state["encoder_state_dict"]
        # Reconstruct head from separate state dict
        raise NotImplementedError("New-format checkpoints not yet supported here")
    else:
        # Flat state dict format (v3b+ saves all weights at top level)
        # Detect model from key patterns
        if any("deberta" in k for k in state.keys()):
            model_name = "microsoft/deberta-v3-small"
        else:
            model_name = "distilbert-base-uncased"

        model = PSQStudent(model_name=model_name, n_dims=len(DIMS))
        # Load matching keys
        model_sd = model.state_dict()
        loaded = {}
        for k, v in state.items():
            # Map flat keys like "encoder.xxx" to model keys
            if k in model_sd:
                loaded[k] = v

        missing = set(model_sd.keys()) - set(loaded.keys())
        if missing:
            print(f"  WARNING: {len(missing)} missing keys: {list(missing)[:5]}...")

        model.load_state_dict(loaded, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    return tokenizer, model


def predict(tokenizer, model, texts, batch_size=32):
    """Run inference on a list of texts."""
    all_scores = []
    all_confs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            scores, confs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        all_scores.append(scores.numpy())
        all_confs.append(confs.numpy())

    return np.vstack(all_scores), np.vstack(all_confs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "psq-student" / "best.pt"))
    args = parser.parse_args()

    if not HELD_OUT.exists():
        print(f"Error: {HELD_OUT} not found. Run held-out labeling first.")
        sys.exit(1)

    # Load held-out data
    with open(HELD_OUT) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(records)} held-out test records")

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer, model = load_model(args.model)

    # Run predictions
    texts = [r["text"] for r in records]
    pred_scores, pred_confs = predict(tokenizer, model, texts)

    # Evaluate per dimension
    print(f"\n{'Dimension':<25} {'r':>7} {'MSE':>8} {'n':>5}  Source")
    print("-" * 60)

    results = {}
    rs = []
    for i, dim in enumerate(DIMS):
        true_scores = []
        true_confs = []
        preds = []

        for j, rec in enumerate(records):
            d = rec.get("dimensions", {}).get(dim, {})
            score = d.get("score")
            conf = d.get("confidence", 0)
            if score is not None and conf > 0.3:
                true_scores.append(score)
                true_confs.append(conf)
                preds.append(pred_scores[j, i])

        if len(true_scores) >= 10:
            r, p = pearsonr(true_scores, preds)
            mse = np.mean((np.array(true_scores) - np.array(preds)) ** 2)
            results[dim] = {"r": round(r, 4), "p": round(p, 4), "mse": round(mse, 4), "n": len(true_scores)}
            rs.append(r)
            sig = "*" if p < 0.05 else " "
            print(f"  {dim:<25} {r:>+.4f} {mse:>8.4f} {len(true_scores):>5}  real-world {sig}")
        else:
            results[dim] = {"r": None, "n": len(true_scores)}
            print(f"  {dim:<25}    n/a      n/a {len(true_scores):>5}  (too few)")

    if rs:
        avg = np.mean(rs)
        print(f"\n  {'AVERAGE (held-out)':<25} {avg:>+.4f}  ({len(rs)}/10 dims)")
        results["_avg_r"] = round(avg, 4)
        results["_n_dims"] = len(rs)

    # Save results alongside the model checkpoint
    out_path = Path(args.model).parent / "held_out_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
