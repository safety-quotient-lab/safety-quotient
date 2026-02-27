#!/usr/bin/env python3
"""Helper for in-conversation LLM labeling.

Extract:  python scripts/label_batch_helper.py extract --batch-size 50 --batch-num 1
Append:   python scripts/label_batch_helper.py append --input /tmp/psq_scored_1.jsonl
Stats:    python scripts/label_batch_helper.py stats
"""

import json
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
COMPOSITE = ROOT / "data" / "composite-ground-truth.jsonl"
POOL_FILE = ROOT / "data" / "unlabeled-pool.jsonl"
LLM_FILE = ROOT / "data" / "train-llm.jsonl"
DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity"
]


def load_labeled_keys():
    keys = set()
    if LLM_FILE.exists():
        with open(LLM_FILE) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                keys.add(rec["text"][:200])
    return keys


def load_unlabeled(labeled_keys):
    """Load texts from unlabeled pool (unique texts not in composite).

    Falls back to composite if pool file doesn't exist.
    """
    source = POOL_FILE if POOL_FILE.exists() else COMPOSITE
    texts = []
    with open(source) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("teacher") in ("llm", "llm_labeled"):
                continue
            key = rec["text"][:200]
            if key in labeled_keys:
                continue
            texts.append({
                "text": rec["text"],
                "source": rec.get("source", "unknown"),
            })
    if source == POOL_FILE:
        print(f"  (using unlabeled pool: {len(texts)} available)")
    else:
        print(f"  (WARNING: pool not found, falling back to composite)")
    return texts


def cmd_extract(args):
    labeled = load_labeled_keys()
    unlabeled = load_unlabeled(labeled)

    # Deterministic shuffle
    random.seed(42)
    random.shuffle(unlabeled)

    start = (args.batch_num - 1) * args.batch_size
    end = start + args.batch_size
    batch = unlabeled[start:end]

    total_batches = (len(unlabeled) + args.batch_size - 1) // args.batch_size

    # Write to temp file for Claude Code to read
    out_path = f"/tmp/psq_batch_{args.batch_num}.json"
    with open(out_path, "w") as f:
        json.dump({
            "batch_num": args.batch_num,
            "total_batches": total_batches,
            "count": len(batch),
            "texts": [{"id": start + i, "source": t["source"], "text": t["text"]}
                      for i, t in enumerate(batch)]
        }, f, indent=1)

    # Also save texts lookup for compact mode
    lookup_path = f"/tmp/psq_lookup_{args.batch_num}.json"
    with open(lookup_path, "w") as f:
        json.dump({str(start + i): {"text": t["text"], "source": t["source"]}
                   for i, t in enumerate(batch)}, f)

    print(f"Batch {args.batch_num}/{total_batches}: {len(batch)} texts → {out_path}")
    print(f"Total unlabeled: {len(unlabeled)}, already labeled: {len(labeled)}")


def cmd_append(args):
    """Append scored results to train-llm.jsonl.

    Supports two formats:
    1. Full: [{"text":"...", "dimensions":{...}}, ...]
    2. Compact: {"batch": N, "scores": {"ID": {"te":3,"hi":2,...}, ...}}
       Where dim keys are 2-char abbreviations. Lookup file provides text+source.
    """
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        return

    DIM_ABBREV = {
        "te": "threat_exposure", "hi": "hostility_index",
        "ad": "authority_dynamics", "ed": "energy_dissipation",
        "rc": "regulatory_capacity", "rb": "resilience_baseline",
        "tc": "trust_conditions", "cc": "cooling_capacity",
        "da": "defensive_architecture", "co": "contractual_clarity"
    }

    with open(input_path) as f:
        data = json.load(f)

    count = 0
    errors = 0

    # Detect format
    if isinstance(data, dict) and "scores" in data:
        # Compact format
        batch_num = data.get("batch", 0)
        lookup_path = Path(f"/tmp/psq_lookup_{batch_num}.json")
        if not lookup_path.exists():
            print(f"ERROR: lookup file {lookup_path} not found")
            return
        with open(lookup_path) as f:
            lookup = json.load(f)

        with open(LLM_FILE, "a") as out:
            for text_id, scores in data["scores"].items():
                if text_id not in lookup:
                    errors += 1
                    continue
                info = lookup[text_id]
                dims = {}
                for d in DIMS:
                    dims[d] = {"score": 5, "confidence": 0.1}
                # Apply sparse scores
                for abbrev, val in scores.items():
                    dim_name = DIM_ABBREV.get(abbrev)
                    if dim_name:
                        if isinstance(val, list):
                            dims[dim_name] = {"score": max(0, min(10, val[0])),
                                              "confidence": max(0.0, min(1.0, val[1]))}
                        else:
                            conf = 0.6 if val != 5 else 0.2
                            dims[dim_name] = {"score": max(0, min(10, val)),
                                              "confidence": conf}

                out_rec = {
                    "text": info["text"],
                    "source": info.get("source", "claude_code"),
                    "teacher": "llm",
                    "dimensions": dims
                }
                out.write(json.dumps(out_rec) + "\n")
                count += 1
    else:
        # Full format (list of records)
        records = data if isinstance(data, list) else [data]
        with open(LLM_FILE, "a") as out:
            for rec in records:
                if "text" not in rec or "dimensions" not in rec:
                    errors += 1
                    continue
                dims = rec["dimensions"]
                for d in DIMS:
                    if d not in dims:
                        dims[d] = {"score": 5, "confidence": 0.1}
                for d in dims:
                    dims[d]["score"] = max(0, min(10, dims[d]["score"]))
                    dims[d]["confidence"] = max(0.0, min(1.0, dims[d]["confidence"]))

                out_rec = {
                    "text": rec["text"],
                    "source": rec.get("source", "claude_code"),
                    "teacher": "llm",
                    "dimensions": dims
                }
                out.write(json.dumps(out_rec) + "\n")
                count += 1

    print(f"Appended {count} records to {LLM_FILE} ({errors} errors)")
    print(f"Total in train-llm.jsonl: {count + len(load_labeled_keys())}")


def cmd_stats(args):
    labeled = load_labeled_keys()
    unlabeled = load_unlabeled(labeled)
    by_source = {}
    for t in unlabeled:
        by_source[t["source"]] = by_source.get(t["source"], 0) + 1

    print(f"Pool: {'unlabeled-pool.jsonl' if POOL_FILE.exists() else 'composite (fallback)'}")
    print(f"Already labeled: {len(labeled)}")
    print(f"Available to label: {len(unlabeled)}")
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")

    # How many 50-text batches
    bs = 50
    total_batches = (len(unlabeled) + bs - 1) // bs
    print(f"\nAt batch_size={bs}: {total_batches} batches needed for all")
    print(f"For 3500 texts: {(3500 + bs - 1) // bs} batches")


def cmd_append_synthetic(args):
    """Append synthetic (generated) texts with scores to train-llm.jsonl.

    Input format: {"texts": [{"text":"...", "scores":{"te":[s,c], ...}}, ...]}
    Same compact scoring as regular append, but text is included inline.
    """
    DIM_ABBREV = {
        "te": "threat_exposure", "hi": "hostility_index",
        "ad": "authority_dynamics", "ed": "energy_dissipation",
        "rc": "regulatory_capacity", "rb": "resilience_baseline",
        "tc": "trust_conditions", "cc": "cooling_capacity",
        "da": "defensive_architecture", "co": "contractual_clarity"
    }

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        return

    with open(input_path) as f:
        data = json.load(f)

    items = data.get("texts", [])
    count = 0

    with open(LLM_FILE, "a") as out:
        for item in items:
            text = item.get("text", "").strip()
            if not text:
                continue
            scores = item.get("scores", {})
            dims = {}
            for d in DIMS:
                dims[d] = {"score": 5, "confidence": 0.1}
            for abbrev, val in scores.items():
                dim_name = DIM_ABBREV.get(abbrev)
                if dim_name:
                    if isinstance(val, list):
                        dims[dim_name] = {"score": max(0, min(10, val[0])),
                                          "confidence": max(0.0, min(1.0, val[1]))}
                    else:
                        conf = 0.6 if val != 5 else 0.2
                        dims[dim_name] = {"score": max(0, min(10, val)),
                                          "confidence": conf}

            out_rec = {
                "text": text,
                "source": "synthetic",
                "teacher": "llm",
                "dimensions": dims
            }
            out.write(json.dumps(out_rec) + "\n")
            count += 1

    print(f"Appended {count} synthetic records to {LLM_FILE}")
    total = sum(1 for l in open(LLM_FILE) if l.strip())
    print(f"Total in train-llm.jsonl: {total}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--batch-size", type=int, default=50)
    p_ext.add_argument("--batch-num", type=int, default=1)

    p_app = sub.add_parser("append")
    p_app.add_argument("--input", required=True)

    p_syn = sub.add_parser("append-synthetic")
    p_syn.add_argument("--input", required=True)

    sub.add_parser("stats")

    args = parser.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "append":
        cmd_append(args)
    elif args.cmd == "append-synthetic":
        cmd_append_synthetic(args)
    elif args.cmd == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
