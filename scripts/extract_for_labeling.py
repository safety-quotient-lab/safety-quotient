#!/usr/bin/env python3
"""Extract unlabeled texts from composite for Claude Code batch labeling.

Outputs batches of texts as JSON that Claude Code can evaluate directly
in-conversation, then a companion script appends results to train-llm.jsonl.

Usage:
    python scripts/extract_for_labeling.py --batch-size 20 --batch-num 1
    python scripts/extract_for_labeling.py --batch-size 20 --batch-num 2
    python scripts/extract_for_labeling.py --stats   # show what's needed
"""

import json
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
COMPOSITE = ROOT / "data" / "composite-ground-truth.jsonl"
LLM_FILE = ROOT / "data" / "train-llm.jsonl"


def load_labeled_keys():
    """Load text keys already in train-llm.jsonl."""
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
    """Load texts from composite that aren't yet LLM-labeled."""
    texts = []
    with open(COMPOSITE) as f:
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
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--batch-num", type=int, default=1)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labeled_keys = load_labeled_keys()
    unlabeled = load_unlabeled(labeled_keys)

    if args.stats:
        print(f"Already labeled: {len(labeled_keys)}")
        print(f"Unlabeled available: {len(unlabeled)}")
        by_source = {}
        for t in unlabeled:
            by_source[t["source"]] = by_source.get(t["source"], 0) + 1
        for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"  {src}: {cnt}")
        return

    # Deterministic shuffle so batch-num is reproducible
    random.seed(args.seed)
    random.shuffle(unlabeled)

    start = (args.batch_num - 1) * args.batch_size
    end = start + args.batch_size
    batch = unlabeled[start:end]

    total_batches = (len(unlabeled) + args.batch_size - 1) // args.batch_size
    print(f"Batch {args.batch_num}/{total_batches} ({len(batch)} texts)")
    print(f"---")

    for i, item in enumerate(batch):
        print(f"\n### Text {start + i + 1} [source: {item['source']}]")
        print(item["text"][:500])
        if len(item["text"]) > 500:
            print(f"... ({len(item['text'])} chars total)")


if __name__ == "__main__":
    main()
