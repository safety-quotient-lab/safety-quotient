#!/usr/bin/env python3
"""Build a pool of unlabeled texts from raw datasets.

Only includes texts NOT already in composite-ground-truth.jsonl.
This ensures LLM labels are always on unique texts, avoiding
conflicting duplicate training signals.

Usage:
    python scripts/build_unlabeled_pool.py [--min-length 40] [--max-per-source 5000]
"""

import json
import csv
import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
COMPOSITE = DATA / "composite-ground-truth.jsonl"
POOL_FILE = DATA / "unlabeled-pool.jsonl"


def load_composite_texts():
    texts = set()
    with open(COMPOSITE) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.add(json.loads(line)["text"])
    return texts


def load_berkeley(exclude, min_len):
    path = DATA / "measuring-hate-speech.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    texts = []
    for t in df["text"].unique():
        if len(t) >= min_len and t not in exclude:
            texts.append({"text": t, "source": "berkeley"})
    return texts


def load_dreaddit(exclude, min_len):
    texts = []
    ddir = DATA / "dreaddit"
    if not ddir.exists():
        return texts
    for fn in ddir.glob("*.csv"):
        with open(fn) as f:
            for row in csv.DictReader(f):
                t = row.get("text", "")
                if len(t) >= min_len and t not in exclude:
                    texts.append({"text": t, "source": "dreaddit"})
                    exclude.add(t)  # dedup within source
    return texts


def load_esconv(exclude, min_len):
    path = DATA / "esconv-train.jsonl"
    if not path.exists():
        return []
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("text", "")
            if len(t) >= min_len and t not in exclude:
                texts.append({"text": t, "source": "esconv"})
                exclude.add(t)
    return texts


def load_empathetic_dialogues(exclude, min_len):
    edir = DATA / "empatheticdialogues"
    if not edir.exists():
        return []
    texts = []
    for fn in edir.glob("*.csv"):
        with open(fn) as f:
            for row in csv.DictReader(f):
                t = row.get("utterance", "")
                if len(t) >= min_len and t not in exclude:
                    texts.append({"text": t, "source": "empathetic_dialogues"})
                    exclude.add(t)
    return texts


def load_prosocial(exclude, min_len):
    pdir = DATA / "prosocial"
    if not pdir.exists():
        return []
    texts = []
    for fn in pdir.glob("*.jsonl"):
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                # prosocial has 'text' or 'response' fields
                for field in ("text", "response"):
                    t = r.get(field, "")
                    if len(t) >= min_len and t not in exclude:
                        texts.append({"text": t, "source": "prosocial"})
                        exclude.add(t)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Build unlabeled text pool")
    parser.add_argument("--min-length", type=int, default=40,
                        help="Minimum text length in characters (default: 40)")
    parser.add_argument("--max-per-source", type=int, default=5000,
                        help="Max texts per source dataset (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading composite texts to exclude...")
    exclude = load_composite_texts()

    # Also exclude texts already in train-llm.jsonl
    llm_file = DATA / "train-llm.jsonl"
    if llm_file.exists():
        with open(llm_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    exclude.add(json.loads(line)["text"])

    print(f"  Excluding {len(exclude)} known texts")

    loaders = [
        ("berkeley", load_berkeley),
        ("dreaddit", load_dreaddit),
        ("esconv", load_esconv),
        ("empathetic_dialogues", load_empathetic_dialogues),
        ("prosocial", load_prosocial),
    ]

    all_texts = []
    for name, loader in loaders:
        print(f"Loading {name}...")
        texts = loader(exclude, args.min_length)
        # Cap per source
        if len(texts) > args.max_per_source:
            random.seed(args.seed)
            random.shuffle(texts)
            texts = texts[:args.max_per_source]
        all_texts.extend(texts)
        # Add to exclude set so no cross-source duplicates
        for t in texts:
            exclude.add(t["text"])
        print(f"  {name}: {len(texts)} texts")

    # Shuffle deterministically
    random.seed(args.seed)
    random.shuffle(all_texts)

    # Write pool
    with open(POOL_FILE, "w") as f:
        for t in all_texts:
            f.write(json.dumps(t) + "\n")

    print(f"\nWrote {len(all_texts)} texts to {POOL_FILE}")
    by_source = {}
    for t in all_texts:
        by_source[t["source"]] = by_source.get(t["source"], 0) + 1
    for src, n in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {src}: {n}")


if __name__ == "__main__":
    main()
