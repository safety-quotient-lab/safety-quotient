#!/usr/bin/env python3
"""Select non-overlapping text sets for scoring experiments.

Usage:
  python scripts/select_experiment_texts.py --db data/psq.db \
    --pool data/unlabeled-pool.jsonl \
    --out-dir /tmp/psq_experiments

Creates:
  exp1_texts.jsonl  (30 texts for Experiment 1: Halo-Awareness)
  exp2_texts.jsonl  (30 texts for Experiment 2: Dissimilar Rubrics)
  exp3_texts.jsonl  (20 texts for Experiment 3: Scale Format)

All text sets are non-overlapping and exclude any text already in the DB.
Selection is deterministic (hash-based) and stratified by source.
"""

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def text_hash(text):
    """Deterministic hash for text assignment."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def load_db_texts(db_path):
    """Get all text hashes already in the database."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT text FROM texts").fetchall()
    conn.close()
    return {r[0] for r in rows}


def load_pool(pool_path, exclude_texts):
    """Load unlabeled pool, excluding texts already in DB."""
    records = []
    with open(pool_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["text"] not in exclude_texts:
                records.append(rec)
    return records


def stratified_select(records, n, seed_offset=0):
    """Select n records stratified by source, deterministic via hash."""
    by_source = defaultdict(list)
    for rec in records:
        by_source[rec.get("source", "unknown")].append(rec)

    # Sort each source's records by hash for determinism
    for src in by_source:
        by_source[src].sort(key=lambda r: text_hash(r["text"]) + seed_offset)

    sources = sorted(by_source.keys())
    n_sources = len(sources)
    per_source = n // n_sources
    remainder = n % n_sources

    selected = []
    for i, src in enumerate(sources):
        take = per_source + (1 if i < remainder else 0)
        selected.extend(by_source[src][:take])

    return selected[:n]


def main():
    parser = argparse.ArgumentParser(description="Select experiment text sets")
    parser.add_argument("--db", default="data/psq.db", help="SQLite database path")
    parser.add_argument("--pool", default="data/unlabeled-pool.jsonl",
                        help="Unlabeled pool JSONL")
    parser.add_argument("--out-dir", default="/tmp/psq_experiments",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DB texts to exclude
    print(f"Loading DB texts from {args.db}...")
    db_texts = load_db_texts(args.db)
    print(f"  {len(db_texts)} texts in DB (excluded)")

    # Load pool
    print(f"Loading pool from {args.pool}...")
    pool = load_pool(args.pool, db_texts)
    print(f"  {len(pool)} available texts")

    sources = defaultdict(int)
    for r in pool:
        sources[r.get("source", "unknown")] += 1
    print(f"  Sources: {dict(sources)}")

    if len(pool) < 80:
        print(f"ERROR: Need 80 texts (30+30+20), only {len(pool)} available")
        return

    # Select non-overlapping sets using different hash offsets
    # Offset ensures different texts are selected for each experiment
    set1 = stratified_select(pool, 30, seed_offset=0)
    set1_texts = {r["text"] for r in set1}

    remaining = [r for r in pool if r["text"] not in set1_texts]
    set2 = stratified_select(remaining, 30, seed_offset=1000)
    set2_texts = {r["text"] for r in set2}

    remaining2 = [r for r in remaining if r["text"] not in set2_texts]
    set3 = stratified_select(remaining2, 20, seed_offset=2000)

    # Verify no overlap
    all_texts = [r["text"] for r in set1 + set2 + set3]
    assert len(all_texts) == len(set(all_texts)), "Overlap detected!"

    # Write output files
    for name, records in [("exp1_texts.jsonl", set1),
                          ("exp2_texts.jsonl", set2),
                          ("exp3_texts.jsonl", set3)]:
        path = out_dir / name
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        src_counts = defaultdict(int)
        for r in records:
            src_counts[r.get("source", "unknown")] += 1
        print(f"\n{name}: {len(records)} texts")
        print(f"  Sources: {dict(src_counts)}")
        print(f"  Written to {path}")

    print(f"\nAll sets written to {out_dir}")
    print(f"No overlap: {len(set(all_texts))} unique texts across {len(all_texts)} total")


if __name__ == "__main__":
    main()
