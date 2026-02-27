"""
Assemble held-out test set from labeled batches A and B.

Maps abbreviated dimension keys to full names, validates format,
and writes data/held-out-test.jsonl for use by eval_held_out.py.

Usage:
    python scripts/assemble_held_out.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Map abbreviated keys to full dimension names
SHORT_TO_FULL = {
    "te": "threat_exposure",
    "hi": "hostility_index",
    "ad": "authority_dynamics",
    "ed": "energy_dissipation",
    "rc": "regulatory_capacity",
    "rb": "resilience_baseline",
    "tc": "trust_conditions",
    "cc": "cooling_capacity",
    "da": "defensive_architecture",
    "co": "contractual_clarity",
}

FULL_NAMES = set(SHORT_TO_FULL.values())

BATCH_FILES = [
    Path("/tmp/held_out_labeled_a.json"),
    Path("/tmp/held_out_labeled_b.json"),
]
OUTPUT = ROOT / "data" / "held-out-test.jsonl"


def normalize_dimensions(dims):
    """Map abbreviated or full dimension keys to canonical full names."""
    out = {}
    for key, val in dims.items():
        full_key = SHORT_TO_FULL.get(key, key)
        if full_key not in FULL_NAMES:
            print(f"  WARNING: unknown dimension key '{key}' → '{full_key}', skipping")
            continue
        if isinstance(val, dict) and "score" in val:
            out[full_key] = val
        else:
            print(f"  WARNING: dimension '{key}' has unexpected format: {val}")
    return out


def main():
    all_records = []

    for batch_file in BATCH_FILES:
        if not batch_file.exists():
            print(f"ERROR: {batch_file} not found")
            sys.exit(1)

        with open(batch_file) as f:
            data = json.load(f)

        print(f"Loaded {len(data)} records from {batch_file.name}")

        for rec in data:
            text = rec.get("text", "")
            dims = rec.get("dimensions", {})

            if not text or not dims:
                print(f"  WARNING: record id={rec.get('id')} has empty text or dims, skipping")
                continue

            normalized = normalize_dimensions(dims)
            if not normalized:
                print(f"  WARNING: record id={rec.get('id')} has no valid dimensions after normalization")
                continue

            all_records.append({
                "text": text,
                "dimensions": normalized,
                "source": rec.get("source", "held_out"),
            })

    # Check for duplicates by text
    texts = [r["text"] for r in all_records]
    unique = set(texts)
    if len(unique) < len(texts):
        print(f"\nWARNING: {len(texts) - len(unique)} duplicate texts found")

    # Summary stats
    from collections import Counter
    dim_counts = Counter()
    for r in all_records:
        for d in r["dimensions"]:
            dim_counts[d] += 1

    print(f"\nAssembled {len(all_records)} held-out records")
    print(f"\nPer-dimension coverage:")
    for dim in sorted(FULL_NAMES):
        count = dim_counts.get(dim, 0)
        print(f"  {dim:<25} {count:>3}/{len(all_records)}")

    # Write JSONL
    with open(OUTPUT, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWritten to {OUTPUT}")


if __name__ == "__main__":
    main()
