"""
Ingest all v10/v11 data improvements into train-llm.jsonl.

Handles:
1. Synthetic batches: ad_8, te_2, ed_2, da_2
2. Relabeled composite texts (250 each for threat, energy, regulatory, defensive)

All relabeled texts get added to train-llm.jsonl. The dedup logic in distill.py
will automatically replace composite versions with these LLM versions at 5x weight.

Usage:
    source venv/bin/activate
    python scripts/ingest_v10_data.py
    # Then launch training:
    python scripts/distill.py --epochs 10 --conf-mode two-phase --patience 3
"""

import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
LLM_FILE = ROOT / "data" / "train-llm.jsonl"

SYNTHETIC_FILES = [
    ("/tmp/psq_synthetic_ad_8.json", "synthetic", "ad_8"),
    ("/tmp/psq_synthetic_te_2.json", "synthetic", "te_2"),
    ("/tmp/psq_synthetic_ed_2.json", "synthetic", "ed_2"),
    ("/tmp/psq_synthetic_da_2.json", "synthetic", "da_2"),
]

RELABEL_FILES = [
    ("/tmp/psq_relabeled_thre.json", "relabeled", "relabel_thre"),
    ("/tmp/psq_relabeled_ener.json", "relabeled", "relabel_ener"),
    ("/tmp/psq_relabeled_regu.json", "relabeled", "relabel_regu"),
    ("/tmp/psq_relabeled_defe.json", "relabeled", "relabel_defe"),
]

DIM_FULL_NAMES = {
    "te": "threat_exposure", "hi": "hostility_index", "ad": "authority_dynamics",
    "ed": "energy_dissipation", "rc": "regulatory_capacity", "rb": "resilience_baseline",
    "tc": "trust_conditions", "cc": "cooling_capacity", "da": "defensive_architecture",
    "co": "contractual_clarity",
}


def normalize_dims(dims):
    """Map abbreviated dimension keys to full names."""
    out = {}
    for k, v in dims.items():
        full = DIM_FULL_NAMES.get(k, k)
        out[full] = v
    return out


def load_synthetic(path, source, batch):
    """Load synthetic file (JSON array format)."""
    with open(path) as f:
        data = json.load(f)

    records = []
    # Handle both flat array and {"texts": [...]} format
    items = data if isinstance(data, list) else data.get("texts", [])

    for item in items:
        text = item.get("text", "")
        if not text:
            continue

        # Handle both {"dimensions": {...}} and {"scores": {...}} formats
        dims = item.get("dimensions", {})
        if not dims and "scores" in item:
            # Old format: {"scores": {"ad": [score, conf], ...}}
            dims = {}
            for k, v in item["scores"].items():
                full = DIM_FULL_NAMES.get(k, k)
                if isinstance(v, list) and len(v) == 2:
                    dims[full] = {"score": v[0], "confidence": v[1]}
                elif isinstance(v, dict):
                    dims[full] = v
        else:
            dims = normalize_dims(dims)

        records.append({
            "text": text,
            "dimensions": dims,
            "source": source,
            "batch": batch,
        })

    return records


def load_relabeled(path, source, batch):
    """Load relabeled file (JSON array of {id, text, dimensions})."""
    with open(path) as f:
        data = json.load(f)

    records = []
    items = data if isinstance(data, list) else data.get("texts", [])

    for item in items:
        text = item.get("text", "")
        if not text:
            continue

        dims = normalize_dims(item.get("dimensions", {}))
        if not dims:
            continue

        records.append({
            "text": text,
            "dimensions": dims,
            "source": source,
            "batch": batch,
        })

    return records


def main():
    # Load existing LLM data
    with open(LLM_FILE) as f:
        existing = [json.loads(l) for l in f if l.strip()]
    print(f"Existing LLM records: {len(existing)}")

    # Ingest all files
    added = 0
    for path, source, batch in SYNTHETIC_FILES + RELABEL_FILES:
        p = Path(path)
        if not p.exists():
            print(f"  SKIP {p.name} (not found)")
            continue

        if source == "synthetic":
            records = load_synthetic(path, source, batch)
        else:
            records = load_relabeled(path, source, batch)

        existing.extend(records)
        added += len(records)
        print(f"  ADD  {p.name}: {len(records)} records ({source}/{batch})")

    if added == 0:
        print("\nNo new files to ingest.")
        sys.exit(0)

    # Dedup by text (keep last occurrence = newest)
    seen = {}
    for rec in existing:
        seen[rec["text"]] = rec
    deduped = list(seen.values())
    removed = len(existing) - len(deduped)

    # Write back
    with open(LLM_FILE, "w") as f:
        for rec in deduped:
            f.write(json.dumps(rec) + "\n")

    # Summary
    sources = Counter(r.get("source", "?") for r in deduped)
    print(f"\n=== Summary ===")
    print(f"  Added: {added}")
    print(f"  Deduped: {removed}")
    print(f"  Final: {len(deduped)} records")
    print(f"  By source: {dict(sources)}")

    # Per-dimension audit
    print(f"\n=== Dimension coverage (LLM records only) ===")
    dim_counts = Counter()
    for r in deduped:
        for d in r.get("dimensions", {}):
            dim_counts[d] += 1
    for d in sorted(dim_counts.keys()):
        print(f"  {d:<25} {dim_counts[d]:>5}")


if __name__ == "__main__":
    main()
