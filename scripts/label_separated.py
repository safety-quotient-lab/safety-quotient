#!/usr/bin/env python3
"""Separated scoring workflow — one dimension at a time, scored by Claude Code.

Eliminates halo effect by preventing cross-dimension anchoring.
Each dimension pass is scored in a separate conversation context.

Workflow:
    1. extract  — prepare per-dimension batch files from input JSONL
    2. (score each dimension batch in Claude Code conversation)
    3. ingest   — import a scored dimension batch
    4. assemble — merge all 10 dimension scores into final output JSONL
    5. status   — show which dimensions are done
    6. validate — compare joint vs separated inter-dimension correlations

Usage:
    # Step 1: Extract batches (one file per dimension)
    python scripts/label_separated.py extract --input data/held-out-test.jsonl

    # Step 2: Score each dimension in Claude Code (read the batch, score, save)
    # Claude Code reads /tmp/psq_sep_threat_exposure.json, scores, writes scores file

    # Step 3: Ingest scored dimensions
    python scripts/label_separated.py ingest --dim threat_exposure --scores /tmp/psq_sep_threat_exposure_scored.json

    # Step 4: When all 10 done, assemble final output
    python scripts/label_separated.py assemble --input data/held-out-test.jsonl --output data/held-out-test-separated.jsonl

    # Check progress
    python scripts/label_separated.py status
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTRUMENTS_PATH = ROOT / "instruments.json"
WORK_DIR = Path("/tmp/psq_separated")

DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]

DIM_ABBREV = {
    "te": "threat_exposure", "hi": "hostility_index",
    "ad": "authority_dynamics", "ed": "energy_dissipation",
    "rc": "regulatory_capacity", "rb": "resilience_baseline",
    "tc": "trust_conditions", "cc": "cooling_capacity",
    "da": "defensive_architecture", "co": "contractual_clarity",
}
ABBREV_DIM = {v: k for k, v in DIM_ABBREV.items()}


def load_instruments():
    with open(INSTRUMENTS_PATH) as f:
        return json.load(f)


def truncate_text(text, max_len=200):
    """Truncate for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def cmd_extract(args):
    """Extract per-dimension batch files from input JSONL."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    with open(input_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    if args.limit > 0:
        records = records[:args.limit]

    instruments = load_instruments()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Save lookup (id → text + source) for all records
    lookup = {}
    for i, rec in enumerate(records):
        lookup[str(i)] = {"text": rec["text"], "source": rec.get("source", "unknown")}

    lookup_path = WORK_DIR / "lookup.json"
    with open(lookup_path, "w") as f:
        json.dump(lookup, f)

    # Extract per-dimension batch files
    dims_to_extract = [args.dim] if args.dim else DIMS

    for dim_id in dims_to_extract:
        dim_def = instruments["dimensions"][dim_id]
        scoring = dim_def["scoring"]
        abbrev = ABBREV_DIM[dim_id]

        batch = {
            "dimension": dim_id,
            "abbreviation": abbrev,
            "description": dim_def["description"],
            "scoring_rubric": {
                "0": scoring["0"],
                "2": scoring["2"],
                "5": scoring["5"],
                "8": scoring["8"],
                "10": scoring["10"],
            },
            "count": len(records),
            "texts": [
                {"id": i, "source": rec.get("source", "unknown"),
                 "text": rec["text"]}
                for i, rec in enumerate(records)
            ],
            "scoring_instructions": (
                f"Score each text on {dim_def['name']} only (0-10 scale). "
                f"5 = neutral (no signal). Use compact format:\n"
                f'{{"dim": "{abbrev}", "scores": {{"0": [score, conf], "1": [score, conf], ...}}}}'
            ),
        }

        batch_path = WORK_DIR / f"{dim_id}.json"
        with open(batch_path, "w") as f:
            json.dump(batch, f, indent=1)

        print(f"  {dim_id} ({abbrev}): {len(records)} texts → {batch_path}")

    print(f"\nLookup: {lookup_path}")
    print(f"Total: {len(records)} texts × {len(dims_to_extract)} dimensions")
    print(f"\nNext: Score each dimension in Claude Code, then run 'ingest' for each.")


def cmd_ingest(args):
    """Ingest a scored dimension batch.

    Accepts two formats:
    1. File: --scores /path/to/scored.json
       Format: {"dim": "te", "scores": {"0": [3, 0.8], "1": [5, 0.4], ...}}
    2. Stdin (pipe): same JSON format

    Scores are saved to WORK_DIR/{dimension}_scores.json
    """
    dim_id = args.dim
    if dim_id in DIM_ABBREV:
        dim_id = DIM_ABBREV[dim_id]

    if dim_id not in DIMS:
        print(f"ERROR: Unknown dimension: {dim_id}")
        print(f"Valid: {', '.join(DIMS)}")
        print(f"Abbrevs: {', '.join(DIM_ABBREV.keys())}")
        sys.exit(1)

    # Load scores
    if args.scores:
        scores_path = Path(args.scores)
        if not scores_path.exists():
            print(f"ERROR: {scores_path} not found")
            sys.exit(1)
        with open(scores_path) as f:
            data = json.load(f)
    else:
        print("Reading scores from stdin...")
        data = json.load(sys.stdin)

    # Extract scores dict
    if "scores" in data:
        scores = data["scores"]
    else:
        scores = data

    # Validate and normalize
    normalized = {}
    for text_id, val in scores.items():
        if isinstance(val, list):
            score = max(0, min(10, float(val[0])))
            conf = max(0.0, min(1.0, float(val[1])))
        elif isinstance(val, (int, float)):
            score = max(0, min(10, float(val)))
            conf = 0.6 if val != 5 else 0.2
        else:
            print(f"  WARNING: skipping {text_id} — unexpected format: {val}")
            continue
        normalized[text_id] = {"score": score, "confidence": conf}

    # Save to work dir
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WORK_DIR / f"{dim_id}_scores.json"
    with open(out_path, "w") as f:
        json.dump(normalized, f, indent=1)

    print(f"Ingested {len(normalized)} scores for {dim_id} → {out_path}")


def cmd_assemble(args):
    """Assemble all 10 dimension scores into final output JSONL."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    with open(input_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    if args.limit > 0:
        records = records[:args.limit]

    # Load all dimension scores
    dim_scores = {}
    missing_dims = []
    for dim_id in DIMS:
        scores_path = WORK_DIR / f"{dim_id}_scores.json"
        if scores_path.exists():
            with open(scores_path) as f:
                dim_scores[dim_id] = json.load(f)
        else:
            missing_dims.append(dim_id)

    if missing_dims:
        print(f"WARNING: Missing dimensions: {', '.join(missing_dims)}")
        if not args.partial:
            print("Use --partial to assemble with available dimensions only.")
            print("Missing dimensions will get score=5, confidence=0.1")
            sys.exit(1)

    # Assemble
    output_path = Path(args.output)
    count = 0
    with open(output_path, "w") as out:
        for i, rec in enumerate(records):
            text_id = str(i)
            dimensions = {}

            for dim_id in DIMS:
                if dim_id in dim_scores and text_id in dim_scores[dim_id]:
                    dimensions[dim_id] = dim_scores[dim_id][text_id]
                else:
                    dimensions[dim_id] = {"score": 5, "confidence": 0.1}

            out_rec = {
                "text": rec["text"],
                "source": rec.get("source", "unknown"),
                "teacher": "separated-llm",
                "dimensions": dimensions,
            }
            # Preserve extra fields
            for k in rec:
                if k not in ("text", "source", "teacher", "dimensions"):
                    out_rec[k] = rec[k]

            out.write(json.dumps(out_rec) + "\n")
            count += 1

    print(f"Assembled {count} records → {output_path}")
    print(f"Dimensions included: {len(DIMS) - len(missing_dims)}/10")
    if missing_dims:
        print(f"Missing (defaulted to 5/0.1): {', '.join(missing_dims)}")


def cmd_status(args):
    """Show scoring progress."""
    if not WORK_DIR.exists():
        print("No work directory found. Run 'extract' first.")
        return

    lookup_path = WORK_DIR / "lookup.json"
    n_texts = 0
    if lookup_path.exists():
        with open(lookup_path) as f:
            n_texts = len(json.load(f))

    print(f"Work dir: {WORK_DIR}")
    print(f"Texts: {n_texts}")
    print()
    print(f"{'Dimension':<25} {'Abbrev':>6}  {'Status':<10} {'Scored':>7}")
    print("-" * 55)

    done = 0
    for dim_id in DIMS:
        abbrev = ABBREV_DIM[dim_id]
        batch_path = WORK_DIR / f"{dim_id}.json"
        scores_path = WORK_DIR / f"{dim_id}_scores.json"

        extracted = batch_path.exists()
        scored = scores_path.exists()
        n_scored = 0
        if scored:
            with open(scores_path) as f:
                n_scored = len(json.load(f))
            done += 1

        status = "done" if scored else ("extracted" if extracted else "—")
        count_str = f"{n_scored}/{n_texts}" if scored else ""
        print(f"  {dim_id:<25} {abbrev:>4}    {status:<10} {count_str:>7}")

    print(f"\nProgress: {done}/10 dimensions scored")
    if done == 10:
        print("\nAll dimensions scored! Run 'assemble' to create final output.")


def _load_score_vectors(path):
    """Load JSONL and return per-pair score lists for texts that have both dims in a pair.

    Returns ({(dim_a, dim_b): ([scores_a], [scores_b])}, n_records, dim_coverage).
    Unlike requiring all 10 dims, this uses all available pairwise data.
    """
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Collect per-record scores (only present dims)
    record_scores = []
    dim_counts = {d: 0 for d in DIMS}
    for rec in records:
        dims = rec.get("dimensions", {})
        scores = {}
        for d in DIMS:
            entry = dims.get(d, {})
            s = entry.get("score")
            if s is not None:
                scores[d] = float(s)
                dim_counts[d] += 1
        record_scores.append(scores)

    # Build pairwise vectors
    pair_vectors = {}
    for a, b in combinations(DIMS, 2):
        xs, ys = [], []
        for scores in record_scores:
            if a in scores and b in scores:
                xs.append(scores[a])
                ys.append(scores[b])
        pair_vectors[(a, b)] = (xs, ys)

    return pair_vectors, len(records), dim_counts


def _is_nan(x):
    return x != x  # NaN != NaN


def _pearson(xs, ys):
    """Pearson r without numpy dependency."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def _corr_matrix(pair_vectors):
    """Compute pairwise Pearson r from pair_vectors dict."""
    matrix = {}
    for (a, b), (xs, ys) in pair_vectors.items():
        r = _pearson(xs, ys)
        matrix[(a, b)] = r
        matrix[(b, a)] = r
    return matrix


# Cluster membership for within/between classification
CLUSTER_MEMBERS = {
    "interpersonal_climate": {"authority_dynamics", "contractual_clarity",
                              "trust_conditions", "threat_exposure"},
    "internal_resources": {"regulatory_capacity", "resilience_baseline",
                           "defensive_architecture"},
    "bridge": {"cooling_capacity", "energy_dissipation", "hostility_index"},
}


def _pair_cluster_type(a, b):
    """Classify a dimension pair as within-cluster, between-cluster, or bridge."""
    for cluster, members in CLUSTER_MEMBERS.items():
        if cluster == "bridge":
            continue
        if a in members and b in members:
            return "within"
    # Check if both are bridge
    if a in CLUSTER_MEMBERS["bridge"] and b in CLUSTER_MEMBERS["bridge"]:
        return "within"
    return "between"


def cmd_validate(args):
    """Compare inter-dimension correlations between joint and separated scoring.

    Checks:
    1. Mean off-diagonal |r| — separated should be lower
    2. Within-cluster vs between-cluster correlations
    3. Discriminant ratio (within / between) — should be higher for separated
    4. Per-pair deltas to identify halo pairs vs genuine overlap
    """
    joint_path = Path(args.joint)
    sep_path = Path(args.separated)

    if not joint_path.exists():
        print(f"ERROR: {joint_path} not found")
        sys.exit(1)
    if not sep_path.exists():
        print(f"ERROR: {sep_path} not found")
        sys.exit(1)

    joint_pairs, joint_n, joint_dims = _load_score_vectors(joint_path)
    sep_pairs, sep_n, sep_dims = _load_score_vectors(sep_path)

    print(f"Joint:     {joint_n} records from {joint_path.name}")
    for d in DIMS:
        print(f"  {d}: {joint_dims[d]} records")
    print(f"Separated: {sep_n} records from {sep_path.name}")
    for d in DIMS:
        print(f"  {d}: {sep_dims[d]} records")
    print()

    joint_corr = _corr_matrix(joint_pairs)
    sep_corr = _corr_matrix(sep_pairs)

    # All unique pairs — only include pairs with enough data in BOTH files
    all_pairs = list(combinations(DIMS, 2))  # 45 pairs
    pairs = []
    skipped = 0
    for a, b in all_pairs:
        j_xs, _ = joint_pairs[(a, b)]
        s_xs, _ = sep_pairs[(a, b)]
        if len(j_xs) >= 5 and len(s_xs) >= 5:
            pairs.append((a, b))
        else:
            skipped += 1

    if not pairs:
        print("ERROR: No dimension pairs with enough data (>=5) in both files.")
        sys.exit(1)

    if skipped:
        print(f"Note: {skipped} pairs skipped (insufficient data in joint file)")
        print(f"Comparing {len(pairs)}/{len(all_pairs)} pairs")
        print()

    # 1. Mean off-diagonal |r|
    joint_rs = [abs(joint_corr[(a, b)]) for a, b in pairs
                if not _is_nan(joint_corr.get((a, b), float("nan")))]
    sep_rs = [abs(sep_corr[(a, b)]) for a, b in pairs
              if not _is_nan(sep_corr.get((a, b), float("nan")))]
    joint_mean = sum(joint_rs) / len(joint_rs)
    sep_mean = sum(sep_rs) / len(sep_rs)

    print("=" * 65)
    print("HALO VALIDATION SUMMARY")
    print("=" * 65)
    print()
    print(f"{'Metric':<35} {'Joint':>8} {'Separated':>10} {'Delta':>8}")
    print("-" * 65)
    print(f"{'Mean off-diagonal |r|':<35} {joint_mean:>+8.3f} {sep_mean:>+10.3f} {sep_mean - joint_mean:>+8.3f}")

    # 2. Within-cluster vs between-cluster
    joint_within, joint_between = [], []
    sep_within, sep_between = [], []
    for a, b in pairs:
        jr = joint_corr.get((a, b), float("nan"))
        sr = sep_corr.get((a, b), float("nan"))
        if _is_nan(jr) or _is_nan(sr):
            continue
        ptype = _pair_cluster_type(a, b)
        if ptype == "within":
            joint_within.append(abs(jr))
            sep_within.append(abs(sr))
        else:
            joint_between.append(abs(jr))
            sep_between.append(abs(sr))

    jw = sum(joint_within) / len(joint_within) if joint_within else 0
    jb = sum(joint_between) / len(joint_between) if joint_between else 0
    sw = sum(sep_within) / len(sep_within) if sep_within else 0
    sb = sum(sep_between) / len(sep_between) if sep_between else 0

    print(f"{'Within-cluster mean |r|':<35} {jw:>+8.3f} {sw:>+10.3f} {sw - jw:>+8.3f}")
    print(f"{'Between-cluster mean |r|':<35} {jb:>+8.3f} {sb:>+10.3f} {sb - jb:>+8.3f}")

    # 3. Discriminant ratio
    j_ratio = jw / jb if jb > 0 else float("inf")
    s_ratio = sw / sb if sb > 0 else float("inf")
    print(f"{'Discriminant ratio (within/between)':<35} {j_ratio:>8.2f}x {s_ratio:>9.2f}x {s_ratio - j_ratio:>+8.2f}")
    print()

    # Verdict
    halo_reduced = sep_mean < joint_mean
    discrimination_improved = s_ratio > j_ratio
    between_dropped = sb < jb

    print("VERDICT:")
    print(f"  Halo reduction:        {'PASS' if halo_reduced else 'FAIL'} "
          f"(mean |r| {'decreased' if halo_reduced else 'increased'} by {abs(sep_mean - joint_mean):.3f})")
    print(f"  Discrimination:        {'PASS' if discrimination_improved else 'FAIL'} "
          f"(ratio {'improved' if discrimination_improved else 'worsened'} from {j_ratio:.2f}x to {s_ratio:.2f}x)")
    print(f"  Between-cluster drop:  {'PASS' if between_dropped else 'FAIL'} "
          f"(between |r| {'decreased' if between_dropped else 'increased'} by {abs(sb - jb):.3f})")
    print()

    overall = halo_reduced and discrimination_improved and between_dropped
    print(f"  OVERALL: {'PASS — halo effect successfully reduced' if overall else 'FAIL — separated scoring did not reduce halo'}")
    print()

    # 4. Per-pair detail table
    if not args.brief:
        print("-" * 65)
        print("PER-PAIR DETAIL (sorted by delta)")
        print(f"{'Pair':<40} {'Joint':>7} {'Sep':>7} {'Delta':>7} {'Type':<8} {'n':>4}")
        print("-" * 70)

        pair_data = []
        for a, b in pairs:
            jr = joint_corr.get((a, b), float("nan"))
            sr = sep_corr.get((a, b), float("nan"))
            if _is_nan(jr) or _is_nan(sr):
                continue
            delta = sr - jr
            ptype = _pair_cluster_type(a, b)
            n_joint = len(joint_pairs[(a, b)][0])
            pair_data.append((a, b, jr, sr, delta, ptype, n_joint))

        pair_data.sort(key=lambda x: x[4])

        strong_halo = 0
        genuine = 0
        for a, b, jr, sr, delta, ptype, n_j in pair_data:
            abbr_a = ABBREV_DIM[a]
            abbr_b = ABBREV_DIM[b]
            marker = ""
            if delta < -0.30:
                marker = " <<HALO"
                strong_halo += 1
            elif abs(delta) < 0.10:
                marker = " (genuine)"
                genuine += 1
            print(f"  {abbr_a} x {abbr_b:<32} {jr:>+.3f} {sr:>+.3f} {delta:>+.3f} {ptype:<8} {n_j:>3}{marker}")

        print()
        print(f"Strong halo pairs (delta < -0.30): {strong_halo}")
        print(f"Genuine overlap (|delta| < 0.10): {genuine}")
        print(f"Total pairs: {len(pairs)}")


def main():
    parser = argparse.ArgumentParser(
        description="Separated scoring workflow for Claude Code (halo-free labeling)")
    sub = parser.add_subparsers(dest="cmd")

    p_ext = sub.add_parser("extract", help="Extract per-dimension batch files")
    p_ext.add_argument("--input", required=True, help="Input JSONL file")
    p_ext.add_argument("--dim", default=None, help="Extract single dimension (optional)")
    p_ext.add_argument("--limit", type=int, default=0, help="Max records (0=all)")

    p_ing = sub.add_parser("ingest", help="Import scored dimension results")
    p_ing.add_argument("--dim", required=True, help="Dimension name or abbreviation")
    p_ing.add_argument("--scores", default=None, help="Path to scores JSON (or stdin)")

    p_asm = sub.add_parser("assemble", help="Merge all dimensions into final JSONL")
    p_asm.add_argument("--input", required=True, help="Original input JSONL")
    p_asm.add_argument("--output", required=True, help="Output JSONL file")
    p_asm.add_argument("--limit", type=int, default=0, help="Max records (0=all)")
    p_asm.add_argument("--partial", action="store_true", help="Allow assembly with missing dims")

    sub.add_parser("status", help="Show scoring progress")

    p_val = sub.add_parser("validate", help="Compare joint vs separated inter-dim correlations")
    p_val.add_argument("--joint", required=True, help="Joint-scored JSONL (original)")
    p_val.add_argument("--separated", required=True, help="Separated-scored JSONL")
    p_val.add_argument("--brief", action="store_true", help="Skip per-pair detail table")

    args = parser.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "ingest":
        cmd_ingest(args)
    elif args.cmd == "assemble":
        cmd_assemble(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
