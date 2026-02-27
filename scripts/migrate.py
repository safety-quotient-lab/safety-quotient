#!/usr/bin/env python3
"""
migrate.py — Bootstrap psq.db from existing JSONL files.

Ingests:
  data/composite-ground-truth.jsonl  → composite-proxy + synthetic + relabeled scores
  data/train-llm.jsonl               → joint-llm scores (teacher='llm') + synthetic/relabeled
  data/held-out-test.jsonl           → separated-llm scores (held-out split)
  data/held-out-test-joint.jsonl     → joint-llm scores for same held-out texts
  data/dataset_mappings.json         → dataset_mappings table

Registers:
  model psq-v13 with per-dimension test + held-out results
  calibration psq-v13-cal from models/psq-student/calibration.json
  split assignments from md5(text) % 100 (matching distill.py exactly)

Usage:
  python scripts/migrate.py [--db data/psq.db] [--dry-run]
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS_DIR = ROOT / "models" / "psq-student"

DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def split_from_text(text: str) -> str:
    """Deterministic split assignment matching distill.py's md5 logic."""
    h = int(hashlib.md5(text.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "val"
    else:
        return "test"


def method_scorer_for_cgt(source: str):
    """Map composite-ground-truth.jsonl source → (method, scorer)."""
    if source == "synthetic":
        return "synthetic", "claude-code"
    elif source == "relabeled":
        return "separated-llm", "claude-code"
    else:
        # e.g. 'goemotions' → 'goemotions-proxy'
        return "composite-proxy", f"{source}-proxy"


def method_scorer_for_train_llm(teacher, source: str):
    """Map train-llm.jsonl teacher + source → (method, scorer)."""
    if teacher is None:
        # These are synthetic/relabeled rows that ended up in train-llm.jsonl
        return method_scorer_for_cgt(source)
    if teacher in ("llm", "llm_labeled"):
        return "joint-llm", "claude-code"
    if teacher == "separated-llm":
        return "separated-llm", "claude-code"
    raise ValueError(f"Unknown teacher value: {teacher!r}")


# ---------------------------------------------------------------------------
# Core migration
# ---------------------------------------------------------------------------

def migrate(db_path: Path, dry_run: bool = False):
    schema_path = DATA / "schema.sql"
    if not schema_path.exists():
        sys.exit(f"ERROR: {schema_path} not found")

    if dry_run:
        print("[DRY RUN] Using in-memory database — nothing will be written to disk")
        con = sqlite3.connect(":memory:")
    else:
        if db_path.exists():
            print(f"WARNING: {db_path} already exists. Continuing will ADD data (no overwrites).")
        con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")

    # Create schema (idempotent — wrap each statement in try/except)
    print("Creating schema...")
    schema_sql = schema_path.read_text()
    # Split on semicolons; strip comments from each fragment to get the
    # actual SQL keyword (comment-only blocks are skipped)
    for raw_stmt in schema_sql.split(";"):
        # Remove leading comment lines to reveal the actual SQL
        lines = [l for l in raw_stmt.splitlines() if not l.strip().startswith("--")]
        stmt = "\n".join(lines).strip()
        if not stmt:
            continue
        try:
            con.execute(stmt)
        except sqlite3.OperationalError as e:
            if "already exists" in str(e):
                pass  # idempotent
            else:
                print(f"  WARNING: {e}")

    # -----------------------------------------------------------------------
    # 1. Ingest texts + scores from composite-ground-truth.jsonl
    # -----------------------------------------------------------------------
    cgt_path = DATA / "composite-ground-truth.jsonl"
    if cgt_path.exists():
        print(f"\nIngesting {cgt_path.name}...")
        _ingest_jsonl(con, cgt_path, file_role="cgt")
    else:
        print(f"  SKIP: {cgt_path} not found")

    # -----------------------------------------------------------------------
    # 2. Ingest texts + scores from train-llm.jsonl
    # -----------------------------------------------------------------------
    train_llm_path = DATA / "train-llm.jsonl"
    if train_llm_path.exists():
        print(f"\nIngesting {train_llm_path.name}...")
        _ingest_jsonl(con, train_llm_path, file_role="train_llm")
    else:
        print(f"  SKIP: {train_llm_path} not found")

    # -----------------------------------------------------------------------
    # 3. Ingest held-out-test.jsonl (separated-llm, split=held-out)
    # -----------------------------------------------------------------------
    ho_path = DATA / "held-out-test.jsonl"
    if ho_path.exists():
        print(f"\nIngesting {ho_path.name}...")
        _ingest_jsonl(con, ho_path, file_role="held_out_separated")
    else:
        print(f"  SKIP: {ho_path} not found")

    # -----------------------------------------------------------------------
    # 4. Ingest held-out-test-joint.jsonl (joint-llm, same texts)
    # -----------------------------------------------------------------------
    hoj_path = DATA / "held-out-test-joint.jsonl"
    if hoj_path.exists():
        print(f"\nIngesting {hoj_path.name}...")
        _ingest_jsonl(con, hoj_path, file_role="held_out_joint")
    else:
        print(f"  SKIP: {hoj_path} not found")

    # -----------------------------------------------------------------------
    # 5. Seed dataset_mappings from dataset_mappings.json
    # -----------------------------------------------------------------------
    dm_path = DATA / "dataset_mappings.json"
    if dm_path.exists():
        print(f"\nSeeding dataset_mappings from {dm_path.name}...")
        _seed_dataset_mappings(con, dm_path)
    else:
        print(f"  SKIP: {dm_path} not found")

    # -----------------------------------------------------------------------
    # 6. Seed models table with v13
    # -----------------------------------------------------------------------
    print("\nSeeding models table with psq-v13...")
    _seed_model_v13(con)

    # -----------------------------------------------------------------------
    # 7. Seed calibration from calibration.json
    # -----------------------------------------------------------------------
    cal_path = MODELS_DIR / "calibration.json"
    if cal_path.exists():
        print(f"\nSeeding calibration from {cal_path.name}...")
        _seed_calibration_v13(con, cal_path)
    else:
        print(f"  SKIP: {cal_path} not found")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n--- Migration summary ---")
    for tbl in ("texts", "scores", "splits", "models", "model_results",
                "calibrations", "calibration_thresholds", "dataset_mappings"):
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:30s} {n:>7,}")

    if not dry_run:
        con.commit()
        print(f"\nDatabase written to: {db_path}")
    else:
        print("\n[DRY RUN] complete — no file written")

    con.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _upsert_text(con, text: str, source: str) -> int:
    """Insert text if not exists, return its id."""
    h = text_hash(text)
    row = con.execute("SELECT id FROM texts WHERE text_hash = ?", (h,)).fetchone()
    if row:
        return row[0]
    cur = con.execute(
        "INSERT INTO texts (text, text_hash, source) VALUES (?, ?, ?)",
        (text, h, source),
    )
    return cur.lastrowid


def _upsert_split(con, text_id: int, split: str):
    con.execute(
        "INSERT OR IGNORE INTO splits (text_id, split) VALUES (?, ?)",
        (text_id, split),
    )


def _interface_for_method(method: str) -> str | None:
    """Derive interface from method for historical migration."""
    if method == "composite-proxy":
        return "proxy"
    if method in ("joint-llm", "separated-llm"):
        return "claude-code"  # all historical LLM labels were via Claude Code
    # synthetic: NULL (author-assigned, no LLM interface)
    return None


def _provider_for_method(method: str) -> str | None:
    """Derive provider from method for historical migration."""
    if method in ("joint-llm", "separated-llm", "synthetic"):
        return "anthropic"  # all LLM and synthetic labels are Anthropic/Claude
    # composite-proxy: no LLM vendor
    return None


def _insert_score(con, text_id: int, dimension: str, score: float,
                  confidence: float, method: str, scorer: str,
                  interface=None, provider=None, session_id=None, notes=None):
    if interface is None:
        interface = _interface_for_method(method)
    if provider is None:
        provider = _provider_for_method(method)
    con.execute(
        """INSERT INTO scores
           (text_id, dimension, score, confidence, method, scorer, provider,
            interface, session_id, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (text_id, dimension, score, confidence, method, scorer, provider,
         interface, session_id, notes),
    )


def _ingest_jsonl(con, path: Path, file_role: str):
    """
    file_role controls how teacher/source are interpreted:
      'cgt'               → composite-ground-truth rules
      'train_llm'         → train-llm rules
      'held_out_separated' → separated-llm, split=held-out
      'held_out_joint'    → joint-llm, split=held-out
    """
    n_texts = 0
    n_scores = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec["text"]
            source = rec.get("source", "unknown")
            teacher = rec.get("teacher")
            dims = rec.get("dimensions", {})

            text_id = _upsert_text(con, text, source)
            n_texts += 1

            # Determine method + scorer + interface
            rec_scorer   = rec.get("scorer")    # present in new assembled JSONLs
            rec_iface    = rec.get("interface") # present in new assembled JSONLs
            rec_provider = rec.get("provider")  # present in new assembled JSONLs
            if file_role == "cgt":
                method, scorer = method_scorer_for_cgt(source)
            elif file_role == "train_llm":
                method, scorer = method_scorer_for_train_llm(teacher, source)
            elif file_role == "held_out_separated":
                method, scorer = "separated-llm", rec_scorer or "claude-code"
            elif file_role == "held_out_joint":
                method, scorer = "joint-llm", rec_scorer or "claude-code"
            elif file_role == "new_separated":
                method, scorer = "separated-llm", rec_scorer or "claude-sonnet-4-6"
            else:
                raise ValueError(f"Unknown file_role: {file_role!r}")
            # interface + provider: use record's value if present, else derive
            interface = rec_iface    or _interface_for_method(method)
            provider  = rec_provider or _provider_for_method(method)

            # Assign split
            if file_role in ("held_out_separated", "held_out_joint"):
                _upsert_split(con, text_id, "held-out")
            else:
                _upsert_split(con, text_id, split_from_text(text))

            # Insert scores
            for dim, val in dims.items():
                score = val.get("score")
                confidence = val.get("confidence", 0.5)
                if score is None:
                    continue
                # Skip placeholder scores (confidence=0.1) from partial assembles
                if file_role == "new_separated" and confidence <= 0.15:
                    continue
                _insert_score(con, text_id, dim, score, confidence, method, scorer,
                             interface=interface, provider=provider)
                n_scores += 1

    print(f"  {n_texts:,} texts, {n_scores:,} score observations")


def _seed_dataset_mappings(con, dm_path: Path):
    cfg = json.loads(dm_path.read_text())
    version = cfg.get("version", "v13")
    n = 0
    for dataset_id, dataset_cfg in cfg.get("datasets", {}).items():
        for mapping in dataset_cfg.get("mappings", []):
            dim = mapping.get("dimension")
            if not dim:
                continue
            enabled = 1 if mapping.get("enabled", True) else 0
            mapping_type = mapping.get("type", "unknown")
            disabled_since = mapping.get("disabled_since")
            disabled_reason = mapping.get("disabled_reason")
            config_json = json.dumps(mapping)
            con.execute(
                """INSERT OR REPLACE INTO dataset_mappings
                   (version, dataset_id, dimension, enabled, mapping_type,
                    config_json, disabled_since, disabled_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (version, dataset_id, dim, enabled, mapping_type,
                 config_json, disabled_since, disabled_reason),
            )
            n += 1
    print(f"  {n} mapping rows seeded")


def _seed_model_v13(con):
    # Load config + results
    config = json.loads((MODELS_DIR / "config.json").read_text())
    test_results = json.loads((MODELS_DIR / "test_results.json").read_text())
    held_out_results = json.loads((MODELS_DIR / "held_out_results.json").read_text())

    # Average r across dims for summary fields
    test_rs = [v["r"] for v in test_results.values() if isinstance(v, dict) and "r" in v]
    test_r_avg = round(sum(test_rs) / len(test_rs), 4) if test_rs else None
    held_rs = [v["r"] for v in held_out_results.values() if isinstance(v, dict) and "r" in v]
    held_r_avg = round(sum(held_rs) / len(held_rs), 4) if held_rs else None

    con.execute(
        """INSERT OR IGNORE INTO models
           (id, version, architecture, checkpoint_path, onnx_path, onnx_quant_path,
            config_json, test_r, held_out_r, status, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "psq-v13",
            13,
            config.get("model_name", "distilbert-base-uncased"),
            "models/psq-student/best.pt",
            "models/psq-student/model.onnx",
            "models/psq-student/model_quantized.onnx",
            json.dumps(config),
            test_r_avg,
            held_r_avg,
            "active",
            "Trained on composite-proxy + joint-llm + separated-llm. "
            "Separated labels on held-out set; halo correction applied.",
        ),
    )

    # Per-dimension model_results
    for dim, res in test_results.items():
        if not isinstance(res, dict) or "r" not in res:
            continue
        con.execute(
            """INSERT OR IGNORE INTO model_results
               (model_id, split, dimension, pearson_r, mse, n)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("psq-v13", "test", dim, res.get("r"), res.get("mse"), res.get("n")),
        )

    for dim, res in held_out_results.items():
        if not isinstance(res, dict) or "r" not in res:
            continue
        con.execute(
            """INSERT OR IGNORE INTO model_results
               (model_id, split, dimension, pearson_r, mse, n)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("psq-v13", "held-out", dim, res.get("r"), res.get("mse"), res.get("n")),
        )

    print(f"  psq-v13 registered (test_r={test_r_avg}, held_out_r={held_r_avg})")


def _seed_calibration_v13(con, cal_path: Path):
    cal = json.loads(cal_path.read_text())
    # n_samples: take from the first dimension's 'n' field
    first_dim_data = next(iter(cal.values()))
    n_samples = first_dim_data.get("n", 0)

    con.execute(
        """INSERT OR IGNORE INTO calibrations
           (id, model_id, fitted_on_split, n_samples, notes)
           VALUES (?, ?, ?, ?, ?)""",
        (
            "psq-v13-cal",
            "psq-v13",
            "held-out",
            n_samples,
            "Isotonic regression on score and confidence. "
            "Migrated from models/psq-student/calibration.json.",
        ),
    )

    n = 0
    for dim, data in cal.items():
        score_x = json.dumps(data.get("x_thresholds", []))
        score_y = json.dumps(data.get("y_thresholds", []))
        conf_cc = data.get("confidence_calibration", {})
        conf_x = json.dumps(conf_cc.get("x_thresholds", []))
        conf_y = json.dumps(conf_cc.get("y_thresholds", []))
        con.execute(
            """INSERT OR IGNORE INTO calibration_thresholds
               (calibration_id, dimension, score_x, score_y, conf_x, conf_y)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("psq-v13-cal", dim, score_x, score_y, conf_x, conf_y),
        )
        n += 1

    print(f"  psq-v13-cal registered ({n} dimension thresholds, n_samples={n_samples})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate PSQ JSONL files to SQLite DB")
    parser.add_argument("--db", default="data/psq.db", help="Output database path")
    parser.add_argument("--dry-run", action="store_true", help="Run without committing")
    parser.add_argument("--ingest", metavar="JSONL", help="Ingest a new assembled JSONL (separated-llm) into existing DB")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path

    if args.ingest:
        # Incremental ingest into existing DB
        ingest_path = Path(args.ingest)
        if not ingest_path.is_absolute():
            ingest_path = ROOT / ingest_path
        con = sqlite3.connect(":memory:" if args.dry_run else str(db_path))
        if args.dry_run:
            src = sqlite3.connect(str(db_path))
            src.backup(con)
            src.close()
        print(f"Ingesting {ingest_path.name} into {db_path.name}...")
        _ingest_jsonl(con, ingest_path, "new_separated")
        if not args.dry_run:
            con.commit()
            print("Done.")
        else:
            print("[DRY RUN] complete — no changes written")
        con.close()
    else:
        migrate(db_path, dry_run=args.dry_run)
