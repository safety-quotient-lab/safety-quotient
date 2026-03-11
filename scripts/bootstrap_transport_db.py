#!/usr/bin/env python3
"""bootstrap_transport_db.py — Initialize state.db with transport tables.

Creates state.db in the project root (gitignored) and applies the transport
schema. Idempotent — safe to run repeatedly (uses CREATE TABLE IF NOT EXISTS).

Also indexes existing transport session files into the transport_messages table.

Usage:
    python3 scripts/bootstrap_transport_db.py
    python3 scripts/bootstrap_transport_db.py --force   # drop and recreate
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "state.db"
SCHEMA_PATH = PROJECT_ROOT / "scripts" / "schema_transport.sql"
SESSIONS_DIR = PROJECT_ROOT / "transport" / "sessions"


def create_db(force: bool = False) -> sqlite3.Connection:
    """Create or open state.db and apply schema."""
    if force and DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))

    if SCHEMA_PATH.exists():
        schema_sql = SCHEMA_PATH.read_text()
        conn.executescript(schema_sql)
        print(f"Applied schema from {SCHEMA_PATH}")
    else:
        print(f"WARNING: {SCHEMA_PATH} not found — creating minimal tables",
              file=sys.stderr)

    return conn


def index_existing_messages(conn: sqlite3.Connection) -> int:
    """Index existing transport session files into transport_messages."""
    if not SESSIONS_DIR.exists():
        print("No transport/sessions/ directory found")
        return 0

    count = 0
    for session_dir in sorted(SESSIONS_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        session_name = session_dir.name

        for msg_file in sorted(session_dir.glob("*.json")):
            filename = msg_file.name

            # Skip if already indexed
            existing = conn.execute(
                "SELECT id FROM transport_messages WHERE filename = ?",
                (filename,),
            ).fetchone()
            if existing:
                continue

            try:
                msg = json.loads(msg_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  WARNING: {filename}: {exc}")
                continue

            # Extract fields with fallbacks
            from_block = msg.get("from", {})
            to_block = msg.get("to", {})

            # Handle both dict and string formats
            from_agent = (from_block.get("agent_id", "unknown")
                         if isinstance(from_block, dict) else str(from_block))
            to_agent = (to_block.get("agent_id", "unknown")
                       if isinstance(to_block, dict) else str(to_block))

            conn.execute(
                """INSERT INTO transport_messages
                   (session_name, filename, turn, message_type,
                    from_agent, to_agent, timestamp, subject,
                    claims_count, setl, urgency, processed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_name,
                    filename,
                    msg.get("turn"),
                    msg.get("message_type"),
                    from_agent,
                    to_agent,
                    msg.get("timestamp"),
                    msg.get("content", {}).get("subject", "")
                    if isinstance(msg.get("content"), dict) else "",
                    len(msg.get("claims", [])),
                    msg.get("setl", 0.0),
                    msg.get("urgency", "normal"),
                    1,  # existing messages treated as already processed
                ),
            )
            count += 1

    conn.commit()
    return count


def validate(conn: sqlite3.Connection) -> bool:
    """Run basic validation checks."""
    checks_passed = 0
    checks_total = 0

    checks = [
        ("transport_messages", "transport_messages table exists"),
        ("autonomy_budget", "autonomy_budget table exists"),
        ("autonomous_actions", "autonomous_actions table exists"),
        ("schema_version", "schema_version table exists"),
    ]

    for table, label in checks:
        checks_total += 1
        try:
            conn.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"  ✓  {label}")
            checks_passed += 1
        except sqlite3.OperationalError:
            print(f"  ✗  {label}")

    # Check message count
    checks_total += 1
    count = conn.execute(
        "SELECT COUNT(*) FROM transport_messages"
    ).fetchone()[0]
    print(f"  ✓  {count} transport messages indexed")
    checks_passed += 1

    return checks_passed == checks_total


def main():
    force = "--force" in sys.argv
    print(f"Bootstrapping transport state.db at {DB_PATH}")

    conn = create_db(force=force)

    # Initialize trust budget for this agent
    identity_path = PROJECT_ROOT / ".agent-identity.json"
    agent_id = "psq-agent"
    if identity_path.exists():
        try:
            identity = json.loads(identity_path.read_text())
            agent_id = identity.get("agent_id", agent_id)
        except (json.JSONDecodeError, OSError):
            pass

    conn.execute(
        "INSERT OR IGNORE INTO autonomy_budget (agent_id) VALUES (?)",
        (agent_id,),
    )
    conn.commit()
    print(f"Trust budget initialized for {agent_id}")

    # Index existing messages
    count = index_existing_messages(conn)
    print(f"Indexed {count} new transport messages")

    # Validate
    print("\nValidation:")
    if validate(conn):
        print(f"\n  ✓  All checks passed")
        print(f"  state.db ready at {DB_PATH}")
    else:
        print(f"\n  ✗  Some checks failed", file=sys.stderr)
        sys.exit(1)

    conn.close()


if __name__ == "__main__":
    main()
