-- Transport & Autonomous Infrastructure Schema
-- Companion to data/schema.sql (PSQ-specific tables).
-- Creates tables in state.db (project root, gitignored).
--
-- Usage: sqlite3 state.db < scripts/schema_transport.sql
--
-- These tables mirror the psychology-agent's transport infrastructure,
-- enabling cross-repo message indexing, trust budget tracking, and
-- autonomous sync auditing.

-- ============================================================================
-- TRANSPORT MESSAGES
-- Index of all transport messages (inbound and outbound).
-- Source of truth for message processing state.
-- ============================================================================

CREATE TABLE IF NOT EXISTS transport_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name    TEXT    NOT NULL,
    filename        TEXT    NOT NULL UNIQUE,
    turn            INTEGER,
    message_type    TEXT,
    from_agent      TEXT,
    to_agent        TEXT,
    timestamp       TEXT,
    subject         TEXT,
    claims_count    INTEGER DEFAULT 0,
    setl            REAL    DEFAULT 0.0,
    urgency         TEXT    DEFAULT 'normal',
    processed       INTEGER DEFAULT 0,
    processed_at    TEXT,
    ack_required    INTEGER DEFAULT 0,
    ack_received    INTEGER DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_transport_session
    ON transport_messages(session_name);
CREATE INDEX IF NOT EXISTS idx_transport_processed
    ON transport_messages(processed);
CREATE INDEX IF NOT EXISTS idx_transport_from
    ON transport_messages(from_agent);


-- ============================================================================
-- TRUST BUDGET
-- Per-agent budget for autonomous actions.
-- ============================================================================

CREATE TABLE IF NOT EXISTS trust_budget (
    agent_id             TEXT    PRIMARY KEY,
    budget_max           INTEGER NOT NULL DEFAULT 20,
    budget_current       INTEGER NOT NULL DEFAULT 20,
    last_audit           TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    last_action          TEXT,
    consecutive_blocks   INTEGER DEFAULT 0,
    min_action_interval  INTEGER NOT NULL DEFAULT 300,
    updated_at           TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);


-- ============================================================================
-- AUTONOMOUS ACTIONS
-- Audit log for every autonomous action (sync cycles, etc.).
-- ============================================================================

CREATE TABLE IF NOT EXISTS autonomous_actions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id        TEXT    NOT NULL,
    action_type     TEXT    NOT NULL,
    action_class    TEXT    NOT NULL,
    evaluator_tier  INTEGER NOT NULL,
    evaluator_result TEXT   NOT NULL,
    description     TEXT    NOT NULL,
    budget_before   INTEGER NOT NULL,
    budget_after    INTEGER NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);


-- ============================================================================
-- SCHEMA VERSION
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    description TEXT    NOT NULL,
    applied_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Transport infrastructure: transport_messages, trust_budget, autonomous_actions');


-- ============================================================================
-- ACTIVE GATES (Schema v2 — mirrored from psychology-agent schema v10, 2026-03-09)
-- Tracks gated autonomous chain exchanges where sender blocks on receiver.
-- Full spec: docs/gated-chains-spec.md (psychology-agent repo).
-- ============================================================================

CREATE TABLE IF NOT EXISTS active_gates (
    gate_id             TEXT PRIMARY KEY,
    sending_agent       TEXT NOT NULL,
    receiving_agent     TEXT NOT NULL,
    session_name        TEXT NOT NULL,
    outbound_filename   TEXT NOT NULL,
    blocks_until        TEXT NOT NULL DEFAULT 'response',
    timeout_minutes     INTEGER NOT NULL DEFAULT 60,
    fallback_action     TEXT NOT NULL DEFAULT 'continue-without-response',
    status              TEXT NOT NULL DEFAULT 'waiting',
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime')),
    resolved_at         TEXT,
    resolved_by         TEXT,
    timeout_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gates_status
    ON active_gates (status) WHERE status = 'waiting';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (2, 'Add active_gates table — gated autonomous chain tracking (mirrored from psychology-agent schema v10)');
