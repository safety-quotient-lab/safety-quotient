-- PSQ Database Schema
-- Single source of truth for all texts, scores, labeling sessions, and model artifacts.
--
-- Design principles:
--   - A score is an *observation*, not a property of a text. Nothing is overwritten.
--   - Labeling sessions are persistent DB objects, not /tmp files.
--   - Models and their calibrations are explicitly linked.
--   - Split assignments are persisted before training, not computed in RAM.
--
-- Field semantics:
--   source    = where the text came from (origin dataset, never changes)
--   method    = how it was scored (the process)
--   scorer    = what did the scoring (the agent)


-- ============================================================================
-- TEXTS
-- ============================================================================

CREATE TABLE texts (
    id          INTEGER PRIMARY KEY,
    text        TEXT    NOT NULL,
    text_hash   TEXT    NOT NULL UNIQUE,  -- sha256(text) for dedup
    source      TEXT    NOT NULL,         -- goemotions | civil_comments | esconv |
                                          -- dreaddit | empathetic_dialogues | casino |
                                          -- prosocial | politeness_* | ucc | berkeley |
                                          -- synthetic | relabeled | claude_code
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_texts_source ON texts(source);
CREATE INDEX idx_texts_hash   ON texts(text_hash);


-- ============================================================================
-- SCORES
-- Every scoring observation — composite proxy, joint LLM, separated LLM, synthetic.
-- Never overwritten. Query best_scores view for canonical training values.
-- ============================================================================

CREATE TABLE scores (
    id          INTEGER PRIMARY KEY,
    text_id     INTEGER NOT NULL REFERENCES texts(id),
    dimension   TEXT    NOT NULL,   -- threat_exposure | hostility_index | authority_dynamics |
                                    -- energy_dissipation | regulatory_capacity |
                                    -- resilience_baseline | trust_conditions |
                                    -- cooling_capacity | defensive_architecture |
                                    -- contractual_clarity
    score       REAL    NOT NULL CHECK (score BETWEEN 0 AND 10),
    confidence  REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    method      TEXT    NOT NULL CHECK (method IN (
                    'composite-proxy',  -- proxy from dataset labels (goemotions, CC, etc.)
                    'joint-llm',        -- all 10 dims in one LLM call (halo-inflated)
                    'separated-llm',    -- one dim per LLM call (halo-free)
                    'synthetic'         -- hand-written text + author-assigned score
                )),
    scorer      TEXT,               -- model ID or proxy name:
                                    --   proxy:  goemotions-proxy | civil-comments-proxy | etc.
                                    --   llm:    claude-sonnet-4-6 | claude-haiku-4-5 | etc.
                                    --   legacy: claude-code (version unknown, pre-provenance)
    provider    TEXT,               -- LLM vendor: 'anthropic' | NULL for proxy/synthetic
    interface   TEXT CHECK (interface IN ('claude-code', 'api', 'proxy', NULL)),
                                    -- how scoring was invoked:
                                    --   'claude-code' — interactive Claude Code session
                                    --   'api'         — Anthropic API call
                                    --   'proxy'       — formula from dataset labels (no LLM)
                                    --   NULL          — synthetic (author-assigned)
    session_id  TEXT    REFERENCES labeling_sessions(id),  -- NULL for composite-proxy/synthetic
    scored_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    notes       TEXT
);

CREATE INDEX idx_scores_text_dim  ON scores(text_id, dimension);
CREATE INDEX idx_scores_method    ON scores(method);
CREATE INDEX idx_scores_dim       ON scores(dimension);
CREATE INDEX idx_scores_session   ON scores(session_id);


-- ============================================================================
-- SPLITS
-- Persisted train/val/test/held-out assignments.
-- Computed from hash(text) % 100 once, then frozen. Never re-derived at training time.
-- A text can be in held-out AND test simultaneously for cross-referencing.
-- ============================================================================

CREATE TABLE splits (
    text_id     INTEGER NOT NULL REFERENCES texts(id),
    split       TEXT    NOT NULL CHECK (split IN ('train', 'val', 'test', 'held-out')),
    assigned_at TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (text_id, split)
);

CREATE INDEX idx_splits_split ON splits(split);


-- ============================================================================
-- LABELING SESSIONS
-- Replaces /tmp/psq_separated. Sessions survive reboots and can be resumed.
-- One session = one input file, one or more dimensions being scored.
-- Scores are written to session_scores (staging), then promoted to scores on commit.
-- ============================================================================

CREATE TABLE labeling_sessions (
    id           TEXT    PRIMARY KEY,   -- e.g. 'sep_20260227_001'
    input_file   TEXT    NOT NULL,      -- data/train-llm.jsonl (what was extracted)
    offset       INTEGER NOT NULL DEFAULT 0,
    limit_n      INTEGER NOT NULL DEFAULT 0,  -- 0 = all
    n_texts      INTEGER NOT NULL,
    dimensions   TEXT    NOT NULL,      -- JSON array of dims being scored in this session
    method       TEXT    NOT NULL DEFAULT 'separated-llm',
    scorer       TEXT    NOT NULL DEFAULT 'claude-code',  -- model ID (e.g. claude-sonnet-4-6)
    provider     TEXT    NOT NULL DEFAULT 'anthropic',    -- LLM vendor
    interface    TEXT    NOT NULL DEFAULT 'claude-code'
                         CHECK (interface IN ('claude-code', 'api')),
    status       TEXT    NOT NULL DEFAULT 'open'
                         CHECK (status IN ('open', 'committed', 'abandoned')),
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    committed_at TEXT,
    notes        TEXT
);

-- Staging table: scores in progress, not yet in the main scores table
CREATE TABLE session_scores (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT    NOT NULL REFERENCES labeling_sessions(id),
    text_id     INTEGER NOT NULL REFERENCES texts(id),
    dimension   TEXT    NOT NULL,
    score       REAL    NOT NULL CHECK (score BETWEEN 0 AND 10),
    confidence  REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    scored_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (session_id, text_id, dimension)  -- no duplicate scores within a session
);

CREATE INDEX idx_session_scores_session  ON session_scores(session_id);
CREATE INDEX idx_session_scores_text_dim ON session_scores(text_id, dimension);


-- ============================================================================
-- MODELS
-- Model registry. "v13" is a row here, not a filename convention.
-- ============================================================================

CREATE TABLE models (
    id              TEXT    PRIMARY KEY,  -- 'psq-v13'
    version         INTEGER NOT NULL,
    architecture    TEXT    NOT NULL,     -- 'distilbert-base-uncased'
    checkpoint_path TEXT    NOT NULL,     -- 'models/psq-student/best.pt'
    onnx_path       TEXT,                 -- 'models/psq-student/model.onnx'
    onnx_quant_path TEXT,                 -- 'models/psq-student/model_quantized.onnx'
    config_json     TEXT    NOT NULL,     -- full training config as JSON string
    test_r          REAL,                 -- average Pearson r on test split
    held_out_r      REAL,                 -- average Pearson r on held-out set
    status          TEXT    NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active', 'archived', 'broken')),
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    notes           TEXT
);

-- Per-dimension results for each model
CREATE TABLE model_results (
    id          INTEGER PRIMARY KEY,
    model_id    TEXT    NOT NULL REFERENCES models(id),
    split       TEXT    NOT NULL CHECK (split IN ('val', 'test', 'held-out')),
    dimension   TEXT    NOT NULL,
    pearson_r   REAL,
    mse         REAL,
    n           INTEGER,
    UNIQUE (model_id, split, dimension)
);


-- ============================================================================
-- CALIBRATION
-- Isotonic regression maps linked to the model they were fitted on.
-- Replacing calibration.json — each calibration row is immutable once written.
-- ============================================================================

CREATE TABLE calibrations (
    id              TEXT    PRIMARY KEY,  -- 'psq-v13-cal'
    model_id        TEXT    NOT NULL REFERENCES models(id),
    fitted_on_split TEXT    NOT NULL,     -- 'held-out' | 'val'
    n_samples       INTEGER NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    notes           TEXT
);

-- Per-dimension calibration thresholds
CREATE TABLE calibration_thresholds (
    id              INTEGER PRIMARY KEY,
    calibration_id  TEXT    NOT NULL REFERENCES calibrations(id),
    dimension       TEXT    NOT NULL,
    score_x         TEXT    NOT NULL,   -- JSON array of raw model output breakpoints
    score_y         TEXT    NOT NULL,   -- JSON array of calibrated score values
    conf_x          TEXT    NOT NULL,   -- JSON array of raw confidence breakpoints
    conf_y          TEXT    NOT NULL,   -- JSON array of calibrated confidence values
    UNIQUE (calibration_id, dimension)
);


-- ============================================================================
-- DATASET MAPPINGS
-- Mirrors data/dataset_mappings.json in the DB for auditability.
-- The JSON file is the source of truth. This table is populated by migrate.py.
-- ============================================================================

CREATE TABLE dataset_mappings (
    id              INTEGER PRIMARY KEY,
    version         TEXT    NOT NULL,           -- 'v13', 'v14', ...
    dataset_id      TEXT    NOT NULL,           -- 'goemotions', 'civil_comments', etc.
    dimension       TEXT    NOT NULL,
    enabled         INTEGER NOT NULL DEFAULT 1, -- 0 = disabled
    mapping_type    TEXT    NOT NULL,           -- 'scale', 'weighted_scale', 'cluster_binary', etc.
    config_json     TEXT    NOT NULL,           -- full mapping config as JSON string
    disabled_since  TEXT,                       -- version when disabled, if applicable
    disabled_reason TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (version, dataset_id, dimension)
);

CREATE INDEX idx_dataset_mappings_version ON dataset_mappings(version);
CREATE INDEX idx_dataset_mappings_dataset ON dataset_mappings(dataset_id);


-- ============================================================================
-- VIEWS
-- ============================================================================

-- best_scores: canonical score per text+dim
--
-- Priority: separated-llm(1) > synthetic(2) > joint-llm(3) > composite-proxy(4)
-- Tiebreak: higher confidence, then most recent
CREATE VIEW best_scores AS
SELECT s.*
FROM scores s
WHERE s.id = (
    SELECT s2.id FROM scores s2
    WHERE s2.text_id = s.text_id AND s2.dimension = s.dimension
    ORDER BY
        CASE s2.method
            WHEN 'separated-llm'   THEN 1
            WHEN 'synthetic'       THEN 2
            WHEN 'joint-llm'       THEN 3
            WHEN 'composite-proxy' THEN 4
        END ASC,
        s2.confidence DESC,
        s2.scored_at DESC
    LIMIT 1
);


-- training_data: what distill.py reads
-- One row per (text, dimension) — same shape as current JSONL loading.
CREATE VIEW training_data AS
SELECT
    t.id          AS text_id,
    t.text,
    t.source,
    bs.dimension,
    bs.score,
    bs.confidence,
    bs.method,
    bs.scorer,
    CASE bs.method
        WHEN 'separated-llm'   THEN 5.0
        WHEN 'synthetic'       THEN 5.0
        WHEN 'joint-llm'       THEN 5.0
        WHEN 'composite-proxy' THEN 1.5
    END AS sample_weight
FROM best_scores bs
JOIN texts  t  ON t.id       = bs.text_id
JOIN splits sp ON sp.text_id = t.id AND sp.split = 'train';


-- held_out_scores: ALL scoring events for held-out texts
-- Returns multiple rows per text+dim when scored by different methods —
-- enables joint-vs-separated comparison and halo validation.
CREATE VIEW held_out_scores AS
SELECT
    t.id        AS text_id,
    t.text,
    t.source,
    s.dimension,
    s.score,
    s.confidence,
    s.method,
    s.scorer,
    s.scored_at
FROM scores s
JOIN texts  t  ON t.id       = s.text_id
JOIN splits sp ON sp.text_id = t.id AND sp.split = 'held-out'
ORDER BY t.id, s.dimension, s.scored_at;


-- session_progress: how far along each open labeling session is
CREATE VIEW session_progress AS
SELECT
    ls.id           AS session_id,
    ls.input_file,
    ls.n_texts,
    ls.dimensions,
    ls.status,
    ls.created_at,
    json_each.value AS dimension,
    COUNT(ss.id)    AS n_scored,
    ls.n_texts - COUNT(ss.id) AS n_remaining
FROM labeling_sessions ls,
     json_each(ls.dimensions)
LEFT JOIN session_scores ss
       ON ss.session_id = ls.id
      AND ss.dimension  = json_each.value
GROUP BY ls.id, json_each.value;


-- coverage: scoring progress per dimension+method across all committed scores
CREATE VIEW coverage AS
SELECT
    s.dimension,
    s.method,
    COUNT(DISTINCT s.text_id) AS n_texts,
    ROUND(AVG(s.confidence), 3) AS avg_conf,
    ROUND(AVG(s.score), 2)      AS avg_score,
    MIN(s.scored_at)            AS first_scored,
    MAX(s.scored_at)            AS last_scored
FROM scores s
GROUP BY s.dimension, s.method
ORDER BY s.dimension, s.method;


-- ============================================================================
-- MIGRATION NOTES
-- ============================================================================
--
-- teacher → method mapping for existing JSONL files:
--
--   composite-ground-truth.jsonl (no teacher field):
--     source = 'synthetic'   → method = 'synthetic',       scorer = 'claude-code'
--     source = 'relabeled'   → method = 'separated-llm',   scorer = 'claude-code'
--     all other sources      → method = 'composite-proxy',  scorer = '{source}-proxy'
--
--   train-llm.jsonl:
--     teacher = 'llm'            → method = 'joint-llm',     scorer = 'claude-code'
--     teacher = 'llm_labeled'    → method = 'joint-llm',     scorer = 'claude-code'
--     teacher = 'separated-llm'  → method = 'separated-llm', scorer = 'claude-code'
--
--   held-out-test.jsonl:
--     teacher = 'separated-llm'  → method = 'separated-llm', scorer = 'claude-code'
--     split = 'held-out'
--
-- Split assignments:
--   Derive from distill.py's hash(text) % 100 logic:
--     < 80  → train
--     < 90  → val
--     else  → test
--   held-out texts get split = 'held-out' regardless of hash.
--
-- Model registry seed row:
--   id = 'psq-v13', version = 13, architecture = 'distilbert-base-uncased'
--   test_r = 0.553, held_out_r = 0.402
--
-- Calibration seed row:
--   id = 'psq-v13-cal', model_id = 'psq-v13'
--   fitted_on_split = 'held-out', n_samples = 100
--   thresholds: migrate from models/psq-student/calibration.json
