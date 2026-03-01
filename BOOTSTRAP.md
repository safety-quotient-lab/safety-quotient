# Bootstrapping a Fresh Claude Code Session

How to bring a new Claude Code installation up to full context with this
project, starting from git clone.

**Last verified:** 2026-03-01

---

## What Lives Where

```
IN GIT (fully portable):
  ├── .claude/skills/         ← /hunt, /cycle skill definitions
  ├── .claude/settings.local.json  ← permission presets (user-specific)
  ├── scripts/                ← all Python scripts
  ├── data/                   ← DB, labeled JSONLs, provenance docs
  ├── *.md                    ← all research documentation
  └── BOOTSTRAP.md            ← this file

NOT IN GIT (user-local, must be reconstructed):
  ├── ~/.claude/projects/<path-hash>/memory/MEMORY.md    ← session orientation
  ├── ~/.claude/projects/<path-hash>/memory/snapshot-*.md ← restore points
  ├── models/                 ← .pt checkpoints, ONNX files (.gitignored)
  └── venv/                   ← Python virtual environment
```

The critical gap: **MEMORY.md** is not in the git repo. It lives in Claude
Code's per-project memory directory, which is user-local and path-dependent.
Without it, a fresh Claude Code session starts with no orientation context.

---

## Step-by-Step Bootstrap

### 1. Clone and set up environment

```bash
git clone git@github.com:safety-quotient-lab/safety-quotient.git
cd safety-quotient

python3 -m venv venv
source venv/bin/activate
pip install torch transformers onnxruntime scipy numpy
```

### 2. Restore MEMORY.md

Claude Code auto-creates the memory directory on first run based on the
absolute path of the project. The path hash is deterministic.

```bash
# Start Claude Code once in the project directory to create the path:
cd /path/to/safety-quotient
claude

# Then exit and find the created directory:
ls ~/.claude/projects/

# Copy MEMORY.md into the correct path-hashed directory:
# The source is committed in this repo for portability:
cp docs/MEMORY-snapshot.md ~/.claude/projects/<path-hash>/memory/MEMORY.md
```

**IMPORTANT:** MEMORY.md is the single most important file for session
continuity. It contains: current model version, DB state, labeling policy,
voice protocol, all key decisions, and file locations.

### 3. Restore snapshots (optional)

Snapshots capture point-in-time decision records. Copy any relevant ones:

```bash
cp docs/snapshots/*.md ~/.claude/projects/<path-hash>/memory/
```

### 4. Retrain or download model checkpoints

Model checkpoints are .gitignored (too large). To reconstruct:

```bash
# Option A: Retrain from DB (canonical, ~45 min on CPU)
source venv/bin/activate
python scripts/distill.py --db data/psq.db --drop-proxy-dims --out models/psq-v23

# Option B: If a model registry/release exists, download from there
# (not yet implemented)
```

### 5. Verify

```bash
# DB integrity
python3 -c "
import sqlite3
db = sqlite3.connect('data/psq.db')
c = db.cursor()
c.execute('SELECT COUNT(*) FROM texts')
print('Texts:', c.fetchone()[0])
c.execute('SELECT COUNT(*) FROM scores')
print('Scores:', c.fetchone()[0])
"

# Model loads
python scripts/eval_held_out.py --model models/psq-v23/best.pt

# Skills available
# Start Claude Code — /hunt and /cycle should auto-discover from .claude/skills/
```

---

## What's Still Not Portable

| Item | Why | Mitigation |
|------|-----|------------|
| **MEMORY.md** | Claude Code stores per-project memory by absolute path hash. Not in git. | Keep `docs/MEMORY-snapshot.md` as a committed copy. Bootstrap step 2 copies it. |
| **Snapshots** | Same path-hash directory as MEMORY.md. | Keep committed copies in `docs/snapshots/`. |
| **Model checkpoints** | 250+ MB each, .gitignored. | Retrain from DB (deterministic given seed) or add GitHub Releases. |
| **ONNX exports** | Same — large binaries. | Re-export via `scripts/export_onnx.py` after retraining. |
| **venv/** | Python environment, platform-specific. | `pip install` from requirements (not yet formalized — see TODO below). |
| **Labeling log timing** | `data/labeling_log.jsonl` is in git but timestamps are session-specific. | Informational only — doesn't affect reproducibility. |

---

## TODO: Full Portability

To make bootstrap truly one-command:

- [ ] Create `docs/MEMORY-snapshot.md` — committed copy of MEMORY.md, updated by /cycle
- [ ] Create `docs/snapshots/` — committed copies of key snapshots
- [ ] Create `requirements.txt` or `pyproject.toml` for venv reproducibility
- [ ] Consider GitHub Releases for model checkpoint distribution
- [ ] Add a `make bootstrap` target that automates steps 1-5
- [ ] Add CLAUDE.md to repo root (Claude Code reads this automatically — would eliminate the MEMORY.md portability gap for orientation context)

---

## Why Not Just Use CLAUDE.md?

Claude Code reads `CLAUDE.md` from the repo root automatically — no path-hash
dependency. This would solve the portability gap for orientation context.

**Current state:** No CLAUDE.md exists. All orientation lives in MEMORY.md
(user-local).

**Recommendation:** Create a CLAUDE.md that contains the stable, non-session-
specific parts of MEMORY.md (labeling policy, dimension names, voice protocol,
key file locations). Keep MEMORY.md for volatile state (current model version,
DB counts, in-progress work). /cycle updates both.

This splits orientation into:
- **CLAUDE.md** (in git) — "how this project works" — portable
- **MEMORY.md** (user-local) — "where we are right now" — session-specific
