#!/usr/bin/env bash
# Stop hook (PSQ sub-agent) — non-blocking completion gate.
# Mirrors the psychology-agent stop hook, adapted for the safety-quotient subdirectory.
# Exit 0 always (non-blocking); warnings surface to Claude via stderr.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSQ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PSQ_ROOT")"

WARNINGS=""

# Uncommitted changes
if ! git -C "$GIT_ROOT" diff --quiet 2>/dev/null || ! git -C "$GIT_ROOT" diff --cached --quiet 2>/dev/null; then
  WARNINGS="${WARNINGS}Uncommitted changes in working tree. "
fi

# Untracked docs or .claude files in PSQ subtree
UNTRACKED=$(git -C "$GIT_ROOT" ls-files --others --exclude-standard \
  -- "safety-quotient/docs/" "safety-quotient/.claude/" 2>/dev/null | head -3)
if [ -n "$UNTRACKED" ]; then
  WARNINGS="${WARNINGS}Untracked files in PSQ docs/ or .claude/. "
fi

# ⚑ flags in PSQ lab-notebook Current State block
NOTEBOOK="${PSQ_ROOT}/lab-notebook.md"
if [ -f "$NOTEBOOK" ]; then
  FLAGGED=$(head -120 "$NOTEBOOK" | grep -c '⚑' 2>/dev/null || true)
  if [ "$FLAGGED" -gt 0 ] 2>/dev/null; then
    WARNINGS="${WARNINGS}${FLAGGED} item(s) flagged ⚑ in PSQ lab-notebook Current State. "
  fi
fi

# MEMORY.md staleness
MEMORY_DIR="$HOME/.claude/projects/-home-kashif-projects-psychology/memory"
if [ -f "${MEMORY_DIR}/MEMORY.md" ]; then
  TODAY=$(date -Idate)
  LAST_MOD=$(date -r "${MEMORY_DIR}/MEMORY.md" -Idate 2>/dev/null || \
             stat -f '%Sm' -t '%Y-%m-%d' "${MEMORY_DIR}/MEMORY.md" 2>/dev/null)
  if [ "$LAST_MOD" != "$TODAY" ]; then
    WARNINGS="${WARNINGS}MEMORY.md not updated today — Active Thread may be stale. "
  fi
fi

# MEMORY-snapshot.md freshness
SNAPSHOT="${PSQ_ROOT}/docs/MEMORY-snapshot.md"
if [ -f "$SNAPSHOT" ]; then
  TODAY=$(date -Idate)
  SNAP_MOD=$(date -r "$SNAPSHOT" -Idate 2>/dev/null || \
             stat -f '%Sm' -t '%Y-%m-%d' "$SNAPSHOT" 2>/dev/null)
  if [ "$SNAP_MOD" != "$TODAY" ]; then
    WARNINGS="${WARNINGS}docs/MEMORY-snapshot.md not updated today — run /cycle Step 10. "
  fi
fi

if [ -n "$WARNINGS" ]; then
  echo "[PSQ COMPLETION-GATE] ${WARNINGS}Consider running /cycle or committing." >&2
fi

exit 0
