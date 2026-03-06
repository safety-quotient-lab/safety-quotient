#!/usr/bin/env bash
# PSQ SessionStart hook — check proposal inbox for pending requests.
# stdout becomes model context at session start.

INBOX="$HOME/.claude/proposals/to-psq"

if [ -d "$INBOX" ]; then
  PENDING=$(find "$INBOX" -maxdepth 1 \( -name '*.json' -o -name '*.md' \) 2>/dev/null | wc -l)
  if [ "$PENDING" -gt 0 ]; then
    echo "[PSQ-INBOX] $PENDING pending proposal(s) in $INBOX:"
    find "$INBOX" -maxdepth 1 \( -name '*.json' -o -name '*.md' \) -exec basename {} \; 2>/dev/null
    echo "[PSQ-INBOX] Read and evaluate each proposal. Accept, modify, or reject. Respond via interagent/v1 format."
  fi
else
  echo "[PSQ-INBOX] WARNING: Inbox directory not found: $INBOX"
fi

exit 0
