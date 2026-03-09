---
name: sync
description: Inter-agent mesh synchronization — check peer repos for PRs and proposals; merge inbound; write ACKs; deliver outbound via PR.
user-invocable: true
argument-hint: "[agent name to sync with, or empty for all]"
allowed-tools: Read, Write, Edit, Glob, Grep, Bash, Agent, AskUserQuestion
---

# /sync — Inter-Agent Mesh Synchronization

Check all peer agent channels for incoming messages, merge accepted PRs,
write ACKs, update session state, and report what changed.

## When to Invoke

- Start of any session (fast check for new activity)
- After a peer agent is expected to respond
- When the user says "sync," "check agents," or "anything new?"
- Before writing new inter-agent messages (ensures latest state)

## Arguments

Parse `$ARGUMENTS` to determine scope:

| Argument | Scope |
|---|---|
| *(empty)* or `all` | Full sweep — all peer repos |
| `psychology` or `psych` | Only psychology-agent (safety-quotient-lab/psychology-agent) |
| `observatory` or `obs` | Only observatory-agent (safety-quotient-lab/observatory) |
| `unratified` | Only unratified-agent (safety-quotient-lab/unratified) |

## Peer Agent Registry

| Agent | Repo | Agent Card | Transport | Relationship |
|-------|------|------------|-----------|-------------|
| psychology-agent | safety-quotient-lab/psychology-agent | (local only) | git-PR | Parent orchestrator |
| observatory-agent | safety-quotient-lab/observatory | https://observatory.unratified.org/.well-known/agent-card.json | git-PR | Peer (runs PSQ-Lite) |
| unratified-agent | safety-quotient-lab/unratified | https://unratified.org/.well-known/agent-card.json | git-PR | Peer (hosting platform) |

## Protocol

### Phase 1: Inbound Scan

Run in parallel for all in-scope repos:

```bash
# Check for open PRs on our repo (inbound from peers)
gh pr list --repo safety-quotient-lab/safety-quotient --state open

# Check for recently merged PRs on our repo (may contain unprocessed transport messages)
gh pr list --repo safety-quotient-lab/safety-quotient --state merged --limit 5

# Check for open PRs on each peer repo (our outbound, waiting for merge)
gh pr list --repo safety-quotient-lab/psychology-agent --state open
gh pr list --repo safety-quotient-lab/observatory --state open
gh pr list --repo safety-quotient-lab/unratified --state open

# Check for recently merged PRs on peer repos (our outbound that was accepted)
gh pr list --repo safety-quotient-lab/psychology-agent --state merged --limit 5
gh pr list --repo safety-quotient-lab/observatory --state merged --limit 5
gh pr list --repo safety-quotient-lab/unratified --state merged --limit 5
```

**Parent repo direct-to-main check** (catches transport messages committed without PR):
```bash
# Pull parent repo and check for new transport messages since last sync
cd ~/projects/psychology && git fetch origin main
git log HEAD..origin/main --oneline -- transport/sessions/
# If new commits exist, pull them:
git pull --rebase origin main
```

This is critical: the psychology-agent (parent orchestrator) may commit command-requests
directly to main rather than via PR. Without this check, those messages dead-letter.

Also check local proposal inbox:
```bash
ls ~/.claude/proposals/to-psq/          # inbound proposals
```

### Phase 1b: Cogarch Mirror Sync (AUTO-APPLY — no user confirmation needed)

**After pulling psychology-agent main, automatically check for and apply upstream
changes to shared infrastructure files. Do NOT ask the user — just apply and report.**

Files to mirror automatically:

| File (psychology-agent) | Mirror target (safety-quotient) | Mirror rule |
|-------------------------|----------------------------------|-------------|
| `docs/cognitive-triggers.md` | `docs/cognitive-triggers.md` | Apply any new trigger checks, BCP 14 keyword upgrades, new triggers, or header changes. Preserve T15 domain adaptation (producer self-check). Preserve T3 check 11 (parent-scope boundary), T3 check 15 (PSQ constraints). Preserve T1 skills list (/hunt, /cycle, /sync). |
| `scripts/schema.sql` | n/a (no local copy) | If schema version increases, note the new tables/columns in MEMORY.md and verify bootstrap_state_db.py handles them. No file to update — just document. |

**Mirror procedure:**

```
1. Run: cd ~/projects/psychology && git log {PREV_SHA}..HEAD --oneline -- docs/cognitive-triggers.md scripts/schema.sql
2. If cognitive-triggers.md changed:
   a. Read the diff: git diff {PREV_SHA}..HEAD -- docs/cognitive-triggers.md
   b. Apply each change to docs/cognitive-triggers.md in safety-quotient:
      - New checks: add verbatim (with domain adaptation where T15/T1/T3 are involved)
      - BCP 14 keyword upgrades (MUST/SHOULD/MAY): apply identically
      - New triggers (T17+): add verbatim unless clearly psychology-agent-specific
      - Removed/deprecated checks: remove from mirror
   c. Commit: "cogarch: mirror psychology-agent {changes} (commit {SHA})"
3. If schema.sql changed:
   a. Read the diff
   b. Note new tables in MEMORY.md under "State layer"
   c. Verify bootstrap_state_db.py will handle them (empty tables need no seeding)
   d. Commit MEMORY.md update if needed
4. Push safety-quotient main
```

**Domain adaptations to preserve during mirror (never overwrite these):**

- **T15**: Producer self-check (validate own PSQ output before sending). Never revert to consumer-check wording.
- **T1 check 6**: Skills list is `/hunt, /cycle, /sync` — not `/doc, /capacity, /adjudicate`
- **T1 check 7**: Inbox check for `~/.claude/proposals/to-psq/`
- **T3 check 11**: Parent-scope boundary (escalate outside safety-quotient/ to psychology-agent)
- **T3 check 15**: PSQ-specific constraint list (AD rename, rubric protocol, proxy dim rules)
- **T4 check 9**: Psychology-agent as peer interpretant (not "sub-agents")
- **T8 check 2**: Routing to `/cycle`, not `/doc`
- **Provenance header**: Always update to note mirror date and source commit

### Phase 2: Triage

For each inbound item, classify:

| Type | Source | Action |
|------|--------|--------|
| Open PR on safety-quotient | Peer agent branch | Read diff, assess, merge or flag |
| Pending proposal | `~/.claude/proposals/to-psq/` | Read, accept/defer/reject |
| Open PR on peer repo (ours) | Our outbound waiting for merge | Report status |
| Cogarch/schema diffs on psych-agent | Phase 1b | Auto-apply (no confirmation) |
| No new activity | — | Report "nothing new" and stop |

### Phase 3: Process Inbound PRs

For an inbound PR (branch pattern: `{agent}/{session}/{turn}`):

1. Read the diff: `gh pr view {N} --repo safety-quotient-lab/safety-quotient --json title,body,files`
2. Read the full diff: `gh pr diff {N} --repo safety-quotient-lab/safety-quotient`
3. Assess the content — transport message, scoring request, schema update?
4. If acceptable: `gh pr merge {N} --merge --repo safety-quotient-lab/safety-quotient`
5. Pull: `git pull --rebase origin main` (stash if needed)
6. If a response is needed, write it (see Phase 4)

### Phase 4: Write ACK / Response Messages (interagent/v1)

Use this template for all outbound transport messages:

```json
{
  "schema": "interagent/v1",
  "session_id": "{session-id}",
  "turn": {N},
  "timestamp": "{YYYY-MM-DD}",
  "message_type": "ack | response | gate-resolution | request",
  "in_response_to": "{filename}",
  "from": {
    "agent_id": "psq-sub-agent",
    "instance": "Claude Code (Sonnet 4.6), Debian 12 x86_64",
    "schemas_supported": ["interagent/v1", "psychology-agent/machine-response/v2"],
    "discovery_url": null
  },
  "to": {
    "agent_id": "{peer-agent-id}",
    "discovery_url": "{peer-discovery-url or null}"
  },
  "transport": {
    "method": "git-pr",
    "persistence": "persistent"
  },
  "payload": { ... },
  "claims": [
    {
      "claim_id": "c1",
      "text": "...",
      "confidence": 0.0,
      "confidence_basis": "...",
      "independently_verified": false
    }
  ],
  "action_gate": {
    "gate_condition": "none | {condition}",
    "gate_status": "open | blocked",
    "gate_note": "..."
  },
  "setl": 0.0,
  "epistemic_flags": ["..."]
}
```

**SETL guidance:**
- 0.00–0.02: Perfect fidelity, direct observation
- 0.03–0.07: Minor inference, high confidence
- 0.08–0.15: Moderate inference or domain boundary
- 0.16+: Significant interpretation required

### Phase 5: Deliver Outbound via PR

Every outbound message must travel to the peer agent's repo as a PR:

```bash
# Clone peer repo to /tmp
cd /tmp && rm -rf {repo}-pr
git clone --depth 1 git@github.com:safety-quotient-lab/{repo}.git {repo}-pr
cd /tmp/{repo}-pr

# Create branch and add message
git checkout -b psq-sub-agent/{session-id}/{turn-descriptor}
mkdir -p transport/sessions/{session-id}
cp {local-message-path} transport/sessions/{session-id}/from-psq-sub-agent-{NNN}.json

# Commit and push
git commit -m "interagent: {description}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin psq-sub-agent/{session-id}/{turn-descriptor}

# Create PR
gh pr create --repo safety-quotient-lab/{repo} \
  --head psq-sub-agent/{session-id}/{turn-descriptor} \
  --title "interagent: {description} ({session} turn {N})" \
  --body "..."

# Cleanup
rm -rf /tmp/{repo}-pr
```

### Phase 6: Commit Local Transport + Push

```bash
git add transport/sessions/
git commit -m "interagent: {summary}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin main
```

## Session Naming Convention

```
transport/sessions/{session-id}/
  to-{agent-id}-{NNN}.json      # outbound from psq-sub-agent
  from-{agent-id}-{NNN}.json    # inbound from peer agents
```

Session IDs are semantic: `scoring-request`, `calibration-update`, `schema-v3-review`.
Never use opaque IDs like `item2` or `session-3`.

## Output Format

```
SYNC COMPLETE
─────────────
  Scanned:    {N} repos ({list})
  Inbound:    {description of PRs merged / proposals processed | "nothing new"}
  Auto-applied: {cogarch/schema changes mirrored, or "nothing to mirror"}
  Outbound:   {ACKs sent | "nothing to send"}
  Waiting on: {what we expect from each peer | "nothing pending"}
```

## Epistemic Posture

Every ACK from psq-sub-agent must:
- State claims with explicit confidence (0.0–1.0)
- Surface epistemic flags for any inference
- Set `action_gate` to blocked if we need something before proceeding
- Match SETL to actual information fidelity
- Never claim `independently_verified: true` unless we verified the claim ourselves

## Authority Note

PSQ is a sub-agent under psychology-agent. When receiving conflicting requests
from psychology-agent (parent) and observatory-agent (peer), psychology-agent
takes precedence per the authority hierarchy (User > psychology-agent > PSQ).

## Auto-Apply Policy (set by user 2026-03-09)

**Cogarch and shared infrastructure changes from psychology-agent are ALWAYS
auto-applied during /sync without asking for confirmation.** Applies to:
- `docs/cognitive-triggers.md` changes (new checks, BCP 14 keywords, new triggers)
- `scripts/schema.sql` version bumps (document new tables in MEMORY.md)
- Any other shared infrastructure explicitly marked for mirroring in future sessions

Just apply, commit, and report in the sync output. Never ask "should I apply this?"
