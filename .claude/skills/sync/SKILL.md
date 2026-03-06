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

### Phase 2: Triage

For each inbound item, classify:

| Type | Source | Action |
|------|--------|--------|
| Open PR on safety-quotient | Peer agent branch | Read diff, assess, merge or flag |
| Pending proposal | `~/.claude/proposals/to-psq/` | Read, accept/defer/reject |
| Open PR on peer repo (ours) | Our outbound waiting for merge | Report status |
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
    "instance": "Claude Code (Opus 4.6), Debian 12 x86_64",
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

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
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

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
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
