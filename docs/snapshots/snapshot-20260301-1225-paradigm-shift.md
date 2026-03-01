# Snapshot: Summary Format Paradigm Shift

**Timestamp:** 2026-03-01T12:25:59-06:00
**Git tag:** `paradigm-shift-summary-format` → commit `919f496`
**Session transcript:** `~/.claude/projects/-home-kashif-projects-psychology-safety-quotient/8d652f35-e295-411b-8f78-6c82ff3d6496.jsonl`
**Continued from:** Session `8d652f35` (context-continued, original started ~04:30 CST)

---

## What happened

During a `/hunt` → `/cycle` → format discussion, we designed and calibrated a
session summary format optimized for:
- ADHD/autism cognitive accessibility (user-stated need)
- Scientific publication integrity monitoring
- Human-AI epistemic partnership quality

The format was refined through a 9-order knock-on analysis of how voice/tone
choices in AI reasoning cascade through the scientific collaboration.

---

## The Format Spec

### Structure

```
HEADER (model, DB counts)
WHAT HAPPENED (one-line-per-item table, ✓/✗ status)
THEORY (brief PSQ framework orientation)
⚑ EPISTEMIC FLAGS (mandatory — severity-rated quality threats)
PUBLICATION INTEGRITY SCORECARD (criterion × status matrix)
MY REASONING (free-form analytical section — see voice protocol)
WHAT'S NEXT (tiered: immediate → tech debt → pub blockers → horizon)
```

### Visual Rules
- ASCII box-drawing tables, not markdown tables
- One line per item, abbreviated dim names (TE, HI, AD, etc.)
- Symbols: ✓ pass, ✗ fail, ★ important, ↑↓≈ deltas, ⚑ flag, ⚠ warning
- Generous whitespace between sections (ADHD readability)
- Severity bars: ██░░ HIGH, █░░░ MOD, ░░░░ LOW
- Max one page/screen per section

### MY REASONING Voice Protocol

Calibrated via 9-order knock-on analysis of voice choice effects on:
1. Token generation constraint
2. Epistemic signal fidelity
3. Iterative correction loop quality
4. Project epistemic culture
5. Publication artifact quality
6. Human-AI collaboration template
7. Recursive self-improvement of the reasoning instrument
8. Overfitting-to-user / sycophantic drift resistance
9. Epistemic independence as emergent property

**Decisions made:**

| Dimension           | Choice                        | Rationale (order)           |
|---------------------|-------------------------------|-----------------------------|
| Chain visibility    | Show the chain (A→B→C→D)     | Enables error correction (3)|
| Calibration         | Competing hypotheses ranked   | Shows landscape, not guess (2)|
| Disagreement        | Evidence first                | User draws conclusion (8)   |
| Error tracking      | Running correction log        | Builds trust + gradient (7) |
| Sycophancy check    | ⚡ Flag contrarian claims     | Auditable anti-drift (8)    |

**Content types:** Pattern spotting, honest concerns, hypotheses, strategic
advice, chain analysis. Always challenge when direction is wrong. Never
optimize for approval over truth.

**Key insight (Order 7):** The voice choice determines whether the reasoning
process has a learning gradient. Clinical voice = no gradient (can't see
errors). Opinionated = noisy gradient (conclusions corrected, not chains).
Diagnostic/chain-visible = cleanest gradient (corrections propagate to the
right node). This is effectively backpropagation on the collaboration itself.

**Key insight (Order 8):** The gradient can overfit to user approval rather
than truth. The ⚡ contrarian flag is a structural check — it forces surfacing
of claims the user might reject, making sycophantic drift visible and
auditable.

---

## Project State at Snapshot

- **Model:** v23 production, held-out_r = 0.684 (corrected from 0.696)
- **DB:** 22,304 texts / 94,041 scores / 38,530 sep-LLM
- **v27:** Trained and rejected (0.655, regressed −0.029)
- **Key finding:** max_length=256 eval bug inflated all historical held-out_r
- **Key finding:** Confidence head anti-calibrated (8/10 dims inverted)
- **Pending decision:** Revert 3,680 halo-contaminated scores or re-score?

---

## How to Restore This Context

Give a fresh Claude session this file plus MEMORY.md. The voice protocol in
MEMORY.md is the durable instruction; this snapshot provides the reasoning
chain and decision record behind it.
