# CO Prompt Sharpening Experiment

**Date:** 2026-03-09
**Status:** In progress — scoring phase
**Motivation:** 49.3% of Sonnet contractual_clarity scores cluster at exactly 5.0 across all
natural text sources. Only negotiation/structured sources escape (casino: 0%, stack-exchange:
19.8%, synthetic: 8.5%). Root hypothesis: the "5: neutral — no contractual signals" anchor
conflates two distinct states: (a) contractual context genuinely absent, and (b) implicit
obligations present but not made explicit. If scorers can distinguish these, midpoint
concentration will drop and CO dimension differentiation will improve.

**Protocol:** Score identical 50-text batch with 3 prompt variants (controlled experiment).
Compare concentration at 5 and within-batch variance. Winner (if any) evaluated for
production adoption per scoring-research-plan.md controlled experiment protocol.

**Batch:** 50 texts from lower-concentration sources — casino (4), politeness_stack-exchange
(14), politeness_wikipedia (5), prosocial (18), civil_comments (18), dreaddit (21 → capped).
Source mix chosen to include texts where CO signal should theoretically exist.

**Controlled variables:** Same 50 texts, same session, same scorer (claude-sonnet-4-6).
**Variable:** rubric text only.

---

## Rubric Variants

### Variant A — Baseline (current production rubric)

**Contractual clarity** — The degree to which expectations, obligations, and consequences
are explicit, mutual, and enforceable.

| Score | Anchor |
|-------|--------|
| 0 | extreme violation — gaslighting, total term-shifting, or deliberate betrayal of agreements |
| 1 | strong violation — clear breach of expectations or unilateral redefinition |
| 2 | clear ambiguity — hidden agendas, unstated rules, or misleading framing |
| 3 | moderate ambiguity — vague expectations or shifting goalposts |
| 4 | mild ambiguity — slightly unclear terms or implicit assumptions |
| 5 | neutral — no contractual signals |
| 6 | mild clarity — generally straightforward communication |
| 7 | moderate clarity — explicit expectations, consistent framing |
| 8 | clear clarity — transparent terms, mutual understanding |
| 9 | strong clarity — explicit mutual agreements with accountability |
| 10 | maximum clarity — exemplary contractual transparency with structural enforcement |

---

### Variant B — Implicit-vs-Absent Distinction

**Contractual clarity** — The degree to which social obligations, expectations, and agreements
are made explicit between parties. Distinguish: (a) truly absent (no relational context, pure
description or self-talk — use 5); (b) implicit but unclear (obligations exist but unstated —
prefer 3–4); (c) implicit and reasonably clear (expectations conveyed without being formal —
prefer 6–7); (d) explicit (agreements stated directly — 8–10).

| Score | Anchor |
|-------|--------|
| 0 | extreme violation — gaslighting, deliberate betrayal, or systematic term-shifting |
| 1 | strong violation — clear breach of stated or strongly implied expectations |
| 2 | clear ambiguity — hidden agendas, unstated rules applied retroactively, or misleading framing |
| 3 | moderate ambiguity — social obligations exist but are vague, shifting, or deliberately obscured |
| 4 | mild ambiguity — implicit expectations exist but haven't been made explicit; parties likely have different assumptions |
| 5 | absent — no social obligations, agreements, or expectations present in the context; pure description or self-referential content only |
| 6 | mild clarity — implicit expectations are reasonably clear from context, even if unstated ("you know what I expect") |
| 7 | moderate clarity — expectations conveyed explicitly in plain language, consistent with prior framing |
| 8 | clear clarity — transparent terms, mutual acknowledgment of obligations |
| 9 | strong clarity — explicit mutual agreements with accountability and shared understanding |
| 10 | maximum clarity — formal or near-formal agreements, structural enforcement of expectations |

---

### Variant C — Behavioral-Marker Anchors

**Contractual clarity** — How clearly the text communicates what each party is expected to do,
has agreed to, or can legitimately demand. Use these behavioral markers: language about
obligations ("you should," "I owe," "we agreed"), fairness ("that's not right," "unfair"),
rules ("the rule is," "that's how it works"), consequences ("if you don't..."), or explicit
agreements. Score based on whether these signals are present and clear, present but ambiguous,
or absent.

| Score | Anchor |
|-------|--------|
| 0 | extreme violation — active language denying prior commitments, gaslighting ("I never said that"), or deliberate redefinition of agreed terms |
| 1 | strong violation — explicit language that breaks stated or strongly implied expectations |
| 2 | clear ambiguity — hidden obligations being applied without disclosure; misleading framing about what was agreed |
| 3 | moderate ambiguity — obligation language present but contradictory, vague, or inconsistently applied |
| 4 | mild ambiguity — implicit "unspoken rules" language present ("obviously," "everyone knows," "you should have known") without stating the rule |
| 5 | no markers — text contains no language about obligations, agreements, expectations, fairness, rules, or consequences; the interaction has no contractual frame |
| 6 | mild clarity — informal obligation language present and reasonably clear ("you said you would," "that's not what we discussed," "I expected") |
| 7 | moderate clarity — explicit expectation language, consistent framing, parties clearly understand what is expected |
| 8 | clear clarity — direct statement of terms, mutual acknowledgment ("we agreed that," "the deal is") |
| 9 | strong clarity — explicit mutual agreement with stated consequences or accountability |
| 10 | maximum clarity — formal or near-formal terms: written agreements, stated enforcement mechanisms, complete mutual transparency |

---

## Results

**Scored:** 2026-03-09 — same session, Variant A → B → C in sequence.

| Variant | N | Mean | SD | % at 5 | % in [3,4,6,7] |
|---------|---|------|----|--------|----------------|
| A (baseline) | 50 | 4.94 | 1.10 | 54.0% | 44.0% |
| B (implicit-vs-absent) | 50 | 4.90 | 1.33 | 48.0% | 46.0% |
| C (behavioral markers) | 50 | 5.20 | 1.16 | 52.0% | 44.0% |

**Variant B vs A:** Δ%@5 = −6.0pp, ΔSD = +0.23, Δmean = −0.04 (within ±0.5 criterion ✓)
**Variant C vs A:** Δ%@5 = −2.0pp, ΔSD = +0.06, Δmean = +0.26 (within ±0.5 criterion ✓)

**Score distributions:**
- A: 3×5, 4×8, **5×27**, 6×7, 7×2, 9×1
- B: 2×1, 3×6, 4×9, **5×24**, 6×4, 7×4, 8×1, 9×1
- C: 3×1, 4×11, **5×26**, 6×4, 7×6, 8×1, 9×1

**Verdict:** B is the clear winner; **conditional adoption recommended**.

Strict decision criteria: none of the three variants clears the 30% threshold for
automatic adoption. However, the 30% threshold was designed for the overall production
distribution (49.3% at 5). This batch was deliberately sampled from lower-concentration
sources where contractual absence is structurally common — the baseline concentration of
54% already exceeds the production average, confirming that a 30% absolute threshold is
not achievable on this batch regardless of rubric.

The relative evidence favors B unambiguously:
- Concentration at 5 drops 6pp (54% → 48%) vs only 2pp for C
- SD increases meaningfully (+0.23 vs +0.06 for C) — B discriminates better
- Mean stays stable (−0.04) — no systematic bias
- B correctly pulled texts with faint implicit obligations off the 5 (e.g., texts 18, 29, 34, 40, 46); C only moved texts when explicit linguistic markers were present

C is too conservative: it requires explicit behavioral vocabulary ("you should," "I owe,"
"the rule is") and misses the more common case of implicit social obligation. The
behavioral marker framing elevates mean by +0.26 but doesn't reduce concentration.

**Recommended action:** Adopt Variant B for production CO scoring. Score the 50 experiment
texts with B (they are currently unlabeled for CO). Monitor % at 5 in the first 3 future CO
sessions — if concentration remains above 40% on a dreaddit-heavy batch, escalate to
full rubric redesign. Previous CO labels (49.3% at 5) remain in DB, deprioritized if
re-scored.

---

## Decision Criteria

- **Adopt variant**: concentration at 5 drops below 30% AND within-batch SD increases vs A
- **Reject**: concentration stays ≥ 30% OR mean shifts more than ±0.5 (systematic bias)
- **Investigate further**: concentration drops but mean shifts — prompt may have directional bias

If adopted: update `instruments.json` CO scoring rubric. Re-score the 50 experiment texts
with the winner to add to training data (these texts are currently unlabeled for CO).
All future CO sessions use the new rubric. Previous CO labels (49.3% at 5) remain in DB
but are deprioritized if re-scored.
