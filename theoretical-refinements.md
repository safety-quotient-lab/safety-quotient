# PSQ Theoretical Refinements

**Date:** 2026-02-26
**Status:** Proposals based on empirical findings from distillation research
**Context:** These refinements are informed by pairwise dimension correlations (§8x), error analysis (§8t), proxy audit findings (§8z), and the psychometric evaluation.

---

## 1. Dimension Reduction: 10 → 8 Factor Model

### Empirical Basis

Pairwise correlations computed across 17,643+ scored texts reveal three pairs with r > 0.78:

| Pair | r | n | Theoretical Overlap |
|---|---|---|---|
| Regulatory Capacity ↔ Resilience Baseline | 0.877 | 3,932 | Both measure internal capacity to manage emotional disruption |
| Hostility Index ↔ Cooling Capacity | 0.840 | 3,949 | Hostile content inherently lacks de-escalation; these are mirror images |
| Authority Dynamics ↔ Trust Conditions | 0.787 | 3,949 | Power asymmetry and trust are deeply entangled |

### Proposed 8-Factor Model

**Merge 1: Regulatory Capacity + Resilience Baseline → "Regulatory Resilience"**

*Rationale:* Both dimensions measure the same underlying construct — the internal capacity to manage emotional disruption and return to equilibrium. The ERQ/DERS instruments (regulation) and CD-RISC/BRS (resilience) measure overlapping phenomena: Can this person (or does this content support the ability to) absorb emotional impact and recover? The distinction between "regulation" (in-the-moment modulation) and "resilience" (return to baseline over time) is theoretically clean but empirically inseparable at the content level. A text that supports emotional regulation also supports resilience, and vice versa.

*New definition:* **Regulatory Resilience** — the degree to which content supports (or undermines) the reader's capacity to modulate emotional states in the moment and recover functional equilibrium over time. High scores indicate content that acknowledges difficulty while supporting adaptive coping; low scores indicate content that overwhelms regulatory capacity with unrelenting demand, invalidation, or emotional flooding.

*Instruments:* ERQ, DERS, CERQ (regulation side) + CD-RISC, BRS, Grit Scale (resilience side) — now unified under one dimension.

*Protective factor. Score 0–10.*

**Merge 2: Hostility Index + Cooling Capacity → "Hostility-Recovery Balance"**

*Rationale:* These are conceptual mirror images. Hostile content (high hostility) inherently prevents de-escalation (low cooling). The error analysis confirms they share 70.6% of variance (r²=0.706). In practice, scoring them separately doubles the measurement effort for nearly the same construct: "How much does this content inflame vs. de-escalate?"

However, merging a threat factor (hostility) with a protective factor (cooling) requires rethinking the PSQ formula. Option: score the merged dimension on a single axis where 0 = maximum hostility with zero recovery pathways, 10 = zero hostility with strong de-escalation support, 5 = neutral.

*New definition:* **Hostility-Recovery Balance** — the balance between inflammatory/hostile signals and de-escalation/recovery mechanisms in the content. Low scores indicate active hostility, escalation triggers, and absence of cooling pathways. High scores indicate absence of hostility with active de-escalation support, cognitive reframing, and temporal buffers.

*Instruments:* Cook-Medley, BPAQ, STAXI-2 (hostility side) + CPI, Gross reappraisal, REQ (cooling side).

*Threat factor (inverted). Score 0–10.*

**Keep Separate: Authority Dynamics and Trust Conditions (r=0.787)**

*Rationale:* Despite high correlation, these dimensions are conceptually distinct in ways that matter for the PSQ's juris pillar. Authority dynamics measures how power is *exercised* — distributed vs. coercive, checked vs. unchecked. Trust conditions measures whether the environment is *safe for vulnerability* — consistency, transparency, promise-keeping. You can have high authority abuse with high trust (authoritarian but reliable: "The boss is harsh but fair") or healthy authority with low trust (egalitarian but inconsistent). The juris framework requires distinguishing power structure from relational safety.

The high correlation likely reflects that in *most* real-world text, abusive authority co-occurs with broken trust. But they have different intervention targets: authority dynamics are structural (power distribution, accountability), while trust conditions are relational (consistency, vulnerability safety).

### Resulting 8-Factor Structure

**Protective Factors (5 → 4):**
1. Regulatory Resilience *(was: Regulatory Capacity + Resilience Baseline)*
2. Trust Conditions
3. Cooling *(absorbed into Hostility-Recovery Balance)*
4. Energy Dissipation
5. Defensive Architecture
6. Contractual Clarity

Wait — merging hostility + cooling collapses a threat + protective pair. Let me reconsider.

**Revised: Keep cooling separate. Only merge regulatory + resilience.**

The hostility↔cooling correlation (0.840) may reflect proxy methodology (both from UCC) more than true construct overlap. And the PSQ formula structurally separates threat and protective factors — merging across that boundary creates complications.

### Recommended 9-Factor Model

| # | Dimension | Type | Source Dimensions |
|---|---|---|---|
| 1 | Threat Exposure | Threat | (unchanged) |
| 2 | Hostility Index | Threat | (unchanged) |
| 3 | Authority Dynamics | Threat | (unchanged) |
| 4 | Energy Dissipation | Threat | (unchanged) |
| 5 | **Regulatory Resilience** | Protective | Regulatory Capacity + Resilience Baseline |
| 6 | Trust Conditions | Protective | (unchanged) |
| 7 | Cooling Capacity | Protective | (unchanged) |
| 8 | Defensive Architecture | Protective | (unchanged) |
| 9 | Contractual Clarity | Protective | (unchanged) |

This is the most conservative reduction supported by the data. The r=0.877 between regulatory capacity and resilience baseline is high enough to justify merging. The other high-correlation pairs have sufficient theoretical distinctiveness to retain.

**Impact on PSQ formula:** `PSQ = ((protective_avg - threat_avg + 10) / 20) × 100` stays the same, but protective_avg is now over 5 factors (not 6) and threat_avg over 4 (unchanged). This slightly increases the weight of each protective factor.

**Important:** This proposal should be tested empirically — score 500+ texts on all dimensions simultaneously and run confirmatory factor analysis (CFA) comparing the 10-factor, 9-factor, and 8-factor models.

---

## 2. Defensive Architecture Redefinition

### The Problem

Defensive architecture has the lowest model performance across all versions (v2d r=0.125 on full dataset, v3b epoch 3 r=0.28). The psychometric evaluation (§5) identified the root cause: defense mechanisms as defined by DSQ-40, DMRS, and Vaillant's hierarchy are *intrapsychic processes* — they operate unconsciously to manage anxiety. Text provides behavioral markers but cannot access:

- Whether patterns are conscious or unconscious
- Whether they're ego-syntonic or ego-dystonic
- Whether they're situational or characterological

### Current Definition (Too Clinical)

> Defensive Architecture: The available protective mechanisms — boundaries, assertive communication, withdrawal options, advocacy structures, legal protections.

The instruments (DSQ-40, DMRS, Vaillant hierarchy) focus on clinical defense mechanisms: projection, denial, sublimation, intellectualization, splitting. These require trained clinician observation or self-report access that text analysis cannot provide.

### Proposed Redefinition: "Boundary & Protection Patterns"

Shift from unmeasurable clinical defense mechanisms to text-observable boundary and protection signals:

> **Boundary & Protection Patterns** — the degree to which content supports, respects, or undermines the establishment and maintenance of interpersonal boundaries and self-protective behaviors. High scores indicate content that validates boundary-setting, respects limits, enables assertiveness, and permits withdrawal without punishment. Low scores indicate content that shames self-protection, violates stated boundaries, punishes assertiveness, or strips protective options.

### Revised Indicators (Text-Observable)

**High-scoring indicators (7-10):**
- Explicit boundary language: "I need X," "That's not okay with me"
- Assertiveness validation: "You have every right to say no"
- Withdrawal permission: "Take the time you need"
- Advocacy enablement: "Here's how to get support"
- Limit acknowledgment: "I respect that boundary"
- Protective option preservation: "You can always leave/report/say no"

**Low-scoring indicators (0-3):**
- Boundary violation: ignoring stated limits, pushing past "no"
- Defense shaming: "You're too sensitive," "Stop being so defensive"
- Self-protection punishment: retaliation for setting limits
- Limit dismissal: "That's a ridiculous boundary"
- Protective option removal: isolation, dependency creation
- Guilt for self-care: "If you really cared, you wouldn't need space"

**Neutral (4-6):**
- No boundary-relevant content
- Factual discussion without interpersonal protection signals

### Revised Instruments

*Remove:* DSQ-40, DMRS, Vaillant hierarchy (require intrapsychic access)

*Retain:* TKI (Thomas & Kilmann 1974, conflict handling modes — observable in text), ROCI-II (Rahim 1983, organizational conflict — observable), Nonviolent Communication (Rosenberg 2003, explicitly about boundary language)

*Add:*
- Interpersonal Boundaries Scale (Gierveld & Tilburg 2006) — measures boundary maintenance in relationships
- Assertiveness Scale (Rathus 1973) — behavioral assertiveness markers
- HealMe cognitive distortion labels — observable distortion patterns in text

### Expected Impact

Redefining from clinical defense mechanisms to observable boundary patterns should:
1. Raise the psychometric ceiling (defense mechanisms: r≈0.40 max from text; boundary patterns: r≈0.65+ achievable)
2. Improve LLM labeling consistency (clearer rubric)
3. Better serve the PSQ's content-evaluation purpose (what does this text *do* to boundaries?)
4. Align with what the 450 existing LLM labels actually measure (they already focus on boundaries, not unconscious mechanisms)

---

## 3. Score Anchors: Concrete Examples Per Dimension

Each dimension's 0-10 scale needs calibrated anchor examples at 0, 2.5, 5, 7.5, and 10 to improve scoring consistency across LLM runs, human raters, and student model targets.

### 3a. Threat Exposure

| Score | Anchor | Example |
|---|---|---|
| 0 | Maximum threat, sustained + severe | "I know where you live. Every day you don't comply, it gets worse for you and your family." |
| 2.5 | Moderate threat, intermittent | "You better not mess this up again or there will be consequences." |
| 5 | Neutral — no threat signals | "The quarterly report is due next Friday. Please submit your sections by Wednesday." |
| 7.5 | Low threat, protective signals | "This is a safe space. No judgement here — share what you're comfortable with." |
| 10 | Zero threat, explicit protection | "Nothing you say here will be used against you. This conversation is confidential and you can stop at any time." |

### 3b. Regulatory Capacity

| Score | Anchor | Example |
|---|---|---|
| 0 | Overwhelms regulation | "You should be ashamed. Everyone is laughing at you. There's no excuse for what you did and you know it." |
| 2.5 | Strains regulation | "I can't believe you would do something so stupid. What were you thinking?" |
| 5 | Neutral | "Here's the update on the project timeline. We're on track for the March milestone." |
| 7.5 | Supports regulation | "I can see this is frustrating. Let's break it down into smaller pieces and tackle one thing at a time." |
| 10 | Actively builds regulation | "It's completely normal to feel overwhelmed right now. What helped you cope last time something like this happened?" |

### 3c. Resilience Baseline

| Score | Anchor | Example |
|---|---|---|
| 0 | Undermines resilience | "You'll never recover from this. This failure defines you. Everyone will remember." |
| 2.5 | Discourages resilience | "Some people just aren't cut out for this. Maybe you should give up." |
| 5 | Neutral | "The meeting has been moved to 3pm. Room 204." |
| 7.5 | Supports resilience | "This is a setback, not the end. You've gotten through harder things before." |
| 10 | Actively builds resilience | "What you went through was genuinely difficult, and the fact that you're here talking about it shows real strength. What did you learn from it?" |

### 3d. Trust Conditions

| Score | Anchor | Example |
|---|---|---|
| 0 | Destroys trust | "I read your private messages and shared them with the team. You should be more careful what you write." |
| 2.5 | Erodes trust | "I said I'd help, but honestly, it's not really my problem. Figure it out yourself." |
| 5 | Neutral | "The store closes at 9pm on weekdays and 6pm on weekends." |
| 7.5 | Builds trust | "I appreciate you telling me that. I won't share it with anyone else unless you want me to." |
| 10 | Maximum trust | "I made a mistake last week and I want to own it. Here's what happened, here's what I should have done, and here's what I'll do differently." |

### 3e. Hostility Index

| Score | Anchor | Example |
|---|---|---|
| 0 | Maximum hostility | "You worthless piece of garbage. I hope you suffer. People like you don't deserve to exist." |
| 2.5 | Moderate hostility | "Your opinion is idiotic and you clearly have no idea what you're talking about." |
| 5 | Neutral | "I picked up groceries on the way home. Dinner should be ready by seven." |
| 7.5 | Warm/supportive | "I really enjoyed our conversation. You have such a thoughtful perspective on things." |
| 10 | Maximum warmth | "You are deeply valued here. Your contributions make everyone's work better, and I'm grateful you're on this team." |

### 3f. Cooling Capacity

| Score | Anchor | Example |
|---|---|---|
| 0 | Maximum escalation | "Don't you DARE walk away from this! We're settling this RIGHT NOW. You don't get to run from consequences!" |
| 2.5 | Promotes escalation | "You always do this! You're just like your mother — impossible to reason with!" |
| 5 | Neutral | "The next bus arrives in 12 minutes according to the schedule." |
| 7.5 | Supports de-escalation | "I can see we're both getting heated. Can we take a break and come back to this in an hour?" |
| 10 | Maximum cooling | "I understand you're angry, and you have reason to be. Let's take a moment to breathe. When you're ready, I'd like to understand your perspective fully." |

### 3g. Energy Dissipation

| Score | Anchor | Example |
|---|---|---|
| 0 | Traps energy | "There's no time to rest. You have to keep going. People are counting on you. If you stop now, everything falls apart." |
| 2.5 | Discourages recovery | "Must be nice to take breaks. Some of us actually have work to do." |
| 5 | Neutral | "The library opens at 8am and closes at 10pm Monday through Saturday." |
| 7.5 | Supports recovery | "You've been working hard. Take the afternoon off — the work will be there tomorrow." |
| 10 | Active dissipation | "Your wellbeing matters more than the deadline. Let's talk about what you need to recharge, and I'll help redistribute the workload." |

### 3h. Defensive Architecture (Boundary & Protection Patterns)

| Score | Anchor | Example |
|---|---|---|
| 0 | Strips defenses | "You don't get to have boundaries with me. After everything I've done for you, how dare you say no?" |
| 2.5 | Shames self-protection | "You're so dramatic. It's not a big deal. Stop being so sensitive about everything." |
| 5 | Neutral | "The conference registration deadline is March 15th. Early bird pricing ends March 1st." |
| 7.5 | Supports boundaries | "If that situation feels unsafe to you, trust that feeling. You don't owe anyone an explanation for protecting yourself." |
| 10 | Maximum protection | "You have the right to say no at any point. Here are your options for reporting. Nothing will happen without your explicit consent, and there will be no retaliation." |

### 3i. Authority Dynamics

| Score | Anchor | Example |
|---|---|---|
| 0 | Maximum authority abuse | "I am the boss. You will do exactly what I say, when I say it. If you don't like it, there's the door. And don't think about complaining — I know people." |
| 2.5 | Coercive authority | "This is not a discussion. The decision is made. Your job is to execute, not to have opinions." |
| 5 | Neutral | "The new policy takes effect on April 1st. Details are in the attached document." |
| 7.5 | Distributed authority | "I have a proposal, but I want to hear everyone's input before we decide. Every perspective matters here." |
| 10 | Maximum healthy authority | "This affects all of us equally, so we'll decide together. I'll facilitate, but my vote counts the same as yours. If anyone feels pressured, we can do anonymous voting." |

### 3j. Contractual Clarity

| Score | Anchor | Example |
|---|---|---|
| 0 | Maximum ambiguity/gaslighting | "I never said that. You're imagining things. We never had that agreement — you must have misunderstood." |
| 2.5 | Shifting expectations | "Well, things change. What I expected last month isn't what I expect now. You should have anticipated that." |
| 5 | Neutral | "It rained 3.2 inches yesterday, above the seasonal average of 2.1 inches." |
| 7.5 | Clear expectations | "Here's what I need from you: the draft by Friday, with the three sections we discussed. I'll provide feedback by Monday." |
| 10 | Maximum clarity | "Let's put this in writing so we're both clear. You'll handle X, I'll handle Y. If either of us can't deliver, we'll notify the other by Z date. If we disagree, we'll use [specific process]. Does this capture your understanding?" |

---

## 4. Validation Study Design

### 4a. The Central Question

**Criterion validity asks:** Do PSQ scores predict real-world outcomes that matter?

If a team's communications consistently score PSQ 35 (low), do bad things actually happen more often than for a team scoring PSQ 70 (high)? Without this evidence, PSQ scores are theoretical — they may correlate with the construct of "psychological safety" as we define it, but we cannot claim they predict anything meaningful in the real world.

### 4b. Outcome Variables (What PSQ Should Predict)

**Tier 1 — Direct safety outcomes (highest priority):**
- Formal safety complaints or reports (HR complaints, bullying reports, hostile environment claims)
- Team member departures / turnover rate
- Sick leave / stress leave frequency
- Employee assistance program (EAP) utilization

**Tier 2 — Wellbeing and engagement outcomes:**
- Edmondson Psychological Safety Scale scores (team-level, survey-based)
- PSC-12 Psychosocial Safety Climate scores (organizational, survey-based)
- Employee engagement scores (pulse surveys)
- Self-reported wellbeing (WHO-5 or similar)
- Burnout scores (MBI — Maslach Burnout Inventory)

**Tier 3 — Performance outcomes (secondary):**
- Team productivity metrics (where available)
- Innovation indicators (idea submission rates, experiment frequency)
- Collaboration quality (peer review scores, 360 feedback)

### 4c. Study Design: Minimum Viable Validation

**Design:** Cross-sectional correlational study with prospective follow-up

**Sample:** 30-50 teams within 2-3 organizations (total n ≈ 150-300 individuals)

**Protocol:**

1. **Collect team communications** (with informed consent):
   - 4 weeks of Slack/Teams messages per team
   - Email threads (internal, work-related)
   - Meeting transcripts (if available)
   - Target: 200+ messages per team

2. **Score communications with PSQ** (student model or LLM):
   - Aggregate per-team PSQ profile (10 dimensions + overall PSQ)
   - Track weekly PSQ trajectory (is it stable or fluctuating?)
   - Flag low-PSQ outlier messages

3. **Administer surveys simultaneously:**
   - Edmondson Psychological Safety Scale (7 items, team level)
   - PSC-12 (12 items, climate level)
   - WHO-5 Wellbeing Index (5 items, individual)
   - Single-item burnout indicator

4. **Collect outcome data (retrospective + prospective):**
   - Prior 6 months: turnover, complaints, sick leave (from HR records)
   - Following 6 months: same metrics (prospective)

**Analysis:**

| Comparison | Method | Target |
|---|---|---|
| PSQ vs Edmondson scale | Pearson r, team-level | r ≥ 0.50 (convergent validity) |
| PSQ vs PSC-12 | Pearson r, org-level | r ≥ 0.40 |
| PSQ vs wellbeing (WHO-5) | Multilevel regression | Significant after controlling for demographics |
| PSQ vs turnover (6mo) | Logistic regression | Low PSQ predicts higher turnover, OR ≥ 1.5 |
| PSQ vs complaints | Poisson regression | Low PSQ predicts more complaints |
| PSQ dimensions vs outcomes | Multiple regression | At least 3 dimensions predict outcomes independently |
| PSQ beyond sentiment | Hierarchical regression | PSQ predicts outcomes after controlling for text sentiment |

**The last comparison is critical.** If PSQ scores don't predict outcomes beyond what a simple positive/negative sentiment score provides, the 10-dimension framework adds complexity without value. PSQ must demonstrate *incremental validity* over sentiment.

### 4d. Power Analysis

- Team-level correlation (PSQ vs Edmondson): for r=0.50, n=30 teams achieves power=0.80 at α=0.05
- Individual-level regression (PSQ vs wellbeing): for R²=0.10, n=200 achieves power=0.95 at α=0.05
- Turnover prediction: for OR=1.5, n=200 with 15% base rate achieves power=0.75 at α=0.05

**Minimum sample: 30 teams / 200 individuals** is feasible and statistically adequate.

### 4e. Ethical Considerations

- **Informed consent** is essential — participants must know their communications are being analyzed
- **Anonymization**: PSQ scores are aggregated at team level; no individual scoring
- **Right to withdraw**: any team member can opt out, removing their messages
- **No individual consequences**: PSQ scores must not be used for individual evaluation, hiring, or discipline
- **IRB approval** required if conducted in an academic context
- **Data minimization**: delete raw text after scoring; retain only aggregate PSQ scores

### 4f. Realistic Timeline

| Phase | Duration | Activities |
|---|---|---|
| Design + IRB | 4-8 weeks | Protocol finalization, ethics approval, organization recruitment |
| Data collection | 4-6 weeks | Communication collection + surveys |
| PSQ scoring | 1-2 weeks | Automated scoring of collected text |
| Analysis | 2-4 weeks | Statistical analysis, write-up |
| Prospective follow-up | 6 months | Collect outcome data |
| Final analysis | 2-4 weeks | Prospective outcomes analysis |
| **Total** | **~9-12 months** | First results at ~4 months, full results at ~12 months |

### 4g. What "Success" Looks Like

**Strong validation:**
- PSQ overall score correlates r ≥ 0.50 with Edmondson Psychological Safety Scale
- At least 5/10 dimensions show r ≥ 0.30 with relevant outcomes
- PSQ predicts turnover/complaints above chance (OR ≥ 1.5)
- PSQ provides incremental prediction beyond sentiment (ΔR² ≥ 0.05)

**Adequate validation:**
- PSQ overall correlates r ≥ 0.35 with established measures
- At least 3/10 dimensions predict outcomes
- PSQ provides marginal incremental prediction beyond sentiment

**Insufficient validation:**
- PSQ overall correlates r < 0.30 with established measures
- PSQ doesn't predict real-world outcomes
- PSQ adds nothing beyond sentiment → dimension structure may need fundamental revision

---

## 5. Summary of Proposals

| Proposal | Impact | Reversibility | Recommended Action |
|---|---|---|---|
| Merge regulatory + resilience (9 factors) | Moderate | Reversible (keep both scores, merge at aggregation) | Test with CFA on 500+ fully-scored texts |
| Redefine defensive architecture | High | Forward-compatible (existing labels already measure boundaries) | Update spec, retrain with clearer rubric |
| Score anchors | High | Additive (no existing work invalidated) | Add to final-state.md, use in future labeling |
| Validation study | Critical | — | Design now, execute when model is stable at r ≥ 0.60 |
