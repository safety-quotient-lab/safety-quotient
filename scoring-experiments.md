# PSQ Scoring Experiments: Protocols

**Created:** 2026-02-28
**Revised:** 2026-02-28 (v2 — addressed confounds: stale control, missing test-retest, familiarity contamination, rescaling artifacts, construct redefinition)
**Purpose:** Controlled experiments to reduce rubric-induced halo in LLM scoring.

## Baseline Measurements (held-out set, N=100, separated-llm)

| Metric | Value |
|---|---|
| Within-text SD | 0.683 (mean), 0.685 (median) |
| Mean inter-dimension \|r\| | 0.656 |
| Pairs with \|r\| > 0.7 | 21/45 (47%) |
| Most concentrated dim | CO: 66% exact-5, sd=1.02 |
| Second most concentrated | AD: 58% exact-5, sd=1.28 |
| Third most concentrated | DA: 56% exact-5, sd=1.15 |

**Target:** Within-text SD > 0.85, mean \|r\| < 0.50, exact-5 rate < 40% for all dims.

---

## Design Principles

### 1. Fresh controls, not stale gold labels

The held-out gold labels were scored weeks ago, possibly by a different Claude version in a different session context. Session-to-session variability (model sampling, context effects, version drift) is a confound if we compare treatment scores against old labels.

**Solution:** Every experiment scores the same texts under BOTH conditions — control (current prompt/rubric) and treatment — in the same experimental window. Old gold labels are used only for construct stability checks, not as the primary comparator.

### 2. Test-retest baseline first

Before testing any intervention, we must establish the noise floor: how much do scores vary when we re-score with the identical prompt? Any treatment effect must exceed this baseline variability to be meaningful.

### 3. Non-overlapping text sets

Scoring the same texts repeatedly across experiments creates familiarity contamination — the scorer anchors to prior sessions' scores. Each experiment uses a **different text set** drawn from the unlabeled pool.

Exception: Phase 0 (test-retest) uses 20 held-out texts to validate against existing gold labels.

### 4. Information-theoretic metrics over raw SD

Comparing within-text SD across different scales (1-5 vs 0-10 vs 0-100) is misleading because rescaling changes SD mechanically. Use scale-invariant metrics instead:

- **Entropy per dimension:** How many effective score levels are used? `H = -Σ p(s) log₂ p(s)` for each dimension's score distribution.
- **Eigenvalue ratio:** First eigenvalue / total variance from PCA on the N×10 score matrix. Lower = less g-factor dominance. Scale-invariant when computed on standardized scores.
- **Rank-order agreement:** Spearman ρ between control and treatment (rank-based, scale-invariant).

### 5. Criterion validity gate

Reducing halo is only valuable if predictive power is maintained. 67.3% of the shared variance is real g-factor — destroying it could worsen criterion validity. Any adopted intervention must pass a **criterion validity replication** before production use: re-run CaSiNo satisfaction prediction (AUC ≥ 0.58, current = 0.599) or CGA-Wiki derailment prediction (AUC ≥ 0.57, current = 0.599) with scores from the new protocol.

### 6. Distinguishing halo reduction from construct redefinition

If we change rubric anchors and scores change, two explanations are possible:
- (a) We reduced halo (same construct, less scorer bias)
- (b) We redefined the construct (different construct, different scores)

**Diagnostic:** Check the 7 unmodified dimensions. If changing 3 rubrics also increases differentiation in the other 7, that's evidence of halo reduction (contagion effect). If only the 3 modified dims change, it's more likely construct redefinition.

---

## Metrics (computed for every experiment)

| # | Metric | What it measures | Scale-invariant? |
|---|---|---|---|
| M1 | Within-text SD | Dimension differentiation per text | No (use for same-scale comparisons only) |
| M2 | Mean inter-dimension \|r\| | Halo magnitude | Yes (correlation is unitless) |
| M3 | Exact-neutral rate | Anchoring to neutral point | No (but comparable at same scale) |
| M4 | Control-treatment ρ | Construct stability (Spearman) | Yes |
| M5 | Eigenvalue ratio (EV1/total) | g-factor dominance | Yes (computed on z-scored data) |
| M6 | Effective information (entropy) | Score resolution | Yes |

### Analysis Script

```python
# scripts/scoring_experiment_analysis.py (to be written)
# Input: control.jsonl, treatment.jsonl (same texts, same format)
# Output: comparison table for M1-M6, per-dimension breakdowns, significance tests
# Includes: paired t-test on within-text SD, Fisher z-test on correlations
```

---

## Phase 0: Test-Retest Baseline

**Purpose:** Establish the noise floor for LLM scoring variability. How much do scores change when we re-score with the *identical* prompt?

**Why this must come first:** If test-retest variability is high (e.g., within-text SD differs by 0.15 between sessions), then only treatment effects exceeding that threshold are meaningful. If it's low (SD difference < 0.05), we have high power to detect small effects.

### Procedure

1. Take 20 held-out texts (texts 0-19, 4 per source)
2. Score all 10 dimensions using the **current production prompt** (no changes)
3. One dimension per session (standard separated protocol)
4. Compare retest scores to existing gold labels

### Outputs

| Metric | Description |
|---|---|
| Test-retest r (per dim) | Pearson r between gold and retest scores |
| Mean absolute difference | Average \|gold - retest\| per dimension |
| Within-text SD difference | \|SD_gold - SD_retest\| per text |
| ICC(3,1) per dim | Single-measures intra-class correlation (consistency) |

### Decision

- If test-retest r < 0.70 for most dimensions: **stop** — scoring is too unstable for intervention experiments. Investigate sources of instability first.
- If test-retest r ≥ 0.80 for most dimensions: **proceed** — scoring is stable enough. Use the mean absolute difference as the noise floor for interpreting treatment effects.
- Record the test-retest within-text SD difference as **Δ_noise**. Treatment effects must exceed 2 × Δ_noise to be considered real.

### Effort

- 10 sessions × 20 texts = 200 scorings total
- ~10 min per dimension = ~100 min total

---

## Experiment 1: Halo-Awareness Instructions (Avenue 5)

**Hypothesis:** Adding explicit halo-awareness text to the scoring prompt will increase within-text SD by reducing global anchoring.

**Literature:** Sulsky & Day (1994) — frame-of-reference training reduces halo in human raters. Counter-evidence: Westbury & King (2024) — LLM halo arises from distributional co-occurrence, not amenable to instruction.

### Text Set

30 texts from `data/unlabeled-pool.jsonl` (never previously scored). Stratified: 6 per source dataset. Selected by hash to avoid cherry-picking.

### Conditions

**Control prompt** (current production):
> "Score each text on [Dimension Name] only (0-10 scale). 5 = neutral (no signal)."

**Treatment prompt** (halo-aware):
> "Score each text on [Dimension Name] only (0-10 scale). 5 = neutral (no signal). IMPORTANT: This dimension is independent of overall text quality. A highly threatening text can have excellent contractual clarity. A calm, well-regulated text can have no contractual content at all. Score ONLY [Dimension Name] — ignore your impression of other dimensions."

### Procedure

1. Extract 30 texts into experiment batch file
2. Score all 10 dimensions with **control prompt** (1 dim per session)
3. Score all 10 dimensions with **treatment prompt** (1 dim per session, fresh sessions)
4. Compare control vs treatment on M1-M6

**Order counterbalancing:** Score control dimensions in order TE→HI→AD→ED→RC→RB→TC→CC→DA→CO. Score treatment dimensions in reverse order CO→DA→CC→TC→RB→RC→ED→AD→HI→TE. This prevents systematic order-of-dimension effects.

### Decision Criteria

- Treatment within-text SD exceeds control by > 2 × Δ_noise AND > 15% relative: **adopt**
- Treatment within-text SD exceeds control by > Δ_noise but < 15%: **note** (combine with other interventions)
- Treatment within-text SD within Δ_noise of control: **reject** (confirms Westbury & King)
- **Hard constraint:** Control-treatment ρ per dimension ≥ 0.70 (we're measuring the same construct)

### Effort

- 20 scoring sessions (10 control + 10 treatment × 30 texts each)
- ~15 min per dimension = ~300 min total

---

## Experiment 2: Structurally Dissimilar Rubrics (Avenue 2)

**Hypothesis:** Rewriting rubric anchors to use dimension-specific behavioral features (instead of the universal safe↔unsafe valence axis) will increase within-text SD by breaking isomorphic rubric structure.

**Literature:** Humphry & Heldsinger (2014) — structurally aligned rubric categories cause halo.

### Text Set

30 **different** texts from `data/unlabeled-pool.jsonl` (no overlap with Experiment 1). Stratified by source.

### Current Rubric Problem

All 10 rubrics follow the same template:
- 0 = "extreme [bad thing]"
- 5 = "neutral — no [dimension] signals"
- 10 = "maximum [good thing]"

This creates a single valence axis: bad↔good. The scorer forms a global impression and maps it to all 10 scales with minor adjustments.

### Treatment: Dissimilar Rubric Anchors

Rewrite anchors for 3 target dimensions (CO, ED, AD) to use **dimension-specific behavioral axes** rather than the universal valence axis. The other 7 dimensions keep current rubrics.

**Key design constraint:** The *construct definitions* stay identical. Only the *anchor language structure* changes. This means:
- Same dimension descriptions
- Same score direction (low = negative, high = positive)
- Different anchor vocabulary: behavioral features instead of valence adjectives
- Different conceptual framing: "what is present/absent" instead of "how good/bad"

#### Contractual Clarity — Rewritten

**New axis:** Absence of contractual content ↔ Presence of explicit contractual content

| Score | Current Anchor | Proposed Anchor |
|---|---|---|
| 0 | extreme violation — gaslighting, total term-shifting | Active contract destruction: explicit breaking of stated terms, denial of prior agreements, deliberate redefinition of what was promised |
| 2 | clear ambiguity — hidden agendas, unstated rules | Implicit expectations operating: rules exist but are unstated, obligations are assumed rather than articulated, terms could be read multiple ways |
| 5 | neutral — no contractual signals | No contractual content: text contains no agreements, expectations, obligations, or terms — the topic is unrelated to any form of contract |
| 8 | clear clarity — transparent terms, mutual understanding | Explicit bilateral terms: both parties' expectations are stated, conditions are named, consequences for breach are specified or implied |
| 10 | maximum clarity — exemplary contractual transparency | Fully operationalized agreement: terms defined, measurable, time-bound, with explicit enforcement mechanism and mutual consent documented |

**What changed:** Anchor vocabulary is now dimension-specific (terms, agreements, expectations) rather than generic quality words (clarity, ambiguity, transparency). Score 5 explicitly means "no contractual content present," not "neutral quality."

#### Energy Dissipation — Rewritten

**New axis:** Energy accumulation/trapping ↔ Energy release/processing pathways

| Score | Current Anchor | Proposed Anchor |
|---|---|---|
| 0 | extreme entrapment — traps all energy with no outlet | Pressure sealed: text describes/creates a situation where emotional or psychological energy has no outlet — suppression demanded, expression punished, no exit available |
| 2 | clear entrapment — blocks healthy dissipation pathways | Accumulation without release: stress or emotional energy is building but healthy outlets are absent, blocked, or stigmatized |
| 5 | neutral — no energy impact | No energy dynamics: text does not engage with emotional energy, stress, or coping — topic is energetically inert |
| 8 | clear dissipation — actively supports recovery | Active processing pathways: text describes or creates space for emotional expression, physical release, creative processing, or reflective rest |
| 10 | maximum dissipation — exemplary support for rest | Full discharge and renewal: comprehensive support for energy release through multiple channels (expression + rest + meaning-making), resulting in restored capacity |

#### Authority Dynamics — Rewritten

**New axis:** Power concentration/exercise ↔ Power distribution/accountability

| Score | Current Anchor | Proposed Anchor |
|---|---|---|
| 0 | extreme abuse — unchecked power wielded to harm | Total power monopoly: one party holds all definitional authority, controls the interaction, and faces no accountability — the other party's reality, needs, and voice are treated as nonexistent |
| 2 | clear imbalance — authority used to dismiss | Unilateral authority: one party sets the terms, defines what counts as valid, and dismisses or overrides the other — power asymmetry is operating but not necessarily malicious |
| 5 | neutral — no power dynamics present | No power dynamics: text contains no status negotiation, hierarchy, deference, dominance, or authority structures — participants are not in a power relationship |
| 8 | clear equity — actively distributes power | Distributed authority: multiple parties contribute to defining reality, decision-making is shared, disagreement is permitted without penalty, expertise is acknowledged across participants |
| 10 | maximum equity — exemplary power-sharing | Accountable collaborative authority: power is explicitly shared with structural mechanisms for challenge, correction, and rotation — authority is exercised transparently with consent |

### Conditions

**Control:** Current rubrics (all 10 dims), current prompt
**Treatment:** 3 modified rubrics (CO, ED, AD) + 7 current rubrics, current prompt

### Procedure

1. Create `instruments-experiment-2.json` with the 3 rewritten rubrics
2. Extract 30 texts into experiment batch file
3. Score all 10 dimensions with **control** rubrics (1 dim per session)
4. Score all 10 dimensions with **treatment** rubrics (1 dim per session, fresh sessions)
5. Compare control vs treatment on M1-M6

**Important:** We score ALL 10 dimensions under both conditions. The 7 unmodified dims serve as a diagnostic: if they also show increased differentiation, that's halo reduction (contagion effect). If only the 3 modified dims change, it's more likely construct redefinition.

### Decision Criteria

- Primary: Treatment within-text SD exceeds control by > 2 × Δ_noise AND > 20% relative: **adopt** the new rubric structure
- Diagnostic: If 7 unmodified dims also show > 10% SD increase: strong evidence of halo reduction
- Diagnostic: If only 3 modified dims change: likely construct redefinition — proceed cautiously
- **Hard constraint:** Control-treatment ρ per dimension ≥ 0.60 for modified dims, ≥ 0.70 for unmodified dims
- **Criterion validity gate (post-adoption only):** Before deploying new rubrics to production labeling, re-run CaSiNo criterion validity. AUC must remain ≥ 0.58.

### Effort

- 20 scoring sessions (10 control + 10 treatment × 30 texts each)
- ~300 min total

---

## Experiment 3: Scale Format Comparison (Avenue 1)

**Hypothesis:** Coarser scales force categorical distinctions rather than fine-grained adjustments from a global anchor, reducing halo.

**Literature:** Preston & Colman (2000) — reliability peaks at 7-10 categories. Li et al. (2026) — 0-5 yields highest LLM-human alignment.

### Text Set

20 texts from `data/unlabeled-pool.jsonl` (no overlap with Experiments 1 or 2). **Pilot size** — if results warrant, expand to 30.

### Conditions

Score the same 20 texts at 3 scales. The 0-10 control is scored fresh (not reused from other experiments).

#### Scale A: 0-10 (control)

Current production scale. Scored with current prompt and rubrics.

#### Scale B: 1-5

| Score | Meaning |
|---|---|
| 1 | strong negative signal |
| 2 | moderate negative signal |
| 3 | neutral / absent |
| 4 | moderate positive signal |
| 5 | strong positive signal |

#### Scale C: 1-7

| Score | Meaning |
|---|---|
| 1 | strong negative |
| 2 | clear negative |
| 3 | mild negative |
| 4 | neutral / absent |
| 5 | mild positive |
| 6 | clear positive |
| 7 | strong positive |

### Metrics (scale-invariant only)

Raw within-text SD is **not comparable** across scales due to mechanical rescaling artifacts. Use only:

- **M2:** Mean inter-dimension |r| (correlation is scale-invariant)
- **M5:** Eigenvalue ratio on z-scored data (standardization removes scale effects)
- **M6:** Entropy per dimension (measures how many effective levels are used)
- **Cross-scale ρ:** Spearman rank correlation between 0-10 and alternative scale scores per dimension (construct stability)

### Procedure

1. Add `--scale 5` and `--scale 7` flags to `label_separated.py` extract
2. Score 20 texts × 10 dims at 0-10 (control)
3. Score 20 texts × 10 dims at 1-5
4. Score 20 texts × 10 dims at 1-7
5. Compare M2, M5, M6 across scales

**Order:** Score 0-10 first, then 1-7, then 1-5. (We don't counterbalance scale order because the scales themselves are the treatment — the text-familiarity confound is acknowledged and noted as a limitation.)

### Decision Criteria

- If an alternative scale beats 0-10 on M2 (lower |r|) AND M5 (lower EV ratio) AND cross-scale ρ ≥ 0.70: **switch** future labeling
- If 0-10 is best or tied on M2 and M5: **retain** (our current format is near-optimal, consistent with prediction)
- **0-100 excluded:** Already confirmed to collapse dimensions (EV1 = 94.1% in pct-scored FA). No need to re-test.

### Effort

- 3 scales × 10 dims × 20 texts = 30 scoring sessions
- ~150 min per scale = ~450 min total
- **Pilot gate:** After scoring 1-7, compare to 0-10. If \|r\| difference < 0.05, skip 1-5 (diminishing returns from coarser scales).

---

## Execution Order

```
Phase 0: Test-Retest Baseline (20 texts, ~100 min)
    ↓ establishes Δ_noise
    ↓ go/no-go: test-retest r ≥ 0.80?

Experiment 1: Halo-Awareness (30 texts, ~300 min)
    ↓ uses different text set

Experiment 2: Dissimilar Rubrics (30 texts, ~300 min)
    ↓ uses different text set

Experiment 3: Scale Format (20 texts, ~450 min, pilot)
    ↓ uses different text set

Combinatorial: best interventions combined (if any positive)
    ↓
Criterion Validity Replication (CaSiNo, ~100 texts)
    ↓ go/no-go: AUC ≥ 0.58?

Production Adoption
```

**Total effort (all phases):** ~1,150 min (~19 hours of scoring across ~70 sessions)

**Minimum viable path:** Phase 0 + Experiment 1 only (~400 min, ~7 hours). If Exp 1 is null, Exp 2 becomes the priority.

---

## Text Set Allocation

To prevent familiarity contamination, each phase uses non-overlapping texts:

| Phase | Source | Text IDs | N |
|---|---|---|---|
| Phase 0 (test-retest) | held-out-test.jsonl | 0-19 | 20 |
| Experiment 1 | unlabeled-pool.jsonl | hash-selected set A | 30 |
| Experiment 2 | unlabeled-pool.jsonl | hash-selected set B | 30 |
| Experiment 3 | unlabeled-pool.jsonl | hash-selected set C | 20 |

Text selection script (to be written): deterministic hash-based sampling from unlabeled pool, stratified by source, excluding any text already in psq.db.

---

## Status

| Phase | Protocol | Scoring | Analysis | Decision |
|---|---|---|---|---|
| 0. Test-Retest | DESIGNED | Not started | - | - |
| 1. Halo-Awareness | DESIGNED | Not started | - | - |
| 2. Dissimilar Rubrics | DESIGNED | Not started | - | - |
| 3. Scale Format | DESIGNED | Not started | - | - |
