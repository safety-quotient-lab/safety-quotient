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

### Results (2026-02-28)

| Dim | Pearson r | Spearman ρ | ICC(3,1) | MAD |
|-----|-----------|------------|----------|-----|
| TE | 0.891 | 0.776 | 0.867 | 0.50 |
| HI | 0.910 | 0.870 | 0.829 | 0.65 |
| AD | 0.156 | -0.078 | 0.144 | 0.70 |
| ED | 0.832 | 0.800 | 0.837 | 0.35 |
| RC | 0.779 | 0.772 | 0.703 | 0.60 |
| RB | 0.814 | 0.848 | 0.790 | 0.45 |
| TC | 0.822 | 0.836 | 0.767 | 0.80 |
| CC | 0.820 | 0.763 | 0.804 | 0.55 |
| DA | 0.602 | 0.647 | 0.563 | 0.45 |
| CO | 0.742 | 0.704 | 0.703 | 0.35 |

**Mean Pearson r:** 0.737 (0.804 excluding AD)
**Δ_noise:** 0.011 (within-text SD: gold=0.730, retest=0.741)
**Go/no-go:** 6/10 dims ≥ 0.80 (8/10 ≥ 0.74). **Qualified GO.**

**Notes:**
- AD severely unstable (r=0.156) — pre-existing construct problem, not scoring prompt issue. Most texts score neutral on AD (limited variance).
- DA also weak (r=0.602) — consistent with known DA construct validity concern.
- Δ_noise of 0.011 means within-text SD treatment effects >0.022 are detectable.
- Mean MAD = 0.54 score points — typical test-retest noise for integer-scale LLM scoring.

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

### Results (2026-02-28)

| Metric | Control | Treatment | Change |
|--------|---------|-----------|--------|
| M1: Within-text SD | 0.542 | 0.685 | +26.4% |
| M2: Mean inter-dim \|r\| | 0.751 | 0.631 | -0.120 |
| M4: Mean control-treatment ρ | — | 0.892 | — |
| M5: Eigenvalue ratio | 78.4% | 68.6% | -9.8pp |

**Per-dimension control-treatment ρ:** TE=0.928, HI=0.882, AD=0.909, ED=0.957, RC=0.938, RB=0.844, TC=0.960, CC=0.785, DA=0.923, CO=0.791. All ≥ 0.70.

**Initial decision: ADOPT** (subsequently reversed — see Post-Hoc Analysis below).

The instruction met all pre-registered adoption criteria:
- Increased within-text SD by 26.4% (>> 2×Δ_noise of 0.022, >> 15% threshold)
- Reduced g-factor eigenvalue from 78.4% → 68.6% (meaningful dimensionality recovery)
- Reduced mean inter-dim |r| from 0.751 → 0.631
- Maintained construct stability (all ρ ≥ 0.79)

**Caveats noted at time of adoption:**
1. Single-scorer experiment — the "halo-aware" mindset is applied by the same scorer who knows the hypothesis. Cannot fully separate instruction effect from scorer intention.
2. Exact-5 rates were mixed (HI and RB increased, TC and DA decreased) — the instruction did not uniformly reduce neutral scoring.
3. Entropy mixed: DA/TC/CO/RC increased (better), HI/CC decreased (worse). Net effect is positive on structural metrics.

### Post-Hoc Analysis: The Range-Extremity Effect (2026-02-28)

Deeper investigation of the g-factor's source reversed the adoption decision. The central question: does the dominant eigenvalue reflect scorer halo (measurement artifact) or genuine co-variation (real psychological structure)? If the latter, the halo-aware instruction would be fighting real signal.

**Finding 1: The g-factor is a range/extremity effect.**

Texts were stratified by their mean score across all 10 dimensions (g):

| Text group | n | EV1 | Mean |r| | PC1 loading SD |
|---|---|---|---|---|
| Extreme (g < 3 or g > 7) | 232 | 82.8% | 0.807 | 0.023 |
| Middle (g ∈ [4, 6]) | 1,447 | 38.7% | 0.285 | 0.117 |
| Informative middle (g ∈ [3, 4.5) ∪ [5.5, 7]) | 681 | 64.2% | 0.595 | — |

Extreme texts have perfectly uniform PC1 loadings (SD = 0.023) — every dimension loads equally, indicating pure valence ("this text is uniformly threatening/safe"). Middle texts have structured PC1 loadings (SD = 0.117) — RC and RB load highest, ED lowest, indicating genuine psychological differentiation. The g-factor is substantially inflated by texts where all dimensions legitimately co-vary because the content is uniformly extreme.

**Finding 2: Text properties don't explain the g-factor.**

- Text length × g: r = 0.012 (zero correlation)
- Partialing out text length + source: EV1 drops 71.5% → 69.8% (trivial, 1.7pp)
- g-factor persists within every source dataset (54.9% – 77.4%)

The g-factor is not a source artifact, length artifact, or scoring artifact — it reflects genuine co-variation in the texts themselves.

**Finding 3: Ipsatization reveals real bipolar structure.**

After subtracting each text's mean score (removing g), mean |r| drops from 0.679 to 0.232. The residual structure is theoretically coherent:
- TE-HI: +0.43 (threat cluster)
- ED-CC: -0.51 (energy depletion vs contractual clarity — genuine tradeoff)
- RC-RB: +0.42 (internal resources cluster)

This matches the EFA 5-factor structure from §26 and is consistent with the hierarchical decomposition PSQ → clusters → dimensions.

**Finding 4: The instruction's individual effects are within noise.**

- 79% of individual scores (237/300) were unchanged between control and treatment
- Mean |Δ| per score = 0.217, which is *below* the test-retest noise floor of 0.54 (Phase 0 MAD)
- CC showed systematic +0.33 upward bias (12/14 changes are +1) — mean shift, not differentiation
- CO decoupled from other dimensions (HI-CO ρ: 0.506 → 0.111) — loss of legitimate co-variation
- ~1/3 of the headline +26.4% SD increase comes from CC bias and CO decoupling

**Finding 5: Halo-awareness would damage the hierarchical model.**

The PSQ is intended to decompose hierarchically: PSQ (g) → 2-3 factor clusters → 5 factor groups → 10 dimensions. In this model, the g-factor *is* the PSQ — it's the construct at its broadest level. A bifactor architecture (which treats g as orthogonal to group factors) would flatten this hierarchy. The halo-aware instruction, by instructing the scorer to "ignore your impression of other dimensions," fights the real co-variation that defines the top of the hierarchy.

For extreme texts (g < 3 or g > 7, n = 232), all dimensions *should* co-vary because the content is uniformly threatening or safe. Telling the scorer to ignore this is asking them to hallucinate dimension-specific signal where none exists.

**Revised decision: REJECT.** The halo-awareness instruction:
1. Fights genuine co-variation, not measurement artifact — the g-factor is primarily a range/extremity effect reflecting real textual properties
2. Produces per-score changes below the test-retest noise floor (0.217 < 0.54)
3. Introduces systematic CC bias (+0.33 mean shift) and CO decoupling
4. Would damage the hierarchical PSQ model by suppressing legitimate g-factor signal
5. Cannot distinguish halo from real co-variation at the scorer level

**Alternative adopted: Structural approaches** (see §51 in distillation-research.md):
- **Middle-g text enrichment:** Select labeling batch texts from g ∈ [3, 4.5) ∪ [5.5, 7] where dimension-specific signal is strongest, giving the student model more training signal about what makes dimensions *different*
- **Hierarchical model architecture:** Preserve PSQ → cluster → dimension decomposition rather than bifactor (which treats g and group factors as orthogonal competitors)

**Files:**
- Control: `/tmp/psq_exp1_control.jsonl`
- Treatment: `/tmp/psq_exp1_treatment.jsonl`

### Criterion Validity Gate (2026-02-28)

**Purpose:** Verify halo-aware scores still predict real-world outcomes before production deployment.

**Method:** 40 CaSiNo negotiation dialogues stratified by satisfaction outcome (15 high-sat, 15 low-sat, 10 mid-sat). All 10 dimensions scored with halo-aware prompt.

| Metric | Result | Gate |
|--------|--------|------|
| AUC (high vs low satisfaction) | 0.971 | ≥ 0.58 **PASS** |
| g-PSQ Spearman ρ with satisfaction | 0.875 | — |
| Mean dim r with satisfaction | 0.793 | — |
| Top predictor (logistic regression) | ED (+0.382) | — |

**Per-dimension correlations with satisfaction:**

| Dim | Pearson r | p-value |
|-----|-----------|---------|
| ED | 0.869 | <0.001 |
| RB | 0.854 | <0.001 |
| RC | 0.849 | <0.001 |
| TC | 0.838 | <0.001 |
| CC | 0.828 | <0.001 |
| HI | 0.812 | <0.001 |
| DA | 0.786 | <0.001 |
| TE | 0.762 | <0.001 |
| CO | 0.755 | <0.001 |
| AD | 0.685 | <0.001 |

**Interpretation:** Gate formally PASSED. However, correlations are unusually high (mean r=0.793), likely inflated by: (a) small sample (N=40), (b) same scorer aware of outcomes, (c) no cross-validation. The directional finding is strong — halo-aware scores clearly maintain criterion validity — but absolute magnitudes should not be compared to the original CaSiNo study (model-scored, N=1,030, AUC=0.599).

**Decision:** Gate formally PASSED — criterion validity preserved. However, halo-aware adoption was subsequently reversed based on structural analysis of the g-factor (see Post-Hoc Analysis above). The gate result is moot given the reversal.

**Files:**
- Texts: `/tmp/psq_casino_gate.jsonl`
- Scored: `/tmp/psq_casino_gate_scored.jsonl`

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

### Results (2026-02-28)

| Metric | Control | Treatment | Change |
|--------|---------|-----------|--------|
| M1: Within-text SD | 0.630 | 0.664 | +5.3% |
| M2: Mean inter-dim \|r\| | 0.810 | 0.793 | -0.017 |
| M3: Exact-5 rate | 39.3% | 36.0% | -3.3pp |
| M5: Eigenvalue ratio | 83.3% | 82.1% | -1.2pp |

**Per-dimension control-treatment ρ:**

| Dim | ρ | Modified? |
|-----|---|-----------|
| TE | 0.997 | No |
| HI | 0.996 | No |
| AD | 0.815 | **Yes** |
| ED | 0.896 | **Yes** |
| RC | 1.000 | No |
| RB | 1.000 | No |
| TC | 1.000 | No |
| CC | 1.000 | No |
| DA | 1.000 | No |
| CO | 0.626 | **Yes** |

**Modified dims exact-5 change:** CO 77%→53% (-24pp), AD 60%→50% (-10pp), ED 40%→37% (-3pp)
**Unmodified dims:** Essentially unchanged (ρ=1.000 for 5/7 dims)

**Diagnostic: Construct redefinition, NOT halo reduction.**
- Only modified dims changed; unmodified dims are identical
- No contagion effect: unmodified dims SD *decreased* by 3.7%
- Modified-Unmodified pair |r| decreased (0.745→0.697) while Unmod-Unmod stayed flat (0.893→0.898)
- The dissimilar rubrics changed what was being measured, not how it was measured

**Decision: REJECT.** The dissimilar rubrics redefine the constructs rather than reducing halo. Within-text SD increase of 5.3% is well below the 20% adoption threshold. CO's ρ=0.626 is borderline — the rewritten anchors are measuring a slightly different concept. No evidence that rubric structure is a driver of halo.

**Files:**
- Control: `/tmp/psq_exp2_control.jsonl`
- Treatment: `/tmp/psq_exp2_treatment.jsonl`

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

### Results (2026-02-28)

Scored 0-10 (control) and 1-7 only. Pilot gate triggered — 1-5 skipped.

| Metric | 0-10 | 1-7 | Difference |
|--------|------|-----|-----------|
| M2: Mean inter-dim \|r\| | 0.666 | 0.660 | -0.006 |
| M5: Eigenvalue ratio | 75.4% | 74.7% | -0.7pp |
| M6: Mean entropy | 1.769 | 1.681 | -0.088 |
| Exact-neutral rate | 45.5% | 45.0% | -0.5pp |

**Cross-scale ρ:** Mean 0.994 (range 0.968–1.000). Near-perfect rank preservation — scorer applies identical relative ordering regardless of scale.

**Pilot gate:** \|r\| difference = 0.006 < 0.05 → **1-5 scale skipped** (diminishing returns).

**Neutral rates unchanged:** AD=65%, CO=95%, DA=50% at both scales. The neutral-anchoring problem is scale-independent.

**Decision: RETAIN 0-10.** Scale format has essentially zero effect on halo. 1-7 is negligibly better on M2 and M5 but entropy is worse (fewer distinct levels available). The scorer applies the same relative judgments regardless of scale granularity, confirming that halo is a scorer-level phenomenon, not a scale artifact. This is consistent with the prediction from the protocol.

**Files:**
- 0-10: `/tmp/psq_exp3_scale10_*.json`
- 1-7: `/tmp/psq_exp3_scale7_*.json`

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
| 0. Test-Retest | DONE | DONE | DONE | Qualified GO (6/10 ≥ 0.80) |
| 1. Halo-Awareness | DONE | DONE | DONE | **REJECT** (post-hoc: g-factor is real, not halo) |
| 2. Dissimilar Rubrics | DONE | DONE | DONE | REJECT (construct redefinition, +5.3% SD) |
| 3. Scale Format | DONE | DONE | DONE | RETAIN 0-10 (negligible effect, pilot gate triggered) |

### Summary of Findings

1. **No scoring intervention is adopted.** All three interventions tested (halo-awareness instruction, dissimilar rubrics, alternative scale formats) are REJECTED. The halo-awareness instruction was initially adopted based on pre-registered criteria but reversed after structural analysis revealed the g-factor is primarily real co-variation, not scorer artifact.
2. **The g-factor is a range/extremity effect.** Extreme texts (g < 3 or g > 7) show EV1 = 82.8% with perfectly uniform PC1 loadings (SD = 0.023) — pure valence, not halo. Middle texts (g 4-6) show EV1 = 38.7% with structured loadings (SD = 0.117) — genuine dimension-specific differentiation. The g-factor reflects real textual properties, not measurement artifact.
3. **Rubric structure does not drive halo.** Dissimilar rubrics changed construct definitions without reducing halo — only modified dims changed, no contagion effect on unmodified dims.
4. **Scale format does not drive halo.** Scorer applies identical rank ordering at 0-10 and 1-7 scales. Neutral-anchoring rates are scale-independent.
5. **The g-factor IS the PSQ at its broadest level.** The hierarchical model PSQ → clusters → dimensions means the g-factor is the construct itself at the top of the hierarchy. Fighting it with scorer instruction would damage the theoretical structure. The correct approach is structural: enrich training data with middle-g texts (where dimension-specific signal is strongest) and preserve the hierarchical decomposition in the model architecture.
6. **Scorer-level interventions cannot distinguish halo from real co-variation.** The halo-awareness instruction's individual score changes (mean |Δ| = 0.217) are below the test-retest noise floor (MAD = 0.54). The headline metrics improvements are partially attributable to CC bias (+0.33 mean shift) and CO decoupling (loss of legitimate co-variation).
