# Psychometric Evaluation: SafetyQuotient (PSQ)

**Date:** 2026-02-27
**Scope:** Evaluation of PSQ against established psychometric best practices
**Status:** v15 DistilBERT — test_r=0.536, held-out_r=0.495 (separated labels). AD batch drove authority_dynamics +0.166; generalization gap down to 7.6%. Contractual clarity regressed (0.498→0.388).

---

## 1. Summary Assessment

The PSQ is a 10-dimension content-level psychological safety measurement system grounded in 170+ validated instruments. It demonstrates genuine methodological innovation — no prior tool assesses psychological safety at the content level across this many dimensions. The theoretical foundation is strong, the operational specification is thorough, and the working implementation produces measurable results (v14 DistilBERT: test_r=0.544, held-out_r=0.482 with halo-free separated labels). Eight dimensions now generalize well to real-world text (r=0.41-0.65); regulatory_capacity requires additional targeted labels. The halo effect in joint LLM scoring has been confirmed and addressed via separated scoring (one dimension per call).

However, against established psychometric standards (AERA/APA/NCME *Standards for Educational and Psychological Testing*, 2014), the project has significant validation gaps. Most standard reliability and validity evidence has not yet been collected.

### Scorecard

| Category | Status | Priority |
|---|---|---|
| Theoretical grounding | **Strong** | — |
| Content validity | **Partial** | High |
| Construct validity | **Preliminary** (pairwise correlations) | Critical |
| Convergent/discriminant validity | **Strong** discriminant (mean |r|=0.167 calibrated vs sentiment) | — |
| Known-groups validity | **Mixed** (10/10 ANOVA sig, 3/8 predictions confirmed) | Medium |
| Criterion validity | **Not measured** | High |
| Internal consistency (Cronbach's α) | **Not measured** | High |
| Test-retest reliability | **Excellent** (ICC=0.935 perturbation stability) | — |
| Inter-rater reliability | **Not measured** | Critical |
| Range utilization / bias | **Measured** (6/10 good, 2/10 poor) | High |
| Measurement invariance / bias | **Planned, not done** | High |
| Normative data | **Not established** | Medium |
| Confidence calibration | **Problematic** (7/10 dims inverted/NaN after isotonic calibration) | Critical |

---

## 2. What the Project Does Well

### 2a. Instrument-Grounded Dimensions

Each of the 10 PSQ dimensions maps to 3-5 established, published instruments with known psychometric properties. This is far more rigorous than typical "new framework" proposals that define constructs ad hoc.

| Dimension | Key Instruments | Instrument Maturity |
|---|---|---|
| Threat Exposure | COPSOQ, NAQ, Abusive Supervision Scale | High (20+ years, cross-cultural) |
| Regulatory Capacity | ERQ, DERS, CERQ | High (process model well-validated) |
| Resilience Baseline | CD-RISC, BRS, Grit Scale | High (extensive norming) |
| Trust Conditions | Rotter ITS, OTI, Trust Quotient | Moderate (individual vs. content gap) |
| Hostility Index | Cook-Medley, BPAQ, STAXI-2 | High (decades of clinical use) |
| Cooling Capacity | CPI, Gross reappraisal, REQ | Moderate (mostly applied frameworks) |
| Energy Dissipation | Effort-Recovery, COR, Flow State | Moderate (theoretical + empirical) |
| Defensive Architecture | DSQ-40, DMRS, Vaillant hierarchy | High (clinical tradition) |
| Authority Dynamics | French & Raven, MLQ, Tepper ABS | High (organizational psych) |
| Contractual Clarity | PCI, Morrison & Robinson | Moderate (workplace-specific) |

**Assessment:** The instrument selection is well-justified and spans multiple traditions (clinical, organizational, social psychology). The key weakness is the leap from *person-level self-report instruments* to *content-level text analysis* — see §3b.

### 2b. Explicit Confidence Framework

The system requires confidence scores alongside every dimension score and gates aggregation at confidence < 0.60. This is sound psychometric practice — it prevents false precision and explicitly models uncertainty.

**Assessment:** The intent is correct. The execution needs validation — see §3d.

### 2c. Protective/Threat Factor Structure

The formula separates 6 protective factors (numerator) from 4 threat factors (denominator, inverted). This mirrors established risk/resilience frameworks in clinical psychology and maps cleanly to the PJE (Psychology-Juris-Engineering) theoretical lens.

### 2d. Non-Diagnostic Positioning

The documentation consistently states PSQ evaluates *content*, not *people*, and is not a clinical diagnostic tool. This is appropriate intellectual humility and avoids the most dangerous misuse patterns.

### 2e. Multi-Source Training Data

The training pipeline combines 13 source datasets with different strengths:
- Berkeley Hate Speech (IRT-derived scores, high methodological rigor)
- Civil Comments (large scale, crowd-annotated toxicity)
- GoEmotions (27 fine-grained emotion labels)
- UCC (unhealthy conversation attributes)
- Diplomacy, CaSiNo, Stanford Politeness, ProsocialDialog (specialized signals)
- 900 LLM gold-standard labels across all 10 dimensions

---

## 3. Critical Psychometric Gaps

### 3a. Reliability Evidence — Preliminary

**Standard:** Any measurement instrument must demonstrate that it produces consistent results under consistent conditions (AERA/APA/NCME Standards 2.0-2.19).

| Reliability Type | What It Tests | PSQ Status |
|---|---|---|
| Internal consistency (α) | Do items within a dimension agree? | Not measured |
| Test-retest (perturbation) | Stable under meaning-preserving text changes? | **Excellent** — ICC(3,1) = 0.935, all 10 dims > 0.90 |
| Test-retest (temporal) | Same content → same score after time gap? | Not measured (LLM teacher) |
| Inter-rater (human) | Do independent raters agree? | Not measured |
| Inter-model (LLM) | Do different LLMs agree? | Not measured |
| Intra-model (student) | Same model, same input → same score? | **Trivially satisfied** (deterministic) |
| Range utilization | Does model use full scoring range? | **Measured** — 6/10 dims good, 2/10 weak, 2/10 poor |
| Split consistency | Stable across train/val/test? | **Measured** — 5/10 dims good, 3/10 moderate, 2/10 overfitting |
| Systematic bias | Consistent directional error? | **Measured** — contractual_clarity -1.81 (severe), rest <0.4 |

**Update (2026-02-26, perturbation stability):** A perturbation-based test-retest analysis was conducted on the v2d ONNX model using 500 test-split samples and 5 perturbation types (typo insertion, punctuation removal, case mutation, word deletion, whitespace noise). This is the neural model analogue of classical test-retest reliability — it measures whether scores reflect underlying constructs or surface-level features.

| Dimension | ICC(3,1) | MAD | Interpretation |
|---|---|---|---|
| Threat Exposure | 0.955 | 0.226 | Excellent |
| Energy Dissipation | 0.952 | 0.135 | Excellent |
| Cooling Capacity | 0.944 | 0.212 | Excellent |
| Hostility Index | 0.941 | 0.240 | Excellent |
| Resilience Baseline | 0.939 | 0.121 | Excellent |
| Trust Conditions | 0.935 | 0.239 | Excellent |
| Authority Dynamics | 0.928 | 0.223 | Excellent |
| Contractual Clarity | 0.928 | 0.138 | Excellent |
| Regulatory Capacity | 0.918 | 0.116 | Excellent |
| Defensive Architecture | 0.909 | 0.123 | Excellent |
| **Average** | **0.935** | **0.177** | **Excellent** |

Key findings:
- All 10 dimensions exceed the ICC > 0.75 "good" threshold; all > 0.90 ("excellent")
- Average MAD = 0.177 on a 0-10 scale — scores shift < 2% under perturbation
- Punctuation removal is the strongest perturbation (MAD=0.392) — the model uses punctuation as a meaningful signal (linguistically valid: "!" vs "." conveys intensity)
- Case changes and whitespace have zero effect (expected: uncased model, tokenizer normalizes whitespace)
- Confidence scores are extremely stable (max Δ = 0.02 across all perturbations)

This satisfies the psychometric standard for test-retest reliability at the model level. It does not address temporal stability of the *LLM teacher* (scoring the same text at different times) or inter-rater reliability (human agreement).

**Update (2026-02-26, spot check):** A reliability spot check was conducted on v2d (see distillation-research.md §9c). Key findings:
- 6 dimensions show good measurement properties: >79% of predictions within 1 point of ground truth, low bias, full range utilization
- 2 dimensions (trust, authority) show moderate properties: mild bias, some range compression
- 2 dimensions (defensive, contractual) show poor properties: severe range compression (pred_std/true_std < 0.60), high error rates
- Contractual_clarity has -1.81 systematic bias (since addressed by removing UCC proxy)

**Why it matters:** Without reliability evidence, we cannot know whether score differences reflect real differences in content or measurement noise. A score of 4.2 vs 5.8 could be meaningful or meaningless.

**Recommendation:** Before any production deployment:
1. Score 200+ diverse texts with 2-3 independent human expert raters (clinical psychologists or organizational psychologists). Compute ICC (Intraclass Correlation Coefficient). Target: ICC ≥ 0.70 for ≥ 8/10 dimensions.
2. Re-score 100 texts with the same LLM after 2+ weeks. Compute test-retest r. Target: r ≥ 0.80.
3. Score 200 texts with Claude, GPT-4, and Gemini. Compute inter-model agreement. This tests whether scores reflect the construct or idiosyncrasies of a single LLM.

### 3b. Construct Validity Gap: Person-Level Instruments → Content-Level Scoring

**The central psychometric challenge.** The 170+ instruments cited were designed to measure *person-level traits and states* via self-report:
- DERS asks "When I'm upset, I have difficulty concentrating" (person self-reports)
- NAQ asks "Have you been exposed to... repeated reminders of your errors" (person recalls)
- DSQ-40 asks "People tend to mistreat me" (person's subjective experience)

PSQ repurposes these as *content-level rubrics*:
- Does this text *cause* difficulty concentrating? (content assessment)
- Does this text *contain* repeated reminders of errors? (text analysis)
- Does this text *strip* defenses? (inferred impact)

This is a **level-of-analysis shift** from trait/state measurement to stimulus assessment. The instruments' validity evidence (factor structure, norms, reliability) does not automatically transfer. Strictly speaking, PSQ is using these instruments as *conceptual guides* for what to look for in text, not as the instruments themselves.

**Recommendation:** Acknowledge this explicitly in documentation. Frame instrument references as "conceptual grounding" rather than "measurement with." Consider: the PSQ dimensions are *inspired by* these instruments, and the validity of the text-level construct must be established independently.

### 3c. Factor Analysis — Dimension Independence Partially Tested

**Standard:** Multi-dimensional instruments should demonstrate that dimensions are empirically distinct (discriminant validity) via factor analysis (AERA/APA/NCME Standard 1.13).

**Update (2026-02-26):** Pairwise Pearson correlations have been computed across all records with overlapping ground-truth scores (see distillation-research.md §8x). Key findings:

**Strongly correlated pairs (|r| > 0.5) — 7 of 45 pairs:**

| Pair | r | n |
|---|---|---|
| Regulatory Capacity ↔ Resilience Baseline | 0.877 | 3,932 |
| Hostility Index ↔ Cooling Capacity | 0.840 | 3,949 |
| Authority Dynamics ↔ Trust Conditions | 0.787 | 3,949 |
| Hostility Index ↔ Authority Dynamics | 0.737 | 3,949 |
| Hostility Index ↔ Trust Conditions | 0.687 | 3,949 |
| Authority Dynamics ↔ Cooling Capacity | 0.650 | 1,949 |
| Trust Conditions ↔ Cooling Capacity | 0.583 | 1,949 |

**Near-zero pairs (good discriminant validity):** 15 pairs with |r| < 0.2, confirming dimensions like Energy Dissipation, Threat Exposure, and Defensive Architecture measure distinct constructs.

**Summary statistics:** Mean off-diagonal |r| = 0.257, median = 0.174.

**Interpretation:** The 10-factor structure is partially supported. Most dimension pairs show low correlation, suggesting distinct constructs. However, three pairs (r > 0.8) raise questions:
1. **Regulatory Capacity ↔ Resilience Baseline (r=0.877):** Both measure emotion regulation capacity from different angles. A 7-factor model merging these may be more parsimonious.
2. **Hostility Index ↔ Cooling Capacity (r=0.840):** Hostile content inherently lacks de-escalation — the theoretical prediction that these are distinct (one is a threat factor, one is protective) is not supported empirically in current data.
3. Some high correlations may reflect shared proxy methodology (e.g., both from UCC) rather than true construct overlap. LLM-only correlations are needed for a cleaner signal.

**Update (2026-02-27, halo effect confirmed):** A halo experiment (§18 of distillation-research.md) demonstrated that joint LLM scoring inflates inter-dimension correlations by ~0.15 on average. Mean |r| = 0.766 (joint) vs 0.656 (separated). Two genuine clusters emerged: Interpersonal Climate (authority_dynamics, contractual_clarity, trust_conditions, threat_exposure) and Internal Resources (regulatory_capacity, resilience_baseline, defensive_architecture), with bridge dimensions (cooling_capacity, energy_dissipation, hostility_index). The held-out test was re-scored with separated calls (one dimension per LLM call) to eliminate this contamination. The 10-dimension structure is retained; clusters are additive reporting layers only.

**Remaining gap:** Full confirmatory factor analysis (CFA) requires records scored on all 10 dimensions simultaneously, which the current training data does not provide (each record has 1-6 dimensions). The 900 LLM-labeled records are closer (each scored on 1 dimension), but a dedicated labeling pass scoring all 10 dimensions on 500+ texts would be needed.

**Recommendation (updated):** The initial pairwise correlation results are encouraging but not conclusive. Priority steps:
1. Score 500 texts on all 10 dimensions simultaneously (LLM or expert raters)
2. Run EFA on the full 10×500 matrix
3. Test both 10-factor and reduced (7- or 8-factor) models
4. Report factor loadings, interfactor correlations, and model fit indices (RMSEA, CFI, TLI)

### 3d. Confidence Calibration — PROBLEMATIC

**Standard:** Confidence estimates should be monotonically related to accuracy — higher confidence should mean lower error.

**Update (2026-02-26):** Comprehensive calibration analysis was run on the v2d ONNX model (`scripts/validate_confidence_calibration.py`, 1,974 test records).

**Confidence-error correlation per dimension:**

| Dimension | r(conf, error) | Direction | Useful? |
|---|---|---|---|
| contractual_clarity | -0.453 | Correct | YES |
| authority_dynamics | -0.268 | Correct | YES |
| defensive_architecture | -0.193 | Correct | YES |
| threat_exposure | -0.014 | Flat | NO |
| hostility_index | +0.100 | INVERTED | NO |
| resilience_baseline | +0.110 | INVERTED | NO |
| regulatory_capacity | +0.180 | INVERTED | NO |
| trust_conditions | +0.236 | INVERTED | NO |
| energy_dissipation | +0.393 | INVERTED | NO |
| cooling_capacity | +0.416 | INVERTED | NO |

**Summary:** 3 correct, 1 flat, 6 inverted. The reliability diagram is non-monotonic (MAE rises from 0.73 at low confidence to 0.92 at high confidence).

**Root cause:** The student model faithfully reproduces teacher confidence (r = 0.51–0.86 with ground truth confidence). But proxy data has *high* confidence on *biased* labels — when the proxy "confidently" maps Berkeley hate_speech_score to threat_exposure, the mapping itself introduces systematic error. The model learns that high-confidence proxy predictions are actually less accurate than low-confidence ones.

**Update (2026-02-26, v3b post-hoc isotonic calibration):** Isotonic regression was applied post-hoc to v3b model confidence outputs. Results:

| Dimension | r(conf, error) | Direction | Useful? |
|---|---|---|---|
| authority_dynamics | negative | Correct | YES |
| trust_conditions | negative | Correct | YES |
| defensive_architecture | negative | Correct | YES |
| energy_dissipation | ~0 (collapsed) | Near-constant confidence | NO |
| resilience_baseline | ~0 (collapsed) | Near-constant confidence | NO |
| threat_exposure | positive | INVERTED | NO |
| hostility_index | positive | INVERTED | NO |
| regulatory_capacity | positive | INVERTED | NO |
| cooling_capacity | positive | INVERTED | NO |
| contractual_clarity | positive | INVERTED | NO |

**Summary (calibrated):** 3/10 correct (authority_dynamics, trust_conditions, defensive_architecture), 7/10 still inverted or NaN after post-hoc isotonic calibration. energy_dissipation and resilience_baseline collapsed to near-constant confidence (no variance left to calibrate). The three dimensions that were correct before calibration remain correct; isotonic regression could not rescue the others.

**Conclusion:** Post-hoc calibration improves *score* accuracy (see §3e-1 below) but cannot fix the fundamental confidence training problem. The confidence head learned to reproduce teacher confidence, which is systematically miscalibrated on proxy data. The two-phase confidence training approach in v4 (Phase 1: train scores only; Phase 2: train confidence to predict actual score error) is needed to address this at the root.

**Recommendation (updated):** Do not expose raw or post-hoc-calibrated confidence to users. The v4 two-phase confidence training (currently in progress) is the correct fix. Interim options:
1. Suppress confidence display entirely, use only internally for gating
2. Replace with MC dropout uncertainty at inference
3. Await v4 model with properly trained confidence head

### 3e. Discriminant Validity vs Sentiment — STRONG

**Standard:** PSQ should measure something beyond simple positive/negative sentiment (AERA/APA/NCME Standard 1.14).

**Update (2026-02-26):** Discriminant analysis was run against VADER sentiment (`scripts/validate_discriminant_sentiment.py`, 800 test samples).

- **Mean |r| with VADER compound = 0.205** — PSQ is clearly distinct from sentiment
- 9/10 dimensions have low correlation (|r| < 0.30); 1 has moderate (regulatory_capacity, r=0.362)
- 0/10 dimensions have high correlation (|r| ≥ 0.60)

**Incremental validity** (PSQ R² over sentiment R² predicting ground truth):

| Dimension | R²(sentiment) | R²(PSQ) | ΔR² |
|---|---|---|---|
| threat_exposure | 0.000 | 0.776 | +0.776 |
| hostility_index | 0.044 | 0.808 | +0.764 |
| cooling_capacity | 0.002 | 0.644 | +0.641 |
| resilience_baseline | 0.147 | 0.710 | +0.563 |
| trust_conditions | 0.000 | 0.558 | +0.558 |
| energy_dissipation | 0.183 | 0.675 | +0.492 |
| regulatory_capacity | 0.035 | 0.525 | +0.490 |
| authority_dynamics | 0.000 | 0.386 | +0.386 |
| defensive_architecture | 0.066 | 0.020 | -0.046 |

PSQ adds massive predictive value over sentiment on 8/9 dimensions. Defensive architecture is the sole exception (R²=0.02 — poor prediction regardless of method).

#### 3e-1. Discriminant Validity — Calibrated (v3b + isotonic regression)

**Update (2026-02-26, v3b post-hoc isotonic calibration):** After applying isotonic regression to v3b model scores, discriminant validity against VADER sentiment *improved*:

- **Mean |r| with VADER compound = 0.167** (improved from 0.205 raw — calibration made PSQ even more distinct from sentiment)
- 9/10 dimensions have LOW correlation (|r| < 0.30)
- 1/10 has MODERATE correlation: regulatory_capacity (r = +0.355, down from r = 0.362 raw)
- 0/10 dimensions have HIGH correlation

**Interpretation:** Isotonic calibration shifted scores toward the true scale center and compressed outliers, which reduced the spurious correlation with sentiment. This is a positive signal — the calibrated scores are *more* construct-specific and *less* contaminated by general positive/negative valence.

#### 3e-2. Score Calibration Effect on Distributions (v3b + isotonic regression)

Isotonic regression had the expected effect on score distributions:

- Calibration shifted dimension means toward the true scale center (5.0 on a 0-10 scale)
- Reduced standard deviation on all dimensions (compressed outliers toward center, as expected with monotonic regression toward the mean)
- Example: cooling_capacity std dropped from 1.61 to 0.92

This compression is a natural property of isotonic regression — it cannot increase score variance, only reduce it. The tradeoff is reduced sensitivity at the extremes in exchange for better calibration in the middle of the distribution. For content moderation (where distinguishing "moderate" from "high" matters more than distinguishing "extreme" from "very extreme"), this is an acceptable tradeoff.

**Note:** These are v3b results with post-hoc isotonic calibration. The v4 model (two-phase confidence training, currently in progress) will incorporate calibration into training rather than applying it post-hoc.

### 3f. Known-Groups Validity — MIXED

**Standard:** The instrument should differentiate groups known to differ on the construct (criterion validity analogue).

**Update (2026-02-26):** Known-groups analysis across 7 source datasets (`scripts/validate_known_groups.py`).

**Group separation (ANOVA):** All 10 dimensions show significant group differences (p < 0.001). Effect sizes:
- Large (η² > 0.14): threat_exposure (0.37), contractual_clarity (0.29), cooling_capacity (0.15), hostility_index (0.14)
- Medium (η² > 0.06): all remaining 6 dimensions

**Theoretical predictions:** 3/8 confirmed (38%). However, the "failed" predictions reveal the model is making *correct* but *unexpected* distinctions:
- Civil Comments (newspaper comments) scores highest on threat_exposure (9.29) — higher than Berkeley hate speech (7.47). This is valid: casual online toxicity creates a more threatening *environment* than targeted hate speech.
- Politeness data scores highest on hostility (6.88) — a known proxy contamination artifact we already identified.

**Verdict:** The model strongly differentiates content types (all η² significant). The naive theoretical predictions were too simplistic — different safety concerns manifest differently across datasets. The model captures this nuance, which is arguably a strength.

### 3g. No Convergent or Criterion Validity

**Convergent validity** asks: Does PSQ correlate with other measures of similar constructs?
- Edmondson's Psychological Safety Scale (team level)
- PSC-12 (Psychosocial Safety Climate)
- COPSOQ (Copenhagen Psychosocial Questionnaire)

None of these comparisons have been conducted.

**Criterion validity** asks: Do PSQ scores predict real-world outcomes?
- Do low-PSQ texts appear in contexts where safety complaints were filed?
- Do high-PSQ communication patterns correlate with team retention/wellbeing?

No outcome data has been collected.

**Recommendation:** This is the most important gap for establishing PSQ as a meaningful measurement. Even a small study (n=50 teams, correlate PSQ scores of team communications with Edmondson survey scores) would provide critical evidence.

### 3h. Formula Inconsistency — RESOLVED

The formal specification (final-state.md) originally defined PSQ as a **ratio** (protective / threat factors) with 0-2 per-dimension scoring. The implementation (detector.js) uses a **linear difference** ((protective - threat + MAX) / (2 * MAX) * 100) with 0-10 per-dimension scoring. These were mathematically different formulas.

**Resolution (2026-02-26):** Updated `final-state.md` and `psq-definition.md` to match the implementation:
- Dimension scores: 0–10 (5 = neutral), not 0–2 (1 = neutral)
- Aggregation: confidence-weighted averages, not sums
- Formula: linear difference `PSQ = ((protective_avg - threat_avg + 10) / 20) × 100`, not ratio
- Output: PSQ 0–100 with protective_avg and threat_avg (both 0–10), not protective_score 0–12 / threat_score 0–8
- Classification thresholds: updated to 0–100 scale (critical < 20, low 20–40, moderate 40–70, high > 70)
- Confidence gating: dimensions below 0.6 confidence excluded (already in code, now in spec)

---

## 4. Proxy Mapping Validity

The training pipeline maps existing dataset labels to PSQ dimensions. These proxy mappings introduce construct mismatch:

| Mapping | Source Construct | PSQ Construct | Gap |
|---|---|---|---|
| Berkeley hate_speech_score → threat_exposure | Hate speech (attacks on protected groups) | Psychoemotional threat (broader) | Moderate |
| Civil Comments toxicity → threat_exposure | Online toxicity (offensive language) | Psychoemotional threat | Moderate |
| GoEmotions anger/fear → regulatory_capacity | Emotion category (present in text) | Emotion regulation support (content provides) | **Large** |
| UCC condescending → authority_dynamics | Conversational attribute (present) | Power-dynamics quality (content creates) | **Large** |
| UCC dismissive → contractual_clarity | Conversational attribute | Expectation clarity (inferred) | **Very large** |
| Diplomacy truthfulness → trust_conditions | Message honesty (fact) | Trust environment quality (inferred) | Moderate |
| Politeness score → authority_dynamics | Linguistic politeness (surface) | Power dynamics quality (deep) | **Large** |
| ProsocialDialog safety_label → defensive_architecture | Safety level (categorical) | Defense mechanism maturity (clinical) | **Very large** |

The error analysis confirms these gaps empirically:
- Proxy-LLM score gaps: authority +2.8, contractual +4.4, threat +3.6
- Proxy ceiling: r ≈ 0.65 maximum — all proxy models hit this wall
- UCC proxy for contractual_clarity has bias of -2.32 (actively harmful)

**Assessment:** The proxy mappings are reasonable *starting points* for getting a model off the ground but should not be treated as ground truth. The project correctly identifies this (halving UCC proxy confidence, increasing LLM label weight to 5x) but should document it more prominently as a known limitation.

---

## 5. The Defensive Architecture Problem

Defensive architecture (r=0.125 on full dataset, essentially random) deserves special attention because it exposes a fundamental question: **can defense mechanisms be scored from text?**

The DSQ-40 measures defenses through self-report ("People tend to mistreat me" → projection). The DMRS requires trained clinician observation of therapy sessions. Vaillant's hierarchy was derived from longitudinal case studies.

All of these require access to the person's *internal experience* — the very thing that defines a defense mechanism is that it operates *unconsciously* to manage anxiety. Text alone provides:
- Behavioral markers (what the person said/wrote)
- Linguistic patterns (absolutist language, blame-shifting, emotional detachment)
- Content themes (splitting, idealization, devaluation)

But it does **not** provide:
- Whether the person is aware of their pattern (conscious vs. unconscious)
- Whether the pattern is ego-syntonic (feels natural) or ego-dystonic (recognized as problematic)
- Whether it's situational or characterological

**Assessment:** Defensive architecture as currently defined may have a low psychometric ceiling for text-based measurement. Options:
1. Reframe as "observable defense patterns in text" (more modest, more defensible)
2. Focus on the text-observable subset: acting out (behavioral), intellectualization (linguistic), humor as defense (identifiable), splitting (absolutist language)
3. Accept a lower target (r ≥ 0.4 rather than r ≥ 0.7) and document why

---

## 6. Appropriate Use Cases (Current State)

### Appropriate Now

- **Research tool** for studying psychological safety patterns in text corpora
- **Exploratory content analysis** with explicit confidence caveats
- **Decision support** in moderation (human reviews PSQ output, not automated action)
- **Longitudinal tracking** of communication safety trends (relative changes, not absolute scores)
- **Training/education** to illustrate psychological safety concepts with examples

### Not Yet Appropriate

- **Automated moderation decisions** (insufficient reliability evidence)
- **Legal/forensic assessment** (no inter-rater reliability, no normative data)
- **Clinical risk screening** (no criterion validity, not validated against outcomes)
- **Hiring/evaluation decisions** (no bias testing completed, no measurement invariance)
- **Diagnostic or classificatory use** (explicitly stated to be non-diagnostic, correctly)

---

## 7. Validation Roadmap

Priority-ordered steps to bring PSQ to psychometric standards:

### Phase 1: Reliability (estimated effort: 2-4 weeks)

1. **Intra-model reliability**: Score 200 texts twice with same LLM, 2-week gap. Compute test-retest r per dimension. Target: r ≥ 0.80.
2. **Inter-model reliability**: Score 200 texts with Claude, GPT-4, Gemini. Compute ICC. Target: ICC ≥ 0.70.
3. **Student model stability**: ~~Score 200 texts twice with student model.~~ **DONE** — perturbation-based test-retest on 500 texts, ICC(3,1) = 0.935 (Excellent). All 10 dimensions > 0.90. See `models/psq-student/test_retest_results.json`.

### Phase 2: Human Validation (estimated effort: 4-8 weeks)

4. **Expert panel content validity**: 3-5 clinical/organizational psychologists rate 100 texts on all 10 dimensions independently. Compute ICC per dimension. Target: ICC ≥ 0.70 for ≥ 8/10 dimensions.
5. **Human-LLM agreement**: Compare expert ratings to LLM scores. Compute r and systematic bias per dimension.
6. **Confidence calibration**: Plot LLM confidence vs. actual error (reliability diagrams). Calibrate if needed.

### Phase 3: Construct Validation (estimated effort: 4-8 weeks)

7. **Factor analysis**: Score 500+ texts, run EFA on 10 dimension scores. Test 10-factor structure.
8. **Convergent validity**: Correlate PSQ scores with Edmondson's scale, PSC-12, or COPSOQ in teams where both content and survey data are available. Target: r ≥ 0.50.
9. **Discriminant validity**: Show that PSQ dimensions are not simply measuring "general positivity/negativity." Correlate with sentiment scores; demonstrate incremental prediction beyond sentiment.

### Phase 4: Norming and Bias (estimated effort: 4-8 weeks)

10. **Bias testing**: Vary demographic markers in matched content. Measure scoring differences across gender, race, age, cultural references.
11. **Normative data**: Score a representative sample (diverse contexts, content types, domains). Establish percentile norms and empirically-derived cutoffs.
12. **Clinical cutoff validation**: If possible, correlate PSQ scores with documented safety outcomes (complaints, incidents, team health surveys).

---

## 8. References for Psychometric Standards

- AERA, APA, & NCME. (2014). *Standards for Educational and Psychological Testing*. AERA.
- Edmondson, A. (1999). Psychological safety and learning behavior in work teams. *Administrative Science Quarterly*, 44(2), 350-383.
- Kadavath, S., et al. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*.
- Xiong, M., et al. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. *ICLR 2024*.
- Bond, M., et al. (1983). An empirical study of self-rated defense styles. *Archives of General Psychiatry*, 40(3), 333-338.
- Gross, J.J. & John, O.P. (2003). Individual differences in two emotion regulation processes. *Journal of Personality and Social Psychology*, 85(2), 348-362.
- Connor, K.M. & Davidson, J.R.T. (2003). Development of a new resilience scale. *Depression and Anxiety*, 18(2), 76-82.
- Rousseau, D.M. (2000). *Psychological Contract Inventory*. Carnegie Mellon University.
- Kristensen, T.S., et al. (2005). The Copenhagen Psychosocial Questionnaire. *Scandinavian Journal of Public Health*, 33, 438-449.
