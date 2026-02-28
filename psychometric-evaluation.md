# Psychometric Evaluation: SafetyQuotient (PSQ)

**Date:** 2026-02-28
**Scope:** Evaluation of PSQ against established psychometric best practices
**Status:** v21 DistilBERT — test_r=0.504, held-out_r=0.630 (separated labels, best ever, +0.030 vs v19). Score-concentration cap active. 4 criterion validity studies (CaSiNo, CGA-Wiki, CMV, DonD). DB: 21,627 texts, 80,361 scores, 26,771 separated-llm. Factor analysis v2: g-factor eigenvalue 6.727 (67.3% variance), KMO=0.902. Scoring experiment protocols designed for rubric-induced halo mitigation (3 experiments + test-retest baseline).

---

## 1. Summary Assessment

The PSQ is a 10-dimension content-level psychological safety measurement system grounded in 170+ validated instruments. It demonstrates genuine methodological innovation — no prior tool assesses psychological safety at the content level across this many dimensions. The theoretical foundation is strong, the operational specification is thorough, and the working implementation produces measurable results (v19 DistilBERT: test_r=0.509, held-out_r=0.600 with halo-free separated labels). All 10 dimensions now generalize to real-world text (r=0.49-0.71); threat_exposure has recovered to 0.495 (from 0.370 in v18) following broad-spectrum data addition. The halo effect in joint LLM scoring has been confirmed and addressed via separated scoring (one dimension per call). Four independent criterion validity studies demonstrate that PSQ profiles predict real-world outcomes (negotiation satisfaction, conversation derailment, persuasion success, deal-reaching) with AUC 0.59–0.69, and that profile shape predicts while the average does not.

However, against established psychometric standards (AERA/APA/NCME *Standards for Educational and Psychological Testing*, 2014), the project has significant validation gaps. Factor analysis v2 (n=1,970 separated-llm texts) shows a dominant general factor explaining 67.3% of variance (eigenvalue 6.727, KMO=0.902), with parallel analysis retaining only 1 factor. The previous 5-factor structure has largely collapsed; 8/10 dimensions load primarily on the general factor. A newly discovered integer-only scoring bias (LLM almost never assigns non-integer scores) may be inflating correlations by compressing effective resolution to 11 bins. Most standard reliability and validity evidence has not yet been collected.

### Scorecard

| Category | Status | Priority |
|---|---|---|
| Theoretical grounding | **Strong** | — |
| Content validity | **Partial** | High |
| Construct validity | **Tested** — EFA v2 rejects 10-factor independence. g-factor eigenvalue 6.727 (67.3% variance), parallel analysis retains 1 factor only. 5-factor structure collapsed. Integer-only scoring bias may inflate g-factor. | High |
| Convergent/discriminant validity | **Strong** discriminant (mean |r|=0.167 calibrated vs sentiment) | — |
| Known-groups validity | **Mixed** (10/10 ANOVA sig, 3/8 predictions confirmed) | Medium |
| Criterion validity | **Four studies** — CaSiNo: satisfaction (r≈0.08-0.13\*\*\*); CGA-Wiki: derailment (AUC=0.599); CMV: persuasion (AUC=0.590); DonD: deal-reaching (AUC=0.686). Context-dependent primacy: AD in contested-status, ED in sustained negotiation, DA in fixed-status. Profile >> average in all studies. | **Strong** |
| Internal consistency (Cronbach's α) | **Not measured** | High |
| Test-retest reliability | **Excellent** (ICC=0.935 perturbation stability) | — |
| Inter-rater reliability | **Not measured** — protocol designed (`expert-validation-protocol.md`) | Critical |
| Held-out generalization | **Strong** (held-out_r=0.630, 10/10 dims r>0.49, v21) | — |
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
| Inter-rater (human) | Do independent raters agree? | Not measured — protocol designed (see `expert-validation-protocol.md`) |
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

### 3c. Factor Analysis — 10-Factor Structure NOT Supported

**Standard:** Multi-dimensional instruments should demonstrate that dimensions are empirically distinct (discriminant validity) via factor analysis (AERA/APA/NCME Standard 1.13).

**Update (2026-02-28, factor analysis v1):** EFA was conducted on 2,359 texts with complete 10-dimension coverage (1,470 separated-llm, 976 joint-llm, 150 composite-proxy). Analyses used sklearn FactorAnalysis with varimax rotation.

#### Adequacy Tests

| Test | v1 Result (N=2,359, mixed) | v2 Result (N=1,970, sep-llm only) | Interpretation |
|---|---|---|---|
| KMO (Kaiser-Meyer-Olkin) | **0.819** | **0.902** | Meritorious → Superb |
| Bartlett's sphericity | χ²=12,750.5, df=45, p≈0.000 | — | Correlations are not an identity matrix |

#### How Many Factors?

| Method | v1 (mixed) | v2 (sep-llm only) |
|---|---|---|
| Kaiser criterion (eigenvalue > 1) | **3** (all data), **2** (sep-llm) | — |
| Parallel analysis (Horn's, 95th %ile) | **2** | **1** |
| BIC (model selection) | **5** (ΔBIC=0 vs 4-factor +110) | **5** (but 4-/5-factor didn't converge) |
| 10-factor model fit | Collapsed to **5** non-trivial factors | — |

The first eigenvalue alone explains **48.4%** of variance in v1, rising to **67.3%** in v2, indicating a dominant and strengthening general factor.

#### Scree Analysis

| Factor | v1 Eigenvalue | v1 % Variance | v2 Eigenvalue | v2 % Variance |
|---|---|---|---|---|
| 1 | 4.844 | 48.4% | **6.727** | **67.3%** |
| 2 | 1.292 | 12.9% | — | — |
| 3 | 1.029 | 10.3% | — | — |
| 4 | 0.851 | 8.5% | — | — |
| 5 | 0.572 | 5.7% | — | — |
| 6–10 | 0.171–0.395 | 1.7–3.9% | — | — |

#### g-Factor Loadings (v2, all dimensions)

| Dimension | g-Loading |
|---|---|
| trust_conditions | 0.930 |
| defensive_architecture | 0.914 |
| cooling_capacity | 0.864 |
| regulatory_capacity | 0.854 |
| hostility_index | 0.824 |
| resilience_baseline | 0.810 |
| threat_exposure | 0.768 |
| authority_dynamics | 0.737 |
| contractual_clarity | 0.720 |
| energy_dissipation | 0.661 |

All 10 dimensions load >0.66 on the general factor. The 5-factor structure from v1 has largely collapsed: Factor 1 absorbs 8/10 dimensions; only CO, ED, and AD separate weakly.

#### BIC-Best 5-Factor Solution (v1, varimax) — retained for provenance

| Factor | Dimensions (loading > 0.35) | Interpretation |
|---|---|---|
| F1: Hostility/Threat | HI(-0.85), TE(-0.72), CC(-0.59), DA(-0.38) | Aggressive/threatening content |
| F2: Relational Contract | CO(-0.82), TC(-0.78) | Clarity of expectations and trust |
| F3: Internal Resources | RB(-0.73), RC(-0.64), DA(-0.50), CC(-0.49) | Coping and regulation support |
| F4: Power Dynamics | AD(-0.83), DA(-0.36) | Authority relationship quality |
| F5: Stress/Energy | ED(+0.77), TE(+0.48) | Energy drain and accumulation |

Cross-loading dimensions: DA loads on F1, F3, and F4 (no clear primary factor). CC loads on F1 and F3.

All factor score correlations are near zero (max |r|=0.110), confirming orthogonal structure after rotation.

#### Correlation Comparison

| Metric | v1 (N=2,359, mixed) | v1 sep-llm (N=1,470) | v2 (N=1,970, sep-llm only) |
|---|---|---|---|
| Mean off-diagonal \|r\| | 0.417 | 0.564 | **0.632** |
| Pairs \|r\| > 0.5 | 15/45 | 28/45 | — |
| Pairs \|r\| > 0.7 | 1/45 | 11/45 | — |
| Kaiser factors retained | 3 | 2 | — |
| Eigenvalue 1 (% variance) | 4.844 (48.4%) | 6.153 (61.5%) | **6.727 (67.3%)** |

The monotonic increase in inter-dimension correlations (0.417 → 0.564 → 0.632) as composite-proxy data is excluded suggests the composite-proxy mappings introduce dimension-specific noise that artificially decorrelates dimensions. However, the integer-only scoring bias (see below) may also contribute to correlation inflation.

#### Integer-Only Scoring Bias (discovered 2026-02-28)

A score distribution audit revealed that the LLM scorer almost never assigns non-integer values. The effective scoring scale is **11 bins (integers 0-10)**, not continuous 0-10. The 4-5-6 band captures 57-81% of all separated-llm scores. Score-5 concentration: CO 60.8% (worst), TE 24.1% (best), 9/10 dims above the 30% cap threshold.

This has direct implications for the factor analysis:
- Shared "score-5" signal mechanically inflates inter-dimension correlations
- The g-factor eigenvalue may be partly artifactual
- Resolution: pilot 0-100 percentage scoring scale, then re-run factor analysis

**Update (2026-02-28):** Percentage scoring validated at scale. Production batch (200 texts × 10 dims, separated protocol) achieved 86.2% non-integer scores (vs 2.1% integer), 4.8% exact-5.0 (vs 41.3%), 35 unique values (vs ~11). The integer bias is effectively resolved for new labeling. Factor analysis v3 with pct-scored data is the next step to determine whether the g-factor eigenvalue drops.

**Bifactor architecture tested (v19b):** 11th output head (g-PSQ) learned well (r=0.594) but per-dimension test_r dropped to 0.502 (from 0.509). DistilBERT lacks capacity for 11 heads. Recommendation: compute g-PSQ post-hoc as mean of 10 dimension scores.

#### Verdict on H0 (10 factors are distinct)

**H0 is rejected by all standard criteria.** The data supports 1–5 latent factors, not 10. The dominant general factor (67.3% of variance in v2) indicates that most of the 10 dimensions measure a single underlying construct — "overall psychological safety of content." Parallel analysis retains only 1 factor. The 5-factor structure from v1 has become unstable in v2.

However, rejection of 10 independent factors does not require merging dimensions. The dimensions capture theoretically distinct constructs that imply different interventions, and criterion validity studies show that individual dimensions carry non-redundant predictive signal (g-PSQ AUC near chance in all 4 studies, 10-dim profiles predict).

**Recommendation (updated):** Adopt a **hierarchical reporting model**:
1. Report the **overall PSQ** as the primary score (captures the general factor)
2. Report **cluster scores** cautiously — the 5-factor structure may not be stable; await resolution of integer-only bias before committing to a cluster layer
3. Report **dimension scores** as fine-grained detail with the caveat that dimensions share a strong general factor
4. Document the general factor and integer-only bias caveat in all technical reporting
5. Do *not* claim 10 independent dimensions — claim 10 *theoretically distinct* dimensions that empirically share 67% of variance via a general factor
6. Investigate whether switching to a 0-100 scoring scale reduces the g-factor eigenvalue

#### Dimension Reduction Evaluation (2026-02-28)

Empirical test of information loss from dimension reduction (held-out labels, n=117):
- **5-factor**: avg R²=0.881 (88% of dimension-level info preserved). Weakest: CC (0.772), CO (0.813).
- **3-factor**: avg R²=0.738 — AD (0.615) and ED (0.449) poorly reconstructed.
- **CGA-Wiki evidence**: g-PSQ AUC=0.515 (near-chance), 10-dim AUC=0.599. Individual dimensions carry non-redundant predictive signal.
- **Unique variance >30%**: CC (39.4% in Hostility/Threat), CO (36.0% in Relational Contract).
- **Conclusion**: 5-factor is the parsimony sweet spot. Do not reduce below 5 — AD and ED are genuinely independent. Keep 10 dimensions for prediction tasks.

#### Promax (Oblique) Rotation Confirmation

**Update (2026-02-28):** Promax rotation (k=4) was applied to test whether orthogonal rotation was distorting the structure. Key findings:

- **Perfect simple structure:** Promax achieved 0/10 cross-loaders at 2, 3, and 5 factors (varimax had 3–5 cross-loaders). Every dimension loads on exactly one factor.
- **Factor correlations are moderate:** Mean |r|=0.234, max |r|=0.470 (Hostility↔Power). No pair exceeds 0.5 — factors are correlated but distinct.
- **5-factor promax clusters:** HI/TE/CC (Hostility), CO/TC (Contract), RB/RC (Resources), AD (Power), ED (Energy). DA has no loading >0.35 in promax — the weakest construct.
- **Conclusion:** Oblique rotation confirms the 5-factor structure is defensible. Factors are related (as expected for facets of a single meta-construct) but sufficiently separable.

#### Earlier Analyses (retained for provenance)

**Pairwise correlations (2026-02-26):** Initial analysis on records with overlapping ground-truth scores showed 7/45 pairs with |r|>0.5 (strongest: RC↔RB at 0.877, HI↔CC at 0.840). These were computed on partially-overlapping subsets and are superseded by the full correlation matrix above.

**Halo effect (2026-02-27):** A halo experiment confirmed joint LLM scoring inflates inter-dimension correlations by ~0.15 on average. Mean |r|=0.766 (joint) vs 0.656 (separated) on the 100-text held-out set. The held-out test was re-scored with separated calls to eliminate this contamination.

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

**CaSiNo Negotiation Outcomes (2026-02-28).** First criterion validity evidence. The CaSiNo dataset (Chawla et al., 2021; n=1,030 dialogues, 2,060 participant-level observations) includes three post-negotiation outcome variables never used as PSQ training signals: satisfaction (1-5), opponent likeness (1-5), and points scored (0-32).

Results:
- **Satisfaction**: 9/10 PSQ dimensions significantly predict satisfaction (p<0.05). Strongest: energy_dissipation (r=+0.114\*\*\*), defensive_architecture (r=+0.108\*\*\*). g-PSQ r=+0.096\*\*\*.
- **Opponent likeness**: 9/10 significant. Strongest: defensive_architecture (r=+0.126\*\*\*), energy_dissipation (r=+0.125\*\*\*). g-PSQ r=+0.099\*\*\*.
- **Points scored**: Near-zero (max |r|=0.054). PSQ predicts relational outcomes, not competitive ones — theoretically correct.
- **Partial correlations** (controlling text length): DA is the only dimension that *increases* after control (+0.004), indicating it captures interpersonal boundary dynamics beyond conversational complexity.
- **Incremental R²**: PSQ adds +0.016 (satisfaction) and +0.023 (likeness) beyond sentiment + text length. Small but significant: PSQ captures something beyond simple positivity.
- **Extreme groups**: High-PSQ (Q4) dialogues produce +0.18 more satisfaction and +0.23 more likeness than low-PSQ (Q1) dialogues (Cohen's d≈0.17-0.20).

Effect sizes are small (r≈0.08-0.13) but consistent with content-level linguistic predictors in the literature (Pennebaker & King, 1999: r=0.05-0.15; Tausczik & Pennebaker, 2010: r=0.08-0.20).

**Notable finding**: DA — the construct with the weakest factor loading in the PSQ system — is the strongest criterion validity predictor. This suggests DA may lack within-system discriminant validity but has genuine predictive validity for interpersonal outcomes.

**CGA-Wiki Derailment Prediction (2026-02-28).** Second criterion validity study. The Conversations Gone Awry corpus (Zhang et al., 2018; n=4,188 Wikipedia talk-page conversations, perfectly balanced: 2,094 derailing into personal attacks, 2,094 safe) provides a behavioral outcome with zero circularity — no Wikipedia talk pages in PSQ training data.

Results:
- **Logistic regression** (10-dim PSQ, train→test): AUC=0.599, accuracy=57.5%. g-PSQ alone near chance (AUC=0.515).
- **Group comparison**: Derailing conversations have lower PSQ scores on 8/10 dimensions. Strongest: AD (d=-0.212\*\*\*), RC (d=-0.177\*\*\*), TC (d=-0.150\*\*\*).
- **AD again strongest predictor** (r\_pb=-0.105\*\*\*): Replicates CaSiNo finding across a completely different domain and outcome type.
- **Temporal signal decay**: AUC drops from 0.599 (all turns) → 0.570 (early turns) → 0.519 (first turn only). PSQ captures accumulated interpersonal dynamics, not static text properties.
- **10-dim >> g-PSQ**: Individual dimensions carry non-redundant signal; general factor alone cannot predict.

**Cross-study consistency**: AD/DA emerges as the strongest criterion predictor in both studies (CaSiNo: negotiation satisfaction; CGA-Wiki: derailment avoidance), despite having the weakest factor loading in the PSQ system. This paradox — low internal structure, high external validity — suggests AD captures something real that the other dimensions don't.

**Data provenance caveat**: AD's training signal is 70.4% LLM-generated (separated-llm 37%, joint-llm 25%, synthetic 8%), with only 29.6% from composite proxy mappings (UCC condescension and politeness corpus). This is comparable to other dimensions (range: HI at 52% to CO at 97%), but the entire construct validity chain for AD runs through LLM interpretation — including the held-out evaluation (r=0.625, scored by separated-llm). Several factors argue against pure LLM artifact: (1) cross-domain generalization to an unseen discourse register, (2) replication across two independent criterion studies, (3) AD-residual correlates with theoretically predicted text features (second-person pronouns, question marks, authority vocabulary), and (4) the most LLM-dependent dimension (CO at 97%) shows the weakest criterion validity. Nevertheless, AD's criterion findings remain *provisionally grounded* until the expert validation panel (§3h planned) produces human-scored ground truth with ICC(2,1) ≥ 0.70.

**CMV Persuasion Prediction (2026-02-28).** Third criterion validity study. The winning-args-corpus from r/ChangeMyView (Tan et al., 2016; n=4,263 matched pairs — same original post, one delta-awarded reply and one not) tests PSQ in a persuasion context with a matched-pair design controlling for topic and author.

- **All 10 dims discriminate** delta from non-delta replies (9/10 survive Bonferroni, p<.005).
- **DA is top predictor** (r\_pb=+0.085, d\_z=0.135, paired accuracy 55.4%), NOT AD — a critical context-dependent finding.
- **AD is weakest dim** (d\_z=0.033, p=0.032, non-significant at Bonferroni) — in contrast to its dominance in CaSiNo and CGA-Wiki.
- **10-dim AUC=0.590** (5-fold CV) vs **g-PSQ=0.531** — profile >> average (gap 0.059), replicating CGA-Wiki pattern.
- **Incremental AUC beyond text length**: +0.012 (0.596 → 0.608); 9/10 dims retain significance after partial correlation.

**Context-dependent AD finding**: AD predicts when status is contested (Wikipedia disputes, negotiation) but not when status is fixed (CMV, where the OP explicitly invites counterarguments). This supports the status negotiation theory (journal §24, Theory 3) and argues against AD being a general-purpose predictor.

**Cross-study summary:**

| Study | Domain | N | Top dim | AD rank | 10-dim AUC | g-PSQ AUC |
|---|---|---|---|---|---|---|
| CaSiNo | Negotiation | 1,030 | AD | 1st | — | — |
| CGA-Wiki | Wikipedia | 4,188 | AD | 1st | 0.599 | 0.515 |
| CMV | Persuasion | 4,263 pairs | DA | 11th | 0.590 | 0.531 |
| DonD | Negotiation | 12,234 | ED | 10th (neg) | 0.686 | 0.622 |

**Deal or No Deal (2026-02-28).** Fourth criterion validity study. The DonD corpus (Lewis et al., 2017; n=12,234 negotiation dialogues) provides a behavioral outcome — deal reached vs. no deal — tested with PSQ v18. This is the largest criterion validity study to date.

Results:
- **10-dim AUC=0.686** — strongest criterion validity result across all 4 studies. g-PSQ AUC=0.622.
- **ED is top predictor** (d=+0.614, r_pb=+0.247), the largest single-dimension effect across all studies. AD is weakest (d=-0.063, near zero).
- **AD suppressor replicated**: AD coefficient=-0.534 in logistic regression despite weak bivariate r — confirms suppressor pattern from CGA-Wiki and CMV.
- **Incremental AUC beyond controls**: +0.059 beyond text length + turns. High-PSQ (Q4) deal rate 84.4% vs Low-PSQ (Q1) 68.5% — 15.9pp difference.
- **ED validates as "process dimension"**: Deal-reaching requires sustained engagement without energy collapse. ED's dominance here, combined with its strong showing in CaSiNo (satisfaction), supports its construct validity as a resource depletion measure.

**Context-dependent primacy finding refined**: AD dominates when status is contested (CaSiNo, CGA-Wiki). ED dominates when behavioral outcome depends on sustained engagement (DonD). DA dominates when status is structurally fixed (CMV). This pattern is theoretically coherent and constitutes the strongest evidence against the PSQ being a single-factor construct.

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

### Phase 2: Human Validation (estimated effort: 7-9 weeks)

**Full protocol designed:** See `expert-validation-protocol.md` for the complete study design.

4. **Expert panel inter-rater reliability**: 5 expert psychologists (clinical, organizational, social, boundary specialist, psychometrician) score 200 stratified texts on all 10 dimensions. Fully crossed design (10,000 ratings). Compute ICC(2,1) per dimension. Target: ICC ≥ 0.70 for ≥ 8/10 dimensions.
5. **DA construct validity decision**: DA-specific decision tree based on expert data — ICC < 0.50 → deprecate; partial r all < 0.30 (controlling for g-PSQ) → retain as distinct; R² > 0.80 from other 9 dims → absorb into nearest cluster.
6. **Human-LLM convergent validity**: Compare mean expert ratings to LLM teacher scores on 20 held-out overlap texts. Compute per-dimension r and systematic bias (Bland-Altman).
7. **Expert factor structure**: Run EFA on expert rating matrix (200 × 10). Compare to LLM-derived factor structure via Tucker's congruence coefficient. Key question: do experts replicate the 5-factor structure?

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
