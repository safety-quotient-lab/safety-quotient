# Expert Panel Validation Protocol: PSQ Dimensions

**Version:** 1.0 (2026-02-28)
**Principal Investigator:** [TBD]
**Status:** Draft — protocol design complete, recruitment not started

---

## 1. Purpose

This study addresses two critical psychometric gaps identified in the PSQ evaluation:

1. **Inter-rater reliability**: No evidence exists that independent human raters agree on PSQ dimension scores. Without this, the instrument rests entirely on LLM teacher consensus (a single model's judgment).

2. **DA construct validity**: Defensive Architecture loads diffusely across all empirical factors (max promax loading = 0.332 at 5-factor level, mean r with other dims = 0.480). The question: is DA a distinct, scorable construct, or an artifact of general safety assessment?

### Primary Research Questions

- **RQ1**: Can trained raters achieve acceptable inter-rater reliability (ICC ≥ 0.70) on each of the 10 PSQ dimensions when scoring text content?
- **RQ2**: Is Defensive Architecture empirically separable from the other 9 dimensions in expert ratings, or does it collapse into a general factor?
- **RQ3**: How do expert ratings compare to LLM teacher scores (convergent validity)?

---

## 2. Design Overview

| Element | Specification |
|---|---|
| Design | Fully crossed rating study: all raters score all texts on all dimensions |
| Raters | 5 expert psychologists (clinical, organizational, or social) |
| Texts | 200 texts (stratified sample from training corpus + unlabeled pool) |
| Dimensions | All 10 PSQ dimensions (each text scored on all 10) |
| Total ratings | 5 raters × 200 texts × 10 dimensions = 10,000 ratings |
| Primary analysis | ICC(2,1) per dimension (two-way random, single measures) |
| Secondary analysis | Factor analysis of expert ratings; DA discriminant validity |

---

## 3. Rater Selection

### Inclusion Criteria

- Doctoral-level (PhD or PsyD) in clinical, organizational, counseling, or social psychology
- Minimum 3 years post-doctoral clinical or research experience
- Familiarity with at least 3 of the 10 PSQ source instrument families (e.g., DERS, NAQ, DSQ-40, CD-RISC, Edmondson)
- No prior involvement in PSQ development

### Desired Expertise Distribution

| Rater | Primary Expertise | Relevant to |
|---|---|---|
| R1 | Clinical psychology (defense mechanisms, emotion regulation) | DA, RC, RB, ED |
| R2 | Organizational psychology (team safety, authority, trust) | AD, TC, CO |
| R3 | Social psychology (hostility, intergroup threat, de-escalation) | HI, TE, CC |
| R4 | Clinical or counseling (boundary work, assertiveness training) | DA (critical for RQ2) |
| R5 | Psychometrics or measurement specialist | All (methodological anchor) |

### Rater Training

1. **Self-study phase (2 hours)**: Raters receive the PSQ construct definitions (`psq-definition.md`), the 0-10 scoring anchors for all 10 dimensions (`instruments.json`), and 10 worked examples with model scores and rationale.

2. **Calibration session (2 hours, synchronous)**: All raters independently score the same 20 calibration texts. Scores are compared, disagreements discussed, and scoring rules clarified. Calibration texts are not included in the study sample.

3. **Practice round (1 hour)**: Raters independently score 10 additional practice texts. ICC is computed. If any dimension falls below 0.50, additional calibration is conducted before proceeding.

### Rater Independence

- Raters score independently (no consultation after calibration)
- Texts presented in randomized order (different randomization per rater)
- Each scoring session limited to 20-25 texts to prevent fatigue effects
- Raters do not see LLM scores or other raters' scores during the study

---

## 4. Text Sample

### Sample Size Justification

200 texts with 5 raters provides adequate power for ICC estimation:
- For ICC = 0.70, 95% CI width ≈ ±0.08 with n=200, k=5 (Bujang & Baharum, 2017)
- For detecting ICC difference from 0.50 (poor) vs 0.70 (good) at α=0.05, power > 0.95

### Stratified Sampling

| Stratum | n | Source | Purpose |
|---|---|---|---|
| **DA-extreme (low)** | 30 | Texts with LLM DA score ≤ 3 | Tests DA at the low end |
| **DA-extreme (high)** | 30 | Texts with LLM DA score ≥ 7 | Tests DA at the high end |
| **DA-neutral** | 20 | Texts with LLM DA score 4.5-5.5 | Tests DA discriminability at midpoint |
| **General safety spread** | 60 | Texts spanning full range on g-PSQ | General reliability across all dims |
| **Factor-informative** | 40 | 8 texts per 5-factor cluster with extreme cluster scores | Tests factor structure in expert ratings |
| **Held-out overlap** | 20 | 20 texts from existing held-out set | Direct LLM-expert comparison on benchmarked texts |

### Text Selection Criteria

- Minimum 30 words, maximum 500 words
- English only
- Remove texts with personally identifiable information
- Include diverse content types: workplace communication, online comments, support conversations, negotiation, news commentary
- Remove exact duplicates

### Text Presentation

Each text presented with:
- Sequential ID (e.g., "Text 037")
- The text content (unmodified)
- NO source label, NO LLM scores, NO other metadata

---

## 5. Rating Procedure

### Rating Interface

Each rater receives a scoring workbook (spreadsheet or web form) with:
- One row per text
- 10 columns for dimension scores (0-10 integer scale)
- 10 columns for dimension confidence (0.0-1.0 in 0.1 increments)
- 1 column for free-text notes (optional)

### Dimension Scoring Instructions

For each text, rate on each of the 10 dimensions using the 0-10 scale:
- 0 = most negative/harmful expression of this construct
- 5 = neutral — this text has no relevance to this construct
- 10 = most positive/supportive expression of this construct

**Critical instruction for DA**: Score Defensive Architecture *specifically* on whether the text supports, respects, or undermines interpersonal boundaries and self-protective behaviors. This is distinct from:
- General safety or positivity (captured by g-PSQ)
- Emotion regulation support (captured by Regulatory Capacity)
- Trust or relational quality (captured by Trust Conditions)
- Power dynamics (captured by Authority Dynamics)

If the text has no relevance to boundaries, defenses, or self-protection, score DA = 5 (neutral).

### Confidence Rating Instructions

For each dimension score, rate your confidence that the score accurately reflects the text on this specific dimension:
- 0.0-0.3: Low confidence — construct is ambiguous for this text
- 0.4-0.6: Moderate confidence — reasonable interpretation but others could differ
- 0.7-0.9: High confidence — clear textual evidence supports this score
- 1.0: Certainty — unambiguous textual evidence

### Session Structure

- 8-10 scoring sessions per rater
- 20-25 texts per session (approximately 60-90 minutes)
- Minimum 24 hours between sessions (prevent carry-over effects)
- Total rater time commitment: approximately 15-20 hours over 2-3 weeks

---

## 6. Analysis Plan

### 6a. Primary Analysis: Inter-Rater Reliability

**Metric:** ICC(2,1) — two-way random effects, single measures, absolute agreement

**Decision criteria:**

| ICC Range | Interpretation | Action |
|---|---|---|
| ≥ 0.75 | Excellent | Dimension validated for scoring |
| 0.60-0.74 | Good | Dimension validated with caveats |
| 0.50-0.59 | Moderate | Dimension needs refinement |
| < 0.50 | Poor | Dimension not reliably scorable from text |

**Report:** ICC with 95% CI for each of the 10 dimensions.

### 6b. DA-Specific Analysis (RQ2)

**Test 1: DA discriminant validity**

Compute partial correlations between expert DA scores and each other dimension, controlling for g-PSQ (mean of all 10 dimensions). If DA's partial correlations are all < 0.30, DA captures something beyond the general factor. If partial correlations remain > 0.50 with multiple dimensions, DA is redundant.

**Test 2: DA incremental validity**

Stepwise regression: predict DA from the other 9 dimensions. If R² > 0.80 (DA is >80% predictable from other dims), DA adds negligible unique information. If R² < 0.60, DA captures substantial unique variance.

**Test 3: DA factor loading in expert data**

Run EFA (promax rotation) on the expert 10-dimension rating matrix. Key question: does DA load >0.40 on a single factor, or is it diffuse (as in LLM data)?

**Decision tree for DA:**

```
Expert ICC(DA) < 0.50?
  YES → DA is not reliably scorable from text
        → DEPRECATE (absorb into nearest factor or remove)
  NO  → Continue to discriminant test

Expert DA partial r with all other dims < 0.30?
  YES → DA is distinct
        → RETAIN (assign to best-fit factor or keep standalone)
  NO  → Continue to incremental test

Expert DA R² from other 9 dims > 0.80?
  YES → DA is redundant
        → ABSORB (merge with highest-correlated cluster)
  NO  → DA has some unique variance but is partially overlapping
        → RETAIN with caveat, assign to best-fit cluster
```

### 6c. Expert vs. LLM Convergent Validity (RQ3)

For each dimension, compute:
- Pearson r between mean expert rating and LLM teacher score (across the 20 held-out overlap texts)
- Systematic bias: mean(expert - LLM)
- Bland-Altman plot: identify dimensions where LLM and experts systematically diverge

**Interpretation:**
- r ≥ 0.70: Strong convergent validity — LLM teacher captures the construct similarly to experts
- r = 0.50-0.69: Moderate — LLM and experts partially agree
- r < 0.50: Poor — LLM may be measuring something different than experts intend

### 6d. Full Factor Analysis (expert ratings)

Run EFA on the 200 × 10 expert-rated matrix (using mean expert scores):
- KMO, Bartlett's test
- Scree analysis, parallel analysis, BIC model selection
- Varimax and promax rotation at 2, 3, 5, and 10 factors
- Compare expert factor structure to LLM-derived factor structure (Tucker's congruence coefficient)

Key question: do experts produce the same 5-factor structure as the LLM, or a different one?

### 6e. Confidence Calibration (exploratory)

Compare expert confidence ratings to actual inter-rater agreement:
- For each text × dimension, compute within-text SD across raters
- Correlate mean expert confidence with (1 - within-text SD)
- Test whether expert confidence is better calibrated than LLM confidence

---

## 7. Materials Checklist

| Material | Status | Notes |
|---|---|---|
| PSQ construct definitions document | **Ready** | `psq-definition.md` |
| Scoring rubric (0-10 anchors, all 10 dims) | **Ready** | `instruments.json`, condensed version needed |
| Worked examples (10 texts with model scores) | **Needed** | Select from held-out set with rationale |
| Calibration texts (20 texts) | **Needed** | Select from diverse sources, full score range |
| Practice texts (10 texts) | **Needed** | Subset of calibration difficulty |
| Study texts (200 texts) | **Needed** | Stratified sample per §4 |
| Scoring workbook template | **Needed** | Spreadsheet or web form |
| Rater instructions document | **Needed** | Detailed, standardized |
| Informed consent form | **Needed** | IRB-dependent |
| Data analysis scripts | **Partially ready** | ICC computation, factor analysis |

---

## 8. Ethical Considerations

- **IRB/Ethics review**: Required if conducted at a university or if results are intended for publication. May qualify for exempt review (analysis of existing text data, expert raters are not research subjects).
- **Text content**: Some texts contain hostile, threatening, or distressing content (by design — the instrument must score such content). Raters should be informed of this during recruitment.
- **Compensation**: Expert raters should be compensated at professional consulting rates (suggested: $75-150/hour, total $1,125-3,000 per rater, $5,625-15,000 total for 5 raters).
- **Data handling**: Expert ratings are research data, not clinical assessments. Store securely, de-identify texts before sharing.

---

## 9. Timeline

| Phase | Duration | Activities |
|---|---|---|
| 1. Preparation | 2 weeks | Finalize text sample, prepare materials, recruit raters |
| 2. Training | 1 week | Self-study, calibration session, practice round |
| 3. Rating | 2-3 weeks | Independent scoring (8-10 sessions per rater) |
| 4. Analysis | 1-2 weeks | ICC, factor analysis, DA decision, LLM comparison |
| 5. Reporting | 1 week | Write up findings, update psychometric evaluation |
| **Total** | **7-9 weeks** | |

---

## 10. Expected Outcomes and Impact

### Best case
- 8+ dimensions achieve ICC ≥ 0.70
- DA achieves ICC ≥ 0.60 and shows partial distinctness (R² < 0.70)
- Expert factor structure aligns with LLM structure (Tucker's φ > 0.85)
- Expert-LLM r ≥ 0.60 on most dimensions
- **Impact**: PSQ validated for research use with human expert endorsement

### Realistic case
- 6-7 dimensions achieve ICC ≥ 0.70
- DA achieves ICC 0.50-0.65 with weak distinctness
- Expert factor structure shows 3-4 factors (less than LLM's 5)
- Expert-LLM r = 0.40-0.65 on most dimensions
- **Impact**: Most dimensions validated; DA absorbed into nearest cluster; LLM teacher recalibrated on dimensions with poor convergence

### Worst case
- <5 dimensions achieve ICC ≥ 0.70
- DA achieves ICC < 0.50
- Expert factor structure fundamentally different from LLM
- Expert-LLM r < 0.40 on multiple dimensions
- **Impact**: Fundamental construct revision needed; several dimensions may be unmeasurable from text; project pivots to fewer, better-defined dimensions

---

## 11. References

- AERA, APA, & NCME. (2014). *Standards for Educational and Psychological Testing*. AERA.
- Bujang, M. A., & Baharum, N. (2017). A simplified guide to determination of sample size requirements for estimating the value of intraclass correlation coefficient. *Archives of Orofacial Sciences, 12*(1), 1-11.
- Cicchetti, D. V. (1994). Guidelines, criteria, and rules of thumb for evaluating normed and standardized assessment instruments in psychology. *Psychological Assessment, 6*(4), 284-290.
- Hallgren, K. A. (2012). Computing inter-rater reliability for observational data: An overview and tutorial. *Tutorials in Quantitative Methods for Psychology, 8*(1), 23-34.
- McGraw, K. O., & Wong, S. P. (1996). Forming inferences about some intraclass correlation coefficients. *Psychological Methods, 1*(1), 30-46.
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. *Psychological Bulletin, 86*(2), 420-428.
- Tucker, L. R. (1951). A method for synthesis of factor analysis studies. *Personnel Research Section Report No. 984*.
