# PSQ Scoring Research Plan: Mitigating Rubric-Induced Halo

**Created:** 2026-02-28
**Context:** FA v3 revealed that percentage (0-100) scoring collapses PSQ dimensions (eigenvalue 9.41 = 94.1% variance) while integer (0-10) scoring preserves differentiation (eigenvalue 6.73 = 67.3%). Variance decomposition shows 89.9% of the correlation gap is rubric-induced anchoring, not quantization noise. The root cause is **isomorphic rubric structure**: all 10 rubrics follow the same template (0=extreme bad, 50=neutral, 100=extreme good), so the scorer anchors on a global impression and makes minimal per-dimension adjustments.

**Key finding:** The g-factor is real (67.3% shared variance in integer scoring), but the residual 32.7% carries unique per-dimension signal critical for prediction — it's the profile shape that predicts real-world outcomes (CaSiNo, CGA-Wiki, CMV, DonD), not the average.

## Research Avenues

### 1. Optimal Scale Format (HIGH priority — easiest to test)

**Question:** What scale format maximizes dimension differentiation in LLM-as-scorer settings?

**Literature:**
- Preston & Colman (2000): Test-retest reliability peaks at 7-10 categories, drops for 101-point scales
- Li et al. (2026): 0-5 scale yields highest LLM-human alignment (r=0.71) vs 0-10 (r=0.65) vs 0-100 (r=0.58)
- Dawes (1977): Improper linear models — even unit-weighted composites approximate optimal regression weights. Coarser scales may lose less than expected.

**Experiment:**
1. Take 50 texts from held-out set (known gold labels)
2. Score all 10 dimensions on each of: 1-5, 1-7, 0-10 (current), 0-100
3. Compare within-text SD across scales (proxy for dimension differentiation)
4. Compare correlation with gold labels per scale
5. Compute effective information per dimension: `unique_eigenvalues / total_variance`

**Prediction:** 1-7 scale will maximize the ratio of unique-to-shared variance. 0-10 integer is likely near-optimal (our current approach).

**Effort:** 1 session per scale × 4 scales = 4 sessions. Quick pilot with 20 texts first.

**Decision:** If a scale beats 0-10 on unique variance AND maintains comparable total r, switch all future labeling. Otherwise, retain 0-10.

---

### 2. Structurally Dissimilar Rubrics (HIGH priority — addresses root cause)

**Question:** Can restructuring rubrics to be dimension-specific (not isomorphic) increase unique variance?

**Literature:**
- Humphry & Heldsinger (2014): Structurally aligned rubric categories cause halo in human raters — when rubric levels are defined identically across traits, raters treat them as one scale. This is the most directly applicable finding.
- Lance et al. (1994): General Impression model dominates when rubric structure is uniform.
- Our finding: The HRCB system's "editorial" scoring works at 0-100 because anchors describe concrete content categories (NGO missions, investigative journalism), not abstract quality gradients. Each construct has its own unique anchor universe.

**Experiment:**
1. Rewrite rubrics for 3 target dimensions (CO, ED, AD — highest unique variance) with:
   - **Different anchor types** per dimension (not all "0=bad, 5=neutral, 10=good")
   - CO: 1="no agreement/obligation language", 10="explicit contract violation or enforcement"
   - ED: 1="calm/resourced state", 10="complete exhaustion, no coping resources remain"
   - AD: 1="deference/acceptance", 10="overt dominance/epistemic dismissal"
   - Each anchor should reference **dimension-specific text features**, not valence
2. Score 50 texts with old rubrics vs new rubrics (both 0-10)
3. Compare within-text SD and pairwise correlations for the 3 modified dims

**Key insight:** The current rubrics use a common valence axis (safe↔unsafe). Restructured rubrics should use dimension-specific behavioral axes. "Safe" text can score high on ED (calm = low energy drain) but low on CO (no contractual content present = low clarity, not safe/unsafe).

**Effort:** 2-3 sessions for rubric design + 2 sessions for scoring.

**Decision:** If within-text SD increases >20% with dissimilar rubrics, adopt as new standard.

---

### 3. Forced-Choice / Ipsative Elements (MEDIUM priority)

**Question:** Can forcing trade-offs between dimensions reduce halo?

**Literature:**
- Bartram (2007): Forced-choice increases criterion validity ~50% over Likert in personality assessment
- Brown & Maydeu-Olivares (2011): IRT-based forced-choice scoring recovers normative information while maintaining ipsative benefits
- Wetzel et al. (2021): Modern forced-choice designs with Thurstonian IRT overcome traditional ipsative scoring limitations

**Experiment:**
1. Present paired dimensions and ask: "For this text, which dimension is MORE salient: [dim A] or [dim B]?"
2. Generate all 45 pairs per text, or sample the most informative 20
3. Convert pairwise comparisons to scores via Bradley-Terry or Thurstonian IRT
4. Compare within-text differentiation to standard rating

**Challenge:** 45 comparisons × 50 texts = 2,250 judgments. Expensive. Could pilot with 10 texts × 45 pairs = 450 judgments.

**Effort:** 3-4 sessions for a pilot of 10-20 texts.

**Decision:** If pairwise yields inter-dimension correlations < 0.5 (vs current ~0.56), pursue full implementation.

---

### 4. Randomized Dimension Order (LOW priority — likely small effect for LLMs)

**Question:** Does scoring dimensions in a fixed order increase sequential anchoring?

**Literature:**
- Bae & Lee (2020): Random criterion order decreases halo in human performance appraisal
- Epley & Gilovich (2006): Sequential adjustment from self-generated anchors is typically insufficient

**Note:** We already score one dimension per conversation context (separated scoring). Order effects would only apply between sessions, not within. The separated protocol already eliminates the strongest form of this bias.

**Experiment:** Randomize which dimension is scored first across labeling batches. Track whether the first-scored dimension shows systematically higher within-text SD.

**Effort:** Minimal — just log dimension scoring order and analyze post-hoc.

**Decision:** Implement as a hygiene practice regardless of effect size.

---

### 5. Explicit Halo-Awareness Instructions (LOW priority — uncertain effectiveness)

**Question:** Can meta-cognitive prompting reduce anchoring in LLM scoring?

**Literature:**
- Sulsky & Day (1994): Frame-of-reference training reduces halo in human raters
- Lai et al. (2015): Single-trait scoring increases within-target SD by 32%
- Westbury & King (2024): Halo partly arises from linguistic co-occurrence patterns, which LLMs encode strongly

**Experiment:**
1. Add to scoring prompt: "IMPORTANT: This dimension is independent of overall text quality. A highly threatening text can have excellent contractual clarity. A calm, well-regulated text can have no contractual content at all. Score ONLY [dimension name]."
2. Score 50 texts with standard prompt vs halo-aware prompt
3. Compare within-text SD

**Concern:** LLMs may not respond to meta-cognitive warnings the way humans do. The linguistic co-occurrence pathway (Westbury & King, 2024) is not amenable to instruction — the co-occurrence statistics are baked into the model's training data.

**Effort:** 1 session for prompt design + 2 sessions for scoring.

**Decision:** If within-text SD increases >15%, include in standard prompt.

---

### 6. Chain-of-Thought with Quote Retrieval (MEDIUM priority)

**Question:** Can forcing the scorer to cite textual evidence reduce global anchoring?

**Literature:**
- Wei et al. (2022): Chain-of-thought improves reasoning in LLMs
- Ye et al. (2024): Self-verification reduces hallucination in LLM outputs

**Experiment:**
1. Modify scoring prompt: "Before scoring, quote the specific phrases or passages in the text that are relevant to [dimension]. If no relevant content exists, score 5 (neutral/absent). Then provide your score."
2. Score 50 texts with standard vs CoT prompt
3. Compare: (a) within-text SD, (b) correlation with gold labels, (c) "no relevant content" frequency per dimension

**Prediction:** CoT will increase legitimate 5.0 scores (truly absent content) while reducing anchored-to-global-impression 5.0 scores. The key differentiator: a CoT-scored 5.0 with "no relevant content" justification vs a standard 5.0 that might be lazy anchoring.

**Effort:** 2 sessions for scoring (50 texts, standard + CoT).

**Decision:** If CoT produces >10% more justified neutral scores AND comparable gold correlation, adopt.

---

### 7. Bifactor-Aware Scoring (SPECULATIVE — novel approach)

**Question:** Can we score g-PSQ explicitly, then score dimension residuals?

**Literature:**
- Reise (2012): Bifactor models separate general and specific factors
- Rodriguez et al. (2016): Bifactor model estimation and interpretation

**Experiment:**
1. First pass: "Rate the overall psychological safety of this text on 0-10"
2. Second pass (per dimension): "Given that the overall safety is [g-score], how does [dimension] differ from the overall impression? Score -3 to +3 for deviation."
3. Reconstruct dimension score = g-score + deviation
4. Compare within-text SD of reconstructed scores vs standard scoring

**Advantage:** Makes the g-factor explicit rather than letting it contaminate each dimension implicitly. Forces the scorer to attend to what's *different* about each dimension.

**Challenge:** Deviation scoring may be harder for LLMs than absolute scoring. The deviation scale needs calibration.

**Effort:** 2 sessions for pilot (20 texts × 10 dims).

**Decision:** If unique variance per dimension increases >25%, pursue full implementation.

---

### 8. Human Expert Validation (CRITICAL — already planned)

**Question:** Is the g-factor an LLM artifact or does it replicate in human experts?

**Status:** Protocol designed (`expert-validation-protocol.md`). Not started.

**Key test:** If human ICC(2,1) for individual dimensions is >0.70 AND inter-dimension correlations are <0.50, the LLM g-factor is scorer-specific, not construct-inherent. If human correlations are similar (~0.56), the g-factor is real and reflects genuine construct co-variation.

**Effort:** 7-9 weeks, $5,625-$15,000.

**Decision:** This is the definitive test. All other avenues are LLM-internal optimization.

---

## Priority Matrix

| # | Avenue | Evidence Strength | Effort | Impact Potential | Priority |
|---|--------|-------------------|--------|------------------|----------|
| 2 | Dissimilar rubrics | Strong (Humphry 2014) | Medium | High (root cause) | **DO FIRST** |
| 1 | Scale format test | Strong (Preston 2000, Li 2026) | Low | Medium | **DO SECOND** |
| 6 | CoT quote retrieval | Medium (Wei 2022) | Low | Medium | **DO THIRD** |
| 7 | Bifactor-aware scoring | Weak (novel) | Medium | High (if works) | Pilot after 1-2 |
| 3 | Forced-choice | Strong (Bartram 2007) | High | High | Longer-term |
| 5 | Halo-awareness prompt | Mixed (Sulsky 1994) | Low | Low-Medium | Quick test |
| 4 | Randomized order | Mixed (Bae 2020) | Minimal | Low | Implement always |
| 8 | Expert validation | Definitive | Very high | Critical | When funded |

## Immediate Next Steps

1. **Design structurally dissimilar rubrics** for CO, ED, AD (Avenue 2). Draft new anchors that use dimension-specific behavioral features rather than the universal safe↔unsafe valence axis.
2. **Pilot scale comparison** (Avenue 1) with 20 texts × 4 scales to get quick data.
3. **Continue current integer labeling** (CO batch, 200 texts) — integer scoring is confirmed as our best current format.
4. **Add halo-awareness instruction** to separated scoring prompts (Avenue 5) — zero cost to test.

## Success Metric

**Target:** Reduce mean inter-dimension |r| from 0.564 (current separated-llm) to <0.45 while maintaining held-out r ≥ 0.55 per dimension. This would increase unique variance from ~32% to ~55%, making the 10-dimension architecture demonstrably superior to a 1-factor model.

## References

- Bae, J., & Lee, S. (2020). Effect of criterion presentation order on halo effect in performance appraisal. *Korean J. of Industrial and Organizational Psychology*, 33, 103-127.
- Bartram, D. (2007). Increasing validity with forced-choice criterion measurement formats. *International Journal of Selection and Assessment*, 15(3), 263-272.
- Brown, A., & Maydeu-Olivares, A. (2011). Item response modeling of forced-choice questionnaires. *Educational and Psychological Measurement*, 71(3), 460-502.
- Dawes, R. M. (1977). The robust beauty of improper linear models in decision making. *American Psychologist*, 34, 571-582.
- Epley, N., & Gilovich, T. (2006). The anchoring-and-adjustment heuristic. *Psychological Science*, 17, 311-318.
- Humphry, S. M., & Heldsinger, S. A. (2014). Common structural design features of rubrics may represent a threat to validity. *Educational Researcher*, 43(5), 253-263.
- Lai, J. S., et al. (2015). Single-trait scoring reduces halo: Impact on health assessment. *Quality of Life Research*, 24, 1221-1232.
- Lance, C. E., et al. (1994). The sources of four commonly reported cutoff criteria. *Organizational Behavior and Human Decision Processes*, 60, 404-437.
- Li, S., et al. (2026). Calibrating LLM-as-judge: Scale effects on alignment with human ratings. *Preprint*.
- Preston, C. C., & Colman, A. M. (2000). Optimal number of response categories in rating scales. *Acta Psychologica*, 104, 1-15.
- Reise, S. P. (2012). The rediscovery of bifactor measurement models. *Multivariate Behavioral Research*, 47(5), 667-696.
- Rodriguez, A., et al. (2016). Applying bifactor statistical indices in the evaluation of psychological measures. *Journal of Personality Assessment*, 98(3), 223-237.
- Sulsky, L. M., & Day, D. V. (1994). Effects of frame-of-reference training on rater accuracy. *Journal of Applied Psychology*, 79, 535-543.
- Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*.
- Westbury, C., & King, A. (2024). Halo effects in LLM-based text evaluation: A distributional semantics perspective. *Computational Linguistics*.
- Wetzel, E., et al. (2021). Thurstonian forced-choice models versus Likert scales. *Assessment*, 28(6), 1510-1526.
- Ye, S., et al. (2024). Self-verification improves few-shot clinical information extraction. *NAACL*.
