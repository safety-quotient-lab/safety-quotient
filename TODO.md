# PSQ Project TODO

Last updated: 2026-02-28

## Priority 1: Immediate

### Score production pct batch (200 texts × 10 dims) [COMPLETE — RETRACTED]

**Status:** COMPLETE but RETRACTED. 200 texts scored and ingested. FA v3 (§47) showed pct scoring *collapses* dimension differentiation (eigenvalue 9.41 = 94.1% shared variance). v20 training confirmed no benefit (held-out_r=0.600, flat vs v19). Integer 0-10 scale retained.

**Lesson:** Pct scoring triggers anchoring-and-adjustment heuristic — scorer locks onto global impression at finer granularity instead of differentiating better.

### Rename authority_dynamics → power_positioning [DEFERRED — WILL NOT RENAME]

**Decision (2026-02-28): DEFERRED indefinitely.** The rename is not being pursued. Rationale: **fidelity with the official psychological safety taxonomy.** The name `authority_dynamics` aligns with how power and authority are discussed in the established psychological safety literature (Edmondson 1999, French & Raven 1959, the instruments grounding this dimension). Renaming to `power_positioning` would be a custom neologism that departs from that taxonomy without clear publication benefit. The construct name stays as `authority_dynamics`.

**What the criterion validity evidence actually shows:** AD measures *status negotiation dynamics* in peer contexts — a specific mechanism within authority dynamics, not a departure from it. This is an empirical finding about *where* AD predicts (contested peer status), not evidence that the name is wrong. The name captures the broader theoretical frame; the criterion studies narrow where within that frame the predictive signal lives.

**What changes instead:**
- [ ] Update the AD description in `psq-definition.md` §9 to reflect the peer-context status negotiation finding (the rubric anchors describe formal/institutional power; they should broaden to include epistemic positioning and relational status moves)
- [ ] Add a note in the criterion validity section of `psq-definition.md` referencing the status negotiation theory (journal §24, Theory 3)
- No DB migration, no script changes, no ONNX compatibility risk

**Prior rename rationale (retained for reference):**
- AD predicts strongly in peer contexts (Wikipedia editors, Reddit commenters, campsite negotiation) where formal authority is absent
- AD is weakest in CMV where status is *fixed*
- AD-residual correlates with epistemic markers (second-person pronouns r=+0.202, question marks r=+0.235)
- Theory 3 (status negotiation, journal §24) has the best explanatory coverage

### Context-aware scoring API design

**Why:** g-PSQ (single score) is near-chance across all three criterion studies (AUC 0.515–0.531). Profile shape carries the signal. But different dimensions matter for different use cases:

| Use case | Key dimensions | Rationale |
|---|---|---|
| Content moderation (derailment risk) | AD, HI, DA | AD strongest predictor in CGA-Wiki; HI/DA in multivariate model |
| Persuasion quality | DA, CC, TC | DA top predictor in CMV; CC/TC strongest multivariate coefficients |
| Negotiation outcomes | AD, DA, HI | AD/DA top in CaSiNo; HI significant |
| Workplace safety assessment | AD, ED, TC | Status negotiation + resource depletion + trust |
| Therapeutic conversation quality | RC, RB, CC, ED | Internal resources + recovery + cooling |

**Design questions to resolve:**
- [ ] Should the API return raw 10-dim scores + a context-weighted composite? Or raw scores + recommended weights per use case?
- [ ] Should context be user-specified ("I'm building a moderation tool") or auto-detected from text features?
- [ ] Does this belong in the ONNX model (custom post-processing) or in the application layer?

### Deal or No Deal criterion study [COMPLETE — v23 RERUN DONE]

**Status:** Complete (§39, original v18). v23 rerun complete (§60, 2026-02-28). AUC=0.732 (was 0.686 — strongest of 4 studies). TE top bivariate predictor (d=+0.801, was ED). T3b confirmed: AD predicts deal (+0.138) but not points (−0.070). Context-dependent primacy confirmed across 4 studies.

## Priority 2: Important

### Publication framing and paper outline

See `journal.md` and `Publication Narrative` section below. The criterion validity battery (3 studies) with context-dependent predictive primacy is the publication centerpiece.

**Target venue:** Computational linguistics / NLP+psychology intersection (e.g., EMNLP, ACL, Behavior Research Methods, Journal of Personality Assessment)

**Key findings to highlight:**
1. 10-dim PSQ trained via knowledge distillation (LLM → DistilBERT)
2. Three independent criterion validity studies with real-world outcomes
3. Profile shape >> average across all studies (the multi-dimensional architecture is psychometrically justified)
4. Context-dependent predictive primacy (AD in contested-status, DA in fixed-status) — dimensions are genuinely distinct
5. Factor structure: general factor + 5 clusters, but singletons (AD, ED) carry unique predictive signal
6. The construct the model learned (power positioning) is more nuanced than what was intended (authority dynamics) — a case study in emergent construct validity

### Bifactor model architecture [EVALUATED — NOT ADOPTED]

**Status:** Option A implemented (`--bifactor` flag in distill.py). v19b evaluated: g-head learned well (g_r=0.594) but per-dim test_r dropped (0.509→0.502) — capacity competition in 384-dim projection layer.

**Decision (2026-02-28):** Bifactor architecture is NOT the right approach. Structural analysis (§51) established that the g-factor is real co-variation (range/extremity effect), not scorer artifact. The PSQ should decompose **hierarchically** (PSQ → clusters → dimensions), not bifactor (which treats g as orthogonal to group factors, flattening the hierarchy). The g-factor IS the construct at its broadest level.

**Alternative adopted:** Middle-g text enrichment — enrich training data with texts from g ∈ [3, 4.5) ∪ [5.5, 7] where dimension-specific signal is strongest, preserving the hierarchical decomposition.

### Score broad-spectrum labeling batch [COMPLETE]

**Status:** Complete. 300 texts × 10 dims = 3,000 scores ingested. Drove v19 improvements (held-out_r=0.600).

## Priority 3: Future

### Expert validation panel

Status: Protocol designed (§19), recruitment not started. 5 expert psychologists, 200 texts, all 10 dims, 10,000 ratings. ICC(2,1) target ≥ 0.70. Estimated 7-9 weeks, $5,625-$15,000.

**New consideration:** Test "authority_dynamics" vs "power_positioning" labels in the expert study to determine optimal construct name.

### Additional criterion validity studies

- **Deal or No Deal** — COMPLETE (§39, §60 v23 rerun — AUC=0.732)
- **Workplace communication** — predict manager ratings, 360 feedback, or exit interview sentiment
- **Therapeutic alliance** — predict WAI scores from therapy transcripts
- **Educational discourse** — predict student engagement or learning outcomes from classroom discussion

### Middle-g text enrichment (Option B) [COMPLETE]

**Status:** Complete. midg batch (250 texts × 10 dims) scored and ingested. v22a ablation confirmed proxy removal is the dominant intervention; midg enrichment alone (v22b) regressed. Middle-g batch data contributed to v22c and v23 training sets.

**Why:** Structural analysis shows the g-factor is a range/extremity effect. Extreme texts (g<3 or g>7) contribute pure valence signal (EV1=82.8%, uniform loadings). Middle texts (g 4-6) show genuine dimension differentiation (EV1=38.7%, structured loadings). Enriching training with middle-g texts will improve dimension-specific prediction without modifying the scoring instrument.

**Approach options:**
1. Model-guided selection from unlabeled pool (pre-score 15K texts with v21, select from informative middle band)
2. Upgrade existing proxy texts (1,900 texts with 5+ dims) to full 10-dim separated-llm labels
3. Combined selection from both pools

**Next:** Analyze v21 pre-scoring results to determine pool yield by g-band and source.

### Scoring rubric review

**Status:** Not started. Priority 3 but substantive.

**Why:** The score anchors in `psq-definition.md` were written at construct inception, before criterion validity studies revealed what the model actually learned. Key discrepancies likely exist:
- AD (authority_dynamics): rubric describes institutional authority/hierarchy, but the model actually measures *status negotiation* in peer contexts (epistemic positioning, moral claims, relational power moves). See journal §24.
- Other dimensions may have similar drift between intended construct and learned construct.

**Approach:**
- [ ] Review each dimension's score anchors (1-3, 4-6, 7-9) against actual high/low-scoring texts in the held-out set
- [ ] For each dimension, sample 5 texts at score extremes (held-out predictions <3 and >7) and verify anchors match observed content
- [ ] Update anchors where the model's learned construct diverges from the original definition
- [ ] Prioritize AD (known discrepancy), ED (singleton, unclear construct), DA (weak factor loading)

### Criterion validity summary table [COMPLETE]

**Status:** Complete. `criterion-validity-summary.md` created and maintained as the canonical reference. Updated to v23 numbers (2026-02-28): CMV AUC=0.5735, DonD AUC=0.732. Cross-referenced from distillation-research.md §59/§60, psychometric-evaluation.md §3g.

### Turn-by-turn temporal analysis

Test prediction T2 from journal §24: does AD deteriorate before HI/TE in CGA-Wiki conversations that derail? Cross-lagged correlation analysis. Requires turn-level scoring (currently score full conversations).

### ONNX model re-export [COMPLETE]

v21 promoted to production. ONNX re-exported (254 MB / 64 MB quantized).
