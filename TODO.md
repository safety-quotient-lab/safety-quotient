# PSQ Project TODO

Last updated: 2026-02-28

## Priority 1: Immediate

### Rename authority_dynamics → power_positioning

**Why:** Three criterion validity studies (CaSiNo, CGA-Wiki, CMV) reveal that AD measures *status negotiation in peer contexts*, not formal authority or hierarchy. The current name "authority_dynamics" implies institutional power structures, but the construct actually measures:
- Epistemic positioning ("I know better than you")
- Moral claims ("My needs should take priority")
- Status contestation ("Who has the right to define reality here?")
- Relational power moves (assertions, challenges, deference)

**Evidence for rename:**
- AD predicts strongly in peer contexts (Wikipedia editors, Reddit commenters, campsite negotiation) where formal authority is absent
- AD is weakest in CMV where status is *fixed* (OP holds delta power, challengers must persuade)
- AD-residual correlates with epistemic markers (second-person pronouns r=+0.202, question marks r=+0.235) not emotional markers
- Theory 3 (status negotiation, journal §24) has the best explanatory coverage

**What needs to change:**
- [ ] `psq-definition.md` — dimension name, description, scoring rubric, score anchors
- [ ] `final-state.md` — dimension specification
- [ ] `data/dataset_mappings.json` — dimension key (affects composite pipeline)
- [ ] `scripts/distill.py` — `DIM_NAMES` list, any hardcoded references
- [ ] `scripts/eval_held_out.py` — dimension references
- [ ] `scripts/label_separated.py` — dimension abbreviations (ad → pp?)
- [ ] `data/schema.sql` — dimension enum/values if constrained
- [ ] `data/psq.db` — migrate existing scores (UPDATE scores SET dimension = 'power_positioning' WHERE dimension = 'authority_dynamics')
- [ ] All criterion validity scripts — variable names
- [ ] Expert validation protocol — test both labels ("authority_dynamics" vs "power_positioning") to see which produces higher inter-rater reliability
- [ ] Consider backward compatibility for any external consumers of the ONNX model

**Risk:** Renaming mid-project creates documentation churn. Alternative: keep internal name as `authority_dynamics` but note in all public-facing docs that the construct is better described as "power positioning." Decide after expert validation.

**Recommendation:** Wait for expert validation (§19) to test both labels. If "power positioning" produces higher ICC, rename everywhere. If "authority_dynamics" is equally reliable, keep the original but add a "also known as" note.

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

### Deal or No Deal criterion study [IN PROGRESS]

**Why:** Tests prediction T3b from journal §24 — AD should predict deal (relational outcome) but NOT points scored (resource allocation). This would provide causal-directional evidence, not just correlations.

**Status:** Script ready (`scripts/criterion_dealornodeal.py`), currently running in background. Uses v16 model from `models/psq-student/`.

**Expected outcomes:**
- If AD predicts deal but not points → Theory 3 (status negotiation) strongly supported
- If AD predicts both → Theory 1/2 not ruled out, AD may be more general
- If AD predicts neither → CaSiNo/CGA-Wiki findings may be domain-specific

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

### Bifactor model architecture for DistilBERT

**Why:** A bifactor model separates g-PSQ from dimension-specific residuals. The residuals are what carry context-dependent predictive information (§22, §25). Current architecture: shared projection → 10 independent heads. Bifactor would: shared projection → g-factor head + 10 residual heads.

**Design considerations:**
- [ ] Bifactor vs hierarchical: bifactor (g + residuals) vs hierarchical (g → clusters → dims). Literature (Reise, 2012; Rodriguez et al., 2016) slightly favors bifactor for prediction tasks.
- [ ] Training objective: should g-factor head predict g-PSQ (average of 10 dims)? Or should it be learned implicitly via structural constraint?
- [ ] Residual orthogonality: enforce orthogonality between g and residual heads (Schmid-Leiman) or allow correlation (exploratory bifactor)?
- [ ] Practical impact: does bifactor improve held-out r? Or is this purely interpretive (same predictions, better decomposition)?
- [ ] Factor-informed clustering: constrain residual heads to load on their assigned cluster (HI/TE/CC → Hostility/Threat, CO/TC → Relational Contract, etc.)

**Prerequisite:** Understand current shared projection layer behavior — is it already implicitly learning a g-factor? Check by computing correlation between shared projection output and g-PSQ across held-out texts.

### Score broad-spectrum labeling batch

**Status:** 300 texts extracted to `/tmp/psq_separated/`, ready for separated scoring across 10 Claude Code sessions.

**Design:** 150 random + 100 single-dim keyword + 50 multi-dim keyword texts. Intended to provide varied scores across ALL dimensions simultaneously (vs previous batches that were dimension-focused).

**Expected impact:** Reduce score-5 concentration for all dimensions. Previous focused batches (CO, RB, CC, TE) produced neutral scores on non-target dimensions.

## Priority 3: Future

### Expert validation panel

Status: Protocol designed (§19), recruitment not started. 5 expert psychologists, 200 texts, all 10 dims, 10,000 ratings. ICC(2,1) target ≥ 0.70. Estimated 7-9 weeks, $5,625-$15,000.

**New consideration:** Test "authority_dynamics" vs "power_positioning" labels in the expert study to determine optimal construct name.

### Additional criterion validity studies

- **Deal or No Deal** — in progress (see above)
- **Workplace communication** — predict manager ratings, 360 feedback, or exit interview sentiment
- **Therapeutic alliance** — predict WAI scores from therapy transcripts
- **Educational discourse** — predict student engagement or learning outcomes from classroom discussion

### Turn-by-turn temporal analysis

Test prediction T2 from journal §24: does AD deteriorate before HI/TE in CGA-Wiki conversations that derail? Cross-lagged correlation analysis. Requires turn-level scoring (currently score full conversations).

### ONNX model re-export after v18

If v18 is promoted, re-export ONNX + INT8 quantization + recalibrate.
