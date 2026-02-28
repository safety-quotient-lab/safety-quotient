# PSQ Project TODO

Last updated: 2026-02-28

## Priority 1: Immediate

### Score production pct batch (200 texts × 10 dims)

**Status:** `data/labeling-batch-pct-200.jsonl` created (200 texts, 5 sources). Ready for `extract --pct`.

**Why:** Pilot (§44) confirmed 0-100 scale breaks integer bias. Need production-scale data scored with proper separated protocol (1 dim per session) to:
1. Add pct-scored data to training pipeline
2. Re-run factor analysis to test if g-factor inflation drops
3. Determine if percentage scoring improves model resolution

**Workflow:**
```
python scripts/label_separated.py extract --input data/labeling-batch-pct-200.jsonl --pct
# Score each dim in separate sessions
python scripts/label_separated.py ingest --dim <dim> --scores /tmp/scored.json
# Assemble when all 10 done
python scripts/label_separated.py assemble --input data/labeling-batch-pct-200.jsonl --output data/labeling-batch-pct-200-scored.jsonl
# Ingest into DB
python scripts/migrate.py --ingest data/labeling-batch-pct-200-scored.jsonl
```

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

### Deal or No Deal criterion study [COMPLETE]

**Status:** Complete (§39). AUC=0.686 (strongest of 4 studies). ED top predictor (d=+0.614), AD suppressor replicated. Context-dependent primacy confirmed across 4 studies.

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

### Bifactor model architecture [IMPLEMENTED, TRAINING]

**Status:** Option A implemented (`--bifactor` flag in distill.py). v19b training in progress. g-PSQ target = mean of available dim scores. g-head is `nn.Linear(384, 1)` with sigmoid*10, loss weight 1.0.

**Prerequisite confirmed:** r(mean_pred, mean_target) = 0.673 on held-out — sufficient headroom for explicit g-head.

**Next:** Evaluate v19b held-out results. If g-PSQ r > 0.95, the g-head is redundant. If lower, consider Option B (orthogonal decomposition) or C (cluster-mediated).

### Score broad-spectrum labeling batch [COMPLETE]

**Status:** Complete. 300 texts × 10 dims = 3,000 scores ingested. Drove v19 improvements (held-out_r=0.600).

## Priority 3: Future

### Expert validation panel

Status: Protocol designed (§19), recruitment not started. 5 expert psychologists, 200 texts, all 10 dims, 10,000 ratings. ICC(2,1) target ≥ 0.70. Estimated 7-9 weeks, $5,625-$15,000.

**New consideration:** Test "authority_dynamics" vs "power_positioning" labels in the expert study to determine optimal construct name.

### Additional criterion validity studies

- **Deal or No Deal** — COMPLETE (§39)
- **Workplace communication** — predict manager ratings, 360 feedback, or exit interview sentiment
- **Therapeutic alliance** — predict WAI scores from therapy transcripts
- **Educational discourse** — predict student engagement or learning outcomes from classroom discussion

### Scoring rubric review

**Why:** The score anchors in `psq-definition.md` were written at construct inception, before criterion validity studies revealed what the model actually learned. Key discrepancies likely exist:
- AD (authority_dynamics): rubric describes institutional authority/hierarchy, but the model actually measures *status negotiation* in peer contexts (epistemic positioning, moral claims, relational power moves). See journal §24.
- Other dimensions may have similar drift between intended construct and learned construct.

**Approach:**
- [ ] Review each dimension's score anchors (1-3, 4-6, 7-9) against actual high/low-scoring texts in the held-out set
- [ ] For each dimension, sample 5 texts at score extremes (held-out predictions <3 and >7) and verify anchors match observed content
- [ ] Update anchors where the model's learned construct diverges from the original definition
- [ ] Prioritize AD (known discrepancy), ED (singleton, unclear construct), DA (weak factor loading)

### Criterion validity summary table

**Why:** Cross-study comparison data is scattered across journal.md §25, distillation-research.md §34, and psychometric-evaluation.md §3g. Need a single canonical table that lives somewhere accessible for quick reference during writing and presentations.

**Approach:**
- [ ] Create a comprehensive cross-study table with: study, dataset, domain, N, outcome, top predictor, AD rank, 10-dim metric, g-PSQ metric, key finding
- [ ] Include all completed studies (CaSiNo, CGA-Wiki, CMV) plus DonD when complete
- [ ] Place in `psq-definition.md` (Section: Criterion Validity Evidence) or as a standalone `criterion-validity-summary.md`
- [ ] Reference from journal.md, distillation-research.md, and psychometric-evaluation.md

### Turn-by-turn temporal analysis

Test prediction T2 from journal §24: does AD deteriorate before HI/TE in CGA-Wiki conversations that derail? Cross-lagged correlation analysis. Requires turn-level scoring (currently score full conversations).

### ONNX model re-export [COMPLETE]

v19 promoted to production. ONNX re-exported (254 MB / 64 MB quantized).
