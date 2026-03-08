# PSQ Project TODO

Last updated: 2026-03-08

## Priority 1: Immediate

### Cross-scorer concordance remediation [ACTIVE]

**Status:** Concordance study COMPLETE — gate FAILS (ICC 0.495, 1/10 dims pass). 10,000 Opus scores in DB are not interchangeable with Sonnet. Production models (v23/v35) are uncontaminated.

**Sequence:**
1. ✓ Design concordance study (docs/concordance-study-protocol.md)
2. ✓ Score 50 texts × 10 dims with Opus (separated-LLM, blind)
3. ✓ Analyze: mean ICC(2,1) = 0.495 ("poor"), 1/10 pass
4. ✓ Notify psychology-agent (PR #71, T19)
5. ✗ **NEXT:** Re-score 999 Opus-only texts with Sonnet (~3 hrs, 10 sessions × 1 dim)
6. ✗ Retrain with Sonnet-only labels
7. ✗ B3 recalibration (quantile-binned isotonic, all 10 dims — from psychology-agent T17)

See distillation-research.md §72/§73.

### AD + HI range compression augmentation (distillation-research.md §69) [BLOCKED — concordance]

**Status:** HI batch scored with Opus (350 texts) but concordance study revealed Opus labels are not interchangeable with Sonnet (HI bias +0.82). v36 diagnostic confirmed HI did not improve. HI batch must be re-scored with Sonnet as part of concordance remediation.

**Completed:**
1. ✓ Source and extract AD batch: 260 synthetic formal authority texts
2. ✓ Label AD batch (1 session, authority_dynamics only) — 260 scores ingested
3. ✓ v34 training → REJECTED (AD improvement but overall regression)
4. ✓ Source HI batch: 350 texts — `data/hi-augmentation-batch.jsonl`
5. ✓ Label HI batch with Opus — 350 HI scores ingested
6. ✓ v36 diagnostic — HI=0.709 (v35=0.714, −0.005). HI did NOT improve.

**Blocked on:** Concordance remediation (re-score with Sonnet).

---

### Score production pct batch (200 texts × 10 dims) [COMPLETE — RETRACTED]

**Status:** COMPLETE but RETRACTED. 200 texts scored and ingested. FA v3 (§47) showed pct scoring *collapses* dimension differentiation (eigenvalue 9.41 = 94.1% shared variance). v20 training confirmed no benefit (held-out_r=0.600, flat vs v19). Integer 0-10 scale retained.

**Lesson:** Pct scoring triggers anchoring-and-adjustment heuristic — scorer locks onto global impression at finer granularity instead of differentiating better.

### Rename authority_dynamics → power_positioning [DEFERRED — WILL NOT RENAME]

**Decision (2026-02-28): DEFERRED indefinitely.** The rename is not being pursued. Rationale: **fidelity with the official psychological safety taxonomy.** The name `authority_dynamics` aligns with how power and authority are discussed in the established psychological safety literature (Edmondson 1999, French & Raven 1959, the instruments grounding this dimension). Renaming to `power_positioning` would be a custom neologism that departs from that taxonomy without clear publication benefit. The construct name stays as `authority_dynamics`.

**What the criterion validity evidence actually shows:** AD measures *status negotiation dynamics* in peer contexts — a specific mechanism within authority dynamics, not a departure from it. This is an empirical finding about *where* AD predicts (contested peer status), not evidence that the name is wrong. The name captures the broader theoretical frame; the criterion studies narrow where within that frame the predictive signal lives.

**What changes instead:**
- [x] Update the AD description in `psq-definition.md` §9 to reflect the peer-context status negotiation finding (the rubric anchors describe formal/institutional power; they should broaden to include epistemic positioning and relational status moves) — **Done 2026-03-01.** Updated `instruments.json` AD description and all 11 scoring anchors.
- [x] Add a note in the criterion validity section of `psq-definition.md` referencing the status negotiation theory (journal §24, Theory 3) — **Done 2026-03-01.**
- No DB migration, no script changes, no ONNX compatibility risk

**Prior rename rationale (retained for reference):**
- AD predicts strongly in peer contexts (Wikipedia editors, Reddit commenters, campsite negotiation) where formal authority is absent
- AD is weakest in CMV where status is *fixed*
- AD-residual correlates with epistemic markers (second-person pronouns r=+0.202, question marks r=+0.235)
- Theory 3 (status negotiation, journal §24) has the best explanatory coverage

### Fix max_length bugs in criterion validity scripts + re-eval historical models [COMPLETE]

**Status:** All items COMPLETE (2026-03-01).

1. [x] Fix CMV + CGA-Wiki T2 scripts, re-run — CMV AUC corrected: 0.5735→**0.5549** (−0.019). CGA-Wiki T2 unchanged.
2. [x] Re-fit production calibration at max_length=128 — saved.
3. [x] Re-eval v14–v22c held-out at max_length=128 — 11 models done. v22a corrected=0.695 > v23=0.684. Relative ordering shifted but v23 held as production (Δ within noise, avoids invalidating 4 criterion studies).
4. [x] Add `--max-length` CLI arg — done.
5. [x] Update EXPERIMENTS.md — corrected values recorded.

### B3 — threat_exposure uniformity: TE unlabeled-pool expansion [CLOSED — 2026-03-07]

**Status:** F4 (350 texts: 200 prosocial + 150 esconv) complete. v33 REJECTED (TE=0.742, overall=0.672).
5 consecutive rejections. 1,550 additional TE texts total (F3=500, F3b=700, F4=350) have not recovered
v23 TE=0.795. SE(r)≈0.10 noise floor at n=99 makes TE gains below ±0.10 unresolvable. B3 CLOSED.
v23 TE=0.795 accepted as production ceiling.

**Fix sequence:**
- [x] F1 (PAVA calibration) diagnostic: PAVA pooling confirmed (2026-03-06)
- [x] F2 (368 re-scored sep-llm): 368 reverted texts re-scored (2026-03-06); v29 trained — REJECTED
- [x] v30 single-task diagnostic: multi-task necessity confirmed (2026-03-07)
- [x] **F3 (unlabeled-pool expansion, 500 texts):** v31 trained — REJECTED (TE=0.773, overall=0.679)
- [x] **F3b (unlabeled-pool expansion, 700 texts):** v32 trained — REJECTED (TE=0.739, overall=0.676)
- [x] **F4 (distribution-rebalanced, 350 texts: prosocial + esconv):** v33 trained — REJECTED (TE=0.742, overall=0.672)
- [x] F1 (recalibrate n_bins=20): DEFERRED PERMANENTLY — no model beat v23; calibration deferred until architecture changes
- **CONCLUSION:** B3 closed. v23 remains production. TE ceiling accepted.

See distillation-research.md §65/§66/§67 and journal.md §37/§38 for full diagnosis.

### Re-score 368 reverted texts (1 dim per session) [COMPLETE — 2026-03-06]

**Status:** COMPLETE. All 10 dimensions scored across 10 sessions. 3,680 scores assembled and ingested.

368 texts (ucc/civil/extreme-adco) had all scores reverted due to same-session halo contamination (mean |r|=0.658). A second re-scoring attempt (all 10 dims in one session) produced even worse contamination (mean |r|=0.811). The one-dim-per-session protocol is validated as essential.

**Completed sessions:** TE, HI, AD, ED, RC, RB, TC, CC, DA, CO — all 368 texts × 10 dims. Score=5 fractions 28–39% (vs 43% composite-proxy baseline). Assembled: `data/rescore-368-assembled.jsonl`. Ingested: `migrate.py --ingest`. Separated-llm is now priority for all 368 texts in `best_scores` view.

### APA 7th edition conversion of publication-facing docs [COMPLETE]

**Status:** Complete (2026-03-06).
- [x] criterion-validity-summary.md — COMPLETE
- [x] psychometric-evaluation.md — COMPLETE
- [x] journal.md — COMPLETE
- [x] distillation-research.md — COMPLETE

### Context-aware scoring API design

**Why:** g-PSQ (single score) is near-chance across all three criterion studies (AUC 0.515–0.531). Profile shape carries the signal. But different dimensions matter for different use cases:

| Use case | Key dimensions | Rationale |
|---|---|---|
| Content moderation (derailment risk) | AD, HI, DA | AD strongest predictor in CGA-Wiki; HI/DA in multivariate model |
| Persuasion quality | DA, CC, TC | DA top predictor in CMV; CC/TC strongest multivariate coefficients |
| Negotiation outcomes | AD, DA, HI | AD/DA top in CaSiNo; HI significant |
| Workplace safety assessment | AD, ED, TC | Status negotiation + resource depletion + trust |
| Therapeutic conversation quality | RC, RB, CC, ED | Internal resources + recovery + cooling |

**Design questions resolved (2026-03-07, distillation-research.md §68):**
- [x] **Return format:** Raw 10-dim + `context_weighted_composite` + `context_weights_used`. Transparent and immediately usable; backward-compatible.
- [x] **Context specification:** User-specified `context` parameter (`moderation` | `persuasion` | `negotiation` | `workplace` | `therapeutic`). Auto-detection deferred.
- [x] **Implementation layer:** Application layer (Node.js `server.js` on Hetzner). ONNX unchanged. Weights in `context-weights.json` config.
- [x] **Schema:** v3 → v3.1 minor bump. Additive fields; no breaking change for v3 consumers.

**Implementation (2026-03-07):**
- [x] `src/context-weights.json` created — 5 contexts, evidence-grounded weights, description + evidence_source per context
- [x] `src/server.js` updated — parses `context` param, validates, computes context_weighted_composite (0-10), spreads into scores block under v3.1 schema
- [x] Weight ratios derived from criterion validity ordinal rankings (bivariate r-values from CGA-Wiki, CMV, CaSiNo, DonD studies)
- [x] Schema: server.js uses v3.1 when context present; v3 otherwise (backward-compatible)
- [x] **Deploy to Hetzner:** rsync to `/opt/psychology-agent/safety-quotient/src/` via `root@178.156.229.103` (gray-box key). `systemctl restart psq-server`. Verified live 2026-03-07: schema=v3.1, context_weighted_composite=3.76/10 for workplace test.
- [x] Add `context` param to `agent-card.json` `.well-known/` capabilities — v3.1, context_aware_scoring block, operations updated, confidence limitation downgraded to MEDIUM, composite_status fixed. Deployed to Hetzner 2026-03-07.

### Deal or No Deal criterion study [COMPLETE — v23 RERUN DONE]

**Status:** Complete (§39, original v18). v23 rerun complete (§60, 2026-02-28). AUC=0.732 (was 0.686 — strongest of 4 studies). TE top bivariate predictor (d=+0.801, was ED). T3b confirmed: AD predicts deal (+0.138) but not points (−0.070). Context-dependent primacy confirmed across 4 studies.

## Priority 2: Important

### Publication framing and paper outline

See `journal.md` and `Publication Narrative` section below. The criterion validity battery (4 studies) with context-dependent predictive primacy is the publication centerpiece.

**Target venue:** Computational linguistics / NLP+psychology intersection (e.g., EMNLP, ACL, Behavior Research Methods, Journal of Personality Assessment)

**Key findings to highlight:**
1. 10-dim PSQ trained via knowledge distillation (LLM → DistilBERT)
2. Four independent criterion validity studies with real-world outcomes (CaSiNo, CGA-Wiki, CMV, DonD)
3. Profile shape >> average across all studies (the multi-dimensional architecture is psychometrically justified)
4. Context-dependent predictive primacy (AD in contested-status, DA in fixed-status, TE+ED in behavioral negotiation) — dimensions are genuinely distinct
5. Factor structure: general factor + 5 clusters, but singletons (AD, ED) carry unique predictive signal
6. The learned authority_dynamics construct measures status negotiation in peer contexts (not just formal/institutional power) — a case study in emergent construct validity revealing the mechanism within the intended construct

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
- [ ] Prioritize AD (known discrepancy), HI (floor compression confirmed), ED (singleton, unclear construct), DA (weak factor loading)

### Criterion validity summary table [COMPLETE]

**Status:** Complete. `criterion-validity-summary.md` created and maintained as the canonical reference. Updated to v23 numbers (2026-02-28): CMV AUC=0.5735, DonD AUC=0.732. Cross-referenced from distillation-research.md §59/§60, psychometric-evaluation.md §3g.

### Turn-by-turn temporal analysis

Test prediction T2 from journal §24: does AD deteriorate before HI/TE in CGA-Wiki conversations that derail? Cross-lagged correlation analysis. Requires turn-level scoring (currently score full conversations).

### ONNX model re-export [COMPLETE]

v35 promoted to production (2026-03-08). ONNX re-exported (254 MB / 64 MB quantized INT8). v23 tagged as rollback.
