# PSQ Lab Notebook

Structured extraction from research sessions. Each entry records what was done, key findings, decisions made, and artifacts produced. Written in terse, factual form — not a narrative.

**Primary sources:** `sessions/*.jsonl` (raw transcripts, Git LFS)
**Derived views:** `journal.md` (curated narrative), `distillation-research.md` (technical reference)
**Note:** Entries prior to 2026-02-27 are reconstructed from documentation; no session transcripts exist for those dates.

---

## Notation

- `→` Decision or action taken as a result of finding
- `▶` Cross-reference to journal.md or distillation-research.md
- `[reconstructed]` Entry derived from docs, no raw transcript

---

## 2022-05-xx [reconstructed]

**Conceptual inception.** 71 operational PJE terms enumerated in email under "Psychology - Juris - Engineering" framework. Terms include *psychoemotional safety quotient*, *psychoemotional cooling*, *psychoemotional energy dissipation*, *psychoemotional contract law*.

No measurement procedures, scoring rubrics, or validation criteria defined. Pre-paradigmatic: vocabulary without methods.

▶ journal.md §1

---

## 2026-02-25 [reconstructed]

**Construct formalization.** External critique: PJE is "a manifesto, not a methodology" — lacks novel constructs, methods, instruments.

**Response (same session):**
- Defined 10-dimension PSQ construct
- Wrote `psq-definition.md` with scoring rubrics (0–10 per dim)
- Mapped ~100 validated instruments from clinical/org/social psych to 10 dims
- Defined measurement procedure: multi-pass LLM-as-judge → distillation

**Dimensions defined:** TE (Threat Exposure), RC (Regulatory Capacity), RB (Resilience Baseline), TC (Trust Conditions), HI (Hostility Index), CC (Cooling Capacity), AD (Authority Dynamics), DA (Defensive Architecture), CO (Contractual Obligations), ED (Energy Dissipation)

▶ journal.md §1–2

---

## 2026-02-26 [reconstructed]

**Early training infrastructure.** Built composite ground truth pipeline (`build_composite_ground_truth.py`), SQLite schema (`data/schema.sql`), and initial distillation script.

**Proxy data ingested:** 30,803 rows from 11 source datasets (dataset_mappings.json). Proxy = composite of keyword/sentiment/emotion classifiers.

**v1–v8 iterations:** Architecture sweeps (DeBERTa vs DistilBERT). DeBERTa: slower, higher capacity. DistilBERT: 6× faster, comparable validation performance at this data size.
→ Settled on DistilBERT as production architecture.

**Data quality issue identified:** Civil Comments dataset adversarially mis-labeled by proxy (CC dimension). CC proxy correlates negatively with LLM scores.
→ Downstream: CC exception in proxy-drop logic.

▶ journal.md §7–8, §12

---

## 2026-02-27

### Session `20260227-1447` (11 KB)

**v14 training and separated-scoring infrastructure.** Implemented `label_separated.py` — one dimension per session to eliminate halo effect. Deleted `batch_label_llm.js` and `relabel_separated.js` (joint scoring, halo problem).

**Labeling batches scored (all 10 dims, separated):**
- `labeling-batch-weak-dims.jsonl` (200 texts)
- `labeling-batch-rc.jsonl` (150 texts)
- `labeling-batch-ad.jsonl` (300 texts)
- `labeling-batch-co.jsonl` (200 texts, CO-focused)
- `labeling-batch-rb.jsonl` (200 texts)
- `labeling-batch-cc.jsonl` (200 texts)
- `labeling-batch-te.jsonl` (200 texts, TE-focused, mean=3.17)
- `labeling-batch-broad.jsonl` (300 texts, broad-spectrum)

**Score concentration cap implemented:** `_cap_score_concentration()` in distill.py. Dims where >30% of scores are the same value → weight 1.5 for minority scores.

→ v14 baseline established. Separated-llm scores now in DB with `scorer=claude-sonnet-4-6, provider=anthropic, interface=claude-code`.

▶ journal.md §16–17, distillation-research.md §§1–25

---

### Session `20260227-1451` (236 bytes)

Trivial continuation or test. No substantive content.

---

### Session `20260227-1740` (8.7 MB)

**Factor analysis v1 → v2. Criterion validity battery.**

**Factor analysis v2** (N=1,970 separated-llm-only texts):
- EV1 = 6.727 (67.3% variance). Up from 4.844 (48.4%) in v1 mixed data.
- KMO = 0.902 ("Superb"). Up from 0.819.
- Parallel analysis: 1 factor only (was 2 in v1).
- Mean inter-dim |r| = 0.632.
- g-factor loadings: TC=0.930, DA=0.914, CC=0.864, RC=0.854.
→ PSQ has a genuine g-factor. g IS the PSQ at broadest level.

**Pct vs integer scoring experiment:**
- pct: within-text SD=0.448, 35 unique values, 8/10 dims <5% unique variance.
- int: within-text SD=0.717, 11 bins, genuine differentiation.
- g-factor EV: int=6.727 (67.3%), pct=9.410 (94.1%) — pct collapses dimensions.
→ **Reverted to integer scoring.** Pct anchoring-and-adjustment destroys differentiation.

**Criterion validity — CaSiNo** (1,030 negotiation dialogues):
- 9/10 dims predict satisfaction (r≈0.08–0.13***), 9/10 predict likeness.
- Incremental R² = +0.016 (sat), +0.023 (like) beyond sentiment + text length.
- DA top predictor after controls.

**Criterion validity — CGA-Wiki** (4,188 Wikipedia talk-page convos, derailment):
- AUC=0.599 (10-dim), g-PSQ near-chance (0.515) — profile shape predicts, average doesn't.
- AD strongest (r_pb=-0.105***). Temporal gradient: AUC 0.519→0.570→0.599.

**Criterion validity — CMV** (4,263 matched pairs, persuasion):
- 10-dim AUC=0.590, g-PSQ=0.531. Profile >> average (gap 0.059).
- DA top predictor (r_pb=+0.085).

**Criterion validity — DonD** (12,234 negotiation dialogues, deal/no-deal):
- AUC=0.686, g-PSQ=0.622 — strongest yet.
- ED top predictor (d=+0.614, largest effect across 4 studies).
- AD suppressor replicated (coef=-0.534).
- High-PSQ Q4 deal rate 84.4% vs Low-PSQ Q1 68.5% (15.9pp).

→ PSQ has multi-dataset criterion validity. Profile >> average consistently. ED is a valid genuine singleton.

▶ journal.md §18–28, distillation-research.md §§26–42, psychometric-evaluation.md §3g

---

### Session `20260227-1901` (1.9 MB)

**Scoring experiment protocol design. Halo mitigation research.**

Designed 4-phase scoring experiment (scoring-research-plan.md, scoring-experiments.md):
- Phase 0: Test-retest reliability (Δ_noise baseline)
- Exp 1: Halo-awareness instruction
- Exp 2: Structurally dissimilar rubrics
- Exp 3: Scale format (0–10 vs 0–4)

Selected 80-text experiment set (`select_experiment_texts.py`). Ran Phase 0 and Exp 1.

**Phase 0 (test-retest):** Δ_noise=0.011, 6/10 dims r≥0.80, AD unstable (r=0.156). → GO.

**Exp 1 (halo-awareness instruction):** Initially adopted pending full analysis.

▶ distillation-research.md §§43–50, scoring-experiments.md

---

### Session `20260227-1948` (4.5 MB)

**Scoring experiments concluded. Proxy audit. v21 → v22a.**

**Exp 2 (dissimilar rubrics):** REJECTED — construct redefinition, not halo reduction.
**Exp 3 (scale format):** RETAINED 0–10. Scale has zero effect on halo.

**G-factor structural analysis (§51):**
- Extreme texts (g<3 or g>7): EV1=82.8%, uniform loadings — pure valence.
- Middle texts (g 4–6): EV1=38.7%, structured loadings — genuine differentiation.
- Halo-aware instruction's individual |Δ|=0.217 < test-retest noise floor 0.54.
- CC bias (+0.33 mean shift) and CO decoupling account for ~1/3 of SD improvement.
→ g-factor is real co-variation (range/extremity effect), NOT scorer halo.
→ **Exp 1 REVERSED.** No changes to scoring prompt. Current prompt is correct.

**Proxy data audit:**
- Proxy: 30,803 rows, 17.8% effective weight. 1 sep-llm row = 5.8× 1 proxy row.
- Proxy-LLM agreement: RB=0.539, RC=0.497, HI=0.488, DA=0.448 (usable).
- AD=0.155, CC=0.102, TC=0.071 (harmful). TE=-0.260 (adversarial). ED=constant (r=NaN).
- 43% proxy rows have confidence <0.3; 7,705 have only 1 dim scored.
→ Drop proxy for TE, TC, CC, AD (harmful agreement). ED separately (zero information).

**Unlabeled pool analysis:** 50.4% informative band (g ∈ [3,4.5)∪[5.5,7]). ~7,700 texts available.
Best sources: dreaddit (62% informative), berkeley (53.5%).
→ Create middle-g labeling batch.

**`--drop-proxy-dims` flag added** to distill.py. Default set: TE, TC, CC, AD.

**`labeling-batch-midg.jsonl`** created: 250 texts, model-selected from pool for informative band.

**v22a trained:** `--drop-proxy-dims` only (TE, TC, CC, AD removed from proxy).
- held-out_r = **0.682** (new best, +0.052 vs v21 0.630).
- TE: 0.492→0.805 (+0.313, largest single-dim improvement ever). 9/10 dims improved.
- CC regression: -0.051 (CC proxy removal costs something; CC exception noted).
- test_r = 0.457 (LOWER than v21 0.504 — test-split paradox).

**Test-split paradox confirmed:** 72.8% of test texts have ONLY proxy labels as ground truth. test_r is unreliable. held-out_r is the valid metric.

▶ journal.md §31–33, distillation-research.md §§51–54

---

## 2026-02-28

### Session `20260228-1105` (68 MB)
`sessions/20260228-1105_9e5127a1-9117-422d-803e-d418971c2f7b.jsonl`

**v22b. Range-dependent g-factor. Curriculum learning. GitHub.**

**v22b trained:** midg data only (no proxy removal).
- held-out_r = 0.578 (WORSE than v21 by -0.052).
- All 10 dims worse than v22a.
→ Data quality > data quantity, conclusively. ±0.052 symmetry with v22a.

**Range-dependent g-factor discovery:**
- Middle-g texts (4≤g≤6, N=1,602): EV1=3.90 (39.0% variance).
- Overall: EV1=7.225 (72.3%).
- The g-factor collapses precisely where dimensions should differentiate.
→ This is good news for the construct: g-dominance is a range/extremity artifact, not fundamental.

**Updated factor analysis** (N=2,319 sep-llm texts):
- EV1=7.225 (72.3%), KMO still excellent, Kaiser retains 1 factor.

**ED added to `--drop-proxy-dims`** default: ED proxy is constant 5.0, r=NaN (zero information).

**Curriculum learning implemented** in distill.py:
- Phase 1: LLM-only data (separated-llm, joint-llm, synthetic).
- Phase 2 (after split epoch): adds proxy data with standard weighting.
- CLI: `--curriculum`, `--curriculum-split` (default 3).
- Smoke test (CPU, 2 epochs, split=1): Phase 1 val_r=0.329, Phase 2 val_r=0.441 (+0.112).

**New labeling batches created** (not yet scored):
- `data/labeling-batch-test-clean.jsonl` — 200 test-split proxy-only texts (for clean test metric)
- `data/labeling-batch-proxy-audit.jsonl` — 200 texts for TC/CC/AD/HI/CO proxy-vs-LLM audit
- `data/labeling-batch-held-out-expand.jsonl` — 150 unlabeled-pool texts for held-out expansion

**Sessions preservation architecture:**
- Raw transcripts copied to `sessions/*.jsonl`, tracked via Git LFS.
- `sessions/README.md` created (index and rationale).
- Document hierarchy: sessions (primary) → lab-notebook.md (structured extraction) → journal.md (curated narrative) → distillation-research.md (technical reference).

**GitHub remote established:**
- Org: `safety-quotient-lab`. Repo: `safety-quotient-lab/safety-quotient`.
- Public, CC BY-NC-SA 4.0. SSH key: `~/.ssh/github-sqlab` (ed25519, passwordless).
- All commits + 86MB LFS objects pushed successfully.
- Topics: psychometrics, psychological-safety, nlp, distilbert, content-analysis, text-classification, pytorch.

▶ journal.md §33, distillation-research.md §55

---

### Session `20260228-current` (this session)

**v22c training. GitHub cleanup.**

**v22c launched:** `--drop-proxy-dims --curriculum --out models/psq-v22c`
- Phase 1: LLM base (5,308 records, epochs 1–3).
- Phase 2: +13,787 proxy records (19,095 total, epochs 4–10).
- Hypothesis: proxy removal + curriculum learning > either alone.

**Old repo deleted:** `kashfshah/safety-quotient` removed.
**Topics applied** to `safety-quotient-lab/safety-quotient`.
**Lab-notebook.md created** (this file).

**Pending:**
- v22c held-out results (ETA ~52 min from launch)
- Score 4 labeling batches: test-clean, proxy-audit, held-out-expand, ccda
- Promote v22c if held-out_r > 0.682 (v22a benchmark)
- lab-notebook.md retrospective fill-in from older session transcripts

---

## v-Series Summary Table

| Version | Key change | test_r | held-out_r | Notes |
|---------|-----------|--------|------------|-------|
| v1–v8   | Architecture sweep | — | — | DeBERTa→DistilBERT |
| v14     | Separated scoring, concentration cap | ~0.42 | ~0.58 | Baseline |
| v21     | Expanded LLM data (8 batches) | 0.504 | 0.630 | **Production** |
| v22a    | `--drop-proxy-dims` (TE/TC/CC/AD) | 0.457 | **0.682** | New best |
| v22b    | midg data only | — | 0.578 | Worse than v21 |
| v22c    | `--drop-proxy-dims + --curriculum` | — | pending | Training |

---

## Open Questions

1. Does curriculum learning add anything beyond proxy removal alone? (v22c vs v22a)
2. What is the clean test_r once `labeling-batch-test-clean.jsonl` is scored and ingested?
3. Does expanding the held-out set change the v22a/v22c ranking?
4. Is CC penalized by proxy removal? (CC regression in v22a was -0.051; need more CC LLM data)
5. Human expert validation: DA construct validity still unresolved by LLM data alone.
