# PSQ Lab Notebook

Structured extraction from research sessions. Each entry records what was done, key findings, decisions made, and artifacts produced. Written in terse, factual form — not a narrative.

**Primary sources:** `sessions/*.jsonl` (raw transcripts, Git LFS)
**Derived views:** `journal.md` (curated narrative), `distillation-research.md` (technical reference)
**Note:** Entries prior to 2026-02-27 are reconstructed from documentation; no session transcripts exist for those dates.

---

## Current State *(overwrite each session)*

### Model: v23 (production, 2026-02-28)

| Metric | Value |
|---|---|
| Architecture | DistilBERT-base-uncased (66.7M params) |
| Held-out r (avg 10 dims) | **0.696** (+0.014 vs v22a, +0.066 vs v21) |
| Test r | 0.387 (test-split paradox — proxy labels as GT for remaining test texts) |
| Production checkpoint | `models/psq-student/best.pt` |
| ONNX | `model.onnx` 254 MB / `model_quantized.onnx` 64 MB INT8 |

### Per-dimension held-out r (v23)

| Dim | v22a | **v23** | Δ |
|---|---|---|---|
| regulatory_capacity | 0.756 | **0.782** | +0.026 |
| threat_exposure | 0.805 | **0.800** | −0.005 |
| energy_dissipation | 0.712 | **0.768** | +0.056 |
| cooling_capacity | 0.719 | **0.739** | +0.020 |
| authority_dynamics | 0.679 | **0.709** | +0.030 |
| trust_conditions | 0.679 | **0.689** | +0.010 |
| hostility_index | 0.719 | 0.691 | −0.028 |
| defensive_architecture | 0.607 | **0.608** | +0.001 |
| resilience_baseline | 0.640 | 0.621 | −0.019 |
| contractual_clarity | 0.504 | **0.549** | +0.045 |
| **Average** | **0.682** | **0.696** | **+0.014** |

### Database (data/psq.db)

| | Count |
|---|---|
| Texts | 22,186 |
| Total scores | 90,361 |
| Separated-LLM (scorer=claude-sonnet-4-6) | 34,850 |
| Held-out set | 100 texts (separate file, not in training) |
| Train / val / test split | ~17,800 / ~2,150 / ~2,200 texts |

### Labeling Batches (ingested)

| Batch | Texts | Focus | Notes |
|---|---|---|---|
| weak-dims | 200 | te/rc/co | — |
| rc | 150 | regulatory_capacity | — |
| ad | 300 | authority_dynamics | — |
| co | 200 | contractual_clarity | keyword-filtered |
| rb | 200 | resilience_baseline | — |
| cc | 200 | cooling_capacity | — |
| te | 200 | threat_exposure | TE mean=3.17 |
| broad | 300 | all dims | 150 random + 100 single-dim + 50 multi-dim |
| pct-200 | 200 | all dims | 0-100 pct scale pilot (ingested, scale RETRACTED) |
| midg | 250 | all dims | g∈[3,4.5)∪[5.5,7] middle-band enrichment |
| ccda | 200 | CO+CC | v23 batch — CO-targeted keyword-filtered |
| proxy-audit | 200 | all dims | source-diverse: goemotions/ucc/casino/berkeley |
| held-out-expand | 150 | all dims | ingested as training data (not held-out) |
| test-clean | 200 | all dims | test-split texts relabeled with LLM |

**Pending (extracted, not yet scored):**

| Batch | Texts | Priority | Rationale |
|---|---|---|---|
| ucc | 150 | **Highest** | 3% sep-llm coverage; worst MAE source (2.296) |
| civil | 100 | High | 1% sep-llm coverage; MAE=1.681 |
| extreme-adco | 118 | Medium | AD compression fix; CO pool sparse (only 118 found) |

### Criterion Validity Studies

| Study | N | Top predictor | 10-dim AUC | g-PSQ AUC | Model |
|---|---|---|---|---|---|
| CaSiNo | 1,030 | AD (r=0.127***) | — | — | v16 |
| CGA-Wiki | 4,188 | AD (r_pb=−0.105***) | 0.599 | 0.515 | v16 |
| CMV | 4,263 pairs | DA (r_pb=+0.059***) | 0.5735 | 0.5227 | **v23** |
| DonD | 12,234 | TE bivariate (d=+0.801) | **0.732** | 0.700 | **v23** |

Cross-study: profile >> average in all studies. AD positive in DonD (r_pb=+0.138, relational). T3b confirmed: AD predicts deal, not points. Context-dependent primacy: AD in contested-status, TE+ED in sustained negotiation, DA in fixed-status.

### Known Issues

| Issue | Status |
|---|---|
| DA construct validity (weak factor loading, 49% scores=5) | Open — requires expert panel ICC(2,1) |
| AD range compression (output std=1.54 vs actual 2.46) | Partially addressed — UCC/extreme-adco batches pending |
| Berkeley/UCC blind spot (MAE 2.5/2.3) | Distribution mismatch, not token length — UCC batch pending |
| CO still weakest dimension (0.549) | Improving — more data needed |
| Expert validation recruitment | Not started — protocol designed |

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

### Session `20260228-current` (this session — continued across context limit)

**v22c training completed. test-clean batch scored (all 10 dims). Curriculum REJECTED.**

**v22c trained:** `--drop-proxy-dims --curriculum --out models/psq-v22c`
- Phase 1: LLM base (5,308 records, epochs 1–3). Phase 2: +10,383 proxy (15,691 total, epochs 4–9).
- Best at epoch 6 (val_r=0.4478). Early stopping at epoch 9.
- held-out_r = **0.638** — WORSE than v22a (0.682) by -0.044. All 10 dims regressed vs v22a.
- Curriculum learning REJECTED. v22a (proxy removal only) remains the production candidate.

**2×2 ablation complete:**

| Version | Proxy removal | Curriculum | held-out_r | Δ vs v21 |
|---------|--------------|------------|------------|----------|
| v21 | No | No | 0.630 | — |
| v22a | Yes | No | **0.682** | **+0.052** |
| v22b | No | — | 0.578 | -0.052 |
| v22c | Yes | Yes | 0.638 | +0.008 |

→ Proxy removal alone is the dominant and sufficient intervention.

**test-clean batch scored:** `data/labeling-batch-test-clean.jsonl` (200 texts from test split)
- All 10 dimensions scored using separated LLM protocol across multiple sessions.
- Assembled: `data/labeling-batch-test-clean-labeled.jsonl`
- Ingested: 200 texts, 2,000 score observations. Partially resolves test-split paradox.

**Old repo deleted:** `kashfshah/safety-quotient` removed.
**Topics applied** to `safety-quotient-lab/safety-quotient`.
**Lab-notebook.md created** (this file).

**Pending:**
- Promote v22a to production slot.
- Score CC-targeted batch (labeling-batch-ccda.jsonl) to improve CO (worst dim: 0.504).
- Score remaining batches: proxy-audit (200 texts), held-out-expand (150 texts).
- Begin expert validation recruitment.

---

## v-Series Summary Table

| Version | Key change | test_r | held-out_r | Notes |
|---------|-----------|--------|------------|-------|
| v1–v8   | Architecture sweep | — | — | DeBERTa→DistilBERT |
| v14     | Separated scoring, concentration cap | ~0.42 | ~0.58 | Baseline |
| v21     | Expanded LLM data (8 batches) | 0.504 | 0.630 | Production (superseded) |
| v22a    | `--drop-proxy-dims` (TE/TC/CC/AD) | 0.457 | 0.682 | New best at time |
| v22b    | midg data only (no proxy removal) | — | 0.578 | Worse than v21 |
| v22c    | `--drop-proxy-dims + --curriculum` | 0.431 | 0.638 | Curriculum REJECTED |
| **v23** | +550 texts (ccda/proxy-audit/held-out-expand) | — | **0.696** | **Current production** |
| v24 | 256-token context (batch 16, grad_accum 2) | — | pending | Training (task bxsm4j1ou) |

---

## Open Questions

1. ~~Does curriculum learning add anything beyond proxy removal alone?~~ **ANSWERED:** v22c 0.638 < v22a 0.682. Curriculum REJECTED.
2. ~~What is the clean test_r once `labeling-batch-test-clean.jsonl` is scored and ingested?~~ **ANSWERED:** v22c test_r=0.431 (proxy-clean test split; not comparable to prior test_r).
3. ~~Does more CO-targeted data (ccda batch) improve CO from 0.504?~~ **ANSWERED:** v23 CO=0.549 (+0.045). YES — CO-targeted ccda batch improved the weakest dimension. Still weakest overall; more data will help further.
4. ~~Is CC penalized by proxy removal?~~ **ANSWERED:** v23 CC=0.739 (+0.020 vs v22a). NO — proxy removal is net-positive for CC. The v22a regression was a data quantity effect, not a proxy removal artifact.
5. Human expert validation: DA construct validity still unresolved by LLM data alone. T3b provides computational evidence (AD predicts deal not points), but ICC(2,1) from expert panel required for final resolution.
6. Does increasing context from 128→256 tokens improve performance on long-text sources? Error analysis identifies berkeley/UCC blind spots as distribution mismatch (not length), but criterion datasets (DonD multi-turn) may still benefit.
7. Can the AD range compression (output std=1.54 vs actual std=2.46) be corrected by the UCC/extreme-adco labeling batches? AD is the most compressed dimension (ratio=0.63) and has 48.4% of sep-llm scores at exactly 5.0.

---

### Session `20260228-1331` (this session)

**v22a promotion. ONNX export. Three labeling batches. Proxy audit. v23 launched.**

**v22a promoted to production slot:**
- Copied `models/psq-v22a/{best.pt,config.json,held_out_results.json}` → `models/psq-student/`
- Re-exported ONNX: `model.onnx`=254.4 MB (full precision, verification diff=0.000004), `model_quantized.onnx`=64.0 MB (INT8, 4.0× smaller)
- Export note: `export_onnx.py` reads config from `models/psq-student/config.json` regardless of `--checkpoint`; must copy config before running.

**Three labeling batches scored and ingested (all 10 dims, separated protocol):**

| Batch | Texts | Sources | Notable distributions |
|---|---|---|---|
| ccda | 200 | prosocial 104, berkeley 38, dreaddit 33, empath 16, esconv 9 | CO mean=5.50, range [1,9] — good CO variance |
| proxy-audit | 200 | goemotions 75, ucc 42, casino 42, berkeley 41 | TE mean=5.91, AD range [3,6] (compressed) |
| held-out-expand | 150 | empath 47, prosocial 45, berkeley 43, esconv 4, dreaddit 11 | TE mean=5.51, full range [1,9] |

**Held-out-expand ingestion decision:** Originally labeled "expand held-out set" but ingested as training data (migrate.py --ingest). No overlap with `data/held-out-test.jsonl` confirmed. Distribution: 118 train / 19 val / 13 test by hash split. Useful as training data; held-out set remains 100 texts.

**Proxy audit findings:**
Source-specific proxy-LLM correlations for goemotions/ucc/casino/berkeley texts:
- DROPPED dims: TE=0.223, AD=-0.129, TC=-0.200, CC=-0.293, ED≈0.106 — all near-zero or negative
- "Retained" dims: HI=-0.126, RC=0.004, RB=-0.203 — also near-zero or negative within these sources
- Key insight: corpus-wide positive r values (RB=0.539, HI=0.488) come from OTHER sources (dreaddit, empathetic_dialogues), not from goemotions/ucc/casino/berkeley. These four sources have near-zero proxy utility for all dimensions.

→ Proxy-drop decision confirmed. The ccda + proxy-audit + held-out-expand batches replace proxy signal with verified LLM signal from the problematic sources.

**v23 training launched:** `python scripts/distill.py --db data/psq.db --drop-proxy-dims --out models/psq-v23`
- +5,500 new separated-llm scores (550 texts × 10 dims) vs v22a
- DB state: 22,186 texts, 90,361 scores (34,850 separated-llm)
- **Results:** held-out_r=**0.696** (new best, +0.014 vs v22a). 7/10 dims improved. ED +0.056, CO +0.045, AD +0.030. v23 promoted to production.

▶ EXPERIMENTS.md (v23 row added), DATA-PROVENANCE.md (Tier 5 table updated)

---

### Session `20260228-1423` (novelty hunt + criterion reruns + error analysis)

**Error analysis (v23), criterion reruns (CMV + DonD), three new labeling batches extracted.**

**Error analysis results** (`scripts/error_analysis.py --checkpoint models/psq-v23/best.pt --split all`):

| Source | MAE | Bias | Notes |
|---|---|---|---|
| berkeley | 2.549 | −2.259 | Worst. Short hate-speech — model predicts safe when text is threatening. |
| ucc | 2.296 | −1.463 | Short hostile political comments. Systematic under-prediction. |
| civil_comments | 1.681 | −0.968 | Still problematic after TE proxy removal. |
| dreaddit | 1.545 | +0.163 | Slight over-prediction. |
| synthetic | 1.088 | −0.037 | Well-calibrated. |
| esconv / claude_code | ~0.83 | ~0.10 | Near-perfect. |
| politeness_stack-exchange | 0.615 | +0.314 | Best source. |

Root cause: **distribution mismatch**, not token length. Berkeley/UCC are short cryptic texts; model trained on emotionally explicit longer texts (dreaddit, esconv). AD is most compressed (output std=1.54 vs actual 2.46).

**CMV v23 rerun:** AUC=0.5735 (was 0.590 v16). DA still top (r_pb=+0.059***). TE p=0.914 — proxy artifact confirmed eliminated. CO p=0.155 (NS). 7/10 dims significant.

**DonD v23 rerun:** AUC=0.732 (was 0.686 v18) — new project best criterion validity result. 5-fold CV: 0.723±0.010. TE displaces ED as top bivariate predictor (d=+0.801) — v18's ED dominance was a TE measurement artifact. After length control: TE partial r=0.203 ≈ ED partial r=0.209. AD bivariate reversed to +0.138 (was −0.026). Q4/Q1 deal gap: 88.7pp (was 15.9pp). T3b CONFIRMED: AD predicts deal (+0.138) but not points (−0.070***).

**Three labeling batches extracted** (not yet scored):
- `data/labeling-batch-ucc.jsonl` — 150 texts from UCC (3% sep-llm coverage; highest priority blind spot)
- `data/labeling-batch-civil.jsonl` — 100 texts from civil_comments
- `data/labeling-batch-extreme-adco.jsonl` — 118 texts keyword-filtered for extreme AD/CO (CO keywords sparse in pool; only 19 extreme CO texts found)

Dimension files extracted to `/tmp/psq_separated/` for all three batches. Ready to score.

▶ distillation-research.md §59/§60, journal.md §36, psychometric-evaluation.md, criterion-validity-summary.md, novelty-hunt-20260228-1423.md

---

### Session `20260228-1530` (v24 launched: 256-token context experiment)

**v24 training started.** Smoke test passed (1 epoch, 649s, no OOM). Full run in background (task bxsm4j1ou).

- Config: `--max-length 256 --batch-size 16 --grad-accum 2 --drop-proxy-dims`
- Effective batch = 16 × 2 = 32 (same as v23). ~11 min/epoch on GTX 1060 6GB.
- Hypothesis: longer context improves held-out_r on texts where 128-token truncation loses signal (DonD multi-turn, long reddit posts). Error analysis showed berkeley/UCC blind spots are distribution mismatch not length — so main gains expected from criterion datasets, not those sources.
- Data: same as v23 (no new labels ingested). Pure architectural ablation.
- Metrics pending.

▶ EXPERIMENTS.md (v24 row added)
