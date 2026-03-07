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
| Held-out r (avg 10 dims) | **0.684** (corrected; was 0.696 with max_length=256 eval bug) |
| Test r | 0.387 (test-split paradox — proxy labels as GT for remaining test texts) |
| Production checkpoint | `models/psq-student/best.pt` |
| ONNX | `model.onnx` 254 MB / `model_quantized.onnx` 64 MB INT8 |

### Per-dimension held-out r (v23 production; v29 rejected; v31 rejected)

| Dim | **v23** | v29 | v31 | Δ (v31−v23) |
|---|---|---|---|---|
| threat_exposure | **0.795** | 0.734 | 0.773 | −0.022 |
| regulatory_capacity | **0.768** | 0.767 | 0.765 | ≈0 |
| energy_dissipation | **0.760** | 0.747 | 0.773 | +0.013 |
| cooling_capacity | **0.736** | 0.754 | 0.747 | +0.011 |
| authority_dynamics | **0.713** | 0.713 | 0.671 | −0.042 |
| trust_conditions | **0.681** | 0.693 | 0.711 | +0.030 |
| hostility_index | **0.669** | 0.652 | 0.716 | +0.047 |
| resilience_baseline | **0.597** | 0.575 | 0.585 | −0.012 |
| defensive_architecture | **0.588** | 0.531 | 0.501 | −0.087 |
| contractual_clarity | **0.538** | 0.517 | 0.553 | +0.015 |
| **Average** | **0.684** | 0.668 | 0.679 | **−0.005** |

v30-te-ceiling (single-task TE only): held-out TE=0.762. Confirms multi-task adds +0.033 bonus (0.795−0.762). Multi-task scaffolding is necessary — single-task TE never viable for production.

v31: expanded TE corpus (+500 texts from unlabeled pool). TE improved +0.039 vs v29 but still −0.022 vs v23. Overall 0.679 < 0.684. REJECTED. v23 remains production. DA trend: 0.588→0.531→0.501 (three versions, but differences within noise at n=88).

### Database (data/psq.db)

| | Count |
|---|---|
| Texts | 22,304 |
| Total scores | 94,041 |
| Separated-LLM (scorer=claude-sonnet-4-6) | 40,451 |
| Held-out set | 100 texts (separate file, not in training) |
| Train / val / test split | 17,800 / 2,170 / 2,251 texts |

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
| ucc | 150 | all dims | **REVERTED then RE-SCORED** — separated-llm, 1 dim/session protocol ✓ |
| civil | 100 | all dims | **REVERTED then RE-SCORED** — separated-llm, 1 dim/session protocol ✓ |
| extreme-adco | 118 | AD/CO | **REVERTED then RE-SCORED** — separated-llm, 1 dim/session protocol ✓ |
| rescore-368 | 368 | all 10 dims | 10-session separated-llm rescore; 3,680 scores ingested 2026-03-06 |
| te-expansion-500 | 500 | threat_exposure | unlabeled-pool: 150 dreaddit + 150 emp.dial. + 100 prosocial + 100 berkeley; TE sep-llm; ingested 2026-03-07; drove v31 |

### Criterion Validity Studies

| Study | N | Top predictor | 10-dim AUC | g-PSQ AUC | Model |
|---|---|---|---|---|---|
| CaSiNo | 1,030 | AD (r=0.127***) | — | — | v16 |
| CGA-Wiki | 4,188 | AD (r_pb=−0.105***) | 0.599 | 0.515 | v16 |
| CMV | 4,263 pairs | DA (r_pb=+0.059***) | 0.5549 | 0.5227 | **v23** (corrected max_length) |
| DonD | 12,234 | TE bivariate (d=+0.801) | **0.732** | 0.700 | **v23** |

Cross-study: profile >> average in all studies. AD positive in DonD (r_pb=+0.138, relational). T3b confirmed: AD predicts deal, not points. Context-dependent primacy: AD in contested-status, TE+ED in sustained negotiation, DA in fixed-status.

### Known Issues

| Issue | Status |
|---|---|
| DA construct validity (weak factor loading, 49% scores=5) | Open — requires expert panel ICC(2,1) |
| AD range compression (output std=1.54 vs actual 2.46) | v29 rejected (DA regressed −0.057). Root cause unresolved. More DA data needed. |
| Berkeley/UCC blind spot (MAE 2.5/2.3) | v29 rejected — 368 rescore not sufficient. More data needed. |
| CO still weakest dimension (0.538 corrected) | Improving — more data needed |
| B1 — Confidence head dead (constant output regardless of input) | **FIXED (2026-03-07).** Model confidence head collapsed to per-dim constants. Production now surfaces static held-out Pearson r as confidence (confidence_type: held_out_r). Deployed on Hetzner via student.js + calibration.json update. |
| B2 — HI calibration dead zone (raw 5.85-7.65 → 6.69) | **FIXED (2026-03-06).** Recalibrated with isotonic-v2 (n_bins=45). HI now differentiates (e.g., 6.55/6.99/7.33/7.39 for 4 test texts). MAE improved −3.9%. |
| max_length eval bug (256 vs 128) | **Fixed** — all historical held-out_r inflated ~0.012. v23 corrected: 0.696→0.684 |
| Same-session halo replication | **Confirmed** — mean |r|=0.811. Even "careful" sequential scoring (|r|=0.777) exceeds threshold. 10 sessions required. |
| 25 residual pre-revert scores | Open — 7 TE (civil), 18 DA (ucc), half-point values from 2026-02-27 |
| Expert validation recruitment | Not started — protocol designed |
| B3 — TE uniformity (calibration dead zone + label degradation) | **F3 COMPLETE, v31 REJECTED (2026-03-07).** 500 TE texts scored from unlabeled pool and ingested. v31 trained: TE=0.773 (+0.039 vs v29), overall=0.679 (< v23 0.684). TE expansion works but 500 texts insufficient — need 500–1,000 more. v23 remains production. |
| v28 — not promoted | v28 held-out r=0.678 < v23 0.684. TE regression (0.762 vs 0.800) and CO regression (0.488 vs 0.538) offset gains elsewhere. v23 remains production. |

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

### Session `20260306-1800` (interagent protocol wiring + /sync skill)

**Interagent/v1 protocol integration complete.** PSQ sub-agent wired into the psychology-agent mesh:
- `.well-known/agent-card.json` created — capability declaration (A2A v0.3.0 structure)
- `~/.claude/proposals/to-psq/` inbox created
- `.claude/hooks/session-start-inbox.sh` — SessionStart hook checks inbox, surfaces pending proposals
- `.claude/settings.local.json` — hooks key added (local, not committed)
- `CLAUDE.md` — Interagent Protocol section added (authority hierarchy, schemas, inbox, response format, namespace)

**`/sync` skill created.** Adapted from observatory-agent's implementation. 6-phase protocol: inbound scan → triage → process PRs → write ACK → deliver outbound PR → commit local transport. Peer registry: psychology-agent, observatory-agent, unratified-agent.

**First /sync executed.** No inbound activity for PSQ. Mesh active between psychology-agent ↔ observatory-agent ↔ unratified-agent (icescr-framing session, 7 turns). PSQ has not yet participated in any PR-based exchange.

**Sync infrastructure audit proposal sent** to psychology-agent (`~/.claude/proposals/to-psychology/sync-infrastructure-audit-2026-03-06.json`). 6 findings (4 HIGH, 2 MEDIUM): no /sync skill on orchestrator, no inbox polling at session start, no agent-card.json, no T13 inbox trigger, format divergence, inverted capability hierarchy. Action gate: BLOCKED — PSQ outbound dead-letters without psychology-agent /sync + inbox hook.

**APA conversion marked COMPLETE** in TODO.md (journal.md + distillation-research.md finished prior session).

**Commits:** `8c0e31a` Wire PSQ into interagent/v1 protocol; `5fd204a` Add /sync skill.

### Session `20260306-2100` (Hetzner deployment + command-response delivery)

**Model-rsync command-request executed.** Psychology-agent requested model file transfer to Hetzner production server (178.156.229.103) via `transport/sessions/psychology-interface/model-rsync-request-001.json`. First use of command-request/v1 protocol.

- rsync: 41 files, 531 MB transferred at ~1.45 MB/s
- SHA256 verified: model.onnx `bc5d7f29...d833c52a`, model_quantized.onnx `28c9a950...2f128239`
- systemd: `User=ubuntu` → `User=psq`, service installed + enabled + started
- onnxruntime-node conflict resolved: removed nested `@1.21.0` (bundled by `@huggingface/transformers@3.8.1`), top-level `@1.24.2` serves both. Fix fragile — won't survive `npm install`.
- Health check: `{status: ok, model: psq-student, calibration_version: isotonic-v1-2026-03-06, ready: true}`
- Scoring test: composite 36/100, 5/10 dims above threshold, 84ms inference

**Command-response delivered** via PR #15 to psychology-agent repo. `model-rsync-response-001.json` with full state_attestation. Action gate: OPEN — psychology-agent can now set `PSQ_ENDPOINT_URL` via wrangler secret.

**Communication gap diagnosed and fixed.** `/sync` Phase 1 only checked GitHub PRs — missed transport messages committed directly to main on the parent repo. Added `git fetch origin main` + log check to Phase 1. Commit `cf13076`.

**Pending:** Re-score 368 texts (1 dim/session × 10). Durable onnxruntime-node fix (npm override or postinstall). Psychology-agent to set PSQ_ENDPOINT_URL. Endpoint security: HTTP only, port 3000 on 0.0.0.0.

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
| **v23** | +550 texts (ccda/proxy-audit/held-out-expand) | 0.387 | **0.684*** | **Current production** |
| v24 | 256-token context (batch 16, grad_accum 2) | 0.391 | 0.670* | 128 tokens superior; NOT promoted |
| v25 | 512-token context (batch 8, grad_accum 4) | 0.390 | 0.692* | Near-equal but 5× slower; NOT promoted |
| v26 | 128-token, LR=1e-5 (slow training test) | — | — | Training failed at startup |
| v27 | +368 texts (ucc/civil/extreme-adco) | 0.390 | 0.655* | **Regressed** — not promoted |
| v28 | Same data, no --drop-proxy-dims | 0.412 | 0.678 | B3 diagnostic — TE=0.762 (−0.033 vs v23) |
| v29 | rescore-368 + --drop-proxy-dims | 0.383 | 0.668 | B3 F2 — TE=0.734, REJECTED |
| v30 | Single-task TE only | — | TE=0.762 | Multi-task bonus +0.033 confirmed |
| v31 | +500 TE expansion texts | 0.384 | 0.679 | B3 F3 — TE=0.773, REJECTED. v23 remains. |

*held-out_r corrected with max_length=128 eval (was inflated ~0.012 with 256-token eval bug).

---

## Open Questions

1. ~~Does curriculum learning add anything beyond proxy removal alone?~~ **ANSWERED:** v22c 0.638 < v22a 0.682. Curriculum REJECTED.
2. ~~What is the clean test_r once `labeling-batch-test-clean.jsonl` is scored and ingested?~~ **ANSWERED:** v22c test_r=0.431 (proxy-clean test split; not comparable to prior test_r).
3. ~~Does more CO-targeted data (ccda batch) improve CO from 0.504?~~ **ANSWERED:** v23 CO=0.549 (+0.045). YES — CO-targeted ccda batch improved the weakest dimension. Still weakest overall; more data will help further.
4. ~~Is CC penalized by proxy removal?~~ **ANSWERED:** v23 CC=0.739 (+0.020 vs v22a). NO — proxy removal is net-positive for CC. The v22a regression was a data quantity effect, not a proxy removal artifact.
5. Human expert validation: DA construct validity still unresolved by LLM data alone. T3b provides computational evidence (AD predicts deal not points), but ICC(2,1) from expert panel required for final resolution.
6. ~~Does increasing context from 128→256 tokens improve performance on long-text sources?~~ **ANSWERED:** v24 (256 tok) held-out_r=0.670 (−0.026 vs v23). 128-token context is superior: 8/10 dims regressed, only CC (+0.022) and AD (+0.014) improved. Confirms that relevant safety-relevant signal is concentrated in early text windows; DistilBERT's 6 layers cannot leverage longer-range dependencies. 128 tokens confirmed as optimal for this hardware/model combination.
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

---

### Session `20260228-1625` (v24 results, context length sweep, T2 temporal analysis)

**v24 results confirmed.** 256-token context regresses −0.026 vs v23 (held-out_r=0.670). 128 tokens superior. v24 NOT promoted.

**Unattended training queue launched.** `scripts/train_queue_v25_v26.sh` (nohup PID 2469760):
- v25 (512 tok, batch=8, grad-accum=4): training on GPU now
- v26 (128 tok, LR=1e-5): queued after v25 completes
- Eval logs: `/tmp/psq_v25_eval.txt`, `/tmp/psq_v26_eval.txt`

**CGA-Wiki T2 temporal analysis launched.** `criterion_cgawiki_temporal.py` scoring 25,351 utterances on CPU. Tests T2: r(AD_t, HI_{t+1}) vs r(HI_t, AD_{t+1}) in derailing conversations. ~70% complete at time of writing.

**Documentation updates:**
- EXPERIMENTS.md: v24 complete row, v25/v26 pending rows; v23 and v24 per-dim held-out tables
- distillation-research.md: §61 added (context length + T2 analysis), status line updated, ToC entry added
- lab-notebook.md: v24/v25/v26 rows in training table; Open Question #6 answered
- MEMORY.md: v24 result, v25/v26 status, T2 analysis note added

▶ EXPERIMENTS.md (v24/v23/v24 held-out sections), distillation-research.md §61

---

### Session `20260301-0430` (max_length bug fix, 3 labeling batches, v27 regression, confidence calibration)

**max_length eval bug discovered and fixed.** `eval_held_out.py`, `calibrate.py`, and `distill.py PSQDataset` all hardcoded `max_length=256` despite training using 128. All 3 scripts fixed to 128. v23 corrected held-out_r: 0.696→**0.684** (−0.012). All historical held-out_r numbers inflated by ~0.012.

**Confidence calibration rewritten.** `validate_confidence_calibration.py` was reading old JSONL files with proxy GT. Rewrote to query psq.db with separated-LLM GT only. Result: 1 correct, 1 flat, **8 inverted** directions (higher conf → higher error). Verdict: **POOR** — confidence head is anti-calibrated.

**3 labeling batches scored and ingested:**
- UCC (150 texts × 10 dims = 1,500 scores) — UCC source enrichment
- civil_comments (100 × 10 = 1,000 scores) — civil_comments enrichment
- extreme-adco (118 × 10 = 1,180 scores) — AD compression + CO extremes
- DB after: 22,304 texts, 94,041 scores, 38,530 sep-LLM

**v27 trained and evaluated.** held-out_r=**0.655** (−0.029 vs v23 corrected). All 10 dims regressed except CC (flat). **Not promoted.** Possible causes: same-session halo in rapid scoring, distribution mismatch, data dilution.

**Hunt: efficiency + human-rights alignment.** Identified: PSQStudent class duplicated 8×, DIMENSIONS list in 10 files, no demographic bias testing, WEIRD assumptions in CO/AD rubrics, Dreaddit consent gap, dual-use ONNX risk.

**Session summary format co-designed.** ADHD/autism-optimized ASCII dashboard: HEADER → WHAT HAPPENED → THEORY → ⚑ EPISTEMIC FLAGS → PUB SCORECARD → MY REASONING → WHAT'S NEXT. Voice protocol calibrated via 9-order knock-on analysis of how tone choices cascade through scientific collaboration. Restore point: git tag `paradigm-shift-summary-format` (919f496).

**BOOTSTRAP.md created.** Documents how to bring a fresh Claude Code install to full project context from git clone. Identifies critical portability gap: MEMORY.md is path-hash dependent.

**CLAUDE.md created.** Stable project conventions split from MEMORY.md into auto-read repo-root file. Solves the MEMORY.md portability gap — fresh sessions get full project context without manual bootstrap.

**max_length audit completed.** Agent found 2 additional bugs: `criterion_validity_cmv.py` uses max_len=512 (should be 128), `criterion_cgawiki_temporal.py` uses MAX_LENGTH=256 (should be 128). DonD and CaSiNo unaffected.

**11 historical models re-evaluated at max_length=128.** Correction is NOT uniform (range: +0.033 to −0.049). v22a corrected=0.706, v23 corrected=0.698 — relative ordering shifted. v22a may be actual best model. Production calibration re-fit saved.

**Discrepancy: v23 held-out_r 0.684 vs 0.698.** Earlier manual run reported 0.684; batch re-eval computes 0.698 from mean of per-dim r values. Needs investigation — averaging method or GT label difference.

**Pending decisions (deferred to next session):**
- Revert 3,680 rapid-scored records from psq.db? (contrarian claim from MY REASONING)
- State "scoring hygiene > quantity" as paper methods principle?
- Test H1 (halo) vs H2 (source mismatch) for v27 regression?
- Resolve v22a vs v23 as production model after discrepancy investigation

▶ distillation-research.md §62 (pending)

### Session `20260301-1800` (re-scoring attempt, halo replication, APA conversion)

**Re-scoring 368 reverted texts attempted — HALO CONTAMINATED.** Scored all 10 dimensions sequentially in a single session, violating the one-dim-per-session protocol. Result: mean |*r*| = .811 — well above the .658 threshold. Three contamination tiers emerged:

| Tier | Dims | Mean |*r*| | Pattern |
|---|---|---|---|
| Careful (per-text reasoning) | TE, HI, AD, ED | .777 | Context window memory |
| Moderate | RC, TC | .973 | Near-identical |
| Rapid (scoring fatigue) | RB, CC, DA, CO | .887 | 54–60% at score 5 |

**All 10 dims discarded.** No scores entered psq.db (label_separated.py writes to /tmp only). Validates one-dim-per-session protocol — session isolation is the minimum decontamination requirement.

**3,680-score revert confirmed clean (prior session).** DB backup: psq.db.bak-pre-revert-20260301. Post-revert: 22,304 texts, 90,361 scores, 36,771 sep-LLM.

**25 residual pre-revert scores found.** 7 TE (civil_comments), 18 DA (ucc), all from 2026-02-27, all half-point values. Survived revert filter. Low-priority cleanup.

**v22a vs v23 resolved.** v22a corrected held-out *r* = .695 > v23 = .684. Δ = .011 < SE ≈ .05 (noise). Promoting v22a would invalidate 4 criterion studies. Parsimony: hold v23.

**APA 7th edition conversion progress:**
- criterion-validity-summary.md — COMPLETE (prior session)
- psychometric-evaluation.md — key sections COMPLETE (prior session)
- journal.md — in progress (background agent)
- distillation-research.md — in progress (background agent)

**Pending:** Re-score 368 texts properly (1 dim/session × 10 sessions). Clean 25 residual scores. Complete APA conversion of journal.md and distillation-research.md.

▶ distillation-research.md §64 (pending), psychometric-evaluation.md §3c

---

### Session `20260306-2100` (10-dim rescore complete; v29 training)

**All 368 reverted texts re-scored across 10 sessions (separated-llm protocol).** Each dimension scored in isolation (one dim per Claude Code session) to prevent halo contamination. The one-dim-per-session protocol was validated as essential in the prior re-scoring attempt (mean |*r*|=.811 when all dims scored in a single session). Ten sessions were executed across multiple conversations; each dim committed separately.

**Score distributions — all well-differentiated (contrasting with 43% score=5 composite-proxy concentration):**

| Dim | N | Mean | Score=5 fraction |
|---|---|---|---|
| TE | 368 | — | — |
| HI | 368 | — | — |
| AD | 368 | 4.27 | 28.5% |
| ED | 368 | 4.84 | 36.7% |
| RC | 368 | 4.51 | 29.1% |
| RB | 368 | 4.25 | 38.9% |
| TC | 368 | 4.21 | 34.0% |
| CC | 368 | 4.12 | 30.4% |
| DA | 368 | 4.36 | 37.0% |
| CO | 368 | — | — |

All score=5 fractions substantially below the 43% composite-proxy baseline, confirming improved discriminative signal.

**Corpus pattern confirmed:** Two main clusters — Canadian/US political news comments (IDs ≈ 0–249; hostile/adversarial; skews 2–4 on most dims) and empathic support dialogs (IDs ≈ 250+; skews 5–8). Bimodal distribution is genuine construct signal, not artifact.

**Assembly and DB ingest:**
- Assembled: `data/rescore-368-assembled.jsonl` (368 records × 10 dims)
- Ingested via `migrate.py --ingest` as `new_separated` file_role: 3,680 score observations
- Separated-llm confirmed as `best_scores` priority for all 368 texts (verified by query)
- DB after ingest: 22,304 texts, 94,041 total scores, 40,451 separated-llm scores

**v29 training launched.** `scripts/distill.py --out models/psq-v29 --drop-proxy-dims`. Training in progress as of session close. Target: held-out TE ≥ 0.800 (v23=0.795, v28=0.762), overall held-out *r* > 0.684.

**Backups committed:** All 10 per-dim score JSONs in `data/labeling-sessions/`. Commits: RC=08ca8ba, RB=1c7f662, TC=7a97c8a, CC=fac0cc3, DA=10b1f25, CO=fd34866. Assembly+ingest: 40342d1.

▶ TODO.md §B3 (F2 now complete), distillation-research.md §65 (v29 pending)

---

### Session `20260307-0950` (v29/v30 evaluation; B3 fully diagnosed; TE expansion started)

**v29 evaluated — REJECTED.** `--drop-proxy-dims`, data includes rescore-368 (3,680 sep-llm scores). Held-out overall *r* = 0.668 (< v23 0.684). TE = 0.734 (< v23 0.795, target was ≥ 0.800). 6 dims regressed, 2 improved (TC +0.012, CC +0.018), 2 approximately equal (AD, RC). **v23 remains production.**

- Root cause: 368 texts = 9.5% of TE training data (368/3,852). Too dilute to shift TE label distribution.

**`--train-dims` flag implemented in `scripts/distill.py`.** Zeroes out non-selected dim masks in the training loop so only target-dim gradients flow. All 10 heads still present; val/test evaluates all dims. Smoke-tested (1 epoch, --no-save, confirmed "Training on 1 dim(s): ['threat_exposure']").

**v30-te-ceiling trained — diagnostic only, NOT promoted.** `--train-dims threat_exposure --drop-proxy-dims`. Held-out TE = **0.762** (< v23 0.795). DB test-split result (−0.035) was misleading; canonical 100-text held-out is the correct metric.

**B3 fully diagnosed:**

| Hypothesis | Test | Result |
|---|---|---|
| F2: Replace 368 proxy TE labels with sep-llm | v29 training | REJECTED — 9.5% dilution, no improvement |
| Single-task interference: multi-task hurts TE | v30 single-task | REJECTED — multi-task HELPS (+0.033 bonus) |
| Natural data ceiling | v30 vs v23 gap | 0.762 single-task vs 0.795 multi-task → ceiling is data volume |

**Root cause confirmed:** TE requires (a) multi-task scaffolding (+0.033) AND (b) more clean TE training data. The 3,852-text TE corpus has insufficient high-quality labels at the extremes.

**Path forward: TE unlabeled-pool expansion.** Score 500+ texts from `data/unlabeled-pool.jsonl` on TE dimension using separated-llm protocol. Target TE ≥ 0.830 (v23 + anticipated gain from ~10% data volume increase at high label quality). Starting immediately.

▶ distillation-research.md §66 (v29/v30 evaluation + B3 root cause diagnosis)

---

### Session `20260307-1200` (B1/B2 bug diagnosis + production fix; interagent sync)

**PSQ sub-agent session (Chromabook).** Executed /sync, diagnosed production bugs from unratified-agent scoring data, deployed B1 fix on Hetzner.

**Interagent activity:**
- Received command-response-ack from psychology-agent (to-psq-sub-agent-001.json). Independent verification: TLS endpoint live at https://psq.unratified.org/score, Caddy installed, firewall hardened (port 3000 closed), onnxruntime-node postinstall fix durable, wrangler secret set.
- MANIFEST.json transport discovery layer introduced by psychology-agent.
- PR #18 delivered to psychology-agent: scoring interpretation response (psq-interpretation-001.json) diagnosing B1/B2 from unratified-agent's 4-text scoring run.

**B1 diagnosed and fixed (confidence head dead):**
- Tested 3 texts (advocacy, constituent guide, "cooking pasta in a sunny kitchen") on both quantized and fp32 ONNX models — ALL returned identical confidence values for ALL 10 dims regardless of input
- Root cause: model confidence head collapsed to per-dimension constants during training
- calibration.json `r_confidence` values were seeded from the dead model output (not actual held-out r)
- Fix: updated calibration.json with correct v23 held-out Pearson r values, modified student.js to use static r as confidence with `confidence_type: "held_out_r"` indicator
- Deployed on Hetzner, service restarted, verified: 38ms inference, correct confidence values in response

**B2 verified as resolved:**
- HI calibration dead zone (raw 5.85-7.65 → 6.69) was already fixed by psychology-agent's isotonic-v2 recalibration (calibration_version: isotonic-v2-2026-03-06)
- Current HI scores differentiate: 6.55, 6.99, 7.33, 7.39 for 4 test texts
- 11 remaining wide flat bins across other dims (TE, AD, ED, TC, CO) — deferred to post-v32 recalibration

**Commits:** 61eece2 (scoring interpretation), 54a1a85 (B1 fix), 9629412 (B2 fix — by psychology-agent)

---

### Session `20260307-1048` (TE expansion ingested; v31 trained and evaluated)

**TE expansion assembled and ingested.** `data/te-expansion-500-assembled.jsonl` assembled (--partial, 1/10 dims scored). Migrate.py confirmed 500 texts × 1 score observation each. Placeholder scores (score=5, confidence=0.1) for remaining 9 dims filtered by the ≤0.15 confidence threshold in migrate.py. DB after ingest: TE sep-llm scores = 6,109.

**Score distribution:** TE training data (after proxy drop): 4,135 rows, score=5 fraction = 34% (down from 41.5% in full training_data view). The expansion texts (dreaddit low-TE, empathetic_dialogues high-TE, berkeley extreme-TE) successfully shifted distribution toward extremes.

**v31 trained** (`--drop-proxy-dims`, no --train-dims): 8 epochs to early stopping (patience=3 at epoch 5, val_r=0.4625). 14,859 train / 2,207 val / 2,291 test. 466s/epoch × 8 = ~37 minutes.

**v31 held-out results:**

| Metric | v23 | v31 | Δ |
|---|---|---|---|
| threat_exposure | 0.795 | 0.773 | −0.022 ↓ |
| hostility_index | 0.669 | 0.716 | +0.047 ↑ |
| authority_dynamics | 0.713 | 0.671 | −0.042 ↓ |
| energy_dissipation | 0.760 | 0.773 | +0.013 ↑ |
| regulatory_capacity | 0.768 | 0.765 | ≈0 |
| resilience_baseline | 0.597 | 0.585 | −0.012 ↓ |
| trust_conditions | 0.681 | 0.711 | +0.030 ↑ |
| cooling_capacity | 0.736 | 0.747 | +0.011 ↑ |
| defensive_architecture | 0.588 | 0.501 | −0.087 ↓ |
| contractual_clarity | 0.538 | 0.553 | +0.015 ↑ |
| **Average** | **0.684** | **0.679** | **−0.005** |

**v31 REJECTED.** TE improved +0.039 vs v29 (expansion strategy confirmed working) but did not recover to v23 (−0.022 below). Overall average 0.679 < 0.684. DA regression (−0.087) likely within noise at n=88 (3 consecutive declines: 0.588→0.531→0.501 worth monitoring). **v23 remains production.**

**B3 path forward:** The data expansion strategy is validated — 500 texts shifted TE from 0.734 (v29) to 0.773 (v31). To recover v23 TE (0.795) or surpass it, an additional 500–1,000 TE texts from the unlabeled pool are needed (v32 attempt). F1 (recalibrate) deferred until v32 shows improvement.

▶ TODO.md §B3 updated, distillation-research.md §67 (v31 TE expansion results)
