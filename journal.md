# PSQ Research Journal

A chronological narrative of the Psychoemotional Safety Quotient (PSQ) project, from initial conception through distillation into a deployable model. Written for scientific reproducibility and to document the research decisions, failures, and breakthroughs that shaped the instrument.

**Principal investigator:** Kashif Shah
**Research assistant:** Claude (Anthropic) — LLM-assisted development, labeling, and analysis
**Start date:** May 2022 (conceptual) / February 25, 2026 (formalization)
**Current date:** 2026-02-27

---

## Table of Contents

1. [Origin: From Manifesto to Methodology](#1-origin-from-manifesto-to-methodology)
2. [Defining the Construct](#2-defining-the-construct)
3. [The Distillation Hypothesis](#3-the-distillation-hypothesis)
4. [Proxy Validation: Can We Cheat?](#4-proxy-validation-can-we-cheat)
5. [Composite Ground Truth: Building a Frankenstein](#5-composite-ground-truth-building-a-frankenstein)
6. [First Models (v1–v2d): The Slow Climb](#6-first-models-v1v2d-the-slow-climb)
7. [The Data Quality Reckoning (v3–v4)](#7-the-data-quality-reckoning-v3v4)
8. [Architecture Wars: DeBERTa vs DistilBERT](#8-architecture-wars-deberta-vs-distilbert)
9. [Signal Starvation and Synthetic Data (v5–v9)](#9-signal-starvation-and-synthetic-data-v5v9)
10. [The Held-Out Test: Facing Reality](#10-the-held-out-test-facing-reality)
11. [Relabeling: Teaching the Teacher (v10–v13)](#11-relabeling-teaching-the-teacher-v10v13)
12. [The Civil Comments Poisoning](#12-the-civil-comments-poisoning)
13. [Construct Validity Crisis](#13-construct-validity-crisis)
14. [Current State and Open Questions](#14-current-state-and-open-questions)

---

## 1. Origin: From Manifesto to Methodology

**May 2022.** The PSQ concept emerged from a late-night email brainstorm — a list of 71 operational terms under the umbrella "Psychology - Juris - Engineering" (PJE). Terms like *psychoemotional safety quotient*, *psychoemotional cooling*, *psychoemotional energy dissipation*. Raw vocabulary, no formal definitions, no measurement strategy.

**February 25, 2026.** An external critique described PJE as "a manifesto, not a methodology." The critique was fair: PJE had operational definitions but no novel constructs, methods, or instruments. In response, we formalized the Psychoemotional Safety Quotient — a 10-dimension instrument for evaluating the psychoemotional safety climate of text content.

The key insight was that each of the 71 PJE terms could be mapped to validated psychological instruments. The PSQ doesn't invent new psychology; it synthesizes existing constructs into a multi-dimensional safety metric.

**Documents produced:** `psq-definition.md` (theoretical foundation), `intermediate-state.md` (instrument mapping), `final-state.md` (operational specification), `psychometric-evaluation.md` (validation framework).

## 2. Defining the Construct

The PSQ measures 10 dimensions of psychoemotional safety in text:

| # | Dimension | Anchored by | Scale |
|---|---|---|---|
| 1 | Threat Exposure | COPSOQ, NAQ, Abusive Supervision Scale | 1–10 (10 = safe) |
| 2 | Regulatory Capacity | ERQ, DERS, CERQ | 1–10 |
| 3 | Resilience Baseline | CD-RISC, BRS, Grit Scale | 1–10 |
| 4 | Trust Conditions | Rotter ITS, OTI, Trust Questionnaire | 1–10 |
| 5 | Hostility Index | Cook-Medley, BPAQ, STAXI-2 | 1–10 |
| 6 | Cooling Capacity | CPI, Gross reappraisal, Recovery Experience | 1–10 |
| 7 | Energy Dissipation | Effort-Recovery, COR, Flow Short Scale | 1–10 |
| 8 | Defensive Architecture | DSQ, DMRS, Vaillant hierarchy | 1–10 |
| 9 | Authority Dynamics | French & Raven, MLQ, Tepper ABS | 1–10 |
| 10 | Contractual Clarity | PCI, Morrison & Robinson, COPSOQ | 1–10 |

Each score runs 1–10 where 10 represents the safest possible configuration. The composite PSQ is a weighted average.

The construct is **novel** in that no existing instrument combines threat exposure with regulatory capacity, defensive architecture, and contractual clarity into a single index. Adjacent work (Edmondson's psychological safety, Lazarus's appraisal theory) addresses subsets of these dimensions but not the full picture.

## 3. The Distillation Hypothesis

The PSQ was first implemented as an LLM-based evaluator: send text to Claude, get 10 dimension scores. This works (validated against psychometric criteria) but costs ~$0.10 per evaluation and takes ~60 seconds (10 sequential API calls).

**Hypothesis:** We can distill the LLM's scoring behavior into a small local model (DistilBERT, 66.7M parameters) that runs in ~20ms for zero API cost.

**Challenge:** Training data. The LLM can label ~500 texts per batch at reasonable cost. We need 10,000+ training examples across 10 dimensions with known scores.

## 4. Proxy Validation: Can We Cheat?

**Feb 26, 2026.** First attempt: use existing toxicity models as free label sources.

We tested `detoxify` (Jigsaw-trained toxicity classifier) against the Berkeley Measuring Hate Speech dataset. Result: detoxify's `toxicity` score correlates at r=0.68 with Berkeley's IRT-derived `hate_speech_score`, and r=0.66 for `insult`. Close to our r>0.70 threshold but not quite.

**Decision:** Detoxify is insufficient as a standalone proxy teacher. Instead, we'll build a **composite ground truth** — map existing labeled datasets to PSQ dimensions using domain-specific proxy formulas, and supplement with LLM-labeled gold-standard samples.

This was the right call. The proxy approach would have locked us into a two-dimensional model (hostility + threat only). The composite approach, while much harder to build, gave us coverage across all 10 dimensions.

## 5. Composite Ground Truth: Building a Frankenstein

We assembled training data from 11 source datasets:

**Tier 1 (Primary):** Berkeley Hate Speech (IRT scores → hostility, threat), Civil Comments (crowd toxicity → hostility), GoEmotions (27 emotions → 7 PSQ dims), UCC Unhealthy Conversations (4 unhealthy types → 5 dims).

**Tier 2 (Expanded):** Dreaddit (stress → energy dissipation), ESConv (support strategies → regulatory capacity), Empathetic Dialogues (emotions → resilience, regulation), CaSiNo (negotiation → contractual clarity), Stanford Politeness (→ authority dynamics), ProsocialDialog (safety → defensive architecture).

**Tier 3 (LLM gold-standard):** Claude-labeled samples from an unlabeled text pool, plus targeted synthetic text generation.

Each proxy mapping is a formula — e.g., Berkeley's `hate_speech_score` (range -8 to +6, where negative = hateful) maps to PSQ `hostility_index` via `10 - ((hs - min) / (max - min)) * 9`. Confidence is set based on the proxy's semantic proximity to the PSQ construct.

**Total training data by v12:** 17,643 composite + 4,199 LLM = ~21,842 records.

## 6. First Models (v1–v2d): The Slow Climb

**v1 (Feb 26).** First training run. DeBERTa-v3-small, 4 source datasets, avg test r=0.492. Hostility was strong (0.74); authority and contractual were near zero. The model was learning *something* but not the right things for most dimensions.

**v2a–v2d.** A cascade of fixes:
- v2a: Added 4 new proxy datasets (Dreaddit, ESConv, CaSiNo, Politeness). Fixed a sample-weighting bug that was zeroing out LLM samples.
- v2b/v2c: Added LLM-labeled gold-standard samples. Introduced confidence-weighted loss (`conf^power * MSE`) so low-confidence proxy labels contribute less.
- v2d: Fixed structural issues in the data pipeline (dedup, split leaks). **avg test r=0.585**. First time all 10 dimensions were positive.

**Key lesson from v2 era:** More data helps, but only if the proxy-to-construct mapping is valid. Bad proxies poison dimensions regardless of volume.

## 7. The Data Quality Reckoning (v3–v4)

**v3 (Feb 26).** Cross-source correlation analysis revealed that some proxy mappings were actively harmful:
- **Diplomacy dataset → trust_conditions:** The Diplomacy game measures *sender deceptive intent*, but trust in text is about *the environment's trustworthiness*. A skilled liar's text reads as high-trust. MAE was 2.405 (worst source). **Removed.**
- **UCC generalisation_unfair → contractual_clarity:** Negative correlation (r=-0.10) with LLM labels, bias of -2.32. The mapping was teaching the opposite signal. **Removed.**
- **UCC condescending → authority_dynamics:** Narrow construct, +2.8 bias. Retained but confidence halved.

**v3b.** Authority_dynamics still collapsed. Root cause: zeroing out the bad proxy data revealed that the remaining sources (Politeness, UCC condescending) had fundamentally compressed ranges (std=0.73 vs LLM std=1.72). The model learned "predict the mean."

**v4.** Introduced squared confidence weighting (`conf^2.0`) and two-phase confidence warmup. Test r improved slightly but authority remained stubborn.

**Key lesson:** You cannot fix a bad proxy mapping by adjusting weights. You need to remove it and supply genuine signal (LLM labels or better proxies).

## 8. Architecture Wars: DeBERTa vs DistilBERT

**Feb 26.** We ran parallel experiments with DeBERTa-v3-small (141M params) and DistilBERT-base-uncased (66.7M params).

| Metric | DeBERTa | DistilBERT |
|---|---|---|
| Test avg r | 0.48–0.52 | 0.50–0.55 |
| Training time/epoch | 45 min | 12 min |
| GPU memory | ~5.8 GB | ~3.2 GB |
| Batch size possible | 8 | 16 |

DistilBERT won on every metric. On our GTX 1060 6GB GPU, DeBERTa required gradient accumulation (effective batch 32 via 8 × 4 accumulation steps) and was 4x slower. The accuracy difference was within noise.

**Decision:** DistilBERT for all subsequent training. The 50% smaller model trains faster, fits more comfortably on consumer GPU, and produces equal or better correlations on our task.

**Hypothesis for why:** PSQ scoring depends more on lexical-semantic features (word choice, emotional tone) than on the deep syntactic understanding where DeBERTa excels. DistilBERT's simpler attention is sufficient.

## 9. Signal Starvation and Synthetic Data (v5–v9)

**v5–v8 (Feb 27).** With proxy quality issues mostly resolved, the remaining bottleneck was **signal starvation** — dimensions with too few training samples or too narrow a score range.

**Diagnosis by dimension:**

| Dimension | Problem | Training records | Score range |
|---|---|---|---|
| authority_dynamics | Proxies zeroed, only LLM data | ~400 | 3–8 (compressed) |
| contractual_clarity | Tiny dataset | ~800 | narrow |
| threat_exposure | Ceiling effect (34.7% at 10.0) | ~6,000 | but 52.5% in [8–10] |
| energy_dissipation | Range ceiling at 6.8 | ~4,000 | [0, 6.8] |

**Strategy: targeted synthetic generation.** We used Claude to generate batches of realistic text scenarios and score them on specific dimensions:

- **Authority dynamics (batches ad_1 through ad_8):** 726 texts across workplace, education, family, institutional, community, and online contexts. Deliberately targeted the full 1–10 range.
- **Contractual clarity (co_2, co_3):** 368 negotiation/agreement scenarios.
- **Threat exposure (te_2):** 200 texts weighted toward low-safety scores (1–4) to counteract the ceiling effect.
- **Energy dissipation (ed_2):** 150 texts including high-drain scenarios (7–10 range).
- **Defensive architecture (da_2):** 191 boundary and coping pattern texts.

We also **relabeled** 1,000 existing composite texts through the LLM, targeting dimensions where proxy labels were most suspect (threat, energy, regulation, defense).

**v9 result:** test_r=0.515, held-out_r=0.385. Four strong dimensions (hostility 0.70, cooling 0.70, trust 0.63, resilience 0.52). Six dimensions still below 0.40 on held-out.

## 10. The Held-Out Test: Facing Reality

**Feb 27.** We built a proper held-out test: 100 texts (20 from each of 5 source datasets), independently LLM-labeled in two batches of 50. These texts had no overlap with training data.

The held-out test revealed a **25% generalization gap**: test_r=0.515 vs held-out_r=0.385. The model was overfitting to the proxy distribution.

**Dimension tiers on held-out data:**

| Tier | Dimensions | Held-out r | Notes |
|---|---|---|---|
| Strong | hostility, cooling, trust, resilience | 0.52–0.70 | Good proxy coverage + LLM supplement |
| Moderate | authority, defensive, contractual | 0.32–0.46 | Mostly LLM/synthetic signal |
| Weak | threat, energy, regulatory | 0.12–0.30 | Proxy data actively misleading or compressed |

**Key insight:** Dimensions with good psychometric proxy coverage generalize well. Dimensions trained primarily on synthetic data don't generalize as well yet — not because synthetic data is bad, but because there isn't enough of it to overcome the noisy composite baseline.

## 11. Relabeling: Teaching the Teacher (v10–v13)

**Feb 27.** New strategy: instead of generating new synthetic texts, relabel existing composite texts with LLM scores for the four weakest dimensions. This gives the model high-quality labels on texts it already sees during training.

- Relabeled 250 texts each for threat_exposure, energy_dissipation, regulatory_capacity, and defensive_architecture.
- Added remaining synthetic batches (ad_8: 305, te_2: 200, ed_2: 150, da_2: 191).

**v10:** test_r=0.534, held-out_r=0.425. A 10% improvement on held-out — relabeling works.

**v13 (current best):** test_r=0.553, held-out_r=0.428. Includes the Civil Comments fix (see next section). Best test_r ever, held-out essentially flat vs v10.

## 12. The Civil Comments Poisoning

**Feb 27.** During v12 training prep, we diagnosed why `threat_exposure` was permanently broken on held-out data (r=0.12, mean bias +4.31 — model predicts everything is safe).

**Root cause:** The Civil Comments dataset mapped `1 - severe_toxicity - threat` to threat_exposure. This seems reasonable: "not threatening" = "safe." But it's backwards. Civil Comments labels whether the *author is making threats*, not whether the *text describes a threatening environment*. Result: texts describing harassment, rape, ethnic cleansing scored 9–10 ("perfectly safe") because the authors weren't directly threatening anyone.

**1,754 out of 1,853 Civil Comments records (95%)** had threat_exposure scores of 9.0 or higher. The model learned "text mentioning violence = safe" from 1,754 examples of exactly this.

**Fix:** Removed threat_exposure entirely from Civil Comments in the composite builder. Threat will be learned from LLM labels + synthetic data only (1,400 records with correct signal).

**Lesson:** Proxy mapping failures can be subtle. "Not threatening" ≠ "safe." The proxy's semantic framing must match the construct's target perspective (author intent ≠ environmental safety).

## 13. Construct Validity Crisis

**Feb 27.** We computed inter-dimension correlations on held-out data (LLM-scored, all 10 dims per text). Nearly all pairs showed r > 0.70. Some pairs exceeded r > 0.90: authority × hostility (0.96), regulatory × resilience (0.95), cooling × defensive (0.94).

This violates discriminant validity. If all dimensions move together, the 10-dimension model may be measuring one underlying factor with noise.

**Three potential explanations:**
1. **p-factor (general safety):** There's a genuine general factor of psychoemotional safety, analogous to the g-factor in intelligence. All 10 dimensions load onto it.
2. **Short text entanglement:** With 50–500 word texts, there isn't enough information to discriminate between dimensions. A threatening text is also hostile, also low-trust, also low-cooling.
3. **LLM halo effect:** When the LLM scores all 10 dimensions simultaneously, it anchors on an overall impression and adjusts individual scores around it.

**Halo effect experiment (Feb 27).** We scored 30 texts two ways: (a) all 10 dimensions in a single LLM call ("joint"), and (b) each dimension independently in separate LLM calls ("separated").

Results:
- **Mean joint inter-correlation:** r=0.641
- **Mean separated inter-correlation:** r=0.494
- **Mean halo inflation:** 0.147 (joint correlations are ~15% higher than separated)

The halo effect is real but not the whole story. Some pairs drop dramatically when separated (auth × resilience: 0.76→0.04, **strong halo**). Others barely change (reg × resilience: 0.95→0.93, **genuine overlap**).

**Emerging structure when halo is removed:**
- **Cluster 1 (Interpersonal climate):** authority, contractual, trust, threat — high mutual correlations (0.70–0.88)
- **Cluster 2 (Internal resources):** regulatory, resilience, defensive — high mutual correlations (0.71–0.93)
- **Bridge dimensions:** cooling, energy, hostility — correlate with both clusters

This is consistent with the research literature's distinction between environmental demands (JD-R model demands) and personal resources (JD-R model resources). A bifactor or hierarchical model may be more appropriate than 10 independent dimensions.

**Status:** Open question. Four restructuring alternatives under evaluation (bifactor, JD-R 2-factor, 4-factor, 3-level hierarchy).

## 14. Current State and Open Questions

### Model Performance (v13, 2026-02-27)

| Metric | Value |
|---|---|
| Architecture | DistilBERT-base-uncased (66.7M params) |
| Training data | 17,643 composite + 4,199 LLM = ~21,842 |
| Test avg r | 0.553 |
| Held-out avg r | 0.428 |
| Generalization gap | 23% |
| ONNX model size | 64 MB (INT8 quantized) |
| Inference time | ~20ms per text |

### Psychometric Properties

| Property | Status | Evidence |
|---|---|---|
| Test-retest reliability | Excellent | ICC = 0.935 (perturbation-based) |
| Discriminant validity vs sentiment | Strong | r = 0.205 vs VADER |
| Confidence calibration | Done | Isotonic regression on score + confidence |
| Held-out generalization | Moderate | r = 0.428 across 100 real-world texts |
| Construct (discriminant) validity | Under investigation | Inter-dim correlations too high; halo effect confirmed |

### Open Questions

1. **Dimensionality:** Should we restructure from 10 independent dimensions to a hierarchical model (g-factor + clusters)? The halo test suggests yes, but sample size is small (n=30).
2. **Threat exposure:** Still the weakest dimension (held-out r=0.12). Civil Comments fix applied but untested at scale.
3. **Training ceiling:** test_r has grown slowly (0.49 → 0.55 over 13 versions). May be approaching the ceiling for this architecture + data mix.
4. **Real-world validation:** The held-out test uses LLM labels as ground truth. A proper validation study would need human expert ratings.
5. **Deployment:** ONNX export complete (64 MB quantized). Node.js inference provider (`student.js`) wired up. Not yet deployed to production.

### What Worked

- Confidence-weighted loss with squared exponent
- LLM 5x weighting over composite
- Targeted synthetic generation for signal-starved dimensions
- Relabeling existing texts instead of only generating new ones
- Hash-based text splitting to prevent train/test leakage
- DistilBERT over DeBERTa on consumer GPU
- Systematic proxy auditing (removing bad sources > tuning bad sources)

### What Didn't Work

- Detoxify as proxy teacher (insufficient correlation)
- Diplomacy dataset for trust (sender intent ≠ textual trust)
- Civil Comments for threat exposure (author threat ≠ environmental safety)
- UCC generalisation_unfair for contractual clarity (negative correlation)
- DeBERTa-v3-small (slower, worse results on our GPU)
- Trying to fix bad proxy mappings by adjusting confidence weights
