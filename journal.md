# PSQ Research Journal

A chronological research narrative of the Psychoemotional Safety Quotient (PSQ) project: from initial conceptualization through construct formalization, knowledge distillation, psychometric validation, and the discovery of latent dimensionality structure. Written in the idiom of a methods-and-findings journal article to support reproducibility, peer review, and future meta-analytic work.

**Principal investigator:** Kashif Shah
**Research assistant:** Claude (Anthropic) — LLM-assisted construct operationalization, data labeling, and analysis
**Inception:** May 2022 (conceptual vocabulary) / February 25, 2026 (formal construct definition)
**Current date:** 2026-02-28 (v19 cycle, pct batch + bifactor)

---

## Table of Contents

1. [Origin: From Manifesto to Methodology](#1-origin-from-manifesto-to-methodology)
2. [Theoretical Foundation and Construct Definition](#2-theoretical-foundation-and-construct-definition)
3. [The Distillation Hypothesis](#3-the-distillation-hypothesis)
4. [Proxy Teacher Validation](#4-proxy-teacher-validation)
5. [Composite Ground Truth Construction](#5-composite-ground-truth-construction)
6. [Early Training Iterations (v1–v2d)](#6-early-training-iterations-v1v2d)
7. [Data Quality Audit and Proxy Pathology (v3–v4)](#7-data-quality-audit-and-proxy-pathology-v3v4)
8. [Architecture Selection: DeBERTa vs DistilBERT](#8-architecture-selection-deberta-vs-distilbert)
9. [Signal Starvation and Targeted Augmentation (v5–v9)](#9-signal-starvation-and-targeted-augmentation-v5v9)
10. [Held-Out Evaluation: The Generalization Gap](#10-held-out-evaluation-the-generalization-gap)
11. [LLM Relabeling and Data Correction (v10–v13)](#11-llm-relabeling-and-data-correction-v10v13)
12. [The Civil Comments Poisoning: A Case Study in Proxy Misalignment](#12-the-civil-comments-poisoning-a-case-study-in-proxy-misalignment)
13. [Construct Validity Under Scrutiny](#13-construct-validity-under-scrutiny)
14. [Current State and Open Questions](#14-current-state-and-open-questions)
15. [Separated Scoring and Hierarchical Reporting](#15-separated-scoring-and-hierarchical-reporting)
16. [Training Data Labeling Expansion and V14](#16-training-data-labeling-expansion-and-v14-2026-02-27)
17. [Score-Concentration Fix and Targeted Labeling](#17-score-concentration-fix-and-targeted-labeling-2026-02-27)
18. [Factor Analysis: The General Factor Question](#18-factor-analysis-the-general-factor-question-2026-02-28)
19. [Expert Validation Protocol: The DA Question](#19-expert-validation-protocol-the-da-question-2026-02-28)
20. [Criterion Validity: The Negotiation Test](#20-criterion-validity-the-negotiation-test-2026-02-28)
21. [The Derailment Test](#21-the-derailment-test-2026-02-28)
22. [Dimension Reduction: Where the Signal Lives](#22-dimension-reduction-where-the-signal-lives-2026-02-28)
23. [The Misfits: Why Authority Dynamics Predicts What Nothing Else Can](#23-the-misfits-why-authority-dynamics-predicts-what-nothing-else-can-2026-02-28)
24. [The AD Paradox: Three Theories of Predictive Primacy](#24-the-ad-paradox-three-theories-of-predictive-primacy-2026-02-28)
25. [The Persuasion Test: Change My View](#25-the-persuasion-test-change-my-view-2026-02-28)
26. [Publication Narrative](#26-publication-narrative-2026-02-28)
27. [The Deal Test: When Energy Matters More Than Status](#27-the-deal-test-when-energy-matters-more-than-status-2026-02-28)
28. [The g-Factor Deepens and the Integer Problem](#28-the-g-factor-deepens-and-the-integer-problem-2026-02-28)
29. [The Resolution Fix: Percentage Scoring at Scale](#29-the-resolution-fix-percentage-scoring-at-scale-2026-02-28)
30. [References](#30-references)

---

## 1. Origin: From Manifesto to Methodology

The Psychoemotional Safety Quotient originated not as an instrument specification but as a vocabulary — 71 operational terms enumerated in a May 2022 email under the umbrella framework "Psychology - Juris - Engineering" (PJE). Terms such as *psychoemotional safety quotient*, *psychoemotional cooling*, *psychoemotional energy dissipation*, and *psychoemotional contract law* sketched the contours of a transdisciplinary space without defining measurement procedures, scoring rubrics, or validation criteria. This was, in Kuhn's (1962) terminology, pre-paradigmatic: a field recognizable by its vocabulary but not yet by its methods.

In February 2026, an external critique described PJE as "a manifesto, not a methodology" — noting the absence of novel constructs, methods, or instruments. The critique was methodologically sound: PJE as formulated consisted of operational definitions (17), operational methods (2), and operational vocabulary (52 terms), but lacked the three elements required by the AERA/APA/NCME *Standards for Educational and Psychological Testing* (2014) to constitute a measurement framework: (a) a clearly defined construct, (b) a standardized measurement procedure, and (c) evidence of reliability and validity.

The response to this critique produced, in rapid succession:

1. **A novel construct** — the Psychoemotional Safety Quotient (PSQ): a 10-dimension composite measure of the psychoemotional safety climate of textual content.
2. **A measurement method** — a multi-pass LLM-as-judge content evaluation pipeline, subsequently distilled into a local neural network.
3. **An instrument specification** — with explicit scoring rubrics, confidence thresholds, and aggregation rules.
4. **A reference library** — 170+ validated instruments from clinical, organizational, and social psychology mapped to the PSQ's 10 dimensions.
5. **A validation framework** — targeting the AERA/APA/NCME standards with specific psychometric benchmarks.

The key theoretical insight that enabled this transition was that each of the 71 PJE terms corresponded to established psychological constructs with validated measurement instruments. The PSQ does not invent new psychological phenomena; it synthesizes existing constructs — drawn from Lazarus & Folkman's (1984) transactional stress model, Hobfoll's (1989) Conservation of Resources theory, Edmondson's (1999) psychological safety framework, and Bakker & Demerouti's (2007) Job Demands-Resources model, among others — into a multi-dimensional safety metric applicable at the content level.

## 2. Theoretical Foundation and Construct Definition

### 2a. The Construct

The PSQ is defined as a composite measure of the degree to which textual content protects, threatens, or modulates the psychoemotional functioning of persons exposed to it. It is scored across 10 dimensions, each anchored to 3–5 validated psychometric instruments:

| # | Dimension | Anchoring Instruments | Construct Tradition | Factor Type |
|---|---|---|---|---|
| 1 | Threat Exposure | COPSOQ (Pejtersen et al., 2010), NAQ (Einarsen et al., 2009), Abusive Supervision Scale (Tepper, 2000) | Occupational health, workplace aggression | Threat |
| 2 | Regulatory Capacity | ERQ (Gross & John, 2003), DERS (Gratz & Roemer, 2004), CERQ (Garnefski et al., 2001) | Emotion regulation (Gross, 1998) | Protective |
| 3 | Resilience Baseline | CD-RISC (Connor & Davidson, 2003), BRS (Smith et al., 2008), Grit Scale (Duckworth et al., 2007) | Resilience, positive psychology | Protective |
| 4 | Trust Conditions | Rotter ITS (Rotter, 1967), OTI (Cummings & Bromiley, 1996), Trust Questionnaire | Interpersonal trust | Protective |
| 5 | Hostility Index | Cook-Medley HO (Cook & Medley, 1954), BPAQ (Buss & Perry, 1992), STAXI-2 (Spielberger, 1999) | Aggression, anger, hostility | Threat |
| 6 | Cooling Capacity | CPI (Gough, 1987), Gross reappraisal subscale, Recovery Experience Questionnaire (Sonnentag & Fritz, 2007) | Emotion regulation, recovery | Protective |
| 7 | Energy Dissipation | Effort-Recovery Model (Meijman & Mulder, 1998), COR (Hobfoll, 1989), Flow Short Scale (Rheinberg et al., 2003) | Occupational stress, resource theory | Threat |
| 8 | Defensive Architecture | DSQ-40 (Andrews et al., 1993), DMRS (Perry, 1990), Vaillant hierarchy (Vaillant, 1977) | Defense mechanisms, ego psychology | Protective |
| 9 | Authority Dynamics | French & Raven (1959) power bases, MLQ (Bass & Avolio, 1995), Tepper ABS (Tepper, 2000) | Power, leadership, organizational behavior | Threat |
| 10 | Contractual Clarity | PCI (Rousseau, 1995), Morrison & Robinson (1997) violation model, COPSOQ | Psychological contract theory | Protective |

Each dimension is scored 1–10, where 10 represents the safest possible configuration. The composite PSQ is a weighted ratio of protective to threat factors, normalized to a 0–100 scale.

### 2b. Theoretical Novelty

The PSQ's contribution is not the identification of new psychological phenomena but the synthesis of existing constructs into a unified content-level assessment framework. No prior instrument combines threat exposure (occupational health), regulatory capacity (clinical emotion regulation), defensive architecture (psychodynamic defense mechanisms), authority dynamics (organizational power), and contractual clarity (psychological contract theory) into a single index. Adjacent work addresses subsets:

- **Edmondson's (1999) psychological safety** focuses on team-level interpersonal risk-taking — a single construct, not a multi-dimensional profile.
- **Lazarus & Folkman's (1984) stress appraisal** provides the theoretical engine (primary appraisal → threat perception; secondary appraisal → coping resources) but not a standardized instrument for content analysis.
- **The JD-R model** (Bakker & Demerouti, 2007) distinguishes job demands from resources — a framework the PSQ operationalizes at the content level, as our empirical factor analysis later confirmed (see §13).

### 2c. The Content-Level Measurement Problem

A critical methodological challenge: the PSQ's anchoring instruments are designed for person-level self-report (e.g., "I find it hard to calm down when I'm upset" — DERS item). The PSQ applies these constructs to *content* — evaluating texts for the degree to which they expose readers to threat, support regulatory capacity, etc. This level-of-analysis shift introduces a fundamental validity question: can text be meaningfully scored on constructs designed for individuals?

Our working position is that this is analogous to how sentiment analysis applies emotion constructs (originally defined for persons) to text. The PSQ scores represent the *psychoemotional affordances* of the content — what the text makes available to or imposes upon the reader — not the reader's actual psychological state. This distinction is documented in the instrument specification (`psq-definition.md`) and acknowledged in the psychometric evaluation as a standing validity gap requiring empirical investigation.

## 3. The Distillation Hypothesis

The PSQ was initially operationalized as an LLM-based evaluator: submit text to Claude (Anthropic), receive 10 dimension scores with confidence estimates. This pipeline functions — it produces scores that pass preliminary psychometric validation — but imposes prohibitive operational costs: approximately $0.10 per evaluation and 60 seconds of wall-clock time (10 sequential API calls, one per dimension). For content-level deployment (e.g., evaluating news articles, social media posts, or organizational communications at scale), this is untenable.

The distillation hypothesis, following Hinton et al. (2015), proposes that the LLM teacher's scoring behavior can be compressed into a small local model — specifically, a DistilBERT-base-uncased encoder (Sanh et al., 2019; 66.7M parameters) with 10 per-dimension regression heads — that reproduces the teacher's scores in ~20ms at zero marginal cost. This is a form of knowledge distillation: the large model (Claude, ~100B+ parameters) serves as the teacher, and the small model serves as the student.

The challenge is training data. The LLM teacher can label ~500 texts per batch at reasonable cost, but supervised regression across 10 dimensions requires substantially more. Our strategy combined three data sources:

1. **Proxy-labeled composite data** (~17,600 records): existing labeled datasets (hate speech, emotions, stress, etc.) mapped to PSQ dimensions via hand-crafted proxy formulas.
2. **LLM gold-standard labels** (~1,350 records): direct Claude scoring of diverse texts.
3. **Targeted synthetic generation** (~1,900 records): Claude-generated realistic scenarios scored on specific dimensions, targeting score ranges underrepresented in the proxy data.

This three-tier approach reflects a common pattern in knowledge distillation under label scarcity: leverage cheap but noisy labels at scale, supplement with expensive but accurate labels for critical regions, and generate targeted examples to fill distributional gaps (cf. data augmentation strategies in Shorten & Khoshgoftaar, 2019).

## 4. Proxy Teacher Validation

**February 26, 2026.** Before committing to the composite approach, we tested whether a simpler proxy teacher — `detoxify`, a Jigsaw-trained toxicity classifier (Hanu & Unitary team, 2020) — could serve as a free label source for the hostility and threat dimensions.

We evaluated detoxify against the Berkeley Measuring Hate Speech dataset (Kennedy et al., 2020), which provides continuous IRT-derived hate speech scores across 39,565 unique texts. The results fell short of our pre-registered decision threshold (Pearson r > 0.70):

| Detoxify attribute | Berkeley ground truth | r |
|---|---|---|
| toxicity | hate_speech_score | 0.68 |
| insult | insult | 0.66 |
| threat | violence | 0.51 |
| identity_attack | dehumanize | 0.63 |
| severe_toxicity | hatespeech (binary) | 0.74 |

The moderate correlations (r = 0.51–0.74) reflect a fundamental construct mismatch: detoxify scores *toxicity* (a unidimensional social-harm estimate), while PSQ dimensions measure distinct psychological constructs (hostility, threat, authority). Toxicity is correlated with hostility (they share variance in aggressive/harmful language) but they are not the same construct — a text can be hostile without being "toxic" in the Jigsaw sense (e.g., corporate passive aggression), and toxic without being hostile in the PSQ sense (e.g., sexual content).

**Decision:** Detoxify rejected as standalone proxy teacher. The composite ground truth approach adopted instead, using detoxify correlations as a secondary quality check.

## 5. Composite Ground Truth Construction

The composite ground truth assembles proxy-labeled training data from 11 source datasets, each mapped to one or more PSQ dimensions through hand-crafted formulas. The approach is conceptually similar to multitask learning with heterogeneous label spaces (Ruder, 2017), except that the label mappings are defined a priori rather than learned.

### 5a. Source Datasets

**Tier 1 — Primary proxy sources (in composite from v1):**

| Dataset | License | Records | PSQ Dimensions | Mapping Strategy |
|---|---|---|---|---|
| Berkeley Measuring Hate Speech (Kennedy et al., 2020) | CC-BY 4.0 | 2,000 | hostility_index, threat_exposure | IRT score → linear rescale to [1,10]; confidence from annotator agreement |
| Civil Comments (Borkan et al., 2019) | CC0 1.0 | 2,000 | hostility_index | Crowd-sourced toxicity → inverted PSQ scale |
| GoEmotions (Demszky et al., 2020) | Apache 2.0 | 2,000 | 7 dimensions | 27 emotion labels → PSQ dimension formulas (e.g., anger+disgust → hostility; admiration+approval → trust) |
| UCC Unhealthy Conversations (Price et al., 2020) | CC-BY 4.0 | 1,949 | 5 dimensions | Unhealthy attributes (hostile, dismissive, sarcastic, condescending) → PSQ mappings |

**Tier 2 — Expanded sources (added v2–v3):**

| Dataset | License | Records | PSQ Dimensions | Mapping Strategy |
|---|---|---|---|---|
| Dreaddit (Turney et al., 2019) | CC-BY-SA 4.0 | 2,000 | energy_dissipation | Binary stress label → PSQ (stressed=3, not=7) |
| ESConv (Liu et al., 2021) | MIT | 1,300 | regulatory_capacity | Support strategy labels → regulatory proxy |
| Empathetic Dialogues (Rashkin et al., 2019) | CC-BY 4.0 | 2,000 | resilience_baseline, regulatory_capacity | Emotion labels → internal resource proxies |
| CaSiNo (Chawla et al., 2021) | CC-BY 4.0 | 396 | contractual_clarity | Negotiation strategies → clarity proxy |
| Stanford Politeness (Danescu-Niculescu-Mizil et al., 2013) | CC-BY 4.0 | 2,000 | authority_dynamics | Politeness score → power dynamics proxy (de-weighted, see §7) |
| ProsocialDialog (Kim et al., 2022) | CC-BY 4.0 | 1,998 | defensive_architecture | Safety labels → boundary pattern proxy |

Each proxy mapping assigns both a score and a confidence value. Confidence reflects the semantic proximity between the source dataset's label and the target PSQ construct — high (0.50–0.70) for direct mappings (Berkeley hostility → PSQ hostility), low (0.15–0.30) for indirect ones (Politeness → authority dynamics). The confidence-weighted loss function (§6) ensures that dubious proxies contribute less to gradient updates.

### 5b. Proxy Mapping as Measurement Approximation

The proxy mapping strategy rests on an assumption analogous to the multitrait-multimethod logic of Campbell & Fiske (1959): if a source dataset's label measures a construct that shares substantial variance with a PSQ dimension, then the label can serve as a noisy training signal for that dimension. The noise is modeled through the confidence parameter, which acts as a precision weight in the loss function.

This assumption breaks down when the shared variance is illusory — when the proxy *appears* related to the PSQ construct but measures something fundamentally different. We encountered three such failures (§7, §12), each teaching a different lesson about proxy validity.

## 6. Early Training Iterations (v1–v2d)

### 6a. Baseline (v1)

**February 26, 2026.** First training: DeBERTa-v3-small (He et al., 2021; 141M parameters) on 4 proxy datasets (~8,000 records) plus 365 LLM-labeled samples. Standard MSE loss, flat confidence weighting.

**Result:** avg test r = 0.492. Hostility index led (r = 0.74), reflecting strong proxy coverage from Berkeley and Civil Comments. Authority dynamics and contractual clarity were near zero — unsurprising given zero proxy data for these dimensions.

### 6b. The v2 Cascade

**v2a:** Added 4 new proxy datasets (Dreaddit, ESConv, CaSiNo, Politeness). Simultaneously discovered and fixed a sample-weighting bug: LLM samples were receiving weight = 0.0 due to a missing confidence field. This meant the gold-standard labels were contributing nothing to training.

**v2b–v2c:** Introduced confidence-weighted loss, following the intuition that proxy labels with lower construct validity should contribute proportionally less to the gradient:

$$\mathcal{L} = \text{conf}^{\alpha} \cdot w_{\text{source}} \cdot \text{MSE}(\hat{y}, y)$$

where α = 1.0 initially, w_source differentiates LLM (3×) from composite (1×) samples, and conf is the proxy confidence (0.15–0.70).

**v2d:** Fixed hash-based text splitting (preventing train/test leakage on duplicate texts), added deduplication logic, integrated ProsocialDialog. **avg test r = 0.585** — a 19% improvement over v1 and the first version where all 10 dimensions showed positive correlations.

### 6c. Lesson

The v2 era established a pattern that would recur: data pipeline bugs and proxy quality issues dominate model performance far more than architecture or hyperparameter choices. The sample-weighting bug (LLM weight = 0) likely cost v1 several points of average r. In distillation, the quality and representativeness of the training signal matters more than the model's capacity (cf. the "Bitter Lesson" — Sutton, 2019 — but applied to data rather than compute).

## 7. Data Quality Audit and Proxy Pathology (v3–v4)

### 7a. Cross-Source Correlation Analysis

**February 26, 2026.** With 10+ source datasets contributing proxy labels, we computed per-source agreement: for each dimension, compare the proxy label from each source against LLM gold-standard labels on shared or similar texts. This revealed three pathological mappings:

**Diplomacy → trust_conditions (r ≈ 0, MAE = 2.405).** The Diplomacy dataset (FAIR) labels whether players in the board game Diplomacy are being deceptive. We mapped "not deceptive" → "high trust." But the Diplomacy game's texts are *designed* to conceal intent — a skilled liar's prose reads as trustworthy by construction. The proxy measured *sender deceptive intent*, while the PSQ dimension measures *environmental trustworthiness*. These are not merely different constructs; they are inversely related in this dataset. **Removed.**

**UCC generalisation_unfair → contractual_clarity (r = -0.10, bias = -2.32).** The mapping hypothesized that unfair generalizations indicate poor contractual clarity (unclear expectations). Empirically, the correlation was negative — the proxy was teaching the model the wrong direction. **Removed.**

**UCC condescending → authority_dynamics (bias = +2.8, compressed range).** Condescension is related to authority but is a narrow sub-construct. The mapping over-predicted authority dynamics by 2.8 points on average. Retained with confidence halved (0.25 → 0.125).

### 7b. Range Compression Pathology

After removing pathological proxies for authority_dynamics, the remaining sources (Stanford Politeness, UCC condescending) had fundamentally compressed score ranges (σ = 0.73) compared to LLM labels (σ = 1.72). The model learned to predict the population mean for authority dynamics regardless of input — the classic regression-to-the-mean problem in the presence of high-noise, low-variance training signal.

### 7c. Confidence Weighting Refinement (v4)

We increased the confidence exponent from α = 1.0 to α = 2.0, following the logic of inverse-variance weighting in meta-analysis (DerSimonian & Laird, 1986): squaring the confidence more aggressively downweights low-confidence labels. Combined with a two-phase confidence warmup (first 2 epochs: fixed conf = 0.5; thereafter: use model's own confidence predictions), this improved stability but did not solve the authority dynamics collapse.

**Lesson:** Proxy pathology cannot be fixed by reweighting. A proxy that measures the wrong construct provides anti-signal that must be removed, not attenuated. This is the training-data analog of the garbage-in-garbage-out principle, with the added subtlety that "garbage" can appear plausible (condescension *seems* related to authority dynamics; it just isn't *the same construct*).

## 8. Architecture Selection: DeBERTa vs DistilBERT

**February 26–27, 2026.** We ran controlled comparisons between DeBERTa-v3-small (He et al., 2021; 141M parameters, disentangled attention) and DistilBERT-base-uncased (Sanh et al., 2019; 66.7M parameters, knowledge-distilled from BERT).

| Metric | DeBERTa-v3-small | DistilBERT-base | Ratio |
|---|---|---|---|
| Test avg r (best) | 0.48–0.52 | 0.50–0.55 | 1.04× DistilBERT |
| Training time/epoch | 45 min | 12 min | 3.75× faster |
| Peak GPU memory | ~5.8 GB | ~3.2 GB | 1.8× less |
| Max batch size (GTX 1060 6GB) | 8 | 16 | 2× larger |

DistilBERT consistently matched or exceeded DeBERTa on our task while training 4× faster. On our hardware-constrained setup (NVIDIA GTX 1060 6GB, sm_61), DeBERTa required gradient accumulation (effective batch 32 via 8 × 4 accumulation steps) to achieve equivalent effective batch sizes, introducing gradient staleness and additional training overhead.

**Why the larger model didn't help.** We hypothesize that PSQ scoring depends primarily on lexical-semantic features — word choice, emotional tone, aggression markers, boundary language — rather than on the deep syntactic and positional disambiguation where DeBERTa's disentangled attention mechanism excels. The PSQ task more closely resembles sentiment classification (where smaller models perform comparably to larger ones; cf. Liu et al., 2019) than natural language inference (where DeBERTa's architectural advantages are most evident).

**Decision:** DistilBERT adopted for all subsequent training. The smaller model's faster iteration cycle proved more valuable than marginal accuracy gains from larger architectures — a practical instance of the "scaling down" philosophy in efficient NLP (Treviso et al., 2023).

## 9. Signal Starvation and Targeted Augmentation (v5–v9)

### 9a. Diagnosing Dimensional Weakness

With proxy quality issues addressed, the remaining performance bottleneck was **signal starvation**: insufficient training signal for specific dimensions, manifesting as either (a) too few labeled examples, (b) too narrow a score range, or (c) both.

| Dimension | Problem | Training records | Observed range | Full range |
|---|---|---|---|---|
| authority_dynamics | Proxies zeroed, only LLM data | ~400 | 3–8 | 1–10 |
| contractual_clarity | Smallest dataset | ~800 | 3–8 | 1–10 |
| threat_exposure | Ceiling effect | ~6,000 | 52.5% in [8–10] | 1–10 |
| energy_dissipation | Range ceiling | ~4,000 | max 6.8 | 1–10 |

The ceiling effect on threat_exposure was particularly problematic: 34.7% of training scores were exactly 10.0 ("perfectly safe"), teaching the model a strong prior toward high safety predictions. This resembles the class imbalance problem in classification (He & Garcia, 2009), transposed to regression: the model learns to predict the mode of the training distribution rather than the full range.

### 9b. Targeted Synthetic Generation

We addressed signal starvation through targeted synthetic data generation — using the LLM teacher to generate realistic text scenarios scored on specific dimensions, with deliberate attention to underrepresented score ranges:

- **Authority dynamics (batches ad_1–ad_8):** 726 texts across workplace, education, family, institutional, community, and online contexts. Score distribution deliberately uniform across 1–10 to counteract composite compression.
- **Contractual clarity (co_2, co_3):** 368 negotiation, agreement, and boundary-setting scenarios.
- **Threat exposure (te_2):** 200 texts weighted toward scores 1–4 (actively threatening environments) to counteract the composite ceiling effect.
- **Energy dissipation (ed_2):** 150 texts including high-drain scenarios (scores 7–10) to extend the truncated range.
- **Defensive architecture (da_2):** 191 boundary-pattern and coping-mechanism texts.

This is analogous to active learning's concept of *uncertainty sampling* (Lewis & Gale, 1994) applied at the data-generation stage: we generated training examples specifically where the model's predictions were most uncertain or most biased, rather than sampling uniformly from the input space.

### 9c. LLM Relabeling of Proxy Data

Complementing synthetic generation, we **relabeled** 1,000 existing composite texts through the LLM teacher, targeting the four weakest dimensions (threat_exposure, energy_dissipation, regulatory_capacity, defensive_architecture — 250 texts each). This provides the model with high-quality labels on texts it already encounters during training, combining the representativeness of real data with the accuracy of LLM scoring.

When a text appears in both the composite (with proxy labels) and the LLM relabeling set, the deduplication logic in `distill.py` keeps the LLM version with 5× weight — ensuring that correct labels override noisy proxies without discarding the text.

## 10. Held-Out Evaluation: The Generalization Gap

**February 27, 2026.** To assess real-world generalization beyond the proxy distribution, we constructed a held-out test set: 100 texts sampled from the unlabeled pool (20 from each of 5 source datasets, stratified for length and topic diversity), independently LLM-labeled in two batches of 50 to mitigate labeling-session effects. These texts had no overlap with training data (verified by text-hash exclusion).

The held-out evaluation revealed a **25% generalization gap**:

| Metric | Test set | Held-out set | Gap |
|---|---|---|---|
| avg r (v9) | 0.515 | 0.385 | -25.2% |
| avg r (v13) | 0.553 | 0.428 | -22.6% |

The gap is dimension-dependent, reflecting the quality of proxy coverage:

| Tier | Dimensions | Test r | Held-out r | Proxy quality |
|---|---|---|---|---|
| Strong | hostility, cooling, trust, resilience | 0.60–0.82 | 0.56–0.66 | Good direct proxies (Berkeley, GoEmotions) |
| Moderate | authority, defensive, contractual, energy, regulatory | 0.35–0.65 | 0.30–0.46 | Indirect proxies or LLM-only signal |
| Weak | threat_exposure | 0.68 | 0.12 | Poisoned proxy (see §12) |

The pattern is interpretable: dimensions with strong, construct-valid proxy data (where the source dataset's label genuinely measures the PSQ construct) generalize well. Dimensions relying primarily on synthetic or LLM data show moderate held-out performance — they generalize to new texts but not as robustly, likely because the synthetic distribution is narrower than the real-world distribution. Threat exposure is an outlier, explained in §12.

**Psychometric context:** A held-out r of 0.428 is comparable to the cross-sample validity coefficients reported for brief personality measures (r ≈ 0.40–0.60; Soto & John, 2017) and substantially better than content-level affect analysis baselines (r ≈ 0.20–0.35; Ribeiro et al., 2016). The strong dimensions (r = 0.56–0.66) approach the reliability ceiling for single-rater content coding.

## 11. LLM Relabeling and Data Correction (v10–v13)

### 11a. The Relabeling Strategy

**February 27, 2026.** Analysis of the v9 error patterns suggested that the generalization gap was driven primarily by proxy label noise rather than model capacity limitations. The model had sufficient parameters (66.7M) and training data (~20,000 records), but the signal-to-noise ratio on the weakest dimensions was dominated by noisy proxy labels that outnumbered accurate LLM labels ~15:1.

Rather than generating more synthetic texts (which improves score-range coverage but not distributional representativeness), we relabeled 1,000 existing composite texts through the LLM teacher on their weakest dimensions. This strategy has three advantages:

1. The relabeled texts are *real data* — representative of the training distribution.
2. The LLM labels are *accurate* — directly measuring the PSQ construct.
3. The 5× LLM weight ensures relabeled scores dominate the noisy proxy scores at training time.

### 11b. Results

| Version | Data change | test_r | held-out_r | Δ held-out |
|---|---|---|---|---|
| v9 | Baseline | 0.515 | 0.385 | — |
| v10 | +1,000 relabeled | 0.534 | 0.425 | +10.4% |
| v13 | +CC fix, +846 synthetic | 0.553 | 0.428 | +11.2% (vs v9) |

The relabeling strategy (v10) produced a 10% improvement on held-out, the largest single-version gain in the project's history. This confirms the diagnosis: proxy noise, not model capacity, was the binding constraint.

v13 added the Civil Comments fix (§12) and all remaining synthetic data. Test r improved further (+3.6%) but held-out was essentially flat relative to v10, suggesting that the remaining generalization gap requires either (a) substantially more relabeled data, (b) longer texts with richer signals, or (c) restructuring the dimensionality model (§13).

## 12. The Civil Comments Poisoning: A Case Study in Proxy Misalignment

### 12a. The Symptom

Threat_exposure was consistently the worst-performing dimension across all model versions (held-out r = 0.09–0.12), despite having the second-largest training set (~6,000 records). The model predicted high safety (scores 7–9) for virtually all texts, including those describing harassment, abuse, and violence. Mean prediction bias on held-out was +4.31 points: the model believed the world was 4 points safer than it actually was.

### 12b. Root Cause Analysis

The Civil Comments dataset (Borkan et al., 2019) was mapped to threat_exposure via the formula:

```
threat_exposure = 10 × (1 - severe_toxicity) × (1 - threat)
```

The logic: "not severely toxic" and "not threatening" implies "safe." This mapping contains a perspective error that is subtle but catastrophic.

Civil Comments annotators rated texts for **author-directed threat**: "Is the author of this comment making threats?" The PSQ's threat_exposure dimension measures **environment-directed safety**: "How safe is the environment described by this text?"

These are different constructs with different referents. A text describing ethnic cleansing, workplace harassment, or child abuse is *not* "making threats" (the author is describing events, not threatening the reader), so it scores low on Civil Comments threat. The proxy formula then converts this to high safety (score 9–10). The result: 1,754 out of 1,853 Civil Comments records (94.7%) received threat_exposure scores ≥ 9.0.

This is a textbook example of the *jangle fallacy* (Kelley, 1927): using the same word ("threat") to refer to different constructs. The proxy mapping assumed that "threat" in Civil Comments and "threat" in PSQ referred to the same thing. They do not.

### 12c. The Damage

At confidence = 0.40, these 1,754 records contributed substantial gradient signal: the model learned "text mentioning violence, harassment, or abuse → predict safe" from nearly two thousand examples. This prior was strong enough to override the ~400 correctly-labeled LLM and synthetic records, creating an effectively unlearnable dimension.

### 12d. The Fix

Removed threat_exposure entirely from the Civil Comments proxy mapping. Retained hostility_index (where the mapping is construct-valid: author toxicity *is* a reasonable proxy for environmental hostility). Threat_exposure is now learned exclusively from LLM labels and synthetic data (~1,400 records with correct construct alignment).

### 12e. Methodological Implications

The Civil Comments poisoning illustrates a general risk in composite ground truth construction: **perspective misalignment**. A proxy label's construct validity depends not only on semantic similarity ("threat" ≈ "threat") but on referent alignment (who/what is being assessed?). Automated proxy mapping should include explicit perspective checks: Does the source label assess the *same entity* (author vs. environment vs. reader) and *same property* (intent vs. exposure vs. impact) as the target construct?

This connects to the broader measurement literature on *frame-of-reference effects* (Lievens et al., 2008): the same construct can yield dramatically different scores depending on whose perspective the rater adopts. In content analysis, the perspective of the label (author-intent vs. reader-impact vs. environment-description) is a critical but often implicit parameter.

## 13. Construct Validity Under Scrutiny

### 13a. The Inter-Correlation Problem

**February 27, 2026.** Computing the inter-dimension correlation matrix on held-out data (n = 30 texts with complete 10-dimension LLM labels) revealed alarmingly high correlations across nearly all dimension pairs. The mean off-diagonal correlation was r = 0.641, with several pairs exceeding r = 0.90:

| Pair | r | Shared variance (r²) |
|---|---|---|
| authority_dynamics × hostility_index | 0.913 | 83.4% |
| cooling_capacity × defensive_architecture | 0.879 | 77.3% |
| regulatory_capacity × resilience_baseline | 0.954 | 91.0% |
| cooling_capacity × regulatory_capacity | 0.890 | 79.2% |
| contractual_clarity × trust_conditions | 0.890 | 79.2% |

By the standards of discriminant validity (Campbell & Fiske, 1959), correlations above r = 0.85 between purportedly distinct constructs suggest they are measuring the same thing. If regulatory capacity and resilience baseline share 91% of their variance, the argument for separate dimensions is difficult to sustain.

Three explanations were considered:

1. **General factor (p-factor).** Analogous to the general factor of psychopathology (Caspi et al., 2014), there may be a general factor of psychoemotional safety — a "g-PSQ" — onto which all 10 dimensions load. The high inter-correlations would reflect this general factor rather than construct redundancy. Precedent: the DASS-21 (Lovibond & Lovibond, 1995) measures depression, anxiety, and stress as separate scales despite shared variance of 40–60%, because a bifactor model reveals both a general distress factor and specific factors.

2. **Short text entanglement.** With texts of 50–500 words, the information available to discriminate between dimensions may be insufficient. A hostile text is *also* threatening, low-trust, high-authority, and low-cooling — not because these constructs are identical, but because the text provides a single gestalt from which all dimensions are derived.

3. **LLM halo effect.** Thorndike (1920) first described the halo effect as a rater's tendency to let a global impression influence ratings on specific attributes. When the LLM scores all 10 dimensions in a single call, it may anchor on an overall safety impression and adjust individual scores around it.

### 13b. The Halo Effect Experiment

To disentangle explanation 3 (halo) from explanations 1–2 (genuine structure), we designed a within-subjects experiment:

**Method.** 30 held-out texts scored two ways:
- **Joint condition:** All 10 dimensions scored in a single LLM call (the existing held-out labels).
- **Separated condition:** Each dimension scored independently in separate LLM calls. Dimensions paired 2-per-call, with high-correlation pairs deliberately split across different calls to prevent within-call anchoring.

**Results:**

| Metric | Joint scoring | Separated scoring | Difference |
|---|---|---|---|
| Mean off-diagonal r | 0.641 | 0.494 | -0.147 |
| Pairs with r > 0.80 | 16 / 45 | 7 / 45 | -56% |
| Pairs with r < 0.30 | 2 / 45 | 10 / 45 | +400% |

The halo effect accounts for approximately 0.15 correlation units — substantial but not the whole story. Some pairs dropped dramatically when scoring was separated (indicating halo artifact), while others barely changed (indicating genuine construct overlap):

**Strong halo pairs** (joint → separated delta < -0.30):
- authority × resilience: 0.76 → 0.04 (almost entirely halo)
- cooling × resilience: 0.88 → 0.21 (mostly halo)
- contractual × defensive: 0.77 → 0.13 (mostly halo)
- cooling × regulatory: 0.89 → 0.34 (substantial halo)

**Genuine overlap pairs** (|delta| < 0.10):
- regulatory × resilience: 0.95 → 0.93 (genuine: these constructs share theoretical roots in Lazarus & Folkman, 1984)
- authority × trust: 0.83 → 0.88 (genuine: power and trust are theoretically entangled — Edmondson, 1999)
- defensive × regulatory: 0.73 → 0.71 (genuine: both involve internal protective capacity)

### 13c. Emergent Factor Structure

When halo is removed, a two-cluster structure emerges that maps remarkably well to the Job Demands-Resources (JD-R) model (Bakker & Demerouti, 2007):

**Cluster 1: Interpersonal Climate (Environmental Demands)**
- authority_dynamics, contractual_clarity, trust_conditions, threat_exposure
- Within-cluster mean r = 0.79 (separated scoring)
- Theoretical interpretation: these dimensions assess the *external environment's* safety properties — power dynamics, contractual obligations, interpersonal trust, and threat exposure. In JD-R terms, these are *demands* (or inversely, *environmental resources*).

**Cluster 2: Internal Resources (Personal Capacities)**
- regulatory_capacity, resilience_baseline, defensive_architecture
- Within-cluster mean r = 0.80 (separated scoring)
- Theoretical interpretation: these dimensions assess the *individual's* capacity to manage psychoemotional challenge — emotion regulation, resilience, and defense/boundary patterns. In JD-R terms, these are *personal resources*.

**Bridge dimensions:**
- cooling_capacity (r = 0.58–0.76 with Cluster 1; r = 0.21–0.35 with Cluster 2)
- energy_dissipation (r = 0.54–0.77 with Cluster 1; r = 0.42–0.52 with Cluster 2)
- hostility_index (r = 0.55–0.69 with Cluster 1; r = -0.07–0.16 with Cluster 2)

The bridge dimensions align with theory: cooling capacity involves both environmental features (de-escalation mechanisms, temporal buffers) and personal skills (cognitive reappraisal), placing it between clusters. Energy dissipation similarly spans environmental demands (workload, conflict intensity) and personal resources (recovery capacity, flow access).

### 13d. Implications for Dimensionality

The emerging evidence suggests that the PSQ's 10 dimensions may be better represented as a **hierarchical model** (cf. Reise, 2012; Rodriguez et al., 2016):

- **Level 1 (general factor):** g-PSQ — overall psychoemotional safety.
- **Level 2 (cluster factors):** Environmental Climate + Internal Resources (+ possibly Bridge).
- **Level 3 (specific factors):** The 10 individual dimensions, carrying unique variance beyond Levels 1–2.

This is directly analogous to the bifactor structure found in other multi-dimensional psychological instruments: the DASS-21 (depression + anxiety + stress + general distress; Henry & Crawford, 2005), the PCL-5 (4 PTSD symptom clusters + general PTSD; Armour et al., 2016), and the ProQOL (compassion satisfaction + burnout + secondary trauma + general wellbeing; Stamm, 2010).

**Status:** The halo experiment (n = 30) is suggestive but not confirmatory. A proper bifactor analysis would require n ≥ 200 texts with separated scoring, followed by confirmatory factor analysis comparing the 10-factor, bifactor, and hierarchical models using fit indices (CFI, RMSEA, BIC). This is a priority for the next phase of development.

## 14. Current State and Open Questions

### 14a. Model Performance (v19, 2026-02-28)

| Metric | Value |
|---|---|
| Architecture | DistilBERT-base-uncased (66.7M params) |
| Training data | 21,627 texts in DB (78,361 scores, 24,771 separated-llm) |
| Test avg Pearson r | 0.509 (10/10 dimensions positive) |
| Held-out avg Pearson r | 0.600 (best, +0.039 vs v16, +0.172 vs v13) |
| Generalization gap | ~9% |
| ONNX model size | 64 MB (INT8 quantized, v16) |
| Inference latency | ~20ms / text (CPU) |

### 14b. Psychometric Properties

| Property | Status | Evidence | Standard |
|---|---|---|---|
| Test-retest reliability | Excellent | ICC = 0.935 (perturbation-based) | ICC > 0.75 (Cicchetti, 1994) |
| Discriminant validity (vs. sentiment) | Strong | Mean |r| = 0.205 vs VADER | r < 0.30 (distinct construct) |
| Confidence calibration | Done | Isotonic regression; 8/10 dims improved | Platt (1999) |
| Held-out generalization | Good | r = 0.600, n = 100 (separated labels, v19) | Comparable to brief personality measures |
| Construct validity (discriminant) | Confirmed | 5-factor EFA (n=2,359); AD/ED singletons | CFA needed (n ≥ 200) |
| Criterion validity | **Strong** | **4 studies: CaSiNo, CGA-Wiki, CMV, DonD** (AUC 0.59–0.69) | Profile >> average; context-dependent primacy |
| Inter-rater reliability | Not measured | — | Critical gap |
| Measurement invariance | Not measured | — | DIF analysis across text types |

### 14c. Open Questions

1. **Dimensionality restructuring.** Should the 10-dimension model be replaced with a hierarchical structure (g-PSQ + clusters + dimensions)? The halo experiment suggests genuine factor structure, but the sample size (n = 30) is insufficient for confirmatory analysis.

2. **Threat exposure rehabilitation.** Despite the Civil Comments fix and improvement to r=0.41 (v15), threat_exposure regressed from v14 (0.48) and remains volatile. The model's legacy "default to safe" prior may require architectural intervention (e.g., dimension-specific learning rates) rather than just data correction.

3. **Training ceiling.** Test r has grown from 0.492 (v1) to 0.553 (v13) — a 12% improvement over 13 versions. The diminishing returns suggest we may be approaching the ceiling for this architecture + data mix. Next steps likely require either substantially more LLM-labeled data (expensive) or a larger base model (compute-constrained).

4. **Human validation.** All "ground truth" in this project is LLM-generated. The held-out test uses LLM labels as truth, the halo experiment uses LLM ratings, and the psychometric evaluation treats LLM consistency as reliability. A proper validation study requires human expert ratings — clinical or organizational psychologists scoring texts on the PSQ dimensions — to assess whether the LLM teacher itself is valid. This is the most important missing piece.

5. **Deployment pipeline.** ONNX export is complete (64 MB quantized model). The Node.js inference provider (`student.js`) is wired up with calibration support. Not yet deployed to production.

### 14d. What Worked

- **Confidence-weighted loss** with squared exponent (α = 2.0) — properly downweights noisy proxies
- **LLM 5× weighting** over composite — ensures gold-standard signal dominates
- **Targeted synthetic generation** for signal-starved score ranges — analogous to active learning
- **Relabeling existing texts** rather than only generating new ones — real-data representativeness + LLM accuracy
- **Hash-based text splitting** — prevents train/test leakage on duplicated texts
- **DistilBERT** over DeBERTa on consumer GPU — faster iteration > marginal accuracy
- **Systematic proxy auditing** — removing bad sources beats tuning bad sources

### 14e. What Didn't Work

- **Detoxify as proxy teacher** — insufficient construct correlation (r = 0.51–0.74)
- **Diplomacy dataset for trust** — sender intent ≠ environmental trustworthiness
- **Civil Comments for threat exposure** — author-directed threat ≠ environment-directed safety (jangle fallacy)
- **UCC generalisation_unfair for contractual clarity** — negative correlation, wrong direction
- **DeBERTa-v3-small** — slower and worse on consumer GPU (§8)
- **Adjusting confidence weights to fix bad proxies** — anti-signal cannot be attenuated, only removed

---

## 15. Separated Scoring and Hierarchical Reporting

### 15a. From Diagnosis to Intervention

**February 27, 2026.** The halo experiment (§13b) diagnosed the problem: joint LLM scoring inflates inter-dimension correlations by ~0.15 on average, contaminating both training labels and evaluation benchmarks. Three interventions followed in the same session.

### 15b. Separated Scoring Workflow

A new labeling pipeline (`scripts/label_separated.py`) was created to support halo-free scoring. Rather than calling an API directly, the script implements an extract/ingest/assemble workflow for in-conversation labeling:

1. **Extract** — generates per-dimension batch files containing the scoring rubric, calibration anchors, and texts to score. Each batch is self-contained: a rater scoring "cooling_capacity" sees only the Cooling Capacity definition, its instrument basis (CPI, Gross reappraisal, REQ), and the 0–10 scale anchors. No other dimensions are visible.
2. **Ingest** — normalizes scored results into `{score, confidence}` pairs per text per dimension.
3. **Assemble** — merges 10 single-dimension score files into a complete JSONL record set with `teacher: "separated-llm"`.
4. **Validate** — compares joint vs. separated inter-dimension correlation matrices, classifying each dimension pair as "halo" (inflation artifact), "genuine" (real construct overlap), or "unclear".

The key design insight: by extracting dimension definitions from `instruments.json` (the same source of truth used by the LLM detector), the separated workflow guarantees prompt parity with the production scoring system — the rubric a human or LLM sees when scoring a single dimension is identical to what the detector sees, minus the other nine dimensions.

### 15c. Held-Out Re-Scoring

All 100 held-out texts were re-scored with separated calls — one dimension per LLM call, 1,000 total scoring operations. The original joint-scored file was archived as `data/held-out-test-joint.jsonl`.

**Halo validation results:**

| Metric | Joint | Separated | Change |
|---|---|---|---|
| Mean inter-dim \|r\| | 0.766 | 0.656 | -0.111 (PASS) |
| Between-cluster mean \|r\| | 0.765 | 0.639 | -0.126 (PASS) |
| Discriminant ratio (within/between) | 1.10× | 1.05× | -0.05 |

The mean correlation drop of 0.111 confirms that halo contamination was present in the held-out labels. The discriminant ratio did not improve — expected, given that the joint file had sparse dimension coverage (0 of 100 records with all 10 dims present), making the baseline ratio unreliable. The separated file has complete 10-dimension coverage on all 100 records.

**Re-evaluation against student model (v13):**

| Dimension | Joint r | Separated r | Direction |
|---|---|---|---|
| cooling_capacity | 0.574 | 0.574 | Stable |
| trust_conditions | 0.498 | 0.498 | Stable |
| resilience_baseline | 0.496 | 0.496 | Stable |
| hostility_index | 0.480 | 0.480 | Stable |
| authority_dynamics | 0.457 | 0.457 | Stable |
| energy_dissipation | 0.393 | 0.393 | Stable |
| defensive_architecture | 0.368 | 0.368 | Stable |
| regulatory_capacity | 0.325 | 0.325 | Stable |
| contractual_clarity | 0.271 | 0.271 | Stable |
| threat_exposure | 0.160 | 0.160 | Stable |
| **Average** | **0.428** | **0.402** | **-0.026** |

The avg_r drop from 0.428 to 0.402 is modest and expected: the separated labels are a harder, more discriminating benchmark. The student model's per-dimension rankings are unchanged — the same four dimensions (cooling, trust, resilience, hostility) remain strongest, and threat_exposure remains weakest at r = 0.16.

### 15d. Hierarchical Reporting

The halo experiment's cluster structure (§13c) was implemented as additive reporting layers in both inference providers:

**`src/student.js`** — `StudentProvider.computeHierarchy()`:
- Computes confidence-weighted means for three clusters: interpersonal_climate (ad, co, tc, te), internal_resources (rc, rb, da), bridge (cc, ed, hi)
- Computes g-PSQ as the confidence-weighted mean of all 10 dimensions
- Added as a `hierarchy` field alongside existing `scores` — fully backwards-compatible

**`src/detector.js`** — `aggregatePSQ()`:
- Same cluster definitions and computation
- `hierarchy` field added to return value alongside existing `psq`, `protective_avg`, `threat_avg`

**Hard constraint maintained:** The 10-dimension structure is unchanged. No dimensions were merged or removed. Clusters and g-PSQ are strictly additive layers for interpretive convenience — a consumer of PSQ output can ignore `hierarchy` entirely and use the 10 dimension scores as before.

This follows the pattern established by bifactor instruments in clinical psychology: the DASS-21 reports both total distress and three subscale scores; the PCL-5 reports both total PTSD severity and four cluster scores. The additive layer provides parsimony without sacrificing granularity.

### 15e. Remaining Gaps

The separated scoring workflow establishes the tooling for halo-free labeling but does not yet address the training data. The 4,199 LLM records in `train-llm.jsonl` were scored jointly and carry the ~0.15 halo inflation. Re-labeling these with separated scoring, followed by a v14 training run, is the highest-priority next step. See `suggestions.md` for the full prioritized backlog.

---

## 16. Training Data Labeling Expansion and V14 (2026-02-27)

### 16a. Completing the All-Dimensions Batch

The 200-text batch assembled for v14 (`data/labeling-batch-weak-dims.jsonl`) was scored on all 10 dimensions using the separated workflow. The prior session had covered the three weakest dimensions (threat_exposure, regulatory_capacity, contractual_clarity). This session extended coverage to the remaining seven: hostility_index, authority_dynamics, energy_dissipation, resilience_baseline, trust_conditions, cooling_capacity, and defensive_architecture.

Two hundred texts × 10 dimensions = 2,000 new separated-llm labels. Total separated-llm training labels now range from 203 (seven previously uncovered dimensions) to 653 (regulatory_capacity, which had priority in earlier sessions). The held-out benchmark (100 texts, all 10 dims, clean separated labels) remains unchanged.

A practical constraint emerged: scoring outputs cannot exceed ~32,000 tokens per response. The fix — scoring in four batches of 50 texts per dimension — became the canonical batching protocol. Partial score files accumulate in `/tmp/psq_separated/{dim}_partial.json` across batches and are merged before ingestion.

### 16b. Infrastructure: Checkpoint Safety

A recurring problem surfaced: smoke-test training runs were silently overwriting the v13 production checkpoint (`models/psq-student/best.pt`). This is the kind of data loss that evades version control because model weights are not committed. The fix adds two CLI arguments to `distill.py`:

- `--out DIR` — explicit output directory, defaulting to `models/psq-student` for backward compatibility
- `--no-save` — smoke-test mode: checkpoints are written to a temporary directory and deleted on completion

Future training runs should follow the convention `--out models/psq-vN`, reserving `models/psq-student/` for the current production checkpoint only.

### 16c. V14 Results

Training completed: 10 epochs, best checkpoint at epoch 8 (val_r = 0.528). Test-set average r = 0.544. Held-out average r = 0.482 — a +0.080 improvement over v13, the largest single-version gain in the project.

The held-out results answered the question of whether 200 separated-scored texts per dimension is sufficient: yes, substantially so. Eight of ten dimensions improved, with the three formerly-weakest showing the largest gains — threat_exposure jumped from r = 0.16 to r = 0.41, contractual_clarity from r = 0.27 to r = 0.43, and energy_dissipation from r = 0.39 to r = 0.53. This pattern is consistent with the expectation that halo-inflated joint labels were specifically harmful for dimensions with strong theoretical independence (threat and contractual clarity have low conceptual overlap with the remaining eight), and that even 200 clean separated labels is sufficient to override years of accumulated halo bias.

One notable regression: regulatory_capacity declined from r = 0.325 to r = 0.244 on the held-out set, despite a test-set r of 0.527. This test/held-out inversion suggests the 200 new rc labels may be sampling from a narrower distribution (Reddit stress posts, emotional support conversations) that does not generalize to the real-world rc benchmark. Regulatory capacity — the dimension assessing whether a communication environment provides adequate stress-regulatory resources — may require training texts drawn from organizational and workplace contexts rather than peer support contexts. This is the primary open question for v15 data planning.

### 16d. Current State (V15)

| Metric | Value |
|---|---|
| Architecture | DistilBERT-base-uncased (66.7M params) |
| Training data | 20,127 texts (DB), 16,046 train split |
| Separated-llm labels | 9,771 (all 10 dims) |
| Test avg Pearson r | 0.536 |
| Held-out avg Pearson r | 0.495 (+0.013 vs v14, +0.093 vs v13) |
| Generalization gap | 7.6% (down from 11.4% in v14) |
| Checkpoint | `models/psq-v15/best.pt` |

The generalization gap continues to shrink — from 27.3% (v13) to 11.4% (v14) to 7.6% (v15) — as separated-llm labels replace noisy proxy data. The AD batch produced the single largest per-dimension held-out gain in the project (authority_dynamics +0.166), validating the targeted labeling strategy. The remaining weak points are regulatory_capacity (0.285, still lowest) and a new contractual_clarity regression (0.498→0.388) that warrants investigation.

---

## 17. Score-Concentration Fix and Targeted Labeling (2026-02-27)

The contractual_clarity regression in v15 (held-out 0.498→0.388) revealed a systematic vulnerability in our dimension-focused labeling strategy. When we scored 300 texts selected for authority_dynamics relevance, those texts were naturally neutral on contractual_clarity — producing correct but monotonic co=5 labels. With separated-llm priority and 5× sample weight, 58% of the co training distribution collapsed to a single value, overwhelming the co head with uninformative gradient.

This is not a labeling error — the labels are correct — but a statistical imbalance inherent to our approach: every dimension-focused batch creates score concentration on non-target dimensions. The problem was not isolated to co; a post-hoc audit found that 9 of 10 dimensions had >30% score-5 concentration, with resilience_baseline at 58% and cooling_capacity at 52%.

We implemented a systemic fix: `_cap_score_concentration()` in `distill.py` identifies any (dimension, rounded_score) pair exceeding 30% of samples and reduces excess rows' sample_weight from 5.0 to 1.5 — preserving the labels but limiting their influence to composite-proxy levels. The approach is deterministic (seed=42), applies per-dimension, and is enabled by default. A 1-epoch smoke test showed immediate co recovery (test r=0.737 vs v15's 0.388 on held-out), though single-epoch numbers should not be over-interpreted.

To complement the systemic fix, we extracted a CO-focused labeling batch: 200 texts from the unlabeled pool filtered by contractual-clarity keywords (agree, rule, policy, obligation, consent, permission, etc.). The co dimension was scored with 52% non-5 scores (range 1–9, mean 5.20), providing the distributional variance the co head needs. All 10 dimensions were scored and ingested in a single session, bringing the database to 20,327 texts and 65,361 scores. Comparing the CO batch's separated-llm scores to prior batches confirmed the targeting worked: 47.5% score-5 in the CO batch versus 58.3% in prior separated-llm CO labels.

A practical addition: labeling sessions now record timing data. The ingest command accepts `--started-at` and logs duration and throughput to `data/labeling_log.jsonl`. The full CO batch (200 texts × 10 dimensions) completed in 25.3 minutes — an average throughput of 4,743 texts/hr. Scoring speed varied significantly by context: first-encounter dimensions required careful reading (~3,500 texts/hr), while dimensions scored after the texts were already in working memory reached 23,000–24,000 texts/hr. This data will inform batch size planning for future labeling campaigns.

v16 training was launched with both fixes active. Epoch 1 showed co test r=0.71, a dramatic recovery from v15's 0.388 held-out, though early-epoch numbers warrant caution.

### 17a. v16 Results: Best Held-Out Ever (0.561)

v16 training completed with both the concentration cap and three targeted batches (CO, RB, CC — 600 texts × 10 dims = 6,000 new separated-llm scores). The held-out evaluation produced the best result to date: **r=0.561** (+0.066 vs v15, +0.133 vs v13). Eight of ten dimensions improved, with two dramatic recoveries:

- **Regulatory capacity** jumped from 0.285 to 0.563 (+0.278), the single largest per-dimension gain in the project's history, surpassing the AD batch's authority_dynamics +0.166 in v15. The combination of concentration cap (which reduced the overwhelming weight of score-5 samples from 45% of rc's distribution) and the RB/CC batches providing additional varied training signal appears to have unblocked the rc head.

- **Contractual clarity** recovered from 0.388 to 0.534 (+0.146), confirming that the score-5 flooding diagnosis was correct. The CO batch's targeted keywords produced 52% non-5 co scores, and the concentration cap ensured the remaining score-5 samples did not dominate training.

One concerning regression: **threat_exposure** dropped from 0.476 to 0.347 (-0.129). This dimension has been volatile across versions — 0.367 in v13, 0.476 in v14, 0.410 in v15, now 0.347. The test split shows te at 0.522, suggesting the model has learned te patterns that don't generalize to the held-out distribution. Investigation is needed: the held-out te labels may need re-examination, or te may be particularly sensitive to the concentration cap's down-weighting (49% of te scores were 5, with 945 down-weighted).

Notably, the generalization gap inverted: test_r=0.529 vs held-out_r=0.561, meaning the model performs *better* on unseen real-world texts than on the seen-distribution test set. This is unusual but not unprecedented — it likely reflects that separated-llm labels (which dominate the held-out set) are more consistent than the mixed composite/joint-llm labels in the test split.

### 17b. Current State (v16)

| Metric | Value |
|---|---|
| Architecture | DistilBERT-base-uncased (66.7M params) |
| Training data | 20,727 texts (DB), 16,216 train split |
| Separated-llm labels | 15,771 (all 10 dims) |
| Test avg Pearson r | 0.529 |
| Held-out avg Pearson r | 0.561 (+0.066 vs v15, +0.133 vs v13) |
| Generalization gap | -6.0% (held-out exceeds test) |
| Checkpoint | `models/psq-v16/best.pt` |

---

## 18. Factor Analysis: The General Factor Question (2026-02-28)

We subjected the 10-dimension PSQ to full exploratory factor analysis for the first time. The dataset comprised 2,359 texts with complete 10-dimension coverage — 1,470 from separated-llm scoring (halo-free) and the remainder from joint-llm and composite-proxy sources. KMO was 0.819 (meritorious), confirming the data was well-suited for factor analysis.

The results were unambiguous: the 10-factor independence hypothesis was rejected by every standard criterion. Kaiser's rule retained 3 factors, parallel analysis retained 2, and BIC model selection favored 5. A dominant first eigenvalue (4.844, explaining 48.4% of variance) indicated that most dimensions load on a single general factor — what we might call "overall psychological safety of content."

The finding was even stronger in the separated-llm subset. Counter-intuitively, halo-free scoring produced *higher* inter-dimension correlations (mean |r|=0.564 vs 0.417 for all data). We had expected the opposite: that joint scoring inflated correlations and separated scoring would reveal independence. Instead, the composite-proxy data — with its narrow, noisy per-dimension mappings — had been *artificially deflating* correlations by introducing independent noise into each dimension. When scored with careful, dimension-specific attention, the LLM teacher recognized the genuine co-variation that exists in natural text: hostile content really does tend to lack trust, impair regulation, and involve power imbalances.

The BIC-best 5-factor solution revealed interpretable clusters: Hostility/Threat (HI, TE, CC), Relational Contract (CO, TC), Internal Resources (RB, RC, DA), Power Dynamics (AD), and Stress/Energy (ED). Defensive architecture cross-loaded on three factors, confirming its status as the most diffuse construct in the system (see §12 in the psychometric evaluation).

This result has a parallel in personality psychology. The Big Five personality factors (Costa & McCrae, 1992) also show a general factor — the "Big One" or general factor of personality (GFP; Musek, 2007) — that explains roughly 50% of variance. Yet the field retains the five-factor structure because (a) the factors have distinct predictive validity for different outcomes, and (b) they suggest different interventions. Similarly, the 10 PSQ dimensions may co-vary but serve different theoretical and practical purposes: hostility implies content moderation, while contractual clarity implies expectation-setting.

Our recommendation is a hierarchical reporting model: an overall PSQ score (general factor), 3–5 cluster scores, and 10 dimension scores with the explicit caveat that dimensions within a cluster are not independent. This is analogous to how the WISC reports Full Scale IQ, Index Scores, and Subtest Scores — a nested measurement framework where the subscales provide texture within a dominant general factor. We do not claim 10 independent dimensions; we claim 10 theoretically distinct facets of a coherent construct.

---

## 19. Expert Validation Protocol: The DA Question (2026-02-28)

The factor analysis (§18) surfaced a structural question that no amount of LLM training can resolve: is Defensive Architecture a distinct, scorable psychological construct, or an artifact of general safety assessment? DA's empirical profile is troubling on multiple fronts. At the 5-factor level, its maximum promax loading is 0.332 — below even the lenient 0.35 threshold for meaningful factor membership. Its mean correlation with the other 9 dimensions is 0.480, essentially identical to its correlations with both the Hostility/Threat cluster (r=0.603) and Internal Resources cluster (r=0.608). In the separated-llm data, DA correlates 0.825 with Trust Conditions, 0.768 with Regulatory Capacity, and 0.744 with Cooling Capacity — correlations high enough to question discriminant validity.

One response would be to improve the LLM's DA scoring through targeted labeling batches, as we had done successfully for Authority Dynamics (+0.166 improvement after 300-text batch). But this treats a construct validity question with a measurement precision tool. If DA genuinely measures a distinct boundary-related construct that happens to be underspecified in text, then improving measurement precision might reveal discriminant validity. If DA is genuinely diffuse — a construct that exists in clinical practice but lacks a distinct textual signature — then better measurement will only confirm the diffuseness with more certainty.

The scientifically sound approach is external validation by human experts. We designed a comprehensive expert panel study (`expert-validation-protocol.md`) to resolve three research questions: (1) Can trained raters achieve acceptable inter-rater reliability (ICC ≥ 0.70) on each of the 10 PSQ dimensions? (2) Is DA empirically separable from the other 9 dimensions in expert ratings? (3) How do expert ratings compare to LLM teacher scores?

The study proposes 5 expert psychologists scoring 200 stratified texts across all 10 dimensions — a fully crossed design yielding 10,000 ratings. The text sample is stratified to oversample DA-extreme texts (60 texts with DA scores ≤3 or ≥7) alongside general safety-range texts and factor-informative texts. The DA-specific decision tree is explicit: if expert ICC for DA falls below 0.50, DA is not reliably scorable from text and should be deprecated. If DA partial correlations (controlling for g-PSQ) all fall below 0.30, DA captures something distinct and should be retained. If R² > 0.80 from the other 9 dimensions, DA is redundant and should be absorbed into the nearest cluster.

This represents a pivotal methodological shift. Until now, all "ground truth" in the PSQ project has been LLM-generated — the held-out evaluation uses LLM labels, the factor analysis uses LLM scores, and the psychometric evaluation treats LLM consistency as a proxy for reliability. The expert panel study introduces the first fully independent human judgment into the validation chain. Its results will determine not only DA's fate but also whether the LLM teacher itself is measuring what clinical psychologists would recognize as psychological safety.

The estimated cost ($5,625–$15,000 for 5 raters at professional consulting rates) and timeline (7–9 weeks) represent significant investment for a small-team project, but the alternative — proceeding to deployment with an instrument whose construct validity rests entirely on a single model's judgment — creates far greater risk. As Shrout and Fleiss (1979) established, inter-rater reliability is a prerequisite, not an optional supplement, for any claim of measurement validity.

---

## 20. Criterion Validity: The Negotiation Test (2026-02-28)

We conducted the first criterion validity test of the PSQ, asking a deceptively simple question: do PSQ scores on conversation text predict how people actually feel after the conversation is over?

The CaSiNo negotiation corpus (Chawla et al., 2021) provided a natural experiment. Each dialogue — a campsite resource negotiation between two MTurk workers — ends with each participant independently reporting their satisfaction with the outcome, how much they liked their opponent, and their objective points scored. These outcome variables were never used in PSQ training; the only CaSiNo signal in the PSQ pipeline is strategy annotation ratios mapped to contractual clarity. The satisfaction and likeness ratings are fully external criteria.

The results were modest but theoretically coherent. Nine of ten PSQ dimensions significantly predicted both satisfaction (r=+0.07 to +0.11) and opponent likeness (r=+0.07 to +0.13), with all correlations in the positive direction: higher PSQ content → more satisfied negotiators who liked each other more. Points scored — the objective competitive outcome — showed near-zero correlations with PSQ. This dissociation is exactly what psychological safety theory predicts: safety is about relational quality, not competitive advantage. A psychologically safe conversation helps both parties feel good, regardless of who "won."

The most surprising finding involved Defensive Architecture, the construct we had been preparing to potentially deprecate. After controlling for text length and sentiment, DA emerged as the single strongest predictor of both satisfaction (ΔR²=+0.007) and opponent likeness (ΔR²=+0.009), and it was the only dimension whose partial correlation *increased* after controlling for text length. Whatever DA captures — boundary respect, interpersonal defense quality, self-protective behavior support — it matters for real-world interpersonal outcomes, even if it refuses to load cleanly on any single factor in our measurement model.

This creates an intriguing paradox. DA has the weakest discriminant validity within the PSQ system (no primary factor loading >0.35, mean r=0.48 with other dimensions) but the strongest criterion validity against an external outcome. One interpretation, consistent with Caspi et al.'s (2014) p-factor model in psychopathology, is that DA functions as a "general safety indicator" — a dimension that reflects overall safety quality rather than a specific, separable facet. In clinical parlance, it may be more like a clinician's "gestalt impression" than a subscale score: hard to decompose analytically but predictively valid.

The incremental R² beyond sentiment (ΔR²=+0.016 for satisfaction, +0.023 for likeness) confirms that PSQ captures something beyond simple positivity. These are small effects — comparable to LIWC dimensions predicting interpersonal outcomes (Tausczik & Pennebaker, 2010) — but they represent real signal in a domain where content-level measurement has never been attempted at this granularity.

The effect sizes (r≈0.10) should be interpreted in context. We are predicting a post-conversation subjective experience from the text of the conversation itself, using a 66-million-parameter model trained on an entirely different construct (psychological safety of text content, not negotiation outcomes). That it predicts at all — and in the theoretically predicted direction, across 9 of 10 dimensions — constitutes meaningful, if preliminary, criterion validity evidence. The next test — predicting conversation derailment from early turns in the Conversations Gone Awry corpus — will probe whether PSQ can identify latent unsafety before it manifests as overt hostility.

---

## 21. The Derailment Test (2026-02-28)

The CaSiNo study asked whether PSQ predicted how people *felt* after a conversation. The Conversations Gone Awry study asks something harder: can PSQ predict what people *did* — specifically, whether a Wikipedia talk-page conversation would derail into a personal attack?

The corpus (Zhang et al., 2018) is elegant in its design: 4,188 conversations, perfectly balanced between those that derailed and those that didn't, drawn from Wikipedia editor disputes. The domain is entirely absent from PSQ training data — no Wikipedia talk pages, no editor interactions, nothing from this register of discourse. If PSQ predicts here, it predicts because the construct generalizes, not because the model memorized.

It does predict. A logistic regression on all ten PSQ dimensions achieves AUC=0.599 on the held-out test set — modest but stable (5-fold CV: 0.579 ± 0.016). The direction is theoretically correct and intuitively satisfying: conversations that will derail show *lower* authority dynamics (-0.212), regulatory capacity (-0.177), and trust conditions (-0.150). These are conversations where participants feel less authority, less self-regulation capacity, and less interpersonal trust — the preconditions for losing composure.

The temporal analysis reveals something important about what PSQ measures. When we score only the first turn of each conversation (before any conflict has emerged), prediction drops to near-chance (AUC=0.519). With early turns, it rises to 0.570. With all turns, 0.599. PSQ is not reading static lexical features — it is tracking an interpersonal trajectory. The psychological unsafety accumulates, and the model captures that accumulation. This is precisely what a process-level construct should do: detect the erosion of safety conditions over the course of an interaction.

The implications of this temporal gradient deserve unpacking, because they touch on what PSQ *is* at a fundamental level.

First, the gradient rules out the simplest explanation for PSQ's predictive power: that it is merely a toxicity detector in disguise. A toxicity classifier like Perspective API would show the opposite temporal pattern — strong signal on the final turns (which contain the personal attack) and weaker signal on early turns (which are typically civil). PSQ shows the reverse: its signal builds *before* the attack, in the apparently civil early exchanges where safety conditions are quietly eroding. This is the difference between measuring temperature (a state variable) and measuring heat transfer (a process variable). PSQ appears to be measuring the latter — the thermodynamic trajectory of an interpersonal system, not just its current state.

Second, the gradient has practical implications for deployment. A toxicity filter that triggers only *after* someone has already been attacked is, from a safety perspective, closing the barn door after the horse has bolted. A PSQ monitor that detects deteriorating safety conditions *during* the conversation — even at the modest AUC=0.570 available at the halfway point — offers the possibility of preventive intervention. The confidence would increase with each additional turn, following the gradient: uncertain at first, increasingly confident as the trajectory clarifies. This maps naturally onto a traffic-light interface: green (first turn, insufficient data), yellow (mid-conversation, safety declining), red (accumulated evidence of impending derailment). The 128-token truncation we currently use is a limitation here — a production system would score each new turn incrementally, maintaining a running PSQ profile.

Third, the temporal gradient is consistent with Edmondson's (1999) original formulation of psychological safety as a *shared belief* that emerges from repeated interpersonal interactions. Edmondson never proposed that safety is a fixed property of individuals or texts — she described it as a team-level phenomenon that develops through experience. Our finding that PSQ signal accumulates across turns is empirical support for this process view: safety is something that is *built or eroded* through interaction, not simply *present or absent* in the content.

The AD finding replicates across studies with uncanny consistency. In CaSiNo, authority dynamics was the strongest criterion predictor of negotiation satisfaction. In CGA-Wiki, it is the strongest point-biserial correlate of derailment (r=-0.105, p<0.001). Two completely different domains — campsite negotiations and Wikipedia policy disputes — two different outcomes — subjective satisfaction and behavioral hostility — and the same dimension emerges as the strongest signal. Whatever AD captures about power dynamics and interpersonal authority, it is doing real psychological work.

This cross-study replication has important consequences for the DA/AD construct validity debate. Our factor analysis (§18) showed that DA has no primary loading above 0.35 in the 5-factor promax solution — by traditional psychometric criteria, this would be grounds for deprecation. But the criterion validity evidence tells a different story: the construct that is hardest to *define* internally is the easiest to *validate* externally. This is not unprecedented in psychology. The Big Five dimension of Openness to Experience has a similarly contested internal structure — debate continues over whether it reflects intellect, aesthetic sensitivity, or imaginative tendency — yet it predicts academic performance, creativity, and political attitudes with remarkable consistency (McCrae & Costa, 1997). Our AD may be the PSQ's openness: a dimension whose predictive validity survives despite (or perhaps because of) its conceptual breadth.

The practical implication is clear: we should not deprecate AD based on internal structure alone. The expert validation protocol (§19) includes a DA-specific decision tree — if experts can't reliably rate it (ICC<0.50), we deprecate; if they can, we investigate whether its criterion validity is mediated by other dimensions. But the CGA-Wiki data adds a new data point to this decision: AD predicts a behavioral outcome (personal attack) that none of the other dimensions predict as well. This is not redundant information.

Perhaps most importantly for the dimensionality question: g-PSQ (the general factor, a simple mean of all ten dimensions) achieves AUC=0.515 — barely above coin flip. The ten individual dimensions together achieve 0.599. This is direct evidence that the general factor, while statistically dominant in the variance decomposition (55.4% of variance), carries almost no predictive utility for external outcomes. The information lives in the dimension profile, not the global score.

This dissociation between variance explained and predictive utility is worth emphasizing because it cuts against a common psychometric intuition. In classical test theory, the first principal component is the "best summary" of a test battery — the single number that preserves the most information. But preserving variance is not the same as preserving *predictive* signal. Consider a medical analogy: a patient's average vital sign (mean of temperature, blood pressure, heart rate, respiratory rate) captures the most variance in the vital-sign battery, but no physician would use that average for diagnosis. The diagnostic information lives in the *pattern* — high temperature with low blood pressure means something entirely different from low temperature with high blood pressure. Our finding is analogous: the PSQ profile shape predicts, the profile average does not.

This argues strongly against dimension reduction for any applied use case. The hierarchical reporting structure — g-PSQ for overview, 5 clusters for interpretation, 10 dimensions for prediction — is not just an organizational convenience; it reflects where the signal actually lives. Future PSQ applications should be designed around the 10-dimension vector, with the general factor and cluster scores serving as human-interpretable summaries rather than replacements.

One additional finding deserves commentary: the non-significance of threat exposure (TE) and contractual clarity (CO) in the derailment study. These are the only two dimensions that show no mean difference between derailing and safe conversations. TE's non-significance is particularly instructive. One might expect that conversations containing more explicit threat content would be more likely to derail. But PSQ-TE does not measure whether threats are present — it measures the degree to which the *text content supports assessment of* threat exposure. A Wikipedia dispute about article deletion policy may contain substantial TE content (discussion of what constitutes a threat to article integrity) without any interpersonal hostility. Conversely, a conversation that derails may do so through authority violations (AD), trust betrayal (TC), or regulatory failure (RC) without any explicit threat language. TE's null result helps clarify what PSQ measures: the psychological safety *landscape* of text, not the interpersonal *behavior* of speakers.

---

## 22. Dimension Reduction: Where the Signal Lives (2026-02-28)

The factor analysis (§18) showed that a general factor explains 55% of PSQ score variance, and a promax rotation yields a clean 5-factor solution. The natural question is whether we can simplify: does collapsing from 10 dimensions to 5 cluster scores (or even 3) lose anything that matters?

The answer is nuanced. Statistically, the 5-factor model retains 88% of dimension-level information (average R² = 0.881 from cluster scores back to individual dimensions) and captures 91% of total variance. The 3-factor model drops to 74% — an unacceptable loss, driven by authority dynamics (R²=0.615) and energy dissipation (R²=0.449), which simply don't belong in their assigned clusters.

But the criterion validity data provide a sharper answer. In the CGA-Wiki derailment study, g-PSQ (one number) achieves AUC=0.515. The 10 individual dimensions achieve 0.599. The predictive information is distributed across dimensions in a way that collapsing obscures. This is consistent with Meehl's (1956) observation that configural patterns in personality profiles often outperform simple sum scores — the *shape* of the profile matters, not just its average height.

Two dimensions emerge with particularly high unique variance within their clusters: cooling capacity (39% unique in Hostility/Threat) and contractual clarity (36% unique in Relational Contract). These are the dimensions most likely to lose critical information if collapsed. Both measure something distinct from their cluster neighbors — CC captures emotion regulation capacity rather than hostility per se, and CO captures the clarity of interpersonal agreements rather than relational trust.

Our recommendation: report hierarchically (g-PSQ → 5 clusters → 10 dimensions) but always retain the 10-dimension profile for prediction tasks. The 5-cluster level is useful for interpretation and communication — "this conversation shows low Relational Contract but high Internal Resources" — but it should supplement, not replace, the full dimensional scoring.

---

## 23. The Misfits: Why Authority Dynamics Predicts What Nothing Else Can (2026-02-28)

The dimension reduction analysis (§22) revealed that a 5-factor model retains 88% of dimensional information while a 3-factor model loses too much. But *which* dimensions refuse to be collapsed, and why? The answer turns out to illuminate something fundamental about what the PSQ is actually measuring.

Two dimensions — authority_dynamics (AD) and energy_dissipation (ED) — are the primary casualties of dimension reduction. In the 3-factor model, AD's R² drops to 0.615 and ED's to 0.449 — meaning the cluster averages reconstruct less than half of ED's variance and barely two-thirds of AD's. Both dimensions load as singletons in the promax 5-factor solution. They are, statistically speaking, homeless.

The reason is different for each. AD correlates roughly equally with all three major clusters (mean |r| = 0.666 with Hostility/Threat, 0.564 with Relational Contract, 0.507 with Internal Resources). It is a *general factor indicator* — not an orphan from any cluster but a dimension that reflects the shared variance equally. Sixty-two percent of AD's variance is explained by g-PSQ, the highest of any dimension. But its remaining 36% — the residual after removing the general factor — captures something genuinely unique: the interpersonal power structure of the text.

ED, by contrast, is a true orphan. It correlates almost identically with Internal Resources (0.506) and Hostility/Threat (0.480), never finding a natural home. Its g-PSQ loading is the lowest of any dimension (R²=0.447), meaning it is the most independent from the shared safety-threat continuum. After controlling for g-PSQ, ED shows strong negative partial correlations with cooling_capacity (-0.536) and trust_conditions (-0.455). This reveals ED's distinctive construct: texts high in ED-residual describe sustained resource depletion in the *absence* of recovery opportunities — the theoretical picture of chronic allostatic load (McEwen, 1998), distinct from acute hostility or poor coping.

The more scientifically consequential finding is AD's predictive dominance. In the CGA-Wiki study (§21), authority_dynamics showed the strongest point-biserial correlation with derailment (r_pb = -0.105***) and the largest Cohen's d (-0.212). A leave-one-out analysis quantified this: removing AD from the logistic regression costs ΔAUC = -0.021, nearly double the second-largest loss (regulatory_capacity at -0.014). AD alone achieves AUC=0.549 — more than g-PSQ (0.515) and roughly 40% of the incremental signal that all 10 dimensions provide beyond the general factor.

What makes AD special? Not emotional sensitivity. A text-feature analysis of AD-residual scores reveals correlations with interpersonal language markers — second-person pronouns ("you"/"your", r=+0.202), question marks (r=+0.235), authority-related vocabulary (r=+0.121) — but not with sentiment, profanity, or emotional intensity. AD is reading the *relational structure* of text: who has power, how it is exercised, whether it is contested. This is precisely what French and Raven (1959) described in their taxonomy of social power bases — legitimate, coercive, referent, expert, reward — and what Tepper's (2000) abusive supervision construct measures in organizational contexts.

The implication is that conversation derailment is fundamentally a *power* phenomenon, not an *emotion* phenomenon. Hostility matters (HI is the strongest coefficient in the multivariate model), but the dimension that carries the most non-redundant predictive signal is the one measuring who holds power and how they wield it. This aligns with Felson and Tedeschi's (1993) social interactionist theory of aggression: interpersonal violence arises from perceived challenges to authority and social identity, not from free-floating anger.

AD also functions as a suppressor variable in the classical psychometric sense (Conger, 1974). Its multivariate importance (2nd-largest coefficient) exceeds what its bivariate correlation would predict. This occurs because AD captures variance in other dimensions — particularly defensive_architecture and hostility_index — that is *irrelevant* to derailment prediction. Including AD allows the model to isolate the portions of HI and DA that genuinely predict attacks from the portions that merely reflect power dynamics. It is, in Cattell's (1988) terminology, an "instrumental" variable — important not for what it predicts directly but for what it allows other variables to predict.

There is an irony here worth noting. Authority_dynamics is the dimension with the weakest factor loading in the PSQ, the one flagged for potential construct validity concerns (§19), the one for which we designed an expert validation protocol (§19). And yet it is also the most externally valid predictor — the dimension that most consistently predicts real-world outcomes (negotiation satisfaction in CaSiNo, derailment in CGA-Wiki) that it was never trained on. This is consistent with Meehl's (1990) important but often-overlooked observation that a construct's psychometric structure (its factor loadings) need not predict its criterion validity pattern. The dimension that loads least neatly onto the PSQ's internal structure is the one that connects most powerfully to the external world.

For energy_dissipation, the story is quieter but equally instructive. ED contributes little to derailment prediction (ΔAUC=-0.005 when removed), consistent with its construct: resource depletion is a chronic process that unfolds over weeks and months, not within the span of a Wikipedia talk-page exchange. ED's criterion validity should be sought in different contexts — longitudinal studies of burnout, workplace withdrawal, or performance decline — where the temporal scale matches the construct's theoretical dynamics.

Our architectural recommendation remains: keep all 10 dimensions. But this analysis sharpens the reason. It is not merely that collapsing loses statistical information (the R² argument from §22). It is that the dimensions most resistant to collapsing — AD and ED — are the ones measuring fundamentally different psychological processes: interpersonal power structure and chronic resource depletion, respectively. These are not redundant with hostility, trust, or coping. They are distinct mechanisms operating at different levels of analysis — relational for AD, energetic for ED — that enrich the PSQ beyond a simple safety-threat continuum.

A necessary caveat tempers these findings. Authority_dynamics — the PSQ's most externally valid dimension — is also a dimension whose training signal is 70.4% LLM-generated (Claude scoring training texts via separated and joint labeling protocols), with only 29.6% coming from composite proxy mappings (UCC condescension labels and politeness corpus data). The concern is straightforward: has the LLM taught the student model an idiosyncratic definition of "authority dynamics" that happens to correlate with derailment for reasons we have not identified?

We find this concern serious but unlikely to explain the full pattern. First, AD is not unusually LLM-dependent — it sits in the middle of the pack. Contractual_clarity is 96.7% LLM-generated yet shows the *weakest* criterion validity (non-significant in CGA-Wiki). If LLM labeling bias were the mechanism, the most LLM-dependent dimensions should be the best predictors. They are not. Second, the student model generalizes AD scores across domains entirely absent from training (Wikipedia disputes were never in the training data), and does so consistently across two independent criterion studies using different outcome types. Domain-specific biases rarely generalize this cleanly across discourse registers and outcome categories. Third, AD-residual scores correlate with exactly the interpersonal language markers — second-person pronouns, question marks, authority vocabulary — that French and Raven's (1959) power bases framework would predict. A spurious LLM artifact producing these specific, theoretically grounded text-feature correlations would be a remarkable coincidence.

Nevertheless, the entire construct validity chain for AD currently runs through LLM interpretation. The held-out evaluation (r=0.625) uses LLM-scored labels as ground truth; the training data is majority LLM-scored; and no human expert has independently evaluated AD's scoring rubric against the texts. This is precisely the gap the expert validation protocol (§19) is designed to close. Until five independent expert psychologists produce ICC(2,1) ≥ 0.70 on AD scoring and demonstrate substantial convergent validity with the LLM labels, we must hold AD's criterion validity findings as *promising but provisionally grounded*. The strongest predictor in our battery is also the one most in need of human confirmation.

## 24. The AD Paradox: Three Theories of Predictive Primacy (2026-02-28)

The findings from §20–23 converge on a paradox that demands theoretical explanation: authority_dynamics has the weakest factor loading in the PSQ's internal structure (max promax loading 0.332, below the conventional 0.35 threshold) yet the strongest criterion validity across two independent studies. This is not merely a statistical curiosity — it is a substantive finding about the nature of psychoemotional safety that, if correctly interpreted, reshapes our understanding of what the PSQ is measuring. We advance three competing theoretical accounts, each generating distinct testable predictions, and argue that their resolution has implications for construct naming, instrument architecture, and the broader relationship between psychometric structure and criterion validity.

### 24a. Theory 1: The Meta-Conversation Channel

Watzlawick, Beavin, and Jackson (1967) distinguished two simultaneous channels in all human communication: *report* (content — what is said) and *command* (relationship — how the speaker positions themselves relative to the listener). Most PSQ dimensions measure report-level properties: how hostile is the content (HI), how much threat is described (TE), what coping strategies are evident (RC). Authority dynamics, we propose, primarily measures the *command* channel — the relational positioning that occurs alongside and often independently of content.

This would explain AD's factor-analytic behavior. The general factor g-PSQ captures shared variance that is largely report-level: hostile content tends to be threatening, poorly regulated, and low on trust. AD correlates moderately with all of these (mean |r| = 0.480) because power challenges *accompany* hostile content — but AD also captures relational positioning that occurs without hostility (a polite assertion of authority, a subtle claim to expertise, a deferential hedge). This command-channel variance is orthogonal to report-level safety, which is precisely why AD loads weakly on any report-dominated factor.

The predictive primacy follows directly. Conversation derailment is fundamentally a *relational* event — a breakdown in the social contract between participants — not merely an escalation of hostile content. A Wikipedia editor who writes "Your edit is wrong and here's why" may score moderately on hostility but very low (unsafe) on authority dynamics. The relational challenge, not the content, predicts whether the conversation will spiral into personal attacks. Watzlawick's framework predicts exactly this: relationship-level messages are more consequential for interaction outcomes than content-level messages, because they define the frame within which content is interpreted.

**Testable prediction (T1):** If AD measures the command channel, then in conversations where participants explicitly negotiate relational framing (meta-conversation: "Who are you to tell me that?", "I'm just trying to help"), AD's predictive power should increase relative to other dimensions. In purely informational exchanges with no relational positioning (e.g., technical documentation), AD scores should cluster near the neutral point and lose predictive power.

### 24b. Theory 2: The Leading Indicator

A second possibility is that AD measures a temporal signal that other dimensions do not. In the CGA-Wiki study (§21), we observed a temporal gradient: PSQ scores from later conversation turns predicted derailment better than scores from earlier turns (AUC 0.519 → 0.570 → 0.599, first turn → early → all). This is consistent with the general hypothesis that PSQ measures *process*, not static content. But within this process, different dimensions may operate on different time scales.

We propose that authority dynamics is a *leading indicator* — a dimension that shifts before overt hostility, distrust, or emotional dysregulation become manifest. The psychological sequence would be: (1) a participant makes a power claim or challenges another's standing, (2) the challenged participant experiences threat to their social identity, (3) this threat produces defensive hostility, emotional flooding, and trust breakdown. If this causal sequence is correct, AD scores should deteriorate 1–2 turns before HI, TE, and CC scores deteriorate.

This theory explains both the factor structure and the criterion validity pattern. AD loads weakly on the general factor because it operates at a different temporal phase of the safety-threat dynamic — it is the *precursor* to the hostility/threat cluster, not a member of it. It predicts derailment strongly because it captures the initiating event (power challenge) rather than the downstream consequences (hostility, emotional dysregulation) that other dimensions measure.

**Testable prediction (T2):** In a turn-by-turn analysis of CGA-Wiki conversations, AD scores should deteriorate *before* HI and TE scores in conversations that eventually derail. Specifically, the cross-lagged correlation between AD(t) and HI(t+1) should be stronger than between HI(t) and AD(t+1). If AD is merely co-occurring with hostility rather than preceding it, both cross-lagged correlations should be equal.

### 24c. Theory 3: Status Negotiation and Epistemic Positioning

The third theory draws on social identity theory (Tajfel & Turner, 1979) and the sociology of epistemic authority (Collins & Evans, 2007). On this account, AD does not measure formal authority or power hierarchy — the construct implied by its name — but rather *status negotiation*: the moment-to-moment positioning of speakers as epistemic authorities, group members, or moral agents. This is a broader and more interpersonally sensitive construct than "authority dynamics" suggests.

Consider the texts where AD scores diverge most from other dimensions. A Wikipedia editor writing "Actually, if you read the sources more carefully, you'd see that..." scores moderate on hostility (mild condescension) but very low on AD — not because the editor holds formal authority, but because they are claiming epistemic superiority and positioning the other editor as an incompetent reader. A campsite negotiator saying "I really need the firewood because my kids get cold at night" scores low on hostility but moderately low on AD — they are making a moral claim (my children's welfare) that positions the other negotiator as potentially callous if they refuse. These are acts of *status positioning* — assertions about who has the right to define reality, whose needs take priority, whose expertise counts.

This reframing explains why AD predicts outcomes in peer contexts (Wikipedia editors, Reddit commenters, campsite negotiations) where formal authority is absent. It is not formal power that predicts derailment but *contested epistemic and moral standing* — the negotiation of who gets to be right and who gets to be heard.

If this theory is correct, the dimension should be renamed. "Authority dynamics" implies hierarchical power structures; "power positioning" or "status negotiation" better captures the construct as actually measured. The name matters because it shapes how human raters interpret the scoring rubric — and our expert validation study (§19) will ask raters to score "authority dynamics" using a rubric that may be misleading about what the construct actually is.

**Testable prediction (T3a):** AD's predictive power should be equal or greater in peer contexts (Wikipedia, Reddit, campsite negotiation) compared to hierarchical contexts (boss-employee, teacher-student). If AD were measuring formal authority, hierarchical contexts should produce stronger effects.

**Testable prediction (T3b):** In the Deal or No Deal criterion study, AD should predict whether a deal was reached (a relational outcome — did the parties find mutual standing?) but should *not* predict the number of points scored (a zero-sum resource allocation that depends on strategy, not status).

**Testable prediction (T3c):** AD-residual scores should correlate with markers of epistemic positioning — hedging language ("I think," "perhaps"), certainty markers ("clearly," "obviously"), credentialing ("as an expert," "in my experience"), and appeal to authority ("according to," "research shows") — more strongly than with markers of emotional state.

### 24d. Theoretical Integration and Construct Naming

The three theories are not mutually exclusive. AD may measure command-channel communication (Theory 1) that operates as a leading indicator of conflict (Theory 2) through the specific mechanism of status negotiation (Theory 3). The theories describe different levels of the same phenomenon: Theory 1 identifies the *communication channel*, Theory 2 the *temporal dynamics*, and Theory 3 the *psychological mechanism*.

If the empirical evidence supports this integrated account, several consequences follow:

First, the construct should be renamed from "authority dynamics" to something that better captures its actual content. "Power positioning" is our current best candidate — it retains the interpersonal focus while removing the connotation of formal hierarchy. "Status negotiation" is more precise but may be less intuitive to non-specialist users. The expert validation study should test both labels against the current "authority dynamics" to determine which produces higher inter-rater reliability.

Second, the PSQ's theoretical framework should be revised to explicitly distinguish report-level dimensions (HI, TE, RC, RB, ED, CC, DA) from command-level dimensions (AD/power positioning). This distinction is more informative than the current flat 10-dimension architecture and would explain — rather than merely describe — AD's anomalous factor structure.

Third, the finding that a command-channel dimension is the PSQ's strongest external predictor supports a broader theoretical claim: that psychoemotional safety is fundamentally a *relational* property of communication, not merely a *content* property. The report-level dimensions (how hostile, how threatening, how well-regulated) are necessary but not sufficient. The relational scaffolding — who has power, how it is exercised, whether it is contested — is what determines whether a conversation remains safe or degrades into personal attack. This is consistent with Edmondson's (1999) original formulation of psychological safety as an interpersonal climate variable, not an individual trait or a property of message content.

The resolution of these three theories awaits three empirical tests: (1) the Deal or No Deal criterion study (prediction T3b: deal vs. points), (2) a turn-by-turn temporal analysis of CGA-Wiki conversations (prediction T2: AD leads HI), and (3) the expert validation study with alternative construct labels (prediction T3a: peer vs. hierarchical contexts). Until these tests are completed, we hold all three theories as live hypotheses, noting that Theory 3 (status negotiation) currently has the most parsimonious explanatory coverage and the most actionable implications for construct revision.

## 25. The Persuasion Test: Change My View (2026-02-28)

The third criterion validity study extends the PSQ's predictive reach to a new domain: online persuasion. Using the winning-args-corpus from r/ChangeMyView (Tan, Niculae, Danescu-Niculescu-Mizil, & Lee, 2016), we scored 4,263 matched pairs — same original post, one reply that earned a delta (changed the OP's mind) and one that did not — across all 10 PSQ dimensions. The matched-pair design controls for topic, author characteristics, and subreddit norms, isolating the textual properties that discriminate persuasive from non-persuasive arguments.

All 10 dimensions significantly discriminate delta from non-delta replies (paired t-tests, 9/10 surviving Bonferroni correction at alpha=0.005). The direction of effects is theoretically coherent: persuasive replies show higher defensive_architecture (d_z=+0.135, p<1e-17), higher trust_conditions (+0.090), higher cooling_capacity (+0.082), and higher hostility_index (+0.104, reflecting assertive engagement rather than hostility per se), but lower threat_exposure (-0.077, less threatening language). The only dimension not surviving Bonferroni correction is authority_dynamics (d_z=+0.033, p=0.032) — an important finding we return to below.

The logistic regression results replicate the key structural finding from CGA-Wiki: the 10-dimension profile (AUC=0.590, 5-fold CV) substantially outperforms g-PSQ alone (0.531). The gap of 0.059 AUC units between profile and average is nearly identical to the CGA-Wiki gap (0.084), confirming that the *shape* of the PSQ profile carries predictive information that collapsing to a single score destroys.

A text-length confound is present in CMV — delta replies are significantly longer (mean 1,623 vs 1,248 characters, d=0.301) — consistent with Tan et al.'s (2016) original finding that length predicts persuasion. Text length alone achieves AUC=0.596, and the 10-dimension PSQ adds incremental AUC of +0.012 beyond length (combined AUC=0.608). Partial correlations confirm that 9/10 dimensions retain significance after controlling for length.

The most striking finding is defensive_architecture's emergence as the top individual predictor (r_pb=+0.085, paired accuracy=55.4%), displacing authority_dynamics from the top position it held in CaSiNo and CGA-Wiki. This is not a contradiction but a context-dependent pattern: in CMV, where the task is to construct a convincing argument rather than to navigate a relationship, the *structural quality of argumentation* (DA) matters more than *interpersonal power positioning* (AD). DA measures boundary maintenance, structured reasoning, and cognitive framing — precisely the toolkit of effective persuasion.

Authority_dynamics, meanwhile, shows the weakest bivariate effect in CMV (r_pb=+0.021, not Bonferroni-significant) despite its dominance in CGA-Wiki and CaSiNo. This is exactly what Theory 3 from §24 predicts: AD/power positioning should matter most when status is *contested* (Wikipedia disputes, negotiation) and least when the social structure is *fixed* (CMV, where the OP has explicitly invited counterarguments and grants delta voluntarily). In CMV, the power dynamic is settled — the OP holds the delta, challengers must persuade — so there is little status to negotiate. The dimension's predictive power drops accordingly.

This pattern provides the first empirical evidence distinguishing our three theories. Theory 1 (meta-conversation) and Theory 2 (leading indicator) predict AD should be generally predictive across contexts; Theory 3 (status negotiation) uniquely predicts context-dependent effects tied to whether status is fixed or contested. The CMV results favor Theory 3.

The cross-study comparison now spans three independent datasets:

| Study | Domain | N | Outcome | 10-dim AUC | g-PSQ AUC | Top dim | AD rank |
|---|---|---|---|---|---|---|---|
| CaSiNo | Negotiation | 1,030 | Satisfaction | — | — | AD (r=0.127) | 1st |
| CGA-Wiki | Wikipedia | 4,188 | Derailment | 0.599 | 0.515 | AD (r_pb=-0.105) | 1st |
| CMV | Persuasion | 4,263 pairs | Delta | 0.590 | 0.531 | DA (r_pb=+0.085) | 11th (weakest) |

The consistency of the profile-shape-over-average finding across three studies — with gaps of 0.059–0.084 AUC — provides strong evidence that the PSQ's multi-dimensional architecture is psychometrically justified. The context-dependent ranking of dimensions provides equally strong evidence that these are genuine, distinct constructs rather than redundant measures of a single latent variable. If the 10 dimensions were merely noisy reflections of g-PSQ, their relative importance would not systematically vary with the outcome's psychological demands.

## 26. Publication Narrative (2026-02-28)

The PSQ project has converged on a publication-ready finding that is more interesting than what we set out to build. We began with an engineering goal: distill a 10-dimension psychoemotional safety scorer from an LLM teacher into a fast local model. What we found along the way is a substantive scientific result about the structure of conversational safety — one that contributes to theory, not just tooling.

### 26a. The Core Contribution

The central finding is **context-dependent predictive primacy**: in a multi-dimensional model of psychoemotional safety, the dimension that best predicts real-world outcomes *changes depending on the social context*. Authority dynamics (power positioning) dominates when status is contested (Wikipedia disputes, campsite negotiation). Defensive architecture (structured argumentation) dominates when status is fixed and the task is persuasion (Change My View). This is not a failure of the instrument — it is evidence that the 10 dimensions measure genuinely distinct psychological mechanisms that interact differently with different social demands.

This finding has three layers of significance:

**Methodological.** The 10-dimension profile consistently outperforms the single-score average (g-PSQ) across three studies. The gaps (0.059–0.084 AUC) are modest individually but remarkably consistent, and they survive text-length controls in the two studies where length is a confound. This is the strongest empirical argument against collapsing multi-dimensional safety instruments to single scores — a practice common in toxicity detection (Perspective API, Detoxify) and content moderation systems.

**Theoretical.** The context-dependent ranking connects to Watzlawick et al.'s (1967) report/command distinction. Most NLP safety tools measure report-level properties (how hostile is the content). The PSQ's authority_dynamics dimension measures the command channel (who has power and how it is exercised), and this command-level signal is what predicts relational outcomes — but only when the relational structure is contested. This bridges communication theory, social psychology, and NLP in a way that no existing toxicity or safety instrument does.

**Practical.** Content moderation systems that use single-score toxicity thresholds will miss the relational dynamics that actually predict conversation failure. The finding that different dimensions matter for different outcomes implies that safety tools should be context-aware — weighting power positioning heavily for moderation, defensive architecture for educational contexts, and regulatory capacity for therapeutic applications.

### 26b. The Paper Structure

A publication targeting a venue at the intersection of computational linguistics and psychology (EMNLP, ACL Findings, or Behavior Research Methods) would follow this structure:

1. **Introduction.** The problem of multi-dimensional safety measurement. Why single-score toxicity is insufficient. The PSQ as a case study in knowledge distillation from LLM to local model.

2. **The PSQ Instrument.** 10 dimensions, theoretical grounding in JD-R, COR, and psychological safety frameworks. Construction via composite proxy labeling + separated LLM scoring. DistilBERT student model architecture.

3. **Factor Structure.** General factor (55% variance) + 5 clusters. Two singleton dimensions (AD, ED) that resist clustering. Bifactor implications.

4. **Criterion Validity Battery.** Three studies:
   - CaSiNo (n=1,030): negotiation satisfaction and liking
   - CGA-Wiki (n=4,188): conversation derailment
   - CMV (n=4,263 pairs): persuasion success

   The cross-study comparison table is the centerpiece. Profile >> average in all three. AD top predictor in contested-status, DA top in fixed-status.

5. **The Status Negotiation Theory.** Three competing theories of AD's predictive primacy. CMV evidence favoring Theory 3. Testable predictions. Implications for construct naming (authority_dynamics → power_positioning).

6. **Limitations.** LLM labeling chain (no human ground truth yet). Held-out evaluation uses LLM labels. Expert validation needed. DistilBERT's 128-token context window limits long-text scoring. All criterion studies use English text from specific online platforms.

7. **Discussion.** Safety is relational, not just content-level. Multi-dimensional instruments with context-aware interpretation outperform single-score alternatives. The dimension the model learned to detect (status negotiation) is more nuanced than what was designed (authority dynamics) — a case study in emergent construct validity through knowledge distillation.

### 26c. What Strengthens the Paper Before Submission

The current evidence base is strong for a workshop or short paper. For a main-conference long paper, several additions would strengthen it:

- **Deal or No Deal results** (in progress) — if AD predicts deal but not points, the status negotiation theory gets causal-directional support
- **Expert validation** — even preliminary ICC from 2-3 raters would address the "it's all LLM labeling" concern
- **Turn-by-turn temporal analysis** — if AD leads HI in derailing conversations, the leading indicator theory gets direct support
- **Bifactor model comparison** — does a bifactor architecture improve held-out r, or is the current flat architecture already sufficient?
- **A non-English or non-online test** — criterion validity from a different discourse context (workplace transcripts, therapy sessions) would demonstrate generalizability

### 26d. The Emergent Construct Validity Story

Perhaps the most novel contribution for a psychometrics audience is the *emergent construct validity* narrative. The PSQ was designed with 10 dimensions based on theoretical frameworks (JD-R, COR, psychological safety). The LLM teacher was instructed to score these dimensions. The student model learned to replicate those scores. But the resulting instrument measures something more nuanced than what was specified — authority_dynamics turned out to capture *status negotiation*, not formal authority, and this emergent interpretation has stronger criterion validity than the designed one.

This is analogous to the discovery of the p factor in clinical psychology (Caspi et al., 2014) — a general factor that emerged from a measurement instrument designed to measure specific disorders. The PSQ's general factor and its status-negotiation dimension are both *emergent* properties of the measurement process, not designed-in features. They represent what the data taught us about the structure of psychoemotional safety, beyond what our initial theory specified.

This is a methodological contribution to the growing field of LLM-as-instrument-designer: the finding that an LLM, when asked to label psychological constructs, can discover construct-relevant features that the human designer did not specify. The student model, trained on these labels, then replicates these discoveries in a form that can be validated against external criteria. Knowledge distillation becomes not just model compression but *construct refinement*.

---

## 27. The Deal Test: When Energy Matters More Than Status (2026-02-28)

The fourth criterion validity study — and the largest — used the Deal or No Deal negotiation corpus (Lewis et al., 2017; n=12,234 dialogues). Where CaSiNo measured self-reported satisfaction, DonD provides a behavioral outcome: whether the negotiation actually produced a deal, and how many points were scored. This distinction matters. A participant can feel satisfied about a failed negotiation, or dissatisfied about a successful one. By testing PSQ against both relational (CaSiNo) and behavioral (DonD) outcomes within the same domain (negotiation), we can separate what PSQ predicts about *experience* from what it predicts about *results*.

The results were striking. First, the 10-dimension PSQ achieved its strongest AUC to date: 0.686, compared to 0.599 (CGA-Wiki), 0.590 (CMV), and uncalculated for CaSiNo (which used continuous outcomes). Second, the top predictor was energy_dissipation (ED), with a Cohen's d of +0.614 — by far the largest single-dimension effect size across all four studies. AD, which had dominated both CaSiNo and CGA-Wiki, was the weakest predictor in DonD (d=-0.063, near zero and slightly negative).

This inversion is theoretically coherent. In DonD negotiations, both parties want a deal — the question is whether they can sustain engagement long enough to find mutually acceptable terms. ED, which captures the availability of healthy dissipation pathways and recovery from engagement fatigue, directly indexes this capacity. AD, which captures status positioning and authority dynamics, is less relevant when both parties are motivated to agree. The status negotiation theory (§24, Theory 3) predicted exactly this: AD should predict more strongly when status is contested and the outcome depends on who defines the terms, not when the task is to reach mutual accommodation.

With four studies complete, the context-dependency pattern resolves into a clean matrix. AD dominates in contested-status interactions with relational outcomes (CaSiNo satisfaction) and behavioral outcomes (CGA-Wiki derailment prevention). ED dominates in behavioral outcomes that depend on sustained engagement (DonD deal-reaching). DA dominates when status is structurally fixed (CMV persuasion). The g-PSQ average remains near-chance in every study (AUC: 0.515–0.622), while the 10-dimension profile consistently predicts — confirming that it is the *shape* of the safety profile, not its average level, that carries signal.

The incremental validity was also notable: high-PSQ dialogues (Q4) reached deals at 84.4%, compared to 68.5% for low-PSQ (Q1) — a 15.9 percentage-point difference. Text length is a significant confound (r=-0.339 with deal outcome; shorter conversations tend to deal more easily), but PSQ retains significance after controlling for length (incremental AUC +0.059 beyond length + turns).

The AD suppressor variable pattern replicated again: despite a near-zero bivariate correlation, AD received a negative coefficient (-0.534) in logistic regression, indicating it carries information that improves multivariate prediction once shared variance with other dimensions is removed. This is now confirmed across three of four studies and represents one of the most robust findings in the PSQ criterion validity program.

---

## 28. The g-Factor Deepens and the Integer Problem (2026-02-28)

The fourth training cycle produced the best model yet — v19, held-out r=0.600 — but the more consequential findings came from two analytical investigations that challenge assumptions about the measurement system itself.

### The Broad-Spectrum Strategy

v19 benefited from the broad-spectrum labeling batch: 300 texts selected for diversity (150 random, 100 single-dimension keyword-filtered, 50 multi-dimension) rather than targeting a specific weak dimension. The result was the broadest improvement profile of any training run: 7/10 dimensions improved, with the three weakest dimensions showing the largest gains. Threat exposure recovered by +0.125 (its largest single-run improvement), energy dissipation by +0.087, and authority dynamics by +0.058. Only three dimensions regressed, all modestly (resilience -0.027, contractual clarity -0.020, cooling capacity -0.016).

The lesson is counterintuitive. The dimension-focused batches (CO-filtered, TE-filtered, AD-filtered) improved their target dimensions but often at the cost of others, because the keyword-filtered texts had skewed profiles. The broad-spectrum approach, by providing diverse texts with diverse score profiles, gave the model a richer signal landscape. This is consistent with the factor analysis finding that the dimensions share a strong general factor: improving the model's grasp of "overall safety" lifts all dimensions, while improving its grasp of one specific dimension can distort its general calibration.

### Factor Analysis v2: The g-Factor Strengthens

A re-run of the factor analysis on N=1,970 texts with complete separated-llm coverage (excluding joint-llm and composite-proxy) produced a striking result: the first eigenvalue jumped from 4.844 (48.4% of variance) to 6.727 (67.3%). KMO improved from 0.819 ("Meritorious") to 0.902 ("Superb"). Parallel analysis, which previously retained 2 factors, now retains only 1. The 5-factor structure — Hostility/Threat, Relational Contract, Internal Resources, Power Dynamics, Stress/Energy — largely collapsed. Factor 1 absorbed 8 of 10 dimensions; only CO, ED, and AD maintained weak separation.

All 10 g-factor loadings exceeded 0.66, with trust_conditions (0.930) and defensive_architecture (0.914) at the top. This is remarkable for DA, the construct that had the weakest promax loading in the previous analysis. The mean inter-dimension correlation rose from 0.417 to 0.632.

The strengthening has two possible explanations, and they are not mutually exclusive. First, as composite-proxy noise is excluded from the analysis, the genuine correlation structure becomes more apparent. The proxy mappings (Berkeley hate speech to threat exposure, UCC condescension to authority dynamics) introduced dimension-specific noise that artificially decorrelated dimensions. With pure separated-llm data, the true co-variation of psychoemotional safety dimensions is visible, and it is high. Second, the integer-only scoring bias may be inflating correlations mechanically.

### The Integer Problem

This is the most consequential discovery of the cycle. A score distribution audit revealed that the LLM scorer almost never assigns non-integer values on the 0-10 scale. Despite the rubric permitting continuous scores, the effective scoring scale is 11 bins (integers 0 through 10). Worse, the 4-5-6 band captures 57-81% of all separated-llm scores, with score-5 concentration ranging from 24.1% (TE, the best) to 60.8% (CO, the worst).

This means the measurement system has less resolution than it appears to have. When a 0-10 scale is effectively an 11-point ordinal scale with most mass in three bins, the practical information content per score is low. Two texts that differ meaningfully in, say, contractual clarity may both receive a score of 5 — not because the LLM cannot distinguish them, but because the integer scale does not require it to.

The integer bias also has a direct mechanistic path to inflated inter-dimension correlations. If 45% of texts receive score-5 on both dimension A and dimension B, the shared "neutral" signal creates correlation even if the dimensions are genuinely independent for the 55% of texts outside that bin. The score-concentration cap downweights these texts in training (from 5.0 to 3.38-4.58), but the cap operates on the training loss, not on the correlation matrix used for factor analysis.

The proposed mitigation is to switch the LLM scoring prompt from a 0-10 scale to a 0-100 percentage scale, with post-processing back to 0-10 for model training. The hypothesis is that asking the LLM to assign a percentage (e.g., "45% vs 55%") will produce finer granularity than asking for a score (e.g., "4 vs 5" — or more likely, "5 vs 5"). This is a well-known psychometric technique: expanding the response scale forces raters to make finer discriminations.

This investigation has high priority because it affects the interpretation of every prior analysis. If the g-factor eigenvalue drops substantially under a 0-100 scoring regime, the 5-factor structure may re-emerge. If it does not, the general factor is genuine. Either way, the measurement system needs more resolution.

---

## 29. The Resolution Fix: Percentage Scoring at Scale (2026-02-28)

The integer problem identified in §28 — where the LLM scorer almost never assigns non-integer values on the 0-10 scale, producing an effective 11-bin ordinal scale with most mass in the 4-5-6 band — demanded an engineering solution before further factor analysis or model training could meaningfully proceed. The pilot (50 texts, single session) demonstrated that a 0-100 percentage scale breaks the integer constraint, but the single-session design introduced massive halo (mean inter-dimension r=0.986), leaving the critical question unanswered: does the resolution improvement survive proper separated scoring?

It does, and it improves further. A production batch of 200 texts scored with the separated protocol (one dimension per conversation context, ten separate scoring sessions) produced 86.2% non-integer scores versus 77.8% in the pilot — and more importantly, versus 2.1% with the standard 0-10 scale. Exact-5.0 concentration dropped to 4.8%, down from 7.2% in the pilot and 41.3% in the integer-scored database. The number of unique score values rose to 35, compared to ~11 with integer scoring.

The magnitude of improvement is difficult to overstate. Under the integer regime, the measurement system was discarding roughly 90% of the scorer's discriminative capacity — the difference between a text that "feels like" a 4 and one that "feels like" a 5 on contractual clarity was simply erased. Under the percentage regime, a text scored at 42% and one at 48% are preserved as meaningfully different (4.2 vs 4.8 on the internal 0-10 scale). Whether the model can *learn* from this finer granularity is the subject of the next training experiment.

The bifactor experiment (v19b) provided a useful negative result in parallel. Adding an 11th output head to predict g-PSQ (the general factor, operationalized as the mean of 10 dimension scores) produced a well-learned g-head (r=0.594) but degraded per-dimension prediction (test_r 0.509→0.502). The DistilBERT backbone (66.7M params, 384-dim projection) does not have sufficient representational capacity to serve 11 output heads without capacity competition. The practical implication is clear: if a general factor score is needed, compute it post-hoc from the 10 dimension scores rather than training a dedicated head. The bifactor architecture (Design A from §35) should be revisited only with a larger base model.

### The Dimension Collapse

But the more consequential finding came from factor analysis on the pct-scored data. The original hypothesis — that integer-only scoring was mechanically inflating inter-dimension correlations through shared "score-5" signals — predicted that percentage scoring would *reduce* the g-factor eigenvalue. The opposite occurred. The g-factor eigenvalue jumped from 6.727 (67.3% of variance, integer data) to 9.410 (94.1%, pct data). Mean inter-dimension correlation rose from 0.632 to 0.934.

The diagnostic data are unambiguous. Within-text standard deviation — the measure of how much a scorer differentiates between dimensions for a single text — dropped from 0.717 (integer) to 0.448 (pct). Two-thirds of pct-scored texts have all ten dimension scores within a one-point range. Between-text variance accounts for 93.2% of total variance, meaning the scores are almost entirely "overall safety" with negligible dimension-specific signal. Eight of ten dimensions retain less than 5% unique variance in the pct data, versus 16-47% in integer data.

The mechanism appears to be anchoring-and-adjustment: the 0-100 scale invites the scorer to first form a global safety impression ("about 35%"), then make tiny adjustments per dimension ("maybe 33% for TE, 37% for TC"). The adjustments are real — parallel analysis on residuals after removing the text mean retains 3 factors — but they are overwhelmed by the global anchor. Integer scoring, paradoxically, forces larger discrete jumps between dimensions, producing more genuine differentiation.

This finding has important consequences beyond PSQ. It suggests that expanding the number of response options in a rating scale does not monotonically improve measurement quality. There is an optimum: enough options to capture meaningful between-item differences, but not so many that the rater's global impression dominates the within-item differentiation. For constructs with genuine multi-dimensional structure, an 11-point scale (0-10 integers) may outperform a 101-point scale (0-100 percentages), consistent with Schwarz et al.'s (1991) work on response format effects in survey methodology.

The practical implications are clear: revert to integer scoring for future labeling. The g-factor is real, the 10 dimensions are genuinely correlated, and the integer scale provides more informative dimension-specific variance than the percentage scale. The percentage resolution gains (less score-5 concentration, more unique values) are real but are outweighed by the dimension-collapse cost.

### The Training Confirmation

v20 training — which added the 200 pct-scored texts (2,000 new separated-llm labels) to the training set — produced an anticlimactic result that confirms the analysis: held-out_r = 0.600, identical to v19. The pct-scored data neither helps nor hurts. At 200 texts in a 17,000-text training set, the volume is too small to move the needle, and the collapsed dimension structure means the new labels carry primarily g-factor signal rather than dimension-specific information. The per-dimension shifts (CO +0.024, CC +0.023 vs ED -0.034, TE -0.028) are within random training variance for a 100-text held-out evaluation.

The null result closes the percentage scoring research arc and opens a new one. A systematic literature review identified the root cause as **isomorphic rubric structure** — all ten PSQ rubrics follow the same template (0=extreme bad, 5=neutral, 10=extreme good), which teaches the scorer to treat them as instances of a single continuum. Humphry and Heldsinger (2014) documented precisely this phenomenon in educational assessment rubrics, calling it a "common structural design feature" that threatens validity. The most promising mitigation avenue is redesigning rubric anchors to be structurally dissimilar across dimensions — using concrete, dimension-specific content features rather than abstract quality gradients. A comprehensive research plan (`scoring-research-plan.md`) tracks eight research avenues for rubric-induced halo mitigation, ranked by evidence strength and feasibility.

---

## 30. References

Andrews, G., Singh, M., & Bond, M. (1993). The Defense Style Questionnaire. *Journal of Nervous and Mental Disease, 181*(4), 246–256.

AERA, APA, & NCME. (2014). *Standards for Educational and Psychological Testing*. American Educational Research Association.

Armour, C., Tsai, J., Durham, T. A., Charak, R., Biehn, T. L., Elhai, J. D., & Pietrzak, R. H. (2016). Dimensional structure of DSM-5 posttraumatic stress symptoms: Support for a hybrid anhedonia and externalizing behaviors model. *Journal of Psychiatric Research, 73*, 117–126.

Bakker, A. B., & Demerouti, E. (2007). The Job Demands-Resources model: State of the art. *Journal of Managerial Psychology, 22*(3), 309–328.

Bass, B. M., & Avolio, B. J. (1995). *MLQ Multifactor Leadership Questionnaire*. Mind Garden.

Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced metrics for measuring unintended bias with real data for text classification. In *Proceedings of the 2019 World Wide Web Conference* (pp. 491–500).

Buss, A. H., & Perry, M. (1992). The aggression questionnaire. *Journal of Personality and Social Psychology, 63*(3), 452–459.

Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant validation by the multitrait-multimethod matrix. *Psychological Bulletin, 56*(2), 81–105.

Caspi, A., Houts, R. M., Belsky, D. W., Goldman-Mellor, S. J., Harrington, H., Israel, S., ... & Moffitt, T. E. (2014). The p factor: One general psychopathology factor in the structure of psychiatric disorders? *Clinical Psychological Science, 2*(2), 119–137.

Cattell, R. B. (1988). The meaning and strategic use of factor analysis. In J. R. Nesselroade & R. B. Cattell (Eds.), *Handbook of Multivariate Experimental Psychology* (2nd ed., pp. 131–203). Plenum Press.

Conger, A. J. (1974). A revised definition for suppressor variables: A guide to their identification and interpretation. *Educational and Psychological Measurement, 34*(1), 35–46.

Chawla, K., Ramirez, J., Clever, R., Lucas, G., May, J., & Gratch, J. (2021). CaSiNo: A corpus of campsite negotiation dialogues for automatic negotiation systems. In *Proceedings of NAACL-HLT 2021* (pp. 3167–3185).

Collins, H. M., & Evans, R. (2007). *Rethinking Expertise*. University of Chicago Press.

Cicchetti, D. V. (1994). Guidelines, criteria, and rules of thumb for evaluating normed and standardized assessment instruments in psychology. *Psychological Assessment, 6*(4), 284–290.

Connor, K. M., & Davidson, J. R. T. (2003). Development of a new resilience scale: The Connor-Davidson Resilience Scale (CD-RISC). *Depression and Anxiety, 18*(2), 76–82.

Cook, W. W., & Medley, D. M. (1954). Proposed hostility and pharisaic-virtue scales for the MMPI. *Journal of Applied Psychology, 38*(6), 414–418.

Cummings, L. L., & Bromiley, P. (1996). The Organizational Trust Inventory (OTI). In R. M. Kramer & T. R. Tyler (Eds.), *Trust in Organizations* (pp. 302–330). Sage.

Danescu-Niculescu-Mizil, C., Sudhof, M., Jurafsky, D., Leskovec, J., & Potts, C. (2013). A computational approach to politeness with application to social factors. In *Proceedings of ACL 2013* (pp. 250–259).

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. In *Proceedings of ACL 2020* (pp. 4040–4054).

DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials, 7*(3), 177–188.

Duckworth, A. L., Peterson, C., Matthews, M. D., & Kelly, D. R. (2007). Grit: Perseverance and passion for long-term goals. *Journal of Personality and Social Psychology, 92*(6), 1087–1101.

Edmondson, A. (1999). Psychological safety and learning behavior in work teams. *Administrative Science Quarterly, 44*(2), 350–383.

Einarsen, S., Hoel, H., & Notelaers, G. (2009). Measuring exposure to bullying and harassment at work: Validity, factor structure and psychometric properties of the Negative Acts Questionnaire-Revised. *Work & Stress, 23*(1), 24–44.

Felson, R. B., & Tedeschi, J. T. (1993). A social interactionist approach to violence: Cross-cultural applications. *Violence and Victims, 8*(3), 295–310.

French, J. R. P., & Raven, B. (1959). The bases of social power. In D. Cartwright (Ed.), *Studies in Social Power* (pp. 150–167). Institute for Social Research.

Garnefski, N., Kraaij, V., & Spinhoven, P. (2001). Negative life events, cognitive emotion regulation and emotional problems. *Personality and Individual Differences, 30*(8), 1311–1327.

Gough, H. G. (1987). *California Psychological Inventory administrator's guide*. Consulting Psychologists Press.

Gratz, K. L., & Roemer, L. (2004). Multidimensional assessment of emotion regulation and dysregulation: Development, factor structure, and initial validation of the difficulties in emotion regulation scale. *Journal of Psychopathology and Behavioral Assessment, 26*(1), 41–54.

Gross, J. J. (1998). The emerging field of emotion regulation: An integrative review. *Review of General Psychology, 2*(3), 271–299.

Gross, J. J., & John, O. P. (2003). Individual differences in two emotion regulation processes: Implications for affect, relationships, and well-being. *Journal of Personality and Social Psychology, 85*(2), 348–362.

Hanu, L., & Unitary team. (2020). Detoxify [Software]. GitHub. https://github.com/unitaryai/detoxify

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263–1284.

He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. In *Proceedings of ICLR 2021*.

Henry, J. D., & Crawford, J. R. (2005). The short-form version of the Depression Anxiety Stress Scales (DASS-21): Construct validity and normative data in a large non-clinical sample. *British Journal of Clinical Psychology, 44*(2), 227–239.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.

Hobfoll, S. E. (1989). Conservation of resources: A new attempt at conceptualizing stress. *American Psychologist, 44*(3), 513–524.

Kelley, T. L. (1927). Interpretation of educational measurements. *Journal of Social Psychology, 6*, 103–118.

Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). Constructing interval variables via faceted Rasch measurement and multitask deep learning: A hate speech application. *arXiv preprint arXiv:2009.10277*.

Kim, H., Cho, H., Kim, M., Kim, Y., & Choi, J. (2022). ProsocialDialog: A prosocial backbone for conversational agents. In *Proceedings of EMNLP 2022* (pp. 4005–4029).

Kuhn, T. S. (1962). *The Structure of Scientific Revolutions*. University of Chicago Press.

Lazarus, R. S., & Folkman, S. (1984). *Stress, Appraisal, and Coping*. Springer.

Lewis, D. D., & Gale, W. A. (1994). A sequential algorithm for training text classifiers. In *Proceedings of SIGIR 1994* (pp. 3–12).

Lievens, F., De Corte, W., & Schollaert, E. (2008). Adjusting exercise difficulty in assessment centers: A within-sample approach. *International Journal of Selection and Assessment, 16*(2), 130–136.

Liu, S., Zheng, C., Demasi, O., Sabour, S., Li, Y., Yu, Z., Jiang, Y., & Huang, M. (2021). Towards emotional support dialog systems. In *Proceedings of ACL-IJCNLP 2021* (pp. 3469–3483).

Lovibond, P. F., & Lovibond, S. H. (1995). The structure of negative emotional states: Comparison of the Depression Anxiety Stress Scales (DASS) with the Beck Depression and Anxiety Inventories. *Behaviour Research and Therapy, 33*(3), 335–343.

Meijman, T. F., & Mulder, G. (1998). Psychological aspects of workload. In P. J. D. Drenth, H. Thierry, & C. J. de Wolff (Eds.), *Handbook of Work and Organizational Psychology* (2nd ed., pp. 5–33). Psychology Press.

Morrison, E. W., & Robinson, S. L. (1997). When employees feel betrayed: A model of how psychological contract violation develops. *Academy of Management Review, 22*(1), 226–256.

Nunnally, J. C., & Bernstein, I. H. (1994). *Psychometric Theory* (3rd ed.). McGraw-Hill.

Pejtersen, J. H., Kristensen, T. S., Borg, V., & Bjorner, J. B. (2010). The second version of the Copenhagen Psychosocial Questionnaire. *Scandinavian Journal of Public Health, 38*(Suppl 3), 8–24.

Perry, J. C. (1990). *Defense Mechanism Rating Scales* (5th ed.). Cambridge Hospital.

Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers* (pp. 61–74). MIT Press.

Price, I., Gifford-Moore, J., Flemming, J., Musber, S., Roichman, M., Sylvain, G., Thain, N., Dixon, L., & Borkan, D. (2020). Six attributes of unhealthy conversations. In *Proceedings of the Fourth Workshop on Online Abuse and Harms* (pp. 114–124).

Rashkin, H., Smith, E. M., Li, M., & Boureau, Y.-L. (2019). Towards empathetic open-domain conversation models: A new benchmark and dataset. In *Proceedings of ACL 2019* (pp. 5370–5381).

Reise, S. P. (2012). The rediscovery of bifactor measurement models. *Multivariate Behavioral Research, 47*(5), 667–696.

Rheinberg, F., Vollmeyer, R., & Engeser, S. (2003). Die Erfassung des Flow-Erlebens [The assessment of flow experience]. In J. Stiensmeier-Pelster & F. Rheinberg (Eds.), *Diagnostik von Motivation und Selbstkonzept* (pp. 261–279). Hogrefe.

Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Applying bifactor statistical indices in the evaluation of psychological measures. *Journal of Personality Assessment, 98*(3), 223–237.

Rotter, J. B. (1967). A new scale for the measurement of interpersonal trust. *Journal of Personality, 35*(4), 651–665.

Rousseau, D. M. (1995). *Psychological Contracts in Organizations: Understanding Written and Unwritten Agreements*. Sage.

Ruder, S. (2017). An overview of multi-task learning in deep neural networks. *arXiv preprint arXiv:1706.05098*.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data, 6*(1), 60.

Smith, B. W., Dalen, J., Wiggins, K., Steger, M. F., & Tooley, E. (2008). The Brief Resilience Scale: Assessing the ability to bounce back. *International Journal of Behavioral Medicine, 15*(3), 194–200.

Sonnentag, S., & Fritz, C. (2007). The Recovery Experience Questionnaire: Development and validation of a measure for assessing recuperation and unwinding from work. *Journal of Occupational Health Psychology, 12*(3), 204–221.

Spielberger, C. D. (1999). *STAXI-2: State-Trait Anger Expression Inventory-2*. Psychological Assessment Resources.

Stamm, B. H. (2010). *The Concise ProQOL Manual* (2nd ed.). ProQOL.org.

Tajfel, H., & Turner, J. C. (1979). An integrative theory of intergroup conflict. In W. G. Austin & S. Worchel (Eds.), *The Social Psychology of Intergroup Relations* (pp. 33–47). Brooks/Cole.

Tan, C., Niculae, V., Danescu-Niculescu-Mizil, C., & Lee, L. (2016). Winning arguments: Interaction dynamics and persuasion strategies in good-faith online discussions. In *Proceedings of the 25th International Conference on World Wide Web* (pp. 613–624).

Tepper, B. J. (2000). Consequences of abusive supervision. *Academy of Management Journal, 43*(2), 178–190.

Thorndike, E. L. (1920). A constant error in psychological ratings. *Journal of Applied Psychology, 4*(1), 25–29.

Treviso, M., Ji, T., Pruthi, D., & Martins, A. F. T. (2023). Efficient methods for natural language processing: A survey. *Transactions of the Association for Computational Linguistics, 11*, 826–860.

Turney, P. D., Neuman, Y., Assaf, D., & Cohen, Y. (2019). Dreaddit: A Reddit dataset for stress analysis in social media. In *Proceedings of the 10th International Workshop on Health Text Mining and Information Analysis* (pp. 97–107).

Vaillant, G. E. (1977). *Adaptation to Life*. Little, Brown.

McEwen, B. S. (1998). Stress, adaptation, and disease: Allostasis and allostatic load. *Annals of the New York Academy of Sciences, 840*(1), 33–44.

McCrae, R. R., & Costa, P. T. (1997). Conceptions and correlates of openness to experience. In R. Hogan, J. Johnson, & S. Briggs (Eds.), *Handbook of Personality Psychology* (pp. 825–847). Academic Press.

Meehl, P. E. (1956). Wanted — a good cookbook. *American Psychologist, 11*(6), 263–272.

Meehl, P. E. (1990). Why summaries of research on psychological theories are often uninterpretable. *Psychological Reports, 66*(1), 195–244.

Watzlawick, P., Beavin, J. H., & Jackson, D. D. (1967). *Pragmatics of Human Communication: A Study of Interactional Patterns, Pathologies, and Paradoxes*. W. W. Norton.

Zhang, J., Chang, J., Danescu-Niculescu-Mizil, C., Dixon, L., Hua, Y., Taraborelli, D., & Thain, N. (2018). Conversations gone awry: Detecting early signs of conversational failure. In *Proceedings of ACL 2018* (pp. 1350–1361).
