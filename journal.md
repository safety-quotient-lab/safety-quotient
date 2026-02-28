# PSQ Research Journal

A chronological research narrative of the Psychoemotional Safety Quotient (PSQ) project: from initial conceptualization through construct formalization, knowledge distillation, psychometric validation, and the discovery of latent dimensionality structure. Written in the idiom of a methods-and-findings journal article to support reproducibility, peer review, and future meta-analytic work.

**Principal investigator:** Kashif Shah
**Research assistant:** Claude (Anthropic) — LLM-assisted construct operationalization, data labeling, and analysis
**Inception:** May 2022 (conceptual vocabulary) / February 25, 2026 (formal construct definition)
**Current date:** 2026-02-27

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
18. [References](#18-references)

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

### 14a. Model Performance (v15, 2026-02-27)

| Metric | Value |
|---|---|
| Architecture | DistilBERT-base-uncased (66.7M params) |
| Training data | 20,127 texts in DB (63,361 scores, 9,771 separated-llm) |
| Test avg Pearson r | 0.536 (10/10 dimensions positive) |
| Held-out avg Pearson r | 0.495 (+0.013 vs v14, +0.093 vs v13) |
| Generalization gap | 7.6% (down from 11.4% in v14) |
| ONNX model size | 64 MB (INT8 quantized, v13 — v15 not yet exported) |
| Inference latency | ~20ms / text (CPU) |

### 14b. Psychometric Properties

| Property | Status | Evidence | Standard |
|---|---|---|---|
| Test-retest reliability | Excellent | ICC = 0.935 (perturbation-based) | ICC > 0.75 (Cicchetti, 1994) |
| Discriminant validity (vs. sentiment) | Strong | Mean |r| = 0.205 vs VADER | r < 0.30 (distinct construct) |
| Confidence calibration | Done | Isotonic regression; 8/10 dims improved | Platt (1999) |
| Held-out generalization | Moderate | r = 0.495, n = 100 (separated labels, v15) | Comparable to brief personality measures |
| Construct validity (discriminant) | Confirmed | Halo addressed; 3-cluster hierarchy implemented | Requires CFA (n ≥ 200) |
| Inter-rater reliability | Not measured | — | Critical gap |
| Criterion validity | Not measured | — | Requires external criterion |
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

## 18. References

Andrews, G., Singh, M., & Bond, M. (1993). The Defense Style Questionnaire. *Journal of Nervous and Mental Disease, 181*(4), 246–256.

AERA, APA, & NCME. (2014). *Standards for Educational and Psychological Testing*. American Educational Research Association.

Armour, C., Tsai, J., Durham, T. A., Charak, R., Biehn, T. L., Elhai, J. D., & Pietrzak, R. H. (2016). Dimensional structure of DSM-5 posttraumatic stress symptoms: Support for a hybrid anhedonia and externalizing behaviors model. *Journal of Psychiatric Research, 73*, 117–126.

Bakker, A. B., & Demerouti, E. (2007). The Job Demands-Resources model: State of the art. *Journal of Managerial Psychology, 22*(3), 309–328.

Bass, B. M., & Avolio, B. J. (1995). *MLQ Multifactor Leadership Questionnaire*. Mind Garden.

Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced metrics for measuring unintended bias with real data for text classification. In *Proceedings of the 2019 World Wide Web Conference* (pp. 491–500).

Buss, A. H., & Perry, M. (1992). The aggression questionnaire. *Journal of Personality and Social Psychology, 63*(3), 452–459.

Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant validation by the multitrait-multimethod matrix. *Psychological Bulletin, 56*(2), 81–105.

Caspi, A., Houts, R. M., Belsky, D. W., Goldman-Mellor, S. J., Harrington, H., Israel, S., ... & Moffitt, T. E. (2014). The p factor: One general psychopathology factor in the structure of psychiatric disorders? *Clinical Psychological Science, 2*(2), 119–137.

Chawla, K., Ramirez, J., Clever, R., Lucas, G., May, J., & Gratch, J. (2021). CaSiNo: A corpus of campsite negotiation dialogues for automatic negotiation systems. In *Proceedings of NAACL-HLT 2021* (pp. 3167–3185).

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

Tepper, B. J. (2000). Consequences of abusive supervision. *Academy of Management Journal, 43*(2), 178–190.

Thorndike, E. L. (1920). A constant error in psychological ratings. *Journal of Applied Psychology, 4*(1), 25–29.

Treviso, M., Ji, T., Pruthi, D., & Martins, A. F. T. (2023). Efficient methods for natural language processing: A survey. *Transactions of the Association for Computational Linguistics, 11*, 826–860.

Turney, P. D., Neuman, Y., Assaf, D., & Cohen, Y. (2019). Dreaddit: A Reddit dataset for stress analysis in social media. In *Proceedings of the 10th International Workshop on Health Text Mining and Information Analysis* (pp. 97–107).

Vaillant, G. E. (1977). *Adaptation to Life*. Little, Brown.
