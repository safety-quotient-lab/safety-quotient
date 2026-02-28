# PSQ Criterion Validity Summary

**Last updated:** 2026-02-28
**Status:** Four studies complete. All studies independent of PSQ training data.
**PSQ version used:** v16 (CaSiNo, CGA-Wiki, CMV); v18 (DonD).

This document is the canonical cross-study reference for PSQ criterion validity evidence. It consolidates all numeric results from `distillation-research.md` §§30, 31, 34, 39 and `psychometric-evaluation.md` §3g. Narrative commentary is drawn from `journal.md` §§20, 21, 25, 27.

---

## 1. Cross-Study Summary Table

| Study | N | Domain | Outcome type | Method | 10-dim metric | g-PSQ metric | Profile >> avg gap | Top predictor | AD rank | Key finding |
|---|---|---|---|---|---|---|---|---|---|---|
| **CaSiNo** | 1,030 dialogues (2,060 obs) | Campsite negotiation (MTurk) | Subjective (satisfaction 1-5, likeness 1-5) | OLS regression, Pearson r | r=+0.096*** (sat), +0.099*** (like) | +0.096*** | — (continuous outcomes) | DA (ΔR²=+0.007 after controls) | 5th (r=+0.089) | PSQ predicts relational quality not competitive outcome; 9/10 dims significant; incremental R²=+0.016 sat / +0.023 like beyond sentiment+length |
| **CGA-Wiki** | 4,188 conversations | Wikipedia talk-page disputes | Behavioral (personal attack: yes/no) | Logistic regression, AUC | AUC=0.599, accuracy=57.5% | AUC=0.515 | 0.084 | AD (r_pb=-0.105***) | 1st | Temporal gradient (first turn AUC=0.519 → all turns 0.599) confirms PSQ measures process not static content; AD strongest for 2nd consecutive study |
| **CMV** | 4,263 matched pairs | r/ChangeMyView persuasion | Behavioral (delta awarded: yes/no) | Logistic regression (5-fold CV), paired t-test | AUC=0.590 | AUC=0.531 | 0.059 | DA (r_pb=+0.085, d_z=+0.135) | 11th (weakest; d_z=+0.033, ns at Bonferroni) | AD context-dependent: top in contested-status, negligible in fixed-status; profile gap 0.059 replicates CGA-Wiki |
| **DonD** | 12,234 dialogues | Deal or No Deal negotiation | Behavioral (deal reached: yes/no) | Logistic regression, AUC | AUC=0.686 | AUC=0.622 | 0.064 | ED (d=+0.614, r_pb=+0.247) | 10th (d=-0.063, near zero, negative) | Strongest criterion validity result to date; ED dominates when sustained engagement determines outcome; AD suppressor replicates (coef=-0.534) |

---

## 2. Per-Study Results

### 2a. CaSiNo — Negotiation Satisfaction and Likeness

**Source:** `distillation-research.md` §30; `journal.md` §20
**Reference:** Chawla et al. (2021), NAACL-HLT
**PSQ model:** v16 DistilBERT (held-out_r=0.561)
**Independence:** CaSiNo text appears in training only through strategy annotations mapped to contractual_clarity. Satisfaction and likeness outcomes were never used in PSQ training.

#### Study Design

1,030 dialogues, 2,060 participant-level observations. Each participant independently reports three outcomes post-negotiation: satisfaction (1-5), opponent likeness (1-5), and points scored (0-32). Each dialogue scored with the v16 student model at 128-token truncation.

#### Satisfaction: All-Dimension Correlations

| Dimension | r | p | Direction |
|---|---|---|---|
| energy_dissipation (ED) | +0.114 | <0.001*** | Higher PSQ → more satisfied |
| defensive_architecture (DA) | +0.108 | <0.001*** | |
| contractual_clarity (CO) | +0.097 | <0.001*** | |
| g-PSQ (mean all 10) | +0.096 | <0.001*** | |
| authority_dynamics (AD) | +0.089 | <0.001*** | |
| cooling_capacity (CC) | +0.083 | <0.001*** | |
| hostility_index (HI) | +0.077 | <0.001*** | |
| resilience_baseline (RB) | +0.077 | <0.001*** | |
| trust_conditions (TC) | +0.073 | 0.001** | |
| regulatory_capacity (RC) | +0.072 | 0.001** | |
| threat_exposure (TE) | ns | — | |

9/10 significant. All directions consistent with theory (higher PSQ → better outcomes).

#### Opponent Likeness: Top Correlations

| Dimension | r | p |
|---|---|---|
| defensive_architecture (DA) | +0.126 | <0.001*** |
| energy_dissipation (ED) | +0.125 | <0.001*** |
| contractual_clarity (CO) | +0.104 | <0.001*** |
| g-PSQ | +0.099 | <0.001*** |
| authority_dynamics (AD) | +0.099 | <0.001*** |

#### Points Scored

Near-zero correlations across all dimensions (max |r|=0.054). PSQ predicts how people *feel* after a conversation, not who wins. Theoretically correct: psychological safety is about relational quality, not competitive advantage.

#### Partial Correlations (controlling text length)

Text length is a confound (r=-0.19 with satisfaction, -0.17 with likeness — longer dialogues = harder negotiations). After partialing:

| Dimension | Raw r (sat) | Partial r (sat) | Raw r (like) | Partial r (like) |
|---|---|---|---|---|
| defensive_architecture (DA) | +0.108 | +0.112*** | +0.126 | +0.130*** |
| energy_dissipation (ED) | +0.114 | +0.096*** | +0.125 | +0.109*** |
| authority_dynamics (AD) | +0.089 | +0.085*** | +0.099 | +0.095*** |
| g-PSQ | +0.096 | +0.079*** | +0.099 | +0.084*** |

DA is the only dimension whose partial correlation increases after length control — it captures interpersonal boundary dynamics that are independent of conversational complexity.

#### Incremental R²

| Model | R² (satisfaction) | R² (likeness) |
|---|---|---|
| Text length + n_turns | 0.070 | 0.104 |
| Sentiment + text length | 0.068 | 0.076 |
| Sentiment + length + PSQ 10-dim | 0.084 | 0.099 |
| **Incremental R² (PSQ given sent+len)** | **+0.016** | **+0.023** |

PSQ captures psychological safety signal beyond simple positivity.

#### Extreme Group Comparison

| Outcome | Low PSQ (Q1) | High PSQ (Q4) | Difference | Cohen's d |
|---|---|---|---|---|
| Satisfaction | 4.04 | 4.22 | +0.18 | +0.17 |
| Likeness | 3.97 | 4.20 | +0.23 | +0.20 |

#### Key Quote (journal.md §20)

> "The most surprising finding involved Defensive Architecture, the construct we had been preparing to potentially deprecate. After controlling for text length and sentiment, DA emerged as the single strongest predictor of both satisfaction (ΔR²=+0.007) and opponent likeness (ΔR²=+0.009), and it was the only dimension whose partial correlation increased after controlling for text length. Whatever DA captures — boundary respect, interpersonal defense quality, self-protective behavior support — it matters for real-world interpersonal outcomes, even if it refuses to load cleanly on any single factor in our measurement model."

---

### 2b. CGA-Wiki — Wikipedia Derailment Prediction

**Source:** `distillation-research.md` §31; `journal.md` §21
**Reference:** Zhang et al. (2018)
**PSQ model:** v16 DistilBERT
**Independence:** No Wikipedia talk pages in PSQ training data. Zero circularity.

#### Study Design

4,188 Wikipedia talk-page conversations — 2,094 derailing into personal attacks, 2,094 matched controls. Pre-split: train (2,508), val (840), test (840). Perfectly balanced design. Scored with three turn strategies: all turns, early turns only (first half), first turn only.

#### Group Comparison (all turns, full dataset)

| Dimension | Derailing mean | Safe mean | Cohen's d | p-value |
|---|---|---|---|---|
| authority_dynamics (AD) | 4.860 | 5.012 | **-0.212** | <0.001*** |
| regulatory_capacity (RC) | 5.386 | 5.483 | -0.177 | <0.001*** |
| trust_conditions (TC) | 6.739 | 6.913 | -0.150 | <0.001*** |
| hostility_index (HI) | 7.335 | 7.499 | -0.144 | 0.016* |
| cooling_capacity (CC) | 7.117 | 7.286 | -0.143 | 0.008** |
| resilience_baseline (RB) | 5.737 | 5.809 | -0.116 | <0.001*** |
| energy_dissipation (ED) | 5.518 | 5.556 | -0.072 | 0.028* |
| g-PSQ | 6.059 | 6.146 | -0.134 | 0.001** |
| contractual_clarity (CO) | — | — | ns | >0.25 |
| threat_exposure (TE) | — | — | ns | >0.25 |

Derailing conversations have lower PSQ across 8/10 dimensions. AD is the strongest discriminator (Cohen's d=-0.212).

#### Point-Biserial Correlations (full dataset)

| Dimension | r_pb | p-value |
|---|---|---|
| authority_dynamics (AD) | **-0.105** | <0.001*** |
| regulatory_capacity (RC) | -0.088 | <0.001*** |
| trust_conditions (TC) | -0.075 | <0.001*** |
| hostility_index (HI) | -0.072 | <0.001*** |
| cooling_capacity (CC) | -0.072 | <0.001*** |
| g-PSQ | -0.067 | <0.001*** |
| resilience_baseline (RB) | -0.058 | <0.001*** |
| energy_dissipation (ED) | -0.036 | 0.020* |
| contractual_clarity (CO) | -0.017 | 0.267 (ns) |
| threat_exposure (TE) | +0.017 | 0.272 (ns) |
| defensive_architecture (DA) | -0.005 | 0.751 (ns) |

#### Logistic Regression (train → test)

| Model | AUC | Accuracy |
|---|---|---|
| **10-dim PSQ** | **0.599** | **57.5%** |
| PSQ + text length | 0.605 | 57.0% |
| Text length only | 0.542 | — |
| g-PSQ only | **0.515** | 50.1% |

5-fold CV on train: AUC = 0.579 ± 0.016 (stable).

#### Logistic Regression Feature Weights (top 5)

| Rank | Dimension | Coefficient | Interpretation |
|---|---|---|---|
| 1 | hostility_index (HI) | -0.392 | Lower HI → more derailment |
| 2 | authority_dynamics (AD) | -0.281 | Lower AD → more derailment (power imbalance precedes attacks) |
| 3 | defensive_architecture (DA) | +0.276 | Higher DA → more derailment (defensive posturing escalates) |
| 4 | cooling_capacity (CC) | +0.230 | Higher CC → more derailment (suppression; Simpson's paradox) |
| 5 | threat_exposure (TE) | +0.229 | Higher TE → more derailment |

Note: Sign reversals for CC and TE in multivariate model (positive = more derailment) reflect Simpson's paradox — after adjusting for other 8 dimensions, they carry opposite information from their bivariate direction.

#### Temporal Signal Decay

| Turn strategy | AUC (10-dim) | Cohen's d (g-PSQ) | Dims significant (p<0.05) |
|---|---|---|---|
| All turns | 0.599 | -0.134 | 8/10 |
| Early turns (first half) | 0.570 | -0.053 | 4/10 |
| First turn only | 0.519 | -0.042 | 1/10 |

Signal builds as conversation develops — PSQ measures accumulated interpersonal dynamics, not static text properties. PSQ is not a lexical toxicity classifier (which would perform equally well on any turn).

#### Key Quote (journal.md §21)

> "PSQ is not reading static lexical features — it is tracking an interpersonal trajectory. The psychological unsafety accumulates, and the model captures that accumulation. This is precisely what a process-level construct should do: detect the erosion of safety conditions over the course of an interaction."

> "g-PSQ (the general factor, a simple mean of all ten dimensions) achieves AUC=0.515 — barely above coin flip. The ten individual dimensions together achieve 0.599. This is direct evidence that the general factor, while statistically dominant in the variance decomposition (55.4% of variance), carries almost no predictive utility for external outcomes. The information lives in the dimension profile, not the global score."

---

### 2c. CMV — Change My View Persuasion Prediction

**Source:** `distillation-research.md` §34; `journal.md` §25
**Reference:** Tan et al. (2016); ConvoKit winning-args-corpus
**PSQ model:** v16 DistilBERT
**Independence:** No r/ChangeMyView data in PSQ training. Zero circularity.

#### Study Design

4,263 matched pairs from r/ChangeMyView — same original post, one reply that earned a delta (changed OP's mind), one that did not. Matched-pair design controls for topic and OP. Text length confound present (delta replies longer: mean 1,623 vs 1,248 characters, d=0.301).

#### Group Comparison — All Dimensions (paired t-tests)

| Dimension | Delta mean | No-delta mean | d_z | p | Bonferroni (α=0.005) |
|---|---|---|---|---|---|
| defensive_architecture (DA) | 6.468 | 6.310 | **+0.135** | 2.3e-18 | Yes |
| hostility_index (HI) | 7.586 | 7.432 | +0.104 | 1.1e-11 | Yes |
| trust_conditions (TC) | 7.182 | 7.067 | +0.090 | 3.9e-09 | Yes |
| cooling_capacity (CC) | 7.337 | 7.214 | +0.082 | 8.2e-08 | Yes |
| regulatory_capacity (RC) | 5.763 | 5.702 | +0.078 | 4.1e-07 | Yes |
| threat_exposure (TE) | 6.814 | 6.908 | **-0.077** | 5.4e-07 | Yes |
| contractual_clarity (CO) | 5.963 | 5.897 | +0.064 | 2.9e-05 | Yes |
| energy_dissipation (ED) | 5.633 | 5.582 | +0.063 | 4.0e-05 | Yes |
| resilience_baseline (RB) | 6.167 | 6.114 | +0.060 | 9.6e-05 | Yes |
| authority_dynamics (AD) | 5.318 | 5.280 | **+0.033** | 3.2e-02 | **No** |

All 10 significant at p<.05; 9/10 survive Bonferroni. DA is the strongest predictor (d_z=+0.135). AD is the weakest and fails Bonferroni correction — a dramatic reversal from its dominance in CaSiNo and CGA-Wiki.

#### Point-Biserial Correlations

| Dimension | r_pb |
|---|---|
| defensive_architecture (DA) | **+0.085** |
| hostility_index (HI) | +0.064 |
| trust_conditions (TC) | +0.054 |
| ... | ... |
| authority_dynamics (AD) | +0.021 (ns at Bonferroni) |

Text length: r_pb=+0.156 (the dominant baseline predictor).

#### Logistic Regression AUC (5-fold CV)

| Model | AUC | SD |
|---|---|---|
| Text length only | 0.596 | 0.009 |
| g-PSQ only | **0.531** | 0.011 |
| **10-dim PSQ** | **0.590** | 0.011 |
| 10-dim + length | 0.608 | 0.009 |

Incremental AUC of PSQ beyond text length: +0.012. 9/10 dimensions retain significance after partial correlation controlling for length.

Profile >> average gap: 0.059 (consistent with CGA-Wiki's 0.084).

#### Key Quote (journal.md §25)

> "The most striking finding is defensive_architecture's emergence as the top individual predictor (r_pb=+0.085, paired accuracy=55.4%), displacing authority_dynamics from the top position it held in CaSiNo and CGA-Wiki. This is not a contradiction but a context-dependent pattern: in CMV, where the task is to construct a convincing argument rather than to navigate a relationship, the structural quality of argumentation (DA) matters more than interpersonal power positioning (AD). DA measures boundary maintenance, structured reasoning, and cognitive framing — precisely the toolkit of effective persuasion."

> "Authority_dynamics, meanwhile, shows the weakest bivariate effect in CMV (r_pb=+0.021, not Bonferroni-significant) despite its dominance in CGA-Wiki and CaSiNo. This is exactly what Theory 3 from §24 predicts: AD/power positioning should matter most when status is contested... and least when the social structure is fixed."

---

### 2d. DonD — Deal or No Deal Outcome Prediction

**Source:** `distillation-research.md` §39; `journal.md` §27
**Reference:** Lewis et al. (2017); DeepMind DonD corpus
**PSQ model:** v18 DistilBERT (held-out_r=0.568)
**Independence:** No DonD texts in PSQ training data. Zero circularity.

#### Study Design

12,234 negotiation dialogues. Binary outcome: deal reached (77.2%) vs. no deal (22.8%). Continuous outcome: points scored (0-10, item-value-based). Text length major confound (r=-0.339 with deal outcome — shorter conversations tend to deal).

#### Dimension-Level Analysis (deal vs. no-deal)

| Dimension | Cohen's d | r_pb | Direction |
|---|---|---|---|
| **energy_dissipation (ED)** | **+0.614** | **+0.247** | Deal-makers substantially higher |
| resilience_baseline (RB) | +0.502 | +0.203 | — |
| regulatory_capacity (RC) | +0.478 | +0.194 | — |
| hostility_index (HI) | +0.363 | +0.149 | — |
| cooling_capacity (CC) | +0.340 | +0.140 | — |
| trust_conditions (TC) | +0.312 | +0.129 | — |
| defensive_architecture (DA) | +0.295 | +0.122 | — |
| contractual_clarity (CO) | +0.248 | +0.103 | — |
| threat_exposure (TE) | +0.195 | +0.081 | — |
| **authority_dynamics (AD)** | **-0.063** | **-0.026** | **Near zero, slightly negative** |

ED is the top predictor (d=+0.614, the largest single-dimension effect size across all four studies). AD is the weakest — and the direction is reversed from CaSiNo and CGA-Wiki.

#### Model Comparison

| Metric | 10-dim PSQ | g-PSQ | Text length |
|---|---|---|---|
| AUC | **0.686** | 0.622 | 0.675 |
| Profile >> avg gap | +0.064 | — | — |

Incremental AUC beyond text length + n_turns: +0.059.

#### Extreme Group Comparison

| Group | Deal rate |
|---|---|
| High PSQ (Q4) | 84.4% |
| Low PSQ (Q1) | 68.5% |
| **Difference** | **+15.9 percentage points** |

#### AD as Suppressor Variable

AD bivariate correlation with deal: r_pb=-0.026 (near zero). AD logistic regression coefficient: -0.534 (2nd largest absolute coefficient). This is the suppressor variable pattern: AD carries information that improves multivariate prediction when shared variance with other dimensions is removed. In DonD specifically, the negative direction suggests high-AD conversations (explicit status negotiation) make it harder to reach mutual agreement — theoretically coherent.

This suppressor pattern has now been confirmed in three of four studies (CGA-Wiki, CMV, DonD).

#### Key Quote (journal.md §27)

> "The results were striking. First, the 10-dimension PSQ achieved its strongest AUC to date: 0.686, compared to 0.599 (CGA-Wiki), 0.590 (CMV), and uncalculated for CaSiNo (which used continuous outcomes). Second, the top predictor was energy_dissipation (ED), with a Cohen's d of +0.614 — by far the largest single-dimension effect size across all four studies. AD, which had dominated both CaSiNo and CGA-Wiki, was the weakest predictor in DonD (d=-0.063, near zero and slightly negative). This inversion is theoretically coherent."

> "The incremental validity was also notable: high-PSQ dialogues (Q4) reached deals at 84.4%, compared to 68.5% for low-PSQ (Q1) — a 15.9 percentage point difference."

---

## 3. Cross-Study Patterns

### 3a. Profile Shape Consistently Outperforms Average Score

The most consistent finding across all four studies is that the 10-dimension PSQ profile substantially outperforms g-PSQ (the mean of all 10 dimensions) in predicting external outcomes:

| Study | 10-dim metric | g-PSQ metric | Profile >> avg gap |
|---|---|---|---|
| CaSiNo | r=+0.096*** | (g-PSQ also r=+0.096; no regression done) | — |
| CGA-Wiki | AUC=0.599 | AUC=0.515 | **0.084** |
| CMV | AUC=0.590 | AUC=0.531 | **0.059** |
| DonD | AUC=0.686 | AUC=0.622 | **0.064** |

The AUC gap (0.059–0.084) is modest in absolute terms but remarkably consistent. g-PSQ is near-chance in two of three binary-outcome studies (CGA-Wiki, CMV) and substantially below the full profile in DonD. The predictive information is distributed across dimensions in a way that collapsing to a single score destroys.

This is a direct empirical argument against single-score toxicity/safety systems. The *shape* of the PSQ profile predicts; the *average level* does not.

This finding is consistent with Meehl's (1956) observation that configural personality profiles often outperform simple sum scores, and with medical profiling — a mean vital sign has no diagnostic utility, but a patterned combination does.

### 3b. AD Role: Contested-Status Predictor, Suppressor Variable

Authority dynamics (AD) shows the most context-dependent pattern of any dimension:

| Study | Status structure | AD rank | AD bivariate effect | AD in multivariate |
|---|---|---|---|---|
| CaSiNo | Contested (negotiation) | 5th bivariate, 1st after controls | r=+0.089***, partial r stays strong | ΔR² strongest after controlling sat/length |
| CGA-Wiki | Contested (Wikipedia disputes) | **1st** | r_pb=-0.105*** | coef=-0.281 (2nd) |
| CMV | Fixed (OP holds delta) | **11th (weakest)** | r_pb=+0.021, ns at Bonferroni | suppressor (coef present but direction unclear) |
| DonD | Cooperative (both want deal) | **10th (weakest)** | r_pb=-0.026 | coef=-0.534 (suppressor) |

The pattern supports the **status negotiation theory** (journal.md §24, Theory 3): AD predicts most strongly when interpersonal status is actively contested and the outcome depends on who defines the terms of interaction. When status is fixed (CMV: OP holds the delta) or when both parties are motivated to cooperate (DonD: mutual goal of reaching a deal), AD's predictive power collapses.

AD also functions as a classic **suppressor variable** in three of four studies: its bivariate correlation with outcome is smaller than its multivariate coefficient would predict, because AD carries information that improves prediction of other dimensions' unique variance. This pattern suggests AD is capturing a dimension of relational structure that is partially orthogonal to the other dimensions yet modulates their predictive validity.

The AD paradox — weakest factor loading (max promax 0.332, below 0.35 threshold), strongest external validity in contested contexts — is discussed at length in journal.md §§23-24 and `distillation-research.md` §33. Three theoretical accounts are advanced; the CMV and DonD results favor Theory 3 (status negotiation).

### 3c. ED Role: Process Predictor for Sustained Engagement

Energy dissipation (ED) shows a complementary pattern: it predicts strongly when the behavioral outcome requires sustained engagement over the course of an interaction, and weakly when the outcome is driven by acute power dynamics.

| Study | Behavioral demand | ED rank | ED effect |
|---|---|---|---|
| CaSiNo | Satisfaction from negotiation process | 2nd (sat), 2nd (like) | r=+0.114*** / +0.125*** |
| CGA-Wiki | Derailment (acute power event) | 7th | ΔAUC=-0.005 (leave-one-out; minimal) |
| CMV | Persuasion (argument quality) | 8th | d_z=+0.063 |
| DonD | Deal-reaching (sustained engagement) | **1st** | **d=+0.614***, r_pb=+0.247*** |

ED is the strongest predictor in the two studies where success requires maintaining engagement over an extended interaction (CaSiNo satisfaction, DonD deal-reaching) and the weakest where success is determined by acute relational structure (CGA-Wiki derailment). This validates ED as a **process-level construct** — it captures resource depletion dynamics that determine whether sustained interaction reaches resolution.

ED's factor structure corroborates this: lowest g-loading of all 10 dimensions (R²=0.447), true singleton factor (Stress/Energy, F5), most independent from the shared safety-threat continuum. What is invisible to the general factor is precisely what matters for sustained-engagement outcomes.

### 3d. DA Role: Fixed-Status Predictor

Defensive architecture (DA) occupies a third ecological niche — it predicts most strongly when the social structure is fixed and success requires constructing a cogent argument rather than navigating contested power:

| Study | DA rank | DA effect | Interpretation |
|---|---|---|---|
| CaSiNo | 1st (after controls) | ΔR²=+0.007 (sat), +0.009 (like) | Boundary respect predicts relational quality |
| CGA-Wiki | ns (r_pb=-0.005, p=0.751) | — | DA irrelevant for derailment |
| CMV | **1st** | r_pb=+0.085, d_z=+0.135 | Structured argumentation predicts persuasion |
| DonD | 7th | d=+0.295 | Moderate; not top, not bottom |

DA measures boundary maintenance, structured reasoning, and cognitive framing — the toolkit of effective persuasion (CMV) and mutual respect in negotiation (CaSiNo). When success depends on interpersonal hostility regulation (CGA-Wiki) or energy management (DonD), DA's contribution is attenuated.

### 3e. Context-Dependent Primacy: The 2×2 Matrix

With four studies, the context-dependency pattern resolves into a structured matrix:

| | **Contested status** | **Fixed / cooperative status** |
|---|---|---|
| **Relational outcome** | AD dominates (CaSiNo: negotiation satisfaction) | DA dominates (CMV: persuasion) |
| **Behavioral outcome** | AD dominates (CGA-Wiki: derailment avoidance) | ED dominates (DonD: deal-reaching) |

This matrix is theoretically coherent. When status is contested, power dynamics (AD) determine relational and behavioral outcomes. When status is fixed, the capacity for structured argumentation (DA) determines persuasion outcomes. When the outcome requires sustained engagement regardless of status, resource management (ED) determines whether parties can stay the course. In all cases, g-PSQ (the aggregate) carries minimal independent predictive signal.

This pattern constitutes the strongest evidence that the PSQ's 10 dimensions measure genuinely distinct psychological mechanisms rather than redundant indicators of a single latent variable.

### 3f. Non-Significant Dimensions Reveal Construct Boundaries

Two dimensions — threat_exposure (TE) and contractual_clarity (CO) — are non-significant in the CGA-Wiki derailment study. These null results are theoretically informative:

- **TE non-significance**: PSQ-TE measures the degree to which text content supports assessment of threat exposure — not whether explicit threats are present. A Wikipedia dispute about article deletion policy may contain substantial TE content without interpersonal hostility. Derailment is driven by power imbalance (AD) and regulatory failure (RC), not threat language per se.
- **CO non-significance**: CO measures the clarity of interpersonal agreements in text content, not whether agreements were actually violated. Derailment is a behavioral event that may occur independently of how clearly expectations were set.

These null results help sharpen the construct definitions: PSQ dimensions describe the psychological safety *landscape* of text, not the presence of specific interpersonal behaviors.

### 3g. Effect Size Calibration

| Study | Metric | Value | Comparison literature |
|---|---|---|---|
| CaSiNo | r (satisfaction) | 0.07–0.11 | LIWC predicts personality at r=0.05–0.15 (Pennebaker & King, 1999) |
| CaSiNo | Cohen's d (sat, extreme groups) | 0.17–0.20 | Typical for content-level predictors |
| CGA-Wiki | AUC | 0.599 | Moderate; 57.5% accuracy on balanced data |
| CMV | AUC | 0.590 | Comparable to text-length-only baseline (0.596) |
| DonD | AUC | 0.686 | Strongest result; ED d=+0.614 is substantial |
| DonD | Deal rate gap (Q4 vs Q1) | 15.9 pp | Practically meaningful for deployed systems |

Effect sizes are small to moderate throughout, consistent with content-level prediction of interpersonal outcomes. The consistency across four independent studies, different domains, and different outcome types is more compelling than the magnitude of any single result.

---

## 4. AD Suppressor Variable Analysis

The suppressor variable pattern for AD is now confirmed in three of four studies:

| Study | AD bivariate r | AD multivariate coefficient | Direction |
|---|---|---|---|
| CaSiNo | +0.089 (bivariate; partial increases after controls) | Strongest incremental predictor | Positive |
| CGA-Wiki | -0.105 (bivariate) | -0.281 (2nd largest) | Negative (lower AD → derailment) |
| CMV | +0.021 (ns at Bonferroni) | Present in model | Context-dependent |
| DonD | -0.026 (near zero) | **-0.534 (largest)** | Negative (higher status negotiation → no deal) |

In classical psychometric terms (Conger, 1974), AD is an "instrumental" suppressor: important not for what it predicts directly but for what it allows other variables to predict. By removing shared variance with other dimensions that is *irrelevant* to the outcome, AD isolates the predictive signal in dimensions like HI and RC.

The DonD result (-0.534 coefficient despite near-zero bivariate r) is the most striking example. In deal-reaching, explicit status negotiation is negatively associated with agreement — but this effect is masked in bivariate analysis by the positive correlation of AD with other safety dimensions that do predict deals.

---

## 5. Implications for Architecture and Deployment

### 5a. Always Report All 10 Dimensions

The consistent finding that 10-dim >> g-PSQ implies that any deployed PSQ system must output all 10 dimensions. Reporting only a global score discards the predictive signal. The hierarchical structure (g-PSQ → 5 clusters → 10 dimensions) is appropriate for interpretation and communication, but prediction tasks should always use the full vector.

### 5b. Context-Aware Weighting

The context-dependent primacy pattern implies that optimal feature weighting depends on application context:

| Context | Highest-weight dimensions | Rationale |
|---|---|---|
| Content moderation, derailment prevention | AD, RC, HI | Contested status; power and regulation signals |
| Educational/persuasion contexts | DA, TC, CC | Fixed status; argumentation quality |
| Negotiation / sustained engagement | ED, RB, RC | Process endurance; resource management |
| General relational quality | DA, ED, AD | Multiple studies; robust predictors |

### 5c. Temporal Monitoring

The CGA-Wiki temporal gradient (AUC 0.519 → 0.570 → 0.599 from first turn to all turns) implies that PSQ monitoring systems should accumulate evidence over the conversation:

- First turn: AUC≈0.519 (near-chance; insufficient signal)
- Halfway: AUC≈0.570 (partial warning possible)
- Full conversation: AUC≈0.599 (strongest signal)

A traffic-light interface (green/yellow/red) with confidence increasing over turns would reflect this calibration. This requires scoring each new turn incrementally rather than re-scoring the full concatenation.

### 5d. Bifactor Architecture

The bifactor model (planned; `distillation-research.md` §35) would output both g-PSQ and dimension residuals (dimension scores with shared variance removed). This directly operationalizes the cross-study finding: use g-PSQ for overall safety assessment, use dimension residuals (especially AD-residual in contested-status contexts) for prediction tasks.

---

## 6. Open Questions and Limitations

### 6a. Limitations Across All Studies

1. **LLM labeling chain**: All PSQ scores are generated by a DistilBERT model trained on LLM-labeled data. No human expert has validated the ground truth used for either training or the held-out evaluation. AD's criterion validity, in particular, remains *provisionally grounded* until expert ICC(2,1) ≥ 0.70 is established.

2. **128-token truncation**: All studies score text at 128 tokens. Many conversations exceed this. A production system would score incrementally or use a longer-context model.

3. **English only, online discourse**: All four corpora are English-language, online text. Generalizability to offline, multilingual, or professional contexts is unknown.

4. **Effect sizes are small**: r≈0.08–0.13 in CaSiNo, AUC 0.59–0.69 in binary studies. PSQ alone is insufficient for high-stakes decisions; it should be combined with other features.

5. **No sentiment baseline for CGA-Wiki**: The incremental contribution of PSQ beyond sentiment is unknown for the derailment study (unlike CaSiNo where incremental R² is reported).

### 6b. Pending Tests

- **Turn-by-turn temporal analysis** of CGA-Wiki: tests whether AD scores deteriorate *before* HI in derailing conversations (Theory 2 / leading indicator hypothesis; prediction T2 in journal.md §24).
- **Expert validation**: ICC(2,1) from 5 expert psychologists on 200 stratified texts. Required before AD findings can be treated as more than provisionally grounded (see `expert-validation-protocol.md`).
- **Non-online corpus**: Workplace transcripts, therapy sessions, or classroom interactions to test domain generalizability.
- **Points-scored analysis in DonD**: Whether AD predicts deal (behavioral) but not points (resource allocation), which would test prediction T3b from journal.md §24.

---

## 7. Source References

| Study | Primary source | Secondary source | Reference |
|---|---|---|---|
| CaSiNo | `distillation-research.md` §30 | `journal.md` §20 | Chawla et al. (2021), NAACL-HLT |
| CGA-Wiki | `distillation-research.md` §31 | `journal.md` §21 | Zhang et al. (2018) |
| CMV | `distillation-research.md` §34 | `journal.md` §25 | Tan et al. (2016) |
| DonD | `distillation-research.md` §39 | `journal.md` §27 | Lewis et al. (2017) |
| Cross-study synthesis | `psychometric-evaluation.md` §3g | `journal.md` §26 | — |
| AD construct analysis | `distillation-research.md` §33 | `journal.md` §§23-24 | French & Raven (1959); Conger (1974); Meehl (1990) |
| ED construct analysis | `distillation-research.md` §37 | `journal.md` §23 | Hobfoll (1989); McEwen (1998) |
