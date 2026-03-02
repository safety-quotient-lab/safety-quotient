# PSQ Criterion Validity Summary

**Last updated:** 2026-03-01
**Status:** Four studies complete. All studies independent of PSQ training data.
**PSQ version used:** v16 (CaSiNo, CGA-Wiki); v23 (CMV rerun 2026-02-28, DonD rerun 2026-02-28).

This document is the canonical cross-study reference for PSQ criterion validity evidence. It consolidates all numeric results from `distillation-research.md` §§30, 31, 34, 39 and `psychometric-evaluation.md` §3g. Narrative commentary is drawn from `journal.md` §§20, 21, 25, 27.

**Citation conventions:** This document follows APA 7th edition formatting. Test statistics are italicized (*r*, *p*, *d*, *N*), leading zeros are suppressed on *p*-values, and degrees of freedom are included where available.

---

## 1. Cross-Study Summary Table

| Study | *N* | Domain | Outcome type | Method | 10-dim metric | g-PSQ metric | Profile >> avg gap | Top predictor | AD rank | Key finding |
|---|---|---|---|---|---|---|---|---|---|---|
| **CaSiNo** | 1,030 dialogues (2,060 obs) | Campsite negotiation (MTurk) | Subjective (satisfaction 1–5, likeness 1–5) | OLS regression, Pearson *r* | *r* = .096, *p* < .001 (sat); *r* = .099, *p* < .001 (like) | .096 | — (continuous outcomes) | DA (Δ*R*² = .007 after controls) | 5th (*r* = .089) | PSQ predicts relational quality not competitive outcome; 9/10 dims significant; incremental *R*² = .016 sat / .023 like beyond sentiment + length |
| **CGA-Wiki** | 4,188 conversations | Wikipedia talk-page disputes | Behavioral (personal attack: yes/no) | Logistic regression, AUC | AUC = .599, accuracy = 57.5% | AUC = .515 | .084 | AD (*r*_pb = −.105, *p* < .001) | 1st | Temporal gradient (first turn AUC = .519 → all turns .599) confirms PSQ measures process not static content; AD strongest for 2nd consecutive study |
| **CMV** | 4,263 matched pairs | r/ChangeMyView persuasion | Behavioral (delta awarded: yes/no) | Logistic regression (5-fold CV), paired *t*-test | AUC = .5549 | AUC = .5227 | .032 | DA (*r*_pb = .059, *p* < .001) | 11th (weakest; *d*_z = .054, *p* = .028, ns after Bonferroni) | v23 rerun: DA still top; 7/10 dims significant; CO not significant (*p* = .155); profile gap .032 replicates pattern |
| **DonD** | 12,234 dialogues | Deal or No Deal negotiation | Behavioral (deal reached: yes/no) | Logistic regression, AUC | AUC = .732 (5-CV: .723 ± .010) | AUC = .700 | .032 | TE (*d* = .801, *r*_pb = .315) | 10th bivariate (*d* = .336, *r*_pb = .138), suppressor in multivariate (coef = −.746) | v23 rerun: +.046 AUC vs v18; TE is now top (was ED) — TE held-out improved from .492 → .800 with v23; T3b confirmed (AD+deal, AD−points); Q4/Q1 gap 88.5%/59.7% = 28.7pp |

---

## 2. Per-Study Results

### 2a. CaSiNo — Negotiation Satisfaction and Likeness

**Source:** `distillation-research.md` §30; `journal.md` §20
**Reference:** Chawla et al. (2021)
**PSQ model:** v16 DistilBERT (held-out *r* = .561)
**Independence:** CaSiNo text appears in training only through strategy annotations mapped to contractual_clarity. Satisfaction and likeness outcomes were never used in PSQ training.

#### Study Design

*N* = 1,030 dialogues (2,060 participant-level observations). Each participant independently reports three outcomes post-negotiation: satisfaction (1–5), opponent likeness (1–5), and points scored (0–32). Each dialogue scored with the v16 student model at 128-token truncation.

#### Satisfaction: All-Dimension Correlations

| Dimension | *r* | *p* | Direction |
|---|---|---|---|
| energy_dissipation (ED) | .114 | < .001 | Higher PSQ → more satisfied |
| defensive_architecture (DA) | .108 | < .001 | |
| contractual_clarity (CO) | .097 | < .001 | |
| g-PSQ (mean all 10) | .096 | < .001 | |
| authority_dynamics (AD) | .089 | < .001 | |
| cooling_capacity (CC) | .083 | < .001 | |
| hostility_index (HI) | .077 | < .001 | |
| resilience_baseline (RB) | .077 | < .001 | |
| trust_conditions (TC) | .073 | .001 | |
| regulatory_capacity (RC) | .072 | .001 | |
| threat_exposure (TE) | ns | — | |

Nine of 10 dimensions significant. All directions consistent with theory (higher PSQ → better outcomes).

#### Opponent Likeness: Top Correlations

| Dimension | *r* | *p* |
|---|---|---|
| defensive_architecture (DA) | .126 | < .001 |
| energy_dissipation (ED) | .125 | < .001 |
| contractual_clarity (CO) | .104 | < .001 |
| g-PSQ | .099 | < .001 |
| authority_dynamics (AD) | .099 | < .001 |

#### Points Scored

Near-zero correlations across all dimensions (max |*r*| = .054). PSQ predicts how people *feel* after a conversation, not who wins. Theoretically correct: psychological safety is about relational quality, not competitive advantage.

#### Partial Correlations (controlling text length)

Text length is a confound (*r* = −.19 with satisfaction, −.17 with likeness — longer dialogues = harder negotiations). After partialing:

| Dimension | Raw *r* (sat) | Partial *r* (sat) | Raw *r* (like) | Partial *r* (like) |
|---|---|---|---|---|
| defensive_architecture (DA) | .108 | .112, *p* < .001 | .126 | .130, *p* < .001 |
| energy_dissipation (ED) | .114 | .096, *p* < .001 | .125 | .109, *p* < .001 |
| authority_dynamics (AD) | .089 | .085, *p* < .001 | .099 | .095, *p* < .001 |
| g-PSQ | .096 | .079, *p* < .001 | .099 | .084, *p* < .001 |

DA is the only dimension whose partial correlation increases after length control — it captures interpersonal boundary dynamics that are independent of conversational complexity.

#### Incremental *R*²

| Model | *R*² (satisfaction) | *R*² (likeness) |
|---|---|---|
| Text length + *n* turns | .070 | .104 |
| Sentiment + text length | .068 | .076 |
| Sentiment + length + PSQ 10-dim | .084 | .099 |
| **Incremental *R*² (PSQ given sent+len)** | **.016** | **.023** |

PSQ captures psychological safety signal beyond simple positivity.

#### Extreme Group Comparison

| Outcome | Low PSQ (Q1) | High PSQ (Q4) | Difference | Cohen's *d* |
|---|---|---|---|---|
| Satisfaction | 4.04 | 4.22 | +0.18 | 0.17 |
| Likeness | 3.97 | 4.20 | +0.23 | 0.20 |

#### Key Quote (journal.md §20)

> "The most surprising finding involved Defensive Architecture, the construct we had been preparing to potentially deprecate. After controlling for text length and sentiment, DA emerged as the single strongest predictor of both satisfaction (ΔR²=+0.007) and opponent likeness (ΔR²=+0.009), and it was the only dimension whose partial correlation increased after controlling for text length. Whatever DA captures — boundary respect, interpersonal defense quality, self-protective behavior support — it matters for real-world interpersonal outcomes, even if it refuses to load cleanly on any single factor in our measurement model."

---

### 2b. CGA-Wiki — Wikipedia Derailment Prediction

**Source:** `distillation-research.md` §31; `journal.md` §21
**Reference:** Zhang et al. (2018)
**PSQ model:** v16 DistilBERT
**Independence:** No Wikipedia talk pages in PSQ training data. Zero circularity.

#### Study Design

*N* = 4,188 Wikipedia talk-page conversations — 2,094 derailing into personal attacks, 2,094 matched controls. Pre-split: train (2,508), val (840), test (840). Perfectly balanced design. Scored with three turn strategies: all turns, early turns only (first half), first turn only.

#### Group Comparison (all turns, full dataset)

| Dimension | Derailing *M* | Safe *M* | Cohen's *d* | *p* |
|---|---|---|---|---|
| authority_dynamics (AD) | 4.860 | 5.012 | **−0.212** | < .001 |
| regulatory_capacity (RC) | 5.386 | 5.483 | −0.177 | < .001 |
| trust_conditions (TC) | 6.739 | 6.913 | −0.150 | < .001 |
| hostility_index (HI) | 7.335 | 7.499 | −0.144 | .016 |
| cooling_capacity (CC) | 7.117 | 7.286 | −0.143 | .008 |
| resilience_baseline (RB) | 5.737 | 5.809 | −0.116 | < .001 |
| energy_dissipation (ED) | 5.518 | 5.556 | −0.072 | .028 |
| g-PSQ | 6.059 | 6.146 | −0.134 | .001 |
| contractual_clarity (CO) | — | — | ns | > .25 |
| threat_exposure (TE) | — | — | ns | > .25 |

Derailing conversations have lower PSQ across 8/10 dimensions. AD is the strongest discriminator (Cohen's *d* = −0.212).

#### Point-Biserial Correlations (full dataset)

| Dimension | *r*_pb | *p* |
|---|---|---|
| authority_dynamics (AD) | **−.105** | < .001 |
| regulatory_capacity (RC) | −.088 | < .001 |
| trust_conditions (TC) | −.075 | < .001 |
| hostility_index (HI) | −.072 | < .001 |
| cooling_capacity (CC) | −.072 | < .001 |
| g-PSQ | −.067 | < .001 |
| resilience_baseline (RB) | −.058 | < .001 |
| energy_dissipation (ED) | −.036 | .020 |
| contractual_clarity (CO) | −.017 | .267 |
| threat_exposure (TE) | +.017 | .272 |
| defensive_architecture (DA) | −.005 | .751 |

#### Logistic Regression (train → test)

| Model | AUC | Accuracy |
|---|---|---|
| **10-dim PSQ** | **.599** | **57.5%** |
| PSQ + text length | .605 | 57.0% |
| Text length only | .542 | — |
| g-PSQ only | **.515** | 50.1% |

Five-fold CV on train: AUC = .579 ± .016 (stable).

#### Logistic Regression Feature Weights (top 5)

| Rank | Dimension | Coefficient | Interpretation |
|---|---|---|---|
| 1 | hostility_index (HI) | −0.392 | Lower HI → more derailment |
| 2 | authority_dynamics (AD) | −0.281 | Lower AD → more derailment (power imbalance precedes attacks) |
| 3 | defensive_architecture (DA) | +0.276 | Higher DA → more derailment (defensive posturing escalates) |
| 4 | cooling_capacity (CC) | +0.230 | Higher CC → more derailment (suppression; Simpson's paradox) |
| 5 | threat_exposure (TE) | +0.229 | Higher TE → more derailment |

Sign reversals for CC and TE in the multivariate model (positive = more derailment) reflect Simpson's paradox — after adjusting for the other eight dimensions, they carry opposite information from their bivariate direction.

#### Temporal Signal Decay

| Turn strategy | AUC (10-dim) | Cohen's *d* (g-PSQ) | Dims significant (*p* < .05) |
|---|---|---|---|
| All turns | .599 | −0.134 | 8/10 |
| Early turns (first half) | .570 | −0.053 | 4/10 |
| First turn only | .519 | −0.042 | 1/10 |

Signal builds as conversation develops — PSQ measures accumulated interpersonal dynamics, not static text properties. PSQ is not a lexical toxicity classifier (which would perform equally well on any turn).

#### Key Quote (journal.md §21)

> "PSQ is not reading static lexical features — it is tracking an interpersonal trajectory. The psychological unsafety accumulates, and the model captures that accumulation. This is precisely what a process-level construct should do: detect the erosion of safety conditions over the course of an interaction."

> "g-PSQ (the general factor, a simple mean of all ten dimensions) achieves AUC=0.515 — barely above coin flip. The ten individual dimensions together achieve 0.599. This is direct evidence that the general factor, while statistically dominant in the variance decomposition (55.4% of variance), carries almost no predictive utility for external outcomes. The information lives in the dimension profile, not the global score."

---

### 2c. CMV — Change My View Persuasion Prediction

**Source:** `distillation-research.md` §34; `journal.md` §25
**Reference:** Tan et al. (2016); ConvoKit winning-args-corpus
**PSQ model:** v23 DistilBERT (held-out *r* = .684) — rerun 2026-02-28, corrected 2026-03-01 (max_length fix: 512 → 128). Previous run was v16 (held-out *r* = .561).
**Independence:** No r/ChangeMyView data in PSQ training. Zero circularity.

#### Study Design

*N* = 4,263 matched pairs from r/ChangeMyView — same original post, one reply that earned a delta (changed OP's mind), one that did not. Matched-pair design controls for topic and OP. Text length confound present (delta replies longer: *M* = 1,623 vs. 1,248 characters, *d* = 0.301).

#### Group Comparison — All Dimensions (paired *t*-tests, v23 model)

| Dimension | Delta *M* | No-delta *M* | *d*_z | *p* | Bonferroni (α = .005) |
|---|---|---|---|---|---|
| defensive_architecture (DA) | 5.887 | 5.788 | **0.093** | 1.4 × 10⁻⁹ | Yes |
| hostility_index (HI) | 6.679 | 6.565 | 0.083 | 6.2 × 10⁻⁸ | Yes |
| trust_conditions (TC) | 5.452 | 5.396 | 0.056 | 2.8 × 10⁻⁴ | Yes |
| cooling_capacity (CC) | 6.355 | 6.285 | 0.054 | 4.7 × 10⁻⁴ | Yes |
| authority_dynamics (AD) | 5.352 | 5.299 | 0.054 | 3.8 × 10⁻⁴ | Yes |
| energy_dissipation (ED) | 5.226 | 5.179 | 0.057 | 2.2 × 10⁻⁴ | Yes |
| resilience_baseline (RB) | 5.701 | 5.660 | 0.047 | .002 | Yes |
| regulatory_capacity (RC) | 5.505 | 5.482 | 0.029 | .060 | **No** |
| threat_exposure (TE) | 5.007 | 5.005 | 0.002 | .910 | No |
| contractual_clarity (CO) | 5.747 | 5.723 | 0.025 | .110 | No |

Seven of 10 significant at *p* < .05; 7/10 survive Bonferroni correction. DA remains the strongest predictor (*d*_z = 0.093). CO and TE are not significant — v23 correctly shows TE as near-zero for CMV (previously significant in v16 due to adversarial TE proxy). AD is now significant (*d*_z = 0.054) whereas it was the weakest in v16 (*d*_z = 0.033, failed Bonferroni).

#### Point-Biserial Correlations (v23)

| Dimension | *r*_pb | *p* |
|---|---|---|
| defensive_architecture (DA) | **.059** | < .001 |
| hostility_index (HI) | .049 | < .001 |
| authority_dynamics (AD) | .034 | .002 |
| energy_dissipation (ED) | .035 | .001 |
| trust_conditions (TC) | .035 | .001 |
| cooling_capacity (CC) | .032 | .004 |
| resilience_baseline (RB) | .030 | .006 |
| regulatory_capacity (RC) | .019 | .094 |
| contractual_clarity (CO) | .015 | .160 |
| threat_exposure (TE) | .001 | .930 |

Text length: *r*_pb = .156 (dominant baseline predictor). CO not significant across any analysis.

#### Logistic Regression AUC (5-fold CV, v23)

| Model | AUC | *SD* |
|---|---|---|
| Text length only | .5961 | .009 |
| g-PSQ only | .5227 | .009 |
| **10-dim PSQ** | **.5549** | .009 |
| 10-dim + length | .5985 | .009 |

Incremental AUC of PSQ beyond text length: +.002 (vs. +.012 in v16). Profile >> average gap: .032 (consistent with pattern across studies). DA single-predictor AUC = .5337 (best single dim).

**Comparison with v16:** v16 achieved AUC = .590 for 10-dim. v23 (corrected) shows .5549. The regression is partly attributable to v23's removal of the adversarial TE proxy (which was significantly negative in v16: *d*_z = −0.077) and partly to the max_length correction (v16 ran at 512 tokens; corrected to 128). The v23 results are more trustworthy.

#### Key Quote (journal.md §25)

> "The most striking finding is defensive_architecture's emergence as the top individual predictor (r_pb=+0.085 in v16, +0.059 in v23 rerun), displacing authority_dynamics from the top position it held in CaSiNo and CGA-Wiki. This is not a contradiction but a context-dependent pattern: in CMV, where the task is to construct a convincing argument rather than to navigate a relationship, the structural quality of argumentation (DA) matters more than interpersonal power positioning (AD). DA measures boundary maintenance, structured reasoning, and cognitive framing — precisely the toolkit of effective persuasion."

> "Authority_dynamics, meanwhile, shows the weakest bivariate effect in CMV (r_pb=+0.021, not Bonferroni-significant) despite its dominance in CGA-Wiki and CaSiNo. This is exactly what Theory 3 from §24 predicts: AD/power positioning should matter most when status is contested... and least when the social structure is fixed."

---

### 2d. DonD — Deal or No Deal Outcome Prediction

**Source:** `distillation-research.md` §39; `journal.md` §27
**Reference:** Lewis et al. (2017); DeepMind DonD corpus
**PSQ model:** v23 DistilBERT (held-out *r* = .684) — rerun 2026-02-28. Previous run was v18 (held-out *r* = .568).
**Independence:** No DonD texts in PSQ training data. Zero circularity.

#### Study Design

*N* = 12,234 negotiation dialogues. Binary outcome: deal reached (77.9%) vs. no deal (22.1%). Continuous outcomes: YOU points (0–10, deals only, *n* = 9,530) and joint points (0–20). Text length confound (*r* = −.339 with deal; shorter dialogues → more deals). Train/test/val split: 10,095/1,052/1,087.

#### Dimension-Level Analysis (deal vs. no-deal, v23)

| Dimension | Cohen's *d* | *r*_pb | Partial *r* (length-controlled) |
|---|---|---|---|
| **threat_exposure (TE)** | **0.801** | **.315** | **.203** |
| resilience_baseline (RB) | 0.720 | .286 | .243 |
| trust_conditions (TC) | 0.658 | .264 | .254 |
| cooling_capacity (CC) | 0.596 | .240 | .166 |
| hostility_index (HI) | 0.556 | .225 | .168 |
| regulatory_capacity (RC) | 0.544 | .220 | .218 |
| energy_dissipation (ED) | 0.535 | .217 | .209 |
| contractual_clarity (CO) | 0.468 | .191 | .287 |
| defensive_architecture (DA) | 0.398 | .163 | .240 |
| **authority_dynamics (AD)** | **0.336** | **.138** | **.201** |

All 10 dimensions significant (*p* < .0001). TE is the top bivariate predictor (*d* = 0.801). Critically, **AD is now positive and significant** (*d* = 0.336, *r*_pb = .138) — a reversal from v18 where AD was *d* = −0.063 (near-zero negative). This reflects v23's improved model quality: TE held-out *r* improved from .492 → .800, freeing AD from the suppression role it held when TE was near-noise.

**Note on v18 vs. v23 comparison.** In v18, TE was estimated near-randomly (held-out *r* = .492). All positive TE signal was absorbed by other dimensions (especially ED). With v23 TE at held-out *r* = .800, TE now contributes its genuine signal. After controlling for text length, ED partial *r* (.209) and TE partial *r* (.203) are nearly identical — TE's dominance in raw analysis is partly a length confound (*r*_TE,length = −.420), while ED retains independent signal.

#### Model Comparison (v23)

| Metric | 10-dim PSQ | g-PSQ | Text length | Turn count |
|---|---|---|---|---|
| AUC (train + test) | **.732** | .700 | .675 | .692 |
| 5-fold CV AUC | .723 ± .010 | — | — | — |
| Profile >> avg gap | +.032 | — | — | — |
| Incremental AUC beyond len + turns | **+.061** | — | — | — |

**Temporal signal.** AUC builds from first-turn (.505) to early-turns (.694) to all-turns (.732) — replicating CGA-Wiki's temporal gradient.

#### Extreme Group Comparison (v23)

| Group | Deal rate | YOU pts *M* |
|---|---|---|
| High PSQ (Q4, *n* = 3,059) | **88.5%** | 7.42 |
| Low PSQ (Q1, *n* = 3,059) | **59.7%** | 7.44 |
| **Difference** | **+28.7 pp** | −0.02 (ns) |

High-PSQ negotiators reach deals at 88.5% vs. 59.7% for low-PSQ — a **28.7 pp gap**, substantially larger than v18's 15.9 pp. The points gap is near-zero (*d* = −0.009): high-PSQ negotiators reach deals more often but do not extract more resources when they do.

#### T3b Finding: AD Predicts Deal But Not Points (Confirmed)

This tests prediction T3b from `journal.md` §24 — whether AD distinguishes relational safety (deal/no-deal) from strategic effectiveness (points scored).

| Outcome | AD correlation | Direction | *p* |
|---|---|---|---|
| Deal (binary) | *r*_pb = .138 | Higher AD → more deals | < .001 |
| YOU points (deals only) | *r* = −.070 | Higher AD → **fewer points** | < .001 |
| Joint points (deals only) | *r* = −.097 | Higher AD → **fewer joint points** | < .001 |

Confirmed: AD predicts cooperative behavioral outcome (deal) positively but predicts resource extraction (points) negatively. This confirms the construct interpretation — AD measures relational safety conditions (whether parties stay cooperative), not strategic effectiveness (who extracts more value). The party with high AD behavior commits to reaching agreement but at the cost of assertiveness in resource allocation.

#### AD as Suppressor Variable (v23)

AD bivariate *r*_pb = .138 (positive). AD logistic regression coefficient: −0.746 (largest absolute coefficient in the model, negative). This is a classic suppressor (Conger, 1974): the bivariate and multivariate directions oppose each other because AD shares variance with other positive predictors (particularly CO and DA), and after removing that shared variance, the unique AD component is negatively associated with deals (status assertion → resistance). The suppressor pattern is replicated from v18 (coef = −0.534), but now the bivariate direction has flipped from negative to positive — consistent with improved model quality.

#### Leave-One-Out AUC (v23)

Largest LODO drops: AD (−.007), TE (−.006), CO (−.005). ED, RB, DA each add marginal positive value in LODO. AD's large LODO contribution despite positive bivariate *r* further confirms its suppressor role.

#### Key Quote (journal.md §27 — updated for v23)

> "The v23 rerun produced a striking improvement in criterion validity: AUC=0.732 vs v18's 0.686 (+0.046). The high-PSQ/low-PSQ deal rate gap expanded from 15.9pp to 28.7pp. The story about which dimension leads has changed: TE is now the top bivariate predictor (d=+0.801), whereas ED led in v18 (d=+0.614). This reversal reflects v23's improved TE model (held-out r: 0.492→0.800) — in v18, TE was estimated near-randomly, causing its predictive signal to be absorbed by ED and others. After controlling for text length, ED (partial r=+0.209) and TE (partial r=+0.203) are effectively equal. Both are informative; neither is definitively 'the' process dimension."

> "Most importantly, T3b was confirmed: AD predicts deal (+0.138) but negatively predicts points (−0.070). The party engaging in more status negotiation commits to reaching agreement but concedes more resources. This sharpens AD's construct interpretation: it measures the interpersonal boundary conditions that sustain cooperative engagement, not the strategic advantage of dominant framing."

---

## 3. Cross-Study Patterns

### 3a. Profile Shape Consistently Outperforms Average Score

The most consistent finding across all four studies is that the 10-dimension PSQ profile substantially outperforms g-PSQ (the mean of all 10 dimensions) in predicting external outcomes:

| Study | 10-dim metric | g-PSQ metric | Profile >> avg gap |
|---|---|---|---|
| CaSiNo | *r* = .096, *p* < .001 | (g-PSQ also *r* = .096; no regression done) | — |
| CGA-Wiki | AUC = .599 | AUC = .515 | **.084** |
| CMV | AUC = .5549 (v23, corrected) | AUC = .5227 | **.032** |
| DonD | AUC = .732 (v23) | AUC = .700 | **.032** |

The AUC gap (.032–.084) is modest in absolute terms but remarkably consistent. g-PSQ is near-chance in two of three binary-outcome studies (CGA-Wiki, CMV) and substantially below the full profile in DonD. The predictive information is distributed across dimensions in a way that collapsing to a single score destroys.

This is a direct empirical argument against single-score toxicity/safety systems. The *shape* of the PSQ profile predicts; the *average level* does not.

This finding is consistent with Meehl's (1956) observation that configural personality profiles often outperform simple sum scores, and with medical profiling — a mean vital sign has no diagnostic utility, but a patterned combination does.

### 3b. AD Role: Contested-Status Predictor, Suppressor Variable

Authority dynamics (AD) shows the most context-dependent pattern of any dimension:

| Study | Status structure | AD rank | AD bivariate effect | AD in multivariate |
|---|---|---|---|---|
| CaSiNo | Contested (negotiation) | 5th bivariate, 1st after controls | *r* = .089, *p* < .001; partial *r* stays strong | Δ*R*² strongest after controlling sat/length |
| CGA-Wiki | Contested (Wikipedia disputes) | **1st** | *r*_pb = −.105, *p* < .001 | coef = −0.281 (2nd) |
| CMV | Fixed (OP holds delta) | **11th (weakest)** | *r*_pb = .021, ns at Bonferroni | suppressor (coef present but direction unclear) |
| DonD | Cooperative (both want deal) | **10th (weakest)** | *r*_pb = −.026 | coef = −0.534 (suppressor) |

The pattern supports the **status negotiation theory** (`journal.md` §24, Theory 3; cf. French & Raven, 1959): AD predicts most strongly when interpersonal status is actively contested and the outcome depends on who defines the terms of interaction. When status is fixed (CMV: OP holds the delta) or when both parties are motivated to cooperate (DonD: mutual goal of reaching a deal), AD's predictive power collapses.

AD also functions as a classic **suppressor variable** (Conger, 1974) in three of four studies: its bivariate correlation with outcome is smaller than its multivariate coefficient would predict, because AD carries information that improves prediction of other dimensions' unique variance. This pattern suggests AD is capturing a dimension of relational structure that is partially orthogonal to the other dimensions yet modulates their predictive validity.

The AD paradox — weakest factor loading (max promax = .332, below .35 threshold), strongest external validity in contested contexts — is discussed at length in `journal.md` §§23–24 and `distillation-research.md` §33. Three theoretical accounts are advanced; the CMV and DonD results favor Theory 3 (status negotiation).

### 3c. ED Role: Process Predictor for Sustained Engagement

Energy dissipation (ED) shows a complementary pattern: it predicts strongly when the behavioral outcome requires sustained engagement over the course of an interaction, and weakly when the outcome is driven by acute power dynamics.

| Study | Behavioral demand | ED rank | ED effect |
|---|---|---|---|
| CaSiNo | Satisfaction from negotiation process | 2nd (sat), 2nd (like) | *r* = .114 / .125, *p* < .001 |
| CGA-Wiki | Derailment (acute power event) | 7th | ΔAUC = −.005 (leave-one-out; minimal) |
| CMV | Persuasion (argument quality) | 8th | *d*_z = 0.063 |
| DonD | Deal-reaching (sustained engagement) | **7th (v23); was 1st (v18)** | *d* = 0.535, *r*_pb = .217, partial *r* = .209, all *p* < .001 |

**Note on v23 DonD update.** ED's demotion from 1st to 7th reflects improved TE estimation in v23 (held-out .492 → .800), not a change in ED's genuine signal. After controlling text length, ED partial *r* (.209) ≈ TE partial *r* (.203) — both are process-level predictors in DonD. The v18 "ED tops" finding was partially an artifact of near-random TE estimation.

ED remains the strongest predictor in CaSiNo satisfaction (*r* = .114) and is a key process predictor in DonD (partial *r* = .209). This validates ED as a **process-level construct** — it captures resource depletion dynamics that modulate sustained engagement (cf. Hobfoll, 1989; McEwen, 1998). The construct interpretation stands; only the rank ordering within DonD changed.

ED's factor structure corroborates this: lowest *g*-loading of all 10 dimensions (*R*² = .447), true singleton factor (Stress/Energy, F5), most independent from the shared safety-threat continuum. What is invisible to the general factor is precisely what matters for sustained-engagement outcomes.

### 3d. DA Role: Fixed-Status Predictor

Defensive architecture (DA) occupies a third ecological niche — it predicts most strongly when the social structure is fixed and success requires constructing a cogent argument rather than navigating contested power:

| Study | DA rank | DA effect | Interpretation |
|---|---|---|---|
| CaSiNo | 1st (after controls) | Δ*R*² = .007 (sat), .009 (like) | Boundary respect predicts relational quality |
| CGA-Wiki | ns (*r*_pb = −.005, *p* = .751) | — | DA irrelevant for derailment |
| CMV | **1st** | *r*_pb = .085, *d*_z = 0.135 | Structured argumentation predicts persuasion |
| DonD | 7th | *d* = 0.295 | Moderate; not top, not bottom |

DA measures boundary maintenance, structured reasoning, and cognitive framing — the toolkit of effective persuasion (CMV) and mutual respect in negotiation (CaSiNo). When success depends on interpersonal hostility regulation (CGA-Wiki) or energy management (DonD), DA's contribution is attenuated.

### 3e. Context-Dependent Primacy: The 2×2 Matrix

With four studies, the context-dependency pattern resolves into a structured matrix:

| | **Contested status** | **Fixed / cooperative status** |
|---|---|---|
| **Relational outcome** | AD dominates (CaSiNo: negotiation satisfaction) | DA dominates (CMV: persuasion) |
| **Behavioral outcome** | AD dominates (CGA-Wiki: derailment avoidance) | TE/ED dominate (DonD: deal-reaching; TE bivariate top, ED partial r ≈ equal after length control) |

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
| CaSiNo | *r* (satisfaction) | .07–.11 | LIWC predicts personality at *r* = .05–.15 (Pennebaker & King, 1999) |
| CaSiNo | Cohen's *d* (sat, extreme groups) | 0.17–0.20 | Typical for content-level predictors |
| CGA-Wiki | AUC | .599 | Moderate; 57.5% accuracy on balanced data |
| CMV | AUC | .5549 (corrected) | Below text-length-only baseline (.596) |
| DonD | AUC | **.732** (v23 rerun; was .686 with v18) | Strongest criterion result; +.046 improvement with v23 |
| DonD | Deal rate gap (Q4 vs. Q1) | **28.7 pp** (was 15.9 pp with v18) | Practically meaningful for deployed systems |

Effect sizes are small to moderate throughout, consistent with content-level prediction of interpersonal outcomes. The consistency across four independent studies, different domains, and different outcome types is more compelling than the magnitude of any single result.

---

## 4. AD Suppressor Variable Analysis

The suppressor variable pattern for AD is now confirmed in three of four studies:

| Study | AD bivariate *r* | AD multivariate coefficient | Direction |
|---|---|---|---|
| CaSiNo | .089 (bivariate; partial increases after controls) | Strongest incremental predictor | Positive |
| CGA-Wiki | −.105 (bivariate) | −0.281 (2nd largest) | Negative (lower AD → derailment) |
| CMV | .021 (ns at Bonferroni) | Present in model | Context-dependent |
| DonD | **.138 (positive, sig)** | **−0.746 (largest)** | **Bivariate positive, multivariate negative** (suppressor confirmed; bivariate direction reversed with v23's improved TE) |

In classical psychometric terms (Conger, 1974), AD is an "instrumental" suppressor: important not for what it predicts directly but for what it allows other variables to predict. By removing shared variance with other dimensions that is *irrelevant* to the outcome, AD isolates the predictive signal in dimensions like HI and RC.

The DonD result (−0.534 coefficient despite near-zero bivariate *r*) is the most striking example. In deal-reaching, explicit status negotiation is negatively associated with agreement — but this effect is masked in bivariate analysis by the positive correlation of AD with other safety dimensions that do predict deals.

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

1. **LLM labeling chain.** All PSQ scores are generated by a DistilBERT model trained on LLM-labeled data. No human expert has validated the ground truth used for either training or the held-out evaluation. AD's criterion validity, in particular, remains *provisionally grounded* until expert ICC(2,1) ≥ .70 is established.

2. **128-token truncation.** All studies score text at 128 tokens. Many conversations exceed this. A production system would score incrementally or use a longer-context model.

3. **English only, online discourse.** All four corpora are English-language, online text. Generalizability to offline, multilingual, or professional contexts is unknown.

4. **Effect sizes are small.** *r* ≈ .08–.13 in CaSiNo, AUC .55–.73 in binary studies. PSQ alone is insufficient for high-stakes decisions; it should be combined with other features.

5. **No sentiment baseline for CGA-Wiki.** The incremental contribution of PSQ beyond sentiment is unknown for the derailment study (unlike CaSiNo where incremental *R*² is reported).

### 6b. Pending Tests

- **Turn-by-turn temporal analysis** of CGA-Wiki: tests whether AD scores deteriorate *before* HI in derailing conversations (Theory 2 / leading indicator hypothesis; prediction T2 in journal.md §24).
- **Expert validation**: ICC(2,1) from 5 expert psychologists on 200 stratified texts. Required before AD findings can be treated as more than provisionally grounded (see `expert-validation-protocol.md`).
- **Non-online corpus**: Workplace transcripts, therapy sessions, or classroom interactions to test domain generalizability.
- ~~**Points-scored analysis in DonD**~~: **CONFIRMED (v23 rerun, 2026-02-28).** AD predicts deal (*r*_pb = .138, *p* < .001) but negatively predicts points (*r* = −.070, *p* < .001). Confirms AD measures relational safety (cooperative engagement), not strategic advantage. See §2d for full results.

---

## 7. Source References

### Internal Cross-References

| Study | Primary source | Secondary source |
|---|---|---|
| CaSiNo | `distillation-research.md` §30 | `journal.md` §20 |
| CGA-Wiki | `distillation-research.md` §31 | `journal.md` §21 |
| CMV | `distillation-research.md` §34 | `journal.md` §25 |
| DonD | `distillation-research.md` §39 | `journal.md` §27 |
| Cross-study synthesis | `psychometric-evaluation.md` §3g | `journal.md` §26 |
| AD construct analysis | `distillation-research.md` §33 | `journal.md` §§23–24 |
| ED construct analysis | `distillation-research.md` §37 | `journal.md` §23 |

### References

Chawla, K., Ramirez, J., Clever, R., Lucas, G., May, J., & Gratch, J. (2021). CaSiNo: A corpus of campsite negotiation dialogues for automatic negotiation systems. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 3167–3185). Association for Computational Linguistics.

Conger, A. J. (1974). A revised definition for suppressor variables: A guide to their identification and interpretation. *Educational and Psychological Measurement*, *34*(1), 35–46.

Edmondson, A. (1999). Psychological safety and learning behavior in work teams. *Administrative Science Quarterly*, *44*(2), 350–383.

French, J. R. P., & Raven, B. (1959). The bases of social power. In D. Cartwright (Ed.), *Studies in social power* (pp. 150–167). Institute for Social Research.

Hobfoll, S. E. (1989). Conservation of resources: A new attempt at conceptualizing stress. *American Psychologist*, *44*(3), 513–524.

Lewis, M., Yarats, D., Dauphin, Y. N., Parikh, D., & Batra, D. (2017). Deal or no deal? End-to-end learning for negotiation dialogues. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing* (pp. 2443–2453). Association for Computational Linguistics.

McEwen, B. S. (1998). Protective and damaging effects of stress mediators. *New England Journal of Medicine*, *338*(3), 171–179.

Meehl, P. E. (1956). Wanted—A good cookbook. *American Psychologist*, *11*(6), 263–272.

Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an individual difference. *Journal of Personality and Social Psychology*, *77*(6), 1296–1312.

Tan, C., Niculae, V., Danescu-Niculescu-Mizil, C., & Lee, L. (2016). Winning arguments: Interaction dynamics and persuasion strategies in good-faith online discussions. In *Proceedings of the 25th International Conference on World Wide Web* (pp. 613–624). ACM.

Zhang, J., Chang, J., Danescu-Niculescu-Mizil, C., Dixon, L., Thain, N., Hua, Y., & Taraborelli, D. (2018). Conversations gone awry: Detecting early signs of conversational failure. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics* (Vol. 1, pp. 1350–1361). Association for Computational Linguistics.
