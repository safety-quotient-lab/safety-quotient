# PSQ Metrics Engine — LLM-Based Content Analysis

## Core Concept

Use the ~100 validated instruments from intermediate-state.md as structured rubrics
for an LLM to analyze text content and produce PSQ profiles. The instruments provide
the operationalized constructs; the LLM provides the pattern recognition at scale.

## What Gets Analyzed (Input Types)

- Text communications (emails, messages, chat logs, letters)
- Transcripts (meetings, therapy sessions, interviews, depositions)
- Documents (policies, contracts, organizational documents)
- Social media content
- Legal documents (complaints, rulings, contracts)
- Workplace communications (Slack, Teams, memos)
- Relationship communications
- Any text where psychoemotional dynamics are present

## Instrument Suitability Tiers for Text Analysis

### Tier 1 — High (linguistic/behavioral patterns directly detectable in text)

These instruments measure constructs that manifest clearly in language:

- **Hostility**: Cook-Medley (1954), Buss-Perry (1992), STAXI-2 (Spielberger 1999), Novaco (1994)
  → hostile language, contempt, cynicism, aggressive speech patterns
- **Humor styles**: HSQ (Martin et al. 2003)
  → affiliative, self-enhancing, aggressive, self-defeating humor in text
- **Conflict modes**: TKI (Thomas & Kilmann 1974), ROCI-II (Rahim 1983)
  → competing, collaborating, compromising, avoiding, accommodating patterns
- **Trust/distrust**: Rotter (1967), Organizational Trust Inventory (Cummings & Bromiley 1996)
  → trust language, verification demands, suspicion markers, faith signals
- **Authority/power**: French & Raven (1959), Abusive Supervision (Tepper 2000), MLQ (Bass & Avolio 1995)
  → directive language, deference, power assertions, coercion, inspiration
- **Psychological safety**: Edmondson (1999), PSC-12 (Dollard & Bakker 2010)
  → question-asking, idea-sharing, error-admitting, punishment for speaking up
- **Defense mechanisms**: DSQ (Bond et al. 1983), Vaillant hierarchy (1977)
  → rationalization, projection, denial, displacement, intellectualization in text
- **Emotion regulation/dysregulation**: ERQ (Gross & John 2003), DERS (Gratz & Roemer 2004)
  → emotional escalation patterns, reappraisal, suppression, flooding
- **Contractual clarity**: Psychological Contract (Rousseau 1989/1995)
  → explicitness of expectations, mutuality, unilateral redefinition
- **Psychosocial hazards**: COPSOQ (Kristensen et al. 2005), NAQ (Einarsen et al. 1994)
  → bullying language, exclusion, unreasonable demands, role ambiguity
- **Attachment patterns**: ECR (Brennan et al. 1998)
  → anxious/avoidant communication patterns, proximity-seeking, distancing
- **Love styles**: LAS (Hendrick & Hendrick 1986)
  → eros/storge/ludus/mania/pragma/agape language patterns
- **Moral reasoning**: Kohlberg (1981), DIT (Rest 1979)
  → pre-conventional, conventional, post-conventional reasoning in text

### Tier 2 — Medium (inferable from patterns across multiple texts or longer content)

- **Resilience**: CD-RISC (Connor & Davidson 2003), BRS (Smith et al. 2008)
  → recovery patterns across communications over time, bounce-back language
- **Coping styles**: COPE (Carver et al. 1989), Ways of Coping (Folkman & Lazarus 1988)
  → problem-focused vs emotion-focused patterns, avoidance, support-seeking
- **Self-regulation/discipline**: SCS (Tangney et al. 2004), Grit (Duckworth et al. 2007)
  → follow-through patterns, delayed gratification language, persistence
- **Energy/burnout**: Effort-Recovery (Meijman & Mulder 1998), COR (Hobfoll 1989)
  → exhaustion markers, resource depletion language, recovery absence
- **Organizational change readiness**: ADKAR (Hiatt 2006), Kotter (1996)
  → resistance language, engagement, awareness, commitment markers
- **Self-determination**: SDT (Deci & Ryan 1985)
  → autonomy, competence, relatedness needs expressed or thwarted
- **Psychological capital**: PsyCap (Luthans et al. 2007)
  → efficacy, optimism, hope, resilience language patterns
- **Community/belonging**: SCI (McMillan & Chavis 1986)
  → membership language, influence, integration, shared emotional connection

### Tier 3 — Lower (requires supplementary data or self-report; text alone insufficient)

- **Physiological measures**: allostatic load (McEwen 1998), polyvagal (Porges 1994)
  → can detect reported symptoms, not actual physiological state
- **Neuroscience constructs**: affective neuroscience (Panksepp 1998), somatic markers (Damasio 1994)
  → can detect behavioral descriptions, not neural activity
- **Clinical diagnoses**: SCID, PHQ-9, GAD-7, BDI-II
  → can flag indicators, not diagnose — important ethical boundary
- **Cognitive load**: NASA-TLX (Hart & Staveland 1988)
  → can infer from complexity/confusion markers, not measure directly
- **Psychometric properties**: IRT, signal detection
  → these are meta-instruments, not content-analyzable

## Detector Architecture

Each instrument becomes a "detector" — a structured prompt component:

```
Detector: {instrument_name}
Source: {authors, year}
Construct: {what it measures}
Subscales: {list of subscales/factors}
Indicators: {specific textual markers mapped from instrument items}
Scoring rubric: {how to score, anchored to original instrument's scale}
Output: {score, subscale_scores, evidence[], confidence}
```

## Multi-Pass Analysis Pipeline

### Pass 1 — Content Characterization
- Type of content (email, transcript, policy, etc.)
- Context (workplace, intimate relationship, legal, clinical, etc.)
- Participants and their roles/power positions
- Temporal span (single moment vs. longitudinal)
- Which detectors are applicable to this content type

### Pass 2 — Instrument Analysis (parallelizable)
- Run each applicable Tier 1 detector against the content
- Run applicable Tier 2 detectors if content has sufficient depth/span
- Flag Tier 3 indicators without scoring
- Each detector returns: score, evidence quotes, confidence level

### Pass 3 — Dimension Aggregation
Roll instrument scores up into the 10 PSQ dimensions:

| PSQ Dimension | Primary Instruments | Secondary Instruments |
|---|---|---|
| 1. Threat Exposure | COPSOQ, NAQ, Abusive Supervision | Effort-Reward Imbalance, JDC |
| 2. Regulatory Capacity | ERQ, DERS, CERQ | DBT constructs, Gross Process Model |
| 3. Resilience Baseline | CD-RISC, BRS, RS | Grit, PsyCap resilience subscale |
| 4. Trust Conditions | Rotter ITS, OTI, TQ | Propensity to Trust, PCI (trust element) |
| 5. Hostility Index | Cook-Medley, BPAQ, STAXI-2, NAS | TKI competing mode |
| 6. Cooling Capacity | CPI Crisis Model, Gross reappraisal | Recovery Experience, de-escalation indicators |
| 7. Energy Dissipation | Effort-Recovery, COR, allostatic indicators | Recovery Experience, Flow State |
| 8. Defensive Architecture | DSQ, DMRS, Vaillant hierarchy | TKI, Rahim, NVC indicators |
| 9. Authority Dynamics | French & Raven, MLQ, Abusive Supervision | Power Distance, Milgram indicators |
| 10. Contractual Clarity | PCI, Psychological Breach/Violation | Role ambiguity (COPSOQ subscale) |

### Pass 4 — PSQ Profile Computation
- Protective factors (dimensions 2,3,4,6,8,10) → numerator
- Threat factors (dimensions 1,5,7,9 inverted) → denominator
- Composite PSQ profile with per-dimension scores
- Narrative synthesis: what the profile means in plain language
- Actionable recommendations grounded in the instrument literature

## Ethical Constraints

- Never diagnose — flag indicators only
- Always cite evidence (specific quotes/patterns)
- Confidence scoring is mandatory — no false precision
- Tier 3 constructs are flagged, not scored
- The engine measures the *content*, not the *person* — critical distinction
- Informed consent considerations for any content analyzed
- Cultural context affects instrument validity — must be accounted for

## Implementation Considerations

- Reference library: structured data (YAML/JSON) of all instruments with items/subscales
- Detector templates: parameterized prompts per instrument
- Aggregation weights: configurable per context (clinical vs. workplace vs. relationship)
- Calibration: test against known-scored content where possible
- Output format: structured (JSON) + narrative (markdown)
- API design: content in → PSQ profile out
