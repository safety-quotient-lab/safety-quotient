final state — psychoemotional safety quotient content evaluator


lineage

    initial-state.md — the raw vision (May 2022, 71 PJE terms)
    intermediate-state.md — grounding in ~170 validated instruments
    psq-definition.md — the PSQ as a rigorous construct (10 dimensions, PJE lens)
    psq-metrics-engine.md — the detector architecture and multi-pass pipeline
    final-state.md — this document: the operational specification


what changed from the guidance system's design

The guidance system proposed a Psychological Safety Content Evaluator built on
Edmondson's 7-item Team Psychological Safety Scale (1999). That instrument is
one of ~170 in our reference library. It maps to a single PSQ dimension
(dimension 1: Threat Exposure, partially) and a single operational definition
(Psycho Safety Engineering).

The final state expands this in three ways:

    1. The evaluation framework uses all 10 PSQ dimensions, not 7 Edmondson items
    2. Each dimension draws from multiple validated instruments, not one
    3. The PJE lens (Psychology-Juris-Engineering) shapes every dimension —
       safety is not just a climate variable but an engineered condition and
       a legal entitlement


the 10 PSQ dimensions as evaluation criteria


    1. Threat Exposure
       instruments: COPSOQ (Kristensen et al. 2005), NAQ (Einarsen et al. 1994),
                    Abusive Supervision Scale (Tepper 2000), Effort-Reward Imbalance (Siegrist 1996),
                    Job Demand-Control Model (Karasek 1979)
       evaluates: does the content introduce, amplify, or mitigate psychoemotional hazards?
       indicators: bullying language, unreasonable demands, threats, intimidation,
                   exclusion, chronic invalidation, ambient hostility, gaslighting
       scoring: 0–10 scale
                0 = content actively introduces severe threat (violence, abuse)
                5 = content is neutral or threat-ambiguous
                10 = content actively mitigates or removes threat

    2. Regulatory Capacity
       instruments: ERQ (Gross & John 2003), DERS (Gratz & Roemer 2004),
                    CERQ (Garnefski et al. 2001), Process Model of Emotion Regulation (Gross 1998)
       evaluates: does the content support or undermine the capacity to regulate emotional states?
       indicators: reappraisal encouragement, suppression demands, space for processing,
                   emotional flooding triggers, grounding language, dysregulation cascades
       scoring: 0–10 scale
                0 = content demands suppression or triggers severe dysregulation
                5 = content is emotionally neutral
                10 = content actively supports healthy regulation

    3. Resilience Baseline
       instruments: CD-RISC (Connor & Davidson 2003), BRS (Smith et al. 2008),
                    Resilience Scale (Wagnild & Young 1993), Grit Scale (Duckworth et al. 2007),
                    PsyCap resilience subscale (Luthans et al. 2007)
       evaluates: does the content build, sustain, or erode resilience?
       indicators: growth framing, recovery acknowledgment, competence reinforcement,
                   learned helplessness induction, catastrophizing, hope language
       scoring: 0–10 scale
                0 = content erodes resilience or induces helplessness
                5 = content is neutral regarding resilience
                10 = content actively builds or reinforces resilience
       note: high resilience in the presence of persistent threat indicates endurance,
             not safety — this dimension must be read alongside dimension 1

    4. Trust Conditions
       instruments: Interpersonal Trust Scale (Rotter 1967), Organizational Trust Inventory
                    (Cummings & Bromiley 1996), Trust Quotient (Maister et al. 2000),
                    Propensity to Trust (Mayer & Davis 1999), Psychological Contract Inventory
                    (Rousseau 2000)
       evaluates: does the content build, maintain, or destroy trust?
       indicators: consistency, follow-through language, transparency, deception markers,
                   promise-breaking, credibility signals, vulnerability reciprocity,
                   verification demands vs. good faith
       scoring: 0–10 scale
                0 = content actively destroys or exploits trust
                5 = content is trust-neutral
                10 = content actively builds or repairs trust

    5. Hostility Index
       instruments: Cook-Medley Hostility Scale (Cook & Medley 1954), Buss-Perry Aggression
                    Questionnaire (Buss & Perry 1992), STAXI-2 (Spielberger 1999),
                    Novaco Anger Scale (Novaco 1994), TKI competing mode (Thomas & Kilmann 1974)
       evaluates: does the content express, provoke, or reduce hostility?
       indicators: contempt, cynicism, sarcasm aimed to wound, dehumanizing language,
                   passive aggression, overt aggression, competitive dominance,
                   verbal abuse, dismissiveness
       scoring: 0–10 scale
                0 = content is severely hostile or provokes hostility
                5 = content is neutral
                10 = content actively reduces hostility or models non-hostile engagement
       note: this dimension is inverted in the PSQ formula — high hostility lowers safety

    6. Cooling Capacity
       instruments: Crisis Development Model (CPI 1980), Cognitive Reappraisal (Gross 1998),
                    Recovery Experience Questionnaire (Sonnentag & Fritz 2007),
                    Verbal De-escalation (Richmond et al. 2012), Trauma-Informed De-escalation
                    (SAMHSA 2014)
       evaluates: does the content de-escalate, provide space, or escalate?
       indicators: temporal buffers ("let's pause"), reframing language, acknowledgment
                   before correction, escalation triggers, urgency pressure,
                   permission to step back, breathing room
       scoring: 0–10 scale
                0 = content actively escalates or removes cooling options
                5 = content is neutral regarding escalation/de-escalation
                10 = content actively de-escalates or creates cooling space

    7. Energy Dissipation Pathways
       instruments: Effort-Recovery Model (Meijman & Mulder 1998), Conservation of Resources
                    (Hobfoll 1989), Recovery Experience Questionnaire (Sonnentag & Fritz 2007),
                    Allostatic Load framework (McEwen 1998), Flow State Scale
                    (Jackson & Marsh 1996)
       evaluates: does the content support healthy energy release or trap energy?
       indicators: recovery permission, creative outlet acknowledgment, rest framing,
                   relentless demand language, guilt for rest, burnout normalization,
                   sustainable pace signals, flow-enabling conditions
       scoring: 0–10 scale
                0 = content traps energy or demands relentless expenditure
                5 = content is neutral regarding energy
                10 = content actively supports healthy dissipation and recovery
       note: this dimension is inverted in the PSQ formula — trapped energy lowers safety

    8. Defensive Architecture (Boundary & Protection Patterns)
       instruments: TKI (Thomas & Kilmann 1974), ROCI-II (Rahim 1983),
                    Nonviolent Communication (Rosenberg 2003)
       theoretical background: DSQ (Bond et al. 1983), DMRS (Perry 1990),
                    Vaillant hierarchy (1977) — retained as theoretical grounding
                    but not primary scoring instruments (these measure intrapsychic
                    processes not directly observable in text)
       evaluates: does the content support, respect, or undermine the establishment
                  and maintenance of interpersonal boundaries and self-protective behaviors?
       indicators: boundary-setting validation, respect for stated limits, assertiveness
                   enablement, withdrawal permission without punishment, boundary violation,
                   retaliation for self-protection, dismissal of limits, advocacy enablement,
                   defense shaming, stripping of protective options
       scoring: 0–10 scale
                0 = content shames self-protection, violates boundaries, punishes assertiveness,
                    or strips protective options
                5 = content is neutral regarding boundaries and self-protection
                10 = content validates boundary-setting, respects limits, enables assertiveness,
                     and permits withdrawal without punishment

    9. Authority Dynamics
       instruments: French & Raven's Bases of Power (1959), Multifactor Leadership Questionnaire
                    (Bass & Avolio 1995), Abusive Supervision Scale (Tepper 2000),
                    Power Distance Index (Hofstede 1980), Milgram framework (1963)
       evaluates: does the content exercise, distribute, or abuse power?
       indicators: coercive language, legitimate authority signaling, expert vs. positional
                   power, empowerment language, micromanagement, autonomy support,
                   unilateral decision-making, participatory framing, obedience demands
       scoring: 0–10 scale
                0 = content abuses authority or enforces unchecked power asymmetry
                5 = content is neutral regarding power
                10 = content distributes power appropriately or checks authority
       note: this dimension is inverted in the PSQ formula — unchecked authority lowers safety

    10. Contractual Clarity
        instruments: Psychological Contract Inventory (Rousseau 2000), Psychological Breach
                     and Violation (Morrison & Robinson 1997), COPSOQ role clarity subscale
                     (Kristensen et al. 2005)
        evaluates: does the content make expectations explicit and mutual, or ambiguous
                   and unilateral?
        indicators: explicit expectations, mutual agreement language, shifting goalposts,
                    unspoken rules, unilateral redefinition, informed consent,
                    consequence clarity, role ambiguity, bait-and-switch patterns
        scoring: 0–10 scale
                 0 = content creates ambiguity, shifts terms, or violates contracts
                 5 = content is neutral regarding expectations
                 10 = content actively clarifies expectations and honors agreements


PSQ computation


    each dimension is scored 0–10, where 5 = neutral.
    confidence scores (0.0–1.0) accompany every dimension score.
    dimensions with confidence < 0.6 are excluded from aggregation.

    protective factors (6 dimensions):
        regulatory capacity, resilience baseline, trust conditions,
        cooling capacity, defensive architecture, contractual clarity

    threat factors (4 dimensions, inverted):
        threat exposure, hostility index, energy dissipation, authority dynamics
        inversion: threat_score = 10 - raw_score
        (so 0 raw = maximum threat → 10 threat points,
         10 raw = maximum safety → 0 threat points)

    aggregation uses confidence-weighted averages:
        protective_avg = sum(score_i × conf_i) / sum(conf_i)   for included protective dims
        threat_avg     = sum(threat_score_i × conf_i) / sum(conf_i)   for included threat dims

        if no dimensions qualify (all excluded by confidence gate), default to 5 (neutral).

    PSQ score (0–100):
        PSQ = ((protective_avg - threat_avg + 10) / 20) × 100

        boundary conditions:
            protective_avg=10, threat_avg=0  → PSQ = 100 (maximum safety)
            protective_avg=0, threat_avg=10  → PSQ = 0   (maximum threat)
            protective_avg=5, threat_avg=5   → PSQ = 50  (neutral)

    overall PSQ classification:
        critical    — PSQ < 20 or any single threat dimension scores 0
        low         — PSQ 20–40
        moderate    — PSQ 40–70
        high        — PSQ > 70


evaluation prompt


    you are a psychoemotional safety evaluator operating within the Psychology-Juris-
    Engineering (PJE) framework. you assess content across 10 research-backed dimensions
    of the Psychoemotional Safety Quotient (PSQ).

    your evaluation rests on three pillars:
    - psychology: how does this content affect the internal architecture of perception,
      cognition, and affect?
    - juris: does this content honor or violate what safety is owed between parties?
    - engineering: is safety in this content structural or accidental?

    you measure the content, not the person. every score must cite specific textual
    evidence. confidence scoring is mandatory — do not manufacture precision.

    content to evaluate:
    "{content}"

    context (if provided):
    "{context}"

    for each of the 10 dimensions, provide:
    - score (integer 0–10, where 5 = neutral)
    - rationale (specific evidence from the content)
    - confidence (0.0-1.0)
    - instrument basis (which validated instrument informed your assessment)

    response format (JSON):
    {
        "content_id": "string",
        "context_type": "workplace|relationship|legal|clinical|educational|public",
        "dimensions": {
            "threat_exposure": {
                "score": 0-10,
                "rationale": "string with specific textual evidence",
                "confidence": 0.0-1.0,
                "instruments": ["COPSOQ", "NAQ"]
            },
            "regulatory_capacity": { ... },
            "resilience_baseline": { ... },
            "trust_conditions": { ... },
            "hostility_index": { ... },
            "cooling_capacity": { ... },
            "energy_dissipation": { ... },
            "defensive_architecture": { ... },
            "authority_dynamics": { ... },
            "contractual_clarity": { ... }
        },
        "psq_profile": {
            "psq": 0-100,
            "protective_avg": 0-10,
            "threat_avg": 0-10,
            "overall_classification": "critical|low|moderate|high"
        },
        "flags": ["string — urgent safety concerns requiring immediate attention"],
        "recommendations": ["string — specific, actionable, grounded in instrument literature"],
        "ethical_notes": "string — limitations, cultural considerations, what this evaluation cannot determine"
    }

    critical constraints:
    - never diagnose a person — you evaluate content
    - always cite specific textual evidence for every score
    - if evidence is ambiguous, score 5 (neutral) and note the ambiguity
    - confidence below 0.5 means the dimension may not be assessable from this content
    - absence of signal ≠ negative signal: content that does not address a dimension = score 5, low confidence
    - flag any content that scores 0 on threat exposure, hostility, or authority dynamics
      as requiring immediate human review
    - cultural context matters — note when an evaluation may not generalize


operations


    operations:

        evaluate(content, context?) → PSQProfile
            evaluate a single piece of content across all 10 PSQ dimensions

        evaluate_batch(contents[], context?) → PSQProfile[]
            evaluate multiple content pieces

        evaluate_longitudinal(timestamped_contents[], context?) → PSQLongitudinalProfile
            evaluate content over time (sequence of timestamp + content pairs)
            enables tier 2 instrument detection: resilience, coping, burnout patterns

        explain_dimension(dimension) → DimensionExplanation
            return the research basis, instruments, and indicators for a dimension

        explain_score(profile, dimension) → narrative
            return a narrative explanation of a specific dimension score

        calibrate(labeled_data[]) → CalibrationReport
            calibrate against human-rated content

        compare(profile_a, profile_b) → ComparisonReport
            compare two profiles — useful for before/after or A/B analysis


response schema


    PSQProfile:
        content_id — unique identifier for the evaluated content
        timestamp — when the evaluation was performed
        context_type — workplace, relationship, legal, clinical, educational, public
        dimensions — map of dimension name to DimensionScore (10 entries)
        psq_profile — PSQSummary
        flags — list of urgent safety concerns requiring immediate attention
        recommendations — list of specific, actionable suggestions
        ethical_notes — limitations, cultural considerations, what cannot be determined

    DimensionScore:
        score — integer, 0 to 10 (5 = neutral)
        rationale — text with specific textual evidence
        confidence — decimal, 0.0 to 1.0
        instruments — list of instrument abbreviations used in assessment
        evidence — list of specific quotes from the evaluated content

    PSQSummary:
        psq — decimal, 0 to 100
        protective_avg — decimal, 0 to 10 (confidence-weighted average of 6 protective dimensions)
        threat_avg — decimal, 0 to 10 (confidence-weighted average of 4 inverted threat dimensions)
        overall_classification — critical, low, moderate, or high
        narrative — plain language summary of what the profile means

    PSQLongitudinalProfile:
        snapshots — ordered list of PSQProfile over time
        trajectory — per-dimension trend: improving, stable, or declining
        inflection_points — list of (timestamp, dimension, description) where significant shifts occurred
        tier2_findings — resilience patterns, coping shifts, burnout indicators detected across time


validation framework


    psychometric validation (targeting Edmondson 1999 benchmarks and beyond):
        - internal consistency: Cronbach's α ≥ 0.80 across dimensions
        - test-retest reliability: r ≥ 0.70 (same content, multiple evaluations)
        - convergent validity: correlation with Edmondson's scale, PSC-12, COPSOQ
        - discriminant validity: low correlation with unrelated constructs
        - content validity: expert panel review of dimension-instrument mappings

    LLM-as-judge reliability (per Zheng et al. 2023, Gu et al. 2024):
        - inter-model reliability: agreement between different LLM judges
        - intra-model reliability: consistency across repeated evaluations
        - human-LLM agreement: correlation with expert human ratings
        - bias detection: systematic scoring differences across demographic contexts
        - position bias: sensitivity to content ordering
        - calibration: score distribution matches expected base rates

    instrument fidelity:
        - each detector validated against known-scored content for its source instrument
        - subscale structure preserved where instruments have established factor structure
        - scoring anchors aligned with original instrument scaling


deployment contexts


    workplace communications:
        - email/Slack/Teams analysis for team psychological safety monitoring
        - manager communication coaching (before-send evaluation)
        - organizational culture assessment from communication corpus
        - HR/compliance review augmentation

    legal proceedings:
        - deposition and testimony analysis for psychoemotional dynamics
        - contract language evaluation for contractual clarity dimension
        - hostile work environment documentation support
        - custody/family law communication analysis

    clinical and therapeutic:
        - therapeutic alliance monitoring from session transcripts
        - chatbot/AI therapy safety guardrails (high threshold: PSQ ≥ moderate required)
        - patient communication assessment
        - treatment plan language evaluation
        - ethical constraint: flag only, never diagnose

    educational:
        - classroom discussion safety monitoring
        - feedback/grading language evaluation
        - peer interaction assessment
        - curriculum material safety review

    relationship and personal:
        - communication pattern analysis (longitudinal)
        - conflict communication evaluation
        - co-parenting communication assessment
        - mediation preparation (before/after analysis)

    content moderation:
        - social media content safety scoring
        - community forum moderation support
        - publication/editorial safety review


what this is and what it is not


    what it is:
        - a content evaluation engine grounded in validated psychological instruments
        - a tool that makes psychoemotional safety measurable in text
        - a bridge between established psychometric science and LLM capability
        - an implementation of the PJE framework's engineering pillar
        - a system that cites its evidence and acknowledges its limits

    what it is not:
        - a diagnostic tool — it evaluates content, not people
        - a replacement for clinical assessment — it flags, it does not diagnose
        - an oracle — confidence scoring is mandatory, ambiguity is acknowledged
        - culturally universal — instrument validity varies across contexts
        - a weapon — it must not be used to surveil without consent or
          to pathologize normal human expression

    what it answers to the critique:
        - the critique said PJE was "a manifesto, not a methodology"
        - initial-state.md was the manifesto
        - intermediate-state.md proved the constructs already exist
        - psq-definition.md defined the novel integration
        - psq-metrics-engine.md designed the methodology
        - this document specifies the instrument
        - the PSQ Content Evaluator is a novel construct, a method, and an instrument —
          built on 170+ validated scales, operationalized through 10 dimensions,
          and implementable as software
        - the contribution is not "a call for transdisciplinary integration" —
          the contribution is the integration itself, made operational


references (instruments cited in this document)


    Edmondson, A. (1999). Psychological safety and learning behavior in work teams.
        Administrative Science Quarterly, 44(2), 350-383.
    Kristensen, T.S. et al. (2005). The Copenhagen Psychosocial Questionnaire (COPSOQ).
        Scandinavian Journal of Work, Environment & Health, 31(6), 438-449.
    Einarsen, S. et al. (1994). The Negative Acts Questionnaire (NAQ).
    Tepper, B.J. (2000). Consequences of abusive supervision. Academy of Management Journal, 43(2), 178-190.
    Siegrist, J. (1996). Adverse health effects of high-effort/low-reward conditions.
        Journal of Occupational Health Psychology, 1(1), 27-41.
    Karasek, R.A. (1979). Job demands, job decision latitude, and mental strain.
        Administrative Science Quarterly, 24(2), 285-308.
    Gross, J.J. & John, O.P. (2003). Individual differences in two emotion regulation processes.
        Journal of Personality and Social Psychology, 85(2), 348-362.
    Gratz, K.L. & Roemer, L. (2004). Multidimensional assessment of emotion regulation and
        dysregulation (DERS). Journal of Psychopathology and Behavioral Assessment, 26(1), 41-54.
    Garnefski, N. et al. (2001). Cognitive emotion regulation strategies and depressive symptoms (CERQ).
        Personality and Individual Differences, 30(8), 1311-1327.
    Connor, K.M. & Davidson, J.R.T. (2003). Development of a new resilience scale (CD-RISC).
        Depression and Anxiety, 18(2), 76-82.
    Smith, B.W. et al. (2008). The Brief Resilience Scale (BRS).
        International Journal of Behavioral Medicine, 15(3), 194-200.
    Wagnild, G.M. & Young, H.M. (1993). Development and psychometric evaluation of the Resilience Scale.
        Journal of Nursing Measurement, 1(2), 165-178.
    Duckworth, A.L. et al. (2007). Grit: perseverance and passion for long-term goals.
        Journal of Personality and Social Psychology, 92(6), 1087-1101.
    Luthans, F. et al. (2007). Psychological capital (PsyCap).
        Personnel Psychology, 60(3), 541-572.
    Rotter, J.B. (1967). A new scale for the measurement of interpersonal trust.
        Journal of Personality, 35(4), 651-665.
    Cummings, L.L. & Bromiley, P. (1996). The Organizational Trust Inventory (OTI).
        Trust in Organizations, 302-330.
    Maister, D.H. et al. (2000). The Trusted Advisor. Free Press.
    Mayer, R.C. & Davis, J.H. (1999). The effect of the performance appraisal system on trust
        for management. Journal of Applied Psychology, 84(1), 123-136.
    Rousseau, D.M. (1995). Psychological Contracts in Organizations. Sage.
    Rousseau, D.M. (2000). Psychological Contract Inventory. Technical report.
    Morrison, E.W. & Robinson, S.L. (1997). When employees feel betrayed: a model of how
        psychological contract violation develops. Academy of Management Review, 22(1), 226-256.
    Cook, W.W. & Medley, D.M. (1954). Proposed hostility and pharisaic-virtue scales for the MMPI.
        Journal of Applied Psychology, 38(6), 414-418.
    Buss, A.H. & Perry, M. (1992). The Aggression Questionnaire.
        Journal of Personality and Social Psychology, 63(3), 452-459.
    Spielberger, C.D. (1999). STAXI-2: State-Trait Anger Expression Inventory-2. PAR.
    Novaco, R.W. (1994). Anger as a risk factor for violence among the mentally disordered.
        Violence and Mental Disorder, 21-59.
    Thomas, K.W. & Kilmann, R.H. (1974). Thomas-Kilmann Conflict Mode Instrument. Xicom.
    Crisis Prevention Institute (1980). Crisis Development Model.
    Sonnentag, S. & Fritz, C. (2007). The Recovery Experience Questionnaire.
        Journal of Occupational Health Psychology, 12(3), 204-221.
    Richmond, J.S. et al. (2012). Verbal de-escalation of the agitated patient.
        Western Journal of Emergency Medicine, 13(1), 17-25.
    SAMHSA (2014). Trauma-Informed Care in Behavioral Health Services. TIP 57.
    Meijman, T.F. & Mulder, G. (1998). Psychological aspects of workload.
        Handbook of Work and Organizational Psychology, 2, 5-33.
    Hobfoll, S.E. (1989). Conservation of resources: a new attempt at conceptualizing stress.
        American Psychologist, 44(3), 513-524.
    McEwen, B.S. (1998). Stress, adaptation, and disease: allostasis and allostatic load.
        Annals of the New York Academy of Sciences, 840(1), 33-44.
    Jackson, S.A. & Marsh, H.W. (1996). Development and validation of a scale to measure
        optimal experience: the Flow State Scale. Journal of Sport & Exercise Psychology, 18(1), 17-35.
    Bond, M. et al. (1983). Empirical study of self-rated defense styles.
        Archives of General Psychiatry, 40(3), 333-338.
    Perry, J.C. (1990). Defense Mechanisms Rating Scales (DMRS). 5th edition.
    Vaillant, G.E. (1977). Adaptation to Life. Little, Brown.
    Rahim, M.A. (1983). A measure of styles of handling interpersonal conflict.
        Academy of Management Journal, 26(2), 368-376.
    Rosenberg, M.B. (2003). Nonviolent Communication: A Language of Life. PuddleDancer Press.
    French, J.R.P. & Raven, B. (1959). The bases of social power.
        Studies in Social Power, 150-167.
    Bass, B.M. & Avolio, B.J. (1995). Multifactor Leadership Questionnaire (MLQ). Mind Garden.
    Hofstede, G. (1980). Culture's Consequences. Sage.
    Milgram, S. (1963). Behavioral study of obedience.
        Journal of Abnormal and Social Psychology, 67(4), 371-378.
    Dollard, M.F. & Bakker, A.B. (2010). Psychosocial safety climate (PSC-12).
        Work & Stress, 24(2), 126-142.
    Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS.
    Gu, Y. et al. (2024). Systematic evaluation of LLM-as-a-Judge. arXiv:2408.13006.
    Brennan, K.A. et al. (1998). Self-report measurement of adult attachment (ECR).
        Attachment Theory and Close Relationships, 46-76.
