session record — what we learned building the PSQ Content Evaluator specification


origin

    date: 2025-02-25
    starting point: initial-state.md (May 2022 email, 71 raw PJE terms)
    ending point: a peer-reviewed-ready instrument specification with external validation
    trigger: an external critique that called PJE "a manifesto, not a methodology"


the critique and our response

    the critique said:
        - PJE is "a call for transdisciplinary integration" — a position paper, not a field
        - it lacks "novel constructs, methods, or instruments"
        - it could be valuable as a manifesto but doesn't yet constitute new science

    what we built in response:
        1. a novel construct — the Psychoemotional Safety Quotient (PSQ)
        2. a method — the multi-pass LLM-as-judge content evaluation pipeline
        3. an instrument — the 10-dimension PSQ Content Evaluator
        4. a reference library — 170+ validated instruments mapped to every PJE term
        5. a validation framework — with specific psychometric targets

    what the research system confirmed:
        - scientific validity: strong
        - novelty: genuinely novel (no existing tool combines these elements)
        - implementability: complex but feasible
        - overall: "significant contribution" — "no longer just a manifesto"


documents produced (in order of creation)

    1. initial-state.md
       what: original PJE terms from May 2022
       content: 17 operational definitions, 2 methods, 52 vocabulary terms
       role: the raw vision — ungrounded, uncited, but structurally complete

    2. psq-definition.md
       what: rigorous definition of the Psychoemotional Safety Quotient
       content: core definition, relationship to all PJE operational definitions and methods,
                10 constituent dimensions with full descriptions, the PSQ formula,
                application targets (individual, relationship, environment, system),
                differentiation from adjacent concepts (Edmondson's psychological safety,
                resilience quotient, intelligence quotient), engineering implications
       role: the theoretical foundation — what the PSQ is, what it measures, why it matters

    3. intermediate-state.md
       what: mapping of all 71 PJE terms to real instruments, scales, and constructs
       content: every operational definition → established field + key instruments + authors + dates
                every operational method → established field + key frameworks + authors + dates
                every vocabulary term → validated scales + authors + dates
       count: ~170 instruments across 71 terms
       role: the empirical grounding — proof that PJE terms are not invented from nothing
              but sit on decades of validated research

    4. psq-metrics-engine.md
       what: the detector architecture and analysis pipeline design
       content: input types, three suitability tiers for text analysis,
                detector architecture (instrument → structured prompt with items/subscales/rubric),
                four-pass pipeline (characterize → detect → aggregate → compute),
                dimension-to-instrument aggregation table, ethical constraints,
                implementation considerations
       role: the engineering blueprint — how to turn the theory into software

    5. final-state.md
       what: the complete operational specification for the PSQ Content Evaluator
       content: lineage, what changed from the guidance system's Edmondson-only design,
                all 10 dimensions with instruments/indicators/scoring,
                PSQ computation formula with classification thresholds,
                full evaluation prompt (tech-stack agnostic),
                operations specification, response schema,
                validation framework (psychometric + LLM-as-judge + instrument fidelity),
                deployment contexts (6 domains), ethical boundaries,
                direct answer to the critique
       role: the implementable specification — everything needed to build

    6. final-state-response.md
       what: the research system's independent assessment
       content: scientific validity (strong), novelty (genuinely novel),
                implementability (complex but feasible), potential value (high),
                limitations and considerations, overall assessment (significant contribution),
                13 peer-reviewed references
       role: external validation — confirmation that this is real science


key concepts established

    the PSQ is not psychological safety (Edmondson)
        Edmondson's construct is about interpersonal risk-taking in teams.
        The PSQ is broader: it includes emotional, relational, contractual, and somatic
        dimensions, and it carries a juris obligation — safety as a right with duties.

    the PSQ measures content, not people
        this is a critical distinction. the evaluator assesses what a piece of text does
        to psychoemotional safety. it does not diagnose the author or the reader.
        every score must cite specific textual evidence.

    safety is not resilience
        resilience is the capacity to endure and recover.
        safety is the condition of not needing to endure in the first place.
        conflating the two places the burden on the endangered rather than the system.
        the PSQ distinguishes between environments that are safe and environments
        that are merely survived by resilient individuals.

    the PJE lens transforms every dimension
        psychology alone describes what happens internally.
        juris adds what is owed — obligation, rights, duty of care.
        engineering adds what can be built — measurement, design, tolerance, intervention.
        no existing framework applies all three to psychoemotional safety in content.

    the formula structure
        PSQ = protective factors / threat factors
        protective: regulatory capacity, resilience, trust, cooling, defense, contractual clarity
        threat: threat exposure, hostility, energy entrapment, authority imbalance
        classification: critical, low, moderate, high
        each dimension scored 0-2 with mandatory confidence and evidence


the 10 PSQ dimensions (summary)

    1. Threat Exposure — hazards present (COPSOQ, NAQ, Abusive Supervision, ERI, JDC)
    2. Regulatory Capacity — ability to regulate emotion (ERQ, DERS, CERQ, Gross model)
    3. Resilience Baseline — capacity to absorb and recover (CD-RISC, BRS, RS, Grit, PsyCap)
    4. Trust Conditions — trust built or destroyed (Rotter, OTI, TQ, Propensity to Trust, PCI)
    5. Hostility Index — hostility expressed or reduced (Cook-Medley, BPAQ, STAXI-2, NAS, TKI)
    6. Cooling Capacity — de-escalation available (CPI, Gross reappraisal, Recovery Experience)
    7. Energy Dissipation — recovery paths open or blocked (Effort-Recovery, COR, allostatic load, Flow)
    8. Defensive Architecture — boundaries supported or stripped (DSQ, DMRS, Vaillant, TKI, NVC)
    9. Authority Dynamics — power distributed or abused (French & Raven, MLQ, Tepper, Hofstede)
    10. Contractual Clarity — expectations explicit or ambiguous (PCI, Morrison & Robinson, COPSOQ)


instrument suitability for LLM text analysis

    tier 1 (high — directly detectable in language):
        hostility, humor styles, conflict modes, trust/distrust, authority/power,
        psychological safety indicators, defense mechanisms, emotion regulation,
        contractual clarity, psychosocial hazards, attachment patterns,
        love styles, moral reasoning

    tier 2 (medium — inferable from longer content or patterns over time):
        resilience, coping styles, self-regulation, energy/burnout,
        change readiness, self-determination, psychological capital, belonging

    tier 3 (lower — requires supplementary data, flag only):
        physiological measures, neuroscience constructs, clinical diagnoses,
        cognitive load, psychometric properties


limitations identified

    methodological:
        - dimension independence is questionable — factor analysis needed during validation
        - LLM-as-judge reliability is unproven for this specific application
        - context dependency means the same content may score differently in different settings

    ethical:
        - surveillance risk if deployed without consent
        - cultural bias — instruments predominantly Western-developed
        - over-pathologizing risk — labeling normal expression as "unsafe"

    practical:
        - 10 dimensions × multiple instruments = high cognitive load for LLM judges
        - validation requires extensive human-expert correlation studies
        - computational cost at scale is nontrivial
        - false positive/negative rates unknown until validated


what the research system referenced (13 sources)

    psychological safety scales:
        - Neuroception of Psychological Safety Scale (NPSS) — PMC11675212
        - Employee perception of psychological safety construct — PubMed 39709484
        - Factor structure and measurement of psychological safety — PMC11507099
        - Psychological Safety Scale (Safety, Communication) — PMC9422763
        - Diagnostic metric for ability to speak up — PubMed 35985041
        - Psychological safety and patient safety systematic review — PMC12021220

    AI and mental health:
        - Early detection of mental health crises through AI — PMC11433454

    content moderation and analysis:
        - Psychological impacts of content moderation — cyberpsychology.eu/33166
        - Abusive content automatic detection review — PMC9680866
        - Detection and moderation of detrimental content — PMC9444091
        - Deep learning text emotion analysis for legal anomie — PMC9247634
        - Sentiment analysis and emotion detection review — PMC8402961


what was decided

    - tech-stack agnostic specification (no language or model dependencies)
    - the evaluator uses all 10 PSQ dimensions, not Edmondson's 7 items alone
    - every dimension draws from multiple validated instruments
    - the PJE lens (psychology + juris + engineering) shapes every dimension
    - confidence scoring is mandatory — no false precision
    - the engine measures content, not people — this is a hard ethical boundary
    - cultural context parameter needed from the start, not retrofitted
    - surveillance constraints must be enforced technically, not just stated


what was not yet decided

    - implementation language and runtime
    - which LLM(s) to use for evaluation
    - database/storage for instrument library, evaluations, and profiles
    - whether to start with all 10 dimensions or a subset for the prototype
    - weighting scheme for dimension aggregation (equal vs. context-dependent)
    - delivery format (API, CLI, library, service, or combination)
    - licensing model (open source, commercial, research-only, or hybrid)
    - how to handle multi-language content
    - the calibration dataset — what known-scored content to validate against
    - whether the longitudinal evaluation mode is in scope for the prototype
