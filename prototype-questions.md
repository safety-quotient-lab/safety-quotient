prototype questions — decision sequence for building the PSQ Content Evaluator


1. scope

    1.1. which deployment context do you want to target first?
         workplace communications, legal proceedings, clinical/therapeutic,
         educational, relationship/personal, or content moderation?

    1.2. do you start with all 10 PSQ dimensions or a subset?
         if a subset — which dimensions are essential for v1?

    1.3. is longitudinal evaluation in scope for the prototype?

    1.4. what is the minimum viable output?


2. instrument library

    2.1. how do you represent the instruments as data?
         YAML, JSON, or something else? one file per instrument, per dimension,
         or a single reference file?

    2.2. which specific instruments do you include per dimension for v1?
         all of them, or pick the single strongest per dimension?

    2.3. how do you handle copyrighted instrument items?
         use published items directly (fair use for research), describe constructs
         without quoting, or develop original indicators?

    2.4. how granular are the textual indicators?
         construct level, subscale level, or item level?


3. detector design

    3.1. one prompt or many prompts per evaluation?
         single mega-prompt (faster, cheaper) vs. one per dimension (more accurate)
         vs. hybrid clusters?

    3.2. how do you structure the evaluation prompt?
         use final-state.md template as-is, or iterate on prompt design first?

    3.3. how do you enforce structured output?
         native JSON mode, parse free-text, or function calling / tool use?

    3.4. how do you handle confidence scoring?
         trust LLM self-assessment, calibrate against ground truth,
         or derive from inter-run consistency?


4. architecture

    4.1. what language and runtime?

    4.2. which LLM(s) do you use?
         one model or multiple? stronger for initial, faster for batch?

    4.3. how do you handle API access?
         direct API calls, local model, or abstraction layer?

    4.4. what storage do you need?
         database, flat files, or both?

    4.5. what is the delivery format?
         CLI, REST API, library, or combination?


5. validation

    5.1. what content do you use for calibration?
         hand-curated, real-world collected, published datasets, or synthetic?

    5.2. who does the human rating?
         you alone, small panel, or deferred?

    5.3. what are your v1 psychometric targets?

    5.4. how do you test for bias?
         include in prototype or defer?


6. ethics and constraints

    6.1. how do you enforce the consent requirement technically?

    6.2. how do you handle the cultural context parameter?

    6.3. what happens when content scores critical?

    6.4. do you build the "not a diagnosis" guardrail into the output?


7. distribution

    7.1. open source, proprietary, or research-only?

    7.2. what do you call it?

    7.3. does the prototype need documentation beyond the spec files?


8. sequencing

    8.1. what is your build order?

    8.2. what is your timeline expectation?
         weekend project, month-long build, or ongoing research program?

    8.3. do you build alone or with collaborators?
