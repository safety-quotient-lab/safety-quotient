prototype decisions — SafetyQuotient


scope

    deployment context: content moderation (social media, forums, public-facing content)
    dimensions: all 10 PSQ dimensions from day one
    longitudinal: primary use case — tracking patterns over time comes first
    minimum viable output: full PSQ profile (all scores, evidence, flags, recommendations, ethical notes)


instrument library

    format: JSON
    instruments per dimension: top 2-3 per dimension (~25 total)
    copyright approach: fair use for research
    indicator granularity: item level (mapped from original instrument items)


detector design

    prompt strategy: hybrid clusters (related dimensions grouped into 2-3 prompts)
    prompt development: iterate on prompt design first, test against sample content before locking in
    structured output: native JSON mode (provider-specific JSON output features)
    confidence scoring: hybrid — LLM self-assessment for v1, inter-run consistency added later


architecture

    runtime: Cloudflare Worker (JavaScript/TypeScript, V8 isolates)
    LLM providers: abstraction layer with two backends — Anthropic Claude + Workers AI
    API access: direct API calls (Anthropic API for Claude, Workers AI binding for Cloudflare models)
    storage: R2 + D1 hybrid + KV
        KV — instrument library (read-heavy, static, cached at edge)
        R2 — full PSQ profile JSON files (raw evaluation storage, portable, exportable)
        D1 — summary rows for querying (source_id, timestamp, scores, classification, longitudinal queries)
    delivery: REST API + web UI + CLI client


validation

    calibration content: real-world collected (smart selection across full scoring range, store results only not raw content)
    human rating: deferred — face-validity checks for prototype
    psychometric targets: set now — Cronbach's α ≥ 0.80, test-retest r ≥ 0.70
    bias testing: included in v1 (vary demographic markers, check for score variance)


ethics and constraints

    consent enforcement: documentation only for v1 (no technical enforcement)
    cultural context: optional parameter with Western baseline default, explicit caveat when omitted
    critical scoring: flag in output + configurable webhook for alerting
    diagnosis guardrail: standard disclaimer on every response


distribution

    license: dual licensing (research-free + commercial-paid, per research system recommendation)
    name: SafetyQuotient
    documentation: existing spec files are sufficient for v1
    timeline: ongoing research program (build for longevity)
    team: solo to start, designed for contributors (need code standards, issue tracking from start)


build order

    1. instrument library as JSON
    2. single-dimension detector proof of concept
    3. test against real-world collected content
    4. expand to all 10 dimensions (hybrid clusters)
    5. aggregation logic (PSQ scoring, classification)
    6. D1/R2/KV storage layer
    7. longitudinal tracking and trajectory analysis
    8. REST API on Cloudflare Worker
    9. web UI
    10. CLI client
    11. bias testing pipeline
    12. webhook for critical flags
    13. packaging and contributor setup
