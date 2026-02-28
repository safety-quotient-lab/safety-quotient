# PSQ Content Evaluator: Practical Applications Analysis

**Date:** 2026-02-28
**Scope:** Entrepreneurial tech and psychology research applications for the 10-dimension PSQ (Psychoemotional Safety Quotient) content evaluator

---

## Executive Summary

The PSQ occupies a unique position in the emerging market for communication intelligence: it is a real-time, multi-dimensional measure of psychoemotional safety climate in text, backed by criterion validity evidence across four independent datasets. Its core differentiator is that the 10-dimension profile consistently outperforms single-score sentiment or toxicity metrics at predicting real-world outcomes (negotiation satisfaction, conversation derailment, persuasion, deal completion). The model runs in real-time on consumer hardware (DistilBERT, 66M params, 64MB quantized ONNX), making edge deployment feasible for privacy-sensitive applications.

This analysis maps PSQ's capabilities to specific product concepts, competitive gaps, and research applications, with attention to what is validated today versus what requires further evidence.

---

## 1. Entrepreneurial Tech Applications

### 1A. Workplace Communication Climate Analytics (B2B SaaS)

**The gap:** Current employee experience platforms — [Qualtrics](https://www.qualtrics.com/articles/employee-experience/employee-sentiment/), [Perceptyx](https://www.perceptyx.com/), [Culture Amp](https://www.selectsoftwarereviews.com/reviews/cultureamp) — rely primarily on periodic survey instruments. Their AI-powered sentiment analysis, while increasingly sophisticated, collapses communication quality to positive/negative/neutral polarity. The $6.7B employee experience management market (2025, growing to $12.9B by 2035) is actively seeking richer signal from passive data sources, but existing tools have well-documented limitations: they struggle with sarcasm, context-dependent meaning, and cannot distinguish *why* communication feels unsafe.

Meanwhile, organizational network analysis tools like [Worklytics](https://www.worklytics.co/) and [Humanyze](https://humanyze.com/) analyze collaboration *patterns* (who talks to whom, response latency, meeting load) but not communication *quality*. They can tell you a team is siloed; they cannot tell you that the silo exists because authority dynamics are poorly managed or that trust conditions have eroded.

[Edmondson's Psychological Safety Index](https://fearlessorganizationscan.com/the-fearless-organization) remains the gold standard for measuring team psychological safety, but it is a 7-item self-report survey — subject to social desirability bias, survey fatigue, and point-in-time measurement limitations. 84% of employees rank psychological safety among their top three workplace needs, yet measurement remains episodic.

**PSQ product concept: "Safety Climate Dashboard"**

A Slack/Teams integration that continuously profiles the psychoemotional safety climate of channels, teams, and organizational units across all 10 PSQ dimensions. Unlike sentiment analysis, it can distinguish between:

- A channel with high Hostility Index but high Cooling Capacity (intense but self-correcting debate)
- A channel with moderate Energy Dissipation and declining Trust Conditions (burnout trajectory)
- A channel with low Authority Dynamics friction and high Contractual Clarity (healthy execution mode)

These distinctions are precisely what Edmondson's survey measures at the team level via self-report; PSQ measures them continuously from the text itself.

**Technical feasibility:** The Slack Discovery API (available to Enterprise Grid customers) provides programmatic access to message history across public, private, and DM channels. The quantized ONNX model (64MB) processes text in <100ms on CPU, making real-time scoring viable at enterprise message volumes. Privacy can be preserved by scoring and aggregating at the channel/team level without storing individual message scores — only aggregate dimension profiles over time windows.

**Competitive positioning:** This sits between Worklytics (structural patterns, no content analysis) and Receptiviti (psycholinguistic analysis via LIWC, 200+ measures but no integrated safety climate framework). [Receptiviti](https://www.receptiviti.com/), co-founded by LIWC inventor Jamie Pennebaker, is the closest competitor — it quantifies psychological traits from language using 94 LIWC categories and additional validated measures. However, Receptiviti's output is a flat vector of psycholinguistic features; PSQ provides an integrated, theoretically grounded safety climate assessment with demonstrated criterion validity for predicting interpersonal outcomes.

**Revenue model:** Per-seat SaaS ($5-15/user/month), tiered by team/org/enterprise. The edge deployment capability (ONNX, no API dependency) enables an on-premise option for regulated industries — a significant differentiator vs. cloud-only competitors.

**Validation requirements before launch:** Inter-rater reliability with human experts (currently not measured, protocol designed), measurement invariance across demographic groups (not yet tested), and organizational-level outcome validation (retention, engagement survey correlation).

---

### 1B. Conversation Intelligence for Sales and Customer Success

**The gap:** Conversation intelligence platforms — [Gong](https://www.gong.io/conversation-intelligence) ($250/user/month, $5K-$50K annual platform fees), [Chorus/ZoomInfo](https://www.claap.io/blog/gong-vs-chorus-which-is-better-and-why), [Observe.AI](https://upsolve.ai/blog/conversation-analytics-intelligence-software) — analyze sales calls and customer interactions for keyword patterns, talk ratios, and deal signals. They tell reps *what* was discussed and provide coaching signals like talk/listen ratio and filler word frequency.

What they do not measure is the interpersonal dynamics that determine whether a deal closes. PSQ's criterion validity evidence directly addresses this:

- **Deal or No Deal (n=12,234):** ED (Energy Dissipation) predicts deal-reaching with d=0.614, AUC=0.686 — the largest single-dimension effect across all four validation studies. High-PSQ (Q4) deal rate was 84.4% vs. Low-PSQ (Q1) at 68.5%, a 15.9 percentage-point gap.
- **CaSiNo (n=1,030):** AD (Authority Dynamics) predicts negotiation satisfaction at r=0.127 (p<0.001). 9/10 dimensions predict satisfaction.
- The 10-dimension profile consistently outperforms the single average score (DonD: AUC 0.686 vs. 0.622; CMV: 0.590 vs. 0.531).

**PSQ product concept: "Deal Climate Score"**

An add-on or standalone tool that processes call transcripts (or chat transcripts in text-based sales) and surfaces a 10-dimension safety climate profile for each deal/account over time. The actionable insight is not "this call had negative sentiment" but rather:

- "Energy Dissipation is increasing across calls with this account — the buyer may be experiencing decision fatigue. Consider simplifying the next interaction."
- "Authority Dynamics shifted after the VP joined the last call. Trust Conditions dropped. Recommend reestablishing shared expectations before the next meeting."
- "Cooling Capacity is low in this thread. The account is not self-correcting after disagreements. Flag for manager review."

**Competitive edge:** Gong and peers optimize for *what to say*. PSQ measures *how the conversation feels* along dimensions that demonstrably predict whether deals close. The DonD validation (AUC=0.686 for deal-reaching) is a directly sellable metric for revenue leaders.

**Integration:** Process transcripts from Gong/Chorus/Observe.AI via API, or operate as a lightweight standalone (the 64MB ONNX model can run in a serverless function). This positions PSQ as an enrichment layer rather than a platform replacement.

---

### 1C. Platform Safety and Community Health (Content Moderation)

**The gap:** The content moderation services market was $12.5B in 2025 and is projected to reach $42.4B by 2035. Current tools — [Perspective API (Jigsaw/Google)](https://www.perspectiveapi.com/), [Spectrum Labs](https://www.spectrumlabsai.com/ai-for-content-moderation/), [ActiveFence](https://huntscreens.com/en/topic/ai-moderation), [Hive](https://thehive.ai/) — are primarily toxicity classifiers: they score text on a toxic/not-toxic continuum. Perspective API explicitly uses only comment text to produce a score, with well-documented limitations on context, nuance, sarcasm, and cross-cultural transferability. As one ACL benchmark study notes, Perspective "exhibits troubling shortcomings across a number of toxicity categories."

The fundamental limitation is conceptual, not technical: toxicity detection is binary (remove or keep). It cannot distinguish between:

- A heated but productive disagreement (high HI, high CC — both parties are de-escalating)
- A calm but corrosive conversation (low HI, low TC, declining CO — polite disengagement)
- A one-sided power play (high AD asymmetry, low DA — one party has no protective boundaries)

Online community managers are already moving beyond toxicity metrics. Community health frameworks now track activation, retention, sentiment, flagged-message rate, and time-to-first-response — but these are behavioral proxies, not direct measures of the conversation climate.

PSQ's CGA-Wiki validation is directly relevant: on 4,188 Wikipedia talk-page conversations, the 10-dimension profile predicted derailment into personal attacks (AUC=0.599) while the single average score was near-chance (AUC=0.515). The temporal signal — AUC rising from 0.519 (first turn only) to 0.599 (all turns) — demonstrates that PSQ captures conversational dynamics, not just static content properties. This means it can flag conversations that are *on a trajectory toward* toxicity before they arrive there.

**PSQ product concept: "Conversation Climate API"**

A developer API that returns a 10-dimension safety climate profile for any text or conversation thread, designed to complement (not replace) existing toxicity classifiers. Use cases:

1. **Pre-emptive moderation:** Flag conversations where AD asymmetry is rising and CC is declining, before any individual message crosses a toxicity threshold. This addresses the "polite hostility" blindspot in current tools.
2. **Community health dashboards:** Give community managers a multi-dimensional view of channel/forum/subreddit health. A channel with high TE but high RB and RC is resilient; the same TE with declining RB triggers an alert.
3. **Conversation design:** Game developers, social platform designers, and LMS providers can use PSQ profiles to evaluate whether their platform design choices promote or suppress psychoemotional safety.

**Technical advantage:** At 64MB quantized, PSQ can run at the edge — in-browser or on mobile — without sending user text to external servers. This is a privacy advantage over cloud-only APIs like Perspective, particularly for platforms operating under GDPR or processing children's data.

**Pricing:** Per-API-call (Perspective API pricing model: free tier + usage-based), or embedded licensing for platforms that deploy the ONNX model directly.

---

### 1D. EdTech and Learning Management Systems

**The gap:** LMS platforms (Canvas, Moodle, Blackboard, Google Classroom) capture extensive behavioral data — clicks, page views, time-on-task, quiz scores — but have minimal insight into the *quality* of student-to-student and student-to-instructor communication. Research confirms that emotional attachments and feeling safe to participate are key drivers of student engagement in online learning, yet text analysis in LMS remains limited to participation counts and, occasionally, rudimentary sentiment analysis.

**PSQ product concept: "Discussion Climate Monitor"**

An LMS plugin that profiles the psychoemotional safety climate of discussion forums, peer review exchanges, and group project communication. For instructors and academic administrators:

- Identify courses where Trust Conditions and Contractual Clarity are low (students are confused about expectations and do not trust the environment enough to ask)
- Flag peer review exchanges with high Hostility Index or Authority Dynamics asymmetry
- Track Cooling Capacity over a semester: does the course community develop self-correction capabilities?

For students: a non-diagnostic, anonymous dashboard showing the class climate profile — "your section's discussion environment has strong Regulatory Capacity and moderate Trust Conditions" — that normalizes awareness of psychoemotional dynamics.

This maps to the DEI and student belonging initiatives that universities are investing heavily in, and provides continuous measurement rather than end-of-semester course evaluations.

---

### 1E. Telehealth and Digital Mental Health

**The gap:** The intersection of AI and clinical psychology is accelerating. [Eleos Health](https://eleos.health/) uses NLP to analyze therapy session transcripts, measuring talk/listen ratio, clinician wait time, and evidence-based technique usage — with results showing 3-4x better symptom improvement for therapists who use the platform. [Lyssn](https://www.technologyreview.com/2021/12/06/1041345/ai-nlp-mental-health-better-therapists-psychology-cbt/) focuses on training therapists by analyzing session recordings for adherence to evidence-based protocols. Both measure *therapist behavior* in the session.

The emerging concept of a [Digital Therapeutic Alliance (DTA)](https://mental.jmir.org/2025/1/e69294) — the therapeutic relationship between a patient and a digital tool — has produced a preliminary 5-dimension measurement scale. But no existing tool measures the psychoemotional safety *climate* of the therapeutic conversation itself, as it unfolds.

**PSQ product concept: "Session Climate Profile"**

A therapy session analysis tool that generates a PSQ profile for each session, tracking how the 10 dimensions evolve over the course of a conversation. Clinical hypotheses:

- A session where Threat Exposure spikes (patient disclosing traumatic material) followed by a rise in Cooling Capacity and Regulatory Capacity indicates successful therapist containment
- Declining Trust Conditions across sessions may predict therapeutic rupture before the patient disengages
- Energy Dissipation patterns could predict burnout in both therapists and patients

**Critical constraint:** This application sits in regulated healthcare territory. Deployment requires IRB-approved validation studies, HIPAA compliance, and explicit positioning as a *clinician decision support* tool, not a diagnostic or outcome measure. The PSQ documentation correctly states it evaluates content, not people — this distinction must be maintained rigorously. The current lack of inter-rater reliability evidence (human expert validation) is a blocking gap for clinical credibility.

---

## 2. Psychology Research Applications

### 2A. Interpersonal Communication Research

PSQ provides a standardized, scalable method for coding the psychoemotional dynamics of text-based communication — something that currently requires manual coding by trained raters (expensive, slow, subjectively variable) or ad hoc NLP feature engineering.

**Research applications:**

- **Conflict and negotiation:** The CaSiNo and DonD validations demonstrate that PSQ captures meaningful variance in negotiation dynamics. Researchers studying integrative vs. distributive bargaining, BATNA effects, or cultural differences in negotiation style could use PSQ profiles as a richer dependent/independent variable than sentiment alone.
- **Online discourse quality:** The CGA-Wiki validation (derailment prediction, AUC=0.599) positions PSQ as a tool for studying conversation trajectories. The temporal gradient finding (AUC improving from first turn to all turns) suggests PSQ captures *process* features that accumulate, making it suitable for studying how conversations go wrong over time.
- **Computer-mediated communication (CMC):** The CMV validation shows PSQ differentiates persuasive from non-persuasive arguments (AUC=0.590), with a context-dependent finding: DA (Defensive Architecture) dominates in persuasion contexts, AD (Authority Dynamics) dominates in contested-status contexts. This context-sensitivity is itself a research finding worth exploring further.

**What PSQ offers researchers:** A pre-trained, freely deployable model (DistilBERT ONNX) that can code 10 theoretically grounded dimensions across thousands of texts in minutes. No API costs, no cloud dependency, no per-text charges. For researchers accustomed to hiring and training human coders at $15-25/hour, the efficiency gain is transformative — even if PSQ requires human validation for any given study.

**Methodological contribution:** The finding that g-PSQ (average) is consistently near-chance while the 10-dimension profile predicts outcomes is a methodological argument against collapsing multi-dimensional constructs into single scores. This replicates a pattern well-known in personality psychology (the bandwidth-fidelity tradeoff) but demonstrates it in a new domain.

### 2B. Therapeutic Process Research

**The gap:** Measuring therapeutic alliance is a cornerstone of psychotherapy research, but existing measures — the Working Alliance Inventory (WAI), the Session Rating Scale (SRS), the Therapeutic Alliance Questionnaire — are self-report instruments administered after the fact. They capture the patient's (or therapist's) subjective experience of the session, not the linguistic features of the session itself.

A 2026 systematic review in *Clinical Psychology & Psychotherapy* ([Orru et al.](https://onlinelibrary.wiley.com/doi/full/10.1002/cpp.70242)) documents 205 studies on LLMs in psychiatry, psychology, and psychotherapy. NLP analysis of therapy transcripts is being used for risk prediction, symptom trajectory modeling, and relapse detection — but no standardized multi-dimensional measure of session *climate* exists.

**Research applications:**

- **Rupture-repair sequences:** PSQ's Trust Conditions and Cooling Capacity dimensions could provide continuous markers of therapeutic rupture (TC declining) and repair (CC activating). This complements Safran and Muran's qualitative rupture-repair model with quantitative text-derived signals.
- **Therapist training:** Yeshiva University's Wurzweiler School of Social Work already uses Eleos for clinician training. PSQ could provide a complementary lens: rather than scoring whether the therapist used specific evidence-based techniques, it measures the *emotional climate* the therapist created. A session might be technically adherent (correct CBT technique) but climatically poor (low TC, high AD asymmetry).
- **Digital therapeutic alliance:** The emerging DTA measurement framework could be enriched by PSQ profiles of chatbot conversations, providing a content-derived complement to self-report DTA scales.

### 2C. Organizational Psychology and Climate Research

**The gap:** Organizational climate and culture research traditionally relies on survey instruments — the Organizational Climate Questionnaire (OCQ), the Competing Values Framework, Denison's Organizational Culture Survey. These measure *perceived* climate. NLP-based analysis of actual organizational communications is an emerging but under-developed methodology.

**Research applications:**

- **Psychological safety as observable behavior:** Edmondson's psychological safety construct is defined behaviorally (willingness to take interpersonal risks), but measured via self-report. PSQ enables measurement from actual communication artifacts — emails, Slack messages, meeting transcripts — without the social desirability bias inherent in surveys. This is a genuine methodological advance: moving from "people say they feel safe" to "the communication artifacts exhibit safety-conducive patterns."
- **Climate change over time:** Survey instruments provide point-in-time snapshots. PSQ can be applied to archived communications to reconstruct how organizational climate evolved over months or years — before and after leadership changes, reorganizations, or crisis events.
- **Cross-cultural comparison:** PSQ's 10 dimensions are grounded in instruments that have varying levels of cross-cultural validation (e.g., COPSOQ has extensive cross-national norming; the Psychological Contract Inventory is more Western-centric). Applying PSQ to communications from different cultural contexts could both test and extend the instrument's cross-cultural validity.

### 2D. Scale Development and Psychometric Innovation

The PSQ project itself contributes to psychometric methodology:

- **Separated scoring to mitigate halo effect:** The methodological innovation of scoring one dimension per LLM call, rather than all dimensions jointly, is a replicable approach for any LLM-based measurement system. The project's data shows joint scoring inflates inter-dimension correlations by approximately 0.15 (halo inflation).
- **Hierarchical measurement model:** The empirical finding that a general factor dominates (67.3% of variance) while individual dimensions carry non-redundant criterion validity suggests a hierarchical model (g-PSQ at the top, 3-5 clusters, 10 dimensions at the base). This structure — where the general score provides a quick summary but dimensional detail drives actionable insight — is applicable to other multi-dimensional text analysis systems.
- **Content-level measurement from person-level constructs:** The conceptual move from "this person has high hostility" (trait, self-report) to "this text exhibits hostility-indicative patterns" (content, observed) is a methodological bridge that other psychometric constructs could cross.

---

## 3. Differentiation from Existing Solutions

### 3A. What Sentiment Analysis Cannot Do

Standard sentiment analysis (positive/negative/neutral polarity, sometimes with emotion categories like joy/anger/fear) fails in several specific ways that PSQ addresses:

| Scenario | Sentiment Analysis Says | PSQ Says |
|---|---|---|
| A manager gives direct, critical feedback with clear expectations | Negative sentiment | Low HI, High CO, Moderate AD — constructive criticism within a clear contractual frame |
| A team uses polite, professional language while systematically excluding a colleague | Neutral/positive sentiment | Low TC, High AD asymmetry, Low CO — trust erosion beneath polite surface |
| A heated brainstorm with passionate disagreement but mutual respect | Mixed/negative sentiment | High TE, High CC, High RB — intense but resilient environment |
| A quiet team meeting where nobody disagrees with the boss | Positive sentiment | Low RC, Low DA, High AD asymmetry — suppressed disagreement, not genuine harmony |
| A negotiation where one party is exhausting the other through prolonged discussion | Neutral sentiment | Rising ED, Declining RC — energy depletion predicting deal failure (DonD: AUC=0.686) |

The core insight: **sentiment measures the affect of the text; PSQ measures the safety climate the text creates.** Negative sentiment can be safe (honest critical feedback in a high-trust environment). Positive sentiment can be unsafe (superficial agreement masking power asymmetry).

### 3B. What Toxicity Detection Cannot Do

[Perspective API](https://www.perspectiveapi.com/) and peers detect *what is toxic*. PSQ measures *what makes conversations go wrong* — a broader and earlier-stage signal. The CGA-Wiki validation demonstrates this: PSQ predicts conversation derailment into personal attacks (AUC=0.599) from the *entire conversation*, not just the toxic utterance. The temporal gradient (AUC 0.519 on first turn alone, rising to 0.599 on full conversation) proves PSQ is measuring conversational process, not static toxicity.

Toxicity detection is retrospective and binary: this message is/is not toxic. PSQ is prospective and graded: this conversation's trajectory suggests declining safety. The difference is the difference between a smoke alarm and a fire prevention inspection.

### 3C. What LIWC/Receptiviti Cannot Do

[Receptiviti's LIWC API](https://www.receptiviti.com/liwc) provides 94+ psycholinguistic categories (function words, cognitive processes, social references, drives, etc.) and 200+ validated psychological measures. It is the closest methodological cousin to PSQ. Key differences:

1. **Theoretical integration:** LIWC categories are linguistic features (word counts in categories). PSQ dimensions are theoretically grounded constructs mapped to validated psychological instruments. LIWC tells you the text has high "social" word usage; PSQ tells you trust conditions are eroding.
2. **Criterion validity for interpersonal outcomes:** PSQ has four independent validation studies showing that its 10-dimension profile predicts negotiation satisfaction, conversation derailment, persuasion, and deal completion. LIWC has extensive validity evidence (Pennebaker's decades of research), but primarily for *individual* psychological states (depression markers, deception detection, personality inference) rather than *interpersonal safety climate*.
3. **Profile vs. features:** PSQ's validated finding is that the 10-dimensional profile shape matters more than any single score. LIWC provides a feature vector for downstream analysis; the user must build the model. PSQ provides the model.
4. **Deployment:** PSQ runs as a self-contained 64MB model with no API dependency. Receptiviti is a commercial API ($pricing varies, typically per-text).

### 3D. What Conversation Intelligence Cannot Do

[Gong](https://www.gong.io/conversation-intelligence), Chorus, and Observe.AI analyze *sales conversations specifically*, optimized for revenue signals (deal stage, competitor mentions, pricing discussion, next steps). They measure what was said and how (talk ratio, filler words, monologue length). They do not measure the interpersonal safety climate of the conversation. PSQ's DonD validation (AUC=0.686 for deal-reaching, ED d=0.614) demonstrates that energy dissipation in negotiation predicts deal failure — a signal these platforms do not capture.

---

## 4. Go-to-Market Considerations

### 4A. Validation Requirements Before Any Product Launch

The PSQ has strong criterion validity and good theoretical grounding, but several psychometric gaps must be addressed before commercial deployment:

| Gap | Current Status | Required for Launch | Estimated Effort |
|---|---|---|---|
| Inter-rater reliability (human experts) | Protocol designed, not executed | Yes — credibility with buyers depends on human-AI agreement | 7-9 weeks, $5,625-$15,000 for 5 raters |
| Measurement invariance (bias testing) | Planned, not done | Yes for any HR/workplace deployment | 2-4 weeks, requires diverse text corpora |
| Normative data | Not established | Helpful for benchmarking products | Accumulates with early customers |
| Confidence calibration | Problematic (7/10 dims inverted/NaN) | Yes — customers need reliable confidence intervals | 1-2 weeks, technical fix |
| Cross-cultural validity | Not tested | Required for international markets | 4-8 weeks, requires multilingual data |

The inter-rater reliability study is the single highest-priority gate. Without evidence that human psychologists agree with PSQ's dimensional scores, the system rests entirely on LLM-generated labels validated against LLM-generated labels. The existing expert validation protocol (5 psychologists, 200 texts, 10,000 ratings, targeting ICC >= 0.70) is well-designed and should be executed before any commercial launch.

### 4B. MVP Scope: Research API First

The lowest-risk first product is a **research API** targeting academic psychology and communication researchers:

- **What it is:** A hosted API (or downloadable ONNX model) that accepts text and returns 10-dimension PSQ profiles. No UI, no integrations, no compliance requirements.
- **Why this first:** Researchers have lower reliability requirements than commercial buyers (they validate against their own criteria), higher tolerance for beta-stage tools, and generate validation evidence through their publications. Each published study using PSQ is a free credibility investment.
- **Pricing:** Free tier (1,000 texts/day), paid tier for bulk processing ($0.001-0.005/text), academic discount.
- **Success metric:** Number of published papers citing PSQ within 18 months.
- **Distribution:** Hugging Face model hub, PyPI package, arXiv preprint documenting the methodology.

### 4C. Second Product: Workplace Communication Dashboard (Pilot)

After the research API generates validation evidence and the expert inter-rater study is complete:

- **Target:** 3-5 mid-size tech companies (200-2,000 employees) already using Slack Enterprise Grid
- **Offer:** Free 90-day pilot in exchange for validation data (anonymous, aggregated)
- **Deployment:** On-premise ONNX model (no data leaves the customer's infrastructure) — a significant differentiator for privacy-conscious companies
- **Metrics:** Correlation between PSQ channel profiles and (a) Edmondson PSI survey scores, (b) team retention, (c) employee engagement survey results
- **Risk mitigation:** Position explicitly as "communication climate analytics" — not "employee monitoring." Report only at the channel/team level, never individual. Require HR and legal sign-off before pilot begins.

### 4D. First Customers

| Segment | Why They Buy | Willingness to Pay | Validation Value |
|---|---|---|---|
| Communication researchers (academia) | Scalable text coding tool | Low ($0-500/year) | High (publications) |
| Org psych consultants | Client engagement analytics | Medium ($2,000-10,000/year) | Medium (case studies) |
| EdTech platforms | Student engagement / DEI metrics | Medium ($5,000-20,000/year) | Medium (institutional validation) |
| HR tech platforms (OEM) | Embedded safety climate signal | High ($50,000+/year licensing) | High (enterprise credibility) |
| Trust & Safety teams (platforms) | Pre-emptive moderation | High ($100,000+/year) | High (reduces content incidents) |

The academic segment is the beachhead: low revenue, high validation. The trust & safety segment is the scaling opportunity: the content moderation market is $42B by 2035 and the gap between toxicity detection and conversation climate measurement is real.

### 4E. Competitive Moat Considerations

**Defensible:**
- The 10-dimension theoretical framework grounded in ~100 validated instruments
- Four criterion validity studies with specific, publishable findings (ED predicts deal completion, AD predicts contested-status outcomes, profile >> average)
- Separated scoring methodology (halo effect mitigation) — a genuine methodological innovation applicable to any LLM-based scoring system
- Edge-deployable model (64MB ONNX) — no cloud dependency, privacy-preserving

**Not defensible:**
- DistilBERT architecture — commodity, reproducible
- Any single dimension score — most dimensions correlate with existing constructs
- ONNX export — standard tooling

The moat is in the integrated framework, the validation evidence, and the methodological innovations — not in the model architecture. A competitor could train a similar model; they cannot easily replicate the theoretical grounding or the four criterion validity studies.

### 4F. Risk Factors

1. **Over-interpretation risk:** PSQ evaluates content, not people. Any product that profiles individuals (rather than channels, teams, or conversation threads) risks ethical and legal exposure. The non-diagnostic positioning must be absolute and enforced architecturally (no individual-level scores stored or displayed).

2. **Reliability gap:** Without inter-rater reliability evidence, a critical reviewer can dismiss PSQ as "an LLM's opinion validated against another LLM's opinion." The expert validation study is not optional — it is existential for credibility.

3. **g-factor dominance:** 67.3% of variance loads on a single general factor. While the criterion validity evidence shows individual dimensions carry non-redundant signal, a skeptical buyer might ask: "Why not just use a single safety score?" The answer — that g-PSQ is near-chance for prediction while profiles predict — must be demonstrated repeatedly and clearly.

4. **Dimensional instability:** AD's construct validity is questionable (factor loading 0.332, below the 0.35 threshold; 49% of separated-llm scores are exact 5.0). DA also shows weak temporal stability (r=0.602). These dimensions need resolution before clinical or high-stakes deployment.

5. **Regulatory exposure:** Any application touching healthcare (therapy session analysis), education (student profiling), or employment (performance assessment) faces regulatory requirements that a research-stage tool does not yet meet.

---

## 5. Summary: Where PSQ Creates Unique Value

The PSQ's unique value proposition is the intersection of three properties no existing tool combines:

1. **Multi-dimensional safety climate measurement** (not sentiment, not toxicity, not psycholinguistic features — an integrated 10-dimension profile of psychoemotional safety)
2. **Criterion validity for interpersonal outcomes** (four studies showing the profile predicts negotiation satisfaction, conversation derailment, persuasion, and deal completion)
3. **Edge-deployable, real-time, privacy-preserving inference** (64MB ONNX model, <100ms on CPU, no cloud dependency)

The highest-impact, lowest-risk path to market is: (1) execute the expert validation study, (2) release a research API / Hugging Face model, (3) accumulate published validation evidence, (4) pilot a workplace communication dashboard with privacy-preserving edge deployment.

---

## Sources

- [Employee Experience Management Market (GMInsights)](https://www.gminsights.com/industry-analysis/employee-experience-management-market)
- [Content Moderation Services Market (ResearchNester)](https://www.researchnester.com/reports/content-moderation-services-market/7630)
- [Perspective API (Jigsaw/Google)](https://www.perspectiveapi.com/)
- [Perspective API Limitations (ACL Anthology)](https://aclanthology.org/2022.nlp4pi-1.2/)
- [Receptiviti LIWC API](https://www.receptiviti.com/liwc)
- [Receptiviti Measures](https://www.receptiviti.com/measures)
- [Gong Conversation Intelligence](https://www.gong.io/conversation-intelligence)
- [Gong vs Chorus Comparison (Claap)](https://www.claap.io/blog/gong-vs-chorus-which-is-better-and-why)
- [Qualtrics Employee Sentiment](https://www.qualtrics.com/articles/employee-experience/employee-sentiment/)
- [Perceptyx AI-Powered Employee Experience](https://www.perceptyx.com/)
- [Fearless Organization / Edmondson PSI](https://fearlessorganizationscan.com/the-fearless-organization)
- [Worklytics ONA](https://www.worklytics.co/organizational-network-analysis/)
- [Humanyze ONA](https://humanyze.com/improving-company-performance-with-organizational-network-analysis/)
- [Slack Discovery API for Analytics (Worklytics)](https://www.worklytics.co/blog/using-slack-discovery-api-for-analytics)
- [Eleos Health](https://eleos.health/)
- [AI in Clinical Psychology Systematic Review (Orru, 2026)](https://onlinelibrary.wiley.com/doi/full/10.1002/cpp.70242)
- [Digital Therapeutic Alliance (JMIR, 2025)](https://mental.jmir.org/2025/1/e69294)
- [NLP for Mental Health Interventions (Nature)](https://www.nature.com/articles/s41398-023-02592-2)
- [Employee Experience Trends (Perceptyx)](https://go.perceptyx.com/employee-experience-trends-web-report)
- [Employee Experience Platforms 2026 (UCToday)](https://www.uctoday.com/employee-engagement-recognition/employee-experience-platforms-2026/)
- [HR Tech 2025 Trends (Deloitte)](https://www.deloitte.com/us/en/services/consulting/articles/latest-hr-technology-trends-influencing-the-way-we-work.html)
- [DistilBERT ONNX Optimization (Hugging Face)](https://huggingface.co/blog/infinity-cpu-performance)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Community Health Metrics (Medium)](https://medium.com/@npavfan2facts/7-community-health-metrics-that-beat-discord-members-201c219b0ab3)
- [AI Moderation Tools 2026 (HuntScreens)](https://huntscreens.com/en/topic/ai-moderation)
- [AI Tools for HR 2026 (Staffbase)](https://staffbase.com/blog/ai-tools-for-hr)
- [Conversation Analytics Software 2025 (Upsolve)](https://upsolve.ai/blog/conversation-analytics-intelligence-software)
