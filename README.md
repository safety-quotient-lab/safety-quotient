# Psychoemotional Safety Quotient (PSQ)

A 10-dimension framework and DistilBERT student model for measuring **psychoemotional safety climate** in text content — from social media posts and workplace communications to negotiation transcripts and online discussions.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## What It Measures

PSQ scores text across 10 psychologically-grounded dimensions, each anchored to 3–5 validated instruments from clinical, organizational, and social psychology:

| Dim | Name | Description |
|-----|------|-------------|
| TE | Threat Exposure | Degree to which content signals danger, aggression, or harm |
| RC | Regulatory Capacity | Presence of emotion regulation strategies |
| RB | Resilience Baseline | Signals of adaptive coping and bounce-back capacity |
| TC | Trust Conditions | Interpersonal trust and cooperative orientation |
| HI | Hostility Index | Anger, aggression, and hostile attribution patterns |
| CC | Cooling Capacity | De-escalation, reappraisal, and recovery signals |
| AD | Authority Dynamics | Power asymmetry, dominance, and status negotiation |
| DA | Defensive Architecture | Psychological boundary maintenance and self-protection |
| CO | Contractual Clarity | Implied relational agreements and norm adherence |
| ED | Energy Dissipation | Sustained engagement capacity and motivational resources |

Each dimension is scored 0–10. The g-PSQ (unweighted average) provides a single overall safety index.

---

## Model Performance

Current model: **v23** (DistilBERT, held-out r = 0.696)

| Metric | v21 | v22a | **v23** |
|--------|-----|------|---------|
| Held-out r | 0.630 | 0.682 | **0.696** |
| TE | 0.492 | 0.805 | **0.800** |
| ED | 0.636 | 0.712 | **0.768** |
| RC | 0.729 | 0.756 | **0.782** |
| CO | 0.555 | 0.504 | **0.549** |

Held-out evaluation on 100 stratified texts scored by separated LLM scoring (one dimension per session to eliminate halo effects).

---

## Criterion Validity

PSQ profiles predict meaningful real-world outcomes across 4 independent datasets:

| Dataset | N | Task | AUC / R² |
|---------|---|------|----------|
| CaSiNo (negotiation) | 1,030 | Satisfaction prediction | R²=+0.016 incremental |
| CGA-Wiki (talk pages) | 4,188 | Conversation derailment | AUC=0.599 |
| CMV (Reddit) | 4,263 | Persuasion success | AUC=0.5735 |
| DonD (negotiation) | 12,234 | Deal outcome | **AUC=0.732** |

Profile shape consistently outperforms the g-PSQ average, demonstrating that the dimensional structure carries genuine predictive information beyond valence.

---

## Data Pipeline

```
Raw text
    │
    ▼
LLM-as-judge (Claude) — separated scoring, one dimension per session
    │
    ▼
SQLite (psq.db) — 22,186 texts, 90,361 scores
    │
    ▼
DistilBERT fine-tuning — curriculum learning (LLM-first → proxy supplement)
    │
    ▼
Held-out evaluation — 100 stratified texts
```

**Separated scoring:** Each dimension is labeled in an independent Claude Code session to eliminate halo effects (inter-dimension correlation inflation from joint scoring).

---

## Repository Structure

```
├── data/
│   ├── psq.db                    # SQLite database (texts + scores)
│   ├── schema.sql                # DB schema
│   ├── unlabeled-pool.jsonl      # ~7K texts available for labeling
│   └── labeling-batch-*.jsonl    # Scored labeling batches
├── models/
│   └── psq-v*/                   # Trained model checkpoints
├── scripts/
│   ├── distill.py                # Training script (DistilBERT)
│   ├── label_separated.py        # LLM labeling tool (separated scoring)
│   ├── build_composite_ground_truth.py
│   └── migrate.py                # DB migration
├── sessions/                     # Raw research session transcripts (Git LFS)
├── psq-definition.md             # Construct definitions and scoring rubrics
├── distillation-research.md      # Full technical research log
├── lab-notebook.md               # Structured session extraction
├── journal.md                    # Curated research narrative
└── psychometric-evaluation.md    # Psychometric properties and validity evidence
```

---

## Key Technical Details

**Training:**
```bash
# Standard run
python scripts/distill.py --out models/psq-vN

# With proxy removal + curriculum learning (current best configuration)
python scripts/distill.py --drop-proxy-dims --curriculum --out models/psq-vN

# Smoke test (no save, 1 epoch)
python scripts/distill.py --no-save --epochs 1
```

**Labeling (separated scoring):**
```bash
# Extract batch for scoring
python scripts/label_separated.py extract --input data/batch.jsonl

# Check progress
python scripts/label_separated.py status

# Ingest scored dimension
python scripts/label_separated.py ingest --dim te --scores /tmp/scored.json

# Assemble final labeled file
python scripts/label_separated.py assemble --input data/batch.jsonl --output data/batch-labeled.jsonl
```

---

## Psychometric Properties

- **g-factor:** EV1 = 7.225 (72.3% variance), KMO = 0.902 ("Superb"), N=2,319
- **Range-dependent structure:** g-factor collapses in middle-g texts (39% variance at g∈[4,6]) — dimensions genuinely differentiate where valence is ambiguous
- **Factor structure:** 1 dominant general factor + residual cluster structure (5 groups)
- **Expert validation:** Protocol designed (5 raters, 200 texts, ICC targets ≥0.70); recruitment not yet started

---

## Research Status

The PSQ is a research-stage instrument. It has not undergone human expert validation. Current validation evidence is:
- ✅ Criterion validity (4 datasets, held-out evaluation)
- ✅ Factor structure (EFA, parallel analysis)
- ✅ Construct operationalization (~100 anchoring instruments)
- ⏳ Expert inter-rater reliability (protocol designed, not executed)
- ⏳ ONNX production export (pending model promotion decision)

---

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](LICENSE)

**Citation:**
```
Shah, K. (2026). Psychoemotional Safety Quotient (PSQ): A 10-dimension
framework for measuring psychoemotional safety climate in text.
Safety Quotient Lab. https://github.com/safety-quotient-lab/safety-quotient
```

---

**Principal Investigator:** Kashif Shah
**Research Assistant:** Claude (Anthropic) — LLM-assisted construct operationalization, data labeling, and analysis
