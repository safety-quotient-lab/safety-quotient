# Scoring Experiments Raw Data (2026-02-28)

Intermediate data from the PSQ scoring experiments (Phase 0-3).
Archived for reproducibility. Full protocol and results in `scoring-experiments.md`.

## Files

### Experiment 1: Halo-Awareness Instructions (N=30 texts)
- `psq_exp1_control.jsonl` — 30 texts scored with standard prompt (all 10 dims)
- `psq_exp1_treatment.jsonl` — same 30 texts scored with halo-aware prompt

### Experiment 2: Dissimilar Rubrics (N=30 texts)
- `psq_exp2_control.jsonl` — assembled control condition
- `psq_exp2_treatment.jsonl` — assembled treatment condition
- `psq_exp2_control_{dim}.json` — per-dimension control scores (10 files)
- `psq_exp2_treatment_{dim}.json` — per-dimension treatment scores (10 files)

### Experiment 3: Scale Format (N=20 texts)
- `psq_exp3_scale10_{dim}.json` — 0-10 scale scores (10 files)
- `psq_exp3_scale7_{dim}.json` — 1-7 scale scores (10 files)

### Criterion Validity Gate
- `psq_casino_gate.jsonl` — 40 CaSiNo texts selected for gate
- `psq_casino_gate_scored.jsonl` — same texts scored with halo-aware prompt

### Other
- `psq_efa_v3.py` — factor analysis script used for pct-scored data analysis

## Decisions
- Exp 1 (halo-awareness): Initially ADOPTED, then REVERSED after g-factor structural analysis
- Exp 2 (dissimilar rubrics): REJECTED (construct redefinition, not halo reduction)
- Exp 3 (scale format): RETAINED 0-10 (negligible effect)
- All three interventions REJECTED. g-factor is real co-variation, not scorer artifact.
