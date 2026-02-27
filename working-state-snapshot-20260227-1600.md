# PSQ Project — Working State

This file captures the current operational state of the PSQ Content Evaluator project.
It is updated at the end of each working session. Snapshots are saved as
`working-state-snapshot-YYYYMMDD-HHMM.md`.

---

## Current Model: psq-v13

| Metric | Value |
|---|---|
| Architecture | DistilBERT → 10-dim regression (knowledge distillation) |
| Test r (avg 10 dims) | 0.564 |
| Val r (best epoch) | 0.543 |
| Held-out r (avg 10 dims) | 0.402 |
| Checkpoint | `models/psq-student/best.pt` |
| Config | `models/psq-student/config.json` |
| Calibration | `models/psq-student/calibration.json` (isotonic) |

### Per-dimension held-out r (v13)

| Dimension | Held-out r | Status |
|---|---|---|
| threat_exposure | 0.160 | weak — more labels needed |
| contractual_clarity | 0.271 | weak — more labels needed |
| regulatory_capacity | 0.325 | weak — more labels needed |
| defensive_architecture | 0.368 | moderate |
| energy_dissipation | 0.393 | moderate |
| authority_dynamics | 0.457 | moderate |
| hostility_index | 0.480 | moderate |
| resilience_baseline | 0.496 | moderate |
| trust_conditions | 0.498 | moderate |
| cooling_capacity | 0.574 | good |

---

## Database: data/psq.db

| Table | Count |
|---|---|
| texts | 19,884 |
| scores | 56,131 |
| splits (train) | 15,859 |
| splits (val) | 1,913 |
| splits (test) | 2,015 |
| splits (held-out) | 100 |

### Score method breakdown (best_scores view)

| Method | Count |
|---|---|
| composite-proxy | ~40,500 |
| joint-llm | ~12,300 |
| separated-llm | ~2,541 |
| synthetic | ~846 |

### Separated-llm labels per dimension

| Dimension | n |
|---|---|
| regulatory_capacity | 550 |
| threat_exposure | 529 |
| defensive_architecture | 350 |
| energy_dissipation | 312 |
| contractual_clarity | 300 |
| authority_dynamics | 100 |
| cooling_capacity | 100 |
| hostility_index | 100 |
| resilience_baseline | 100 |
| trust_conditions | 100 |

---

## Labeling

### Completed batches

| Batch | File | Dims | n texts | Date |
|---|---|---|---|---|
| held-out separated | `data/held-out-test.jsonl` | all 10 | 100 | pre-v13 |
| weak dims batch | `data/labeling-batch-weak-dims.jsonl` | te, rc, co | 200 | 2026-02-27 |

### Provenance fields (since 2026-02-27)

All new separated-llm labels carry:
- `scorer: claude-sonnet-4-6`
- `provider: anthropic`
- `interface: claude-code`

---

## Key Scripts

| Script | Purpose |
|---|---|
| `scripts/distill.py` | Train DistilBERT student model |
| `scripts/label_separated.py` | Extract/ingest/assemble separated-scoring batches |
| `scripts/migrate.py` | Bootstrap and incrementally ingest into `data/psq.db` |
| `scripts/evaluate.py` | Evaluate model against test/held-out splits |
| `scripts/build_composite_ground_truth.py` | Rebuild training JSONL files |

---

## Architecture Decisions

- **Separated scoring**: one LLM call per dimension per text (eliminates halo effect)
- **Longitudinal SQLite schema**: scores as observations, texts/dimensions as axes
- **best_scores view**: priority separated-llm(1) > synthetic(2) > joint-llm(3) > composite-proxy(4)
- **Per-dim sample weights [N_DIMS]**: distill.py uses tensor weights per dimension
- **distill.py DB mode**: `--db data/psq.db` reads directly from `training_data` view
- **Splits**: frozen md5(text) hash assignments, persisted in `splits` table

---

## What's Next

1. **Train v14** — `python scripts/distill.py --db data/psq.db`
   - Incorporates 600 new separated-llm labels for te/rc/co
   - Expect improvement on the 3 weak dims
2. **More labeling** — authority_dynamics, cooling_capacity, hostility_index, etc. each have only 100 separated labels
3. **Evaluate v14** — compare held-out r per dim vs v13 baseline
4. **Cycle docs** — update EXPERIMENTS.md, journal.md after v14 run

---

*Last updated: 2026-02-27*
