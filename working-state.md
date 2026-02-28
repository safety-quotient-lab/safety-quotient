# PSQ Project — Working State

This file captures the current operational state of the PSQ Content Evaluator project.
It is updated at the end of each working session. Snapshots are saved as
`working-state-snapshot-YYYYMMDD-HHMM.md`.

---

## Current Model: psq-v15 (complete)

| Metric | Value |
|---|---|
| Architecture | DistilBERT → 10-dim regression (knowledge distillation) |
| Test r (avg 10 dims) | 0.536 |
| Val r (best epoch) | 0.523 (epoch 7 of 10) |
| Held-out r (avg 10 dims) | 0.495 (+0.013 vs v14, +0.093 vs v13) |
| Generalization gap | 7.6% (down from 11.4% in v14) |
| Checkpoint | `models/psq-v15/best.pt` |
| Config | `models/psq-v15/config.json` |

### Per-dimension held-out r (v15)

| Dimension | Held-out r | Δ vs v14 | Status |
|---|---|---|---|
| regulatory_capacity | 0.285 | +0.041 | weak — partial recovery, still worst |
| contractual_clarity | 0.388 | -0.110 | moderate — regressed, investigate |
| threat_exposure | 0.410 | -0.066 | moderate — regressed |
| resilience_baseline | 0.507 | +0.063 | good — improved |
| energy_dissipation | 0.511 | -0.020 | good — slight regression |
| defensive_architecture | 0.523 | +0.017 | good — improved |
| hostility_index | 0.538 | +0.050 | good — improved |
| trust_conditions | 0.564 | -0.008 | good — stable |
| authority_dynamics | 0.573 | +0.166 | good — massive improvement from AD batch |
| cooling_capacity | 0.653 | 0.000 | good — unchanged |

---

## Database: data/psq.db

| Table | Count |
|---|---|
| texts | 20,127 |
| scores | 63,361 |
| splits (train) | 15,859 |
| splits (val) | 1,913 |
| splits (test) | 2,015 |
| splits (held-out) | 100 |

### Score method breakdown

| Method | Count |
|---|---|
| composite-proxy | 40,487 |
| joint-llm | 12,257 |
| separated-llm | 9,771 |
| synthetic | 846 |

---

## Labeling

### Completed batches

| Batch | File | Dims | n texts | Date |
|---|---|---|---|---|
| held-out separated | `data/held-out-test.jsonl` | all 10 | 100 | pre-v13 |
| weak dims batch | `data/labeling-batch-weak-dims.jsonl` | all 10 | 200 | 2026-02-27 |
| rc focus batch | `data/labeling-batch-rc.jsonl` | all 10 | 150 | 2026-02-27 |

| ad focus batch | `data/labeling-batch-ad.jsonl` | all 10 | 300 | 2026-02-27 |

### Provenance fields (since 2026-02-27)

All new separated-llm labels carry:
- `scorer: claude-sonnet-4-6`
- `provider: anthropic`
- `interface: claude-code`

---

## Architecture Decisions

- **Separated scoring**: one LLM call per dimension per text (eliminates halo effect)
- **Longitudinal SQLite schema**: scores as observations, texts/dimensions as axes
- **best_scores view**: priority separated-llm(1) > synthetic(2) > joint-llm(3) > composite-proxy(4)
- **Per-dim sample weights [N_DIMS]**: distill.py uses tensor weights per dimension
- **distill.py DB mode**: `--db data/psq.db` reads directly from `training_data` view
- **Splits**: frozen md5(text) hash assignments, persisted in `splits` table

---

## Key Scripts

| Script | Purpose |
|---|---|
| `scripts/distill.py` | Train DistilBERT student model (`--out DIR`, `--no-save` for smoke tests) |
| `scripts/label_separated.py` | Extract/ingest/assemble separated-scoring batches |
| `scripts/migrate.py` | Bootstrap and incrementally ingest (`--ingest JSONL`) into `data/psq.db` |
| `scripts/eval_held_out.py` | Evaluate model against 100-text held-out benchmark |
| `scripts/build_composite_ground_truth.py` | Rebuild training JSONL files |

---

## Operational Lessons

### Context Limit on Large Labeling Sessions

The RC batch (150 texts × 10 dims) exhausted the Claude Code context window before assemble/ingest could run. Score files in `/tmp/psq_separated/` persisted safely and were recovered in the next session.

**Mitigations:**
- Assemble after every 2-3 dimensions instead of waiting for all 10
- Budget context for post-processing (assemble + ingest + docs)
- Use `label_separated.py status` to verify progress before ending a session
- Sub-batch with `--offset`/`--limit` for batches >100 texts

---

## What's Next

1. **Investigate co regression** — contractual_clarity held-out dropped 0.498→0.388 despite test_r improving; distributional shift?
2. **Promote v15 to production** — copy `models/psq-v15/best.pt` → `models/psq-student/best.pt`, re-run calibration
3. **Plan v16 labeling** — priorities: co (regressed), rc (still weakest), te (regressed)
4. **Score more batches from unlabeled pool** — ~7K texts available in `data/unlabeled-pool.jsonl`

---

*Last updated: 2026-02-27*
