# SafetyQuotient — Next Steps

Remaining tasks from the halo-correction planning session (2026-02-27).

| # | Suggestion | Priority | Rationale | Depends on |
|---|---|---|---|---|
| 3 | Large-scale relabeling of train-llm.jsonl with separated scoring | High | Removes halo from training signal (4,199 records × 10 calls each) | `label_separated.py` ready |
| 4 | Threat exposure rehabilitation (specialized synthetic + eval) | High | Worst dimension (r=0.12 on held-out); needs targeted data | — |
| 6 | V14 training run on halo-corrected data | High | Re-train on separated labels; expect ~0.05–0.10 correlation lift | #3 done |
| 7 | Larger halo test (100+ texts) for publication-quality evidence | Medium | Current N=30 is pilot-scale; need N≥100 for stable effect sizes | — |
| 8 | Human validation study (SME ratings vs model scores) | Medium | Gold standard for construct validity; no SME data exists yet | — |
| 9 | Cross-domain generalization (clinical, educational, organizational) | Medium | Held-out is workplace-heavy; untested on other domains | — |
| 10 | Deployment pipeline (API endpoint, batch scoring, monitoring) | Low | After psychometric validation is satisfactory | #6, #8 done |

## Already completed this session

- **Separated scoring script** — `scripts/label_separated.py` (one dimension per API call)
- **Held-out re-scoring** — 100 texts re-labeled with separated calls → `data/held-out-test.jsonl`
- **Hierarchical reporting** — g-PSQ + cluster subscales in `student.js` and `detector.js`
