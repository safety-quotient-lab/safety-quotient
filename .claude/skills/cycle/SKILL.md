---
name: cycle
description: Post-development checklist — update all research documentation, experiment tracking, psychometric evaluation, project memory, and data provenance to keep the PSQ project in a publishable state. Then commit and clean up.
user-invocable: true
argument-hint: [summary of what changed]
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, Task
---

# Post-Development Cycle

This cycle ensures that every code, data, or training change in the PSQ project propagates through the full documentation chain. The goal is that a reviewer picking up this project cold can reconstruct the full research narrative from the committed files alone — no institutional memory required.

**Design principle:** The PSQ project maintains several overlapping documents at different levels of abstraction. Each serves a distinct audience and purpose:

| Document | Audience | Purpose | Abstraction |
|---|---|---|---|
| `journal.md` | Peer reviewers, future self | Narrative research story — why decisions were made, what failed, what the data revealed. Written in journal-article prose with citations. | Highest — tells the story |
| `distillation-research.md` | Technical collaborators | Detailed technical log — every training run, data analysis, correlation matrix, error diagnosis. | Medium — shows the work |
| `EXPERIMENTS.md` | Reproducibility | Version-by-version parameter table — architecture, data changes, metrics. Machine-parseable where possible. | Lowest — just the facts |
| `data/DATA-PROVENANCE.md` | Data auditors, licensing | Dataset-level provenance — sources, licenses, record counts, audit findings. | Dataset-focused |
| `psychometric-evaluation.md` | Psychometricians | Validation evidence against AERA/APA/NCME (2014) standards. | Standards-focused |
| `psq-definition.md` | Theoreticians | Construct definition — what the PSQ is, its 10 dimensions, theoretical grounding. | Conceptual |
| `MEMORY.md` | Claude across sessions | Orientation state — current version, key metrics, file locations, patterns. | Cross-session context |

When something changes, the relevant documents must be updated *at the appropriate level of abstraction*. A new training version needs a row in EXPERIMENTS.md (facts), a subsection in distillation-research.md (analysis), and possibly a narrative paragraph in journal.md (if the results reveal something scientifically interesting). A routine parameter tweak might only need EXPERIMENTS.md.

## Checklist

Work through each step. Skip any that don't apply to the changes described in $ARGUMENTS. When in doubt about whether a step applies, check the document's current state — if it's already accurate, skip.

### 1. Identify what changed

- Read the recent git diff or summarize from context what files were modified
- Categorize the changes: training results, data pipeline, new scripts, new analysis, conceptual/theoretical, bugfix, documentation-only
- This categorization determines which downstream documents need updates

### 2. Update distillation-research.md

The technical research log. This is the most detailed document and the primary record of what was tried and what happened.

- **Status line** (top of file): Must reflect the current best model version, test_r, held-out_r, and the immediate next priority. Example: `v13 complete (test_r=0.553, held-out_r=0.428). CC threat_exposure removed. Next: resolve construct validity.`
- **New sections:** Major findings (new training version, new evaluation, new analysis) get their own numbered subsection. Use the existing numbering scheme (§17a, §17b, etc.). Include raw numbers, tables, and interpretation.
- **Table of Contents:** If sections were added, update the ToC. Keep the ToC format consistent with existing entries (number, title, brief descriptor).
- **Comparisons:** When reporting new training results, always include a comparison table against the previous best version (at minimum: test_r, held-out_r, key dimension changes).
- **Tone:** Technical and precise. Include specific numbers, dataset names, record counts. This is a lab notebook, not a narrative.

### 3. Update psq-definition.md and final-state.md

The construct specification documents. Only update these when the *definition* of the PSQ or its dimensions changes — not when training results change.

- Dimension definitions changed (e.g., defensive architecture redefinition from ego-defense to boundary patterns)
- Scoring rubric or score anchors updated
- Model architecture description no longer matches reality (e.g., architecture switch from DeBERTa to DistilBERT)
- Dimensionality restructuring (e.g., 10 → bifactor model, cluster identification)
- Skip if changes are purely training/data — the construct definition is independent of the model's current performance

### 4. Update data/DATA-PROVENANCE.md

The data audit trail. This document must be accurate enough that a licensing review or data ethics board could verify every training record's provenance.

- New datasets: add to the appropriate tier table with license, record count, PSQ dimensions mapped, quality assessment, and audit status
- Synthetic data batches: add to the Tier 3 table with batch name, file path, count, target dimension, and date
- Relabeled records: document in the relabeled records table
- Data removed or filtered: note the change, the rationale (with evidence — e.g., "negative correlation, bias of -2.32"), and the version in which it was removed
- Composite summary table: update total record counts, composite/LLM split, train/val/test proportions
- Pipeline diagram: update if the data flow changed (new scripts, new ingestion paths)

### 5. Update journal.md

The scientific narrative. This document tells the *story* of the PSQ research — written in the idiom of a journal article's methods and results sections. It should be readable by a peer reviewer who has never seen the project before.

**When to update:**
- A significant finding was made (data poisoning discovered, construct validity crisis, architecture decision with rationale)
- A new research direction opened (dimensionality restructuring, new validation approach)
- A training version produced results that change the narrative (held-out improvement, generalization gap analysis)

**How to write entries:**
- First-person plural ("we found," "our analysis revealed")
- Honest about failures — document what didn't work and why, not just successes
- Include citations where the work connects to established literature (e.g., "This is consistent with the JD-R model — Bakker & Demerouti, 2007")
- Interpret results in psychometric terms where applicable (e.g., "This held-out r of 0.43 is comparable to cross-sample validity coefficients for brief personality measures — Soto & John, 2017")
- Update the metrics table in §14 (Current State) with any new numbers
- Update the Table of Contents if sections were added
- Add new references to §15 if cited for the first time

**When to skip:**
- Minor code changes with no research significance
- Parameter tweaks that didn't produce notable results
- Documentation-only changes

### 5b. Update EXPERIMENTS.md

The experiment tracking table. Every training run gets a row, successful or not.

- Add a row to the Training Runs table: version, date, architecture, epochs, LR, batch size, test_r, held-out_r, data changes (brief), config changes (brief)
- For killed/aborted runs: still add a row, note "Killed — [reason]" in data changes
- If held-out evaluation was re-run on an existing model, update the per-dimension results table
- If calibration was re-run, update the calibration summary section
- If hyperparameters changed, update the Key Hyperparameters block

### 5c. Update psychometric-evaluation.md

The validation evidence document, organized against the AERA/APA/NCME (2014) psychometric standards.

- New held-out evaluation → update the scorecard's "Held-out generalization" row and relevant subsections
- New test-retest results → update reliability section
- New discriminant/convergent validity → update relevant validity section
- New calibration results → update confidence calibration section
- New construct validity evidence (e.g., halo test, factor analysis) → update construct validity section
- Update the status line at the top with current model version and key metrics
- Skip if no new psychometric evidence was generated

### 6. Update project memory

Memory file: `~/.claude/projects/-home-kashif-projects/memory/MEMORY.md`

This file orients Claude at the start of each new session. It should contain the minimum information needed to resume work without re-reading the full documentation.

- Update the SafetyQuotient status line (current model version, test_r, held-out_r, immediate next priority)
- Update training data counts if they changed
- Update key scripts list if new scripts were added
- Update key documents list if new documents were created
- Update the "key insight" line if the research direction shifted
- **Don't duplicate** what's in distillation-research.md — memory is for orientation, not detail
- Keep the PSQ section under ~60 lines

### 7. Check for orphaned references and files

- Grep for references to removed functions, renamed scripts, or old file paths across all `.md` and `.py` files
- Check if any `/tmp/psq_*` scratch files should be cleaned up (synthetic batches already ingested, old training logs, completed halo test data)
- Verify no dead code was left behind in `scripts/`
- If a new script replaces an old one, remove the old one
- Check for stale TODO comments in code

### 8. Verify scripts work

Run a quick smoke test on any modified scripts:

```bash
cd /home/kashif/projects/psychology/safety-quotient
source venv/bin/activate
# Check Python syntax on modified scripts
python -m py_compile scripts/<modified_script>.py
```

If training was done, verify the model checkpoint exists and is loadable:

```bash
python -c "import torch; cp = torch.load('models/psq-student/best.pt', map_location='cpu', weights_only=False); print('OK:', list(cp.keys())[:3])"
```

If ONNX export was done, verify the model files exist:

```bash
ls -lh models/psq-student/model.onnx models/psq-student/model_quantized.onnx
```

### 9. Git commit

- Run `git status` and `git diff --stat` to review all staged and unstaged changes
- Stage relevant files — prefer naming specific files over `git add -A`
- **Never stage:** `.env`, `venv/`, `node_modules/`, `models/*.pt` (large binaries), `models/*.onnx` (large binaries), `/tmp/` scratch files, `__pycache__/`
- **Always stage:** `*.md` documentation, `scripts/*.py`, `scripts/*.sh`, `.claude/skills/**`, `data/DATA-PROVENANCE.md`, `models/psq-student/*.json` (small config/results files)
- Write a commit message that summarizes the *why* (research outcome), not the *what* (files changed)
- Commit using the standard Co-Authored-By trailer
- Run `git status` after to verify clean working tree

### 10. Cleanup

- Remove `/tmp/psq_*` scratch files that are no longer needed:
  - Synthetic batch files already ingested into `train-llm.jsonl` → safe to remove
  - Old training logs → safe to remove
  - Halo test data files → keep if analysis is ongoing
- Check for `print()` debug statements in modified scripts
- Verify `.gitignore` covers: `venv/`, `node_modules/`, `models/`, `__pycache__/`, `*.pyc`
- Flag any new untracked files that remain after commit

### 11. Summary

Report to the user:
- **Documentation updated:** which files, what was added/changed
- **Committed:** files + commit message (abbreviated)
- **Skipped:** which steps, with reason
- **Current model status:** version, test_r, held-out_r
- **Pending work:** anything flagged for next session
- **Data integrity:** any concerns about stale references, orphaned files, or documentation inconsistencies
