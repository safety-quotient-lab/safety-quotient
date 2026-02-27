---
name: cycle
description: Post-development checklist — update research docs, memory, data provenance, verify scripts, commit, and cleanup after code or data changes
user-invocable: true
argument-hint: [summary of what changed]
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, Task
---

# Post-Development Cycle

Run this after completing code or data changes to ensure all documentation, research notes, and project memory stay in sync, then commit and clean up.

## Checklist

Work through each step. Skip any that don't apply to the changes described in $ARGUMENTS.

### 1. Identify what changed

- Read the recent git diff or summarize from context what files were modified
- Note any new scripts, changed training pipeline, new data sources, updated results, or modified model architecture

### 2. Update distillation-research.md

- Check if the status line at the top is still accurate (model version, avg_r, held-out_r)
- If new training results exist, update the comparison table in §14e
- If new findings were documented in a section, ensure the Table of Contents is current
- Add new sections for significant findings (new training versions, new evaluation results, methodology changes)
- Keep existing sections intact — append, don't rewrite history

### 3. Update psq-definition.md and final-state.md

- Check if dimension definitions changed (e.g., defensive architecture redefinition)
- Check if scoring rubric or anchors were updated
- Check if the model architecture description is still accurate
- Skip if no conceptual changes were made (purely training/data work)

### 4. Update data/DATA-PROVENANCE.md

- If new datasets were added or existing ones modified, update provenance records
- If synthetic data batches were generated, document them (batch name, count, target dimension, date)
- If data was removed or filtered, note the change and rationale
- Verify license information is current

### 5. Update psychometric-evaluation.md

- If new validation results exist (held-out eval, test-retest, discriminant, known-groups), update the relevant sections
- If confidence calibration was re-run, update calibration results
- Skip if no new psychometric work was done

### 6. Update project memory

Memory file: `~/.claude/projects/-home-kashif-projects/memory/MEMORY.md`

- Update the SafetyQuotient status line (current model version, key metrics)
- Record any new stable patterns confirmed during this session
- Update training data counts if they changed
- Update key scripts list if new scripts were added
- Don't duplicate what's in distillation-research.md — memory is for cross-session orientation

### 7. Check for orphaned references and files

- Grep for any references to removed functions, renamed scripts, or old file paths
- Check if any `/tmp/psq_*` scratch files should be cleaned up
- Verify no dead code was left behind in scripts/
- If a new script replaces an old one, remove the old one

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

### 9. Git commit

- Run `git status` and `git diff --stat` to review all staged and unstaged changes
- Stage relevant files — prefer naming specific files over `git add -A`
- Never stage: `.env`, `venv/`, `node_modules/`, `models/*.pt` (large binaries), `/tmp/` scratch files, `__pycache__/`
- Write a concise commit message summarizing the "why"
- Commit using the standard Co-Authored-By trailer
- Run `git status` after to verify clean working tree

### 10. Cleanup

- Remove any `/tmp/psq_*` scratch files that are no longer needed (synthetic batches already ingested, old training logs)
- Check for any `print()` debug statements that should be removed from scripts
- Verify `.gitignore` covers: `venv/`, `node_modules/`, `models/`, `__pycache__/`, `*.pyc`, `/tmp/`
- If new untracked files remain after commit, flag them

### 11. Summary

Report:
- What documentation was updated
- What was committed (files + message)
- What was skipped (with reason)
- Current model status (version, test_r, held-out_r)
- Any pending work flagged for next session
