#!/bin/bash
# V3b launch sequence — run after v3 completes
set -e
cd /home/kashif/projects/psychology/safety-quotient
source venv/bin/activate

echo "=== Step 1: Back up v3 results ==="
cp models/psq-student/best_results.json models/psq-student/v3_best_results.json 2>/dev/null || echo "No best_results.json yet"
cp models/psq-student/test_results.json models/psq-student/v3_test_results.json 2>/dev/null || echo "No test_results.json yet"
cp models/psq-student/config.json models/psq-student/v3_config.json 2>/dev/null || echo "No config.json yet"
cp models/psq-student/v3_train.log models/psq-student/v3_train_final.log
echo "  Backed up v3 artifacts"

echo "=== Step 2: Verify v3b composite ==="
COMPOSITE_COUNT=$(wc -l < data/composite-ground-truth.jsonl)
LLM_COUNT=$(wc -l < data/train-llm.jsonl)
echo "  Composite: $COMPOSITE_COUNT records"
echo "  LLM labels: $LLM_COUNT records"
echo "  Total: $((COMPOSITE_COUNT + LLM_COUNT)) records"

echo "=== Step 3: Launch v3b training ==="
nohup python scripts/distill.py --epochs 10 --max-length 256 > models/psq-student/v3b_train.log 2>&1 &
echo "  PID: $!"
echo "  Log: models/psq-student/v3b_train.log"
echo "  Monitor: tail -f models/psq-student/v3b_train.log"
