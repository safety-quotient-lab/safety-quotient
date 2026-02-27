#!/bin/bash
# v10 Launch Script
# Ingests all pending data (synthetic + relabeled), deduplicates, launches training
# Usage: source venv/bin/activate && bash scripts/launch_v10.sh

set -e
cd "$(dirname "$0")/.."

echo "=== v10 Data Ingestion ==="
python scripts/ingest_v10_data.py

echo ""
echo "=== Launching v10 DistilBERT ==="
nohup python scripts/distill.py \
    --epochs 10 \
    --conf-mode two-phase \
    --patience 3 \
    > /tmp/psq_v10_training.log 2>&1 &

echo "v10 PID: $!"
echo "Log: /tmp/psq_v10_training.log"
echo "Monitor: tail -f /tmp/psq_v10_training.log | grep -E 'Epoch|Early|Test|AVER'"
