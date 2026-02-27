#!/bin/bash
# V4 DeBERTa launch — architecture upgrade
# Use after v3b results are evaluated
set -e
cd /home/kashif/projects/psychology/safety-quotient
source venv/bin/activate

echo "=== Back up v3b results ==="
cp models/psq-student/best_results.json models/psq-student/v3b_best_results.json 2>/dev/null || true
cp models/psq-student/test_results.json models/psq-student/v3b_test_results.json 2>/dev/null || true
cp models/psq-student/config.json models/psq-student/v3b_config.json 2>/dev/null || true
cp models/psq-student/v3b_train.log models/psq-student/v3b_train_final.log 2>/dev/null || true

echo "=== Launch DeBERTa-v3-small training ==="
# 141M params (vs DistilBERT 66.7M), ~3-5 points better on NLU benchmarks
# GTX 1060 6GB: batch 8 + grad_accum 4 = effective batch 32
# Two-phase confidence: no conf loss epochs 1-2, then accuracy-based conf loss
# Confidence power 2.0: squared weighting to suppress low-conf proxy noise
#   (fixes authority_dynamics negative r caused by politeness data at conf=0.26)
nohup python scripts/distill.py \
    --model-name microsoft/deberta-v3-small \
    --epochs 10 \
    --max-length 256 \
    --batch-size 8 \
    --grad-accum 4 \
    --conf-mode two-phase \
    --conf-warmup-epochs 2 \
    --conf-power 2.0 \
    > models/psq-student/v4_deberta_train.log 2>&1 &
echo "  PID: $!"
echo "  Log: models/psq-student/v4_deberta_train.log"
echo "  Monitor: tail -f models/psq-student/v4_deberta_train.log"
