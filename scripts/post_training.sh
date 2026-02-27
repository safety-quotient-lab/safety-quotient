#!/bin/bash
# Post-training pipeline: eval → calibrate → export
# Usage: source venv/bin/activate && bash scripts/post_training.sh
set -e
cd "$(dirname "$0")/.."

echo "=== 1/3: Held-Out Evaluation ==="
python scripts/eval_held_out.py
echo ""

echo "=== 2/3: Calibration ==="
python scripts/calibrate.py
echo ""

echo "=== 3/3: ONNX Export ==="
python scripts/export_onnx.py
echo ""

echo "=== Done ==="
echo "Artifacts in models/psq-student/:"
ls -lh models/psq-student/*.{onnx,json} 2>/dev/null
