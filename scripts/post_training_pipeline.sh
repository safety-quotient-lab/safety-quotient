#!/bin/bash
# Post-Training Pipeline — run after best model is identified
# Usage: source venv/bin/activate && bash scripts/post_training_pipeline.sh
#
# Prerequisites: models/psq-student/best.pt exists with the winning model

set -e
cd "$(dirname "$0")/.."

echo "=== Post-Training Pipeline ==="
echo ""

# 1. Calibration (score + confidence isotonic regression)
echo "Step 1: Calibration..."
python scripts/calibrate.py
echo "  Done — calibration params saved"
echo ""

# 2. ONNX Export
echo "Step 2: ONNX Export..."
python scripts/export_onnx.py
echo "  Done — ONNX model exported"
echo ""

# 3. Validation Battery
echo "Step 3: Test-Retest Reliability..."
python scripts/test_retest_reliability.py
echo ""

echo "Step 4: Discriminant Validity (vs VADER sentiment)..."
python scripts/validate_discriminant_sentiment.py
echo ""

echo "Step 5: Confidence Calibration Check..."
python scripts/validate_confidence_calibration.py
echo ""

echo "Step 6: Known-Groups Validation..."
python scripts/validate_known_groups.py
echo ""

echo "=== Pipeline Complete ==="
echo "Review results above. If avg_r >= 0.60:"
echo "  - Update distillation-research.md with final results"
echo "  - Update psychometric-evaluation.md"
echo "  - Commit model + calibration files"
echo "  - Consider launching validation study (see theoretical-refinements.md §4)"
