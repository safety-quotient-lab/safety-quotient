#!/usr/bin/env bash
# PSQ Production Deploy — Hetzner CX (178.156.229.103)
#
# Runs the full post-training pipeline locally, then syncs model files
# to Hetzner and restarts the scoring service.
#
# Prerequisites:
#   - models/psq-vN/best.pt exists (training complete)
#   - venv activated: source venv/bin/activate
#   - SSH access: ssh root@178.156.229.103 works without password (key auth)
#
# Usage:
#   bash deploy/hetzner-deploy.sh --model models/psq-v28
#
# WARNING: Do NOT run npm install on Hetzner after deploying.
#   onnxruntime-node has a nested conflict that was manually resolved.
#   npm install will recreate the conflict and break the service.
#   See: transport/sessions/psychology-interface/model-rsync-response-001.json

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

REMOTE_HOST="root@178.156.229.103"
REMOTE_WORKING_DIR="/opt/psychology-agent/safety-quotient"
REMOTE_MODEL_DIR="${REMOTE_WORKING_DIR}/models/psq-student"
SERVICE_NAME="psq-server"
HEALTH_URL="https://psq.unratified.org/health"
SCORE_URL="https://psq.unratified.org/score"

LOCAL_STUDENT_DIR="models/psq-student"

# ── Argument parsing ──────────────────────────────────────────────────────────

MODEL_DIR=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_DIR" ]]; then
    echo "Usage: bash deploy/hetzner-deploy.sh --model models/psq-vN [--dry-run]"
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/best.pt" ]]; then
    echo "ERROR: ${MODEL_DIR}/best.pt not found. Training complete?"
    exit 1
fi

echo "=== PSQ Hetzner Deploy ==="
echo "Source model:   ${MODEL_DIR}/best.pt"
echo "Target:         ${REMOTE_HOST}:${REMOTE_MODEL_DIR}"
echo "Dry run:        ${DRY_RUN}"
echo ""

# ── Step 1: Post-training pipeline (local) ────────────────────────────────────

echo "Step 1: Copying best.pt to ${LOCAL_STUDENT_DIR}/best.pt ..."
if [[ "$DRY_RUN" == "false" ]]; then
    cp "${MODEL_DIR}/best.pt" "${LOCAL_STUDENT_DIR}/best.pt"
fi

echo "Step 2: Calibration ..."
if [[ "$DRY_RUN" == "false" ]]; then
    python scripts/calibrate.py --model-dir "${MODEL_DIR}" --out "${LOCAL_STUDENT_DIR}/calibration.json"
fi

echo "Step 3: ONNX export ..."
if [[ "$DRY_RUN" == "false" ]]; then
    python scripts/export_onnx.py --checkpoint "${MODEL_DIR}/best.pt"
fi

echo "Step 4: Held-out evaluation ..."
if [[ "$DRY_RUN" == "false" ]]; then
    python scripts/eval_held_out.py --model "${MODEL_DIR}/best.pt"
fi

# ── Step 2: SHA256 local ──────────────────────────────────────────────────────

echo ""
echo "Step 5: Local SHA256 checksums ..."
LOCAL_ONNX_HASH=$(sha256sum "${LOCAL_STUDENT_DIR}/model.onnx" | cut -d' ' -f1)
LOCAL_QUANT_HASH=$(sha256sum "${LOCAL_STUDENT_DIR}/model_quantized.onnx" | cut -d' ' -f1)
echo "  model.onnx:           ${LOCAL_ONNX_HASH}"
echo "  model_quantized.onnx: ${LOCAL_QUANT_HASH}"

# ── Step 3: Backup current model on Hetzner ─────────────────────────────────

echo ""
echo "Step 6: Backing up current model on Hetzner ..."
if [[ "$DRY_RUN" == "false" ]]; then
    ssh "${REMOTE_HOST}" "cp ${REMOTE_MODEL_DIR}/model_quantized.onnx ${REMOTE_MODEL_DIR}/model_quantized.onnx.bak 2>/dev/null && echo '  Backup created ✓' || echo '  No existing model to back up (first deploy?)'"
    ssh "${REMOTE_HOST}" "cp ${REMOTE_MODEL_DIR}/calibration.json ${REMOTE_MODEL_DIR}/calibration.json.bak 2>/dev/null || true"
else
    echo "  (dry run — skipping backup)"
fi

# ── Step 4: rsync to Hetzner ──────────────────────────────────────────────────

echo ""
echo "Step 7: rsync model files to Hetzner ..."
RSYNC_FLAGS="-avz --progress"
if [[ "$DRY_RUN" == "true" ]]; then
    RSYNC_FLAGS="${RSYNC_FLAGS} --dry-run"
fi

rsync ${RSYNC_FLAGS} \
    --include="*.onnx" \
    --include="*.json" \
    --include="*.pt" \
    --include="tokenizer/" \
    --include="tokenizer/**" \
    --exclude="*" \
    "${LOCAL_STUDENT_DIR}/" \
    "${REMOTE_HOST}:${REMOTE_MODEL_DIR}/"

# ── Step 5: Verify SHA256 on Hetzner ─────────────────────────────────────────

echo ""
echo "Step 8: Verifying SHA256 on Hetzner ..."
REMOTE_ONNX_HASH=$(ssh "${REMOTE_HOST}" "sha256sum ${REMOTE_MODEL_DIR}/model.onnx" | cut -d' ' -f1)
REMOTE_QUANT_HASH=$(ssh "${REMOTE_HOST}" "sha256sum ${REMOTE_MODEL_DIR}/model_quantized.onnx" | cut -d' ' -f1)

echo "  model.onnx:           ${REMOTE_ONNX_HASH}"
echo "  model_quantized.onnx: ${REMOTE_QUANT_HASH}"

if [[ "$LOCAL_ONNX_HASH" != "$REMOTE_ONNX_HASH" ]]; then
    echo "ERROR: model.onnx SHA256 mismatch — aborting restart"
    exit 1
fi
if [[ "$LOCAL_QUANT_HASH" != "$REMOTE_QUANT_HASH" ]]; then
    echo "ERROR: model_quantized.onnx SHA256 mismatch — aborting restart"
    exit 1
fi
echo "  SHA256 verified ✓"

# ── Step 6: Restart service ───────────────────────────────────────────────────

echo ""
echo "Step 9: Restarting ${SERVICE_NAME} ..."
if [[ "$DRY_RUN" == "false" ]]; then
    ssh "${REMOTE_HOST}" "systemctl restart ${SERVICE_NAME}"
    sleep 10  # wait for ONNX model load (~8s)
fi

# ── Step 7: Health check ──────────────────────────────────────────────────────

echo ""
echo "Step 10: Health check ..."
if [[ "$DRY_RUN" == "false" ]]; then
    HEALTH=$(curl -sf "${HEALTH_URL}" || echo '{"status":"error"}')
    echo "  ${HEALTH}"
    if ! echo "${HEALTH}" | grep -q '"status":"ok"'; then
        echo "ERROR: Health check failed"
        exit 1
    fi
    echo "  Health check passed ✓"
fi

# ── Step 8: Scoring smoke test ────────────────────────────────────────────────

echo ""
echo "Step 11: Scoring smoke test ..."
if [[ "$DRY_RUN" == "false" ]]; then
    SCORE=$(curl -sf -X POST "${SCORE_URL}" \
        -H "Content-Type: application/json" \
        -d '{"text": "The team felt safe raising concerns in this meeting."}' \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'composite={d[\"scores\"][\"psq_composite\"][\"value\"]:.1f}, calibration={d[\"scores\"][\"calibration_version\"]}, dims={len(d[\"dimensions\"])}')" \
        2>/dev/null || echo "score request failed")
    echo "  ${SCORE}"
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "=== Deploy complete ==="
echo "  Local SHA256 model.onnx:  ${LOCAL_ONNX_HASH:0:16}..."
echo "  Remote SHA256 model.onnx: ${REMOTE_ONNX_HASH:0:16}..."
echo ""
echo "Reminder: do NOT run npm install on Hetzner (onnxruntime-node conflict)."
