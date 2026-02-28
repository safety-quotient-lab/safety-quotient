#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Unattended training queue: v25 and v26
#
# v25: 512-token context (completes 128→256→512 sweep; same flags as v24 but
#      max-length doubled again, batch 8, grad-accum 4 → effective batch 32)
# v26: 128-token, LR=1e-5 (half the v23 LR) — tests if slow training helps at
#      the proven v23 configuration
#
# Run from project root with venv active:
#   source venv/bin/activate
#   bash scripts/train_queue_v25_v26.sh
#
# Waits for v24 to finish before starting v25.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Wait for v24 ──────────────────────────────────────────────────────────────
V24_CKPT="models/psq-v24/best.pt"
log "Waiting for v24 to finish ($V24_CKPT)..."
while [ ! -f "$V24_CKPT" ]; do
    sleep 60
done
log "v24 checkpoint found. Running held-out evaluation..."

python scripts/eval_held_out.py --model "$V24_CKPT" 2>&1 | tee /tmp/psq_v24_eval.txt
log "v24 eval complete. Results in /tmp/psq_v24_eval.txt"

# ── v25: 512-token context ────────────────────────────────────────────────────
log "Starting v25 (512 tokens, batch=8, grad-accum=4)..."
python scripts/distill.py \
    --db data/psq.db \
    --drop-proxy-dims \
    --max-length 512 \
    --batch-size 8 \
    --grad-accum 4 \
    --out models/psq-v25 \
    2>&1 | tee /tmp/psq_v25_train.txt

log "v25 training complete. Running held-out evaluation..."
python scripts/eval_held_out.py --model models/psq-v25/best.pt 2>&1 | tee /tmp/psq_v25_eval.txt
log "v25 eval complete. Results in /tmp/psq_v25_eval.txt"

# ── v26: 128-token, LR=1e-5 ──────────────────────────────────────────────────
log "Starting v26 (128 tokens, LR=1e-5, slow-training test)..."
python scripts/distill.py \
    --db data/psq.db \
    --drop-proxy-dims \
    --max-length 128 \
    --lr 1e-5 \
    --out models/psq-v26 \
    2>&1 | tee /tmp/psq_v26_train.txt

log "v26 training complete. Running held-out evaluation..."
python scripts/eval_held_out.py --model models/psq-v26/best.pt 2>&1 | tee /tmp/psq_v26_eval.txt
log "v26 eval complete. Results in /tmp/psq_v26_eval.txt"

# ── Summary ───────────────────────────────────────────────────────────────────
log "All done. Summary:"
log "  v24 eval: /tmp/psq_v24_eval.txt"
log "  v25 eval: /tmp/psq_v25_eval.txt"
log "  v26 eval: /tmp/psq_v26_eval.txt"
log ""
log "Next: run /cycle to document results."
