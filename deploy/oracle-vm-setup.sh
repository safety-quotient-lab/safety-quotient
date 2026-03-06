#!/usr/bin/env bash
# PSQ Scoring Endpoint — Oracle Ampere A1 VM Setup
# Run as the default user (ubuntu) on a fresh Ubuntu 22.04 ARM64 VM.
# Idempotent — safe to re-run.

set -euo pipefail

REPO_URL="https://github.com/safety-quotient-lab/psychology-agent.git"
INSTALL_DIR="/opt/psychology-agent"
SERVICE_USER="ubuntu"

# ──────────────────────────────────────────────────────────────────────────────
# 1. System dependencies
# ──────────────────────────────────────────────────────────────────────────────
echo "==> Installing Node.js 20 LTS + git..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs git
node --version   # Expected: v20.x.x
npm --version

# ──────────────────────────────────────────────────────────────────────────────
# 2. Clone repo
# ──────────────────────────────────────────────────────────────────────────────
echo "==> Cloning repository to ${INSTALL_DIR}..."
if [ -d "${INSTALL_DIR}/.git" ]; then
  echo "    Repo already present — pulling latest..."
  git -C "${INSTALL_DIR}" pull
else
  sudo git clone "${REPO_URL}" "${INSTALL_DIR}"
  sudo chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 3. Install npm dependencies (safety-quotient only)
# ──────────────────────────────────────────────────────────────────────────────
echo "==> Installing npm dependencies..."
cd "${INSTALL_DIR}/safety-quotient"
npm install --production
# onnxruntime-node@1.24.2 ships pre-built ARM64 Linux binaries — no recompile needed.

# ──────────────────────────────────────────────────────────────────────────────
# 4. Verify model files (rsynced separately from Debian psq-agent)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR="${INSTALL_DIR}/safety-quotient/models/psq-student"
REQUIRED_FILES=("model.onnx" "config.json" "calibration.json")
echo "==> Checking model files in ${MODEL_DIR}..."
all_present=true
for required_file in "${REQUIRED_FILES[@]}"; do
  if [ -f "${MODEL_DIR}/${required_file}" ]; then
    file_size=$(du -h "${MODEL_DIR}/${required_file}" | cut -f1)
    echo "    ✓  ${required_file} (${file_size})"
  else
    echo "    ✗  MISSING: ${required_file}"
    all_present=false
  fi
done
if [ -d "${MODEL_DIR}/tokenizer" ]; then
  echo "    ✓  tokenizer/"
else
  echo "    ✗  MISSING: tokenizer/"
  all_present=false
fi
if [ "$all_present" = false ]; then
  echo ""
  echo "ERROR: Model files missing. Run rsync from Debian psq-agent first:"
  echo "  rsync -avz --progress <DEBIAN_USER>@<DEBIAN_IP>:${MODEL_DIR}/ ${MODEL_DIR}/"
  echo "Then re-run this script."
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# 5. Install systemd service
# ──────────────────────────────────────────────────────────────────────────────
echo "==> Installing systemd service..."
sudo cp "${INSTALL_DIR}/safety-quotient/deploy/psq-server.service" \
        /etc/systemd/system/psq-server.service
sudo systemctl daemon-reload
sudo systemctl enable psq-server
sudo systemctl restart psq-server

# ──────────────────────────────────────────────────────────────────────────────
# 6. Health check
# ──────────────────────────────────────────────────────────────────────────────
echo "==> Waiting 15s for server startup (ONNX model init ~8s)..."
sleep 15
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/health 2>/dev/null || echo "000")
if [ "${HTTP_STATUS}" = "200" ]; then
  echo "    ✓  /health → 200 OK"
  curl -s http://localhost:3000/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:3000/health
else
  echo "    ✗  /health → HTTP ${HTTP_STATUS} (server may still be initializing)"
  echo "    Check: sudo journalctl -u psq-server -n 50 --no-pager"
fi

echo ""
echo "==> Done. PSQ server running on http://localhost:3000"
echo "    Public URL: configure nginx + TLS or open port 3000 in Oracle VCN security list."
