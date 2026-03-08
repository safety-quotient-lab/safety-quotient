# PSQ Production Deployment

Last updated: 2026-03-08

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Hetzner CX22 — 178.156.229.103                            │
│  Debian 13 (trixie) · 3 vCPU AMD EPYC · 4 GB RAM · 75 GB  │
│                                                             │
│  ┌──────────┐     ┌──────────────────┐     ┌────────────┐  │
│  │  Caddy    │────▶│  Node.js v20     │────▶│  ONNX      │  │
│  │  :443 TLS │     │  server.js :3000 │     │  Runtime   │  │
│  └──────────┘     └──────────────────┘     └────────────┘  │
│                                                             │
│  psq.unratified.org  →  reverse_proxy localhost:3000        │
└─────────────────────────────────────────────────────────────┘
```

**Request flow:** Client → Caddy (TLS termination, auto-cert) → Node.js `server.js`
→ `student.js` (ONNX inference) → JSON response (machine-response/v3 or v3.1)

**Typical latency:** ~38 ms inference (end-to-end ~50-80 ms including TLS)


## Server Access

| Field | Value |
|-------|-------|
| Host | `178.156.229.103` |
| User | `root` |
| Auth | SSH key (on-disk, no password) |
| DNS | `psq.unratified.org` |

```bash
ssh root@178.156.229.103
```


## Endpoints

### GET /health

Liveness check. Returns model readiness and calibration version.

```bash
curl https://psq.unratified.org/health
```

```json
{"status": "ok", "model": "psq-student", "calibration_version": "isotonic-v2-2026-03-06"}
```

### POST /score

Score text across all 10 PSQ dimensions.

```bash
curl -X POST https://psq.unratified.org/score \
  -H "Content-Type: application/json" \
  -d '{"text": "The team felt safe raising concerns."}'
```

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | yes | Text to score |
| `session_id` | string | no | Client session ID (auto-generated if absent) |
| `context` | string | no | Use case context for weighted composite. Valid: `moderation`, `persuasion`, `negotiation`, `workplace`, `therapeutic` |

**Response:** `psychology-agent/machine-response/v3` JSON. When `context` is specified,
response schema bumps to `v3.1` and adds `context_weighted_composite` (0–10) and
`context_weights_used`.


## Firewall

UFW active, allow-list only:

| Port | Service |
|------|---------|
| 22/tcp | SSH |
| 80/tcp | HTTP (Caddy redirect → 443) |
| 443/tcp | HTTPS (Caddy TLS, auto-cert via Let's Encrypt) |
| 3000/tcp | Node.js direct (should be localhost-only — see Hardening below) |


## Service Management

The scoring server runs as a systemd service.

```bash
# Status
systemctl status psq-server

# Restart (after model update)
systemctl restart psq-server

# Logs (live tail)
journalctl -u psq-server -f

# Logs (last 100 lines)
journalctl -u psq-server -n 100
```

**Service file:** `/etc/systemd/system/psq-server.service` (source: `deploy/psq-server.service`)

- Runs as: `ubuntu` user
- Working directory: `/opt/psychology-agent/safety-quotient`
- Auto-restart on crash (5s delay)
- ONNX model load takes ~8s on cold start


## Caddy Configuration

```
# /etc/caddy/Caddyfile
psq.unratified.org {
    reverse_proxy localhost:3000
}
```

Caddy handles TLS certificate provisioning automatically via Let's Encrypt.
No manual cert management needed.


## File Layout (Remote)

```
/opt/psychology-agent/safety-quotient/
├── src/
│   ├── server.js              # HTTP endpoint
│   ├── student.js             # ONNX inference provider
│   ├── detector.js            # PSQ composite aggregation
│   ├── context-weights.json   # Context-aware scoring weights
│   └── ...
├── models/psq-student/
│   ├── model.onnx             # Full precision (254 MB)
│   ├── model_quantized.onnx   # INT8 quantized (64 MB) — used in production
│   ├── calibration.json       # Isotonic calibration parameters
│   ├── best.pt                # PyTorch checkpoint (for reference)
│   └── tokenizer/             # DistilBERT tokenizer files
├── node_modules/              # DO NOT REINSTALL — see Landmine below
└── package.json
```


## Deploy Procedure

All deploys run from the local development machine (where GPU and training venv live).

```bash
source venv/bin/activate
bash deploy/hetzner-deploy.sh --model models/psq-vN
```

The deploy script automates:

1. Copy `best.pt` to `models/psq-student/`
2. Run `calibrate.py` (isotonic regression on val set)
3. Run `export_onnx.py` (fp32 + INT8 quantized export)
4. Run `eval_held_out.py` (100-text held-out evaluation)
5. SHA256 checksum of local ONNX files
6. Backup current model on Hetzner (`.bak` files)
7. `rsync` model files to Hetzner (ONNX, JSON, tokenizer)
8. SHA256 verify on remote (abort on mismatch)
9. `systemctl restart psq-server`
10. Health check (`GET /health`)
11. Scoring smoke test

**Dry run:** `bash deploy/hetzner-deploy.sh --model models/psq-vN --dry-run`

**Manual deploy** (if script is unavailable):

```bash
# 1. Export ONNX locally
python scripts/export_onnx.py
python scripts/calibrate.py

# 2. rsync to Hetzner
rsync -avz models/psq-student/{model_quantized.onnx,calibration.json,tokenizer/} \
  root@178.156.229.103:/opt/psychology-agent/safety-quotient/models/psq-student/

# 3. Restart
ssh root@178.156.229.103 "systemctl restart psq-server"

# 4. Verify
curl https://psq.unratified.org/health
```


## Rollback

The previous ONNX model is overwritten by rsync. To rollback:

1. Retrain or locate the previous model version locally (`models/psq-vN/best.pt`)
2. Re-export ONNX: `python scripts/export_onnx.py --model models/psq-vN/best.pt`
3. Re-deploy: `bash deploy/hetzner-deploy.sh --model models/psq-vN`

**Improvement needed:** The deploy script does not keep remote backups. Consider adding
`ssh root@... "cp model_quantized.onnx model_quantized.onnx.bak"` before rsync.


## onnxruntime-node Landmine

**DO NOT run `npm install` on the Hetzner server.**

The `onnxruntime-node` package has a nested dependency conflict that was manually
resolved on the server. Running `npm install` will recreate the conflict and break
the service. The fix involved manually patching the `node_modules` tree.

If `npm install` is accidentally run:
1. The service will fail to start (ONNX runtime import error)
2. Recovery: `rsync` the working `node_modules/onnxruntime-node/` from a backup
   or manually resolve the nested dependency again
3. See `transport/sessions/psychology-interface/model-rsync-response-001.json` for
   the original diagnosis

**Root cause:** `onnxruntime-node` ships platform-specific binaries that conflict
with npm's dependency resolution when other packages in the tree have different
version constraints. A `postinstall` script in `package.json` may mitigate this
but has not been tested.


## Monitoring

**Health check (cron or external):**

```bash
curl -sf https://psq.unratified.org/health | grep -q '"status":"ok"' || echo "PSQ DOWN"
```

**No uptime monitoring is currently configured.** The service relies on systemd
auto-restart (5s delay) for crash recovery.

**Resource usage:**
- Memory: ~300-500 MB (Node.js + ONNX runtime with quantized model)
- CPU: near-zero at idle, brief spike per request
- Disk: 3.4 GB used / 75 GB total


## Known Issues

**User mismatch (partially resolved 2026-03-08):** The deploy script now uses
`root@178.156.229.103`. The systemd service file (`psq-server.service`) in the repo
specifies `User=ubuntu`, but the on-server copy runs as root. Files under
`/opt/psychology-agent/` are owned by UID 1000 (orphaned user).

**Remaining:** Consider creating a dedicated service user for security, or updating
the repo's service file to match the server reality (`User=root`).


## Hardening (Open Items)

- [ ] Port 3000 is exposed in UFW but should be localhost-only (Caddy handles
  external traffic on 443). Remove `ufw allow 3000/tcp`.
- [ ] No rate limiting configured (Caddy or application-level)
- [ ] No authentication on `/score` endpoint (public API)
- [ ] No request logging beyond systemd journal
- [x] Remote backups of ONNX model files before deploy (added to deploy script 2026-03-08)
- [x] Deploy script user mismatch fixed (`ubuntu` → `root`, 2026-03-08)
- [ ] Reconcile repo service file with on-server service file (User=ubuntu vs root)
- [ ] Consider adding uptime monitoring (UptimeRobot, healthchecks.io, or cron)
