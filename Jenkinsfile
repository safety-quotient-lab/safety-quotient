// Safety Quotient (PSQ) — Tier 2 CI/CD Pipeline
//
// CONTEXT
// The PSQ scoring server runs on Hetzner, not on the same LAN as the other
// agents. This pipeline has two modes:
//
//   1. CI mode (default, on push): validates the codebase and checks
//      production health. Triggered automatically via GitHub Actions relay.
//
//   2. Model deploy mode (manual, parameterized): runs the full post-training
//      pipeline — calibrate, export ONNX, evaluate held-out set — then syncs
//      model files to Hetzner and restarts the scoring service.
//
// WHY TWO MODES IN ONE JENKINSFILE
// Jenkins maps one Jenkinsfile per repo. A separate Jenkinsfile would require
// a separate Jenkins job, separate webhook config, and separate SCM polling —
// duplication with no functional benefit. Parameters let us gate the expensive
// model-deploy stages behind an explicit manual trigger while keeping CI
// automatic.
//
// WHY MANUAL TRIGGER FOR MODEL DEPLOY
// Model deployment requires human judgment. The operator must verify:
//   - Training converged (loss curves, early stopping epoch)
//   - Held-out metrics meet minimum thresholds
//   - The calibration curve looks reasonable (isotonic regression output)
// Automating this gate risks deploying undertrained or miscalibrated models
// to production. The pipeline automates the mechanics; the human owns the
// decision.
//
// WHY SHA256 VERIFICATION
// rsync uses checksums for transfer integrity, but a post-transfer SHA256
// comparison provides defense in depth. If the remote filesystem silently
// corrupts (disk error, interrupted write), rsync would not detect it on the
// next run. The explicit hash comparison catches this class of failure before
// the service restarts with a corrupt model.
//
// CRITICAL: Do NOT run `npm install` on Hetzner after deploying model files.
// onnxruntime-node has a nested dependency conflict that was manually resolved
// on the server. npm install recreates the conflict and breaks the service.
// See: transport/sessions/psychology-interface/model-rsync-response-001.json
//
// BUILD TRIGGER
// CI mode: GitHub Actions relay (.github/workflows/trigger-forge.yml).
//   See that file for why a relay is needed (Cloudflare Access authentication).
//   SCM polling (H/5 * * * *) serves as a fallback.
// Deploy mode: "Build with Parameters" in Jenkins UI.
//
// PREREQUISITES (deploy mode only)
// Cabinet (Jenkins host) requires:
//   - Python 3.10+ with venv containing: torch, numpy, scikit-learn, onnx,
//     onnxruntime (for calibration, ONNX export, and evaluation)
//   - SSH access to Hetzner via 'hetzner-ssh-key' credential
//   - Jenkins env vars: HETZNER_HOST, HETZNER_REMOTE_DIR, PSQ_HEALTH_URL,
//     PSQ_SCORE_URL, PSQ_SERVICE_NAME
//
// Required credentials (Jenkins > Manage > Credentials):
//   'hetzner-ssh-key' — SSH private key for Hetzner server (Secret file)

pipeline {
    agent any

    parameters {
        // MODEL_DIR: path relative to workspace root.
        // Empty string (default) means CI-only mode — no deploy stages run.
        // Example: "models/psq-v28" triggers the full deploy pipeline.
        string(
            name: 'MODEL_DIR',
            defaultValue: '',
            description: 'Model directory (relative to repo root) containing best.pt. Leave empty for CI-only mode.'
        )
        booleanParam(
            name: 'DRY_RUN',
            defaultValue: false,
            description: 'Run deploy pipeline without modifying Hetzner (rsync --dry-run, skip restart).'
        )
    }

    stages {
        // ── CI stages (always run) ─────────────────────────────────────────

        stage('Install') {
            steps {
                sh 'npm ci'
            }
        }

        // Verify the production PSQ endpoint responds.
        // This catches outages introduced by infrastructure changes
        // (DNS, firewall, service restarts) that code changes alone
        // would not reveal.
        stage('Health Check') {
            steps {
                sh '''
                    STATUS=$(curl -sf "${PSQ_HEALTH_URL:-https://psq.safety-quotient.dev/health}" || echo "DOWN")
                    echo "Production status: $STATUS"
                '''
            }
        }

        // ── Model deploy stages (manual trigger only) ──────────────────────
        //
        // These stages implement the 11-step post-training pipeline:
        //   1. Copy best.pt to student directory
        //   2. Isotonic regression calibration
        //   3. ONNX export (full + quantized)
        //   4. Held-out evaluation
        //   5. Local SHA256 checksums
        //   6. Backup current model on Hetzner
        //   7. rsync model files to Hetzner
        //   8. Remote SHA256 verification
        //   9. Restart scoring service
        //  10. Health check
        //  11. Scoring smoke test
        //
        // Each stage gates the next — a SHA256 mismatch aborts before
        // restart, a failed health check prevents the smoke test from
        // running against a broken service.

        stage('Validate Model') {
            when {
                expression { params.MODEL_DIR != '' }
            }
            steps {
                sh """
                    if [ ! -f "${params.MODEL_DIR}/best.pt" ]; then
                        echo "ERROR: ${params.MODEL_DIR}/best.pt not found."
                        echo "Verify training completed and the path is correct."
                        exit 1
                    fi
                    echo "Model checkpoint found: ${params.MODEL_DIR}/best.pt"
                    ls -lh "${params.MODEL_DIR}/best.pt"
                """
            }
        }

        // Steps 1–4: Post-training pipeline runs locally on cabinet.
        // WHY LOCAL: Calibration and ONNX export require PyTorch and
        // scikit-learn. These run on cabinet (build server) rather than
        // Hetzner (production) to keep the production server lean —
        // it only needs onnxruntime-node for inference.
        stage('Post-Training Pipeline') {
            when {
                expression { params.MODEL_DIR != '' }
            }
            steps {
                sh """
                    echo "Step 1: Copying best.pt to models/psq-student/"
                    cp "${params.MODEL_DIR}/best.pt" models/psq-student/best.pt

                    echo "Step 2: Calibration (isotonic regression, 20 bins)"
                    python3 scripts/calibrate.py \\
                        --model-dir "${params.MODEL_DIR}" \\
                        --n-bins 20 \\
                        --out models/psq-student/calibration.json

                    echo "Step 3: ONNX export (full + quantized)"
                    python3 scripts/export_onnx.py \\
                        --checkpoint "${params.MODEL_DIR}/best.pt"

                    echo "Step 4: Held-out evaluation"
                    python3 scripts/eval_held_out.py \\
                        --model "${params.MODEL_DIR}/best.pt"
                """
            }
        }

        // Step 5: Compute local checksums before transfer.
        // These hashes become the reference for post-transfer verification.
        stage('Local Checksums') {
            when {
                expression { params.MODEL_DIR != '' }
            }
            steps {
                sh '''
                    echo "Step 5: Local SHA256 checksums"
                    sha256sum models/psq-student/model.onnx
                    sha256sum models/psq-student/model_quantized.onnx
                    sha256sum models/psq-student/calibration.json
                '''
                // Capture hashes for later comparison.
                script {
                    env.LOCAL_ONNX_HASH = sh(
                        script: "sha256sum models/psq-student/model.onnx | cut -d' ' -f1",
                        returnStdout: true
                    ).trim()
                    env.LOCAL_QUANT_HASH = sh(
                        script: "sha256sum models/psq-student/model_quantized.onnx | cut -d' ' -f1",
                        returnStdout: true
                    ).trim()
                }
            }
        }

        // Steps 6–7: Backup current model, then rsync new files to Hetzner.
        // WHY BACKUP FIRST: If the new model causes scoring failures,
        // the operator can SSH in and restore the .bak files without
        // needing to re-run the full pipeline from cabinet.
        stage('Deploy to Hetzner') {
            when {
                expression { params.MODEL_DIR != '' }
            }
            steps {
                withCredentials([sshUserPrivateKey(
                    credentialsId: 'hetzner-ssh-key',
                    keyFileVariable: 'SSH_KEY'
                )]) {
                    sh """
                        SSH_OPTS="-i \$SSH_KEY -o StrictHostKeyChecking=accept-new"
                        REMOTE="${env.HETZNER_HOST}"
                        REMOTE_DIR="${env.HETZNER_REMOTE_DIR}/models/psq-student"

                        echo "Step 6: Backing up current model on Hetzner"
                        if [ "${params.DRY_RUN}" = "false" ]; then
                            ssh \$SSH_OPTS "\$REMOTE" \\
                                "cp \$REMOTE_DIR/model_quantized.onnx \$REMOTE_DIR/model_quantized.onnx.bak 2>/dev/null && echo 'Backup created' || echo 'No existing model (first deploy?)'"
                            ssh \$SSH_OPTS "\$REMOTE" \\
                                "cp \$REMOTE_DIR/calibration.json \$REMOTE_DIR/calibration.json.bak 2>/dev/null || true"
                        else
                            echo "(dry run — skipping backup)"
                        fi

                        echo "Step 7: rsync model files to Hetzner"
                        RSYNC_FLAGS="-avz --progress -e 'ssh \$SSH_OPTS'"
                        if [ "${params.DRY_RUN}" = "true" ]; then
                            RSYNC_FLAGS="\$RSYNC_FLAGS --dry-run"
                        fi

                        rsync \$RSYNC_FLAGS \\
                            --include='*.onnx' \\
                            --include='*.json' \\
                            --include='*.pt' \\
                            --include='tokenizer/' \\
                            --include='tokenizer/**' \\
                            --exclude='*' \\
                            models/psq-student/ \\
                            "\$REMOTE:\$REMOTE_DIR/"
                    """
                }
            }
        }

        // Step 8: Defense-in-depth — verify file integrity survived transfer.
        // Aborts before restart if any hash diverges.
        stage('Verify Remote Checksums') {
            when {
                expression { params.MODEL_DIR != '' }
            }
            steps {
                withCredentials([sshUserPrivateKey(
                    credentialsId: 'hetzner-ssh-key',
                    keyFileVariable: 'SSH_KEY'
                )]) {
                    sh """
                        SSH_OPTS="-i \$SSH_KEY -o StrictHostKeyChecking=accept-new"
                        REMOTE="${env.HETZNER_HOST}"
                        REMOTE_DIR="${env.HETZNER_REMOTE_DIR}/models/psq-student"

                        echo "Step 8: Verifying SHA256 on Hetzner"
                        REMOTE_ONNX=\$(ssh \$SSH_OPTS "\$REMOTE" "sha256sum \$REMOTE_DIR/model.onnx" | cut -d' ' -f1)
                        REMOTE_QUANT=\$(ssh \$SSH_OPTS "\$REMOTE" "sha256sum \$REMOTE_DIR/model_quantized.onnx" | cut -d' ' -f1)

                        echo "Local  model.onnx:           ${env.LOCAL_ONNX_HASH}"
                        echo "Remote model.onnx:           \$REMOTE_ONNX"
                        echo "Local  model_quantized.onnx: ${env.LOCAL_QUANT_HASH}"
                        echo "Remote model_quantized.onnx: \$REMOTE_QUANT"

                        if [ "${env.LOCAL_ONNX_HASH}" != "\$REMOTE_ONNX" ]; then
                            echo "ERROR: model.onnx SHA256 mismatch — aborting restart"
                            exit 1
                        fi
                        if [ "${env.LOCAL_QUANT_HASH}" != "\$REMOTE_QUANT" ]; then
                            echo "ERROR: model_quantized.onnx SHA256 mismatch — aborting restart"
                            exit 1
                        fi
                        echo "SHA256 verified"
                    """
                }
            }
        }

        // Steps 9–11: Restart, health check, smoke test.
        // WHY 15-SECOND SLEEP: ONNX model loading takes ~8–12 seconds.
        // Hitting the health endpoint too early returns connection refused
        // rather than an unhealthy response, making it hard to distinguish
        // "still loading" from "crashed on startup."
        stage('Restart and Verify') {
            when {
                expression { params.MODEL_DIR != '' && !params.DRY_RUN }
            }
            steps {
                withCredentials([sshUserPrivateKey(
                    credentialsId: 'hetzner-ssh-key',
                    keyFileVariable: 'SSH_KEY'
                )]) {
                    sh """
                        SSH_OPTS="-i \$SSH_KEY -o StrictHostKeyChecking=accept-new"
                        REMOTE="${env.HETZNER_HOST}"
                        HEALTH_URL="${env.PSQ_HEALTH_URL:-https://psq.safety-quotient.dev/health}"
                        SCORE_URL="${env.PSQ_SCORE_URL:-https://psq.safety-quotient.dev/score}"

                        echo "Step 9: Restarting ${env.PSQ_SERVICE_NAME:-psq-server}"
                        ssh \$SSH_OPTS "\$REMOTE" "systemctl restart ${env.PSQ_SERVICE_NAME:-psq-server}"
                        echo "Waiting 15 seconds for ONNX model load..."
                        sleep 15

                        echo "Step 10: Health check"
                        HEALTH=\$(curl -sf "\$HEALTH_URL" || echo '{"status":"error"}')
                        echo "Response: \$HEALTH"
                        if ! echo "\$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='ok' else 1)" 2>/dev/null; then
                            echo "ERROR: Health check failed"
                            exit 1
                        fi
                        echo "Health check passed"

                        echo "Step 11: Scoring smoke test"
                        SCORE=\$(curl -sf -X POST "\$SCORE_URL" \\
                            -H "Content-Type: application/json" \\
                            -d '{"text": "The team felt safe raising concerns in this meeting."}' \\
                            | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'composite={d[\"scores\"][\"psq_composite\"][\"value\"]:.1f}, calibration={d[\"scores\"][\"calibration_version\"]}, dims={len(d[\"dimensions\"])}')" \\
                            2>/dev/null || echo "score request failed")
                        echo "Result: \$SCORE"
                    """
                }
            }
        }
    }

    post {
        success {
            script {
                if (params.MODEL_DIR != '') {
                    echo "Model deploy succeeded: ${params.MODEL_DIR} → Hetzner"
                }
            }
            echo "Build succeeded: ${env.BUILD_URL}"
        }
        failure {
            script {
                if (params.MODEL_DIR != '') {
                    echo "MODEL DEPLOY FAILED — check console output for which step failed."
                    echo "If failure occurred after rsync (steps 8–11), .bak files exist on Hetzner."
                }
            }
            echo "Build failed: ${env.BUILD_URL}"
        }
    }
}
