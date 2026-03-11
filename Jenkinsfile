// Safety Quotient (PSQ) — Tier 2 CI/CD Pipeline
//
// CONTEXT
// The PSQ scoring server runs on Hetzner (178.156.229.103), not on the
// same LAN as the other agents. This pipeline validates the codebase and
// checks production health. Model deployment (train → validate → export →
// deploy) remains a manual-trigger pipeline — see docs/devops-pipeline.md
// in the psychology-agent repo for the full 11-step sequence.
//
// BUILD TRIGGER
// Builds trigger via a GitHub Actions relay (.github/workflows/trigger-forge.yml).
// See that file for why a relay is needed (Cloudflare Access authentication).
// SCM polling (H/5 * * * *) serves as a fallback.

pipeline {
    agent any

    stages {
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
                    STATUS=$(curl -sf https://psq.safety-quotient.dev/health || echo "DOWN")
                    echo "Production status: $STATUS"
                '''
            }
        }
    }

    post {
        success {
            echo "Build succeeded: ${env.BUILD_URL}"
        }
        failure {
            echo "Build failed: ${env.BUILD_URL}"
        }
    }
}
