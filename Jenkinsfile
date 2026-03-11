pipeline {
    agent any

    stages {
        stage('Install') {
            steps {
                sh 'npm ci'
            }
        }

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
        failure {
            echo "Build failed: ${env.BUILD_URL}"
        }
    }
}
