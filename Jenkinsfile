def REGISTRY_IP = '192.168.1.165:30002'
def PROJECT_NAME = 'kt-moa'
def REGISTRYCREDENTIAL = 'harbor.credentials'
def IMAGE_NAME = 'hardcap-api-server'
def appImage
def saved_path

pipeline {
  agent {
    kubernetes {
      label 'kt-moa'
      yaml '''
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: "git"
      image: "alpine/git"
      command:
        - "cat"
      tty: true

    - name: "docker"
      image: "docker"
      command:
        - "cat"
      tty: true
      volumeMounts:
        - mountPath: "/root/kt-moa/"
          name: "kt-moa"
        - mountPath: "/var/run/docker.sock"
          name: "docker-sock"

    - name: "kt-moa-docker"
      image: "192.168.1.165:30002/kt-moa/hardcap-api-server:latest"
      command:
        - "cat"
      tty: true
      volumeMounts:
        - mountPath: "/root/kt-moa/"
          name: "kt-moa"

  volumes:
    - name: "docker-sock"
      hostPath:
        path: "/var/run/docker.sock"
    - name: "kt-moa"
      emptyDir: {}
      '''
    }
  }

  // Start script
  stages {
    // Checkout from git
    stage('Checkout') {
      steps {
        container('git') {
          script {
            checkout scm
            sh 'ls'
          }
        }
      }
    }

    // BentoML packing
    stage('bento-packing') {
      steps {
        container('kt-moa-docker') {
          script {
            sh 'ls'
            sh 'cd /root/kt-moa'
            sh 'ls'
            sh 'export MLFLOW_TRACKING_URI=http://192.168.1.145:31442;'
            sh 'export MLFLOW_S3_ENDPOINT_URL=http://192.168.1.145:30575;'
            sh 'export AWS_ACCESS_KEY_ID=minio; export AWS_SECRET_ACCESS_KEY=minio123;'
            sh 'python3 bento-packing.py'

            saved_path = sh(
              script: 'bentoml get api_service:latest --print-location --quiet',
              returnStdout: true
            ).trim()

            println(saved_path)
          }
        }
      }
    }

    // // bento svc dockerbuild
    // stage('Bento docker build') {
    //   steps {
    //     container('kt-moa-docker') {
    //       script {
    //         sh 'ls /root/kt-moa/'
    //         sh 'cd /root/bentoml; ls'
    //         sh 'systemctl start docker'
    //         sh 'systemctl enable docker'
    //         appImage = docker.build(
    //         "$REGISTRY_IP/$PROJECT_NAME/$IMAGE_NAME:$BUILD_NUMBER",
    //         "--network=host $saved_path")
    //       }
    //     }
    //   }
    // }

    // // Bento svc image push
    // stage('Image Push') {
    //   steps {
    //     container('kt-moa-docker') {
    //       script {
    //         docker.withRegistry("http://$REGISTRY_IP", REGISTRYCREDENTIAL) {
    //           appImage.push('latest')
    //           appImage.push(BUILD_NUMBER)
    //         }
    //       }
    //     }
    //   }
    // }
  }
}
