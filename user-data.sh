#!/bin/bash
set -euo pipefail
ECR_REGISTRY="491085385944.dkr.ecr.ap-south-1.amazonaws.com/hackeval-agent"   # replace
IMAGE_NAME="hackeval-agent"
CONTAINER_NAME="hackeval-agent"
APP_PORT=8000
AWS_REGION="ap-south-1"

yum update -y
yum install -y docker jq
systemctl enable docker
systemctl start docker
usermod -a -G docker ec2-user

# Install AWS CLI v2 if not present
if ! command -v aws >/dev/null 2>&1; then
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  yum install -y unzip
  unzip -q /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install
fi

# Login to ECR (instance needs IAM role permitting ECR pull)
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# initial tag (can be overridden by SSM update script)
IMAGE_TAG=${IMAGE_TAG:-latest}
IMAGE_URI="${ECR_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

cat >/usr/local/bin/hackeval_update_image.sh <<'EOF'
#!/bin/bash
set -euo pipefail
ECR_REGISTRY="123456789012.dkr.ecr.ap-south-1.amazonaws.com" # replace
IMAGE_NAME="hackeval-agent"
IMAGE_TAG="$1"
IMAGE_URI="${ECR_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin ${ECR_REGISTRY}
docker pull "${IMAGE_URI}"
docker rm -f hackeval-agent || true
docker run -d --name hackeval-agent -p 8000:8000 --restart always "${IMAGE_URI}"
EOF

chmod +x /usr/local/bin/hackeval_update_image.sh

# Start the initial container (if image exists in ECR)
if aws ecr describe-images --repository-name ${IMAGE_NAME} --image-ids imageTag=latest --region ${AWS_REGION} >/dev/null 2>&1; then
  /usr/local/bin/hackeval_update_image.sh latest || true
fi
