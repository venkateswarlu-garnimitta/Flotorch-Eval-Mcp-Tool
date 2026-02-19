#!/bin/bash
# Push Docker image to AWS ECR (required for ASG/ECS deployment)
# Usage: ./push-to-ecr.sh [region] [account-id]
# Run from project root: ./deploy/ec2/push-to-ecr.sh us-east-1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

REGION="${1:-us-east-1}"
ACCOUNT_ID="${2:-$(aws sts get-caller-identity --query Account --output text)}"
REPO_NAME="flotorch-eval-mcp"
IMAGE_TAG="latest"

echo "Region: $REGION, Account: $ACCOUNT_ID"
echo "Creating ECR repo if not exists..."
aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION"

echo "Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

echo "Building image..."
docker build -t "$REPO_NAME:$IMAGE_TAG" .

echo "Tagging and pushing..."
docker tag "$REPO_NAME:$IMAGE_TAG" "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"
docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"

echo "Done! Image: $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"
