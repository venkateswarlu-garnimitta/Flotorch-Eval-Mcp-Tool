#!/bin/bash
# User data script for EC2 Launch Template (ASG deployment)
# Replace ACCOUNT_ID and REGION before using in Launch Template

set -e

REGION="us-east-1"
ACCOUNT_ID="YOUR_ACCOUNT_ID"   # e.g. 123456789012
ECR_REPO="flotorch-eval-mcp"

# Install Docker
yum update -y
yum install -y docker
systemctl start docker && systemctl enable docker

# Fetch secrets from SSM (instance role must have ssm:GetParameter)
FLOTORCH_API_KEY=$(aws ssm get-parameter --name "/flotorch-eval-mcp/FLOTORCH_API_KEY" --with-decryption --query Parameter.Value --output text --region "$REGION")
FLOTORCH_BASE_URL=$(aws ssm get-parameter --name "/flotorch-eval-mcp/FLOTORCH_BASE_URL" --query Parameter.Value --output text --region "$REGION")

# Login to ECR
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# Pull and run
docker pull "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest"
docker run -d --name flotorch-eval-mcp -p 8080:8080 \
  -e FLOTORCH_API_KEY="$FLOTORCH_API_KEY" \
  -e FLOTORCH_BASE_URL="$FLOTORCH_BASE_URL" \
  --restart unless-stopped \
  "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest"
