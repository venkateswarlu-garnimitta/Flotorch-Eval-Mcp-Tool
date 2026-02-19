# AWS EC2 Deployment Guide – Simple, ASG, and Best Practices

This guide covers three deployment options for the Flotorch Evaluation MCP Server on AWS.

---

## Deployment Options Comparison

| Option | Best For | Complexity | Cost | Auto-Scale | HA |
|--------|----------|------------|------|------------|-----|
| **A. Single EC2** | Dev, low traffic | Low | $ | No | No |
| **B. EC2 + ASG + ALB** | Production, variable load | Medium | $$ | Yes | Yes |
| **C. ECS Fargate** | Production, containers | Medium | $$ | Yes | Yes |

**Recommendation:** Use **Option B (ASG)** for production with auto-scaling, or **Option A** for a quick single-instance setup.

---

## Prerequisites

- AWS account
- AWS CLI configured (`aws configure`)
- Flotorch API key and base URL
- Your Git repo URL (or Docker image in ECR)

---

# Option A: Single EC2 (Quick Start)

Best for: development, low traffic, simple setup.

## Step 1: Launch EC2

1. **EC2 Console** → Launch instance
2. **AMI:** Amazon Linux 2023 or Ubuntu 22.04
3. **Instance type:** t3.small (or t3.medium for heavier load)
4. **Key pair:** Create or select one for SSH
5. **Security group:** Create new, add rules:
   - SSH (22) from your IP
   - Custom TCP (8080) from 0.0.0.0/0 (or restrict to your IPs)
6. **Storage:** 20 GB default
7. Launch

## Step 2: Connect and Deploy

```bash
# SSH (replace with your key and instance IP)
ssh -i your-key.pem ec2-user@<instance-public-ip>

# Install Docker (Amazon Linux 2023)
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker && sudo systemctl enable docker
sudo usermod -aG docker ec2-user
# Log out and back in, or run: newgrp docker

# Clone and deploy
git clone https://github.com/YOUR_ORG/Flotorch-Eval-Mcp-Tool.git
cd Flotorch-Eval-Mcp-Tool

# Create .env
echo "FLOTORCH_API_KEY=your_key" > .env
echo "FLOTORCH_BASE_URL=https://gateway.flotorch.cloud" >> .env

# Deploy
chmod +x deploy/ec2/deploy.sh
./deploy/ec2/deploy.sh --build
```

## Step 3: Verify

```bash
curl http://localhost:8080/.well-known/flotorch-mcp
```

From outside: `http://<ec2-public-ip>:8080/.well-known/flotorch-mcp`

---

# Option B: EC2 + ASG + ALB (Production, Auto-Scaling)

Best for: production, high availability, auto-scaling on load.

## Architecture

```
Internet → ALB (port 80/443) → Target Group → ASG (EC2 instances)
                                         ↓
                              Each instance runs Docker on 8080
```

## Step 1: Push Docker Image to ECR

From the project root:

```bash
# Using the helper script (recommended)
chmod +x deploy/ec2/push-to-ecr.sh
./deploy/ec2/push-to-ecr.sh us-east-1

# Or manually:
aws ecr create-repository --repository-name flotorch-eval-mcp
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t flotorch-eval-mcp .
docker tag flotorch-eval-mcp:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/flotorch-eval-mcp:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/flotorch-eval-mcp:latest
```

## Step 2: Store Secrets in AWS Systems Manager

```bash
aws ssm put-parameter --name "/flotorch-eval-mcp/FLOTORCH_API_KEY" --value "your_api_key" --type SecureString
aws ssm put-parameter --name "/flotorch-eval-mcp/FLOTORCH_BASE_URL" --value "https://gateway.flotorch.cloud" --type String
```

## Step 3: Create Launch Template

1. **EC2** → Launch Templates → Create launch template
2. **Name:** flotorch-eval-mcp
3. **AMI:** Amazon Linux 2023
4. **Instance type:** t3.small
5. **Key pair:** Your SSH key
6. **Security group:** Create one that allows:
   - SSH (22) from your IP
   - All traffic from the ALB security group (created in Step 4)
7. **Advanced details** → User data (paste the script below)

### User Data Script for Launch Template

Replace `ACCOUNT_ID`, `REGION`, and `REPO` with your values:

```bash
#!/bin/bash
set -e

# Install Docker
yum update -y
yum install -y docker
systemctl start docker && systemctl enable docker

# Install AWS CLI (if not present) and fetch secrets
yum install -y amazon-cloudwatch-agent 2>/dev/null || true

# Get secrets from SSM
export FLOTORCH_API_KEY=$(aws ssm get-parameter --name "/flotorch-eval-mcp/FLOTORCH_API_KEY" --with-decryption --query Parameter.Value --output text --region us-east-1)
export FLOTORCH_BASE_URL=$(aws ssm get-parameter --name "/flotorch-eval-mcp/FLOTORCH_BASE_URL" --query Parameter.Value --output text --region us-east-1)

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Pull and run
docker pull ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/flotorch-eval-mcp:latest
docker run -d --name flotorch-eval-mcp -p 8080:8080 \
  -e FLOTORCH_API_KEY="$FLOTORCH_API_KEY" \
  -e FLOTORCH_BASE_URL="$FLOTORCH_BASE_URL" \
  --restart unless-stopped \
  ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/flotorch-eval-mcp:latest
```

**IAM:** The EC2 instance role must have:
- `AmazonSSMReadOnlyAccess` (to read parameters)
- `AmazonEC2ContainerRegistryReadOnly` (to pull from ECR)

## Step 4: Create Application Load Balancer

1. **EC2** → Load Balancers → Create
2. **Type:** Application Load Balancer
3. **Scheme:** Internet-facing
4. **Listeners:** HTTP:80 (or add HTTPS:443 with ACM certificate)
5. **Availability Zones:** Select at least 2
6. **Security group:** Allow 80 (and 443) from 0.0.0.0/0
7. Create target group:
   - **Target type:** Instances
   - **Protocol:** HTTP, port 8080
   - **Health check path:** `/.well-known/flotorch-mcp`
   - **Health check interval:** 30s

## Step 5: Create Auto Scaling Group

1. **EC2** → Auto Scaling Groups → Create
2. **Name:** flotorch-eval-mcp-asg
3. **Launch template:** flotorch-eval-mcp (from Step 3)
4. **VPC and subnets:** Use default or your VPC (at least 2 subnets in different AZs)
5. **Load balancing:** Attach to the ALB and target group from Step 4
6. **Group size:**
   - Min: 1
   - Desired: 1
   - Max: 3 (or higher for more scale)
7. **Scaling policies (optional):**
   - Add: CPU > 70% for 2 min → add 1 instance
   - Remove: CPU < 30% for 5 min → remove 1 instance

## Step 6: Verify

Use the ALB DNS name (e.g. `flotorch-eval-mcp-123456789.us-east-1.elb.amazonaws.com`):

```bash
curl http://<alb-dns>/.well-known/flotorch-mcp
```

---

# Option C: ECS Fargate (Serverless Containers)

Best for: fully managed containers, no EC2 to maintain.

## Quick Steps

1. **Push image to ECR** (same as Option B, Step 1)
2. **ECS** → Create cluster → Networking only
3. **ECS** → Task Definitions → Create:
   - Family: flotorch-eval-mcp
   - Launch type: Fargate
   - Task size: 0.5 vCPU, 1 GB memory
   - Container: Your ECR image, port 8080
   - Environment: FLOTORCH_API_KEY, FLOTORCH_BASE_URL (or use Secrets)
4. **ECS** → Create Service:
   - Cluster, task definition
   - Desired tasks: 1 (or 2+ for HA)
   - Load balancer: ALB, target group on port 8080
   - Health check: `/.well-known/flotorch-mcp`

---

# Summary

| If you want... | Use |
|----------------|-----|
| Quick test / dev | Option A (Single EC2) |
| Production + auto-scale | Option B (ASG + ALB) |
| No EC2 management | Option C (ECS Fargate) |

---

# Security Checklist

- [ ] Store `FLOTORCH_API_KEY` in SSM Parameter Store (SecureString) or Secrets Manager
- [ ] Restrict security groups (e.g. 8080 only from ALB, not 0.0.0.0/0)
- [ ] Use HTTPS on ALB with ACM certificate for production
- [ ] Attach Elastic IP only if you need a static IP (single EC2)
