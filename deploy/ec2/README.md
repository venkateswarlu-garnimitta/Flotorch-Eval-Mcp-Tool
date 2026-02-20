# Deploy to AWS EC2

Deploy the Flotorch Evaluation MCP Server on EC2 using Docker.

## Prerequisites

- EC2 instance (Amazon Linux 2 or Ubuntu 22.04+)
- Port 8080 open in security group
- Docker installed

## Deploy

```bash
git clone <repo> flotorch-eval-mcp
cd flotorch-eval-mcp

# Set credentials
echo "FLOTORCH_API_KEY=your_key" > .env
echo "FLOTORCH_BASE_URL=https://gateway.flotorch.cloud" >> .env

# Run
./deploy/ec2/deploy.sh
```

Or manually: `docker compose up -d --build`

## Verify

```bash
curl http://localhost:8080/.well-known/flotorch-mcp
```

Public URL: `http://<ec2-public-ip>:8080/.well-known/flotorch-mcp`
