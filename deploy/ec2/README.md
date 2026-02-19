# Deploy to AWS EC2

Deploy the Flotorch Evaluation MCP Server on an AWS EC2 instance using Docker.

## Prerequisites

- AWS account
- EC2 instance (Amazon Linux 2, Ubuntu 22.04, or similar)
- SSH access to the instance
- Security group allowing inbound traffic on port 8080 (or your chosen port)

## Step 1: Launch EC2 Instance

1. Launch an EC2 instance (e.g., `t3.small` or larger)
2. Use Amazon Linux 2 or Ubuntu 22.04 AMI
3. Open port **8080** in the security group (or 80/443 if using a reverse proxy)
4. Attach an Elastic IP if you need a static address

## Step 2: Install Docker on EC2

### Amazon Linux 2

```bash
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user
```

### Ubuntu 22.04

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker ubuntu
```

Log out and back in for group changes to take effect.

## Step 3: Deploy Using the Script

```bash
# Clone or copy the project to the EC2 instance
git clone <your-repo-url> flotorch-eval-mcp
cd flotorch-eval-mcp

# Create .env file with credentials
cat > .env << 'EOF'
FLOTORCH_API_KEY=your_api_key_here
FLOTORCH_BASE_URL=https://gateway.flotorch.cloud
EOF

# Run the deployment script
chmod +x deploy/ec2/deploy.sh
./deploy/ec2/deploy.sh
```

## Step 4: Manual Deployment (Alternative)

```bash
# Build and run with docker-compose
docker compose up -d --build

# Or build and run standalone
docker build -t flotorch-eval-mcp:latest .
docker run -d --name flotorch-eval-mcp -p 8080:8080 \
  -e FLOTORCH_API_KEY=your_key \
  -e FLOTORCH_BASE_URL=https://gateway.flotorch.cloud \
  --restart unless-stopped \
  flotorch-eval-mcp:latest
```

## Step 5: Verify

```bash
# Check container status
docker ps

# Test discovery endpoint
curl http://localhost:8080/.well-known/flotorch-mcp
```

From outside: `http://<ec2-public-ip>:8080/.well-known/flotorch-mcp`

## Optional: Reverse Proxy (Nginx)

For HTTPS and a custom domain:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Optional: Systemd Service

Create `/etc/systemd/system/flotorch-eval-mcp.service`:

```ini
[Unit]
Description=Flotorch Evaluation MCP Server
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ec2-user/flotorch-eval-mcp
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

| Issue              | Solution                                      |
|--------------------|-----------------------------------------------|
| Port 8080 blocked  | Update security group inbound rules            |
| Container exits    | Check `docker logs flotorch-eval-mcp-server`  |
| API key invalid    | Verify `.env` or environment variables        |
| Out of memory      | Use a larger instance type (e.g., t3.medium)  |
