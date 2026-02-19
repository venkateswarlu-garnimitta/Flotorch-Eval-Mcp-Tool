# Flotorch Evaluation MCP Server – Deployment Guide

This directory contains deployment configurations and scripts for running the Flotorch Evaluation MCP Server on **Hugging Face Spaces** and **AWS EC2**.

## Project Structure

```
Flotorch-Eval-Mcp-Tool/
├── src/
│   └── flotorch_eval_mcp/     # Application package
│       ├── __init__.py
│       ├── config.py
│       ├── evaluator.py
│       ├── server.py
│       ├── utils.py
│       └── __main__.py
├── deploy/
│   ├── ec2/
│   └── huggingface/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Quick Reference

| Target           | Config / Script              | Port |
|------------------|------------------------------|------|
| Hugging Face     | `huggingface/README.md`       | 7860 |
| AWS EC2          | `ec2/deploy.sh`              | 8080 |
| Local / Generic  | `docker-compose.yml` (root)  | 8080 |

---

## Prerequisites

- Docker 20.10+
- Docker Compose v2+ (for EC2)
- Flotorch API key and base URL

## Environment Variables

| Variable             | Description                          | Required |
|----------------------|--------------------------------------|----------|
| `FLOTORCH_API_KEY`   | Flotorch API key                     | Yes      |
| `FLOTORCH_BASE_URL`  | Flotorch gateway URL                 | Yes      |
| `PORT`               | Server port (default: 8080)          | No       |
| `HOST`               | Bind address (default: 0.0.0.0)      | No       |
| `LOG_LEVEL`          | Logging level (default: INFO)        | No       |

---

## Local Build & Run

```bash
# Build image
docker build -t flotorch-eval-mcp:latest .

# Run container
docker run -d -p 8080:8080 \
  -e FLOTORCH_API_KEY=your_key \
  -e FLOTORCH_BASE_URL=https://gateway.flotorch.cloud \
  flotorch-eval-mcp:latest

# Or with docker-compose
docker compose up -d
```

---

## Deployment Targets

- **[Hugging Face Spaces](huggingface/README.md)** – Docker-based Space deployment
- **[AWS EC2](ec2/README.md)** – EC2 deployment with Docker Compose
