# Deployment

Deploy the Flotorch Evaluation MCP Server.

| Target | Config | Port |
|--------|--------|------|
| Hugging Face Spaces | [huggingface/README.md](huggingface/README.md) | 7860 |
| AWS EC2 | [ec2/README.md](ec2/README.md) | 8080 |
| Local Docker | `docker compose up -d` (root) | 8080 |

## Environment

| Variable | Required | Description |
|----------|----------|-------------|
| `FLOTORCH_API_KEY` | Yes | Flotorch API key |
| `FLOTORCH_BASE_URL` | Yes | Flotorch gateway URL |
| `PORT` | No | Server port (default: 8080) |
| `HOST` | No | Bind address (default: 0.0.0.0) |
