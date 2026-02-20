# Deploy to Hugging Face Spaces

Deploy the Flotorch Evaluation MCP Server as a Docker Space.

## Steps

1. **Create Space** – [huggingface.co/new-space](https://huggingface.co/new-space), choose Docker SDK.

2. **Add YAML to README**:
   ```yaml
   ---
   title: Flotorch Evaluation MCP Server
   sdk: docker
   app_port: 7860
   ---
   ```

3. **Set Variables** (Settings → Variables and secrets):
   | Name | Value | Secret |
   |------|-------|--------|
   | `PORT` | 7860 | No |
   | `FLOTORCH_API_KEY` | your_key | Yes |
   | `FLOTORCH_BASE_URL` | https://gateway.flotorch.cloud | No |

4. **Copy files** – Use `deploy/huggingface/Dockerfile` as the Space Dockerfile. Include `src/`, `requirements.txt`, `pyproject.toml`, `.dockerignore`.

5. **Push** – Hugging Face builds and runs the container.

## MCP Client Config

```json
{
  "transport": "HTTP_STREAMABLE",
  "url": "https://your-space.hf.space/",
  "headers": {
    "X-Flotorch-Api-Key": "your_api_key",
    "X-Flotorch-Base-Url": "https://gateway.flotorch.cloud"
  },
  "timeout": 120000,
  "sse_read_timeout": 120000
}
```

## Endpoints

- Discovery: `/.well-known/flotorch-mcp`
- MCP: `/`
