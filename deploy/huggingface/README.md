# Deploy to Hugging Face Spaces (Docker)

Deploy the Flotorch Evaluation MCP Server as a **Docker Space** on Hugging Face.

## Prerequisites

- Hugging Face account
- Flotorch API key and base URL

## Step 1: Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Docker** as the SDK
3. Select hardware (CPU is sufficient; GPU optional)
4. Create the Space

## Step 2: Configure Space README

Add this YAML block at the top of your Space's `README.md`:

```yaml
---
title: Flotorch Evaluation MCP Server
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
```

## Step 3: Add Environment Variables

In your Space → **Settings** → **Variables and secrets**:

| Name               | Value                            | Secret |
|--------------------|----------------------------------|--------|
| `PORT`             | `7860`                           | No     |
| `FLOTORCH_API_KEY` | Your Flotorch API key            | Yes    |
| `FLOTORCH_BASE_URL`| `https://gateway.flotorch.cloud` | No     |

> **Important:** Hugging Face Spaces expects the app to listen on port **7860** by default. Set `PORT=7860` as a Variable so the server binds correctly.

## Step 4: Copy Project Files

Copy these files and folders into your Space repository root:

```
README.md          # With YAML header above
Dockerfile         # Use deploy/huggingface/Dockerfile (port 7860 preconfigured)
requirements.txt
pyproject.toml
src/
  flotorch_eval_mcp/
    __init__.py
    config.py
    evaluator.py
    server.py
    utils.py
    __main__.py
.dockerignore
```

Use `deploy/huggingface/Dockerfile` as your Space's `Dockerfile`—it is preconfigured for port 7860.

## Step 5: Deploy

Push your changes to the Space. Hugging Face will build and run the Docker container. The Dockerfile installs the `flotorch_eval_mcp` package from `src/` and runs it via `python -m flotorch_eval_mcp`.

## Verification

- **Discovery:** `https://your-username-your-space.hf.space/.well-known/flotorch-mcp`
- **MCP endpoint:** `https://your-username-your-space.hf.space/`

## MCP Client Configuration

```json
{
  "transport": "HTTP_STREAMABLE",
  "url": "https://your-username-your-space.hf.space/",
  "headers": {
    "X-Flotorch-Api-Key": "your_api_key",
    "X-Flotorch-Base-Url": "https://gateway.flotorch.cloud"
  }
}
```

## Troubleshooting

| Issue                    | Solution                                                |
|--------------------------|---------------------------------------------------------|
| Build fails              | Check `requirements.txt` and Dockerfile paths           |
| Port mismatch            | Set `PORT=7860` in Space Variables                     |
| API key errors           | Add `FLOTORCH_API_KEY` as a **Secret** in Settings     |
| Permission errors        | Ensure Dockerfile uses `USER appuser` (UID 1000)        |
