---
title: Flotorch Evaluation MCP Server
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
suggested_hardware: cpu-upgrade
---

# Flotorch Evaluation MCP Server

MCP server for LLM evaluation on the Flotorch platform. Supports standard evaluation, RAG evaluation with knowledge base retrieval, and multi-model comparison.

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
python -m flotorch_eval_mcp
```

Server runs at `http://0.0.0.0:8080`.

## Configuration

**Environment**

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port |
| `HOST` | 0.0.0.0 | Bind address |
| `FLOTORCH_API_KEY` | — | Flotorch API key (required) |
| `FLOTORCH_BASE_URL` | — | Flotorch gateway URL (required) |

**MCP client headers**: `X-Flotorch-Api-Key`, `X-Flotorch-Base-Url`

**MCP client timeouts** (evaluations take 30–90+ seconds): set `timeout` and `sse_read_timeout` to `120000` (ms) in your agent config.

## Tools

| Tool | Description |
|------|-------------|
| `evaluate_llm` | Evaluate pre-computed Q&A pairs |
| `evaluate_rag` | RAG pipeline: retrieve, generate, evaluate |
| `compare_llm_models` | Compare multiple models on the same dataset |
| `list_evaluation_metrics` | List available metrics |

## Metrics

**Normal** (no context): Answer Relevance

**RAG**: Faithfulness, Context Relevancy, Context Precision, Context Recall, Answer Relevance, Hallucination

## MCP Configuration

```json
{
  "transport": "HTTP_STREAMABLE",
  "url": "http://localhost:8080",
  "headers": {
    "X-Flotorch-Api-Key": "your_api_key",
    "X-Flotorch-Base-Url": "https://gateway.flotorch.cloud"
  },
  "timeout": 120000,
  "sse_read_timeout": 120000
}
```

## Deployment

**Docker**

```bash
docker build -t flotorch-eval-mcp .
docker run -d -p 8080:8080 \
  -e FLOTORCH_API_KEY=your_key \
  -e FLOTORCH_BASE_URL=https://gateway.flotorch.cloud \
  flotorch-eval-mcp
```

**Hugging Face Spaces**: Use `deploy/huggingface/Dockerfile`; set `PORT=7860` and credentials in Space Variables.

**EC2**: See `deploy/ec2/README.md`.

## Project Structure

```
src/flotorch_eval_mcp/
├── config.py      # Credentials, metrics config
├── evaluator.py   # Dataset generation, evaluation
├── server.py      # FastMCP server and tools
└── utils.py       # Validation, formatting
```

## Requirements

- Python 3.11+
- flotorch, flotorch-eval, mcp
