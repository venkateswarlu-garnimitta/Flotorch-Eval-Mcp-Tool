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

**Gateway timeouts**: Evaluations are long-running (up to 50 Q/A pairs across all tools). Set `metadata.timeout` and `metadata.sse_read_timeout` to **600000** (10 minutes). The gateway uses a fixed timeout per request; completion before the limit is unaffected.

## Tools

| Tool | Description |
|------|-------------|
| `evaluate_llm` | Evaluate pre-computed Q&A pairs against selected metrics |
| `evaluate_rag` | Full RAG pipeline: retrieve from KB, generate answers, evaluate |
| `compare_llm_models` | Compare multiple models in parallel on the same dataset |
| `list_evaluation_metrics` | List available metrics with descriptions and engine support |

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `evaluation_engine` | `deepeval` | Evaluation engine (`deepeval` or `ragas`). DeepEval is recommended for speed and broadest metric coverage. |
| `metrics` | (auto) | Optional JSON array of metric names (e.g. `["faithfulness", "answer_relevance"]`). If omitted, uses defaults for the evaluation type. |
| `max_concurrent` | `10` | Maximum concurrent LLM calls per model during dataset generation. |
| `query_level_metrics` | `false` | Include per-query metric breakdown in results. |
| `gateway_metrics` | `false` | Include performance data (tokens, latency) per query and aggregated totals. Only computed when enabled. |

## Metrics

All metrics can be selected individually via the `metrics` parameter.

| Metric | Engines | Requires Context | Description |
|--------|---------|:-:|-------------|
| `faithfulness` | deepeval, ragas | Yes | Is the answer factually consistent with the context? |
| `answer_relevance` | deepeval, ragas | No | Does the answer directly address the question? |
| `context_relevancy` | deepeval | Yes | Is the retrieved context relevant to the question? |
| `context_precision` | deepeval, ragas | Yes | Is the retrieved context precise and focused? |
| `context_recall` | deepeval | Yes | Does the context cover the information needed? |
| `hallucination` | deepeval | Yes | Does the answer contain fabricated information? |

**Defaults**:
- **Normal** evaluation: `answer_relevance`
- **RAG** evaluation: all six metrics above

### Performance Metrics

Available when `gateway_metrics` is set to `true`:

- **Per-query**: `latency_ms` (wall-clock), `input_tokens`, `output_tokens`
- **Aggregated**: `avg_latency_ms`, `min_latency_ms`, `max_latency_ms`, `total_latency_ms`, `total_input_tokens`, `total_output_tokens`

## Parallel Execution

The server is optimised for throughput:

- **Dataset generation**: All questions are dispatched concurrently, bounded by `max_concurrent` (default 10).
- **RAG pipeline**: Two-phase parallelism — retrieval for all questions in parallel, then answer generation for all questions in parallel.
- **Model comparison**: All models are evaluated in full parallel with no model-level concurrency limit. Each model independently parallelises its questions up to `max_concurrent`.

## MCP Configuration

**Gateway template** (transport and metadata):

```ts
{
  templateId: "flotorch-eval",
  name: "Flotorch Evaluation MCP",
  description: "LLM evaluation: standard and RAG evaluation, multi-model comparison on the Flotorch platform. Credentials are passed per request via headers: X-Flotorch-Api-Key, X-Flotorch-Base-Url.",
  category: "Evaluation",
  icon: "i-lucide-clipboard-check",
  url: "", // Set in database (e.g. deployed server URL)
  transport: "HTTP_STREAMABLE" as const,
  metadata: {
    transport: "HTTP_STREAMABLE" as const,
    timeout: 600000,
    sse_read_timeout: 600000,
    terminate_on_close: true,
  },
  requiredFields: [],
  baseHeaders: {},
  isEnabled: false, // Enable after URL is configured in database
}
```

**Local MCP client** (JSON):

```json
{
  "transport": "HTTP_STREAMABLE",
  "url": "http://localhost:8080",
  "headers": {
    "X-Flotorch-Api-Key": "your_api_key",
    "X-Flotorch-Base-Url": "https://gateway.flotorch.cloud"
  },
  "timeout": 600000,
  "sse_read_timeout": 600000
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
├── config.py      # Credentials, metrics config, metric resolution
├── evaluator.py   # Parallel dataset generation, evaluation orchestration
├── server.py      # FastMCP server and tool definitions
└── utils.py       # Validation, formatting, gateway metadata enrichment
```

## Requirements

- Python 3.11+
- flotorch, flotorch-eval, mcp
