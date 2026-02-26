# Flotorch Evaluation MCP Server

MCP server for LLM evaluation on the Flotorch platform. It supports standard evaluation, RAG evaluation with knowledge-base retrieval, and multi-model comparison. Designed for production deployment (e.g. on AWS) behind a gateway or load balancer.

---

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
python -m flotorch_eval_mcp
```

The server listens on `http://0.0.0.0:8081` by default.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8081 | Server port |
| `HOST` | 0.0.0.0 | Bind address |
| `FLOTORCH_API_KEY` | — | Flotorch API key (required at runtime via header or env) |
| `FLOTORCH_BASE_URL` | — | Flotorch gateway URL (required at runtime via header or env) |

Credentials can be supplied via HTTP headers: `X-Flotorch-Api-Key`, `X-Flotorch-Base-Url`.

**Timeouts (gateway / MCP client):** Evaluations can run for several minutes. Set `timeout` and `sse_read_timeout` to **18000000** (ms) in the gateway and MCP client configuration so long-running requests are not closed prematurely.

---

## Tools

| Tool | Description |
|------|-------------|
| `evaluate_llm` | Evaluate pre-computed Q&A pairs against selected metrics |
| `evaluate_rag` | Full RAG pipeline: retrieve from KB, generate answers, evaluate |
| `compare_llm_models` | Compare multiple models in parallel on the same dataset |
| `list_evaluation_metrics` | List available metrics and engine support |

**Common parameters:** `evaluation_engine` (default: deepeval), `metrics` (optional JSON array), `max_concurrent` (default: 10), `query_level_metrics`, `gateway_metrics`.

---

## Metrics

Available metrics (selectable via the `metrics` parameter): `faithfulness`, `answer_relevance`, `context_relevancy`, `context_precision`, `context_recall`, `hallucination`. Defaults depend on evaluation type (normal vs RAG). When `gateway_metrics` is true, results include token counts and latency (per-query and aggregated).

---

## Agent Goal and System Prompt

The agent should run exactly one evaluation tool per user request, with correctly prepared inputs, and return the tool result.

**Tool selection:** Use `compare_llm_models` to compare models on a shared Q&A set; `evaluate_llm` for pre-computed answers; `evaluate_rag` for retrieve-then-generate-then-evaluate; `list_evaluation_metrics` to list metrics.

**Data preparation:** `ground_truth` must be a JSON array of `{"question": "...", "answer": "..."}`. Use only ASCII in JSON (no Unicode symbols or control characters). Required parameters include `evaluation_model`, `embedding_model` (e.g. `flotorch/embed`), `system_prompt`, and `user_prompt_template` (with `{question}` placeholder; for RAG, also `{context}`).

**Example system prompt for the agent:**

```text
You are a Flotorch Evaluation Agent. Run LLM evaluations by calling the correct MCP tool once with valid parameters, then return the tool result.

Rules: (1) Choose the right tool: compare_llm_models, evaluate_llm, evaluate_rag, or list_evaluation_metrics. (2) For compare_llm_models, provide ground_truth (JSON array of {"question","answer"}), inference_models (JSON array of model IDs), evaluation_model, embedding_model, system_prompt, user_prompt_template. (3) Use only ASCII in JSON. (4) Call exactly one tool once; after receiving the result, return it to the user. (5) If the user does not ask for an evaluation, respond briefly and ask what they need. (6) Default evaluation_model: flotorch/gpt-4-1-mini; default embedding_model: flotorch/embed.
```

---

## MCP Configuration

**Gateway / tool metadata (timeouts in ms):**

```ts
{
  templateId: "flotorch-eval",
  name: "Flotorch Evaluation MCP",
  description: "LLM evaluation: standard and RAG evaluation, multi-model comparison. Credentials via headers: X-Flotorch-Api-Key, X-Flotorch-Base-Url.",
  category: "Evaluation",
  icon: "i-lucide-clipboard-check",
  url: "",
  transport: "HTTP_STREAMABLE" as const,
  metadata: {
    transport: "HTTP_STREAMABLE" as const,
    timeout: 18000000,
    sse_read_timeout: 18000000,
    terminate_on_close: true,
  },
  requiredFields: [],
  baseHeaders: {},
  isEnabled: false,
}
```

**Local MCP client (JSON):**

```json
{
  "transport": "HTTP_STREAMABLE",
  "url": "http://localhost:8081",
  "headers": {
    "X-Flotorch-Api-Key": "your_api_key",
    "X-Flotorch-Base-Url": "https://gateway.flotorch.cloud"
  },
  "timeout": 18000000,
  "sse_read_timeout": 18000000
}
```

**Timeout / connection issues:** If you see "Request timed out" or "Connection closed", ensure the gateway and the MCP client both use `timeout` and `sse_read_timeout` of **18000000** (ms). If the client expects seconds, use **18000**. Run a small test (e.g. 3–5 Q&A, one model) to confirm the tool works before scaling up.

---

## Docker: Build and Verify

**1. Build the image**

```bash
docker build -t flotorch-eval-mcp .
```

**2. Run the container**

```bash
docker run -d -p 8081:8081 -e FLOTORCH_API_KEY=your_key -e FLOTORCH_BASE_URL=https://gateway.flotorch.cloud --name flotorch-eval-mcp flotorch-eval-mcp
```

**3. Verify the server**

```bash
curl -s http://localhost:8081/flotorch-eval/mcp
```

Expected: JSON with `"transport": "HTTP_STREAMABLE"` and a short message. A healthy container passes the Docker HEALTHCHECK (discovery endpoint returns 200).

**PowerShell (Windows):** Use `;` instead of `\` for line continuation, or run the `docker run` command as a single line as above.

**Optional – docker-compose:** The repo includes `docker-compose.yml` for convenience (e.g. `docker-compose up -d`). It is not required if you run the container with `docker run` or deploy via another orchestrator. You can remove `docker-compose.yml` if you do not use it.

---

## Example Agent Prompt (single line)

Single-line prompt for an agent with access to this MCP tool (model comparison; use ASCII-only in JSON):

```text
Use compare_llm_models with ground_truth=[{"question":"What is the time complexity of binary search?","answer":"O(log n)."},{"question":"What does CAP theorem state?","answer":"You cannot have consistency, availability, and partition tolerance all at once."},{"question":"Prime factorization of 84?","answer":"2^2 * 3 * 7."}], inference_models=["flotorch/nova-pro","flotorch/gemini-flash"], evaluation_model="flotorch/gpt-4-1-mini", embedding_model="flotorch/embed", system_prompt="You are a precise assistant. Answer concisely.", user_prompt_template="Question: {question}".
```

---

## Project Structure

```
src/flotorch_eval_mcp/
├── config.py      # Credentials, metrics, metric resolution
├── evaluator.py   # Dataset generation, evaluation orchestration
├── server.py      # FastMCP server and tool definitions
└── utils.py       # Validation, formatting, gateway metadata
```

---

## Requirements

- Python 3.11+
- flotorch, flotorch-eval, mcp (see `requirements.txt`)
