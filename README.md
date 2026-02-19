---
title: Flotorch Evaluation MCP Server
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
# CPU basic may OOM (exit 137) - use cpu-upgrade in Space Settings
suggested_hardware: cpu-upgrade
---

# Flotorch Evaluation MCP Server

A high-performance MCP server for comprehensive LLM evaluation using the Flotorch platform.

## Features

- **Standard LLM Evaluation**: Evaluate pre-computed question-answer pairs
- **RAG Evaluation**: Full RAG pipeline with knowledge base retrieval and generation
- **Multi-Model Comparison**: Compare multiple LLMs on the same dataset
- **Parallel Processing**: 5-10x faster with configurable concurrency
- **Gateway Metrics**: Track latency, cost, and token usage
- **Query-Level Results**: Detailed per-question metric breakdown

## Project Structure

```
Flotorch-Eval-Mcp-Tool/
├── src/flotorch_eval_mcp/     # Application package
│   ├── config.py              # Credentials, metrics config
│   ├── evaluator.py           # Evaluation workflows
│   ├── server.py              # FastMCP server & tools
│   └── utils.py               # Helpers, validation
├── deploy/                    # Deployment configs (EC2, Hugging Face)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Credentials

Set environment variables or use headers:

```bash
FLOTORCH_API_KEY=your_api_key_here
FLOTORCH_BASE_URL=https://gateway.flotorch.cloud
```

### 3. Start Server

```bash
# From project root (with src/ on PYTHONPATH)
pip install -e .
python -m flotorch_eval_mcp

# Or run directly
python -m flotorch_eval_mcp
```

Server starts on `http://0.0.0.0:8080`

## Usage

### Standard LLM Evaluation

```python
# Evaluate pre-computed answers
result = evaluate_llm(
    evaluation_items=json.dumps([
        {
            "question": "What is AI?",
            "generated_answer": "AI is artificial intelligence...",
            "expected_answer": "AI is artificial intelligence",
            "context": ["Context chunk 1"]
        }
    ]),
    evaluation_model="flotorch/gpt-4o",
    embedding_model="flotorch/text-embedding-model",
    evaluation_type="rag"  # or "normal"
)
```

### RAG Evaluation

```python
# Full RAG pipeline
result = evaluate_rag(
    ground_truth=json.dumps([
        {"question": "What is AI?", "answer": "AI is artificial intelligence"}
    ]),
    knowledge_base_id="your_kb_id",
    inference_model="flotorch/gpt-4o",
    evaluation_model="flotorch/gemini-flash",
    embedding_model="flotorch/text-embedding-model"
)
```

### Model Comparison

```python
# Compare multiple models
result = compare_llm_models(
    ground_truth=json.dumps([
        {"question": "What is AI?", "answer": "AI is artificial intelligence"}
    ]),
    inference_models=json.dumps(["flotorch/gpt-4o", "flotorch/gemini-flash"]),
    evaluation_model="flotorch/gpt-4o",
    embedding_model="flotorch/text-embedding-model"
)
```

## Configuration

### Environment Variables

```bash
PORT=8080                    # Server port (default: 8080)
HOST=0.0.0.0                # Server host (default: 0.0.0.0)
LOG_LEVEL=INFO              # Logging level (default: INFO)
```

### HTTP Headers (Recommended)

When calling from MCP client:

```
X-Flotorch-Api-Key: your_api_key
X-Flotorch-Base-Url: https://gateway.flotorch.cloud
```

## Evaluation Types & Metrics

### Normal Evaluation (no context)
- Answer Relevance

### RAG Evaluation (with context)
- Faithfulness
- Context Relevancy
- Context Precision
- Context Recall
- Answer Relevance
- Hallucination Detection

## Performance Tuning

### Concurrency Settings

- **Small datasets** (<10 items): `max_concurrent=3`
- **Medium datasets** (10-50 items): `max_concurrent=5`
- **Large datasets** (>50 items): `max_concurrent=10`
- **Rate-limited APIs**: `max_concurrent=3`

### Performance Benefits

- **Sequential**: ~500s for 100 items
- **Parallel (5x)**: ~100s for 100 items
- **Parallel (10x)**: ~50s for 100 items

## Agent Integration

Add to your Flotorch agent configuration:

```json
{
  "tools": [
    {
      "name": "flotorch-eval",
      "type": "MCP",
      "config": {
        "transport": "HTTP_STREAMABLE",
        "url": "http://localhost:8080",
        "headers": {
          "X-Flotorch-Base-Url": "https://dev-gateway.flotorch.cloud",
          "X-Flotorch-Api-Key": "your_api_key"
        }
      }
    }
  ]
}
```

## API Reference

### Tools

1. **evaluate_llm** - Standard evaluation with pre-computed answers
2. **evaluate_rag** - Full RAG evaluation pipeline
3. **compare_llm_models** - Multi-model comparison
4. **list_evaluation_metrics** - List available metrics

### Parameters

All tools support:
- `evaluation_engine`: "deepeval" or "ragas"
- `query_level_metrics`: Include per-question breakdown
- `gateway_metrics`: Include latency/cost metrics
- `max_concurrent`: Parallel processing concurrency

## Troubleshooting

### Common Issues

- **"API key not found"**: Set credentials in headers or environment
- **"Model not available"**: Check model names in Flotorch console
- **"Knowledge Base not found"**: Verify KB ID exists
- **Slow performance**: Increase `max_concurrent` parameter
- **Rate limit errors**: Decrease `max_concurrent` parameter

### Debugging

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python -m flotorch_eval_mcp
```

## Deployment

The server can be deployed via Docker to **Hugging Face Spaces** or **AWS EC2**.

### Docker (Local / EC2)

```bash
# Build and run
docker build -t flotorch-eval-mcp:latest .
docker run -d -p 8080:8080 \
  -e FLOTORCH_API_KEY=your_key \
  -e FLOTORCH_BASE_URL=https://gateway.flotorch.cloud \
  flotorch-eval-mcp:latest

# Or with docker-compose
cp .env.example .env   # Edit with your credentials
docker compose up -d
```

### Hugging Face Spaces

1. Create a new Space with **Docker** SDK
2. Set `app_port: 7860` in README YAML
3. Copy project files and use `deploy/huggingface/Dockerfile`
4. Add `FLOTORCH_API_KEY` and `FLOTORCH_BASE_URL` as Space secrets/variables

See [deploy/huggingface/README.md](deploy/huggingface/README.md) for details.

### AWS EC2

```bash
# On your EC2 instance
git clone <repo> && cd flotorch-eval-mcp
cp .env.example .env   # Edit with your credentials
./deploy/ec2/deploy.sh --build
```

See [deploy/ec2/README.md](deploy/ec2/README.md) for full instructions.

## Requirements

- Python 3.11+
- Flotorch account with API access
- Valid model IDs and knowledge base IDs (if using RAG)

## License

Part of the Flotorch evaluation toolkit.
