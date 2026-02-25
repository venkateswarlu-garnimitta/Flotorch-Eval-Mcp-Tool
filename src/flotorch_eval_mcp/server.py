#!/usr/bin/env python3
"""
Flotorch Evaluation MCP Server.

Exposes evaluation tools via FastMCP: evaluate_llm, evaluate_rag,
compare_llm_models, and list_evaluation_metrics.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

logging.basicConfig(
    format="[%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

try:
    from flotorch_eval_mcp.config import (
        DEFAULT_EVALUATION_ENGINE,
        EvaluationType,
        get_flotorch_credentials,
        resolve_metrics,
    )
    from flotorch_eval_mcp.evaluator import (
        KBRetrievalError,
        LLMGenerationError,
        generate_dataset_parallel,
        generate_rag_dataset_parallel,
        run_evaluation,
    )
    from flotorch_eval_mcp.utils import (
        enrich_eval_results_with_gateway_metadata,
        format_api_error,
        validate_evaluation_items,
        validate_ground_truth_data,
    )
    from flotorch.sdk.llm import FlotorchLLM
    from flotorch.sdk.memory import FlotorchAsyncVectorStore
    from flotorch_eval.llm_eval import EvaluationItem, MetricKey, LLMEvaluator
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    logger.warning(f"Failed to import heavy dependencies: {e}. Tool calls will return error messages.")

mcp = FastMCP(
    "Flotorch Evaluation",
    instructions=(
        "Flotorch Evaluation MCP Server: Provides comprehensive LLM evaluation tools. "
        "Supports standard evaluation, RAG evaluation with knowledge bases, and multi-model comparison. "
        "API credentials are read from HTTP headers (X-Flotorch-Api-Key, X-Flotorch-Base-Url)."
    ),
    json_response=True,
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    stateless_http=True,
)


@mcp.custom_route("/.well-known/flotorch-mcp", methods=["GET"])
async def discovery(_request: Request) -> JSONResponse:
    """Discovery endpoint for transport detection."""
    return JSONResponse({
        "transport": "HTTP_STREAMABLE",
        "protocol": "streamable-http",
        "message": "Flotorch Evaluation MCP Server - Set transport to HTTP_STREAMABLE",
    })


def _generate_comparison_summary(model_results: List[Dict], model_ids: List[str]) -> List[str]:
    """Generate a summary comparing model performance."""
    if not model_results:
        return []

    lines = [
        "=" * 80,
        "MODEL COMPARISON SUMMARY",
        "=" * 80,
    ]

    first_metrics = model_results[0]
    metric_keys = list(first_metrics.keys())

    lines.append("")
    lines.append("Average Scores by Model:")
    lines.append("-" * 40)

    for metric_key in metric_keys:
        lines.append(f"\n{metric_key.replace('_', ' ').title()}:")
        for i, result in enumerate(model_results):
            score = result.get(metric_key, "N/A")
            if isinstance(score, (int, float)):
                score = f"{score:.4f}"
            lines.append(f"  {model_ids[i]}: {score}")

    lines.append("")
    return lines


async def _evaluate_model_for_comparison(
    model_id: str,
    gt_data: List[Dict],
    api_key: str,
    base_url: str,
    system_prompt: str,
    user_prompt_template: str,
    evaluation_model: str,
    embedding_model: str,
    evaluation_engine: str,
    max_concurrent: int,
    metrics: List,
    gateway_metrics: bool = False,
) -> Dict:
    """Evaluate a single model for comparison."""
    try:
        llm = FlotorchLLM(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
        )

        evaluation_items_list = await generate_dataset_parallel(
            ground_truth=gt_data,
            llm=llm,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            max_concurrent=max_concurrent,
            gateway_metrics=gateway_metrics,
        )

        if not evaluation_items_list:
            raise ValueError("No evaluation items generated")

        try:
            eval_results = await run_evaluation(
                evaluation_items=evaluation_items_list,
                api_key=api_key,
                base_url=base_url,
                evaluation_model=evaluation_model,
                embedding_model=embedding_model,
                metrics=metrics,
                evaluation_engine=evaluation_engine,
            )
        except Exception as e:
            logger.error(f"Evaluation failed for model {model_id}: {e}")
            raise ValueError(f"Model evaluation failed: {format_api_error(e)}") from e

        overall_metrics = eval_results.get("evaluation_metrics", {})
        if gateway_metrics:
            enrich_eval_results_with_gateway_metadata(eval_results, evaluation_items_list)

        return {
            "metrics": overall_metrics,
            "eval_results": eval_results,
        }

    except Exception as e:
        logger.exception(f"Failed to evaluate model {model_id}")
        raise


async def _evaluate_single_model(
    ground_truth: List[Dict],
    inference_model: str,
    evaluation_model: str,
    system_prompt: str,
    user_prompt_template: str,
    embedding_model: str,
    evaluation_type: str,
    evaluation_engine: str,
    max_concurrent: int,
    metrics: List,
    gateway_metrics: bool,
    api_key: str,
    base_url: str,
) -> Dict[str, Any]:
    """Evaluate a single model (fallback from comparison when only one model)."""
    try:
        llm = FlotorchLLM(
            model_id=inference_model,
            api_key=api_key,
            base_url=base_url,
        )

        evaluation_items_list = await generate_dataset_parallel(
            ground_truth=ground_truth,
            llm=llm,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            max_concurrent=max_concurrent,
            gateway_metrics=gateway_metrics,
        )

        if not evaluation_items_list:
            return {"error": "No evaluation items generated"}

        eval_results = await run_evaluation(
            evaluation_items=evaluation_items_list,
            api_key=api_key,
            base_url=base_url,
            evaluation_model=evaluation_model,
            embedding_model=embedding_model,
            metrics=metrics,
            evaluation_engine=evaluation_engine,
        )

        if gateway_metrics:
            enrich_eval_results_with_gateway_metadata(eval_results, evaluation_items_list)
        return eval_results

    except Exception as e:
        logger.exception(f"Failed to evaluate single model {inference_model}")
        return {"error": format_api_error(e)}


def _extract_headers_from_context(ctx: Optional[Context]) -> Dict[str, str]:
    """Extract HTTP headers from request context as lowercase dict."""
    if not ctx:
        return {}
    try:
        request_context = ctx.request_context
        if hasattr(request_context, "request") and request_context.request:
            request = request_context.request
            return {name.lower(): request.headers[name] for name in request.headers.keys()}
    except Exception as e:
        logger.debug(f"Could not extract headers: {e}")
    return {}


def _parse_metric_names(metrics_json: str) -> Optional[List[str]]:
    """Parse optional JSON metrics list. Returns None if empty/unset."""
    if not metrics_json:
        return None
    try:
        parsed = json.loads(metrics_json) if isinstance(metrics_json, str) else metrics_json
        if isinstance(parsed, list) and parsed:
            return [str(m) for m in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def evaluate_llm(
    evaluation_items: str,
    evaluation_model: str,
    embedding_model: str = "flotorch/text-embedding-model",
    evaluation_type: str = "normal",
    evaluation_engine: str = DEFAULT_EVALUATION_ENGINE,
    metrics: str = "",
    query_level_metrics: bool = False,
    gateway_metrics: bool = False,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Evaluate pre-computed LLM responses using question-answer pairs.

    Args:
        evaluation_items: JSON string - list of objects with question, generated_answer, expected_answer, and optional context list
        evaluation_model: Flotorch model ID for evaluation scoring (required)
        embedding_model: Flotorch embedding model ID (default: flotorch/text-embedding-model)
        evaluation_type: "normal" or "rag" (default: normal)
        evaluation_engine: "deepeval" or "ragas" (default: deepeval)
        metrics: Optional JSON array of metric names to evaluate (e.g. ["faithfulness", "answer_relevance"]). If omitted, uses defaults for the evaluation_type.
        query_level_metrics: Include per-query breakdown (default: false)
        gateway_metrics: Include performance data - tokens, latency per query and aggregated totals (default: false)

    Returns:
        Evaluation results with overall metrics and optionally per-query scores and performance data
    """
    if not IMPORTS_SUCCESSFUL:
        return {"error": "Required dependencies are not installed. Please install flotorch, flotorch-eval packages."}

    try:
        headers = _extract_headers_from_context(ctx)
        api_key, base_url = get_flotorch_credentials(headers)
    except ValueError as e:
        return {"error": f"API credentials invalid or missing. {e}"}

    try:
        items_data = json.loads(evaluation_items) if isinstance(evaluation_items, str) else evaluation_items
    except json.JSONDecodeError as e:
        return {"error": f"evaluation_items must be valid JSON. Parse error: {e}"}

    is_valid, error_msg = validate_evaluation_items(items_data)
    if not is_valid:
        return {"error": error_msg}

    evaluation_items_list: List[EvaluationItem] = []
    for item in items_data:
        context = item.get("context", [])
        if not isinstance(context, list):
            context = [str(context)] if context else []
        metadata = item.get("metadata", {})
        if metadata and isinstance(metadata, dict):
            metadata = {str(k): v for k, v in metadata.items()}

        evaluation_items_list.append(
            EvaluationItem(
                question=item.get("question", ""),
                generated_answer=item.get("generated_answer", ""),
                expected_answer=item.get("expected_answer", ""),
                context=context,
                metadata=metadata,
            )
        )

    try:
        resolved = resolve_metrics(_parse_metric_names(metrics), evaluation_type, MetricKey)
    except ValueError as e:
        return {"error": str(e)}

    try:
        eval_results = await run_evaluation(
            evaluation_items=evaluation_items_list,
            api_key=api_key,
            base_url=base_url,
            evaluation_model=evaluation_model,
            embedding_model=embedding_model,
            metrics=resolved,
            evaluation_engine=evaluation_engine,
        )
        if gateway_metrics:
            enrich_eval_results_with_gateway_metadata(eval_results, evaluation_items_list)
        return eval_results

    except Exception as e:
        logger.exception("Evaluation failed")
        return {
            "error": f"Evaluation failed. {format_api_error(e)}",
            "evaluation_model": evaluation_model,
            "embedding_model": embedding_model,
        }


@mcp.tool()
async def evaluate_rag(
    ground_truth: str,
    knowledge_base_id: str,
    inference_model: str,
    evaluation_model: str,
    system_prompt: str,
    user_prompt_template: str,
    embedding_model: str = "flotorch/text-embedding-model",
    evaluation_engine: str = DEFAULT_EVALUATION_ENGINE,
    metrics: str = "",
    query_level_metrics: bool = False,
    gateway_metrics: bool = False,
    max_concurrent: int = 10,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Run full RAG evaluation: retrieve from knowledge base, generate answers, then evaluate.

    Args:
        ground_truth: JSON string - list of {question, answer} objects (answer = ground truth)
        knowledge_base_id: Flotorch Knowledge Base ID for retrieval
        inference_model: Flotorch model ID for generating answers (required)
        evaluation_model: Flotorch model ID for evaluation scoring (required)
        system_prompt: System prompt for the inference LLM (required)
        user_prompt_template: User prompt template with {context} and {question} placeholders (required)
        embedding_model: Flotorch embedding model ID (default: flotorch/text-embedding-model)
        evaluation_engine: "deepeval" or "ragas" (default: deepeval)
        metrics: Optional JSON array of metric names (e.g. ["faithfulness", "hallucination"]). If omitted, uses all RAG metrics.
        query_level_metrics: Include per-query breakdown (default: false)
        gateway_metrics: Include performance data - tokens, latency per query and aggregated totals (default: false)
        max_concurrent: Maximum concurrent operations per phase (default: 10)

    Returns:
        Evaluation results with RAG metrics and optionally per-query scores and performance data
    """
    if not IMPORTS_SUCCESSFUL:
        return {"error": "Required dependencies are not installed. Please install flotorch, flotorch-eval packages."}

    try:
        headers = _extract_headers_from_context(ctx)
        api_key, base_url = get_flotorch_credentials(headers)
    except ValueError as e:
        return {"error": f"API credentials invalid or missing. {e}"}

    try:
        gt_data = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except json.JSONDecodeError as e:
        return {"error": f"ground_truth must be valid JSON. Parse error: {e}"}

    is_valid, error_msg = validate_ground_truth_data(gt_data)
    if not is_valid:
        return {"error": error_msg}

    try:
        inference_llm = FlotorchLLM(
            model_id=inference_model,
            api_key=api_key,
            base_url=base_url,
        )
        kb = FlotorchAsyncVectorStore(
            base_url=base_url,
            api_key=api_key,
            vectorstore_id=knowledge_base_id,
        )
    except Exception as e:
        return {"error": f"Failed to initialize inference model or knowledge base. {format_api_error(e)}"}

    try:
        resolved = resolve_metrics(_parse_metric_names(metrics), EvaluationType.RAG.value, MetricKey)
    except ValueError as e:
        return {"error": str(e)}

    try:
        logger.info(f"Generating RAG dataset for {len(gt_data)} questions...")
        try:
            evaluation_items_list = await generate_rag_dataset_parallel(
                ground_truth=gt_data,
                kb=kb,
                llm=inference_llm,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                max_concurrent=max_concurrent,
                gateway_metrics=gateway_metrics,
            )
        except LLMGenerationError as e:
            return (
                f"Experiment stopped: Inference model failed to generate answers.\n"
                f"Reason: {format_api_error(e)}\n"
                f"Inference model: {inference_model}"
            )
        except KBRetrievalError as e:
            return (
                f"Experiment stopped: Knowledge base retrieval failed.\n"
                f"Reason: {e}\n"
                f"Knowledge base ID: {knowledge_base_id}"
            )

        if not evaluation_items_list:
            return "Error: No evaluation items generated"

        logger.info("Dataset generated. Running evaluation...")

        try:
            eval_results = await run_evaluation(
                evaluation_items=evaluation_items_list,
                api_key=api_key,
                base_url=base_url,
                evaluation_model=evaluation_model,
                embedding_model=embedding_model,
                metrics=resolved,
                evaluation_engine=evaluation_engine,
            )
        except Exception as e:
            return (
                f"Experiment stopped: Evaluation model or embedding model failed.\n"
                f"Reason: {format_api_error(e)}\n"
                f"Evaluation model: {evaluation_model}, Embedding model: {embedding_model}"
            )

        if gateway_metrics:
            enrich_eval_results_with_gateway_metadata(eval_results, evaluation_items_list)
        return eval_results

    except Exception as e:
        logger.exception("RAG evaluation failed")
        return {"error": f"RAG evaluation failed: {format_api_error(e)}"}


@mcp.tool()
async def compare_llm_models(
    ground_truth: str,
    inference_models: str,
    evaluation_model: str,
    system_prompt: str,
    user_prompt_template: str,
    embedding_model: str = "flotorch/text-embedding-model",
    evaluation_type: str = "normal",
    evaluation_engine: str = DEFAULT_EVALUATION_ENGINE,
    metrics: str = "",
    query_level_metrics: bool = False,
    gateway_metrics: bool = False,
    max_concurrent: int = 10,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Compare multiple LLM models on the same ground truth dataset.

    All models are evaluated in full parallel — dataset generation and evaluation
    run concurrently across models. Within each model, questions are processed
    in parallel up to max_concurrent.

    Args:
        ground_truth: JSON string - list of {question, answer} or {question, answer, context} objects
        inference_models: JSON string - array of model IDs (required)
        evaluation_model: Flotorch model ID for evaluation scoring (required)
        system_prompt: System prompt for inference LLMs (required)
        user_prompt_template: Template with {context} and {question} placeholders (required)
        embedding_model: Flotorch embedding model ID (default: flotorch/text-embedding-model)
        evaluation_type: "normal" or "rag" (default: normal)
        evaluation_engine: "deepeval" or "ragas" (default: deepeval)
        metrics: Optional JSON array of metric names (e.g. ["faithfulness", "answer_relevance"]). If omitted, uses defaults for the evaluation_type.
        query_level_metrics: Include per-query breakdown (default: false)
        gateway_metrics: Include performance data - tokens, latency per query and aggregated totals (default: false)
        max_concurrent: Maximum concurrent calls per model (default: 10)

    Returns:
        Comparison report with per-model metrics and optionally performance data and summary
    """
    if not IMPORTS_SUCCESSFUL:
        return {"error": "Required dependencies are not installed. Please install flotorch, flotorch-eval packages."}

    try:
        headers = _extract_headers_from_context(ctx)
        api_key, base_url = get_flotorch_credentials(headers)
    except ValueError as e:
        return {"error": f"API credentials invalid or missing. {e}"}

    try:
        gt_data = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except json.JSONDecodeError as e:
        return {"error": f"ground_truth must be valid JSON. Parse error: {e}"}

    try:
        models_list = json.loads(inference_models) if isinstance(inference_models, str) else inference_models
    except json.JSONDecodeError as e:
        return {"error": f"inference_models must be valid JSON array. Parse error: {e}"}

    is_valid, error_msg = validate_ground_truth_data(gt_data)
    if not is_valid:
        return {"error": error_msg}

    if not isinstance(models_list, list) or not models_list:
        return {"error": "inference_models must be a non-empty list of model IDs"}

    try:
        resolved = resolve_metrics(_parse_metric_names(metrics), evaluation_type, MetricKey)
    except ValueError as e:
        return {"error": str(e)}

    if len(models_list) == 1:
        return await _evaluate_single_model(
            ground_truth=gt_data,
            inference_model=models_list[0],
            evaluation_model=evaluation_model,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            embedding_model=embedding_model,
            evaluation_type=evaluation_type,
            evaluation_engine=evaluation_engine,
            max_concurrent=max_concurrent,
            metrics=resolved,
            gateway_metrics=gateway_metrics,
            api_key=api_key,
            base_url=base_url,
        )

    # All models evaluated in full parallel (no model-level semaphore)
    async def evaluate_model(model_id: str) -> dict:
        return await _evaluate_model_for_comparison(
            model_id=model_id,
            gt_data=gt_data,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            evaluation_model=evaluation_model,
            embedding_model=embedding_model,
            evaluation_engine=evaluation_engine,
            max_concurrent=max_concurrent,
            metrics=resolved,
            gateway_metrics=gateway_metrics,
        )

    tasks = [evaluate_model(model_id) for model_id in models_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    comparison_results: Dict[str, Any] = {}
    model_metrics = []

    for i, result in enumerate(results):
        model_id = models_list[i]
        if isinstance(result, Exception):
            comparison_results[model_id] = {"error": format_api_error(result)}
        else:
            comparison_results[model_id] = result["eval_results"]
            model_metrics.append(result["metrics"])

    if len(model_metrics) > 1:
        successful_ids = [models_list[i] for i, r in enumerate(results) if not isinstance(r, Exception)]
        comparison_results["comparison_summary"] = _generate_comparison_summary(model_metrics, successful_ids)

    return comparison_results


@mcp.tool()
def list_evaluation_metrics() -> str:
    """
    List all available evaluation metrics and their descriptions.

    Returns:
        JSON with metric names, descriptions, and supported engines
    """
    metrics = {
        "available_metrics": [
            {
                "name": "faithfulness",
                "description": "Is the answer factually consistent with the retrieved context?",
                "engines": ["deepeval", "ragas"],
                "requires_context": True,
            },
            {
                "name": "answer_relevance",
                "description": "Does the answer directly address the question?",
                "engines": ["deepeval", "ragas"],
                "requires_context": False,
            },
            {
                "name": "context_relevancy",
                "description": "Is the retrieved context relevant to the question?",
                "engines": ["deepeval"],
                "requires_context": True,
            },
            {
                "name": "context_precision",
                "description": "Is the retrieved context precise and focused?",
                "engines": ["deepeval", "ragas"],
                "requires_context": True,
            },
            {
                "name": "context_recall",
                "description": "Does the context cover the information needed to answer?",
                "engines": ["deepeval"],
                "requires_context": True,
            },
            {
                "name": "hallucination",
                "description": "Does the answer contain fabricated information not in the context?",
                "engines": ["deepeval"],
                "requires_context": True,
            },
        ],
        "default_engine": DEFAULT_EVALUATION_ENGINE,
        "supported_engines": ["deepeval", "ragas"],
        "defaults": {
            "normal": ["answer_relevance"],
            "rag": [
                "faithfulness",
                "context_relevancy",
                "context_precision",
                "context_recall",
                "answer_relevance",
                "hallucination",
            ],
        },
        "performance_metrics": {
            "description": "Automatically included in all evaluation results",
            "metrics": [
                "latency_ms (per-query wall-clock latency)",
                "input_tokens / output_tokens (per-query token counts)",
                "avg_latency_ms, total_latency_ms (aggregated in gateway_metrics)",
            ],
        },
    }

    return json.dumps(metrics, indent=2)


async def main() -> None:
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")

    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.log_level = "INFO"

    logger.info(
        f"Flotorch Evaluation MCP Server starting on http://{host}:{port}\n"
        "Streamable HTTP at / | Discovery at /.well-known/flotorch-mcp"
    )

    await mcp.run_streamable_http_async()


if __name__ == "__main__":
    asyncio.run(main())
