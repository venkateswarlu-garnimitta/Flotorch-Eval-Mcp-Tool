#!/usr/bin/env python3
"""
Flotorch Evaluation MCP Server using FastMCP.

A robust MCP tool for evaluating LLMs with support for:
- Standard LLM evaluation with pre-computed answers
- RAG evaluation with knowledge base retrieval
- Multi-model comparison experiments
- Gateway metrics and per-query results
- Parallel processing for speed
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

# FastMCP imports
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

# Configure logging
logging.basicConfig(
    format="[%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Heavy dependencies (imported at startup for fast tool calls)
try:
    from flotorch_eval_mcp.config import (
        EvaluationType,
        get_flotorch_credentials,
        get_metrics_for_evaluation_type,
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

# Create FastMCP server
# Disable host-header validation to allow deployment behind reverse proxies
# (e.g. Hugging Face Spaces, load balancers) where Host differs from origin
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

    # Extract metric keys from first result
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
    evaluation_type: str,
    evaluation_engine: str,
    query_level_metrics: bool,
    gateway_metrics: bool,
    max_concurrent: int,
    metrics: List,
) -> Dict:
    """Evaluate a single model for comparison, with robust error handling."""
    try:
        # Initialize LLM
        llm = FlotorchLLM(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
        )

        # Generate dataset
        evaluation_items_list = await generate_dataset_parallel(
            ground_truth=gt_data,
            llm=llm,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            max_concurrent=max_concurrent,
            return_headers=gateway_metrics,
        )

        if not evaluation_items_list:
            raise ValueError("No evaluation items generated")

        # Run evaluation
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

        # Extract overall metrics for comparison
        overall_metrics = eval_results.get("evaluation_metrics", {})

        # Enrich with gateway metadata
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
    query_level_metrics: bool,
    gateway_metrics: bool,
    max_concurrent: int,
    api_key: str,
    base_url: str,
) -> Dict[str, Any]:
    """Evaluate a single model (fallback from comparison)."""
    try:
        # Initialize LLM
        llm = FlotorchLLM(
            model_id=inference_model,
            api_key=api_key,
            base_url=base_url,
        )

        # Generate dataset
        evaluation_items_list = await generate_dataset_parallel(
            ground_truth=ground_truth,
            llm=llm,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            max_concurrent=max_concurrent,
            return_headers=gateway_metrics,
        )

        if not evaluation_items_list:
            return {"error": "No evaluation items generated"}

        # Run evaluation
        eval_results = await run_evaluation(
            evaluation_items=evaluation_items_list,
            api_key=api_key,
            base_url=base_url,
            evaluation_model=evaluation_model,
            embedding_model=embedding_model,
            metrics=get_metrics_for_evaluation_type(evaluation_type, MetricKey),
            evaluation_engine=evaluation_engine,
        )

        # Enrich with gateway metadata
        enrich_eval_results_with_gateway_metadata(eval_results, evaluation_items_list)

        return eval_results

    except Exception as e:
        logger.exception(f"Failed to evaluate single model {inference_model}")
        return {"error": format_api_error(e)}


def _extract_headers_from_context(ctx: Context) -> Dict[str, str]:
    """Extract HTTP headers from context as lowercase dict."""
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


@mcp.tool()
async def evaluate_llm(
    evaluation_items: str,
    evaluation_model: str,
    embedding_model: str = "flotorch/text-embedding-model",
    evaluation_type: str = "normal",
    evaluation_engine: str = "deepeval",
    query_level_metrics: bool = False,
    gateway_metrics: bool = False,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Evaluate pre-computed LLM responses using question-answer pairs.

    Args:
        evaluation_items: JSON string - list of evaluation items with question, generated_answer, expected_answer
        evaluation_model: Flotorch model ID for evaluation scoring (required)
        embedding_model: Flotorch embedding model ID (default: flotorch/text-embedding-model)
        evaluation_type: "normal" or "rag" (default: normal)
        evaluation_engine: "deepeval" or "ragas" (default: deepeval)
        query_level_metrics: Include per-query breakdown (default: false)
        gateway_metrics: Include gateway metrics (input/output tokens) (default: false)

    Returns:
        Formatted evaluation report
    """
    if not IMPORTS_SUCCESSFUL:
        return {"error": "Required dependencies are not installed. Please install flotorch, flotorch-eval packages."}

    # Get credentials
    try:
        headers = _extract_headers_from_context(ctx)
        api_key, base_url = get_flotorch_credentials(headers)
    except ValueError as e:
        return {"error": f"API credentials invalid or missing. {e}"}

    # Parse evaluation items
    try:
        items_data = json.loads(evaluation_items) if isinstance(evaluation_items, str) else evaluation_items
    except json.JSONDecodeError as e:
        return {"error": f"evaluation_items must be valid JSON. Parse error: {e}"}

    # Validate structure
    is_valid, error_msg = validate_evaluation_items(items_data)
    if not is_valid:
        return {"error": error_msg}

    # Build EvaluationItem list
    evaluation_items_list: List[EvaluationItem] = []
    for item in items_data:
        question = item.get("question", "")
        generated_answer = item.get("generated_answer", "")
        expected_answer = item.get("expected_answer", "")
        context = item.get("context", [])
        metadata = item.get("metadata", {})

        # Normalize context
        if not isinstance(context, list):
            context = [str(context)] if context else []

        # Normalize metadata
        if metadata and isinstance(metadata, dict):
            metadata = {str(k): v for k, v in metadata.items()}

        evaluation_items_list.append(
            EvaluationItem(
                question=question,
                generated_answer=generated_answer,
                expected_answer=expected_answer,
                context=context,
                metadata=metadata,
            )
        )

    # Get appropriate metrics for evaluation type
    metrics = get_metrics_for_evaluation_type(evaluation_type, MetricKey)

    try:
        # Run evaluation (uses evaluation_model and embedding_model)
        eval_results = await run_evaluation(
            evaluation_items=evaluation_items_list,
            api_key=api_key,
            base_url=base_url,
            evaluation_model=evaluation_model,
            embedding_model=embedding_model,
            metrics=metrics,
            evaluation_engine=evaluation_engine,
        )

        enrich_eval_results_with_gateway_metadata(eval_results, evaluation_items_list)

        return eval_results

    except Exception as e:
        logger.exception("Evaluation failed")
        return {
            "error": f"Evaluation failed. {format_api_error(e)}",
            "evaluation_model": evaluation_model,
            "embedding_model": embedding_model
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
    evaluation_engine: str = "deepeval",
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
        evaluation_engine: Evaluation engine - "deepeval" or "ragas" (default: deepeval)
        query_level_metrics: Include per-query breakdown (default: false)
        gateway_metrics: Include gateway metrics (input/output tokens) (default: false)
        max_concurrent: Maximum concurrent operations (default: 5)

    Returns:
        Formatted evaluation report with RAG metrics
    """
    if not IMPORTS_SUCCESSFUL:
        return {"error": "Required dependencies are not installed. Please install flotorch, flotorch-eval packages."}

    # Get credentials
    try:
        headers = _extract_headers_from_context(ctx)
        api_key, base_url = get_flotorch_credentials(headers)
    except ValueError as e:
        return {"error": f"API credentials invalid or missing. {e}"}

    # Parse ground truth
    try:
        gt_data = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except json.JSONDecodeError as e:
        return {"error": f"ground_truth must be valid JSON. Parse error: {e}"}

    # Validate structure
    is_valid, error_msg = validate_ground_truth_data(gt_data)
    if not is_valid:
        return {"error": error_msg}

    # Import SDK components (lazy loading for faster startup)
    try:
        # Initialize inference LLM and knowledge base
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

    # Get RAG metrics
    metrics = get_metrics_for_evaluation_type(EvaluationType.RAG.value, MetricKey)

    try:
        # Generate dataset in parallel (retrieve + generate)
        logger.info(f"Generating RAG dataset for {len(gt_data)} questions...")
        try:
            evaluation_items_list = await generate_rag_dataset_parallel(
                ground_truth=gt_data,
                kb=kb,
                llm=inference_llm,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                max_concurrent=max_concurrent,
                return_headers=gateway_metrics,
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

        logger.info(f"Dataset generated. Running evaluation...")

        # Run evaluation (uses evaluation_model and embedding_model)
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
            return (
                f"Experiment stopped: Evaluation model or embedding model failed.\n"
                f"Reason: {format_api_error(e)}\n"
                f"Evaluation model: {evaluation_model}, Embedding model: {embedding_model}"
            )

        # Inject per-query gateway metadata (flotorch_eval drops it from question results)
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
    evaluation_engine: str = "deepeval",
    query_level_metrics: bool = False,
    gateway_metrics: bool = False,
    max_concurrent: int = 10,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Compare multiple LLM models on the same ground truth dataset.

    Args:
        ground_truth: JSON string - list of {question, answer} or {question, answer, context} objects
        inference_models: JSON string - array of model IDs (required)
        evaluation_model: Flotorch model ID for evaluation scoring (required)
        system_prompt: System prompt for inference LLMs (required)
        user_prompt_template: Template with {context} and {question} placeholders (required)
        embedding_model: Flotorch embedding model ID (default: flotorch/text-embedding-model)
        evaluation_type: "normal" or "rag" (default: normal)
        evaluation_engine: "deepeval" or "ragas" (default: deepeval)
        query_level_metrics: Include per-query breakdown (default: false)
        gateway_metrics: Include gateway metrics (input/output tokens) (default: false)
        max_concurrent: Maximum concurrent calls per model (default: 5)

    Returns:
        Comparison report showing metrics across models
    """
    if not IMPORTS_SUCCESSFUL:
        return {"error": "Required dependencies are not installed. Please install flotorch, flotorch-eval packages."}

    # Get credentials
    try:
        headers = _extract_headers_from_context(ctx)
        api_key, base_url = get_flotorch_credentials(headers)
    except ValueError as e:
        return {"error": f"API credentials invalid or missing. {e}"}

    # Parse inputs
    try:
        gt_data = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except json.JSONDecodeError as e:
        return {"error": f"ground_truth must be valid JSON. Parse error: {e}"}

    try:
        models_list = json.loads(inference_models) if isinstance(inference_models, str) else inference_models
    except json.JSONDecodeError as e:
        return {"error": f"inference_models must be valid JSON array. Parse error: {e}"}

    # Validate
    is_valid, error_msg = validate_ground_truth_data(gt_data)
    if not is_valid:
        return {"error": error_msg}

    if not isinstance(models_list, list) or not models_list:
        return "Error: inference_models must be a non-empty list of model IDs"

    # Import SDK components (lazy loading for faster startup)
    try:
        from flotorch.sdk.llm import FlotorchLLM
        from flotorch_eval.llm_eval import MetricKey
    except ImportError as e:
        return {"error": f"Missing required packages: {e}. Run: pip install flotorch flotorch-eval"}

    # Get appropriate metrics
    metrics = get_metrics_for_evaluation_type(evaluation_type, MetricKey)

    # If only one model, evaluate normally instead of comparison
    if len(models_list) == 1:
        single_model = models_list[0]
        return await _evaluate_single_model(
            ground_truth=gt_data,
            inference_model=single_model,
            evaluation_model=evaluation_model,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            embedding_model=embedding_model,
            evaluation_type=evaluation_type,
            evaluation_engine=evaluation_engine,
            query_level_metrics=query_level_metrics,
            gateway_metrics=gateway_metrics,
            max_concurrent=max_concurrent,
            api_key=api_key,
            base_url=base_url,
        )

    # Build comparison report header
    report_lines = [
        "=" * 80,
        f"LLM COMPARISON REPORT ({evaluation_engine.upper()} ENGINE)",
        f"Evaluation Type: {evaluation_type.upper()}",
        f"Models: {len(models_list)} | Questions: {len(gt_data)}",
        "=" * 80,
        "",
    ]

    model_results = []

    # Evaluate each model in parallel
    semaphore = asyncio.Semaphore(min(len(models_list), 3))  # Limit concurrent model evaluations

    async def evaluate_model(model_id: str) -> dict:
        async with semaphore:
            return await _evaluate_model_for_comparison(
                model_id=model_id,
                gt_data=gt_data,
                api_key=api_key,
                base_url=base_url,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                evaluation_model=evaluation_model,
                embedding_model=embedding_model,
                evaluation_type=evaluation_type,
                evaluation_engine=evaluation_engine,
                query_level_metrics=query_level_metrics,
                gateway_metrics=gateway_metrics,
                max_concurrent=max_concurrent,
                metrics=metrics,
            )

    tasks = [evaluate_model(model_id) for model_id in models_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Build structured results
    comparison_results = {}

    for i, result in enumerate(results):
        model_id = models_list[i]

        if isinstance(result, Exception):
            comparison_results[model_id] = {"error": format_api_error(result)}
        else:
            comparison_results[model_id] = result["eval_results"]
            model_results.append(result["metrics"])

    # Add comparison summary if we have multiple successful results
    if len(model_results) > 1:
        comparison_results["comparison_summary"] = _generate_comparison_summary(model_results, models_list)

    return comparison_results


@mcp.tool()
def list_evaluation_metrics() -> str:
    """
    List all available evaluation metrics and their descriptions.

    Returns:
        JSON string with metric information
    """
    metrics = {
        "normal_evaluation": {
            "description": "Metrics for standard LLM evaluation (no context)",
            "metrics": [
                {
                    "name": "answer_relevance",
                    "description": "Does the answer directly address the question?"
                }
            ]
        },
        "rag_evaluation": {
            "description": "Metrics for RAG system evaluation (with context)",
            "metrics": [
                {
                    "name": "faithfulness",
                    "description": "Is the answer supported by the retrieved context?"
                },
                {
                    "name": "context_relevancy",
                    "description": "Is the retrieved context relevant to the question?"
                },
                {
                    "name": "context_precision",
                    "description": "Is the retrieved context precise and focused?"
                },
                {
                    "name": "context_recall",
                    "description": "Does the context cover the information needed to answer?"
                },
                {
                    "name": "answer_relevance",
                    "description": "Does the answer directly address the question?"
                },
                {
                    "name": "hallucination",
                    "description": "Does the answer contain fabricated information not in context?"
                }
            ]
        },
        "supported_engines": ["deepeval", "ragas"]
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
