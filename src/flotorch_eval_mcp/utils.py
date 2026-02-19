"""
Utility functions for the Flotorch Evaluation MCP Server.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def format_api_error(exc: Exception) -> str:
    """
    Extract a clear user-facing message from API/gateway errors.
    Handles common patterns: ERROR_MODEL_NOT_FOUND, 500 JSON, etc.
    """
    msg = str(exc).strip()
    if not msg:
        return f"API error: {type(exc).__name__}"
    # Extract from [500] {"error":{"message":"ERROR_MODEL_NOT_FOUND",...}}
    m = re.search(r'"message"\s*:\s*"([^"]+)"', msg)
    if m:
        return m.group(1)
    if "ERROR_MODEL_NOT_FOUND" in msg:
        return "Model not found. Check that the model ID exists and is available on the gateway."
    if "401" in msg or "unauthorized" in msg.lower() or "api" in msg.lower() and "key" in msg.lower():
        return "API key invalid or missing. Check X-Flotorch-Api-Key header or FLOTORCH_API_KEY."
    if "404" in msg or "not found" in msg.lower():
        return "Resource not found. Check knowledge base ID, model ID, or base URL."
    if "500" in msg or "internal" in msg.lower():
        return f"Gateway error: {msg[:200]}"
    return msg[:500]


def headers_to_metadata(headers: Any) -> Dict[str, Any]:
    """
    Convert HTTP headers to metadata dict for gateway metrics.

    Args:
        headers: HTTP headers object or dict

    Returns:
        Dict with lowercase keys for consistent gateway metrics
    """
    if headers is None:
        return {}

    try:
        if hasattr(headers, "items"):
            return {str(k).lower(): v for k, v in headers.items()}
        elif hasattr(headers, "get"):
            return {str(k).lower(): headers.get(k) for k in headers.keys()}
        return {}
    except Exception:
        return {}


def llm_response_to_metadata(response: Any) -> Dict[str, Any]:
    """
    Extract gateway metrics from Flotorch LLMResponse and map to flotorch_eval format.

    FlotorchLLM returns LLMResponse with .metadata (inputTokens, outputTokens, totalTokens).
    GatewayMetrics expects: x-total-tokens, x-gateway-total-latency, x-total-cost.

    Args:
        response: LLMResponse from FlotorchLLM.ainvoke

    Returns:
        Dict with gateway-expected keys for GatewayMetrics aggregation
    """
    if response is None:
        return {}
    result: Dict[str, Any] = {}
    try:
        meta = getattr(response, "metadata", None)
        if not meta or not isinstance(meta, dict):
            return result
        # Map Flotorch metadata - keep inputTokens and outputTokens for display
        input_tok = meta.get("inputTokens") or meta.get("inputtokens")
        output_tok = meta.get("outputTokens") or meta.get("outputtokens")
        if input_tok is not None:
            result["inputTokens"] = int(input_tok) if isinstance(input_tok, str) else input_tok
        if output_tok is not None:
            result["outputTokens"] = int(output_tok) if isinstance(output_tok, str) else output_tok
        total = meta.get("totalTokens") or meta.get("totaltokens")
        if total is not None:
            result["x-total-tokens"] = int(total) if isinstance(total, str) else total
        elif input_tok is not None or output_tok is not None:
            inp = int(input_tok) if input_tok else 0
            out = int(output_tok) if output_tok else 0
            result["x-total-tokens"] = inp + out
        raw = meta.get("raw_response")
        if isinstance(raw, dict):
            usage = raw.get("usage", {})
            if "x-total-tokens" not in result and usage:
                tot = usage.get("total_tokens") or usage.get("total")
                if tot is not None:
                    result["x-total-tokens"] = int(tot)
            ext = raw.get("usage_metadata") or raw.get("gateway_metadata") or {}
            if isinstance(ext, dict):
                if "x-gateway-total-latency" not in result and ext.get("latency_ms"):
                    result["x-gateway-total-latency"] = float(ext["latency_ms"])
                if "x-total-cost" not in result and ext.get("cost") is not None:
                    result["x-total-cost"] = float(ext["cost"])
    except Exception:
        pass
    return result


def create_prompt_messages(
    system_prompt: str,
    user_prompt_template: str,
    question: str,
    context: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Build messages list for LLM inference.

    Args:
        system_prompt: System prompt for the LLM
        user_prompt_template: Template with {context} and {question} placeholders
        question: User question
        context: Optional list of context strings

    Returns:
        List of message dicts with role and content
    """
    context_text = ""

    if context:
        if isinstance(context, list):
            context_text = "\n\n---\n\n".join(context)
        elif isinstance(context, str):
            context_text = context

    # Replace placeholders in template
    user_content = (
        user_prompt_template
        .replace("{context}", context_text)
        .replace("{question}", question)
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def _compute_token_totals(evaluation_items: List[Any]) -> Dict[str, int]:
    """Sum input and output tokens from evaluation item metadata."""
    total_input, total_output = 0, 0
    for item in evaluation_items or []:
        meta = getattr(item, "metadata", None) or {}
        if isinstance(meta, dict):
            total_input += int(meta.get("inputTokens") or meta.get("inputtokens") or 0)
            total_output += int(meta.get("outputTokens") or meta.get("outputtokens") or 0)
    return {"total_input_tokens": total_input, "total_output_tokens": total_output}


def enrich_eval_results_with_gateway_metadata(
    eval_results: Dict[str, Any],
    evaluation_items: List[Any],
) -> None:
    """
    Inject per-query gateway metadata and token totals.
    flotorch_eval drops EvaluationItem.metadata when building question results;
    this restores it and adds total_input_tokens/total_output_tokens for display.
    """
    # Add token totals for GATEWAY METRICS section (input/output only)
    token_totals = _compute_token_totals(evaluation_items)
    if token_totals.get("total_input_tokens") or token_totals.get("total_output_tokens"):
        gw = eval_results.setdefault("gateway_metrics", {})
        gw["total_input_tokens"] = token_totals["total_input_tokens"]
        gw["total_output_tokens"] = token_totals["total_output_tokens"]
    qlr = eval_results.get("question_level_results")
    if not qlr or not evaluation_items:
        return
    for i, result in enumerate(qlr):
        if i < len(evaluation_items):
            item = evaluation_items[i]
            meta = getattr(item, "metadata", None) or {}
            if meta and isinstance(meta, dict):
                # Build readable per-query gateway metrics (flotorch_eval drops these)
                display_meta = {}
                if "x-total-tokens" in meta:
                    display_meta["tokens"] = meta["x-total-tokens"]
                if "x-gateway-total-latency" in meta:
                    display_meta["latency_ms"] = meta["x-gateway-total-latency"]
                if "x-total-cost" in meta:
                    display_meta["cost"] = meta["x-total-cost"]
                if "inputTokens" in meta:
                    display_meta["input_tokens"] = meta["inputTokens"]
                if "outputTokens" in meta:
                    display_meta["output_tokens"] = meta["outputTokens"]
                result["metadata"] = display_meta if display_meta else meta


def format_evaluation_results(
    eval_results: Dict[str, Any],
    show_query_level: bool = False,
    show_gateway: bool = False
) -> str:
    """
    Format evaluation results into a readable string report.

    Args:
        eval_results: Raw evaluation results dict
        show_query_level: Include per-query metrics breakdown
        show_gateway: Include gateway metrics (latency, cost, tokens)

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Overall metrics
    eval_metrics = eval_results.get("evaluation_metrics", {})
    if eval_metrics:
        lines.append("OVERALL METRICS:")
        lines.append("-" * 40)
        for metric_name, value in sorted(eval_metrics.items()):
            if isinstance(value, (int, float)):
                lines.append(f"  {metric_name}: {value:.4f}")
        lines.append("")

    # Gateway metrics (input/output tokens only)
    if show_gateway:
        gateway_metrics = eval_results.get("gateway_metrics", {})
        total_in = gateway_metrics.get("total_input_tokens", 0)
        total_out = gateway_metrics.get("total_output_tokens", 0)
        if total_in or total_out:
            lines.append("GATEWAY METRICS:")
            lines.append("-" * 40)
            lines.append(f"  total_input_tokens: {total_in:,}")
            lines.append(f"  total_output_tokens: {total_out:,}")
            lines.append("")

    # Query-level results
    if show_query_level:
        query_results = eval_results.get("question_level_results", [])
        if query_results:
            lines.append("QUERY-LEVEL RESULTS:")
            lines.append("-" * 40)
            for idx, result in enumerate(query_results, 1):
                lines.append(f"\nQuery {idx}:")
                question = result.get("question", "")[:100]
                lines.append(f"  Question: {question}{'...' if len(result.get('question', '')) > 100 else ''}")

                metrics = result.get("metrics", {})
                if metrics:
                    lines.append("  Metrics:")
                    for metric_name, value in sorted(metrics.items()):
                        if isinstance(value, (int, float)):
                            lines.append(f"    {metric_name}: {value:.4f}")

                if show_gateway:
                    metadata = result.get("metadata", {})
                    if metadata and ("input_tokens" in metadata or "output_tokens" in metadata or "tokens" in metadata):
                        tok_in = metadata.get("input_tokens", metadata.get("inputTokens", "–"))
                        tok_out = metadata.get("output_tokens", metadata.get("outputTokens", "–"))
                        lines.append(f"  Tokens: input={tok_in}, output={tok_out}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def validate_ground_truth_data(data: Any) -> Tuple[bool, str]:
    """
    Validate ground truth data structure.

    Args:
        data: Data to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, list):
        return False, "Ground truth must be a list of objects"

    if not data:
        return False, "Ground truth list cannot be empty"

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"Item at index {idx} must be a dict"

        if "question" not in item:
            return False, f"Item at index {idx} missing 'question' field"

        if "answer" not in item:
            return False, f"Item at index {idx} missing 'answer' field"

    return True, ""


def validate_evaluation_items(data: Any) -> Tuple[bool, str]:
    """
    Validate evaluation items structure.

    Args:
        data: Data to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, list):
        return False, "Evaluation items must be a list of objects"

    if not data:
        return False, "Evaluation items list cannot be empty"

    required_fields = ["question", "generated_answer", "expected_answer"]

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"Item at index {idx} must be a dict"

        for field in required_fields:
            if field not in item:
                return False, f"Item at index {idx} missing '{field}' field"

    return True, ""
