"""
Utility functions for the Flotorch Evaluation MCP Server.

Provides error formatting, prompt building, validation, and result formatting.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def format_api_error(exc: Exception) -> str:
    """Extract a clear user-facing message from API or gateway errors."""
    msg = str(exc).strip()
    if not msg:
        return f"API error: {type(exc).__name__}"

    match = re.search(r'"message"\s*:\s*"([^"]+)"', msg)
    if match:
        return match.group(1)

    if "ERROR_MODEL_NOT_FOUND" in msg:
        return "Model not found. Check that the model ID exists on the gateway."
    if "401" in msg or "unauthorized" in msg.lower():
        return "API key invalid or missing. Set X-Flotorch-Api-Key or FLOTORCH_API_KEY."
    if "api" in msg.lower() and "key" in msg.lower():
        return "API key invalid or missing. Set X-Flotorch-Api-Key or FLOTORCH_API_KEY."
    if "404" in msg or "not found" in msg.lower():
        return "Resource not found. Check knowledge base ID, model ID, or base URL."
    if "500" in msg or "internal" in msg.lower():
        return f"Gateway error: {msg[:200]}"

    return msg[:500]


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
    context: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Build messages list for LLM inference.

    Ensures the template includes {question}; appends it if missing so the model
    always receives the question. For RAG, {context} must be present.

    Args:
        system_prompt: System prompt for the LLM.
        user_prompt_template: Template with {context} and {question} placeholders.
        question: The user question.
        context: Optional list of context strings for RAG.

    Returns:
        List of message dicts with role and content.
    """
    template = (user_prompt_template or "").strip()
    if "{question}" not in template:
        template = template + "\n\nQuestion: {question}" if template else "Question: {question}"
    if context and "{context}" not in template:
        template = "Context:\n{context}\n\n" + template

    context_text = ""
    if context:
        context_text = "\n\n---\n\n".join(
            str(c) for c in (context if isinstance(context, list) else [context])
        )

    user_content = template.replace("{context}", context_text).replace("{question}", question)

    return [
        {"role": "system", "content": (system_prompt or "").strip() or "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]


def _compute_gateway_totals(evaluation_items: List[Any]) -> Dict[str, Any]:
    """Aggregate tokens and latency across all evaluation items."""
    total_input = 0
    total_output = 0
    latencies: list = []

    for item in evaluation_items or []:
        meta = getattr(item, "metadata", None) or {}
        if not isinstance(meta, dict):
            continue
        total_input += int(meta.get("inputTokens") or meta.get("inputtokens") or 0)
        total_output += int(meta.get("outputTokens") or meta.get("outputtokens") or 0)
        lat = meta.get("latency_ms") or meta.get("x-gateway-total-latency")
        if lat is not None:
            latencies.append(float(lat))

    totals: Dict[str, Any] = {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
    }
    if latencies:
        totals["total_latency_ms"] = round(sum(latencies), 2)
        totals["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
        totals["min_latency_ms"] = round(min(latencies), 2)
        totals["max_latency_ms"] = round(max(latencies), 2)
    return totals


def enrich_eval_results_with_gateway_metadata(
    eval_results: Dict[str, Any],
    evaluation_items: List[Any],
) -> None:
    """
    Inject per-query gateway metadata, token totals, and latency stats.

    flotorch_eval drops EvaluationItem.metadata when building question results;
    this restores it and adds aggregated gateway_metrics for display.
    """
    totals = _compute_gateway_totals(evaluation_items)
    has_data = any(v for v in totals.values() if v)
    if has_data:
        eval_results["gateway_metrics"] = totals

    qlr = eval_results.get("question_level_results")
    if not qlr or not evaluation_items:
        return
    for i, result in enumerate(qlr):
        if i >= len(evaluation_items):
            break
        item = evaluation_items[i]
        meta = getattr(item, "metadata", None) or {}
        if not meta or not isinstance(meta, dict):
            continue
        display_meta: Dict[str, Any] = {}
        if "inputTokens" in meta:
            display_meta["input_tokens"] = meta["inputTokens"]
        if "outputTokens" in meta:
            display_meta["output_tokens"] = meta["outputTokens"]
        if "x-total-tokens" in meta:
            display_meta["tokens"] = meta["x-total-tokens"]
        if "latency_ms" in meta:
            display_meta["latency_ms"] = meta["latency_ms"]
        elif "x-gateway-total-latency" in meta:
            display_meta["latency_ms"] = meta["x-gateway-total-latency"]
        if "x-total-cost" in meta:
            display_meta["cost"] = meta["x-total-cost"]
        result["metadata"] = display_meta if display_meta else meta


def format_evaluation_results(
    eval_results: Dict[str, Any],
    show_query_level: bool = False,
) -> str:
    """Format evaluation results into a readable string report."""
    lines = []
    lines.append("=" * 80)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 80)
    lines.append("")

    eval_metrics = eval_results.get("evaluation_metrics", {})
    if eval_metrics:
        lines.append("OVERALL METRICS:")
        lines.append("-" * 40)
        for metric_name, value in sorted(eval_metrics.items()):
            if isinstance(value, (int, float)):
                lines.append(f"  {metric_name}: {value:.4f}")
        lines.append("")

    gateway_metrics = eval_results.get("gateway_metrics", {})
    if gateway_metrics:
        lines.append("PERFORMANCE:")
        lines.append("-" * 40)
        total_in = gateway_metrics.get("total_input_tokens", 0)
        total_out = gateway_metrics.get("total_output_tokens", 0)
        if total_in or total_out:
            lines.append(f"  total_input_tokens: {total_in:,}")
            lines.append(f"  total_output_tokens: {total_out:,}")
        if "avg_latency_ms" in gateway_metrics:
            lines.append(f"  avg_latency_ms: {gateway_metrics['avg_latency_ms']}")
            lines.append(f"  min_latency_ms: {gateway_metrics['min_latency_ms']}")
            lines.append(f"  max_latency_ms: {gateway_metrics['max_latency_ms']}")
            lines.append(f"  total_latency_ms: {gateway_metrics['total_latency_ms']}")
        lines.append("")

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

                metadata = result.get("metadata", {})
                if metadata:
                    parts = []
                    if "input_tokens" in metadata:
                        parts.append(f"in={metadata['input_tokens']}")
                    if "output_tokens" in metadata:
                        parts.append(f"out={metadata['output_tokens']}")
                    if "latency_ms" in metadata:
                        parts.append(f"latency={metadata['latency_ms']}ms")
                    if parts:
                        lines.append(f"  Gateway: {', '.join(parts)}")

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
