"""
Configuration for the Flotorch Evaluation MCP Server.

Provides credential resolution, evaluation types, and metric selection.
"""

import os
from enum import Enum
from typing import Optional, Tuple


class EvaluationType(Enum):
    """Supported evaluation types."""
    NORMAL = "normal"  # No context, answer relevance only
    RAG = "rag"  # Full RAG metrics with context


class EvaluationEngine(Enum):
    """Supported evaluation engines."""
    DEEPEVAL = "deepeval"
    RAGAS = "ragas"


def get_flotorch_credentials(headers: Optional[dict] = None) -> Tuple[str, str]:
    """
    Extract Flotorch API key and base URL from headers or environment.

    Priority:
    1. HTTP headers (X-Flotorch-Api-Key, X-Flotorch-Base-Url)
    2. Environment variables (FLOTORCH_API_KEY, FLOTORCH_BASE_URL)

    Args:
        headers: Optional dict of HTTP headers (lowercase keys)

    Returns:
        Tuple of (api_key, base_url)

    Raises:
        ValueError: If credentials are not found
    """
    api_key = ""
    base_url = ""

    if headers:
        api_key = (
            headers.get("x-flotorch-api-key")
            or headers.get("flotorch-api-key")
            or headers.get("authorization", "").replace("Bearer ", "").strip()
        )
        base_url = (
            headers.get("x-flotorch-base-url")
            or headers.get("flotorch-base-url", "")
        )

    # Fallback to environment variables
    api_key = (api_key or os.getenv("FLOTORCH_API_KEY", "")).strip()
    base_url = (base_url or os.getenv("FLOTORCH_BASE_URL", "")).strip()

    if not api_key:
        raise ValueError(
            "Flotorch API key not found. Set X-Flotorch-Api-Key header or FLOTORCH_API_KEY env variable."
        )
    if not base_url:
        raise ValueError(
            "Flotorch base URL not found. Set X-Flotorch-Base-Url header or FLOTORCH_BASE_URL env variable."
        )

    return api_key, base_url


def get_metrics_for_evaluation_type(
    evaluation_type: str, metric_key_class: type
) -> list:
    """
    Return appropriate metrics based on evaluation type.

    Args:
        evaluation_type: Type of evaluation (normal/rag)
        metric_key_class: MetricKey class from flotorch_eval

    Returns:
        List of metric keys appropriate for the evaluation type
    """
    eval_type = evaluation_type.lower().strip()

    if eval_type == EvaluationType.RAG.value:
        # Full RAG metrics: faithfulness, context metrics, answer relevance, hallucination
        return [
            metric_key_class.FAITHFULNESS,
            metric_key_class.CONTEXT_RELEVANCY,
            metric_key_class.CONTEXT_PRECISION,
            metric_key_class.CONTEXT_RECALL,
            metric_key_class.ANSWER_RELEVANCE,
            metric_key_class.HALLUCINATION,
        ]
    else:
        # Normal evaluation: only answer relevance (no context-based metrics)
        return [metric_key_class.ANSWER_RELEVANCE]
