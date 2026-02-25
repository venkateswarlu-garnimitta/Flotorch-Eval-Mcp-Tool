"""
Configuration for the Flotorch Evaluation MCP Server.

Provides credential resolution, evaluation types, metric selection,
and user-facing metric resolution.
"""

import os
from enum import Enum
from typing import List, Optional, Tuple


class EvaluationType(Enum):
    """Supported evaluation types."""
    NORMAL = "normal"
    RAG = "rag"


class EvaluationEngine(Enum):
    """Supported evaluation engines."""
    DEEPEVAL = "deepeval"
    RAGAS = "ragas"


METRIC_NAME_MAP = {
    "faithfulness": "FAITHFULNESS",
    "answer_relevance": "ANSWER_RELEVANCE",
    "context_relevancy": "CONTEXT_RELEVANCY",
    "context_precision": "CONTEXT_PRECISION",
    "context_recall": "CONTEXT_RECALL",
    "hallucination": "HALLUCINATION",
}

DEFAULT_EVALUATION_ENGINE = "deepeval"


def get_flotorch_credentials(headers: Optional[dict] = None) -> Tuple[str, str]:
    """
    Extract Flotorch API key and base URL from headers or environment.

    Priority:
    1. HTTP headers (X-Flotorch-Api-Key, X-Flotorch-Base-Url)
    2. Environment variables (FLOTORCH_API_KEY, FLOTORCH_BASE_URL)
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
    """Return default metrics based on evaluation type."""
    eval_type = evaluation_type.lower().strip()

    if eval_type == EvaluationType.RAG.value:
        return [
            metric_key_class.FAITHFULNESS,
            metric_key_class.CONTEXT_RELEVANCY,
            metric_key_class.CONTEXT_PRECISION,
            metric_key_class.CONTEXT_RECALL,
            metric_key_class.ANSWER_RELEVANCE,
            metric_key_class.HALLUCINATION,
        ]
    else:
        return [metric_key_class.ANSWER_RELEVANCE]


def resolve_metrics(
    metric_names: Optional[List[str]],
    evaluation_type: str,
    metric_key_class: type,
) -> list:
    """
    Resolve user-specified metric names to MetricKey values.

    If metric_names is None or empty, falls back to the default set
    for the given evaluation_type.

    Accepts human-friendly names like "faithfulness", "answer_relevance",
    "context-relevancy" (hyphens / spaces are normalised).

    Raises:
        ValueError: If any metric name is unrecognised.
    """
    if not metric_names:
        return get_metrics_for_evaluation_type(evaluation_type, metric_key_class)

    resolved = []
    invalid = []
    for name in metric_names:
        key = name.lower().strip().replace(" ", "_").replace("-", "_")
        attr_name = METRIC_NAME_MAP.get(key)
        if attr_name and hasattr(metric_key_class, attr_name):
            resolved.append(getattr(metric_key_class, attr_name))
        else:
            invalid.append(name)

    if invalid:
        available = list(METRIC_NAME_MAP.keys())
        raise ValueError(f"Unknown metrics: {invalid}. Available: {available}")

    return resolved if resolved else get_metrics_for_evaluation_type(evaluation_type, metric_key_class)
