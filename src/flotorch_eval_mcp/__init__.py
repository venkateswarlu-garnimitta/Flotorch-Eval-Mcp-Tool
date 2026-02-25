"""
Flotorch Evaluation MCP Server.

MCP tools for LLM evaluation via the Flotorch platform.
"""

__version__ = "1.0.0"

from flotorch_eval_mcp.config import (
    DEFAULT_EVALUATION_ENGINE,
    EvaluationEngine,
    EvaluationType,
    get_flotorch_credentials,
    get_metrics_for_evaluation_type,
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
    format_evaluation_results,
    validate_evaluation_items,
    validate_ground_truth_data,
)

__all__ = [
    "__version__",
    "DEFAULT_EVALUATION_ENGINE",
    "EvaluationEngine",
    "EvaluationType",
    "get_flotorch_credentials",
    "get_metrics_for_evaluation_type",
    "resolve_metrics",
    "KBRetrievalError",
    "LLMGenerationError",
    "generate_dataset_parallel",
    "generate_rag_dataset_parallel",
    "run_evaluation",
    "enrich_eval_results_with_gateway_metadata",
    "format_api_error",
    "format_evaluation_results",
    "validate_evaluation_items",
    "validate_ground_truth_data",
]
