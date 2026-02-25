"""
Evaluation workflows for the Flotorch Evaluation MCP Server.

Handles dataset generation (normal and RAG), parallel processing,
and evaluation execution via flotorch-eval.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from flotorch.sdk.llm import FlotorchLLM
from flotorch.sdk.memory import FlotorchAsyncVectorStore
from flotorch.sdk.utils import memory_utils
from flotorch_eval.llm_eval import EvaluationItem, LLMEvaluator

from flotorch_eval_mcp.utils import (
    create_prompt_messages,
    format_api_error,
    llm_response_to_metadata,
)


def _apply_deepeval_patch() -> None:
    """Apply CacheConfig patch to prevent DeepEval disk writes."""
    try:
        from deepeval.evaluate import CacheConfig
        import flotorch_eval.llm_eval.core.deepeval_evaluator as mod
        _orig = mod.evaluate

        def _patched(*args, **kwargs):
            kwargs["cache_config"] = CacheConfig(write_cache=False)
            return _orig(*args, **kwargs)

        mod.evaluate = _patched
    except Exception:
        pass


_apply_deepeval_patch()

logger = logging.getLogger(__name__)


class LLMGenerationError(Exception):
    """Raised when LLM fails to generate answers."""

    def __init__(self, message: str, model_id: str = "", question: str = ""):
        self.model_id = model_id
        self.question = question
        super().__init__(message)


class KBRetrievalError(Exception):
    """Raised when knowledge base retrieval fails."""

    def __init__(self, message: str, knowledge_base_id: str = "", question: str = ""):
        self.knowledge_base_id = knowledge_base_id
        self.question = question
        super().__init__(message)


async def generate_answer_for_question(
    question: str,
    expected_answer: str,
    llm: FlotorchLLM,
    system_prompt: str,
    user_prompt_template: str,
    context: Optional[List[str]] = None,
    gateway_metrics: bool = False,
) -> EvaluationItem:
    """
    Generate answer for a single question using the LLM.

    When gateway_metrics is True, captures token counts and wall-clock
    latency per call. Otherwise metadata is left empty.
    """
    messages = create_prompt_messages(
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        question=question,
        context=context,
    )

    try:
        if gateway_metrics:
            t0 = time.perf_counter()

        raw_result = await llm.ainvoke(messages=messages)
        generated_answer = getattr(raw_result, "content", str(raw_result))

        if gateway_metrics:
            wall_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            metadata = llm_response_to_metadata(raw_result)
            metadata["latency_ms"] = wall_latency_ms
        else:
            metadata = {}
    except Exception as e:
        logger.error(f"Failed to generate answer for question: {e}")
        raise

    return EvaluationItem(
        question=question,
        generated_answer=generated_answer,
        expected_answer=expected_answer,
        context=context or [],
        metadata=metadata,
    )


async def retrieve_and_generate(
    question: str,
    expected_answer: str,
    kb: FlotorchAsyncVectorStore,
    llm: FlotorchLLM,
    system_prompt: str,
    user_prompt_template: str,
    gateway_metrics: bool = False,
) -> EvaluationItem:
    """Retrieve context from KB and generate answer for a single question."""
    try:
        search_results = await kb.search(query=question)
        context_texts = memory_utils.extract_vectorstore_texts(search_results)
    except Exception as e:
        logger.error(f"Failed to retrieve context for question: {e}")
        raise KBRetrievalError(
            f"Knowledge base retrieval failed: {e}",
            question=question,
        ) from e

    return await generate_answer_for_question(
        question=question,
        expected_answer=expected_answer,
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        context=context_texts,
        gateway_metrics=gateway_metrics,
    )


async def generate_dataset_parallel(
    ground_truth: List[Dict[str, Any]],
    llm: FlotorchLLM,
    system_prompt: str,
    user_prompt_template: str,
    max_concurrent: int = 10,
    gateway_metrics: bool = False,
) -> List[EvaluationItem]:
    """
    Generate evaluation dataset in parallel for all questions.

    All questions are dispatched concurrently, bounded by max_concurrent.
    Metadata (tokens, latency) is captured only when gateway_metrics is True.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(idx: int, qa: Dict[str, Any]) -> Tuple[int, EvaluationItem]:
        async with semaphore:
            question = qa.get("question", "")
            expected_answer = qa.get("answer", "")
            context = qa.get("context", [])
            if not isinstance(context, list):
                context = [str(context)] if context else []

            item = await generate_answer_for_question(
                question=question,
                expected_answer=expected_answer,
                llm=llm,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                context=context,
                gateway_metrics=gateway_metrics,
            )
            return idx, item

    tasks = [process_with_semaphore(i, qa) for i, qa in enumerate(ground_truth)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            err_msg = str(result)
            question = ground_truth[idx].get("question", "")[:80]
            raise LLMGenerationError(
                f"LLM failed to generate answer: {err_msg}",
                question=question,
            )

    items_with_idx = [r for r in results if not isinstance(r, Exception)]
    items_with_idx.sort(key=lambda x: x[0])
    return [item for _, item in items_with_idx]


async def generate_rag_dataset_parallel(
    ground_truth: List[Dict[str, Any]],
    kb: FlotorchAsyncVectorStore,
    llm: FlotorchLLM,
    system_prompt: str,
    user_prompt_template: str,
    max_concurrent: int = 10,
    gateway_metrics: bool = False,
) -> List[EvaluationItem]:
    """
    Generate RAG evaluation dataset with maximum parallelism.

    Two-phase pipeline:
    1. Retrieve contexts for ALL questions in parallel
    2. Generate answers for ALL questions in parallel

    Metadata (tokens, latency) is captured only when gateway_metrics is True.
    """
    # Phase 1: Retrieve contexts in parallel
    logger.info(f"Phase 1: Retrieving contexts for {len(ground_truth)} questions (max_concurrent={max_concurrent})...")
    retrieval_semaphore = asyncio.Semaphore(max_concurrent)

    async def retrieve_context(idx: int, qa: Dict[str, Any]) -> Tuple[int, str, str, List[str], Optional[Exception]]:
        async with retrieval_semaphore:
            question = qa.get("question", "")
            expected_answer = qa.get("answer", "")
            try:
                search_results = await kb.search(query=question)
                context_texts = memory_utils.extract_vectorstore_texts(search_results)
                return idx, question, expected_answer, context_texts, None
            except Exception as e:
                logger.error(f"Failed to retrieve context for question {idx}: {e}")
                return idx, question, expected_answer, [], KBRetrievalError(
                    f"Knowledge base retrieval failed: {e}",
                    question=question,
                )

    retrieval_tasks = [retrieve_context(i, qa) for i, qa in enumerate(ground_truth)]
    retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

    questions_with_context = []
    failed_retrievals = []

    for result in retrieval_results:
        if isinstance(result, Exception):
            continue
        idx, question, expected_answer, context_texts, error = result
        if error:
            failed_retrievals.append((idx, error))
        else:
            questions_with_context.append((idx, question, expected_answer, context_texts))

    if failed_retrievals:
        idx, error = failed_retrievals[0]
        question = ground_truth[idx].get("question", "")[:80]
        raise KBRetrievalError(str(error), question=question) from error

    # Phase 2: Generate answers in parallel
    logger.info(f"Phase 2: Generating answers for {len(questions_with_context)} questions (max_concurrent={max_concurrent})...")
    generation_semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_answer(idx: int, question: str, expected_answer: str, context_texts: List[str]) -> Tuple[int, EvaluationItem]:
        async with generation_semaphore:
            try:
                item = await generate_answer_for_question(
                    question=question,
                    expected_answer=expected_answer,
                    llm=llm,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    context=context_texts,
                    gateway_metrics=gateway_metrics,
                )
                return idx, item
            except Exception as e:
                logger.error(f"Failed to generate answer for question {idx}: {e}")
                raise LLMGenerationError(
                    f"LLM failed to generate answer: {e}",
                    question=question[:80],
                ) from e

    generation_tasks = [
        generate_answer(idx, question, expected_answer, context_texts)
        for idx, question, expected_answer, context_texts in questions_with_context
    ]
    generation_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

    items_with_idx = []
    for result in generation_results:
        if isinstance(result, Exception):
            if isinstance(result, LLMGenerationError):
                raise result
            continue
        idx, item = result
        items_with_idx.append((idx, item))

    items_with_idx.sort(key=lambda x: x[0])
    return [item for _, item in items_with_idx]


async def run_evaluation(
    evaluation_items: List[EvaluationItem],
    api_key: str,
    base_url: str,
    evaluation_model: str,
    embedding_model: str,
    metrics: List[Any],
    evaluation_engine: str = "deepeval",
) -> Dict[str, Any]:
    """
    Run evaluation on the dataset using LLMEvaluator.

    Args:
        evaluation_items: List of EvaluationItem objects
        api_key: Flotorch API key
        base_url: Flotorch base URL
        evaluation_model: Model for evaluation scoring
        embedding_model: Embedding model for evaluation
        metrics: List of MetricKey values
        evaluation_engine: Evaluation engine (deepeval/ragas)

    Returns:
        Dict containing evaluation results with metrics
    """
    evaluator = LLMEvaluator(
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model,
        inferencer_model=evaluation_model,
        evaluation_engine=evaluation_engine,
        metrics=metrics,
    )

    valid_items = []
    invalid_count = 0
    for item in evaluation_items:
        answer_str = str(item.generated_answer or "").strip()
        is_invalid = (
            not answer_str
            or answer_str.startswith("Error:")
            or answer_str.lower() in ("none", "null", "nan")
        )
        if is_invalid:
            invalid_count += 1
            logger.debug(f"Skipping invalid item: question='{item.question[:50]}...', answer='{answer_str[:50]}...'")
        else:
            valid_items.append(item)

    if not valid_items:
        raise ValueError(f"Evaluation failed: No valid generated answers found. {invalid_count} out of {len(evaluation_items)} items had invalid/empty/error responses.")

    logger.info(f"Evaluating {len(valid_items)} valid items out of {len(evaluation_items)} total items ({invalid_count} skipped due to errors/empty responses)")

    try:
        eval_results = await evaluator.aevaluate(valid_items)
    except Exception as e:
        error_msg = str(e)
        if "'NoneType' object has no attribute 'hyperparameters'" in error_msg:
            raise ValueError(
                f"Evaluation failed: DeepEval test run was None (often when .deepeval temp file is missing). "
                f"Ensure the CacheConfig(write_cache=False) patch is applied. Error: {error_msg}"
            ) from e
        raise ValueError(f"Evaluation failed: {format_api_error(e)}") from e

    # Handle case where evaluation completely fails and returns None
    if eval_results is None:
        raise ValueError("Evaluation failed: No results returned from evaluation engine. This may be due to invalid test cases, model errors, or evaluation configuration issues.")

    return eval_results
