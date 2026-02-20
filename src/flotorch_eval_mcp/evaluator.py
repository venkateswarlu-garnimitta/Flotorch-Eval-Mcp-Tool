"""
Evaluation workflows for the Flotorch Evaluation MCP Server.

Handles dataset generation (normal and RAG), parallel processing,
and evaluation execution via flotorch-eval.
"""

import asyncio
import logging
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
    """Raised when LLM fails to generate answers. Stops experiment with reason."""

    def __init__(self, message: str, model_id: str = "", question: str = ""):
        self.model_id = model_id
        self.question = question
        super().__init__(message)


class KBRetrievalError(Exception):
    """Raised when knowledge base retrieval fails. Stops experiment with reason."""

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
    return_headers: bool = False,
) -> EvaluationItem:
    """
    Generate answer for a single question using the LLM.

    Args:
        question: User question
        expected_answer: Ground truth answer
        llm: FlotorchLLM instance
        system_prompt: System prompt for LLM
        user_prompt_template: User prompt template
        context: Optional context for RAG
        return_headers: Whether to return headers for gateway metrics

    Returns:
        EvaluationItem with generated answer and metadata
    """
    messages = create_prompt_messages(
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        question=question,
        context=context,
    )

    try:
        # Do NOT pass return_headers - FlotorchLLM returns LLMResponse (content, metadata).
        # Passing return_headers can cause gateway to return different format → "too many values to unpack".
        raw_result = await llm.ainvoke(messages=messages)
        generated_answer = getattr(raw_result, "content", str(raw_result))
        metadata = llm_response_to_metadata(raw_result) if return_headers else {}
    except Exception as e:
        logger.error(f"Failed to generate answer for question: {e}")
        raise  # Re-raise so caller can stop experiment and report reason

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
    return_headers: bool = False,
) -> EvaluationItem:
    """
    Retrieve context from KB and generate answer for a single question.

    Args:
        question: User question
        expected_answer: Ground truth answer
        kb: FlotorchAsyncVectorStore instance
        llm: FlotorchLLM instance
        system_prompt: System prompt for LLM
        user_prompt_template: User prompt template
        return_headers: Whether to return headers for gateway metrics

    Returns:
        EvaluationItem with retrieved context, generated answer, and metadata
    """
    try:
        # Retrieve context from knowledge base
        search_results = await kb.search(query=question)
        context_texts = memory_utils.extract_vectorstore_texts(search_results)
    except Exception as e:
        logger.error(f"Failed to retrieve context for question: {e}")
        raise KBRetrievalError(
            f"Knowledge base retrieval failed: {e}",
            question=question,
        ) from e

    # Generate answer with retrieved context
    return await generate_answer_for_question(
        question=question,
        expected_answer=expected_answer,
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        context=context_texts,
        return_headers=return_headers,
    )


async def generate_dataset_parallel(
    ground_truth: List[Dict[str, Any]],
    llm: FlotorchLLM,
    system_prompt: str,
    user_prompt_template: str,
    max_concurrent: int = 10,
    return_headers: bool = False,
) -> List[EvaluationItem]:
    """
    Generate evaluation dataset in parallel for all questions.

    Args:
        ground_truth: List of {question, answer} dicts
        llm: FlotorchLLM instance
        system_prompt: System prompt for LLM
        user_prompt_template: User prompt template
        max_concurrent: Maximum concurrent LLM calls
        return_headers: Whether to return headers for gateway metrics

    Returns:
        List of EvaluationItem objects
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(idx: int, qa: Dict[str, Any]) -> Tuple[int, EvaluationItem]:
        async with semaphore:
            question = qa.get("question", "")
            expected_answer = qa.get("answer", "")
            context = qa.get("context", [])

            # Normalize context to list
            if not isinstance(context, list):
                context = [str(context)] if context else []

            item = await generate_answer_for_question(
                question=question,
                expected_answer=expected_answer,
                llm=llm,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                context=context,
                return_headers=return_headers,
            )
            return idx, item

    # Process all questions in parallel
    tasks = [process_with_semaphore(i, qa) for i, qa in enumerate(ground_truth)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Fail fast: if any LLM generation failed, stop and report reason
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            err_msg = str(result)
            question = ground_truth[idx].get("question", "")[:80]
            raise LLMGenerationError(
                f"LLM failed to generate answer: {err_msg}",
                question=question,
            )

    # Sort by index and extract items
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
    return_headers: bool = False,
) -> List[EvaluationItem]:
    """
    Generate RAG evaluation dataset with maximum parallelism.

    Optimized two-phase approach:
    1. Retrieve contexts for ALL questions in parallel (up to max_concurrent)
    2. Generate answers for ALL questions in parallel (up to max_concurrent)

    This maximizes concurrency compared to sequential retrieve+generate per question.

    Args:
        ground_truth: List of {question, answer} dicts
        kb: FlotorchAsyncVectorStore instance
        llm: FlotorchLLM instance
        system_prompt: System prompt for LLM
        user_prompt_template: User prompt template
        max_concurrent: Maximum concurrent operations per phase
        return_headers: Whether to return headers for gateway metrics

    Returns:
        List of EvaluationItem objects with retrieved context
    """
    # Phase 1: Retrieve contexts for all questions in parallel
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

    # Process retrieval results and prepare for generation
    questions_with_context = []
    failed_retrievals = []

    for result in retrieval_results:
        if isinstance(result, Exception):
            # This shouldn't happen with our error handling above
            continue

        idx, question, expected_answer, context_texts, error = result
        if error:
            failed_retrievals.append((idx, error))
        else:
            questions_with_context.append((idx, question, expected_answer, context_texts))

    # Fail fast on retrieval errors
    if failed_retrievals:
        idx, error = failed_retrievals[0]
        question = ground_truth[idx].get("question", "")[:80]
        raise KBRetrievalError(str(error), question=question) from error

    # Phase 2: Generate answers for all questions in parallel
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
                    return_headers=return_headers,
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

    # Process generation results
    items_with_idx = []
    for result in generation_results:
        if isinstance(result, Exception):
            if isinstance(result, LLMGenerationError):
                raise result
            # This shouldn't happen with our error handling
            continue

        idx, item = result
        items_with_idx.append((idx, item))

    # Sort by original index and return
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
