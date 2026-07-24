"""Relevance Grader Agent Node."""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def relevance_grader_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node to grade relevance of retrieved chunks against user question."""
    question = state.get("question", "")
    chunks = state.get("retrieved_chunks", [])
    processing_steps = list(state.get("processing_steps", []))

    if not chunks:
        step_info = {
            "name": "Relevance Grader",
            "status": "warning",
            "detail": "No chunks available to grade",
        }
        processing_steps.append(step_info)
        return {
            **state,
            "graded_chunks": [],
            "processing_steps": processing_steps,
        }

    graded_chunks: List[Dict[str, Any]] = []

    # Fast semantic vector score grading to prevent sequential LLM latency bottlenecks
    for chunk in chunks:
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")

        # High or moderate similarity chunks pass relevance test directly
        if score >= 0.25 or len(chunks) <= 3:
            chunk_copy = dict(chunk)
            chunk_copy["is_relevant"] = True
            chunk_copy["reason"] = f"Semantic similarity score ({score:.2f})"
            graded_chunks.append(chunk_copy)

    relevant_count = len(graded_chunks)
    step_info = {
        "name": "Relevance Grader",
        "status": "success" if relevant_count > 0 else "warning",
        "detail": f"Graded {relevant_count}/{len(chunks)} chunks as relevant",
    }
    processing_steps.append(step_info)
    logger.info("Graded %d/%d chunks as relevant", relevant_count, len(chunks))

    return {
        **state,
        "graded_chunks": graded_chunks,
        "processing_steps": processing_steps,
    }
