"""Main graph execution interface for Agentic RAG workflow."""
import logging
from typing import Dict, Any
from agents.workflow import create_agentic_rag_graph
from config import DEFAULT_TOP_K, DEFAULT_TEMPERATURE, MAX_RETRIES

logger = logging.getLogger(__name__)

# Compile graph once
rag_graph = create_agentic_rag_graph()


def run_agentic_rag(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """Execute the LangGraph Agentic RAG workflow for a user question.

    Args:
        question: User query string.
        top_k: Top-K context passages to retrieve.
        temperature: LLM temperature setting.
        max_retries: Maximum rewrite attempts.

    Returns:
        Dict containing answer, confidence, sources, processing_steps, and full state.
    """
    initial_state = {
        "question": question,
        "rewritten_question": "",
        "retrieved_chunks": [],
        "graded_chunks": [],
        "confidence": 0.0,
        "retry_count": 0,
        "max_retries": max_retries,
        "top_k": top_k,
        "temperature": temperature,
        "answer": "",
        "sources": [],
        "processing_steps": [],
    }

    try:
        final_state = rag_graph.invoke(initial_state)
        return final_state
    except Exception as e:
        logger.error("LangGraph execution error: %s", str(e), exc_info=True)
        return {
            **initial_state,
            "answer": f"An error occurred during workflow execution: {str(e)}",
            "confidence": 0.0,
            "processing_steps": [
                {
                    "name": "Workflow Execution",
                    "status": "error",
                    "detail": f"Execution failed: {str(e)}",
                }
            ],
        }
